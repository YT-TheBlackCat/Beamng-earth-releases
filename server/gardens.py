"""Gardens in the world from the baked recipes (gardenbake.py, tools/bake_gardens.py): lawn, beds, terraces and
driveways as the terrain's material layers, hedges as rows of the stock shrub, trees and shrubs at their measured
places and heights in place of the canopy model's in the garden parcels.

    paint(idx, lat, lon, cache_dir)         -> idx with the garden ground painted (Surface.paint_ground_pads)
    digest(lat0, lat1, lon0, lon1, cache_dir) -> bytes: the recipes the box reads (the block cache's key)
    hedge_rows(frame, box, z_at, cache_dir, ...) -> Forest rows of the hedges the tile owns (streetfront.tile)
    swap_vegetation(items, key, cache_dir)  -> vegetation.Items: the garden parcels' trees and shrubs (Vegetation.cell)

Only where a recipe was baked (Baden-Wuerttemberg, the squares tools/bake_gardens.py made); everywhere else nothing
changes. Switches: BNE_GARDENS=off (all of it), BNE_GARDEN_GROUND / BNE_GARDEN_HEDGES / BNE_GARDEN_TREES=off (one part).
"""
import hashlib
import logging
import math
import os
import threading
import zipfile
import zlib

import numpy as np

import dnp
import gardenbake
import lod2geom

log = logging.getLogger("earthd.gardens")

ON = os.environ.get("BNE_GARDENS", "on") != "off"
GROUND_ON = ON and os.environ.get("BNE_GARDEN_GROUND", "on") != "off"
HEDGES_ON = ON and os.environ.get("BNE_GARDEN_HEDGES", "on") != "off"
TREES_ON = ON and os.environ.get("BNE_GARDEN_TREES", "on") != "off"
BW = lod2geom.BW
MAX_TILES = 24
HEDGE_H = (1.2, 2.6)        # m: a hedge row's shrubs (the stock shrub grows as wide as high: taller hedges are drawn lower)
HEDGE_STEP_M = 1.6          # m between a baked hedge's shrubs (each as wide as high: they still overlap)
HEDGE_TRI_BUDGET = 120000   # per road tile (fences.DE_TRIS["hedge"] = 400 a shrub: ~480 m of hedge; U8 has ~300 m a tile)
HEDGE_REACH_M = 80.0        # m around a road tile whose recipes' hedges are read (a hedge is kept in its middle's square)
CONIFER_NDVI = 0.30         # a crown greener than this (NDVI) is a broadleaf (conifers read darker in the near infrared)
CONIFER_MIN_H = 6.0         # m: ... lower crowns are broadleaves anyway


def folder(cache_dir):
    d = os.environ.get("BNE_GARDENS_DIR")
    return d if d else os.path.join(cache_dir, "gardens", gardenbake.VERSION)


class Recipes:
    """The baked squares, read once (a few MB each in memory), the most recent MAX_TILES kept."""
    _tiles = {}                 # (path, mtime) -> dict of arrays
    _lock = threading.Lock()
    _bad = set()

    @classmethod
    def get(cls, cache_dir, e_km, n_km):
        path = gardenbake.km_path(folder(cache_dir), e_km, n_km)
        try:
            key = (path, os.path.getmtime(path))
        except OSError:
            return None
        with cls._lock:
            d = cls._tiles.get(key)
        if d is not None:
            return d
        try:
            with np.load(path) as z:
                d = {k: z[k] for k in z.files}
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile, zlib.error) as e:
            with cls._lock:
                first = key not in cls._bad
                cls._bad.add(key)
            if first:
                log.warning("gardens: unreadable recipe %s (%s)", path, e)
            return None
        with cls._lock:
            cls._tiles[key] = d
            while len(cls._tiles) > MAX_TILES:
                cls._tiles.pop(next(iter(cls._tiles)))
        return d

    @classmethod
    def stamp(cls, cache_dir, e_km, n_km):
        path = gardenbake.km_path(folder(cache_dir), e_km, n_km)
        try:
            st = os.stat(path)
        except OSError:
            return None
        return f"{e_km}_{n_km}:{st.st_size}:{st.st_mtime_ns}"


def _in_bw(lat, lon):
    return BW[0] <= lat <= BW[2] and BW[1] <= lon <= BW[3]


def _squares(E, N):
    """The km squares (e_km, n_km) the UTM points reach."""
    return [(e, n) for e in range(int(math.floor(float(np.min(E)) / 1000)), int(math.floor(float(np.max(E)) / 1000)) + 1)
            for n in range(int(math.floor(float(np.min(N)) / 1000)), int(math.floor(float(np.max(N)) / 1000)) + 1)]


def _utm_of_grid(lat, lon):
    """UTM32 (E, N) of a block's sample grid: exact at a sparse lattice, an affine fit in between (a 1 km block is
    affine to well under a millimetre)."""
    lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    if lat.ndim != 2 or lat.size < 64:
        return lod2geom.utm32_from_wgs(lat, lon)
    h, w = lat.shape
    step = max(1, min(h, w) // 16)
    la, lo = lat[::step, ::step].ravel(), lon[::step, ::step].ravel()
    E, N = lod2geom.utm32_from_wgs(la, lo)
    lo0, la0 = float(lo.mean()), float(la.mean())
    A = np.stack([lo - lo0, la - la0, np.ones_like(lo)], 1)
    coef, *_ = dnp.lstsq(A, np.stack([E, N], 1), rcond=None)
    dlo, dla = lon - lo0, lat - la0
    return coef[0, 0] * dlo + coef[1, 0] * dla + coef[2, 0], coef[0, 1] * dlo + coef[1, 1] * dla + coef[2, 1]


def classes_at(lat, lon, cache_dir):
    """The recipes' ground class (gardenbake.G_*) at the points (0 where no recipe or no garden parcel)."""
    E, N = _utm_of_grid(lat, lon)
    E, N = np.asarray(E), np.asarray(N)
    out = np.zeros(E.shape, np.uint8)
    for e_km, n_km in _squares(E, N):
        d = Recipes.get(cache_dir, e_km, n_km)
        if d is None:
            continue
        g, px = d["ground"], float(d["px"])
        sel = (E >= e_km * 1000) & (E < (e_km + 1) * 1000) & (N >= n_km * 1000) & (N < (n_km + 1) * 1000)
        if not sel.any():
            continue
        col = np.clip(((E[sel] - e_km * 1000) / px).astype(np.int64), 0, g.shape[1] - 1)
        row = np.clip((((n_km + 1) * 1000 - N[sel]) / px).astype(np.int64), 0, g.shape[0] - 1)
        out[sel] = g[row, col]
    return out


def _material_lut():
    import surface
    lut = np.full(256, 255, np.uint8)
    G = gardenbake
    for c, m in ((G.G_LAWN, surface.LAWN), (G.G_BED, surface.DUSTY), (G.G_PLANTING, surface.DUSTY),
                 (G.G_TERRACE, surface.PAVERS), (G.G_DRIVEWAY, surface.PAVERS), (G.G_PATH, surface.PAVERS),
                 (G.G_GRAVEL, surface.LOOSE)):
        lut[c] = m
    return lut


def paint(idx, lat, lon, cache_dir):
    """idx (terrain material indices, surface.MATERIALS) with the gardens of the baked recipes painted in: lawn LAWN,
    beds and the ground under hedges and shrubs DUSTY (the scrub layer's), terraces, driveways and paths PAVERS,
    gravel LOOSE. Only natural ground is painted (DIRT, LAWN, GRASS, DUSTY: roads, sidewalks, lots and squares keep
    theirs); water and buildings keep what they have. Pure but for the recipe files read."""
    import surface
    if not GROUND_ON or not cache_dir:
        return idx
    lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    if not _in_bw(float(lat.mean()), float(lon.mean())):
        return idx
    cls = classes_at(lat, lon, cache_dir)
    if not cls.any():
        return idx
    mat = _material_lut()[cls]
    ok = (mat != 255) & np.isin(idx, (surface.DIRT, surface.LAWN, surface.GRASS, surface.DUSTY))
    if not ok.any():
        return idx
    out = idx.copy()
    out[ok] = mat[ok]
    return out


def digest(lat0, lat1, lon0, lon1, cache_dir):
    """A hash of the switches and of the recipe files (name, size, time) the box (degrees) reads: blocks built
    before a square was baked (or baked again) are built anew."""
    h = hashlib.sha256(repr((GROUND_ON, gardenbake.VERSION)).encode())
    if not GROUND_ON or not cache_dir or not _in_bw((lat0 + lat1) / 2, (lon0 + lon1) / 2):
        return h.digest()
    E, N = lod2geom.utm32_from_wgs(np.array([lat0, lat0, lat1, lat1]), np.array([lon0, lon1, lon0, lon1]))
    for e_km, n_km in _squares(E, N):
        st = Recipes.stamp(cache_dir, e_km, n_km)
        if st:
            h.update(st.encode())
    return h.digest()


def _to_enu(frame, e, n):
    lat, lon = lod2geom.wgs_from_utm32(e, n)
    x, y, _ = frame.geodetic_to_enu(np.asarray(lat), np.asarray(lon), np.zeros(np.size(lat)))
    return np.asarray(x, np.float64), np.asarray(y, np.float64)


def _recipes_around(frame, box, cache_dir, margin=20.0):
    x0, y0, x1, y1 = box
    lat, lon, _ = frame.enu_to_geodetic(np.array([x0 - margin, x1 + margin, x1 + margin, x0 - margin]),
                                        np.array([y0 - margin, y0 - margin, y1 + margin, y1 + margin]), np.zeros(4))
    if not _in_bw(float(np.mean(lat)), float(np.mean(lon))):
        return []
    E, N = lod2geom.utm32_from_wgs(lat, lon)
    out = []
    for e_km, n_km in _squares(E, N):
        d = Recipes.get(cache_dir, e_km, n_km)
        if d is not None:
            out.append(d)
    return out


def hedge_lines(frame, box, cache_dir):
    """[(line (k, 2) ENU, height m, width m)] of the baked hedges reaching within HEDGE_REACH_M of the box."""
    out = []
    x0, y0, x1, y1 = box
    for d in _recipes_around(frame, box, cache_dir, margin=HEDGE_REACH_M):
        st = d["hedge_start"]
        if len(st) < 2:
            continue
        ox, oy = float(d["e_km"]) * 1000.0, float(d["n_km"]) * 1000.0
        xy = d["hedge_xy"].astype(np.float64)
        x, y = _to_enu(frame, xy[:, 0] + ox, xy[:, 1] + oy)
        for i in range(len(st) - 1):
            a, b = int(st[i]), int(st[i + 1])
            if b - a < 2:
                continue
            lx, ly = x[a:b], y[a:b]
            if lx.max() < x0 - 2 or lx.min() > x1 + 2 or ly.max() < y0 - 2 or ly.min() > y1 + 2:
                continue
            out.append((np.stack([lx, ly], 1), float(d["hedge_h"][i]), float(d["hedge_w"][i])))
    return out


def hedge_rows(frame, box, z_at, cache_dir, carriageways=(), rings=(), walks=(), drives=(), driveways=(),
               existing=()):
    """Forest rows [group, x, y, z, yaw, scale, height, rank] of the baked hedges whose shrubs' centres are in the
    box (x0, y0, x1, y1, ENU m): the stock hedge shrub every HEDGE_STEP_M, as high as measured (HEDGE_H),
    kept off roads, drives, sidewalks and buildings like the mapped German boundaries (fences.boundary_rows), none
    within a crown of a mapped hedge already in existing (rows), at most HEDGE_TRI_BUDGET triangles (nearest the
    box's centre first). Off with BNE_GARDEN_HEDGES=off or the German boundaries' BNE_DE_BOUNDARIES=off."""
    import fences
    if not HEDGES_ON or not cache_dir or not fences.de_enabled():
        return []
    lines = hedge_lines(frame, box, cache_dir)
    if not lines:
        return []
    x0, y0, x1, y1 = box
    mapped = np.array([r[1:3] for r in existing if r and r[0] == fences.G_HEDGE], np.float64).reshape(-1, 2)
    pending, heights = [], []
    for line, h, _ in lines:
        for centre, tangent in fences._centres(line, HEDGE_STEP_M):
            if not (x0 <= centre[0] < x1 and y0 <= centre[1] < y1):
                continue
            if len(mapped) and np.min(dnp.hypot(*(mapped - centre).T)) < 1.2:
                continue
            pending.append(("hedge", np.asarray(centre, np.float64), np.asarray(tangent, np.float64),
                            HEDGE_STEP_M))
            heights.append(float(np.clip(h, *HEDGE_H)))
    if not pending:
        return []
    mid = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    order = np.argsort([float(dnp.hypot(*(p[1] - mid))) for p in pending], kind="stable")
    pending, heights = [pending[i] for i in order], [heights[i] for i in order]
    return fences.boundary_rows(pending, [], z_at, carriageways, rings, walks, drives, driveways, heights=heights,
                                budget=HEDGE_TRI_BUDGET, what="garden hedges")


def swap_vegetation(items, key, cache_dir):
    """vegetation.Items of the map cell key ((z, x, y)) with the canopy model's trees and bushes in the baked garden
    parcels replaced by the recipes' (a tree at its measured top, height and crown; conifer when its crown is dark in
    the near infrared; shrubs as bushes). Rocks stay. items unchanged where nothing was baked."""
    if not TREES_ON or not cache_dir:
        return items
    import mapcells
    import vegetation as V
    k = key if isinstance(key, mapcells.CellKey) else mapcells.CellKey(*[int(v) for v in key])
    s, w, n, e = _cell_bounds(k)
    if not _in_bw((s + n) / 2, (w + e) / 2):
        return items
    E, N = lod2geom.utm32_from_wgs(np.array([s, s, n, n]), np.array([w, e, w, e]))
    recipes = [d for d in (Recipes.get(cache_dir, a, b) for a, b in _squares(E, N)) if d is not None]
    if not recipes:
        return items
    keep = np.ones(len(items), bool)
    if len(items):
        garden = classes_at(items.lat, items.lon, cache_dir) > 0
        # (the far clumps stay: a clump stands for trees around it, some outside the garden; vegetation.clumps)
        keep = ~(garden & np.isin(items.kind, (V.KIND_TREE, V.KIND_BUSH)) & ~np.isin(items.group, V.CLUMP_GROUPS))
    lon, lat, hh, rad, kind, group = [], [], [], [], [], []
    for d in recipes:
        ox, oy = float(d["e_km"]) * 1000.0, float(d["n_km"]) * 1000.0
        t = d["tree"].astype(np.float64)       # (float32 northings would snap to 0.5 m)
        if len(t):
            la, lo = lod2geom.wgs_from_utm32(t[:, 0] + ox, t[:, 1] + oy)
            conifer = (t[:, 4] < CONIFER_NDVI) & (t[:, 2] >= CONIFER_MIN_H)
            lon.append(lo), lat.append(la), hh.append(t[:, 2]), rad.append(t[:, 3])
            kind.append(np.full(len(t), V.KIND_TREE, np.uint8))
            group.append(np.where(conifer, V.GROUPS.index("conifer"), V.GROUPS.index("deciduous")).astype(np.uint8))
        b = d["shrub"].astype(np.float64)
        if len(b):
            la, lo = lod2geom.wgs_from_utm32(b[:, 0] + ox, b[:, 1] + oy)
            lon.append(lo), lat.append(la), hh.append(b[:, 2]), rad.append(b[:, 3])
            kind.append(np.full(len(b), V.KIND_BUSH, np.uint8))
            group.append(np.full(len(b), V.GROUPS.index("shrub"), np.uint8))
    if not lon:
        return items.take(keep) if not keep.all() else items
    lon, lat = np.concatenate(lon), np.concatenate(lat)
    inside = (lat >= s) & (lat < n) & (lon >= w) & (lon < e)
    lon, lat = lon[inside], lat[inside]
    hh = np.concatenate(hh)[inside].astype(items.height.dtype if len(items) else np.float32)
    rad = np.concatenate(rad)[inside].astype(items.radius.dtype if len(items) else np.float32)
    kind = np.concatenate(kind)[inside]
    group = np.concatenate(group)[inside]
    # deterministic yaw, rank and size variation from the position (vegetation._hash01 keys on integer cells)
    qa, qb = np.round(lon * 1e6).astype(np.int64), np.round(lat * 1e6).astype(np.int64)
    yaw = (V._hash01(qa, qb, 11) * 360.0).astype(np.float32)
    rank = V._hash01(qa, qb, 12).astype(np.float32)
    scale = np.ones(len(lon), np.float32)
    add = V.Items(lon.astype(items.lon.dtype if len(items) else np.float64),
                  lat.astype(items.lat.dtype if len(items) else np.float64), hh, rad,
                  kind.astype(np.uint8), group.astype(np.uint8), rank, yaw, scale)
    kept = items.take(keep)
    return V._concat_all([kept, add]) if len(kept) else add


def _cell_bounds(k):
    """(south, west, north, east) of a map cell (mapcells.CellKey.bounds: west, south, east, north)."""
    w, s, e, n = (float(v) for v in k.bounds())
    return s, w, n, e
