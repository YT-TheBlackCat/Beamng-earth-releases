#!/usr/bin/env python3
"""How accurately the built Leonberg buildings match the LGL LoD2 models (the reference), per test spot.

    .venv/bin/python tools/leonberg_building_accuracy.py [--spots T1,T8,HW18] [--radius 150] [--out DIR]
                                                          [--exp NAME[,NAME]] [--cache DIR] [--detail E,N]

Reads only (the real cache as the other Leonberg tools; LoD2 geometry from ~/.cache/beamng-earth/lod2geom/v1, built by
tools/build_lgl_lod2_geom.py). Deterministic: a 0.5 m raster in the spot's ENU frame.

Per spot, the items of buildings.road_tile_items (what the mesh is built from) and the LoD2 units are rastered:
  footprint  built: the ring its walls stand on (buildings.drawn_ring, the oriented box for a pitched estimated roof)
                    and its roof's cells; LoD2: the unit's ground polygons
  surface    height over the footprint's lowest ground: built from the roof triangles of buildings.build([item]),
             LoD2 from its roof polygons (z over the unit's lowest ground)
Built items and LoD2 units overlapping by >= 20 % of the smaller one form a cluster (one real building, any number of
units and items). Per cluster (centre within the radius): IoU of the footprints, ridge error (highest roof built -
LoD2), eave error (the largest item's eave - the largest unit's eave), mean |surface difference| over the union.
  missing   LoD2 unit (>= 12 m2, code != 9999) < 30 % covered by any built footprint
  extra     built item on the ground (>= 12 m2) < 30 % of it on LoD2 ground
  double    two items on the ground overlapping by >= 50 % of the smaller (lod2+fp: one of them LoD2, one not)
Source of each item: LoD2 (from a footprint), LoD2-Luecke (leftover unit), Teil (OSM building:part), Hoehe:<height_src>
/Stockwerke/geschaetzt; for an item without LoD2 on LoD2 ground the reason (Teile, Innenhof, Verhaeltnis r, ...).
--exp: prototype changes, monkeypatched in this process only: lod2ring (LoD2 walls on the LoD2 ring, not its oriented
box), lod2noclear (road clearance leaves LoD2 items alone), xtile (leftover sees neighbouring tiles' footprints),
ratio (take window 0.3-3.0), nogeom (no LoD2 geometry: footprints + LoD2 records, as without the lod2geom cache),
fillfix (records without roof shape pass their roof height). lod2ring, xtile and the passage part of lod2noclear are in
buildings.py since 2026-11-24 (lod2ring and xtile change nothing there any more). Writes <out>/<spot>_clusters.csv, prints a summary table.
"""
import argparse
import csv
import logging
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "server"))
os.environ.setdefault("BNE_ROOF_PHOTO", "off")

CELL = 0.5
LINK = 0.2          # overlap share (of the smaller) that links an item and a unit into one cluster
COVER = 0.3
DOUBLE = 0.5
MIN_M2 = 12.0
EXTRA_SPOTS = [("HW18", "Hohensteiner Weg 18", 48.789383, 8.990135)]
EXPERIMENTS = {}


def experiment(name):
    def deco(fn):
        EXPERIMENTS[name] = fn
        return fn
    return deco


# ------------------------------------------------------------------------------------------- raster

class Grid:
    def __init__(self, half):
        self.half = half
        self.n = int(math.ceil(2 * half / CELL))

    def _sub(self, xy):
        xy = np.asarray(xy, np.float64)[:, :2]
        i0 = max(0, int(math.floor((xy[:, 0].min() + self.half) / CELL)))
        i1 = min(self.n, int(math.ceil((xy[:, 0].max() + self.half) / CELL)) + 1)
        j0 = max(0, int(math.floor((xy[:, 1].min() + self.half) / CELL)))
        j1 = min(self.n, int(math.ceil((xy[:, 1].max() + self.half) / CELL)) + 1)
        return i0, i1, j0, j1

    def cells_in(self, ring):
        """Flat indices of the cells whose centre is in the ring (ENU)."""
        from polygon import Path
        ring = np.asarray(ring, np.float64)[:, :2]
        if len(ring) < 3:
            return np.zeros(0, np.int64)
        i0, i1, j0, j1 = self._sub(ring)
        if i1 <= i0 or j1 <= j0:
            return np.zeros(0, np.int64)
        I, J = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
        pts = np.stack([(I.ravel() + 0.5) * CELL - self.half, (J.ravel() + 0.5) * CELL - self.half], 1)
        ins = Path(ring).contains_points(pts)
        return (I.ravel()[ins] * self.n + J.ravel()[ins]).astype(np.int64)

    def surface(self, tris):
        """{cell: max z} of the triangles (k, 3, 3), ENU, vertical ones left out."""
        out = {}
        for t in tris:
            a, b, c = t
            d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(d) < 1e-6:
                continue
            i0, i1, j0, j1 = self._sub(t)
            if i1 <= i0 or j1 <= j0:
                continue
            I, J = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
            px = (I.ravel() + 0.5) * CELL - self.half
            py = (J.ravel() + 0.5) * CELL - self.half
            l1 = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / d
            l2 = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / d
            l3 = 1 - l1 - l2
            ins = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
            if not ins.any():
                continue
            z = l1[ins] * a[2] + l2[ins] * b[2] + l3[ins] * c[2]
            for k, zz in zip((I.ravel()[ins] * self.n + J.ravel()[ins]).tolist(), z.tolist()):
                if zz > out.get(k, -1e9):
                    out[k] = zz
        return out


def _tri_poly(P):
    """Triangles (k, 3, 3) of a planar 3-D polygon by its xy projection (fan where ear clipping fails)."""
    import buildings as B
    P = np.asarray(P, np.float64).reshape(-1, 3)
    if len(P) < 3:
        return np.zeros((0, 3, 3))
    xy = P[:, :2]
    s = B._area2(xy)
    if abs(s) < 1e-6:
        return np.zeros((0, 3, 3))
    order = np.arange(len(P)) if s > 0 else np.arange(len(P))[::-1]
    try:
        T = B.triangulate(xy[order])
        T = order[T]
    except Exception:
        T = np.array([[0, k, k + 1] for k in range(1, len(P) - 1)])
    if len(T) < len(P) - 2:
        T = np.array([[0, k, k + 1] for k in range(1, len(P) - 1)])
    return P[T]


# ------------------------------------------------------------------------------------------- recording hooks

class Rec:
    attrs = {}          # id(spec) -> (spec, attrs)
    takes = []          # (cx, cy, ring area, unit area inside, n inside, n taken)
    unit_of = {}        # id(spec.lod2) -> (spec.lod2, LoD2 unit id)
    clear = Counter()   # LoD2 items the road clearance touched
    ref = False         # True while the reference LoD2 is loaded (experiment nogeom)


def _install_hooks():
    import buildings as B
    import lod2geom
    plan0 = B.plan

    def plan(ring, attrs, *a, **k):
        s = plan0(ring, attrs, *a, **k)
        Rec.attrs[id(s)] = (s, attrs)
        return s
    B.plan = plan
    copy0 = B._spec_copy

    def spec_copy(spec, **kw):
        s = copy0(spec, **kw)
        got = Rec.attrs.get(id(spec))
        if got is not None:
            Rec.attrs[id(s)] = (s, got[1])
        return s
    B._spec_copy = spec_copy
    unit0 = B._lod2_unit

    def lod2_unit(g, i, x0, y0):
        got = unit0(g, i, x0, y0)
        if got is not None:
            Rec.unit_of[id(got[1]["_lod2"])] = (got[1]["_lod2"], g.unit(i)["id"])
        return got
    B._lod2_unit = lod2_unit
    sp0 = B._split_passages

    def split_passages(items, split, *a, **k):
        for kk, _r, _b in split:
            spec = items[kk][0]
            if spec.lod2:
                Rec.clear["lod2_split"] += 1
                Rec.clear["lod2_split_h>=%s" % ("tall" if (spec.height or 0) >= B.ROAD_CLEAR_PASSAGE_H + B.ROAD_CLEAR_LINTEL_M else "low")] += 1
        return sp0(items, split, *a, **k)
    B._split_passages = split_passages
    cr0 = B._clear_roads

    def clear_roads(items, *a, **k):
        before = {id(it[0]): np.asarray(it[0].ring).copy() for it in items if it[0].lod2}
        out = cr0(items, *a, **k)
        for it in out:
            r0 = before.get(id(it[0]))
            if r0 is not None and (r0.shape != np.asarray(it[0].ring).shape or not np.allclose(r0, it[0].ring)):
                Rec.clear["lod2_trimmed"] += 1
        return out
    B._clear_roads = clear_roads
    take0 = lod2geom.Geoms.take

    def take(self, e, n, area_ratio=None):
        from polygon import Path
        e_, n_ = np.asarray(e, np.float64), np.asarray(n, np.float64)
        if len(e_) > 1 and e_[0] == e_[-1] and n_[0] == n_[-1]:
            e_, n_ = e_[:-1], n_[:-1]
        info = [float(e_.mean()), float(n_.mean()), abs(lod2geom._signed(e_, n_)), 0.0, 0, 0]
        if len(e_) >= 3 and self.n:
            ii = np.array([i for i in range(self.n) if i not in self._taken], np.int64)
            if len(ii):
                ins = Path(np.stack([e_, n_], 1)).contains_points(np.stack([self.cx[ii], self.cy[ii]], 1))
                ii = ii[ins]
                info[3] = float(sum(self.unit(int(i))["area"] for i in ii))
                info[4] = len(ii)
        got = take0(self, e, n) if area_ratio is None else take0(self, e, n, area_ratio)
        info[5] = len(got)
        Rec.takes.append(info)
        return got
    lod2geom.Geoms.take = take


# ------------------------------------------------------------------------------------------- experiments

@experiment("lod2ring")
def _exp_lod2ring():
    """LoD2 items: walls on the LoD2 ground ring itself, not its oriented box (drawn_ring)."""
    import buildings as B
    dr0 = B.drawn_ring

    def drawn_ring(spec):
        if getattr(spec, "lod2", None):
            return spec.ring
        return dr0(spec)
    B.drawn_ring = drawn_ring


@experiment("ratio")
def _exp_ratio():
    """Geoms.take window 0.3 .. 3.0 (instead of 0.45 .. 2.0)."""
    import lod2geom
    t0 = lod2geom.Geoms.take

    def take(self, e, n, area_ratio=(0.3, 3.0)):
        return t0(self, e, n, area_ratio)
    lod2geom.Geoms.take = take


@experiment("nogeom")
def _exp_nogeom():
    """No LoD2 geometry for the generator (as on a machine without the lod2geom cache): footprints + LoD2 records."""
    import lod2geom
    load0 = lod2geom.Geoms.load.__func__

    def load(cls, *a, **k):
        return load0(cls, *a, **k) if Rec.ref else None
    lod2geom.Geoms.load = classmethod(load)


@experiment("lod2noclear")
def _exp_lod2noclear():
    """The road clearance leaves LoD2 items alone (no trim, no pieces that each redraw the whole LoD2 roof)."""
    import buildings as B
    c0 = B._clearable

    def clearable(spec):
        return c0(spec) and not getattr(spec, "lod2", None)
    B._clearable = clearable


@experiment("xtile")
def _exp_xtile():
    """Geoms.leftover also sees the outlines of footprints whose box reaches into the tile but whose centre is another
    tile's (a footprint of tile A takes a unit whose centre is in tile B; B drew it again as a leftover)."""
    import buildings as B
    import lod2geom
    import mapgeom
    tb0 = B.tile_buildings
    ctx = {}

    def tile_buildings(frame, x0, y0, x1, y1, layers, *a, **k):
        ctx["frame"], ctx["layers"] = frame, layers
        return tb0(frame, x0, y0, x1, y1, layers, *a, **k)
    B.tile_buildings = tile_buildings
    lo0 = lod2geom.Geoms.leftover

    def leftover(self, x0, y0, x1, y1, outlines, own=None, **k):
        frame, extra = ctx.get("frame"), []
        for f in ctx.get("layers") or ():
            if f is None or not isinstance(f, mapgeom.Features) or len(f.kind) == 0:
                continue
            a = f.attrs
            bb = f.boxes(frame)
            rows = np.flatnonzero((bb[:, 2] >= x0) & (bb[:, 0] < x1) & (bb[:, 3] >= y0) & (bb[:, 1] < y1))
            for i in rows.tolist():
                if a.get("is_underground") is not None and bool(a["is_underground"][i]):
                    continue
                lv = a["level"][i] if a.get("level") is not None else None
                if lv is not None and np.isfinite(lv) and lv < 0:
                    continue
                parts = f.parts(i)
                if not parts or not parts[0] or len(parts[0][0]) < 4:
                    continue
                extra.append(mapgeom.enu_xy(f, frame, np.asarray(parts[0][0], np.float64)))
        return lo0(self, x0, y0, x1, y1, list(outlines) + extra, own, **k)
    lod2geom.Geoms.leftover = leftover


@experiment("fillfix")
def _exp_fillfix():
    """lod2roofs.Records.fill without a known roof shape: the record's roof height is passed on (roof_height) and the
    height is the whole top, so plan's estimated pitched roof takes the measured rise (eave = top - roof)."""
    import lod2roofs
    f0 = lod2roofs.Records.fill

    def fill(self, at, e, n):
        i = self.find(e, n)
        out = f0(self, at, e, n)
        if i is not None and not lod2roofs.SHAPES.get(int(self.code[i])) and self.roof[i] > 0 \
                and out.get("height_src") == "LoD2":
            out["height"] = float(self.height[i])
            if not lod2roofs._known(out.get("roof_height")):
                out["roof_height"] = float(self.roof[i])
        return out
    lod2roofs.Records.fill = fill


# ------------------------------------------------------------------------------------------- measuring

def _source(spec, attrs):
    if getattr(spec, "lod2", None):
        sid = str(attrs.get("seed_id") or "")
        return "LoD2-Luecke" if sid.startswith(("DEBW_", "UUID_")) else "LoD2"
    if attrs.get("building_height") is not None:
        return "Teil"
    hs = str(attrs.get("height_src") or "")
    h = attrs.get("height")
    if h is not None and np.isfinite(h) and h > 0:
        if hs == "LoD2":
            return "LoD2-Datensatz"
        if hs in ("Microsoft ML Buildings", "Google Open Buildings"):
            return "ML-Hoehe"
        return "Karte"
    nf = attrs.get("num_floors")
    if nf is not None and np.isfinite(nf) and nf > 0:
        return "Stockwerke"
    return "geschaetzt"


def _no_lod2_reason(attrs, ring_enu, takes):
    if attrs.get("building_height") is not None:
        return "Teil"
    if attrs.get("_parts_drawn"):
        return "Umriss+Teile"
    if attrs.get("_holes"):
        return "Innenhof"
    c = np.asarray(ring_enu, np.float64).mean(0)
    best = None
    for t in takes:
        d = math.hypot(t[0] - c[0], t[1] - c[1])
        if d < 3.0 and (best is None or d < best[0]):
            best = (d, t)
    if best is None:
        return "nicht versucht"
    t = best[1]
    if t[4] == 0:
        return "kein Unit-Mittelpunkt"
    return f"Verhaeltnis {t[3] / max(t[2], 1e-6):.2f}"


def measure_spot(st, spot, radius, detail=None):
    import buildings as B
    import geodesy
    import lod2geom
    import roads
    import leonberg_buildings_report as R

    sid, name, lat, lon = spot
    frame = geodesy.EnuFrame(lat, lon)
    reach = radius + 60.0
    grid = Grid(reach + 40.0)
    Rec.attrs.clear()
    Rec.unit_of.clear()
    Rec.clear.clear()
    Rec.takes.clear()
    k_lo, k_hi = math.floor(-reach / roads.TILE), math.floor(reach / roads.TILE)
    items = []      # (spec, gmin, gmax, x0, y0)
    for tx in range(k_lo, k_hi + 1):
        for ty in range(k_lo, k_hi + 1):
            x0, y0 = roads.tile_bbox(tx, ty)[:2]
            its, _c = B.road_tile_items(frame, tx, ty, st.md, R._ground(frame, st),
                                        urban_fn=st.surface.urban if st.surface is not None else None,
                                        widths=st.widths)
            for it in its:
                c = np.asarray(it[0].ring, np.float64)[:, :2].mean(0) + (x0, y0)
                if math.hypot(c[0], c[1]) <= reach:
                    items.append((it[0], it[1], it[2], x0, y0))
    # --- built
    built = []
    for spec, gmin, gmax, x0, y0 in items:
        got = Rec.attrs.get(id(spec))
        attrs = got[1] if got is not None and got[0] is spec else {}
        if spec.cls in ("toll_booth",):
            continue
        try:
            wr = np.asarray(B.drawn_ring(spec), np.float64)[:, :2] + (x0, y0)
        except Exception:
            wr = np.asarray(spec.ring, np.float64)[:, :2] + (x0, y0)
        ring = np.asarray(spec.ring, np.float64)[:, :2] + (x0, y0)
        mesh = B.build([(spec, gmin, gmax)])
        surf = {}
        ntri = 0 if mesh is None else len(np.asarray(mesh[2]).reshape(-1, 3))
        if mesh is not None:
            P, _uv, T, M, mats, _C = mesh
            P = np.asarray(P, np.float64).reshape(-1, 3)
            T = np.asarray(T, np.int64).reshape(-1, 3)
            M = np.asarray(M, np.int64).reshape(-1)
            isr = np.array([0 <= m < len(mats) and "roof" in str(mats[m]) for m in M], bool)
            tri = P[T[isr]].copy()
            tri[:, :, 0] += x0
            tri[:, :, 1] += y0
            tri[:, :, 2] -= gmin
            surf = grid.surface(tri)
        fp = np.union1d(grid.cells_in(wr), np.fromiter(surf.keys(), np.int64, len(surf)))
        mh = float(spec.min_h or 0.0)
        if spec.lod2:
            L = spec.lod2
            rs = [np.asarray(P_, np.float64).reshape(-1, 3) for k_, P_ in L["polys"] if k_ == 2 and len(P_) >= 3]
            e0 = min(float(r_[:, 2].min()) for r_ in rs) if rs else float(L["eave"])
            eave = max(0.0, (gmax - gmin) + B.LOD2_EAVE_CLEAR_M - e0) + e0
        else:
            eave = B.base_z(spec, gmin, gmax) + mh + float(spec.height or 0.0) - gmin
        uo = Rec.unit_of.get(id(spec.lod2)) if spec.lod2 else None
        built.append(dict(ntri=ntri, spec=spec, attrs=attrs, uid=uo[1] if uo is not None and uo[0] is spec.lod2 else None, ring=ring, wring=wr, fp=fp, surf=surf, mh=mh, eave=eave,
                          src=_source(spec, attrs), area=abs(B._area2(ring)) / 2,
                          box_area=abs(B._area2(wr)) / 2, c=ring.mean(0)))
    # --- LoD2
    Rec.ref = True
    geo = lod2geom.Geoms.load(st.cache_dir if hasattr(st, "cache_dir") else os.path.expanduser("~/.cache/beamng-earth"),
                              frame, -reach, -reach, reach, reach, margin=0.0)
    Rec.ref = False
    units = []
    if geo is not None:
        for i in range(geo.n):
            if math.hypot(geo.cx[i], geo.cy[i]) > reach:
                continue
            u = geo.unit(i)
            fp = np.zeros(0, np.int64)
            for k_, P_ in u["polys"]:
                if k_ == 0:
                    fp = np.union1d(fp, grid.cells_in(P_[:, :2]))
            if not len(fp):
                fp = grid.cells_in(u["ring"])
            tris = [t for k_, P_ in u["polys"] if k_ == 2 for t in _tri_poly(P_)]
            surf = grid.surface(np.array(tris)) if tris else {}
            units.append(dict(u=u, fp=fp, surf=surf, c=np.array([u["cx"], u["cy"]]), area=u["area"]))
    # --- links
    cell_b = defaultdict(list)
    for k, b in enumerate(built):
        for c in b["fp"].tolist():
            cell_b[c].append(k)
    cell_u = defaultdict(list)
    for k, u in enumerate(units):
        for c in u["fp"].tolist():
            cell_u[c].append(k)
    parent = list(range(len(built) + len(units)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    ov = Counter()
    for c, bl in cell_b.items():
        for ul in [cell_u.get(c, ())]:
            for kb in bl:
                for ku in ul:
                    ov[(kb, ku)] += 1
    for (kb, ku), n in ov.items():
        if n >= LINK * min(len(built[kb]["fp"]), len(units[ku]["fp"])):
            parent[find(kb)] = find(len(built) + ku)
    clusters = defaultdict(lambda: ([], []))
    for kb in range(len(built)):
        clusters[find(kb)][0].append(kb)
    for ku in range(len(units)):
        clusters[find(len(built) + ku)][1].append(ku)
    # --- per cluster
    rows = []
    for _, (bl, ul) in sorted(clusters.items(), key=lambda kv: kv[0]):
        cs = [built[k]["c"] for k in bl] + [units[k]["c"] for k in ul]
        cen = np.mean(cs, 0)
        if math.hypot(cen[0], cen[1]) > radius:
            continue
        ground_b = [k for k in bl if built[k]["mh"] <= 0.5]
        fb = np.unique(np.concatenate([built[k]["fp"] for k in bl])) if bl else np.zeros(0, np.int64)
        fu = np.unique(np.concatenate([units[k]["fp"] for k in ul])) if ul else np.zeros(0, np.int64)
        inter = len(np.intersect1d(fb, fu))
        union = len(np.union1d(fb, fu))
        row = dict(e=round(float(cen[0]), 1), n=round(float(cen[1]), 1), n_built=len(bl), n_units=len(ul),
                   area_built=round(len(fb) * CELL * CELL, 1), area_lod2=round(len(fu) * CELL * CELL, 1),
                   iou=round(inter / union, 3) if union and bl and ul else None)
        if bl:
            big = max(bl, key=lambda k: built[k]["area"])
            row["src"] = built[big]["src"]
            row["srcs"] = "+".join(sorted(set(built[k]["src"] for k in bl)))
            row["box_excess"] = round(sum(built[k]["box_area"] for k in bl) / max(sum(built[k]["area"] for k in bl), 1e-6), 3)
            if ul and not any(built[k]["spec"].lod2 for k in bl):
                row["why_no_lod2"] = _no_lod2_reason(built[big]["attrs"], built[big]["ring"], Rec.takes)
        if bl and ul:
            sb, su = {}, {}
            for k in bl:
                for c, z in built[k]["surf"].items():
                    if z > sb.get(c, -1e9):
                        sb[c] = z
            for k in ul:
                for c, z in units[k]["surf"].items():
                    if z > su.get(c, -1e9):
                        su[c] = z
            if sb and su:
                row["top_built"] = round(max(sb.values()), 2)
                row["top_lod2"] = round(max(su.values()), 2)
                row["d_top"] = round(row["top_built"] - row["top_lod2"], 2)
            bigu = max(ul, key=lambda k: units[k]["area"])
            row["eave_lod2"] = round(units[bigu]["u"]["eave"], 2)
            row["eave_built"] = round(built[big]["eave"], 2)
            row["d_eave"] = round(row["eave_built"] - row["eave_lod2"], 2)
            cells = np.union1d(fb, fu).tolist()
            row["surf_mae"] = round(float(np.mean([abs(sb.get(c, 0.0) - su.get(c, 0.0)) for c in cells])), 2)
            row["code"] = units[bigu]["u"]["code"]
            row["lod2_id"] = units[bigu]["u"]["id"]
        rows.append(row)
    # --- missing / extra / double
    bcells = set().union(*[set(b["fp"].tolist()) for b in built]) if built else set()
    ucells = set().union(*[set(u["fp"].tolist()) for u in units]) if units else set()
    missing = []
    for u in units:
        if math.hypot(*u["c"]) > radius or u["area"] < MIN_M2 or u["u"]["code"] == 9999 or not len(u["fp"]):
            continue
        cov = sum(1 for c in u["fp"].tolist() if c in bcells) / len(u["fp"])
        if cov < COVER:
            missing.append((u["u"]["id"], round(u["area"]), u["u"]["code"], round(float(u["c"][0])), round(float(u["c"][1]))))
    extra = []
    for b in built:
        if math.hypot(*b["c"]) > radius or b["mh"] > 0.5 or b["area"] < MIN_M2 or not len(b["fp"]):
            continue
        cov = sum(1 for c in b["fp"].tolist() if c in ucells) / len(b["fp"])
        if cov < COVER:
            extra.append((b["src"], round(b["area"]), round(float(b["c"][0])), round(float(b["c"][1]))))
    double = []
    pairs = Counter()
    for c, bl in cell_b.items():
        g = [k for k in bl if built[k]["mh"] <= 0.5]
        for a_ in range(len(g)):
            for b_ in range(a_ + 1, len(g)):
                pairs[(g[a_], g[b_])] += 1
    for (ka, kb), n in pairs.items():
        A, Bb = built[ka], built[kb]
        if math.hypot(*A["c"]) > radius and math.hypot(*Bb["c"]) > radius:
            continue
        sm = min(len(A["fp"]), len(Bb["fp"]))
        if sm and n >= DOUBLE * sm:
            if A["uid"] is not None and A["uid"] == Bb["uid"]:
                kind = "same-unit"
            elif "Teil" in (A["src"], Bb["src"]):
                kind = "teil"
            elif bool(A["spec"].lod2) != bool(Bb["spec"].lod2):
                kind = "lod2+fp"
            elif A["spec"].lod2:
                kind = "lod2+lod2"
            else:
                kind = "fp+fp"
            double.append((kind, A["src"], Bb["src"], A["uid"], Bb["uid"], round(float(A["c"][0])), round(float(A["c"][1])),
                           round(n / sm, 2)))
    if detail is not None:
        dx, dy = detail
        print(f"--- Detail um ({dx:.0f}, {dy:.0f}) ---")
        for r in sorted(rows, key=lambda r: math.hypot(r["e"] - dx, r["n"] - dy))[:6]:
            print("  ", r)
        for b in sorted(built, key=lambda b: math.hypot(b["c"][0] - dx, b["c"][1] - dy))[:4]:
            s = b["spec"]
            print(f"   item {b['src']} c=({b['c'][0]:.1f},{b['c'][1]:.1f}) ring {b['area']:.0f} m2, walls {b['box_area']:.0f}"
                  f" m2, roof={s.roof} h={s.height} floors={s.floors} roof_h={s.roof_h} pitch={s.pitch} lod2={bool(s.lod2)}"
                  f" cls={s.cls} attrs={{{', '.join(f'{k}={v}' for k, v in b['attrs'].items() if v is not None and not str(k).startswith('_lod2'))}}}")
        for u in sorted(units, key=lambda u: math.hypot(u["c"][0] - dx, u["c"][1] - dy))[:4]:
            uu = u["u"]
            print(f"   unit {uu['id']} c=({uu['cx']:.1f},{uu['cy']:.1f}) {uu['area']:.0f} m2 code {uu['code']} eave {uu['eave']:.2f}"
                  f" top {uu['top']:.2f}")
    return name, rows, missing, extra, double, built, units


def _q(v, p):
    v = [x for x in v if x is not None]
    return round(float(np.percentile(v, p)), 2) if v else None


def summary(sid, rows, missing, extra, double, built, radius):
    both = [r for r in rows if r.get("iou") is not None]
    iou = [r["iou"] for r in both]
    dt = [abs(r["d_top"]) for r in both if r.get("d_top") is not None]
    de = [abs(r["d_eave"]) for r in both if r.get("d_eave") is not None]
    sm = [r["surf_mae"] for r in both if r.get("surf_mae") is not None]
    nb = sum(1 for b in built if math.hypot(*b["c"]) <= radius)
    nu = sum(r["n_units"] for r in rows)
    out = dict(spot=sid, tris=sum(b["ntri"] for b in built if math.hypot(*b["c"]) <= radius), n_built=nb, n_lod2=nu, n_clusters=len(both), iou_med=_q(iou, 50), iou_p10=_q(iou, 10),
               iou_lt07=sum(1 for x in iou if x < 0.7), top_med=_q(dt, 50), top_p90=_q(dt, 90),
               top_gt2=sum(1 for x in dt if x > 2.0), eave_med=_q(de, 50), eave_p90=_q(de, 90),
               surf_med=_q(sm, 50), surf_p90=_q(sm, 90), missing=len(missing), extra=len(extra),
               double=len(double), **{"d_" + k: v for k, v in sorted(Counter(d[0] for d in double).items())})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spots", default="T1,T2,T3,T7,T8,HW18")
    ap.add_argument("--radius", type=float, default=150.0)
    ap.add_argument("--out", default=os.path.expanduser("~/bnge-tests/leonberg/accuracy"))
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/beamng-earth"))
    ap.add_argument("--exp", default="", help="comma list of prototype changes: " + ", ".join(EXPERIMENTS))
    ap.add_argument("--detail", default=None, help="E,N (m, spot frame): print the clusters/items/units near it")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)
    import leonberg_buildings_report as R
    import blockworker as bw

    spots = {s[0]: s for s in R.SPOTS + EXTRA_SPOTS}
    want = [s.strip() for s in args.spots.split(",") if s.strip()]
    bad = [s for s in want if s not in spots]
    if bad:
        ap.error(f"unknown spot(s): {', '.join(bad)}")
    _install_hooks()
    for e in [e.strip() for e in args.exp.split(",") if e.strip()]:
        EXPERIMENTS[e]()
    os.makedirs(args.out, exist_ok=True)
    st = bw._State(args.cache, False, True, True, True, True)
    bw._s = st
    st.cache_dir = args.cache
    detail = tuple(float(v) for v in args.detail.split(",")) if args.detail else None
    sums = []
    for sid in want:
        name, rows, missing, extra, double, built, units = measure_spot(st, spots[sid], args.radius, detail)
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0].keys()).index(k)
                      if rows and k in rows[0] else 99)
        with open(os.path.join(args.out, f"{sid}_clusters.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        s = summary(sid, rows, missing, extra, double, built, args.radius)
        sums.append(s)
        print(sid, name, s, flush=True)
        both = [r for r in rows if r.get("iou") is not None]
        multi = Counter(b["uid"] for b in built if b["uid"] is not None)
        print("  LoD2-Units mehrfach gezeichnet:", sum(1 for v in multi.values() if v > 1),
              "Items:", sum(v for v in multi.values() if v > 1), "Strassen-Freihaltung an LoD2:", dict(Rec.clear))
        print("  Quelle (groesstes Item je Cluster):", dict(Counter(r["src"] for r in both)))
        print("  ohne LoD2 auf LoD2-Grund, Grund:", dict(Counter(r.get("why_no_lod2") for r in both if r.get("why_no_lod2"))))
        if args.verbose:
            print("  fehlend:", missing[:12])
            print("  ueberzaehlig:", extra[:12])
            print("  doppelt:", double[:12])
            for r in sorted(both, key=lambda r: r["iou"])[:8]:
                print("  IoU schlecht:", r)
            for r in sorted(both, key=lambda r: -abs(r.get("d_top") or 0))[:6]:
                print("  First schlecht:", r)
    print()
    cols = []
    for s in sums:
        cols += [k for k in s if k not in cols]
    print("| " + " | ".join(cols) + " |")
    for s in sums:
        print("| " + " | ".join(str(s.get(c, 0)) for c in cols) + " |")


if __name__ == "__main__":
    main()
