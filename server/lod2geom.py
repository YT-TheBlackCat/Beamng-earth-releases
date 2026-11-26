"""LoD2 building geometry (ground, wall and roof surfaces) from the LGL BW open data.

tools/build_lgl_lod2_geom.py prepares each 2x2 km UTM32 tile of the CityGML zips as a compact npz (one row per
building or building part, its polygons, heights relative to its ground). Geoms.load places the tiles around a box
in the ENU frame of the scene; Geoms.take gives the units that stand in a footprint ring.
"""
import logging
import math
import os
import threading
import xml.etree.ElementTree as ET
import zipfile
import zlib

import dnp
import numpy as np

from polygon import Path

log = logging.getLogger("earthd.lod2geom")

ON = os.environ.get("BNE_LOD2_GEOM", "on") != "off"
# LoD2 units no footprint covers are drawn too (Geoms.leftover); BNE_LOD2_GAPS=off keeps them out
GAPS = os.environ.get("BNE_LOD2_GAPS", "on") != "off"
# a tile not on disk is downloaded from the LGL and prepared the first time a scene needs it (Geoms.load with a fetcher;
# before, only tools/build_lgl_lod2_geom.py made them: a game without it had none and fell back to the footprints with
# one height each, the Hohensteiner Weg's houses with 3 m eaves, 2026-11-24); BNE_LOD2_GEOM_FETCH=off: the disk only
FETCH = os.environ.get("BNE_LOD2_GEOM_FETCH", "on") != "off"
FETCH_RETRY_S = 600.0                    # s: a download that failed is tried again after this (a missing tile never)
BW = (47.5, 7.4, 49.85, 10.55)           # south, west, north, east
VERSION = "v1"
URL = "https://opengeodata.lgl-bw.de/data/lod2/LoD2_32_{e}_{n}_2_bw.zip"
CREDIT = "LoD2-Gebäudemodelle © LGL BW, www.lgl-bw.de, dl-de/by-2-0"

GID = "{http://www.opengis.net/gml}id"
GRID_M = 32.0                            # centroid grid of take()
MAX_TILES = 16
_SURF = {"GroundSurface": 0, "WallSurface": 1, "RoofSurface": 2, "ClosureSurface": None}

def _ipow(x, k):
    r = 1.0
    for _ in range(k):
        r = r * x
    return r


# UTM32 / GRS80 transverse Mercator, Krueger series to n^6 (Karney 2011)
_A, _F = 6378137.0, 1 / 298.257222101
_N = _F / (2 - _F)
_E = math.sqrt(_F * (2 - _F))
_K0, _FE, _LON0 = 0.9996, 500000.0, 9.0
_A0 = _A / (1 + _N) * (1 + _ipow(_N, 2) / 4 + _ipow(_N, 4) / 64 + _ipow(_N, 6) / 256)
_ALPHA = [
    {1: 1 / 2, 2: -2 / 3, 3: 5 / 16, 4: 41 / 180, 5: -127 / 288, 6: 7891 / 37800},
    {2: 13 / 48, 3: -3 / 5, 4: 557 / 1440, 5: 281 / 630, 6: -1983433 / 1935360},
    {3: 61 / 240, 4: -103 / 140, 5: 15061 / 26880, 6: 167603 / 181440},
    {4: 49561 / 161280, 5: -179 / 168, 6: 6601661 / 7257600},
    {5: 34729 / 80640, 6: -3418889 / 1995840},
    {6: 212378941 / 319334400},
]
_BETA = [
    {1: 1 / 2, 2: -2 / 3, 3: 37 / 96, 4: -1 / 360, 5: -81 / 512, 6: 96199 / 604800},
    {2: 1 / 48, 3: 1 / 15, 4: -437 / 1440, 5: 46 / 105, 6: -1118711 / 3870720},
    {3: 17 / 480, 4: -37 / 840, 5: -209 / 4480, 6: 5569 / 90720},
    {4: 4397 / 161280, 5: -11 / 504, 6: -830251 / 7257600},
    {5: 4583 / 161280, 6: -108847 / 3991680},
    {6: 20648693 / 638668800},
]


def _series(coef):
    return [sum(c * _ipow(_N, k) for k, c in d.items()) for d in coef]


_AL, _BE = _series(_ALPHA), _series(_BETA)


def folder(cache_dir):
    d = os.environ.get("BNE_LOD2GEOM_DIR")
    return d if d else os.path.join(cache_dir, "lod2geom", VERSION)


def utm32_from_wgs(lat, lon):
    phi = np.radians(np.asarray(lat, np.float64))
    dl = np.radians(np.asarray(lon, np.float64) - _LON0)
    s = dnp.sin(phi)
    tau = dnp.sinh(dnp.arctanh(s) - _E * dnp.arctanh(_E * s))
    xi0 = dnp.arctan2(tau, dnp.cos(dl))
    eta0 = dnp.arcsinh(dnp.sin(dl) / np.sqrt(tau * tau + dnp.sq(dnp.cos(dl))))
    xi, eta = xi0, eta0
    for j, a in enumerate(_AL, 1):
        xi = xi + a * dnp.sin(2 * j * xi0) * dnp.cosh(2 * j * eta0)
        eta = eta + a * dnp.cos(2 * j * xi0) * dnp.sinh(2 * j * eta0)
    return _FE + _K0 * _A0 * eta, _K0 * _A0 * xi


def wgs_from_utm32(E, N):
    xi0 = np.asarray(N, np.float64) / (_K0 * _A0)
    eta0 = (np.asarray(E, np.float64) - _FE) / (_K0 * _A0)
    xi, eta = xi0, eta0
    for j, b in enumerate(_BE, 1):
        xi = xi - b * dnp.sin(2 * j * xi0) * dnp.cosh(2 * j * eta0)
        eta = eta - b * dnp.cos(2 * j * xi0) * dnp.sinh(2 * j * eta0)
    tau = dnp.sin(xi) / np.sqrt(dnp.sq(dnp.sinh(eta)) + dnp.sq(dnp.cos(xi)))
    dl = dnp.arctan2(dnp.sinh(eta), dnp.cos(xi))
    psi = dnp.arcsinh(tau)
    phi = dnp.arcsin(dnp.tanh(psi))
    for _ in range(8):      # fixed point of atanh(sin phi) - e atanh(e sin phi) = psi
        phi = dnp.arcsin(dnp.tanh(psi + _E * dnp.arctanh(_E * dnp.sin(phi))))
    return np.degrees(phi), _LON0 + np.degrees(dl)


def zip_tile(E, N):
    """(e_km, n_km) of the zip whose UTM32 square holds (E, N) in metres."""
    return 2 * math.floor((E / 1000 - 1) / 2) + 1, 2 * math.floor(N / 2000)


def zip_tiles_over(e0, n0, e1, n1):
    """Sorted (e_km, n_km) of every zip whose square touches the UTM rectangle."""
    e0, e1 = min(e0, e1), max(e0, e1)
    n0, n1 = min(n0, n1), max(n0, n1)
    out = []
    for e in range(zip_tile(e0, 0)[0], zip_tile(e1, 0)[0] + 1, 2):
        for n in range(zip_tile(0, n0)[1], zip_tile(0, n1)[1] + 1, 2):
            out.append((e, n))
    return out


def _tile_text(el):
    return el.tag.rsplit("}", 1)[-1]


def parse_gml(fileobj):
    """Units of one CityGML file: every Building or BuildingPart that has boundedBy surfaces of its own."""
    frames, surf, stack = [], [], []
    outer, interior = None, 0
    for ev, el in ET.iterparse(fileobj, events=("start", "end")):
        t = _tile_text(el)
        if ev == "start":
            stack.append(t)
            if t in ("Building", "BuildingPart"):
                gid = el.get(GID)
                if not frames:
                    outer = gid
                frames.append({"id": gid, "bldg": outer, "code": 0, "polys": []})
            elif t in _SURF:
                surf.append(_SURF[t])
            elif t == "interior":
                interior += 1
            continue
        if t == "roofType" and frames:
            try:
                frames[-1]["code"] = int(el.text.strip())
            except (AttributeError, TypeError, ValueError):
                pass
        elif (t == "posList" and frames and surf and surf[-1] is not None and interior == 0
              and len(stack) >= 2 and stack[-2] == "LinearRing"):
            pts = np.array(el.text.split(), np.float64).reshape(-1, 3)[:-1]     # the last point closes the ring
            if len(pts) >= 3:
                frames[-1]["polys"].append((surf[-1], pts))
        elif t in ("Building", "BuildingPart"):
            f = frames.pop()
            if f["polys"]:
                yield {"id": f["id"], "bldg": f["bldg"], "code": f["code"], "polys": f["polys"]}
            if not frames:
                el.clear()          # the outermost building is done: free its tree
        elif t in _SURF:
            surf.pop()
        elif t == "interior":
            interior -= 1
        elif t == "cityObjectMember":
            el.clear()
        stack.pop()


def prepare(zip_path, out_npz):
    """Units of all building GML files in a LoD2 zip -> out_npz (atomic). Returns the number of units."""
    ids, bldgs, codes, cen_e, cen_n = [], [], [], [], []
    pu, pk, plen, verts = [], [], [], []
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(m for m in z.namelist() if m.lower().endswith(".gml") and "meta" not in m.lower())
        for name in names:
            with z.open(name) as f:
                for u in parse_gml(f):
                    j = len(ids)
                    ground = [p for k, p in u["polys"] if k == 0]
                    g = np.concatenate(ground) if ground else np.concatenate([p for _, p in u["polys"]])
                    base = float(g[:, 2].min())
                    ids.append(u["id"])
                    bldgs.append(u["bldg"])
                    codes.append(u["code"])
                    cen_e.append(float(g[:, 0].mean()))
                    cen_n.append(float(g[:, 1].mean()))
                    for k, p in u["polys"]:
                        pu.append(j)
                        pk.append(k)
                        plen.append(len(p))
                        verts.append(np.column_stack([p[:, 0], p[:, 1], p[:, 2] - base]))
    V = np.concatenate(verts) if verts else np.zeros((0, 3))
    vlat, vlon = wgs_from_utm32(V[:, 0], V[:, 1])
    ulat, ulon = wgs_from_utm32(np.array(cen_e, np.float64), np.array(cen_n, np.float64))
    arrays = dict(
        unit_id=np.array(ids, str), unit_bldg=np.array(bldgs, str), unit_code=np.array(codes, np.int32),
        unit_lat=np.asarray(ulat, np.float64), unit_lon=np.asarray(ulon, np.float64),
        poly_unit=np.array(pu, np.int32), poly_kind=np.array(pk, np.int8),
        poly_start=np.concatenate([[0], np.cumsum(np.array(plen, np.int64))]).astype(np.int64),
        vlat=np.asarray(vlat, np.float64), vlon=np.asarray(vlon, np.float64), vz=V[:, 2].astype(np.float32))
    os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
    tmp = f"{out_npz}.{os.getpid()}-{threading.get_ident()}.tmp.npz"     # (.npz last: savez adds one otherwise)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, out_npz)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return len(ids)


def _signed(x, y):
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _hull(pts):
    """Convex hull (monotone chain), counter-clockwise."""
    p = sorted(set(map(tuple, pts.tolist())))
    if len(p) < 3:
        return np.array(p, np.float64).reshape(-1, 2)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1], np.float64)


def _cover(ring, area, boxes):
    """Share of the ring's sample points (a grid of ~400 points over it) that lie in one of the boxes' outlines
    (boxes: (e0, e1, n0, n1, Path) each). The ring's vertices stand in for a sliver with no grid point inside."""
    e0, n0 = float(ring[:, 0].min()), float(ring[:, 1].min())
    e1, n1 = float(ring[:, 0].max()), float(ring[:, 1].max())
    step = max(1.0, math.sqrt(area / 400.0))
    g_e, g_n = np.meshgrid(np.arange(e0 + step / 2, e1, step), np.arange(n0 + step / 2, n1, step))
    pts = np.stack([g_e.ravel(), g_n.ravel()], 1)
    if len(pts):
        pts = pts[Path(ring).contains_points(pts)]
    if not len(pts):
        pts = ring
    hit = np.zeros(len(pts), bool)
    for b0, b1, c0, c1, path in boxes:
        if b1 < e0 or b0 > e1 or c1 < n0 or c0 > n1:
            continue
        hit |= path.contains_points(pts)
    return float(hit.mean()) if len(pts) else 0.0


class Geoms:
    _tiles = {}                  # (path, mtime) -> arrays, oldest first
    _missing = set()
    _bad = set()                 # (path, mtime) of tiles that did not read: logged once
    _lock = threading.Lock()
    _fetch_locks = {}            # path -> its download's lock (one download of a tile at a time in a process)
    _fetch_failed = {}           # path -> time.monotonic() of the failed download, None: the LGL has no such tile

    @classmethod
    def _fetch(cls, fetcher, e, n, path):
        """Download the zip of tile (e, n) with fetcher (netfetch) and prepare it as path; True when path is there."""
        import time
        import netfetch
        with cls._lock:
            lk = cls._fetch_locks.setdefault(path, threading.Lock())
        with lk:
            if os.path.exists(path):        # (another thread had it meanwhile)
                return True
            with cls._lock:
                if path in cls._fetch_failed:
                    at = cls._fetch_failed[path]
                    if at is None or time.monotonic() - at < FETCH_RETRY_S:
                        return False
            url = URL.format(e=e, n=n)
            tmp = f"{path}.{os.getpid()}-{threading.get_ident()}.zip"
            try:
                body = fetcher.get(url, kind="demand").body
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                with open(tmp, "wb") as fh:
                    fh.write(body)
                units = prepare(tmp, path)
                log.info("LoD2 geometry tile %s_%s: %d units from %s", e, n, units, url)
                return True
            except netfetch.NotFound:
                with cls._lock:
                    cls._fetch_failed[path] = None
                log.info("LoD2 geometry tile %s_%s: the LGL has none", e, n)
            except Exception as err:       # (offline, cut short, a portal page: the footprints meanwhile)
                with cls._lock:
                    cls._fetch_failed[path] = time.monotonic()
                log.warning("LoD2 geometry tile %s_%s not had: %s: %s", e, n, type(err).__name__, err)
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            return False

    def __init__(self, frame, tiles, ti, ui, cx, cy):
        self.frame, self.tiles = frame, tiles
        self.ti, self.ui = ti, ui
        self.cx, self.cy = cx, cy
        self.n = len(ui)
        self._units = {}
        self._taken = set()
        self._grid = {}
        for i, (x, y) in enumerate(zip(cx.tolist(), cy.tolist())):
            self._grid.setdefault((math.floor(x / GRID_M), math.floor(y / GRID_M)), []).append(i)

    @classmethod
    def _tile(cls, path):
        key = (path, os.path.getmtime(path))
        with cls._lock:
            d = cls._tiles.get(key)
        if d is None:
            try:
                with np.load(path) as z:
                    d = {k: z[k] for k in z.files}
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile, zlib.error) as e:
                with cls._lock:
                    first = key not in cls._bad
                    cls._bad.add(key)
                if first:
                    log.warning("LoD2 geometry tile unreadable: %s (%s)", path, e)
                return None
            with cls._lock:
                cls._tiles[key] = d
                while len(cls._tiles) > MAX_TILES:
                    cls._tiles.pop(next(iter(cls._tiles)))
        return d

    @classmethod
    def load(cls, cache_dir, frame, x0, y0, x1, y1, margin=40.0, fetcher=None):
        """The units around the box (ENU, margin m more) from the tiles on disk; with a fetcher (and FETCH) a tile not
        there is downloaded and prepared first. None: outside BW, off, or no tile had."""
        if not ON:
            return None
        x0, x1 = min(x0, x1) - margin, max(x0, x1) + margin
        y0, y1 = min(y0, y1) - margin, max(y0, y1) + margin
        lat, lon, _ = frame.enu_to_geodetic(np.array([(x0 + x1) / 2]), np.array([(y0 + y1) / 2]), 0.0)
        if not (BW[0] <= float(lat[0]) <= BW[2] and BW[1] <= float(lon[0]) <= BW[3]):
            return None
        lat, lon, _ = frame.enu_to_geodetic(np.array([x0, x1, x1, x0]), np.array([y0, y0, y1, y1]), 0.0)
        E, N = utm32_from_wgs(lat, lon)
        tiles, ti, ui, cx, cy, loaded = [], [], [], [], [], 0
        for e, n in zip_tiles_over(float(E.min()), float(N.min()), float(E.max()), float(N.max())):
            path = os.path.join(folder(cache_dir), f"{e}_{n}.npz")
            if not os.path.exists(path) and not (fetcher is not None and FETCH and cls._fetch(fetcher, e, n, path)):
                if path not in cls._missing:
                    cls._missing.add(path)
                    log.info("LoD2 geometry tile missing: %s", path)
                continue
            d = cls._tile(path)
            if d is None:
                continue
            loaded += 1
            ce, cn, _ = frame.geodetic_to_enu(d["unit_lat"], d["unit_lon"], 0.0)
            keep = np.nonzero((ce >= x0) & (ce <= x1) & (cn >= y0) & (cn <= y1))[0]
            tiles.append(d)
            ti += [len(tiles) - 1] * len(keep)
            ui += keep.tolist()
            cx += ce[keep].tolist()
            cy += cn[keep].tolist()
        if not loaded:
            return None
        return cls(frame, tiles, np.array(ti, np.int64), np.array(ui, np.int64),
                   np.array(cx, np.float64), np.array(cy, np.float64))

    def unit(self, i):
        u = self._units.get(i)
        if u is None:
            u = self._units[i] = self._build(i)
        return u

    def _build(self, i):
        d, j = self.tiles[self.ti[i]], int(self.ui[i])
        p0 = int(np.searchsorted(d["poly_unit"], j, "left"))
        p1 = int(np.searchsorted(d["poly_unit"], j, "right"))
        v0, v1 = int(d["poly_start"][p0]), int(d["poly_start"][p1])
        e, n, _ = self.frame.geodetic_to_enu(d["vlat"][v0:v1], d["vlon"][v0:v1], 0.0)
        V = np.stack([np.asarray(e, np.float64), np.asarray(n, np.float64), d["vz"][v0:v1].astype(np.float64)], 1)
        starts = d["poly_start"][p0:p1 + 1] - v0
        polys = [(int(d["poly_kind"][p0 + k]), V[starts[k]:starts[k + 1]]) for k in range(p1 - p0)]
        ground = [a[:, :2] for kk, a in polys if kk == 0 and len(a) >= 3]
        if ground:
            areas = [abs(_signed(a[:, 0], a[:, 1])) for a in ground]
            ring = ground[int(np.argmax(areas))]
            area = float(sum(areas))
        else:
            ring = _hull(V[:, :2])
            area = abs(_signed(ring[:, 0], ring[:, 1])) if len(ring) >= 3 else 0.0
        if len(ring) >= 3 and _signed(ring[:, 0], ring[:, 1]) < 0:
            ring = ring[::-1]
        top = float(V[:, 2].max())
        roof = [a[:, 2] for kk, a in polys if kk == 2]
        eave = float(min(r.min() for r in roof)) if roof else top
        return {"ring": ring, "area": area, "cx": float(self.cx[i]), "cy": float(self.cy[i]),
                "polys": polys, "code": int(d["unit_code"][j]), "top": top, "eave": eave,
                "id": str(d["unit_id"][j]), "bldg": str(d["unit_bldg"][j])}

    def take(self, e, n, area_ratio=(0.45, 2.0)):
        """Units standing in the footprint ring (e, n) whose footprint area sums to area_ratio of the ring; they are
        marked as taken. Returns their sorted indices, or [] (nothing marked)."""
        e, n = np.asarray(e, np.float64), np.asarray(n, np.float64)
        if len(e) > 1 and e[0] == e[-1] and n[0] == n[-1]:
            e, n = e[:-1], n[:-1]
        if len(e) < 3 or self.n == 0:
            return []
        ring_area = abs(_signed(e, n))
        if ring_area <= 0:
            return []
        gx0, gx1 = math.floor(e.min() / GRID_M), math.floor(e.max() / GRID_M)
        gy0, gy1 = math.floor(n.min() / GRID_M), math.floor(n.max() / GRID_M)
        cand = [i for gx in range(gx0, gx1 + 1) for gy in range(gy0, gy1 + 1)
                for i in self._grid.get((gx, gy), ()) if i not in self._taken]
        if not cand:
            return []
        ii = np.array(cand, np.int64)
        inside = Path(np.stack([e, n], 1)).contains_points(np.stack([self.cx[ii], self.cy[ii]], 1))
        ii = ii[inside]
        if not len(ii):
            return []
        total = sum(self.unit(int(i))["area"] for i in ii)
        if not (area_ratio[0] <= total / ring_area <= area_ratio[1]):
            return []
        self._taken.update(int(i) for i in ii)
        return sorted(int(i) for i in ii)

    def leftover(self, x0, y0, x1, y1, outlines, own=None, max_cover=0.3, min_area=12.0):
        """Units of this tile (centre in it: own, else the box [x0, x1) x [y0, y1)) that no footprint covers: fewer than
        max_cover of their ring lies in the outlines (ENU (e, n) arrays), and at least min_area m2. Marked as taken;
        returns their sorted indices. [] when GAPS is off."""
        if not GAPS or self.n == 0:
            return []
        cx, cy = self.cx, self.cy
        mine = np.asarray(own.owns(cx, cy), bool) if own is not None else (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
        cand = [int(i) for i in np.flatnonzero(mine) if int(i) not in self._taken]
        boxes = []
        for e, n in outlines:
            e, n = np.asarray(e, np.float64), np.asarray(n, np.float64)
            if len(e) >= 3:
                boxes.append((float(e.min()), float(e.max()), float(n.min()), float(n.max()),
                              Path(np.stack([e, n], 1))))
        out = []
        for i in cand:
            u = self.unit(i)
            ring = np.asarray(u["ring"], np.float64)
            if u["area"] < min_area or len(ring) < 3:
                continue
            if _cover(ring, u["area"], boxes) >= max_cover:
                continue
            self._taken.add(i)
            out.append(i)
        return sorted(out)
