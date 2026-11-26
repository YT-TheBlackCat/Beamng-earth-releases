"""Ground classes of Leonberg read from the state's orthophoto (DOP20 RGBI, 20 cm, with the infrared band) and the object
height model (nDOM1), baked once by tools/bake_ground_classes.py (tools/groundbake.py has the classifier) into
server/data/leonberg_ground.npz: one class per 0.8 m cell and the paved areas' outlines. This module is what the server reads of it (numpy only).

Why: the map says little about the ground between the roads and the houses (squares, footways, parking, field tracks,
meadows, fields), and WorldCover's 10 m pixels cannot tell a paved square from a lawn. The photo can: infrared and
height separate plants from hard ground, brightness and warmth separate light paving stones from asphalt. The map still
wins where it says something (a mapped surface=* tag, a road, a square): the consumers read this only for what the map
leaves open.

Consumers (each switchable, see ON):
  surface.classify        the physics ground layers per 2 m sample (`layers`)
  plazas.mesh             crisp paving where the photo shows a paved square or path (`paving_rings`)
  roads.road_specs        the surface of a track / service road no tag describes (`road_verdict`)

The data is a derived work of "Digitales Orthophoto 20 cm" and "nDOM1", (c) LGL BW, www.lgl-bw.de, dl-de/by-2-0, changed.
"""
import logging
import math
import os
import threading

import numpy as np

log = logging.getLogger("earthd.groundclass")

ON = os.environ.get("BNE_GROUND_CLASSES", "on").lower() not in ("off", "0", "false", "no")
# one switch per consumer (the physics layers, the paved meshes, the road verdicts), all behind ON
LAYERS_ON = os.environ.get("BNE_GROUND_LAYERS", "on").lower() not in ("off", "0", "false", "no")
PAVING_ON = os.environ.get("BNE_GROUND_PAVING", "on").lower() not in ("off", "0", "false", "no")
ROADS_ON = os.environ.get("BNE_GROUND_ROADS", "on").lower() not in ("off", "0", "false", "no")

VERSION = 1
BAKE_CELL_M = 0.4                 # m per cell the classifier works at (the outlines are traced on it)
CELL_M = 0.8                      # m per cell of the stored raster (every second cell of the classified one)
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "leonberg_ground.npz")

# class codes of the baked raster
NONE, ASPHALT, STONE, RED, GRAVEL, SOIL, GRASS, SHRUB, TREE, OBJECT = range(10)
NAMES = ("none", "asphalt", "stone", "red", "gravel", "soil", "grass", "shrub", "tree", "object")
HARD = (ASPHALT, STONE, RED, GRAVEL)


def _utm32(lat, lon):
    import lod2geom
    e, n = lod2geom.utm32_from_wgs(np.asarray(lat, np.float64), np.asarray(lon, np.float64))
    return np.asarray(e, np.float64), np.asarray(n, np.float64)


class GroundMap:
    """The baked class raster: classes (rows north to south, columns west to east) of the box whose lower left corner
    is (e0, n0) in UTM 32N metres, CELL_M per cell; rings: the paved areas' outlines (see paving_rings)."""

    def __init__(self, classes, e0, n0, rings=None, ring_class=None, ring_off=None, holes=None):
        self.classes = np.asarray(classes, np.uint8)
        self.e0, self.n0 = float(e0), float(n0)
        self.h, self.w = self.classes.shape
        self.rings = rings                  # (k, 2) float32 metres east / north of (e0, n0), all rings concatenated
        self.ring_class = ring_class        # (m,) class of each ring
        self.ring_off = ring_off            # (m + 1,) first point of each ring in rings
        self.ring_hole = holes              # (m,) outer ring index of a hole, -1 for an outer ring
        self._boxes = None                  # (m, 4) e0, n0, e1, n1 of each outer ring (UTM), built when asked
        self._holes_of = None               # outer ring index -> [hole ring indices]

    def contains(self, e, n):
        return (e >= self.e0) & (e < self.e0 + self.w * CELL_M) & (n > self.n0) & (n <= self.n0 + self.h * CELL_M)

    def at(self, e, n):
        """Class codes at UTM points (arrays); NONE outside the box."""
        e, n = np.asarray(e, np.float64), np.asarray(n, np.float64)
        ok = self.contains(e, n)
        c = np.clip(np.floor((e - self.e0) / CELL_M).astype(np.int64), 0, self.w - 1)
        r = np.clip(np.floor((self.n0 + self.h * CELL_M - n) / CELL_M).astype(np.int64), 0, self.h - 1)
        return np.where(ok, self.classes[r, c], np.uint8(NONE))

    def _ring(self, i):
        a, b = int(self.ring_off[i]), int(self.ring_off[i + 1])
        r = self.rings[a:b].astype(np.float64)
        return r + np.array([self.e0, self.n0])

    def _index(self):
        if self._boxes is None:
            m = 0 if self.ring_class is None else len(self.ring_class)
            boxes = np.zeros((m, 4))
            holes = {}
            for i in range(m):
                r = self._ring(i)
                boxes[i] = (r[:, 0].min(), r[:, 1].min(), r[:, 0].max(), r[:, 1].max())
                if int(self.ring_hole[i]) >= 0:
                    holes.setdefault(int(self.ring_hole[i]), []).append(i)
            self._boxes, self._holes_of = boxes, holes

    def rings_in(self, e0, n0, e1, n1, classes=None):
        """[(outer ring (k, 2) UTM metres, [hole rings], class)] of the paved areas whose outline box meets the UTM box
        (e0, n0, e1, n1), of classes (default all with an outline)."""
        if self.ring_class is None or not len(self.ring_class):
            return []
        self._index()
        b = self._boxes
        sel = (self.ring_hole < 0) & (b[:, 2] >= e0) & (b[:, 0] <= e1) & (b[:, 3] >= n0) & (b[:, 1] <= n1)
        if classes is not None:
            sel &= np.isin(self.ring_class, np.asarray(classes, np.uint8))
        return [(self._ring(i), [self._ring(h) for h in self._holes_of.get(i, [])], int(self.ring_class[i]))
                for i in np.flatnonzero(sel)]

    def share_in(self, e, n, radius_m, classes):
        """Share (0..1) of the cells within radius_m of each UTM point (a box of that half width) that are in classes."""
        e, n = np.atleast_1d(np.asarray(e, np.float64)), np.atleast_1d(np.asarray(n, np.float64))
        k = max(1, int(round(radius_m / CELL_M)))
        offs = np.arange(-k, k + 1)
        out = np.zeros(len(e))
        sel = np.isin(self.classes, np.asarray(classes, np.uint8))
        for i in range(len(e)):
            c = int(math.floor((e[i] - self.e0) / CELL_M))
            r = int(math.floor((self.n0 + self.h * CELL_M - n[i]) / CELL_M))
            rr, cc = np.clip(r + offs, 0, self.h - 1), np.clip(c + offs, 0, self.w - 1)
            out[i] = float(sel[np.ix_(rr, cc)].mean())
        return out


_LOCK = threading.Lock()
_MAP = {}


def load(path=None):
    """The baked GroundMap, or None (switched off, not baked, unreadable). Cached for the process."""
    if not ON:
        return None
    path = path or DATA
    with _LOCK:
        if path in _MAP:
            return _MAP[path]
        gm = None
        try:
            if os.path.exists(path):
                with np.load(path) as z:
                    if int(z["version"]) == VERSION:
                        gm = GroundMap(z["classes"], float(z["e0"]), float(z["n0"]), z["rings"], z["ring_class"],
                                       z["ring_off"], z["ring_hole"])
        except (OSError, ValueError, KeyError) as ex:
            log.warning("ground classes not read (%s): %s", path, ex)
        _MAP[path] = gm
        return gm


def reset():
    """Forget the loaded map (tests)."""
    with _LOCK:
        _MAP.clear()


# ---- what the server takes from it -------------------------------------------------------------------------------------

TAG = "gc1"                       # in the cache keys of what is built from the classes (roads.road_specs): bump with a rule


def active():
    """Is there a baked map (and is it switched on)?"""
    return load() is not None


def _bounds(gm):
    """(lat0, lon0, lat1, lon1) of the map's box (the corners; the box is small and the meridian convergence tiny)."""
    import lod2geom
    e = np.array([gm.e0, gm.e0 + gm.w * CELL_M])
    n = np.array([gm.n0, gm.n0 + gm.h * CELL_M])
    la, lo = lod2geom.wgs_from_utm32(np.array([e[0], e[1], e[0], e[1]]), np.array([n[0], n[0], n[1], n[1]]))
    return float(np.min(la)), float(np.min(lo)), float(np.max(la)), float(np.max(lo))


def _utm_grid(lat, lon, k=9):
    """UTM 32N (E, N) of the sample grids lat, lon (H, W): worked out on a k x k net of samples and interpolated
    bilinearly (a block is about a kilometre: the error is far below a centimetre), exactly for a small grid."""
    lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    h, w = lat.shape
    if h * w <= 4096 or min(h, w) < 3:
        return _utm32(lat, lon)
    ri = np.unique(np.linspace(0, h - 1, k).round().astype(np.int64))
    ci = np.unique(np.linspace(0, w - 1, k).round().astype(np.int64))
    ce, cn = _utm32(lat[np.ix_(ri, ci)], lon[np.ix_(ri, ci)])
    fr = np.interp(np.arange(h), ri, np.arange(len(ri)))
    fc = np.interp(np.arange(w), ci, np.arange(len(ci)))
    i0 = np.minimum(fr.astype(np.int64), len(ri) - 2)
    j0 = np.minimum(fc.astype(np.int64), len(ci) - 2)
    u = (fr - i0)[:, None]
    v = (fc - j0)[None, :]

    def lerp(a):
        return (a[np.ix_(i0, j0)] * (1 - u) * (1 - v) + a[np.ix_(i0 + 1, j0)] * u * (1 - v)
                + a[np.ix_(i0, j0 + 1)] * (1 - u) * v + a[np.ix_(i0 + 1, j0 + 1)] * u * v)
    return lerp(ce), lerp(cn)


def apply_layers(idx, classes, lat, lon, layer):
    """idx (material per 2 m sample) with the photo's ground classes painted: layer {class code: material index}. Where
    land cover says built-up (WorldCover class 50) every class in layer is painted (a lawn, a yard's gravel or asphalt,
    a bare patch); paving stone and red paving also where it says otherwise (a small square the 10 m pixels miss).
    Pure given the baked map."""
    gm = load() if LAYERS_ON else None
    if gm is None or not layer:
        return idx
    la0, lo0, la1, lo1 = _bounds(gm)
    lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    if lat.max() < la0 or lat.min() > la1 or lon.max() < lo0 or lon.min() > lo1:
        return idx
    e, n = _utm_grid(lat, lon)
    cls = gm.at(e, n)
    built = np.asarray(classes) == 50
    out = idx.copy()
    for code, material in layer.items():
        sel = (cls == code) & (built | np.isin(code, (STONE, RED)))
        out[sel] = material
    return out


# a stone area inside the old town wall is natural stone setts (europewalks.OLD_TOWNS), elsewhere flat paving stones
RING_SURFACE = {STONE: "paving_stones", RED: "bricks"}
PHOTO_PREFIX = "photo:"           # in plazas.mesh a surface value with this prefix is an area read from the photo: the
                                  # mapped squares keep the ground they share with it


def paving_rings(lat0, lon0, lat1, lon1):
    """[(outer ring lon / lat (k, 2), [hole rings], "photo:<surface>")] of the paved areas (stone, red paving) the photo
    shows within the lat / lon box: what plazas.mesh paves as it paves a mapped square."""
    gm = load() if PAVING_ON else None
    if gm is None or gm.ring_class is None or not len(gm.ring_class):
        return []
    import lod2geom
    import europewalks
    e, n = _utm32(np.array([lat0, lat0, lat1, lat1]), np.array([lon0, lon1, lon0, lon1]))
    out = []
    for outer, holes, c in gm.rings_in(float(e.min()), float(n.min()), float(e.max()), float(n.max()),
                                       classes=tuple(RING_SURFACE)):
        def ll(r):
            la, lo = lod2geom.wgs_from_utm32(r[:, 0], r[:, 1])
            return np.stack([lo, la], 1)
        ring = ll(outer)
        surface = RING_SURFACE[c]
        if c == STONE and europewalks.in_old_town(float(ring[:, 0].mean()), float(ring[:, 1].mean())):
            surface = "sett"
        out.append((ring, [ll(h) for h in holes], PHOTO_PREFIX + surface))
    return out


# a road no map tag describes (track, service, unclassified): what the photo shows along its centre line
ROAD_CLASSES = ("track", "service", "unclassified")
ROAD_OFFSETS = (-0.8, 0.0, 0.8)
ROAD_MIN_KNOWN = 12               # samples on a known ground class (not a tree, a roof, off the map)
ROAD_KNOWN_SHARE = 0.5            # ... of all
ROAD_SHARE = 0.55                 # of the known ones for a verdict
# a service or unclassified road is paved unless the map says otherwise (a yard's lane, its edge in a building's
# shade, a bit of moss, read beige): more proof before it turns to gravel or grass
STRICT_CLASSES = ("service", "unclassified")
STRICT_SHARE = 0.8
STRICT_MIN_KNOWN = 25
PAVED = (ASPHALT, STONE, RED)


def road_verdicts(feats, surface_of):
    """{feature index: "asphalt", "gravel" or "grass"} for the roads of feats (a roads layer) of ROAD_CLASSES with no
    map surface (surface_of(i) None) the photo shows paved, bare and light (gravel) or as green as a meadow. A road
    the photo cannot tell (under trees, off the baked box) keeps the rule it had."""
    gm = load() if ROADS_ON else None
    cls = feats.attrs.get("class") if gm is not None else None
    if cls is None:
        return {}
    import photosurface
    out = {}
    for i in range(len(feats)):
        if str(cls[i]) not in ROAD_CLASSES or surface_of(i) is not None:
            continue
        best = None
        for part in feats.parts(i):
            for ring in part:
                ll, nrm = photosurface._along(ring)
                if not len(ll):
                    continue
                pts = np.concatenate([ll + nrm * o for o in ROAD_OFFSETS])
                e, n = _utm32(pts[:, 1], pts[:, 0])
                c = gm.at(e, n)
                known = np.isin(c, PAVED + (GRAVEL, SOIL, GRASS))
                strict = str(cls[i]) in STRICT_CLASSES
                share = STRICT_SHARE if strict else ROAD_SHARE
                if known.sum() >= (STRICT_MIN_KNOWN if strict else ROAD_MIN_KNOWN) and known.mean() >= ROAD_KNOWN_SHARE:
                    k = c[known]
                    v = None
                    if np.isin(k, PAVED).mean() >= ROAD_SHARE:
                        v = "asphalt"
                    elif np.isin(k, (GRAVEL, SOIL)).mean() >= share:
                        v = "gravel"
                    elif (k == GRASS).mean() >= share:
                        v = "grass"
                    if v is not None and (best is None or known.sum() > best[0]):
                        best = (int(known.sum()), v)
        if best is not None:
            out[i] = best[1]
    return out


def digest(la0=None, la1=None, lo0=None, lo1=None):
    """A short text for the cache keys of what is built from the classes in a lat / lon box: empty when they play no
    part there (switched off, not baked, box off the map), else the tag, the switches and the baked file's stamp."""
    gm = load()
    if gm is None:
        return ""
    if la0 is not None:
        b = _bounds(gm)
        if max(la0, la1) < b[0] or min(la0, la1) > b[2] or max(lo0, lo1) < b[1] or min(lo0, lo1) > b[3]:
            return ""
    try:
        st = os.stat(DATA)
        stamp = f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        stamp = ""
    return f"{TAG}:{int(LAYERS_ON)}{int(PAVING_ON)}{int(ROADS_ON)}:{stamp}"


def roads_active():
    """Do the road verdicts (road_verdicts) take part?"""
    return ROADS_ON and load() is not None
