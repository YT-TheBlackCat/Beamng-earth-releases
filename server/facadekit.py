"""Facade kit (Fassaden-Bausteine, 2026-11-25): houses dressed from parts that fit the real house.

The walls and roofs of Leonberg's houses are measured (LoD2, buildings._lod2_draw), their windows were not: every
house had the same window sheet laid on its walls at one bay width, party walls included. Here a house gets its windows,
sills, surrounds, shutters, roller-shutter boxes, balconies, a canopy over the front door and gable windows from rules
that read what the data says about it:
  * every wall's real length (the LoD2 outline): window axes fitted to it, kept aligned from floor to floor, one of
    them on the front door (spec.door);
  * party walls: where a neighbour's wall touches (mark_neighbours: the other footprints of the tile) there are no
    windows (a row house's side walls were full of them);
  * storeys and heights from the LoD2 eave (spec.floors, floor_h), the roof form (flat or pitched), the footprint's
    size, its class (OSM house / apartments), its year where mapped (era), the old town (Fachwerk circle), a mapped
    timber-framed or coloured facade (the walls keep their own material and colour);
  * everything else (which house has shutters, their colour, window pairs, flower boxes, ...) hashed from the
    building's id, the same in every build and tile (buildings._hash01).
The parts are plain quads and boxes in the existing materials (the plaster sheet's window cells and its plain spot, tinted
by vertex colour), no new texture. Near tiles only (lite keeps the sheet's windows), only where the walls are plain:
wall_spec() turns the sheet houses' walls into plain stucco, so the sheet's windows do not show under the kit's.

BNE_FACADE_KIT=off switches it off, =all lifts the area limit (AREA: Leonberg and its surroundings).
"""
import math
import os

import numpy as np

import dmath
import dnp

ON = os.environ.get("BNE_FACADE_KIT", "on").lower() not in ("off", "0", "false", "no")
EVERYWHERE = os.environ.get("BNE_FACADE_KIT", "").lower() == "all"
AREA = (48.80, 9.01, 8000.0)            # lat, lon, radius m: Leonberg, Ramtel, the villages round it
MAX_FLOORS = 6
MIN_AREA_M2, MAX_AREA_M2 = 30.0, 1500.0
MIN_EDGE_M = 2.4                        # a wall shorter than this gets no windows
MAX_WINDOWS = 160                       # per building (triangle budget: ~12 triangles a window with its parts)
NEIGHBOUR_TOL_M = 0.6                   # a neighbour's wall this near a wall (and facing it) is its party wall
NEIGHBOUR_MIN_M = 0.4                   # ... over at least this length

TRIM_UV = ((0.002, 0.995), (0.012, 0.995), (0.012, 0.999), (0.002, 0.999))     # the plaster sheet's plain spot

# colours (the plain spot takes the vertex colour as it is)
SHUTTERS = ((0.27, 0.38, 0.27), (0.33, 0.22, 0.16), (0.45, 0.20, 0.17), (0.84, 0.84, 0.80), (0.30, 0.38, 0.48),
            (0.22, 0.32, 0.24))
STONE_TRIM = (0.80, 0.76, 0.68)
WHITE_TRIM = (0.96, 0.95, 0.92)
TIMBER_TRIM = (0.25, 0.17, 0.12)
SILL = (0.74, 0.72, 0.68)
ROLLER = (0.88, 0.87, 0.84)
CANOPY = (0.30, 0.32, 0.34)
SLAB = (0.70, 0.70, 0.68)
RAILS = ((0.92, 0.92, 0.90), (0.45, 0.50, 0.55), (0.30, 0.34, 0.36), (0.62, 0.34, 0.28))
BOX_WOOD = (0.40, 0.28, 0.20)
BOX_GREEN = (0.28, 0.42, 0.22)

# bay: m between window axes; w, h: a window's width and height (m); the shares are of the buildings (hashed)
FAMILIES = {
    "altstadt": dict(bay=(2.5, 3.2), w=(0.95, 1.15), h=(1.35, 1.6), pair=0.3, shutter=0.45, roller=0.0, surround=0.55,
                     flower=0.12, canopy=0.10, balcony=0.0),
    "siedlung": dict(bay=(3.0, 3.7), w=(1.1, 1.4), h=(1.3, 1.5), pair=0.1, shutter=0.30, roller=0.45, surround=0.20,
                     flower=0.10, canopy=0.50, balcony=0.15),
    "block": dict(bay=(3.0, 3.6), w=(1.2, 1.6), h=(1.3, 1.45), pair=0.0, shutter=0.0, roller=0.6, surround=0.0,
                  flower=0.0, canopy=0.70, balcony=0.70),
    "neubau": dict(bay=(3.2, 4.2), w=(1.5, 2.1), h=(1.5, 2.0), pair=0.0, shutter=0.0, roller=0.3, surround=0.0,
                   flower=0.0, canopy=0.60, balcony=0.50),
}
# The house's epoch (when it was built decides its windows, shutters, balconies and plaster):
#   alt (before 1920), zw (1920-1949), nk (1950-1969), s70 (1970-1989), s90 (1990-2009), neu (2010 on).
# Known from a mapped construction year (OSM start_date, 3DBAG), else guessed from what the data shows (estimate_epoch).
EPOCH_YEARS = ((1920, "alt"), (1950, "zw"), (1970, "nk"), (1990, "s70"), (2010, "s90"))
# overrides of the family's numbers per epoch (the family keeps what is not here)
EPOCH_ORDER = ("alt", "zw", "nk", "s70", "s90", "neu")
EPOCHS = {
    "alt": dict(bay=(2.5, 3.2), w=(0.95, 1.15), h=(1.4, 1.7), pair=0.3, shutter=0.55, roller=0.0, surround=0.6,
                flower=0.12, canopy=0.08, balcony=0.0),
    "zw": dict(bay=(2.7, 3.4), w=(1.0, 1.2), h=(1.3, 1.5), pair=0.2, shutter=0.45, roller=0.05, surround=0.35,
               flower=0.10, canopy=0.25, balcony=0.05),
    "nk": dict(bay=(3.0, 3.6), w=(1.1, 1.3), h=(1.2, 1.4), pair=0.0, shutter=0.15, roller=0.35, surround=0.0,
               flower=0.08, canopy=0.55, balcony=0.35),
    "s70": dict(bay=(3.2, 3.9), w=(1.3, 1.7), h=(1.3, 1.5), pair=0.0, shutter=0.05, roller=0.6, surround=0.0,
                flower=0.05, canopy=0.50, balcony=0.55),
    "s90": dict(bay=(3.2, 4.0), w=(1.3, 1.7), h=(1.4, 1.7), pair=0.0, shutter=0.10, roller=0.45, surround=0.15,
                flower=0.05, canopy=0.60, balcony=0.45),
    "neu": dict(bay=(3.2, 4.2), w=(1.5, 2.2), h=(1.6, 2.2), pair=0.0, shutter=0.0, roller=0.25, surround=0.0,
                flower=0.0, canopy=0.60, balcony=0.55),
}
# plaster by epoch where no colour is mapped: (the colour the epoch's plaster leans to, how far the sheet's tint goes there)
# (2026-11-25: much lighter and nearer the epoch's colour; the sheet's own tints (olive, brown, grey) made the new districts dull)
EPOCH_PLASTER = {"nk": ((0.95, 0.92, 0.84), 0.70), "s70": ((0.94, 0.90, 0.80), 0.70),
                 "s90": ((0.96, 0.94, 0.90), 0.80), "neu": ((0.97, 0.97, 0.95), 0.90)}
# The kit (single windows, shutters, balconies) dresses only the old houses; a house of 1950 or later keeps the sheet's solid
# window bands (the base as before the kit, 2026-11-25), in the epoch's light plaster. BNE_FACADE_KIT_NEW=on: the kit on them too.
KIT_EPOCHS = ("alt", "zw")
KIT_NEW = os.environ.get("BNE_FACADE_KIT_NEW", "off").lower() in ("on", "1", "true", "yes")
# What is known of a house without a year (a prior for Leonberg, then how its roof, its place, its size and its aerial
# roof colour move the odds): epoch -> weight.
EPOCH_PRIOR = {"alt": 0.06, "zw": 0.08, "nk": 0.22, "s70": 0.24, "s90": 0.20, "neu": 0.20}
EPOCH_OLD_TOWN = {"alt": 20.0, "zw": 3.0, "nk": 0.6, "s70": 0.4, "s90": 0.5, "neu": 0.5}
EPOCH_ROOF = {                                  # by the main roof's pitch: steep >= 40 deg, mid >= 25, low >= 10
    "steep": {"alt": 2.5, "zw": 2.5, "nk": 1.2, "s70": 0.7, "s90": 0.5, "neu": 0.2},
    "mid": {"alt": 0.7, "zw": 1.2, "nk": 1.6, "s70": 1.4, "s90": 1.2, "neu": 0.7},
    "low": {"alt": 0.2, "zw": 0.4, "nk": 1.2, "s70": 1.8, "s90": 0.8, "neu": 1.0},
    "flat": {"alt": 0.1, "zw": 0.2, "nk": 1.0, "s70": 2.0, "s90": 1.0, "neu": 3.0},
}
EPOCH_BIG = {"alt": 0.5, "zw": 0.7, "nk": 1.5, "s70": 2.5, "s90": 1.2, "neu": 1.5}      # four floors up or over 600 m2
EPOCH_CLAY = {"alt": 1.4, "zw": 1.4, "nk": 1.4, "s70": 1.0, "s90": 0.8, "neu": 0.8}     # red-brown clay-tile roofs
EPOCH_GREY = {"alt": 0.7, "zw": 0.7, "nk": 0.7, "s70": 1.0, "s90": 1.3, "neu": 1.3}     # grey/anthracite roofs
OLD_TOWN_BIG_M2 = 400.0                          # an unmapped big building in the old town is the castle, the town hall...
TAN_STEEP, TAN_MID, TAN_LOW = 0.839, 0.466, 0.176      # tan 40, 25, 10 degrees
WINDOW_BASES = ("bne_bld_stucco", "bne_bld_fachwerk", "bne_bld_stone", "bne_bld_stone_rubble", "bne_bld_siding")


def _b():
    import buildings
    return buildings


def in_area(lat, lon):
    if EVERYWHERE:
        return True
    if lat is None or lon is None:
        return False
    k = dmath.cos(math.radians(AREA[0]))
    return math.hypot((lat - AREA[0]) * 111320.0, (lon - AREA[1]) * 111320.0 * k) <= AREA[2]


def wall_look(spec, ep=None):
    """The plain wall material (a LOOKS name) a sheet house gets under the kit: stucco in the sheet's tint (leaning to the
    epoch's plaster when ep is given), or its own plain wall; None when its walls are not a house's (curtain walls,
    panels, sheds)."""
    b = _b()
    if spec.wall:
        return spec.wall if str(spec.wall).split("|")[0] in WINDOW_BASES else None
    if spec.plain or spec.band is None:
        return None
    if spec.band == b.BAND_BRICK:
        return "bne_brick_red"
    mat, rgb = b.LOOKS.get(spec.facade, (None, None))
    if mat != b.FACADE:
        return None
    return b.custom_look(b.STUCCO, plaster(rgb, ep))


def epoch_of_year(year):
    """The epoch of a construction year (int or text), None when it is not a year."""
    y = _b()._year_of(year)
    if y is None:
        return None
    for lim, name in EPOCH_YEARS:
        if y < lim:
            return name
    return "neu"


def roof_class(spec):
    """"steep", "mid", "low" or "flat": the main roof (the largest LoD2 roof face) of a house; None when unknown."""
    L = getattr(spec, "lod2", None)
    if not L:
        return "flat" if spec.roof == "flat" else None
    if not L.get("pitched"):
        return "flat"
    b = _b()
    best, tan = 0.0, None
    for kind, P in L["polys"]:
        if kind != 2 or len(P) < 3:
            continue
        nv = b._newell(P)
        nh = math.hypot(float(nv[0]), float(nv[1]))
        nz = abs(float(nv[2]))
        area = math.hypot(nh, nz)
        if area > best and (nz > 1e-9 or nh > 1e-9):
            best, tan = area, nh / max(nz, 1e-9)
    if tan is None:
        return None
    return "steep" if tan >= TAN_STEEP else "mid" if tan >= TAN_MID else "low" if tan >= TAN_LOW else "flat"


def estimate_epoch(spec, attrs, old_town, area):
    """A guess at a house's epoch when no year is mapped: a prior, moved by the main roof's pitch, the old town, the size
    and the roof's colour in the aerial photo; one pick by the building's hash (the same in every build)."""
    w = dict(EPOCH_PRIOR)

    def scale(t):
        for k in w:
            w[k] *= t[k]
    if old_town:
        scale(EPOCH_OLD_TOWN)
    rc = roof_class(spec)
    if rc is not None:
        scale(EPOCH_ROOF[rc])
    if int(spec.floors or 0) >= 4 or area > 600.0:
        scale(EPOCH_BIG)
    ph = attrs.get("_roof_photo")
    if ph is not None and not attrs.get("roof_color"):
        r, g, bl = (float(v) for v in ph)
        if r > g * 1.12 and r > bl * 1.25:
            scale(EPOCH_CLAY)
        elif max(r, g, bl) - min(r, g, bl) < 0.07:
            scale(EPOCH_GREY)
    u = _b()._hash01(spec.seed, "kit", "epoch") * sum(w.values())
    for name in EPOCH_ORDER:
        u -= w[name]
        if u <= 0:
            return name
    return "neu"


def plaster(rgb, ep):
    """The sheet's plaster tint leaning to the epoch's plaster (cream in the fifties, ochre in the seventies, white now)."""
    t = EPOCH_PLASTER.get(ep)
    if t is None:
        return tuple(rgb)
    to, k = t
    return tuple(float(rgb[i]) * (1.0 - k) + to[i] * k for i in range(3))


def light_sheet(spec, ep):
    """A sheet house (the walls with the texture's window bands) in the epoch's plaster instead of the style's tint."""
    b = _b()
    if spec.band is None or spec.plain or spec.band == b.BAND_BRICK:
        return
    if spec.wall:
        if str(spec.wall).split("|")[0] not in (b.STONE, b.STONE_RUBBLE):
            return
        spec.wall = None                    # (dressed stone on a house of 1950 or later: plaster, like its neighbours)
        spec.facade = b.custom_look(b.FACADE, plaster((0.95, 0.92, 0.86), ep))
        return
    mat, rgb = b.LOOKS.get(spec.facade, (None, None))
    if mat == b.FACADE:
        spec.facade = b.custom_look(b.FACADE, plaster(rgb, ep))


def decide(spec, attrs, lat, lon, part=False):
    """The kit's plan for a building (a dict kept in spec.kit) or None when it keeps the sheet's windows."""
    if not ON or part or not in_area(lat, lon):
        return None
    b = _b()
    if (spec.min_h or 0.0) > 0.5 or spec.fr_house or spec.holes or spec.deck or spec.steeple or spec.portal \
            or spec.tank or spec.shop or spec.windows is False or spec.skel or spec.ship:
        return None
    if spec.cls in b.SHED_CLASSES or spec.cls in b.CHURCH_CLASSES or spec.cls in b.BARN_CLASSES \
            or spec.cls in b.CANOPY_CLASSES or spec.cls in b.INDUSTRIAL_CLASSES \
            or spec.cls in ("pyramid", "ship", b.landmarks.LANDMARK):
        return None
    if not (1 <= int(spec.floors or 0) <= MAX_FLOORS) or float(spec.height or 0.0) < 2.8:
        return None
    ring = b.drawn_ring(spec)
    area = abs(b._area2(ring)) / 2
    if not (MIN_AREA_M2 <= area <= MAX_AREA_M2):
        return None
    mapped = bool(attrs.get("facade_color") or attrs.get("facade_material"))
    ep0 = epoch_of_year(attrs.get("_year") or attrs.get("_osm_year"))
    cls = str(attrs.get("class") or spec.cls or "").lower()
    flat = spec.roof == "flat"
    in_old = b.in_fachwerk_area(lat, lon)
    if ep0 is None and in_old and area > OLD_TOWN_BIG_M2 and cls != "apartments":
        ep0 = "alt"                                   # (the castle, the town hall: not a 1970s block)
    ep = ep0 or estimate_epoch(spec, attrs, in_old, area)
    if ep not in KIT_EPOCHS and not KIT_NEW and not str(spec.wall or "").startswith(b.FACHWERK):
        if not mapped:
            light_sheet(spec, ep)
        return None
    wall = wall_look(spec, None if mapped else ep)
    if wall is None:
        return None
    if wall.startswith(b.FACHWERK):
        fam, ep = "altstadt", "alt"
    elif ep == "neu" and spec.floors >= 2:
        fam = "neubau"
    elif cls == "apartments" or (ep in ("nk", "s70") and spec.floors >= 3) or (ep == "s90" and spec.floors >= 4):
        fam = "block"
    elif ep in ("alt", "zw") and (in_old or ep == "alt"):
        fam = "altstadt"
    elif area > 500 or (spec.floors >= 3 and flat):
        fam = "block"
    else:
        fam = "siedlung"
    f = dict(FAMILIES[fam], **EPOCHS[ep])
    seed = spec.seed

    def h(k):
        return b._hash01(seed, "kit", k)

    def pick(r, k):
        return r[0] + (r[1] - r[0]) * h(k)
    timber = wall.startswith(b.FACHWERK)
    shutter = SHUTTERS[int(h("sc") * len(SHUTTERS))] if h("sh") < f["shutter"] else None
    roller = shutter is None and h("ro") < f["roller"]
    surround = None
    if timber:
        surround = TIMBER_TRIM
    elif h("su") < f["surround"]:
        surround = STONE_TRIM if h("sk") < 0.6 else WHITE_TRIM
    p = dict(fam=fam, bay=pick(f["bay"], "bay"), w=pick(f["w"], "w"), h=pick(f["h"], "h"), pair=h("pair") < f["pair"],
             shutter=shutter, roller=roller, surround=surround, flower=h("fl") < f["flower"],
             canopy=h("ca") < f["canopy"], balcony=spec.floors >= 2 and h("ba") < f["balcony"],
             rail=RAILS[int(h("rl") * len(RAILS))], flat=flat)
    return {"fam": fam, "wall": wall, "p": p, "nb": None, "ep": ep, "ep_src": "year" if ep0 else "guess"}


def wall_spec(spec, lite):
    """The spec the walls are drawn from: the kit's plain walls without laid-over windows, else spec itself."""
    kit = getattr(spec, "kit", None)
    if not kit or lite:
        return spec
    b = _b()
    return b._spec_copy(spec, wall=kit["wall"], facade=kit["wall"], windows=False, front=None, shop=False,
                        street=None)


def mark_neighbours(items, ring_of):
    """For every kit building of items [(spec, gmin, gmax)] the walls of the other footprints near it (a (k, 4) array of
    x0 y0 x1 y1), what the party walls are found from."""
    rings = []
    for spec, _g0, _g1 in items:
        if (spec.min_h or 0.0) > 0.5:
            rings.append(None)
            continue
        r = np.asarray(ring_of(spec), np.float64)[:, :2]
        rings.append((r, r.min(0), r.max(0)))
    for i, (spec, _g0, _g1) in enumerate(items):
        kit = getattr(spec, "kit", None)
        if not kit or rings[i] is None:
            continue
        r, lo, hi = rings[i]
        segs = []
        for j, other in enumerate(rings):
            if j == i or other is None:
                continue
            r2, lo2, hi2 = other
            if (lo2 > hi + 1.0).any() or (hi2 < lo - 1.0).any():
                continue
            segs.append(np.concatenate([r2, np.roll(r2, -1, 0)], 1))
        kit["nb"] = np.concatenate(segs) if segs else None


def blind_intervals(nb, a, d, length):
    """The stretches (lo, hi) m along the wall a + d s (s in [0, length]) that a neighbour's wall touches."""
    out = []
    if nb is None or not len(nb):
        return out
    a = np.asarray(a, np.float64)
    d = np.asarray(d, np.float64)
    nrm = np.array([-d[1], d[0]])
    p, q = nb[:, :2] - a, nb[:, 2:] - a
    dp, dq = dnp.matmul(p, nrm), dnp.matmul(q, nrm)
    sp, sq = dnp.matmul(p, d), dnp.matmul(q, d)
    dn = q - p
    ln = dnp.hypot(dn[:, 0], dn[:, 1])
    facing = dnp.matmul(dn, d) < -0.8 * np.maximum(ln, 1e-9)         # a shared wall runs the other way round
    ok = facing & (np.abs(dp) <= NEIGHBOUR_TOL_M) & (np.abs(dq) <= NEIGHBOUR_TOL_M)
    for k in np.nonzero(ok)[0]:
        lo, hi = max(0.0, min(sp[k], sq[k])), min(length, max(sp[k], sq[k]))
        if hi - lo >= NEIGHBOUR_MIN_M:
            out.append((float(lo), float(hi)))
    return out


def axes(length, bay, w, margin=0.9, door=None, blind=()):
    """Window centres (m along a wall of this length): bays of about `bay` between the corner margins; where the wall has
    the front door, the bays start from it (one axis on the door, the others a bay apart, all inside the margins);
    none over a blind stretch."""
    free = length - 2 * margin
    if free < w + 0.2:
        return []
    n = max(1, int(free / bay + 0.5))
    sp = free / n
    lo, hi = margin + w / 2, length - margin - w / 2
    if door is not None and lo <= door <= hi:
        k0 = int(math.ceil((lo - door) / sp - 1e-9))
        k1 = int(math.floor((hi - door) / sp + 1e-9))
        cs = [door + k * sp for k in range(k0, k1 + 1)]
    else:
        cs = [margin + sp * (k + 0.5) for k in range(n)]
    return [c for c in cs if not any(c + w / 2 + 0.25 > a and c - w / 2 - 0.25 < b for a, b in blind)]


def _face(m, pts, outward, mat, uv=TRIM_UV):
    """A convex quad, wound counter-clockwise seen from the side `outward` (a 3-vector) points to."""
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = pts[0], pts[1], pts[2]
    ux, uy, uz, vx, vy, vz = bx - ax, by - ay, bz - az, cx - ax, cy - ay, cz - az
    if ((uy * vz - uz * vy) * outward[0] + (uz * vx - ux * vz) * outward[1] + (ux * vy - uy * vx) * outward[2]) < 0:
        pts, uv = pts[::-1], uv[::-1]
    m.quad(*pts, *uv, mat)


class Wall:
    """A wall's frame: origin a, along d, out (outward, horizontal). s along, z up, o out of the wall."""
    __slots__ = ("a", "d", "o")

    def __init__(self, a, d, o):
        self.a, self.d, self.o = (float(a[0]), float(a[1])), (float(d[0]), float(d[1])), (float(o[0]), float(o[1]))

    def pt(self, s, z, off):
        return (self.a[0] + self.d[0] * s + self.o[0] * off, self.a[1] + self.d[1] * s + self.o[1] * off, float(z))

    def quad(self, m, s0, s1, z0, z1, off, mat, uv=TRIM_UV):
        _face(m, [self.pt(s0, z0, off), self.pt(s1, z0, off), self.pt(s1, z1, off), self.pt(s0, z1, off)],
              (self.o[0], self.o[1], 0.0), mat, uv)

    def box(self, m, s0, s1, z0, z1, o0, o1, mat, sides=True, bottom=False):
        """A box standing on the wall: front, top, (sides, bottom)."""
        self.quad(m, s0, s1, z0, z1, o1, mat)
        _face(m, [self.pt(s0, z1, o0), self.pt(s1, z1, o0), self.pt(s1, z1, o1), self.pt(s0, z1, o1)],
              (0.0, 0.0, 1.0), mat)
        if sides:
            for s, sgn in ((s0, -1.0), (s1, 1.0)):
                _face(m, [self.pt(s, z0, o0), self.pt(s, z0, o1), self.pt(s, z1, o1), self.pt(s, z1, o0)],
                      (self.d[0] * sgn, self.d[1] * sgn, 0.0), mat)
        if bottom:
            _face(m, [self.pt(s0, z0, o0), self.pt(s1, z0, o0), self.pt(s1, z0, o1), self.pt(s0, z0, o1)],
                  (0.0, 0.0, -1.0), mat)


def _tint(mat_idx, rgb):
    b = _b()
    return mat_idx[b.custom_look(b.FACADE, rgb)]


def window(m, wl, c, z, w, h, p, mat_idx, cell, door=False):
    """One window (centre c along the wall, sill z): its sash from the plaster sheet's window cell, a surround, a sill,
    shutters or a roller-shutter box, a flower box. door: a balcony door (down to the floor: no sill, no shutters)."""
    b = _b()
    glass = mat_idx["bne_bld_fac_white"]
    u0, u1 = b.WIN_CELLS[cell % len(b.WIN_CELLS)]
    v0, v1 = b.WIN_UV[2], b.WIN_UV[3]
    uvw = ((u0, v0), (u1, v0), (u1, v1), (u0, v1))
    pair = p["pair"] and not door
    full = 2 * w + 0.14 if pair else w
    if p["surround"] is not None:
        t = 0.13
        wl.quad(m, c - full / 2 - t, c + full / 2 + t, z - (0.0 if door else 0.04), z + h + t, 0.015,
                _tint(mat_idx, p["surround"]))
    if pair:
        for sgn in (-1, 1):
            cc = c + sgn * (w / 2 + 0.07)
            wl.quad(m, cc - w / 2, cc + w / 2, z, z + h, 0.03, glass, uvw)
    else:
        wl.quad(m, c - w / 2, c + w / 2, z, z + h, 0.03, glass, uvw)
    if door:
        return
    wl.quad(m, c - full / 2 - 0.08, c + full / 2 + 0.08, z - 0.10, z, 0.07, _tint(mat_idx, SILL))      # (the sill: its front)
    if p["shutter"] is not None:
        sw = w * 0.5
        col = _tint(mat_idx, p["shutter"])
        for sgn in (-1, 1):
            e0 = c + sgn * (full / 2 + 0.04)
            wl.quad(m, min(e0, e0 + sgn * sw), max(e0, e0 + sgn * sw), z - 0.02, z + h + 0.02, 0.045, col)
    elif p["roller"]:
        wl.quad(m, c - full / 2 - 0.05, c + full / 2 + 0.05, z + h + 0.04, z + h + 0.30, 0.08, _tint(mat_idx, ROLLER))
    if p["flower"] and z > 0:
        wl.box(m, c - full / 2, c + full / 2, z - 0.01, z + 0.16, 0.0, 0.2, _tint(mat_idx, BOX_WOOD), sides=False)
        _face(m, [wl.pt(c - full / 2 + 0.03, z + 0.2, 0.03), wl.pt(c + full / 2 - 0.03, z + 0.2, 0.03),
                  wl.pt(c + full / 2 - 0.03, z + 0.2, 0.18), wl.pt(c - full / 2 + 0.03, z + 0.2, 0.18)],
              (0.0, 0.0, 1.0), _tint(mat_idx, BOX_GREEN))


def balcony(m, wl, c, width, z, depth, p, mat_idx):
    """A balcony slab (its top at z) with a solid parapet of 1 m, sides and front."""
    slab, rail = _tint(mat_idx, SLAB), _tint(mat_idx, p["rail"])
    s0, s1 = c - width / 2, c + width / 2
    wl.box(m, s0, s1, z - 0.16, z, 0.0, depth, slab, bottom=True)
    wl.quad(m, s0, s1, z, z + 1.0, depth, rail)
    for s, sgn in ((s0, -1.0), (s1, 1.0)):
        _face(m, [wl.pt(s, z, 0.0), wl.pt(s, z, depth), wl.pt(s, z + 1.0, depth), wl.pt(s, z + 1.0, 0.0)],
              (wl.d[0] * sgn, wl.d[1] * sgn, 0.0), rail)


def dress(m, spec, ring, z_eave, mat_idx):
    """The kit's parts on the walls of ring (the drawn one, counter-clockwise) of a building with spec.kit, its walls
    (wall_spec) already drawn plain up to z_eave."""
    b = _b()
    kit = spec.kit
    p = kit["p"]
    ring = np.asarray(ring, np.float64)
    n = len(ring)
    floors, floor_h = int(spec.floors), float(spec.floor_h)
    base = z_eave - floors * floor_h
    door_edge, door_s = (spec.door if spec.door else (None, None))
    edges = []
    for i in range(n):
        a, bb = ring[i], ring[(i + 1) % n]
        L = float(math.hypot(*(bb - a)))
        if L < MIN_EDGE_M:
            continue
        d = (bb - a) / L
        edges.append((i, a, d, L, blind_intervals(kit["nb"], a, d, L)))
    balcony_edge = None
    if p["balcony"]:
        cand = [e for e in edges if e[0] != door_edge and sum(hi - lo for lo, hi in e[4]) < 0.2 * e[3] and e[3] >= 5.5]
        if cand:
            balcony_edge = max(cand, key=lambda e: e[3])[0]
    count = 0
    for i, a, d, L, blind in edges:
        o = (d[1], -d[0])
        wl = Wall(a, d, o)
        w, h = p["w"], p["h"]
        full = 2 * w + 0.14 if p["pair"] else w
        ds = door_s if i == door_edge else None
        cs = axes(L, p["bay"], full, 0.9, ds, blind)
        pe = p
        if len(cs) >= 2 and p["shutter"] is not None and (cs[1] - cs[0]) < full + 2 * w * 0.5 + 0.3:
            pe = dict(p, shutter=None)            # (bays too narrow for open shutters beside the windows)
        for f in range(floors):
            sill = 0.95 if f == 0 else 0.85
            hh = min(h, floor_h - sill - 0.5)
            if hh < 0.9:
                continue
            zf = base + f * floor_h + sill
            if zf + hh > z_eave - 0.2:
                continue
            for k, c in enumerate(cs):
                if f == 0 and ds is not None and abs(c - ds) < 0.5 + full / 2 + 0.2:
                    continue
                if count >= MAX_WINDOWS:
                    break
                cell = int(b._hash01(spec.seed, "wincell", i, f, k) * 1000)
                door = False
                if balcony_edge == i and f >= 1 and k % 2 == 0 and not (f == floors - 1 and not p["flat"]):
                    bw = min(2.8, (cs[1] - cs[0]) * 0.9 if len(cs) > 1 else 2.4)
                    if bw >= 1.6:
                        balcony(m, wl, c, bw, base + f * floor_h, 1.4, p, mat_idx)
                        window(m, wl, c, base + f * floor_h + 0.05, min(w, bw - 0.5), min(2.1, floor_h - 0.45), pe,
                               mat_idx, cell, door=True)
                        count += 1
                        continue
                window(m, wl, c, zf, w, hh, pe, mat_idx, cell, door)
                count += 1
        if i == door_edge and p["canopy"] and floor_h >= 2.7 and z_eave > base + 3.2:
            gd = b.door_ground(spec, i, ds, base) if spec.gring is not None else base
            zc = gd + 2.45
            if zc + 0.1 < z_eave - 0.2:
                wl.box(m, ds - 0.95, ds + 0.95, zc, zc + 0.1, 0.0, 1.0, _tint(mat_idx, CANOPY), bottom=True)


def slice_at(Q, d, z):
    """Where the polygon Q (k, 3) crosses height z: the two end points (3-vectors) with the smallest and largest
    coordinate along the horizontal direction d, or None."""
    pts = []
    for i in range(len(Q)):
        a, c = Q[i], Q[(i + 1) % len(Q)]
        if (a[2] - z) * (c[2] - z) < 0 or a[2] == z:
            t = (z - a[2]) / (c[2] - a[2]) if c[2] != a[2] else 0.0
            pts.append(a + (c - a) * t)
    if len(pts) < 2:
        return None
    s = [float(q[0] * d[0] + q[1] * d[1]) for q in pts]
    return pts[int(np.argmin(s))], pts[int(np.argmax(s))]


def gable_window(m, spec, Q, out, z_eave, mat_idx):
    """An attic window in the gable face Q (k, 3, above the eave, facing `out`) of a kit building, if it is high and wide
    enough and no neighbour's wall touches it."""
    kit = spec.kit
    Q = np.asarray(Q, np.float64)
    rise = float(Q[:, 2].max()) - z_eave
    if rise < 1.9:
        return
    d = (-out[1], out[0])
    w, h = 0.85, 1.1
    sill = z_eave + max(0.35, rise * 0.28)
    if sill + h > float(Q[:, 2].max()) - 0.45:
        return
    top = slice_at(Q, d, sill + h)
    mid = slice_at(Q, d, sill + h / 2)
    if top is None or mid is None:
        return
    if float((top[1] - top[0])[0] * d[0] + (top[1] - top[0])[1] * d[1]) < w + 0.7:
        return
    c = (mid[0] + mid[1]) / 2
    if kit["nb"] is not None and len(kit["nb"]):
        iv = blind_intervals(kit["nb"], (c[0], c[1]), d, 0.0001 + 1.0)
        if iv:
            return
    b = _b()
    wl = Wall((c[0], c[1]), d, out)
    cell = int(b._hash01(spec.seed, "gablecell", round(float(c[0]), 1), round(float(c[1]), 1)) * 1000)
    p = dict(kit["p"], flower=False, roller=False, shutter=kit["p"]["shutter"])
    window(m, wl, 0.0, sill, w, h, dict(p, pair=False), mat_idx, cell)


# ---------------------------------------------------------------------------------------------------- roofs

DORMER_W, DORMER_H = 1.3, 1.3       # m: a shed dormer's front
DORMER_LID_DEG = 10.0               # its lid's pitch (the roof's: 45 degrees and more)
DORMER_MIN_EAVE_M = 6.5             # a roof face with an eave at least this long gets dormers (every 4.5 m, up to 3)
DORMER_MIN_RISE_M = 3.3             # ... and rising this much (a dormer needs the height to sit in)
CHIMNEY = (0.55, 0.32, 0.26)
CHIMNEY_PLASTER = (0.82, 0.80, 0.75)
ROOF_SHARES = {"altstadt": (0.45, 0.65), "siedlung": (0.35, 0.50), "block": (0.05, 0.20), "neubau": (0.0, 0.10)}


def _plane_z(Q0, n, x, y):
    return float(Q0[2] - (n[0] * (x - Q0[0]) + n[1] * (y - Q0[1])) / n[2])


def _tri3(m, a, b, c, outward, mat, uv=((0.01, 0.997),) * 3):
    n = np.cross(np.subtract(b, a), np.subtract(c, a))
    if float(dnp.dot(n, outward)) < 0:
        b, c = c, b
        uv = uv[::-1]
    m.tri(tuple(a), tuple(b), tuple(c), *uv, mat)


def roof_details(m, spec, planes, z_eave, mat_idx):
    """Dormers and a chimney on a pitched LoD2 roof of a kit building. planes: the roof faces (k, 3) in the tile's
    frame (lifted as drawn). Dormers: shed dormers with a window on the long steep faces; chimney: on the ridge."""
    b = _b()
    kit = spec.kit
    p = kit["p"]
    if p["flat"] or not planes:
        return
    d_share, c_share = ROOF_SHARES[kit["fam"]]
    wall = mat_idx[kit["wall"]]
    top = max(float(np.max(np.asarray(Q)[:, 2])) for Q in planes)
    tg = dmath.tan(math.radians(DORMER_LID_DEG))
    glass = mat_idx["bne_bld_fac_white"]
    u0, u1 = b.WIN_CELLS[0]
    uvw = ((u0, b.WIN_UV[2]), (u1, b.WIN_UV[2]), (u1, b.WIN_UV[3]), (u0, b.WIN_UV[3]))
    dormers = 0
    for pi, Q in enumerate(planes):
        Q = np.asarray(Q, np.float64)
        nv = b._newell(Q)
        if nv[2] < 0:
            nv = -nv
        nh = float(math.hypot(nv[0], nv[1]))
        if nv[2] < 1e-6 or nh < 1e-6:
            continue
        ta = nh / float(nv[2])
        zmin, zmax = float(Q[:, 2].min()), float(Q[:, 2].max())
        if not (0.55 <= ta <= 1.9) or zmax - zmin < DORMER_MIN_RISE_M or dormers >= 6:
            continue
        if b._hash01(spec.seed, "dormer", pi) >= d_share:
            continue
        out = np.array([nv[0], nv[1]]) / nh
        d = (-out[1], out[0])
        ends = slice_at(Q, d, zmin + 0.05)
        if ends is None:
            continue
        P0 = ends[0]
        L = float((ends[1] - ends[0])[0] * d[0] + (ends[1] - ends[0])[1] * d[1])
        if L < DORMER_MIN_EAVE_M:
            continue
        nd = 1 if L < 9.0 else 2 if L < 14.0 else 3
        tf = DORMER_H / ta
        delta = DORMER_H / max(ta - tg, 0.2)
        for j in range(nd):
            s = L * (j + 0.5) / nd
            c = np.array([P0[0] + d[0] * s - out[0] * tf, P0[1] + d[1] * s - out[1] * tf])
            bk = c - out * delta
            zf = _plane_z(Q[0], nv, c[0], c[1])
            zb = _plane_z(Q[0], nv, bk[0], bk[1])
            if zb > zmax - 0.45 or zf < zmin + 0.4:
                continue
            ctr = P0[:2] + np.array(d) * s
            hi = slice_at(Q, d, zf + DORMER_H)                       # (the face is wide enough up there)
            if hi is None:
                continue
            lo_s = float((hi[0] - P0)[0] * d[0] + (hi[0] - P0)[1] * d[1])
            hi_s = float((hi[1] - P0)[0] * d[0] + (hi[1] - P0)[1] * d[1])
            if s - DORMER_W / 2 - 0.5 < lo_s or s + DORMER_W / 2 + 0.5 > hi_s:
                continue
            fl, fr = c - np.array(d) * DORMER_W / 2, c + np.array(d) * DORMER_W / 2
            bl, br = bk - np.array(d) * DORMER_W / 2, bk + np.array(d) * DORMER_W / 2
            zt = zf + DORMER_H
            _face(m, [(fl[0], fl[1], zf), (fr[0], fr[1], zf), (fr[0], fr[1], zt), (fl[0], fl[1], zt)],
                  (out[0], out[1], 0.0), wall, ((0.01, 0.997),) * 4)
            gc = c + out * 0.03
            gl, gr = gc - np.array(d) * 0.42, gc + np.array(d) * 0.42
            _face(m, [(gl[0], gl[1], zf + 0.2), (gr[0], gr[1], zf + 0.2), (gr[0], gr[1], zf + 1.1), (gl[0], gl[1], zf + 1.1)],
                  (out[0], out[1], 0.0), glass, uvw)
            b._poly3d(m, [(fl[0], fl[1], zt), (fr[0], fr[1], zt), (br[0], br[1], zb), (bl[0], bl[1], zb)],
                      mat_idx[spec.roof_mat])
            for e, sg in ((fl, -1.0), (fr, 1.0)):
                bq = bl if sg < 0 else br
                _tri3(m, (e[0], e[1], zf), (bq[0], bq[1], zb), (e[0], e[1], zt), (d[0] * sg, d[1] * sg, 0.0), wall)
            dormers += 1
    if b._hash01(spec.seed, "chimney") < c_share:
        pts = [np.asarray(Q, np.float64)[k] for Q in planes for k in range(len(Q)) if Q[k][2] >= top - 0.05]
        if len(pts) >= 2:
            best = max(((q, r) for i, q in enumerate(pts) for r in pts[i + 1:]), key=lambda qr: float(dnp.hypot(*(qr[0][:2] - qr[1][:2]))))
            a_, c_ = best[0][:2], best[1][:2]
            ln = float(dnp.hypot(*(c_ - a_)))
            if ln >= 2.5:
                u = (c_ - a_) / ln
                t = 0.22 + 0.12 * b._hash01(spec.seed, "chimney t") if b._hash01(spec.seed, "chimney side") < 0.5 \
                    else 0.66 + 0.12 * b._hash01(spec.seed, "chimney t")
                pos = a_ + u * ln * t
                col = CHIMNEY if b._hash01(spec.seed, "chimney col") < 0.7 else CHIMNEY_PLASTER
                b._box(m, pos, u, 0.28, 0.28, top - 0.5, top + 0.95, _tint(mat_idx, col))
