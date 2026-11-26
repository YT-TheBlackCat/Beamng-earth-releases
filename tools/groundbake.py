"""The classifier behind groundclass.py: ground classes from the state orthophoto's four bands (red, green, blue,
near infrared) and the object height model. Pure numpy / scipy, one 1 km block at a time; tools/bake_ground_classes.py
reads the LGL files and writes server/data/leonberg_ground.npz. scipy is needed here only (not by the running server).

Rules (the numbers were read off the photo around Leonberg's Neues Rathaus and the Ezach field tracks, 2026-11-25:
asphalt lum 150-175 and neutral (warm <= 0.0), the plaza's paving 200-222 and warm (0.02-0.05), the grass NDVI 0.25-0.35):
  plants       NDVI >= NDVI_VEG (and the infrared bright enough: shade and dark water are not plants), by the object
               height over the ground: GRASS < H_SHRUB <= SHRUB < H_TREE <= TREE
  OBJECT       not a plant, higher than H_OBJECT over the ground (roofs, walls; the 1 m model smears them a metre
               over the ground beside them, which this only costs a metre of ground)
  RED          red paving (clinker, red asphalt of a cycle lane): redder than beige gravel or brown soil
  STONE        light and warm: paving stones, slabs, natural stone, concrete
  GRAVEL / SOIL  beige and saturated, or a little plant signal (NDVI_DRY): bare ground; a large patch is SOIL (a field), a small one GRAVEL (a yard, a track)
  ASPHALT      the rest of the hard ground: neutral grey
Dark cells (shade) take the nearest lit cell's class, then a majority filter takes the speckle (cars, markings, joints).
"""
import numpy as np

import groundclass as gc

NDVI_VEG = 0.20
NDVI_DRY = 0.09               # between asphalt (0-0.05) and green plants: stubble, bare soil, dirt paths
NIR_MIN = 110.0               # a plant's infrared (8 bit) is above this; shade on asphalt is below
H_SHRUB = 0.8                 # m over the ground
H_TREE = 3.0
H_OBJECT = 2.5
SHADE_LUM = 75.0
STONE_LUM = 188.0
STONE_WARM = 0.005            # (r - b) / luminance
RED_RG = 0.30                 # (r - g) / luminance: red paving is red, beige gravel and brown soil are not that red
RED_SAT = 0.35
BEIGE_SAT = 0.09
BEIGE_WARM = 0.055
BEIGE_LUM = (95.0, 235.0)
SOIL_MIN_M2 = 500.0           # a bare patch at least this large is a field
MAJORITY_PX = 5
SHADE_FILL_M = 6.0


def downsample2(a):
    """2 x 2 mean of an (H, W[, C]) array with even sides."""
    a = np.asarray(a, np.float32)
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    a = a[:h, :w]
    return (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]) * np.float32(0.25)


def features(rgbi):
    """(luminance, warmth, saturation, ndvi, nir, redness) of float cells (H, W, 4); redness (r - g) / luminance."""
    r, g, b, n = (rgbi[..., i] for i in range(4))
    lum = np.float32(.299) * r + np.float32(.587) * g + np.float32(.114) * b
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = (mx - mn) / np.maximum(mx, 1.0)
    warm = (r - b) / np.maximum(lum, 1.0)
    ndvi = (n - r) / np.maximum(n + r, 1.0)
    red = (r - g) / np.maximum(lum, 1.0)
    return lum, warm, sat, ndvi, n, red


def majority(cls, k=MAJORITY_PX, n_classes=len(gc.NAMES)):
    """The most common class in the k x k window of each cell (ties: the lower code)."""
    from scipy import ndimage as ndi
    best = np.full(cls.shape, -1.0, np.float32)
    out = np.zeros(cls.shape, np.uint8)
    for c in range(n_classes):
        share = ndi.uniform_filter((cls == c).astype(np.float32), size=k, mode="nearest")
        better = share > best
        out[better] = c
        best[better] = share[better]
    return out


def classify(rgbi, ndom):
    """Class codes (groundclass.NONE ...) of the cells of rgbi (H, W, 4) float, bands R G B NIR 0-255, with the object
    height over the ground ndom (H, W) in metres at the same cells. Pure."""
    from scipy import ndimage as ndi
    # (a 3 x 3 mean first: single bright or dark cells, a car, a marking, are not a class)
    smooth = np.stack([ndi.uniform_filter(rgbi[..., i], size=3, mode="nearest") for i in range(4)], -1)
    lum, warm, sat, ndvi, nir, redness = features(smooth)
    veg = (ndvi >= NDVI_VEG) & (nir >= NIR_MIN)
    obj = ~veg & (ndom >= H_OBJECT)
    shade = ~veg & ~obj & (lum < SHADE_LUM)
    hard = ~veg & ~obj & ~shade
    cls = np.full(lum.shape, gc.NONE, np.uint8)
    cls[hard] = gc.ASPHALT
    cls[hard & (lum >= STONE_LUM) & (warm >= STONE_WARM)] = gc.STONE
    beige = hard & (sat >= BEIGE_SAT) & (warm >= BEIGE_WARM) & (lum >= BEIGE_LUM[0]) & (lum <= BEIGE_LUM[1])
    cls[beige | (hard & (ndvi >= NDVI_DRY))] = gc.GRAVEL
    cls[hard & (redness >= RED_RG) & (sat >= RED_SAT)] = gc.RED
    cls[veg & (ndom < H_SHRUB)] = gc.GRASS
    cls[veg & (ndom >= H_SHRUB) & (ndom < H_TREE)] = gc.SHRUB
    cls[veg & (ndom >= H_TREE)] = gc.TREE
    cls[obj] = gc.OBJECT
    # shade (and the penumbra round it, where the 3 x 3 mean mixes lit and dark cells): the nearest lit cell's class (up
    # to SHADE_FILL_M), asphalt past it
    if shade.any():
        core = ndi.binary_erosion(~shade, iterations=2, border_value=1)      # (lit cells, two cells in from the shade)
        dist, (ri, ci) = ndi.distance_transform_edt(~core, return_indices=True)
        fill = cls[ri, ci]
        redo = ~core & ~obj
        near = redo & (dist * gc.BAKE_CELL_M <= SHADE_FILL_M)
        cls[near] = fill[near]
        cls[shade & ~near] = gc.ASPHALT
    cls = majority(cls)
    # a large bare patch is a field
    bare = cls == gc.GRAVEL
    if bare.any():
        lab, n = ndi.label(bare)
        area = np.bincount(lab.ravel(), minlength=n + 1) * gc.BAKE_CELL_M ** 2
        big = area[lab] >= SOIL_MIN_M2
        cls[bare & big] = gc.SOIL
    return cls


# ---- outlines -----------------------------------------------------------------------------------------------------

def _simplify(pts, eps):
    """Douglas-Peucker of a closed ring's points (k, 2) (kept corners: the farthest from the chord of the whole)."""
    pts = np.asarray(pts, np.float64)
    n = len(pts)
    if n < 5:
        return pts
    # split the ring at its two most distant points so the open recursion has fixed ends
    i0 = 0
    i1 = int(np.argmax(np.hypot(*(pts - pts[0]).T)))
    keep = np.zeros(n, bool)
    keep[[i0, i1]] = True
    stack = [(i0, i1), (i1, n)]           # (a, b): points a..b, b may be n meaning the start again
    while stack:
        a, b = stack.pop()
        pa = pts[a]
        pb = pts[b % n]
        if b - a < 2:
            continue
        seg = pts[a + 1:b]
        d = pb - pa
        ln = float(np.hypot(*d))
        if ln < 1e-12:
            dist = np.hypot(*(seg - pa).T)
        else:
            dist = np.abs(d[0] * (seg[:, 1] - pa[1]) - d[1] * (seg[:, 0] - pa[0])) / ln
        j = int(np.argmax(dist))
        if dist[j] > eps:
            m = a + 1 + j
            keep[m] = True
            stack.append((a, m))
            stack.append((m, b))
    return pts[keep]


def _area(ring):
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def trace(mask, eps=0.6):
    """Outlines of the True cells of mask (H, W): [(ring (k, 2) of (col, row) corner coordinates, signed area)]: an
    outer ring has a negative area (_area, with rows counted downwards), a hole a positive one. The cell boundary
    edges are linked (inside on the left of each edge) and each ring simplified to eps cells."""
    mask = np.asarray(mask, bool)
    p = np.pad(mask, 1)
    # edges between a True cell and a False one, as (x0, y0, x1, y1) in padded corner coordinates, inside on the left
    # (y down): going along +x with the inside below is "left" when y points down? Use a fixed convention instead:
    edges = {}
    ys, xs = np.nonzero(p[1:-1, 1:-1] & ~p[:-2, 1:-1])       # top edge of a cell: inside below
    for y, x in zip(ys.tolist(), xs.tolist()):
        edges.setdefault((x + 1, y), []).append((x, y))       # runs right to left
    ys, xs = np.nonzero(p[1:-1, 1:-1] & ~p[2:, 1:-1])        # bottom edge: inside above
    for y, x in zip(ys.tolist(), xs.tolist()):
        edges.setdefault((x, y + 1), []).append((x + 1, y + 1))   # left to right
    ys, xs = np.nonzero(p[1:-1, 1:-1] & ~p[1:-1, :-2])       # left edge: inside right
    for y, x in zip(ys.tolist(), xs.tolist()):
        edges.setdefault((x, y), []).append((x, y + 1))       # runs down
    ys, xs = np.nonzero(p[1:-1, 1:-1] & ~p[1:-1, 2:])        # right edge: inside left
    for y, x in zip(ys.tolist(), xs.tolist()):
        edges.setdefault((x + 1, y + 1), []).append((x + 1, y))   # runs up
    rings = []
    while edges:
        start = next(iter(edges))
        ring = [start]
        cur = start
        prev = None
        while True:
            outs = edges.get(cur)
            if not outs:
                break
            if len(outs) > 1 and prev is not None:       # a vertex shared by two corners: turn the same way every time
                dx, dy = cur[0] - prev[0], cur[1] - prev[1]
                outs.sort(key=lambda q: (dx * (q[1] - cur[1]) - dy * (q[0] - cur[0])))
            nxt = outs.pop(0)
            if not outs:
                del edges[cur]
            prev, cur = cur, nxt
            if cur == start:
                break
            ring.append(cur)
        if len(ring) >= 4:
            rings.append(np.asarray(ring, np.float64) - 1.0)
    out = []
    for r in rings:
        s = _simplify(r, eps)
        if len(s) >= 3:
            out.append((s, _area(s)))
    return out


def smooth_mask(mask, sigma=1.2):
    """A mask with its staircase and pinholes smoothed (gaussian, threshold one half)."""
    from scipy import ndimage as ndi
    return ndi.gaussian_filter(np.asarray(mask, np.float32), sigma, mode="nearest") > 0.5


def polygons(mask, min_area_m2=20.0, min_hole_m2=6.0, eps_m=0.8, sigma=1.5):
    """[(outer ring (k, 2) in cells (col, row), [hole rings])] of the True cells of mask (rings simplified to eps_m,
    holes assigned to the smallest outer ring that contains their first point). Areas in m2 by gc.BAKE_CELL_M."""
    cell2 = gc.BAKE_CELL_M ** 2
    outs, holes = [], []
    for ring, area in trace(smooth_mask(mask, sigma), eps_m / gc.BAKE_CELL_M):
        if abs(area) * cell2 < (min_area_m2 if area < 0 else min_hole_m2):
            continue
        (outs if area < 0 else holes).append(ring)
    result = [(o, []) for o in outs]
    areas = [abs(_area(o)) for o in outs]
    for h in holes:
        pt = h[0]
        best = None
        for k, o in enumerate(outs):
            if (best is None or areas[k] < areas[best]) and _inside(o, pt):
                best = k
        if best is not None:
            result[best][1].append(h)
    return result


def _inside(ring, pt):
    """Even-odd test of the point (x, y) against a ring (k, 2)."""
    x, y = pt
    xs, ys = ring[:, 0], ring[:, 1]
    xn, yn = np.roll(xs, -1), np.roll(ys, -1)
    cross = ((ys > y) != (yn > y)) & (x < (xn - xs) * (y - ys) / np.where(yn == ys, 1e-12, yn - ys) + xs)
    return bool(cross.sum() % 2)
