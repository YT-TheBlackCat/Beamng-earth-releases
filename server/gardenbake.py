"""Gardens from the LGL BW open data: one parcel at a time, lawn, beds, hedges, trees, terraces and driveways.

The recipe for a garden (docs/leonberg/RECHERCHE_BAUSTEINE_UND_DATEN.md, step 1 and 2) comes from three sources of the
LGL Baden-Wuerttemberg (dl-de/by-2-0), all in ETRS89 / UTM32:

- DOP20 RGBI: the true orthophoto, 0.2 m, red, green, blue and near infrared (houses do not lean: roofs lie on their
  ground plan), 1 km GeoTIFFs in 2 km zips.
- nDOM1: the height of the objects over the ground, 1 m, from the same leaf-on flight.
- ALKIS (Shape, one zip per Gemarkung): the parcels (flurstueck, with the actual use) and the buildings.

bake_km() turns one 1 km square of them into a recipe file (npz, a few hundred KB): a ground class raster at
GROUND_PX_M over the garden parcels only, hedge lines, trees and shrubs with heights, and one summary row per parcel.
gardens.py reads these in the server; tools/bake_gardens.py downloads the sources and calls bake_km. No machine
learning: near infrared and height separate plants from paving and trees from hedges from lawn (fixed thresholds,
calibrated on U8 Ramtel), texture separates lawn from beds.
"""
import io
import math
import os
import re
import struct
import zipfile

import numpy as np

import dnp

VERSION = "v1"
CREDIT = "Datenquelle: LGL, www.lgl-bw.de, DOP20/nDOM1/ALKIS, dl-de/by-2-0 (https://www.govdata.de/dl-de/by-2-0), Daten verändert"

DOP_URL = "https://opengeodata.lgl-bw.de/data/dop20/dop20rgbi_32_{e}_{n}_2_bw.zip"
NDOM_URL = "https://opengeodata.lgl-bw.de/data/ndom1/ndom1_32_{e}_{n}_2_bw.zip"
ALKIS_URL = "https://opengeodata.lgl-bw.de/data/alkis/{name}"
# Gemarkungen of Leonberg and its neighbours (Verw-Grenzen_ALKIS / the portal's Gemarkungen layer): file name of the
# Shape zip -> (west, south, east, north) of the Gemarkung in UTM32 metres, rounded out to whole km (bake picks the
# zips whose box meets the square)
ALKIS_ZIPS = {
    "ALKIS-oE_081500_Leonberg_shp.zip": (496000, 5399000, 506000, 5407000),
    "ALKIS-oE_081501_Gebersheim_shp.zip": (497000, 5405000, 500000, 5409000),
    "ALKIS-oE_081502_Höfingen_shp.zip": (498000, 5405000, 504000, 5409000),
    "ALKIS-oE_081503_Warmbronn_shp.zip": (497000, 5400000, 502000, 5403000),
    "ALKIS-oE_081165_Gerlingen_shp.zip": (502000, 5400000, 507000, 5408000),
    "ALKIS-oE_081155_Ditzingen_shp.zip": (502000, 5406000, 507000, 5410000),
    "ALKIS-oE_081495_Rutesheim_shp.zip": (493000, 5404000, 498000, 5409000),
    "ALKIS-oE_081510_Renningen_shp.zip": (492000, 5397000, 499000, 5406000),
    "ALKIS-oE_081525_Magstadt_shp.zip": (494000, 5397000, 503000, 5402000),
}

# ---------------------------------------------------------------------------------------------------------------------
# ALKIS Shape files (no GDAL: the format is simple)


def read_dbf(data, fields=None, encoding="cp1252"):
    """Rows of a dBASE table (bytes) as dicts of str (stripped; the ALKIS Shape files are Windows-1252); only the named
    fields when given."""
    n, hl, rl = struct.unpack("<4xIHH20x", data[:32])
    cols, p, pos = [], 32, 1
    while data[p] != 0x0D:
        name = data[p:p + 11].split(b"\0")[0].decode("ascii", "replace")
        size = data[p + 16]
        if fields is None or name in fields:
            cols.append((name, pos, size))
        pos += size
        p += 32
    out = []
    for i in range(n):
        r = data[hl + i * rl: hl + (i + 1) * rl]
        if not r or r[:1] == b"*":          # (deleted record)
            out.append(None)
            continue
        out.append({name: r[a:a + s].decode(encoding, "replace").strip() for name, a, s in cols})
    return out


def read_shp_polygons(data):
    """The polygons of a Shape file (bytes): one list of rings (float64 (k, 2) arrays, closed point dropped) per record,
    None for a null shape. Polygon (5), PolygonZ (15) and PolygonM (25) records."""
    out, p, size = [], 100, len(data)
    while p + 8 <= size:
        _, words = struct.unpack(">ii", data[p:p + 8])
        rec = data[p + 8:p + 8 + 2 * words]
        p += 8 + 2 * words
        st = struct.unpack("<i", rec[:4])[0] if len(rec) >= 4 else 0
        if st not in (5, 15, 25):
            out.append(None)
            continue
        nparts, npts = struct.unpack("<ii", rec[36:44])
        parts = list(struct.unpack(f"<{nparts}i", rec[44:44 + 4 * nparts])) + [npts]
        xy = np.frombuffer(rec, "<f8", 2 * npts, 44 + 4 * nparts).reshape(npts, 2)
        rings = []
        for a, b in zip(parts[:-1], parts[1:]):
            r = xy[a:b]
            if len(r) > 1 and r[0, 0] == r[-1, 0] and r[0, 1] == r[-1, 1]:
                r = r[:-1]
            if len(r) >= 3:
                rings.append(np.array(r, np.float64))
        out.append(rings)
    return out


def signed_area(ring):
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def read_alkis(zip_bytes_or_path, box=None):
    """(parcels, buildings) of an ALKIS-oE Shape zip, those whose bounding box meets box (w, s, e, n UTM m) if given.
    parcels: [{"id", "use", "area", "rings"}]: use = the actual uses ("Wohnbaufläche;612|Garten;80" -> the names),
    rings: outer rings first (shapefile order: clockwise outers, counter-clockwise holes). buildings: [{"id",
    "function", "kind", "floors", "rings"}]."""
    zf = zipfile.ZipFile(zip_bytes_or_path if not isinstance(zip_bytes_or_path, bytes)
                         else io.BytesIO(zip_bytes_or_path))
    names = {os.path.basename(n): n for n in zf.namelist()}

    def table(stem, fields):
        shp = read_shp_polygons(zf.read(names[stem + ".shp"]))
        dbf = read_dbf(zf.read(names[stem + ".dbf"]), fields)
        rows = []
        for rings, row in zip(shp, dbf):
            if not rings or row is None:
                continue
            if box is not None:
                allxy = np.concatenate(rings)
                if (allxy[:, 0].max() < box[0] or allxy[:, 0].min() > box[2]
                        or allxy[:, 1].max() < box[1] or allxy[:, 1].min() > box[3]):
                    continue
            rows.append((rings, row))
        return rows

    parcels = []
    for rings, row in table("flurstueck", {"idflurst", "tntxt", "flaeche"}):
        uses = [u.split(";")[0] for u in row.get("tntxt", "").split("|") if u]
        try:
            area = float(row.get("flaeche") or 0)
        except ValueError:
            area = 0.0
        parcels.append({"id": row.get("idflurst", ""), "use": uses, "area": area, "rings": rings})
    buildings = []
    for rings, row in table("gebaeudeBauwerke", {"oid", "funktion", "gebnutzbez", "anzahlgs"}):
        try:
            floors = int(row.get("anzahlgs") or 0)
        except ValueError:
            floors = 0
        buildings.append({"id": row.get("oid", ""), "function": row.get("funktion", ""),
                          "kind": row.get("gebnutzbez", ""), "floors": floors, "rings": rings})
    return parcels, buildings


# ---------------------------------------------------------------------------------------------------------------------
# Rasters: DOP20 RGBI (0.2 m), nDOM1 (1 m), both 1 km GeoTIFFs, row 0 north

DOP_PX_M = 0.2
GROUND_PX_M = 0.5               # the recipe's ground raster (the game's terrain squares are 1 m)
KM_PX = int(round(1000 / DOP_PX_M))
GARDEN_USES = ("Wohnbaufläche", "Garten", "Fläche gemischter Nutzung")     # parcels baked (prefix match)
ROAD_USES = ("Straßenverkehr", "Weg", "Platz", "Parkplatz")                 # parcels a driveway leads to
SKIP_BUILDINGS = ("Tiefgarage",)                                            # underground: their roof is garden
SKIP_KINDS = ("Sonstiges Bauwerk",)        # pools, walls, steps: seen in the photo instead of drawn as a house


def read_tiff_raw(data):
    """(array (rows, cols, bands) or (rows, cols), dtype) of an uncompressed strip TIFF (bytes): the LGL's DOP20 RGBI
    (4 x uint8, which Pillow reads as RGB and drops the infrared) and nDOM1 (float32). Pillow for anything else."""
    from PIL import Image
    im = Image.open(io.BytesIO(data))
    t = im.tag_v2
    w, h = im.size
    bits = t.get(258, (8,))
    bands = len(bits) if isinstance(bits, tuple) else 1
    if t.get(259, 1) == 1 and t.get(284, 1) == 1 and 273 in t:
        off = np.asarray(t[273], np.int64)
        cnt = np.asarray(t[279], np.int64)
        fmt = t.get(339, (1,))
        fmt = fmt[0] if isinstance(fmt, tuple) else fmt
        dt = {(8, 1): np.uint8, (32, 3): np.float32, (16, 1): np.uint16}.get(
            (bits[0] if isinstance(bits, tuple) else bits, fmt))
        if dt is not None and np.all(off[1:] == off[:-1] + cnt[:-1]):
            a = np.frombuffer(data, np.dtype(dt).newbyteorder("<" if data[:2] == b"II" else ">"),
                              w * h * bands, int(off[0]))
            return a.reshape((h, w, bands) if bands > 1 else (h, w)).astype(dt)
    return np.asarray(im)


def _km_member(zf, kind, e_km, n_km):
    """The name of the 1 km GeoTIFF of (e_km, n_km) in a DOP20 / nDOM1 zip, or None."""
    pat = re.compile(rf"{kind}_32_{e_km}_{n_km}_1_bw(_\d{{4}})?\.tif$")
    for name in zf.namelist():
        if pat.search(name):
            return name
    return None


def zip_of_km(e_km, n_km):
    """(e, n) of the 2 km zip holding the 1 km square (odd east, even north km, as lod2geom.zip_tile)."""
    return 2 * math.floor((e_km - 1) / 2) + 1, 2 * math.floor(n_km / 2)


class Mosaic:
    """DOP20 RGBI and nDOM1 over a UTM rectangle [e0, e1) x [n0, n1) (m, multiples of 0.2 m), read from the 2 km zips
    in src_dir (dop20/dop20rgbi_32_<e>_<n>_2_bw.zip, ndom1/ndom1_32_<e>_<n>_2_bw.zip). rgbi: uint8 (rows, cols, 4), row 0 at n1;
    ndom: float32 (rows, cols) at the same pixels (bilinear from 1 m); valid: bool, where both had data."""

    def __init__(self, src_dir, e0, n0, e1, n1):
        self.e0, self.n0, self.e1, self.n1 = e0, n0, e1, n1
        cols, rows = int(round((e1 - e0) / DOP_PX_M)), int(round((n1 - n0) / DOP_PX_M))
        self.rgbi = np.zeros((rows, cols, 4), np.uint8)
        self.ndom = np.zeros((rows, cols), np.float32)
        self.valid = np.zeros((rows, cols), bool)
        self.dates = set()
        zips = {}
        for e_km in range(math.floor(e0 / 1000), math.ceil(e1 / 1000)):
            for n_km in range(math.floor(n0 / 1000), math.ceil(n1 / 1000)):
                ze, zn = zip_of_km(e_km, n_km)
                dop = self._member(zips, os.path.join(src_dir, "dop20", f"dop20rgbi_32_{ze}_{zn}_2_bw.zip"), "dop20rgbi",
                                   e_km, n_km)
                ndom = self._member(zips, os.path.join(src_dir, "ndom1", f"ndom1_32_{ze}_{zn}_2_bw.zip"), "ndom1", e_km, n_km)
                if dop is None or ndom is None:
                    continue
                self._paste(e_km, n_km, read_tiff_raw(dop), read_tiff_raw(ndom))
        for zf in zips.values():
            if zf is not None:
                zf.close()

    def _member(self, zips, path, kind, e_km, n_km):
        if path not in zips:
            zips[path] = zipfile.ZipFile(path) if os.path.exists(path) else None
        zf = zips[path]
        if zf is None:
            return None
        name = _km_member(zf, kind, e_km, n_km)
        if name is None:
            return None
        m = re.search(r"_(\d{4})\.tif$", name)
        if m:
            self.dates.add(m.group(1))
        return zf.read(name)

    def _paste(self, e_km, n_km, dop, ndom):
        from scipy import ndimage
        # the square's pixels in the mosaic (row 0 = north)
        c0 = int(round((e_km * 1000 - self.e0) / DOP_PX_M))
        r0 = int(round((self.n1 - (n_km + 1) * 1000) / DOP_PX_M))
        a0, a1 = max(c0, 0), min(c0 + KM_PX, self.rgbi.shape[1])
        b0, b1 = max(r0, 0), min(r0 + KM_PX, self.rgbi.shape[0])
        if a0 >= a1 or b0 >= b1:
            return
        if dop.ndim == 3 and dop.shape[2] >= 4 and dop.shape[:2] == (KM_PX, KM_PX):
            self.rgbi[b0:b1, a0:a1] = dop[b0 - r0:b1 - r0, a0 - c0:a1 - c0, :4]
        else:
            return
        # nDOM: 1 m pixels, centre of (row i, col j) at (j + 0.5, i + 0.5) m from the square's NW corner
        jj = (np.arange(a0 - c0, a1 - c0) + 0.5) * DOP_PX_M - 0.5
        ii = (np.arange(b0 - r0, b1 - r0) + 0.5) * DOP_PX_M - 0.5
        nd = np.nan_to_num(np.asarray(ndom, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        nd[nd < -100] = 0.0
        rows = ndimage.map_coordinates(nd, np.meshgrid(ii, jj, indexing="ij"), order=1, mode="nearest")
        self.ndom[b0:b1, a0:a1] = np.clip(rows, 0.0, 60.0)
        self.valid[b0:b1, a0:a1] = True

    def to_px(self, e, n):
        """(col, row) float pixel coordinates (pixel centres at .5) of UTM points."""
        return (np.asarray(e, np.float64) - self.e0) / DOP_PX_M, (self.n1 - np.asarray(n, np.float64)) / DOP_PX_M

    def to_utm(self, col, row):
        """UTM (e, n) of the centres of pixels (col, row)."""
        return self.e0 + (np.asarray(col, np.float64) + 0.5) * DOP_PX_M, \
            self.n1 - (np.asarray(row, np.float64) + 0.5) * DOP_PX_M


def rasterize(shape, polys, e0, n1, px=DOP_PX_M, values=None, dtype=np.uint8):
    """A raster (rows, cols) of the polygons (each a list of UTM rings, outer first, holes after) drawn in order with
    their values (1 by default; later ones on top, holes cleared to 0 within their own polygon's outline)."""
    from PIL import Image, ImageDraw
    mode = "I" if np.dtype(dtype).itemsize > 1 else "L"
    im = Image.new(mode, (shape[1], shape[0]), 0)
    dr = ImageDraw.Draw(im)
    for k, rings in enumerate(polys):
        v = 1 if values is None else int(values[k])
        for j, ring in enumerate(rings):
            xy = [((x - e0) / px, (n1 - y) / px) for x, y in ring]
            if len(xy) < 3:
                continue
            hole = j > 0 and signed_area(np.asarray(ring)) > 0      # (shapefile: counter-clockwise = hole)
            dr.polygon(xy, fill=0 if hole else v)
    return np.asarray(im).astype(dtype)


# ---------------------------------------------------------------------------------------------------------------------
# Pixel classes (0.2 m)

P_NONE, P_BUILDING, P_TREE, P_SHRUB, P_LAWN, P_BED, P_PAVED, P_GRAVEL, P_WATER, P_OBJECT = range(10)
P_NAMES = ("none", "building", "tree", "shrub", "lawn", "bed", "paved", "gravel", "water", "object")
NDVI_VEG = 0.22         # (NIR - R) / (NIR + R) from which a pixel is a plant (U8: roads 0.05-0.12, lawns 0.3-0.5)
NDVI_SOIL = 0.08        # ... bare soil and mulch between this and NDVI_VEG when brownish (beds)
H_TREE = 3.0            # m over the ground (nDOM): trees from here, ...
H_SHRUB = 1.0           # ... shrubs and hedges from here (nDOM is +-0.4 m, 1 m pixels: a lawn beside a tree reads
H_LAWN_MAX = 1.8        # up to ~1.5 m); a smooth plant surface below H_LAWN_MAX is still lawn (U8, 2026-11-25)
TEX_LAWN = 8.0          # the near infrared's standard deviation over 1 m, averaged over 2 m: smoother is lawn
                        # (U8: mown lawns 1.6-4, crowns 6-12; low plants up to this are lawn, the rest beds)
GRAVEL_BRIGHT = 205     # mean of r, g, b from which grey ground is gravel or light stone (asphalt: 150-185)
GRAVEL_SPREAD = 22      # ... at most this between its channels
WATER_BLUE = 25         # b - r of a pool
SMOOTH_PX = 5           # majority filter (1 m)


def _local_std(x, k):
    from scipy import ndimage
    m = ndimage.uniform_filter(x, k)
    m2 = ndimage.uniform_filter(x * x, k)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def classify_pixels(rgbi, ndom, building):
    """The class (P_*) of each pixel from the orthophoto (uint8 (rows, cols, 4) R, G, B, NIR), the height over the
    ground (float32 m) and the building mask (bool), smoothed by a 1 m majority filter. Pure."""
    from scipy import ndimage
    r, g, b, n = (rgbi[..., i].astype(np.float32) for i in range(4))
    ndvi = (n - r) / (n + r + 1.0)
    bright = (r + g + b) / 3.0
    spread = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    veg = ndvi >= NDVI_VEG
    tex = ndimage.uniform_filter(_local_std(n, 5), 9)
    ndom = ground_height(ndom, building)
    cls = np.full(r.shape, P_PAVED, np.uint8)
    cls[(~veg) & (ndvi >= NDVI_SOIL) & (r >= g) & (r >= b + 8)] = P_BED
    cls[(~veg) & (bright >= GRAVEL_BRIGHT) & (spread <= GRAVEL_SPREAD)] = P_GRAVEL
    cls[(~veg) & (b >= r + WATER_BLUE) & (b >= g + 5) & (ndvi < 0)] = P_WATER
    smooth = tex < TEX_LAWN
    cls[veg & (ndom < H_SHRUB)] = P_BED
    cls[veg & smooth & (ndom < H_LAWN_MAX)] = P_LAWN
    cls[veg & ~smooth & (ndom >= H_SHRUB)] = P_SHRUB
    cls[veg & smooth & (ndom >= H_LAWN_MAX)] = P_SHRUB
    cls[veg & (ndom >= H_TREE)] = P_TREE
    cls = majority(cls, len(P_NAMES), SMOOTH_PX)
    cls[building] = P_BUILDING
    return cls


NEAR_HOUSE_M = 1.4      # m: the 1 m nDOM carries a roof's height this far out onto the ground beside it


def ground_height(ndom, building):
    """The nDOM with the roofs' bleed taken out: within NEAR_HOUSE_M of a building, the lowest height within that
    distance (a lawn along a house read 2-6 m from the bilinear 1 m raster, U8). Pure."""
    from scipy import ndimage
    if not building.any():
        return ndom
    k = int(round(NEAR_HOUSE_M / DOP_PX_M)) * 2 + 1
    near = ndimage.binary_dilation(building, np.ones((k, k), bool)) & ~building
    out = ndom.copy()
    out[near] = ndimage.minimum_filter(np.where(building, 99.0, ndom).astype(np.float32), k)[near]
    return out


def majority(cls, k, size):
    """Each pixel's most common class within a size x size window (ties: the lower class). Pure."""
    from scipy import ndimage
    best = np.zeros(cls.shape, np.float32)
    out = cls.copy()
    for c in range(k):
        m = cls == c
        if not m.any():
            continue
        f = ndimage.uniform_filter(m.astype(np.float32), size)
        win = f > best + 1e-6
        out[win] = c
        best[win] = f[win]
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Parcels: what the pixels are in a garden

# ground classes of the recipe (GROUND_PX_M raster; 0 = not a garden parcel: the server leaves the ground as it is)
G_NONE, G_LAWN, G_BED, G_PLANTING, G_TERRACE, G_DRIVEWAY, G_PATH, G_GRAVEL, G_WATER, G_BUILDING = range(10)
G_NAMES = ("none", "lawn", "bed", "planting", "terrace", "driveway", "path", "gravel", "water", "building")
MARGIN_M = 40.0             # m of the sources read around a km square (parcels across its edge are seen whole)
ROAD_TOUCH_M = 1.0          # m: paving this close to a road parcel is a driveway (or the path to the door)
PATH_W_M = 1.6              # m: paving at most this wide is a path; wider paving not on a road is a terrace (by the
                            # house or not: a yard, a seat; all paving is drawn alike, PAVERS)
MIN_PAVED_M2 = 1.0          # m2: smaller paved specks take their surroundings' class
UNDER_TREE_M = 4.0          # m: ground under a crown farther than this from open ground is planting (no lawn)
# Hedges: along a parcel's boundary, a band of plants 0.9-4.5 m high, at most HEDGE_MAX_W_M wide
HEDGE_H = (0.9, 4.5)
HEDGE_STEP = 0.5            # m between the stations along the boundary
HEDGE_BAND = (-0.6, 1.6)    # m across the boundary (+ = into the parcel) a hedge's plants are looked for
HEDGE_MIN_W_M = 0.8         # m of hedge plants across the band at a hedge station (nDOM 1 m: narrower ones blur away)
HEDGE_TREE_MAX = 0.5        # at most this share of a station's hedge plants under a tree crown (a row of trees)
HEDGE_BEYOND = (2.6, 3.6)   # m into the parcel: a hedge is narrow, its plants end before this ...
HEDGE_BEYOND_MAX = 0.5      # ... (at most this share there; else a wood or a shrub bed, not a hedge)
HEDGE_MIN_M = 3.0           # m: shorter runs are shrubs
HEDGE_GAP_M = 1.0           # m: a gap this short does not end a run
HEDGE_MAX_W_M = 2.5
# Trees and shrubs: tops of the nDOM
TREE_BANDS = ((H_TREE, 6.0, 1.5), (6.0, 10.0, 2.0), (10.0, 15.0, 3.0), (15.0, 99.0, 4.0))  # (from, to m: radius m a top
                            # is the highest within; vegetation.py's bands)
SHRUB_SPACING_M = 2.5         # m: a shrub is the highest within this window, ...
SHRUB_MIN_H = 1.2           # m: ... at least this high ...
SHRUB_OFF_TREE_M = 1.0      # m: ... and this far from a crown (the nDOM's blur around a tree reads as shrubs)


def _dist_px(m):
    """Distance (px) of each pixel to the nearest True of m (inf when none)."""
    from scipy import ndimage
    if not m.any():
        return np.full(m.shape, np.inf, np.float32)
    return ndimage.distance_transform_edt(~m).astype(np.float32)


def _dp(pts, tol):
    """Douglas-Peucker simplification of a polyline (k, 2)."""
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    d = b - a
    L = math.hypot(d[0], d[1])
    rel = pts - a
    dist = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0]) / L if L > 1e-9 else dnp.hypot(rel[:, 0], rel[:, 1])
    i = int(np.argmax(dist))
    if dist[i] <= tol:
        return np.stack([a, b])
    return np.concatenate([_dp(pts[:i + 1], tol)[:-1], _dp(pts[i:], tol)])


def _resample(ring, step):
    """Stations every step m along a closed ring: (points (k, 2), unit tangents (k, 2))."""
    closed = np.vstack([ring, ring[:1]])
    seg = np.diff(closed, axis=0)
    L = dnp.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(L)])
    if cum[-1] < step:
        return np.zeros((0, 2)), np.zeros((0, 2))
    s = np.arange(step / 2, cum[-1], step)
    k = np.clip(np.searchsorted(cum, s, "right") - 1, 0, len(L) - 1)
    t = (s - cum[k]) / np.maximum(L[k], 1e-9)
    pts = closed[k] + seg[k] * t[:, None]
    tan = seg[k] / np.maximum(L[k], 1e-9)[:, None]
    return pts, tan


class Garden:
    """The per-parcel reading of a classified mosaic (Mosaic + classify_pixels): ground classes, hedges, trees and
    shrubs in UTM metres."""

    def __init__(self, mosaic, cls, parcels, road_mask):
        self.m, self.cls = mosaic, cls
        self.parcels = parcels                  # [(id, rings)], label k + 1 in self.pid
        shape = cls.shape
        self.pid = rasterize(shape, [r for _, r in parcels], mosaic.e0, mosaic.n1,
                             values=np.arange(1, len(parcels) + 1), dtype=np.int32)
        self.road_d = _dist_px(road_mask) * DOP_PX_M
        self.h = h = ground_height(mosaic.ndom, cls == P_BUILDING)
        rgbi = mosaic.rgbi
        r, n = rgbi[..., 0].astype(np.float32), rgbi[..., 3].astype(np.float32)
        self.veg = (n - r) / (n + r + 1.0) >= NDVI_VEG
        self.hedgy = self.veg & (h >= HEDGE_H[0]) & (h <= HEDGE_H[1]) & (cls != P_BUILDING)
        self.used = np.zeros(shape, bool)        # pixels of a hedge already found
        self.ground = np.zeros(shape, np.uint8)
        self.hedges, self.trees, self.shrubs, self.stats = [], [], [], []

    # -- ground ----------------------------------------------------------------------------------------------------
    def build_ground(self):
        from scipy import ndimage
        cls, pid, g = self.cls, self.pid, self.ground
        inside = pid > 0
        g[inside & (cls == P_LAWN)] = G_LAWN
        g[inside & (cls == P_BED)] = G_BED
        g[inside & (cls == P_SHRUB)] = G_PLANTING
        g[inside & (cls == P_GRAVEL)] = G_GRAVEL
        g[inside & (cls == P_WATER)] = G_WATER
        g[inside & (cls == P_BUILDING)] = G_BUILDING
        paved = inside & np.isin(cls, (P_PAVED, P_OBJECT))
        lab, nlab = ndimage.label(paved, structure=np.ones((3, 3), bool))
        if nlab:
            idx = np.arange(1, nlab + 1)
            area = ndimage.sum(paved, lab, idx) * dnp.sq(DOP_PX_M)
            road = ndimage.minimum(self.road_d, lab, idx)
            width = 2 * ndimage.maximum(ndimage.distance_transform_edt(paved), lab, idx) * DOP_PX_M
            kind = np.full(nlab + 1, G_TERRACE, np.uint8)
            kind[1:][width <= PATH_W_M] = G_PATH
            kind[1:][road <= ROAD_TOUCH_M] = np.where(width[road <= ROAD_TOUCH_M] <= PATH_W_M, G_PATH, G_DRIVEWAY)
            kind[1:][area < MIN_PAVED_M2] = G_NONE
            g[paved] = kind[lab[paved]]
        # under the crowns (and specks): the nearest open ground of the same parcel, planting deep under a wood
        open_ground = inside & np.isin(g, (G_LAWN, G_BED, G_PLANTING, G_TERRACE, G_DRIVEWAY, G_PATH, G_GRAVEL))
        hole = inside & (g == G_NONE)
        if hole.any() and open_ground.any():
            d, (ii, jj) = ndimage.distance_transform_edt(~open_ground, return_indices=True)
            fill = g[ii, jj]
            same = pid[ii, jj] == pid
            fill = np.where(same & (d * DOP_PX_M <= UNDER_TREE_M), fill, G_PLANTING)
            g[hole] = fill[hole]
        elif hole.any():
            g[hole] = G_PLANTING

    # -- hedges ----------------------------------------------------------------------------------------------------
    def find_hedges(self):
        for k, (_, rings) in enumerate(self.parcels):
            for j, ring in enumerate(rings):
                if j > 0 and signed_area(ring) > 0:
                    continue
                self._ring_hedges(k + 1, ring)

    def _sample(self, mask, pts):
        col, row = self.m.to_px(pts[..., 0], pts[..., 1])
        c, r = np.floor(col).astype(int), np.floor(row).astype(int)
        ok = (c >= 0) & (r >= 0) & (c < mask.shape[1]) & (r < mask.shape[0])
        out = np.zeros(c.shape, mask.dtype)
        out[ok] = mask[r[ok], c[ok]]
        return out

    def _ring_hedges(self, label, ring):
        pts, tan = _resample(ring, HEDGE_STEP)
        if not len(pts):
            return
        inward = np.stack([tan[:, 1], -tan[:, 0]], 1)     # right of the direction: inside a clockwise ring
        if signed_area(ring) > 0:                          # (a counter-clockwise outer ring)
            inward = -inward
        offs = np.arange(HEDGE_BAND[0], HEDGE_BAND[1] + 1e-6, 0.2)
        band = pts[:, None, :] + offs[None, :, None] * inward[:, None, :]
        hedgy = self._sample(self.hedgy, band) & ~self._sample(self.used, band)
        beyond = pts[:, None, :] + np.arange(*HEDGE_BEYOND, 0.2)[None, :, None] * inward[:, None, :]
        wood = self._sample(self.hedgy | (self.cls == P_TREE), beyond).mean(1) > HEDGE_BEYOND_MAX
        own = self._sample(self.pid, pts + 0.3 * inward) == label
        treed = (self._sample(self.cls == P_TREE, band) & hedgy).sum(1) > HEDGE_TREE_MAX * hedgy.sum(1)
        on = (hedgy.sum(1) * 0.2 >= HEDGE_MIN_W_M) & ~wood & ~treed & own
        # runs of stations (gaps up to HEDGE_GAP_M bridged), the ring's start joined to its end
        gap = int(round(HEDGE_GAP_M / HEDGE_STEP))
        idx = np.flatnonzero(on)
        if not len(idx):
            return
        runs, cur = [], [idx[0]]
        for i in idx[1:]:
            if i - cur[-1] <= gap + 1:
                cur.append(i)
            else:
                runs.append(cur)
                cur = [i]
        runs.append(cur)
        if len(runs) > 1 and runs[0][0] + len(pts) - runs[-1][-1] <= gap + 1:
            runs[0] = runs.pop() + runs[0]
        for run in runs:
            sel = np.arange(run[0], run[-1] + 1) % len(pts) if run[-1] >= run[0] else \
                np.concatenate([np.arange(run[0], len(pts)), np.arange(0, run[-1] + 1)])
            if len(sel) * HEDGE_STEP < HEDGE_MIN_M:
                continue
            hb = hedgy[sel]
            # the hedge's middle across the band and its width, per station (smoothed)
            cnt = np.maximum(hb.sum(1), 1)
            mid = (hb * offs[None, :]).sum(1) / cnt
            mid = dnp.convolve(np.pad(mid, 2, mode="edge"), np.ones(5) / 5, "valid")
            width = float(np.clip(np.median(hb.sum(1)) * 0.2, 0.6, HEDGE_MAX_W_M))
            line = pts[sel] + mid[:, None] * inward[sel]
            hs = self._sample(self.h, band[sel])[hb]
            height = float(np.clip(np.percentile(hs, 75) if len(hs) else 1.5, HEDGE_H[0], HEDGE_H[1]))
            line = _dp(line, 0.3)
            self.hedges.append((line, height, width))
            self._mark_used(line, width)

    def _mark_used(self, line, width):
        from PIL import Image, ImageDraw
        col, row = self.m.to_px(line[:, 0], line[:, 1])
        pad = int(round((width + 0.6) / DOP_PX_M)) + 2
        c0, r0 = max(int(col.min()) - pad, 0), max(int(row.min()) - pad, 0)
        c1, r1 = min(int(col.max()) + pad + 1, self.used.shape[1]), min(int(row.max()) + pad + 1, self.used.shape[0])
        if c0 >= c1 or r0 >= r1:
            return
        im = Image.new("L", (c1 - c0, r1 - r0), 0)
        ImageDraw.Draw(im).line(list(zip((col - c0).tolist(), (row - r0).tolist())), fill=1,
                                width=max(1, int(round((width + 0.6) / DOP_PX_M))))
        self.used[r0:r1, c0:c1] |= np.asarray(im, bool)

    # -- trees and shrubs ------------------------------------------------------------------------------------------
    def find_plants(self):
        from scipy import ndimage
        h = ndimage.uniform_filter(self.h, 3)
        inside = self.pid > 0
        tree = inside & (self.cls == P_TREE)
        tops = np.zeros(h.shape, bool)
        for lo, hi, w in TREE_BANDS:
            size = int(round(2 * w / DOP_PX_M)) | 1
            tops |= tree & (h >= lo) & (h < hi) & (h >= ndimage.maximum_filter(h, size) - 1e-3)
        tl, nt = ndimage.label(tops)
        if nt:
            cy, cx = np.array(ndimage.center_of_mass(tops, tl, np.arange(1, nt + 1))).T
            ri, ci = np.round(cy).astype(int), np.round(cx).astype(int)
            seeds = np.zeros(h.shape, np.int32)
            seeds[ri, ci] = np.arange(1, nt + 1)
            _, (ii, jj) = ndimage.distance_transform_edt(seeds == 0, return_indices=True)
            owner = np.where(tree, seeds[ii, jj], 0)
            area = ndimage.sum(tree, owner, np.arange(1, nt + 1)) * dnp.sq(DOP_PX_M)
            r = self.m.rgbi[..., 0].astype(np.float32)
            n = self.m.rgbi[..., 3].astype(np.float32)
            ndvi = ndimage.mean((n - r) / (n + r + 1.0), owner, np.arange(1, nt + 1))
            e, nn = self.m.to_utm(ci, ri)
            for k in range(nt):
                ht = float(h[ri[k], ci[k]])
                rad = float(np.clip(math.sqrt(max(area[k], 0.5) / math.pi), 1.0, max(1.0, 0.5 * ht)))
                self.trees.append((float(e[k]), float(nn[k]), ht, rad, float(ndvi[k])))
        shrub = inside & (self.cls == P_SHRUB) & ~self.used & (_dist_px(tree) * DOP_PX_M >= SHRUB_OFF_TREE_M)
        size = int(round(SHRUB_SPACING_M / DOP_PX_M)) | 1
        st = shrub & (h >= ndimage.maximum_filter(np.where(shrub, h, 0), size) - 1e-3) & (h >= SHRUB_MIN_H)
        sl, ns = ndimage.label(st)
        if ns:
            cy, cx = np.array(ndimage.center_of_mass(st, sl, np.arange(1, ns + 1))).T
            ri, ci = np.round(cy).astype(int), np.round(cx).astype(int)
            e, nn = self.m.to_utm(ci, ri)
            for k in range(ns):
                hs = float(np.clip(h[ri[k], ci[k]], 0.6, H_TREE))
                self.shrubs.append((float(e[k]), float(nn[k]), hs, float(np.clip(0.35 * hs + 0.3, 0.5, 1.5))))


# ---------------------------------------------------------------------------------------------------------------------
# One km square -> recipe file


def km_path(folder, e_km, n_km):
    return os.path.join(folder, f"{e_km}_{n_km}.npz")


def alkis_zips_over(e0, n0, e1, n1):
    return [name for name, (w, s, e, n) in ALKIS_ZIPS.items() if w < e1 and e > e0 and s < n1 and n > n0]


def is_garden(uses):
    return any(u.startswith(GARDEN_USES) for u in uses)


def is_road(uses):
    return any(u.startswith(ROAD_USES) for u in uses)


def bake_km(e_km, n_km, src_dir, out_path, log=print):
    """Bake the recipe of the 1 km UTM32 square (e_km, n_km) from the sources in src_dir (dop20/, ndom1/, alkis/ as
    tools/bake_gardens.py fills it) into out_path (npz). Returns the summary dict, or None when the sources do not
    cover it (nothing written)."""
    import json
    import time
    t0 = time.time()
    e0, n0 = e_km * 1000 - MARGIN_M, n_km * 1000 - MARGIN_M
    e1, n1 = (e_km + 1) * 1000 + MARGIN_M, (n_km + 1) * 1000 + MARGIN_M
    m = Mosaic(src_dir, e0, n0, e1, n1)
    if not m.valid.any():
        log(f"gardens {e_km}_{n_km}: no photo or heights (zips missing, or not an uncompressed 4-band DOP20 RGBI)")
        return None
    if not m.valid.all():
        log(f"gardens {e_km}_{n_km}: the sources cover {100 * m.valid.mean():.0f} % of the square and its "
            f"{MARGIN_M:.0f} m margin (neighbouring zips missing: parcels across that edge are cut)")
    parcels, buildings, seen = [], [], set()
    for name in alkis_zips_over(e0, n0, e1, n1):
        path = os.path.join(src_dir, "alkis", name)
        if not os.path.exists(path):
            continue
        p, b = read_alkis(path, (e0, n0, e1, n1))
        for q in p:
            if q["id"] not in seen:
                seen.add(q["id"])
                parcels.append(q)
        buildings += b
    if not parcels:
        return None
    shape = m.valid.shape
    house = rasterize(shape, [b["rings"] for b in buildings if b["function"] not in SKIP_BUILDINGS
                              and not b["kind"].startswith(SKIP_KINDS)], e0, n1).astype(bool)
    cls = classify_pixels(m.rgbi, m.ndom, house)
    gardens = [(q["id"], q["rings"]) for q in sorted(parcels, key=lambda q: -q["area"]) if is_garden(q["use"])]
    roads = rasterize(shape, [q["rings"] for q in parcels if is_road(q["use"])], e0, n1).astype(bool)
    g = Garden(m, cls, gardens, roads)
    g.pid[~m.valid] = 0
    g.build_ground()
    g.find_hedges()
    g.find_plants()
    # the ground raster over the square, GROUND_PX_M cells (row 0 north), nearest 0.2 m pixel
    k = int(round(1000 / GROUND_PX_M))
    ce = e_km * 1000 + (np.arange(k) + 0.5) * GROUND_PX_M
    cn = (n_km + 1) * 1000 - (np.arange(k) + 0.5) * GROUND_PX_M
    col = np.floor((ce - e0) / DOP_PX_M).astype(int)
    row = np.floor((n1 - cn) / DOP_PX_M).astype(int)
    ground = g.ground[np.ix_(row, col)]
    ox, oy = e_km * 1000.0, n_km * 1000.0

    def mine(e, n):
        return ox <= e < ox + 1000 and oy <= n < oy + 1000

    hedges = []
    for line, h, w in g.hedges:
        seg = dnp.hypot(*np.diff(line, axis=0).T)
        half = np.searchsorted(np.cumsum(seg), seg.sum() / 2)
        if mine(*line[min(half, len(line) - 1)]):
            hedges.append((line, h, w))
    trees = [t for t in g.trees if mine(t[0], t[1])]
    shrubs = [t for t in g.shrubs if mine(t[0], t[1])]
    # one row per garden parcel whose largest ring's centre is in the square: area, shares of the ground classes
    counts = np.zeros((len(gardens) + 1, len(G_NAMES)), np.int64)
    np.add.at(counts, (g.pid.ravel(), g.ground.ravel()), 1)
    pids, pstats, pxy = [], [], []
    for i, (pid, rings) in enumerate(gardens):
        ring = max(rings, key=lambda r: abs(signed_area(r)))
        c = ring.mean(0)
        if not mine(c[0], c[1]) or counts[i + 1].sum() == 0:
            continue
        pids.append(pid)
        pstats.append(counts[i + 1] / counts[i + 1].sum())
        pxy.append(c - (ox, oy))
    starts = np.concatenate([[0], np.cumsum([len(h[0]) for h in hedges])]).astype(np.int32)
    arrays = dict(
        e_km=np.int32(e_km), n_km=np.int32(n_km), px=np.float32(GROUND_PX_M), ground=ground.astype(np.uint8),
        hedge_xy=(np.concatenate([h[0] for h in hedges]) - (ox, oy)).astype(np.float32) if hedges
        else np.zeros((0, 2), np.float32),
        hedge_start=starts, hedge_h=np.array([h[1] for h in hedges], np.float32),
        hedge_w=np.array([h[2] for h in hedges], np.float32),
        tree=np.array([(e - ox, n - oy, h, r, nd) for e, n, h, r, nd in trees], np.float32).reshape(-1, 5),
        shrub=np.array([(e - ox, n - oy, h, r) for e, n, h, r in shrubs], np.float32).reshape(-1, 4),
        parcel_id=np.array(pids, str), parcel_xy=np.array(pxy, np.float32).reshape(-1, 2),
        parcel_share=np.array(pstats, np.float32).reshape(-1, len(G_NAMES)),
        meta=np.array(json.dumps({"version": VERSION, "dop_years": sorted(m.dates), "credit": CREDIT,
                                  "classes": G_NAMES})))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = f"{out_path}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out_path)
    share = np.bincount(ground.ravel(), minlength=len(G_NAMES)) / ground.size
    out = {"km": (e_km, n_km), "parcels": len(pids), "hedges": len(hedges),
           "hedge_m": round(float(sum(dnp.hypot(*np.diff(h[0], axis=0).T).sum() for h in hedges)), 1),
           "trees": len(trees), "shrubs": len(shrubs), "seconds": round(time.time() - t0, 1),
           "share": {G_NAMES[i]: round(float(v), 4) for i, v in enumerate(share) if v > 0}}
    log(f"gardens {e_km}_{n_km}: {out}")
    return out
