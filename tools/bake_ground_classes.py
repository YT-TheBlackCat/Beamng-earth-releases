#!/usr/bin/env python3
"""Bake Leonberg's ground classes from the LGL orthophoto (DOP20 RGBI) and the object height model (nDOM1).

    python tools/bake_ground_classes.py [--tiles 499_5402,501_5402,499_5404,501_5404] [--out server/data/leonberg_ground.npz]
                                         [--cache ~/.cache/beamng-earth] [--no-download] [--preview DIR]

Reads the 2 x 2 km zips of the LGL Open GeoData (dl-de/by-2-0) from <cache>/lgl/dop20 and <cache>/lgl/ndom1 (named as
the LGL names them, downloaded when missing: ~360 MB and ~10 MB a tile; the raw data never goes into the repo), runs
tools/groundbake.py's classifier on every 1 km block with a margin so that no seam shows, and writes one compact npz:
the class raster (0.8 m cells) and the outlines of the paved areas (stone, red paving, gravel) as rings. The server
reads that file (server/groundclass.py). The photo's own date is in the .csv beside each tif ("Aktualitaet"): the
flight of 2025 shows plants of the season it was taken, leaf-on or not.
"""
import argparse
import io
import os
import sys
import time
import urllib.request
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
sys.path.insert(0, HERE)

import groundbake as gb  # noqa: E402
import groundclass as gc  # noqa: E402

DEFAULT_TILES = "499_5402,501_5402,499_5404,501_5404"
URL = "https://opengeodata.lgl-bw.de/data/{kind}/{kind}{rgbi}_32_{e}_{n}_2_bw.zip"
MARGIN_M = 20                   # m of neighbouring photo around each 1 km block
SEMI = {"dop20": "rgbi", "ndom1": ""}


def zip_path(cache, kind, e, n):
    return os.path.join(cache, "lgl", kind, f"{kind}{SEMI[kind]}_32_{e}_{n}_2_bw.zip")


def fetch(cache, kind, e, n, download=True):
    p = zip_path(cache, kind, e, n)
    if os.path.exists(p):
        return p
    if not download:
        return None
    os.makedirs(os.path.dirname(p), exist_ok=True)
    url = URL.format(kind=kind, rgbi=SEMI[kind], e=e, n=n)
    print("download", url, flush=True)
    urllib.request.urlretrieve(url, p + ".part")
    os.replace(p + ".part", p)
    return p


class Mosaic:
    """1 km tiles of the photo (4 bands, uncompressed TIFF, memory mapped) and of the nDOM, read from the zips."""

    def __init__(self, cache, download=True):
        self.cache, self.download = cache, download
        self.dop, self.ndom = {}, {}

    def _extract(self, kind, e2, n2, member_end):
        """Extracts the zip's members that end with member_end into <cache>/lgl/<kind>/x/ (kept); {path}."""
        zp = fetch(self.cache, kind, e2, n2, self.download)
        if zp is None:
            return {}
        out = {}
        d = os.path.join(self.cache, "lgl", kind, "x")
        os.makedirs(d, exist_ok=True)
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if name.endswith(member_end):
                    dst = os.path.join(d, os.path.basename(name))
                    if not os.path.exists(dst):
                        with z.open(name) as src, open(dst + ".part", "wb") as f:
                            while True:
                                chunk = src.read(1 << 24)
                                if not chunk:
                                    break
                                f.write(chunk)
                        os.replace(dst + ".part", dst)
                    out[os.path.basename(name)] = dst
        return out

    def photo(self, e, n):
        """The photo of the 1 km tile whose lower left corner is (e, n) km: (5000, 5000, 4) uint8 or None."""
        key = (e, n)
        if key not in self.dop:
            files = self._extract("dop20", _tile2_e(e), _tile2_n(n), ".tif")
            path = next((p for nm, p in files.items() if nm.startswith(f"dop20rgbi_32_{e}_{n}_1_bw")), None)
            self.dop[key] = None if path is None else _open_rgbi(path)
        return self.dop[key]

    def height(self, e, n):
        """The nDOM of the 1 km tile (e, n) km: (1000, 1000) float32 (m over the ground) or None."""
        key = (e, n)
        if key not in self.ndom:
            files = self._extract("ndom1", _tile2_e(e), _tile2_n(n), ".tif")
            path = next((p for nm, p in files.items() if nm.startswith(f"ndom1_32_{e}_{n}_1_bw")), None)
            if path is None:
                self.ndom[key] = None
            else:
                from PIL import Image
                self.ndom[key] = np.asarray(Image.open(path), np.float32)
        return self.ndom[key]


def _tile2_e(e):
    """The 2 km tile (its lower left km, as the LGL names it) that holds the 1 km column e: eastings start at odd km."""
    return e if e % 2 == 1 else e - 1


def _tile2_n(n):
    """... and the 1 km row n: northings start at even km."""
    return n if n % 2 == 0 else n - 1


def _open_rgbi(path):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        w, h = im.size
        tile = im.tile[0]
        if im.tag_v2.get(277) != 4 or tile[0] != "raw":
            raise ValueError(f"{path}: not an uncompressed 4 band TIFF")
        offset = tile[2]
    # (the 4th band: Pillow reads such a file as RGBX and drops it; the strips follow each other without a gap)
    return np.memmap(path, dtype=np.uint8, mode="r", offset=offset, shape=(h, w, 4))


def window(mos, e0_m, n0_m, w_m, h_m):
    """The photo (rows north to south, 0.2 m) and the nDOM at 0.4 m of the box (e0_m, n0_m, w_m, h_m) in UTM metres
    (all whole metres): (rgbi (H, W, 4) uint8, ndom (H/2, W/2) float32 at the same cells as the 2 x 2 mean of the photo)."""
    from PIL import Image
    w, h = int(w_m * 5), int(h_m * 5)
    rgbi = np.zeros((h, w, 4), np.uint8)
    top = n0_m + h_m
    nd = np.zeros((int(h_m), int(w_m)), np.float32)
    for e in range(int(e0_m // 1000), int((e0_m + w_m - 1) // 1000) + 1):
        for n in range(int(n0_m // 1000), int((top - 1) // 1000) + 1):
            a = mos.photo(e, n)
            d = mos.height(e, n)
            xa, xb = max(e0_m, e * 1000), min(e0_m + w_m, e * 1000 + 1000)
            ya, yb = max(n0_m, n * 1000), min(top, n * 1000 + 1000)
            if xb <= xa or yb <= ya:
                continue
            if a is not None:
                r0, r1 = int((n * 1000 + 1000 - yb) * 5), int((n * 1000 + 1000 - ya) * 5)
                c0, c1 = int((xa - e * 1000) * 5), int((xb - e * 1000) * 5)
                o_r, o_c = int((top - yb) * 5), int((xa - e0_m) * 5)
                rgbi[o_r:o_r + r1 - r0, o_c:o_c + c1 - c0] = a[r0:r1, c0:c1]
            if d is not None:
                r0, r1 = int(n * 1000 + 1000 - yb), int(n * 1000 + 1000 - ya)
                c0, c1 = int(xa - e * 1000), int(xb - e * 1000)
                nd[int(top - yb):int(top - yb) + r1 - r0, int(xa - e0_m):int(xa - e0_m) + c1 - c0] = d[r0:r1, c0:c1]
    ndom = np.asarray(Image.fromarray(nd).resize((w // 2, h // 2), Image.BILINEAR), np.float32)
    return rgbi, ndom


def bake(cache, tiles, download=True, log=print):
    """(classes (H, W) uint8, e0, n0) of the 2 km tiles (list of (e_km, n_km) lower left corners)."""
    e_km = [t[0] for t in tiles]
    n_km = [t[1] for t in tiles]
    e0, n0 = min(e_km) * 1000, min(n_km) * 1000
    w_km, h_km = max(e_km) + 2 - min(e_km), max(n_km) + 2 - min(n_km)
    cells = int(1000 / gc.BAKE_CELL_M)
    out = np.zeros((h_km * cells, w_km * cells), np.uint8)
    mos = Mosaic(cache, download)
    have = {(t[0] + i, t[1] + j) for t in tiles for i in (0, 1) for j in (0, 1)}
    for (e, n) in sorted(have):
        t0 = time.time()
        rgbi, ndom = window(mos, e * 1000 - MARGIN_M, n * 1000 - MARGIN_M, 1000 + 2 * MARGIN_M, 1000 + 2 * MARGIN_M)
        cls = gb.classify(gb.downsample2(rgbi), ndom)
        m = int(MARGIN_M / gc.BAKE_CELL_M)
        block = cls[m:-m, m:-m]
        r0 = (n0 // 1000 + h_km - 1 - n) * cells
        c0 = (e - e0 // 1000) * cells
        out[r0:r0 + cells, c0:c0 + cells] = block
        log(f"block {e}_{n}: {time.time() - t0:.1f} s")
    return out, e0, n0


RING_CLASSES = (gc.STONE, gc.RED, gc.GRAVEL)


def outlines(classes, e0, n0):
    """The outlines of RING_CLASSES in classes: (rings float32 (K, 2) metres east / north of (e0, n0), ring_class (M,),
    ring_off (M + 1,), ring_hole (M,): the index of the outer ring a hole belongs to, -1 for an outer one)."""
    h = classes.shape[0]
    pts, cls_, off, hole = [], [], [0], []
    for c in RING_CLASSES:
        for outer, holes in gb.polygons(classes == c, min_area_m2=25.0 if c != gc.GRAVEL else 40.0):
            idx = len(cls_)
            for ring, is_hole in [(outer, False)] + [(r, True) for r in holes]:
                xy = np.stack([ring[:, 0] * gc.BAKE_CELL_M, (h - ring[:, 1]) * gc.BAKE_CELL_M], 1)
                pts.append(xy.astype(np.float32))
                cls_.append(c)
                off.append(off[-1] + len(xy))
                hole.append(idx if is_hole else -1)
    rings = np.concatenate(pts) if pts else np.zeros((0, 2), np.float32)
    return rings, np.asarray(cls_, np.uint8), np.asarray(off, np.int64), np.asarray(hole, np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", default=DEFAULT_TILES, help="2 km tiles as e_n (lower left km), comma separated")
    ap.add_argument("--out", default=gc.DATA)
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/beamng-earth"))
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--preview", default=None, help="folder for a class picture of the whole area (downscaled)")
    args = ap.parse_args()
    tiles = [tuple(int(v) for v in t.split("_")) for t in args.tiles.split(",") if t]
    t0 = time.time()
    classes, e0, n0 = bake(args.cache, tiles, not args.no_download)
    rings, ring_class, ring_off, ring_hole = outlines(classes, e0, n0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    buf = io.BytesIO()
    step = int(round(gc.CELL_M / gc.BAKE_CELL_M))
    np.savez_compressed(buf, version=np.int64(gc.VERSION), classes=classes[::step, ::step], e0=np.float64(e0),
                        n0=np.float64(n0),
                        rings=rings, ring_class=ring_class, ring_off=ring_off, ring_hole=ring_hole)
    with open(args.out, "wb") as f:
        f.write(buf.getvalue())
    share = np.bincount(classes.ravel(), minlength=len(gc.NAMES)) / classes.size
    print(f"{args.out}: {len(buf.getvalue()) / 1e6:.1f} MB, {len(ring_class)} rings, {time.time() - t0:.0f} s")
    print(", ".join(f"{n} {s:.1%}" for n, s in zip(gc.NAMES, share)))
    if args.preview:
        from PIL import Image
        os.makedirs(args.preview, exist_ok=True)
        Image.fromarray(PALETTE[classes[::6, ::6]]).save(os.path.join(args.preview, "classes.png"))


PALETTE = np.array([[0, 0, 0], [90, 90, 100], [240, 230, 150], [200, 60, 40], [250, 170, 60], [150, 100, 50],
                    [120, 220, 90], [40, 140, 40], [10, 70, 20], [255, 0, 255]], np.uint8)

if __name__ == "__main__":
    main()
