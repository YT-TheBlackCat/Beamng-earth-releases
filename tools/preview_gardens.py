#!/usr/bin/env python3
"""Offline before/after previews of the baked gardens at a test spot (no game needed).

    .venv/bin/python tools/preview_gardens.py [--spot U8 | --lat LAT --lon LON] [--radius 100] [--cache DIR] [--out DIR]

Writes <name>_luftbild.png, _klassen.png, _vorher.png, _nachher.png and _vorher_nachher.png (name = the spot id, or
pos_<lat>_<lon>). The view is the UTM square of +-radius m around the spot, north up, at 0.2 m per pixel. Reads the
recipes (gardens/v1), the DOP20 and ALKIS sources of the cache (~/.cache/beamng-earth); no network.
"""
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import gardenbake  # noqa: E402
import gardens  # noqa: E402
import leonberg_surroundings_report as report  # noqa: E402
import lod2geom  # noqa: E402
import surface  # noqa: E402

PX = gardenbake.DOP_PX_M                    # m per pixel of the view (the aerial's)
CLASS_RGB = np.array([(0, 0, 0), (150, 230, 110), (150, 100, 50), (60, 140, 60), (230, 170, 120), (200, 200, 200),
                      (255, 255, 140), (240, 230, 190), (60, 120, 255), (90, 90, 90)], np.float32)  # G_NAMES order
BUILDING_RGB = (90, 90, 90)
HEDGE_RGB, TREE_RGB, SHRUB_RGB = (0, 90, 0), (255, 0, 0), (0, 80, 255)
TREE_FILL, TREE_EDGE, SHRUB_FILL = (40, 110, 40), (20, 60, 20), (70, 150, 70)
MAT_COLOURS = {"LAWN": (120, 190, 90), "GRASS": (90, 160, 60), "DUSTY": (140, 120, 70), "LOOSE": (215, 205, 170),
               "PAVERS": (185, 175, 165), "ASPHALT": (110, 110, 110), "DIRT": (160, 135, 100)}
MAT_LUT = np.full((len(surface.MATERIALS), 3), (255, 0, 255), np.uint8)
MAT_NAMES = {}
for _name, _rgb in MAT_COLOURS.items():
    MAT_LUT[getattr(surface, _name)] = _rgb
    MAT_NAMES[getattr(surface, _name)] = _name


def squares(E, N):
    """The km squares (e_km, n_km) the UTM points reach."""
    return [(ek, nk) for ek in range(int(math.floor(E.min() / 1000)), int(math.floor(E.max() / 1000)) + 1)
            for nk in range(int(math.floor(N.min() / 1000)), int(math.floor(N.max() / 1000)) + 1)]


def ground_classes(E, N, cache):
    """Recipe ground class (gardenbake.G_*) per UTM point, read from the recipe arrays (0.5 m px)."""
    out = np.zeros(E.shape, np.uint8)
    for ek, nk in squares(E, N):
        d = gardens.Recipes.get(cache, ek, nk)
        if d is None:
            continue
        g, px = d["ground"], float(d["px"])
        sel = (E >= ek * 1000) & (E < (ek + 1) * 1000) & (N >= nk * 1000) & (N < (nk + 1) * 1000)
        col = np.clip(((E[sel] - ek * 1000) / px).astype(np.int64), 0, g.shape[1] - 1)
        row = np.clip((((nk + 1) * 1000 - N[sel]) / px).astype(np.int64), 0, g.shape[0] - 1)
        out[sel] = g[row, col]
    return out


def recipe_features(sq, cache):
    """Hedges [(UTM points (k,2), width m)], trees and shrubs (n,4) as UTM x, y, height, radius m, of the squares."""
    hedges, trees, shrubs = [], [], []
    for ek, nk in sq:
        d = gardens.Recipes.get(cache, ek, nk)
        if d is None:
            continue
        off = np.array([ek * 1000.0, nk * 1000.0])
        xy, st = d["hedge_xy"].astype(np.float64), d["hedge_start"]
        for i in range(len(st) - 1):
            if st[i + 1] - st[i] >= 2:
                hedges.append((xy[st[i]:st[i + 1]] + off, float(d["hedge_w"][i])))
        for key, out in (("tree", trees), ("shrub", shrubs)):
            a = d[key].astype(np.float64)
            if len(a):
                a = a[:, :4].copy()
                a[:, :2] += off
                out.append(a)
    trees = np.concatenate(trees) if trees else np.zeros((0, 4))
    shrubs = np.concatenate(shrubs) if shrubs else np.zeros((0, 4))
    return hedges, trees, shrubs


def draw_features(img, view, feats, filled):
    """Hedges as lines, trees and shrubs as circles: outlines (filled=False) or filled (the gardens' planting)."""
    e0, n1 = view
    hedges, trees, shrubs = feats
    dr = ImageDraw.Draw(img)

    def to_px(E, N):
        return (np.asarray(E, np.float64) - e0) / PX, (n1 - np.asarray(N, np.float64)) / PX

    for pts, w in hedges:
        c, r = to_px(pts[:, 0], pts[:, 1])
        dr.line(list(zip(c.tolist(), r.tolist())), fill=HEDGE_RGB, width=int(max(2, round(w / PX))), joint="curve")
    for x, y, _h, rad in trees:
        c, r = (float(v) for v in to_px(x, y))
        R = rad / PX
        box = [c - R, r - R, c + R, r + R]
        if filled:
            dr.ellipse(box, fill=TREE_FILL, outline=TREE_EDGE, width=2)
        else:
            dr.ellipse(box, outline=TREE_RGB, width=2)
    for x, y, _h, rad in shrubs:
        c, r = (float(v) for v in to_px(x, y))
        R = rad / PX
        box = [c - R, r - R, c + R, r + R]
        if filled:
            dr.ellipse(box, fill=SHRUB_FILL)
        else:
            dr.ellipse(box, outline=SHRUB_RGB, width=1)


def draw_legend(img):
    font = ImageFont.load_default()
    rows = [(name, tuple(int(v) for v in CLASS_RGB[i]), "box") for i, name in enumerate(gardenbake.G_NAMES) if i]
    rows += [("Hecke", HEDGE_RGB, "line"), ("Baum", TREE_RGB, "ring"), ("Strauch", SHRUB_RGB, "ring")]
    dr = ImageDraw.Draw(img)
    dr.rectangle([6, 6, 190, 6 + 18 * len(rows) + 10], fill=(255, 255, 255), outline=(60, 60, 60))
    for i, (label, colour, kind) in enumerate(rows):
        y = 12 + 18 * i
        if kind == "box":
            dr.rectangle([12, y, 30, y + 12], fill=colour, outline=(0, 0, 0))
        elif kind == "line":
            dr.line([(12, y + 6), (30, y + 6)], fill=colour, width=4)
        else:
            dr.ellipse([12, y, 30, y + 12], outline=colour, width=2)
        dr.text((38, y), label, fill=(0, 0, 0), font=font)


def material_rgb(idx, bld):
    rgb = MAT_LUT[idx]
    rgb[bld] = BUILDING_RGB
    return Image.fromarray(rgb)


def side_by_side(vor, nach, path):
    font = ImageFont.load_default()
    w, h = vor.width, vor.height
    canvas = Image.new("RGB", (2 * w + 10, h + 28), (255, 255, 255))
    canvas.paste(vor, (0, 28))
    canvas.paste(nach, (w + 10, 28))
    dr = ImageDraw.Draw(canvas)
    dr.text((6, 8), "Vorher (heute: Rasen-Zufallsflecken)", fill=(0, 0, 0), font=font)
    dr.text((w + 16, 8), "Nachher (Gaerten aus Luftbild)", fill=(0, 0, 0), font=font)
    canvas.save(path)


def shares(idx, mask):
    n = np.bincount(idx[mask].ravel(), minlength=len(surface.MATERIALS))
    tot = max(1, int(n.sum()))
    named = [(MAT_NAMES[i], int(n[i])) for i in MAT_NAMES]
    named.append(("andere", int(n.sum()) - sum(c for _, c in named)))
    return ", ".join(f"{k} {100.0 * c / tot:.1f} %" for k, c in sorted(named, key=lambda p: -p[1]) if c)


def main():
    ap = argparse.ArgumentParser(description="Before/after previews of the baked gardens at a test spot.")
    ap.add_argument("--spot", help="spot id from tools/leonberg_surroundings_report.SPOTS (default U8)")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--radius", type=float, default=100.0, help="half size of the square, m")
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/beamng-earth"))
    ap.add_argument("--out", default=os.path.expanduser("~/bnge-tests/leonberg/gardens"))
    a = ap.parse_args()
    if (a.lat is None) != (a.lon is None):
        ap.error("--lat und --lon gehören zusammen")
    if a.lat is not None:
        name, lat, lon = f"pos_{a.lat:.5f}_{a.lon:.5f}", a.lat, a.lon
    else:
        spots = {s[0].upper(): s for s in report.SPOTS}
        s = spots.get((a.spot or "U8").upper())
        if s is None:
            print(f"unbekannter Spot {a.spot}; bekannt: {', '.join(spots)}", file=sys.stderr)
            return 2
        name, lat, lon = s[0], s[2], s[3]
    bw = lod2geom.BW
    if not (bw[0] <= lat <= bw[2] and bw[1] <= lon <= bw[3]):
        print(f"{name}: ausserhalb Baden-Württemberg (lat {lat}, lon {lon})", file=sys.stderr)
        return 2
    E, N = (float(v) for v in lod2geom.utm32_from_wgs(lat, lon))
    ek, nk = int(E // 1000), int(N // 1000)
    if gardens.Recipes.get(a.cache, ek, nk) is None:
        print(f"{name}: kein Rezept für das Quadrat {ek}_{nk} in {gardens.folder(a.cache)} - "
              f"erst tools/bake_gardens.py --spots {name} laufen lassen", file=sys.stderr)
        return 2
    if not gardens.GROUND_ON:
        print("Warnung: BNE_GARDEN_GROUND=off, die Gärten werden nicht gemalt")

    r = float(a.radius)
    n = int(round(2 * r / PX))
    e0, n0 = round((E - r) / PX) * PX, round((N - r) / PX) * PX
    e1, n1 = e0 + n * PX, n0 + n * PX
    os.makedirs(a.out, exist_ok=True)
    E2, N2 = np.meshgrid(e0 + (np.arange(n) + 0.5) * PX, n1 - (np.arange(n) + 0.5) * PX)
    lat2, lon2 = lod2geom.wgs_from_utm32(E2, N2)

    m = gardenbake.Mosaic(os.path.join(a.cache, "lgl"), e0, n0, e1, n1)
    aerial = m.rgbi[:, :, :3].copy() if m.valid.any() else np.full((n, n, 3), 128, np.uint8)

    alkis = os.path.join(a.cache, "lgl", "alkis")
    zips = [z for z in gardenbake.alkis_zips_over(e0, n0, e1, n1) if os.path.exists(os.path.join(alkis, z))]
    parcels, buildings = [], []
    if not zips:
        print(f"Warnung: keine ALKIS-Zips in {alkis}; Straßen und Gebäude fehlen")
    for z in zips:
        p, b = gardenbake.read_alkis(os.path.join(alkis, z), box=(e0, n0, e1, n1))
        parcels += p
        buildings += b
    road = gardenbake.rasterize((n, n), [p["rings"] for p in parcels if gardenbake.is_road(p["use"])], e0, n1) > 0
    bld = gardenbake.rasterize((n, n), [b["rings"] for b in buildings], e0, n1) > 0
    valid = ~bld & ~road

    # vorher: today's terrain materials (lawns on built-up land, dirt elsewhere), roads asphalt
    idx = np.where(surface.lawns(lat2, lon2), surface.LAWN, surface.DIRT).astype(np.uint8)
    idx[road] = surface.ASPHALT
    vor_img = material_rgb(idx, bld)
    # nachher: the same, with the gardens painted in
    idx2 = gardens.paint(idx.copy(), lat2, lon2, a.cache)
    nach_img = material_rgb(idx2, bld)

    view = (e0, n1)
    feats = recipe_features(squares(E2, N2), a.cache)
    cls = ground_classes(E2, N2, a.cache)
    dim = aerial.astype(np.float32) * 0.5
    sel = cls > 0
    dim[sel] = 0.3 * dim[sel] + 0.7 * CLASS_RGB[cls[sel]]
    klass_img = Image.fromarray(dim.clip(0, 255).astype(np.uint8))
    draw_features(klass_img, view, feats, filled=False)
    draw_legend(klass_img)
    draw_features(nach_img, view, feats, filled=True)

    outs = {"luftbild": Image.fromarray(aerial), "klassen": klass_img, "vorher": vor_img, "nachher": nach_img}
    for key, img in outs.items():
        img.save(os.path.join(a.out, f"{name}_{key}.png"))
    side_by_side(vor_img, nach_img, os.path.join(a.out, f"{name}_vorher_nachher.png"))

    hedges, trees, shrubs = feats

    def inside(x, y):
        return (x >= e0) & (x < e1) & (y >= n0) & (y < n1)

    nh, hl = 0, 0.0
    for pts, _w in hedges:
        mid = (pts[:-1] + pts[1:]) / 2
        seg = np.hypot(*np.diff(pts, axis=0).T)
        ins = inside(mid[:, 0], mid[:, 1])
        if ins.any():
            nh += 1
            hl += float(seg[ins].sum())
    print(f"{name} (lat {lat:.5f}, lon {lon:.5f}): {2 * r:.0f} m, {n} x {n} px, Mosaik gültig {m.valid.mean() * 100:.0f} %")
    print(f"Vorher (ohne Gebäude/Straßen): {shares(idx, valid)}")
    print(f"Nachher (ohne Gebäude/Straßen): {shares(idx2, valid)}")
    print(f"In der Sicht: {nh} Hecken ({hl:.0f} m), {int(inside(trees[:, 0], trees[:, 1]).sum())} Bäume, "
          f"{int(inside(shrubs[:, 0], shrubs[:, 1]).sum())} Sträucher")
    print(f"PNGs in {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
