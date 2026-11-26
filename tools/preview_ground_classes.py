#!/usr/bin/env python3
"""A picture of the baked ground classes around a point, next to the photo they come from (the before / after of the PR).

    python tools/preview_ground_classes.py --spot U3 [--radius 90] [--out DIR] [--data server/data/leonberg_ground.npz]
    python tools/preview_ground_classes.py --lat 48.79838 --lon 9.01353 --name rathaus

Left: the DOP20 photo (reads the LGL zips of <cache>/lgl as the bake tool does; --no-photo draws the classes only); with
--roads the mapped tracks / service roads / unclassified roads no tag describes, coloured by the photo's verdict
(groundclass.road_verdicts: asphalt dark blue, gravel orange, grass green, no verdict white; reads the map data, network).
Middle: the classes read back from the baked file (groundclass.GroundMap.at) over the photo.
Right: the outlines the paving mesh is built from (groundclass.paving_rings) over the photo: stone / red paving / gravel.
Last: the ground layers surface.classify paints from them on the 2 m lattice (land cover taken as built-up everywhere).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
sys.path.insert(0, HERE)

import groundclass as gc  # noqa: E402

SPOTS = {"U1": (48.80140, 9.01520), "U2": (48.79321, 9.01574), "U3": (48.79838, 9.01353), "U4": (48.79974, 9.00432),
         "U5": (48.79872, 9.02900), "U6": (48.79285, 8.99376), "U7": (48.78168, 9.01279), "U8": (48.79252, 9.03213),
         "T2": (48.80128, 9.01217), "T3": (48.80155, 9.01328), "FELD": (48.79758, 8.99471)}
OUTLINE = {gc.STONE: (255, 255, 0), gc.RED: (255, 40, 40), gc.GRAVEL: (255, 150, 0)}


LAYER_COLOURS = {  # surface.MATERIALS index -> colour
    0: (70, 70, 80), 1: (120, 200, 90), 2: (190, 160, 110), 3: (240, 200, 90), 4: (130, 130, 130), 5: (255, 255, 255),
    6: (110, 80, 50), 7: (150, 120, 130), 8: (80, 170, 70), 13: (225, 215, 150), 14: (200, 60, 40), 15: (150, 100, 50)}
VERDICT_COLOURS = {"asphalt": (0, 60, 220), "gravel": (255, 140, 0), "grass": (40, 220, 40), None: (255, 255, 255)}


def layer_panel(lat, lon, radius):
    """The physics layers of the 2 m lattice around the point as a picture of 4 px a sample."""
    import surface
    n = int(radius)                 # 2 m samples across 2 * radius m
    xs = np.linspace(-radius, radius, n, endpoint=False) + 1.0
    X, Y = np.meshgrid(xs, xs[::-1])
    import geodesy
    frame = geodesy.EnuFrame(lat, lon)
    la, lo, _ = frame.enu_to_geodetic(X.ravel(), Y.ravel(), np.zeros(X.size))
    la, lo = la.reshape(X.shape), lo.reshape(X.shape)
    idx = surface.classify(np.full(X.shape, 50, np.uint8), np.zeros(X.shape), 2.0, la, lo)
    rgb = np.zeros(idx.shape + (3,), np.uint8)
    for k, c in LAYER_COLOURS.items():
        rgb[idx == k] = c
    return rgb


def road_lines(draw, lat, lon, e0, n0, size):
    """Draws the roads gc.road_verdicts looks at (the map's roads of ROAD_CLASSES) on draw, in the window's pixels."""
    import geodesy
    import mapcells
    import blockworker as bw
    import lod2geom
    frame = geodesy.EnuFrame(lat, lon)
    st = bw._State(os.path.expanduser("~/.cache/beamng-earth"), False, True, True, True, True)
    bw._s = st
    la, lo, _ = frame.enu_to_geodetic(np.array([-size, -size, size, size], float), np.array([-size, size, -size, size], float),
                                      np.zeros(4))
    for key in mapcells.cells_for_points(la, lo, pad_lat=0.00005, pad_lon=0.00005, max_cells=16):
        f = st.md.cell(key, layers=("roads",), kind="demand").layers.get("roads")
        if f is None or not len(f.kind):
            continue
        v = gc.road_verdicts(f, lambda i: None)
        cls = f.attrs.get("class")
        for i in range(len(f.kind)):
            if str(cls[i]) not in gc.ROAD_CLASSES:
                continue
            for part in f.parts(i):
                ring = np.asarray(part[0], np.float64)
                e, n = lod2geom.utm32_from_wgs(ring[:, 1], ring[:, 0])
                pts = [((float(a) - e0) / gc.BAKE_CELL_M, (n0 + size - float(b)) / gc.BAKE_CELL_M) for a, b in zip(e, n)]
                draw.line(pts, fill=VERDICT_COLOURS[v.get(i)], width=2)


def render(gm, mos, lat, lon, radius, with_photo=True, roads=False):
    """Four panels side by side: photo (with the road verdicts), classes, paving outlines, ground layers."""
    from PIL import Image, ImageDraw
    import bake_ground_classes as bake
    e, n = (float(v) for v in gc._utm32(lat, lon))
    e0, n0 = int(e - radius) // 2 * 2, int(n - radius) // 2 * 2
    size = int(radius * 2) // 2 * 2
    cells = int(round(size / gc.BAKE_CELL_M))
    ee = e0 + (np.arange(cells) + 0.5) * gc.BAKE_CELL_M
    nn = n0 + size - (np.arange(cells) + 0.5) * gc.BAKE_CELL_M
    cls = gm.at(ee[None, :].repeat(cells, 0), nn[:, None].repeat(cells, 1))
    if with_photo and mos is not None:
        rgbi, _ = bake.window(mos, e0, n0, size, size)
        base = np.asarray(Image.fromarray(rgbi[..., :3]).resize((cells, cells), Image.BILINEAR))
    else:
        base = np.full((cells, cells, 3), 90, np.uint8)
    mid = (0.45 * base + 0.55 * bake.PALETTE[cls]).astype(np.uint8)
    left = Image.fromarray(base.copy())
    if roads:
        road_lines(ImageDraw.Draw(left), lat, lon, e0, n0, size)
    right = Image.fromarray(base.copy())
    d = ImageDraw.Draw(right)
    for outer, holes, c in gm.rings_in(e0, n0, e0 + size, n0 + size):
        for ring, w in [(outer, 2)] + [(h, 1) for h in holes]:
            pts = [((x - e0) / gc.BAKE_CELL_M, (n0 + size - y) / gc.BAKE_CELL_M) for x, y in ring]
            d.line(pts + [pts[0]], fill=OUTLINE.get(c, (255, 255, 255)), width=w)
    layers = np.asarray(Image.fromarray(layer_panel(lat, lon, size / 2)).resize((cells, cells), Image.NEAREST))
    return np.concatenate([np.asarray(left), mid, np.asarray(right), layers], 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spot", default=None, help="U1..U8, T2, T3 or FELD")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--name", default=None)
    ap.add_argument("--radius", type=float, default=90.0)
    ap.add_argument("--data", default=gc.DATA)
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/beamng-earth"))
    ap.add_argument("--out", default=os.path.expanduser("~/bnge-tests/leonberg/ground"))
    ap.add_argument("--no-photo", action="store_true")
    ap.add_argument("--roads", action="store_true", help="draw the road verdicts on the photo (reads the map data)")
    args = ap.parse_args()
    if args.spot:
        lat, lon = SPOTS[args.spot.upper()]
        name = args.name or args.spot.lower()
    else:
        lat, lon, name = args.lat, args.lon, args.name or "spot"
    gm = gc.load(args.data)
    if gm is None:
        sys.exit(f"no ground classes in {args.data}")
    mos = None
    if not args.no_photo:
        import bake_ground_classes as bake
        mos = bake.Mosaic(args.cache, download=False)
    from PIL import Image
    os.makedirs(args.out, exist_ok=True)
    img = render(gm, mos, lat, lon, args.radius, not args.no_photo, args.roads)
    path = os.path.join(args.out, f"{name}.png")
    Image.fromarray(img).save(path)
    print(path)


if __name__ == "__main__":
    main()
