"""Offline preview of the painted lane lines at a spot: what the plan wants, what is drawn, what is lost.

    .venv/bin/python tools/preview_lane_markings.py --spot 48.79360,9.00000 --radius 90 --out /tmp/b1.png
    .venv/bin/python tools/preview_lane_markings.py --spot ... --set junctions.X=1 --out after.png

Grey: carriageways. White: centre / lane / edge lines that are drawn. Red: centre / lane lines the plan has but the
build removed (a junction cut, another road, a stub). Needs the Overture cells of the spot (online). Docs:
docs/leonberg/kreuzungen/SPUREN.md.
"""
import argparse
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "server"), str(ROOT / "server" / "tests"), str(ROOT / "tools")]


def _segments(parts):
    for p in parts:
        for a, b in zip(p[:-1], p[1:]):
            yield a, b


def lost_pieces(plan, drawn, step=0.5):
    """Sub-segments of the planned polylines more than 0.8 m from every drawn piece: [(a, b)]. Pure (numpy)."""
    import numpy as np
    drawn_pts = []
    for p in drawn:
        p = np.asarray(p, np.float64)[:, :2]
        for a, b in zip(p[:-1], p[1:]):
            n = max(2, int(np.hypot(*(b - a)) / step) + 1)
            drawn_pts.append(np.linspace(a, b, n))
    D = np.vstack(drawn_pts) if drawn_pts else np.zeros((0, 2))
    out = []
    for p in plan:
        p = np.asarray(p, np.float64)[:, :2]
        for a, b in zip(p[:-1], p[1:]):
            n = max(2, int(np.hypot(*(b - a)) / step) + 1)
            q = np.linspace(a, b, n)
            miss = np.ones(len(q), bool) if not len(D) else \
                (np.hypot(q[:, None, 0] - D[None, :, 0], q[:, None, 1] - D[None, :, 1]).min(1) > 0.8)
            i = 0
            while i < len(q) - 1:
                if miss[i] and miss[i + 1]:
                    out.append((q[i], q[i + 1]))
                i += 1
    return out


def unpainted_roads(run):
    """[(pts, spec)] of the roads of a centre-line class (roads.CENTRE_CLASSES) that are not one-way, with a width over
    5.5 m, where no centre / lane line is drawn within half the road's width. Roads of one lane or in a 30 zone are
    not told apart here: the overview shows where to look, the tags tell why."""
    import numpy as np
    from scipy.spatial import cKDTree
    import roads
    pts = []
    for diag in run.diags:
        for rec in diag.get("lines", []):
            if rec["kind"] in ("centre", "lane"):
                for p in rec["stages"]["drawn"]:
                    p = np.asarray(p)[:, :2]
                    for a, b in zip(p[:-1], p[1:]):
                        pts.append(np.linspace(a, b, max(2, int(np.hypot(*(b - a)) / 2.0) + 1)))
    tree = cKDTree(np.vstack(pts)) if pts else None
    out = []
    for diag in run.diags:
        for road in diag.get("roads", ()):
            spec, line = road[1], np.asarray(road[2])[:, :2]
            if spec.cls not in roads.CENTRE_CLASSES or spec.oneway or spec.width < 5.5 or len(line) < 2:
                continue
            L = np.hypot(*np.diff(line, axis=0).T)
            for (a, b), n in zip(zip(line[:-1], line[1:]), L):
                q = np.linspace(a, b, max(2, int(n / 3.0) + 1))
                near = tree.query(q)[0] if tree is not None else np.full(len(q), 99.0)
                if (near > spec.width / 2 + 1.0).all():
                    out.append((np.array([a, b]), spec))
    return out


def render(run, out, size=1400, show_edges=False, overview=False):
    import numpy as np
    from PIL import Image, ImageDraw
    cx, cy = run.origin
    r = run.radius
    scale = size / (2 * r)

    def px(p):
        return ((p[0] - cx) * scale + size / 2, size / 2 - (p[1] - cy) * scale)
    im = Image.new("RGB", (size, size), (62, 74, 62))
    dr = ImageDraw.Draw(im)
    for diag in run.diags:                      # (every tile's piece of a road: run.roads keeps only the first)
        for road in diag.get("roads", ()):
            spec, pts = road[1], np.asarray(road[2])
            dr.line([px(p) for p in pts], fill=(96, 96, 100), width=max(2, int(spec.width * scale)), joint="curve")
    lost_n = 0
    for diag in run.diags:
        for rec in diag.get("lines", []):
            kind = rec["kind"]
            if kind == "edge" and not show_edges:
                continue
            st = rec["stages"]
            w = max(1, int(0.15 * scale))
            for a, b in _segments(st["drawn"]):
                dr.line([px(a), px(b)], fill=(240, 240, 240), width=w)
            if kind in ("centre", "lane") and not rec["dash"]:
                for a, b in lost_pieces(st["tile"], st["drawn"]):
                    dr.line([px(a), px(b)], fill=(230, 40, 40), width=w + 1)
                    lost_n += 1
    if overview:
        for seg, spec in unpainted_roads(run):
            dr.line([px(p) for p in seg], fill=(255, 150, 0), width=max(3, int(spec.width * scale * 0.5)))
    im.save(out)
    return lost_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spot", required=True, help="lat,lon")
    ap.add_argument("--radius", type=float, default=90.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--edges", action="store_true")
    ap.add_argument("--overview", action="store_true", help="orange: two-way roads of a centre-line class with no line")
    ap.add_argument("--size", type=int, default=1400)
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/beamng-earth"))
    ap.add_argument("--urban", default="all")
    args = ap.parse_args()
    lat, lon = (float(v) for v in args.spot.split(","))
    os.environ["BNGE_PLANCACHE"] = "off"
    os.environ["BNE_REGIONAL_DIR"] = str(ROOT / "server" / "tests" / "fixtures" / "regional")
    import mapdata
    import netfetch
    import leonberg_junction_core as core
    import leonberg_junction_probe as P
    md = mapdata.MapData(args.cache, fetcher=netfetch.Fetcher())
    a = SimpleNamespace(radius=args.radius, cache=args.cache, urban=args.urban, no_widths=False)
    lat, lon, roads, infra, routes, urban_fn, widths, _ = P.online_inputs(lat, lon, a, md)
    frame, origin = core.build_frame(lat, lon)
    run = core.run(frame, origin, args.radius, roads, infra, routes, urban_fn=urban_fn, widths=widths)
    n = render(run, args.out, size=args.size, show_edges=args.edges, overview=args.overview)
    print(f"{args.out}: {n} lost centre/lane pieces")


if __name__ == "__main__":
    main()
