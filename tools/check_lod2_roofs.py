#!/usr/bin/env python3
"""Checks the LoD2 roofs of a Leonberg spot's buildings: ridge height, roof area, and gaps (LoD2 units never drawn).

    python3 tools/check_lod2_roofs.py [--spots T1,T3] [--radius 150] [--out DIR] [--cache DIR]

Per building with a LoD2 model (spec.lod2), built alone with buildings.build:
  lift_expected = max(0, (gmax - gmin) + 0.5 - e0), e0 = lowest z of the LoD2 roof polygons (the eave)
  ridge_mesh    = highest roof triangle z - gmin;  ridge_dev = ridge_mesh - lod2["top"]
  ridge_err     = ridge_dev - lift_expected          (wanted: |ridge_err| <= 0.05 m)
  area_ratio    = roof triangle area in the mesh / Newell area of the LoD2 roof polygons (wanted: 0.98 .. 1.02)
Gaps: LoD2 units with centre within radius that no drawn footprint took (Geoms._taken in no instance).
Writes <out>/<spot>_lod2roofs.csv (one row per building) and prints a summary. Exit status 0 (a measuring tool).
"""
import argparse
import csv
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "server"))

RIDGE_TOL = 0.05
AREA_LO, AREA_HI = 0.98, 1.02
LIFT_MARGIN = 0.5      # = buildings.LOD2_EAVE_CLEAR_M
COLUMNS = ["e", "n", "area", "top_lod", "eave_lod", "lift", "ridge_mesh", "ridge_dev", "ridge_err",
           "roof_area_mesh", "roof_area_lod", "area_ratio"]


def _eave0(L):
    """e0: the lowest z of the LoD2 roof polygons (kind 2), else the model's eave."""
    zs = [float(np.asarray(P, np.float64).reshape(-1, 3)[:, 2].min()) for kind, P in L["polys"]
          if kind == 2 and len(P) >= 3]
    return min(zs) if zs else float(L["eave"])


def lift_expected(L, gmin, gmax):
    """How far buildings.py raises the roof above the eave: z_eave = zb + e0 (Fix B: whole roof)."""
    return max(0.0, (gmax - gmin) + LIFT_MARGIN - _eave0(L))


def roof_metrics(mesh, spec, gmin, gmax):
    """The check's numbers for one building. mesh: the 6-tuple of buildings.build (or None when it is empty)."""
    import buildings as B
    L = spec.lod2
    lift = lift_expected(L, gmin, gmax)
    ridge_mesh, area_mesh = None, 0.0
    if mesh is not None:
        P, _uv, T, M, mats, _C = mesh
        P = np.asarray(P, np.float64).reshape(-1, 3)
        T = np.asarray(T, np.int64).reshape(-1, 3)
        M = np.asarray(M, np.int64).reshape(-1)
        is_roof = np.array([0 <= m < len(mats) and "roof" in str(mats[m]) for m in M], bool)
        if is_roof.any():
            tri = P[T[is_roof]]
            ridge_mesh = float(tri[:, :, 2].max()) - gmin
            area_mesh = float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                                                   axis=1).sum())
    area_lod = 0.0
    for kind, P in L["polys"]:
        if kind == 2 and len(P) >= 3:
            area_lod += float(np.linalg.norm(B._newell(np.asarray(P, np.float64).reshape(-1, 3))) / 2)
    ridge_dev = None if ridge_mesh is None else ridge_mesh - float(L["top"])
    ridge_err = None if ridge_dev is None else ridge_dev - lift
    return {"lift": lift, "ridge_mesh": ridge_mesh, "ridge_dev": ridge_dev, "ridge_err": ridge_err,
            "roof_area_mesh": area_mesh, "roof_area_lod": area_lod,
            "area_ratio": area_mesh / area_lod if area_lod > 0 else None}


def _footprint_area(ring):
    x, y = ring[:, 0], ring[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2


def _record_loads(lod2geom, sink):
    """Wrap Geoms.load so every instance built during the run is kept in sink (for the gap check)."""
    orig = lod2geom.Geoms.load.__func__

    def load(cls, *a, **k):
        g = orig(cls, *a, **k)
        if g is not None:
            sink.append(g)
        return g

    lod2geom.Geoms.load = classmethod(load)


def gaps(instances, radius):
    """LoD2 units with centre within radius that no instance took: [(id, cx, cy, area, code)], largest first."""
    seen, taken = {}, set()
    for g in instances:
        for i in range(g.n):
            if math.hypot(g.cx[i], g.cy[i]) > radius:
                continue
            u = g.unit(i)
            seen[u["id"]] = (u["area"], float(g.cx[i]), float(g.cy[i]), u["code"])
        for i in g._taken:
            taken.add(g.unit(int(i))["id"])
    out = [(k, v[1], v[2], v[0], v[3]) for k, v in seen.items() if k not in taken]
    return sorted(out, key=lambda r: -r[3])


def check_spot(st, spot, radius, instances):
    """The rows (one per LoD2 building within radius) of one spot, and the gap list."""
    import buildings as B
    import geodesy
    import roads
    import leonberg_buildings_report as R
    import preview_leonberg_buildings as PV

    sid, name, lat, lon = spot
    frame = geodesy.EnuFrame(lat, lon)
    k_lo, k_hi = math.floor(-radius / roads.TILE), math.floor(radius / roads.TILE)
    rows = []
    for tx in range(k_lo, k_hi + 1):
        for ty in range(k_lo, k_hi + 1):
            try:
                x0, y0 = roads.tile_bbox(tx, ty)[:2]
                items, _custom = B.road_tile_items(frame, tx, ty, st.md, R._ground(frame, st),
                                                   urban_fn=st.surface.urban if st.surface is not None else None,
                                                   widths=st.widths)
                for item in PV.within(items, x0, y0, radius):
                    spec, gmin, gmax = item[0], item[1], item[2]
                    if getattr(spec, "lod2", None) is None:
                        continue
                    ring = np.asarray(spec.ring, np.float64)[:, :2]
                    e, n = ring.mean(0) + (x0, y0)
                    m = roof_metrics(B.build([item]), spec, gmin, gmax)
                    m.update(e=float(e), n=float(n), area=_footprint_area(ring),
                             top_lod=float(spec.lod2["top"]), eave_lod=float(spec.lod2["eave"]))
                    rows.append(m)
            except Exception:
                logging.exception("tile %s,%s of %s failed", tx, ty, sid)
                print(f"{sid} tile {tx},{ty} FAILED, skipped", flush=True)
    return name, rows, gaps(instances, radius)


def _fmt(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def summarize(sid, name, rows, gap_list):
    ok = [r for r in rows if r["ridge_err"] is not None]
    errs = [abs(r["ridge_err"]) for r in ok]
    devs = [abs(r["ridge_dev"]) for r in ok]
    ratios = [r["area_ratio"] for r in rows if r["area_ratio"] is not None]
    print(f"{sid} {name}: {len(rows)} Units, ohne Dach-Dreiecke {len(rows) - len(ok)}", flush=True)
    print(f"  max |ridge_dev| {_fmt(max(devs, default=None))} m; max |ridge_err| {_fmt(max(errs, default=None))} m, "
          f"n mit |ridge_err| > {RIDGE_TOL}: {sum(e > RIDGE_TOL for e in errs)}", flush=True)
    print(f"  min area_ratio {_fmt(min(ratios, default=None), 3)}, n mit area_ratio < {AREA_LO}: "
          f"{sum(r < AREA_LO for r in ratios)}, n mit > {AREA_HI}: {sum(r > AREA_HI for r in ratios)}", flush=True)
    print(f"  Lücken (LoD2-Units im Radius nie gezeichnet): {len(gap_list)}", flush=True)
    for gid, cx, cy, area, code in gap_list[:10]:
        print(f"    {gid} {area:.0f} m2 code {code} bei ({cx:.0f}, {cy:.0f})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spots", default=None, help="comma list of spot ids (default all)")
    ap.add_argument("--radius", type=float, default=150.0)
    ap.add_argument("--out", default=os.path.expanduser("~/bnge-tests/leonberg/lod2roofs"))
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/beamng-earth"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    os.environ.setdefault("BNE_ROOF_PHOTO", "off")
    import leonberg_buildings_report as R
    import blockworker as bw
    import lod2geom

    spots = R.SPOTS
    if args.spots:
        want = [s.strip() for s in args.spots.split(",") if s.strip()]
        known = {s[0] for s in R.SPOTS}
        bad = [s for s in want if s not in known]
        if bad:
            ap.error(f"unknown spot(s): {', '.join(bad)}")
        spots = [s for s in R.SPOTS if s[0] in want]
    os.makedirs(args.out, exist_ok=True)
    st = bw._State(args.cache, False, True, True, True, True)
    bw._s = st
    instances = []
    _record_loads(lod2geom, instances)
    for spot in spots:
        sid = spot[0]
        instances.clear()
        try:
            name, rows, gap_list = check_spot(st, spot, args.radius, instances)
        except Exception as ex:
            logging.exception("%s failed", sid)
            print(f"{sid} FAILED {type(ex).__name__}: {ex}", flush=True)
            continue
        path = os.path.join(args.out, f"{sid}_lod2roofs.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: (None if r.get(k) is None else round(r[k], 4)) for k in COLUMNS})
        summarize(sid, name, rows, gap_list)
        print(f"  -> {path}", flush=True)


if __name__ == "__main__":
    main()
