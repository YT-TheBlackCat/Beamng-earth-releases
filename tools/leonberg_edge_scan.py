#!/usr/bin/env python3
"""tools/leonberg_edge_scan.py: scan the roads round Leonberg for edges (steps) a car cannot drive over, offline.

A grid of sites over Leonberg is captured (as tools/capture_interchange_fixture.py does, into a scratch dir, never the
repo), each site is replayed as the game plans it and measured with tests/drivability.py (step_m: a ledge or kink on the
wheel tracks, grade, lift_g). Every part with a step over --min-step is listed with its place (lat, lon), surface, class
and size, the worst first; sites overlap, so the same edge is merged.

    .venv/bin/python tools/leonberg_edge_scan.py --out DIR [--spacing 450] [--radius 320] [--min-step 0.3]
        [--bbox S,W,N,E] [--jobs N] [--capture-only | --measure-only]

Writes DIR/edges.md (the table), DIR/edges.json and DIR/sites/*.npz (the captures: reruns reuse them).
Severity: a step of 0.3 m or more is a kerb a car stops at (blocking from ~0.5 m).
"""
import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "server" / "tests"))
sys.path.insert(0, str(ROOT / "tools"))

BBOX = (48.778, 8.980, 48.812, 9.045)    # S, W, N, E: Leonberg, A8, A81 and the Dreieck


def sites(bbox, spacing):
    s, w, n, e = bbox
    dlat = spacing / 111320.0
    out = []
    la, row = s, 0
    while la <= n:
        dlon = spacing / (111320.0 * math.cos(math.radians(la)))
        lo = w + (dlon / 2 if row % 2 else 0)
        col = 0
        while lo <= e:
            out.append((f"leonberg_scan_{row:02d}_{col:02d}", round(la, 6), round(lo, 6)))
            lo += dlon
            col += 1
        la += dlat
        row += 1
    return out


def capture(job):
    name, lat, lon, outdir, cache = job
    out = Path(outdir) / f"drive_{name}.npz"
    if out.exists():
        return name, "cached"
    import capture_interchange_fixture as cap
    import dem as demmod
    import geodesy
    import mapcells
    import mapdata as mapdatamod
    import netfetch
    net = netfetch.Fetcher()
    md = mapdatamod.MapData(cache, fetcher=net)
    dem = demmod.Dem(str(Path(cache) / "dem"), fetcher=net)
    pad = 1200.0 / 111320.0
    keys = sorted(mapcells.cells_for_points(np.array([lat]), np.array([lon]), pad,
                                            pad / np.cos(np.radians(lat))), key=str)
    data = {"origin": np.array([lat, lon]), "cells": np.array([str(k) for k in keys]),
            "release": np.array(md.release), "dem_zoom": np.array(demmod.FINE_ZOOM)}
    for i, key in enumerate(keys):
        f = md.cell(key, layers=("roads",)).layers.get("roads")
        if f is not None and len(f):
            data.update(f.to_arrays(f"cell{i}."))
    frame = geodesy.EnuFrame(lat, lon)

    def sample(la, lo):
        return dem.sample(la, lo, demmod.FINE_ZOOM)
    data["fine_axis"], data["fine_z"] = cap.grid(frame, sample, cap.FINE_M, cap.FINE_STEP_M)
    data["coarse_axis"], data["coarse_z"] = cap.grid(frame, sample, cap.COARSE_M, cap.COARSE_STEP_M)
    tmp = out.with_suffix(".tmp.npz")
    cap.save(tmp, data)
    tmp.rename(out)
    return name, "captured"


def measure_site(job):
    name, lat, lon, outdir, radius = job
    import drivability
    import interchanges
    interchanges.FIXTURES = Path(outdir)
    drivability.SITES[name] = ("drive", name)
    drivability.RADIUS_M = radius
    try:
        res = drivability.measure(name)
    except Exception as e:      # a site with no roads or a failed replay is no reason to lose the others
        return {"site": name, "error": f"{type(e).__name__}: {e}", "lat": lat, "lon": lon}
    res.update(lat=lat, lon=lon)
    return res


def edges(results, min_step):
    """One row per edge: (step, lat, lon, surface, class, part, site), merged over overlapping sites (within 12 m)."""
    rows = []
    for r in results:
        if "parts" not in r:
            continue
        for key, part in r["parts"].items():
            step, at = part.get("step_m", 0.0), part.get("step_m_at")
            if step < min_step or not at:
                continue
            lat = r["lat"] + at[1] / 111320.0
            lon = r["lon"] + at[0] / (111320.0 * math.cos(math.radians(r["lat"])))
            surface, _, fid = key.partition(":")
            rows.append(dict(step=round(step, 2), lat=round(lat, 6), lon=round(lon, 6), surface=surface,
                             cls=part.get("class"), part=fid[:36], site=r["site"]))
    rows.sort(key=lambda x: -x["step"])
    kept = []
    for x in rows:
        if any(abs(x["lat"] - k["lat"]) * 111320 < 12 and abs(x["lon"] - k["lon"]) * 74000 < 12 for k in kept):
            continue
        kept.append(x)
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spacing", type=float, default=450.0, help="m between sites")
    ap.add_argument("--radius", type=float, default=320.0, help="m of roads measured round a site")
    ap.add_argument("--min-step", type=float, default=0.3)
    ap.add_argument("--bbox", help="S,W,N,E")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--capture-only", action="store_true")
    ap.add_argument("--measure-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    sdir = out / "sites"
    sdir.mkdir(parents=True, exist_ok=True)
    bbox = tuple(float(v) for v in a.bbox.split(",")) if a.bbox else BBOX
    grid = sites(bbox, a.spacing)
    print(f"{len(grid)} sites, spacing {a.spacing:.0f} m, radius {a.radius:.0f} m", flush=True)
    if not a.measure_only:
        jobs = [(n, la, lo, str(sdir), str(out / "cache")) for n, la, lo in grid]
        with ProcessPoolExecutor(min(a.jobs, 4)) as pool:     # the fetcher is polite: few at once
            for i, (name, what) in enumerate(pool.map(capture, jobs), 1):
                print(f"[{i}/{len(grid)}] {name} {what}", flush=True)
    if a.capture_only:
        return
    jobs = [(n, la, lo, str(sdir), a.radius) for n, la, lo in grid]
    results = []
    with ProcessPoolExecutor(a.jobs) as pool:
        for i, r in enumerate(pool.map(measure_site, jobs), 1):
            results.append(r)
            print(f"[{i}/{len(grid)}] {r['site']} " + (r.get("error") or f"steps {r['steps']} m"), flush=True)
    rows = edges(results, a.min_step)
    errors = [r for r in results if "error" in r]
    (out / "edges.json").write_text(json.dumps({"edges": rows, "errors": errors}, indent=1))
    md = [f"# Kanten auf Straßen, Leonberg (Stufe >= {a.min_step} m)", "",
          f"{len(grid)} Messpunkte, Raster {a.spacing:.0f} m, Radius {a.radius:.0f} m, {len(errors)} Fehler. "
          f"{sum(r['step'] >= 0.5 for r in rows)} Kanten >= 0,5 m (unbefahrbar), {len(rows)} insgesamt.", "",
          "| Stufe m | Lat, Lon | Fläche | Klasse | Straßen-ID |", "|---|---|---|---|---|"]
    md += [f"| {x['step']:.2f} | {x['lat']:.5f}, {x['lon']:.5f} | {x['surface']} | {x['cls']} | {x['part']} |"
           for x in rows]
    if errors:
        md += ["", "Fehler:"] + [f"- {r['site']}: {r['error']}" for r in errors]
    (out / "edges.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:30]))


if __name__ == "__main__":
    main()
