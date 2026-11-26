#!/usr/bin/env python3
"""tools/leonberg_profile_bench.py: edges and roughness of the road surface round Leonberg, offline, in absolute terms.

tests/drivability.py counts a Leonberg step only as its excess over the raw lidar's, so a ledge the lidar itself has (a
bridge's hole in the DTM, an interpolation seam, a wall) stays on the road unseen. This bench measures what a car meets:
the final ground (Grader.apply with ROAD_LIFT, as the physics block) along the centre line and the wheel tracks of every
road on the ground (its line where the road is drawn: roads.placement_context), against the class's limit, and how far
that ground is from the lidar (the counter-metric: smoothing must not destroy real heights).

    .venv/bin/python tools/leonberg_profile_bench.py --out DIR [--sites a,b,...] [--fixtures DIR] [--jobs N] [--tag T]

Sites: the Leonberg replays of tests/fixtures (leonberg_*) and the captures of tools/leonberg_edge_scan.py in --fixtures
(drive_<name>.npz). Writes DIR/<tag>.json (per site and class) and prints a table.

Metrics, z every metre along a track (1 m samples, the tracks at 0 and +-(half width - 0.8 m) off the line smoothed with
a 3 m Gaussian, as a car drives a sharp corner of the way):
  step     |z(s-1) - 2 z(s) + z(s+1)| m: a ledge or a kink (a steady grade has none)
  curv5    |z(s-5) - 2 z(s) + z(s+5)| / 25 1/m: the bend a wheelbase feels (1/curv5 the crest or sag radius)
  edges    m of track with step over 0.02 / 0.05 / 0.10 / 0.30 m (a 4 cm clearance car scrapes on ~0.05 m kinks)
  over     m of track with step over the class's limit (lidarsmooth.KAPPA: 1 / its minimum radius)
  rough    the RMS of step over all tracks (the "Ruckel": a washboard has many small steps)
  dev      |z - lidar| m on the centre line: p50, p95, max (ground parts only; a bridge's approach can be off the DTM)
  grade    the largest change of the 10 m grade against the lidar's (a real ramp kept: small)
"""
import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "server" / "tests"))

TRACK_IN_M = 0.8
PATH_SIGMA_M = 3.0      # the line a car drives: the road's line smoothed so (a sharp corner of the way as drawn: the
#                         wheel tracks across it would jump metres in one sample, a false edge)
EDGE_LIMITS = (0.02, 0.05, 0.10, 0.30)
FIXTURE_SITES = ("leonberg_dreieck", "leonberg_altstadt", "leonberg_a8_g3", "leonberg_a81_link")


def _class_limit(cls):
    import lidarsmooth
    return lidarsmooth.kappa(cls)


def extract(name, fixtures=None, radius=None):
    """[{fid, cls, width, xy (k, 2), raw (k, 3), z (k, 3)}] of every road part on the ground within the radius: the raw
    lidar and the final ground on the three tracks (left, centre, right), every metre."""
    import dnp
    import drivability
    import interchanges
    if fixtures:
        interchanges.FIXTURES = Path(fixtures)
        drivability.SITES.setdefault(name, ("drive", name))
    if radius:
        drivability.RADIUS_M = radius
    out = []
    with drivability.replay(name) as r:
        seen = set()
        for feats in r.cells:
            placement = r.roads.placement_context([feats])       # (the road where it is drawn, as grading's keeps)
            for i, spec in enumerate(r.roads.road_specs(feats)):
                if spec is None:
                    continue
                fid = str(feats.ids[i])
                for j, ring in enumerate(r.roads.ground_parts(feats, i)):
                    if (fid, j) in seen:
                        continue
                    seen.add((fid, j))
                    ring = placement.ring(r.site.frame, feats, i, np.asarray(ring, np.float64), spec)
                    e, n, _ = r.site.frame.geodetic_to_enu(ring[:, 1], ring[:, 0], 0.0)
                    c = drivability._resample(np.c_[np.asarray(e), np.asarray(n)])
                    if c is None:
                        continue
                    inside = drivability._inside(c)
                    if not inside.any():
                        continue
                    lo, hi = np.flatnonzero(inside)[[0, -1]]
                    c = c[max(lo - 10, 0):hi + 11]
                    if len(c) < 5:
                        continue
                    import grading
                    c = np.stack([grading.smooth_profile(c[:, 0], 1.0, PATH_SIGMA_M),
                                  grading.smooth_profile(c[:, 1], 1.0, PATH_SIGMA_M)], axis=1)
                    d = np.gradient(c, axis=0)
                    nor = np.c_[-d[:, 1], d[:, 0]] / np.maximum(dnp.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
                    side = max(spec.width / 2 - TRACK_IN_M, 0.0)
                    pts = [c + off * nor for off in (side, 0.0, -side)]
                    out.append(dict(fid=f"{fid}#{j}", cls=spec.cls, width=float(spec.width), xy=c,
                                    raw=np.stack([r.raw(p) for p in pts], axis=1),
                                    z=np.stack([r.ground(p) for p in pts], axis=1)))
    return out


def _run(job):
    name, fixtures, outdir, tag = job
    path = Path(outdir) / f"{tag}.{name}.pkl"
    if path.exists():
        return name, pickle.loads(path.read_bytes())
    try:
        parts = extract(name, fixtures)
    except Exception as e:                                  # one failed replay is no reason to lose the others
        return name, {"error": f"{type(e).__name__}: {e}"}
    path.write_bytes(pickle.dumps(parts))
    return name, parts


def _second(z, h):
    return np.abs(z[:-2 * h] - 2 * z[h:-h] + z[2 * h:])


def metrics(parts):
    """{class: {...}} and "all" over the parts (the module's docstring)."""
    acc = {}

    def bucket(cls):
        return acc.setdefault(cls, dict(m=0, edges=[0] * len(EDGE_LIMITS), over=0, sq=0.0, nstep=0, step_max=0.0,
                                         step_at=None, curv5_max=0.0, dev=[], grade=0.0))
    for p in parts:
        z, raw, xy = p["z"], p["raw"], p["xy"]
        if len(z) < 12:
            continue
        lim = _class_limit(p["cls"])
        for b in (bucket(p["cls"]), bucket("all")):
            b["m"] += len(z)
            for t in range(z.shape[1]):
                st = _second(z[:, t], 1)
                st = np.where(np.isfinite(st), st, 0.0)
                for k, e in enumerate(EDGE_LIMITS):
                    b["edges"][k] += int((st > e).sum())
                b["over"] += int((st > lim * 1.0 + 1e-9).sum())
                b["sq"] += float((st * st).sum())
                b["nstep"] += len(st)
                k = int(np.argmax(st))
                if st[k] > b["step_max"]:
                    b["step_max"], b["step_at"] = float(st[k]), [round(float(v), 1) for v in xy[k + 1]] + [p["fid"]]
                c5 = _second(z[:, t], 5) / 25.0
                if len(c5):
                    b["curv5_max"] = max(b["curv5_max"], float(np.nanmax(c5)))
            d = np.abs(z[:, 1] - raw[:, 1])
            b["dev"].append(d[np.isfinite(d)])
            if len(z) > 22:
                gz = (z[10:, 1] - z[:-10, 1]) / 10.0
                gr = (raw[10:, 1] - raw[:-10, 1]) / 10.0
                b["grade"] = max(b["grade"], float(np.nanmax(np.abs(gz - gr))))
    out = {}
    for cls, b in acc.items():
        dev = np.concatenate(b["dev"]) if b["dev"] else np.zeros(1)
        out[cls] = dict(m=b["m"], **{f"edges_{e:g}": v for e, v in zip(EDGE_LIMITS, b["edges"])}, over=b["over"],
                        rough=round((b["sq"] / max(b["nstep"], 1)) ** 0.5, 4), step_max=round(b["step_max"], 3),
                        step_at=b["step_at"], curv5_max=round(b["curv5_max"], 5),
                        dev_p50=round(float(np.percentile(dev, 50)), 3),
                        dev_p95=round(float(np.percentile(dev, 95)), 3),
                        dev_max=round(float(dev.max()), 3), grade=round(b["grade"], 4))
    return out


def table(res):
    cols = ["m"] + [f"edges_{e:g}" for e in EDGE_LIMITS] + ["over", "rough", "step_max", "dev_p95", "dev_max", "grade"]
    lines = [f"{'site':34} " + " ".join(f"{c:>10}" for c in cols)]
    for site, by in res.items():
        if "error" in by:
            lines.append(f"{site:34} {by['error']}")
            continue
        a = by.get("all")
        if a:
            lines.append(f"{site:34} " + " ".join(f"{a[c]:>10}" for c in cols))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sites", help="comma-separated (default: the fixtures' and every capture in --fixtures)")
    ap.add_argument("--fixtures", help="dir of drive_<name>.npz captures (tools/leonberg_edge_scan.py DIR/sites)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--tag", default="now", help="name of this run (env switches differ between runs)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.sites:
        names = a.sites.split(",")
    else:
        names = list(FIXTURE_SITES)
        if a.fixtures:
            names += sorted(p.stem[len("drive_"):] for p in Path(a.fixtures).glob("drive_*.npz"))
    jobs = [(n, None if n in FIXTURE_SITES else a.fixtures, str(out), a.tag) for n in names]
    res = {}
    with ProcessPoolExecutor(max(1, min(a.jobs, len(jobs)))) as pool:
        for name, parts in pool.map(_run, jobs):
            res[name] = parts if isinstance(parts, dict) else metrics(parts)
            print(f"{name}: " + (res[name].get("error") or "ok"), flush=True)
    allparts = []
    for name in names:
        got = pickle.loads((out / f"{a.tag}.{name}.pkl").read_bytes()) if (out / f"{a.tag}.{name}.pkl").exists() else []
        allparts += got
    res["TOTAL"] = metrics(allparts)
    (out / f"{a.tag}.json").write_text(json.dumps(res, indent=1))
    print("\n".join(table(res)))


if __name__ == "__main__":
    main()
