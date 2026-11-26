#!/usr/bin/env python3
"""tools/leonberg_elevation_probe.py: the ground the server builds (DEM shaped + road grading) against the raw lidar DEM
at a few spots, offline. Per spot: metrics (diff, artificial edges, artificial waviness), hillshade / diff / edge
images (PNG, cells upscaled 4x) and <name>.json; a summary table in <out>/summary.md.

    cd server && python ../tools/leonberg_elevation_probe.py            (or from the repo root with .venv python)
    python tools/leonberg_elevation_probe.py --spots "X:48.8,9.0;Y:48.81,9.02" --half 30 --out ~/tmp/probe
"""
import argparse
import ast
import csv
import importlib
import json
import logging
import multiprocessing
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

DEFAULT_SPOTS = [
    ("R1", 48.798180, 9.014340), ("R3", 48.800550, 9.016777), ("R4", 48.801977, 9.016582),
    ("R5", 48.801257, 9.016607), ("R6", 48.802486, 9.018365), ("REF_Ramtel", 48.792524, 9.032133),
]
EDGE_STEP_M = 0.12      # m per step: a graded step this big ...
EDGE_EXCESS_M = 0.10    # ... that the raw DEM does not have (|dg - dr|) is an artificial edge
HP_SIGMA_M = 8.0        # waviness: high-pass with this Gaussian sigma
DIFF_CLIP_M = 1.5       # the diff image's colour range is +-this
UP = 4                  # image upscale (cells)
HOT_M = 1.0             # hotspot: cells with |graded - raw| above this, clustered into blobs
HOT_ROWS = 15           # hotspot rows kept per spot (largest first); top 5 printed
SUMMARY_HEAD = ("| spot | mean diff | p95 abs diff | max abs diff | edges | hp RMS graded | hp RMS raw | ratio | "
                "shape p95 | shape max | grade p95 | grade max |")


def parse_spots(text):
    """'NAME:lat,lon;NAME:lat,lon' -> [(name, lat, lon)]."""
    spots = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        name, coords = part.split(":", 1)
        lat, lon = (float(v) for v in coords.split(","))
        spots.append((name.strip(), lat, lon))
    return spots


def grid_xy(half, step):
    """East and north (m) of every cell, row 0 = north. Returns (X, Y) of shape (n, n)."""
    v = np.arange(-half, half + 1e-9, step)
    X, Y = np.meshgrid(v, v[::-1])
    return X, Y


def high_pass(a, step):
    return a - ndimage.gaussian_filter(a, sigma=HP_SIGMA_M / step, mode="nearest")


def metrics(raw, graded, step):
    """Pure: the diff, the artificial edges and the waviness of graded against raw (2-D, row 0 = north)."""
    raw = np.asarray(raw, np.float64)
    graded = np.asarray(graded, np.float64)
    n_r, n_c = raw.shape
    diff = graded - raw
    ad = np.abs(diff)
    dgx, drx = np.diff(graded, axis=1), np.diff(raw, axis=1)
    dgy, dry = np.diff(graded, axis=0), np.diff(raw, axis=0)
    vx = np.zeros(raw.shape)
    vy = np.zeros(raw.shape)
    ex = (np.abs(dgx) > EDGE_STEP_M) & (np.abs(dgx - drx) > EDGE_EXCESS_M)
    ey = (np.abs(dgy) > EDGE_STEP_M) & (np.abs(dgy - dry) > EDGE_EXCESS_M)
    vx[:, 1:] = np.where(ex, np.abs(dgx), 0.0)
    vy[1:, :] = np.where(ey, np.abs(dgy), 0.0)
    val = np.maximum(vx, vy)
    rows, cols = np.nonzero(val > 0)
    order = np.argsort(-val[rows, cols])[:5]
    worst = []
    for k in order:
        i, j = rows[k], cols[k]
        worst.append({"value": round(float(val[i, j]), 3),
                      "east": round(float((j - (n_c - 1) / 2) * step), 1),
                      "north": round(float(((n_r - 1) / 2 - i) * step), 1)})
    hg = float(np.sqrt(np.mean(high_pass(graded, step) ** 2)))
    hr = float(np.sqrt(np.mean(high_pass(raw, step) ** 2)))
    if hr < 1e-9:
        ratio = 1.0 if hg < 1e-9 else float("inf")
    else:
        ratio = hg / hr
    return {
        "mean_diff": round(float(np.mean(diff)), 4),
        "p95_abs_diff": round(float(np.percentile(ad, 95)), 4),
        "max_abs_diff": round(float(np.max(ad)), 4),
        "edge_count": int(rows.size),
        "edges_worst": worst,
        "hp_rms_graded": round(hg, 4),
        "hp_rms_raw": round(hr, 4),
        "hp_ratio": round(float(ratio), 3) if np.isfinite(ratio) else None,
    }


def blobs(diff, threshold):
    """Pure: 4-connected blobs of cells with |diff| > threshold. Returns [{"size", "rows", "cols"}], largest first."""
    lab, n = ndimage.label(np.abs(np.asarray(diff, np.float64)) > threshold)
    out = []
    for k in range(1, n + 1):
        r, c = np.nonzero(lab == k)
        out.append({"size": int(r.size), "rows": r, "cols": c})
    out.sort(key=lambda b: -b["size"])
    return out


def hillshade(z, step, az=315.0, alt=45.0):
    """Grey hillshade (uint8, 2-D). Rows run north to south, so the north slope is minus the row gradient."""
    z = np.asarray(z, np.float64)
    d_row, d_col = np.gradient(z, step)
    nx, ny, nz = -d_col, d_row, np.ones_like(z)     # surface normal (east, north, up)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    a, e = np.radians(az), np.radians(alt)
    lx, ly, lz = np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)
    shade = np.clip((nx * lx + ny * ly + nz * lz) / norm, 0.0, 1.0)
    return (shade * 255).astype(np.uint8)


def to_rgb(grey):
    return np.repeat(np.asarray(grey, np.uint8)[..., None], 3, axis=2)


def diverging(diff, clip=DIFF_CLIP_M):
    """Blue (below) - white (0) - red (above) colour map of diff clipped to +-clip. Returns uint8 (n, m, 3)."""
    t = np.clip(np.asarray(diff, np.float64) / clip, -1.0, 1.0)[..., None]
    white = np.array([255.0, 255.0, 255.0])
    blue = np.array([30.0, 90.0, 220.0])
    red = np.array([220.0, 50.0, 40.0])
    rgb = np.where(t < 0, white * (1 + t) - blue * t, white * (1 - t) + red * t)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def save_png(rgb, path, legend, centre):
    """Upscale UP x (nearest), mark the spot centre with a cross, write a legend in the top-left corner."""
    img = Image.fromarray(rgb).resize((rgb.shape[1] * UP, rgb.shape[0] * UP), Image.NEAREST)
    d = ImageDraw.Draw(img)
    cx, cy = (centre[1] + 0.5) * UP, (centre[0] + 0.5) * UP
    d.line([(cx - 8, cy), (cx + 8, cy)], fill=(255, 0, 255), width=2)
    d.line([(cx, cy - 8), (cx, cy + 8)], fill=(255, 0, 255), width=2)
    d.rectangle([0, 0, 7 * len(legend) + 8, 14], fill=(0, 0, 0))
    d.text((4, 2), legend, fill=(255, 255, 255))
    img.save(path)


def run_spot(blockworker, geodesy, grading, s, name, lat, lon, half, step, zoom, out):
    """One spot: grids, metrics, images and <name>.json. Returns the metrics dict."""
    frame = geodesy.EnuFrame(lat, lon)
    g, _ = s.anchor(1, lat, lon)
    X, Y = grid_xy(half, step)
    la, lo, _ = frame.enu_to_geodetic(X.ravel(), Y.ravel(), 0.0)
    la, lo = np.asarray(la, np.float64), np.asarray(lo, np.float64)
    raw = np.asarray(s.dem.sample(la, lo, zoom), np.float64).reshape(X.shape)
    shaped = np.asarray(blockworker.ground(frame, la, lo, raw.ravel()), np.float64).reshape(X.shape)
    if g is not None:
        graded = np.asarray(g.apply(la, lo, shaped, grading.ROAD_LIFT), np.float64)
    else:
        graded = shaped
    graded = graded.reshape(X.shape)
    m = metrics(raw, graded, step)
    diff = graded - raw
    d_shape, d_grade = shaped - raw, graded - shaped
    m.update({"shape_p95": round(float(np.percentile(np.abs(d_shape), 95)), 4),
              "shape_max": round(float(np.max(np.abs(d_shape))), 4),
              "grade_p95": round(float(np.percentile(np.abs(d_grade), 95)), 4),
              "grade_max": round(float(np.max(np.abs(d_grade))), 4)})
    centre = (X.shape[0] // 2, X.shape[1] // 2)
    rng = lambda a: f"min {a.min():.2f} max {a.max():.2f} m"
    gh, rh = hillshade(graded, step), hillshade(raw, step)
    save_png(to_rgb(gh), os.path.join(out, f"{name}_hillshade_graded.png"), "graded " + rng(graded), centre)
    save_png(to_rgb(rh), os.path.join(out, f"{name}_hillshade_raw.png"), "raw " + rng(raw), centre)
    save_png(diverging(diff), os.path.join(out, f"{name}_diff.png"), f"graded-raw +-{DIFF_CLIP_M} m", centre)
    save_png(diverging(d_shape), os.path.join(out, f"{name}_diff_shape.png"), f"shaped-raw +-{DIFF_CLIP_M} m", centre)
    save_png(diverging(d_grade), os.path.join(out, f"{name}_diff_grade.png"), f"graded-shaped +-{DIFF_CLIP_M} m", centre)
    hot = []
    n_r, n_c = X.shape
    for b in blobs(diff, HOT_M)[:HOT_ROWS]:
        r, c = b["rows"], b["cols"]
        k = int(np.abs(diff[r, c]).argmax())
        east = (c[k] - (n_c - 1) / 2) * step
        north = ((n_r - 1) / 2 - r[k]) * step
        bla, blo, _ = frame.enu_to_geodetic(np.array([east]), np.array([north]), 0.0)
        hot.append({"cells": b["size"], "gr_mean": float(diff[r, c].mean()), "gr_min": float(diff[r, c].min()),
                    "gr_max": float(diff[r, c].max()), "sr_mean": float(d_shape[r, c].mean()),
                    "gs_mean": float(d_grade[r, c].mean()), "lat": float(np.ravel(bla)[0]),
                    "lon": float(np.ravel(blo)[0])})
    with open(os.path.join(out, f"{name}_hotspots.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hot[0]) if hot else ["cells"])
        w.writeheader()
        w.writerows(hot)
    print(f"{name} hotspots (|graded-raw| > {HOT_M} m, {len(hot)} blobs):", flush=True)
    for h in hot[:5]:
        print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in h.items()), flush=True)
    edges = to_rgb(gh)
    edge_mask = np.zeros(X.shape, bool)
    _mask_edges(graded, raw, step, edge_mask)
    edges[edge_mask] = (255, 0, 0)
    save_png(edges, os.path.join(out, f"{name}_edges.png"), f"edges {m['edge_count']}", centre)
    spot = {"name": name, "lat": lat, "lon": lon}
    settings = {"half_m": half, "step_m": step, "zoom": zoom, "grading_on": g is not None}
    with open(os.path.join(out, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump({"spot": spot, "settings": settings, "metrics": m}, f, indent=2)
    return m


def _mask_edges(graded, raw, step, mask):
    """Fills mask with the edge cells of metrics() (same rule, kept separate so metrics() stays a pure number)."""
    dgx, drx = np.diff(graded, axis=1), np.diff(raw, axis=1)
    dgy, dry = np.diff(graded, axis=0), np.diff(raw, axis=0)
    mask[:, 1:] |= (np.abs(dgx) > EDGE_STEP_M) & (np.abs(dgx - drx) > EDGE_EXCESS_M)
    mask[1:, :] |= (np.abs(dgy) > EDGE_STEP_M) & (np.abs(dgy - dry) > EDGE_EXCESS_M)


def summary_row(name, m):
    if "error" in m:
        return f"| {name} | FAILED: {m['error']} |" + " |" * 10
    return (f"| {name} | {m['mean_diff']:+.3f} | {m['p95_abs_diff']:.3f} | {m['max_abs_diff']:.3f} | "
            f"{m['edge_count']} | {m['hp_rms_graded']:.3f} | {m['hp_rms_raw']:.3f} | "
            f"{'inf' if m['hp_ratio'] is None else m['hp_ratio']} | {m['shape_p95']:.3f} | {m['shape_max']:.3f} | "
            f"{m['grade_p95']:.3f} | {m['grade_max']:.3f} |")


def parse_set(text):
    """'MODULE.ATTR=VALUE' -> (module, attr, value). Split at the first '='; module and attr at the last '.' before it.
    The value is a Python literal (ast.literal_eval). Raises ValueError for a bad spec."""
    if "=" not in text:
        raise ValueError(f"expected MODULE.ATTR=VALUE, got {text!r}")
    target, raw = text.split("=", 1)
    module, dot, attr = target.strip().rpartition(".")
    if not dot or not module or not attr:
        raise ValueError(f"expected MODULE.ATTR before '=', got {target!r}")
    try:
        value = ast.literal_eval(raw.strip())
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"value of {target.strip()} is not a Python literal: {raw.strip()!r} ({e})") from e
    return module, attr, value


def apply_settings(sets):
    """setattr every --set on its (already imported) module. Unknown module or attr exits with an error.
    Returns [(MODULE.ATTR, old value, new value)] for the summary."""
    applied = []
    for text in sets:
        try:
            module, attr, value = parse_set(text)
        except ValueError as e:
            raise SystemExit(f"--set: {e}")
        try:
            mod = importlib.import_module(module)
        except ImportError as e:
            raise SystemExit(f"--set {text!r}: unknown module {module!r} ({e})")
        if not hasattr(mod, attr):
            raise SystemExit(f"--set {text!r}: {module} has no attribute {attr!r}")
        old = getattr(mod, attr)
        setattr(mod, attr, value)
        applied.append((f"{module}.{attr}", old, value))
    return applied


def main():
    os.environ["BNGE_PLANCACHE"] = "off"      # no cached road plans reused between variants (set before the server imports)
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server"))
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/beamng-earth"))
    ap.add_argument("--spots", default=None, help="NAME:lat,lon;NAME:lat,lon (default: the built-in list)")
    ap.add_argument("--half", type=float, default=60.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--zoom", type=int, default=16)
    ap.add_argument("--out", default=os.path.expanduser("~/bnge-tests/leonberg/elevation"))
    ap.add_argument("--tag", default="base", help="outputs go to <out>/<TAG>/ (default: base)")
    ap.add_argument("--set", action="append", default=[], metavar="MODULE.ATTR=VALUE",
                    help="override a server constant before the run, e.g. grading.SIGMA_M=3.0 (repeatable)")
    args = ap.parse_args()
    if not args.tag or any(c in args.tag for c in "/\\") or args.tag in (".", ".."):
        raise SystemExit(f"--tag must be a plain folder name, got {args.tag!r}")
    spots = parse_spots(args.spots) if args.spots else DEFAULT_SPOTS
    sys.path.insert(0, args.server)

    import blockworker                  # noqa: E402
    import geodesy                      # noqa: E402
    import grading as gradingmod        # noqa: E402

    applied = apply_settings(args.set)  # before blockworker.init
    out = os.path.join(args.out, args.tag)
    os.makedirs(out, exist_ok=True)
    blockworker.init(args.cache, False, True, True, True, False, multiprocessing.Queue(1), logging.ERROR, True, True)
    s = blockworker._s
    rows = []
    for name, lat, lon in spots:
        t0 = time.time()
        try:
            m = run_spot(blockworker, geodesy, gradingmod, s, name, lat, lon, args.half, args.step, args.zoom, out)
            print(f"{name}: done in {time.time() - t0:.0f} s, {m['edge_count']} edge cells", flush=True)
        except Exception as e:           # one bad spot must not stop the others
            lines = (str(e).strip().splitlines() or [""])
            m = {"error": f"{type(e).__name__}: {lines[0]}"[:200]}
            print(f"{name}: {m['error']}", flush=True)
        rows.append(summary_row(name, m))
    settings_lines = [f"- `{k}`: {old!r} -> {new!r}" for k, old, new in applied] or ["- (none, defaults)"]
    head = (f"## Run {time.strftime('%Y-%m-%d %H:%M')} tag `{args.tag}` (half {args.half:g} m, step {args.step:g} m, "
            f"zoom {args.zoom}, BNGE_PLANCACHE=off)\n\nApplied settings:\n" + "\n".join(settings_lines) + "\n\n")
    table = "\n".join([SUMMARY_HEAD, "|" + "---|" * 12] + rows)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:   # overwritten per run
        f.write(head + table + "\n")
    print()
    print(head + table)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
