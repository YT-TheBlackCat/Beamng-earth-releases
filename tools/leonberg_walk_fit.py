"""Walk fit diagnostics for Leonberg test spots: do the sidewalk slabs sit in the drawn ground, float above it, and
join the corners? Design GEHWEGE_DESIGN.md Abschnitt 3 (Brief C1).

Per spot: the sidewalk runs (no median, islands, splitters or mats) and corners of the tiles around it, measured on
the drawn ground (roadmesh.drawn_ground). Runs are rigid slabs (top = cz + h + w * CROSS_SLOPE, bottom = cz - SINK)
unless a run carries walk["back"], which this tool does not evaluate yet (the count is in the JSON).

Usage:
  .venv/bin/python tools/leonberg_walk_fit.py [--spots U1,U8] [--radius 120] [--zoom 16]
      [--cache ~/.cache/beamng-earth] [--out DIR] [--png]

Default OUT: ~/bnge-tests/leonberg/walkfit, plus -eu (EU details on) and -off (BNE_WALK_DRAPE=off).
Writes <id>.json per spot, report.md, and with --png <id>_worst<n>.png (cross sections of the 6 worst points).
"""

import argparse
import json
import logging
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))

import leonberg_surroundings_report as lsr  # noqa: E402  (SPOTS, _walks_in_frame)
import blockworker as bw  # noqa: E402
import europewalks  # noqa: E402
import furniture  # noqa: E402
import geodesy  # noqa: E402
import roadmesh  # noqa: E402
import roads  # noqa: E402
import sidewalks  # noqa: E402

CS = sidewalks.CROSS_SLOPE
SINK = sidewalks.SINK
CORNER_LIFT = sidewalks.CORNER_LIFT
JOINT_MAX_M = 0.5                  # corner end to the nearest run back edge: a joint up to this far


def resample(pts, step=1.0, extra=()):
    """(n, 3) polyline -> points about every `step` m along it, z interpolated; returns (points, arc length, and
    each (n_orig, c) array in `extra` interpolated along the same arc length)."""
    p = np.asarray(pts, np.float64)[:, :3]
    seg = np.hypot(*np.diff(p[:, :2], axis=0).T)
    d = np.r_[0.0, np.cumsum(seg)]
    if d[-1] <= 0:
        return p[:1], d[:1], [np.asarray(e, np.float64)[:1] for e in extra]
    n = max(2, int(round(d[-1] / step)) + 1)
    s = np.linspace(0.0, d[-1], n)
    out = np.column_stack([np.interp(s, d, p[:, k]) for k in range(3)])
    ex = [np.column_stack([np.interp(s, d, np.asarray(e, np.float64)[:, k]) for k in range(np.shape(e)[1])])
          for e in extra]
    return out, s, ex


def is_run(wk):
    """A plain sidewalk run: no median band, mat, island or splitter (Brief C1)."""
    return ("pts" in wk and "w" in wk and "corner" not in wk and "disc" not in wk and not wk.get("median")
            and not wk.get("mat") and wk.get("fid") not in ("island", "splitter"))


def run_geometry(wk, rigid):
    """A run on the curb line: its stations, and (unless --rigid or no walk["back"]) the draped back of
    sidewalks.drape_back, columns top (outer edge), bottom (back), run, interpolated to the stations."""
    has_back = wk.get("back") is not None and len(wk["back"]) == len(wk["pts"])
    p, s, (back,) = resample(wk["pts"], extra=[wk["back"] if has_back else np.full((len(wk["pts"]), 3), np.nan)])
    side, w = wk.get("side", 1), float(wk["w"])
    h = float(wk.get("height", wk.get("h", sidewalks.CURB_H)))
    curb, out = sidewalks.run_geometry(p, side, w)
    use_back = has_back and not rigid
    return {"curb": curb, "out": out, "s": s, "w": w, "h": h, "cz": curb[:, 2].copy(), "fid": wk.get("fid"),
            "side": side, "mid": p[len(p) // 2, :2], "has_back": has_back, "use_back": use_back,
            "back": back}


def measure_tile(P, frame, runs):
    """Penetration, float and slope per station of the runs, on the drawn ground of this tile (one drawn call).
    The plate runs linearly from the curb top (cz + h) to its outer top: the draped one (walk["back"][:, 0]) or
    the rigid one (cz + h + w * CROSS_SLOPE). Float: the draped back's bottom minus the ground at w + 0.5 m, else
    the rigid bottom (cz - SINK) minus the lower of the ground at w and w + 0.5 m."""
    if not runs:
        return
    xs_all = [np.array([0.3, r["w"] / 2, r["w"], r["w"] + 0.5]) for r in runs]
    xy = np.concatenate([(r["curb"][None, :, :2] + r["out"][None] * xs[:, None, None]).reshape(-1, 2)
                         for r, xs in zip(runs, xs_all)])
    ground = roadmesh.drawn_ground(frame, P.drawn, xy)
    off = 0
    for r, xs in zip(runs, xs_all):
        n = len(r["cz"])
        G = ground[off:off + 4 * n].reshape(4, n)
        off += 4 * n
        cz, h, w = r["cz"], r["h"], r["w"]
        rigid_top = cz + h + w * CS
        if r["use_back"]:
            top = np.where(np.isnan(r["back"][:, 0]), rigid_top, r["back"][:, 0])
            bottom = r["back"][:, 1]
            r["float"] = bottom - G[3]
        else:
            top = rigid_top
            r["float"] = (cz - SINK) - np.minimum(G[2], G[3])
        plate = np.stack([(cz + h) + (top - cz - h) * xs[k] / w for k in range(3)])
        r["G"] = G
        r["pen"] = (G[:3] - plate).max(axis=0)
        r["top"] = top
        r["slope"] = (top - cz - h) / w
        r["back_xy"] = r["curb"][:, :2] + r["out"] * w


def stats(v):
    v = np.asarray(v, np.float64)
    return v if len(v) else None


def pct(v, q):
    return None if v is None else float(np.percentile(v, q))


def summarise(acc):
    """The JSON numbers of one spot (metres, fractions and percent; rounded)."""
    pen = stats(np.concatenate(acc["pen"])) if acc["pen"] else None
    flo = stats(np.concatenate(acc["float"])) if acc["float"] else None
    slo = stats(np.concatenate(acc["slope"])) if acc["slope"] else None
    gap = stats(acc["gap"])
    step = stats(acc["step"])
    r3 = lambda x: None if x is None else round(float(x), 3)  # noqa: E731
    r4 = lambda x: None if x is None else round(float(x), 4)  # noqa: E731
    frac = lambda v, t: None if v is None else r4(np.mean(v > t))  # noqa: E731
    return {
        "runs": acc["runs"], "corners": acc["corners"], "metres": round(acc["metres"], 1),
        "runs_with_back_field": acc["back_field"], "runs_draped": acc["draped"], "mode": acc["mode"],
        "penetration_m": {"p95": r3(pct(pen, 95)), "max": r3(pen.max()) if pen is not None else None,
                          "gt0": frac(pen, 0.0), "gt5cm": frac(pen, 0.05), "gt15cm": frac(pen, 0.15)},
        "float_m": {"p95": r3(pct(flo, 95)), "gt2cm": frac(flo, 0.02)},
        "slope_pct": {"p50": None if slo is None else round(100 * pct(slo, 50), 2),
                      "p95": None if slo is None else round(100 * pct(slo, 95), 2),
                      "clamp6": None if slo is None else r4(np.mean(slo >= 0.06))},
        "corner_gap_m": {"n": None if gap is None else len(gap), "p95": r3(pct(gap, 95)),
                         "max": r3(gap.max()) if gap is not None else None},
        "corner_step_m": {"p95": r3(pct(step, 95)), "max": r3(step.max()) if step is not None else None},
        "corner_no_partner": acc["no_partner"],
        "widened_runs": None,   # no field for widened or ground-cut runs in the current stand
        "eu_on": bool(europewalks.enabled()),
    }


def worst_points(acc, n=6):
    cand = []
    for ri, r in enumerate(acc["recs"]):
        for i, v in enumerate(r["pen"]):
            cand.append((float(v), ri, i))
    cand.sort(reverse=True)
    return [(v, acc["recs"][ri], i) for v, ri, i in cand[:n]]


def draw_section(path, rec, i, sid, frame, P):
    """A cross section of one station (-1 m .. w + 1.5 m), 200 px/m across, 400 px/m up (Brief C1)."""
    from PIL import Image, ImageDraw
    w, h, cz = rec["w"], rec["h"], rec["cz"][i]
    top_w = float(rec["top"][i])
    xs = np.arange(-1.0, w + 1.5 + 1e-9, 0.25)
    xy = rec["curb"][i, :2][None] + rec["out"][i][None] * xs[:, None]
    g = roadmesh.drawn_ground(frame, P.drawn, xy)
    zlo = min(g.min(), cz - SINK) - 0.2
    zhi = max(g.max(), cz + h + w * CS) + 0.4
    W, H = int((w + 2.5) * 200) + 40, int((zhi - zlo) * 400) + 90

    def px(x, z):
        return (x + 1.0) * 200 + 20, (zhi - z) * 400 + 70

    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    bottom = float(rec["back"][i, 1]) if rec["use_back"] else cz - SINK
    plate = [px(0, cz - SINK), px(0, cz + h), px(w, top_w), px(w, bottom)]
    d.polygon(plate, fill=(190, 190, 190), outline=(60, 60, 60))
    d.line([px(x, z) for x, z in zip(xs, g)], fill=(120, 70, 20), width=2)
    d.line([px(-1.0, cz), px(w + 1.5, cz)], fill=(180, 180, 255))
    d.text((20, 8), f"{sid} s={rec['s'][i]:.0f} m fid={rec['fid']} side={rec['side']}", fill=(0, 0, 0))
    d.text((20, 26), f"Eindringen {rec['pen'][i] * 100:+.1f} cm  Schweben {rec['float'][i] * 100:+.1f} cm  "
                     f"w {w:.2f} m h {h * 100:.0f} cm", fill=(0, 0, 0))
    d.text((20, 44), "grau: Platte (Bord, Oberkante, Außenkante, Rückseite)   braun: Boden", fill=(0, 0, 0))
    img.save(path)


def run_spot(spot, args, log):
    rigid = bool(args.rigid)
    sid, name, lat, lon = spot
    frame = geodesy.EnuFrame(lat, lon)
    lamp = furniture.lamp_region("DE")
    acc = {"pen": [], "float": [], "slope": [], "gap": [], "step": [], "runs": 0, "corners": 0, "metres": 0.0,
           "no_partner": 0, "back_field": 0, "draped": 0, "mode": "rigid" if args.rigid else "back",
           "recs": []}
    backs, corner_ends = [], []
    k_lo, k_hi = math.floor(-args.radius / roads.TILE), math.floor(args.radius / roads.TILE)
    for tx in range(k_lo, k_hi + 1):
        for ty in range(k_lo, k_hi + 1):
            try:
                P = bw._tile_parts(lambda *a, **k: None, frame, (tx, ty), args.zoom, None, True, False, False,
                                   False, 0.0, lamp, "DE", None, ("leonberg", lat, lon), (lat, lon),
                                   cache_dir=args.cache)
            except Exception as ex:          # (one tile failing: the others go on)
                logging.exception("%s tile %s failed", sid, (tx, ty))
                print(f"{sid} tile {(tx, ty)} FAILED {type(ex).__name__}: {ex}", flush=True)
                continue
            walks, _ = lsr._walks_in_frame(P)
            runs = []
            for wk in walks:
                if not isinstance(wk, dict):
                    continue
                if "corner" in wk and "outer" in wk["corner"]:
                    outer = np.asarray(wk["corner"]["outer"], np.float64)
                    if outer.ndim != 2 or outer.shape[1] < 3 or not len(outer):
                        continue
                    h = float(wk.get("height", wk.get("h", sidewalks.CURB_H)))
                    if np.hypot(*outer[[0, -1], :2].T).min() > args.radius:
                        continue
                    acc["corners"] += 1
                    for e in (outer[0], outer[-1]):
                        corner_ends.append((e[:2], float(e[2]) + h + CORNER_LIFT))
                    continue
                if not is_run(wk):
                    continue
                r = run_geometry(wk, rigid)
                if np.hypot(*r["mid"]) > args.radius:
                    continue
                if len(r["cz"]) < 2:
                    continue
                acc["back_field"] += int("back" in wk)
                runs.append(r)
            measure_tile(P, frame, runs)
            for r in runs:
                acc["draped"] += int(r["use_back"])
                acc["runs"] += 1
                acc["metres"] += float(r["s"][-1])
                acc["pen"].append(r["pen"])
                acc["float"].append(r["float"])
                acc["slope"].append(r["slope"])
                backs.append((r["back_xy"], r["top"]))
                r["P"], r["frame"] = P, frame
                acc["recs"].append(r)
            log.info("%s tile %s: %d runs", sid, (tx, ty), len(runs))
    # Corner joints across the whole spot: each corner end to the nearest run back edge
    if backs and corner_ends:
        bxy = np.concatenate([b[0] for b in backs])
        btop = np.concatenate([b[1] for b in backs])
        for e, ztop in corner_ends:
            d = np.hypot(*(bxy - e).T)
            j = int(np.argmin(d))
            if d[j] <= JOINT_MAX_M:
                acc["gap"].append(float(d[j]))
                acc["step"].append(abs(float(btop[j]) - ztop))
            else:
                acc["no_partner"] += 1
    else:
        acc["no_partner"] += len(corner_ends)
    return acc


def md_row(sid, name, s):
    pen, flo, sl, gp, st = s["penetration_m"], s["float_m"], s["slope_pct"], s["corner_gap_m"], s["corner_step_m"]
    f = lambda v: "–" if v is None else f"{v * 100:.1f}"  # noqa: E731 (fraction -> %)
    cm = lambda v: "–" if v is None else f"{v * 100:.1f}"  # noqa: E731 (m -> cm)
    return (f"| {sid} {name} | {s['metres']:.0f} | {f(pen['gt0'])} / {f(pen['gt5cm'])} / {f(pen['gt15cm'])} | "
            f"{cm(pen['p95'])} | {f(flo['gt2cm'])} | {cm(flo['p95'])} | "
            f"{sl['p50'] if sl['p50'] is not None else '–'} / {sl['p95'] if sl['p95'] is not None else '–'} | "
            f"{f(sl['clamp6'])} | {cm(gp['p95'])} / {cm(gp['max'])} | {cm(st['p95'])} / {cm(st['max'])} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spots", default=None, help="comma list of spot ids (default all)")
    ap.add_argument("--radius", type=float, default=120.0)
    ap.add_argument("--zoom", type=int, default=16)
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/beamng-earth"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--png", action="store_true", help="cross sections of the 6 worst points per spot")
    ap.add_argument("--rigid", action="store_true", help="ignore walk['back']: the rigid plate (before-state)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    eu_on = bool(europewalks.enabled())
    drape_on = bool(sidewalks.DRAPE_ON)
    if args.out is None:
        args.out = os.path.expanduser("~/bnge-tests/leonberg/walkfit" + ("-eu" if eu_on else "")
                                      + ("" if drape_on else "-off"))
    spots = lsr.SPOTS
    if args.spots:
        want = [s.strip() for s in args.spots.split(",") if s.strip()]
        bad = [s for s in want if s not in {x[0] for x in lsr.SPOTS}]
        if bad:
            ap.error(f"unknown spot(s): {', '.join(bad)}")
        spots = [s for s in lsr.SPOTS if s[0] in want]
    os.makedirs(args.out, exist_ok=True)
    st = bw._State(args.cache, False, True, True, True, True)
    bw._s = st
    log = logging.getLogger("walkfit")
    mode = ("starre Platte (--rigid)" if args.rigid
            else "walk['back'] (Außenkante top, Unterkante bottom), starre Platte nur ohne back")
    lines = ["# Gehweg-Anpassung (walk fit) Leonberg", "",
             f"Radius {args.radius:g} m, Zoom {args.zoom}; EU-Details {'an' if eu_on else 'aus'}; "
             f"Modus: {mode}; BNE_WALK_DRAPE {'an' if drape_on else 'aus'}.", "",
             "| Stelle | Meter | Eindringen >0 / >5 / >15 cm (%) | p95 cm | Schweben >2 cm (%) | p95 cm | "
             "Neigung p50 / p95 (%) | Klemme 6 % (%) | Fuge Grundriss p95 / max cm | Fuge Höhe p95 / max cm |",
             "|---|---:|---|---:|---:|---:|---|---:|---|---|"]
    for spot in spots:
        sid, name = spot[0], spot[1]
        try:
            acc = run_spot(spot, args, log)
        except Exception as ex:              # (one spot failing: the others go on)
            logging.exception("%s failed", sid)
            print(f"{sid} FAILED {type(ex).__name__}: {ex}", flush=True)
            continue
        s = summarise(acc)
        with open(os.path.join(args.out, f"{sid}.json"), "w", encoding="utf-8") as f:
            json.dump({"spot": list(spot), "radius": args.radius, **s}, f, ensure_ascii=False, indent=1)
        lines.append(md_row(sid, name, s))
        if args.png:
            for n, (v, rec, i) in enumerate(worst_points(acc), 1):
                draw_section(os.path.join(args.out, f"{sid}_worst{n}.png"), rec, i, sid, rec["frame"], rec["P"])
    report = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
