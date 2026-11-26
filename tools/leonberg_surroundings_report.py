#!/usr/bin/env python3
"""Leonberg test spots: the surroundings within a radius, as the generator builds them and as the map has them.

    BNE_EU_ROAD_DETAILS=on python3 tools/leonberg_surroundings_report.py [--spots U1,U3] [--radius 150] [--out DIR]

Per spot (the radius around the spot, measured from the frame's origin):
  generated.walks   the sidewalks the road tiles build (roads.build_tile's walks, sidewalks.add_walk's rendering):
                    runs, corners, length, source (gemappt / geschaetzt), surface, width, height, EU / detached / median
  generated.props   the forest rows of the fences and noise walls (fences.rows) and the other props (street lamps,
                    benches, ...): count per group, noise wall length and heights, chain-link length
  mapped            what the map (Overture / OSM) has there: sidewalk tags of the roads, footway=sidewalk and other
                    footways, barrier lines by kind (with fences.kind_of: built or not), point features by class
  gaps              short sentences where the map and the generator disagree
Outputs: <out>/<id>.json per spot and <out>/report.md (all spots, also printed). Reads the real cache
(~/.cache/beamng-earth, or --cache), fetching what it lacks. The EU road details are not switched here: the caller sets
BNE_EU_ROAD_DETAILS=on; the value europewalks.enabled() sees is written into the output.
Default OUT: ~/bnge-tests/leonberg/surroundings-eu (EU details on) or ~/bnge-tests/leonberg/surroundings.
"""
import argparse
import json
import logging
import math
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))

import numpy as np  # noqa: E402

SPOTS = [
    ("U1", "Marktplatz", 48.80140, 9.01520),
    ("U2", "Stadtpark", 48.79321, 9.01574),
    ("U3", "Neues Rathaus", 48.79838, 9.01353),
    ("U4", "Bahnhof", 48.79974, 9.00432),
    ("U5", "Engelbergtunnel A81", 48.79872, 9.02900),
    ("U6", "Brennerstraße Lärmschutz", 48.79285, 8.99376),
    ("U7", "Autobahndreieck", 48.78168, 9.01279),
    ("U8", "Ramtel", 48.79252, 9.03213),
    ("U9", "Brennerstraße Ezach", 48.79308, 8.99910),
    ("U10", "Einmündung Ebinger Straße", 48.79320, 8.99950),
]

FOOTWAY_CLASSES = ("footway", "path", "pedestrian", "steps")


def length_within(xy, radius):
    """Length (m) of the polyline xy (k, 2) inside the circle of radius m around (0, 0). Each segment is cut into
    pieces of about 1 m (np.linspace); a piece counts when its midpoint is inside the circle."""
    xy = np.asarray(xy, np.float64)
    if xy.ndim != 2 or xy.shape[0] < 2 or xy.shape[1] < 2:
        return 0.0
    xy = xy[:, :2]                          # (the walks' points are x, y, z)
    total = 0.0
    for a, b in zip(xy[:-1], xy[1:]):
        seg = float(np.hypot(*(b - a)))
        if not math.isfinite(seg) or seg <= 0.0:
            continue
        k = max(1, int(round(seg)))
        t = np.linspace(0.0, 1.0, k + 1)
        pts = a[None, :] + t[:, None] * (b - a)[None, :]
        mids = (pts[:-1] + pts[1:]) / 2.0
        inside = np.hypot(mids[:, 0], mids[:, 1]) <= radius
        total += (seg / k) * float(inside.sum())
    return total


def _any_within(points, radius):
    pts = np.asarray(points, np.float64)
    if pts.ndim != 2 or not len(pts):
        return False
    return bool((np.hypot(pts[:, 0], pts[:, 1]) <= radius).any())


def _stats(vals):
    if not vals:
        return None
    a = np.asarray(vals, np.float64)
    return {"median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def summarise_walks(walks, radius):
    """Summary of the walk records of one spot (roads.build_tile's walks; discs are skipped, corners only counted)."""
    import sidewalks

    out = {"n_runs": 0, "corners": 0, "length_m": 0.0, "by_source": Counter(), "by_surface": Counter(),
           "widths": [], "heights": [], "europe_m": 0.0, "detached_m": 0.0, "median_m": 0.0}
    for w in walks or []:
        if not isinstance(w, dict) or "disc" in w:
            continue
        if "corner" in w:
            arc = (w["corner"] or {}).get("arc")
            if arc is not None and _any_within(arc, radius):
                out["corners"] += 1
            continue
        pts = w.get("pts")
        length = length_within(pts, radius) if pts is not None else 0.0
        if length <= 0.0:
            continue
        out["n_runs"] += 1
        out["length_m"] += length
        out["by_source"]["gemappt" if w.get("mapped") else "geschätzt"] += length
        out["by_surface"]["us_slab" if w.get("surface") is None else str(w.get("surface"))] += length
        if w.get("w") is not None:
            out["widths"].append(float(w["w"]))
        out["heights"].append(float(w.get("height", w.get("h", sidewalks.CURB_H))))
        if w.get("europe"):
            out["europe_m"] += length
        if w.get("detached"):
            out["detached_m"] += length
        if w.get("median"):
            out["median_m"] += length
    return {"n_runs": out["n_runs"], "corners": out["corners"], "length_m": out["length_m"],
            "by_source": dict(out["by_source"]), "by_surface": dict(out["by_surface"]),
            "width_m": _stats(out["widths"]), "height_m": _stats(out["heights"]),
            "europe_m": out["europe_m"], "detached_m": out["detached_m"], "median_m": out["median_m"]}


def summarise_props(rows, radius):
    """{group: count} of the prop rows [group, x, y, z, yaw, scale, height, rank] within radius, plus noisewall_m
    (panels x fences.WALL_LEN), noisewall_heights (the distinct heights) and chainlink_m (panels x CHAIN_LEN)."""
    import fences

    counts, heights = Counter(), set()
    for r in rows or []:
        if len(r) < 3 or math.hypot(float(r[1]), float(r[2])) > radius:
            continue
        counts[str(r[0])] += 1
        if str(r[0]) == "noisewall" and len(r) > 6 and _num(r[6]) is not None:
            heights.add(round(float(r[6]), 2))
    out = dict(counts)
    out["noisewall_m"] = counts.get("noisewall", 0) * fences.WALL_LEN
    out["noisewall_heights"] = sorted(heights)
    out["chainlink_m"] = counts.get("chainlink", 0) * fences.CHAIN_LEN
    return out


def barrier_kind(cls, tags):
    """The barrier kind of a mapped infrastructure line (its class and tags), or None (not a barrier)."""
    import fences

    cls = str(cls or "").strip().lower()
    tags = {str(k).strip().lower(): str(v).strip().lower() for k, v in (tags or {}).items()}
    barrier = tags.get("barrier", "")
    wall = tags.get("wall", "")
    fence_type = tags.get("fence_type") or tags.get("fence:type") or ""
    if cls in fences.NOISE_TYPES or barrier in fences.NOISE_TYPES or wall in fences.NOISE_TYPES:
        return "noise_barrier"
    if cls == "fence" or barrier == "fence":
        return f"fence:{fence_type or '?'}"
    if cls == "wall" or barrier == "wall":
        return f"wall:{wall or '?'}"
    if cls == "hedge" or barrier == "hedge":
        return "hedge"
    if cls == "guard_rail" or barrier == "guard_rail":
        return "guard_rail"
    if cls == "retaining_wall" or barrier == "retaining_wall":
        return "retaining_wall"
    if cls:
        return cls
    if barrier:
        return f"barrier:{barrier}"
    return None


def gaps(summary):
    """Short German sentences: mapped but not generated, or generated differently."""
    w, p, m = summary["walks"], summary["props"], summary["mapped"]
    out = []
    for kind, b in sorted(m["barriers"].items()):
        if b["m_unbuilt"] > 0:
            out.append(f"{kind} {b['m_unbuilt']:.0f} m gemappt, nicht gebaut")
    nm = m["barriers"].get("noise_barrier", {}).get("m", 0.0)
    ng = p.get("noisewall_m", 0.0)
    if max(nm, ng) > 0 and abs(nm - ng) > max(5.0, 0.1 * max(nm, ng)):
        out.append(f"noise_barrier gemappt {nm:.0f} m, gebaut {ng:.0f} m")
    nl = m["points"].get("street_lamp", 0)
    gl = p.get("streetlight", 0)
    if nl != gl:
        out.append(f"street_lamp: {nl} Punkte gemappt, {gl} streetlight-Props erzeugt")
    fs = m["footway_sidewalk"]["m"]
    if fs > 0 and w["by_source"].get("gemappt", 0) <= 0:
        out.append(f"footway=sidewalk {fs:.0f} m gemappt, kein gemappter Gehweg erzeugt")
    for value in ("no", "none"):
        if value in m["sidewalk_tags"] and m["sidewalk_tags"][value]["m"] > 0:
            out.append(f"Straßen mit sidewalk={value}: {m['sidewalk_tags'][value]['m']:.0f} m (Hinweis)")
    return out


def _round(obj, nd=1):
    if isinstance(obj, dict):
        return {str(k): _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, nd) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return round(obj, nd) if math.isfinite(obj) else None
    return obj


def _kv(d):
    return ", ".join(f"{k}: {v:.1f}" for k, v in sorted(d.items())) or "-"


def _mmm(s):
    return "-" if not s else f"{s['median']:.2f} / {s['min']:.2f} / {s['max']:.2f}"


def markdown(spot, eu_on, summary):
    sid, name, lat, lon = spot
    w, p, m = summary["walks"], summary["props"], summary["mapped"]
    out = [f"## {sid} {name} ({lat:.5f}, {lon:.5f})", "", f"EU-Details (BNE_EU_ROAD_DETAILS): {'an' if eu_on else 'aus'}",
           "", "### Gehwege erzeugt", "", "| Kennzahl | Wert |", "| --- | --- |",
           f"| Läufe | {w['n_runs']} |", f"| Ecken | {w['corners']} |", f"| Länge (m) | {w['length_m']:.1f} |",
           f"| Quelle (m) | {_kv(w['by_source'])} |", f"| Belag (m) | {_kv(w['by_surface'])} |",
           f"| Breite (m) median / min / max | {_mmm(w['width_m'])} |",
           f"| Höhe (m) median / min / max | {_mmm(w['height_m'])} |",
           f"| EU / abgesetzt / Mittelstreifen (m) | {w['europe_m']:.1f} / {w['detached_m']:.1f} / {w['median_m']:.1f} |",
           "", "### Karte", "", "| Merkmal | Anzahl | Länge (m) | Details |", "| --- | --- | --- | --- |"]
    for v, b in sorted(m["sidewalk_tags"].items()):
        out.append(f"| sidewalk={v} | {b['n']} | {b['m']:.1f} | {_kv(b['classes'])} |")
    fs = m["footway_sidewalk"]
    out.append(f"| footway=sidewalk | {fs['n']} | {fs['m']:.1f} | - |")
    for c, b in sorted(m["footways"].items()):
        out.append(f"| Fußweg {c} | {b['n']} | {b['m']:.1f} | - |")
    for k, b in sorted(m["barriers"].items()):
        built = ", ".join(f"{a}: {n}" for a, n in sorted(b["built_as"].items()))
        hs = ", ".join(f"{h:g}" for h in b["heights"]) or "-"
        out.append(f"| Barriere {k} | {b['n']} | {b['m']:.1f} | Höhe-Tags: {hs}; gebaut als: {built} "
                   f"(nicht gebaut: {b['m_unbuilt']:.1f} m) |")
    for c, n in sorted(m["points"].items()):
        out.append(f"| Punkt {c} | {n} | - | - |")
    out += ["", "### Props erzeugt", "", "| Gruppe | Anzahl |", "| --- | --- |"]
    for k in sorted(k for k in p if not k.endswith("_m") and not k.endswith("_heights")):
        out.append(f"| {k} | {p[k]} |")
    out += ["", f"Noise walls: {p['noisewall_m']:.1f} m, Höhen {p['noisewall_heights'] or '-'}; "
            f"Chain-link: {p['chainlink_m']:.1f} m", "", "### Lücken", ""]
    out += [f"- {g}" for g in summary["gaps"]] or ["- keine"]
    out += ["", "_gebaut als_: fences.kind_of (nur Maschendraht und Lärmschutz); andere Barrieren werden anderswo "
            "gebaut (Leitplanke, Hecke, Stützmauer), dort kann 'nicht gebaut' falsch sein._", ""]
    return "\n".join(out) + "\n"


def _line_length(feats, frame, rings, radius):
    import mapgeom

    total = 0.0
    for ring in rings:
        if ring is None or len(ring) < 2:
            continue
        e, n = mapgeom.enu_xy(feats, frame, np.asarray(ring, np.float64))
        total += length_within(np.column_stack((e, n)), radius)
    return total


def _val(col, i):
    """The lower-case text of attribute column col at i ('' for no column, a missing or NaN value)."""
    if col is None:
        return ""
    v = col[i]
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return str(v).strip().lower()


def _mapped_roads(st, frame, cells, radius, box, out):
    import europewalks
    import mapgeom
    import roads

    seen = {"sw": set(), "fw": set(), "fwc": set()}
    x0, y0, x1, y1 = box
    for key in cells:
        f = st.md.cell(key, layers=("roads",), kind="demand").layers.get("roads")
        if f is None or not len(f.kind):
            continue
        sw_col, cls_col = f.attrs.get("osm_sidewalk"), f.attrs.get("class")
        boxes = f.boxes(frame)
        for i in range(len(f.kind)):
            if int(f.kind[i]) != mapgeom.LINE:
                continue
            b = boxes[i]
            if b[2] < x0 or b[0] > x1 or b[3] < y0 or b[1] > y1:
                continue
            fid = str(f.ids[i])
            cls = _val(cls_col, i)
            if sw_col is not None and fid not in seen["sw"]:
                seen["sw"].add(fid)
                length = _line_length(f, frame, roads.ground_parts(f, i), radius)
                if length > 0:
                    v = _val(sw_col, i) or "kein Tag"
                    e = out["sidewalk_tags"].setdefault(v, {"n": 0, "m": 0.0, "classes": Counter()})
                    e["n"] += 1
                    e["m"] += length
                    e["classes"][cls or "?"] += length
            if fid not in seen["fw"] and europewalks.mapped_walk(f, i) is not None:
                seen["fw"].add(fid)
                length = _line_length(f, frame, roads.ground_parts(f, i), radius)
                if length > 0:
                    out["footway_sidewalk"]["n"] += 1
                    out["footway_sidewalk"]["m"] += length
            if cls in FOOTWAY_CLASSES and fid not in seen["fwc"]:
                seen["fwc"].add(fid)
                length = _line_length(f, frame, roads.ground_parts(f, i), radius)
                if length > 0:
                    e = out["footways"].setdefault(cls, {"n": 0, "m": 0.0})
                    e["n"] += 1
                    e["m"] += length


def _mapped_infra(st, frame, cells, radius, box, out):
    import fences
    import mapgeom

    seen = set()
    x0, y0, x1, y1 = box
    for key in cells:
        f = st.md.cell(key, layers=("infrastructure",), kind="demand").layers.get("infrastructure")
        if f is None or not len(f.kind):
            continue
        boxes = f.boxes(frame)
        cls_col = f.attrs.get("class")
        xy = None
        for i in range(len(f.kind)):
            fid = str(f.ids[i])
            if fid in seen:
                continue
            b = boxes[i]
            if b[2] < x0 or b[0] > x1 or b[3] < y0 or b[1] > y1:
                continue
            cls = _val(cls_col, i)
            kind = int(f.kind[i])
            if kind == mapgeom.POINT:
                if xy is None:
                    xy = f.xy_in(frame)
                px, py = xy[f.coord_off[f.ring_off[f.part_off[i]]]]
                if math.hypot(px, py) <= radius:
                    seen.add(fid)
                    out["points"][cls or "?"] = out["points"].get(cls or "?", 0) + 1
                continue
            if kind != mapgeom.LINE:
                continue
            tags = fences._tags(f, i)
            bk = barrier_kind(cls, tags)
            if bk is None:
                continue
            length = _line_length(f, frame, [part[0] for part in f.parts(i) if part], radius)
            if length <= 0:
                continue
            seen.add(fid)
            built = fences.kind_of(f, i, tags) or "nicht gebaut"
            e = out["barriers"].setdefault(bk, {"n": 0, "m": 0.0, "m_unbuilt": 0.0, "heights": [], "built_as": {}})
            e["n"] += 1
            e["m"] += length
            if built == "nicht gebaut":
                e["m_unbuilt"] += length
            e["built_as"][built] = e["built_as"].get(built, 0) + 1
            h = _num(tags.get("height"))
            if h is not None:
                e["heights"].append(h)


def _walks_in_frame(P):
    """P.walks with their points in the frame (roads.build_tile's points are in the frame; if they lie far from the
    tile's box but near its origin, the origin is added)."""
    walks = [w for w in (P.walks or []) if isinstance(w, dict)]
    sample = [np.asarray(w["pts"], np.float64)[:, :2] for w in walks if w.get("pts") is not None and len(w["pts"])]
    if not sample or P.box is None:
        return walks, "frame"
    mid = np.vstack(sample[:50]).mean(0)
    x0, y0, x1, y1 = [float(v) for v in P.box]
    if x0 - 50 <= mid[0] <= x1 + 50 and y0 - 50 <= mid[1] <= y1 + 50:
        return walks, "frame"
    off = np.asarray(P.origin, np.float64)[:2]
    shifted = []
    for w in walks:
        w = dict(w)
        if w.get("pts") is not None:
            w["pts"] = np.asarray(w["pts"], np.float64)[:, :2] + off
        shifted.append(w)
    return shifted, "origin+"


def collect(st, spot, radius, cache, eu_on):
    """The summary of one spot: generated walks and props of its tiles, the mapped features in the radius."""
    import blockworker as bw
    import furniture
    import geodesy
    import mapcells
    import roads

    sid, name, lat, lon = spot
    frame = geodesy.EnuFrame(lat, lon)
    k_lo, k_hi = math.floor(-radius / roads.TILE), math.floor(radius / roads.TILE)
    tiles = [(tx, ty) for tx in range(k_lo, k_hi + 1) for ty in range(k_lo, k_hi + 1)]
    box = (-radius, -radius, radius, radius)
    walks, props = [], []
    lamp = furniture.lamp_region("DE")
    for t in tiles:
        try:
            P = bw._tile_parts(lambda *a, **k: None, frame, t, 15, None, True, False, False, False, 0.0, lamp, "DE",
                               None, ("leonberg", lat, lon), (lat, lon), cache_dir=cache)
        except Exception as ex:             # (one tile failing: the others go on)
            logging.exception("%s tile %s failed", sid, t)
            print(f"{sid} tile {t} FAILED {type(ex).__name__}: {ex}", flush=True)
            continue
        tw, how = _walks_in_frame(P)
        walks += tw
        props += list(P.props or [])
        logging.info("%s tile %s: %d walks (%s), %d props", sid, t, len(tw), how, len(P.props or []))

    la, lo, _ = frame.enu_to_geodetic(np.array([-radius, -radius, radius, radius]),
                                      np.array([-radius, radius, -radius, radius]), np.zeros(4))
    try:
        cells = mapcells.cells_for_points(la, lo, pad_lat=0.00005, pad_lon=0.00005, max_cells=64)
    except (ValueError, OverflowError) as ex:
        logging.warning("%s: no map cells: %s", sid, ex)
        cells = []
    mapped = {"sidewalk_tags": {}, "footway_sidewalk": {"n": 0, "m": 0.0}, "footways": {}, "barriers": {},
              "points": {}}
    _mapped_roads(st, frame, cells, radius, box, mapped)
    _mapped_infra(st, frame, cells, radius, box, mapped)
    for v in mapped["sidewalk_tags"].values():
        v["classes"] = dict(v["classes"])
    for b in mapped["barriers"].values():
        b["heights"] = sorted(set(b["heights"]))

    summary = {"walks": summarise_walks(walks, radius), "props": summarise_props(props, radius), "mapped": mapped}
    summary["gaps"] = gaps(summary)
    return _round(summary)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spots", default=None, help="comma list of spot ids (default all)")
    ap.add_argument("--radius", type=float, default=150.0)
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/beamng-earth"))
    ap.add_argument("--out", default=None, help="default ~/bnge-tests/leonberg/surroundings(-eu)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    import europewalks
    eu_on = bool(europewalks.enabled())
    if args.out is None:
        args.out = os.path.expanduser("~/bnge-tests/leonberg/" + ("surroundings-eu" if eu_on else "surroundings"))
    spots = SPOTS
    if args.spots:
        want = [s.strip() for s in args.spots.split(",") if s.strip()]
        known = {s[0] for s in SPOTS}
        bad = [s for s in want if s not in known]
        if bad:
            ap.error(f"unknown spot(s): {', '.join(bad)}")
        spots = [s for s in SPOTS if s[0] in want]
    os.makedirs(args.out, exist_ok=True)
    import blockworker as bw
    st = bw._State(args.cache, False, True, True, True, True)
    bw._s = st
    sections = ["# Umgebungs-Bestandsaufnahme Leonberg", "",
                f"Radius {args.radius:g} m um jeden Testpunkt; EU-Details (BNE_EU_ROAD_DETAILS): "
                f"{'an' if eu_on else 'aus'}. Erzeugt: Gehwege, Lärmschutz/Zäune, Props. Karte: Overture/OSM.", ""]
    for spot in spots:
        sid = spot[0]
        try:
            summary = collect(st, spot, args.radius, args.cache, eu_on)
        except Exception as ex:             # (one spot failing: the others go on)
            logging.exception("%s failed", sid)
            print(f"{sid} FAILED {type(ex).__name__}: {ex}", flush=True)
            continue
        with open(os.path.join(args.out, f"{sid}.json"), "w", encoding="utf-8") as f:
            json.dump({"spot": list(spot), "eu_details": eu_on, "generated": {"walks": summary["walks"],
                       "props": summary["props"]}, "mapped": summary["mapped"], "gaps": summary["gaps"]},
                      f, ensure_ascii=False, indent=1)
        sections.append(markdown(spot, eu_on, summary))
    report = "\n".join(sections)
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
