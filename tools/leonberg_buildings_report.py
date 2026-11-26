#!/usr/bin/env python3
"""Leonberg test spots: every building within a radius, as the source data says it is and as the generator builds it.

    python3 tools/leonberg_buildings_report.py [--spots T1,T3] [--radius 150] [--out DIR]

Per spot, one row per building footprint (the raw map footprints, their centroid within the radius):
  footprint_src   OSM / ML (Microsoft ML Buildings, Google Open Buildings) / the dataset's own name
  map_height, map_floors, map_height_src, map_roof_shape, map_roof_height   the map's own values
  lod2_match, lod2_code, lod2_shape, lod2_height, lod2_roof_h, lod2_eave   the measured LoD2 roof that contains the centroid
  height_src      Karte (map height) / LoD2 / Karte:Stockwerke (map floors) / geschaetzt (estimated)
  roof_src        Karte (map roof shape) / LoD2 (measured roof type) / geschaetzt (estimated)
  built_*         what the generator builds on that footprint (buildings.road_tile_items: parts, height, roof, ...)
  ml, facade_color, roof_color, year   the building's ML flag, colours and OSM construction year where known
Outputs per spot: <out>/<id>.json (rows, summary, LoD2 status), <out>/<id>.csv (rows); <out>/report.md (all spots,
also printed). Reads the real cache (~/.cache/beamng-earth, or --cache), fetching what it lacks; no roof photos.
Default OUT: ~/bnge-tests/leonberg/report.
"""
import argparse
import csv
import json
import logging
import math
import os
import sys
from collections import Counter

os.environ.setdefault("BNE_ROOF_PHOTO", "off")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))

import numpy as np  # noqa: E402

SPOTS = [
    ("T1", "Marktplatz", 48.80140, 9.01520),
    ("T2", "Schloss", 48.80128, 9.01217),
    ("T3", "Stadtkirche", 48.80155, 9.01328),
    ("T4", "Neues Rathaus", 48.79838, 9.01353),
    ("T5", "LEO-Center", 48.79524, 9.01127),
    ("T6", "Bahnhof", 48.79974, 9.00432),
    ("T7", "Krankenhaus", 48.80284, 9.00034),
    ("T8", "Ramtel", 48.79252, 9.03213),
]

# height sources that are ML estimates (same values as server/lod2roofs.ML_HEIGHT_SOURCES, copied to keep this light)
ML_HEIGHT_SOURCES = {"Microsoft ML Buildings", "Google Open Buildings"}

# AdV roofType -> roof:shape (copied from server/lod2roofs.py SHAPES)
LOD2_SHAPES = {1000: "flat", 2100: "skillion", 2200: "skillion", 3100: "gabled", 3200: "hipped", 3300: "half-hipped",
               3400: "mansard", 3500: "pyramidal", 3600: "pyramidal", 3700: "dome", 3800: "skillion", 3900: "round",
               4000: "pyramidal"}

ROW_KEYS = ["id", "lat", "lon", "dist_m", "area_m2", "footprint_src", "cls", "name", "map_height", "map_floors",
            "map_height_src", "map_roof_shape", "map_roof_height", "lod2_match", "lod2_code", "lod2_shape",
            "lod2_height", "lod2_roof_h", "lod2_eave", "height_src", "roof_src", "built_parts", "built_height", "built_roof",
            "built_floors", "built_roof_mat", "built_facade", "ml", "facade_color", "roof_color", "year"]
SCALARS = ["n", "lod2_matched", "lod2_share", "dh_built_lod2_median", "dh_built_lod2_max", "roof_mismatch",
           "not_built", "with_colour", "with_year"]


def _finite_pos(v):
    """True for a finite number > 0 (numpy scalars ok; None, NaN and strings are not)."""
    if v is None or isinstance(v, (str, bytes)):
        return False
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f) and f > 0


def height_source(map_height, map_floors, lod2_height, map_height_src=None):
    """Where the height the generator uses comes from (mirrors lod2roofs.Records.fill: the map's height counts, except
    an ML estimate where a measured LoD2 height is there; the floors are a fallback inside plan())."""
    ml = str(map_height_src or "") in ML_HEIGHT_SOURCES
    if _finite_pos(map_height) and (not ml or not _finite_pos(lod2_height)):
        return "Karte"
    if _finite_pos(lod2_height):
        return "LoD2"
    if _finite_pos(map_floors):
        return "Karte:Stockwerke"
    return "geschätzt"


def roof_source(map_roof_shape, lod2_code):
    """Where the roof shape comes from: the map's, else the measured roof's AdV code, else estimated."""
    if isinstance(map_roof_shape, str) and map_roof_shape.strip():
        return "Karte"
    if lod2_code is not None:
        try:
            if int(lod2_code) in LOD2_SHAPES:
                return "LoD2"
        except (TypeError, ValueError):
            pass
    return "geschätzt"


def footprint_source(src_dataset):
    if not src_dataset:
        return "?"
    if src_dataset == "OpenStreetMap":
        return "OSM"
    if src_dataset in ("Microsoft ML Buildings", "Google Open Buildings"):
        return "ML"
    return str(src_dataset)


def summarise(rows):
    n = len(rows)
    dh_rows = []
    for r in rows:
        if _finite_pos(r["built_height"]) and _finite_pos(r["lod2_eave"]):
            dh_rows.append((abs(r["built_height"] - r["lod2_eave"]), r))
    matched = sum(1 for r in rows if r["lod2_match"] is True)
    dh_vals = [d for d, _ in dh_rows]
    worst = sorted(dh_rows, key=lambda t: -t[0])[:5]
    return {
        "n": n,
        "by_footprint": dict(Counter(r["footprint_src"] for r in rows)),
        "by_height_src": dict(Counter(r["height_src"] for r in rows)),
        "by_roof_src": dict(Counter(r["roof_src"] for r in rows)),
        "lod2_matched": matched,
        "lod2_share": round(matched / n, 2) if n else 0,
        "dh_built_lod2_median": round(float(np.median(dh_vals)), 1) if dh_vals else None,
        "dh_built_lod2_max": round(float(max(dh_vals)), 1) if dh_vals else None,
        "roof_mismatch": sum(1 for r in rows if r["lod2_shape"] and r["built_roof"]
                             and r["built_roof"] != r["lod2_shape"]),
        "not_built": sum(1 for r in rows if not r["built_parts"]),
        "with_colour": sum(1 for r in rows if r["facade_color"] or r["roof_color"]),
        "with_year": sum(1 for r in rows if r["year"]),
        "worst": [{"id": r["id"], "name": r["name"], "cls": r["cls"], "map_height": r["map_height"],
                   "lod2_height": r["lod2_height"], "lod2_eave": r["lod2_eave"],
                   "built_height": r["built_height"]} for _, r in worst],
    }


def markdown(spot, lod2_status, summary):
    sid, name, lat, lon = spot
    out = [f"## {sid} {name} ({lat:.5f}, {lon:.5f})", "", f"LoD2: {lod2_status}", "",
           "| Kennzahl | Wert |", "| --- | --- |"]
    for k in SCALARS:
        out.append(f"| {k} | {summary[k]} |")
    for k in ("by_footprint", "by_height_src", "by_roof_src"):
        v = summary[k]
        out.append(f"| {k} | {', '.join(f'{a}: {b}' for a, b in sorted(v.items())) or '-'} |")
    if summary["worst"]:
        out += ["", "Grösste Abweichung Traufhöhe gebaut zu LoD2 (m):", "",
                "| id | Name | Klasse | Karte | LoD2 Traufe | gebaut |", "| --- | --- | --- | --- | --- | --- |"]
        for w in summary["worst"]:
            out.append(f"| {w['id']} | {w['name'] or ''} | {w['cls'] or ''} | {w['map_height']} | "
                       f"{w['lod2_eave']} | {w['built_height']} |")
    return "\n".join(out) + "\n"


def _clean(v, nd=2):
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float):
        return round(v, nd) if math.isfinite(v) else None
    return v


def collect(st, spot, radius):
    """(rows, lod2_status) of the buildings of one spot within radius m of it."""
    import buildings as B
    import geodesy
    import mapgeom
    import osmcolours
    import osmyears
    import polygon
    import roads

    sid, name, lat, lon = spot
    frame = geodesy.EnuFrame(lat, lon)
    k_lo, k_hi = math.floor(-radius / roads.TILE), math.floor(radius / roads.TILE)
    tiles = [(tx, ty) for tx in range(k_lo, k_hi + 1) for ty in range(k_lo, k_hi + 1)]
    boxes = {t: roads.tile_bbox(*t) for t in tiles}
    xmin = min(b[0] for b in boxes.values())
    ymin = min(b[1] for b in boxes.values())
    xmax = max(b[2] for b in boxes.values())
    ymax = max(b[3] for b in boxes.values())

    # LoD2: the measured roofs per tile
    lod2 = []                                   # (box, Records or None)
    for t in tiles:
        rec = B.measured_roofs(st.md, frame, *boxes[t])
        lod2.append((boxes[t], rec))
    got = [r for _, r in lod2 if r is not None]
    if not got:
        lod2_status = "keine Shards erreichbar/vorhanden"
    else:
        lod2_status = f"{sum(len(r.e) for r in got)} Datensätze in {len(got)} Kachel(n)"

    # OSM construction years and colours over the spot's box
    try:
        years = osmyears.Fetch(frame, xmin, ymin, xmax, ymax, st.md).years()
    except Exception as ex:                     # (network may be blocked)
        logging.warning("osm years: %s", ex)
        years = None
    try:
        col_fetch = osmcolours.Fetch(frame, xmin, ymin, xmax, ymax, st.md)
        col_idx = osmcolours.Index(col_fetch.colours())
        if col_idx.c is None:
            col_idx = None
    except Exception as ex:
        logging.warning("osm colours: %s", ex)
        col_idx = None

    # raw footprints of the map cells the tiles touch, within the radius
    cell_keys = list(dict.fromkeys(k for t in tiles for k in roads.cells_of_tile(frame, *t, margin=0.0)))
    fps = []                                    # dicts: e, n, cx, cy, attrs, src_record, id
    seen = set()
    for ck in cell_keys:
        f = st.md.cell(ck, layers=("buildings",), kind="demand").layers.get("buildings")
        if f is None or len(f.kind) == 0:
            continue
        a = f.attrs
        ids = getattr(f, "ids", None)

        def get(k, i):
            return a[k][i] if k in a else None
        for i in range(len(f.kind)):
            if get("is_underground", i) is not None and bool(get("is_underground", i)):
                continue
            lv = get("level", i)
            if lv is not None and np.isfinite(lv) and lv < 0:
                continue
            parts = f.parts(i)
            if not parts or not parts[0]:
                continue
            outer = np.asarray(parts[0][0], np.float64)
            if len(outer) < 4:
                continue
            e, n = mapgeom.enu_xy(f, frame, outer)
            k = len(e) - 1
            cx, cy = float(e[:-1].sum() / k), float(n[:-1].sum() / k)
            if math.hypot(cx, cy) > radius:
                continue
            key = ids[i] if ids is not None else (round(cx, 2), round(cy, 2))
            if key in seen:
                continue
            seen.add(key)
            src_record = get("src_record", i)
            fps.append({"e": e, "n": n, "cx": cx, "cy": cy, "key": key, "src_record": src_record,
                        "attrs": {k2: get(k2, i) for k2 in
                                  ("class", "name", "height", "num_floors", "roof_shape", "roof_height",
                                   "height_src", "src_dataset", "facade_color", "roof_color")}})

    # years and colours by src_record (both optional)
    for fp in fps:
        fp["year"] = None
        if years is not None and fp["src_record"] is not None:
            y = int(osmyears.lookup(years, [fp["src_record"]])[0])
            fp["year"] = y or None
        if col_idx is not None and fp["src_record"] is not None:
            fac, roof = col_idx.lookup([fp["src_record"]])[0]
            if fac is not None and not fp["attrs"]["facade_color"]:
                fp["attrs"]["facade_color"] = fac
            if roof is not None and not fp["attrs"]["roof_color"]:
                fp["attrs"]["roof_color"] = roof

    # LoD2 match per footprint
    for fp in fps:
        fp["lod2"] = None
        for box, rec in lod2:
            if rec is None or not (box[0] <= fp["cx"] < box[2] and box[1] <= fp["cy"] < box[3]):
                continue
            j = rec.find(fp["e"], fp["n"])
            if j is not None:
                code = int(rec.code[j])
                fp["lod2"] = {"code": code, "shape": LOD2_SHAPES.get(code), "height": float(rec.height[j]),
                              "roof_h": float(rec.roof[j])}
            break

    # what the generator builds: specs by the footprint that contains their centroid (first match wins)
    specs = []                                  # (centroid (2,), spec)
    for t in tiles:
        x0, y0 = boxes[t][0], boxes[t][1]
        try:
            items, _custom = B.road_tile_items(frame, t[0], t[1], st.md, _ground(frame, st),
                                               urban_fn=st.surface.urban if st.surface is not None else None,
                                               widths=st.widths)
        except Exception:
            logging.exception("road_tile_items %s failed", t)
            continue
        for spec, _g0, _g1 in items:
            c = np.asarray(spec.ring, np.float64)[:, :2].mean(0) + (x0, y0)
            specs.append((c, spec))
    assigned = [None] * len(specs)
    if specs and fps:
        C = np.array([c for c, _ in specs])
        for j, fp in enumerate(fps):
            todo = np.array([i for i in range(len(specs)) if assigned[i] is None], np.int64)
            if len(todo) == 0:
                break
            ring = np.stack([fp["e"], fp["n"]], 1)
            P = C[todo]
            box = np.array([ring[:, 0].min(), ring[:, 1].min(), ring[:, 0].max(), ring[:, 1].max()])
            m = (P[:, 0] >= box[0]) & (P[:, 0] <= box[2]) & (P[:, 1] >= box[1]) & (P[:, 1] <= box[3])
            cand = todo[m]
            if len(cand) == 0:
                continue
            inside = np.asarray(polygon.contains_points(np.ascontiguousarray(ring),
                                                        np.ascontiguousarray(C[cand])), bool)
            for i in cand[inside]:
                assigned[i] = j
    per_fp = {}
    for i, j in enumerate(assigned):
        if j is not None:
            per_fp.setdefault(j, []).append(specs[i][1])

    rows = []
    for j, fp in enumerate(fps):
        mine = per_fp.get(j, [])
        hs = [s for s in mine if _finite_pos(s.height)]
        best = max(hs, key=lambda s: s.height) if hs else (mine[0] if mine else None)
        at = fp["attrs"]
        lod = fp["lod2"]
        area = abs(float(np.sum(fp["e"][:-1] * fp["n"][1:] - fp["e"][1:] * fp["n"][:-1]))) / 2
        la, lo, _ = frame.enu_to_geodetic(fp["cx"], fp["cy"], 0.0)
        lod_h = lod["height"] if lod else None
        lod_code = lod["code"] if lod else None
        map_roof = at["roof_shape"]
        row = {
            "id": f"{sid}-{len(rows) + 1:04d}",
            "lat": _clean(float(la), 6), "lon": _clean(float(lo), 6),
            "dist_m": _clean(math.hypot(fp["cx"], fp["cy"]), 1),
            "area_m2": _clean(area, 1),
            "footprint_src": footprint_source(at["src_dataset"]),
            "cls": at["class"], "name": at["name"],
            "map_height": at["height"], "map_floors": at["num_floors"],
            "map_height_src": at["height_src"], "map_roof_shape": map_roof,
            "map_roof_height": at["roof_height"],
            "lod2_match": lod is not None, "lod2_code": lod_code,
            "lod2_shape": lod["shape"] if lod else None, "lod2_height": lod_h,
            "lod2_roof_h": lod["roof_h"] if lod else None,
            "lod2_eave": _clean(lod["height"] - lod["roof_h"], 2) if lod else None,
            "height_src": height_source(at["height"], at["num_floors"], lod_h, at["height_src"]),
            "roof_src": roof_source(map_roof, lod_code),
            "built_parts": len(mine),
            "built_height": max(s.height for s in hs) if hs else None,
            "built_roof": best.roof if best is not None else None,
            "built_floors": best.floors if best is not None else None,
            "built_roof_mat": best.roof_mat if best is not None else None,
            "built_facade": best.facade if best is not None else None,
            "ml": bool(any(bool(s.ml) for s in mine)),
            "facade_color": at["facade_color"], "roof_color": at["roof_color"],
            "year": fp["year"],
        }
        rows.append({k: _clean(row[k], 6 if k in ("lat", "lon") else 2) for k in ROW_KEYS})
    return rows, lod2_status


def _ground(frame, st):
    import blockworker as bw

    def ground(la, lo):
        return bw.ground(frame, la, lo, st.dem.sample(la, lo, 15))
    return ground


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spots", default=None, help="comma list of spot ids (default all)")
    ap.add_argument("--radius", type=float, default=150.0)
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/beamng-earth"))
    ap.add_argument("--out", default=os.path.expanduser("~/bnge-tests/leonberg/report"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
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
    sections = ["# Gebäude-Bestandsaufnahme Leonberg", "",
                f"Radius {args.radius:g} m um jeden Testpunkt; Quellen: Karte (OSM / ML), LoD2, geschätzt.", ""]
    for spot in spots:
        sid = spot[0]
        try:
            rows, lod2_status = collect(st, spot, args.radius)
        except Exception as ex:             # (one spot failing: the others go on)
            logging.exception("%s failed", sid)
            print(f"{sid} FAILED {type(ex).__name__}: {ex}", flush=True)
            continue
        summary = summarise(rows)
        with open(os.path.join(args.out, f"{sid}.json"), "w", encoding="utf-8") as f:
            json.dump({"spot": list(spot), "lod2_status": lod2_status, "summary": summary, "rows": rows},
                      f, ensure_ascii=False, indent=1)
        with open(os.path.join(args.out, f"{sid}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ROW_KEYS)
            w.writeheader()
            w.writerows(rows)
        sections.append(markdown(spot, lod2_status, summary))
    report = "\n".join(sections)
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
