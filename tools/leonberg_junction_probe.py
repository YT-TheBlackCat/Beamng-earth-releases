"""Junction probe (Phase 0, Brief C5): what the junction code makes of a spot, as JSON and report.md.

    .venv/bin/python tools/leonberg_junction_probe.py                       # K1, K2, K3, W online (Overture)
    .venv/bin/python tools/leonberg_junction_probe.py --fixture K1 --fixture K2 --tag base   # offline, deterministic
    .venv/bin/python tools/leonberg_junction_probe.py --tag mergeM5 --set junctions.MERGE_M=5.0
    .venv/bin/python tools/leonberg_junction_probe.py --diff base mergeM5

Output: <out>/<tag>/<SPOT>.json and <out>/<tag>/report.md; --diff writes <out>/diff_A_B.md. Format: PHASE0_DESIGN.md 2.
"""
import argparse
import ast
import importlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPOTS = "K1:48.798180,9.014340;K2:48.801977,9.016582;K3:48.801257,9.016607;W:48.791984,9.005522"
PAD_M = 300.0                   # m: map cells and urban prefetch beyond the radius
URBAN_WAIT_S = 60.0             # s: the worldcover prefetch is polled this long at most
LINE_END_KEYS = (("junction_cut", "K"), ("cut_by_road", "S"), ("corner", "E"), ("tile_edge", "T"), ("road_end", "R"))
DIFF_NUM = ("d_edge", "stop_line.d", "crosswalk.d", "wait_line.d", "mast.d", "ai.d", "spread_m")
DIFF_TXT = ("regulation", "mapped_sign")
DIFF_MOVE_M = 0.3               # m: distances change only from this much
DIFF_NODE_M = 5.0               # m: junctions and graph nodes of two runs match within this
DIFF_END_M = 1.0                # m: line ends of two runs match within this


def all_urban(la, lo):
    return np.ones(np.shape(la), bool)


def parse_spots(text):
    """'NAME:lat,lon;NAME:lat,lon' -> [(name, lat, lon)]."""
    out = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        name, _, coords = part.rpartition(":")
        try:
            lat, lon = (float(v) for v in coords.split(","))
        except ValueError as e:
            raise SystemExit(f"--spots: expected NAME:lat,lon, got {part!r} ({e})")
        if not name:
            raise SystemExit(f"--spots: expected NAME:lat,lon, got {part!r}")
        out.append((name.strip(), lat, lon))
    return out


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


def git_info():
    """Short sha, '+dirty' when the tree has changes; None when git fails."""
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return None
    sha, dirty = git("rev-parse", "--short", "HEAD"), git("status", "--porcelain")
    if sha is None or dirty is None:
        return None
    return sha.strip() + ("+dirty" if dirty.strip() else "")


def fixture_inputs(name):
    """(lat, lon, roads, infra, routes, urban_fn, widths, sources) of a kreuz fixture (offline, flat, no widths)."""
    import kreuzfixtures
    fx = kreuzfixtures.load(name)
    sources = {"mode": f"fixture:{name}", "release": None, "lane_tags": bool(fx.sources.get("lane_tags")),
               "urban": "all", "widths": False}
    return fx.origin[0], fx.origin[1], fx.roads, fx.infra, fx.routes, all_urban, None, sources


def online_inputs(lat, lon, args, md):
    """The same inputs from Overture's map cells round the spot (blockworker's layers and urban/width sources)."""
    import leonberg_junction_fixture as jf
    import mapcells
    import roadwidth
    import surface
    pad = args.radius + PAD_M
    dlat = pad / 111320.0
    dlon = dlat / math.cos(math.radians(lat))
    keys = mapcells.cells_for_bbox(lat - dlat, lon - dlon, lat + dlat, lon + dlon)
    layers = {}
    for layer in ("roads", "infrastructure", "road_routes"):
        got = []
        for key in keys:
            try:
                f = md.cell(key, layers=(layer,), kind="demand").layers.get(layer)
            except Exception as e:      # (a cell that fails leaves its layer out, as blockworker._de_routes)
                print(f"  {layer} {key}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            if f is not None and len(f):
                got.append(f)
        layers[layer] = got
    urban_fn, urban = all_urban, "all"
    if args.urban == "worldcover":
        s = surface.Surface(md)
        s.prefetch(lat, lon, pad)
        t0, complete = time.time(), False
        while time.time() - t0 < URBAN_WAIT_S:
            _, complete = s.classes(np.array([lat]), np.array([lon]))
            if bool(complete):
                break
            time.sleep(2)
        if complete:
            urban_fn, urban = s.urban, "worldcover"
        else:
            print("worldcover not complete after 60 s: urban all", file=sys.stderr)
    widths = None
    if not args.no_widths:
        widths = roadwidth.RoadWidths(
            lambda k: md.cell(k, layers=("buildings",), kind="demand").layers.get("buildings"),
            roadwidth.store_dir(args.cache, md)).get
    sources = {"mode": "online", "release": md.release, "lane_tags": any(jf.has_lane_tags(f) for f in layers["roads"]),
               "urban": urban, "widths": widths is not None}
    return (lat, lon, layers["roads"], layers["infrastructure"], layers["road_routes"], urban_fn, widths, sources)


def probe_spot(name, lat, lon, inputs, args, sources, meta):
    """The probe's result dict for one spot: the core run, its junctions with their approaches' lanes, and collect()."""
    import furniture
    import leonberg_junction_core as core
    import leonberg_junction_lines as lines
    roads, infra, routes, urban_fn, widths = inputs
    frame, origin = core.build_frame(lat, lon, args.grid_shift)
    run = core.run(frame, origin, args.radius, roads, infra, routes, urban_fn=urban_fn, widths=widths,
                   de_on=not args.no_de)
    js = core.junction_table(run)
    gt = core.graph_table(run, js)
    ends, stubs = lines.line_ends(run, js)
    arms = furniture._arms(run.roads)
    for j in js:
        for a in j["approaches"]:
            if j["decided_by"] == "model":      # (the model's arm: from the junction's centre; arm_pieces needs fid, p)
                c = np.asarray(run.origin, np.float64)[:2] + np.array([j["x"], j["y"]])
                arm = (a["fid"], None, None, c, a["end"] == "start", 0.0)
            else:
                arm = next((x for x in arms.get(tuple(j["key"]), []) if str(x[0]) == a["fid"]
                            and bool(x[4]) == (a["end"] == "start")), None)
            a["lanes"] = lines.arm_pieces(run, arm) if arm is not None else []
    spot = {"name": name, "lat": lat, "lon": lon, "radius_m": args.radius}
    return collect(spot, run, js, gt, ends, stubs, sources, meta)


def collect(spot, run, junctions, graph, ends, stubs, sources, settings):
    """The result dict of one spot (schema 1, PHASE0_DESIGN 2.2). settings: the run block (tag, time, git, argv,
    settings, de_on). Sets summary from the junctions, line ends and props of run."""
    apps = [a for j in junctions for a in j["approaches"]]
    reasons = {}
    for e in ends:
        reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
    spreads = [a["spread_m"] for a in apps if a.get("spread_m") is not None]
    seen, us_stop = set(), 0
    for row in run.props:
        if len(row) < 5:
            continue
        key = (str(row[0]), round(float(row[1]), 1), round(float(row[2]), 1))
        if key in seen:
            continue
        seen.add(key)
        if key[0] == "stop" and math.hypot(float(row[1]) - run.origin[0], float(row[2]) - run.origin[1]) <= run.radius:
            us_stop += 1
    summary = {
        "junctions": len(junctions), "approaches": len(apps),
        "inbound": sum(1 for a in apps if a["inbound"]),
        "signal_approaches": sum(1 for a in apps if a["inbound"] and a["regulation"] == "signal"),
        "masts": sum(1 for a in apps if a["mast"]),
        "ai_signal": sum(1 for a in apps if a.get("ai") and a["ai"]["kind"] == "signal"),
        "mast_without_ai": sum(1 for a in apps if a["mast"] and not a.get("ai")),
        "spread_max_m": max(spreads) if spreads else None,
        "stop_lines": sum(1 for a in apps if a["stop_line"]),
        "crosswalks_us": sum(1 for a in apps if a["crosswalk"] and a["crosswalk"]["look"] == "us"),
        "zebras": sum(1 for a in apps if a["crosswalk"] and a["crosswalk"]["look"] == "zebra"),
        "wait_lines": sum(1 for a in apps if a["wait_line"]),
        "us_stop_signs": us_stop,
        "line_ends": dict(sorted(reasons.items())), "stubs": len(stubs),
    }
    return {"schema": 2, "tool": "leonberg_junction_probe", "run": dict(settings), "sources": sources, "spot": spot,
            "timing": run.timing, "junctions": junctions, "line_ends": ends, "line_stubs": stubs, "graph": graph,
            "summary": summary}


# ---- report.md ----

def _f(v, nd=1):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "/").replace("\n", " ") for c in r) + " |")
    return out


def _get(d, path):
    for part in path.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(part)
    return d


def _mast(a):
    m = a.get("mast")
    return f"{m['d']:.1f} r{m['lat']:.1f}" if m else "–"


def _approach_rows(junctions):
    rows = []
    for j in junctions:
        node = f"{j['id']} ({j['x']:.1f}, {j['y']:.1f}) {j['n_arms']} Arme"
        if j["cluster"]:
            node += ", Cluster " + ", ".join(j["cluster"])
        for a in j["approaches"]:
            arm = f"{a['name'] or '–'} {a['fid'][:8]} {a['end']} {_f(a['bearing'], 0)}°"
            sl, cw, wl, ai = a.get("stop_line"), a.get("crosswalk"), a.get("wait_line"), a.get("ai")
            pieces = "; ".join(f"{str(p['fid'])[:8]} {p['len_m']:.0f} m "
                               f"{p['layout'][1] if p.get('layout') else '–'} Sp." for p in a.get("lanes", []))
            rows.append([node, arm, a["cls"], ("ein" if a["inbound"] else "aus") + ("/1" if a["oneway"] else "/2"),
                         a["regulation"], a["mapped_sign"] or "–", _f(a["d_edge"]),
                         f"{sl['d']:.1f} ({sl['w']:.2f} m, {sl['src']})" if sl else "–",
                         f"{cw['d']:.1f} {cw['look']}" if cw else "–",
                         f"{wl['d']:.1f} n{wl['n']}" if wl else "–", _mast(a),
                         f"{ai['d']:.1f} {ai['kind']}" if ai else "–", _f(a.get("spread_m")), pieces or "–",
                         a.get("group") or "–", a.get("role") or "–", _f(a.get("d_stop")),
                         ", ".join(f"{k} {v}" for k, v in a.get("why", {}).items()) or "–"])
            node = ""
    return rows


def _group_rows(junctions):
    """One row per signal group (the approaches' group key): its junctions, its approaches, its signalled inbound."""
    groups = {}
    for j in junctions:
        for a in j["approaches"]:
            if a.get("group"):
                g = groups.setdefault(a["group"], {"j": [], "n": 0, "sig": 0})
                if j["id"] not in g["j"]:
                    g["j"].append(j["id"])
                g["n"] += 1
                g["sig"] += 1 if a["inbound"] and a["regulation"] == "signal" else 0
    return [[k, ", ".join(v["j"]), v["n"], v["sig"]] for k, v in sorted(groups.items())]


def _graph_rows(graph):
    rows = []
    for g in graph:
        ins = []
        for e in g["in"]:
            s = (f"{e['from']} {e['lanes']} Sp. {e['kmh']} km/h {e['drivability']} "
                 f"Ph {e['phase']} d {_f(e['stop_d'])}")
            if e.get("skipped"):
                s += f" übersprungen: {e['skipped']}"
            ins.append(s)
        rows.append([g["node"], g["junction"] or "–", g["controller"] or "–", "; ".join(ins) or "–"])
    return rows


def _spot_section(res):
    sp, s = res["spot"], res["summary"]
    out = [f"## {sp['name']} ({sp['lat']:.6f}, {sp['lon']:.6f}, r {sp['radius_m']:.0f} m)", "",
           "### Zufahrten"]
    out += _table(["Knoten", "Arm", "Klasse", "ein/2", "Regelung", "Karte", "d_edge", "Haltlinie", "Furt",
                   "Wartelinie", "Mast", "KI-Halt", "Spread", "Teilstücke", "Gruppe", "Rolle", "d_stop", "Gründe"],
                  _approach_rows(res["junctions"]))
    out += ["", "### Gruppen"]
    out += _table(["Gruppe", "Knoten", "Zufahrten", "Ampel-Zuf. (einfahrend)"], _group_rows(res["junctions"]) or [["–"] * 4])
    out += ["", "### Linienenden (Anzahl je Grund)"]
    out += _table(["Grund", "Anzahl"], list(s["line_ends"].items()) or [["–", 0]])
    out += ["", "### Linienenden cut_by_road / unknown / fully_cut (Einzelliste)"]
    odd = [e for e in res["line_ends"] if e["reason"] in ("cut_by_road", "unknown", "fully_cut")]
    out += _table(["fid", "Art", "Seite", "x, y", "Grund", "Detail", "Verursacher", "Knoten", "d"],
                  [[str(e["fid"])[:8], e["kind"] or "–", e["side"] or "–", f"{e['x']:.1f}, {e['y']:.1f}",
                    e["reason"], e["detail"] or "–", str(e["culprit"])[:8] if e["culprit"] else "–",
                    e["junction"] or "–", _f(e["d_junction"])] for e in odd])
    out += ["", "### Fahrspurgraph"]
    out += _table(["Knoten", "J", "Controller", "Zufahrt (von, Spuren, km/h, Fahrb., Ampel/Phase, d_stop, übersprungen)"],
                  _graph_rows(res["graph"]))
    return out + [""]


def _overview_row(res):
    sp = res["spot"]
    if "error" in res:
        return [sp["name"], f"FEHLER: {res['error']}"] + ["–"] * 15
    s, t = res["summary"], res["timing"]
    ends = s["line_ends"]
    kstr = "/".join(str(ends.get(reason, 0)) for reason, _ in LINE_END_KEYS)
    return [sp["name"], s["junctions"], s["approaches"], s["signal_approaches"], s["masts"], s["ai_signal"],
            s["mast_without_ai"], _f(s["spread_max_m"]), s["stop_lines"], s["crosswalks_us"], s["zebras"],
            s["wait_lines"], s["us_stop_signs"], kstr, s["stubs"], _f(sum(t["build_tile_s"])), _f(t["navgraph_s"], 2)]


def report_md(results, head):
    """report.md: overview table, then per spot the approaches, line ends and the graph (PHASE0_DESIGN 2.3).
    head: {tag, time, git, settings: [[name, old, new]]}."""
    sources = results[0]["sources"] if results and "sources" in results[0] else {}
    out = [f"# Junction probe · tag `{head['tag']}` ({head['time']}, git {head['git'] or '–'})",
           f"Quellen: {_fmt_sources(sources)}",
           "Gesetzte Werte: " + ("; ".join(f"{n} {o} → {v}" for n, o, v in head["settings"]) or "(keine)"), "",
           "## Übersicht"]
    out += _table(["Stelle", "Knoten", "Zufahrten", "Ampel-Zuf.", "Masten", "KI-Ampel", "Mast ohne KI",
                   "max Spread m", "Haltlinien", "Furten US", "Zebras", "Wartelinien", "US-STOP",
                   "Linienenden K/S/E/T/R", "Stummel", "build s", "graph s"], [_overview_row(r) for r in results])
    out.append("")
    for res in results:
        if "error" in res:
            out += [f"## {res['spot']['name']}", "", f"Fehler: {res['error']}", ""]
        else:
            out += _spot_section(res)
    return "\n".join(out) + "\n"


def _fmt_sources(src):
    if not src:
        return "–"
    grid = src.get("grid_shift")
    return (f"{src.get('mode')} · release {src.get('release') or '–'} · lane_tags "
            f"{'ja' if src.get('lane_tags') else 'nein'} · borders {src.get('borders', '–')} · "
            f"urban {src.get('urban')} · widths {'ja' if src.get('widths') else 'nein'} · elev flat · "
            f"grid_shift {grid or '–'}")


# ---- --diff ----

def _near(p, q, m):
    return math.hypot(p[0] - q[0], p[1] - q[1]) <= m


def _by_name(results):
    return {r["spot"]["name"]: r for r in results}


def _approach_changes(a, b):
    ch = []
    for path in DIFF_TXT:
        va, vb = _get(a, path), _get(b, path)
        if va != vb:
            ch.append(f"{path} {va or '–'} → {vb or '–'}")
    for path in DIFF_NUM:
        va, vb = _get(a, path), _get(b, path)
        if va is None or vb is None:
            if va != vb:
                ch.append(f"{path} {_f(va, 2)} → {_f(vb, 2)}")
        elif abs(va - vb) > DIFF_MOVE_M:
            ch.append(f"{path} {va:.2f} → {vb:.2f}")
    if (a.get("ai") is None) != (b.get("ai") is None):
        ch.append(f"ai {'ja' if a.get('ai') else 'nein'} → {'ja' if b.get('ai') else 'nein'}")
    pa = [p["layout"][1] for p in a.get("lanes", [])]
    pb = [p["layout"][1] for p in b.get("lanes", [])]
    if pa != pb:
        ch.append(f"Teilstücke Spuren {pa} → {pb}")
    return ch


def _diff_spot(ra, rb):
    out = []
    sa, sb = ra["summary"], rb["summary"]
    for k in sa:
        if k == "line_ends":
            for reason in sorted(set(sa[k]) | set(sb[k])):
                va, vb = sa[k].get(reason, 0), sb[k].get(reason, 0)
                if va != vb:
                    out.append(f"- Linienenden {reason}: {va} → {vb}")
        elif sa[k] != sb[k]:
            out.append(f"- {k}: {_f(sa[k], 2)} → {_f(sb[k], 2)}")
    ja, jb = ra["junctions"], rb["junctions"]
    for j in jb:
        if not any(_near((j["x"], j["y"]), (i["x"], i["y"]), DIFF_NODE_M) for i in ja):
            out.append(f"- Knoten neu: {j['id']} ({j['x']:.1f}, {j['y']:.1f})")
    for i in ja:
        if not any(_near((i["x"], i["y"]), (j["x"], j["y"]), DIFF_NODE_M) for j in jb):
            out.append(f"- Knoten entfallen: {i['id']} ({i['x']:.1f}, {i['y']:.1f})")
    apa = {(a["fid"], a["end"]): a for j in ja for a in j["approaches"]}
    apb = {(a["fid"], a["end"]): a for j in jb for a in j["approaches"]}
    for key in sorted(set(apa) | set(apb)):
        if key not in apb:
            out.append(f"- Zufahrt entfallen: {apa[key]['name'] or '–'} {key[0][:8]} {key[1]}")
        elif key not in apa:
            out.append(f"- Zufahrt neu: {apb[key]['name'] or '–'} {key[0][:8]} {key[1]}")
        else:
            ch = _approach_changes(apa[key], apb[key])
            if ch:
                out.append(f"- Zufahrt {apa[key]['name'] or '–'} {key[0][:8]} {key[1]}: " + "; ".join(ch))
    ends_a = [(e["fid"], e["kind"], e["side"], e["x"], e["y"]) for e in ra["line_ends"]]
    ends_b = [(e["fid"], e["kind"], e["side"], e["x"], e["y"]) for e in rb["line_ends"]]
    for name, src, other in (("neu", ends_b, ends_a), ("entfallen", ends_a, ends_b)):
        missing = [e for e in src if not any(e[:3] == o[:3] and _near(e[3:], o[3:], DIFF_END_M) for o in other)]
        for e in missing[:50]:
            out.append(f"- Linienende {name}: {e[0][:8]} {e[1]} {e[2]} ({e[3]:.1f}, {e[4]:.1f})")
        if len(missing) > 50:
            out.append(f"- Linienenden {name}: +{len(missing) - 50} weitere")
    ga, gb = ra["graph"], rb["graph"]
    for g in gb:
        match = next((h for h in ga if _near((g["x"], g["y"]), (h["x"], h["y"]), DIFF_NODE_M)), None)
        if match is None:
            out.append(f"- Graphknoten neu: {g['node']} ({g['controller'] or '–'}, {len(g['in'])} Zufahrten)")
            continue
        sk_a = sorted(e["from"] for e in match["in"] if e.get("skipped"))
        sk_b = sorted(e["from"] for e in g["in"] if e.get("skipped"))
        if match["controller"] != g["controller"] or len(match["in"]) != len(g["in"]) or sk_a != sk_b:
            out.append(f"- Graphknoten {g['node']}: Controller {match['controller'] or '–'} → {g['controller'] or '–'}, "
                       f"Zufahrten {len(match['in'])} → {len(g['in'])}, übersprungen {len(sk_a)} → {len(sk_b)}")
    for h in ga:
        if not any(_near((h["x"], h["y"]), (g["x"], g["y"]), DIFF_NODE_M) for g in gb):
            out.append(f"- Graphknoten entfallen: {h['node']} ({h['controller'] or '–'})")
    return out


def diff_md(a, b):
    """Diff of two runs' results (lists of spot dicts): changed fields, matched by spot, position and (fid, end)."""
    ha = a[0]["run"] if a else {}
    hb = b[0]["run"] if b else {}
    out = [f"# Junction-Diff · `{ha.get('tag')}` → `{hb.get('tag')}`", "",
           f"A: {ha.get('tag')} ({ha.get('time')}, git {ha.get('git') or '–'}), "
           f"Gesetzte Werte: {ha.get('settings') or '(keine)'}",
           f"B: {hb.get('tag')} ({hb.get('time')}, git {hb.get('git') or '–'}), "
           f"Gesetzte Werte: {hb.get('settings') or '(keine)'}"]
    src_a = {r["spot"]["name"]: r.get("sources") for r in a}
    src_b = {r["spot"]["name"]: r.get("sources") for r in b}
    if src_a != src_b:
        out.append("WARNUNG: Quellen unterscheiden sich, der Vergleich gilt nur bedingt.")
    out.append("")
    byA, byB = _by_name(a), _by_name(b)
    for name in list(dict.fromkeys([r["spot"]["name"] for r in a] + [r["spot"]["name"] for r in b])):
        out.append(f"## {name}")
        ra, rb = byA.get(name), byB.get(name)
        if ra is None or rb is None:
            out += [f"- nur in {'B' if ra is None else 'A'}", ""]
            continue
        if "error" in ra or "error" in rb:
            out += [f"- Fehler A: {ra.get('error', '–')} · Fehler B: {rb.get('error', '–')}", ""]
            continue
        lines = _diff_spot(ra, rb)
        out += lines if lines else ["- keine Änderung"]
        out.append("")
    return "\n".join(out) + "\n"


def load_tag(out_dir, tag):
    tag_dir = out_dir / tag
    if not tag_dir.is_dir():
        raise SystemExit(f"--diff: no run {tag!r} in {tag_dir.parent}")
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(tag_dir.glob("*.json"))]


def check_tag(tag):
    if not tag or any(c in tag for c in "/\\") or tag in (".", ".."):
        raise SystemExit(f"--tag must be a plain folder name, got {tag!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spots", default=DEFAULT_SPOTS, help="NAME:lat,lon;NAME:lat,lon (online spots)")
    ap.add_argument("--fixture", action="append", default=[], metavar="NAME",
                    help="a kreuz fixture (fixtures/kreuz_NAME.npz) instead of the online spots (repeatable)")
    ap.add_argument("--radius", type=float, default=80.0)
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/beamng-earth"))
    ap.add_argument("--out", default=os.path.expanduser("~/bnge-tests/leonberg/junctions"))
    ap.add_argument("--tag", default="base", help="outputs go to <out>/<TAG>/ (default: base)")
    ap.add_argument("--set", action="append", default=[], metavar="MODULE.ATTR=VALUE",
                    help="override a server constant before the run, e.g. junctions.MERGE_M=5.0 (repeatable)")
    ap.add_argument("--urban", choices=("worldcover", "all"), default="worldcover")
    ap.add_argument("--no-widths", action="store_true")
    ap.add_argument("--no-de", action="store_true")
    ap.add_argument("--borders", choices=("fixture", "online"), default="fixture")
    ap.add_argument("--grid-shift", default="0,0", metavar="DX,DY", help="tile grid shift against the spot (m)")
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"), help="compare two tags: writes <out>/diff_A_B.md")
    args = ap.parse_args()
    out_dir = Path(args.out).expanduser()
    if args.diff:
        for tag in args.diff:
            check_tag(tag)
        text = diff_md(load_tag(out_dir, args.diff[0]), load_tag(out_dir, args.diff[1]))
        path = out_dir / f"diff_{args.diff[0]}_{args.diff[1]}.md"
        path.write_text(text, encoding="utf-8")
        print(path)
        return
    check_tag(args.tag)
    try:
        args.grid_shift = tuple(float(v) for v in args.grid_shift.split(","))
        assert len(args.grid_shift) == 2
    except (ValueError, AssertionError):
        raise SystemExit(f"--grid-shift: expected DX,DY, got {args.grid_shift!r}")

    # the server reads these at import: set them first
    os.environ["BNGE_PLANCACHE"] = "off"
    if args.borders == "fixture":
        os.environ["BNE_REGIONAL_DIR"] = str(ROOT / "server" / "tests" / "fixtures" / "regional")
    sys.path[:0] = [str(ROOT / "server"), str(ROOT / "server" / "tests"), str(ROOT / "tools")]
    import germany                      # noqa: E402
    import mapdata as mapdatamod        # noqa: E402
    import netfetch                     # noqa: E402

    borders = "absent" if germany.ABSENT in germany.signature() else args.borders
    applied = apply_settings(args.set)
    meta = {"tag": args.tag, "time": time.strftime("%Y-%m-%d %H:%M"), "git": git_info(), "argv": sys.argv[1:],
            "settings": [[n, repr(o), repr(v)] for n, o, v in applied], "de_on": not args.no_de}
    jobs = [("fixture", n, None, None) for n in args.fixture] or \
           [("online", n, la, lo) for n, la, lo in parse_spots(args.spots)]
    tag_dir = out_dir / args.tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    results, md = [], None
    for kind, name, lat, lon in jobs:
        try:
            if kind == "fixture":
                lat, lon, roads, infra, routes, urban_fn, widths, sources = fixture_inputs(name)
            else:
                if md is None:
                    md = mapdatamod.MapData(args.cache, fetcher=netfetch.Fetcher())
                lat, lon, roads, infra, routes, urban_fn, widths, sources = online_inputs(lat, lon, args, md)
            sources.update({"borders": borders, "elev": "flat", "grid_shift": list(args.grid_shift)})
            res =probe_spot(name, lat, lon, (roads, infra, routes, urban_fn, widths), args, sources, meta)
        except Exception as e:      # a spot that throws does not stop the run
            res = {"spot": {"name": name, "lat": lat, "lon": lon, "radius_m": args.radius},
                   "error": f"{type(e).__name__}: {e}"}
            print(f"{name}: {res['error']}", file=sys.stderr)
        results.append(res)
        (tag_dir / f"{name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str),
                                              encoding="utf-8")
        print(f"{name}: {'FEHLER' if 'error' in res else 'ok'}", flush=True)
    report = report_md(results, meta)
    (tag_dir / "report.md").write_text(report, encoding="utf-8")
    print(tag_dir / "report.md")


if __name__ == "__main__":
    main()
