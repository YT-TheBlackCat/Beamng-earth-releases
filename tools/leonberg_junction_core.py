"""Junction probe core (Phase 0, Brief C3): one spot's tiles and KI graph, built offline from the features it is given.

The caller sets BNE_REGIONAL_DIR and sys.path (server/) first; the server modules are imported inside the functions.
run() gives the tiles' diagnostics, junction_table() the junctions with their approaches (the JSON of PHASE0_DESIGN
section 2.2, lines and stubs left to Brief C4), graph_table() the graph nodes. Coordinates are frame metres.
"""
import math
import time
from types import SimpleNamespace

import numpy as np

SPOT_OFF = 128.0           # m: the spot's place in its frame (build_frame)
CLUSTER_M = 35.0           # m: junctions this close share a cluster
ARM_M = 60.0               # m: an approach's own road, from the junction out
TOUCH_M = 3.0              # m: a junction's graph node is this close to its centre
ARM_STEP_M = 0.25          # m: steps of the edge search (d_edge)
GRAPH_MIN_NB = 3           # graph nodes with fewer distinct neighbours are not tabled
REGULATION = {0: "none", 1: "z205", 2: "z206", 3: "z306", 4: "z301", 5: "rvl", 6: "exit"}   # designs' what codes
REG_STR = {"signal": "signal", "signal_left": "signal", "giveway": "yield", "stop": "stop"}  # furniture's G_*
MAPPED_SIGN = {"stop": "stop", "giveway": "give_way"}                                       # _mapped_priority
SIGN_GROUPS = {"stop", "giveway", "giveway_b1"}
MAST_GROUPS = {"signal", "signal_left"}


def _f(v, n=2):
    return round(float(v), n)


def _xy(p, origin):
    return [_f(p[0] - origin[0]), _f(p[1] - origin[1])]


def build_frame(lat, lon, shift=(0.0, 0.0)):
    """(frame, origin_xy): the frame's origin is the spot - (128, 128) m + shift; origin_xy is the spot in that frame."""
    import geodesy
    base = geodesy.EnuFrame(lat, lon)
    olat, olon, _ = base.enu_to_geodetic(float(shift[0]) - SPOT_OFF, float(shift[1]) - SPOT_OFF, 0.0)
    frame = geodesy.EnuFrame(float(olat), float(olon))
    e, n, _ = frame.geodetic_to_enu(lat, lon, 0.0)
    return frame, np.array([float(e), float(n)])


def tiles_for(origin_xy, radius):
    """Every tile (roads.TILE square) whose box cuts the square origin +- radius."""
    import roads
    T = roads.TILE
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    xs = range(math.floor((ox - radius) / T), math.ceil((ox + radius) / T))
    ys = range(math.floor((oy - radius) / T), math.ceil((oy + radius) / T))
    return [(tx, ty) for tx in xs for ty in ys]


def _flat(la, lo):
    return np.zeros(np.shape(la))


def run(frame, origin_xy, radius, roads_cells, infra_cells=(), routes_cells=(), urban_fn=None, widths=None,
        de_on=True, signals=None, nodes=None):
    """Tiles and graph of the spot's features; see the module doc. Returns a SimpleNamespace (fields in the brief)."""
    import furniture
    import navgraph
    import roads
    import secrossings
    origin_xy = np.asarray(origin_xy, np.float64)[:2]
    if infra_cells:
        signals = furniture.mapped_signals(frame, infra_cells)
        nodes = secrossings.nodes(frame, infra_cells)
    tiles = tiles_for(origin_xy, radius)
    diags, props, build_s, seen, all_roads = [], [], [], set(), []
    for tx, ty in tiles:
        diag, tile_props = {}, []
        t0 = time.perf_counter()
        roads.build_tile(frame, tx, ty, roads_cells, _flat, urban_fn, walks=[], widths=widths, props=tile_props,
                         signals=signals, nodes=nodes, country="DE", de_on=de_on, de_routes=list(routes_cells),
                         diag=diag)
        build_s.append(_f(time.perf_counter() - t0, 3))
        diags.append(diag)
        props.extend(tile_props)
        for r in diag.get("roads", ()):
            key = (str(r[0]), round(float(r[2][0][0]), 1), round(float(r[2][0][1]), 1))
            if key not in seen:
                seen.add(key)
                all_roads.append(r)
    graph_diag = {}
    t0 = time.perf_counter()
    graph = navgraph.build(frame, roads_cells, _flat, tuple(origin_xy), urban_fn, window=radius + 300, widths=widths,
                           signals=signals, country="DE", priority=nodes["priority"] if nodes else None,
                           nodes=nodes, de_on=de_on, de_routes=list(routes_cells), diag=graph_diag)
    navgraph_s = _f(time.perf_counter() - t0, 3)
    feats_of = {}
    for f in roads_cells:
        for i, fid in enumerate(f.ids):
            feats_of[str(fid)] = (f, i)
    return SimpleNamespace(frame=frame, origin=origin_xy, radius=radius, tiles=tiles, diags=diags, props=props,
                           roads=all_roads, feats_of=feats_of, graph=graph, graph_diag=graph_diag, signals=signals,
                           nodes=nodes, timing={"build_tile_s": build_s, "navgraph_s": navgraph_s})


def _line60(line):
    """The first ARM_M metres of a centreline (k, 2) and its unit direction (the chord of that piece)."""
    seg = np.hypot(*np.diff(line, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    keep = line[cum <= ARM_M]
    if len(keep) < 2:
        keep = line[:2]
    chord = keep[-1] - keep[0]
    n = float(np.hypot(*chord))
    return keep, (chord / n if n > 1e-9 else np.array([0.0, 1.0]))


def _arm(a, pieces):
    """Arm record: the arm tuple, its centreline from the junction (k, 2), its first 60 m and unit vector u."""
    import designs
    line = np.asarray(designs._arm_line(a, pieces), np.float64)[:, :2]
    keep, u = _line60(line)
    inc = -u
    right = np.array([inc[1], -inc[0]])
    return {"a": a, "fid": str(a[0]), "start": bool(a[4]), "line": line, "keep": keep, "u": u, "right": right,
            "width": float(a[1].width)}


def _arm_model(arm):
    """Arm record of a junction model arm: its own line from the junction out (junctionmodel's Arm.line), as _arm."""
    line = np.asarray(arm.line, np.float64)[:, :2]
    keep, u = _line60(line)
    inc = -u
    right = np.array([inc[1], -inc[0]])
    return {"a": arm.as_tuple(), "fid": str(arm.fid), "start": bool(arm.at_start), "line": line, "keep": keep, "u": u,
            "right": right, "width": float(arm.spec.width), "arm": arm}


def _approach_info(ap):
    """The model's fields of an approach (junctionapproach.Approach.as_dict); none without the model's approach."""
    if ap is None:
        return {"group": None, "role": None, "d_stop": None, "link": False, "why": {}}
    d = ap.as_dict()
    return {k: d[k] for k in ("group", "role", "d_stop", "link", "why")}


def _model_junctions(run):
    """[(Junction, apps)] of the junction model's junctions (three or more arms, centre within run.radius). Each comes
    from the first tile whose reach (designs.tile_reach) holds its centre; apps: that tile's Approach by arm id.
    [] without a model (junctionmodel off or not Germany)."""
    import designs
    import roads
    out, seen = [], set()
    for (tx, ty), d in zip(run.tiles, run.diags):
        model = d.get("model")
        if model is None:
            continue
        x0, y0, x1, y1 = designs.tile_reach(roads.tile_bbox(tx, ty))
        for k, J in model.junctions.items():
            c = np.asarray(J.centre, np.float64)[:2]
            if k in seen or len(J.arms) < 3 or not (x0 <= c[0] <= x1 and y0 <= c[1] <= y1):
                continue
            if float(np.hypot(*(c - run.origin))) > run.radius:
                continue
            seen.add(k)
            out.append((J, d.get("approaches") or {}))
    return out


def _dist_line(q, line):
    if len(line) < 2:
        return float(np.hypot(*(line[0] - q))) if len(line) else math.inf
    a, b = line[:-1], line[1:]
    d = b - a
    L2 = np.maximum((d * d).sum(1), 1e-12)
    t = np.clip(((q - a) * d).sum(1) / L2, 0.0, 1.0)
    return float(np.hypot(*(a + t[:, None] * d - q).T).min())


def _reg(what):
    if isinstance(what, (int, np.integer)):
        return REGULATION.get(int(what), str(what))
    return REG_STR.get(str(what), str(what))


def _graph_node(p, gnodes):
    best = None
    for n, v in gnodes.items():
        d = math.hypot(v[0] - p[0], v[1] - p[1])
        if d <= TOUCH_M and (best is None or d < best[0]):
            best = (d, n)
    return None if best is None else best[1]


def junction_table(run):
    """[Junction] of run's junctions (three or more arms, centre within run.radius), nearest first."""
    import furniture
    import designs
    import islands
    origin = run.origin
    J_all = furniture._arms(run.roads)
    pieces = designs._pieces(run.roads)
    de = {}                       # junction key -> the kernel's controls (decide's [(arm, what, plate)])
    for d in run.diags:
        for k, v in (d.get("de_ctl") or {}).items():
            if v is not None and v[1] is not None and k not in de:
                de[k] = v[1]
    cands = []                    # (dist, key, arm tuples | Junction, apps | None, centre)
    model_js = _model_junctions(run)
    if model_js:                  # Germany: the junction model's junctions and their approaches
        for J, apps in model_js:
            p = np.asarray(J.centre, np.float64)[:2]
            cands.append((float(np.hypot(*(p - origin))), J.key, J, apps, p))
    else:                         # no model (junctionmodel off or not Germany): furniture's arms
        for k, arms in J_all.items():
            if len(arms) < 3:
                continue
            p = np.mean([a[3][:2] for a in arms], axis=0)
            dist = float(np.hypot(*(p - origin)))
            if dist <= run.radius:
                cands.append((dist, k, arms, None, p))
    cands.sort(key=lambda c: c[0])
    link_ok = furniture._link_check(J_all, run.signals)
    ring = islands.ring_fids(run.roads)
    priority = run.nodes["priority"] if run.nodes else None
    gnodes = {n: v for n, v in run.graph.get("nodes", {}).items() if str(n).startswith("c")}
    appr = run.graph_diag.get("approaches", {})
    skipped = run.graph_diag.get("skipped", [])
    signals_by_j = {}
    for s in run.graph.get("signals") or []:
        signals_by_j.setdefault(s["j"], []).append(s)
    # the crosswalks, stop lines, zebras, teeth and props of all tiles, once each
    cws, seen_cw = [], set()
    for d in run.diags:
        for cw in d.get("crosswalks") or []:
            a, b = np.asarray(cw["a"], np.float64)[:2], np.asarray(cw["b"], np.float64)[:2]
            key = (round(a[0], 1), round(a[1], 1), round(b[0], 1), round(b[1], 1))
            if key not in seen_cw:
                seen_cw.add(key)
                cws.append((a, b, cw))
    for d in run.diags:
        for ends, half in d.get("zebras") or []:
            e = np.asarray(ends, np.float64)[:, :2]
            key = ("z", round(e[0, 0], 1), round(e[0, 1], 1), round(e[1, 0], 1), round(e[1, 1], 1))
            if key not in seen_cw:
                seen_cw.add(key)
                cws.append((e[0], e[1], {"w": 2 * float(half), "look": "zebra", "zebra": True}))
    teeth, seen_t = [], set()
    for d in run.diags:
        for poly in d.get("teeth") or []:
            c = np.asarray(poly, np.float64)[:, :2].mean(axis=0)
            key = (round(c[0], 1), round(c[1], 1))
            if key not in seen_t:
                seen_t.add(key)
                teeth.append(c)
    props, seen_p = [], set()
    for row in run.props:
        if len(row) < 5:
            continue
        key = (str(row[0]), round(float(row[1]), 1), round(float(row[2]), 1))
        if key not in seen_p:
            seen_p.add(key)
            props.append(row)

    # the junctions with their arms, for the assignment of points to (junction, arm)
    juncs = []
    for dist, k, arms, apps, p in cands:
        if apps is None:
            recs = [_arm(a, pieces) for a in arms]
            juncs.append({"k": k, "arms": arms, "p": p, "recs": recs, "dist": dist, "id": None, "model": None})
        else:
            recs = [_arm_model(arm) for arm in arms.arms]
            juncs.append({"k": k, "arms": arms.arm_tuples(), "p": p, "recs": recs, "dist": dist, "id": None,
                          "model": (arms, apps)})
    for i, J in enumerate(juncs, 1):
        J["id"] = f"J{i}"

    def assign(q, yaw=None):
        """(J, arm rec, d, lat) of a point; yaw (a signal head's): only the arms whose approach it faces (at a corner
        the nearest arm line is as often the cross street's), unless none does."""
        best = None
        for J in juncs:
            dd = math.hypot(q[0] - J["p"][0], q[1] - J["p"][1])
            if dd <= designs.JUNCTION_PAINT_M and (best is None or dd < best[0]):
                best = (dd, J)
        if best is None:
            return None
        J, pick = best[1], None
        recs = J["recs"]
        if yaw is not None:
            def faces(A):
                want = furniture.approach_pose(A["u"], furniture.G_SIGNAL)[1]
                return abs((float(yaw) - want + 180.0) % 360.0 - 180.0) <= 20.0
            recs = [A for A in recs if faces(A)] or recs
        for A in recs:
            v = q - J["p"]
            d = float(v @ A["u"])
            if d < -3.0:
                continue
            dist = _dist_line(q, A["keep"])
            if pick is None or dist < pick[0]:
                pick = (dist, A, d, float(v @ A["right"]))
        if pick is None:
            return None
        return J, pick[1], pick[2], pick[3]

    # assigned objects per (junction id, arm key)
    obj = {}

    def slot(J, A):
        return obj.setdefault((J["id"], A["fid"], A["start"]), {"stop": [], "cw": [], "teeth": [], "mast": [],
                                                                "signs": []})

    stop_xy = {}                  # junction id -> stop line midpoints (in its paint reach)
    for a, b, cw in cws:
        got = assign((a + b) / 2)
        if got is None:
            continue
        J, A, d, _ = got
        if cw.get("stop"):
            lat = sorted(float((x - J["p"]) @ A["right"]) for x in (a, b))
            w = _f(cw.get("w", 0.0))
            slot(J, A)["stop"].append({"d": d, "lat0": _f(lat[0]), "lat1": _f(lat[1]), "w": w,
                                       "src": "z294" if w == _z294() else "us"})
            stop_xy.setdefault(J["id"], []).append(_xy((a + b) / 2, run.origin))
        else:
            look = "zebra" if cw.get("look") or cw.get("zebra") else "us"
            slot(J, A)["cw"].append({"d": d, "w": _f(cw.get("w", 0.0)), "look": look})
    for c in teeth:
        got = assign(c)
        if got is not None:
            J, A, d, _ = got
            slot(J, A)["teeth"].append(d)
    for row in props:
        got = assign(np.array([float(row[1]), float(row[2])]),
                     yaw=float(row[4]) if str(row[0]) in MAST_GROUPS and row[4] is not None else None)
        if got is None:
            continue
        J, A, d, lat = got
        group = str(row[0])
        if group in MAST_GROUPS:
            slot(J, A)["mast"].append((d, lat, group, _f(row[4])))
        elif group.startswith("de_") or group in SIGN_GROUPS:
            slot(J, A)["signs"].append((group, d, lat))

    rows = []
    for J in juncs:
        rows.append(_junction(run, J, de, ring, link_ok, priority, obj, appr, skipped, signals_by_j, gnodes,
                              stop_xy, juncs, pieces))
    groups = {J["id"]: {ap["group"] for ap in row["approaches"] if ap["group"]} for J, row in zip(juncs, rows)}
    for J, row in zip(juncs, rows):
        if J["model"] is None:
            row["cluster"] = [o["id"] for o in juncs if o is not J and math.hypot(*(o["p"] - J["p"])) <= CLUSTER_M]
        else:                     # the other junctions of its signal group
            row["cluster"] = [o["id"] for o in juncs if o is not J and groups[o["id"]] & groups[J["id"]]]
    return rows


_Z294 = None


def _z294():
    global _Z294
    if _Z294 is None:
        import markings
        _Z294 = markings.Z294_W
    return _Z294


def _junction(run, J, de, ring, link_ok, priority, obj, appr, skipped, signals_by_j, gnodes, stop_xy, juncs, pieces):
    import furniture
    import designs
    origin, frame = run.origin, run.frame
    p, arms, recs = J["p"], J["arms"], J["recs"]
    gnode = _graph_node(p, gnodes)
    # the controls: the model's approaches where the junction model decided (its own arms), else the kernel's where it
    # answered, else junction_controls
    model = J["model"]
    ctl = de.get(J["k"]) if model is None else None
    if model is not None:
        decided, found = "model", {}
    elif ctl is not None:
        decided, found = "de_kernel", {(a[0], bool(a[4])): (_reg(w), pl) for a, w, pl in ctl}
    else:
        found_list = furniture.junction_controls(arms, J["k"], link_ok, ring=ring, signals=run.signals,
                                                 country="DE", priority=priority)
        found = {(a[0], bool(a[4])): (REG_STR.get(str(w), str(w)), None) for a, w in found_list}
        decided = "junction_controls" if found_list else "none"
    sig_entries = signals_by_j.get(gnode, []) if gnode else []
    approaches = []
    for A in recs:
        a = A["a"]
        key = (A["fid"], A["start"])
        ap = model[1].get(A["arm"].id) if model is not None else None
        if model is not None:
            reg, plate = ("none", None) if ap is None else (ap.regulation, ap.plate)
            if ap is not None and ap.link and reg == "signal":
                reg = "signal_link"                 # (a link between two signalled junctions: no head of its own)
        else:
            reg, plate = found.get(key, ("none", None))
        o = obj.get((J["id"], A["fid"], A["start"]), {"stop": [], "cw": [], "teeth": [], "mast": [], "signs": []})
        name = None
        feat = run.feats_of.get(A["fid"])
        if feat is not None and "name" in feat[0].attrs:
            name = str(feat[0].attrs["name"][feat[1]]) or None
        mapped = furniture._mapped_priority(a, priority)
        if model is None:
            d_edge = _d_edge(A, recs)
        else:
            d_edge = None if A["arm"].d_edge is None else _f(A["arm"].d_edge)
        stop_line = min(o["stop"], key=lambda x: x["d"]) if o["stop"] else None
        crosswalk = min(o["cw"], key=lambda x: x["d"]) if o["cw"] else None
        wait_line = {"d": _f(np.mean(o["teeth"])), "n": len(o["teeth"])} if o["teeth"] else None
        mast = None
        if o["mast"]:
            d, lat, group, yaw = min(o["mast"], key=lambda x: x[0])
            mast = {"d": _f(d), "lat": _f(lat), "group": group, "yaw": yaw, "count": len(o["mast"])}
        signs = {}
        for group, d, lat in o["signs"]:
            if group not in signs or d < signs[group]["d"]:
                signs[group] = {"group": group, "d": _f(d), "lat": _f(lat)}
        ai, ai_skipped = None, None
        end_key = 0 if A["start"] else -1
        nb_inc = appr.get((gnode, A["fid"], end_key)) if gnode else None
        if model is not None and ap is not None and ap.group is not None and reg == "signal" and ap.inbound:
            # (the model's signal group has one controller for all its junctions, at its first one: navgraph's
            # _signals_model; its heads stand on this arm when their station and heading are this approach's)
            for s in run.graph.get("signals") or []:
                for app in s["app"]:
                    q = np.array([app[1], app[2]], np.float64) - p
                    d, lat = float(q @ A["u"]), float(q @ A["right"])
                    heading_in = float(np.array([app[4], app[5]]) @ A["u"]) < -0.7
                    if ai is None and heading_in and -0.5 <= d <= 40.0 and abs(lat) <= float(a[1].width) / 2 + 1.0:
                        ai = {"d": _f(d), "lat": _f(lat), "kind": s["kind"], "phase": app[6], "from": app[0]}
        if nb_inc is not None and ai is None:
            for s in sig_entries:
                for app in s["app"]:
                    if app[0] == nb_inc[0] and ai is None:
                        q = np.array([app[1], app[2]], np.float64) - p
                        ai = {"d": _f(q @ A["u"]), "lat": _f(q @ A["right"]), "kind": s["kind"],
                              "phase": app[6], "from": app[0]}
            for sk in skipped:
                if gnode and sk["j"] == gnode and str(sk["fid"]) == A["fid"] and sk["end"] == end_key and ai is None:
                    ai_skipped = (f"short {_f(sk['dist'])} m" if sk["why"] == "short" else sk["why"])
        ds = [x for x in ([stop_line["d"]] if stop_line else []) + ([mast["d"]] if mast else []) +
              ([ai["d"]] if ai else [])]
        spread = _f(max(ds) - min(ds)) if len(ds) >= 2 else None
        inbound = not furniture._outbound(a)
        bearing = math.degrees(math.atan2(A["u"][0], A["u"][1])) % 360
        approaches.append({
            "fid": A["fid"], "end": "start" if A["start"] else "end", "name": name, "cls": str(a[1].cls),
            "width": _f(a[1].width), "oneway": bool(a[1].oneway), "inbound": bool(inbound), "bearing": _f(bearing, 1),
            "regulation": reg, "plate": plate, "mapped_sign": MAPPED_SIGN.get(mapped) if mapped else None,
            "d_edge": d_edge, "stop_line": stop_line, "crosswalk": crosswalk, "wait_line": wait_line,
            "mast": mast, "signs": sorted(signs.values(), key=lambda x: x["d"]), "ai": ai,
            "ai_skipped": ai_skipped, "spread_m": spread, "lanes": [], **_approach_info(ap)})
    jkey = str(J["k"]) if model is not None else [int(v) for v in J["k"]]
    mapped = {"signals": [], "priority": [], "stop_lines": stop_xy.get(J["id"], [])}
    for s in (run.signals if run.signals is not None else []):
        if np.hypot(*(np.asarray(s[:2], np.float64) - p)) <= designs.JUNCTION_PAINT_M:
            mapped["signals"].append(_xy(s, origin))
    if run.nodes and len(run.nodes.get("priority", ())):
        for row in run.nodes["priority"]:
            if np.hypot(row[0] - p[0], row[1] - p[1]) <= designs.JUNCTION_PAINT_M:
                mapped["priority"].append([_f(row[0] - origin[0]), _f(row[1] - origin[1]), bool(row[2])])
    lat, lon, _ = frame.enu_to_geodetic(float(p[0]), float(p[1]), 0.0)
    return {"id": J["id"], "x": _xy(p, origin)[0], "y": _xy(p, origin)[1], "lat": _f(lat, 7), "lon": _f(lon, 7),
            "key": jkey, "n_arms": len(arms), "decided_by": decided, "graph_node": gnode, "cluster": [],
            "mapped": mapped, "approaches": approaches}


def _d_edge(A, recs):
    """The first distance s (0.25 m steps, at most ARM_M) where both road edges lie outside every other arm's road;
    None if there is none."""
    line = A["line"]
    seg = np.hypot(*np.diff(line, axis=0).T) if len(line) > 1 else np.zeros(0)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    S = np.arange(0.0, ARM_M + ARM_STEP_M / 2, ARM_STEP_M)
    if len(line) < 2:
        return None
    X, Y = np.interp(S, cum, line[:, 0]), np.interp(S, cum, line[:, 1])
    T = np.column_stack([np.gradient(X), np.gradient(Y)])
    T /= np.maximum(np.hypot(T[:, 0], T[:, 1]), 1e-9)[:, None]
    N = np.column_stack([-T[:, 1], T[:, 0]])
    others = [B for B in recs if B is not A and len(B["line"]) >= 2]
    half = A["width"] / 2
    for i, s in enumerate(S):
        ok = True
        for sign in (1.0, -1.0):
            q = np.array([X[i], Y[i]]) + sign * half * N[i]
            for B in others:
                if _dist_line(q, B["line"]) <= B["width"] / 2:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return _f(s)
    return None


def graph_table(run, junctions):
    """[GraphNode] of run's graph nodes within run.radius with at least GRAPH_MIN_NB distinct neighbours."""
    G = run.graph
    nodes = G.get("nodes", {})
    origin = run.origin
    appr = run.graph_diag.get("approaches", {})
    skipped = run.graph_diag.get("skipped", [])
    node_j = {j["graph_node"]: j["id"] for j in junctions if j.get("graph_node")}
    ai_of = {(j["id"], ap["ai"]["from"]): ap["ai"]["d"] for j in junctions for ap in j["approaches"] if ap["ai"]}
    sigs = {}
    for s in G.get("signals") or []:
        sigs.setdefault(s["j"], []).append(s)
    edges = G.get("edges", [])
    nbs = {}
    for e in edges:
        nbs.setdefault(e[0], set()).add(e[1])
        nbs.setdefault(e[1], set()).add(e[0])

    def fid_of(n, other):
        for (j, fid, end), (nb, _inc) in appr.items():
            if j == n and nb == other:
                return str(fid), end
        return None, None

    def skip_of(n, other):
        for sk in skipped:
            if sk["j"] == n and sk["nb"] == other:
                return f"short {_f(sk['dist'])} m" if sk["why"] == "short" else sk["why"]
        return None

    def sig_of(n, other):
        for s in sigs.get(n, []):
            for app in s["app"]:
                if app[0] == other:
                    return app
        return None

    out = []
    for n, v in nodes.items():
        if len(nbs.get(n, ())) < GRAPH_MIN_NB:
            continue
        if math.hypot(v[0] - origin[0], v[1] - origin[1]) > run.radius:
            continue
        ins, outs = [], []                  # (other node, lanes, edge)
        for e in edges:
            a, b, oneway = e[0], e[1], bool(e[2])
            if b == n and a != n:
                ins.append((a, e[3], e))
            if a == n and b != n and not oneway:
                ins.append((b, e[4], e))
            if a == n and b != n:
                outs.append((b, e[3], e))
            if b == n and a != n and not oneway:
                outs.append((a, e[4], e))
        ins_rows, outs_rows = [], []
        for other, lanes, e in ins:
            app = sig_of(n, other)
            fid, _end = fid_of(n, other)
            ins_rows.append({"from": str(other), "fid": fid,
                             "dist": _f(math.hypot(nodes[other][0] - v[0], nodes[other][1] - v[1])),
                             "lanes": lanes, "kmh": _f(e[5] * 3.6, 1), "drivability": _f(e[6], 3),
                             "oneway": bool(e[2]), "controlled": app is not None,
                             "phase": app[6] if app is not None else None,
                             "stop_d": ai_of.get((node_j.get(n), other)) if app is not None else None,
                             "skipped": skip_of(n, other)})
        for other, lanes, e in outs:
            fid, _end = fid_of(n, other)
            outs_rows.append({"to": str(other), "fid": fid, "lanes": lanes, "kmh": _f(e[5] * 3.6, 1)})
        ctl = sigs.get(n, [])
        pos = _xy(v, origin)
        out.append({"node": str(n), "x": pos[0], "y": pos[1], "junction": node_j.get(n),
                    "controller": ctl[0]["kind"] if ctl else None, "in": ins_rows, "out": outs_rows})
    return out
