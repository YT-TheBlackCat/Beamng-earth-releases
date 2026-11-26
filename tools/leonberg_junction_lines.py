"""Junction probe, lines (Phase 0, Brief C4): why each painted line ends, and the road pieces an arm of a junction runs
over with their lane tags and paint counts.

line_ends(run, junctions) -> (ends, stubs): the line stages of roads.build_tile's diag, grouped per line (fid, kind,
side, offset), each painted end with its reason. arm_pieces(run, arm) -> [Piece] of one junction arm.
run.roads and run.origin are in the tile frame (m); the x, y returned are m from the spot (run.origin).
Pure helpers (arc_of, merge) need no server module; roads is imported only by arm_pieces.
"""
import numpy as np

TOL_M = 0.6              # m: stage ends this close are one end (a line's pieces across a tile seam)
DASH_GAP_MAX_M = 15.0    # m: the largest dash gap a dashed line's tolerance takes
END_M = 1.0              # m: road ends this close meet at one point
JUNCTION_NEAR_M = 50.0   # m: a line end this close to a junction node belongs to it
OBJECT_NEAR_M = 3.0      # m: a junction cut this close to a crossing, stop line, teeth or zebra is cut by it
LINE_KINDS = ("centre", "lane", "edge")
STAGES = ("plan", "tile", "junction", "corner", "cut", "drawn")
CUT_ORDER = (("tile", "tile_edge"), ("junction", "junction_cut"), ("corner", "corner"), ("cut", "cut_by_road"))
EMPTY_ORDER = ("tile", "junction", "corner", "cut", "drawn")


def _xy(a):
    return np.atleast_2d(np.asarray(a, np.float64))[:, :2]


def _project(line, pts):
    """(s, lat, dist) per point of pts: arc length of its foot on the polyline line, the signed distance from the foot
    to the line (positive left of the direction of travel) and the distance. Pure."""
    line, P = _xy(line), _xy(pts)
    k = len(P)
    if len(line) < 2 or k == 0:
        return np.zeros(k), np.zeros(k), np.full(k, np.inf)
    a = line[:-1]
    d = line[1:] - a
    L2 = (d * d).sum(1)
    L = np.sqrt(L2)
    ok = L > 1e-9
    cum = np.concatenate(([0.0], np.cumsum(L)))
    s, lat, dist = np.empty(k), np.empty(k), np.empty(k)
    for i0 in range(0, k, 256):             # (chunks: a long line against a long line is k x m)
        Q = P[i0:i0 + 256]
        t = np.clip(((Q[:, None] - a[None]) * d[None]).sum(2) / np.where(ok, L2, 1.0), 0.0, 1.0)
        off = Q[:, None] - (a[None] + t[..., None] * d[None])
        d2 = np.where(ok, (off * off).sum(2), np.inf)
        j = np.argmin(d2, axis=1)
        r = np.arange(len(Q))
        lj, okj = L[j], ok[j]
        nrm = np.stack([-d[j, 1], d[j, 0]], 1) / np.where(okj, lj, 1.0)[:, None]
        sl = slice(i0, i0 + len(Q))
        lat[sl] = np.where(okj, (off[r, j] * nrm).sum(1), 0.0)
        s[sl] = cum[j] + t[r, j] * lj
        dist[sl] = np.sqrt(d2[r, j])
    return s, lat, dist


def arc_of(line, pts):
    """Arc length along the polyline line of each point's foot, (k,). Pure."""
    return _project(line, pts)[0]


def merge(intervals, tol):
    """Intervals (s0, s1) sorted and joined where the gap between two is at most tol. Pure."""
    out = []
    for a, b in sorted((min(x, y), max(x, y)) for x, y in intervals):
        if out and a <= out[-1][1] + tol:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _at(line, s):
    """(point, unit direction) at arc length s of the polyline line, clamped to its ends. Pure."""
    line = _xy(line)
    a = line[:-1]
    d = line[1:] - a
    L = np.hypot(d[:, 0], d[:, 1])
    cum = np.concatenate(([0.0], np.cumsum(L)))
    j = min(max(int(np.searchsorted(cum, s, side="right")) - 1, 0), len(L) - 1)
    if L[j] <= 1e-9:
        return line[j], np.array([1.0, 0.0])
    t = min(max((s - cum[j]) / L[j], 0.0), 1.0)
    return a[j] + t * d[j], d[j] / L[j]


def _side_point(c, s, off):
    p, tan = _at(c, s)
    return p + off * np.array([-tan[1], tan[0]])


def _cached(run, name, fn):
    """fn(run), kept on the run object (one run is read by many calls)."""
    if getattr(run, name, None) is None:
        setattr(run, name, fn(run))
    return getattr(run, name)


def _index(run):
    pieces, specs, end_fid, end_xy = {}, {}, [], []
    for r in run.roads:
        fid, spec, pts = r[0], r[1], _xy(r[2])
        if len(pts) < 2:
            continue
        pieces.setdefault(fid, []).append(pts)
        specs.setdefault(fid, spec)
        for q in (pts[0], pts[-1]):
            end_fid.append(fid)
            end_xy.append(q)
    return {"pieces": pieces, "specs": specs, "end_fid": end_fid, "end_xy": np.array(end_xy).reshape(-1, 2)}


def _ends_at(index, q, exclude=None):
    """The fids (but exclude) with a road end within END_M of q."""
    near = np.hypot(index["end_xy"][:, 0] - q[0], index["end_xy"][:, 1] - q[1]) <= END_M
    return {f for f, m in zip(index["end_fid"], near) if m and f != exclude}


def _centre_of(o):
    if isinstance(o, dict):                 # a crossing, stop line or zebra: {"a", "b", ...}
        return 0.5 * (np.asarray(o["a"], float)[:2] + np.asarray(o["b"], float)[:2])
    if isinstance(o, tuple) and len(o) == 2 and np.ndim(o[0]) == 2:     # (segment, half width)
        o = o[0]
    return _xy(o).mean(0)                   # a teeth polygon: the mean of its corners


def _objects(run):
    out = []
    for diag in run.diags:
        for o in diag.get("crosswalks", ()):
            out.append((_centre_of(o), "stop_line" if o.get("stop") else "crosswalk"))
        for o in diag.get("zebras", ()):
            out.append((_centre_of(o), "zebra"))
        for o in diag.get("teeth", ()):
            out.append((_centre_of(o), "wait_line"))
    return out


def _object_near(objects, p):
    best = None
    for c, kind in objects:
        d = float(np.hypot(c[0] - p[0], c[1] - p[1]))
        if d <= OBJECT_NEAR_M and (best is None or d < best[0]):
            best = (d, kind)
    return best[1] if best else None


def _culprit(index, fid, p):
    """The other road whose centre line lies nearest the end, less its half width: (fid, its class)."""
    best = None
    for f2, pcs in index["pieces"].items():
        if f2 == fid:
            continue
        d = min(float(_project(pc, p)[2][0]) for pc in pcs) - float(index["specs"][f2].width) / 2
        if best is None or d < best[0]:
            best = (d, f2, index["specs"][f2].cls)
    return (best[1], best[2]) if best else (None, None)


def _junction_near(junctions, x, y):
    best = None
    for j in junctions:
        d = float(np.hypot(j["x"] - x, j["y"] - y))
        if d <= JUNCTION_NEAR_M and (best is None or d < best[1]):
            best = (j["id"], d)
    return best if best else (None, None)


def _spans(c, pieces):
    out = []
    for p in pieces:
        if len(p):
            s = arc_of(c, p)
            out.append((float(s.min()), float(s.max())))
    return out


def _view(g):
    """A line group with its stages as merged intervals of arc length on its centre line. Pure."""
    c = g["centre"]
    iv = {n: _spans(c, g["st"][n]) for n in STAGES}
    plan = sorted(iv["plan"])
    gap = max([max(0.0, b[0] - a[1]) for a, b in zip(plan, plan[1:])], default=0.0)
    tol = TOL_M + (min(gap, DASH_GAP_MAX_M) if g["dash"] else 0.0)     # (a dash: its largest plan gap)
    return {**{k: v for k, v in g.items() if k != "st"}, "tol": tol, "M": {n: merge(v, tol) for n, v in iv.items()}}


def _groups(run):
    """The line groups of run.diags: records by (fid, kind, side, offset), offset = median signed distance of the
    plan to the road's centre line piece nearest to it."""
    index = _cached(run, "_kreuz_index", _index)
    groups = {}
    for diag in run.diags:
        for rec in diag.get("lines", ()):
            cands = index["pieces"].get(rec["fid"])
            plan = rec["stages"].get("plan") or []
            if not cands or not plan or not len(plan[0]):
                continue
            pp = _xy(plan[0])
            fits = [(c, _project(c, pp)) for c in cands]
            c, (_, lat, _) = min(fits, key=lambda f: float(np.median(f[1][2])))
            off = round(float(np.median(lat)), 1)
            key = (rec["fid"], rec.get("kind"), rec.get("side"), off)
            g = groups.get(key)
            if g is None:
                g = groups[key] = {"fid": rec["fid"], "kind": rec.get("kind"), "side": rec.get("side"), "off": off,
                                   "centre": c, "dash": False, "st": {n: [] for n in STAGES}}
            g["dash"] = g["dash"] or bool(rec.get("dash"))
            for n in STAGES:
                g["st"][n].extend(_xy(q) for q in rec["stages"].get(n, ()) if len(q))
    return [_view(g) for g in groups.values()]


def _rel(p, origin):
    return round(float(p[0] - origin[0]), 2), round(float(p[1] - origin[1]), 2)


def _unique(items):
    """One item per (fid, kind, side, point to 0.5 m)."""
    seen, out = set(), []
    for it in items:
        key = (it["fid"], it["kind"], it.get("side"), round(it["x"] / 0.5), round(it["y"] / 0.5))
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out


def line_ends(run, junctions):
    """(ends, stubs). Each painted end inside run.radius: {fid, kind, side, off, dash, x, y, reason, detail, culprit,
    junction, d_junction}, reason road_end / tile_edge / junction_cut / corner / cut_by_road / fully_cut / unknown
    (skipped roads: reason internal / no_lines, kind None). A stub: {fid, kind, x, y, len_m}: cut run that no drawn run
    covers."""
    origin = np.asarray(run.origin, np.float64)[:2]
    radius = float(run.radius)
    index = _cached(run, "_kreuz_index", _index)
    objects = _cached(run, "_kreuz_objects", _objects)
    views = _cached(run, "_kreuz_groups", _groups)

    def inside(p):
        return float(np.hypot(p[0] - origin[0], p[1] - origin[1])) <= radius

    def place(g, p, reason, detail=None, culprit=None):
        x, y = _rel(p, origin)
        j, dj = _junction_near(junctions, x, y)
        return {"fid": g["fid"], "kind": g["kind"], "side": g["side"], "off": g["off"], "dash": g["dash"],
                "x": x, "y": y, "reason": reason, "detail": detail, "culprit": culprit, "junction": j,
                "d_junction": None if dj is None else round(dj, 2)}

    ends, stubs, lined = [], [], set()
    for g in views:
        lined.add(g["fid"])
        c, M, tol, off = g["centre"], g["M"], g["tol"], g["off"]
        plan_ends = [x for a, b in M["plan"] for x in (a, b)]
        for a, b in M["drawn"]:
            for e in (a, b):
                p = _side_point(c, e, off)
                if not inside(p):
                    continue
                if any(abs(e - x) <= tol for x in plan_ends):
                    n = len(_ends_at(index, _at(c, e)[0]))
                    ends.append(place(g, p, "road_end", "junction" if n >= 3 else "split" if n == 2 else "dead_end"))
                    continue
                reason = "unknown"
                for name, label in CUT_ORDER:
                    if any(abs(e - x) <= tol for pa, pb in M[name] for x in (pa, pb)):
                        reason = label
                        break
                if reason == "junction_cut":
                    ends.append(place(g, p, reason, _object_near(objects, p)))
                elif reason == "cut_by_road":
                    culprit, detail = _culprit(index, g["fid"], p)
                    ends.append(place(g, p, reason, detail, culprit))
                else:
                    ends.append(place(g, p, reason))
        if not M["drawn"]:
            for a, b in M["plan"]:
                p = _side_point(c, 0.5 * (a + b), off)
                if inside(p):
                    ends.append(place(g, p, "fully_cut", next(n for n in EMPTY_ORDER if not M[n])))
                    break
        for a, b in M["cut"]:
            if any(a < db and da < b for da, db in M["drawn"]):
                continue
            p = _side_point(c, 0.5 * (a + b), off)
            if inside(p):
                x, y = _rel(p, origin)
                stubs.append({"fid": g["fid"], "kind": g["kind"], "side": None, "x": x, "y": y,
                              "len_m": round(b - a, 2)})

    why = {}
    for diag in run.diags:
        for fid, w in diag.get("skipped", {}).items():
            why.setdefault(fid, w)
    for fid, w in why.items():
        if fid in lined:
            continue
        for pts in index["pieces"].get(fid, ()):
            hit = [p for p in pts if inside(p)]
            if hit:
                x, y = _rel(hit[0], origin)
                j, dj = _junction_near(junctions, x, y)
                ends.append({"fid": fid, "kind": None, "side": None, "off": None, "dash": False, "x": x, "y": y,
                             "reason": w, "detail": None, "culprit": None, "junction": j,
                             "d_junction": None if dj is None else round(dj, 2)})
                break

    ends = _unique(ends)
    ends.sort(key=lambda e: (str(e["fid"]), str(e["kind"]), str(e["side"]), e["x"], e["y"]))
    return ends, _unique(stubs)


def _drawn_counts(run):
    """{fid: {kind: groups with a drawn run}} of the line groups."""
    out = {}
    for g in _cached(run, "_kreuz_groups", _groups):
        if g["M"]["drawn"] and g["kind"] in LINE_KINDS:
            d = out.setdefault(g["fid"], {})
            d[g["kind"]] = d.get(g["kind"], 0) + 1
    return out


def _from(pieces, start):
    """The piece nearest start, oriented to begin there."""
    best = None
    for pts in pieces:
        for q, flip in ((pts[0], False), (pts[-1], True)):
            d = float(np.hypot(*(q - start)))
            if best is None or d < best[0]:
                best = (d, pts[::-1] if flip else pts)
    return best[1]


def _piece(run, fid, spec, length, counts):
    import roads
    drive, lanes = roads.lane_layout(spec)
    parts, osm = [], {}
    hit = (run.feats_of or {}).get(fid)
    if hit is not None:
        feats, i = hit
        for a, b, way, tags in roads.lane_tag_parts(feats, i):
            t = tags or {}
            parts.append([float(a), float(b), None if way is None else int(way), t.get("lanes"),
                          t.get("turn:lanes") or t.get("turn:lanes:forward") or t.get("turn:lanes:backward")])
        osm = dict(roads.lane_tags(feats, i) or {})
    return {"fid": fid, "len_m": round(length, 2), "cls": spec.cls, "width": float(spec.width),
            "oneway": bool(spec.oneway), "lanes_spec": None if spec.lanes is None else int(spec.lanes),
            "layout": [round(float(drive), 2), int(lanes)], "parts": parts, "osm": osm,
            "drawn": {k: counts.get(fid, {}).get(k, 0) for k in LINE_KINDS}}


def arm_pieces(run, arm, max_m=60.0):
    """The road pieces an arm of a junction runs over, from the junction outwards, up to max_m m. The walk goes on
    through a road end where exactly one other road meets (degree 2) and never loops. arm: the furniture._arms tuple
    (fid, spec, unit vector, point, at_start, length). Each piece: {fid, len_m, cls, width, oneway, lanes_spec,
    layout [drive_m, lanes], parts [[a, b, way, lanes, turn]], osm {tag: value}, drawn {centre, lane, edge}}."""
    index = _cached(run, "_kreuz_index", _index)
    counts = _drawn_counts(run)
    fid, _, _, p, _, _ = arm
    start = np.asarray(p, np.float64)[:2]
    out, seen, total = [], set(), 0.0
    while fid not in seen and fid in index["pieces"]:
        seen.add(fid)
        pts = _from(index["pieces"][fid], start)
        length = float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))
        out.append(_piece(run, fid, index["specs"][fid], length, counts))
        total += length
        far = pts[-1]
        if total >= max_m:
            break
        nxt = _ends_at(index, far, exclude=fid)
        if len(nxt) != 1:                   # (a junction of three or more, or a dead end)
            break
        fid, start = next(iter(nxt)), far
    return out
