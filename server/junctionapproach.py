"""Junction approaches (Phase 2 section 3, Brief B9): one Approach per arm of a model junction, with its regulation, stop
station and stop line, signal groups with their phases, and the side cuts of the arms' lines (3.4, first point).

Pure: no paint, props or graph. build() fills decided.apps ({arm id: Approach}) and decided.groups ({group key: [junction
keys]}) from decided.model and the controls of decided ({junction key: (arm tuples, ctl)}, as designs.decide_model
gives them). A junction with ctl None (outside Germany) gets no approaches.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import dnp

import markings

GROUP_LINK_M = 40.0     # m: a road this short between two signalised junctions is one controller (R3-08)
CROSSING_W = 3.0        # m: the width given to a mapped crossing on an arm (its stop station is pushed behind it)
PHASE_DOT = 0.7         # |u . axis| above this: the same phase as the axis

REG_OF_INT = {0: "none", 1: "z205", 2: "z206", 3: "z306", 4: "z301", 5: "rvl", 6: "exit"}
REG_OF_STR = {"signal": "signal", "signal_left": "signal", "giveway": "yield", "stop": "stop"}
STOP_LINE_KIND = {"signal": "z294", "z206": "z294", "stop": "z294", "z205": "z341", "yield": "z341"}
STATION_REG = {"signal", "z206", "stop", "z205", "yield", "rvl"}   # an inbound arm with these has a stop station
AI_REG = {"signal", "z206", "stop", "rvl"}                          # ... and an AI stop there
HALT_REG = {"signal", "z205", "z206", "yield", "stop"}              # regulations that take a stop line (role rule)


def regulation_of(what):
    """The regulation of a controls answer: an int (designs.controls' codes) or a string (its group strings)."""
    if what is None:
        return "none"
    if isinstance(what, (int, np.integer)) and not isinstance(what, bool):
        return REG_OF_INT.get(int(what), "none")
    return REG_OF_STR.get(str(what), "none")


class Decided(dict):
    """{junction key: (arm tuples, ctl)} plus .model (junctionmodel.Model), .apps and .groups."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.apps = {}
        self.groups = {}
        self.drawn = None       # {secrossings.drawn_key: drawn width (m) at each point of a tapered row} (roads.build_tile)


@dataclass
class Approach:
    id: tuple               # the arm's id (junction key, index)
    junction: str
    arm: object             # junctionmodel.Arm
    regulation: str
    what: object            # the controls answer as given
    plate: int
    inbound: bool
    link: bool
    group: str | None
    phase: int | None
    role: str               # "through" | "end"
    d_edge: float | None
    d_stop: float | None
    stop_line: dict | None  # {"d", "a" (3,), "b" (3,), "w", "kind": "z294" | "z341"}
    crossings: tuple        # m along the arm's line of the mapped crossings on it
    mast_d: float | None
    ai: dict | None         # {"d", "lat"}
    lanes_in: int
    why: dict = field(default_factory=dict)

    def as_dict(self):
        def num(v):
            return None if v is None else round(float(v), 2)
        sl = self.stop_line
        return {
            "id": [self.id[0], int(self.id[1])], "junction": self.junction,
            "arm": {"fid": str(self.arm.fid), "end": bool(self.arm.at_start), "node_key": str(self.arm.node_key)},
            "regulation": self.regulation, "plate": int(self.plate), "inbound": self.inbound, "link": self.link,
            "group": self.group, "phase": self.phase, "role": self.role, "d_edge": num(self.d_edge),
            "d_stop": num(self.d_stop),
            "stop_line": None if sl is None else {"d": num(sl["d"]), "a": [num(c) for c in sl["a"]],
                                                  "b": [num(c) for c in sl["b"]], "w": num(sl["w"]),
                                                  "kind": sl["kind"]},
            "crossings": [num(s) for s in self.crossings], "mast_d": num(self.mast_d),
            "ai": None if self.ai is None else {"d": num(self.ai["d"]), "lat": num(self.ai["lat"])},
            "lanes_in": int(self.lanes_in), "why": dict(self.why),
        }


def _stations(line):
    """m along the centreline line (k, 3) from its first point to each of its points."""
    return np.r_[0.0, np.cumsum(dnp.hypot(*np.diff(line[:, :2], axis=0).T))]


def _xy(v):
    return np.zeros((0, 2)) if v is None else np.asarray(v, np.float64).reshape(-1, 2)


def _crossings(arm, spec, nodes, signal):
    """([{"a", "b", "w"}], [m along]) of the mapped crossings on arm: Germany's paint crossings (de_crossings with de_paint
    != 0) and at a signal also the crossings at signals (de_signal_crossings, B3). A crossing counts when its point is
    within half the arm's road plus 1 m of the arm's line."""
    pts = [(float(x), float(y)) for x, y, paint in (nodes.get("de_crossings") or []) if paint]
    if signal:
        pts += [(float(x), float(y)) for x, y in _xy(nodes.get("de_signal_crossings"))]
    cw, st = [], []
    if not pts:
        return cw, st
    s = _stations(arm.line)
    P, Q = arm.line[:-1, :2], arm.line[1:, :2]
    v = Q - P
    vv = np.maximum((v * v).sum(1), 1e-12)
    for x, y in pts:
        t = np.clip(((np.array([x, y]) - P) * v).sum(1) / vv, 0.0, 1.0)      # (point to each segment of the line)
        dist = dnp.hypot(*(P + t[:, None] * v - [x, y]).T)
        i = int(np.argmin(dist))
        if dist[i] <= spec.width / 2 + 1.0:
            p = np.array([x, y])
            cw.append({"a": p, "b": p, "w": CROSSING_W})
            st.append(float(s[i] + t[i] * (s[i + 1] - s[i])))
    return cw, st


def _role(arms, i, reg):
    """through: the arm has a straight partner, neither has a stop line regulation, and every other arm of the node is a
    stub (three arms, or a lesser road class than the pair's; roads.LINE_RANK); else end."""
    import roads
    a = arms[i]
    if a.through is None:
        return "end"
    p = arms[a.through]
    if reg[a.id][0] in HALT_REG or reg[p.id][0] in HALT_REG:
        return "end"
    if len(arms) == 3:
        return "through"
    rank = lambda x: roads.LINE_RANK.get(x.spec.cls, 0)
    lo = min(rank(a), rank(p))
    return "through" if all(rank(o) < lo for j, o in enumerate(arms) if j not in (i, a.through)) else "end"


def _drawn_width(model, arm, drawn, d):
    """The drawn width (m) of arm's road d m from its junction where it tapers (roads.roundabout_flares: a roundabout entry
    is flared at the ring), else None. drawn: Decided.drawn."""
    if not drawn:
        return None
    import secrossings
    fid, _sp, pts, _lon = model.rows[arm.row][:4]
    w = drawn.get(secrossings.drawn_key(fid, pts))
    if w is None:
        return None
    pts = np.asarray(pts, np.float64)
    idx = np.arange(arm.vi, len(pts) if arm.step > 0 else -1, arm.step)
    s = np.r_[0.0, np.cumsum(dnp.hypot(*np.diff(pts[idx, :2], axis=0).T))]
    return float(np.interp(d, s, np.asarray(w, np.float64)[idx]))


def _approach(k, arms, i, reg, nodes, left_hand, group, link, model=None, drawn=None):
    """The Approach of arms[i] of junction k (arms: its model arms in order, reg: {arm id: (regulation, what, plate)})."""
    import furniture
    import roads
    arm = arms[i]
    spec = arm.spec
    regn, what, plate = reg[arm.id]
    inbound = bool(arm.inbound)
    others = [o.spec.width / 2 for j, o in enumerate(arms) if j != i]
    base = arm.d_edge if arm.d_edge is not None else (max(others) if others else 0.0)
    why, d_stop, stop_line, mast_d, ai, cw_st = {}, None, None, None, None, ([], [])
    if link:
        why["stop_line"] = "link"
    elif not inbound:
        why["stop_line"] = "outbound"
    elif regn == "exit":
        why["stop_line"] = "exit"
    elif regn in STATION_REG:
        kind = STOP_LINE_KIND.get(regn)
        signal = regn == "signal"
        length = float(_stations(arm.line)[-1])
        if nodes is not None:
            cw_st = _crossings(arm, spec, nodes, signal)
        mapped = nodes.get("stop_lines") if nodes is not None and kind == "z294" else None
        d_stop = markings.stop_station(arm.line, spec, base, mapped, cw_st[0],
                                       furniture.G_SIGNAL if signal else furniture.G_STOP)
        if kind is None:
            why["stop_line"] = "no_regulation"      # rvl: the AI stops there, nothing is painted
        elif d_stop is None:
            why["stop_line"] = "short"
        else:
            dw = None if spec.oneway or model is None else _drawn_width(model, arm, drawn, d_stop)
            a, b = markings.line_span(arm.line, d_stop, spec if dw is None else spec._replace(width=max(spec.width, dw)),
                                      left_hand)
            w = markings.Z294_W if kind == "z294" else markings.Z341_W
            stop_line = {"d": d_stop, "a": a, "b": b, "w": w, "kind": kind}
        if signal:
            mast_d = d_stop
        if regn in AI_REG:
            ai_d = d_stop
            if ai_d is None and length - 1.0 > 0:
                # the line is short: the AI stops at most one metre before the line's end (markings.stop_station gives no
                # station then, so base + 1,0 stands in for its 'at')
                ai_d = min(base + 1.0, length - 1.0)
            if ai_d is not None:
                ai = {"d": ai_d, "lat": spec.width / 4}
    else:
        why["stop_line"] = "no_regulation"
    lanes = 0
    if inbound:
        n = roads.lane_layout(spec)[1]
        lanes = n if spec.oneway else max(1, n // 2)
    return Approach(id=arm.id, junction=k, arm=arm, regulation=regn, what=what, plate=plate, inbound=inbound,
                    link=link, group=group, phase=None, role=_role(arms, i, reg), d_edge=arm.d_edge, d_stop=d_stop,
                    stop_line=stop_line, crossings=tuple(round(s, 2) for s in cw_st[1]), mast_d=mast_d, ai=ai,
                    lanes_in=lanes, why=why)


def build(decided, nodes=None, left_hand=False):
    """Fills decided.apps and decided.groups: the signal groups (links of at most GROUP_LINK_M between signalised
    junctions), the phases of the outer signal approaches, and each arm's Approach (3.2)."""
    model = decided.model
    decided.apps, decided.groups = {}, {}
    keys = sorted(k for k, v in decided.items() if v[1] is not None)
    reg = {}                                    # arm id -> (regulation, what, plate)
    for k in keys:
        by_id = {e[0][7]: e for e in decided[k][1]}
        for tup in decided[k][0]:
            e = by_id.get(tup[7])
            what = e[1] if e is not None else None
            reg[tup[7]] = (regulation_of(what), what, e[2] if e is not None else -1)
    signalled = {k for k in keys if any(reg[t[7]][0] == "signal" for t in decided[k][0])}
    parent = {k: k for k in signalled}

    def root(k):
        while parent[k] != k:
            k = parent[k]
        return k
    link_ids = set()
    for k in sorted(signalled):
        for arm in model.junction(k).arms:
            if arm.far in signalled and arm.length <= GROUP_LINK_M:
                link_ids.add(arm.id)
                a, b = root(k), root(arm.far)
                if a != b:
                    parent[max(a, b)] = min(a, b)       # the group's key: its smallest junction key
    groups = {}
    for k in sorted(signalled):
        groups.setdefault(root(k), []).append(k)
    decided.groups = groups
    for k in keys:
        arms = [model.arm(t[7]) for t in decided[k][0]]
        group = root(k) if k in signalled else None
        for i, arm in enumerate(arms):
            decided.apps[arm.id] = _approach(k, arms, i, reg, nodes, left_hand, group, arm.id in link_ids, model,
                                             decided.drawn)
    axis = {}                                   # group key -> unit vector of its first outer signal approach
    for ap in decided.apps.values():
        if ap.group is None or not ap.inbound or ap.link or ap.regulation != "signal":
            continue
        u = np.asarray(ap.arm.u, np.float64)[:2]
        axis.setdefault(ap.group, u)
        ap.phase = 0 if abs(float(dnp.dot(u, axis[ap.group]))) > PHASE_DOT else 1


def side_cuts(decided):
    """{fid: [(X, u, t_in, t_out)]} for the arms with role end: X the node's point, u the arm's unit vector, t_out its
    d_edge, t_in the stop line's station plus half its width (else t_out), for every row of the arm's chain."""
    out = {}
    model = decided.model
    for ap in decided.apps.values():
        if ap.role != "end" or ap.d_edge is None:
            continue
        t_out = float(ap.d_edge)
        t_in = ap.stop_line["d"] + ap.stop_line["w"] / 2 if ap.stop_line is not None else t_out
        for r, _rv in ap.arm.chain:
            out.setdefault(model.rows[r][0], []).append((ap.arm.p[:2], ap.arm.u, t_in, t_out))
    return out
