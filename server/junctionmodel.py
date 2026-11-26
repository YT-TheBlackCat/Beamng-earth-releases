"""Junction model: connectors, nodes and split chains of the road rows (Phase 2, PHASE2_DESIGN.md 2.1-2.4, steps 1-4).

A row is (fid, spec, pts (k, 3), lon) as build_tile's roads. A connector ties a row end or inner point to a connector
ID of the map data. Nodes are the shared points of two or more rows; a node of degree 2 is a split (a plain join of two
row ends), the rows between the nodes of degree 3 and more form the chains. Arms and geometry come later.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

import dmath
import dnp

ON = os.environ.get("BNE_JUNCTION_MODEL", "on") != "off"   # switch (probe comparison: --set junctionmodel.ON=False)
NODE_TOL = 0.5           # m: row points this close are one node (fallback without a connector)
CONN_SNAP_M = 2.5        # m: connector point -> nearest vertex of its row (covers the placement offset)
MERGE_M = 3.0            # m: nodes with >= 3 arms this close are one junction node
OVERLAP_M = 0.5          # m: slack of the overlap rule (R2-02)
SPAN_MAX_M = 25.0        # m: largest distance between two part nodes of one junction node
BRANCH_M = 10.0          # m: arm direction (as junctions.BRANCH_M)
ARM_M = 60.0             # m: length of an arm line (R2-01/R3-09)
CHAIN_MAX_M = 250.0      # m: a chain is followed at most this far (arm length up to the next node)
STRAIGHT_DEG = 30.0      # degrees: two arms this close to opposite are a straight pair
HALO_M = 100.0           # m: rows this far beyond the tile reach the model needs (R1-22)
EDGE_STEP_M = 0.25       # m: step of the d_edge search
EDGE_REFINE = 2          # refinements of the last step, 32 stations each (0.25 m / 1024: the edge to 0.3 mm)


@dataclass
class Node:
    id: int
    key: str            # "c<smallest connector ID>", or "v<x>,<y>" without a connector
    xy: np.ndarray      # (2,) float64: the vertex of the row with the widest spec
    slots: tuple        # ((row, vi), ...), caller indices
    degree: int         # 1 per row end, 2 per inner vertex; a run of consecutive vertices counts once
    cids: tuple         # connector IDs in the node


@dataclass
class Chain:
    id: int
    rows: tuple         # ((row, reversed), ...) in run order, caller indices
    start: int | None   # node at the first row's start (junction nodes only), else None
    end: int | None     # node at the last row's end, else None
    length: float       # m: sum of the rows' lengths
    cyclic: bool        # a closed ring without any end


def _length(xy):
    if len(xy) < 2:
        return 0.0
    return float(np.sum(dnp.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))))


def row_connectors(rows, feats_of, frame):
    """{row: [(vi, cid)]}: the connector at each row end or inner vertex (2.4 step 1).

    feats_of = {fid: (Features, i)}; frame = geodesy.EnuFrame. Each connector point sits at start * L along the full
    line of its feature; per (row, cid) the nearest vertex, per (row, vi) the nearest connector (both within CONN_SNAP_M).
    """
    cand = {}                                   # (row, cid) -> (distance, vi)
    for j, (fid, _spec, pts, _lon) in enumerate(rows):
        f_i = feats_of.get(str(fid))
        if len(pts) < 2 or f_i is None:
            continue
        f, i = f_i
        table = f.rules.get("connectors")
        if table is None:
            continue
        ll = np.asarray(f.parts(i)[0][0], np.float64)
        if len(ll) < 2:
            continue
        e, n, _ = frame.geodetic_to_enu(ll[:, 1], ll[:, 0], 0.0)
        s = np.concatenate([[0.0], np.cumsum(dnp.hypot(np.diff(e), np.diff(n)))])
        if s[-1] < 1e-6:
            continue
        xy = np.asarray(pts, np.float64)[:, :2]
        for r in table.rows(i):
            d_at = float(table.start[r]) * float(s[-1])
            px, py = float(np.interp(d_at, s, e)), float(np.interp(d_at, s, n))
            dd = dnp.hypot(xy[:, 0] - px, xy[:, 1] - py)
            vi = int(np.argmin(dd))
            if dd[vi] > CONN_SNAP_M:
                continue
            key = (j, str(table.value[r]))
            if key not in cand or (float(dd[vi]), vi) < cand[key]:
                cand[key] = (float(dd[vi]), vi)
    best = {}                                   # (row, vi) -> (distance, cid)
    for (j, cid), (d, vi) in cand.items():
        if (j, vi) not in best or (d, cid) < best[(j, vi)]:
            best[(j, vi)] = (d, cid)
    out = {}
    for (j, vi), (_d, cid) in best.items():
        out.setdefault(j, []).append((vi, cid))
    return {j: sorted(v) for j, v in out.items()}


def _canonical(rows):
    """(caller index, fid, width, xy) of the rows with >= 2 points, in the canonical order of 2.4 (order-free result)."""
    items = []
    for i, (fid, spec, pts, _lon) in enumerate(rows):
        if len(pts) < 2:
            continue
        xy = np.asarray(pts, np.float64)[:, :2]
        k = (str(fid), round(float(xy[0, 0]), 2), round(float(xy[0, 1]), 2),
             round(float(xy[-1, 0]), 2), round(float(xy[-1, 1]), 2), i)
        items.append((k, i, str(fid), float(getattr(spec, "width", 0.0)), xy))
    items.sort(key=lambda t: t[0])
    return [(i, fid, w, xy) for _k, i, fid, w, xy in items]


def nodes_and_chains(rows, connectors=None):
    """(nodes, chains, node_of, chain_of): 2.4 steps 2-3. node_of {(row, vi): node id}, chain_of {row: chain id}."""
    connectors = connectors or {}
    R = _canonical(rows)
    m = len(R)
    if m == 0:
        return [], [], {}, {}
    cnt = [len(r[3]) for r in R]
    off = np.concatenate([[0], np.cumsum(cnt)]).astype(int)
    S = int(off[-1])
    slot_row = np.empty(S, int)
    slot_vi = np.empty(S, int)
    xs, ys = np.empty(S), np.empty(S)
    for j, (_ci, _fid, _w, xy) in enumerate(R):
        a, b = off[j], off[j + 1]
        slot_row[a:b], slot_vi[a:b] = j, np.arange(cnt[j])
        xs[a:b], ys[a:b] = xy[:, 0], xy[:, 1]
    pos = {ci: j for j, (ci, _f, _w, _xy) in enumerate(R)}
    cid_of = [None] * S
    for ci, lst in connectors.items():
        j = pos.get(ci)
        if j is None:
            continue
        for vi, cid in lst:
            if 0 <= vi < cnt[j]:
                cid_of[off[j] + vi] = str(cid)

    parent = list(range(S))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    # (a) same connector ID; (b) exactly equal points of rows
    by_cid = {}
    for s in range(S):
        if cid_of[s] is not None:
            by_cid.setdefault(cid_of[s], []).append(s)
    by_xy = {}
    for s in range(S):
        by_xy.setdefault((round(float(xs[s]), 6), round(float(ys[s]), 6)), []).append(s)
    for g in list(by_cid.values()) + list(by_xy.values()):
        for s in g[1:]:
            union(g[0], s)
    # (c) fallback: a row end within NODE_TOL of a point of another row (grid of NODE_TOL, 3x3 neighbours)
    grid = {}
    for s in range(S):
        grid.setdefault((math.floor(xs[s] / NODE_TOL), math.floor(ys[s] / NODE_TOL)), []).append(s)
    for s in range(S):
        j, vi = int(slot_row[s]), int(slot_vi[s])
        if vi not in (0, cnt[j] - 1):
            continue
        cx, cy = math.floor(xs[s] / NODE_TOL), math.floor(ys[s] / NODE_TOL)
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for t in grid.get((gx, gy), ()):
                    if slot_row[t] != j and float(dnp.hypot(xs[s] - xs[t], ys[s] - ys[t])) <= NODE_TOL:
                        union(s, t)

    classes = {}
    for s in range(S):
        classes.setdefault(find(s), []).append(s)
    # node = class with slots from >= 2 rows; degree and end attachments per run of consecutive vertices
    recs = []
    for g in classes.values():
        if len({int(slot_row[s]) for s in g}) < 2:
            continue
        g.sort()
        degree, atts, k = 0, [], 0
        while k < len(g):
            e = k
            while (e + 1 < len(g) and g[e + 1] == g[e] + 1 and slot_row[g[e + 1]] == slot_row[g[k]]
                   and slot_vi[g[e + 1]] == slot_vi[g[e]] + 1):
                e += 1
            j = int(slot_row[g[k]])
            ends = []
            for t in g[k:e + 1]:
                if slot_vi[t] == 0:
                    ends.append((j, 0))
                if slot_vi[t] == cnt[j] - 1:
                    ends.append((j, 1))
            if ends:
                degree += len(ends)
                atts += ends
            else:
                degree += 2
            k = e + 1
        best = min(g, key=lambda s: (-R[slot_row[s]][2], R[slot_row[s]][1], R[slot_row[s]][0], slot_vi[s]))
        xy = (float(xs[best]), float(ys[best]))
        cids = tuple(sorted({cid_of[s] for s in g if cid_of[s] is not None}))
        key = "c" + cids[0] if cids else "v%.2f,%.2f" % xy
        slots = tuple(sorted((R[slot_row[s]][0], int(slot_vi[s])) for s in g))
        recs.append({"g": g, "key": key, "xy": xy, "slots": slots, "degree": degree, "cids": cids, "atts": atts})
    recs.sort(key=lambda r: (round(r["xy"][0], 2), round(r["xy"][1], 2), r["key"]))

    nodes, node_of_slot = [], [-1] * S
    for nid, r in enumerate(recs):
        for s in r["g"]:
            node_of_slot[s] = nid
        nodes.append(Node(nid, r["key"], np.array(r["xy"], np.float64), r["slots"], r["degree"], r["cids"]))

    # splits (degree 2, two row ends of two rows) join rows; chains are the rows joined over splits
    other = {}
    for r in recs:
        if r["degree"] == 2 and len(r["atts"]) == 2 and r["atts"][0][0] != r["atts"][1][0]:
            (j1, e1), (j2, e2) = r["atts"]
            other[(j1, e1)] = (j2, e2)
            other[(j2, e2)] = (j1, e1)

    def node_at(j, e):
        t = off[j] + (0 if e == 0 else cnt[j] - 1)
        return node_of_slot[t] if node_of_slot[t] >= 0 else None

    def walk(j0, e0):
        seq, cur, ent = [], j0, e0
        while cur is not None and all(cur != x[0] for x in seq):
            seq.append((cur, ent == 1))
            nxt = other.get((cur, 1 - ent))
            if nxt is None:
                break
            cur, ent = nxt
        return seq

    seen, chain_recs = set(), []
    for j in range(m):
        if j in seen:
            continue
        comp, todo = set(), [j]
        while todo:
            x = todo.pop()
            if x in comp:
                continue
            comp.add(x)
            for e in (0, 1):
                if (x, e) in other:
                    todo.append(other[(x, e)][0])
        seen |= comp
        dangling = sorted((x, e) for x in comp for e in (0, 1) if (x, e) not in other)
        if dangling:
            j0, e0 = dangling[0]
            seq = walk(j0, e0)
            last, last_rev = seq[-1]
            start, end, cyclic = node_at(j0, e0), node_at(last, 1 - int(last_rev)), False
        else:
            j0 = min(comp)
            seq = walk(j0, 0)
            start, end, cyclic = None, None, True
        length = float(sum(_length(R[x][3]) for x, _rv in seq))
        chain_recs.append((min(R[x][0] for x, _rv in seq), seq, start, end, length, cyclic))
    chain_recs.sort(key=lambda c: c[0])

    chains, chain_of = [], {}
    for cid, (_mn, seq, start, end, length, cyclic) in enumerate(chain_recs):
        rows_c = tuple((R[x][0], bool(rv)) for x, rv in seq)
        for x, _rv in seq:
            chain_of[R[x][0]] = cid
        chains.append(Chain(cid, rows_c, start, end, np.float64(length), cyclic))
    node_of = {}
    for nid, r in enumerate(recs):
        for slot in r["slots"]:
            node_of[slot] = nid
    return nodes, chains, node_of, chain_of


def chain_ext(rows, nodes, chains, node_of):
    """(n, 2): m of chain beyond each row's start (col 0) and end (col 1), over split nodes up to the first non-split,
    at most ARM_M (2.4 step 3; rows with < 2 points get 0). chains is unused: the splits are the degree-2 nodes."""
    out = np.zeros((len(rows), 2))
    lens = [_length(np.asarray(r[2], np.float64)[:, :2]) if len(r[2]) >= 2 else 0.0 for r in rows]
    for r, row in enumerate(rows):
        if len(row[2]) < 2:
            continue
        for e in (0, 1):
            total, cur, exit_e, seen = 0.0, r, e, {r}
            while True:
                nid = node_of.get((cur, 0 if exit_e == 0 else len(rows[cur][2]) - 1))
                if nid is None or nodes[nid].degree != 2:
                    break
                other = [s for s in nodes[nid].slots
                         if s[0] != cur and s[1] in (0, len(rows[s[0]][2]) - 1)]
                if len(other) != 1 or other[0][0] in seen:
                    break
                nxt, vi2 = other[0]
                seen.add(nxt)
                total += lens[nxt]
                cur, exit_e = nxt, (0 if vi2 == len(rows[nxt][2]) - 1 else 1)
            out[r, e] = min(total, ARM_M)
    return out


# ---- Brief B5: branches, junction nodes, arms, splits, kernel topology (2.4 steps 5-10, 2.5, 4.2) ----

@dataclass
class Arm:
    id: tuple           # (junction key, index); splits: ("s" + node key, 0 | 1)
    junction: str       # junction key ("s" + node key for splits)
    node: int           # part node id (split node id for splits)
    node_key: str
    row: int            # caller row index of the arm's first row
    vi: int             # vertex of that row at the node
    step: int           # +1 along the row's point order, -1 against it
    fid: str
    spec: object
    u: np.ndarray       # (2,) unit vector out of the arm, from p
    p: np.ndarray       # (3,) the node's vertex
    at_start: bool      # step == +1
    length: float       # m to the next junction node or end (<= CHAIN_MAX_M; splits: row length)
    far: str | None     # junction key at the other end (None: dead end, cap, split)
    line: np.ndarray    # (k, 3) polyline from p, cut at min(length, ARM_M)
    chain: tuple        # ((row, reversed), ...) of the rows on that line, in run order
    through: int | None # index of the straight partner in the junction
    ring: bool
    inbound: bool
    outbound: bool
    angle: float
    d_edge: float | None = None

    def as_tuple(self):
        return (self.fid, self.spec, self.u, self.p, self.at_start, self.length, self.node_key, self.id)


@dataclass
class Junction:
    key: str
    index: int
    centre: np.ndarray
    nodes: tuple        # part node ids
    arms: list          # [Arm] in angle order
    internal_chains: tuple
    ring: bool          # >= 2 arms on a roundabout ring
    polygon: np.ndarray | None = None
    islands: tuple = ()

    def arm_tuples(self):
        return [a.as_tuple() for a in self.arms]


@dataclass
class Model:
    rows: list
    nodes: list
    chains: list
    junctions: dict     # key -> Junction, in index order
    splits: dict        # "s" + node key -> (Arm, Arm)
    ring_fids: frozenset
    node_of: dict       # (row, vi) -> node id
    chain_of: dict      # row -> chain id
    internal: frozenset # chain ids inside one junction

    def junction(self, key):
        return self.junctions[key]

    def arm(self, arm_id):
        key, i = arm_id
        return self.splits[key][i] if key in self.splits else self.junctions[key].arms[i]

    def arms_dict(self, splits=False):
        out = {k: j.arm_tuples() for k, j in self.junctions.items()}
        if splits:
            out.update({k: [a.as_tuple() for a in v] for k, v in self.splits.items()})
        return out

    def link_ok(self, signals):
        import furniture
        def ok(a):
            arm = self.arm(a[7])
            if arm.far is None:
                return False
            arms = self.junctions[arm.far].arm_tuples()
            return len(arms) >= 3 and furniture._controlled_here(arms, signals) and \
                furniture._is_signalled(arms, signals)
        return ok

    def kernel_topology(self, n):
        """The topology arrays of section 4.2 for the first n rows (the halo rows are left out)."""
        cnt = [len(r[2]) for r in self.rows]
        part = [nd.id for nd in self.nodes if nd.degree >= 3]
        newid = {nid: k for k, nid in enumerate(part)}
        jidx = {}
        for jn in self.junctions.values():
            for a in jn.nodes:
                jidx[a] = jn.index
        node = np.full(sum(cnt[:n]), -1, np.int64)
        off = 0
        for j in range(n):
            for vi in range(cnt[j]):
                node[off + vi] = newid.get(self.node_of.get((j, vi)), -1)
            off += cnt[j]
        group = np.array([jidx[a] for a in part], np.int64)
        rank = np.empty(n, np.int64)
        dense = {}
        for j in range(n):
            c = self.chain_of.get(j)
            rank[j] = dense.setdefault(c if c is not None else ("solo", j), len(dense))
        internal = np.zeros(n, bool)
        members = {}
        revd = {}
        for c in self.chains:
            for r, rv in c.rows:
                revd[r] = rv
            if c.id in self.internal:
                for r, _rv in c.rows:
                    if r < n:
                        internal[dense[c.id]] = True
        for j in range(n):
            c = self.chain_of.get(j)
            if c is not None:
                members.setdefault(c, []).append(j)
        same = {}
        for j in range(n):
            c = self.chain_of.get(j)
            if c is None:
                continue
            same[j] = [(k, revd[j] != revd[k]) for k in members[c] if k != j and k < n]
        ext = chain_ext(self.rows, self.nodes, self.chains, self.node_of)[:n]
        return {"node": node, "group": group, "rank": rank, "internal": internal, "ext": ext, "same": same}

    def station(self, arm, d):
        """2.6: (xy (2,), z, t (2,)) of arm.line at station d (m from arm.p, clamped), t the outward unit tangent."""
        return _along(arm.line, d)


def _cut(pts, m):
    """pts (k, 3) cut at arc length m (the end point interpolated); all of pts when shorter."""
    out, tot = [pts[0]], 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        seg = float(dnp.hypot(b[0] - a[0], b[1] - a[1]))
        if tot + seg >= m:
            if seg > 0.0:
                out.append(a + (b - a) * ((m - tot) / seg))
            break
        out.append(b)
        tot += seg
    return np.array(out, np.float64)


def _unit(v):
    n = float(dnp.hypot(v[0], v[1]))
    return np.array([v[0] / n, v[1] / n]) if n > 1e-9 else np.array([1.0, 0.0])


def _span(xys):
    """Largest distance between two of the points xys (k, 2)."""
    p = np.asarray(xys, np.float64)
    if len(p) < 2:
        return 0.0
    return float(np.max(dnp.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])))


def _split_other(nodes, node_of, cnt, j, e):
    """(row, end) on the other side of the split at row j's end e (0 start, 1 end); None where that end is no split."""
    nid = node_of.get((j, 0 if e == 0 else cnt[j] - 1))
    if nid is None or nodes[nid].degree != 2:
        return None
    for r, vi in nodes[nid].slots:
        if r != j:
            return r, (0 if vi == 0 else 1)
    return None


@dataclass
class _Branch:
    node: int           # part node the walk starts at
    j: int              # start row
    vi: int             # start vertex
    s: int              # start step
    far: int | None     # part node the walk ends at
    pts: np.ndarray     # (k, 3) walk polyline from the start vertex
    chain: list         # [(row, reversed, index of its first point)]
    length: float
    last: tuple         # (row, vertex, step) where the walk ended
    u: np.ndarray       # (2,) direction at BRANCH_M


def _walk(XYZ, cnt, nodes, node_of, j, vi, s):
    """2.4 step 5: from vertex vi of row j in direction s, over splits, to the next part node (far), a dead end or
    CHAIN_MAX_M. Returns (far, pts, chain, length, last); a row already visited ends the walk."""
    pts, chain, seen = [XYZ[j][vi]], [(j, s < 0, 0)], {j}
    cur, idx, d, length, far = j, vi, s, 0.0, None
    while length < CHAIN_MAX_M:
        if 0 <= idx + d < cnt[cur]:
            idx += d
        else:
            nb = _split_other(nodes, node_of, cnt, cur, 0 if idx == 0 else 1)
            if nb is None or nb[0] in seen:
                break
            cur, seen = nb[0], seen | {nb[0]}
            d = 1 if nb[1] == 0 else -1
            idx = 0 if nb[1] == 0 else cnt[cur] - 1
            chain.append((cur, d < 0, len(pts)))
        pts.append(XYZ[cur][idx])
        length += float(dnp.hypot(pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]))
        nid = node_of.get((cur, idx))
        if nid is not None and nodes[nid].degree >= 3:
            far = nid
            break
    return far, np.array(pts, np.float64), chain, min(length, CHAIN_MAX_M), (cur, idx, d)


def _span_pts(nodes, group):
    return _span([nodes[n].xy for n in group])


def _along(pts, d):
    """(xy (2,), z, t (2,)): the point at arc length d along pts (k, 3), clamped to its ends, and the unit tangent."""
    p = np.asarray(pts, np.float64)
    if len(p) < 2:
        return p[0, :2].copy(), float(p[0, 2]), np.array([1.0, 0.0])
    seg = np.asarray(dnp.hypot(p[1:, 0] - p[:-1, 0], p[1:, 1] - p[:-1, 1]), np.float64)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    d = min(max(float(d), 0.0), float(cum[-1]))
    i = max(min(int(np.searchsorted(cum, d, side="right")) - 1, len(seg) - 1), 0)
    f = (d - float(cum[i])) / float(seg[i]) if seg[i] > 0.0 else 0.0
    a, b = p[i], p[i + 1]
    return a[:2] + (b[:2] - a[:2]) * f, float(a[2] + (b[2] - a[2]) * f), _unit(b[:2] - a[:2])


def _along_many(pts, ds):
    """_along at each arc length of ds: (xy (n, 2), t (n, 2)), the same numbers point by point."""
    p = np.asarray(pts, np.float64)
    ds = np.asarray(ds, np.float64)
    if len(p) < 2:
        return np.repeat(p[:1, :2], len(ds), 0), np.tile([1.0, 0.0], (len(ds), 1))
    seg = np.asarray(dnp.hypot(p[1:, 0] - p[:-1, 0], p[1:, 1] - p[:-1, 1]), np.float64)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    d = np.minimum(np.maximum(ds, 0.0), float(cum[-1]))
    i = np.clip(np.searchsorted(cum, d, side="right") - 1, 0, len(seg) - 1)
    sg = seg[i]
    f = np.where(sg > 0.0, (d - cum[i]) / np.where(sg > 0.0, sg, 1.0), 0.0)
    a, b = p[i, :2], p[i + 1, :2]
    v = b - a
    n = np.asarray(dnp.hypot(v[:, 0], v[:, 1]), np.float64)
    ok = n > 1e-9
    t = np.where(ok[:, None], v / np.where(ok, n, 1.0)[:, None], np.array([1.0, 0.0]))
    return a + v * f[:, None], t


def _line_len(pts):
    p = np.asarray(pts, np.float64)
    return float(np.sum(dnp.hypot(p[1:, 0] - p[:-1, 0], p[1:, 1] - p[:-1, 1]))) if len(p) > 1 else 0.0


def _dist_many(q, line):
    """Distance (m) of each point of q (m, 2) to the polyline line (k, >= 2; xy used); cf. leonberg_junction_core._dist_line."""
    q = np.asarray(q, np.float64)
    p = np.asarray(line, np.float64)[:, :2]
    if len(p) < 2:
        return np.asarray(dnp.hypot(*(q - p[0]).T), np.float64)
    a, b = p[:-1], p[1:]
    d = b - a
    L2 = np.maximum((d * d).sum(1), 1e-12)
    t = np.clip(((q[:, None, :] - a[None]) * d[None]).sum(2) / L2[None], 0.0, 1.0)
    diff = a[None] + t[..., None] * d[None] - q[:, None, :]
    return np.asarray(dnp.hypot(diff[..., 0], diff[..., 1]), np.float64).min(1)


def _seg_cross(P, Q):
    """True if a segment of the polyline P (k, 2) meets one of Q (m, 2) (touching counts)."""
    if len(P) < 2 or len(Q) < 2:
        return False
    p1, p2 = P[:-1][None], P[1:][None]
    q1, q2 = Q[:-1][:, None], Q[1:][:, None]

    def orient(o, a, b):
        return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])

    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    return bool(np.any((d1 * d2 <= 0.0) & (d3 * d4 <= 0.0)))


def _geometry(rows, jn, chain_by):
    """2.6: d_edge of each arm of junction jn, and its polygon (CCW ring of the arm edges, arms in angle order)."""
    for arm in jn.arms:
        total = _line_len(arm.line)
        if total <= 0.0:
            continue
        half = float(arm.spec.width) / 2
        lines = [(o.line, float(o.spec.width) / 2) for o in jn.arms if o is not arm]
        lines += [(rows[r][2], float(rows[r][1].width) / 2)
                  for cid in jn.internal_chains for r, _rv in chain_by[cid].rows]
        S = np.arange(0.0, min(ARM_M, total) + 1e-9, EDGE_STEP_M)

        def clear(S):
            xy, t = _along_many(arm.line, S)
            n = np.c_[-t[:, 1], t[:, 0]] * half
            Q = np.stack([xy + n, xy - n], 1).reshape(-1, 2)
            ok = np.ones(len(Q), bool)
            for pts, h in lines:
                ok &= _dist_many(Q, pts) > h
            return ok.reshape(-1, 2).all(1)
        ok = clear(S)
        if ok.any():
            k = int(np.argmax(ok))
            lo, hi = (float(S[k - 1]), float(S[k])) if k else (float(S[0]), float(S[0]))
            for _ in range(EDGE_REFINE):        # (the raster's first clear station, refined to the edge itself:
                if hi - lo <= 0.0:              # 32 stations at once, twice)
                    break
                T = lo + (hi - lo) * np.arange(1, 33) / 32.0
                ok = clear(T)
                k2 = int(np.argmax(ok))
                lo, hi = (float(T[k2 - 1]) if k2 else lo), float(T[k2])
            arm.d_edge = hi
    if len(jn.arms) < 3:
        jn.polygon = None
        return
    ring = []
    for arm in jn.arms:
        xy, _z, t = _along(arm.line, arm.d_edge if arm.d_edge is not None else 0.0)
        n = np.array([-t[1], t[0]]) * (float(arm.spec.width) / 2)
        ring += [xy - n, xy + n]
    jn.polygon = np.array(ring, np.float64)


def ring_specs(rows, left_hand=False):
    """(rows, ring_dir): the roundabout block of roads.build_tile (roads.py:11444-11460) as a function, so that tile
    and graph make the same spec changes. ring_dir: {fid: direction along the geometry} of the rings."""
    ring_dir = {}
    if rows:
        import islands
        ring_dir = islands.ring_directions(rows, left_hand)
        if ring_dir:
            apps = islands.ring_approaches(rows, ring_dir)
            short_apps = {fid for fid, sp, pts, _ in rows if fid in apps
                          and float(np.sum(dnp.hypot(*np.diff(pts[:, :2], axis=0).T))) < islands.APPROACH_NO_WALK_M}
            rows = [(fid,
                     (sp._replace(urban=False, median=None, oneway=True, dir=ring_dir[fid], street=None) if fid in ring_dir
                      else sp._replace(urban=False, median=None, street=None) if fid in short_apps
                      else sp),
                     pts, lon) for fid, sp, pts, lon in rows]
    return rows, ring_dir


def crossing_pairs(model):
    """2.7 (R1-05): {(fid, fid2)}: fid2 really crosses fid, and then (fid2, fid) is in the set too. Rows that meet at
    a node of the model (junction or split), or are arms or internal rows of one junction, do not cross."""
    rows = model.rows
    fids = [str(r[0]) for r in rows]
    XY = [np.asarray(r[2], np.float64)[:, :2] for r in rows]
    W = [float(r[1].width) / 2 for r in rows]
    lo = [x.min(0) - w for x, w in zip(XY, W)]
    hi = [x.max(0) + w for x, w in zip(XY, W)]
    chain_by = {c.id: c for c in model.chains}
    excl = set()

    def pairs_of(names):
        fs = sorted(set(names))
        excl.update((fs[x], fs[y]) for x in range(len(fs)) for y in range(x + 1, len(fs)))

    for nd in model.nodes:
        pairs_of(fids[r] for r, _vi in nd.slots)
    for jn in model.junctions.values():
        pairs_of([a.fid for a in jn.arms] + [fids[r] for cid in jn.internal_chains for r, _rv in chain_by[cid].rows])

    def hits(a, b):
        """Row b crosses row a: the centre lines meet, or an end of b lies within a's half width of its centre line."""
        return _seg_cross(XY[a], XY[b]) or bool(np.any(_dist_many(XY[b][[0, -1]], XY[a]) <= W[a]))

    out = set()
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            fi, fj = fids[i], fids[j]
            if fi == fj or (min(fi, fj), max(fi, fj)) in excl:
                continue
            if np.any(lo[i] > hi[j]) or np.any(lo[j] > hi[i]):
                continue
            if hits(i, j) or hits(j, i):
                out.update({(fi, fj), (fj, fi)})
    return out


def build(rows, connectors=None, ring_fids=()):
    """The junction model of rows (2.4 steps 1-10): part nodes, bundled junctions with their arms, split arms."""
    from furniture import _outbound
    nodes, chains, node_of, chain_of = nodes_and_chains(rows, connectors)
    ring = frozenset(str(f) for f in ring_fids)
    XYZ = [np.asarray(r[2], np.float64) for r in rows]
    cnt = [len(p) for p in XYZ]

    def is_ring(j):
        return bool(getattr(rows[j][1], "roundabout", False)) or str(rows[j][0]) in ring

    part = [nd.id for nd in nodes if nd.degree >= 3]
    # step 5: branches of the part nodes
    branches, at, by_start = [], {}, {}
    for a in part:
        for j, vi in sorted(nodes[a].slots, key=lambda t: (str(rows[t[0]][0]), t[1])):
            for s in (1, -1):
                if not 0 <= vi + s < cnt[j] or node_of.get((j, vi + s)) == a:
                    continue
                far, pts, chain, length, last = _walk(XYZ, cnt, nodes, node_of, j, vi, s)
                u = _unit(_cut(pts, BRANCH_M)[-1][:2] - pts[0][:2])
                b = _Branch(a, j, vi, s, far, pts, chain, length, last, u)
                branches.append(b)
                at.setdefault(a, []).append(b)
                by_start[(j, vi, s)] = b

    def reverse(b):
        cur, idx, d = b.last
        return by_start.get((cur, idx, -d))

    def half(bs, b):
        """step 6, hA: half the widest arm at this part node that is neither b nor its straight partner."""
        others = [o for o in bs if o is not b]
        if not others:
            return 0.0
        partner = min(others, key=lambda o: float(dnp.dot(o.u, b.u)))
        if float(dnp.dot(partner.u, b.u)) > -dmath.cos(math.radians(STRAIGHT_DEG)):
            partner = None
        return max([float(rows[o.j][1].width) / 2 for o in others if o is not partner], default=0.0)

    # step 6: bundling candidates (a) close nodes, (b) branches ending at another node (R2-02)
    cands = []
    for x, A in enumerate(part):
        for B in part[x + 1:]:
            dd = float(dnp.hypot(*(nodes[A].xy - nodes[B].xy)))
            if dd <= MERGE_M:
                cands.append((dd, nodes[A].key, nodes[B].key, A, B, None))
    for b in branches:
        if b.far is None or b.far == b.node or any(is_ring(c[0]) for c in b.chain):
            continue
        r = reverse(b)
        h_b = half(at[b.node], b) + (half(at.get(b.far, []), r) if r is not None else 0.0)
        if b.length <= h_b + OVERLAP_M:
            cands.append((b.length, nodes[b.node].key, nodes[b.far].key, b.node, b.far, chain_of[b.chain[0][0]]))
    par = {a: a for a in part}
    mem = {a: [a] for a in part}

    def root(a):
        while par[a] != a:
            a = par[a]
        return a

    merged_chains = set()
    for _v, _ka, _kb, A, B, cid in sorted(cands, key=lambda c: c[:3]):
        ra, rb = root(A), root(B)
        if ra == rb or _span_pts(nodes, mem[ra] + mem[rb]) > SPAN_MAX_M:
            continue
        ra, rb = min(ra, rb), max(ra, rb)
        par[rb] = ra
        mem[ra] = mem[ra] + mem.pop(rb)
        if cid is not None:
            merged_chains.add(cid)

    # step 7: junctions, indexed by their centre
    info = []
    for g in mem.values():
        g = sorted(g)
        centre = np.mean([nodes[a].xy for a in g], axis=0)
        head = min(g, key=lambda a: (-nodes[a].degree, nodes[a].key))
        info.append((round(float(centre[0]), 2), round(float(centre[1]), 2), nodes[head].key, g, centre))
    info.sort(key=lambda t: t[:3])
    jof = {a: jk for _x, _y, jk, g, _c in info for a in g}
    # step 8: internal chains (R2-14)
    internal = set()
    for c in chains:
        if c.start is None or c.end is None or jof[c.start] != jof[c.end]:
            continue
        if float(c.length) <= 3 * MERGE_M or c.id in merged_chains:
            internal.add(c.id)
    jint = {}
    for c in chains:
        if c.id in internal:
            jint.setdefault(jof[c.start], []).append(c.id)
    # step 9: arms
    centre_of = {jk: cen for _x, _y, jk, _g, cen in info}
    pend = {}
    for b in branches:
        a = b.node
        if b.far is not None and jof[b.far] == jof[a] and b.length <= SPAN_MAX_M:
            continue
        if any(chain_of.get(r) in internal for r, _rv, _p in b.chain):
            continue
        ln = _cut(b.pts, min(b.length, ARM_M))
        kept = [(r, rv) for r, rv, p0 in b.chain if p0 < len(ln)]
        ahead = _cut(b.pts, BRANCH_M)[-1][:2] - centre_of[jof[a]]
        arm = Arm(None, jof[a], a, nodes[a].key, b.j, b.vi, b.s, str(rows[b.j][0]), rows[b.j][1], b.u, b.pts[0],
                  b.s == 1, b.length, jof[b.far] if b.far is not None else None, ln, tuple(kept), None,
                  is_ring(b.j), False, False, float(dnp.arctan2(ahead[1], ahead[0])))
        pend.setdefault(jof[a], []).append(arm)
    junctions = {}
    for idx_j, (_x, _y, jk, g, cen) in enumerate(info):
        arms = sorted(pend.get(jk, []), key=lambda a: (a.angle, a.fid, not a.at_start))
        for i, a in enumerate(arms):
            a.id = (jk, i)
        if len(arms) >= 2:      # through: mutually best partners, opposite within STRAIGHT_DEG
            dot = [[float(dnp.dot(x.u, y.u)) if x is not y else np.inf for y in arms] for x in arms]
            best = [min(range(len(arms)), key=lambda k: dot[i][k]) for i in range(len(arms))]
            lim = -dmath.cos(math.radians(STRAIGHT_DEG))
            for i, a in enumerate(arms):
                if best[best[i]] == i and dot[i][best[i]] <= lim:
                    a.through = best[i]
        for a in arms:
            a.outbound = bool(_outbound(a.as_tuple()))
            a.inbound = not a.outbound
        junctions[jk] = Junction(jk, idx_j, cen, tuple(g), arms, tuple(sorted(jint.get(jk, []))),
                                 sum(a.ring for a in arms) >= 2)
    # 2.6: d_edge and polygon of the junctions
    chain_by = {c.id: c for c in chains}
    for jn in junctions.values():
        _geometry(rows, jn, chain_by)
    # step 10: split arms (designs.end_way_starts)
    splits = {}
    for nd in nodes:
        if nd.degree != 2 or len(nd.slots) != 2:
            continue
        key, pair = "s" + nd.key, []
        for k, (j, vi) in enumerate(sorted(nd.slots, key=lambda t: (str(rows[t[0]][0]), t[1]))):
            e = 0 if vi == 0 else 1
            pts = XYZ[j] if e == 0 else XYZ[j][::-1]
            length = _length(pts[:, :2])
            arm = Arm((key, k), key, nd.id, nd.key, j, vi, 1 if e == 0 else -1, str(rows[j][0]), rows[j][1],
                      _unit(_cut(pts, BRANCH_M)[-1][:2] - pts[0][:2]), pts[0], e == 0, length, None,
                      _cut(pts, min(length, ARM_M)), ((j, e == 1),), None, is_ring(j), False, False, 0.0)
            arm.outbound = bool(_outbound(arm.as_tuple()))
            arm.inbound = not arm.outbound
            pair.append(arm)
        splits[key] = tuple(pair)
    return Model(rows, nodes, chains, junctions, splits, ring, node_of, chain_of, frozenset(internal))
