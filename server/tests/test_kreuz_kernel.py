"""The junction kernel's topology mode (earthfast junctions_find, native 306; Phase 2 section 4.2, brief B2).

pin = the topo=None outcome of today, bit for bit; topo mode on the R2 fixtures with hand-built topology (nodes = the
shared points of degree >= 3, one group each; a chain = rows meeting end to end straight through; ext = the chain's
length beyond a row's end where it runs on through a split point).
"""
import math
import os
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import junctions  # noqa: E402


def sp(cls="residential", w=7.0, oneway=False, d=1, link=False):
    return types.SimpleNamespace(cls=cls, width=w, walk=2.0, oneway=oneway, dir=d, roundabout=False, street=None,
                                 kind="paved", link=link)


def road(fid, pts, spec):
    return (fid, spec, np.array([[x, y, 0.0] for x, y in pts], float), 9.0)


def _dist(p, pts):
    best = math.inf
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        t = min(1.0, max(0.0, float(np.dot(p - a, ab)) / max(float(np.dot(ab, ab)), 1e-12)))
        best = min(best, float(np.hypot(*(p - (a + t * ab)))))
    return best


def _nearest_fid(xy, roads):
    return min(roads, key=lambda r: _dist(xy, r[2][:, :2]))[0]


def outcome(roads, topo=None):
    """Corners (count, radii as 2/pi x arc length rounded to 0.5 m), the fids of the crosswalks, stop lines, and the
    line cuts' fids and the internal rows."""
    corners, _cuts, cws, _ramps, lc, internal = junctions.find(roads, lambda s: True, None, topo=topo)
    radii = []
    for k in corners:
        arc = np.asarray(k["arc"], float)[:, :2]
        length = float(np.sum(np.hypot(*np.diff(arc, axis=0).T)))
        radii.append(round(2 / math.pi * length * 2) / 2)
    cw, stops = [], []
    for x in cws:
        mid = (np.asarray(x["a"], float)[:2] + np.asarray(x["b"], float)[:2]) / 2
        (stops if x.get("stop") else cw).append(_nearest_fid(mid, roads))
    return {"corners": len(corners), "radii": sorted(radii), "crosswalks": sorted(cw), "stops": sorted(stops),
            "line_cuts": sorted(lc), "internal": internal}


def topo_for(roads, nodes, rank, same, ext=None, internal=None, group=None):
    """The topology a model would hand over: nodes {(x, y): node id} (every point there gets the id), group per node
    (default: each its own), rank per row, same {row: [(row2, flip)]}, ext per row ((n, 2), default 0)."""
    node = []
    for _f, _s, p, _l in roads:
        for x, y in p[:, :2]:
            node.append(nodes.get((round(float(x), 6), round(float(y), 6)), -1))
    n = len(roads)
    ngroup = (max(nodes.values()) + 1) if nodes else 0
    return {"node": node, "group": group if group is not None else list(range(ngroup)), "rank": rank,
            "internal": internal if internal is not None else [False] * n,
            "ext": ext if ext is not None else [[0.0, 0.0]] * n, "same": same}


MAIN = sp("secondary", 9.0)
SIDE = sp("residential", 7.0)
TERT = sp("tertiary", 8.0)
SLIP_PTS = [(-30, 0)] + [(-30 + 30 * np.sin(t) * 0.95, -30 * (1 - np.cos(t)) * 0.95)
                         for t in np.linspace(0, np.pi / 2, 8)[1:-1]] + [(0, -30)]
SLIP = sp("secondary", 4.5, True, 1, True)

T4_PLAIN = [road("a", [(-60, 0), (0, 0)], MAIN), road("b", [(0, 0), (60, 0)], MAIN),
            road("c", [(0, 0), (0, -60)], SIDE), road("d", [(0, 0), (0, 60)], SIDE)]


def t4_split(L):
    return [road("a", [(-60, 0), (0, 0)], MAIN), road("b", [(0, 0), (60, 0)], MAIN),
            road("c1", [(0, 0), (0, -L)], SIDE), road("c2", [(0, -L), (0, -60)], SIDE),
            road("d", [(0, 0), (0, 60)], SIDE)]


def slip_split():
    return [road("w1", [(-80, 0), (-30, 0)], MAIN), road("w2", [(-30, 0), (0, 0)], MAIN),
            road("e", [(0, 0), (80, 0)], MAIN), road("s2", [(0, 0), (0, -30)], TERT),
            road("s1", [(0, -30), (0, -80)], TERT), road("slip", SLIP_PTS, SLIP),
            road("n", [(0, 0), (0, 80)], TERT)]


def t4_split_topo(roads):
    # chains: a+b straight through, c1+c2 through the split point (0,-12); the c arm runs 60 m from the node
    rank = [0, 0, 1, 1, 2]
    same = {0: [(1, False)], 1: [(0, False)], 2: [(3, False)], 3: [(2, False)]}
    ext = [[0.0, 0.0], [0.0, 0.0], [0.0, 48.0], [12.0, 0.0], [0.0, 0.0]]
    return topo_for(roads, {(0.0, 0.0): 0}, rank, same, ext=ext)


def slip_topo(roads):
    rank = [0, 0, 1, 2, 2, 3, 4]
    same = {0: [(1, False)], 1: [(0, False)], 3: [(4, False)], 4: [(3, False)]}
    nodes = {(0.0, 0.0): 0, (0.0, -30.0): 1, (-30.0, 0.0): 2}
    return topo_for(roads, nodes, rank, same, internal=[False] * 7)


class KernelPinTests(unittest.TestCase):
    """topo=None: today's outcome (the kernel's numbers are the same)."""

    def test_t4_plain(self):
        s = outcome(T4_PLAIN)
        self.assertEqual(s["corners"], 4)
        self.assertEqual(s["radii"], [6, 6, 6, 6])
        self.assertEqual(s["crosswalks"], ["a", "b", "c", "d"])
        self.assertEqual(s["stops"], ["a", "b", "c", "d"])
        self.assertEqual(s["line_cuts"], ["a", "b", "c", "d"])

    def test_t4_split_without_topo(self):
        s = outcome(t4_split(12))
        self.assertEqual(s["crosswalks"], ["a", "b", "d"])

    def test_slip_split_without_topo(self):
        s = outcome(slip_split())
        self.assertEqual(s["stops"], ["e", "n"])
        self.assertEqual(s["line_cuts"], ["e", "n"])


class KernelTopoTests(unittest.TestCase):
    def test_t4_split_topo_corners_and_crossing(self):
        roads = t4_split(12)
        s = outcome(roads, t4_split_topo(roads))
        self.assertEqual(s["corners"], 4)
        self.assertEqual(s["radii"], [6, 6, 6, 6])
        self.assertTrue(set(s["crosswalks"]) & {"c1", "c2"})
        self.assertTrue(set(s["stops"]) & {"c1", "c2"})
        self.assertTrue({"c1", "c2"} <= set(s["line_cuts"]))

    def test_knick_is_no_junction(self):
        roads = [road("p", [(-60, 0), (0, 0)], MAIN), road("q", [(0, 0), (42.4, 42.4)], MAIN)]
        topo = topo_for(roads, {}, [0, 0], {0: [(1, False)], 1: [(0, False)]})
        self.assertEqual(outcome(roads, topo)["corners"], 0)

    def test_internal_rows_follow_rank(self):
        roads = [road("p", [(-60, 0), (0, 0)], MAIN), road("q", [(0, 0), (42.4, 42.4)], MAIN),
                 road("r", [(90, 90), (100, 100)], SIDE)]
        topo = topo_for(roads, {}, [0, 0, 1], {0: [(1, False)], 1: [(0, False)]},
                        internal=[True, False, False])
        self.assertEqual(outcome(roads, topo)["internal"], {0, 1})

    def test_slip_topo_adds_line_cuts(self):
        roads = slip_split()
        s = outcome(roads, slip_topo(roads))
        self.assertTrue({"w2", "s2"} <= set(s["line_cuts"]))


if __name__ == "__main__":
    unittest.main()
