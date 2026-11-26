"""The R2 junction cases (docs/leonberg/kreuzungen/r2_faelle/t1-t4.py as unit tests) and the K1/K2 junction fixtures.

pin = holds today and must stay green; xfail = a known fault, marked expectedFailure with its phase: the phase that
fixes it removes the decorator (an unexpected success then fails the run until it does).
"""
import math
import os
import sys
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import islands  # noqa: E402
import junctionmodel  # noqa: E402
import junctions  # noqa: E402
import kreuzfixtures  # noqa: E402


def sp(cls="residential", w=7.0, oneway=False, d=1, link=False):
    return types.SimpleNamespace(cls=cls, width=w, walk=2.0, oneway=oneway, dir=d, roundabout=False, street=None,
                                 kind="paved", link=link)


def road(fid, pts, spec):
    return (fid, spec, np.array([[x, y, 0.0] for x, y in pts], float), 9.0)


def _dist(p, pts):
    """Distance from point p (2,) to the polyline pts (k, 2)."""
    best = math.inf
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        t = min(1.0, max(0.0, float(np.dot(p - a, ab)) / max(float(np.dot(ab, ab)), 1e-12)))
        best = min(best, float(np.hypot(*(p - (a + t * ab)))))
    return best


def _nearest_fid(xy, roads):
    """The road whose centreline is nearest to xy."""
    return min(roads, key=lambda r: _dist(xy, r[2][:, :2]))[0]


def summary(roads, has_walk=lambda s: True, alias=None):
    """Junction outcome of roads: corners (count, radii 2/pi x arc length rounded to 0.5 m), the fids of the
    crosswalks, stop lines and line cuts. Phase 2 replaces only the body (the new junction model); the expectations stay.
    alias maps split-road fids to their street's name."""
    alias = alias or {}
    name = lambda f: alias.get(f, f)  # noqa: E731
    m = junctionmodel.build(roads)
    corners, _cuts, cws, _ramps, lc, _internal = junctions.find(roads, has_walk, None,
                                                                topo=m.kernel_topology(len(roads)))
    radii = []
    for k in corners:
        arc = np.asarray(k["arc"], float)[:, :2]
        length = float(np.sum(np.hypot(*np.diff(arc, axis=0).T)))
        radii.append(round(2 / math.pi * length * 2) / 2)
    mids = lambda x: (np.asarray(x["a"], float)[:2] + np.asarray(x["b"], float)[:2]) / 2  # noqa: E731
    return {
        "corners": len(corners),
        "radii": sorted(radii),
        "crosswalks": sorted({name(_nearest_fid(mids(x), roads)) for x in cws if not x.get("stop")}),
        "stops": sorted({name(_nearest_fid(mids(x), roads)) for x in cws if x.get("stop")}),
        # per street like the other fields: Phase 2 copies a cut to every piece of the chain (junctions._copy_same)
        "line_cuts": sorted({name(f) for f, cut in lc.items() if cut}),
    }


def islands_of(roads):
    return [(sorted(f["inner"]), f["ring"] is not None)
            for f in islands.find(roads, model=junctionmodel.build(roads))]


MAIN = sp("secondary", 9.0)
SIDE = sp("residential", 7.0)
TERT = sp("tertiary", 8.0)
SLIP_PTS = [(-30, 0)] + [(-30 + 30 * np.sin(t) * 0.95, -30 * (1 - np.cos(t)) * 0.95)
                         for t in np.linspace(0, np.pi / 2, 8)[1:-1]] + [(0, -30)]
SLIP = sp("secondary", 4.5, True, 1, True)


def tee(c_pts=((0, 0), (0, -60))):
    return [road("a", [(-60, 0), (0, 0)], MAIN), road("b", [(0, 0), (60, 0)], MAIN),
            road("c", list(c_pts), SIDE)]


def t4_split(L):
    return [road("a", [(-60, 0), (0, 0)], MAIN), road("b", [(0, 0), (60, 0)], MAIN),
            road("c1", [(0, 0), (0, -L)], SIDE), road("c2", [(0, -L), (0, -60)], SIDE),
            road("d", [(0, 0), (0, 60)], SIDE)]


T4_PLAIN = [road("a", [(-60, 0), (0, 0)], MAIN), road("b", [(0, 0), (60, 0)], MAIN),
            road("c", [(0, 0), (0, -60)], SIDE), road("d", [(0, 0), (0, 60)], SIDE)]
ALIAS_C = {"c1": "c", "c2": "c"}
T_PLAIN = summary(tee())


class TeeTests(unittest.TestCase):
    def test_tee_plain(self):
        s = summary(tee())
        self.assertEqual(s["stops"], ["c"])
        self.assertEqual(s["crosswalks"], ["c"])
        self.assertEqual(s["line_cuts"], ["c"])

    def test_tee_side_split_6(self):
        R = [road("a", [(-60, 0), (0, 0)], MAIN), road("b", [(0, 0), (60, 0)], MAIN),
             road("c1", [(0, 0), (0, -6)], SIDE), road("c2", [(0, -6), (0, -60)], SIDE)]
        self.assertEqual(summary(R, alias=ALIAS_C), T_PLAIN)

    def test_tee_side_split_12(self):
        """Phase 2, R2-01: a side road split 12 m from the junction is still one junction."""
        R = [road("a", [(-60, 0), (0, 0)], MAIN), road("b", [(0, 0), (60, 0)], MAIN),
             road("c1", [(0, 0), (0, -12)], SIDE), road("c2", [(0, -12), (0, -60)], SIDE)]
        self.assertEqual(summary(R, alias=ALIAS_C), T_PLAIN)

    def test_tee_main_split_8(self):
        """Phase 2, R2-02: a main road split 8 m from the junction is still one junction."""
        R = [road("a", [(-60, 0), (-8, 0)], MAIN), road("a2", [(-8, 0), (0, 0)], MAIN),
             road("b", [(0, 0), (60, 0)], MAIN), road("c", [(0, 0), (0, -60)], SIDE)]
        self.assertEqual(summary(R, alias={"a2": "a"}), T_PLAIN)

    def test_offset_tee_15(self):
        R = [road("a", [(-60, 0), (0, 0)], MAIN), road("m", [(0, 0), (15, 0)], MAIN),
             road("b", [(15, 0), (60, 0)], MAIN), road("c", [(0, 0), (0, -60)], SIDE),
             road("d", [(15, 0), (15, 60)], SIDE)]
        s = summary(R)
        self.assertEqual(s["stops"], ["c", "d"])
        for part in ("crosswalks", "stops", "line_cuts"):
            self.assertFalse({"a", "m", "b"} & set(s[part]), part)

    def test_offset_tee_8(self):
        """Phase 2, R2-02: two side roads 8 m apart on opposite sides act like the 15 m offset tee."""
        R = [road("a", [(-60, 0), (0, 0)], MAIN), road("m", [(0, 0), (8, 0)], MAIN),
             road("b", [(8, 0), (60, 0)], MAIN), road("c", [(0, 0), (0, -60)], SIDE),
             road("d", [(8, 0), (8, 60)], SIDE)]
        s = summary(R)
        self.assertEqual(s["stops"], ["c", "d"])
        for part in ("crosswalks", "stops", "line_cuts"):
            self.assertFalse({"a", "m", "b"} & set(s[part]), part)

    def test_split_near_and_far(self):
        ref = summary(T4_PLAIN)
        for L in (6, 8, 18):
            with self.subTest(L=L):
                self.assertEqual(summary(t4_split(L), alias=ALIAS_C), ref)

    def test_split_mid(self):
        """Phase 2, R2-01: a side road split 9.5 to 16 m from a 4-way junction is still one 4-way junction."""
        ref = summary(T4_PLAIN)
        for L in (9.5, 10, 11, 12, 14, 16):
            with self.subTest(L=L):
                self.assertEqual(summary(t4_split(L), alias=ALIAS_C), ref)

    def test_divided_stops_incoming_only(self):
        cw1, cw2 = sp("primary", 7.0, True, 1), sp("primary", 7.0, True, -1)
        R = [road("n1", [(-80, 7), (0, 7)], cw1), road("n2", [(0, 7), (80, 7)], cw1),
             road("s1", [(-80, -7), (0, -7)], cw2), road("s2", [(0, -7), (80, -7)], cw2),
             road("x0", [(0, -7), (0, 7)], TERT), road("xs", [(0, -7), (0, -80)], TERT),
             road("xn", [(0, 7), (0, 80)], TERT)]
        self.assertEqual(summary(R)["stops"], ["n1", "s2", "xn", "xs"])


class SlipTests(unittest.TestCase):
    def slip_split(self):
        return [road("w1", [(-80, 0), (-30, 0)], MAIN), road("w2", [(-30, 0), (0, 0)], MAIN),
                road("e", [(0, 0), (80, 0)], MAIN), road("s2", [(0, 0), (0, -30)], TERT),
                road("s1", [(0, -30), (0, -80)], TERT), road("slip", SLIP_PTS, SLIP),
                road("n", [(0, 0), (0, 80)], TERT)]

    def slip_unsplit(self):
        return [road("w", [(-80, 0), (-30, 0), (0, 0)], MAIN), road("e", [(0, 0), (80, 0)], MAIN),
                road("s", [(0, 0), (0, -30), (0, -80)], TERT), road("slip", SLIP_PTS, SLIP),
                road("n", [(0, 0), (0, 80)], TERT)]

    def test_slip_island_split(self):
        isl = islands_of(self.slip_split())
        self.assertEqual(len(isl), 1)
        self.assertTrue({"w2", "s2", "slip"} <= set(isl[0][0]))

    def test_slip_island_unsplit(self):
        """Phase 2, R2-06: a slip lane on a road without split junctions still makes an island."""
        isl = islands_of(self.slip_unsplit())
        self.assertEqual(len(isl), 1)
        self.assertIn("slip", isl[0][0])

    @unittest.expectedFailure
    def test_slip_island_has_curb(self):
        """Phase 5, R2-06: the slip island has a curb ring."""
        self.assertTrue(islands_of(self.slip_split())[0][1])

    def test_slip_arms_keep_stop_lines(self):
        """Phase 2, R2-08: the arms next to a slip lane keep their stop lines."""
        self.assertTrue({"w2", "s2"} <= set(summary(self.slip_split())["line_cuts"]))


class BendTests(unittest.TestCase):
    def bend(self, deg):
        a = np.radians(deg)
        return [road("a", [(-60, 0), (0, 0)], TERT), road("b", [(0, 0), (60 * np.cos(a), 60 * np.sin(a))], TERT)]

    def test_bend_20(self):
        self.assertEqual(summary(self.bend(20))["corners"], 0)

    def test_bend_no_corner_at_split(self):
        """Phase 2, R2-10: a bend of 35 to 60 degrees is no junction and gets no corner."""
        for deg in (35, 45, 60):
            with self.subTest(deg=deg):
                self.assertEqual(summary(self.bend(deg))["corners"], 0)

    def test_width_change(self):
        R = [road("a", [(-60, 0), (0, 0)], sp("tertiary", 10.0)), road("b", [(0, 0), (60, 0)], sp("tertiary", 7.0))]
        self.assertEqual(summary(R)["corners"], 0)

    @unittest.expectedFailure
    def test_mixed_sidewalks_get_corners(self):
        """Phase 5, R2-04: a 4-way junction with sidewalks on some arms gets its four corners."""
        R = [road("a", [(-60, 0), (0, 0)], TERT), road("b", [(0, 0), (60, 0)], TERT),
             road("c", [(0, 0), (0, -60)], sp("residential", 6.0)), road("d", [(0, 0), (0, 60)], sp("residential", 6.0))]
        self.assertEqual(summary(R, has_walk=lambda s: s.cls != "residential")["corners"], 4)


class KreuzFixtureTests(unittest.TestCase):
    def test_k1_round_trip(self):
        r = kreuzfixtures.load("K1")
        self.assertGreaterEqual(sum(len(f) for f in r.roads), 1)
        self.assertGreaterEqual(sum(len(f) for f in r.infra), 1)
        self.assertIn("lane_tags", r.sources)


if __name__ == "__main__":
    unittest.main()
