"""Junction approaches, Phase 2 Brief B9 (junctionapproach.py 3.1-3.4): synthetic models from junctionmodel.build and
hand-set controls (designs.controls' answers: 2 = Z206, "signal", ...).
"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("BNE_REGIONAL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                                                       "regional"))
os.environ.setdefault("BNGE_PLANCACHE", "off")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tools"))

import junctionapproach as ja  # noqa: E402
import junctionmodel as jm  # noqa: E402
import roads  # noqa: E402


def spec(cls, w, oneway=False, d=1):
    return roads.Spec(w, "paved", True, cls, oneway, 2.0, d)


def row(fid, pts, s):
    return (fid, s, np.array([[x, y, 0.0] for x, y in pts], float), 9.0)


def cross(L=60.0):
    """A secondary road west-east, residential north-south (N and S are the stubs)."""
    main, side = spec("secondary", 9.0), spec("residential", 7.0)
    return [row("w", [(-L, 0), (0, 0)], main), row("e", [(0, 0), (L, 0)], main),
            row("n", [(0, 0), (0, L)], side), row("s", [(0, 0), (0, -L)], side)]


def junction_at(model, x, y):
    return next(k for k, j in model.junctions.items() if np.hypot(*(j.centre[:2] - [x, y])) < 3.0)


def decide_by(model, what_of):
    """ja.Decided of the model: every junction with the controls what_of(arm) gives for each of its arms."""
    d = ja.Decided()
    d.model = model
    for k, j in model.junctions.items():
        d[k] = (j.arm_tuples(), [(a.as_tuple(), what_of(a), -1) for a in j.arms])
    return d


def apps_by_fid(d, model, key):
    return {a.fid: d.apps[a.id] for a in model.junction(key).arms}


class TestApproaches(unittest.TestCase):

    def test_z206_stop_line_north_south_only(self):
        """1: N and S with Z206 get a Z294 at d_edge + 1,0; W and E (none) are through; N and S are end arms."""
        model = jm.build(cross())
        k = junction_at(model, 0, 0)
        d = decide_by(model, lambda a: 2 if a.fid in ("n", "s") else 0)
        ja.build(d, None, False)
        ap = apps_by_fid(d, model, k)
        self.assertEqual(ap["n"].regulation, "z206")
        self.assertAlmostEqual(ap["n"].d_stop, ap["n"].d_edge + 1.0, places=2)
        self.assertEqual(ap["n"].stop_line["kind"], "z294")
        self.assertEqual(ap["n"].role, "end")
        self.assertEqual(ap["s"].role, "end")
        for fid in ("w", "e"):
            self.assertEqual(ap[fid].regulation, "none")
            self.assertEqual(ap[fid].role, "through")
            self.assertIsNone(ap[fid].stop_line)

    def test_signal_mapped_crossing_pushes_the_station(self):
        """2: a signal at all four arms, a mapped signal crossing 3 m beyond d_edge on N: N's station lies behind it
        (crossing station + 1,5 + 2,0), with the mast and the AI stop at the same station."""
        model = jm.build(cross())
        k = junction_at(model, 0, 0)
        dn = next(a for a in model.junction(k).arms if a.fid == "n").d_edge
        nodes = {"de_crossings": [], "de_signal_crossings": np.array([[0.0, dn + 3.0]])}
        d = decide_by(model, lambda a: "signal")
        ja.build(d, nodes, False)
        ap = apps_by_fid(d, model, k)
        self.assertGreaterEqual(ap["n"].d_stop, round(dn + 3.0) + 3.5 - 1e-6)
        self.assertEqual(ap["n"].mast_d, ap["n"].d_stop)
        self.assertEqual(ap["n"].ai["d"], ap["n"].d_stop)
        self.assertEqual(len(d.groups), 1)

    def test_links_between_signals(self):
        """3: two signalised junctions 30 m apart are one group, their connecting arms link (no line, mast or AI);
        50 m apart: two groups, and the arms have their own approaches."""
        s = spec("primary", 9.0)

        def pair(gap):
            return [row("aw", [(-60, 0), (0, 0)], s), row("ab", [(0, 0), (gap, 0)], s),
                    row("an", [(0, 0), (0, 60)], s), row("as", [(0, 0), (0, -60)], s),
                    row("be", [(gap, 0), (gap + 60, 0)], s), row("bn", [(gap, 0), (gap, 60)], s),
                    row("bs", [(gap, 0), (gap, -60)], s)]
        model = jm.build(pair(30.0))
        ka, kb = junction_at(model, 0, 0), junction_at(model, 30, 0)
        d = decide_by(model, lambda a: "signal")
        ja.build(d, None, False)
        self.assertEqual(len(d.groups), 1)
        for key in (ka, kb):
            ap = apps_by_fid(d, model, key)["ab"]
            self.assertTrue(ap.link)
            self.assertEqual(ap.why["stop_line"], "link")
            self.assertIsNone(ap.stop_line)
            self.assertIsNone(ap.mast_d)
            self.assertIsNone(ap.ai)
        model = jm.build(pair(50.0))
        ka, kb = junction_at(model, 0, 0), junction_at(model, 50, 0)
        d = decide_by(model, lambda a: "signal")
        ja.build(d, None, False)
        self.assertEqual(len(d.groups), 2)
        self.assertFalse(apps_by_fid(d, model, ka)["ab"].link)
        self.assertFalse(apps_by_fid(d, model, kb)["ab"].link)

    def test_outbound_one_way_arm_has_no_line(self):
        """4: a one-way arm leaving the junction: why stop_line == outbound."""
        r = spec("residential", 7.0)
        model = jm.build([row("w", [(-60, 0), (0, 0)], r), row("n", [(0, 0), (0, 60)], r),
                          row("e", [(0, 0), (60, 0)], spec("residential", 6.0, True, 1))])
        k = junction_at(model, 0, 0)
        d = decide_by(model, lambda a: 0)
        ja.build(d, None, False)
        ap = apps_by_fid(d, model, k)
        self.assertFalse(ap["e"].inbound)
        self.assertEqual(ap["e"].why["stop_line"], "outbound")
        self.assertIsNone(ap["e"].d_stop)

    def test_side_cuts_only_end_arms_with_a_line(self):
        """5: side_cuts has entries for the end arms only, and t_in > t_out at a stop line."""
        model = jm.build(cross())
        k = junction_at(model, 0, 0)
        d = decide_by(model, lambda a: 2 if a.fid in ("n", "s") else 0)
        ja.build(d, None, False)
        cuts = ja.side_cuts(d)
        self.assertIn("n", cuts)
        self.assertIn("s", cuts)
        self.assertNotIn("w", cuts)
        self.assertNotIn("e", cuts)
        for X, u, t_in, t_out in cuts["n"]:
            self.assertGreater(t_in, t_out)
        self.assertEqual(len(apps_by_fid(d, model, k)), 4)


if __name__ == "__main__":
    unittest.main()
