"""Unit tests for the review fixes B03-B06 in grading.py (tunnel mouth plans, slip-road links, _replan lengths)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import grading  # noqa: E402
import roads  # noqa: E402


def _sec(fid, length, kind=1, cls="primary"):
    """A covered tunnel section (fid, spec, xy, z, kind) running along x for `length` metres."""
    spec = roads.Spec(7.0, "paved", False, cls)
    xy = np.array([[0.0, 0.0], [length, 0.0]])
    z = np.array([10.0, 10.0])
    return (fid, spec, xy, z, kind)


class MouthPlansTest(unittest.TestCase):
    def test_sections_of_one_tunnel_share_the_guard(self):
        # "a.0#0" and "a.0#1" are one tunnel: the inner ends get no mouth plan, only the outer ones do
        secs = [_sec("a.0#0", 20.0), _sec("a.0#1", 20.0)]
        out = grading._Planner._mouth_plans(secs)
        fids = sorted(f for f, _p in out)
        self.assertEqual(len(out), 2, fids)
        self.assertTrue(fids[0].startswith("a.0#0"), fids)
        self.assertTrue(fids[1].startswith("a.0#1"), fids)

    def test_short_section_one_mouth_no_duplicate_vertex(self):
        # 2.5 m < 2 * MOUTH_PLAN_M: the whole section is the start mouth's plan, once
        secs = [_sec("b.0#0", 2.5)]
        out = grading._Planner._mouth_plans(secs)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0][1].xy), 2)

    def test_very_short_section_gets_no_plan(self):
        secs = [_sec("c.0#0", 0.5)]
        self.assertEqual(grading._Planner._mouth_plans(secs), [])


class RoadRankLinkTest(unittest.TestCase):
    def test_link_flag_ranks_just_under_class(self):
        plain = roads.Spec(7.0, "paved", False, "primary")
        link = plain._replace(link=True)
        self.assertLess(grading.road_rank(link), grading.road_rank(plain))


class ReplanTest(unittest.TestCase):
    def _plan(self, n):
        xy = np.array([[float(i), 0.0] for i in range(n)])
        p = grading.Plan(xy, np.full(n, 10.0), 3.0)
        p.rid, p.max_wall = 1, None
        p.bank = np.ones(n)
        return p

    def test_different_length_drops_bank(self):
        p = self._plan(3)
        q = grading._replan(p, np.array([[float(i), 0.0] for i in range(5)]), np.full(5, 10.0))
        self.assertIsNone(q.bank)

    def test_same_length_keeps_bank(self):
        p = self._plan(3)
        q = grading._replan(p, p.xy.copy(), np.full(3, 11.0))
        self.assertIsNotNone(q.bank)


if __name__ == "__main__":
    unittest.main()
