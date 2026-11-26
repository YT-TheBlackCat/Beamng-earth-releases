"""Lidar keeps fade (grading._Planner._keep_weight): an ungraded road on lidar (a keep with a negative half width) keeps the
lidar on its carriageway and blends into the plan over UNGRADED_MARGIN_M, instead of a hard step at its edge. Raceways
(a positive half width) stay on _on_kept and _raceway_ground."""
import os
import sys
import threading
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import grading  # noqa: E402

LINE = np.array([[-50.0, 0.0], [50.0, 0.0]])     # a straight keep along the x axis


def planner(keeps):
    p = object.__new__(grading._Planner)
    p.keeps = keeps
    p.lock = threading.Lock()
    return p


class KeepWeightTest(unittest.TestCase):
    def test_weight_is_one_on_carriageway_zero_at_edge_and_monotone(self):
        # hw = -4: the carriageway's half width 2 + UNGRADED_MARGIN_M (2)
        self.assertEqual(grading.UNGRADED_MARGIN_M, 2.0)
        d = np.arange(0.0, 6.0 + 1e-9, 0.25)            # distance from the line, points (0, d)
        w = planner({"c": [(LINE, -4.0)]})._keep_weight(np.zeros_like(d), d)
        self.assertEqual(w.shape, d.shape)
        self.assertTrue(np.all(w[d <= 2.0] == 1.0))
        self.assertTrue(np.all(w[d >= 4.0] == 0.0))
        self.assertTrue(np.all(np.diff(w) <= 1e-12))

    def test_blend_height_second_difference_small_on_margin(self):
        # the blend of a plan 0.4 m under the lidar: its height change over the margin, samples 0.25 m apart
        d = np.arange(0.0, 6.0 + 1e-9, 0.25)
        w = planner({"c": [(LINE, -4.0)]})._keep_weight(np.zeros_like(d), d)
        z = 0.4 * w
        self.assertLess(float(np.max(np.abs(np.diff(z, 2)))), 0.12)

    def test_no_zone_near_is_none(self):
        self.assertIsNone(planner({})._keep_weight(np.array([0.0]), np.array([0.0])))
        self.assertIsNone(planner({"c": [(LINE, -4.0)]})._keep_weight(np.array([0.0]), np.array([100.0])))

    def test_raceway_zone_is_no_keep_weight(self):
        self.assertIsNone(planner({"c": [(LINE, 4.0)]})._keep_weight(np.array([0.0]), np.array([0.0])))

    def test_lidar_keeps_are_boxed_apart_from_raceways(self):
        other = LINE + np.array([0.0, 10.0])
        p = planner({"r": [(LINE, 4.0)], "l": [(other, -4.0)]})
        zones, bb = p._keep_boxes()
        self.assertEqual(len(zones), 1)
        self.assertEqual(zones[0][1], 4.0)
        self.assertEqual(bb.shape, (1, 4))
        lzones, lbb = p._lidar_keep_boxes()
        self.assertEqual(len(lzones), 1)
        self.assertEqual(lzones[0][1], 4.0)
        self.assertEqual(lbb.shape, (1, 4))
        self.assertIs(p._keep_boxes()[1], bb)           # memoised while the keeps stay

    def test_raceway_keep_still_on_kept_not_lidar_keep(self):
        # the raceway keep marks its points (_on_kept, then _raceway_ground); the lidar keep's points are not marked
        p = planner({"r": [(LINE, 4.0)], "l": [(LINE + np.array([0.0, 10.0]), -4.0)]})
        mask = p._on_kept(np.array([0.0, 0.0]), np.array([0.0, 10.0]))
        self.assertTrue(np.array_equal(mask, np.array([True, False])))
        self.assertIsNone(planner({"l": [(LINE, -4.0)]})._on_kept(np.array([0.0]), np.array([0.0])))


if __name__ == "__main__":
    unittest.main()
