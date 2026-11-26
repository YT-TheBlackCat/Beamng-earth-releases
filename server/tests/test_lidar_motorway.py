"""On lidar elevation the motorway and trunk grading follows the lidar: _Planner._lidar (True where the DEM is not coarse,
the same decision _ungraded uses), and Grader.deck_ground leaves the base alone there (the DTM has the bridges out)."""
import os
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import grading  # noqa: E402


class FakeDem:
    def __init__(self, coarse, raises=False):
        self.coarse, self.raises, self.calls = coarse, raises, []

    def coarse_at(self, lat, lon):
        self.calls.append((lat, lon))
        if self.raises:
            raise RuntimeError("no tile")
        return self.coarse


class FakeKey:
    def bounds(self):
        return (8.9, 48.7, 9.1, 48.9)      # (lon0, lat0, lon1, lat1)


def planner(dem=None, frame=None):
    p = object.__new__(grading._Planner)
    p.md = None
    p.dem = dem
    p.frame = frame or types.SimpleNamespace(lat0=48.8, lon0=9.0)
    return p


class LidarPlannerTest(unittest.TestCase):
    def test_lidar_cell_is_lidar(self):
        self.assertTrue(planner(FakeDem(False))._lidar(FakeKey()))

    def test_coarse_cell_is_not_lidar(self):
        self.assertFalse(planner(FakeDem(True))._lidar(FakeKey()))

    def test_no_dem_is_not_lidar(self):
        self.assertFalse(planner(None)._lidar(FakeKey()))
        self.assertFalse(planner(None)._lidar())

    def test_region_without_key_uses_frame_origin(self):
        dem = FakeDem(False)
        self.assertTrue(planner(dem)._lidar())
        self.assertEqual(dem.calls, [(48.8, 9.0)])
        self.assertFalse(planner(FakeDem(True))._lidar())

    def test_exception_reads_as_not_coarse_as_in_ungraded(self):
        self.assertTrue(planner(FakeDem(True, raises=True))._lidar(FakeKey()))

    def test_memoised_per_key(self):
        dem = FakeDem(False)
        p = planner(dem)
        key = FakeKey()
        self.assertTrue(p._lidar(key))
        self.assertTrue(p._lidar(key))
        self.assertEqual(len(dem.calls), 1)

    def test_ungraded_follows_lidar(self):
        self.assertEqual(planner(FakeDem(False))._ungraded(FakeKey()), grading.UNGRADED | grading.LIDAR_UNGRADED)
        self.assertEqual(planner(FakeDem(True))._ungraded(FakeKey()), grading.UNGRADED)

    def test_lidar_cw_constants_default(self):
        self.assertEqual(grading.LIDAR_CW_MEDIAN_M, 8.0)
        self.assertEqual(grading.LIDAR_CW_SIGMA_M, 6.0)
        self.assertEqual(grading.LIDAR_CW_GROUND_W, 1.0)


class DeckGroundOnLidarTest(unittest.TestCase):
    def grader(self, dem):
        g = object.__new__(grading.Grader)
        g.md = None
        g.store = types.SimpleNamespace(dem=dem)
        return g

    def test_lidar_returns_base_unchanged(self):
        base = np.array([3.0, 4.5, 6.0])
        got = self.grader(FakeDem(False)).deck_ground(np.array([48.8, 48.8, 48.81]),
                                                      np.array([9.0, 9.0, 9.01]), base)
        np.testing.assert_array_equal(got, base)

    def test_lidar_base_is_not_lowered_even_with_planners_present(self):
        g = self.grader(FakeDem(False))
        g.store.planners_of = lambda la, lo: (_ for _ in ()).throw(AssertionError("must not read decks"))
        base = np.array([10.0, 11.0])
        np.testing.assert_array_equal(g.deck_ground(np.array([48.8, 48.81]), np.array([9.0, 9.01]), base), base)


class DeckGroundInitTest(unittest.TestCase):
    def test_lidar_helper_is_false_without_dem(self):
        self.assertFalse(grading._lidar_at(None, 48.8, 9.0))

    def test_lidar_helper_matches_coarse_flag(self):
        self.assertTrue(grading._lidar_at(FakeDem(False), 48.8, 9.0))
        self.assertFalse(grading._lidar_at(FakeDem(True), 48.8, 9.0))
        self.assertFalse(grading._lidar_at(object(), 48.8, 9.0))


if __name__ == "__main__":
    unittest.main()
