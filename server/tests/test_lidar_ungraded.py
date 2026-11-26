"""Town and minor roads are ungraded on lidar elevation (grading._Planner._ungraded, LIDAR_UNGRADED): on coarse global
elevation (dem.coarse_at) or without a DEM only UNGRADED, as before; the switch BNE_LIDAR_UNGRADED=off gives UNGRADED."""
import os
import sys
import types
import unittest
import unittest.mock

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
    p.md = None                             # (no mapdata DEM: _ungraded reads self.md.dem as a fallback)
    p.dem = dem
    p.frame = frame or types.SimpleNamespace(lat0=48.8, lon0=9.0)
    return p


class UngradedOnLidarTest(unittest.TestCase):
    def test_lidar_cell_ungrades_town_and_minor_roads(self):
        got = planner(FakeDem(False))._ungraded(FakeKey())
        self.assertIn("residential", got)
        self.assertIn("raceway", got)
        self.assertNotIn("motorway", got)
        self.assertNotIn("trunk", got)
        self.assertEqual(got, grading.UNGRADED | grading.LIDAR_UNGRADED)

    def test_coarse_cell_keeps_only_ungraded(self):
        self.assertEqual(planner(FakeDem(True))._ungraded(FakeKey()), grading.UNGRADED)

    def test_no_dem_keeps_only_ungraded(self):
        self.assertEqual(planner(None)._ungraded(FakeKey()), grading.UNGRADED)
        self.assertEqual(planner(None)._ungraded(), grading.UNGRADED)

    def test_switch_off_keeps_only_ungraded(self):
        with unittest.mock.patch.object(grading, "LIDAR_UNGRADED_ON", False):
            self.assertEqual(planner(FakeDem(False))._ungraded(FakeKey()), grading.UNGRADED)
            self.assertEqual(planner(FakeDem(False))._ungraded(), grading.UNGRADED)

    def test_region_decides_from_frame_origin(self):
        dem = FakeDem(False)
        got = planner(dem)._ungraded()
        self.assertEqual(got, grading.UNGRADED | grading.LIDAR_UNGRADED)
        self.assertEqual(dem.calls, [(48.8, 9.0)])
        self.assertEqual(planner(FakeDem(True))._ungraded(), grading.UNGRADED)

    def test_exception_in_coarse_is_lidar_as_in_coarse_check(self):
        # (grading._coarse answers False on an exception: the cell counts as lidar)
        self.assertEqual(planner(FakeDem(True, raises=True))._ungraded(FakeKey()),
                         grading.UNGRADED | grading.LIDAR_UNGRADED)

    def test_result_is_memoised_per_key(self):
        dem = FakeDem(False)
        p = planner(dem)
        key = FakeKey()
        first = p._ungraded(key)
        self.assertIs(p._ungraded(key), first)
        self.assertEqual(len(dem.calls), 1)

    def test_class_sets(self):
        self.assertFalse(grading.LIDAR_UNGRADED & {"motorway", "trunk", "motorway_link", "raceway"})
        self.assertIn("busway", grading.LIDAR_UNGRADED)
        self.assertEqual(grading.UNGRADED, {"raceway"})


if __name__ == "__main__":
    unittest.main()
