"""tools/check_lod2_roofs.py: the pure helpers (lift, roof metrics) on a LoD2 house built by buildings.build."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CK = _load("check_lod2_roofs", ROOT / "tools" / "check_lod2_roofs.py")
TB = _load("test_buildings_lod2geom", ROOT / "server" / "tests" / "test_buildings_lod2geom.py")
import buildings as bld  # noqa: E402


class LiftExpected(unittest.TestCase):
    def test_flat_ground_has_no_lift(self):
        L = TB.lod2_spec().lod2                       # eave 6 m
        self.assertEqual(CK.lift_expected(L, 0.0, 0.0), 0.0)

    def test_slope_lifts_the_roof_by_the_height_difference_plus_two(self):
        L = TB.lod2_spec().lod2
        self.assertAlmostEqual(CK.lift_expected(L, 0.0, 6.0), 0.5)
        self.assertAlmostEqual(CK.lift_expected(L, 1.0, 7.0), 0.5)   # only gmax - gmin counts


class RoofMetrics(unittest.TestCase):
    def test_flat_ground_house_matches_its_lod2_model(self):
        spec = TB.lod2_spec()
        m = CK.roof_metrics(bld.build([(spec, 0.0, 0.0)]), spec, 0.0, 0.0)
        self.assertIsNotNone(m["ridge_mesh"])
        self.assertAlmostEqual(m["ridge_mesh"], 9.0, places=6)
        self.assertAlmostEqual(m["ridge_err"], 0.0, places=6)
        self.assertAlmostEqual(m["area_ratio"], 1.0, places=3)

    def test_no_roof_triangles_gives_none(self):
        spec = TB.lod2_spec()
        m = CK.roof_metrics(None, spec, 0.0, 0.0)
        self.assertIsNone(m["ridge_mesh"])
        self.assertIsNone(m["ridge_err"])
        self.assertEqual(m["area_ratio"], 0.0)       # no roof area: flagged as below 0.98


class Gaps(unittest.TestCase):
    def test_untaken_unit_in_radius_is_a_gap(self):
        class FakeGeoms:
            def __init__(self):
                self.n = 2
                self.cx = np.array([0.0, 500.0])
                self.cy = np.array([0.0, 0.0])
                self._taken = {0}
                self._units = [{"id": "A", "area": 50.0, "code": 1000}, {"id": "B", "area": 30.0, "code": 9999}]

            def unit(self, i):
                return self._units[i]

        out = CK.gaps([FakeGeoms()], 150.0)
        self.assertEqual(out, [])                     # B lies outside the radius, A is taken
        g = FakeGeoms()
        g._taken = set()
        out = CK.gaps([g], 150.0)
        self.assertEqual([r[0] for r in out], ["A"])


if __name__ == "__main__":
    unittest.main()
