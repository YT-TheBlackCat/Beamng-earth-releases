"""tools/leonberg_elevation_probe.py: the pure helpers (metrics, colour mapping, spot parsing)."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("leonberg_elevation_probe", ROOT / "tools" / "leonberg_elevation_probe.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


class MetricsTest(unittest.TestCase):
    def test_step_is_an_edge(self):
        raw = np.zeros((41, 41))
        graded = raw.copy()
        graded[:, 20:] += 0.3                      # one artificial 0.3 m step, the raw DEM is flat
        m = P.metrics(raw, graded, 1.0)
        self.assertGreater(m["edge_count"], 0)
        self.assertAlmostEqual(m["max_abs_diff"], 0.3, places=6)
        self.assertLessEqual(len(m["edges_worst"]), 5)
        self.assertAlmostEqual(m["edges_worst"][0]["value"], 0.3, places=3)

    def test_identical_has_no_edges_and_unit_ratio(self):
        rng = np.random.default_rng(1)
        a = np.cumsum(rng.normal(0, 0.05, (41, 41)), axis=1)
        m = P.metrics(a, a.copy(), 1.0)
        self.assertEqual(m["edge_count"], 0)
        self.assertEqual(m["max_abs_diff"], 0.0)
        self.assertAlmostEqual(m["hp_ratio"], 1.0, places=6)

    def test_sine_on_flat_is_wavy(self):
        raw = np.zeros((81, 81))
        x = np.arange(81)
        graded = np.tile(0.05 * np.sin(2 * np.pi * x / 6.0), (81, 1))   # 6 m waves
        m = P.metrics(raw, graded, 1.0)
        self.assertEqual(m["hp_rms_raw"], 0.0)
        self.assertIsNone(m["hp_ratio"])            # flat raw: the ratio is infinite, reported as None
        m2 = P.metrics(raw + 0.01 * np.random.default_rng(2).normal(size=raw.shape), graded, 1.0)
        self.assertGreater(m2["hp_ratio"], 2.0)

    def test_positions_are_centred_and_north_up(self):
        raw = np.zeros((41, 41))
        graded = raw.copy()
        graded[0, 40] += 0.5                       # top-right corner: north +20 m, east +20 m
        m = P.metrics(raw, graded, 1.0)
        self.assertEqual(m["edges_worst"][0]["east"], 20.0)
        self.assertEqual(m["edges_worst"][0]["north"], 20.0)


class ColourTest(unittest.TestCase):
    def test_diverging_shape_and_range(self):
        d = np.linspace(-3, 3, 12).reshape(3, 4)
        rgb = P.diverging(d)
        self.assertEqual(rgb.shape, (3, 4, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertTrue((rgb[0, 0, 2] > rgb[0, 0, 0]))     # strongly negative is blue
        self.assertTrue((rgb[2, 3, 0] > rgb[2, 3, 2]))     # strongly positive is red

    def test_hillshade_flat_and_grey_rgb(self):
        z = np.zeros((9, 9))
        g = P.hillshade(z, 1.0)
        self.assertEqual(g.dtype, np.uint8)
        self.assertEqual(g.shape, (9, 9))
        rgb = P.to_rgb(g)
        self.assertEqual(rgb.shape, (9, 9, 3))
        self.assertEqual(int(rgb[4, 4, 0]), int(np.sin(np.radians(45)) * 255))


class SpotTest(unittest.TestCase):
    def test_parse(self):
        spots = P.parse_spots("A:48.5,9.0; B:48.6,9.1 ;")
        self.assertEqual(spots, [("A", 48.5, 9.0), ("B", 48.6, 9.1)])

    def test_grid_row0_is_north(self):
        X, Y = P.grid_xy(2.0, 1.0)
        self.assertEqual(X.shape, (5, 5))
        self.assertEqual(Y[0, 0], 2.0)
        self.assertEqual(Y[-1, 0], -2.0)


class BlobTest(unittest.TestCase):
    def test_two_separated_blobs(self):
        d = np.zeros((20, 20))
        d[2:4, 2:4] = 1.5          # 2x2 = 4 cells
        d[12:14, 10:13] = -2.0     # 2x3 = 6 cells, negative diff counts by |diff|
        d[17, 17] = 0.5            # below threshold: ignored
        out = P.blobs(d, 1.0)
        self.assertEqual([b["size"] for b in out], [6, 4])
        self.assertEqual(len(out[0]["rows"]), 6)
        self.assertTrue(set(out[1]["rows"].tolist()) == {2, 3})


class SetTest(unittest.TestCase):
    def test_parse_set_splits_at_first_eq_and_last_dot(self):
        self.assertEqual(P.parse_set("grading.SIGMA_M=3.0"), ("grading", "SIGMA_M", 3.0))
        mod, attr, val = P.parse_set('grading.CLASS_SIGMA_M={"secondary":4.0,"residential":4.0}')
        self.assertEqual((mod, attr), ("grading", "CLASS_SIGMA_M"))
        self.assertEqual(val, {"secondary": 4.0, "residential": 4.0})
        self.assertEqual(P.parse_set("a.b.c=2"), ("a.b", "c", 2))

    def test_parse_set_missing_eq_raises(self):
        with self.assertRaises(ValueError):
            P.parse_set("grading.SIGMA_M 3.0")


if __name__ == "__main__":
    unittest.main()
