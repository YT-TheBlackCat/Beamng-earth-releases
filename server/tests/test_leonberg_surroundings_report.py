"""tools/leonberg_surroundings_report.py: the pure helpers (the length in the radius, the walk and prop summaries, the
barrier kinds, the gap sentences, the markdown)."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))
spec = importlib.util.spec_from_file_location("leonberg_surroundings_report", ROOT / "tools" / "leonberg_surroundings_report.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

import fences  # noqa: E402
import sidewalks  # noqa: E402


def _summary(walks=None, props=None, mapped=None, **extra):
    """A summary dict as collect() builds it (only the keys the helpers read); walks = the by_source counts."""
    m = {"sidewalk_tags": {}, "footway_sidewalk": {"n": 0, "m": 0.0}, "footways": {}, "barriers": {}, "points": {}}
    m.update(mapped or {})
    s = {"walks": {"by_source": walks or {}}, "props": props or {}, "mapped": m}
    s.update(extra)
    return s


class LengthWithinTest(unittest.TestCase):
    def test_line_inside_the_circle(self):
        self.assertAlmostEqual(R.length_within([[-100.0, 0.0], [100.0, 0.0]], 50.0), 100.0, delta=1.0)

    def test_line_half_inside(self):
        self.assertAlmostEqual(R.length_within([[0.0, 0.0], [200.0, 0.0]], 50.0), 50.0, delta=1.0)

    def test_outside_and_degenerate(self):
        self.assertEqual(R.length_within([[500.0, 0.0], [600.0, 0.0]], 50.0), 0.0)
        self.assertEqual(R.length_within([[1.0, 1.0]], 50.0), 0.0)
        self.assertEqual(R.length_within(np.zeros((0, 2)), 50.0), 0.0)


class SummariseWalksTest(unittest.TestCase):
    def setUp(self):
        self.walks = [
            {"pts": [[-10.0, 0.0], [10.0, 0.0]], "w": 2.0, "mapped": True, "surface": "light_asphalt", "height": 0.1,
             "europe": True},
            {"pts": [[0.0, 20.0], [0.0, 40.0]], "w": 1.5, "surface": None, "median": True},
            {"disc": {"ring": [[0, 0], [1, 0], [0, 1]]}, "pts": [[0.0, 0.0], [1.0, 0.0]]},
            {"corner": {"arc": [[1.0, 1.0], [2.0, 2.0]]}},
            {"corner": {"arc": [[400.0, 400.0], [401.0, 401.0]]}},
            {"pts": [[500.0, 0.0], [600.0, 0.0]], "w": 3.0},
        ]
        self.s = R.summarise_walks(self.walks, 150.0)

    def test_runs_and_corners(self):
        self.assertEqual(self.s["n_runs"], 2)
        self.assertEqual(self.s["corners"], 1)
        self.assertAlmostEqual(self.s["length_m"], 40.0, delta=1.0)

    def test_source_and_surface(self):
        self.assertAlmostEqual(self.s["by_source"]["gemappt"], 20.0, delta=1.0)
        self.assertAlmostEqual(self.s["by_source"]["geschätzt"], 20.0, delta=1.0)
        self.assertIn("us_slab", self.s["by_surface"])
        self.assertIn("light_asphalt", self.s["by_surface"])

    def test_flags_and_stats(self):
        self.assertAlmostEqual(self.s["europe_m"], 20.0, delta=1.0)
        self.assertAlmostEqual(self.s["median_m"], 20.0, delta=1.0)
        self.assertEqual(self.s["detached_m"], 0.0)
        self.assertEqual(self.s["width_m"]["min"], 1.5)
        self.assertEqual(self.s["width_m"]["max"], 2.0)
        self.assertEqual(self.s["height_m"]["min"], 0.1)
        self.assertEqual(self.s["height_m"]["max"], sidewalks.CURB_H)

    def test_empty(self):
        s = R.summarise_walks([], 150.0)
        self.assertEqual(s["n_runs"], 0)
        self.assertIsNone(s["width_m"])


class SummarisePropsTest(unittest.TestCase):
    def test_radius_and_counts(self):
        rows = [["noisewall", 10.0, 0.0, 0.0, 0.0, 1.0, 2.5, 1], ["noisewall", 500.0, 0.0, 0.0, 0.0, 1.0, 9.0, 1],
                ["chainlink", 0.0, 10.0, 0.0, 0.0, 1.0, 1.76, 1], ["streetlight", 5.0, 5.0, 0.0, 0.0, 1.0, 8.0, 1]]
        s = R.summarise_props(rows, 50.0)
        self.assertEqual(s["noisewall"], 1)
        self.assertEqual(s["chainlink"], 1)
        self.assertEqual(s["streetlight"], 1)
        self.assertAlmostEqual(s["noisewall_m"], fences.WALL_LEN)
        self.assertEqual(s["noisewall_heights"], [2.5])
        self.assertAlmostEqual(s["chainlink_m"], fences.CHAIN_LEN)

    def test_no_rows(self):
        s = R.summarise_props(None, 50.0)
        self.assertEqual(s["noisewall_m"], 0)
        self.assertEqual(s["noisewall_heights"], [])


class BarrierKindTest(unittest.TestCase):
    def test_branches(self):
        self.assertEqual(R.barrier_kind("noise_barrier", {}), "noise_barrier")
        self.assertEqual(R.barrier_kind("wall", {"barrier": "noise_barrier"}), "noise_barrier")
        self.assertEqual(R.barrier_kind("fence", {"fence_type": "wood"}), "fence:wood")
        self.assertEqual(R.barrier_kind("fence", {}), "fence:?")
        self.assertEqual(R.barrier_kind("", {"barrier": "fence"}), "fence:?")
        self.assertEqual(R.barrier_kind("wall", {}), "wall:?")
        self.assertEqual(R.barrier_kind("", {"barrier": "wall", "wall": "brick"}), "wall:brick")
        self.assertEqual(R.barrier_kind("", {"barrier": "hedge"}), "hedge")
        self.assertEqual(R.barrier_kind("guard_rail", {}), "guard_rail")
        self.assertEqual(R.barrier_kind("retaining_wall", {}), "retaining_wall")
        self.assertEqual(R.barrier_kind("bollard", {}), "bollard")
        self.assertEqual(R.barrier_kind("", {"barrier": "bollard"}), "barrier:bollard")
        self.assertIsNone(R.barrier_kind("", {}))
        self.assertIsNone(R.barrier_kind(None, None))


class GapsTest(unittest.TestCase):
    def test_unbuilt_barrier_and_lamps(self):
        s = _summary(mapped={"barriers": {"fence:wood": {"n": 2, "m": 120.0, "m_unbuilt": 120.0, "heights": [],
                                                       "built_as": {"nicht gebaut": 2}}},
                             "points": {"street_lamp": 3}},
                     props={"streetlight": 1})
        g = R.gaps(s)
        self.assertIn("fence:wood 120 m gemappt, nicht gebaut", g)
        self.assertTrue(any(x.startswith("street_lamp: 3") for x in g))

    def test_noise_wall_mismatch_and_footway(self):
        s = _summary(mapped={"barriers": {"noise_barrier": {"n": 1, "m": 100.0, "m_unbuilt": 0.0, "heights": [3.0],
                                                           "built_as": {"noisewall": 1}}},
                             "footway_sidewalk": {"n": 1, "m": 30.0}},
                     props={"noisewall_m": 40.0})
        g = R.gaps(s)
        self.assertIn("noise_barrier gemappt 100 m, gebaut 40 m", g)
        self.assertTrue(any(x.startswith("footway=sidewalk 30 m") for x in g))

    def test_no_gap_when_equal(self):
        s = _summary(mapped={"barriers": {"noise_barrier": {"n": 1, "m": 40.0, "m_unbuilt": 0.0, "heights": [],
                                                           "built_as": {"noisewall": 1}}}, "points": {"street_lamp": 2}},
                     props={"noisewall_m": 40.0, "streetlight": 2}, walks={"gemappt": 5.0})
        self.assertEqual(R.gaps(s), [])


class MarkdownTest(unittest.TestCase):
    def test_contains_spot_and_sections(self):
        s = _summary(mapped={"sidewalk_tags": {"no": {"n": 1, "m": 12.0, "classes": {"primary": 12.0}}}})
        s["walks"] = R.summarise_walks([], 150.0)
        s["props"] = R.summarise_props([], 150.0)
        s["gaps"] = R.gaps(s)
        md = R.markdown(("U1", "Marktplatz", 48.8014, 9.0152), False, s)
        self.assertIsInstance(md, str)
        self.assertIn("U1", md)
        self.assertIn("Gehwege erzeugt", md)
        self.assertIn("sidewalk=no", md)
        self.assertIn("Lücken", md)


if __name__ == "__main__":
    unittest.main()
