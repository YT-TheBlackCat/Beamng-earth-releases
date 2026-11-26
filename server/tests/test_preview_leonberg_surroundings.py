"""tools/preview_leonberg_surroundings.py: the pure helpers (symbols, surface colours, prop and walk filters)."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("preview_leonberg_surroundings",
                                              ROOT / "tools" / "preview_leonberg_surroundings.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


class SymbolTest(unittest.TestCase):
    def test_groups(self):
        self.assertEqual(P.symbol_for("streetlight"), ("lamp", P.YELLOW))
        self.assertEqual(P.symbol_for("pole"), ("lamp", P.YELLOW))
        self.assertEqual(P.symbol_for("deciduous"), ("tree", P.GREEN))
        self.assertEqual(P.symbol_for("noisewall"), ("wall", P.RED))
        self.assertEqual(P.symbol_for("chainlink")[0], "chain")
        self.assertEqual(P.symbol_for("something_unknown"), ("other", P.GREY))


class SurfaceTest(unittest.TestCase):
    def test_colours(self):
        self.assertTrue(P.surface_colour("asphalt"))
        self.assertTrue(P.surface_colour(None))
        self.assertNotEqual(P.surface_colour("pavers"), P.surface_colour("asphalt"))
        self.assertEqual(P.surface_colour(None), P.surface_colour("us_slab"))


class PropsTest(unittest.TestCase):
    def test_radius_filter(self):
        rows = [["streetlight", 10.0, 0.0, 0.0, 0.0, 1.0, 9.0, 0.0],
                ["streetlight", 200.0, 0.0, 0.0, 0.0, 1.0, 9.0, 0.0]]
        out = P.props_points(rows, 150)
        self.assertEqual([(g, x) for g, x, _, _ in out], [("streetlight", 10.0)])

    def test_short_rows_skipped(self):
        self.assertEqual(P.props_points([["streetlight", 1.0]], 150), [])


class WalksTest(unittest.TestCase):
    def test_disc_and_corner_skipped(self):
        walks = [{"disc": True, "pts": [[0, 0], [1, 1]]},
                 {"corner": {"arc": [[0, 0], [1, 1]]}},
                 {"pts": [[0, 0, 0], [5, 0, 0]], "w": 1.8, "surface": "concrete"}]
        out = P.walks_lines(walks)
        self.assertEqual(len(out), 1)
        xy, width, surface = out[0]
        self.assertEqual(xy, [(0.0, 0.0), (5.0, 0.0)])
        self.assertEqual(width, 1.8)
        self.assertEqual(surface, "concrete")


if __name__ == "__main__":
    unittest.main()
