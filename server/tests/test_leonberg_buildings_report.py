"""tools/leonberg_buildings_report.py: the pure helpers (the sources of a building's height and roof, the summary)."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("leonberg_buildings_report", ROOT / "tools" / "leonberg_buildings_report.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def _row(**kw):
    base = {"id": "x", "name": None, "cls": None, "footprint_src": "OSM", "height_src": "Karte", "roof_src": "Karte",
            "lod2_match": False, "lod2_shape": None, "built_roof": None, "built_parts": 1, "map_height": None,
            "lod2_height": None, "lod2_eave": None, "built_height": None, "facade_color": None, "roof_color": None, "year": None}
    base.update(kw)
    return base


class HeightSourceTest(unittest.TestCase):
    def test_branches(self):
        self.assertEqual(R.height_source(12.0, 3, 9.0), "Karte")
        self.assertEqual(R.height_source(None, 3, 9.0), "LoD2")
        self.assertEqual(R.height_source(None, 3, None), "Karte:Stockwerke")
        self.assertEqual(R.height_source(None, None, None), "geschätzt")

    def test_ml_map_height_gives_way_to_lod2(self):
        self.assertEqual(R.height_source(10.9, 3, 13.2, "Microsoft ML Buildings"), "LoD2")
        self.assertEqual(R.height_source(10.9, 3, None, "Microsoft ML Buildings"), "Karte")
        self.assertEqual(R.height_source(10.9, 3, 13.2, "Google Open Buildings"), "LoD2")
        self.assertEqual(R.height_source(12.0, 3, 9.0, "OpenStreetMap"), "Karte")

    def test_nan_and_zero_are_not_heights(self):
        self.assertEqual(R.height_source(float("nan"), np.nan, np.float64(0.0)), "geschätzt")
        self.assertEqual(R.height_source(np.float32(7.5), None, None), "Karte")
        self.assertEqual(R.height_source("12", None, None), "geschätzt")


class RoofSourceTest(unittest.TestCase):
    def test_map_wins(self):
        self.assertEqual(R.roof_source("gabled", 5000), "Karte")
        self.assertEqual(R.roof_source("  ", 3100), "LoD2")

    def test_lod2_code(self):
        self.assertEqual(R.roof_source(None, 3100), "LoD2")
        self.assertEqual(R.roof_source(None, 5000), "geschätzt")
        self.assertEqual(R.roof_source(None, None), "geschätzt")


class FootprintSourceTest(unittest.TestCase):
    def test_names(self):
        self.assertEqual(R.footprint_source("OpenStreetMap"), "OSM")
        self.assertEqual(R.footprint_source("Microsoft ML Buildings"), "ML")
        self.assertEqual(R.footprint_source("Google Open Buildings"), "ML")
        self.assertEqual(R.footprint_source(""), "?")
        self.assertEqual(R.footprint_source(None), "?")
        self.assertEqual(R.footprint_source("Other"), "Other")


class SummariseTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            _row(id="a", name="A", lod2_match=True, lod2_shape="gabled", built_roof="hipped", built_height=10.0,
                 lod2_height=8.0, lod2_eave=7.0, map_height=None, facade_color="#fff"),
            _row(id="b", name="B", lod2_match=True, lod2_shape="flat", built_roof="flat", built_height=20.0,
                 lod2_height=12.0, lod2_eave=11.0, footprint_src="ML", year=1950, built_parts=0),
            _row(id="c", name="C", lod2_match=False, built_height=5.0, lod2_height=None, roof_color="#000"),
        ]
        self.s = R.summarise(self.rows)

    def test_counts(self):
        self.assertEqual(self.s["n"], 3)
        self.assertEqual(self.s["lod2_matched"], 2)
        self.assertEqual(self.s["lod2_share"], 0.67)
        self.assertEqual(self.s["by_footprint"], {"OSM": 2, "ML": 1})
        self.assertEqual(self.s["roof_mismatch"], 1)
        self.assertEqual(self.s["not_built"], 1)
        self.assertEqual(self.s["with_colour"], 2)
        self.assertEqual(self.s["with_year"], 1)

    def test_dh_and_worst_order(self):
        # the wall (built) against the eave of the LoD2 roof: |10-7|, |20-11|
        self.assertEqual(self.s["dh_built_lod2_median"], 6.0)
        self.assertEqual(self.s["dh_built_lod2_max"], 9.0)
        self.assertEqual(self.s["worst"][0]["lod2_eave"], 11.0)
        self.assertEqual([w["id"] for w in self.s["worst"]], ["b", "a"])

    def test_empty(self):
        s = R.summarise([])
        self.assertEqual(s["n"], 0)
        self.assertEqual(s["lod2_share"], 0)
        self.assertIsNone(s["dh_built_lod2_median"])


class MarkdownTest(unittest.TestCase):
    def test_heading_and_status(self):
        s = R.summarise([_row(id="a", lod2_height=8.0, lod2_eave=7.0, built_height=10.0)])
        md = R.markdown(("T1", "Marktplatz", 48.8014, 9.0152), "3 Datensätze in 1 Kachel(n)", s)
        self.assertIn("## T1 Marktplatz (48.80140, 9.01520)", md)
        self.assertIn("LoD2: 3 Datensätze in 1 Kachel(n)", md)
        self.assertIn("| Kennzahl | Wert |", md)
        self.assertIn("Grösste Abweichung Traufhöhe gebaut zu LoD2 (m):", md)
        self.assertIn("| LoD2 Traufe |", md)
        self.assertIn("| a |  |  | None | 7.0 | 10.0 |", md)


if __name__ == "__main__":
    unittest.main()
