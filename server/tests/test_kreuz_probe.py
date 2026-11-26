"""The junction probe's core (Phase 0, Brief C3, tools/leonberg_junction_core.py) on a 4-way round Schrobenhausen."""
import os
os.environ.setdefault("BNE_REGIONAL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                                                       "regional"))     # (PR #505's data: the tests' slice)
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tools"))

import borderfixtures  # noqa: E402
import leonberg_junction_core as core  # noqa: E402
import numpy as np  # noqa: E402
import test_navgraph as tn  # noqa: E402
from test_roads import features, rules  # noqa: E402

CLASSES = {"w": "secondary", "e": "secondary", "n": "residential", "s": "residential"}


def setUpModule():
    borderfixtures.require()        # (no border data: the German tests would pass on anything)


def four_way():
    """The 4-way's roads in Germany (de_graph's features and rules), the junction at the DE frame's origin."""
    lines, classes, con = tn.junction(CLASSES)
    f = tn.in_germany(features(lines, classes))
    f.rules["connectors"] = rules(list(con))
    f.rules["oneway"] = rules([], "float")
    f.rules["max_speed_kmh"] = rules([], "float")
    f.rules["subclass"] = rules([])
    return f


def spot():
    lat, lon, _ = tn.DE_FRAME.enu_to_geodetic(np.array(0.0), np.array(0.0), 0.0)
    return float(lat), float(lon)


class FrameTest(unittest.TestCase):
    def test_spot_at_128_in_first_tile(self):
        frame, origin = core.build_frame(*spot())
        np.testing.assert_allclose(origin, (128.0, 128.0), atol=1e-2)   # (geodetic round trip: ~3 mm)
        self.assertEqual(core.tiles_for(origin, 80), [(0, 0)])
        self.assertEqual(len(core.tiles_for(origin, 200)), 9)


class JunctionProbeTest(unittest.TestCase):
    def probe(self, signals):
        frame, origin = core.build_frame(*spot())
        r = core.run(frame, origin, 80, [four_way()], signals=signals, urban_fn=tn.town, de_on=True)
        js = core.junction_table(r)
        return r, js, core.graph_table(r, js)

    def test_signalled_junction(self):
        frame, origin = core.build_frame(*spot())
        sig = np.array([[origin[0] + 3.0, origin[1] + 3.0]])          # a mapped signal 3 m beside the node
        r, js, gt = self.probe(sig)
        self.assertEqual(len(js), 1, [j["id"] for j in js])
        j = js[0]
        self.assertEqual(len(j["approaches"]), 4)
        self.assertTrue(all(ap["regulation"] == "signal" for ap in j["approaches"]),
                        [ap["regulation"] for ap in j["approaches"]])
        for ap in j["approaches"]:
            if ap["inbound"]:
                self.assertIsNotNone(ap["mast"], ap["fid"])
                self.assertGreater(ap["mast"]["d"], 0.0)
                self.assertIsNotNone(ap["ai"], ap["fid"])
                self.assertGreater(ap["ai"]["d"], 0.0)
        self.assertIsNotNone(j["graph_node"])
        self.assertEqual(len(gt), 1)
        self.assertEqual(gt[0]["controller"], "signal")
        self.assertEqual(gt[0]["junction"], j["id"])

    def test_unsignalled_junction_has_no_lights(self):
        _, js, _ = self.probe(np.zeros((0, 2)))
        self.assertEqual(len(js), 1)
        for ap in js[0]["approaches"]:
            self.assertNotEqual(ap["regulation"], "signal", ap["fid"])
            self.assertIsNone(ap["mast"], ap["fid"])

    def test_model_fields(self):
        """JSON schema 2: the junction model decides (decided_by "model"); each approach carries its group, role,
        d_stop, link and why, the inbound ones a d_stop."""
        frame, origin = core.build_frame(*spot())
        _, js, _ = self.probe(np.array([[origin[0] + 3.0, origin[1] + 3.0]]))
        j = js[0]
        self.assertEqual(j["decided_by"], "model")
        self.assertEqual(j["cluster"], [])
        for ap in j["approaches"]:
            for key in ("group", "role", "d_stop", "link", "why"):
                self.assertIn(key, ap)
            self.assertEqual(ap["group"], j["approaches"][0]["group"])
            self.assertFalse(ap["link"], ap["fid"])
            if ap["inbound"]:
                self.assertIsNotNone(ap["d_stop"], ap["fid"])


if __name__ == "__main__":
    unittest.main()
