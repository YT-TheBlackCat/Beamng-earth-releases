"""The junction probe on the captured round junctions K1 and K2 (Phase 0, Brief C5; fixtures/kreuz_K1.npz, K2.npz).

The probe runs offline on each fixture once (radius 80 m, flat, no widths); xfail tests are known faults with their
phase in the docstring (Phase N removes the decorator).
"""
import math
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("BNE_REGIONAL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                                                       "regional"))
os.environ.setdefault("BNGE_PLANCACHE", "off")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tools"))

import borderfixtures  # noqa: E402
import leonberg_junction_core as core  # noqa: E402
import leonberg_junction_probe as probe  # noqa: E402

RADIUS = 80.0
_RESULTS = {}


def setUpModule():
    borderfixtures.require()        # (no border data: the German junction rules would not be the ones under test)


def result(name):
    """The probe's result for fixture name, run once per test process."""
    if name not in _RESULTS:
        lat, lon, roads, infra, routes, urban_fn, widths, sources = probe.fixture_inputs(name)
        args = SimpleNamespace(radius=RADIUS, grid_shift=(0.0, 0.0), no_de=False)
        meta = {"tag": "test", "time": "", "git": None, "argv": [], "settings": [], "de_on": True}
        _RESULTS[name] = probe.probe_spot(name, lat, lon, (roads, infra, routes, urban_fn, widths), args, sources,
                                          meta)
    return _RESULTS[name]


def approaches(res):
    return [(j, a) for j in res["junctions"] for a in j["approaches"]]


class KreuzFixtureTests(unittest.TestCase):
    def test_runs_and_finds_signal_junctions(self):
        res = result("K1")
        sig = [j for j in res["junctions"] if math.hypot(j["x"], j["y"]) <= 40.0
               and any(a["regulation"] == "signal" for a in j["approaches"])]
        self.assertGreaterEqual(len(sig), 2)

    def test_no_us_stop_signs(self):
        for name in ("K1", "K2"):
            with self.subTest(fixture=name):
                res = result(name)
                self.assertEqual(res["summary"]["us_stop_signs"], 0)
                bad = [j["id"] for j in res["junctions"] for a in j["approaches"]
                       if j["decided_by"] in ("junction_controls", "model") and a["regulation"] == "stop"]
                self.assertEqual(bad, [])

    def test_graph_signals_belong_to_junctions(self):
        for name in ("K1", "K2"):
            with self.subTest(fixture=name):
                res = result(name)
                bad = [g["node"] for g in res["graph"] if g["controller"] == "signal"
                       and math.hypot(g["x"], g["y"]) <= RADIUS and g["junction"] is None]
                self.assertEqual(bad, [])

    def test_every_mast_has_ai(self):
        """Phase 2, R3-04: every approach with a signal mast has its AI stop point."""
        bad = [a["fid"] for _, a in approaches(result("K2")) if a["mast"] and not a["ai"]]
        self.assertEqual(bad, [])

    def test_signal_cluster_one_controller(self):
        """Phase 2, R3-08: signal graph nodes closer than CLUSTER_M form one cluster with one controller, so no two
        of them lie that close."""
        sig = [g for g in result("K2")["graph"] if g["controller"] == "signal"]
        close = [(g["node"], h["node"]) for i, g in enumerate(sig) for h in sig[i + 1:]
                 if math.hypot(g["x"] - h["x"], g["y"] - h["y"]) < core.CLUSTER_M]
        self.assertEqual(close, [])

    def test_stop_points_agree(self):
        """Phase 2, R3-10: an inbound signal approach's mast and AI stop point lie within 1.0 m of each other."""
        for name in ("K1", "K2"):
            bad = [(a["fid"], a["end"]) for _, a in approaches(result(name))
                   if a["inbound"] and a["regulation"] == "signal" and a["mast"] and a["ai"]
                   and abs(a["mast"]["d"] - a["ai"]["d"]) > 1.0]
            self.assertEqual(bad, [], name)

    def test_every_signal_approach_has_stop_line(self):
        """Phase 3, R1-04/R3-05: every inbound signal approach has a stop line."""
        bad = [(a["fid"], a["end"]) for _, a in approaches(result("K1"))
               if a["inbound"] and a["regulation"] == "signal" and not a["stop_line"]]
        self.assertEqual(bad, [])

    @unittest.expectedFailure
    def test_mapped_give_way_arm_no_signal(self):
        """Phase 3, R3-02: an arm the map marks give_way is no signal approach."""
        gw = [a for _, a in approaches(result("K2")) if a["mapped_sign"] == "give_way"]
        self.assertGreaterEqual(len(gw), 1)
        self.assertEqual([a["fid"] for a in gw if a["regulation"] == "signal"], [])

    def test_report_and_diff(self):
        res = result("K1")
        head = {"tag": "test", "time": "", "git": None, "settings": []}
        md = probe.report_md([res], head)
        self.assertIn("K1", md)
        self.assertIn("| Stelle | Knoten | Zufahrten", md)
        self.assertIn("| Knoten | Arm | Klasse", md)
        self.assertIn("keine Änderung", probe.diff_md([res], [res]))


if __name__ == "__main__":
    unittest.main()
