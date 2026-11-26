"""The diagnostics hooks (Phase 0, Brief C1): roads.build_tile(diag=), navgraph.build(diag=) and navgraph._signals(
skipped=). Without them nothing changes; with them the stages of each line, the roads without lines and the signal
approaches left out are reported."""
import os
os.environ.setdefault("BNE_REGIONAL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                                                       "regional"))     # (PR #505's data: the tests' slice)
import pickle
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import borderfixtures  # noqa: E402
import designs  # noqa: E402
import furniture  # noqa: E402
import numpy as np  # noqa: E402
import roads  # noqa: E402
import navgraph  # noqa: E402
import test_designs as td  # noqa: E402     (its classes are not collected here: only their helpers are used)
from test_navgraph import de_graph  # noqa: E402

DIAG_KEYS = {"roads", "corners", "crosswalks", "jlines", "jinternal", "teeth", "zebras", "de_ctl", "corner_cuts",
             "skipped", "lines", "model", "approaches", "side_cuts"}
STAGES = {"plan", "tile", "junction", "corner", "cut", "drawn"}


def setUpModule():
    borderfixtures.require()        # (no border data: the German tests would pass on anything)


class TileDiagTest(unittest.TestCase):
    """The 4-way of MarkingsTest (Schrobenhausen, secondary east and west, residential north and south), in Germany."""

    def setUp(self):
        self.m = td.MarkingsTest()
        self.feats = self.m.layer(self.m.STRETCH)

    def tile(self, diag=None, roads_out=None):
        walks = []
        return roads.build_tile(self.m.FRAME, 0, 0, [self.feats], td._flat, td._everywhere, walks=walks,
                                signals=self.m.NO_SIGNALS, country="DE", de_on=True, diag=diag,
                                roads_out=roads_out)

    def test_chunks_unchanged_by_diag(self):
        diag = {}
        self.assertEqual(pickle.dumps(self.tile(diag=diag)), pickle.dumps(self.tile()))
        self.assertTrue(DIAG_KEYS <= set(diag), DIAG_KEYS - set(diag))

    def test_halo_cells_repeating_the_tiles_own_change_nothing(self):
        """build_tile(halo_cells=): the halo reads only roads the tile's own cells do not have (R1-22)."""
        walks = []
        with_halo = roads.build_tile(self.m.FRAME, 0, 0, [self.feats], td._flat, td._everywhere, walks=walks,
                                     signals=self.m.NO_SIGNALS, country="DE", de_on=True, halo_cells=[self.feats, None])
        self.assertEqual(pickle.dumps(with_halo), pickle.dumps(self.tile()))

    def test_lines_have_all_stages(self):
        diag = {}
        self.tile(diag=diag)
        self.assertTrue(diag["lines"], "no line recorded on the 4-way")
        for rec in diag["lines"]:
            self.assertIn(rec["kind"], ("centre", "lane", "edge"))
            self.assertEqual(set(rec["stages"]), STAGES)
            for stage in rec["stages"].values():
                for arr in stage:
                    self.assertEqual(np.asarray(arr).shape[1], 2)
        self.assertTrue(any(rec["stages"]["drawn"] for rec in diag["lines"]), "no line drawn")

    def test_drawn_points_lie_on_the_cut_stage(self):
        diag = {}
        self.tile(diag=diag)
        for rec in diag["lines"]:
            st = rec["stages"]
            if not st["drawn"]:
                continue
            cut = np.vstack(st["cut"]) if st["cut"] else np.zeros((0, 2))
            for run in st["drawn"]:
                for p in np.asarray(run):
                    self.assertTrue(len(cut) and np.min(np.hypot(*(cut - p).T)) < 1e-6, (rec["fid"], p))

    def test_roads_match_roads_out(self):
        diag, out = {}, []
        self.tile(diag=diag, roads_out=out)
        self.assertEqual([r[0] for r in diag["roads"]], [r[0] for r in out])
        self.assertTrue(diag["roads"])


class SignalsSkippedTest(unittest.TestCase):
    def test_short_approach_is_reported(self):
        # test_navgraph's SignalsTest case: the 8 m link from the connector cK is left out, and reported as such
        sp = roads.Spec(8.0, "paved", True, "residential", False)
        nodes = {"cJ": [0.0, 0.0, 0.0, 4], "cK": [0.0, 8.0, 0.0, 4], "s1": [0.0, -40.0, 0.0, 2],
                 "cW": [-40.0, 0.0, 0.0, 2], "cE": [40.0, 0.0, 0.0, 2]}
        arms = {"cJ": [("n", sp, np.array([0.0, 1.0]), 0.0, 1), ("s", sp, np.array([0.0, -1.0]), 0.0, 0),
                       ("w", sp, np.array([-1.0, 0.0]), 0.0, 1), ("e", sp, np.array([1.0, 0.0]), 0.0, 0)]}
        approaches = {("cJ", "n", 1): ("cK", True), ("cJ", "s", 0): ("s1", True), ("cJ", "w", 1): ("cW", True),
                      ("cJ", "e", 0): ("cE", True)}
        all_stop = lambda a, k, link_ok=None, ring=None, signals=None, country=None, priority=None: \
            [(x, furniture.G_STOP) for x in a]      # an all-way stop
        skipped = []
        with mock.patch.object(furniture, "junction_controls", all_stop):
            out = navgraph._signals(arms, approaches, nodes, skipped=skipped)
        self.assertEqual(len(skipped), 1, skipped)
        self.assertEqual(skipped[0]["why"], "short")
        self.assertEqual(skipped[0]["nb"], "cK")
        self.assertEqual(skipped[0]["dist"], 8.0)
        self.assertEqual({ap[0] for ap in out[0]["app"]}, {"s1", "cW", "cE"})


class GraphDiagTest(unittest.TestCase):
    def test_approaches_reported_and_graph_unchanged(self):
        lines = [[(0, 0), (300, 0)]]
        conn = [(0, "A", 0.0, 0.0), (0, "B", 1.0, 1.0)]
        diag = {}
        with_diag = de_graph(lines, ["primary"], conn, diag=diag)
        without = de_graph(lines, ["primary"], conn)
        self.assertEqual(pickle.dumps(with_diag), pickle.dumps(without))
        self.assertTrue(diag["approaches"], "no approaches reported")
        self.assertIn("skipped", diag)


if __name__ == "__main__":
    unittest.main()
