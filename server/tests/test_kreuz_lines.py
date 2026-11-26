"""The probe's lines (Phase 0, Brief C4): tools/leonberg_junction_lines.py on hand-made runs (no map data): the reason of
each painted end, the stubs, and the road pieces of a junction arm."""
import os
os.environ.setdefault("BNE_REGIONAL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                                                       "regional"))     # (PR #505's data: the tests' slice)
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unittest import mock  # noqa: E402

import borderfixtures  # noqa: E402
import designs  # noqa: E402
import numpy as np  # noqa: E402
import roads  # noqa: E402
import leonberg_junction_lines as jl  # noqa: E402
import test_designs as td  # noqa: E402     (its classes are not collected here: only MarkingsTest's helpers are used)

STAGE_NAMES = ("plan", "tile", "junction", "corner", "cut", "drawn")
A_SPEC = SimpleNamespace(cls="secondary", width=8.0, oneway=False, lanes=None)
B_SPEC = SimpleNamespace(cls="tertiary", width=8.0, oneway=False, lanes=None)
Y = 1.5                                   # the line lies 1.5 m left of road a (its centre line along x)


def seg(x0, x1, y=Y):
    return np.array([[x0, y], [x1, y]], float)


def rec(fid, stages, dash=False, kind="centre", side="left"):
    """One line record of roads.build_tile's diag: stages maps a stage name to a list of polylines."""
    st = {n: [np.asarray(p, float) for p in stages.get(n, [])] for n in STAGE_NAMES}
    return {"fid": fid, "kind": kind, "side": side, "dash": dash, "w": 0.15, "stages": st}


def run_of(diags, roads_=None, radius=200.0):
    if roads_ is None:
        roads_ = [("a", A_SPEC, np.array([[0.0, 0, 0], [100.0, 0, 0]]), 0.0),
                  ("b", B_SPEC, np.array([[50.0, -50, 0], [50.0, 50, 0]]), 0.0)]
    return SimpleNamespace(roads=roads_, diags=diags, feats_of={}, origin=np.zeros(2), radius=radius)


def all_stages(**over):
    base = {n: [seg(0, 100)] for n in STAGE_NAMES}
    base.update(over)
    return base


def ends_of(ends, fid="a", kind="centre"):
    return sorted((e["reason"], e["x"]) for e in ends if e["fid"] == fid and e["kind"] == kind)


class LineEndTest(unittest.TestCase):

    def test_plain_line_ends_at_road_ends(self):
        ends, stubs = jl.line_ends(run_of([{"lines": [rec("a", all_stages())]}]), [])
        self.assertEqual(ends_of(ends), [("road_end", 0.0), ("road_end", 100.0)])
        self.assertEqual({e["detail"] for e in ends}, {"dead_end"})
        self.assertEqual(stubs, [])

    def test_tile_edge(self):
        st = all_stages(tile=[seg(0, 80)], junction=[seg(0, 80)], corner=[seg(0, 80)], cut=[seg(0, 80)],
                        drawn=[seg(0, 80)])
        ends, _ = jl.line_ends(run_of([{"lines": [rec("a", st)]}]), [])
        self.assertEqual(ends_of(ends), [("road_end", 0.0), ("tile_edge", 80.0)])

    def test_junction_cut_with_stop_line(self):
        st = all_stages(junction=[seg(0, 40)], corner=[seg(0, 40)], cut=[seg(0, 40)], drawn=[seg(0, 40)])
        diag = {"lines": [rec("a", st)],
                "crosswalks": [{"a": [41.0, -3.0], "b": [41.0, 3.0], "w": 0.5, "stop": True}]}
        ends, _ = jl.line_ends(run_of([diag]), [])
        cut = [e for e in ends if e["reason"] == "junction_cut"]
        self.assertEqual(len(cut), 1)
        self.assertEqual(cut[0]["x"], 40.0)
        self.assertEqual(cut[0]["detail"], "stop_line")

    def test_cut_by_road_culprit(self):
        st = all_stages(cut=[seg(0, 46)], drawn=[seg(0, 46)])
        ends, _ = jl.line_ends(run_of([{"lines": [rec("a", st)]}]), [])
        cut = [e for e in ends if e["reason"] == "cut_by_road"]
        self.assertEqual([(e["x"], e["culprit"], e["detail"]) for e in cut], [(46.0, "b", "tertiary")])

    def test_dashed_line_is_two_road_ends(self):
        # dashes of 3 m with 6 m gaps from 0 to 93 m: their gaps are bridged, only the two ends remain
        recs = []
        for k in range(11):
            d = seg(9.0 * k, 9.0 * k + 3.0)
            recs.append(rec("a", {n: [d] for n in STAGE_NAMES}, dash=True))
        ends, _ = jl.line_ends(run_of([{"lines": recs}]), [])
        self.assertEqual(ends_of(ends), [("road_end", 0.0), ("road_end", 93.0)])

    def test_short_cut_piece_is_a_stub(self):
        st = all_stages(cut=[seg(0, 40), seg(41, 42), seg(50, 100)], drawn=[seg(0, 40), seg(50, 100)])
        _, stubs = jl.line_ends(run_of([{"lines": [rec("a", st)]}]), [])
        self.assertEqual([(s["fid"], s["x"], s["len_m"]) for s in stubs], [("a", 41.5, 1.0)])

    def test_skipped_road_reason(self):
        ends, _ = jl.line_ends(run_of([{"lines": [], "skipped": {"b": "internal"}}]), [])
        self.assertEqual([(e["fid"], e["reason"], e["kind"]) for e in ends], [("b", "internal", None)])

    def test_fully_cut_group(self):
        st = {n: [] for n in STAGE_NAMES}
        st["plan"] = [seg(0, 100)]
        st["tile"] = [seg(0, 100)]
        ends, _ = jl.line_ends(run_of([{"lines": [rec("a", st)]}]), [])
        self.assertEqual([(e["reason"], e["detail"]) for e in ends], [("fully_cut", "junction")])


class HelperTest(unittest.TestCase):

    def test_merge(self):
        self.assertEqual(jl.merge([(3.0, 5.0), (0.0, 3.0), (7.0, 8.0), (3.5, 4.0)], 0.6), [(0.0, 5.0), (7.0, 8.0)])
        self.assertEqual(jl.merge([], 0.6), [])

    def test_arc_of(self):
        line = np.array([[0.0, 0], [10.0, 0], [10.0, 10.0]])
        s = jl.arc_of(line, np.array([[4.0, 2.0], [12.0, 5.0]]))
        self.assertTrue(np.allclose(s, [4.0, 15.0]), s)


class ArmPiecesTest(unittest.TestCase):

    def setUp(self):
        self.spec = roads.Spec(8.0, "paved", False, "secondary", False, 0.0, 0, None, None, None, 2)

    def arm(self, fid="a"):
        return (fid, self.spec, np.array([1.0, 0.0]), np.array([0.0, 0.0, 0.0]), True, 30.0)

    def road_run(self, extra=()):
        rs = [("a", self.spec, np.array([[0.0, 0, 0], [30.0, 0, 0]]), 0.0),
              ("a2", self.spec, np.array([[30.0, 0, 0], [60.0, 0, 0]]), 0.0),
              ("a3", self.spec, np.array([[60.0, 0, 0], [100.0, 0, 0]]), 0.0)] + list(extra)
        return run_of([{"lines": []}], roads_=rs)

    def test_chain_runs_to_sixty_metres(self):
        pieces = jl.arm_pieces(self.road_run(), self.arm())
        self.assertEqual([p["fid"] for p in pieces], ["a", "a2"])
        self.assertEqual(pieces[0]["len_m"], 30.0)

    def test_stops_at_a_junction_of_three(self):
        c = ("c", self.spec, np.array([[30.0, 0, 0], [30.0, 40, 0]]), 0.0)
        pieces = jl.arm_pieces(self.road_run([c]), self.arm())
        self.assertEqual([p["fid"] for p in pieces], ["a"])
        self.assertEqual(pieces[0]["drawn"], {"centre": 0, "lane": 0, "edge": 0})


def setUpModule():
    borderfixtures.require()        # (no border data: the German tests would pass on anything)


def _length(lines):
    return sum(float(np.hypot(*np.diff(np.asarray(p, np.float64)[:, :2], axis=0).T).sum()) for p in lines)


FOUR = [("e", "secondary", [(120.0, 120.0), (300.0, 120.0)]), ("w", "secondary", [(-60.0, 120.0), (120.0, 120.0)]),
        ("n", "secondary", [(120.0, 120.0), (120.0, 300.0)]), ("s", "secondary", [(120.0, 120.0), (120.0, -60.0)])]
T = [("e", "secondary", [(120.0, 120.0), (200.0, 120.0)]), ("w", "secondary", [(40.0, 120.0), (120.0, 120.0)]),
     ("st", "residential", [(120.0, 120.0), (120.0, 200.0)])]


class CutLinesTest(unittest.TestCase):
    """Germany's road lines (B14, roads.build_tile with the junction model): an approach's lines stop at its stop line
    or its edge (side_cuts), and a line is blocked by the roads that really cross it (crossing_pairs; an edge line also
    by the arms it meets). Lost length: the junction stage less the cut stage (diag)."""

    def setUp(self):
        self.m = td.MarkingsTest()

    def build(self, lines, rural=True, controls=None):
        """Chunks and diag of a tile with lines [(fid, class, [pts])]; rural: no town (edge lines), else a town."""
        feats = self.m.layer({}, lines=lines)
        urban = (lambda la, lo: np.zeros(np.shape(la), bool)) if rural else td._everywhere
        diag = {}
        with mock.patch.object(designs, "controls", controls or designs.controls):
            ch = roads.build_tile(self.m.FRAME, 0, 0, [feats], td._flat, urban, walks=[], signals=self.m.NO_SIGNALS,
                                  country="DE", de_on=True, diag=diag)
        return ch, diag

    @staticmethod
    def lost(diag, fid, kind, side=None):
        return sum(_length(r["stages"]["junction"]) - _length(r["stages"]["cut"]) for r in diag["lines"]
                   if r["fid"] == fid and r["kind"] == kind and (side is None or r["side"] == side))

    def test_side_cut_list(self):
        """A centre line at its approach's stop line (t_in) when it lies on the arm's own half (lateral >= -0.3 m over
        0..15 m), else at the edge (t_out); edge lines always at the edge; left-hand traffic flips the sign."""
        X, u = np.array([120.0, 120.0]), np.array([0.0, 1.0])
        cuts = [(X, u, 7.3, 6.05)]
        north = [(120.0, y) for y in np.arange(121.0, 136.0)]
        self.assertEqual(roads._side_cut_list(north, cuts, False, False)[0][2], 7.3)
        east = [(126.0, y) for y in np.arange(121.0, 136.0)]           # 6 m to the east of the arm: lateral -6
        self.assertEqual(roads._side_cut_list(east, cuts, False, False)[0][2], 6.05)
        self.assertEqual(roads._side_cut_list(east, cuts, True, False)[0][2], 7.3, "(left-hand traffic)")
        self.assertEqual(roads._side_cut_list(north, cuts, False, True)[0][2], 6.05, "(edge line)")
        far = [(120.0, y) for y in np.arange(200.0, 210.0)]              # no point within 0..15 m of the arm
        self.assertEqual(roads._side_cut_list(far, cuts, False, False)[0][2], 6.05)

    def test_stop_line_ends_the_centre_lines_of_the_206_arms(self):
        """A 4-way in town with Z 206 on N and S: each Haltlinie (0.50 m wide) stands at 7.05 m, the centre lines
        of N and S do not run into it (their cut lies at its far edge, 7.3 m; the dashes' phase may leave the gap)."""
        real = designs.controls

        def stop_ns(arms, *a, **kw):
            return [(x, designs.Z206 if x[0] in "ns" else w, p) for x, w, p in real(arms, *a, **kw)]
        ch, diag = self.build(FOUR, rural=False, controls=stop_ns)
        halt = {self.m.arm(h.mean(0))[0]: self.m.arm(h.mean(0))[1] for h in self.m.z294(ch)}
        self.assertEqual(set(halt), {"n", "s"})
        for fid in ("n", "s"):
            self.assertAlmostEqual(halt[fid], 7.05, delta=0.01)
            pts = np.vstack([np.asarray(p, np.float64)[:, :2] for r in diag["lines"] if r["fid"] == fid
                             and r["kind"] == "centre" for p in r["stages"]["junction"]])
            self.assertGreaterEqual(float(np.abs(pts[:, 1] - 120.0).min()), halt[fid] + 0.25 - 0.3, fid)

    def test_parallel_road_six_metres_away_blocks_nothing(self):
        """A main road (12 m) with a residential road 5.5 m beside it, no junction: its lines run on (the parallel road
        does not cross it: the ends of neither lie within the other's half width)."""
        ch, diag = self.build([("m", "secondary", [(40.0, 120.0), (200.0, 120.0)]),
                               ("sv", "residential", [(-100.0, 125.5), (300.0, 125.5)])])
        self.assertEqual(self.lost(diag, "m", "centre"), 0.0)
        self.assertEqual(self.lost(diag, "m", "edge"), 0.0, "(both sides)")

    def test_t_junction_main_road_lines(self):
        """A T with a stem on the north: the main road's centre line runs on through the junction; its edge line on the
        mouth side is interrupted there, the far side's only by the stem's own cut (about 2 m at the junction)."""
        ch, diag = self.build(T)
        self.assertEqual(self.lost(diag, "e", "centre"), 0.0)
        self.assertEqual(self.lost(diag, "w", "centre"), 0.0)
        self.assertGreaterEqual(self.lost(diag, "e", "edge", side=1), 15.0, "(mouth side)")
        self.assertLessEqual(self.lost(diag, "e", "edge", side=-1), 2.5, "(far side)")


if __name__ == "__main__":
    unittest.main()
