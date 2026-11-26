"""lidarsmooth: the lidar roads' surface (a curvature cap per class, zones of many kinks smoothed more, structure ends
fixed) and grading.keep_ground's junction precedence. Synthetic profiles: no fixture, no network."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import grading  # noqa: E402
import lidarsmooth as ls  # noqa: E402

K = ls.kappa("residential")     # 1 / 50 m


def d2(z):
    z = np.asarray(z, np.float64)
    return np.abs(z[:-2] - 2.0 * z[1:-1] + z[2:])


class KappaTest(unittest.TestCase):
    def test_classes(self):
        self.assertAlmostEqual(ls.kappa("residential"), 1.0 / ls.DEFAULT_RADIUS_M)
        self.assertAlmostEqual(ls.kappa("motorway"), 1.0 / 400.0)
        self.assertLess(ls.kappa("primary"), ls.kappa("tertiary"))
        self.assertIsNone(ls.kappa("steps"))

    def test_priority_follows_the_radius(self):
        self.assertGreater(ls.priority(ls.kappa("primary")), ls.priority(ls.kappa("residential")))


class CapTest(unittest.TestCase):
    def test_a_ledge_becomes_a_ramp_within_the_limit(self):
        z = np.r_[np.zeros(80), np.full(80, 0.3)]
        out = ls.cap(z, K)
        self.assertLessEqual(float(d2(out).max()), K * (1 + 1e-6))
        self.assertAlmostEqual(float(out[0]), 0.0, places=6)
        self.assertAlmostEqual(float(out[-1]), 0.3, places=6)

    def test_a_steady_grade_is_kept(self):
        z = 400.0 + 0.12 * np.arange(200.0)
        np.testing.assert_allclose(ls.cap(z, K), z, atol=1e-9)

    def test_a_gentle_crest_is_kept(self):
        s = np.arange(300.0)
        z = 0.4 * K * 50 * 50 * np.sin(s / 50.0)       # curvature 0.4 K at most
        out = ls.cap(z, K)
        # (within a few mm: the window of the cap is limited, CAP_DEPTH_M, on a steep stretch the parabolas touch a
        # sample or two off; the ends free, along their trend)
        self.assertLess(float(np.abs(out - z).max()), 0.008)
        self.assertLess(float(np.abs(out - z)[60:240].mean()), 0.0005)

    def test_a_spike_becomes_a_low_bump(self):
        z = np.zeros(100)
        z[50] = 0.3
        out = ls.cap(z, K)
        self.assertLess(float(np.abs(out).max()), 0.1)
        self.assertLessEqual(float(d2(out).max()), K * (1 + 1e-6))

    def test_columns_each_capped(self):
        z = np.stack([np.r_[np.zeros(50), np.full(50, 0.2)], np.zeros(100)], axis=1)
        out = ls.cap(z, K)
        self.assertEqual(out.shape, z.shape)
        self.assertLessEqual(float(d2(out[:, 0]).max()), K * (1 + 1e-6))
        np.testing.assert_allclose(out[:, 1], 0.0, atol=1e-12)


class ProfileTest(unittest.TestCase):
    def test_washboard_smoothed_more_than_the_cap_alone(self):
        rng = np.random.default_rng(7)
        z = 0.05 * np.arange(200.0)
        z[60:140] += rng.normal(0.0, 0.06, 80)
        capped, out = ls.cap(z, K), ls.profile(z, K)
        self.assertLessEqual(float(d2(out).max()), K * (1 + 1e-6))
        self.assertLess(float(np.std(np.diff(out[60:140], 2))), float(np.std(np.diff(capped[60:140], 2))))
        np.testing.assert_allclose(out[:30], z[:30], atol=1e-6)        # (the clean grade away from it: as it was)

    def test_none_leaves_it(self):
        z = np.r_[np.zeros(10), np.ones(10)]
        np.testing.assert_array_equal(ls.profile(z, None), z)


class EndFixTest(unittest.TestCase):
    def test_bridge_hole_filled_to_the_trend(self):
        z = 300.0 + 0.04 * np.arange(60.0)
        trend = z.copy()
        z[-8:] -= np.linspace(0.3, 2.0, 8)              # the DTM falls into the hole under the deck
        out = ls.end_fix(z, "bridge", at_end=True)
        np.testing.assert_allclose(out[-8:], trend[-8:], atol=1e-6)
        np.testing.assert_allclose(out[:-8], z[:-8], atol=1e-12)

    def test_tunnel_embankment_at_a_short_part(self):
        z = np.r_[np.linspace(348.0, 342.0, 8), np.full(6, 341.6)]    # 14 m: the portal's bank, then the track
        out = ls.end_fix(z, "tunnel", at_end=False)
        self.assertLess(float(np.abs(out - 341.6).max()), 0.1)

    def test_a_real_ramp_is_no_hole(self):
        z = 300.0 + 0.08 * np.arange(60.0)
        np.testing.assert_array_equal(ls.end_fix(z, "bridge", at_end=True), z)

    def test_no_structure_no_change(self):
        z = np.r_[np.zeros(30), np.full(10, -2.0)]
        np.testing.assert_array_equal(ls.end_fix(z, None, at_end=True), z)


class HoleFixTest(unittest.TestCase):
    def test_an_underpass_hole_is_bridged_by_the_straight_road(self):
        # (Eltinger Strasse over the Rathaus underpass: the DTM sank 1.8 m in a V 40 m long round the crossing)
        s = np.arange(90.0)
        road = 375.0 - 0.02 * s
        z = road - np.clip(1.8 - np.abs(s - 30.0) * 0.09, 0.0, None)
        out = ls.hole_fix(z, 28.0, 1.5, K, sag=0.0)
        np.testing.assert_allclose(out, road, atol=0.12)
        self.assertTrue((out >= z - 1e-12).all())

    def test_a_real_sag_over_a_stream_stays(self):
        s = np.arange(120.0)
        z = 300.0 + 0.5 * (K * 0.6) * (s - 60.0) ** 2        # a valley's sag, 0.6 of the limit
        np.testing.assert_allclose(ls.hole_fix(z, 60.0, 1.0, K, sag=1.0), z, atol=1e-9)

    def test_a_narrow_bed_under_the_road_is_filled(self):
        s = np.arange(80.0)
        z = 300.0 + 0.03 * s
        trend = z.copy()
        z[38:43] -= np.array([0.6, 1.4, 1.6, 1.3, 0.5])     # a stream's bed (a bridge not mapped)
        out = ls.hole_fix(z, 40.0, 1.0, K, sag=1.0)
        np.testing.assert_allclose(out, trend, atol=0.02)

    def test_a_shallow_dip_is_left(self):
        s = np.arange(80.0)
        z = 300.0 + 0.03 * s
        z[39:42] -= 0.1
        np.testing.assert_array_equal(ls.hole_fix(z, 40.0, 1.0, K, sag=0.0), z)

    def test_a_hole_away_from_the_crossing_is_not_this_ones(self):
        s = np.arange(120.0)
        z = 300.0 + 0.03 * s
        z[95:100] -= 1.5
        np.testing.assert_array_equal(ls.hole_fix(z, 20.0, 1.0, K, sag=0.0), z)

    def test_fit2_curvature_bounds(self):
        x = np.arange(-10.0, 11.0)
        a, b, c = ls._fit2(x, 1.0 + 0.5 * x + 0.3 * x * x, 0.1)
        self.assertAlmostEqual(c, 0.05)
        a, b, c = ls._fit2(x, 1.0 + 0.5 * x + 0.01 * x * x, 0.1)
        self.assertAlmostEqual(c, 0.01)
        self.assertAlmostEqual(b, 0.5)
        self.assertAlmostEqual(a, 1.0)


class SurfaceTest(unittest.TestCase):
    def test_a_bridge_end_mended_column_by_column(self):
        # (Renninger Strasse past the Glems bridge: the bed under one side only; the centre's correction lifted the
        # other side 0.85 m over the road)
        offs = np.linspace(-3.0, 3.0, 7)
        n = 60
        raw = np.full((n, 7), 365.2)
        raw[:8, :3] -= np.linspace(2.0, 0.6, 3)[None, :] * np.linspace(1.0, 0.1, 8)[:, None]
        out = ls.surface(raw, offs, K, ends=("bridge", None))
        np.testing.assert_allclose(out, 365.2, atol=0.03)

    def test_holes_on_their_columns(self):
        offs = np.linspace(-3.0, 3.0, 7)
        raw = np.full((80, 7), 200.0)
        raw[38:43, 0] -= 1.5
        out = ls.surface(raw, offs, K, holes=[(0, 40.0, 1.0, 1.0)])
        np.testing.assert_allclose(out, 200.0, atol=0.02)

    def test_crossfall_and_grade_kept(self):
        offs = np.linspace(-3.0, 3.0, 7)
        s = np.arange(120.0)
        raw = 200.0 + 0.06 * s[:, None] - 0.025 * offs[None, :]
        np.testing.assert_allclose(ls.surface(raw, offs, K), raw, atol=1e-6)

    def test_a_ledge_across_the_whole_road_is_ramped_on_every_column(self):
        offs = np.linspace(-3.0, 3.0, 7)
        raw = np.r_[np.zeros(60), np.full(60, 0.25)][:, None] + 0.0 * offs[None, :]
        out = ls.surface(raw, offs, K)
        for j in range(len(offs)):
            self.assertLessEqual(float(d2(out[:, j]).max()), K * (1 + 1e-6))


class PinTest(unittest.TestCase):
    def test_pinned_samples_take_the_target_and_ease_off(self):
        z = np.zeros((80, 3))
        mask = np.zeros((80, 3), bool)
        mask[:10] = True
        target = np.full((80, 3), 0.3)
        out = ls.pin(z, mask, target, K)
        np.testing.assert_allclose(out[:10], 0.3)
        self.assertLessEqual(float(d2(out[:, 1]).max()), K * 0.5 * (1 + 1e-6) + 1e-9)
        np.testing.assert_allclose(out[-20:], 0.0, atol=1e-12)


def _zone(line, z, hw, rank):
    """A keep_ground zone: a flat surface z on a straight line."""
    st = np.r_[0.0, np.cumsum(np.hypot(*np.diff(line, axis=0).T))]
    offs = np.linspace(-hw, hw, 5)
    surf = np.full((len(st), len(offs)), float(z))
    return (line, st, offs, (surf, surf - 0.0), hw + grading.UNGRADED_MARGIN_M, rank)


class KeepGroundTest(unittest.TestCase):
    def setUp(self):
        x = np.arange(-40.0, 41.0)
        self.main = _zone(np.stack([x, np.zeros_like(x)], axis=1), 10.0, 3.0, (3.0, 80, "a"))
        self.side = _zone(np.stack([np.zeros(41), np.arange(-40.0, 1.0)], axis=1), 9.0, 2.0, (1.0, 40, "b"))

    def test_the_higher_rank_holds_its_carriageway(self):
        w, v = grading.keep_ground(np.array([0.0, 0.0]), np.array([0.0, -1.0]), np.zeros(2), [self.side, self.main])
        np.testing.assert_allclose(w, 1.0)
        np.testing.assert_allclose(v, 10.0)

    def test_the_lesser_road_off_the_junction(self):
        w, v = grading.keep_ground(np.array([0.0]), np.array([-20.0]), np.zeros(1), [self.main, self.side])
        self.assertAlmostEqual(float(v[0]), 9.0)

    def test_margin_is_the_ground_moved_by_the_edge_correction(self):
        # (beside the main road, 1 m into its margin: the DTM there, 5.0, plus the correction at its edge, 10 - 0)
        w, v = grading.keep_ground(np.array([30.0]), np.array([4.0]), np.array([5.0]), [self.main])
        self.assertGreater(float(w[0]), 0.0)
        self.assertLess(float(w[0]), 1.0)
        self.assertAlmostEqual(float(v[0]), 15.0)


class KeepHolesTest(unittest.TestCase):
    def test_each_column_where_it_crosses(self):
        line = np.stack([np.arange(0.0, 61.0), np.zeros(61)], axis=1)
        offs = np.linspace(-2.0, 2.0, 5)
        under = [(np.array([[20.0, -10.0], [40.0, 10.0]]), 1.5, 0.0)]      # (45 degrees)
        got = grading.keep_holes(line, offs, under)
        self.assertEqual([g[0] for g in got], [0, 1, 2, 3, 4])
        # (the left side, positive offsets: further along on this diagonal)
        np.testing.assert_allclose([g[1] for g in got], 30.0 + offs, atol=1e-9)
        self.assertTrue(all(g[2] == 1.5 and g[3] == 0.0 for g in got))

    def test_a_way_along_the_road_crosses_nothing(self):
        line = np.stack([np.arange(0.0, 61.0), np.zeros(61)], axis=1)
        under = [(np.array([[0.0, -3.0], [60.0, 2.0]]), 1.0, 1.0)]        # (a stream beside it, 5 degrees)
        self.assertEqual(grading.keep_holes(line, np.linspace(-2.0, 2.0, 5), under), [])


if __name__ == "__main__":
    unittest.main()
