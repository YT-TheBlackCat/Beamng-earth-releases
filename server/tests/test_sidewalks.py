"""The drape formulas (sidewalks.drape_back, drape_corner): a slab follows the ground behind its curb (GEHWEGE_DESIGN.md 2.1, 2.4)."""
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sidewalks  # noqa: E402
from sidewalks import SINK, BACK_RUN, CROSS_SLOPE, CORNER_LIFT, BACK_DROP_MAX_M, CORNER_RISE_M  # noqa: E402

TOL = dict(rtol=1e-12, atol=1e-12)


def flat_cross_slope():
    # the old Rust profile's 1.5 % cross slope, so the drape reproduces it exactly on flat ground
    return mock.patch.object(sidewalks, "DRAPE_CROSS_SLOPE", CROSS_SLOPE)


class DrapeBackTest(unittest.TestCase):
    def test_flat_ground_matches_rust_profile(self):
        with flat_cross_slope():
            for h in (0.13, 0.15):
                for w in (1.65, 2.5, 4.8):
                    top, bottom, run = sidewalks.drape_back(np.array([0.0]), np.array([0.0]), np.array([0.0]), h, w)
                    p = sidewalks.profile(h, w)
                    np.testing.assert_allclose(top[0], p[6][1], **TOL)
                    np.testing.assert_allclose(bottom[0], p[7][1], **TOL)
                    np.testing.assert_allclose(run[0], p[7][0] - p[6][0], **TOL)
                    np.testing.assert_allclose(run[0], BACK_RUN * (top[0] - bottom[0]), **TOL)

    def test_ground_slightly_above_design_lifts_slab(self):
        h, w = 0.15, 2.5
        design = 0.0 + h + w * sidewalks.DRAPE_CROSS_SLOPE
        zb = np.array([design + 0.05])
        top, bottom, run = sidewalks.drape_back(np.array([0.0]), zb, zb, h, w)
        np.testing.assert_allclose(top[0], zb[0] + 0.02, **TOL)
        np.testing.assert_allclose(run[0], 0.5 * (top[0] - bottom[0]), **TOL)

    def test_ground_far_above_clamps_to_max_cross_slope(self):
        h, w = 0.15, 2.5
        hi = h + w * sidewalks.DRAPE_CROSS_MAX
        top, _, _ = sidewalks.drape_back(np.array([0.0]), np.array([1.0]), np.array([1.0]), h, w)
        np.testing.assert_allclose(top[0], hi, **TOL)

    def test_ground_below_curb_drops_back_to_ground(self):
        h, w = 0.15, 2.5
        cz = np.array([0.0])
        top, bottom, run = sidewalks.drape_back(cz, np.array([-0.4]), np.array([-0.4]), h, w)
        design = h + w * sidewalks.DRAPE_CROSS_SLOPE
        np.testing.assert_allclose(top[0], design, **TOL)
        np.testing.assert_allclose(bottom[0], -0.4 - SINK, **TOL)
        self.assertLessEqual(run[0], 0.75 + 1e-12)

    def test_deep_drop_limited(self):
        h, w = 0.15, 2.5
        _, bottom, _ = sidewalks.drape_back(np.array([0.0]), np.array([-5.0]), np.array([-5.0]), h, w)
        np.testing.assert_allclose(bottom[0], -SINK - BACK_DROP_MAX_M, **TOL)

    def test_nan_ground(self):
        h, w = 0.15, 2.5
        design = h + w * sidewalks.DRAPE_CROSS_SLOPE
        nan = np.array([np.nan])
        top, bottom, _ = sidewalks.drape_back(np.array([0.0]), nan, np.array([0.0]), h, w)
        np.testing.assert_allclose(top[0], design, **TOL)
        np.testing.assert_allclose(bottom[0], -SINK, **TOL)
        # NaN only in zbb: the back bottom follows zb
        top, bottom, _ = sidewalks.drape_back(np.array([0.0]), np.array([-0.4]), nan, h, w)
        np.testing.assert_allclose(bottom[0], -0.4 - SINK, **TOL)

    def test_low_curb_follows_ground_within_15_percent(self):
        h, w = 0.035, 2.5
        top, _, _ = sidewalks.drape_back(np.array([0.0]), np.array([0.10 * w]), np.array([0.10 * w]), h, w)
        np.testing.assert_allclose(top[0], 0.10 * w + h, **TOL)
        top, _, _ = sidewalks.drape_back(np.array([0.0]), np.array([0.30 * w]), np.array([0.30 * w]), h, w)
        np.testing.assert_allclose(top[0], h + w * sidewalks.LOW_DRAPE_SLOPE, **TOL)


class DrapeCornerTest(unittest.TestCase):
    def test_flat_corner_matches_rust(self):
        h, w = 0.15, 2.5
        with flat_cross_slope():
            z_first = z_last = 0.0
            g = np.zeros(3)
            outer_z, outer_back = sidewalks.drape_corner(z_first, z_last, g, (w, w), h)
            np.testing.assert_allclose(outer_z, [z_last, 0.0, z_first], **TOL)
            np.testing.assert_allclose(outer_back[:, 0], g - SINK, **TOL)
            np.testing.assert_allclose(outer_back[:, 1], BACK_RUN * (h + SINK + CORNER_LIFT), **TOL)

    def test_sloped_corner_middle_and_ends(self):
        h, w = 0.15, 2.5
        with flat_cross_slope():
            # ground 0.3 m above the ends in the middle: the middle rises to max(top_ends) + CORNER_RISE_M
            g = np.array([0.0, 0.3, 0.0])
            outer_z, outer_back = sidewalks.drape_corner(0.0, 0.0, g, (w, w), h)
            top_ends = h + w * sidewalks.DRAPE_CROSS_SLOPE
            np.testing.assert_allclose(outer_z[1] + h + CORNER_LIFT, top_ends + CORNER_RISE_M, **TOL)
            np.testing.assert_allclose(outer_z[[0, 2]], [0.0, 0.0], **TOL)

    def test_nan_middle_uses_mean_of_ends(self):
        h, w = 0.15, 2.5
        with flat_cross_slope():
            g = np.array([0.0, np.nan, 0.0])
            outer_z, _ = sidewalks.drape_corner(0.0, 0.0, g, (w, w), h)
            np.testing.assert_allclose(outer_z, [0.0, 0.0, 0.0], **TOL)



ORIGIN = (100.0, 200.0, 3.0)


def straight(k=6, length=30.0):
    """A straight curb line along x at y = 200 with its road at z = 3 (the origin's z) and the curb top 0.15 above."""
    x = np.linspace(0.0, length, k)
    curb = np.c_[100.0 + x, np.full(k, 200.0), np.full(k, 3.15)]
    return curb, np.c_[np.zeros(k), np.ones(k)]


def flat_back(curb, h=sidewalks.CURB_H, w=2.5):
    """The drape of a slab on flat ground at the curb's height (its design cross slope): what the flat design is."""
    with flat_cross_slope():
        top, bottom, run = sidewalks.drape_back(curb[:, 2], curb[:, 2], curb[:, 2], h, w)
    return np.c_[top, bottom, run]


def build_run(curb, out, back=None, lean=False, **kw):
    b = sidewalks.Builder(ORIGIN, collision=lean)
    b.add_run(curb, out, 2.5, back=back, **kw)
    return b.col.arrays() if lean else b.arrays()


def assert_meshes_equal(test, a, b, atol=1e-9):
    for x, y in zip(a, b):
        test.assertEqual(len(x), len(y))
    np.testing.assert_allclose(a[0], b[0], rtol=0, atol=atol)
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_allclose(a[2], b[2], rtol=0, atol=atol)
    np.testing.assert_array_equal(a[3], b[3])


class BuilderDrapeTest(unittest.TestCase):
    def test_layout_full_and_lean(self):
        k = 6
        curb, out = straight(k)
        P, _, _, _ = build_run(curb, out, back=flat_back(curb))
        self.assertEqual(len(P), 14 * k + 16)
        rel = curb - np.array(ORIGIN)
        # profile point 6 (the slab's outer edge) is the first half of block 6, point 7 (the back) the second half
        np.testing.assert_allclose(P[2 * k * 6:2 * k * 6 + k, :2], rel[:, :2] + out * 2.5, **TOL)
        np.testing.assert_allclose(P[2 * k * 6 + k:2 * k * 6 + 2 * k, 2], rel[:, 2] - SINK, **TOL)
        PL = build_run(curb, out, back=flat_back(curb), lean=True)[0]
        self.assertEqual(len(PL), 26)            # a straight line keeps its two ends: 8 k' + 5 * 2 caps, k' = 2

    def test_flat_ground_is_the_flat_design(self):
        curb, out = straight(6)
        back = flat_back(curb)
        for lean in (False, True):
            with self.subTest(lean=lean):
                assert_meshes_equal(self, build_run(curb, out, lean=lean), build_run(curb, out, back, lean=lean))
        ramped = dict(ramps=[(5.0, 7.0)])
        with self.subTest(ramp=True):
            flat = flat_back(curb)
            assert_meshes_equal(self, build_run(curb, out, **ramped), build_run(curb, out, flat, **ramped))
        eu = dict(europe=True, taper=True, surface=None)
        with self.subTest(europe=True):
            curb30, out30 = straight(12, 60.0)
            assert_meshes_equal(self, build_run(curb30, out30, **eu),
                                build_run(curb30, out30, flat_back(curb30), **eu))

    def test_tilted_ground_shears_points_four_and_five(self):
        k = 6
        curb, out = straight(k)
        flat = flat_back(curb)
        tilt = flat.copy()
        tilt[:, 0] += 0.08                       # the ground rises behind: the slab's outer edge rises 8 cm
        tilt[:, 2] = 2.5
        P, _, _, _ = build_run(curb, out, back=tilt)
        Pf, _, _, _ = build_run(curb, out, back=flat)
        rel = curb - np.array(ORIGIN)
        x5 = sidewalks.profile(sidewalks.CURB_H, 2.5)[5, 0]
        x4 = sidewalks.profile(sidewalks.CURB_H, 2.5)[4, 0]
        for idx, x in ((2 * k * 5, x5), (2 * k * 4, x4)):
            np.testing.assert_allclose(P[idx:idx + k, 2] - Pf[idx:idx + k, 2],
                                       0.08 * (x - sidewalks.CURB_W) / (2.5 - sidewalks.CURB_W), **TOL)
        np.testing.assert_allclose(P[2 * k * 6:2 * k * 6 + k, 2], tilt[:, 0] - ORIGIN[2], **TOL)
        np.testing.assert_allclose(P[2 * k * 6 + k:2 * k * 6 + 2 * k, 2], tilt[:, 1] - ORIGIN[2], **TOL)
        # the collision mesh: its outer edge (profile point 6, lean slot 3) is the drawn one at the same place
        PL = build_run(curb, out, back=tilt, lean=True)[0]
        np.testing.assert_allclose(PL[12, :], P[2 * k * 6], **TOL)           # section 0 of slot 3
        np.testing.assert_allclose(PL[14, :], P[2 * k * 6 + k], **TOL)       # section 0 of slot 4 (the back)

    def test_valley_side_back_follows_ground(self):
        k = 6
        curb, out = straight(k)
        fb = flat_back(curb)
        back = np.c_[fb[:, 0], fb[:, 1] - 0.5, np.full(k, 0.4)]      # the ground 0.5 m lower, back 0.4 wide
        P, _, _, _ = build_run(curb, out, back=back)
        back7 = np.r_[2 * k * 6 + k:2 * k * 6 + 2 * k, 14 * k + 7, 14 * k + 8 + 7]
        np.testing.assert_allclose(P[back7[:k], 2], back[:, 1] - ORIGIN[2], **TOL)
        xy6 = P[2 * k * 6:2 * k * 6 + k, :2]
        step = P[2 * k * 6 + k:2 * k * 6 + 2 * k, :2] - xy6
        np.testing.assert_allclose(np.hypot(step[:, 0], step[:, 1]), 0.4, **TOL)
        np.testing.assert_allclose(step, out * 0.4, **TOL)
        # both end caps carry the back too
        self.assertAlmostEqual(P[14 * k + 7, 2], back[0, 1] - ORIGIN[2], places=9)
        self.assertAlmostEqual(P[14 * k + 8 + 7, 2], back[-1, 1] - ORIGIN[2], places=9)

    def test_corner_back_faces(self):
        arc = np.array([[100.0 + x, 200.0, 3.15] for x in (0.0, 1.0, 2.0, 3.0)])
        inward = np.tile([0.0, 1.0], (4, 1))
        outer = np.array([[103.0, 202.5, 3.15], [101.5, 204.0, 3.15], [100.0, 202.5, 3.15]])
        back = np.array([[2.9, 0.5], [2.9, 0.5], [2.9, 0.5]])
        m, k = 3, 4
        plain = sidewalks.Builder(ORIGIN)
        plain.add_corner(arc, inward, outer)
        P0 = plain.arrays()[0]
        b = sidewalks.Builder(ORIGIN)
        b.add_corner(arc, inward, outer, back=back)
        P = b.arrays()[0]
        self.assertEqual(len(P), 7 * k + m + 4 * (m - 1))
        base = 7 * k + m
        for q in range(m - 1):
            a, bb, bq, aa = base + 4 * q, base + 4 * q + 1, base + 4 * q + 2, base + 4 * q + 3
            self.assertAlmostEqual(P[aa, 2], 2.9 - ORIGIN[2], places=9)
            self.assertAlmostEqual(P[bq, 2], 2.9 - ORIGIN[2], places=9)
            self.assertAlmostEqual(np.hypot(*(P[aa, :2] - P[a, :2])), 0.5, places=9)
            self.assertAlmostEqual(np.hypot(*(P[bq, :2] - P[bb, :2])), 0.5, places=9)
            np.testing.assert_allclose(P[a], P0[a], **TOL)      # the top edge is the Rust one
        # the lean collision mesh: 3 k' + m + 4 (m - 1) vertices; a straight arc keeps two of its k = 4 sections
        c = sidewalks.Builder(ORIGIN, collision=True)
        c.add_corner(arc, inward, outer, back=back)
        PL = c.col.arrays()[0]
        kl = (len(PL) - m - 4 * (m - 1)) // 3
        self.assertEqual((kl, len(PL)), (2, 3 * 2 + m + 4 * (m - 1)))
        for q in range(m - 1):
            self.assertAlmostEqual(PL[3 * kl + m + 4 * q + 3, 2], 2.9 - ORIGIN[2], places=9)
            self.assertAlmostEqual(PL[3 * kl + m + 4 * q + 2, 2], 2.9 - ORIGIN[2], places=9)
        # a corner without outer_back is the plain one; a wrong-length back is dropped by add_walk
        c = sidewalks.Builder(ORIGIN)
        c.add_corner(arc, inward, outer, back=None)
        assert_meshes_equal(self, (c.arrays()[0], c.arrays()[1], c.arrays()[2], c.arrays()[3]),
                            (P0, *plain.arrays()[1:]))

    def test_wrong_back_length_is_ignored(self):
        curb, out = straight(6)
        walk = {"pts": np.c_[curb[:, :2] - np.array(ORIGIN[:2]), curb[:, 2]], "side": 1, "caps": (True, True)}
        plain = sidewalks.Builder(ORIGIN)
        plain.add_walk(walk, 2.5)
        bad = sidewalks.Builder(ORIGIN)
        bad.add_walk(dict(walk, back=flat_back(curb)[:5]), 2.5)
        assert_meshes_equal(self, bad.arrays(), plain.arrays())
        good = sidewalks.Builder(ORIGIN)
        good.add_walk(dict(walk, back=flat_back(curb)), 2.5)
        assert_meshes_equal(self, good.arrays(), plain.arrays())


if __name__ == "__main__":
    unittest.main()
