"""buildings: a footprint with LoD2 units is drawn from their real roof faces and upper walls (_lod2_draw), and
Baden-Wuerttemberg has its own region style ("swabia")."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import buildings as bld  # noqa: E402


def rect(x0, y0, w, h):
    return np.array([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)], np.float64)


def lod2_house():
    """A 10 x 8 m house (ground 0, eave 6 m, ridge 9 m) as its LoD2 polygons: walls (kind 1), gabled roof (kind 2)."""
    walls = []
    corners = [(0, 0), (10, 0), (10, 8), (0, 8)]
    for (ax, ay), (bx, by) in zip(corners, corners[1:] + corners[:1]):
        walls.append((1, np.array([(ax, ay, 0), (bx, by, 0), (bx, by, 6), (ax, ay, 6)], np.float64)))
    roof = [(2, np.array([(0, 0, 6), (10, 0, 6), (10, 4, 9), (0, 4, 9)], np.float64)),
            (2, np.array([(0, 8, 6), (10, 8, 6), (10, 4, 9), (0, 4, 9)], np.float64))]
    return walls + roof


def lod2_spec(ring=None):
    """A Spec as tile_buildings makes it for a LoD2 unit: plan(), then the same fields the plan loop sets."""
    ring = rect(0, 0, 10, 8) if ring is None else ring
    spec = bld.plan(ring, {"class": "house", "height": 6.0, "roof_shape": "gabled", "roof_height": 3.0}, 0, 48.8, 9.0)
    spec.lod2 = {"polys": lod2_house(), "eave": 6.0, "top": 9.0, "pitched": True}
    spec.floors, spec.floor_h, spec.height = 2, 3.0, 6.0
    spec.hf = spec.skel = spec.steeple = spec.porch = spec.deck = False
    spec.holes = None
    spec.tank = False
    return spec


class ClipAbove(unittest.TestCase):
    def test_clip_keeps_only_the_part_above_the_plane(self):
        tri = np.array([(0, 0, 5), (10, 0, 5), (5, 0, 9)], np.float64)
        out = bld._clip_above(tri, 6.0)
        self.assertGreaterEqual(len(out), 3)
        self.assertTrue(np.all(out[:, 2] >= 6.0 - 1e-9))


class Poly3d(unittest.TestCase):
    def test_roof_square_clockwise_input_faces_up(self):
        m = bld.Mesh()
        sq = [(0, 0, 5), (0, 10, 5), (10, 10, 5), (10, 0, 5)]        # clockwise seen from above
        bld._poly3d(m, sq, (0, (255, 255, 255, 255)))
        P, _, T, _, _ = m.arrays()
        self.assertEqual(len(T), 2)
        for t in T:
            a, b, c = P[t]
            self.assertGreater(np.cross(b - a, c - a)[2], 0.0)


class Lod2Draw(unittest.TestCase):
    def test_roof_reaches_the_ridge_and_takes_the_roof_material(self):
        spec = lod2_spec()
        out = bld.build([(spec, 300.0, 300.5)])
        self.assertIsNotNone(out)
        P, _, T, _, MATERIALS, _ = out
        self.assertAlmostEqual(float(P[:, 2].max()), 309.0, delta=0.1)
        roof_tris = P[T].min(1)[:, 2] >= 305.9              # the roof faces lie above the eave (306 m)
        self.assertTrue(roof_tris.any())
        roof_mat = MATERIALS.index(spec.roof_mat.split("|")[0])
        self.assertIn(roof_mat, np.asarray(out[3])[roof_tris].tolist())


def mesh_roof(out, spec):
    """The triangles (k, 3, 3) of the spec's roof material in a bld.build() result, and their unit normals."""
    P, _, T, tri_mat, MATERIALS, _ = out
    sel = np.asarray(tri_mat) == MATERIALS.index(spec.roof_mat.split("|")[0])
    tris = P[np.asarray(T)[sel]]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    return tris, n / np.linalg.norm(n, axis=1, keepdims=True)


def area(tris):
    return float(np.sum(0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)))


L_RING = np.array([(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)], np.float64)


class Lod2Lift(unittest.TestCase):
    def test_flat_ground_keeps_the_roof_and_its_area(self):
        spec = lod2_spec()
        out = bld.build([(spec, 300.0, 300.0)])
        tris, _ = mesh_roof(out, spec)
        self.assertAlmostEqual(float(out[0][:, 2].max()), 309.0, delta=0.01)
        self.assertAlmostEqual(area(tris), 100.0, delta=0.005 * 100.0)

    def test_slope_lifts_the_whole_roof_without_clamping(self):
        spec = lod2_spec()
        out = bld.build([(spec, 300.0, 306.0)])                 # gmax - gmin = 6 m: lift 0.5 m, eave at 306 + 0.5
        tris, n = mesh_roof(out, spec)
        self.assertAlmostEqual(float(tris[:, :, 2].min()), 306.5, delta=0.01)
        self.assertAlmostEqual(float(out[0][:, 2].max()), 309.5, delta=0.01)
        self.assertAlmostEqual(area(tris), 100.0, delta=0.005 * 100.0)
        n0 = n[0]                                               # each roof triangle lies in one of the two planes
        n1 = next(v for v in n if abs(float(v @ n0)) < 0.99)
        self.assertTrue(np.all(np.maximum(np.abs(n @ n0), np.abs(n @ n1)) > 1 - 1e-6))


    def test_max_lift_caps_the_slope_lift(self):
        spec = lod2_spec()
        old = bld.LOD2_MAX_LIFT_M
        bld.LOD2_MAX_LIFT_M = 0.2
        try:
            out = bld.build([(spec, 300.0, 306.0)])             # lift 0.5 m uncapped, 0.2 m capped
        finally:
            bld.LOD2_MAX_LIFT_M = old
        self.assertAlmostEqual(float(out[0][:, 2].max()), 309.2, delta=0.01)


class WallOutward(unittest.TestCase):
    def test_inward_data_normal_on_an_l_ring_is_turned_outwards(self):
        # the L's inner step edge (x 4..10 at y = 4): the building lies at y < 4, so outward is +y
        fwd = np.array([(10, 4, 6), (4, 4, 6), (4, 4, 9), (10, 4, 9)], np.float64)
        for Q in (fwd, fwd[::-1]):
            out = bld._wall_outward(Q, L_RING)
            self.assertIsNotNone(out)
            self.assertGreater(float(out @ np.array([0.0, 1.0])), 0.99)

    def test_outward_data_normal_is_kept_even_if_the_centre_says_otherwise(self):
        # the arm's east face (x = 4, y 4..10) faces +x; the ring's mean lies to its east, so a centre test would flip it
        fwd = np.array([(4, 4, 6), (4, 10, 6), (4, 10, 9), (4, 4, 9)], np.float64)
        for Q in (fwd, fwd[::-1]):
            out = bld._wall_outward(Q, L_RING)
            self.assertIsNotNone(out)
            self.assertGreater(float(out @ np.array([1.0, 0.0])), 0.99)

    def test_horizontal_wall_has_no_direction(self):
        self.assertIsNone(bld._wall_outward(np.array([(0, 0, 6), (1, 0, 6), (1, 1, 6)], np.float64), L_RING))


def l_house():
    """The L of L_RING (eave 6 m, flat roof at 6 m on both arms) as its LoD2 polygons."""
    roof = [(2, np.array([(0, 0, 6), (10, 0, 6), (10, 4, 6), (0, 4, 6)], np.float64)),
            (2, np.array([(0, 4, 6), (4, 4, 6), (4, 10, 6), (0, 10, 6)], np.float64))]
    return roof


def l_spec():
    spec = lod2_spec(L_RING.copy())
    spec.lod2 = {"polys": l_house(), "eave": 6.0, "top": 6.0, "pitched": False}
    return spec


class Lod2Outline(unittest.TestCase):
    """A LoD2 building's walls (and collision) stand on its measured outline, not on its box (2026-11-24: up to 61 %
    of the wall area of an L or U stood outside its roof)."""

    def test_drawn_ring_is_the_lod2_ring(self):
        spec = l_spec()
        self.assertIs(bld.drawn_ring(spec), spec.ring)
        plain = lod2_spec(L_RING.copy())
        plain.lod2 = None
        self.assertEqual(len(bld.drawn_ring(plain)), 4, "a plain pitched house keeps its box")

    def test_no_wall_in_the_l_s_open_corner(self):
        spec = l_spec()
        P = bld.build([(spec, 300.0, 300.0)])[0]
        inside_corner = (P[:, 0] > 4.5) & (P[:, 1] > 4.5)
        self.assertFalse(inside_corner.any(), "nothing drawn in the corner the L leaves open")
        cm = bld.collision_mesh([(spec, 300.0, 300.0)])
        CP = np.asarray(cm[0])
        self.assertFalse(((CP[:, 0] > 4.5) & (CP[:, 1] > 4.5)).any(), "no collision wall there either")


class Lod2OverRoad(unittest.TestCase):
    """A LoD2 building the road clearance makes a passage of: its walls from the passage's height, one LoD2 roof."""

    def test_min_h_lifts_the_walls_and_closes_the_underside(self):
        spec = lod2_spec()
        spec.min_h = 4.5
        P = bld.build([(spec, 300.0, 300.0)])[0]
        self.assertGreaterEqual(float(P[:, 2].min()), 304.5 - 1e-6, "nothing below the passage's height")
        self.assertAlmostEqual(float(P[:, 2].max()), 309.0, delta=0.01)

    def test_pieces_beside_the_road_do_not_redraw_the_lod2_roof(self):
        import geodesy
        spec = lod2_spec(rect(0, 0, 20, 8))         # eave 9 m: a passage under it (ROAD_CLEAR_PASSAGE_H + lintel)
        spec.lod2 = {"polys": [(2, np.array([(0, 0, 9), (20, 0, 9), (20, 8, 9), (0, 8, 9)], np.float64))],
                     "eave": 9.0, "top": 9.0, "pitched": False}
        spec.floors, spec.floor_h, spec.height = 3, 3.0, 9.0
        items = [(spec, 300.0, 300.0)]
        seg = np.array([[10.0, -20.0, 10.0, 30.0]])           # a road through its middle, 3 m each side
        half = np.array([3.0])
        drop = set()
        extra = bld._split_passages(items, [(0, np.asarray(spec.ring), False)], seg, half,
                                    geodesy.EnuFrame(48.8, 9.0), 0.0, 0.0, lambda la, lo: np.zeros(np.shape(la)),
                                    lambda s, k: None, drop)
        self.assertTrue(extra, "pieces beside the road")
        self.assertTrue(all(s.lod2 is None for s, _, _ in extra), "the pieces are plain blocks")
        self.assertEqual(drop, set())
        self.assertEqual(sum(1 for s, _, _ in items + extra if s.lod2), 1, "one LoD2 roof drawn")
        self.assertEqual(items[0][0].min_h, bld.ROAD_CLEAR_PASSAGE_H, "the LoD2 house goes on over the road")
        P = bld.build(items + extra)[0]
        on_road = (np.abs(P[:, 0] - 10.0) < 2.0) & (P[:, 2] < 304.5 - 1e-6)
        self.assertFalse(on_road.any(), "nothing on the road under the passage")


class Regions(unittest.TestCase):
    def test_baden_wuerttemberg_is_swabia_and_north_germany_is_not(self):
        self.assertEqual(bld.region_style(48.80, 9.01).name, "swabia")
        self.assertEqual(bld.region_style(53.5, 10.0).name, "europe_north")

    def test_leonberg_pitched_roofs_are_clay_tiles_not_slate(self):
        """BW (Leonberg): the swabia style's pitched roofs are red clay tiles only, no dark slate."""
        roofs = bld.region_style(48.80, 9.01).roofs
        self.assertFalse(any("slate" in mat for mat, _ in roofs))
        self.assertFalse(any(mat in (bld.ROOF_SLATE, bld.ROOF_SLATE_DARK) for mat, _ in roofs))
        for k in range(20):
            spec = bld.plan(rect(40 * k, 0, 10, 8), {"class": "house", "roof_shape": "gabled", "height": 9.0,
                                                     "roof_height": 3.0}, k / 20.0, 48.80, 9.01)
            self.assertTrue(spec.roof_mat.startswith(bld.ROOF_TILES + "|"), spec.roof_mat)

    def test_leonberg_houses_are_plaster_not_pastel(self):
        """BW (Leonberg): plastered houses are white, cream, sand or grey, only rarely tinted."""
        self.assertLessEqual(bld.region_style(48.80, 9.01).paint, 0.1)
        specs = [bld.plan(rect(40 * k, 0, 10, 8), {"class": "house", "height": 9.0}, k / 200.0, 48.80, 9.01)
                 for k in range(200)]
        tinted = [s for s in specs if s.facade in bld.PAINT_TINTS]
        self.assertLessEqual(len(tinted), 20, f"{len(tinted)} of 200 houses are tinted")


if __name__ == "__main__":
    unittest.main()
