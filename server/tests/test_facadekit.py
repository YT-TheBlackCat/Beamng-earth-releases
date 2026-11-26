"""facadekit: windows and parts fitted to a house's real walls (axes, party walls, door), outward faces, deterministic,
off outside its area and in far tiles."""
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import buildings as bld  # noqa: E402
import facadekit as fk  # noqa: E402

LAT, LON = 48.8013, 9.0137          # the old town of Leonberg (Fachwerk circle)
OUT_LAT, OUT_LON = 52.37, 4.90      # Amsterdam


def setUpModule():
    fk.KIT_NEW = True           # (the kit's own rules are tested on every epoch; NewHousesTest has the default)


def tearDownModule():
    fk.KIT_NEW = False


def rect(x, y, w, d):
    return np.array([(x, y), (x + w, y), (x + w, y + d), (x, y + d)], np.float64)


def house(ring, cls="house", lat=LAT, lon=LON, **attrs):
    a = dict({"class": cls, "num_floors": 3, "roof_shape": "gabled"}, **attrs)
    spec = bld.plan(ring, a, 0, lat, lon)
    spec.gring = np.zeros(len(spec.ring))
    spec.kit = fk.decide(spec, a, lat, lon)
    return spec, a


def tri_normals(P, T):
    P = np.asarray(P, float).reshape(-1, 3)
    T = np.asarray(T, np.int64).reshape(-1, 3)
    n = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    ln = np.linalg.norm(n, axis=1)
    return n / np.maximum(ln, 1e-12)[:, None], P[T].mean(1), ln


class NewHousesTest(unittest.TestCase):
    """2026-11-25: a house of 1950 or later keeps the sheet's solid window bands, in the epoch's light plaster."""

    def test_new_house_has_no_kit_but_light_plaster(self):
        ring = rect(0, 0, 11, 9)
        with mock.patch.object(fk, "KIT_NEW", False):
            spec, a = house(ring, lat=48.795, lon=9.03, _year=2015)
        self.assertIsNone(spec.kit)
        mat, rgb = bld.LOOKS[spec.facade]
        self.assertEqual(mat, bld.FACADE)
        self.assertGreater(min(rgb), 0.85)

    def test_old_house_keeps_the_kit(self):
        with mock.patch.object(fk, "KIT_NEW", False):
            spec, a = house(rect(0, 0, 11, 9), _year=1890)
        self.assertIsNotNone(spec.kit)

    def test_switch_brings_the_kit_back(self):
        with mock.patch.object(fk, "KIT_NEW", True):
            spec, a = house(rect(0, 0, 11, 9), lat=48.795, lon=9.03, _year=2015)
        self.assertEqual(spec.kit["ep"], "neu")

    def test_mapped_colour_is_left_alone(self):
        spec, a = house(rect(0, 0, 11, 9), lat=48.795, lon=9.03, _year=2015, facade_color="#8a6a50")
        before = spec.facade
        with mock.patch.object(fk, "KIT_NEW", False):
            self.assertIsNone(fk.decide(spec, a, 48.795, 9.03))
        self.assertEqual(spec.facade, before)

    def test_stone_on_a_new_house_becomes_plaster(self):
        spec, a = house(rect(0, 0, 11, 9), lat=48.795, lon=9.03, _year=2015)
        spec = bld._spec_copy(spec, wall=f"{bld.STONE}|0", facade=f"{bld.STONE}|0")
        with mock.patch.object(fk, "KIT_NEW", False):
            self.assertIsNone(fk.decide(spec, a, 48.795, 9.03))
        self.assertFalse(spec.wall)
        self.assertEqual(bld.LOOKS[spec.facade][0], bld.FACADE)


class AxesTest(unittest.TestCase):
    def test_even_bays_inside_the_corner_margins(self):
        cs = fk.axes(12.0, 3.0, 1.1, 0.9)
        self.assertEqual(len(cs), 3)
        self.assertGreaterEqual(cs[0] - 0.55, 0.9)
        self.assertLessEqual(cs[-1] + 0.55, 11.1)
        gaps = np.diff(cs)
        self.assertLess(gaps.max() - gaps.min(), 1e-9)

    def test_short_wall_none(self):
        self.assertEqual(fk.axes(2.2, 3.0, 1.1, 0.9), [])

    def test_one_axis_on_the_door(self):
        cs = fk.axes(12.0, 3.0, 1.1, 0.9, door=4.3)
        self.assertTrue(any(abs(c - 4.3) < 1e-9 for c in cs))
        self.assertGreaterEqual(min(cs) - 0.55, 0.9)
        self.assertLessEqual(max(cs) + 0.55, 11.1)

    def test_blind_stretch_leaves_no_window(self):
        cs = fk.axes(12.0, 3.0, 1.1, 0.9, blind=[(0.0, 6.0)])
        self.assertTrue(cs and min(cs) > 6.0)


class NeighbourTest(unittest.TestCase):
    def test_shared_wall_found_only_facing(self):
        a, d = np.array([10.0, 0.0]), np.array([0.0, 1.0])          # the east wall of rect(0, 0, 10, 8), going north
        nb = np.array([[10.0, 8.0, 10.0, 0.0]])                      # the neighbour's west wall, going south
        self.assertEqual(fk.blind_intervals(nb, a, d, 8.0), [(0.0, 8.0)])
        same = np.array([[10.0, 0.0, 10.0, 8.0]])                    # the same direction: not a wall against it
        self.assertEqual(fk.blind_intervals(same, a, d, 8.0), [])
        far = np.array([[14.0, 8.0, 14.0, 0.0]])
        self.assertEqual(fk.blind_intervals(far, a, d, 8.0), [])

    def test_partly_shared(self):
        a, d = np.array([10.0, 0.0]), np.array([0.0, 1.0])
        nb = np.array([[10.0, 12.0, 10.0, 5.0]])
        (lo, hi), = fk.blind_intervals(nb, a, d, 8.0)
        self.assertAlmostEqual(lo, 5.0)
        self.assertAlmostEqual(hi, 8.0)

    def test_mark_neighbours_collects_rings(self):
        s1, _ = house(rect(0, 0, 10, 8))
        s2, _ = house(rect(10, 0, 10, 8))
        far, _ = house(rect(100, 100, 10, 8))
        items = [(s1, 0.0, 0.0), (s2, 0.0, 0.0), (far, 0.0, 0.0)]
        fk.mark_neighbours(items, bld.drawn_ring)
        self.assertEqual(len(s1.kit["nb"]), 4)                      # the touching house's four walls
        self.assertIsNone(far.kit["nb"])


class DecideTest(unittest.TestCase):
    def test_old_town_house_is_kit(self):
        spec, _ = house(rect(0, 0, 9, 12), _year=1850)
        self.assertIsNotNone(spec.kit)
        self.assertEqual(spec.kit["fam"], "altstadt")
        self.assertTrue(spec.kit["wall"].startswith("bne_bld_"))

    def test_outside_the_area_or_off(self):
        spec, a = house(rect(0, 0, 9, 12), lat=OUT_LAT, lon=OUT_LON)
        self.assertIsNone(spec.kit)
        spec, a = house(rect(0, 0, 9, 12))
        with mock.patch.object(fk, "ON", False):
            self.assertIsNone(fk.decide(spec, a, LAT, LON))
        with mock.patch.object(fk, "EVERYWHERE", True):
            self.assertIsNotNone(fk.decide(spec, a, OUT_LAT, OUT_LON))

    def test_shop_shed_tower_part_not(self):
        spec, a = house(rect(0, 0, 9, 12))
        for change in (dict(shop=True), dict(cls="shed"), dict(floors=9), dict(min_h=3.0), dict(windows=False)):
            s = bld._spec_copy(spec, **change)
            self.assertIsNone(fk.decide(s, a, LAT, LON), change)
        self.assertIsNone(fk.decide(spec, a, LAT, LON, part=True))

    def test_big_old_town_building_stays_old_town(self):
        spec, a = house(rect(0, 0, 30, 20), cls="", num_floors=3)       # (the castle: no balconies)
        self.assertEqual(spec.kit["fam"], "altstadt")
        self.assertFalse(spec.kit["p"]["balcony"])

    def test_apartments_are_blocks(self):
        spec, a = house(rect(0, 0, 14, 40), cls="apartments", num_floors=5, roof_shape="flat")
        self.assertEqual(spec.kit["fam"], "block")

    def test_same_house_same_plan(self):
        s1, _ = house(rect(0, 0, 9, 12), seed_id="w123")
        s2, _ = house(rect(50, 50, 9, 12), seed_id="w123")
        self.assertEqual(s1.kit["p"], s2.kit["p"])


class EpochTest(unittest.TestCase):
    def test_epoch_of_year(self):
        got = [fk.epoch_of_year(y) for y in (1850, 1919, 1920, 1949, 1950, 1969, 1970, 1989, 1990, 2009, 2010, 2024)]
        self.assertEqual(got, ["alt", "alt", "zw", "zw", "nk", "nk", "s70", "s70", "s90", "s90", "neu", "neu"])
        self.assertIsNone(fk.epoch_of_year(None))
        self.assertIsNone(fk.epoch_of_year("early 1900s"))

    def test_mapped_year_decides_the_style(self):
        old, _ = house(rect(0, 0, 9, 12), _year=1880)
        new, _ = house(rect(0, 0, 9, 12), _year=2016, lat=48.82, lon=9.05)
        self.assertEqual((old.kit["ep"], old.kit["ep_src"]), ("alt", "year"))
        self.assertEqual((new.kit["ep"], new.kit["ep_src"]), ("neu", "year"))
        self.assertLess(old.kit["p"]["w"], 1.2)                      # old: narrow, tall windows ...
        self.assertGreater(new.kit["p"]["w"], 1.45)                  # ... new: wide ones
        self.assertFalse(new.kit["p"]["shutter"])

    def test_old_town_big_building_is_old(self):
        spec, _ = house(rect(0, 0, 30, 20), cls="", num_floors=3)
        self.assertEqual(spec.kit["ep"], "alt")

    def test_guess_follows_the_roof_and_the_place(self):
        def shares(roof, lat, lon, n=300):
            out = {}
            for i in range(n):
                s, _ = house(rect(0, 0, 9, 12), lat=lat, lon=lon, roof_shape=roof, seed_id=f"w{i}")
                out[s.kit["ep"]] = out.get(s.kit["ep"], 0) + 1
            return out
        flat = shares("flat", 48.82, 9.05)
        gabled = shares("gabled", 48.82, 9.05)
        old = shares("gabled", LAT, LON)
        self.assertGreater(flat.get("neu", 0) + flat.get("s70", 0), flat.get("alt", 0) + flat.get("zw", 0))
        self.assertGreater(old.get("alt", 0) + old.get("zw", 0), 2 * (gabled.get("alt", 0) + gabled.get("zw", 0)))
        self.assertGreater(old.get("alt", 0) + old.get("zw", 0), 0.4 * 300)

    def test_guess_is_stable(self):
        a, _ = house(rect(0, 0, 9, 12), seed_id="w77", lat=48.82, lon=9.05)
        b, _ = house(rect(40, 9, 9, 12), seed_id="w77", lat=48.82, lon=9.05)
        self.assertEqual(a.kit["ep"], b.kit["ep"])

    def test_roof_class_from_lod2(self):
        spec, _ = house(rect(0, 0, 8, 10))
        face = lambda z1: np.array([(0.0, 0.0, 6.0), (8.0, 0.0, 6.0), (8.0, 5.0, z1), (0.0, 5.0, z1)])
        for z1, want in ((10.6, "steep"), (8.5, "mid"), (6.9, "low")):
            spec.lod2 = {"pitched": True, "polys": [(2, face(z1))]}
            self.assertEqual(fk.roof_class(spec), want, z1)
        spec.lod2 = {"pitched": False, "polys": []}
        self.assertEqual(fk.roof_class(spec), "flat")

    def test_plaster_leans_to_the_epoch_unless_mapped(self):
        base = house(rect(0, 0, 10, 8), cls="apartments", _year=2016, lat=48.82, lon=9.05)[0]
        old = house(rect(0, 0, 10, 8), cls="apartments", _year=1850, lat=48.82, lon=9.05)[0]
        self.assertNotEqual(base.kit["wall"], old.kit["wall"])
        mapped = house(rect(0, 0, 10, 8), cls="apartments", _year=2016, lat=48.82, lon=9.05, facade_color="#aa5533")[0]
        keep = bld._spec_copy(mapped, kit=None)
        self.assertEqual(mapped.kit["wall"], fk.wall_look(keep))


class DressTest(unittest.TestCase):
    def mesh(self, spec, lite=False):
        out = bld.build([(spec, 0.0, 0.0)], lite)
        return out

    def test_kit_adds_faces_facing_out(self):
        spec, _ = house(rect(0, 0, 10, 8), cls="apartments", num_floors=3, roof_shape="flat")
        m = bld.Mesh()
        ring = bld.drawn_ring(spec)
        fk.dress(m, spec, ring, spec.floors * spec.floor_h, bld._Looks(spec.seed))
        P, UV, T, M, C = m.arrays()
        self.assertGreater(len(T), 40)
        n, cen, ln = tri_normals(P, T)
        e0, e1 = ring, np.roll(ring, -1, 0)
        for k in range(len(T)):
            if ln[k] < 1e-9 or abs(n[k][2]) > 0.5:
                continue                                              # (tops and undersides)
            q = cen[k][:2]
            d = e1 - e0
            t = np.clip(((q - e0) * d).sum(1) / (d * d).sum(1), 0, 1)
            near = int(np.argmin(np.hypot(*(e0 + d * t[:, None] - q).T)))
            out = np.array([d[near][1], -d[near][0]]) / np.hypot(*d[near])
            self.assertGreater(n[k][:2] @ out, -1e-6, f"triangle {k} faces into the building")

    def test_party_wall_has_no_windows(self):
        s1, _ = house(rect(0, 0, 10, 8), cls="apartments", num_floors=3, roof_shape="flat")
        s2, _ = house(rect(10, 0, 10, 8), cls="apartments", num_floors=3, roof_shape="flat")
        fk.mark_neighbours([(s1, 0.0, 0.0), (s2, 0.0, 0.0)], bld.drawn_ring)
        m = bld.Mesh()
        fk.dress(m, s1, bld.drawn_ring(s1), s1.floors * s1.floor_h, bld._Looks(s1.seed))
        P, UV, T, M, C = m.arrays()
        near_x10 = np.abs(P[T][:, :, 0] - 10.0) < 0.25                # parts standing on the shared wall (x = 10)
        flat_on_wall = near_x10.all(axis=1)
        self.assertFalse(flat_on_wall.any())
        m2 = bld.Mesh()
        s1.kit["nb"] = None
        fk.dress(m2, s1, bld.drawn_ring(s1), s1.floors * s1.floor_h, bld._Looks(s1.seed))
        P2, _, T2, _, _ = m2.arrays()
        self.assertTrue((np.abs(P2[T2][:, :, 0] - 10.0) < 0.25).all(axis=1).any())   # ... and there are some without it

    def test_deterministic(self):
        spec, _ = house(rect(0, 0, 9, 12))
        a, b = self.mesh(spec), self.mesh(spec)
        self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[2], b[2]))

    def test_build_uses_plain_walls_and_more_triangles(self):
        spec, a = house(rect(0, 0, 9, 12), cls="apartments", num_floors=3, roof_shape="flat")
        with_kit = self.mesh(spec)
        off = bld._spec_copy(spec, kit=None)
        without = self.mesh(off)
        self.assertGreater(len(with_kit[2]), len(without[2]))

    def test_lite_is_the_old_mesh(self):
        spec, _ = house(rect(0, 0, 9, 12))
        lite_kit = self.mesh(spec, lite=True)
        lite_off = self.mesh(bld._spec_copy(spec, kit=None), lite=True)
        self.assertTrue(np.array_equal(lite_kit[0], lite_off[0]))

    def test_door_edge_keeps_the_ground_floor_free(self):
        spec, _ = house(rect(0, 0, 10, 8), cls="apartments", num_floors=3, roof_shape="flat")
        spec.door = (0, 5.0)                                          # the south wall, in the middle
        m = bld.Mesh()
        fk.dress(m, spec, bld.drawn_ring(spec), spec.floors * spec.floor_h, bld._Looks(spec.seed))
        P, UV, T, M, C = m.arrays()
        low = P[T].mean(1)
        door_zone = (np.abs(low[:, 0] - 5.0) < 0.6) & (low[:, 1] < 0.5) & (low[:, 2] < 2.2)
        # (only the canopy or nothing: no window below the canopy over the door)
        self.assertFalse(door_zone.any())


class GableTest(unittest.TestCase):
    def test_gable_window_on_a_high_gable(self):
        spec, _ = house(rect(0, 0, 8, 10))
        Q = np.array([(0.0, 0.0, 6.0), (8.0, 0.0, 6.0), (4.0, 0.0, 10.0)])      # a gable on the south wall, eave 6 m
        spec.kit["nb"] = None
        m = bld.Mesh()
        fk.gable_window(m, spec, Q, np.array([0.0, -1.0]), 6.0, bld._Looks(spec.seed))
        P, _, T, _, _ = m.arrays()
        self.assertGreater(len(T), 0)
        self.assertTrue((P[:, 2] > 6.0).all() and (P[:, 2] < 10.0).all())
        self.assertTrue((np.abs(P[:, 0] - 4.0) < 1.0).all())

    def test_low_gable_none(self):
        spec, _ = house(rect(0, 0, 8, 10))
        Q = np.array([(0.0, 0.0, 6.0), (8.0, 0.0, 6.0), (4.0, 0.0, 7.2)])
        m = bld.Mesh()
        fk.gable_window(m, spec, Q, np.array([0.0, -1.0]), 6.0, bld._Looks(spec.seed))
        self.assertEqual(len(m.arrays()[2]), 0)


if __name__ == "__main__":
    unittest.main()
