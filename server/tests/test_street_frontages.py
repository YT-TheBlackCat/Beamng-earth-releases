import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import earthd
import fences
import geodesy
import mapgeom
import roads
import sidewalks
import streetfront
import streetparking
import surface


FRAME = geodesy.EnuFrame(48.82, 7.78)
BOX = (-6., -12., 30., 28.)
STREET = roads.Spec(6., "paved", True, "residential")
STREETS = [("street", STREET, np.array([[-10., 0., 0.], [35., 0., 0.]]), 7.78)]


def flat(x, y):
    return np.zeros(len(x))


def feature(line, cls, kind, tags=None):
    line = np.asarray(line)
    lat, lon, _ = FRAME.enu_to_geodetic(line[:, 0], line[:, 1], 0.)
    return mapgeom.Features("infrastructure", np.array(["one"]), np.array([kind]), np.array([0, 1]),
                            np.array([0, 1]), np.array([0, len(line)]), np.c_[lon, lat],
                            attrs={"class": np.array([cls]), "tags": np.array([json.dumps(tags or {})])})


def rectangle(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])


class StreetParkingTest(unittest.TestCase):
    def test_narrow_mapped_bays_supply_markings_and_slots_in_the_same_place(self):
        rect = rectangle(1., 3.5, 15.6, 5.66)
        features = [feature(rect, "parking", mapgeom.POLYGON, {
            "parking": "street_side", "orientation": "parallel", "capacity": "3"})]
        args = (FRAME, BOX, features, STREETS, [], [], [], [], [], [], flat)
        rows, marks = streetparking.tile_layout(*args)
        self.assertEqual(len(rows), 3)
        self.assertTrue(marks)
        self.assertTrue(all(r[4] == 0 and r[5:] == [2., 4.65] and abs(r[3] - 180.) < .01 for r in rows))
        for mark in marks:
            self.assertTrue(np.all((np.asarray(mark["n"])[:, 1] >= 3.5) & (np.asarray(mark["n"])[:, 1] <= 5.66)))
        left, _ = streetparking.tile_layout(*args, left_hand=True)
        self.assertTrue(all(abs(r[3]) < .01 for r in left))

    def test_mapped_driveway_only_parks_clear_of_street_and_footway(self):
        drive = ("drive", roads.Spec(3.1, "paved", True, "service"), np.array([[8., 0., 0.], [8., 15., 0.]]))
        home = rectangle(4., 16., 13., 24.)
        walk = {"pts": np.array([[0., 4., 0.], [16., 4., 0.]]), "w": 1.5}
        rows, marks = streetparking.tile_layout(FRAME, BOX, [], STREETS, [drive], [walk], [home], [home],
                                                [], [], flat)
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0][1], 9.)
        self.assertEqual(rows[0][4], 1)
        self.assertEqual(marks, [])
        barrier = np.array([[5., 12.], [11., 12.]])
        rows, _ = streetparking.tile_layout(FRAME, BOX, [], STREETS, [drive], [walk], [home], [home],
                                            [barrier], [], flat)
        self.assertEqual(rows, [])


class PavementTest(unittest.TestCase):
    def test_default_asphalt_reaches_a_setback_without_a_sidewalk_but_spares_green(self):
        import roadmesh

        house = rectangle(0., 11., 24., 21.)
        args = (FRAME, [], [house], [], STREETS, [], BOX)
        paved = streetfront.chunks(*args)
        self.assertTrue(paved)
        self.assertEqual({c["m"] for c in paved}, {roadmesh.FRONTAGE_ASPHALT})
        self.assertAlmostEqual(max(max(y for _, y in c["poly"]) for c in paved), 11. - .07)
        self.assertFalse(streetfront.chunks(*args, extend=False, residential=False))
        green = streetfront.PhotoPaving(np.full((80, 80, 3), [65, 100, 60], np.uint8), -16., -16., 60.)
        grass = []
        self.assertFalse(streetfront.chunks(*args, photo=green, grass=grass))
        self.assertTrue(grass)

    def test_terrain_apron_pads_do_not_remove_lawn_outside_the_pavement(self):
        rings = streetfront.pad_items(FRAME, BOX, "tile", [
            streetfront._paving(rectangle(2., 4., 6., 8.)[:-1], "concrete")],
            [rectangle(4., 4., 8., 8.)[:-1]])
        y, x = np.mgrid[0:16:2., 0:16:2.]
        lat, lon, _ = FRAME.enu_to_geodetic(x, y, 0.)
        ground = np.full(x.shape, surface.DIRT, np.uint8)
        sf = surface.Surface(None)
        self.addCleanup(sf.close)
        self.assertTrue(sf.add_fuel_pads(rings))
        self.assertFalse(sf.add_fuel_pads(rings))
        self.assertFalse(sf.fuel_pads(lat, lon))
        sf.clear_fuel_pads()
        paved = sf.paint_ground_pads(ground, lat, lon)
        self.assertTrue(np.all(paved[2:4, 1:3] == surface.ASPHALT))
        self.assertEqual(np.count_nonzero(paved == surface.ASPHALT), 4)
        self.assertTrue(np.all(paved[2:4, 3] == surface.GRASS))
        self.assertEqual(np.count_nonzero(paved == surface.GRASS), 2)
        self.assertTrue(sf.add_fuel_pads([(key, []) for key, _ in rings]))
        np.testing.assert_array_equal(sf.paint_ground_pads(ground, lat, lon), ground)

    def test_paved_widening_keeps_curb_position_and_respects_gardens(self):
        rgb = np.full((128, 128, 3), 139, np.uint8)
        photo = streetfront.PhotoPaving(rgb, -10., -12., 40.)
        house = rectangle(0., 8.3, 24., 16.)
        points = np.array([[1., 3.75, 0.], [20., 3.75, 0.]])
        walk = {"pts": points, "side": 1, "w": 1.5, "europe": True}
        curb, _ = sidewalks.run_geometry(points, 1, 1.5)
        self.assertTrue(streetfront.widen_walks([walk], photo, [house], [], [], STREETS))
        widened, _ = sidewalks.run_geometry(walk["pts"], 1, walk["w"])
        self.assertGreater(walk["w"], 3.)
        np.testing.assert_allclose(widened[[0, -1], :2], curb[:, :2])
        beside_garden = {"pts": points, "side": 1, "w": 1.5, "europe": True}
        garden = rectangle(0., 4.6, 24., 8.2)
        self.assertFalse(streetfront.widen_walks([beside_garden], photo, [house], [garden], [], STREETS))

    def test_pending_road_paint_can_refresh_after_the_data_arrives(self):
        pending = set()
        cached = SimpleNamespace(roads=lambda la, lo: ([], False))
        rgb = np.zeros((16, 16, 3), np.uint8)
        surface.paint_square(rgb, FRAME, 0., 0., 20., cached, simple=True, pending=pending)
        # (every photo, not only with the European road details on: Discord's gravel mountain roads)
        self.assertEqual(pending, {surface.PHOTO_ROADS_PENDING})
        done = SimpleNamespace(roads=lambda la, lo: ([], True))
        later = set()
        surface.paint_square(rgb, FRAME, 0., 0., 20., done, simple=True, pending=later)
        self.assertEqual(later, set())          # (its cells here: nothing to wait for)
        key = (1, 0, 0)
        server = SimpleNamespace(
            base_pending={}, base_stale=set(), layout=SimpleNamespace(block=16),
            imagery=SimpleNamespace(lazy_ready=lambda *args: True),
            mapdata=SimpleNamespace(has=lambda *args: True),
        )
        server.base_ready = lambda a, x, y: earthd.Server.base_ready(server, a, x, y)
        anchor = SimpleNamespace(id=1, frame=FRAME)
        earthd.Server._base_kept(server, anchor, 0, 0, pending)
        self.assertIn(key, server.base_stale)
        self.assertTrue(server.base_ready(anchor, 0, 0))
        server.mapdata.has = lambda *args: False
        self.assertFalse(server.base_ready(anchor, 0, 0))


class FrenchBoundaryTest(unittest.TestCase):
    def test_hedge_crowns_clear_the_road_and_the_crossing(self):
        along = feature([[0., 3.2], [14., 3.2]], "hedge", mapgeom.LINE)
        across = feature([[0., -.2], [0., 9.]], "hedge", mapgeom.LINE)
        with patch("house_regions.profile", return_value="alsace"):
            for hedge in (along, across):
                rows = fences.rows(FRAME, lambda _: hedge, BOX, flat, french=True, carriageways=STREETS)
                self.assertTrue(rows)
                self.assertTrue(all(r[2] >= 3. + r[6] * fences.HEDGE_CROWN_PER_M + .14 for r in rows))
                if hedge is across:
                    self.assertTrue(all(abs(r[1]) < .01 for r in rows))

    def test_regional_models_follow_mapped_boundaries_only(self):
        line = feature([[0., 10.], [10., 10.]], "fence", mapgeom.LINE, {"material": "wood"})
        with patch("house_regions.profile", return_value="alsace"):
            self.assertFalse(fences.rows(FRAME, lambda _: line, BOX, flat))
            rows = fences.rows(FRAME, lambda _: line, BOX, flat, french=True)
            self.assertTrue(rows)
            self.assertTrue(all(row[0] == "fr_fence_wood" and abs(row[2] - 10.) < .02 for row in rows))
            self.assertFalse(fences.rows(FRAME, lambda _: None, BOX, flat, french=True))
        self.assertEqual(fences.french_panel("wall", {}, "north", "one"), "brick")
        self.assertEqual(fences.french_panel("wall", {}, "central", "one"), "dark_stone")

    def test_unmapped_garden_gets_a_boundary_but_an_access_stays_open(self):
        ring = rectangle(4., 12., 17., 23.)[:-1]
        house = SimpleNamespace(door=(0, 6.5), seed=99)
        green = streetfront.PhotoPaving(np.full((80, 80, 3), [65, 100, 60], np.uint8), -16., -16., 60.)
        args = (FRAME, [(house, ring)], [ring], [], STREETS, [], [], [])
        with patch("furniture._h01", return_value=.3), patch("house_regions.profile", return_value="alsace"):
            rows = fences.inferred_rows(*args, [], flat, green)
            self.assertGreaterEqual(len(rows), 4)
            self.assertTrue(all(abs(r[1] - 10.5) > 1.6 and r[2] == 4. for r in rows))
            drive = {"n": [[6., 0., 0.], [6., 13., 0.]], "w": 2.8}
            self.assertFalse(fences.inferred_rows(*args, [drive], flat, green))
            house.door = None
            self.assertTrue(fences.inferred_rows(*args, [], flat, None))
            self.assertFalse(fences.inferred_rows(*args, [drive], flat, None))
        hedge = feature([[0., 10.], [10., 10.]], "hedge", mapgeom.LINE)
        with patch("house_regions.profile", return_value="alsace"):
            rows = fences.rows(FRAME, lambda _: hedge, BOX, flat, french=True)
        self.assertGreaterEqual(len(rows), 8)
        self.assertEqual({row[0] for row in rows}, {fences.G_HEDGE})
        self.assertTrue(all(1.6 <= row[6] <= 2. for row in rows))


class GermanBoundaryTest(unittest.TestCase):
    """Aufgabe 5 (TODO_UMGEBUNG): mapped fences, garden walls and hedges in Germany (fences.german_rows)."""
    BOX = (-6., -12., 30., 28.)

    def rows(self, line, cls="fence", tags=None, **parts):
        feats = feature(line, cls, mapgeom.LINE, tags)
        return fences.german_rows(FRAME, [feats], self.BOX, lambda x, y: np.full(len(x), 2.), **parts)

    def test_a_mapped_fence_gets_a_panel_model_on_the_line(self):
        rows = self.rows([[0., 10.], [10., 10.]], tags={"material": "wood"})
        self.assertTrue(rows)
        self.assertTrue(all(r[0] == "fr_fence_wood" and abs(r[2] - 10.) < .02 and r[3] >= 2. for r in rows))
        self.assertAlmostEqual(float(np.mean([r[1] for r in rows])), 5., delta=.5)

    def test_model_follows_type_and_material_tags(self):
        pick = lambda cls, tags: fences.german_panel(cls, tags, "w1")
        self.assertEqual(pick("wall", {"barrier": "wall"}), "limestone")
        self.assertEqual(pick("wall", {"material": "brick"}), "brick")
        self.assertEqual(pick("fence", {"fence_type": "railing"}), "iron")
        self.assertEqual(pick("fence", {"fence_type": "wire"}), "mesh")
        self.assertEqual(pick("fence", {"material": "metal"}), "rails")
        self.assertEqual(pick("hedge", {}), "hedge")
        for tags in ({"wall": "retaining_wall"}, {"wall": "castle_wall"}):
            self.assertIsNone(pick("wall", tags))
        self.assertIsNone(pick("bollard", {}))
        mix = {pick("fence", {}) if False else fences.german_panel("fence", {}, "id%d" % n) for n in range(60)}
        self.assertTrue(mix <= {"mesh", "wood", "iron", "rails", "slats"} and len(mix) >= 4)
        self.assertEqual(fences.german_panel("fence", {}, "id7"), fences.german_panel("fence", {}, "id7"))

    def test_mapped_height_scales_the_panel_and_a_tall_way_is_left_out(self):
        low = self.rows([[0., 10.], [12., 10.]], tags={"material": "wood", "height": "0.6"})
        normal = self.rows([[0., 10.], [12., 10.]], tags={"material": "wood"})
        self.assertTrue(all(abs(r[6] - .6) < .12 for r in low))
        self.assertGreater(len(low), len(normal) * 1.5)      # (a lower panel is shorter: more of them)
        self.assertFalse(self.rows([[0., 10.], [12., 10.]], "wall", {"barrier": "wall", "height": "3"}))

    def test_hedge_rows_clear_the_road_and_nothing_stands_on_it(self):
        rows = self.rows([[0., 3.2], [14., 3.2]], "hedge", carriageways=STREETS)
        self.assertTrue(rows and all(r[0] == fences.G_HEDGE for r in rows))
        self.assertTrue(all(r[2] >= 3. + r[6] * fences.HEDGE_CROWN_PER_M + .14 for r in rows))
        across = self.rows([[10., -5.], [10., 14.]], tags={"material": "wood"}, carriageways=STREETS)
        self.assertTrue(across)
        self.assertTrue(all(abs(r[2]) > 3.2 for r in across))      # (no panel in the carriageway: gap at the road)

    def test_a_mapped_gate_a_driveway_and_a_footway_leave_gaps(self):
        base = self.rows([[0., 10.], [24., 10.]], tags={"material": "wood"})
        gate = feature([[12., 10.]], "gate", mapgeom.POINT)
        gate.ids = np.array(["gate"])
        line = feature([[0., 10.], [24., 10.]], "fence", mapgeom.LINE, {"material": "wood"})
        both = fences.german_rows(FRAME, [line, gate], self.BOX, flat)
        self.assertTrue(base and both and len(both) < len(base))
        self.assertFalse([r for r in both if abs(r[1] - 12.) < 1.3])
        drive = {"n": [[12., 4.], [12., 16.]], "w": 3.}
        open_ = self.rows([[0., 10.], [24., 10.]], tags={"material": "wood"}, drives=[drive])
        self.assertFalse([r for r in open_ if abs(r[1] - 12.) < 1.5])
        walk = {"pts": np.array([[4., 0., 0.], [4., 20., 0.]]), "w": 1.6}
        beside = self.rows([[0., 10.], [24., 10.]], tags={"material": "wood"}, walks=[walk])
        self.assertFalse([r for r in beside if abs(r[1] - 4.) < .8])
        self.assertTrue([r for r in beside if r[1] > 8.])

    def test_nothing_stands_in_a_building(self):
        house = rectangle(8., 8., 16., 14.)
        rows = self.rows([[0., 10.], [24., 10.]], tags={"material": "wood"}, rings=[house])
        self.assertTrue(rows)
        self.assertFalse([r for r in rows if 8.3 < r[1] < 15.7])
        self.assertTrue([r for r in rows if r[1] < 7.5] and [r for r in rows if r[1] > 16.5])

    def test_switch_off_and_ownership(self):
        with patch.dict(os.environ, {"BNE_DE_BOUNDARIES": "off"}):
            self.assertFalse(self.rows([[0., 10.], [10., 10.]]))
        rows = self.rows([[0., 10.], [60., 10.]], tags={"material": "wood"})
        self.assertTrue(rows and all(r[1] < 30. for r in rows))     # (the part past the box belongs to the next tile)
        self.assertEqual(rows, self.rows([[0., 10.], [60., 10.]], tags={"material": "wood"}))

    def test_the_triangle_budget_cuts_a_long_boundary(self):
        with patch.object(fences, "DE_TRI_BUDGET", 5000):
            rows = self.rows([[-6., 20.], [30., 20.], [30., 26.], [-6., 26.]], tags={"fence_type": "wire"})
        self.assertTrue(rows)
        self.assertLessEqual(len(rows) * fences.DE_TRIS["mesh"], 5000)

    def test_a_town_wall_is_a_thick_stone_wall_in_germany_only(self):
        for lat, lon, expect in ((48.8014, 9.0152, True), (48.82, 7.78, False)):
            frame = geodesy.EnuFrame(lat, lon)
            line = np.array([[0., 10.], [20., 10.]])
            la, lo, _ = frame.enu_to_geodetic(line[:, 0], line[:, 1], 0.)
            feats = mapgeom.Features("infrastructure", np.array(["one"]), np.array([mapgeom.LINE]), np.array([0, 1]),
                                     np.array([0, 1]), np.array([0, 2]), np.c_[lo, la],
                                     attrs={"class": np.array(["city_wall"]),
                                            "tags": np.array([json.dumps({"barrier": "city_wall"})])})
            with patch("streetfront.country_at_tile", return_value="DE" if expect else "FR"):
                walls = fences.stone_walls(frame, lambda _k: feats, (-5., -5., 30., 30.),
                                           lambda la_, lo_: np.zeros(np.shape(la_)))
            self.assertEqual(bool(walls), expect)
            if expect:
                self.assertTrue(np.allclose(walls[0]["top"] - walls[0]["foot"], fences.CITY_H_M + fences.STONE_FOOT_M))
                self.assertTrue(np.allclose(walls[0]["thick"], fences.CITY_THICK_M))


class CityFrontageTest(unittest.TestCase):
    """Outside Europe's details a city's sidewalk runs to the buildings' faces (city_frontage), houses keep lawns."""

    WALK = {"pts": np.array([[-8., 4.25, 0.], [33., 4.25, 0.]]), "side": 1, "w": 2.5}     # (curb 3.0, back 5.5)

    def test_the_sidewalk_itself_is_widened_to_the_walls(self):
        walk = dict(self.WALK, pts=np.array([[x, 4.25, 0.] for x in np.arange(-8., 33.5, 1.5)]))
        shops = [rectangle(-8., 8., 10., 18.), rectangle(10., 9., 33., 18.)]      # (fronts 2.5 and 3.5 m back)
        walks = [walk]
        self.assertEqual(streetfront.widen_city_walks(walks, shops, shops, [], [], STREETS), 1)
        curb, back = streetfront._walk_outline(walks[0])
        self.assertAlmostEqual(float(np.median(curb[:, 1])), 3.0, places=2, msg="the curb stays")
        self.assertGreaterEqual(float(np.median(back[:, 1])), 8.5, "the slab's back at (or in) the walls")
        self.assertGreater(walks[0]["w"], 5.0)
        lawn = [dict(self.WALK, pts=walk["pts"].copy(), w=2.5)]
        green = rectangle(-8., 6.5, 33., 7.5)
        streetfront.widen_city_walks(lawn, shops, shops, [green], [], STREETS)
        self.assertLess(lawn[0]["w"], 2.5 + 1.01, "a mapped lawn in front: no further than it")
        self.assertEqual(streetfront.widen_city_walks([dict(self.WALK)], [], [], [], [], STREETS), 0)

    def test_city_buildings_and_houses(self):
        from types import SimpleNamespace as NS
        ring = rectangle(0., 0., 10., 10.)[:-1]
        def spec(cls, floors, shop=False):
            return (NS(ring=ring, cls=cls, floors=floors, shop=shop, deck=None, min_h=0., outline=None), None, None)
        with patch("buildings.drawn_ring", lambda s: s.ring):
            pick = lambda items, dense: len(streetfront.city_targets(items, np.zeros(2), dense))
            self.assertEqual(pick([spec("retail", 1), spec("", 4), spec("", 1, shop=True)], False), 3)
            self.assertEqual(pick([spec("house", 2), spec("", 2), spec("detached", 1)], False), 0)
            self.assertEqual(pick([spec("", 3)], True), 1)
            self.assertEqual(pick([spec("terrace", 2), spec("", 2)], True), 0, "row houses of two floors: their yards")
            self.assertEqual(pick([spec("shed", 1), spec("greenhouse", 1)], True), 0)


class GroundFitCornerTest(unittest.TestCase):
    """Widening stops where the ground under the widened back leaves the design band (E-F1), and a corner's
    outer points follow a widened run at its joint (E-F4)."""

    def test_widening_stops_where_the_ground_leaves_the_design_band(self):
        curb = np.c_[np.arange(0., 10.5, .5), np.zeros(21), np.zeros(21)]
        out = np.tile([0., 1.], (21, 1))
        flat_core = lambda x, y: np.where(y <= 2.15, 0., .5 * (y - 2.15))          # 1:2 rising past 2.15 m
        fit = streetfront._ground_fit(curb, out, 2., .15, 3., flat_core)
        self.assertGreaterEqual(fit, .5)
        self.assertLessEqual(fit, .75, "the back stops at 2.75 m: 0.6 m past the flat core")
        self.assertEqual(streetfront._ground_fit(curb, out, 2., .15, 3., lambda x, y: np.zeros(len(x))), 3.)
        self.assertEqual(streetfront._ground_fit(curb, out, 2., .15, 3., None), 3.)

    def test_a_steep_ground_keeps_the_city_sidewalk_as_it_is(self):
        steep = lambda x, y: 3. * np.asarray(y, np.float64)
        walk = dict(GroundFitCornerTest.WALK, pts=np.array([[x, 4.25, 0.] for x in np.arange(-8., 33.5, 1.5)]))
        shops = [rectangle(-8., 8., 10., 18.), rectangle(10., 9., 33., 18.)]
        self.assertEqual(streetfront.widen_city_walks([walk], shops, shops, [], [], STREETS, z_at=steep), 0)
        self.assertEqual(walk["w"], 2.5)

    WALK = {"pts": np.array([[-8., 4.25, 0.], [33., 4.25, 0.]]), "side": 1, "w": 2.5}

    def test_a_widened_run_moves_the_corner_point_and_the_neighbour_to_the_new_back_edge(self):
        arc = np.array([[6., 5., 0.], [5., 3., 0.], [4., -1.25, 0.]])          # the arc's end at the run's curb
        outer = [(4., 1.25, 0.5), (6., 1.25, 0.5), (6., 8., 0.5)]                # E, the house corner M, F
        corner = {"corner": {"arc": arc, "outer": list(outer)}}
        run = {"pts": np.array([[-10., 1., 0.], [4., 1., 0.]]), "side": 1, "w": 4.5, "caps": (True, False)}
        walks = [run, corner]
        self.assertEqual(streetfront.fit_corners(walks), 1)
        new = walks[1]["corner"]["outer"]
        np.testing.assert_allclose(new[0][:2], [4., 3.25], atol=1e-6)
        np.testing.assert_allclose(new[1][:2], [6., 3.25], atol=1e-6, err_msg="the house corner on the intersection")
        self.assertEqual(new[0][2], 0.5, "heights kept")
        unwidened = [dict(run, w=2.5, pts=np.array([[-10., 0., 0.], [4., 0., 0.]])), {"corner": {"arc": arc,
                     "outer": list(outer)}}]
        self.assertEqual(streetfront.fit_corners(unwidened), 0)
        self.assertEqual(unwidened[1]["corner"]["outer"], list(outer))

    def test_the_old_town_sidewalk_runs_to_the_wall_and_a_front_garden_stops_it(self):
        walk = dict(GroundFitCornerTest.WALK, pts=np.array([[x, 4.25, 0.] for x in np.arange(-8., 33.5, 1.5)]))
        walls = [rectangle(-8., 7., 10., 17.), rectangle(10., 7., 33., 17.)]          # 1.5 m behind the back edge
        walled = dict(walk)
        kw = dict(front_m=streetfront.WALL_FRONT_M, hit=streetfront.WALL_HIT, pct=streetfront.WALL_PCT,
                  max_w=streetfront.WALL_WALK_MAX, min_extra=streetfront.WALL_MIN_EXTRA_M)
        self.assertEqual(streetfront.widen_city_walks([walled], walls, walls, [], [], STREETS, **kw), 1)
        self.assertAlmostEqual(walled["w"], 4.0, places=2, msg="the back at the wall")
        gardens = [rectangle(-8., 10.5, 10., 17.), rectangle(10., 10.5, 33., 17.)]   # 5 m behind: front gardens
        front = dict(walk)
        self.assertEqual(streetfront.widen_city_walks([front], gardens, gardens, [], [], STREETS, **kw), 0)
        self.assertEqual(front["w"], 2.5)


class GroundFitNoGroundTest(unittest.TestCase):
    def test_no_finite_ground_at_the_curb_or_back_means_it_fits(self):
        curb = np.c_[np.arange(0., 10.5, .5), np.zeros(21), np.zeros(21)]
        out = np.tile([0., 1.], (21, 1))
        no_curb_ground = lambda x, y: np.where(np.asarray(y) < .5, np.nan, 0.)      # NaN at the curb (y = 0)
        self.assertEqual(streetfront._ground_fit(curb, out, 2., .15, 3., no_curb_ground), 3.)
        all_nan = lambda x, y: np.full(len(x), np.nan)
        self.assertEqual(streetfront._ground_fit(curb, out, 2., .15, 3., all_nan), 3.)
