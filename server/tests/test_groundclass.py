"""Ground classes read from the state orthophoto (tools/groundbake.py: the classifier; groundclass.py: what the server takes
from the baked map): classes by band, outlines, the physics layers, the paved areas' meshes, the road verdicts."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))   # (the bake classifier lives with its tool)
import geodesy  # noqa: E402
import groundclass as gc  # noqa: E402
import plazas  # noqa: E402

try:
    import scipy  # noqa: F401
    import groundbake as gb  # noqa: E402
    HAVE_SCIPY = True
except ImportError:         # (the bake tool's classifier only: the running server does not need scipy)
    HAVE_SCIPY = False

FRAME = geodesy.EnuFrame(48.79838, 9.01353)         # (Leonberg's Neues Rathaus)


def block(rgbn, ndom, size=40):
    """A uniform photo block of size x size cells (R, G, B, NIR) with a uniform object height."""
    rgbi = np.empty((size, size, 4), np.float32)
    rgbi[:] = np.asarray(rgbn, np.float32)
    return rgbi, np.full((size, size), float(ndom), np.float32)


@unittest.skipUnless(HAVE_SCIPY, "scipy (the bake tool's)")
class ClassifierTest(unittest.TestCase):
    """The numbers are what the photo shows around Leonberg's Neues Rathaus (groundbake.py's docstring)."""

    def cls(self, rgbn, ndom=0.0):
        c = gb.classify(*block(rgbn, ndom))
        return int(np.bincount(c.ravel()).argmax())

    def test_hard_ground(self):
        self.assertEqual(self.cls((165, 165, 166, 168)), gc.ASPHALT, "neutral grey: the road's asphalt")
        self.assertEqual(self.cls((215, 212, 204, 214)), gc.STONE, "light and warm: the plaza's paving stones")
        self.assertEqual(self.cls((180, 90, 70, 140)), gc.RED, "red paving")

    def test_plants_by_height(self):
        grass = (90, 125, 60, 200)
        self.assertEqual(self.cls(grass, 0.2), gc.GRASS)
        self.assertEqual(self.cls(grass, 1.6), gc.SHRUB)
        self.assertEqual(self.cls(grass, 9.0), gc.TREE)

    def test_roof_is_an_object(self):
        self.assertEqual(self.cls((150, 110, 100, 140), 8.0), gc.OBJECT)

    def test_bare_ground(self):
        self.assertIn(self.cls((150, 125, 95, 160)), (gc.GRAVEL, gc.SOIL), "beige, warm and saturated")

    def test_large_bare_patch_is_a_field(self):
        c = gb.classify(*block((150, 125, 95, 160), 0.0, size=240))     # 240 cells = 96 m a side
        self.assertEqual(int(np.bincount(c.ravel()).argmax()), gc.SOIL)

    def test_shade_takes_the_lit_neighbours_class(self):
        rgbi, nd = block((215, 212, 204, 214), 0.0, size=60)
        rgbi[:, 30:] = (40, 42, 50, 45)                 # the right half in shade
        c = gb.classify(rgbi, nd)
        self.assertEqual(int(c[30, 40]), gc.STONE)      # (within SHADE_FILL_M of the lit paving)


@unittest.skipUnless(HAVE_SCIPY, "scipy (the bake tool's)")
class OutlineTest(unittest.TestCase):
    def test_rectangle_with_a_hole(self):
        m = np.zeros((120, 160), bool)
        m[20:100, 20:140] = True
        m[50:70, 60:90] = False
        (outer, holes), = gb.polygons(m, min_area_m2=1.0, min_hole_m2=1.0, sigma=0.5)
        self.assertEqual(len(holes), 1)
        self.assertAlmostEqual(abs(gb._area(outer)) * gc.BAKE_CELL_M ** 2, 80 * 120 * 0.16, delta=40.0)
        self.assertAlmostEqual(abs(gb._area(holes[0])) * gc.BAKE_CELL_M ** 2, 20 * 30 * 0.16, delta=15.0)

    def test_small_areas_dropped(self):
        m = np.zeros((60, 60), bool)
        m[10:14, 10:14] = True              # 1.6 m x 1.6 m
        self.assertEqual(gb.polygons(m, min_area_m2=25.0), [])

    def test_two_areas(self):
        m = np.zeros((100, 200), bool)
        m[10:60, 10:60] = True
        m[10:60, 120:180] = True
        self.assertEqual(len(gb.polygons(m, min_area_m2=10.0)), 2)


def make_map(path, classes, rings=None):
    """A baked map whose lower left corner is Leonberg's (E, N) = (501000, 5405000); classes at CELL_M."""
    rings = rings or []
    pts = [np.asarray(r[0], np.float32) for r in rings]
    np.savez_compressed(path, version=np.int64(gc.VERSION), classes=np.asarray(classes, np.uint8), e0=np.float64(501000),
                        n0=np.float64(5405000),
                        rings=np.concatenate(pts) if pts else np.zeros((0, 2), np.float32),
                        ring_class=np.asarray([r[1] for r in rings], np.uint8),
                        ring_off=np.cumsum([0] + [len(p) for p in pts]).astype(np.int64),
                        ring_hole=np.full(len(rings), -1, np.int64))


class MapTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "g.npz")
        n = 100                              # 80 m x 80 m
        cls = np.full((n, n), gc.ASPHALT, np.uint8)
        cls[:50, :50] = gc.STONE             # the north west quarter (rows count from the north)
        cls[50:, 50:] = gc.GRASS
        make_map(self.path, cls, [(np.array([[10, 50], [30, 50], [30, 70], [10, 70]]), gc.STONE)])
        gc.reset()
        self.old = gc.DATA
        gc.DATA = self.path
        self.gm = gc.load()
        e, n_ = 501000.0, 5405000.0
        self.at = lambda de, dn: int(self.gm.at(e + de, n_ + dn))

    def tearDown(self):
        gc.DATA = self.old
        gc.reset()
        self.dir.cleanup()

    def test_samples(self):
        self.assertEqual(self.at(10, 70), gc.STONE)
        self.assertEqual(self.at(70, 10), gc.GRASS)
        self.assertEqual(self.at(70, 70), gc.ASPHALT)
        self.assertEqual(self.at(-5, 10), gc.NONE, "off the map")

    def test_rings_in(self):
        got = self.gm.rings_in(501000, 5405000, 501100, 5405100)
        self.assertEqual(len(got), 1)
        outer, holes, c = got[0]
        self.assertEqual(c, gc.STONE)
        self.assertAlmostEqual(float(outer[:, 0].min()), 501010.0, delta=1e-3)
        self.assertEqual(self.gm.rings_in(502000, 5405000, 502100, 5405100), [])

    def test_paving_rings_are_lon_lat_and_prefixed(self):
        import lod2geom
        la, lo = lod2geom.wgs_from_utm32(501020.0, 5405060.0)
        got = gc.paving_rings(float(la) - 0.001, float(lo) - 0.001, float(la) + 0.001, float(lo) + 0.001)
        self.assertEqual(len(got), 1)
        ring, holes, surface = got[0]
        self.assertTrue(surface.startswith(gc.PHOTO_PREFIX))
        self.assertAlmostEqual(float(ring[:, 0].mean()), float(lo), delta=0.0005)
        self.assertAlmostEqual(float(ring[:, 1].mean()), float(la), delta=0.0005)

    def test_layers_paint_built_up_land_and_paving_anywhere(self):
        import lod2geom
        e = 501000.0 + np.array([[10.0, 70.0], [10.0, 70.0]])
        n = 5405000.0 + np.array([[70.0, 10.0], [70.0, 10.0]])
        la, lo = lod2geom.wgs_from_utm32(e, n)
        layer = {gc.STONE: 7, gc.GRASS: 3}
        idx = np.full((2, 2), 1, np.uint8)
        built = np.full((2, 2), 50)
        rural = np.full((2, 2), 40)
        got = gc.apply_layers(idx, built, la, lo, layer)
        self.assertEqual(got[0, 0], 7)
        self.assertEqual(got[0, 1], 3)
        got = gc.apply_layers(idx, rural, la, lo, layer)
        self.assertEqual(got[0, 0], 7, "paving stones also outside built-up land")
        self.assertEqual(got[0, 1], 1, "grass only where built-up")

    def test_utm_grid_matches_the_exact_conversion(self):
        n = 128
        x = np.linspace(-250, 250, n)
        X, Y = np.meshgrid(x, x)
        la, lo, _ = FRAME.enu_to_geodetic(X.ravel(), Y.ravel(), np.zeros(X.size))
        la, lo = la.reshape(n, n), lo.reshape(n, n)
        e, nn = gc._utm_grid(la, lo)
        e2, n2 = gc._utm32(la, lo)
        self.assertLess(float(np.abs(e - e2).max()), 0.01)
        self.assertLess(float(np.abs(nn - n2).max()), 0.01)

    def test_switch_off(self):
        old = gc.LAYERS_ON
        try:
            gc.LAYERS_ON = False
            idx = np.full((2, 2), 1, np.uint8)
            self.assertIs(gc.apply_layers(idx, np.full((2, 2), 50), np.zeros((2, 2)), np.zeros((2, 2)), {gc.STONE: 7}),
                          idx)
        finally:
            gc.LAYERS_ON = old

    def test_digest_empty_off_the_map(self):
        self.assertEqual(gc.digest(0.0, 1.0, 0.0, 1.0), "")
        self.assertTrue(gc.digest().startswith(gc.TAG))


class Feats:
    """The few things of a roads layer road_verdicts reads: attrs["class"], len, parts(i)."""

    def __init__(self, classes, lines):
        self.attrs = {"class": np.asarray(classes)}
        self.lines = lines

    def __len__(self):
        return len(self.lines)

    def parts(self, i):
        return [[np.asarray(self.lines[i], np.float64)]]


class RoadVerdictTest(unittest.TestCase):
    """The map: 100 m x 100 m from (E, N) = (501000, 5405000), rows count from the north (dn = 100 - 0.8 row): grass,
    but a paved lane at dn 76-84, a gravel lane at dn 45-52, a strip of trees at dn 12-20."""

    def setUp(self):
        import lod2geom
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "g.npz")
        cls = np.full((125, 125), gc.GRASS, np.uint8)
        cls[20:30, :] = gc.ASPHALT
        cls[60:70, :] = gc.GRAVEL
        cls[100:110, :] = gc.TREE
        make_map(self.path, cls)
        gc.reset()
        self.old = gc.DATA
        gc.DATA = self.path
        self.lonlat = lambda e, n: tuple(float(v) for v in lod2geom.wgs_from_utm32(e, n))[::-1]

    def tearDown(self):
        gc.DATA = self.old
        gc.reset()
        self.dir.cleanup()

    def line(self, dn):
        return [self.lonlat(501000.0 + e, 5405000.0 + dn) for e in range(5, 95, 3)]

    def test_verdicts(self):
        feats = Feats(["track", "service", "unclassified", "residential"],
                      [self.line(80.0), self.line(48.0), self.line(4.0), self.line(80.0)])
        got = gc.road_verdicts(feats, lambda i: None)
        self.assertEqual(got.get(0), "asphalt")
        self.assertEqual(got.get(1), "gravel")
        self.assertEqual(got.get(2), "grass")
        self.assertNotIn(3, got, "a residential road is not asked")

    def test_mapped_surface_stands(self):
        feats = Feats(["track"], [self.line(80.0)])
        self.assertEqual(gc.road_verdicts(feats, lambda i: "dirt"), {})

    def test_under_trees_no_verdict(self):
        feats = Feats(["track"], [self.line(16.0)])
        self.assertEqual(gc.road_verdicts(feats, lambda i: None), {})

    def test_off_the_map_no_verdict(self):
        feats = Feats(["track"], [[self.lonlat(505000.0 + e, 5405050.0) for e in range(0, 90, 3)]])
        self.assertEqual(gc.road_verdicts(feats, lambda i: None), {})


class PlazaPriorityTest(unittest.TestCase):
    """A paved area read from the photo ("photo:<surface>") is paved after the mapped squares: they keep the ground
    they share with it."""

    def test_mapped_square_wins_its_overlap(self):
        def ring(pts):
            la, lo, _ = FRAME.enu_to_geodetic(np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float),
                                              np.zeros(len(pts)))
            return np.c_[lo, la]
        big_photo = (ring([(10, 10), (110, 10), (110, 50), (10, 50), (10, 10)]), [], "photo:paving_stones")
        mapped = (ring([(40, 10), (80, 10), (80, 50), (40, 50), (40, 10)]), [], "asphalt")      # (smaller, and mapped)
        flat = lambda la, lo: np.full(np.shape(la), 5.0)        # noqa: E731
        P, T, UV, mats, M = plazas.mesh(FRAME, (0.0, 0.0, 256.0, 256.0), [big_photo, mapped], flat, (0.0, 0.0, 0.0))
        self.assertEqual(set(mats), {plazas.ASPHALT[0], plazas.PAVERS[0]})
        c = P[T].mean(1)
        a = M == mats.index(plazas.ASPHALT[0])
        self.assertTrue(((c[a, 0] > 38) & (c[a, 0] < 82)).all(), "asphalt only where it is mapped")
        p = M == mats.index(plazas.PAVERS[0])
        self.assertFalse(((c[p, 0] > 41) & (c[p, 0] < 79)).any(), "no photo paving over the mapped square")

    def test_photo_alone_is_paved(self):
        ring_ = np.array([[9.0135, 48.7984], [9.0140, 48.7984], [9.0140, 48.7987], [9.0135, 48.7987], [9.0135, 48.7984]])
        flat = lambda la, lo: np.full(np.shape(la), 5.0)        # noqa: E731
        m = plazas.mesh(FRAME, (-256.0, -256.0, 256.0, 256.0), [(ring_, [], "photo:sett")], flat, (0.0, 0.0, 0.0))
        self.assertEqual(m[3], [plazas.STONE[0]])


if __name__ == "__main__":
    unittest.main()
