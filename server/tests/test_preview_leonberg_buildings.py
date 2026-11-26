"""tools/preview_leonberg_buildings.py: the pure helpers (camera, z-buffer render, OBJ writer)."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("preview_leonberg_buildings",
                                              ROOT / "tools" / "preview_leonberg_buildings.py")
PV = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PV)


def _cube():
    """A unit-size cube centred on the origin: 8 vertices, 12 triangles, wound outward."""
    P = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)], float)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    T = []
    for a, b, c, d in quads:
        T += [(a, b, c), (a, c, d)]
    T = np.array(T, np.int64)
    for i, tri in enumerate(T):                      # outward: the normal points away from the centre
        n = np.cross(P[tri[1]] - P[tri[0]], P[tri[2]] - P[tri[0]])
        if np.dot(n, P[tri].mean(0)) < 0:
            T[i] = tri[[0, 2, 1]]
    C = np.full((8, 4), 200, np.uint8)
    C[:, 3] = 255
    return P, T, C


class CameraTest(unittest.TestCase):
    def test_orthonormal(self):
        for az, el in ((315, 35), (135, 35), (0, 80), (77, 10)):
            A = PV.camera_axes(az, el)
            self.assertEqual(A.shape, (3, 3))
            np.testing.assert_allclose(A.T @ A, np.eye(3), atol=1e-9)

    def test_viewer_is_at_azimuth(self):
        A = PV.camera_axes(315, 35)
        self.assertLess(A[:, 2][0], 0)               # towards the viewer: west ...
        self.assertGreater(A[:, 2][1], 0)            # ... and north


class RenderTest(unittest.TestCase):
    def test_cube_renders_object_in_centre(self):
        P, T, C = _cube()
        size = (200, 150)
        img = PV.render(P, T, C, PV.camera_axes(315, 35), size, 40.0, np.zeros(3))
        self.assertEqual(img.size, size)
        arr = np.asarray(img).astype(int)
        background = np.round(PV.BACKGROUND * 255)
        self.assertFalse(np.all(np.abs(arr[75, 100] - background) <= 2))


class TintTest(unittest.TestCase):
    def test_a_roof_shows_its_texture_colour_not_grey(self):
        P, T, C = _cube()
        C[:] = 255                                       # white vertex tint: the texture alone shows
        M = np.zeros(12, np.int64)
        axes, size = PV.camera_axes(315, 35), (200, 150)
        P2, T2, _, C2 = PV._tinted(P, T, M, C, ["bne_bld_roof_tiles"])
        roof = np.asarray(PV.render(P2, T2, C2, axes, size, 40.0, np.zeros(3))).astype(float)[75, 100]
        white = np.asarray(PV.render(P, T, C, axes, size, 40.0, np.zeros(3))).astype(float)[75, 100]
        self.assertTrue(roof[0] > roof[1] > roof[2])
        ratio = roof / white                             # the lighting cancels: the texture mean is left
        np.testing.assert_allclose(ratio, (0.625, 0.465, 0.361), atol=0.03)


class ObjTest(unittest.TestCase):
    def test_write_obj(self):
        P, T, C = _cube()
        M = np.zeros(12, np.int64)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cube.obj"
            PV.write_obj(path, P, T, M, C, ["bne_test_wall"])
            lines = path.read_text().splitlines()
            self.assertGreaterEqual(sum(l.startswith("v ") for l in lines), 8)
            self.assertEqual(sum(l.startswith("f ") for l in lines), 12)
            self.assertTrue(path.with_suffix(".mtl").exists())
            faces = [l for l in lines if l.startswith("f ")]
            self.assertTrue(all(int(v) >= 1 for f in faces for v in f.split()[1:]))


class WithinTest(unittest.TestCase):
    def test_keeps_whole_buildings_by_centroid(self):
        from types import SimpleNamespace as NS
        inside = (NS(ring=np.array([(0, 0), (10, 0), (10, 10), (0, 10)], float)), 0.0, 5.0)
        outside = (NS(ring=np.array([(190, 0), (210, 0), (210, 10), (190, 10)], float)), 0.0, 5.0)
        straddling = (NS(ring=np.array([(-10, 0), (150, 0), (150, 10), (-10, 10)], float)), 0.0, 5.0)
        out = PV.within([inside, outside, straddling], 0.0, 0.0, 170.0)
        self.assertEqual([item[0] for item in out], [inside[0], straddling[0]])   # the whole straddler is kept

    def test_tile_offset_is_added_to_the_ring(self):
        from types import SimpleNamespace as NS
        item = (NS(ring=np.array([(0, 0), (10, 0), (10, 10), (0, 10)], float)), 0.0, 5.0)
        self.assertEqual(len(PV.within([item], 200.0, 0.0, 170.0)), 0)
        self.assertEqual(len(PV.within([item], -160.0, 0.0, 170.0)), 1)


if __name__ == "__main__":
    unittest.main()
