"""buildings.tile_buildings: a LoD2 unit that no footprint covers is drawn as a building of its own (leftover), and the
footprint's own LoD2 unit is not drawn twice."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import buildings as bld  # noqa: E402
import geodesy  # noqa: E402


class Feats:
    def __init__(self, ids, rings, **attrs):
        self.ids = np.array(ids, object)
        self.kind = np.full(len(ids), 3, np.uint8)
        self.rings = rings
        self.attrs = {k: np.array(v, object if isinstance(v[0], str) else np.float64) for k, v in attrs.items()}

    def parts(self, i):
        return [[np.asarray(self.rings[i], np.float64)]]


class FakeLod2:
    """Two LoD2 units, 10 x 10 m each: 0 at (0, 0) under the footprint, 1 at (60, 0) with none."""
    n = 2

    def __init__(self, leftover_ids):
        self.leftover_ids = leftover_ids
        self.U = {}
        for i, x in ((0, 0.0), (1, 60.0)):
            ring = np.array([(x - 5, -5), (x + 5, -5), (x + 5, 5), (x - 5, 5)], np.float64)
            self.U[i] = {"ring": ring, "area": 100.0, "cx": x, "cy": 0.0, "polys": [], "code": 1000, "top": 8.0,
                         "eave": 5.0, "id": f"U{i}", "bldg": ""}
        self.taken = []

    def take(self, e, n, area_ratio=(0.45, 2.0)):
        e, n = np.asarray(e, float), np.asarray(n, float)
        if np.hypot(e.mean(), n.mean()) < 20 and 0 not in self.taken:
            self.taken.append(0)
            return [0]
        return []

    def unit(self, i):
        return self.U[i]

    def leftover(self, x0, y0, x1, y1, outlines, own=None, **kw):
        return list(self.leftover_ids)


def ll(f, x, y, w, d):
    e = np.array([x, x + w, x + w, x, x], float)
    n = np.array([y, y, y + d, y + d, y], float)
    la, lo, _ = f.enu_to_geodetic(e, n, 0.0)
    return np.stack([lo, la], 1)


class LeftoverTest(unittest.TestCase):
    def _run(self, leftover_ids):
        f = geodesy.EnuFrame(48.8, 9.01)
        b = Feats(["B"], [ll(f, -5, -5, 10, 10)], **{"class": ["house"], "height": [8.0], "has_parts": [0.0],
                                                     "min_height": [np.nan]})
        return bld.tile_buildings(f, -256, -256, 256, 256, [b], lambda la, lo: np.zeros(np.shape(la)),
                                  lod2geom=FakeLod2(leftover_ids))

    def test_a_leftover_unit_is_a_building_of_its_own(self):
        items = self._run([1])
        self.assertEqual(len(items), 2, "the footprint's own unit and the leftover one")
        centres = sorted(round(float(s.ring[:, 0].mean()) - 256) for s, _, _ in items)   # tile-relative, ring centre e
        self.assertEqual(centres, [0, 60])

    def test_no_leftover_no_extra_building(self):
        self.assertEqual(len(self._run([])), 1)


class BoxedFeats(Feats):
    """Feats with mapgeom.Features' boxes(frame) (the [min e, min n, max e, max n] of each ring)."""

    def boxes(self, frame):
        out = []
        for r in self.rings:
            e, n, _ = frame.geodetic_to_enu(r[:, 1], r[:, 0], 0.0)
            out.append([e.min(), n.min(), e.max(), n.max()])
        return np.array(out, np.float64)


class ReachingOutlines(unittest.TestCase):
    """A footprint whose centre is another tile's but which reaches into this one covers this tile's LoD2 units too
    (the other tile took them): lod2geom.leftover sees it and does not draw them twice."""

    def test_only_the_ones_reaching_in_from_elsewhere(self):
        f = geodesy.EnuFrame(48.8, 9.01)
        rings = [ll(f, 250, 0, 20, 10),        # centre 260: the next tile's, reaches in
                 ll(f, 300, 0, 10, 10),        # wholly in the next tile
                 ll(f, 0, 0, 10, 10),          # this tile's own (near)
                 ll(f, 240, 20, 30, 10)]       # reaches in, but under the ground
        b = BoxedFeats(["A", "B", "C", "D"], rings, level=[np.nan, np.nan, np.nan, -1.0])
        near = np.array([False, False, True, False])
        got = bld._reaching_outlines(b, f, -256, -256, 256, 256, near)
        self.assertEqual(len(got), 1)
        e, n = got[0]
        self.assertAlmostEqual(float(np.min(e)), 250.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
