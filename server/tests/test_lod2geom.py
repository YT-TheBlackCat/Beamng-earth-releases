"""lod2geom: UTM32 projection and tile names, the CityGML units, and the per-tile npz placed in the ENU frame."""
import io
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import geodesy  # noqa: E402
import lod2geom as lg  # noqa: E402

LAT, LON = 48.8, 9.01
NS = ('xmlns:core="http://www.opengis.net/citygml/1.0" '
      'xmlns:bldg="http://www.opengis.net/citygml/building/1.0" xmlns:gml="http://www.opengis.net/gml"')


def ring_xml(pts, z):
    pts = list(pts) + [pts[0]]
    pos = " ".join(f"{x:.3f} {y:.3f} {z + h:.3f}" for (x, y), h in pts)
    return (f'<gml:Polygon><gml:exterior><gml:LinearRing><gml:posList srsDimension="3">{pos}</gml:posList>'
            '</gml:LinearRing></gml:exterior></gml:Polygon>')


def surface(tag, pts, z):
    return (f"<bldg:boundedBy><bldg:{tag}><bldg:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember>"
            f"{ring_xml(pts, z)}</gml:surfaceMember></gml:MultiSurface></bldg:lod2MultiSurface>"
            f"</bldg:{tag}></bldg:boundedBy>")


def square(cx, cy, s):
    h = s / 2
    return [((cx - h, cy - h), 0.0), ((cx + h, cy - h), 0.0), ((cx + h, cy + h), 0.0), ((cx - h, cy + h), 0.0)]


def gml_text(cx, cy):
    """B1: ground, wall, roof and a closure surface (skipped). B2 has only a BuildingPart with a roof."""
    base = square(cx, cy, 10)
    roof = [((x, y), 5.0 if x > cx else 8.0) for (x, y), _ in base]    # eave 5 m on one side, ridge 8 m
    b1 = ('<core:cityObjectMember><bldg:Building gml:id="B1"><bldg:roofType>3100</bldg:roofType>'
          + surface("GroundSurface", base, 400.0)
          + surface("WallSurface", [((cx - 5, cy - 5), 0.0), ((cx + 5, cy - 5), 0.0),
                                    ((cx + 5, cy - 5), 5.0), ((cx - 5, cy - 5), 5.0)], 400.0)
          + surface("RoofSurface", roof, 400.0)
          + surface("ClosureSurface", base, 400.0)
          + "</bldg:Building></core:cityObjectMember>")
    b2 = ('<core:cityObjectMember><bldg:Building gml:id="B2"><bldg:consistsOfBuildingPart>'
          '<bldg:BuildingPart gml:id="P2"><bldg:roofType>1000</bldg:roofType>'
          + surface("RoofSurface", square(cx + 30, cy, 6), 402.0)
          + "</bldg:BuildingPart></bldg:consistsOfBuildingPart></bldg:Building></core:cityObjectMember>")
    return f'<?xml version="1.0" encoding="UTF-8"?><core:CityModel {NS}>{b1}{b2}</core:CityModel>'


class Projection(unittest.TestCase):
    def test_round_trip(self):
        for lat, lon in [(48.8, 9.0), (47.6, 7.5), (49.8, 10.5), (48.80014, 9.01523), (49.1, 8.2)]:
            E, N = lg.utm32_from_wgs(lat, lon)
            la, lo = lg.wgs_from_utm32(E, N)
            self.assertLess(abs(float(la) - lat), 1e-8)
            self.assertLess(abs(float(lo) - lon), 1e-8)

    def test_central_meridian(self):
        E, _ = lg.utm32_from_wgs(48.8, 9.0)
        self.assertLess(abs(float(E) - 500000), 1)

    def test_zip_tile(self):
        self.assertEqual(lg.zip_tile(501079.8, 5405465.3), (501, 5404))

    def test_tiles_over(self):
        self.assertEqual(lg.zip_tiles_over(501100, 5404100, 501900, 5404900), [(501, 5404)])
        self.assertEqual(lg.zip_tiles_over(500900, 5404900, 501100, 5405100), [(499, 5404), (501, 5404)])


class Parse(unittest.TestCase):
    def test_units(self):
        units = list(lg.parse_gml(io.BytesIO(gml_text(500000, 5400000).encode())))
        self.assertEqual([u["id"] for u in units], ["B1", "P2"])
        self.assertEqual(units[0]["bldg"], "B1")
        self.assertEqual(units[1]["bldg"], "B2")
        self.assertEqual(units[0]["code"], 3100)
        self.assertEqual(units[1]["code"], 1000)
        self.assertEqual([k for k, _ in units[0]["polys"]], [0, 1, 2])
        self.assertEqual(units[1]["polys"][0][0], 2)
        for _, p in units[0]["polys"]:
            self.assertEqual(len(p), 4)          # the closing point is gone
            self.assertEqual(p.shape[1], 3)


class Tile(unittest.TestCase):
    def test_prepare_and_load(self):
        cx, cy = lg.utm32_from_wgs(LAT, LON)
        cx, cy = float(cx), float(cy)
        with tempfile.TemporaryDirectory() as tmp:
            zpath = os.path.join(tmp, "LoD2_32_test.zip")
            with zipfile.ZipFile(zpath, "w") as z:
                z.writestr("LoD2_32_test/test_1_BW.gml", gml_text(cx, cy))
                z.writestr("LoD2_32_test/meta.txt", "ignored")
            e, n = lg.zip_tile(cx, cy)
            folder = os.path.join(tmp, "geom")
            os.environ["BNE_LOD2GEOM_DIR"] = folder
            try:
                self.assertEqual(lg.folder(tmp), folder)
                out = os.path.join(folder, f"{e}_{n}.npz")
                self.assertEqual(lg.prepare(zpath, out), 2)
                self.assertFalse(os.path.exists(out + ".tmp.npz"))
                self.assertEqual(os.listdir(folder), [os.path.basename(out)])
                frame = geodesy.EnuFrame(LAT, LON)
                g = lg.Geoms.load(tmp, frame, -150, -150, 150, 150)
                self.assertIsNotNone(g)
                self.assertEqual(g.n, 2)
                u = g.unit(0)
                self.assertEqual(set(u), {"ring", "area", "cx", "cy", "polys", "code", "top", "eave", "id", "bldg"})
                self.assertLess(u["eave"], u["top"])
                self.assertAlmostEqual(u["area"], 100, delta=1)
                self.assertEqual(u["id"], "B1")
                # footprint ring around the building: one unit, area in range
                ring = np.array([[-6, -6], [6, -6], [6, 6], [-6, 6], [-6, -6]], float)
                self.assertEqual(g.take(ring[:, 0], ring[:, 1]), [0])
                self.assertEqual(g.take(ring[:, 0], ring[:, 1]), [])
                g2 = lg.Geoms.load(tmp, frame, -150, -150, 150, 150)
                big = np.array([[-30, -30], [30, -30], [30, 30], [-30, 30]], float)
                self.assertEqual(g2.take(big[:, 0], big[:, 1]), [])
                self.assertIsNone(lg.Geoms.load(tmp, geodesy.EnuFrame(50.9, 9.1), -150, -150, 150, 150))
            finally:
                del os.environ["BNE_LOD2GEOM_DIR"]

    def test_a_damaged_tile_is_skipped(self):
        from unittest import mock
        cx, cy = lg.utm32_from_wgs(LAT, LON)
        cx, cy = float(cx), float(cy)
        with tempfile.TemporaryDirectory() as tmp:
            zpath = os.path.join(tmp, "LoD2_32_test.zip")
            with zipfile.ZipFile(zpath, "w") as z:
                z.writestr("LoD2_32_test/test_1_BW.gml", gml_text(cx, cy))
            e, n = lg.zip_tile(cx, cy)
            folder = os.path.join(tmp, "geom")
            os.environ["BNE_LOD2GEOM_DIR"] = folder
            try:
                good = os.path.join(folder, f"{e}_{n}.npz")
                self.assertEqual(lg.prepare(zpath, good), 2)
                other = [t for t in lg.zip_tiles_over(cx - 3000, cy - 3000, cx + 3000, cy + 3000) if t != (e, n)]
                with open(os.path.join(folder, f"{other[0][0]}_{other[0][1]}.npz"), "wb") as fh:
                    fh.write(b"junk")
                frame = geodesy.EnuFrame(LAT, LON)
                with self.assertLogs("earthd.lod2geom", "WARNING") as logs:
                    g = lg.Geoms.load(tmp, frame, -3000, -3000, 3000, 3000)
                self.assertEqual(len(logs.output), 1)
                self.assertIsNotNone(g)
                self.assertEqual(g.n, 2)
                with mock.patch.object(lg.log, "warning") as warn:
                    self.assertIsNotNone(lg.Geoms.load(tmp, frame, -3000, -3000, 3000, 3000))
                warn.assert_not_called()
                os.remove(good)
                self.assertIsNone(lg.Geoms.load(tmp, frame, -3000, -3000, 3000, 3000))
            finally:
                del os.environ["BNE_LOD2GEOM_DIR"]


class FakeFetcher:
    """netfetch's get(): the zip's bytes for its URL, NotFound, or an error (and the URLs asked)."""

    def __init__(self, body=None, exc=None):
        self.body, self.exc, self.asked = body, exc, []

    def get(self, url, kind="demand", headers=None, lane=None):
        self.asked.append(url)
        if self.exc is not None:
            raise self.exc
        from types import SimpleNamespace
        return SimpleNamespace(status=200, body=self.body)


class Fetch(unittest.TestCase):
    """A tile not on disk is downloaded from the LGL and prepared the first time a scene needs it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = os.path.join(self.tmp.name, "geom")
        os.environ["BNE_LOD2GEOM_DIR"] = self.folder
        self.addCleanup(os.environ.pop, "BNE_LOD2GEOM_DIR", None)
        lg.Geoms._fetch_failed.clear()
        self.addCleanup(lg.Geoms._fetch_failed.clear)
        cx, cy = lg.utm32_from_wgs(LAT, LON)
        self.cx, self.cy = float(cx), float(cy)
        self.e, self.n = lg.zip_tile(self.cx, self.cy)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("LoD2_32_test/test_1_BW.gml", gml_text(self.cx, self.cy))
        self.zip = buf.getvalue()

    def load(self, fetcher):
        return lg.Geoms.load(self.tmp.name, geodesy.EnuFrame(LAT, LON), -150, -150, 150, 150, fetcher=fetcher)

    def test_a_missing_tile_is_downloaded_prepared_and_kept(self):
        f = FakeFetcher(self.zip)
        g = self.load(f)
        self.assertIsNotNone(g)
        self.assertEqual(g.n, 2)
        self.assertEqual(f.asked, [lg.URL.format(e=self.e, n=self.n)])
        self.assertEqual(os.listdir(self.folder), [f"{self.e}_{self.n}.npz"], "the npz alone, no zip left")
        self.assertIsNotNone(self.load(FakeFetcher(exc=AssertionError("asked again"))))

    def test_a_tile_the_lgl_lacks_is_not_asked_again(self):
        import netfetch
        f = FakeFetcher(exc=netfetch.NotFound("404"))
        self.assertIsNone(self.load(f))
        self.assertIsNone(self.load(f))
        self.assertEqual(len(f.asked), 1)

    def test_a_failed_download_waits_before_the_next_try(self):
        f = FakeFetcher(exc=OSError("offline"))
        with self.assertLogs("earthd.lod2geom", "WARNING"):
            self.assertIsNone(self.load(f))
        self.assertIsNone(self.load(f))
        self.assertEqual(len(f.asked), 1)
        self.assertFalse(os.path.exists(self.folder) and os.listdir(self.folder), "nothing half-written kept")

    def test_off_or_no_fetcher_reads_the_disk_only(self):
        self.assertIsNone(self.load(None))
        old = lg.FETCH
        lg.FETCH = False
        try:
            f = FakeFetcher(self.zip)
            self.assertIsNone(self.load(f))
            self.assertEqual(f.asked, [])
        finally:
            lg.FETCH = old


class Gaps(unittest.TestCase):
    """take() accepts area ratios 0.45-2.0; leftover() returns the units no footprint covers (GAPS)."""

    @staticmethod
    def geoms(units):
        """Geoms over fake units: [(cx, cy, area, ring_square_side)] -> instance whose unit() is preset."""
        cx = np.array([u[0] for u in units], np.float64)
        cy = np.array([u[1] for u in units], np.float64)
        g = lg.Geoms(None, [], np.zeros(len(units), np.int64), np.arange(len(units)), cx, cy)
        for i, (x, y, area, side) in enumerate(units):
            h = side / 2
            ring = np.array([[x - h, y - h], [x + h, y - h], [x + h, y + h], [x - h, y + h]], float)
            g._units[i] = {"ring": ring, "area": float(area), "cx": x, "cy": y, "polys": [], "code": 1000,
                           "top": 8.0, "eave": 5.0, "id": f"U{i}", "bldg": ""}
        return g

    @staticmethod
    def sq(x, y, h):
        return np.array([x - h, x + h, x + h, x - h]), np.array([y - h, y - h, y + h, y + h])

    def test_ratio_window(self):
        ring = self.sq(0, 0, 5)                                  # 100 m2
        self.assertEqual(self.geoms([(0, 0, 180, 13.4)]).take(*ring), [0])     # 1.8: taken now
        self.assertEqual(self.geoms([(0, 0, 220, 14.8)]).take(*ring), [])      # 2.2: still no
        self.assertEqual(self.geoms([(0, 0, 49, 7.0)]).take(*ring), [0])       # 0.49: taken now

    def test_leftover(self):
        g = self.geoms([
            (0, 0, 100, 10),      # 0: its footprint is the outline: covered
            (50, 0, 100, 10),     # 1: no footprint: the leftover
            (0, 50, 100, 10),     # 2: 60 % under an outline: covered
            (100, 0, 5, 1),       # 3: under min_area
            (500, 0, 100, 10),    # 4: another tile's
        ])
        outlines = [self.sq(0, 0, 5), (np.array([-5.0, 5, 5, -5]), np.array([45.0, 45, 51, 51]))]
        box = (-100, -100, 200, 100)
        self.assertEqual(g.leftover(*box, outlines), [1])
        self.assertEqual(g.leftover(*box, outlines), [])          # marked taken
        self.assertIn(1, g._taken)

    def test_gaps_off(self):
        from unittest import mock
        g = self.geoms([(50, 0, 100, 10)])
        with mock.patch.object(lg, "GAPS", False):
            self.assertEqual(g.leftover(-100, -100, 200, 100, []), [])
        self.assertEqual(g.leftover(-100, -100, 200, 100, []), [0])


if __name__ == "__main__":
    unittest.main()
