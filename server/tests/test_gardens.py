"""Garden recipes: ALKIS Shape reading, pixel classes, the garden ground, hedges, and the server's paint, digest,
vegetation swap and hedge rows (gardenbake.py, gardens.py). Synthetic data only; no network, no real cache."""
import io
import json
import os
import struct
import sys
import tempfile
import unittest
import zipfile
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fences  # noqa: E402
import gardenbake as G  # noqa: E402
import gardens  # noqa: E402
import geodesy  # noqa: E402
import lod2geom  # noqa: E402
import mapcells  # noqa: E402
import surface  # noqa: E402
import vegetation as V  # noqa: E402


def make_shp(records):
    """A Polygon (5) Shape file: records = [[parts]], each part a list of (x, y) incl. the closing point."""
    body = b""
    allxy = []
    for n, parts in enumerate(records, 1):
        pts = [p for part in parts for p in part]
        allxy += pts
        xy = np.array(pts, np.float64)
        starts, acc = [], 0
        for part in parts:
            starts.append(acc)
            acc += len(part)
        content = struct.pack("<i", 5) + struct.pack("<4d", xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(),
                                                      xy[:, 1].max())
        content += struct.pack("<ii", len(parts), len(pts)) + struct.pack(f"<{len(parts)}i", *starts)
        content += xy.astype("<f8").tobytes()
        body += struct.pack(">ii", n, len(content) // 2) + content
    if allxy:
        a = np.array(allxy, np.float64)
        box = (a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max())
    else:
        box = (0, 0, 0, 0)
    head = struct.pack(">i", 9994) + b"\0" * 20 + struct.pack(">i", (100 + len(body)) // 2)
    head += struct.pack("<ii", 1000, 5) + struct.pack("<4d", *box) + b"\0" * 32
    return head + body


def make_dbf(fields, rows):
    """A dBASE III table: fields [(name, size)] (all C), rows [[str]]."""
    hl = 32 + 32 * len(fields) + 1
    rl = 1 + sum(s for _, s in fields)
    out = bytes([3]) + b"\0" * 3 + struct.pack("<I", len(rows)) + struct.pack("<HH", hl, rl) + b"\0" * 20
    for name, size in fields:
        out += name.encode("ascii").ljust(11, b"\0") + b"C" + b"\0" * 4 + bytes([size]) + b"\0" * 15
    out += b"\x0d"
    for row in rows:
        out += b" "
        for (_, size), val in zip(fields, row):
            out += val.encode("cp1252").ljust(size, b" ")[:size]
    return out + b"\x1a"


CW_SQUARE = [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)]           # clockwise outer (shapefile)
CCW_HOLE = [(4, 4), (6, 4), (6, 6), (4, 6), (4, 4)]                # counter-clockwise hole


class ShapefileTest(unittest.TestCase):
    def test_polygons_rings_without_closing_point_and_hole(self):
        shp = make_shp([[CW_SQUARE], [CW_SQUARE, CCW_HOLE]])
        recs = G.read_shp_polygons(shp)
        self.assertEqual(len(recs), 2)
        self.assertEqual(len(recs[0]), 1)
        np.testing.assert_array_equal(recs[0][0], np.array(CW_SQUARE[:-1], float))
        self.assertEqual(len(recs[1]), 2)
        self.assertLess(G.signed_area(recs[1][0]), 0)
        np.testing.assert_array_equal(recs[1][1], np.array(CCW_HOLE[:-1], float))
        self.assertGreater(G.signed_area(recs[1][1]), 0)

    def test_dbf_decodes_windows_1252(self):
        dbf = make_dbf([("idflurst", 16), ("tntxt", 40), ("flaeche", 10)],
                       [["DEBWL0001", "Wohnbaufläche;100", "100"], ["DEBWL0002", "Wohnbaufläche;96", "96"]])
        rows = G.read_dbf(dbf)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], {"idflurst": "DEBWL0001", "tntxt": "Wohnbaufläche;100", "flaeche": "100"})
        self.assertEqual(rows[1]["tntxt"], "Wohnbaufläche;96")
        only = G.read_dbf(dbf, {"flaeche"})
        self.assertEqual(only[1], {"flaeche": "96"})

    def test_alkis_zip_parcels(self):
        flur = make_shp([[CW_SQUARE], [CW_SQUARE, CCW_HOLE]])
        flur_dbf = make_dbf([("idflurst", 16), ("tntxt", 40), ("flaeche", 10)],
                            [["DEBWL0001", "Wohnbaufläche;100", "100"], ["DEBWL0002", "Wohnbaufläche;96", "96"]])
        geb = make_shp([])
        geb_dbf = make_dbf([("oid", 16), ("funktion", 10), ("gebnutzbez", 40), ("anzahlgs", 4)], [])
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("ALKIS/flurstueck.shp", flur)
            zf.writestr("ALKIS/flurstueck.dbf", flur_dbf)
            zf.writestr("ALKIS/gebaeudeBauwerke.shp", geb)
            zf.writestr("ALKIS/gebaeudeBauwerke.dbf", geb_dbf)
        parcels, buildings = G.read_alkis(buf.getvalue())
        self.assertEqual(buildings, [])
        self.assertEqual([p["id"] for p in parcels], ["DEBWL0001", "DEBWL0002"])
        self.assertEqual(parcels[0]["use"], ["Wohnbaufläche"])
        self.assertEqual(parcels[0]["area"], 100.0)
        self.assertEqual(parcels[1]["area"], 96.0)
        self.assertEqual(len(parcels[0]["rings"]), 1)
        self.assertEqual(len(parcels[1]["rings"]), 2)


def checker(shape, lo=150, hi=230):
    ii, jj = np.indices(shape)
    return np.where(((ii // 2 + jj // 2) % 2) == 0, lo, hi).astype(np.uint8)


class ClassifyTest(unittest.TestCase):
    def test_four_quadrants_and_building(self):
        rgbi = np.zeros((200, 200, 4), np.uint8)
        ndom = np.zeros((200, 200), np.float32)
        rgbi[:100, :100] = (60, 110, 60, 200)                                   # (a) lawn, smooth
        rgbi[:100, 100:, :3] = (60, 110, 60)                                    # (b) tree, rough
        rgbi[:100, 100:, 3] = checker((100, 100))
        ndom[:100, 100:] = 8.0
        rgbi[100:, :100, :3] = (60, 110, 60)                                    # (c) shrub, rough
        rgbi[100:, :100, 3] = checker((100, 100))
        ndom[100:, :100] = 2.0
        rgbi[100:, 100:] = (160, 160, 160, 150)                                 # (d) paving
        building = np.zeros((200, 200), bool)
        building[110:130, 110:130] = True
        cls = G.classify_pixels(rgbi, ndom, building)
        self.assertEqual(cls[50, 50], G.P_LAWN)
        self.assertEqual(cls[50, 150], G.P_TREE)
        self.assertEqual(cls[150, 50], G.P_SHRUB)
        self.assertEqual(cls[150, 150], G.P_PAVED)
        self.assertEqual(cls[120, 120], G.P_BUILDING)


def fake_mosaic(rgbi, ndom):
    m = object.__new__(G.Mosaic)
    m.e0, m.n0, m.e1, m.n1 = 0.0, 0.0, 40.0, 40.0
    m.rgbi, m.ndom = rgbi, ndom
    m.valid = np.ones(ndom.shape, bool)
    return m


def at(m, arr, e, n):
    col, row = m.to_px(e, n)
    return int(arr[int(np.floor(row)), int(np.floor(col))])


SQUARE_40 = np.array([(0, 0), (0, 40), (40, 40), (40, 0)], float)        # clockwise


class GardenGroundTest(unittest.TestCase):
    def test_ground_classes(self):
        rgbi = np.zeros((200, 200, 4), np.uint8)
        rgbi[...] = (60, 110, 60, 200)
        m = fake_mosaic(rgbi, np.zeros((200, 200), np.float32))
        cls = np.full((200, 200), G.P_LAWN, np.uint8)
        cls[100:150, 50:100] = G.P_BUILDING                 # house x 10-20, y 10-20
        cls[150:200, 65:80] = G.P_PAVED                     # strip x 13-16, y 0-10: to the road
        cls[80:100, 60:80] = G.P_PAVED                      # patch x 12-16, y 20-24: at the house
        cls[35:75, 150:155] = G.P_PAVED                     # strip x 30-31, y 25-33: 1 m wide, 8 m long
        road = np.zeros((200, 200), bool)
        road[198:, :] = True                                # y < 0.4 m
        g = G.Garden(m, cls, [("p1", [SQUARE_40])], road)
        g.build_ground()
        self.assertEqual(at(m, g.ground, 14.5, 5), G.G_DRIVEWAY)
        self.assertEqual(at(m, g.ground, 14, 22), G.G_TERRACE)
        self.assertEqual(at(m, g.ground, 30.5, 29), G.G_PATH)
        self.assertEqual(at(m, g.ground, 35, 35), G.G_LAWN)
        self.assertEqual(at(m, g.ground, 15, 15), G.G_BUILDING)


class HedgeTest(unittest.TestCase):
    def test_one_hedge_along_east_edge(self):
        rgbi = np.zeros((200, 200, 4), np.uint8)
        rgbi[...] = (60, 110, 60, 200)
        ndom = np.zeros((200, 200), np.float32)
        cls = np.full((200, 200), G.P_LAWN, np.uint8)
        rgbi[25:175, 194:199] = (50, 100, 50, 210)          # x 38.8-39.8 m, y 5-35 m: shrubs
        ndom[25:175, 194:199] = 2.0
        cls[25:175, 194:199] = G.P_SHRUB
        m = fake_mosaic(rgbi, ndom)
        g = G.Garden(m, cls, [("p1", [SQUARE_40])], np.zeros((200, 200), bool))
        g.find_hedges()
        self.assertEqual(len(g.hedges), 1)
        line, height, _ = g.hedges[0]
        length = float(np.hypot(*np.diff(line, axis=0).T).sum())
        self.assertTrue(26 <= length <= 32, length)
        self.assertTrue(1.8 <= height <= 2.2, height)
        self.assertTrue(np.all((line[:, 0] >= 38) & (line[:, 0] <= 40.5)))


class RecipeServerTest(unittest.TestCase):
    CENTRE = (502360.0, 5404392.0)

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.tmp = self.tmp_dir.name
        env = mock.patch.dict(os.environ, {"BNE_GARDENS_DIR": ""})
        env.start()
        self.addCleanup(env.stop)
        for flag in ("GROUND_ON", "HEDGES_ON", "TREES_ON"):
            p = mock.patch.object(gardens, flag, True)
            p.start()
            self.addCleanup(p.stop)
        lat, lon = lod2geom.wgs_from_utm32(*self.CENTRE)
        self.frame = geodesy.EnuFrame(float(lat), float(lon))
        self.write_recipe()

    def rc(self, e, n):
        """(row, col) of a UTM point in the 0.5 m ground raster of square 502_5404 (row 0 north)."""
        return int((5405000 - n) / 0.5), int((e - 502000) / 0.5)

    def ground(self):
        g = np.zeros((2000, 2000), np.uint8)
        r0, c0 = self.rc(502310, 5404442)
        r1, c1 = self.rc(502410, 5404342)
        g[r0:r1, c0:c0 + 100] = G.G_LAWN               # west half: E 502310-502360
        g[r0:r1, c0 + 100:c1] = G.G_DRIVEWAY           # east half: E 502360-502410
        return g

    def write_recipe(self, ground=None, tree=None):
        folder = gardens.folder(self.tmp)
        os.makedirs(folder, exist_ok=True)
        path = G.km_path(folder, 502, 5404)
        np.savez_compressed(
            path, e_km=np.int32(502), n_km=np.int32(5404), px=np.float32(0.5),
            ground=self.ground() if ground is None else ground,
            hedge_xy=np.array([[340, 380], [340, 410]], np.float32), hedge_start=np.array([0, 2], np.int32),
            hedge_h=np.array([3.5], np.float32), hedge_w=np.array([1.0], np.float32),
            tree=np.array([[360, 392, 9, 3, 0.4]] if tree is None else tree, np.float32).reshape(-1, 5),
            shrub=np.array([[365, 392, 1.5, 0.8]], np.float32),
            parcel_id=np.array(["DEBWL1"]), parcel_xy=np.array([[360, 392]], np.float32),
            parcel_share=np.zeros((1, len(G.G_NAMES)), np.float32),
            meta=np.array(json.dumps({"version": G.VERSION, "classes": G.G_NAMES})))
        return path

    def enu_grid(self, half, step=1.0, cy=0.0):
        xs = np.arange(-half, half, step) + step / 2
        X, Y = np.meshgrid(xs, xs + cy)
        lat, lon, _ = self.frame.enu_to_geodetic(X, Y, np.zeros_like(X))
        return X, Y, lat, lon

    def cell_at(self, x, y):
        lat, lon, _ = self.frame.enu_to_geodetic(np.array([float(x)]), np.array([float(y)]), np.zeros(1))
        return mapcells.cells_for_bbox(float(lat[0]) - 1e-4, float(lon[0]) - 1e-4,
                                       float(lat[0]) + 1e-4, float(lon[0]) + 1e-4)[0]

    def test_paint_lawn_pavers_and_keeps_other_materials(self):
        X, Y, lat, lon = self.enu_grid(30)
        idx = np.full(X.shape, surface.DIRT, np.uint8)
        asph = (np.abs(X + 20) < 1.5) & (np.abs(Y - 10) < 1.5)       # a 3 m asphalt square on the lawn side
        idx[asph] = surface.ASPHALT
        out = gardens.paint(idx, lat, lon, self.tmp)

        def pick(x, y):
            return np.unravel_index(np.argmin(np.abs(X - x) + np.abs(Y - y)), X.shape)

        self.assertEqual(out[pick(-20, 0)], surface.LAWN)
        self.assertEqual(out[pick(20, 0)], surface.PAVERS)
        self.assertTrue(np.all(out[asph] == surface.ASPHALT))

        X2, Y2, lat2, lon2 = self.enu_grid(10, cy=80.0)             # ENU (0, 80) +-10 m: outside the block
        idx2 = np.full(X2.shape, surface.DIRT, np.uint8)
        far = gardens.paint(idx2, lat2, lon2, self.tmp)
        self.assertTrue(np.all(far == surface.DIRT))

    def test_digest_changes_when_recipe_is_rewritten(self):
        X, Y, lat, lon = self.enu_grid(30)
        box = (lat.min(), lat.max(), lon.min(), lon.max())
        d1 = gardens.digest(*box, self.tmp)
        self.assertEqual(d1, gardens.digest(*box, self.tmp))
        path = self.write_recipe(tree=[[300, 300, 12, 4, 0.5]])
        os.utime(path, ns=(10**18, 10**18 + 5 * 10**9))
        self.assertNotEqual(d1, gardens.digest(*box, self.tmp))

    def test_swap_vegetation_replaces_only_garden_trees(self):
        lat0, lon0, _ = self.frame.enu_to_geodetic(np.array([-20.0, 0.0]), np.array([0.0, 80.0]), np.zeros(2))
        items = V.Items(lon=np.asarray(lon0, np.float64), lat=np.asarray(lat0, np.float64),
                        height=np.array([5.0, 7.0]), radius=np.array([2.0, 2.0]),
                        kind=np.full(2, V.KIND_TREE, np.uint8), group=np.zeros(2, np.uint8),
                        rank=np.array([0.1, 0.9], np.float32), yaw=np.zeros(2, np.float32),
                        scale=np.ones(2, np.float32))
        key = self.cell_at(0, 0)
        out = gardens.swap_vegetation(items, key, self.tmp)
        self.assertEqual(len(out), 3)
        trees = out.kind == V.KIND_TREE
        self.assertEqual(int(trees.sum()), 2)
        kept = np.isclose(out.lat, lat0[1], atol=1e-9) & np.isclose(out.lon, lon0[1], atol=1e-9)
        self.assertEqual(int(kept.sum()), 1)
        self.assertFalse(np.any(np.isclose(out.lat, lat0[0], atol=1e-9) & np.isclose(out.lon, lon0[0], atol=1e-9)))
        rec_tree = trees & np.isclose(out.height, 9.0)
        self.assertEqual(int(rec_tree.sum()), 1)
        self.assertEqual(int(out.group[rec_tree][0]), V.GROUPS.index("deciduous"))
        bush = out.kind == V.KIND_BUSH
        self.assertEqual(int(bush.sum()), 1)
        self.assertEqual(int(out.group[bush][0]), V.GROUPS.index("shrub"))

    def test_hedge_rows_along_the_baked_hedge(self):
        rows = gardens.hedge_rows(self.frame, (-128, -128, 128, 128), lambda x, y: np.zeros(len(x)), self.tmp)
        self.assertTrue(rows)
        self.assertTrue(all(r[0] == fences.G_HEDGE for r in rows))
        self.assertTrue(all(r[6] == gardens.HEDGE_H[1] for r in rows))
        lat, lon = lod2geom.wgs_from_utm32(np.array([502340.0, 502340.0]), np.array([5404380.0, 5404410.0]))
        hx, _, _ = self.frame.geodetic_to_enu(np.asarray(lat), np.asarray(lon), np.zeros(2))
        xs = np.array([r[1] for r in rows])
        self.assertTrue(np.all(np.abs(xs - np.mean(hx)) <= 0.5), (xs.min(), xs.max(), hx))
        self.assertLessEqual(abs(len(rows) - 30 / gardens.HEDGE_STEP_M), 2)

    def test_switch_off_returns_idx_unchanged(self):
        X, Y, lat, lon = self.enu_grid(30)
        idx = np.full(X.shape, surface.DIRT, np.uint8)
        with mock.patch.object(gardens, "GROUND_ON", False):
            out = gardens.paint(idx, lat, lon, self.tmp)
        self.assertIs(out, idx)
        self.assertTrue(np.all(out == surface.DIRT))


if __name__ == "__main__":
    unittest.main()
