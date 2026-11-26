import os
"""lod2roofs: a measured record fills what the map leaves out of the footprint it stands in; the map's own values stay,
nothing is asked for outside Germany, and a missing shard means the predicted roofs."""
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import buildings as bld  # noqa: E402
import geodesy  # noqa: E402
import lod2roofs as lr  # noqa: E402
import netfetch  # noqa: E402

LAT, LON = 50.9, 9.1


def shard(rows):
    """npz bytes of [(lat, lon, code, height m, roof m)]."""
    a = np.array(rows, np.float64)
    buf = io.BytesIO()
    np.savez_compressed(buf, lat=np.round(a[:, 0] * 1e6).astype(np.int32), lon=np.round(a[:, 1] * 1e6).astype(np.int32),
                        code=a[:, 2].astype(np.int16), height=np.round(a[:, 3] * 10).astype(np.uint16),
                        roof=np.round(a[:, 4] * 10).astype(np.uint16))
    return buf.getvalue()


class Body:
    def __init__(self, body):
        self.body = body


class Fetcher:
    def __init__(self, body=None, error=None):
        self.body, self.error, self.urls = body, error, []

    def get(self, url, kind="demand", **_):
        self.urls.append(url)
        if self.error:
            raise self.error
        return Body(self.body)


def ring(cx, cy, r=6.0):
    return np.array([cx - r, cx + r, cx + r, cx - r]), np.array([cy - r, cy - r, cy + r, cy + r])


class Base(unittest.TestCase):
    def records(self, rows):
        frame = geodesy.EnuFrame(LAT, LON)
        src = lr.Source(None, Fetcher(shard(rows)))
        return src.tile(frame, -200, -200, 200, 200), src


class FillTest(Base):
    def test_a_measured_roof_fills_an_unknown_building(self):
        rec, _ = self.records([(LAT, LON, 3300, 9.4, 3.2)])
        at = rec.fill({"roof_shape": None, "height": None}, *ring(0, 0))
        self.assertEqual(at["roof_shape"], "half-hipped")
        self.assertAlmostEqual(at["height"], 9.4, 3)
        self.assertAlmostEqual(at["roof_height"], 3.2, 3)

    def test_the_maps_own_values_stay(self):
        rec, _ = self.records([(LAT, LON, 3100, 9.4, 3.2)])
        at = rec.fill({"roof_shape": "hipped", "height": 12.0, "roof_height": 2.0}, *ring(0, 0))
        self.assertEqual((at["roof_shape"], at["height"], at["roof_height"]), ("hipped", 12.0, 2.0))

    def test_a_measured_height_replaces_an_ml_estimate(self):
        rec, _ = self.records([(LAT, LON, 3100, 13.2, 3.0)])
        at = rec.fill({"roof_shape": None, "height": 10.9, "height_src": "Microsoft ML Buildings"}, *ring(0, 0))
        self.assertAlmostEqual(at["height"], 13.2, 3)
        self.assertEqual(at["height_src"], "LoD2")

    def test_an_osm_height_stays_and_a_low_roof_is_kept(self):
        rec, _ = self.records([(LAT, LON, 3100, 9.4, 3.2)])
        at = rec.fill({"roof_shape": None, "height": 12.0, "height_src": "OpenStreetMap", "roof_height": None},
                      *ring(0, 0))
        self.assertEqual((at["height"], at["height_src"]), (12.0, "OpenStreetMap"))
        self.assertAlmostEqual(at["roof_height"], 3.2, 3)

    def test_a_roof_as_tall_as_the_osm_wall_is_not_set(self):
        rec, _ = self.records([(LAT, LON, 3100, 9.4, 7.0)])
        at = rec.fill({"roof_shape": None, "height": 8.0, "height_src": "OpenStreetMap", "roof_height": None},
                      *ring(0, 0))
        self.assertEqual(at["height"], 8.0)
        self.assertIsNone(at["roof_height"])

    def test_a_mixed_roof_gives_its_height_but_no_shape(self):
        rec, _ = self.records([(LAT, LON, 5000, 8.0, 3.0)])
        at = rec.fill({"roof_shape": None, "height": None}, *ring(0, 0))
        self.assertIsNone(at["roof_shape"])
        self.assertAlmostEqual(at["height"], 6.5, 3)     # walls to mid-roof: 8 - 0.5 * 3

    def test_an_unknown_roof_shape_gives_walls_up_to_mid_roof(self):
        rec, _ = self.records([(LAT, LON, 5000, 20.0, 8.0)])
        at = rec.fill({"roof_shape": None, "height": None}, *ring(0, 0))
        self.assertIsNone(at["roof_shape"])
        self.assertAlmostEqual(at["height"], 16.0, 3)

    def test_a_record_beside_the_footprint_does_not_join(self):
        rec, _ = self.records([(LAT, LON, 3100, 9.4, 3.2)])
        at = rec.fill({"roof_shape": None, "height": None}, *ring(40, 0))
        self.assertIsNone(at["roof_shape"])

    def test_the_tallest_part_says_what_the_building_is(self):
        d = 2e-5
        rec, _ = self.records([(LAT, LON, 1000, 4.0, 0.0), (LAT + d, LON, 3200, 11.0, 4.0)])
        at = rec.fill({"roof_shape": None, "height": None}, *ring(0, 0))
        self.assertEqual(at["roof_shape"], "hipped")
        self.assertAlmostEqual(at["height"], 11.0, 3)

    def test_plan_draws_the_filled_shape(self):
        rec, _ = self.records([(LAT, LON, 3400, 9.0, 3.5)])
        e, n = ring(0, 0)
        at = rec.fill({"class": "house", "roof_shape": None, "height": None, "seed_id": "a"}, e, n)
        spec = bld.plan(np.stack([e, n], 1), at)
        self.assertEqual(spec.roof, "mansard")


class ShardTest(Base):
    def test_every_code_in_the_table_is_a_shape_plan_knows(self):
        known = {"gabled", "hipped", "flat", "pyramidal", "half-hipped", "mansard", "skillion", "round"} | set(bld.DOME_ROOFS)
        self.assertLessEqual(set(lr.SHAPES.values()), known)

    def test_nothing_is_asked_for_outside_germany(self):
        self.assertEqual(lr.tiles_over(40.0, -3.0, 40.2, -2.8), [])
        f = Fetcher(shard([(LAT, LON, 3100, 9.0, 3.0)]))
        self.assertIsNone(lr.Source(None, f).tile(geodesy.EnuFrame(40.0, -3.0), -200, -200, 200, 200))
        self.assertEqual(f.urls, [])

    def test_a_missing_shard_means_the_predicted_roofs_and_is_remembered(self):
        f = Fetcher(error=netfetch.NotFound("none"))
        src = lr.Source(None, f)
        frame = geodesy.EnuFrame(LAT, LON)
        self.assertIsNone(src.tile(frame, -200, -200, 200, 200))
        asked = len(f.urls)
        self.assertGreater(asked, 0)
        self.assertIsNone(src.tile(frame, -200, -200, 200, 200))
        self.assertEqual(len(f.urls), asked)

    def test_a_missing_shard_kept_on_disk_is_asked_again_after_a_week(self):
        """A 404 is kept as an empty file (no asking again every session), but not for good: before the shards were
        published every German tile answered 404, and those players would never have got the roofs."""
        import os, tempfile, time
        with tempfile.TemporaryDirectory() as d:
            f = Fetcher(error=netfetch.NotFound("none"))
            frame = geodesy.EnuFrame(LAT, LON)
            self.assertIsNone(lr.Source(d, f).tile(frame, -200, -200, 200, 200))
            kept = [os.path.join(dp, n) for dp, _, ns in os.walk(d) for n in ns if n.endswith(".npz")]
            self.assertTrue(kept and all(os.path.getsize(p) == 0 for p in kept))
            asked = len(f.urls)
            lr.Source(d, f).tile(frame, -200, -200, 200, 200)
            self.assertEqual(len(f.urls), asked, "a fresh miss on disk: not asked again")
            old = time.time() - lr.MISS_TTL_S - 60
            for p in kept:
                os.utime(p, (old, old))
            lr.Source(d, f).tile(frame, -200, -200, 200, 200)
            self.assertGreater(len(f.urls), asked, "a week-old miss: asked again")

    def test_offline_asks_again_only_after_a_while(self):
        f = Fetcher(error=OSError("offline"))
        src = lr.Source(None, f)
        frame = geodesy.EnuFrame(LAT, LON)
        self.assertIsNone(src.tile(frame, -200, -200, 200, 200))
        asked = len(f.urls)
        self.assertIsNone(src.tile(frame, -200, -200, 200, 200))
        self.assertEqual(len(f.urls), asked)

    def test_a_shard_on_disk_is_not_fetched_again(self):
        with tempfile.TemporaryDirectory() as d:
            f = Fetcher(shard([(LAT, LON, 3100, 9.0, 3.0)]))
            frame = geodesy.EnuFrame(LAT, LON)
            self.assertIsNotNone(lr.Source(d, f).tile(frame, -200, -200, 200, 200))
            asked = len(f.urls)
            self.assertIsNotNone(lr.Source(d, f).tile(frame, -200, -200, 200, 200))
            self.assertEqual(len(f.urls), asked)

    def test_shards_from_the_host_the_folders_manifest_names(self):
        import json
        body = shard([(LAT, LON, 3100, 9.0, 3.0)])

        class Site(Fetcher):
            def get(self, url, kind="demand", **_):
                self.urls.append(url)
                if url.endswith("/manifest.json"):
                    return Body(json.dumps({"built": "2026-11-09", "data": "https://tiles.test/data/lod2/k1"}).encode())
                return Body(body)
        with tempfile.TemporaryDirectory() as d:
            f = Site()
            self.assertIsNotNone(lr.Source(d, f).tile(geodesy.EnuFrame(LAT, LON), -200, -200, 200, 200))
            x, y = lr.tile_of(LAT, LON)
            self.assertIn(f"https://tiles.test/data/lod2/k1/{x}/{y}.npz", f.urls)
            self.assertFalse([u for u in f.urls if u.endswith(".npz") and u.startswith(lr.railtags.PAGES)])

    def test_a_broken_shard_reads_as_none(self):
        self.assertEqual(lr.decode(b"not a shard"), {})

    def test_two_workers_fetching_the_same_shard_write_with_their_own_temporary_names(self):
        import threading
        from unittest import mock
        body = shard([(LAT, LON, 3100, 9.0, 3.0)])
        frame = geodesy.EnuFrame(LAT, LON)
        x, y = lr.tile_of(LAT, LON)
        barrier = threading.Barrier(2)

        class Waiting(Fetcher):
            def get(self, url, kind="demand", **_):
                if url.endswith(".npz"):
                    barrier.wait(timeout=10)
                return super().get(url, kind, **_)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pagesdata", "lod2", f"{x}_{y}.npz")
            errors, out = [], []

            def worker():
                try:
                    out.append(lr.Source(d, Waiting(body)).tile(frame, -200, -200, 200, 200))
                except Exception as e:
                    errors.append(e)
            with mock.patch("os.replace", wraps=os.replace) as rep:
                ts = [threading.Thread(target=worker) for _ in range(2)]
                for t in ts:
                    t.start()
                for t in ts:
                    t.join()
            self.assertEqual(errors, [])
            self.assertTrue(all(o is not None for o in out))
            sources = {c.args[0] for c in rep.call_args_list if c.args[1] == path}
            self.assertEqual(len(sources), 2)
            self.assertNotIn(path + ".part", sources)
            self.assertEqual([n for _, _, ns in os.walk(d) for n in ns if n.endswith(".part")], [])
            with open(path, "rb") as fh:
                self.assertTrue(lr.decode(fh.read()))

    def test_an_expired_miss_in_memory_is_asked_again(self):
        import time
        from unittest import mock
        f = Fetcher(error=netfetch.NotFound("none"))
        src = lr.Source(None, f)
        frame = geodesy.EnuFrame(LAT, LON)
        self.assertIsNone(src.tile(frame, -200, -200, 200, 200))
        asked = len(f.urls)
        self.assertIsNone(src.tile(frame, -200, -200, 200, 200))
        self.assertEqual(len(f.urls), asked)
        with mock.patch.object(lr.time, "time", return_value=time.time() + lr.MISS_TTL_S + 60):
            src.tile(frame, -200, -200, 200, 200)
        self.assertGreater(len(f.urls), asked)

    def test_a_network_error_is_not_remembered_as_a_miss(self):
        f = Fetcher(error=OSError("offline"))
        src = lr.Source(None, f)
        frame = geodesy.EnuFrame(LAT, LON)
        self.assertIsNone(src.tile(frame, -200, -200, 200, 200))
        self.assertEqual(src.negatives, {})
        f.error, f.body = None, shard([(LAT, LON, 3100, 9.0, 3.0)])
        src.down_at = 0.0
        self.assertIsNotNone(src.tile(frame, -200, -200, 200, 200))

    def test_a_damaged_shard_on_disk_is_loaded_again_once(self):
        frame = geodesy.EnuFrame(LAT, LON)
        x, y = lr.tile_of(LAT, LON)
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNotNone(lr.Source(d, Fetcher(shard([(LAT, LON, 3100, 9.0, 3.0)]))).tile(frame, -200, -200, 200, 200))
            path = os.path.join(d, "pagesdata", "lod2", f"{x}_{y}.npz")
            with open(path, "wb") as fh:
                fh.write(b"garbage")
            f = Fetcher(shard([(LAT, LON, 3100, 9.0, 3.0)]))
            src = lr.Source(d, f)
            self.assertIsNotNone(src.tile(frame, -200, -200, 200, 200))
            self.assertGreater(len(f.urls), 0)
            asked = len(f.urls)
            with open(path, "rb") as fh:
                self.assertTrue(lr.decode(fh.read()))
            self.assertIsNotNone(src.tile(frame, -200, -200, 200, 200))
            self.assertEqual(len(f.urls), asked)

    def test_a_damaged_download_is_not_kept(self):
        frame = geodesy.EnuFrame(LAT, LON)
        with tempfile.TemporaryDirectory() as d:
            f = Fetcher(b"<html>portal</html>")
            src = lr.Source(d, f)
            self.assertIsNone(src.tile(frame, -200, -200, 200, 200))
            self.assertTrue(any(u.endswith(".npz") for u in f.urls))
            self.assertEqual([n for _, _, ns in os.walk(d) for n in ns if n.endswith(".npz")], [])
            asked = len(f.urls)
            self.assertIsNone(src.tile(frame, -200, -200, 200, 200))
            self.assertEqual(len(f.urls), asked)

    def test_a_damaged_local_shard_is_remembered_not_deleted(self):
        frame = geodesy.EnuFrame(LAT, LON)
        x, y = lr.tile_of(LAT, LON)
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "lod2", str(x)))
            path = os.path.join(d, "lod2", str(x), f"{y}.npz")
            with open(path, "wb") as fh:
                fh.write(b"garbage")
            src = lr.Source(None, None, local=d)
            self.assertIsNone(src.tile(frame, -200, -200, 200, 200))
            self.assertTrue(os.path.exists(path))
            self.assertIn(("lod2", x, y), src.negatives)
            self.assertIsNone(src.tile(frame, -200, -200, 200, 200))
            self.assertEqual(src.negatives, {("lod2", x, y): src.negatives[("lod2", x, y)]})



class CreditTest(unittest.TestCase):
    def test_credited_only_with_a_shard_on_disk(self):
        import tempfile, credits
        d = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, d, True)
        lat, lon = 52.52, 13.405                    # (Berlin)
        self.assertFalse(lr.credited_near(d, lat, lon, 500.0))
        x, y = lr.tile_of(lat, lon)
        os.makedirs(os.path.join(d, "pagesdata", "lod2"))
        path = os.path.join(d, "pagesdata", "lod2", f"{x}_{y}.npz")
        open(path, "wb").close()                    # (a 404 kept as an empty file: no records, no credit)
        self.assertFalse(lr.credited_near(d, lat, lon, 500.0))
        with open(path, "wb") as fh:
            fh.write(b"x")
        self.assertTrue(lr.credited_near(d, lat, lon, 500.0))
        self.assertFalse(lr.credited_near(d, 48.85, 2.35, 500.0))     # (Paris: outside Germany)
        self.assertEqual(lr.credited_near(d, lat, lon, 500.0), {"roof:lod2-de"})
        self.assertIn("GeoBasis-DE", credits.full_label("roof:lod2-de"))
        self.assertIn("roof:lod2-de", credits.ORDER)


class CountriesTest(unittest.TestCase):
    """Several countries' shards, each in its own folder and box; a local folder (BNE_LOD2_DIR) instead of the site."""

    def folders(self, lat, lon):
        f = Fetcher(error=netfetch.NotFound("none"))
        lr.Source(None, f).tile(geodesy.EnuFrame(lat, lon), -200, -200, 200, 200)
        return {u.split(f"{lr.railtags.PAGES}/", 1)[1].split("/")[0] for u in f.urls}

    def test_each_place_asks_only_the_countries_whose_box_it_is_in(self):
        self.assertEqual(self.folders(59.437, 24.7536), {"lod2-ee"})                 # (Tallinn)
        self.assertEqual(self.folders(46.948, 7.4474), {"lod2-ch"})                  # (Bern)
        self.assertEqual(self.folders(47.3769, 8.5417), {"lod2", "lod2-ch"})         # (Zurich: in both boxes)
        self.assertEqual(self.folders(52.52, 13.405), {"lod2"})                      # (Berlin)
        self.assertEqual(self.folders(48.2082, 16.3738), {"lod2-wien"})              # (Vienna)
        self.assertEqual(self.folders(47.07, 15.44), set())                          # (Graz: Austria, not Vienna)
        self.assertEqual(self.folders(48.85, 2.35), set())                           # (Paris)

    def local(self, d, folder, lat, lon, rows):
        x, y = lr.tile_of(lat, lon)
        os.makedirs(os.path.join(d, folder, str(x)), exist_ok=True)
        with open(os.path.join(d, folder, str(x), f"{y}.npz"), "wb") as fh:
            fh.write(shard(rows))

    def test_a_local_folder_is_read_and_nothing_fetched(self):
        lat, lon = 59.437, 24.7536
        with tempfile.TemporaryDirectory() as d:
            self.local(d, "lod2-ee", lat, lon, [(lat, lon, 3200, 12.0, 4.0)])
            f = Fetcher(error=AssertionError("fetched"))
            rec = lr.Source(None, f, local=d).tile(geodesy.EnuFrame(lat, lon), -200, -200, 200, 200)
            self.assertEqual(f.urls, [])
            at = rec.fill({"roof_shape": None, "height": None}, *ring(0, 0))
            self.assertEqual((at["roof_shape"], at["height"]), ("hipped", 12.0))

    def test_a_border_tile_joins_both_countries_records(self):
        lat, lon = 47.3769, 8.5417
        with tempfile.TemporaryDirectory() as d:
            self.local(d, "lod2", lat, lon, [(lat, lon + 0.002, 3100, 9.0, 3.0)])
            self.local(d, "lod2-ch", lat, lon, [(lat, lon, 3300, 14.0, 5.0)])
            frame = geodesy.EnuFrame(lat, lon)
            rec = lr.Source(None, None, local=d).tile(frame, -400, -400, 400, 400)
            self.assertEqual(rec.fill({"roof_shape": None, "height": None}, *ring(0, 0))["roof_shape"], "half-hipped")
            e, _, _ = frame.geodetic_to_enu(np.array([lat]), np.array([lon + 0.002]), 0.0)
            self.assertEqual(rec.fill({"roof_shape": None}, *ring(float(e[0]), 0))["roof_shape"], "gabled")

    def test_each_country_credited_near_its_own_shards(self):
        import credits
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(lr.credited_near(None, 59.437, 24.7536, 500.0, local=d), set())
            self.local(d, "lod2-ee", 59.437, 24.7536, [(59.437, 24.7536, 3100, 9.0, 3.0)])
            self.local(d, "lod2-ch", 46.948, 7.4474, [(46.948, 7.4474, 3100, 9.0, 3.0)])
            self.assertEqual(lr.credited_near(None, 59.437, 24.7536, 500.0, local=d), {"roof:lod2-ee"})
            self.assertEqual(lr.credited_near(None, 46.948, 7.4474, 500.0, local=d), {"roof:lod2-ch"})
            self.assertEqual(lr.credited_near(None, 52.52, 13.405, 500.0, local=d), set())
        for c in lr.SOURCES:
            self.assertIn(c.credit, credits.ORDER)
        self.assertIn("swisstopo", credits.full_label("roof:lod2-ch"))
        self.assertIn("Land and Spatial Development Board", credits.full_label("roof:lod2-ee"))


if __name__ == "__main__":
    unittest.main()
