"""routeconditions + routing.plan: the GPS routes round closures and counts traffic delays (the user, 2026-11-26).
OSRM is faked: a straight road west-east (the direct route) and a longer one round a block (the alternative)."""
import json
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import earthd  # noqa: E402
import routeconditions as rc  # noqa: E402
import routing  # noqa: E402

LAT = 48.80
DIRECT = [[9.00, LAT], [9.01, LAT], [9.02, LAT]]                       # (lon, lat) as OSRM sends it
AROUND = [[9.00, LAT], [9.00, LAT + 0.004], [9.02, LAT + 0.004], [9.02, LAT]]


def osrm(*routes):
    return json.dumps({"code": "Ok", "routes": [{"distance": d, "duration": t, "geometry": {"coordinates": c}}
                                                for c, d, t in routes]}).encode()


class FakeNet:
    def __init__(self, answer):
        self.urls, self.answer = [], answer

    def get(self, url, kind=None):
        self.urls.append(url)
        return types.SimpleNamespace(body=self.answer(url) if callable(self.answer) else self.answer)


CLOSURE = {"id": "roadworks", "lat": LAT, "lon": 9.01, "radius_m": 20}
JAM = {"id": "jam", "points": [[LAT, 9.005], [LAT, 9.015]], "extra_s": 300}


class ConditionsTest(unittest.TestCase):
    route = [(LAT, 9.00), (LAT, 9.01), (LAT, 9.02)]

    def test_touches(self):
        self.assertTrue(rc.touches(self.route, CLOSURE))
        self.assertFalse(rc.touches(self.route, dict(CLOSURE, lat=LAT + 0.001)), "110 m beside the road")
        self.assertTrue(rc.touches(self.route, dict(CLOSURE, lat=LAT + 0.001, radius_m=150)))
        cross = {"id": "x", "points": [[LAT - 0.01, 9.012], [LAT + 0.01, 9.012]], "radius_m": 5}
        self.assertTrue(rc.touches(self.route, cross), "a closed road crossing the route between its vertices")
        self.assertFalse(rc.touches(self.route, {"id": "bad", "lat": "x"}), "an invalid place never blocks")
        self.assertFalse(rc.touches([], CLOSURE))

    def test_check(self):
        r = rc.check(self.route, [CLOSURE], [JAM, dict(JAM, id="far", points=[[LAT + 1, 9.0]])])
        self.assertEqual(r, {"blocked": ["roadworks"], "delay": 300.0})

    def test_choose_prefers_the_unblocked_alternative_and_counts_delay(self):
        direct = {"points": [list(p) for p in self.route], "dur": 100.0}
        around = {"points": [[LAT, 9.0], [LAT + 0.004, 9.0], [LAT + 0.004, 9.02], [LAT, 9.02]], "dur": 250.0}
        got = rc.choose([direct, around], [CLOSURE])
        self.assertEqual((got["dur"], got["blocked"], got["avoided"]), (250.0, [], ["roadworks"]))
        got = rc.choose([direct, around], [], [JAM])        # the jam (+300 s) makes the longer way the quicker one
        self.assertEqual((got["dur"], got["delay"], got["avoided"]), (250.0, 0.0, []))
        got = rc.choose([direct], [CLOSURE])                # nothing else: still blocked, and says so
        self.assertEqual(got["blocked"], ["roadworks"])
        self.assertEqual(rc.choose([direct], [], [])["dur"], 100.0)

    def test_detours_are_beside_the_road(self):
        spots = rc.detours(self.route, [CLOSURE])
        self.assertEqual(len(spots), 4)
        for la, lo in spots:
            self.assertAlmostEqual(lo, 9.01, places=5)          # (a west-east road: the detours are north / south)
        self.assertAlmostEqual(abs(spots[0][0] - LAT) * 111320.0, 300.0, delta=1.0)
        self.assertEqual(rc.detours(self.route, []), [])

    def test_store_versions_and_validates(self):
        s = rc.Store()
        self.assertFalse(s.active())
        self.assertEqual(s.set([CLOSURE, {"id": "broken"}, "junk"], None), 1)
        self.assertEqual(s.closures, [CLOSURE])
        self.assertEqual(s.set([CLOSURE]), 1, "no change: same version")
        self.assertEqual(s.set(None, [JAM]), 2)
        self.assertEqual(len(s.closures), 1, "None leaves the other list as it is")
        self.assertEqual(s.set([], None), 3)
        self.assertTrue(s.active())


class PlanTest(unittest.TestCase):
    def test_no_conditions_asks_once_without_alternatives(self):
        net = FakeNet(osrm((DIRECT, 2000, 100)))
        r = routing.plan(LAT, 9.0, LAT, 9.02, net, None, (), rc.Store())
        self.assertEqual(len(net.urls), 1)
        self.assertNotIn("alternatives", net.urls[0])
        self.assertNotIn("blocked", r)

    def test_a_closure_takes_the_alternative(self):
        net = FakeNet(osrm((DIRECT, 2000, 100), (AROUND, 3500, 250)))
        s = rc.Store()
        s.set([CLOSURE])
        r = routing.plan(LAT, 9.0, LAT, 9.02, net, None, (), s)
        self.assertEqual(len(net.urls), 1)
        self.assertIn("alternatives=3", net.urls[0])
        self.assertEqual((r["dur"], r["blocked"], r["avoided"]), (250.0, [], ["roadworks"]))

    def test_delay_is_in_the_duration(self):
        net = FakeNet(osrm((DIRECT, 2000, 100)))
        s = rc.Store()
        s.set(None, [JAM])
        r = routing.plan(LAT, 9.0, LAT, 9.02, net, None, (), s)
        self.assertEqual((r["dur"], r["delay"]), (400.0, 300.0))

    def test_detour_waypoint_when_every_alternative_is_blocked(self):
        def answer(url):
            if ";" in url.split("/driving/")[1].split("?")[0] and url.count(";") == 2:     # with a via: round the block
                return osrm((AROUND, 3500, 250))
            return osrm((DIRECT, 2000, 100))
        net = FakeNet(answer)
        s = rc.Store()
        s.set([CLOSURE])
        r = routing.plan(LAT, 9.0, LAT, 9.02, net, None, (), s)
        self.assertEqual(r["blocked"], [])
        self.assertEqual(len(net.urls), 2, "the first plan and one detour")
        # no way round at all: bounded tries, then blocked is reported
        net = FakeNet(osrm((DIRECT, 2000, 100)))
        r = routing.plan(LAT, 9.0, LAT, 9.02, net, None, (), s)
        self.assertEqual(r["blocked"], ["roadworks"])
        self.assertEqual(len(net.urls), 1 + routing.MAX_DETOUR_TRIES)
        # with waypoints the user chose: no detour search, still reported
        net = FakeNet(osrm((DIRECT, 2000, 100)))
        r = routing.plan(LAT, 9.0, LAT, 9.02, net, None, [(LAT, 9.005)], s)
        self.assertEqual((r["blocked"], len(net.urls)), (["roadworks"], 1))


class ServerMessageTest(unittest.TestCase):
    def test_route_conditions_message_pushes_to_every_client(self):
        got = [[], []]
        conns = [types.SimpleNamespace(send=got[0].append), types.SimpleNamespace(send=got[1].append)]
        me = types.SimpleNamespace(send=got[0].append,
                                   s=types.SimpleNamespace(route_conditions=rc.Store(), connections=conns))
        earthd.Connection.on_route_conditions(me, {"closures": [CLOSURE]})
        self.assertEqual(got[0][0]["version"], 1)
        self.assertEqual(got[0][0]["closures"], 1)
        self.assertEqual(got[1][0]["version"], 1, "the other client hears of it too")
        earthd.Connection.on_route_conditions(me, {"closures": [CLOSURE]})
        self.assertEqual((len(got[0]), len(got[1])), (2, 1), "no change: only the sender is answered")


if __name__ == "__main__":
    unittest.main()
