"""Lane lines where the map data looks odd (Lasse, 2026-11-26, docs/leonberg/kreuzungen/SPUREN.md):
- roads.doubtful_no_markings: OSM lane_markings=no is not believed on a long two-way carriageway of primary to tertiary
  class that is wide enough for two lanes and not slow (Renninger Straße from the Ezach roundabout), but is believed on
  small streets, in 30 zones, on one-way roads and on short pieces;
- designs.demote_bypass_nodes: the split node of a right-turn slip lane next to a signalled crossing is no signalled
  junction (its stop line cut every lane line for 40 to 110 m of Brennerstraße / Römerstraße)."""
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import designs  # noqa: E402
import furniture  # noqa: E402
import roads  # noqa: E402

NO = {"lane_markings": "no"}


def spec(cls="tertiary", width=6.5, oneway=False, kmh=50, lanes=None):
    return roads.Spec(width, "paved", True, cls, oneway, kmh=kmh, lanes=lanes)


def line(length):
    return np.array([[0.0, 0.0, 0.0], [float(length), 0.0, 0.0]])


class NoMarkingsDoubtTest(unittest.TestCase):
    def doubt(self, sp=None, length=495, osm=None):
        return roads.doubtful_no_markings(sp or spec(), line(length), osm if osm is not None else NO)

    def test_a_long_wide_street_of_tertiary_class_is_doubted(self):
        self.assertTrue(self.doubt())
        self.assertTrue(self.doubt(spec("secondary", 7.5)))

    def test_nothing_to_doubt_without_the_tag(self):
        self.assertFalse(self.doubt(osm={}))
        self.assertFalse(self.doubt(osm={"lane_markings": "yes"}))

    def test_the_tag_is_believed_where_it_is_plausible(self):
        self.assertFalse(self.doubt(spec("residential")), "a residential street")
        self.assertFalse(self.doubt(spec(width=5.0)), "a narrow carriageway")
        self.assertFalse(self.doubt(spec(oneway=True)), "a one-way carriageway")
        self.assertFalse(self.doubt(spec(kmh=30)), "a slow street")
        self.assertFalse(self.doubt(length=40), "a short piece, a junction's or a roundabout's")
        self.assertFalse(self.doubt(osm={**NO, "zone:maxspeed": "DE:30"}), "a 30 zone")
        self.assertFalse(self.doubt(osm={**NO, "junction": "roundabout"}), "a ring")

    def test_extras_paints_the_doubted_street_and_not_the_small_one(self):
        def centre(sp, length, osm):
            return [lk for lk, _w, _ln in roads.extras(sp, line(length), 9.0, country="DE", osm=osm)
                    if lk.get("kind") == "centre"]
        self.assertTrue(centre(spec(), 495, NO), "the 495 m of Renninger Straße get their centre line")
        self.assertFalse(centre(spec("residential"), 495, NO))
        self.assertFalse(centre(spec(), 40, NO))

    def test_trust_switch(self):
        old = roads.NO_MARKINGS_TRUST
        roads.NO_MARKINGS_TRUST = True
        try:
            self.assertFalse(self.doubt())
        finally:
            roads.NO_MARKINGS_TRUST = old


def _arm(index, fid, sp, through, far=None, length=200.0):
    return SimpleNamespace(id=("k", index), fid=fid, spec=sp, through=through, far=far, length=length)


def _model(**junctions):
    return SimpleNamespace(junctions={k: SimpleNamespace(arms=arms) for k, arms in junctions.items()})


def _signal(arms):
    return (arms, [(a, furniture.G_SIGNAL, -1) for a in arms])


class BypassNodeTest(unittest.TestCase):
    """A crossing X (four arms) and, 44 m along the main road, the split node S of a right-turn slip lane."""

    def scene(self, slip_lanes=1, slip_oneway=True, far_signalled=True, length=44.0, x_arms=4):
        main = spec("secondary", 14.0)
        slip = spec("secondary", 3.5 * slip_lanes, oneway=slip_oneway, lanes=slip_lanes)
        s_arms = [_arm(0, "m1", main, through=1, far="X", length=length), _arm(1, "m2", main, through=0, far=None),
                  _arm(2, "slip", slip, through=None)]
        x_arms = [_arm(i, f"x{i}", main, through=(i + 2) % 4, far="S" if i == 0 else None, length=length)
                  for i in range(x_arms)]
        model = _model(S=s_arms, X=x_arms)
        out = {"S": _signal([a.as_tuple() if hasattr(a, "as_tuple") else a for a in s_arms]),
               "X": _signal(x_arms) if far_signalled else (x_arms, [(a, 0, -1) for a in x_arms])}
        return model, out

    def regulations(self, out, key):
        return [w for _a, w, _p in out[key][1]]

    def test_the_split_node_loses_its_signal_the_crossing_keeps_it(self):
        model, out = self.scene()
        designs.demote_bypass_nodes(out, model)
        self.assertEqual(set(self.regulations(out, "S")), {0})
        self.assertEqual(set(self.regulations(out, "X")), {furniture.G_SIGNAL})

    def test_not_for_a_two_lane_or_two_way_side_road_a_far_crossing_or_an_unsignalled_one(self):
        for kw in ({"slip_lanes": 2}, {"slip_oneway": False}, {"length": 90.0}, {"far_signalled": False},
                   {"x_arms": 3}):
            model, out = self.scene(**kw)
            designs.demote_bypass_nodes(out, model)
            self.assertEqual(set(self.regulations(out, "S")), {furniture.G_SIGNAL}, kw)


if __name__ == "__main__":
    unittest.main()
