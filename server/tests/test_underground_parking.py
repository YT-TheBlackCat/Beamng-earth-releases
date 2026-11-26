"""An underground car-park service road (a parking aisle or driveway, or a service road at level -1 or below) is no
tunnel: roads._underground_service, and tunnel_specs leaves it out (Leonberg, 2026-11-22)."""
import os
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import roads  # noqa: E402


class FakeTable:
    """A rule table as roads reads it: feature / value / start / end arrays and major(i)."""

    def __init__(self, rows):
        self.feature = np.array([r[0] for r in rows], dtype=np.int64)
        self.value = np.array([r[1] for r in rows], dtype=object)
        self.start = np.zeros(len(rows))
        self.end = np.ones(len(rows))

    def major(self, i):
        hit = np.flatnonzero(self.feature == i)
        return int(hit[0]) if len(hit) else -1


def fake_feats(subclass=(), level=()):
    rules = {}
    if subclass:
        rules["subclass"] = FakeTable(list(subclass))
    if level:
        rules["level"] = FakeTable(list(level))
    return types.SimpleNamespace(rules=rules)


def spec(cls):
    return types.SimpleNamespace(cls=cls)


class UndergroundServiceTest(unittest.TestCase):
    def test_parking_aisle_service_is_underground(self):
        feats = fake_feats(subclass=[(0, "parking_aisle")])
        self.assertTrue(roads._underground_service(feats, 0, spec("service")))

    def test_driveway_service_is_underground(self):
        feats = fake_feats(subclass=[(3, "driveway")])
        self.assertTrue(roads._underground_service(feats, 3, spec("service")))

    def test_service_at_level_minus_two_is_underground(self):
        feats = fake_feats(level=[(1, "-2")])
        self.assertTrue(roads._underground_service(feats, 1, spec("service")))

    def test_service_at_level_zero_without_subclass_is_not(self):
        feats = fake_feats(level=[(2, "0")])
        self.assertFalse(roads._underground_service(feats, 2, spec("service")))
        feats = fake_feats()
        self.assertFalse(roads._underground_service(feats, 2, spec("service")))

    def test_residential_at_level_minus_one_is_not(self):
        feats = fake_feats(level=[(4, "-1")])
        self.assertFalse(roads._underground_service(feats, 4, spec("residential")))

    def test_spec_none_is_not(self):
        feats = fake_feats(subclass=[(5, "parking_aisle")], level=[(5, "-2")])
        self.assertFalse(roads._underground_service(feats, 5, None))

    def test_unparsable_level_is_skipped(self):
        feats = fake_feats(level=[(6, "x"), (6, "-1")])
        self.assertTrue(roads._underground_service(feats, 6, spec("service")))


if __name__ == "__main__":
    unittest.main()
