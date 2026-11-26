"""Germany's junction controls (furniture.junction_controls with country DE: no guessed lights, no US stop signs, only
the map's own stop and give-way signs) and the switch (designs.enabled, always on unless BNE_DEJUNCTIONS=off), and
mapped_signals' filter of the crossing and other non-junction signals."""
import os
import sys
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import designs  # noqa: E402
import furniture  # noqa: E402
import roads  # noqa: E402


def _arm(fid, cls, u, p, oneway=False):
    """An arm (fid, spec, unit vector away, point, at its start, length) of a junction at the origin."""
    return (fid, roads.Spec(width=8.0, kind="paved", urban=True, cls=cls, oneway=oneway), np.array(u, float),
            np.array(p, float), True, 30.0)


def _three_arms():
    """Two primary arms through (east and west) and a residential arm (north)."""
    return [_arm("p1", "primary", (1, 0), (8, 0)), _arm("p2", "primary", (-1, 0), (-8, 0)),
            _arm("r", "residential", (0, 1), (0, 8))]


def _four_arms():
    """Three primary arms and a residential one (north)."""
    return [_arm("p1", "primary", (1, 0), (8, 0)), _arm("p2", "primary", (-1, 0), (-8, 0)),
            _arm("p3", "primary", (0, -1), (0, -8)), _arm("r", "residential", (0, 1), (0, 8))]


class _Feature:
    """One point per feature, the attrs Overture's infrastructure layer carries (mapped_signals reads these)."""
    def __init__(self, attrs, ids, lonlat):
        self.attrs = {k: np.asarray(v) for k, v in attrs.items()}
        self.ids = np.asarray(ids)
        self.lonlat = np.asarray(lonlat, np.float64)
        n = len(self.ids)
        self.coord_off = np.arange(n + 1)
        self.ring_off = np.arange(n + 1)
        self.part_off = np.arange(n + 1)

    def __len__(self):
        return len(self.ids)


class _Frame:
    def geodetic_to_enu(self, lat, lon, h):
        return np.asarray(lon) * 1.0, np.asarray(lat) * 1.0, np.zeros(len(np.atleast_1d(lat)))


class DeControlsTest(unittest.TestCase):
    def test_enabled_always_unless_off(self):
        self.assertTrue(designs.enabled(False))
        with mock.patch.object(designs, "ENV", "off"):
            self.assertFalse(designs.enabled(True))

    def test_de_three_arms_no_stop(self):
        out = furniture.junction_controls(_three_arms(), 0, country="DE", signals=np.zeros((0, 2)), priority=None)
        self.assertEqual(out, [])

    def test_us_three_arms_residential_stops(self):
        out = furniture.junction_controls(_three_arms(), 0, country="US", signals=np.zeros((0, 2)), priority=None)
        self.assertEqual([(a[0], w) for a, w in out], [("r", furniture.G_STOP)])

    def test_de_no_signals_not_guessed(self):
        out = furniture.junction_controls(_four_arms(), 0, country="DE", signals=None, priority=None)
        self.assertEqual(out, [])

    def test_de_mapped_stop_sign_kept(self):
        # a mapped stop sign (x, y, True) 4 m beyond the residential arm's point: that arm stops, nothing else
        pri = np.array([[0.0, 12.0, 1.0]])
        out = furniture.junction_controls(_three_arms(), 0, country="DE", signals=np.zeros((0, 2)), priority=pri)
        self.assertEqual([(a[0], w) for a, w in out], [("r", furniture.G_STOP)])

    def test_junction_signal_filter(self):
        self.assertFalse(furniture._junction_signal('{"traffic_signals":"crossing"}'))
        self.assertFalse(furniture._junction_signal('{"traffic_signals":"ramp_meter"}'))
        self.assertTrue(furniture._junction_signal('{"highway":"traffic_signals"}'))
        self.assertTrue(furniture._junction_signal(None))
        self.assertTrue(furniture._junction_signal("not json"))

    def test_mapped_signals_skips_crossings(self):
        feats = _Feature({"class": ["traffic_signals"] * 2, "subtype": ["transportation"] * 2,
                          "tags": ['{"traffic_signals":"crossing"}', '{"highway":"traffic_signals"}']},
                         ["c", "j"], [[9.0, 48.0], [9.0, 48.0001]])
        out = furniture.mapped_signals(_Frame(), [feats])
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
