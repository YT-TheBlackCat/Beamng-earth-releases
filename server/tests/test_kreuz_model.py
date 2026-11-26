"""Junction model, Phase 2 Brief B1: connectors, nodes, split chains and chain extensions (junctionmodel.py 2.1-2.4).

Synthetic rows (fid, spec, pts, lon) first; then the K1 fixture's real rows (probe run, radius 80 m, offline).
"""
import os
import random
import sys
import types
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("BNE_REGIONAL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                                                       "regional"))
os.environ.setdefault("BNGE_PLANCACHE", "off")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tools"))

import junctionmodel as jm  # noqa: E402


def sp(w=7.0, cls="residential"):
    return types.SimpleNamespace(width=w, cls=cls, oneway=False, dir=1, urban=True, roundabout=False, walk=2.0)


def row(fid, pts, w=7.0):
    return (fid, sp(w), np.array([[x, y, 0.0] for x, y in pts], float), 9.0)


def nodes_by_key(nodes):
    return {n.key: n for n in nodes}


class TestJunctionModel(unittest.TestCase):

    def test_t_junction_is_one_node_of_degree_3(self):
        rows = [row("a", [(0, 0), (0, 20)]), row("b", [(0, 0), (-20, 0)]), row("c", [(0, 0), (20, 0)])]
        nodes, chains, node_of, chain_of = jm.nodes_and_chains(rows)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].key, "v0.00,0.00")
        self.assertEqual(nodes[0].degree, 3)
        self.assertTrue(all(len(c.rows) == 1 for c in chains))
        self.assertEqual(sorted(node_of), [(0, 0), (1, 0), (2, 0)])

    def test_split_joins_two_rows_into_one_chain(self):
        rows = [row("c1", [(0, 0), (0, -12)]), row("c2", [(0, -12), (0, -60)])]
        nodes, chains, _node_of, _chain_of = jm.nodes_and_chains(rows)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].degree, 2)
        self.assertEqual([r for r, _rev in chains[0].rows], [0, 1])
        self.assertEqual(len(chains), 1)
        self.assertIsNone(chains[0].start)
        self.assertIsNone(chains[0].end)
        self.assertAlmostEqual(float(chains[0].length), 60.0)
        ext = jm.chain_ext(rows, nodes, chains, _node_of)
        self.assertAlmostEqual(ext[0, 1], 48.0)     # beyond c1's end: c2
        self.assertAlmostEqual(ext[1, 0], 12.0)     # beyond c2's start: c1

    def test_ends_0_05_m_apart_across_a_grid_line_are_one_node(self):
        # 0.5 m grid: 249.98 and 250.03 lie in neighbouring cells (R2-15)
        rows = [row("a", [(-10, 0), (0, 249.98)]), row("b", [(10, 0), (0, 250.03)]),
                row("c", [(0, 250.0), (0, 260)])]
        nodes, _chains, node_of, _ = jm.nodes_and_chains(rows)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(node_of[(0, 1)], node_of[(1, 1)])
        self.assertEqual(node_of[(0, 1)], node_of[(2, 0)])

    def test_connectors_join_rows_2_m_apart(self):
        a = row("a", [(0, -30), (0, 0)])
        b = row("b", [(2, -30), (2, 0)])
        nodes, _c, _n, _k = jm.nodes_and_chains([a, b])
        self.assertEqual(nodes, [])                  # 2 m apart without a shared ID: two loose ends
        nodes, _c, _n, _k = jm.nodes_and_chains([a, b], connectors={0: [(1, "7")], 1: [(1, "7")]})
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].key, "c7")
        self.assertEqual(nodes[0].cids, ("7",))

    def test_row_order_does_not_matter(self):
        rows = [row("a", [(0, 0), (0, 20)]), row("b", [(0, 0), (-20, 0)]), row("c", [(0, 0), (20, 0)]),
                row("d", [(0, -12), (0, -60)]), row("e", [(0, -60), (30, -60)]), row("f", [(0, -12), (0, 0)])]

        def summary(rs):
            nodes, chains, _n, _k = jm.nodes_and_chains(rs)
            ns = sorted((n.key, n.degree, tuple(sorted((rs[r][0], vi) for r, vi in n.slots))) for n in nodes)
            cs = sorted((sorted(rs[r][0] for r, _ in c.rows), round(float(c.length), 6)) for c in chains)
            return ns, cs

        base = summary(rows)
        for seed in (7, 8, 9):
            shuffled = list(rows)
            random.Random(seed).shuffle(shuffled)
            self.assertEqual(summary(shuffled), base)

    def test_k1_fixture_connectors_and_junction_nodes(self):
        import leonberg_junction_core as core
        import leonberg_junction_probe as probe
        lat, lon, roads, infra, routes, urban_fn, widths, _sources = probe.fixture_inputs("K1")
        frame, origin = core.build_frame(lat, lon)
        run = core.run(frame, origin, 80.0, roads, infra, routes, urban_fn=urban_fn, widths=widths)
        rows = run.roads
        conn = jm.row_connectors(rows, run.feats_of, run.frame)
        pairs = sum(len(v) for v in conn.values())
        self.assertGreaterEqual(pairs, 220)
        nodes, _chains, _node_of, _chain_of = jm.nodes_and_chains(rows, conn)
        spot = np.asarray(run.origin, np.float64)[:2]
        near = [n for n in nodes if n.degree >= 3 and float(np.hypot(*(n.xy - spot))) <= 90.0]
        prefixes = ["c563c04fc", "cb1393f29", "c131a9a24", "c30ed6e02", "c7c9be5fb"]
        self.assertEqual(len(near), len(prefixes), [n.key for n in near])
        for p in prefixes:
            self.assertTrue(any(n.key.startswith(p) for n in near), p)


def ow(w, cls, d):
    """A one-way spec with direction d (+1 or -1), as the divided road's carriageways."""
    s = sp(w, cls)
    s.oneway, s.dir = True, d
    return s


def t4_rows(L=12):
    return [row("a", [(-60, 0), (0, 0)], 9.0), row("b", [(0, 0), (60, 0)], 9.0),
            row("c1", [(0, 0), (0, -L)], 7.0), row("c2", [(0, -L), (0, -60)], 7.0), row("d", [(0, 0), (0, 60)], 7.0)]


def offset_rows(off=6.0):
    return [row("a", [(-60, 0), (0, 0)], 9.0), row("m", [(0, 0), (off, 0)], 9.0),
            row("b", [(off, 0), (60, 0)], 9.0), row("c", [(0, 0), (0, -60)], 12.0),
            row("d", [(off, 0), (off, 60)], 12.0)]


def by_fid(m, key):
    return {a.fid: a for a in m.junction(key).arms}


class TestJunctionArms(unittest.TestCase):

    def test_t_arms_length_start_and_through(self):
        rows = [row("a", [(-8, 0), (0, 0)]), row("b", [(0, 0), (8, 0)]), row("c", [(0, 0), (0, -8)])]
        m = jm.build(rows)
        self.assertEqual(len(m.junctions), 1)
        jn = m.junction(next(iter(m.junctions)))
        self.assertEqual(len(jn.arms), 3)
        arms = by_fid(m, jn.key)
        self.assertAlmostEqual(arms["a"].length, 8.0)
        self.assertEqual(m.arms_dict()[jn.key][[t[0] for t in m.arms_dict()[jn.key]].index("a")][5], 8.0)
        self.assertEqual(arms["a"].as_tuple()[4], False)           # a ends at the node: step -1
        self.assertEqual(arms["b"].as_tuple()[4], True)
        self.assertEqual(jn.arms[arms["a"].through].fid, "b")
        self.assertEqual(jn.arms[arms["b"].through].fid, "a")
        self.assertIsNone(arms["c"].through)

    def test_t4_split_arm_runs_over_split_to_60_m(self):
        m = jm.build(t4_rows(12))
        self.assertEqual(len(m.junctions), 1)
        jn = m.junction(next(iter(m.junctions)))
        self.assertEqual(len(jn.arms), 4)
        c = by_fid(m, jn.key)["c1"]
        self.assertAlmostEqual(c.length, 60.0)
        self.assertAlmostEqual(float(np.sum(np.hypot(*np.diff(c.line[:, :2], axis=0).T))), 60.0)
        self.assertEqual(c.chain, ((2, False), (3, False)))

    def test_offset_tee_8(self):
        """Phase 2, R2-02: offsetT with 6 m offset and 12 m side roads is one junction; the middle road is internal."""
        m = jm.build(offset_rows(6.0))
        self.assertEqual(len(m.junctions), 1)
        self.assertIn(m.chain_of[1], m.internal)
        k = m.kernel_topology(5)
        self.assertTrue(k["internal"][k["rank"][1]])
        self.assertEqual(sorted(a.fid for a in m.junction(next(iter(m.junctions))).arms), ["a", "b", "c", "d"])

    def test_divided_stops_incoming_only(self):
        cw1, cw2 = ow(7.0, "primary", 1), ow(7.0, "primary", -1)
        x0 = sp(8.0, "tertiary")
        rows = [row("n1", [(-80, 7), (0, 7)], 7.0), row("n2", [(0, 7), (80, 7)], 7.0),
                row("s1", [(-80, -7), (0, -7)], 7.0), row("s2", [(0, -7), (80, -7)], 7.0),
                row("x0", [(0, -7), (0, 7)], 8.0), row("xs", [(0, -7), (0, -80)], 8.0),
                row("xn", [(0, 7), (0, 80)], 8.0)]
        rows[0] = ("n1", cw1, rows[0][2], 9.0)
        rows[1] = ("n2", cw1, rows[1][2], 9.0)
        rows[2] = ("s1", cw2, rows[2][2], 9.0)
        rows[3] = ("s2", cw2, rows[3][2], 9.0)
        rows[4] = ("x0", x0, rows[4][2], 9.0)
        m = jm.build(rows)
        self.assertEqual(len(m.junctions), 2)
        self.assertNotIn(m.chain_of[4], m.internal)

    def test_interior_t_has_two_main_arms(self):
        rows = [row("main", [(-60, 0), (0, 0), (60, 0)], 9.0), row("c", [(0, 0), (0, -60)], 7.0)]
        m = jm.build(rows)
        self.assertEqual(len(m.junctions), 1)
        jn = m.junction(next(iter(m.junctions)))
        self.assertEqual(len(jn.arms), 3)
        main = [a for a in jn.arms if a.fid == "main"]
        self.assertEqual(sorted(a.at_start for a in main), [False, True])

    def test_r2_14_only_the_short_part_is_internal(self):
        rows = [row("x", [(0, 0), (2, 0)]), row("a1", [(-30, 0), (0, 0)]), row("a2", [(0, 0), (0, 30)]),
                row("b1", [(2, 0), (2, -30)]), row("b2", [(2, 0), (30, 0)]),
                row("x", [(100, 0), (130, 0)]), row("c1", [(100, 0), (100, -30)]),
                row("c2", [(100, 0), (100, 30)]), row("d1", [(130, 0), (130, -30)]),
                row("d2", [(130, 0), (130, 30)])]
        m = jm.build(rows)
        self.assertEqual(len(m.junctions), 3)
        internal_rows = {r for c in m.chains if c.id in m.internal for r, _rv in c.rows}
        self.assertEqual(internal_rows, {0})

    def test_row_order_gives_same_arms(self):
        def summary(rs):
            m = jm.build(rs)
            return sorted((k, [a[0] for a in v]) for k, v in m.arms_dict(splits=True).items())

        for base in (t4_rows(12), offset_rows(6.0)):
            ref = summary(base)
            for seed in (3, 4, 5):
                shuffled = list(base)
                random.Random(seed).shuffle(shuffled)
                self.assertEqual(summary(shuffled), ref)

    def test_k1_three_junctions_unbundled_and_far_link(self):
        import islands
        import leonberg_junction_core as core
        import leonberg_junction_probe as probe
        lat, lon, roads, infra, routes, urban_fn, widths, _sources = probe.fixture_inputs("K1")
        frame, origin = core.build_frame(lat, lon)
        run = core.run(frame, origin, 80.0, roads, infra, routes, urban_fn=urban_fn, widths=widths)
        rows = run.roads
        conn = jm.row_connectors(rows, run.feats_of, run.frame)
        m = jm.build(rows, conn, ring_fids=islands.ring_fids(rows))
        for p in ("c563c04fc", "cb1393f29", "c131a9a24"):
            found = [j for k, j in m.junctions.items() if k.startswith(p)]
            self.assertEqual(len(found), 1, p)
            self.assertEqual(len(found[0].arms), 3, p)
            self.assertEqual(len(found[0].nodes), 1, p)          # not bundled
        j1 = next(j for k, j in m.junctions.items() if k.startswith("c563c04fc"))
        j2 = next(k for k in m.junctions if k.startswith("cb1393f29"))
        arm = next(a for a in j1.arms if a.fid.startswith("725d6ac7"))
        self.assertEqual(arm.far, j2)
        self.assertIsInstance(bool(m.link_ok(None)(arm.as_tuple())), bool)

    def test_kernel_topology_shapes(self):
        m = jm.build(t4_rows(12))
        k = m.kernel_topology(5)
        total = sum(len(r[2]) for r in m.rows)
        self.assertEqual(k["node"].shape, (total,))
        self.assertEqual(int((k["node"] >= 0).sum()), 4)     # a, b, c1, d start at the node; the split is -1
        self.assertEqual(k["rank"].shape, (5,))
        self.assertTrue(all(0 <= r < 5 for r in k["rank"]))
        self.assertEqual(k["ext"].shape, (5, 2))
        self.assertIn(3, [r for r, _f in k["same"][2]])


def arms_by_fid(m, key):
    return {a.fid: a for a in m.junction(key).arms}


class TestJunctionGeometry(unittest.TestCase):

    def test_cross_d_edge_of_9_and_7_m_roads(self):
        rows = [row("w", [(-60, 0), (0, 0)], 9), row("e", [(0, 0), (60, 0)], 9),
                row("n", [(0, 0), (0, 60)], 7), row("s", [(0, -60), (0, 0)], 7)]
        m = jm.build(rows)
        jn = m.junction(next(iter(m.junctions)))
        arms = arms_by_fid(m, jn.key)
        self.assertAlmostEqual(arms["n"].d_edge, 4.5, delta=0.25)
        self.assertAlmostEqual(arms["e"].d_edge, 3.5, delta=0.25)

    def test_oblique_stub_d_edge(self):
        # main road 9 m (W-E), stub 7 m at 60 degrees to it (30 degrees off the perpendicular)
        stub = 40 * np.array([0.5, np.sqrt(3) / 2])
        rows = [row("w", [(-40, 0), (0, 0)], 9), row("e", [(0, 0), (40, 0)], 9),
                row("s", [(0, 0), tuple(stub)], 7)]
        m = jm.build(rows)
        jn = m.junction(next(iter(m.junctions)))
        d = arms_by_fid(m, jn.key)["s"].d_edge
        self.assertIsNotNone(d)
        self.assertGreater(d, 4.5 / np.sin(np.radians(60)) - 0.25)

    def test_cross_polygon_is_8_points_ccw(self):
        rows = [row("w", [(-60, 0), (0, 0)], 9), row("e", [(0, 0), (60, 0)], 9),
                row("n", [(0, 0), (0, 60)], 7), row("s", [(0, -60), (0, 0)], 7)]
        m = jm.build(rows)
        poly = m.junction(next(iter(m.junctions))).polygon
        self.assertEqual(poly.shape, (8, 2))
        x, y = poly[:, 0], poly[:, 1]
        self.assertGreater(0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)), 0.0)

    def test_station_after_a_bend(self):
        rows = [row("w", [(-30, 0), (0, 0)], 7), row("e", [(0, 0), (30, 0)], 7),
                row("n", [(0, 0), (0, 20), (20, 20)], 7)]
        m = jm.build(rows)
        jn = m.junction(next(iter(m.junctions)))
        arm = arms_by_fid(m, jn.key)["n"]
        xy, _z, t = m.station(arm, 10.0)
        np.testing.assert_allclose(xy, [0.0, 10.0], atol=1e-9)
        np.testing.assert_allclose(t, [0.0, 1.0], atol=1e-9)
        xy, _z, t = m.station(arm, 30.0)                # 10 m past the bend
        np.testing.assert_allclose(xy, [10.0, 20.0], atol=1e-9)
        np.testing.assert_allclose(t, [1.0, 0.0], atol=1e-9)
        xy, _z, _t = m.station(arm, 100.0)              # clamped to the end of the line
        np.testing.assert_allclose(xy, [20.0, 20.0], atol=1e-9)

    def test_ring_specs_on_a_roundabout(self):
        import roads

        def arc(r, a0, a1, n=12):
            t = np.linspace(a0, a1, n)
            return np.stack([r * np.cos(t), r * np.sin(t), np.zeros(n)], 1)

        ow = roads.Spec(6.0, "paved", True, "residential", True)
        two = roads.Spec(9.0, "paved", True, "residential", False)
        rs = [("r1", ow, arc(16, 0, np.pi), 0.0), ("r2", ow, arc(16, np.pi, 2 * np.pi), 0.0),
              ("app", two, np.array([(16.0, 0.0, 0.0), (80.0, 0.0, 0.0)]), 0.0)]
        out, ring_dir = jm.ring_specs(rs)
        self.assertEqual(set(ring_dir), {"r1", "r2"})
        spec = {fid: sp for fid, sp, _pts, _lon in out}
        for fid in ("r1", "r2"):
            self.assertEqual(spec[fid].dir, ring_dir[fid])
            self.assertTrue(spec[fid].oneway)
            self.assertFalse(spec[fid].urban)
            self.assertIsNone(spec[fid].median)
        self.assertFalse(spec["app"].urban)             # approach shorter than islands.APPROACH_NO_WALK_M
        self.assertIsNone(spec["app"].street)
        self.assertIs(rs[2][1], two)                    # input rows not changed
        self.assertEqual(jm.ring_specs([]), ([], {}))

    def test_crossing_pairs_parallel_no(self):
        m = jm.build([row("A", [(-40, 0), (40, 0)], 9), row("B", [(-40, 6), (40, 6)], 7)])
        self.assertEqual(jm.crossing_pairs(m), set())

    def test_crossing_pairs_cross_without_node_yes(self):
        m = jm.build([row("A", [(-40, 0), (40, 0)], 9), row("C", [(0, -40), (0, 40)], 7)])
        self.assertEqual(jm.crossing_pairs(m), {("A", "C"), ("C", "A")})

    def test_crossing_pairs_t_arms_no(self):
        m = jm.build([row("w", [(-40, 0), (0, 0)], 9), row("e", [(0, 0), (40, 0)], 9),
                      row("n", [(0, 0), (0, 40)], 7)])
        self.assertEqual(jm.crossing_pairs(m), set())


if __name__ == "__main__":
    unittest.main()
