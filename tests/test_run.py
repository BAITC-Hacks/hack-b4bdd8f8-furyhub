"""Проверки приоритетов правил, ограничений данных и готовых CSV.

Запуск после пайплайна: python -m unittest discover -s tests -v
"""

from pathlib import Path
import unittest

import networkx as nx
import pandas as pd

import run


class RulesTests(unittest.TestCase):
    def test_first_match_and_boundaries(self):
        base = dict(
            gid=0, in_deg=1, out_deg=1, in_kzt=100_000, out_kzt=100_000,
            pass_through=1.0, depth=1, is_seed=False, seed_payers=0,
            betweenness=0.0,
        )
        cases = [
            ({"in_deg": 0, "out_deg": 0, "depth": 4}, "peripheral", 0.2),
            # Coordinator wins over distributor and consolidator.
            ({"in_deg": 5, "out_deg": 10, "out_kzt": 10_000,
              "seed_payers": 2, "betweenness": 1.0}, "coordinator", 0.9),
            ({"in_deg": 5, "out_deg": 10, "out_kzt": 10_000}, "distributor", 0.55),
            ({"in_deg": 5, "out_deg": 0, "out_kzt": 0}, "consolidator", 0.75),
            # Seed exclusion applies even when the consolidation condition matches.
            ({"in_deg": 5, "out_deg": 0, "out_kzt": 0, "is_seed": True}, "terminal", 0.7),
            ({"in_deg": 5, "out_kzt": 50_000, "pass_through": 0.5}, "peripheral", 0.4),
            ({"pass_through": 0.8}, "transit", 0.7),
            ({"pass_through": 1.2}, "transit", 0.7),
            ({"pass_through": 1.0}, "transit", 0.9),
            ({"out_deg": 0, "in_kzt": 50_000, "depth": 3}, "terminal", 0.7),
            ({"out_deg": 0, "in_kzt": 49_999}, "peripheral", 0.4),
            ({"out_deg": 0, "depth": 4}, "peripheral", 0.3),
            ({"out_deg": 9, "pass_through": 2.0}, "peripheral", 0.4),
        ]
        frame = pd.DataFrame([base | patch | {"gid": i} for i, (patch, _, _) in enumerate(cases)])
        actual = run.assign_roles(frame)
        for row, (_, role, score) in zip(actual.itertuples(), cases):
            with self.subTest(gid=row.gid):
                self.assertEqual(row.role, role)
                self.assertAlmostEqual(row.role_score, score)

    def test_minmax_penalty_and_score_range(self):
        frame = pd.DataFrame({column: [0.0, 10.0, 0.0] for column in run.PRIORITY_WEIGHTS})
        frame["truncated_by_depth"] = [False, True, True]
        result = run.add_priority(frame)
        self.assertEqual(result.priority_score.round(8).tolist(), [0.0, 0.7, 0.0])
        self.assertEqual(result.priority_score_raw.round(8).tolist(), [0.0, 0.7, -0.3])
        self.assertEqual(run.minmax(pd.Series([7.0, 7.0])).tolist(), [0.0, 0.0])

    def test_projection_preserves_reciprocal_flows_and_isolates(self):
        graph = nx.DiGraph()
        graph.add_node(3)
        graph.add_edge(1, 2, sum_kzt=100.0)
        graph.add_edge(2, 1, sum_kzt=200.0)
        projection = run.undirected_projection(graph)
        self.assertEqual(projection[1][2]["sum_kzt"], 300.0)
        self.assertEqual(set(projection), {1, 2, 3})


class OutputTests(unittest.TestCase):
    def test_provided_dataset_and_exports(self):
        root = Path(__file__).resolve().parents[1]
        nodes = pd.read_parquet(root / "data/nodes.parquet")
        edges = pd.read_parquet(root / "data/edges.parquet")
        roles = pd.read_csv(root / "out/nodes_roles.csv")
        clusters = pd.read_csv(root / "out/clusters.csv")
        top = pd.read_csv(root / "out/top_nodes.csv")
        self.assertEqual(len(roles), 2248)
        self.assertTrue(roles.gid.is_unique)
        self.assertEqual(set(roles.gid), set(nodes.gid))
        self.assertTrue(roles.evidence.str.len().between(1, 200).all())
        self.assertTrue(roles.evidence.str.contains(r"\d").all())
        self.assertTrue(roles.loc[roles.is_seed, "evidence"].str.contains(run.SEED_NOTE, regex=False).all())
        self.assertFalse((roles.is_seed & roles.role.eq("consolidator")).any())
        self.assertFalse((roles.truncated_by_depth & roles.role.eq("terminal")).any())
        required = ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"]
        self.assertFalse(roles[required].isna().any().any())
        self.assertTrue(roles.role.isin(run.ROLES).all())
        self.assertTrue(roles.priority_score.between(0, 1).all())
        self.assertTrue(roles.role_score.between(0, 1).all())
        self.assertEqual(len(roles.loc[(roles.in_deg + roles.out_deg).eq(0)]), 19)
        self.assertEqual(len(top), 25)
        self.assertEqual(top["rank"].tolist(), list(range(1, 26)))
        self.assertTrue(top.priority_score.is_monotonic_decreasing)
        self.assertEqual(top.gid.tolist(), run.rank_nodes(roles).head(25).gid.tolist())
        self.assertFalse(top.why.isna().any())
        # Reconstruct scores independently from the published formula.
        expected = pd.Series(0.0, index=roles.index)
        for weight, values in [
            (0.35, roles.pagerank), (0.25, roles.in_deg),
            (0.20, roles.in_kzt + roles.out_kzt),
            (0.10, roles.seed_payers_share), (0.10, roles.betweenness),
        ]:
            expected += weight * (values - values.min()) / (values.max() - values.min())
        expected -= 0.30 * roles.truncated_by_depth
        self.assertLess((roles.priority_score - expected.clip(0, 1)).abs().max(), 1e-12)
        seed_set = set(nodes.loc[nodes.is_seed, "gid"])
        counts = edges.loc[edges.src.isin(seed_set)].groupby("dst").src.nunique()
        payers = roles.gid.map(counts).fillna(0)
        self.assertLess((roles.seed_payers_share - payers / roles.in_deg.clip(lower=1)).abs().max(), 1e-12)
        self.assertEqual(clusters.n_nodes.sum(), len(nodes))
        self.assertEqual(clusters.n_seed.sum(), int(nodes.is_seed.sum()))
        self.assertEqual(set(clusters.cluster_id), set(roles.cluster_id))
        membership = roles.set_index("gid").cluster_id
        for cluster in clusters.itertuples(index=False):
            group = roles.loc[roles.cluster_id.eq(cluster.cluster_id)]
            internal = (edges.src.map(membership).eq(cluster.cluster_id)
                        & edges.dst.map(membership).eq(cluster.cluster_id))
            self.assertAlmostEqual(cluster.sum_kzt_internal, edges.loc[internal, "sum_kzt"].sum())
            self.assertEqual(cluster.n_nodes, len(group))
            self.assertEqual(cluster.top_gids, ";".join(run.rank_nodes(group).head(5).gid.astype(str)))


if __name__ == "__main__":
    unittest.main()
