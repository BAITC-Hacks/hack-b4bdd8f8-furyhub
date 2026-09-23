"""Проверки приоритетов правил, ограничений данных и готовых CSV.

Запуск после пайплайна: python -m unittest discover -s tests -v
"""

from pathlib import Path
import re
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
            ({"in_deg": 5, "out_deg": 0, "out_kzt": 0, "pass_through": 0.0}, "consolidator", 0.75),
            ({"in_deg": 5, "out_deg": 0, "out_kzt": 0, "pass_through": 0.0,
              "is_seed": True}, "consolidator", 0.75),
            ({"in_deg": 3, "seed_payers": 2, "pass_through": 0.49}, "consolidator", 0.65),
            ({"in_deg": 3, "in_kzt": 400_000, "pass_through": 0.49}, "consolidator", 0.65),
            ({"in_deg": 3, "in_kzt": 399_999, "seed_payers": 1,
              "pass_through": 0.49}, "peripheral", 0.4),
            ({"in_deg": 2, "seed_payers": 2, "in_kzt": 400_000,
              "pass_through": 0.49}, "peripheral", 0.4),
            ({"in_deg": 5, "out_kzt": 50_000, "pass_through": 0.5}, "peripheral", 0.4),
            ({"pass_through": 0.8}, "transit", 0.7),
            ({"pass_through": 1.2}, "transit", 0.7),
            ({"pass_through": 1.0}, "transit", 0.9),
            ({"out_deg": 0, "in_kzt": 50_000, "depth": 3}, "terminal", 0.7),
            ({"out_deg": 0, "in_kzt": 49_999}, "peripheral", 0.4),
            ({"out_deg": 0, "depth": 4}, "peripheral", 0.3),
            ({"out_deg": 9, "pass_through": 2.0}, "transit", 0.5),
            ({"pass_through": 1.200001}, "transit", 0.5),
            ({"is_seed": True, "pass_through": 2.0}, "peripheral", 0.4),
            ({"is_seed": True, "pass_through": 1.0}, "transit", 0.9),
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
        frame["role"] = ["peripheral", "coordinator", "terminal"]
        result = run.add_priority(frame)
        self.assertEqual(result.priority_score.round(8).tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(result.priority_score_raw.round(8).tolist(), [0.0, 0.7, -0.3])
        self.assertEqual(run.minmax(pd.Series([7.0, 7.0])).tolist(), [0.0, 0.0])

    def test_role_weights(self):
        roles = ["peripheral", "coordinator", "consolidator", "distributor", "transit", "terminal", "peripheral"]
        frame = pd.DataFrame({column: [0.0] + [1.0] * 6 for column in run.PRIORITY_WEIGHTS})
        frame["truncated_by_depth"] = False
        frame["role"] = roles
        self.assertEqual(run.add_priority(frame).priority_score.round(8).tolist(),
                         [0.0, 1.0, 1.0, 0.9, 0.8, 0.6, 0.4])

    def test_coordinator_percentile_and_seed_payers(self):
        frame = pd.DataFrame({
            "betweenness": range(100), "in_deg": 3, "out_deg": 3,
            "seed_payers": 1, "is_seed": False, "pass_through": 0.6,
        })
        frame.loc[99, "seed_payers"] = 0
        result = run.assign_roles(frame)
        self.assertEqual(result.index[result.role.eq("coordinator")].tolist(), [95, 96, 97, 98])

    def test_evidence_declension(self):
        for count in [1, 3, 11, 21, 61, 111]:
            singular = count in [1, 21, 61]
            self.assertEqual(run.counted_people(count, "плательщика", "плательщиков"),
                             f"{count} " + ("плательщика" if singular else "плательщиков"))
            self.assertEqual(run.counted_people(count, "получателя", "получателей"),
                             f"{count} " + ("получателя" if singular else "получателей"))

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
        seed_consolidators = roles.loc[roles.is_seed & roles.role.eq("consolidator")]
        self.assertFalse(seed_consolidators.empty)
        for row in seed_consolidators.itertuples():
            self.assertIn(f"из них seed-плательщиков {row.seed_payers}", row.evidence)
        self.assertEqual(int(roles.role.eq("coordinator").sum()), 23)
        self.assertTrue(roles.loc[roles.role.eq("coordinator"), "evidence"].str.contains(
            "в топ-5% по посреднической роли в сети", regex=False).all())
        missing_incoming = roles.loc[roles.role_rule.eq("transit_missing_incoming")]
        self.assertFalse(missing_incoming.empty)
        self.assertFalse(missing_incoming.is_seed.any())
        self.assertTrue(missing_incoming.role_score.eq(0.5).all())
        self.assertTrue(missing_incoming.evidence.str.contains(
            "кандидат на запрос входящих переводов", regex=False).all())
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
        self.assertFalse(top.head(10).role.eq("peripheral").any())
        self.assertEqual(roles.priority_score.max(), 1.0)
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
        expected = expected.clip(0, 1) * roles.role.map({
            "coordinator": 1.0, "consolidator": 1.0, "distributor": 0.9,
            "transit": 0.8, "terminal": 0.6, "peripheral": 0.4,
        })
        expected = (expected - expected.min()) / (expected.max() - expected.min())
        self.assertLess((roles.priority_score - expected).abs().max(), 1e-12)
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
            self.assertTrue(cluster.hypothesis.startswith("Гипотеза для проверки:"))
            self.assertIn(f"seed — {int(group.is_seed.sum())}", cluster.hypothesis)
            for role in run.ROLES:
                self.assertIn(f"{role} — {int(group.role.eq(role).sum())}", cluster.hypothesis)
            share = group.turnover_kzt.sum() / roles.turnover_kzt.sum()
            self.assertIn(f"доля общего оборота (вход + выход) — {share:.2%}", cluster.hypothesis)
            if cluster.n_seed == 0:
                self.assertIn("связь с делом не подтверждена", cluster.hypothesis)
            self.assertEqual(len(re.findall(r"[.!?](?:\s|$)", cluster.hypothesis)), 2)
            self.assertIn("Для проверки следует", cluster.hypothesis)
            self.assertNotIn("курьеров", cluster.hypothesis)


if __name__ == "__main__":
    unittest.main()
