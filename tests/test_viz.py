"""Данные экрана аналитика, автономность и точность длинных gid."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import llm_layer
import viz


class VisualizationTests(unittest.TestCase):
    def setUp(self):
        self.ids = [str(100000000000000001 + i) for i in range(55)]
        self.nodes = pd.DataFrame({
            "gid": self.ids, "role": ["transit"] * 55,
            "priority_score": [1 - i / 100 for i in range(55)],
            "role_score": [0.8] * 55,
            "evidence": ['</script><img src=x onerror="alert(1)"> & > \u2028\u2029'] * 55,
            "is_seed": [True] + [False] * 54,
            "cluster_id": ["0"] * 2 + ["1"] * 53,
            "depth": [0] + [1] * 54,
            "truncated_by_depth": [False] * 55,
        })
        # 50/51 — прямые соседи топ-50; 52 — через два шага, 54 изолирован.
        self.edges = pd.DataFrame({
            "src": [self.ids[50], self.ids[49], self.ids[51], self.ids[52], self.ids[0], self.ids[1]],
            "dst": [self.ids[49], self.ids[51], self.ids[52], self.ids[53], self.ids[1], self.ids[0]],
            "sum_kzt": [10., 20., 30., 40., 50., 60.],
            "n_tx": [1, 2, 3, 4, 5, 6],
        })
        for direction, endpoint, peer in (("in", "dst", "src"), ("out", "src", "dst")):
            grouped = self.edges.groupby(endpoint)
            self.nodes[f"{direction}_deg"] = self.nodes.gid.map(grouped[peer].nunique()).fillna(0).astype(int)
            self.nodes[f"{direction}_kzt"] = self.nodes.gid.map(grouped.sum_kzt.sum()).fillna(0)
            self.nodes[f"{direction}_tx"] = self.nodes.gid.map(grouped.n_tx.sum()).fillna(0).astype(int)
        self.top = pd.DataFrame({
            "gid": [self.ids[2], self.ids[49], self.ids[0]],
            "rank": [7, 1, 3], "why": ["7 связей — проверить", "2 потока — проверить", "5 переводов — проверить"],
        })
        self.graph = viz.make_graph(self.nodes, self.edges)

    def write_inputs(self, directory):
        directory = Path(directory)
        nodes_path, edges_path, top_path = (directory / "nodes.csv", directory / "edges.parquet",
                                             directory / "top.csv")
        self.nodes.to_csv(nodes_path, index=False)
        self.edges.to_parquet(edges_path, index=False)
        self.top.to_csv(top_path, index=False)
        return nodes_path, edges_path, top_path

    def test_overview_is_top_50_and_one_hop(self):
        selected, _ = viz.select_nodes(self.nodes, self.graph)
        self.assertEqual(selected, set(self.ids[:52]))

    def test_cluster_selection_and_unknown_gid(self):
        data = viz.build_data(self.nodes, self.edges, self.top, cluster="0")
        self.assertEqual(data["overviewIds"], self.ids[:2])
        self.assertEqual(data["initialMode"], "overview")
        self.assertIn(data["initialGid"], data["overviewIds"])
        self.assertEqual(data["initialGid"], self.ids[0])
        for kwargs in ({"gid": "missing"}, {"cluster": "missing"}):
            with self.assertRaises(ValueError):
                viz.build_data(self.nodes, self.edges, self.top, **kwargs)

    def test_cluster_without_top_nodes_uses_its_highest_priority_node(self):
        self.nodes.loc[self.nodes.gid.isin(self.ids[-2:]), "cluster_id"] = "isolated-group"
        self.nodes.loc[self.nodes.gid.eq(self.ids[-1]), "priority_score"] = 0.99
        data = viz.build_data(self.nodes, self.edges, self.top, cluster="isolated-group")
        self.assertEqual(data["initialGid"], self.ids[-1])
        self.assertIn(data["initialGid"], data["overviewIds"])
        self.assertNotIn(data["initialGid"], self.top.gid.tolist())

    def test_incomplete_incoming_and_rule_are_preserved_for_card(self):
        self.nodes["role_rule"] = "transit"
        self.nodes["incoming_incomplete"] = False
        self.nodes.loc[1, ["role_rule", "incoming_incomplete"]] = ["transit_missing_incoming", True]
        self.nodes.loc[1, ["in_kzt", "out_kzt"]] = [100, 300]
        data = viz.build_data(self.nodes, self.edges, self.top)
        missing = data["nodes"][1]
        self.assertTrue(missing["incoming_incomplete"])
        self.assertEqual(missing["role_rule"], "transit_missing_incoming")
        self.assertFalse(data["nodes"][2]["incoming_incomplete"])

    def test_imported_coverage_flags_survive_loading_and_html_serialization(self):
        self.nodes["depth"] = 0
        for field in ("depth_known", "outgoing_coverage_known", "seed_incoming_incomplete", "incoming_incomplete"):
            self.nodes[field] = "false"
        with tempfile.TemporaryDirectory() as directory:
            nodes_path, edges_path, top_path = self.write_inputs(directory)
            nodes, edges = viz.load_data(nodes_path, edges_path)
            top = viz.load_top(top_path, nodes)
            data = viz.build_data(nodes, edges, top)
            serialized = json.loads(re.search(r"const DATA = (.*);", viz.build_html(data)).group(1))
        imported_seed = next(node for node in serialized["nodes"] if node["id"] == self.ids[0])
        self.assertIs(imported_seed["is_seed"], True)
        self.assertEqual(imported_seed["depth"], 0)
        for field in ("depth_known", "outgoing_coverage_known", "seed_incoming_incomplete", "incoming_incomplete"):
            with self.subTest(field=field):
                self.assertIs(imported_seed[field], False)

    def test_funnel_ranked_queue_and_initial_node_come_from_data(self):
        data = viz.build_data(self.nodes, self.edges, self.top)
        self.assertEqual(data["funnel"], {"seeds": 1, "nodes": 55, "top": 3})
        self.assertEqual([row["rank"] for row in data["top"]], [1, 3, 7])
        self.assertEqual([row["gid"] for row in data["top"]], [self.ids[49], self.ids[0], self.ids[2]])
        self.assertEqual(data["top"][0]["why"], self.top.iloc[1].why)
        self.assertEqual(data["initialGid"], self.ids[49])
        self.assertEqual(data["initialMode"], "neighborhood")
        self.assertEqual(data["nodes"][0]["in_kzt"], 60.)
        self.assertEqual(data["nodes"][0]["in_tx"], 6)
        self.assertEqual(data["nodes"][0]["out_deg"], 1)

    def test_gid_overrides_rank_one_including_isolated_node(self):
        data = viz.build_data(self.nodes, self.edges, self.top, gid=self.ids[54])
        self.assertEqual(data["initialGid"], self.ids[54])
        self.assertEqual(data["initialMode"], "neighborhood")
        self.assertEqual(data["nodes"][-1]["in_deg"], 0)
        self.assertEqual(data["nodes"][-1]["out_deg"], 0)

    def test_short_suffixes_are_unique_and_minimal(self):
        ids = ["100000003684369100", "100000002684369100", "100000003887654100"]
        length = viz.unique_suffix_length(ids)
        self.assertEqual(length, 10)
        self.assertEqual(len({gid[-length:] for gid in ids}), len(ids))
        self.assertLess(len({gid[-(length - 1):] for gid in ids}), len(ids))
        data = viz.build_data(self.nodes, self.edges, self.top)
        self.assertEqual(len({gid[-data["shortLen"]:] for gid in self.ids}), len(self.ids))

    def test_html_preserves_precision_directions_and_escapes_text(self):
        data = viz.build_data(self.nodes, self.edges, self.top)
        data["hints"] = {self.ids[49]: {"attention": "<текст> & \u2028", "next_request": "\u2029 запрос"}}
        html = viz.build_html(data)
        self.assertNotIn('<img src=x onerror=', html)
        self.assertNotRegex(html, r'<(?:script|link)\b[^>]*(?:src|href)=["\']https?://')
        matches = re.findall(r"const DATA = (.*);", html)
        self.assertEqual(len(matches), 1)
        serialized = matches[0]
        for character in "<>&\u2028\u2029":
            self.assertNotIn(character, serialized)
        parsed = json.loads(serialized)
        self.assertEqual([node["id"] for node in parsed["nodes"]], self.ids)
        self.assertTrue(all(isinstance(node["id"], str) for node in parsed["nodes"]))
        self.assertEqual(parsed["nodes"][0]["evidence"], self.nodes.evidence.iloc[0])
        self.assertEqual(parsed["hints"], data["hints"])
        self.assertEqual({(edge["from"], edge["to"]) for edge in parsed["edges"]},
                         set(zip(self.edges.src, self.edges.dst)))
        self.assertEqual(sum(edge["n_tx"] for edge in parsed["edges"]), int(self.edges.n_tx.sum()))

    def test_loading_preserves_gid_booleans_and_sums_both_edge_metrics(self):
        self.nodes["incoming_incomplete"] = ["true", "1"] + ["false"] * 53
        with tempfile.TemporaryDirectory() as directory:
            nodes_path, edges_path, top_path = self.write_inputs(directory)
            pd.concat([self.edges, self.edges]).to_parquet(edges_path, index=False)
            nodes, edges = viz.load_data(nodes_path, edges_path)
            top = viz.load_top(top_path, nodes)
        self.assertEqual(nodes.gid.tolist(), self.ids)
        self.assertEqual(nodes.is_seed.tolist(), [True] + [False] * 54)
        self.assertFalse(nodes.truncated_by_depth.any())
        self.assertEqual(nodes.incoming_incomplete.tolist(), [True, True] + [False] * 53)
        self.assertEqual(len(edges), len(self.edges))
        self.assertEqual(edges.sum_kzt.sum(), 2 * self.edges.sum_kzt.sum())
        self.assertEqual(edges.n_tx.sum(), 2 * self.edges.n_tx.sum())
        self.assertEqual(top["rank"].tolist(), [1, 3, 7])
        self.assertEqual(top.gid.tolist(), [self.ids[49], self.ids[0], self.ids[2]])

    def test_offline_hints_are_available_without_api_key(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"OPENAI_API_KEY": ""}), \
                patch.object(llm_layer, "OpenAI") as api:
            hints = viz.collect_hints(self.nodes, self.edges, self.top, directory, offline=True)
            data = viz.build_data(self.nodes, self.edges, self.top, hints=hints)
        api.assert_not_called()
        self.assertEqual(set(data["hints"]), set(self.top.gid))
        self.assertTrue(all(hint["attention"] and hint["next_request"] for hint in hints.values()))

    def test_hints_are_one_batch_and_fail_without_breaking_page(self):
        expected = [{"attention": "проверить потоки", "next_request": "запросить переводы"}] * len(self.top)
        with patch.object(viz, "LLMLayer") as layer:
            layer.return_value.node_hints.return_value = expected
            hints = viz.collect_hints(self.nodes, self.edges, self.top, "unused")
        layer.assert_called_once_with("unused", timeout_seconds=10.0, budget_seconds=10.0, offline=False)
        layer.return_value.node_hints.assert_called_once()
        payloads = layer.return_value.node_hints.call_args.args[0]
        self.assertEqual([payload["node"]["gid"] for payload in payloads], self.top.gid.tolist())
        self.assertEqual(set(hints), set(self.top.gid))
        for result in (RuntimeError("не выводить секрет"), [{"attention": "неполный ответ"}]):
            with patch.object(viz, "LLMLayer") as layer, redirect_stderr(io.StringIO()) as stderr:
                if isinstance(result, Exception):
                    layer.return_value.node_hints.side_effect = result
                else:
                    layer.return_value.node_hints.return_value = result
                self.assertEqual(viz.collect_hints(self.nodes, self.edges, self.top, "unused"), {})
            self.assertNotIn("не выводить секрет", stderr.getvalue())

    def test_cli_reads_each_input_once_and_no_llm_skips_hints(self):
        with tempfile.TemporaryDirectory() as directory:
            nodes_path, edges_path, top_path = self.write_inputs(directory)
            output = Path(directory) / "screen.html"
            argv = ["--nodes", str(nodes_path), "--edges", str(edges_path), "--top", str(top_path),
                    "--output", str(output), "--gid", self.ids[54], "--no-llm"]
            with patch.object(pd, "read_csv", wraps=pd.read_csv) as read_csv, \
                    patch.object(pd, "read_parquet", wraps=pd.read_parquet) as read_parquet, \
                    patch.object(viz, "collect_hints") as collect, \
                    patch.object(viz, "build_html", return_value="<html></html>") as build, \
                    redirect_stdout(io.StringIO()):
                viz.main(argv)
            self.assertEqual([call.args[0] for call in read_csv.call_args_list], [nodes_path, top_path])
            read_parquet.assert_called_once_with(edges_path)
            collect.assert_not_called()
            self.assertEqual(build.call_args.args[0]["initialGid"], self.ids[54])
            self.assertEqual(build.call_args.args[0]["hints"], {})
            self.assertTrue(output.exists())

    def test_cli_unknown_gid_exits_one_without_output_or_llm(self):
        with tempfile.TemporaryDirectory() as directory:
            nodes_path, edges_path, top_path = self.write_inputs(directory)
            output = Path(directory) / "screen.html"
            with patch.object(viz, "collect_hints") as collect, redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as raised:
                viz.main(["--nodes", str(nodes_path), "--edges", str(edges_path), "--top", str(top_path),
                          "--output", str(output), "--gid", "unknown"])
            self.assertEqual(raised.exception.code, 1)
            collect.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
