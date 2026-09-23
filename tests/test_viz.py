"""Проверка направлений, границ выборки и сериализации длинных gid."""

import json
from pathlib import Path
import re
import tempfile
import unittest

import pandas as pd

import viz


class VisualizationTests(unittest.TestCase):
    def setUp(self):
        self.ids = [str(100000000000000001 + i) for i in range(55)]
        self.nodes = pd.DataFrame({
            "gid": self.ids, "role": ["transit"]*55,
            "priority_score": [1-i/100 for i in range(55)],
            "evidence": ['</script><img src=x onerror="alert(1)">']*55,
            "is_seed": [True]+[False]*54,
            "cluster_id": ["0"]*2+["1"]*53,
        })
        # Top-50 includes index 49; 50 and 51 are incoming/outgoing neighbors.
        # 52 is two steps away, 53 three steps away, 54 isolated.
        self.edges = pd.DataFrame({
            "src": [self.ids[50], self.ids[49], self.ids[51], self.ids[52], self.ids[0], self.ids[1]],
            "dst": [self.ids[49], self.ids[51], self.ids[52], self.ids[53], self.ids[1], self.ids[0]],
            "sum_kzt": [10., 20., 30., 40., 50., 60.],
        })
        self.graph = viz.make_graph(self.nodes, self.edges)

    def test_default_is_exactly_top_50_and_one_hop(self):
        selected, _ = viz.select_nodes(self.nodes, self.graph)
        self.assertEqual(selected, set(self.ids[:52]))

    def test_gid_includes_incoming_and_outgoing_at_most_two_hops(self):
        selected, _ = viz.select_nodes(self.nodes, self.graph, gid=self.ids[49])
        self.assertEqual(selected, set(self.ids[49:53]))
        selected, _ = viz.select_nodes(self.nodes, self.graph, gid=self.ids[54])
        self.assertEqual(selected, {self.ids[54]})

    def test_cluster_and_missing_ids(self):
        selected, _ = viz.select_nodes(self.nodes, self.graph, cluster="0")
        self.assertEqual(selected, set(self.ids[:2]))
        for kwargs in ({"gid": "missing"}, {"cluster": "missing"}):
            with self.assertRaises(ValueError):
                viz.select_nodes(self.nodes, self.graph, **kwargs)

    def test_html_preserves_precision_directions_and_escapes_evidence(self):
        selected, mode = viz.select_nodes(self.nodes, self.graph, cluster="0")
        html = viz.build_html(self.nodes, self.edges, self.graph, selected, mode)
        self.assertNotIn('<img src=x onerror=', html)
        self.assertNotRegex(html, r'<(?:script|link)\b[^>]*(?:src|href)=["\']https?://')
        records = json.loads(re.search(r"const allNodes = (.*);", html)[1])
        edges = json.loads(re.search(r"const allEdges = (.*);", html)[1])
        self.assertEqual([n["id"] for n in records], self.ids)
        self.assertEqual(records[0]["evidence"], self.nodes.evidence.iloc[0])
        self.assertGreater(records[0]["borderWidth"], records[1]["borderWidth"])
        self.assertGreater(records[0]["size"], records[1]["size"])
        self.assertEqual({(e["from"], e["to"]) for e in edges}, set(zip(self.edges.src, self.edges.dst)))
        self.assertEqual(json.loads(re.search(r"const initialIds = (.*);", html)[1]), self.ids[:2])

    def test_loading_strings_booleans_and_duplicate_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            nodes_path, edges_path = Path(directory)/"nodes.csv", Path(directory)/"edges.parquet"
            self.nodes.to_csv(nodes_path, index=False)
            pd.concat([self.edges, self.edges]).to_parquet(edges_path, index=False)
            nodes, edges = viz.load_data(nodes_path, edges_path)
            self.assertEqual(nodes.gid.tolist(), self.ids)
            self.assertEqual(nodes.is_seed.tolist(), [True]+[False]*54)
            self.assertEqual(edges.sum_kzt.sum(), 2*self.edges.sum_kzt.sum())


if __name__ == "__main__":
    unittest.main()
