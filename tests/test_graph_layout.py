"""Контракт раскладки обзора: сохранение узлов, разделение и детерминизм."""

import itertools
import math
import unittest

import networkx as nx
import pandas as pd

from graph_layout import MIN_NODE_DISTANCE, overview_positions


class OverviewLayoutTests(unittest.TestCase):
    def setUp(self):
        graph = nx.disjoint_union(nx.star_graph(11), nx.path_graph(13))
        graph.add_edge(0, 12)
        graph.add_nodes_from(range(25, 29))
        self.graph = nx.relabel_nodes(graph, lambda gid: f"node{gid}")
        self.nodes = pd.DataFrame({
            "gid": [f"node{gid}" for gid in range(29)],
            "cluster_id": ["star"] * 12 + ["path"] * 13 + [f"isolated{i}" for i in range(4)],
        })
        self.selected = set(self.nodes.gid[:-1])

    def test_all_selected_nodes_are_separated_with_disjoint_communities(self):
        saved = self.nodes.copy(deep=True)
        positions = overview_positions(self.nodes, self.graph, self.selected)
        self.assertEqual(set(positions), self.selected)
        self.assertTrue(all(math.isfinite(value) for point in positions.values() for value in point))
        for left, right in itertools.combinations(positions.values(), 2):
            self.assertGreaterEqual(math.dist(left, right), MIN_NODE_DISTANCE - 1e-8)
        boxes = []
        for _, group in self.nodes.loc[self.nodes.gid.isin(self.selected)].groupby("cluster_id"):
            points = [positions[gid] for gid in group.gid]
            boxes.append((
                [min(point[axis] for point in points) - MIN_NODE_DISTANCE / 2 for axis in range(2)],
                [max(point[axis] for point in points) + MIN_NODE_DISTANCE / 2 for axis in range(2)],
            ))
        for (lo, hi), (other_lo, other_hi) in itertools.combinations(boxes, 2):
            self.assertTrue(any(hi[axis] <= other_lo[axis] or other_hi[axis] <= lo[axis]
                                for axis in range(2)))
        pd.testing.assert_frame_equal(self.nodes, saved)

    def test_row_edge_and_selection_order_do_not_change_positions(self):
        expected = overview_positions(self.nodes, self.graph, self.selected)
        reordered = nx.Graph()
        reordered.add_nodes_from(reversed(list(self.graph)))
        reordered.add_edges_from((dst, src) for src, dst in reversed(list(self.graph.edges)))
        actual = overview_positions(self.nodes.iloc[::-1], reordered, reversed(sorted(self.selected)))
        self.assertEqual(actual, expected)

    def test_empty_and_singleton_selections(self):
        self.assertEqual(overview_positions(self.nodes, self.graph, []), {})
        gid = self.nodes.gid.iloc[-1]
        positions = overview_positions(self.nodes, self.graph, [gid])
        self.assertEqual(set(positions), {gid})
        self.assertTrue(all(math.isfinite(value) for value in positions[gid]))


if __name__ == "__main__":
    unittest.main()
