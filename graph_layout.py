"""Детерминированный обзор: локальная структура сообществ без наложения узлов."""

import math

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree


MIN_NODE_DISTANCE = 76.0
CLUSTER_MARGIN = 46.0
TARGET_ASPECT_RATIO = 1.15
LAYOUT_SEED = 42


def _separate(points):
    """Раздвигаем соседние центры, затем гарантируем минимальное расстояние."""
    for _ in range(100):
        pairs = cKDTree(points).query_pairs(MIN_NODE_DISTANCE, output_type="ndarray")
        if not len(pairs):
            break
        delta = points[pairs[:, 0]] - points[pairs[:, 1]]
        distance = np.linalg.norm(delta, axis=1)
        coincident = distance < 1e-9
        if coincident.any():
            angles = (pairs[coincident, 0] + pairs[coincident, 1] + 1) * 2.399963229728653
            delta[coincident] = np.column_stack((np.cos(angles), np.sin(angles)))
            distance[coincident] = 1.0
        moves = delta * ((MIN_NODE_DISTANCE - distance + 0.1) / (2 * distance))[:, None]
        displacement = np.zeros_like(points)
        np.add.at(displacement, pairs[:, 0], moves)
        np.add.at(displacement, pairs[:, 1], -moves)
        neighbors = np.bincount(pairs.ravel(), minlength=len(points))
        points += displacement / np.sqrt(np.maximum(neighbors, 1))[:, None]
    nearest = cKDTree(points).query(points, k=2)[0][:, 1].min()
    if nearest < MIN_NODE_DISTANCE:
        points *= MIN_NODE_DISTANCE / max(nearest, 1e-9)
    return points


def _local_positions(graph, gids):
    if len(gids) == 1:
        return np.zeros((1, 2))
    # Порядок Graph и начальная случайная выборка не зависят от порядка CSV.
    local = nx.Graph()
    local.add_nodes_from(gids)
    local.add_edges_from(sorted({tuple(sorted((src, dst)))
                                 for src, dst in graph.subgraph(gids).edges()
                                 if src != dst}))
    if not local.number_of_edges():
        # У полностью разрозненной группы пружинная модель не добавляет смысла.
        columns = math.ceil(math.sqrt(len(gids)))
        positions = np.array([
            ((i % columns) * MIN_NODE_DISTANCE, (i // columns) * MIN_NODE_DISTANCE)
            for i in range(len(gids))
        ], dtype=float)
    else:
        layout = nx.spring_layout(
            local, seed=LAYOUT_SEED, weight=None, iterations=65,
            scale=math.sqrt(len(gids)) * 40,
        )
        positions = _separate(np.array([layout[gid] for gid in gids], dtype=float))
    return positions - positions.min(axis=0)


def _pack_shelves(blocks, width):
    """Мелкие сообщества занимают остатки полок, без больших пустых ячеек."""
    shelves, origins = [], {}
    height = 0.0
    for cid, _, block_width, block_height in blocks:
        candidates = [shelf for shelf in shelves
                      if shelf["height"] >= block_height and shelf["used"] + block_width <= width]
        if candidates:
            shelf = min(candidates, key=lambda item: width - item["used"] - block_width)
        else:
            shelf = {"y": height, "height": block_height, "used": 0.0}
            shelves.append(shelf)
            height += block_height
        origins[cid] = (shelf["used"], shelf["y"])
        shelf["used"] += block_width
    return origins, max(shelf["used"] for shelf in shelves), height


def overview_positions(nodes, graph, selected):
    """Вернуть dict[gid, (x, y)] только для selected; входные данные не меняются.

    gid и cluster_id — строки. Внутри сообщества применяется невзвешенный
    spring_layout, между сообществами — упаковка непересекающихся областей.
    Расстояние между любыми центрами не меньше MIN_NODE_DISTANCE; масштаб
    координат сохраняется, чтобы размеры точек можно было выбирать независимо.
    """
    gids = sorted({str(gid) for gid in selected})
    if not gids:
        return {}
    membership = dict(zip(nodes.gid.astype(str), nodes.cluster_id.astype(str)))
    if any(gid not in membership for gid in gids):
        raise ValueError("В обзоре есть gid, отсутствующий в таблице узлов")
    communities = {}
    for gid in gids:
        communities.setdefault(membership[gid], []).append(gid)
    local_positions, blocks = {}, []
    for cid, group in sorted(communities.items()):
        points = _local_positions(graph, group)
        local_positions[cid] = points
        width, height = points.max(axis=0) + 2 * CLUSTER_MARGIN
        blocks.append((cid, group, float(width), float(height)))
    blocks.sort(key=lambda block: (-block[3], -block[2], block[0]))
    area = sum(width * height for _, _, width, height in blocks)
    widest = max(width for _, _, width, _ in blocks)
    ideal_width = math.sqrt(area * TARGET_ASPECT_RATIO)
    layouts = []
    for factor in np.linspace(0.8, 1.8, 21):
        packed, width, height = _pack_shelves(blocks, max(widest, ideal_width * factor))
        # Учитываем одновременно пустое пространство и форму всей сцены.
        score = (width * height / area) * (1 + abs(math.log(width / height / TARGET_ASPECT_RATIO)))
        layouts.append((score, packed, width, height))
    _, origins, width, height = min(layouts, key=lambda item: item[0])
    positions = {}
    for cid, group, _, _ in blocks:
        offset = np.array(origins[cid]) + CLUSTER_MARGIN - np.array([width, height]) / 2
        for gid, point in zip(group, local_positions[cid] + offset):
            positions[gid] = (float(point[0]), float(point[1]))
    return positions
