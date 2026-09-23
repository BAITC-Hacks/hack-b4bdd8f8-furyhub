#!/usr/bin/env python3
"""Граф денег: python run.py --data data --out out.

Роли — проверяемые гипотезы по разделу 2 docs/PLAN.md, не разметка виновности.
PageRank и HITS взвешены суммами. Для betweenness сумма служит длиной пути
согласно заданию (не обратной силой связи). Louvain использует сумму потоков
обоих направлений. Изолированные узлы сохраняются как отдельные кластеры.
Неопределённый pass_through остаётся пустым, как в стартере; обязательные
поля ТЗ заполнены. role_score — эвристическая сила правила, не вероятность.
"""

import argparse
from pathlib import Path
import sys
from time import perf_counter

import networkx as nx
import pandas as pd

from starter.starter import ROLES, basic_features, build_graph, load, sanity_check


THRESHOLDS = {
    "coordinator_in_deg": 3,
    "coordinator_out_deg": 3,
    "coordinator_betweenness_quantile": 0.99,
    "coordinator_seed_payers": 2,
    "distributor_out_deg": 10,
    "consolidator_in_deg": 5,
    "consolidator_out_in_ratio": 0.5,
    "transit_in_deg": 1,
    "transit_out_deg": 1,
    "transit_pass_min": 0.8,
    "transit_pass_max": 1.2,
    "terminal_in_kzt": 50_000,
    "depth_limit": 4,
}

PRIORITY_WEIGHTS = {
    "pagerank": 0.35,
    "in_deg": 0.25,
    "turnover_kzt": 0.20,
    "seed_payers_share": 0.10,
    "betweenness": 0.10,
}
DEPTH_PENALTY = 0.30
RANDOM_SEED = 42
BETWEENNESS_SAMPLES = 500
TOP_N = 25
EVIDENCE_LIMIT = 200
SEED_NOTE = "; seed: входящие суммы занижены методом сбора"


def validate_inputs(edges, nodes, tx):
    """Проверяем уникальность узлов/пар и соответствие сумм и числа переводов."""
    for frame, columns in (
        (nodes, ["gid", "depth", "is_seed"]),
        (edges, ["src", "dst", "sum_kzt", "n_tx", "depth"]),
        (tx, ["src", "dst", "date", "sum_kzt"]),
    ):
        if frame[columns].isna().any().any():
            raise ValueError(f"Пропуски во входных колонках: {columns}")
    if nodes.empty or nodes.gid.duplicated().any():
        raise ValueError("nodes должен содержать непустой список уникальных gid")
    if edges.duplicated(["src", "dst"]).any():
        raise ValueError("edges должен содержать одну строку на направленную пару")
    if (set(edges.src) | set(edges.dst)) - set(nodes.gid):
        raise ValueError("В edges есть узлы, отсутствующие в nodes")
    if not nodes.is_seed.isin([True, False]).all():
        raise ValueError("is_seed должен содержать только булевы значения")
    if not edges.sum_kzt.gt(0).all() or not tx.sum_kzt.gt(0).all():
        raise ValueError("Для взвешенных кратчайших путей суммы должны быть > 0")
    aggregated = tx.groupby(["src", "dst"]).agg(
        sum_kzt=("sum_kzt", "sum"), n_tx=("sum_kzt", "size")
    )
    original = edges.set_index(["src", "dst"])[["sum_kzt", "n_tx"]]
    pd.testing.assert_frame_equal(
        original.sort_index(), aggregated.sort_index(),
        check_dtype=False, check_exact=False, rtol=1e-9, atol=0.01,
    )


def minmax(values):
    """Константная метрика даёт нулевой вклад, без деления на ноль."""
    span = values.max() - values.min()
    if span == 0:
        return pd.Series(0.0, index=values.index)
    return (values - values.min()) / span


def undirected_projection(graph):
    projection = nx.Graph()
    projection.add_nodes_from(graph)
    for src, dst, attrs in graph.edges(data=True):
        previous = projection.get_edge_data(src, dst, {}).get("sum_kzt", 0.0)
        projection.add_edge(src, dst, sum_kzt=previous + attrs["sum_kzt"])
    return projection


def compute_features(edges, nodes):
    graph = build_graph(edges)
    graph.add_nodes_from(nodes.gid)  # Включая seed без единого ребра.
    df = basic_features(graph, nodes)
    df["truncated_by_depth"] = (
        df.depth.eq(THRESHOLDS["depth_limit"]) & df.out_deg.eq(0)
    )
    df["turnover_kzt"] = df.in_kzt + df.out_kzt
    between = nx.betweenness_centrality(
        graph, weight="sum_kzt", k=min(BETWEENNESS_SAMPLES, len(graph)),
        seed=RANDOM_SEED,
    )
    df["betweenness"] = df.gid.map(between)

    if graph.number_of_edges():
        # nx.hits использует атрибут weight, отдельного аргумента weight нет.
        nx.set_edge_attributes(graph, nx.get_edge_attributes(graph, "sum_kzt"), "weight")
        hubs, authorities = nx.hits(
            graph, max_iter=1000, tol=1e-10,
            nstart=dict.fromkeys(graph, 1.0), normalized=True,
        )
    else:
        hubs = authorities = dict.fromkeys(graph, 0.0)
    for name, scores, degree in (
        ("hubs", hubs, df.out_deg), ("authorities", authorities, df.in_deg)
    ):
        # Убираем машинный шум SVD, особенно на узлах с нулевой степенью.
        values = df.gid.map(scores).clip(lower=0.0).where(degree.gt(0), 0.0)
        df[name] = values / values.sum() if values.sum() else values

    seed_gids = set(nodes.loc[nodes.is_seed, "gid"])
    seed_payers = edges.loc[edges.src.isin(seed_gids)].groupby("dst").src.nunique()
    df["seed_payers"] = df.gid.map(seed_payers).fillna(0).astype(int)
    df["seed_payers_share"] = df.seed_payers / df.in_deg.clip(lower=1)

    projection = undirected_projection(graph)
    if projection.number_of_edges():
        communities = nx.community.louvain_communities(
            projection, weight="sum_kzt", seed=RANDOM_SEED
        )
    else:
        communities = [{gid} for gid in graph]
    # Стабильная нумерация: сначала крупные сообщества, при равенстве — min gid.
    communities.sort(key=lambda group: (-len(group), min(group)))
    membership = {gid: cid for cid, group in enumerate(communities) for gid in group}
    df["cluster_id"] = df.gid.map(membership).astype(int)
    return df


def assign_roles(df):
    """Порядок if/elif в точности соответствует таблице PLAN.md."""
    df = df.copy()
    t = THRESHOLDS
    cutoff = df.betweenness.quantile(t["coordinator_betweenness_quantile"])
    max_between = df.betweenness.max()
    roles, scores, reasons = [], [], []
    for row in df.itertuples(index=False):
        if row.in_deg == 0 and row.out_deg == 0:
            role, score, reason = "peripheral", 0.2, "no_edges"
        elif (
            row.in_deg >= t["coordinator_in_deg"]
            and row.out_deg >= t["coordinator_out_deg"]
            and row.betweenness >= cutoff
            and row.seed_payers >= t["coordinator_seed_payers"]
        ):
            # Внутри верхнего процентиля: 0.6 у порога, 0.9 у максимума.
            strength = ((row.betweenness - cutoff) / (max_between - cutoff)
                        if max_between > cutoff else 0.0)
            role, score, reason = "coordinator", 0.6 + 0.3 * strength, "coordinator"
        elif row.out_deg >= t["distributor_out_deg"]:
            role = reason = "distributor"
            score = min(0.95, 0.5 + 0.05 * (row.out_deg // 10))
        elif (
            not row.is_seed
            and row.in_deg >= t["consolidator_in_deg"]
            and row.out_kzt < t["consolidator_out_in_ratio"] * row.in_kzt
        ):
            role = reason = "consolidator"
            # PLAN задаёт зависимость от in_deg без формулы: +0.05 за плательщика.
            score = min(0.95, 0.5 + 0.05 * row.in_deg)
        elif (
            row.in_deg >= t["transit_in_deg"]
            and row.out_deg >= t["transit_out_deg"]
            and t["transit_pass_min"] <= row.pass_through <= t["transit_pass_max"]
        ):
            role = reason = "transit"
            score = max(0.0, 0.9 - abs(1.0 - row.pass_through))
        elif (
            row.out_deg == 0 and row.depth < t["depth_limit"]
            and row.in_kzt >= t["terminal_in_kzt"]
        ):
            role, score, reason = "terminal", 0.7, "terminal"
        elif row.out_deg == 0 and row.depth == t["depth_limit"]:
            role, score, reason = "peripheral", 0.3, "truncated"
        else:
            role, score, reason = "peripheral", 0.4, "other"
        roles.append(role)
        scores.append(score)
        reasons.append(reason)
    df["role"], df["role_score"], df["role_rule"] = roles, scores, reasons
    return df


def add_priority(df):
    df = df.copy()
    df["priority_score_raw"] = sum(
        weight * minmax(df[column]) for column, weight in PRIORITY_WEIGHTS.items()
    ) - DEPTH_PENALTY * df.truncated_by_depth.astype(int)
    # ТЗ требует 0..1. Сохраняем сырой результат для проверки штрафа.
    df["priority_score"] = df.priority_score_raw.clip(0.0, 1.0)
    return df


def evidence(row):
    incoming = f"{row.in_kzt:,.0f}".replace(",", " ")
    outgoing = f"{row.out_kzt:,.0f}".replace(",", " ")
    if row.role_rule == "no_edges":
        text = "0 входящих и 0 исходящих рёбер; недостаточно данных для гипотезы о роли"
    elif row.role == "coordinator":
        text = (f"Вход/выход: {row.in_deg}/{row.out_deg} связей; seed-плательщиков "
                f"{row.seed_payers}; betweenness={row.betweenness:.5f}; признаки координации")
    elif row.role == "distributor":
        text = (f"Отправляет {outgoing} KZT на {row.out_deg} получателей, "
                f"{row.out_tx} переводов; признаки распределения")
    elif row.role == "consolidator":
        text = (f"Получает {incoming} KZT от {row.in_deg} плательщиков, "
                f"передаёт {row.pass_through:.1%}; признаки консолидации")
    elif row.role == "transit":
        text = (f"Вход {incoming}, выход {outgoing} KZT; передаёт {row.pass_through:.1%}; "
                f"{row.in_deg}/{row.out_deg} связей; признаки транзита")
    elif row.role == "terminal":
        text = (f"Вход {incoming} KZT от {row.in_deg} плательщиков, выход 0; "
                f"глубина {row.depth}; признаки конечного получателя")
    elif row.role_rule == "truncated":
        text = (f"Глубина {row.depth}, выход 0, вход {incoming} KZT; "
                "обход обрезан, данных для гипотезы о конечном получателе недостаточно")
    else:
        text = (f"Вход/выход: {row.in_deg}/{row.out_deg} связей, "
                f"{incoming}/{outgoing} KZT; признаков специальной роли недостаточно")
    note = SEED_NOTE if row.is_seed else ""
    # Оговорка о seed сохраняется целиком даже при необычно длинных числах.
    budget = EVIDENCE_LIMIT - len(note)
    if len(text) > budget:
        text = text[:budget - 1] + "…"
    return text + note


def rank_nodes(df):
    return df.sort_values(["priority_score", "gid"], ascending=[False, True])


def cluster_summary(df, edges):
    membership = df.set_index("gid").cluster_id
    src_cluster = edges.src.map(membership)
    dst_cluster = edges.dst.map(membership)
    internal = edges.loc[src_cluster.eq(dst_cluster)].copy()
    internal["cluster_id"] = src_cluster.loc[internal.index]
    sums = internal.groupby("cluster_id").sum_kzt.sum()
    rows = []
    for cid, group in df.groupby("cluster_id", sort=True):
        n_nodes, n_seed = len(group), int(group.is_seed.sum())
        amount = float(sums.get(cid, 0.0))
        rows.append({
            "cluster_id": cid, "n_nodes": n_nodes, "n_seed": n_seed,
            "sum_kzt_internal": amount,
            "top_gids": ";".join(rank_nodes(group).head(5).gid.astype(str)),
            "hypothesis": (f"Гипотеза для проверки: группа из {n_nodes} узлов, "
                           f"seed — {n_seed}; внутренние переводы {amount:,.0f} KZT; "
                           "возможная общность денежных потоков требует проверки."),
        })
    return pd.DataFrame(rows)


def write_outputs(df, edges, out_dir):
    required = ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"]
    if df[required].isna().any().any() or df.gid.duplicated().any():
        raise ValueError("Пустые обязательные поля или повторные gid в результате")
    if not df.evidence.str.len().between(1, EVIDENCE_LIMIT).all():
        raise ValueError("evidence должен содержать от 1 до 200 символов")
    if not df.role.isin(ROLES).all():
        raise ValueError("Неизвестная роль")
    for column in ["role_score", "priority_score"]:
        if not df[column].between(0, 1).all():
            raise ValueError(f"{column} выходит за диапазон 0..1")
    top = rank_nodes(df).head(TOP_N)[["gid", "role", "priority_score", "evidence"]]
    top = top.rename(columns={"evidence": "why"})
    top.insert(0, "rank", range(1, len(top) + 1))
    clusters = cluster_summary(df, edges)
    out_dir.mkdir(parents=True, exist_ok=True)
    df[required + [c for c in df if c not in required]].to_csv(
        out_dir / "nodes_roles.csv", index=False, encoding="utf-8"
    )
    clusters.to_csv(out_dir / "clusters.csv", index=False, encoding="utf-8")
    top.to_csv(out_dir / "top_nodes.csv", index=False, encoding="utf-8")
    return len(clusters)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()
    started = perf_counter()
    edges, nodes, tx = load(args.data)
    validate_inputs(edges, nodes, tx)
    sanity_check(edges, nodes, tx)
    print("Расчёт метрик и Louvain...", flush=True)
    df = add_priority(assign_roles(compute_features(edges, nodes)))
    df["evidence"] = [evidence(row) for row in df.itertuples(index=False)]
    n_clusters = write_outputs(df, edges, args.out)
    print(f"Записаны 3 CSV в {args.out.resolve()}")
    print(f"Узлов: {len(df)}; кластеров: {n_clusters}; топ: {min(TOP_N, len(df))}")
    print(f"Время полного прогона: {perf_counter() - started:.2f} с")
    print("Сводка ролей:")
    for role in ROLES:
        print(f"  {role}: {int(df.role.eq(role).sum())}")


if __name__ == "__main__":
    main()
