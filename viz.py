#!/usr/bin/env python3
"""Экран аналитика: python viz.py [--gid ID | --cluster ID] [--no-llm]."""

import argparse
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import networkx as nx
import pandas as pd
from pyvis.network import Network

from graph_layout import overview_positions
from llm_layer import LLMLayer, node_payload


ROLES = {
    "coordinator": ("Координатор", "#ad5145"),
    "distributor": ("Распределитель", "#b16a33"),
    "consolidator": ("Консолидатор", "#876586"),
    "transit": ("Транзит", "#567d98"),
    "terminal": ("Конечный получатель", "#557b65"),
    "peripheral": ("Периферия", "#858176"),
}
GRAPH_COLORS = {
    "coordinator": "#f2847b", "distributor": "#efb26d",
    "consolidator": "#c2a0e8", "transit": "#80b8ec",
    "terminal": "#77c6a3", "peripheral": "#9aa9bc",
}
ROLE_MEANINGS = {
    "coordinator": "структура связей даёт основания проверить координацию потоков",
    "distributor": "наблюдаются переводы многим получателям",
    "consolidator": "видимый выход меньше половины входа от нескольких плательщиков",
    "transit": "вход и выход за период сопоставимы; последовательность переводов требует проверки",
    "terminal": "в доступной части сети исходящие переводы не наблюдаются",
    "peripheral": "данных недостаточно для более определённой гипотезы о роли",
}


def _booleans(values, name):
    text = values.astype(str).str.lower().str.strip()
    if not text.isin(["true", "false", "1", "0"]).all():
        raise ValueError(f"{name} должен содержать true/false или 1/0")
    return text.isin(["true", "1"])


def load_data(nodes_path, edges_path):
    # gid нельзя пропускать через float или JavaScript Number: он длиннее 53 бит.
    nodes = pd.read_csv(nodes_path, dtype={"gid": str, "cluster_id": str},
                        keep_default_na=False)
    edges = pd.read_parquet(edges_path)
    required = {"gid", "role", "priority_score", "evidence", "is_seed", "cluster_id"}
    if missing := required - set(nodes.columns):
        raise ValueError(f"В nodes_roles.csv отсутствуют колонки: {', '.join(sorted(missing))}")
    if missing := {"src", "dst", "sum_kzt", "n_tx"} - set(edges.columns):
        raise ValueError(f"В edges.parquet отсутствуют колонки: {', '.join(sorted(missing))}")
    for column in ["gid", "cluster_id"]:
        nodes[column] = nodes[column].str.strip()
        if nodes[column].eq("").any():
            raise ValueError(f"Пустой {column} в nodes_roles.csv")
    if nodes.empty or nodes.gid.duplicated().any():
        raise ValueError("Нужен непустой список уникальных gid")
    if not nodes.role.isin(ROLES).all():
        raise ValueError("Неизвестная роль в nodes_roles.csv")
    for column in ["priority_score", "role_score"]:
        if column in nodes:
            nodes[column] = pd.to_numeric(nodes[column], errors="raise")
            if not nodes[column].between(0, 1).all():
                raise ValueError(f"{column} должен быть в диапазоне 0..1")
    nodes["is_seed"] = _booleans(nodes.is_seed, "is_seed")
    for column in ("truncated_by_depth", "incoming_incomplete", "depth_known",
                   "outgoing_coverage_known", "seed_incoming_incomplete"):
        if column in nodes:
            nodes[column] = _booleans(nodes[column], column)
    for column in ["depth", "in_deg", "out_deg", "in_kzt", "out_kzt", "in_tx", "out_tx"]:
        if column in nodes:
            nodes[column] = pd.to_numeric(nodes[column], errors="raise")
            if not nodes[column].map(lambda value: math.isfinite(value) and value >= 0).all():
                raise ValueError(f"{column} должен содержать конечные неотрицательные числа")
    for column in ["src", "dst"]:
        if edges[column].isna().any() or pd.api.types.is_float_dtype(edges[column]):
            raise ValueError(f"{column}: нужны целые числа или строки без пропусков")
        edges[column] = edges[column].astype(str).str.strip()
    if (set(edges.src) | set(edges.dst)) - set(nodes.gid):
        raise ValueError("В edges есть gid, отсутствующий в nodes_roles.csv")
    for column in ["sum_kzt", "n_tx"]:
        edges[column] = pd.to_numeric(edges[column], errors="raise")
        if not edges[column].map(lambda value: math.isfinite(value) and value >= 0).all():
            raise ValueError(f"{column} должен содержать конечные неотрицательные числа")
    if not edges.n_tx.mod(1).eq(0).all():
        raise ValueError("n_tx должен содержать целое число переводов")
    # Складываем оба показателя; встречные потоки остаются разными рёбрами.
    edges = edges.groupby(["src", "dst"], as_index=False, sort=True)[["sum_kzt", "n_tx"]].sum()
    return nodes, edges


def load_top(top_path, nodes):
    top = pd.read_csv(top_path, dtype={"gid": str}, keep_default_na=False)
    if missing := {"rank", "gid", "why"} - set(top.columns):
        raise ValueError(f"В top_nodes.csv отсутствуют колонки: {', '.join(sorted(missing))}")
    top["gid"] = top.gid.str.strip()
    if top.gid.eq("").any() or top.gid.duplicated().any():
        raise ValueError("В top_nodes.csv нужны непустые уникальные gid")
    if set(top.gid) - set(nodes.gid):
        raise ValueError("В top_nodes.csv есть gid, отсутствующий в nodes_roles.csv")
    top["rank"] = pd.to_numeric(top["rank"], errors="raise")
    if (not top["rank"].map(lambda value: math.isfinite(value) and value > 0).all()
            or not top["rank"].mod(1).eq(0).all() or top["rank"].duplicated().any()):
        raise ValueError("rank должен содержать уникальные положительные целые числа")
    top["rank"] = top["rank"].astype(int)
    return top.sort_values(["rank", "gid"], kind="stable").reset_index(drop=True)


def make_graph(nodes, edges):
    graph = nx.Graph()
    graph.add_nodes_from(nodes.gid)
    graph.add_edges_from(zip(edges.src, edges.dst))
    return graph


def select_nodes(nodes, graph, gid=None, cluster=None):
    """Выборка для вторичного обзора; направления сохраняются в данных рёбер."""
    if gid is not None:
        if gid not in graph:
            raise ValueError(f"gid {gid} не найден")
        return set(nx.single_source_shortest_path_length(graph, gid, cutoff=1)), f"Окружение {gid} · 1 шаг"
    if cluster is not None:
        selected = set(nodes.loc[nodes.cluster_id.eq(str(cluster)), "gid"])
        if not selected:
            raise ValueError(f"Группа {cluster} не найдена")
        return selected, f"Группа {cluster}"
    top = nodes.sort_values(["priority_score", "gid"], ascending=[False, True]).head(50)
    selected = set(top.gid)
    for node in top.gid:
        selected.update(graph.neighbors(node))
    return selected, f"Топ-{len(top)} + прямые соседи"


def unique_suffix_length(gids):
    gids = list(gids)
    for length in range(1, max(map(len, gids), default=1) + 1):
        if len({gid[-length:] for gid in gids}) == len(gids):
            return length
    raise ValueError("Невозможно различить повторяющиеся gid")


def collect_hints(nodes, edges, top, out_dir, *, offline=False):
    """Все карточки топа одним обращением; сбой подсказок не ломает экран."""
    if top.empty:
        return {}
    try:
        payloads = [node_payload(nodes, edges, gid) for gid in top.gid]
        layer = LLMLayer(out_dir, timeout_seconds=10.0, budget_seconds=10.0, offline=offline)
        hints = layer.node_hints(payloads)
        if len(hints) != len(payloads) or not all(
            isinstance(hint, dict)
            and all(isinstance(hint.get(key), str) and hint[key].strip()
                    for key in ("attention", "next_request"))
            for hint in hints
        ):
            raise ValueError("Некорректный пакет подсказок")
        return {gid: {key: hint[key] for key in ("attention", "next_request")}
                for gid, hint in zip(top.gid, hints)}
    except Exception:
        # Текст исключения может содержать секреты или данные запроса.
        print("Подсказки аналитика недоступны; экран собран без них.", file=sys.stderr)
        return {}


def build_data(nodes, edges, top, gid=None, cluster=None, hints=None):
    graph = make_graph(nodes, edges)
    if gid is not None and gid not in graph:
        raise ValueError(f"gid {gid} не найден")
    top = top.sort_values(["rank", "gid"], kind="stable")
    selected, overview_label = select_nodes(nodes, graph, cluster=cluster)
    candidates = nodes.loc[nodes.gid.isin(selected)].sort_values(
        ["priority_score", "gid"], ascending=[False, True]
    )
    initial_gid = (str(candidates.iloc[0].gid) if cluster is not None else
                   gid or (str(top.iloc[0].gid) if len(top) else str(nodes.iloc[0].gid)))
    # Раскладка нужна только обзору. Окружение расставляет браузер по потокам.
    positions = overview_positions(nodes, graph, selected)
    max_score = max(float(nodes.priority_score.max()), 1e-12)
    incoming = edges.groupby("dst").agg(in_deg=("src", "nunique"),
                                        in_kzt=("sum_kzt", "sum"), in_tx=("n_tx", "sum"))
    outgoing = edges.groupby("src").agg(out_deg=("dst", "nunique"),
                                        out_kzt=("sum_kzt", "sum"), out_tx=("n_tx", "sum"))
    metrics = incoming.join(outgoing, how="outer").fillna(0).to_dict("index")
    records = []
    for row in nodes.to_dict("records"):
        node_id = str(row["gid"])
        label, color = ROLES[row["role"]]
        x, y = positions.get(node_id, (0, 0))
        score = float(row["priority_score"])
        item = {
            "id": node_id, "role": row["role"], "role_label": label,
            "role_meaning": ROLE_MEANINGS[row["role"]], "color": color,
            "graph_color": GRAPH_COLORS[row["role"]],
            "priority_score": score, "role_score": float(row.get("role_score", 0)),
            "evidence": str(row["evidence"]), "is_seed": bool(row["is_seed"]),
            "cluster_id": str(row["cluster_id"]), "depth": int(row.get("depth", 0)),
            "truncated_by_depth": bool(row.get("truncated_by_depth", False)),
            "incoming_incomplete": bool(row.get("incoming_incomplete", False)),
            "depth_known": bool(row.get("depth_known", True)),
            "outgoing_coverage_known": bool(row.get("outgoing_coverage_known", True)),
            "seed_incoming_incomplete": bool(row.get("seed_incoming_incomplete", row["is_seed"])),
            "role_rule": str(row.get("role_rule", "")),
            "x": float(x), "y": float(y), "size": 8 + 28 * score / max_score,
        }
        for field in ("in_deg", "out_deg", "in_kzt", "out_kzt", "in_tx", "out_tx"):
            value = row.get(field, metrics.get(node_id, {}).get(field, 0))
            item[field] = float(value) if field.endswith("kzt") else int(value)
        records.append(item)
    edge_records = [{"id": f"e{index}", "from": str(row.src), "to": str(row.dst),
                     "sum_kzt": float(row.sum_kzt), "n_tx": int(row.n_tx)}
                    for index, row in enumerate(edges.itertuples(index=False))]
    return {
        "nodes": records, "edges": edge_records,
        "top": [{"gid": str(row.gid), "rank": int(row.rank), "why": str(row.why)}
                for row in top.itertuples(index=False)],
        "funnel": {"seeds": int(nodes.is_seed.sum()), "nodes": len(nodes), "top": len(top)},
        "shortLen": unique_suffix_length(nodes.gid), "initialGid": initial_gid,
        "initialMode": "overview" if cluster is not None else "neighborhood",
        "overviewIds": sorted(selected), "overviewLabel": overview_label,
        "hints": hints or {},
    }


PAGE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FuryHub · Граф денег</title>
<style>__VIS_CSS__</style>
<style>
:root{color-scheme:light;--bg:#f8f6f0;--panel:#fffdf8;--sidebar:#efede6;--line:#dedbd2;--text:#302f2b;--muted:#6f685e;--accent:#ab5335;--accent-soft:#f3e6dc;--serif:Georgia,"Times New Roman",serif}
*{box-sizing:border-box}[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;overflow:hidden}
button,input{font:inherit}button{cursor:pointer;color:inherit;transition:background .15s,border-color .15s}
button:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
button{border:1px solid var(--line);border-radius:9px;background:var(--panel);padding:9px 13px;white-space:nowrap}button:hover{background:#eae5da;border-color:#c7bfb0}
header{height:96px;padding:18px 26px;display:flex;align-items:center;gap:30px;border-bottom:1px solid var(--line);background:var(--bg)}
.brand{display:flex;align-items:center;gap:12px;flex:0 0 272px}.brand-name{font:29px/1.1 var(--serif);letter-spacing:-1px}.brand-name b{font-weight:400}.brand small{display:block;font:11px/1.5 system-ui,sans-serif;color:var(--muted);letter-spacing:.4px;margin-top:5px}
.funnel{display:flex;align-items:center;justify-content:center;gap:27px;flex:1}.funnel-step{min-width:106px}.funnel strong{display:block;font:30px/1.12 var(--serif);font-variant-numeric:tabular-nums;letter-spacing:-1px}.funnel span{font-size:11px;color:var(--muted);white-space:nowrap}.funnel .accent strong{color:var(--accent)}.funnel-arrow{color:#b5aa99;font-size:20px;font-weight:300}
#overview{margin-left:auto;color:var(--accent);border-color:#d8bbaa;background:transparent;font-size:12px}#overview.active{background:var(--accent-soft)}
#app{height:calc(100dvh - 96px);display:grid;grid-template-columns:304px minmax(0,1fr) 350px;min-height:0}
.queue-panel{display:flex;flex-direction:column;overflow:hidden;min-height:0;border-right:1px solid var(--line);background:var(--sidebar);scrollbar-width:thin;scrollbar-color:#c9c3b7 transparent}
.queue-head{padding:24px 18px 16px;flex-shrink:0;z-index:2;border-bottom:1px solid var(--line)}
.search-label{display:block;font-size:12px;margin-bottom:9px}#search-form{display:flex;gap:6px}#gid{width:100%;min-width:0;background:#f9f7f2;border:1px solid #d4cec2;color:var(--text);border-radius:8px;padding:10px;font-size:12px}#gid::placeholder{color:#6f685e}#search-form button{padding:8px 10px;background:var(--accent);color:#fffdf8;border-color:var(--accent);font-size:12px}#search-form button:hover{background:#8f432a}
#status{font-size:11px;line-height:1.5;color:var(--muted);min-height:33px;margin:8px 0 20px;overflow-wrap:anywhere}#status.error{color:#a53d2b}
h2.section-label{font:23px/1.2 var(--serif);color:var(--text);margin:0;letter-spacing:-.5px}.queue-subtitle{color:var(--muted);font-size:11px;margin-top:6px}
#queue{padding:8px 10px 18px;overflow:auto;min-height:0;scrollbar-width:thin}.queue-item{border:1px solid transparent;border-radius:9px;display:block;width:100%;padding:13px 10px;text-align:left;background:transparent;white-space:normal;margin:3px 0}.queue-item:hover{background:#e7e1d6}.queue-item.active{background:#fffaf2;border-color:#d7b7a4;box-shadow:inset 3px 0 var(--accent)}
.queue-line{display:flex;align-items:center;gap:6px;min-width:0}.rank{color:#6f685e;width:19px;font-size:12px;flex-shrink:0;font-variant-numeric:tabular-nums}.queue-item.active .rank,.queue-item.active .queue-role{color:var(--accent)}.role-dot{display:inline-block;width:7px;height:7px;border-radius:50%;flex:0 0 7px}.queue-role{font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px}.queue-gid{margin-left:auto;color:#6f685e;font:11px ui-monospace,SFMono-Regular,monospace;flex-shrink:0}.queue-why{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;color:#777166;font-size:11px;line-height:1.6;padding-left:25px;margin-top:6px}
.center{min-width:0;min-height:0;overflow:auto;display:flex;flex-direction:column;position:relative;background:#151c25}
.canvas-header{background:var(--bg);display:flex;align-items:center;gap:12px;padding:25px 26px 18px;flex-shrink:0}.canvas-heading{min-width:0;flex:1}.eyebrow{font-size:10px;letter-spacing:1.5px;color:var(--accent);font-weight:600;margin-bottom:6px}.canvas-title{font:28px/1.2 var(--serif);letter-spacing:-.6px;margin:0 0 7px}#mode{font-size:11px;color:var(--muted);overflow:hidden;white-space:nowrap;text-overflow:ellipsis}#back{white-space:nowrap;flex-shrink:0;padding:7px 11px;font-size:12px;background:transparent}#back:disabled{opacity:.35;cursor:default}
.canvas-wrap{flex:1;min-height:360px;position:relative;background-color:#151c25;background-image:radial-gradient(#2a3544 .65px,transparent .65px);background-size:24px 24px}#network{width:100%;height:100%;position:absolute;inset:0}#directions{display:flex;justify-content:space-between;position:absolute;left:26px;right:26px;top:10px;pointer-events:none;font-size:11px;color:#9ec4ed}#directions span{padding:4px 8px;background:#151c25;border-radius:5px}#directions span:last-child{color:#edb986}.canvas-hint{color:#afbdce;font-size:11px;text-align:center;margin:0;padding:12px 16px}#isolated{position:absolute;bottom:22%;left:20px;right:20px;text-align:center;color:#b8c6d8;pointer-events:none;font-size:13px}
#hidden-peers{margin:0 20px 12px;padding:10px 13px;background:#1c2735;border:1px solid #334254;border-radius:8px;font-size:11px;line-height:1.8;color:#bdc9d8;flex-shrink:0}#hidden-peers p{margin:0}
.center-footer{padding:14px 22px 18px;border-top:1px solid #303d4e;flex-shrink:0}#legend{display:flex;flex-wrap:wrap;gap:8px 13px;font-size:11px;color:#c3cedd}#legend span{display:inline-flex;align-items:center;gap:5px}.seed-dot{width:10px;height:10px;border:2px solid #f8fbff;border-radius:50%;background:#51647a}
#details{min-height:0;overflow:auto;padding:26px 22px;border-left:1px solid var(--line);background:var(--panel);scrollbar-width:thin;scrollbar-color:#c9c3b7 transparent}.detail-eyebrow{font-size:10px;letter-spacing:1.5px;color:var(--accent);margin:0 0 17px;font-weight:600}.card-head{display:flex;align-items:center;gap:9px}#detail-role{font:28px/1.15 var(--serif);margin:0;letter-spacing:-.6px}.card-role-dot{width:9px;height:9px;flex:0 0 9px}.confidence{font-size:11px;color:var(--muted);margin:9px 0 0}#detail-meaning{font-size:12px;color:#777064;line-height:1.7;margin:12px 0 20px}.gid-row{display:flex;align-items:center;justify-content:space-between;gap:7px;margin-bottom:14px}#detail-gid{font:12px ui-monospace,SFMono-Regular,monospace;overflow-wrap:anywhere;min-width:0}#copy-gid{padding:4px 6px;font-size:10px;color:var(--accent);background:transparent;border-color:transparent}#copy-gid:hover{background:var(--accent-soft)}
#badges{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 20px}.badge{font-size:10px;line-height:1.6;padding:3px 7px;border:1px solid var(--line);border-radius:5px;color:#746b5e;background:#f7f4ed}.badge.queue-badge{color:var(--accent);background:var(--accent-soft);border-color:#e6d0c0}.badge.seed-badge{border-color:#8c8375;color:#534c40;background:transparent}.badge.warning{border-color:#dcc390;color:#7a5d26;background:#f8efdc}
.flow-tiles{display:grid;grid-template-columns:1fr 1fr;gap:10px}.flow-tile{border:1px solid var(--line);border-radius:10px;padding:13px 11px;background:#f8f6f0}.tile-label{font-size:11px;color:#687e89}.flow-tile.out .tile-label{color:var(--accent)}.tile-amount{font:25px/1.3 var(--serif);letter-spacing:-.5px;margin:6px 0}.tile-meta{font-size:10px;line-height:1.6;color:#6f685e}#flow-ratio{font-size:11px;font-weight:500;color:#706351;line-height:1.7;margin:14px 0 7px}#caveats{font-size:11px;color:#937043;line-height:1.65}#caveats p{margin:6px 0}
.card-section{border-top:1px solid var(--line);margin-top:23px;padding-top:18px}.card-section h3{margin:0 0 10px;font-size:10px;font-weight:600;letter-spacing:1px;color:#8c6f57}.card-section p{font-size:12px;line-height:1.75;margin:8px 0;overflow-wrap:anywhere}#hints{padding:15px;margin-top:22px;border:1px solid #e6d3c2;border-radius:10px;background:#f6ede3}#hints h3{color:var(--accent)}#hints p{font-size:11px;color:#806449}#hint-request{color:#674d35}
.peer-filter{width:100%;border:1px solid var(--line);border-radius:7px;background:#f8f6f0;padding:7px 9px;font-size:11px;color:var(--text);margin:0 0 6px}.peer-filter::placeholder{color:#6f685e}.peer-row{display:grid;grid-template-columns:7px minmax(0,1fr) auto;align-items:center;gap:8px;width:100%;background:none;border:0;border-radius:6px;padding:8px 3px;text-align:left}.peer-row:hover{background:#f0ebe1}.peer-id{font:11px ui-monospace,SFMono-Regular,monospace}.peer-role{display:block;font-size:10px;color:#6f685e;white-space:normal}.peer-amount{font-size:11px;font-weight:500}.peer-more{width:100%;margin-top:8px;padding:7px;font-size:11px;color:var(--accent);background:transparent;border-style:dashed}.no-peers{font-size:11px;color:#6f685e}.disclaimer{font-size:10px;line-height:1.7;color:#6f685e;border-top:1px solid var(--line);padding-top:17px;margin:23px 0 0}
.canvas-controls{position:absolute;right:16px;bottom:16px;display:flex;align-items:center;gap:5px;z-index:2;padding:5px;border:1px solid #405168;border-radius:10px;background:#1c2735;box-shadow:0 3px 16px #0003}.canvas-controls button{background:#253345;border:1px solid #405168;color:#e2ebf6;padding:6px 10px;min-width:32px;font-size:13px}.canvas-controls button:hover{background:#354960}.canvas-controls button:focus-visible{outline-color:#edb986}
div.vis-tooltip{max-width:390px;white-space:pre-wrap!important;overflow-wrap:anywhere;background:#fffdf8!important;color:#403a31!important;border:1px solid #cfc3b2!important;border-radius:8px!important;box-shadow:0 5px 20px #50443215!important;padding:12px!important;font:12px/1.7 system-ui,sans-serif!important}
@media(min-width:1700px){#app{grid-template-columns:330px minmax(0,1fr) 380px}header{gap:42px}.funnel{gap:44px}.queue-item{padding-top:15px;padding-bottom:15px}.queue-role{font-size:13px}.queue-why{font-size:12px}.card-section p{font-size:13px}.center-footer{padding-bottom:20px}}
@media(max-width:1200px){#app{grid-template-columns:270px minmax(0,1fr) 310px}header{gap:16px;padding:18px}.brand{flex-basis:230px}.brand-name{font-size:26px}.funnel{gap:14px}.funnel-step{min-width:80px}.funnel span{font-size:10px}#details{padding:22px 16px}.queue-gid{font-size:10px}.queue-role{font-size:11px}.canvas-header{padding:22px 18px 16px}.canvas-title{font-size:24px}.tile-amount{font-size:23px}}
@media(max-width:1000px){body{overflow:auto}header{height:auto;flex-wrap:wrap}.brand{flex:1}.funnel{order:3;flex-basis:100%;justify-content:flex-start;padding:14px 0 0;border-top:1px solid var(--line)}#app{height:auto;min-height:800px;grid-template-columns:260px minmax(0,1fr)}.queue-panel{max-height:740px}.center{height:740px}#details{grid-column:1/-1;overflow:visible;border-top:1px solid var(--line);border-left:0;padding:26px}.flow-tiles{max-width:500px}}
@media(max-width:620px){header{padding:17px}.brand-name{font-size:25px}.brand{gap:9px}#overview{font-size:10px;padding:8px}.funnel{gap:14px}.funnel strong{font-size:25px}.funnel-step{min-width:64px}.funnel-arrow{font-size:16px}#app{display:flex;flex-direction:column}.queue-panel{max-height:360px;border-right:0;border-bottom:1px solid var(--line)}.queue-head{padding:18px 16px 12px}h2.section-label{font-size:21px}#status{min-height:0;margin-bottom:15px}.center{height:570px;flex:none}.canvas-header{padding:20px 16px}.canvas-title{font-size:25px}.canvas-wrap{min-height:300px}.queue-gid{font-size:12px}.queue-role{font-size:13px}.queue-why{-webkit-line-clamp:1}.center-footer{padding:13px 16px}#details{padding:24px 20px}.peer-row{padding:11px 3px}.peer-id,.peer-amount{font-size:12px}}
@media(prefers-reduced-motion:reduce){button{transition:none}}
</style>
<script>__VIS_JS__</script>
</head>
<body>
<header>
 <div class="brand"><div class="brand-name"><b>FuryHub</b><small>Граф денег · рабочее место аналитика</small></div></div>
 <div class="funnel" aria-label="От известных клиентов к очереди проверки">
  <div class="funnel-step"><strong id="seed-count"></strong><span>известны из дела</span></div><div class="funnel-arrow" aria-hidden="true">→</div>
  <div class="funnel-step"><strong id="node-count"></strong><span>в сети переводов</span></div><div class="funnel-arrow" aria-hidden="true">→</div>
  <div class="funnel-step accent"><strong id="top-count"></strong><span>на проверку</span></div>
 </div>
 <button id="overview" type="button">Обзор сети</button>
</header>
<main id="app">
 <aside class="queue-panel" aria-label="Очередь проверки">
  <div class="queue-head"><label class="search-label" for="gid">Найти клиента во всей сети</label>
   <form id="search-form"><input id="gid" inputmode="numeric" autocomplete="off" spellcheck="false" placeholder="Полный gid или последние цифры"><button type="submit" aria-label="Найти клиента">Найти</button></form>
   <div id="status" role="status" aria-live="polite">Полный gid или его хвост — от 4 цифр.</div>
   <h2 class="section-label">Очередь проверки</h2><div class="queue-subtitle">Начните с самого важного</div>
  </div><div id="queue"></div>
 </aside>
 <section class="center" aria-label="Схема переводов">
  <div class="canvas-header"><div class="canvas-heading"><div class="eyebrow">ИССЛЕДОВАНИЕ СВЯЗЕЙ</div><h1 class="canvas-title">За каждым переводом — связь.</h1><div id="mode"></div></div><button id="back" type="button" disabled>← Назад</button></div>
  <div class="canvas-wrap"><div id="network" aria-label="Интерактивная схема денежных переводов"></div><div id="directions"><span>← кто платит</span><span>кому платит →</span></div><p id="isolated" hidden>Видимых переводов у этого клиента нет</p><div class="canvas-controls" aria-label="Масштаб схемы"><button id="zoom-out" type="button" aria-label="Уменьшить масштаб">−</button><button id="zoom-in" type="button" aria-label="Увеличить масштаб">+</button><button id="fit-view" type="button">Вписать</button></div></div>
  <p class="canvas-hint" id="canvas-hint">Клик по клиенту — его окружение · колесо — масштаб</p>
  <div id="hidden-peers" hidden></div><div class="center-footer"><div id="legend"></div></div>
 </section>
 <aside id="details" aria-label="Карточка клиента">
  <p class="detail-eyebrow">КАРТОЧКА КЛИЕНТА</p>
  <div class="card-head"><i id="detail-dot" class="role-dot card-role-dot"></i><h2 id="detail-role"></h2></div><p class="confidence" id="detail-confidence"></p><p id="detail-meaning"></p>
  <div class="gid-row"><span id="detail-gid"></span><button type="button" id="copy-gid">копировать</button></div><div id="badges"></div>
  <div class="flow-tiles"><div class="flow-tile"><div class="tile-label">Получил</div><div class="tile-amount" id="received-amount"></div><div class="tile-meta" id="received-meta"></div></div><div class="flow-tile out"><div class="tile-label">Отправил</div><div class="tile-amount" id="sent-amount"></div><div class="tile-meta" id="sent-meta"></div></div></div>
  <p id="flow-ratio"></p><div id="caveats"></div>
  <section class="card-section"><h3>ОСНОВАНИЕ ДЛЯ ПРОВЕРКИ</h3><p id="detail-why"></p></section>
  <section class="card-section" id="hints" hidden><h3>ПОДСКАЗКА АНАЛИТИКУ</h3><p id="hint-attention"></p><p id="hint-request"></p></section>
  <section class="card-section"><h3>ПЛАТЕЛЬЩИКИ · ПО СУММЕ</h3><input id="payers-filter" class="peer-filter" aria-label="Поиск среди плательщиков" placeholder="Поиск по gid" autocomplete="off"><div id="payers"></div><button id="payers-more" class="peer-more" type="button" hidden>Показать ещё</button></section>
  <section class="card-section"><h3>ПОЛУЧАТЕЛИ · ПО СУММЕ</h3><input id="recipients-filter" class="peer-filter" aria-label="Поиск среди получателей" placeholder="Поиск по gid" autocomplete="off"><div id="recipients"></div><button id="recipients-more" class="peer-more" type="button" hidden>Показать ещё</button></section>
  <p class="disclaimer">Роль — гипотеза по видимой части переводов. Сила правила не является вероятностью нарушения. Назначение платежей требует проверки.</p>
 </aside>
</main>
<script>
"use strict";
const DATA = __DATA__;
const byId = new Map(DATA.nodes.map(n => [n.id, n]));
const topById = new Map(DATA.top.map(n => [n.gid, n]));
const incoming = new Map(DATA.nodes.map(n => [n.id, []]));
const outgoing = new Map(DATA.nodes.map(n => [n.id, []]));
for (const edge of DATA.edges) {incoming.get(edge.to).push(edge); outgoing.get(edge.from).push(edge);}
const el = id => document.getElementById(id);
const setText = (id,text) => {el(id).textContent=text;};
function element(tag, text, className) {
    const item=document.createElement(tag);
    if(text !== undefined) item.textContent=text;
    if(className) item.className=className;
    return item;
}
function number(value,digits=0) {return Number(value).toLocaleString('ru-RU',{maximumFractionDigits:digits});}
function money(value) {
    const amount=Number(value)||0;
    if(Math.abs(amount)>=1e6) return `${number(amount/1e6,1)} млн ₸`;
    if(Math.abs(amount)>=1e3) return `${number(amount/1e3)} тыс ₸`;
    return `${number(amount,2)} ₸`;
}
function plural(value,one,few,many) {
    const n=Math.abs(Number(value))%100;
    return n>=11 && n<=14 ? many : n%10===1 ? one : n%10>=2 && n%10<=4 ? few : many;
}
function transfers(n) {return `${number(n)} ${plural(n,'перевод','перевода','переводов')}`;}
function shortGid(gid) {return gid.length>DATA.shortLen ? `…${gid.slice(-DATA.shortLen)}` : gid;}
function message(text,error=false) {setText('status',text); el('status').classList.toggle('error',error);}
function dot(node) {const item=element('i',undefined,'role-dot'); item.style.background=node.color; return item;}
function title(text) {return element('div',text);}
const nodes = new vis.DataSet();
const edges = new vis.DataSet();
const network = new vis.Network(el('network'), {nodes,edges}, {
    physics:{enabled:false}, layout:{improvedLayout:false,randomSeed:42},
    interaction:{hover:true,tooltipDelay:120,selectConnectedEdges:false,hoverConnectedEdges:false,keyboard:{enabled:true},hideEdgesOnDrag:true},
    nodes:{shape:'dot',font:{color:'#e6edf7',face:'system-ui',size:27,strokeWidth:0},chosen:false},
    edges:{smooth:false,arrows:{to:{enabled:true,scaleFactor:.7}},chosen:false},
});
let currentGid=null;
let currentMode=DATA.initialMode;
let overviewFocus=null;
const history=[];
const queueRows=new Map();
const peerLimits={payers:5,recipients:5};
function remember(gid,mode) {
    if(currentGid!==null && (currentGid!==gid || currentMode!==mode)) history.push({gid:currentGid,mode:currentMode});
}
// Непрозрачный светлый оттенок цвета роли: рёбра не просвечивают сквозь плашки.
function tint(hex,share) {
    const v=parseInt(hex.slice(1),16), mix=c=>Math.round(255-(255-c)*share);
    return `rgb(${mix(v>>16)},${mix((v>>8)&255)},${mix(v&255)})`;
}
function highlightQueue() {
    for(const [gid,row] of queueRows) {
        const active=gid===currentGid;
        row.classList.toggle('active',active);
        row.setAttribute('aria-pressed',String(active));
    }
    queueRows.get(currentGid)?.scrollIntoView({block:'nearest',inline:'nearest'});
}
function badge(text,className='') {el('badges').append(element('span',text,`badge ${className}`));}
function peers(gid,direction) {
    const list=direction==='incoming' ? incoming.get(gid) : outgoing.get(gid);
    return list.map(edge=>({edge,id:direction==='incoming' ? edge.from : edge.to}))
        .sort((a,b)=>b.edge.sum_kzt-a.edge.sum_kzt || a.id.localeCompare(b.id));
}
function renderPeers(gid,direction,target) {
    const container=el(target);container.replaceChildren();
    const query=el(`${target}-filter`).value.trim();
    const all=peers(gid,direction).filter(peer=>peer.id.includes(query));
    const list=all.slice(0,peerLimits[target]);
    for(const peer of list) {
        const node=byId.get(peer.id), row=element('button',undefined,'peer-row');row.type='button';row.dataset.gid=peer.id;
        row.title=`${peer.id} · ${transfers(peer.edge.n_tx)}`;
        const description=element('span');description.append(element('span',shortGid(peer.id),'peer-id'),element('span',node.role_label,'peer-role'));
        row.append(dot(node),description,element('span',money(peer.edge.sum_kzt),'peer-amount'));
        row.addEventListener('click',()=>openNeighborhood(peer.id));container.append(row);
    }
    if(!list.length) container.append(element('span',query ? 'Совпадений нет' : 'Нет видимых переводов','no-peers'));
    const more=el(`${target}-more`),remaining=all.length-list.length;
    more.hidden=remaining<=0;
    more.textContent=`Показать ещё ${Math.min(20,remaining)} · осталось ${remaining}`;
}
function renderCard(gid) {
    if(currentGid!==gid) {
        for(const target of ['payers','recipients']) {peerLimits[target]=5;el(`${target}-filter`).value='';}
    }
    const node=byId.get(gid);currentGid=gid;
    const top=topById.get(gid);
    setText('detail-role',node.role_label);el('detail-dot').style.background=node.color;
    setText('detail-confidence',`Сила правила ${number(node.role_score,2)} / 1`);
    el('detail-confidence').title='Сила сработавшего правила; не вероятность нарушения';
    setText('detail-meaning',node.role_meaning);setText('detail-gid',gid);setText('copy-gid','копировать');
    el('badges').replaceChildren();
    if(top) badge(`№${top.rank} в очереди проверки`,'queue-badge');
    if(node.is_seed) badge('из списка по делу (seed)','seed-badge');
    if(node.truncated_by_depth) badge('обход оборвался на 4-м шаге: дальнейшие переводы не видны','warning');
    if(node.incoming_incomplete && !node.seed_incoming_incomplete) badge('неполный видимый вход','warning');
    if(!node.outgoing_coverage_known) badge('полнота исходящих неизвестна','warning');
    badge(node.depth_known ? `${node.depth}-й шаг от известных клиентов` : 'глубина обхода неизвестна');badge(`группа ${node.cluster_id}`);
    setText('received-amount',money(node.in_kzt));setText('sent-amount',money(node.out_kzt));
    setText('received-meta',`от ${number(node.in_deg)} ${plural(node.in_deg,'плательщика','плательщиков','плательщиков')} · ${transfers(node.in_tx)}`);
    setText('sent-meta',`${number(node.out_deg)} ${plural(node.out_deg,'получателю','получателям','получателям')} · ${transfers(node.out_tx)}`);
    const ratio=node.in_kzt ? node.out_kzt/node.in_kzt : null;
    setText('flow-ratio',ratio===null ? 'Входящих переводов в выборке нет' : `Выход составляет ${number(ratio*100)}% видимого входа за период`);
    el('caveats').replaceChildren();
    if(node.seed_incoming_incomplete) el('caveats').append(element('p','У клиентов из дела входящие занижены: выгрузка собрана от них.'));
    if(node.incoming_incomplete && !node.seed_incoming_incomplete) el('caveats').append(element('p','Возможны пропущенные поступления или остаток на начало периода. Превышение выхода над входом само по себе не подтверждает транзит.'));
    if(!node.outgoing_coverage_known) el('caveats').append(element('p','Полнота исходящих не подтверждена. Нельзя делать вывод об удержании средств или конечном получателе.'));
    if(node.truncated_by_depth) el('caveats').append(element('p','Нулевой выход не доказывает, что деньги осели.'));
    setText('detail-why',top ? top.why : node.evidence);
    const hint=top && DATA.hints[gid];el('hints').hidden=!hint;
    setText('hint-attention',hint ? `На что обратить внимание: ${hint.attention}` : '');
    setText('hint-request',hint ? `Следующий запрос: ${hint.next_request}` : '');
    renderPeers(gid,'incoming','payers');renderPeers(gid,'outgoing','recipients');highlightQueue();
    el('details').scrollTop=0;
    if(window.parent && window.parent!==window && window.location.protocol!=='file:') {
        window.parent.postMessage({type:'furyhub:node-selected',gid},window.location.origin);
    }
}
function updateChrome() {
    const overview=currentMode==='overview';
    el('overview').classList.toggle('active',overview);setText('overview',overview ? '← К окружению' : 'Обзор сети');
    el('overview').setAttribute('aria-pressed',String(overview));el('directions').hidden=overview;el('back').disabled=!history.length;
    setText('canvas-hint',overview ? 'Клик — выделить связи · пустое место — вся схема · двойной клик — окружение' : 'Клик по клиенту — его окружение · колесо или + / − — масштаб');
}
function fitGraph() {
    network.fit({animation:false});
    if(network.getScale()>0.8) network.moveTo({scale:0.8});
    if(currentMode==='overview') styleOverview();
}
function canvasNode(node,position,kind,amountText) {
    const center=kind==='center', both=kind==='both';
    const color=node.graph_color || node.color;
    const border=node.is_seed ? '#f8fbff' : color;
    const rank=topById.get(node.id)?.rank;
    return {
        id:node.id,...position,physics:false,fixed:true,shape:center ? 'dot' : 'box',size:center ? 44 : node.size,
        label:center ? `${shortGid(node.id)}${rank ? ` · №${rank}` : ''}` : `${shortGid(node.id)}${both ? '\n' : '   '}<b>${amountText}</b>`,
        color:{background:center ? color : '#202d3d',border,highlight:{background:center ? color : '#344960',border},hover:{background:center ? color : '#344960',border}},
        borderWidth:node.is_seed ? 3 : 1.5,borderWidthSelected:node.is_seed ? 3 : 1.5,
        font:{size:center ? 30 : 27,multi:center ? false : 'html',color:'#e6edf7',face:'system-ui',strokeWidth:0,bold:{color:'#f4c28d',size:27}},
        margin:{top:8,bottom:8,left:13,right:13},...(center ? {} : {widthConstraint:{minimum:300,maximum:310}}),
        title:title(`${node.id}\n${node.role_label} · приоритет ${number(node.priority_score,3)}\n${node.evidence}`),
    };
}
function canvasEdge(edge,gid,maxAmount,bidirectional=false) {
    const color=gid===null ? '#91a4bc' : edge.to===gid ? '#90bce9' : '#edb17b';
    return {...edge,arrows:'to',width:1+3*Math.log1p(edge.sum_kzt)/Math.max(Math.log1p(maxAmount),1e-12),
        color:{color,highlight:color,hover:color,opacity:.8},
        smooth:bidirectional ? {enabled:true,type:'curvedCW',roundness:.2} : false,
        title:title(`${shortGid(edge.from)} → ${shortGid(edge.to)} / ${money(edge.sum_kzt)} · ${transfers(edge.n_tx)}`),
    };
}
function openNeighborhood(gid,rememberState=true) {
    if(!byId.has(gid)) return;
    if(rememberState) remember(gid,'neighborhood');currentMode='neighborhood';
    const inPeers=peers(gid,'incoming').filter(p=>p.id!==gid),outPeers=peers(gid,'outgoing').filter(p=>p.id!==gid);
    const inMap=new Map(inPeers.map(p=>[p.id,p])),outMap=new Map(outPeers.map(p=>[p.id,p]));
    const both=inPeers.filter(p=>outMap.has(p.id)).map(p=>({...p,total:p.edge.sum_kzt+outMap.get(p.id).edge.sum_kzt}))
        .sort((a,b)=>b.total-a.total || a.id.localeCompare(b.id));
    const left=inPeers.filter(p=>!outMap.has(p.id)),right=outPeers.filter(p=>!inMap.has(p.id));
    const shownLeft=left.slice(0,12),shownRight=right.slice(0,12),shownBoth=both.slice(0,5);
    const visibleNodes=[canvasNode(byId.get(gid),{x:0,y:0},'center')];
    for(const [list,x] of [[shownLeft,-450],[shownRight,450]]) list.forEach((p,i)=>visibleNodes.push(canvasNode(byId.get(p.id),{x,y:(i-(list.length-1)/2)*64},'side',money(p.edge.sum_kzt))));
    const upperY=-Math.max((Math.max(shownLeft.length,shownRight.length)-1)*32+135,190);
    shownBoth.forEach((p,i)=>visibleNodes.push(canvasNode(byId.get(p.id),{x:(i-(shownBoth.length-1)/2)*350,y:upperY},'both',`← ${money(p.edge.sum_kzt)} · ${money(outMap.get(p.id).edge.sum_kzt)} →`)));
    const selected=new Set(visibleNodes.map(n=>n.id)),bothSet=new Set(shownBoth.map(p=>p.id));
    const visibleEdges=DATA.edges.filter(e=>(e.from===gid || e.to===gid) && selected.has(e.from) && selected.has(e.to));
    const maxAmount=Math.max(0,...visibleEdges.map(e=>e.sum_kzt));
    network.unselectAll();edges.clear();nodes.clear();nodes.add(visibleNodes);
    edges.add(visibleEdges.map(e=>canvasEdge(e,gid,maxAmount,bothSet.has(e.from) || bothSet.has(e.to))));
    setText('mode',`Окружение ${shortGid(gid)} · ${inPeers.length} ${plural(inPeers.length,'плательщик','плательщика','плательщиков')} слева, ${outPeers.length} ${plural(outPeers.length,'получатель','получателя','получателей')} справа${both.length ? ` · ${both.length} встречных сверху` : ''}`);
    el('mode').title=el('mode').textContent;
    el('hidden-peers').replaceChildren();
    for(const [list,limit,label] of [[left,12,'плательщиков'],[right,12,'получателей'],[both,5,'встречных клиентов']]) {
        const hidden=list.slice(limit);
        if(hidden.length) el('hidden-peers').append(element('p',`+${hidden.length} ${label} ещё, суммой ${money(hidden.reduce((sum,p)=>sum+(p.total ?? p.edge.sum_kzt),0))} (показаны крупнейшие)`));
    }
    el('hidden-peers').hidden=!el('hidden-peers').children.length;el('isolated').hidden=visibleEdges.length!==0;
    renderCard(gid);network.selectNodes([gid],false);updateChrome();fitGraph();
}
function showOverview(rememberState=true) {
    if(rememberState) remember(currentGid,'overview');currentMode='overview';
    overviewFocus=null;
    const ids=new Set(DATA.overviewIds);
    network.unselectAll();edges.clear();nodes.clear();
    nodes.add(DATA.nodes.filter(n=>ids.has(n.id)).map(n=>({
        id:n.id,x:n.x,y:n.y,size:10+18*Math.sqrt(n.priority_score),shape:'dot',physics:false,fixed:true,label:'',
        color:{background:n.graph_color || n.color,border:n.is_seed ? '#f8fbff' : n.graph_color || n.color},borderWidth:n.is_seed ? 3 : 1,
        font:{size:14},title:title(`${n.id}\n${n.role_label} · приоритет ${number(n.priority_score,3)}\n${n.evidence}`),
    })));
    const visible=DATA.edges.filter(e=>ids.has(e.from) && ids.has(e.to));
    const max=Math.max(0,...visible.map(e=>e.sum_kzt));
    const pairs=new Set(visible.map(e=>JSON.stringify([e.from,e.to])));
    edges.add(visible.map(e=>canvasEdge(e,null,max,e.from!==e.to && pairs.has(JSON.stringify([e.to,e.from])))));
    setText('mode',`Обзор · ${DATA.overviewLabel} · ${number(ids.size)} из ${number(DATA.nodes.length)} клиентов`);el('mode').title=el('mode').textContent;
    el('hidden-peers').hidden=true;el('isolated').hidden=true;
    if(currentGid) {renderCard(currentGid);if(ids.has(currentGid)) network.selectNodes([currentGid],false);}
    updateChrome();fitGraph();
}
function styleOverview() {
    if(currentMode!=='overview') return;
    const related=new Set(overviewFocus ? [overviewFocus] : []);
    const totals=new Map();
    if(overviewFocus) {
        for(const edge of [...incoming.get(overviewFocus),...outgoing.get(overviewFocus)]) {
            const other=edge.from===overviewFocus ? edge.to : edge.from;
            related.add(other);totals.set(other,(totals.get(other)||0)+edge.sum_kzt);
        }
    }
    const scale=network.getScale(),expanded=scale>=0.55;
    // На общем масштабе сотни подписей перекрывают схему: показываем важные.
    const labelled=new Set(overviewFocus ? [overviewFocus,...[...totals]
        .sort((a,b)=>b[1]-a[1] || a[0].localeCompare(b[0]))
        .slice(0,expanded ? totals.size : 12).map(([id])=>id)] :
        DATA.top.filter(item=>item.rank<=10).map(item=>item.gid));
    nodes.update(nodes.get().map(item=>{
        const node=byId.get(item.id),active=!overviewFocus || related.has(item.id);
        return {id:item.id,opacity:active ? 1 : .22,
            label:(overviewFocus ? labelled.has(item.id) : expanded || labelled.has(item.id)) ? shortGid(item.id) : '',
            font:{color:'#e6edf7',size:Math.max(14,Math.min(80,11/Math.max(scale,.06))),strokeWidth:3,strokeColor:'#151c25'},
            borderWidth:item.id===overviewFocus ? 5 : node.is_seed ? 3 : 1};
    }));
    edges.update(edges.get().map(edge=>{
        const active=overviewFocus && (edge.from===overviewFocus || edge.to===overviewFocus);
        const color=active ? edge.to===overviewFocus ? '#90bce9' : '#edb17b' : '#91a4bc';
        return {id:edge.id,width:active ? 2.5 : .8,
            arrows:{to:{enabled:!!active || expanded,scaleFactor:active ? .8 : .4}},
            color:{color,highlight:color,hover:color,opacity:active ? .95 : overviewFocus ? .08 : .16}};
    }));
}
function goBack() {
    const previous=history.pop();if(!previous) return;
    if(previous.mode==='overview') {renderCard(previous.gid);showOverview(false);} else openNeighborhood(previous.gid,false);
}
function searchGid(query) {
    const text=query.trim();let gid=text;
    if(!byId.has(text)) {
        if(!/^\d{4,}$/.test(text)) {message(text ? `gid ${text} не найден. Введите полный gid или хвост от 4 цифр.` : 'Введите полный gid или его хвост — от 4 цифр.',true);return;}
        const matches=DATA.nodes.filter(n=>n.id.endsWith(text));
        if(matches.length>1) {message(`Найдено ${matches.length} gid на …${text} — введите больше цифр`,true);return;}
        if(!matches.length) {message(`gid ${text} не найден`,true);return;}gid=matches[0].id;
    }
    openNeighborhood(gid);
    const top=topById.get(gid);message(top ? `Найден: №${top.rank} в очереди проверки` : `Найден: ${byId.get(gid).role_label.toLowerCase()}, вне топа`);
}
for(const top of DATA.top) {
    const node=byId.get(top.gid), row=element('button',undefined,'queue-item');row.type='button';row.dataset.gid=top.gid;row.title=top.why;
    const line=element('span',undefined,'queue-line');line.append(element('span',String(top.rank),'rank'),dot(node),element('span',node.role_label,'queue-role'),element('span',shortGid(top.gid),'queue-gid'));
    row.append(line,element('span',top.why,'queue-why'));row.addEventListener('click',()=>openNeighborhood(top.gid));
    queueRows.set(top.gid,row);el('queue').append(row);
}
const roles=new Map(DATA.nodes.map(n=>[n.role,n]));
for(const role of ['coordinator','distributor','consolidator','transit','terminal','peripheral']) {
    const node=roles.get(role);if(!node) continue;
    const entry=element('span');entry.title=node.role_meaning;entry.append(dot({...node,color:node.graph_color || node.color}),element('span',node.role_label));el('legend').append(entry);
}
const seedLegend=element('span');seedLegend.append(element('i',undefined,'seed-dot'),element('span','светлая рамка — клиент из дела'));el('legend').append(seedLegend);
setText('seed-count',number(DATA.funnel.seeds));setText('node-count',number(DATA.funnel.nodes));setText('top-count',number(DATA.funnel.top));
el('search-form').addEventListener('submit',event=>{event.preventDefault();searchGid(el('gid').value);});
for(const [target,direction] of [['payers','incoming'],['recipients','outgoing']]) {
    el(`${target}-more`).addEventListener('click',()=>{peerLimits[target]+=20;renderPeers(currentGid,direction,target);});
    el(`${target}-filter`).addEventListener('input',()=>{peerLimits[target]=5;renderPeers(currentGid,direction,target);});
}
el('back').addEventListener('click',goBack);
el('overview').addEventListener('click',()=>currentMode==='overview' ? openNeighborhood(currentGid) : showOverview());
for(const [id,factor] of [['zoom-in',1.3],['zoom-out',1/1.3]]) {
    el(id).addEventListener('click',()=>{
        network.moveTo({scale:Math.max(.06,Math.min(2.5,network.getScale()*factor)),animation:false});
        styleOverview();
    });
}
el('fit-view').addEventListener('click',fitGraph);
network.on('zoom',styleOverview);
el('copy-gid').addEventListener('click',async()=>{
    try {await navigator.clipboard.writeText(currentGid);setText('copy-gid','скопировано');}
    catch {const range=document.createRange();range.selectNodeContents(el('detail-gid'));const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);setText('copy-gid','нажмите Ctrl/Cmd+C');}
});
network.on('click',event=>{
    if(!event.nodes.length) {if(currentMode==='overview') {overviewFocus=null;network.unselectAll();styleOverview();}return;}
    const gid=event.nodes[0];
    if(currentMode==='overview') {remember(gid,'overview');overviewFocus=gid;renderCard(gid);network.selectNodes([gid],false);styleOverview();updateChrome();}
    else openNeighborhood(gid);
});
network.on('doubleClick',event=>{if(event.nodes.length) openNeighborhood(event.nodes[0]);});
if(DATA.initialMode==='overview') {currentGid=DATA.initialGid;showOverview(false);} else openNeighborhood(DATA.initialGid,false);
</script>
</body>
</html>
"""


def build_html(data):
    # Не Jinja: последовательность {# допустима в CSS и пользовательском тексте.
    environment = Network().templateEnv
    css = environment.loader.get_source(environment, "lib/vis-9.1.2/vis-network.css")[0]
    javascript = environment.loader.get_source(environment, "lib/vis-9.1.2/vis-network.min.js")[0]
    serialized = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for character, escaped in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"),
                               ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        serialized = serialized.replace(character, escaped)
    return PAGE.replace("__VIS_CSS__", css).replace("__VIS_JS__", javascript).replace("__DATA__", serialized)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, default=Path("out/nodes_roles.csv"))
    parser.add_argument("--edges", type=Path, default=Path("data/edges.parquet"))
    parser.add_argument("--top", type=Path, default=Path("out/top_nodes.csv"))
    parser.add_argument("--output", type=Path, default=Path("out/graph.html"))
    parser.add_argument("--no-llm", action="store_true", help="Собрать экран без подсказок аналитика")
    parser.add_argument("--offline", action="store_true", help="Локальные подсказки без обращения к API")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gid", help="Открыть окружение этого клиента")
    group.add_argument("--cluster", help="Открыть обзор этой группы")
    args = parser.parse_args(argv)
    started = perf_counter()
    try:
        nodes, edges = load_data(args.nodes, args.edges)
        top = load_top(args.top, nodes)
        data = build_data(nodes, edges, top, gid=args.gid, cluster=args.cluster)
        if not args.no_llm:
            data["hints"] = collect_hints(nodes, edges, top, args.output.parent, offline=args.offline)
        html = build_html(data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(html, encoding="utf-8")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Ошибка: {error}\n")
    print(f"Готово: {args.output.resolve()} ({perf_counter() - started:.2f} с)")
    print("Откройте HTML в браузере. Поиск работает по всем gid без интернета.")


if __name__ == "__main__":
    main()
