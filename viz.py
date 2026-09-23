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

from llm_layer import LLMLayer, node_payload


ROLES = {
    "coordinator": ("Координатор", "#e15759"),
    "distributor": ("Распределитель", "#f28e2b"),
    "consolidator": ("Консолидатор", "#b279d0"),
    "transit": ("Транзит", "#4e9ee8"),
    "terminal": ("Конечный получатель", "#39b89b"),
    "peripheral": ("Периферия", "#929eaf"),
}
ROLE_MEANINGS = {
    "coordinator": "связывает несколько групп, кандидат в организаторы",
    "distributor": "раздаёт деньги многим получателям",
    "consolidator": "собирает деньги от нескольких участников",
    "transit": "пропускает деньги дальше, не удерживая",
    "terminal": "деньги приходят и остаются",
    "peripheral": "признаков роли не выявлено",
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
    if "truncated_by_depth" in nodes:
        nodes["truncated_by_depth"] = _booleans(nodes.truncated_by_depth, "truncated_by_depth")
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


def collect_hints(nodes, edges, top, out_dir):
    """Все карточки топа одним обращением; сбой подсказок не ломает экран."""
    if top.empty:
        return {}
    try:
        payloads = [node_payload(nodes, edges, gid) for gid in top.gid]
        layer = LLMLayer(out_dir, timeout_seconds=10.0, budget_seconds=10.0)
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
    initial_gid = gid or (str(top.iloc[0].gid) if len(top) else str(nodes.iloc[0].gid))
    selected, overview_label = select_nodes(nodes, graph, cluster=cluster)
    # Раскладка нужна только обзору. Окружение расставляет браузер по потокам.
    positions = nx.spring_layout(
        graph.subgraph(sorted(selected)), seed=42, iterations=45, weight=None,
        scale=max(350, math.sqrt(len(selected)) * 85),
    )
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
            "priority_score": score, "role_score": float(row.get("role_score", 0)),
            "evidence": str(row["evidence"]), "is_seed": bool(row["is_seed"]),
            "cluster_id": str(row["cluster_id"]), "depth": int(row.get("depth", 0)),
            "truncated_by_depth": bool(row.get("truncated_by_depth", False)),
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
:root{color-scheme:dark;--bg:#0b121d;--panel:#111c2a;--line:#243448;--text:#edf3fc;--muted:#91a4bd;--accent:#94d7be}
*{box-sizing:border-box}[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;overflow:hidden}
button,input{font:inherit}button{cursor:pointer;color:inherit}button:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
button{border:1px solid #38516a;border-radius:8px;background:#1b2b3e;padding:9px 13px;white-space:nowrap}button:hover{background:#263d54}
header{height:104px;padding:20px 26px;display:flex;align-items:center;gap:30px;border-bottom:1px solid var(--line);background:#101a28}
.brand{flex:0 0 284px;font-size:22px;font-weight:760;letter-spacing:-.6px}.brand b{color:var(--accent)}.brand small{display:block;font-size:12px;font-weight:400;color:var(--muted);letter-spacing:.2px;margin-top:3px}
.funnel{display:flex;align-items:center;justify-content:center;gap:26px;flex:1}.funnel-step{min-width:120px}.funnel strong{display:block;font-size:30px;line-height:1.15;font-variant-numeric:tabular-nums;letter-spacing:-1px}.funnel span{font-size:13px;color:#acbdd1;white-space:nowrap}.funnel .accent strong{color:var(--accent)}.funnel-arrow{color:#516980;font-size:24px}
#overview{min-width:160px;margin-left:auto}#overview.active{background:#26463f;border-color:var(--accent);color:#c4f1df}
#app{height:calc(100dvh - 104px);display:grid;grid-template-columns:340px minmax(0,1fr) 390px;min-height:0}
.queue-panel{display:flex;flex-direction:column;overflow:hidden;min-height:0;border-right:1px solid var(--line);background:#101926;scrollbar-width:thin;scrollbar-color:#34465c transparent}
.queue-head{padding:23px 18px 14px;flex-shrink:0;background:#101926;z-index:2;border-bottom:1px solid var(--line)}
.search-label{display:block;color:#b6c6d9;font-size:13px;margin-bottom:9px}#search-form{display:flex;gap:7px}#gid{width:100%;min-width:0;background:#0a121e;border:1px solid #3b4d64;color:#fff;border-radius:8px;padding:10px 11px;font:14px ui-monospace,SFMono-Regular,monospace}#search-form button{padding:8px 11px;color:#c4f1df}
#status{font-size:12px;line-height:1.4;color:var(--muted);min-height:34px;margin:9px 0 14px;overflow-wrap:anywhere}#status.error{color:#ff9fa8}
h2.section-label{font-size:11px;letter-spacing:1.3px;color:#b5c8df;margin:0;font-weight:750}.queue-subtitle{color:var(--muted);font-size:12px;margin-top:4px}
#queue{padding:7px 10px 18px;overflow:auto;min-height:0;scrollbar-width:thin}.queue-item{border:1px solid transparent;border-radius:9px;display:block;width:100%;padding:12px 9px;text-align:left;background:transparent;white-space:normal;margin:3px 0}.queue-item:hover{background:#1c2b3d}.queue-item.active{background:#1e3341;border-color:#50756c;box-shadow:inset 3px 0 var(--accent)}
.queue-line{display:flex;align-items:center;gap:7px;min-width:0}.rank{color:#a2b5ca;width:20px;font-size:14px;flex-shrink:0;font-variant-numeric:tabular-nums}.role-dot{display:inline-block;width:8px;height:8px;border-radius:50%;flex:0 0 8px}.queue-role{font-weight:650;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:14px}.queue-gid{margin-left:auto;color:#aebfd3;font:14px ui-monospace,SFMono-Regular,monospace;flex-shrink:0}.queue-why{display:block;color:#97abc3;font-size:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding-left:27px;margin-top:6px}
.center{min-width:0;min-height:0;overflow:auto;display:flex;flex-direction:column;position:relative;background:radial-gradient(ellipse at 50% 45%,#152334 0%,#0b131f 75%)}
.canvas-header{display:flex;align-items:center;gap:12px;padding:20px 20px 13px;flex-shrink:0}.canvas-heading{min-width:0;flex:1}.eyebrow{font-size:10px;letter-spacing:1.6px;color:#778fa9;margin-bottom:4px}#mode{font-size:14px;font-weight:650;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}#back{white-space:nowrap;flex-shrink:0;padding:7px 11px;font-size:13px}#back:disabled{opacity:.35;cursor:default}
.canvas-wrap{flex:1;min-height:300px;position:relative}#network{width:100%;height:100%;position:absolute;inset:0}#directions{display:flex;justify-content:space-between;position:absolute;left:22px;right:22px;top:6px;pointer-events:none;font-size:13px;color:#97b5da;opacity:.78}#directions span:last-child{color:#e4b273}.canvas-hint{color:#6f859e;font-size:11px;text-align:center;margin:0 16px 8px}#isolated{position:absolute;bottom:22%;left:20px;right:20px;text-align:center;color:#9caec3;pointer-events:none;font-size:15px}
#hidden-peers{margin:0 20px 13px;padding:10px 13px;background:#172334;border:1px solid #304158;border-radius:8px;font-size:12px;line-height:1.8;color:#b9cadd;flex-shrink:0}#hidden-peers p{margin:0}
.center-footer{padding:14px 20px 17px;border-top:1px solid var(--line);flex-shrink:0;background:#0d1724}#legend{display:flex;flex-wrap:wrap;gap:7px 14px;font-size:11px;color:#abbed4}#legend span{display:inline-flex;align-items:center;gap:5px}.seed-dot{width:10px;height:10px;border:2px solid white;border-radius:50%;background:#6b7889}
#details{min-height:0;overflow:auto;padding:23px 22px 22px;border-left:1px solid var(--line);background:#111c2a;scrollbar-width:thin;scrollbar-color:#34465c transparent}.card-head{display:flex;align-items:center;gap:9px}#detail-role{font-size:22px;line-height:1.25;font-weight:730;margin:0;letter-spacing:-.4px}.card-role-dot{width:11px;height:11px;flex:0 0 11px}.confidence{font-size:14px;color:var(--muted);margin:8px 0 0}#detail-meaning{font-size:14px;color:#acbfd6;line-height:1.55;margin:12px 0 17px}.gid-row{display:flex;align-items:center;justify-content:space-between;gap:9px;margin-bottom:14px}#detail-gid{font:14px ui-monospace,SFMono-Regular,monospace;overflow-wrap:anywhere;min-width:0}#copy-gid{padding:5px 8px;font-size:14px;color:#a7c0d9;background:transparent}
#badges{display:flex;flex-wrap:wrap;gap:7px;margin:0 0 20px}.badge{font-size:14px;line-height:1.5;padding:4px 8px;border:1px solid #3b5068;border-radius:5px;color:#becde0}.badge.queue-badge{color:#bcebd6;background:#203d35;border-color:#3f6b5c}.badge.seed-badge{border-color:#ccd9e6;color:#edf3fc}.badge.warning{border-color:#826b32;color:#ebd292;background:#342e1d}
.flow-tiles{display:grid;grid-template-columns:1fr 1fr;gap:9px}.flow-tile{border:1px solid #2c4057;border-radius:9px;padding:12px 11px;background:#132234}.tile-label{font-size:14px;color:#a9c1dd}.flow-tile.out .tile-label{color:#ddb47e}.tile-amount{font-size:22px;font-weight:700;line-height:1.3;letter-spacing:-.4px;margin:6px 0}.tile-meta{font-size:14px;line-height:1.55;color:#9bafc7}#flow-ratio{font-size:14px;font-weight:550;color:#c3d4e8;line-height:1.6;margin:13px 0 7px}#caveats{font-size:14px;color:#899db6;line-height:1.6}#caveats p{margin:6px 0}
.card-section{border-top:1px solid var(--line);margin-top:20px;padding-top:18px}.card-section h3{margin:0 0 11px;font-size:14px;letter-spacing:1.1px;color:#9eb4ce}.card-section p{font-size:14px;line-height:1.65;margin:8px 0;overflow-wrap:anywhere}#hints{padding:14px;margin-top:18px;border:1px solid #354e5f;border-radius:9px;background:#172b38}#hints h3{color:#96d5bb}#hints p{font-size:14px;color:#bad0e2}#hint-request{color:#d6e7ef!important}
.peer-row{display:grid;grid-template-columns:8px minmax(0,1fr) auto;align-items:center;gap:8px;width:100%;background:none;border:0;border-radius:6px;padding:8px 2px;text-align:left}.peer-row:hover{background:#203247}.peer-id{font:14px ui-monospace,SFMono-Regular,monospace}.peer-role{display:block;font-size:14px;color:#8fa7c2;white-space:normal}.peer-amount{font-size:14px;font-weight:600}.no-peers{font-size:14px;color:#8096af}.disclaimer{font-size:11px;line-height:1.6;color:#7e95af;border-top:1px solid var(--line);padding-top:16px;margin:23px 0 0}
div.vis-tooltip{max-width:390px;white-space:pre-wrap!important;overflow-wrap:anywhere;background:#172639!important;color:#edf3fc!important;border:1px solid #5a7390!important;border-radius:8px!important;padding:11px!important;font:13px/1.6 system-ui,sans-serif!important}
@media(min-width:1700px){header{gap:42px}.funnel{gap:44px}.queue-item{padding-top:14px;padding-bottom:14px}.center-footer{padding-bottom:20px}.canvas-header{padding-top:24px}}
@media(max-width:1200px){#app{grid-template-columns:290px minmax(0,1fr) 340px}header{gap:16px;padding:18px}.brand{flex-basis:245px;font-size:20px}.funnel{gap:14px}.funnel-step{min-width:100px}.funnel span{font-size:11px}#details{padding:20px 16px}.queue-gid{font-size:14px}.queue-role{font-size:14px}}
@media(max-width:950px){body{overflow:auto}header{height:auto;flex-wrap:wrap}.brand{flex:1}.funnel{order:3;flex-basis:100%}#app{height:auto;min-height:800px;grid-template-columns:280px minmax(0,1fr)}.queue-panel{max-height:850px}.center{height:850px}#details{grid-column:1/-1;overflow:visible}.flow-tiles{max-width:500px}}
</style>
<script>__VIS_JS__</script>
</head>
<body>
<header>
 <div class="brand"><b>FuryHub</b> · Граф денег<small>Рабочее место аналитика</small></div>
 <div class="funnel" aria-label="От известных клиентов к очереди проверки">
  <div class="funnel-step"><strong id="seed-count"></strong><span>известны из дела</span></div><div class="funnel-arrow" aria-hidden="true">→</div>
  <div class="funnel-step"><strong id="node-count"></strong><span>в сети переводов</span></div><div class="funnel-arrow" aria-hidden="true">→</div>
  <div class="funnel-step accent"><strong id="top-count"></strong><span>на проверку</span></div>
 </div>
 <button id="overview" type="button">Вся сеть</button>
</header>
<main id="app">
 <aside class="queue-panel" aria-label="Очередь проверки">
  <div class="queue-head"><label class="search-label" for="gid">Найти клиента во всей сети</label>
   <form id="search-form"><input id="gid" inputmode="numeric" autocomplete="off" spellcheck="false" placeholder="Полный gid или последние цифры"><button type="submit" aria-label="Найти клиента">Найти</button></form>
   <div id="status" role="status" aria-live="polite">Полный gid или его хвост — от 4 цифр.</div>
   <h2 class="section-label">КОГО ПРОВЕРЯТЬ ПЕРВЫМ</h2><div class="queue-subtitle">По приоритету · нажмите, чтобы изучить</div>
  </div><div id="queue"></div>
 </aside>
 <section class="center" aria-label="Схема переводов">
  <div class="canvas-header"><div class="canvas-heading"><div class="eyebrow">ДВИЖЕНИЕ ДЕНЕГ</div><div id="mode"></div></div><button id="back" type="button" disabled>← Назад</button></div>
  <div class="canvas-wrap"><div id="network" aria-label="Интерактивная схема денежных переводов"></div><div id="directions"><span>← кто платит</span><span>кому платит →</span></div><p id="isolated" hidden>Видимых переводов у этого клиента нет</p></div>
  <p class="canvas-hint" id="canvas-hint">Клик по клиенту — его окружение · колесо — масштаб</p>
  <div id="hidden-peers" hidden></div><div class="center-footer"><div id="legend"></div></div>
 </section>
 <aside id="details" aria-label="Карточка клиента">
  <div class="card-head"><i id="detail-dot" class="role-dot card-role-dot"></i><h2 id="detail-role"></h2></div><p class="confidence" id="detail-confidence"></p><p id="detail-meaning"></p>
  <div class="gid-row"><span id="detail-gid"></span><button type="button" id="copy-gid">копировать</button></div><div id="badges"></div>
  <div class="flow-tiles"><div class="flow-tile"><div class="tile-label">Получил</div><div class="tile-amount" id="received-amount"></div><div class="tile-meta" id="received-meta"></div></div><div class="flow-tile out"><div class="tile-label">Отправил</div><div class="tile-amount" id="sent-amount"></div><div class="tile-meta" id="sent-meta"></div></div></div>
  <p id="flow-ratio"></p><div id="caveats"></div>
  <section class="card-section"><h3>ПОЧЕМУ В СПИСКЕ</h3><p id="detail-why"></p></section>
  <section class="card-section" id="hints" hidden><h3>ПОДСКАЗКА АНАЛИТИКУ</h3><p id="hint-attention"></p><p id="hint-request"></p></section>
  <section class="card-section"><h3>КРУПНЕЙШИЕ ПЛАТЕЛЬЩИКИ</h3><div id="payers"></div></section>
  <section class="card-section"><h3>КРУПНЕЙШИЕ ПОЛУЧАТЕЛИ</h3><div id="recipients"></div></section>
  <p class="disclaimer">Роль — гипотеза для проверки по видимой части переводов, а не утверждение о виновности.</p>
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
    interaction:{hover:true,tooltipDelay:120,selectConnectedEdges:false,hoverConnectedEdges:false,keyboard:{enabled:true}},
    nodes:{shape:'dot',font:{color:'#e7eef8',face:'system-ui',size:27,strokeWidth:0},chosen:false},
    edges:{smooth:false,arrows:{to:{enabled:true,scaleFactor:.7}},chosen:false},
});
let currentGid=null;
let currentMode=DATA.initialMode;
const history=[];
const queueRows=new Map();
function remember(gid,mode) {
    if(currentGid!==null && (currentGid!==gid || currentMode!==mode)) history.push({gid:currentGid,mode:currentMode});
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
    const list=peers(gid,direction).slice(0,5);
    for(const peer of list) {
        const node=byId.get(peer.id), row=element('button',undefined,'peer-row');row.type='button';row.dataset.gid=peer.id;
        row.title=`${peer.id} · ${transfers(peer.edge.n_tx)}`;
        const description=element('span');description.append(element('span',shortGid(peer.id),'peer-id'),element('span',node.role_label,'peer-role'));
        row.append(dot(node),description,element('span',money(peer.edge.sum_kzt),'peer-amount'));
        row.addEventListener('click',()=>openNeighborhood(peer.id));container.append(row);
    }
    if(!list.length) container.append(element('span','Нет видимых переводов','no-peers'));
}
function renderCard(gid) {
    const node=byId.get(gid);currentGid=gid;
    const top=topById.get(gid);
    setText('detail-role',node.role_label);el('detail-dot').style.background=node.color;
    setText('detail-confidence',`уверенность ${number(node.role_score*100)}%`);
    el('detail-confidence').title='Сила сработавшего правила; не вероятность нарушения';
    setText('detail-meaning',node.role_meaning);setText('detail-gid',gid);setText('copy-gid','копировать');
    el('badges').replaceChildren();
    if(top) badge(`№${top.rank} в очереди проверки`,'queue-badge');
    if(node.is_seed) badge('из списка по делу (seed)','seed-badge');
    if(node.truncated_by_depth) badge('обход оборвался на 4-м шаге: дальнейшие переводы не видны','warning');
    badge(`${node.depth}-й шаг от известных клиентов`);badge(`группа ${node.cluster_id}`);
    setText('received-amount',money(node.in_kzt));setText('sent-amount',money(node.out_kzt));
    setText('received-meta',`от ${number(node.in_deg)} ${plural(node.in_deg,'плательщика','плательщиков','плательщиков')} · ${transfers(node.in_tx)}`);
    setText('sent-meta',`${number(node.out_deg)} ${plural(node.out_deg,'получателю','получателям','получателям')} · ${transfers(node.out_tx)}`);
    const ratio=node.in_kzt ? node.out_kzt/node.in_kzt : null;
    setText('flow-ratio',ratio===null ? 'Входящих переводов в выборке нет' : ratio<=1.5 ? `Передаёт дальше ${number(ratio*100)}% полученного` : `Отправил в ${number(ratio,1)} раза больше, чем видно на входе`);
    el('caveats').replaceChildren();
    if(node.is_seed) el('caveats').append(element('p','У клиентов из дела входящие занижены: выгрузка собрана от них.'));
    if(node.truncated_by_depth) el('caveats').append(element('p','Нулевой выход не доказывает, что деньги осели.'));
    setText('detail-why',top ? top.why : node.evidence);
    const hint=top && DATA.hints[gid];el('hints').hidden=!hint;
    setText('hint-attention',hint ? `На что обратить внимание: ${hint.attention}` : '');
    setText('hint-request',hint ? `Следующий запрос: ${hint.next_request}` : '');
    renderPeers(gid,'incoming','payers');renderPeers(gid,'outgoing','recipients');highlightQueue();
    el('details').scrollTop=0;
}
function updateChrome() {
    const overview=currentMode==='overview';
    el('overview').classList.toggle('active',overview);setText('overview',overview ? '← К окружению' : 'Вся сеть');
    el('overview').setAttribute('aria-pressed',String(overview));el('directions').hidden=overview;el('back').disabled=!history.length;
    setText('canvas-hint',overview ? 'Клик — карточка · двойной клик — окружение · колесо — масштаб' : 'Клик по клиенту — его окружение · колесо — масштаб');
}
function fitGraph() {
    network.fit({animation:false});
    if(network.getScale()>0.8) network.moveTo({scale:0.8});
}
function canvasNode(node,position,kind,amountText) {
    const center=kind==='center', both=kind==='both';
    const border=node.is_seed ? '#ffffff' : node.color;
    const rank=topById.get(node.id)?.rank;
    return {
        id:node.id,...position,physics:false,fixed:true,shape:center ? 'dot' : 'box',size:center ? 44 : node.size,
        label:center ? `${shortGid(node.id)}${rank ? ` · №${rank}` : ''}` : `${shortGid(node.id)}${both ? '\n' : '   '}<b>${amountText}</b>`,
        color:{background:center ? node.color : node.color+'38',border,highlight:{background:node.color+'55',border},hover:{background:node.color+'66',border}},
        borderWidth:node.is_seed ? 3 : 1.5,borderWidthSelected:node.is_seed ? 3 : 1.5,
        font:{size:center ? 30 : 27,multi:center ? false : 'html',color:'#e7eef8',face:'system-ui',strokeWidth:0,bold:{color:'#ffffff',size:27}},
        margin:{top:8,bottom:8,left:13,right:13},...(center ? {} : {widthConstraint:{minimum:300,maximum:310}}),
        title:title(`${node.id}\n${node.role_label}\n${node.evidence}`),
    };
}
function canvasEdge(edge,gid,maxAmount,bidirectional=false) {
    const color=gid===null ? '#72859d' : edge.to===gid ? '#6f93c4' : '#d9a45f';
    return {...edge,arrows:'to',width:1+6*Math.log1p(edge.sum_kzt)/Math.max(Math.log1p(maxAmount),1e-12),
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
    const ids=new Set(DATA.overviewIds);
    network.unselectAll();edges.clear();nodes.clear();
    nodes.add(DATA.nodes.filter(n=>ids.has(n.id)).map(n=>({
        id:n.id,x:n.x,y:n.y,size:n.size,shape:'dot',physics:false,fixed:true,label:shortGid(n.id),
        color:{background:n.color,border:n.is_seed ? '#ffffff' : n.color},borderWidth:n.is_seed ? 3 : 1,
        font:{size:14},title:title(`${n.id}\n${n.role_label}\n${n.evidence}`),
    })));
    const visible=DATA.edges.filter(e=>ids.has(e.from) && ids.has(e.to));
    const max=Math.max(0,...visible.map(e=>e.sum_kzt));
    edges.add(visible.map(e=>canvasEdge(e,null,max)));
    setText('mode',`Вся сеть · ${DATA.overviewLabel} · ${number(ids.size)} клиентов`);el('mode').title=el('mode').textContent;
    el('hidden-peers').hidden=true;el('isolated').hidden=true;
    if(currentGid) {renderCard(currentGid);if(ids.has(currentGid)) network.selectNodes([currentGid],false);}
    updateChrome();fitGraph();
}
function goBack() {
    const previous=history.pop();if(!previous) return;
    if(previous.mode==='overview') {currentGid=previous.gid;showOverview(false);} else openNeighborhood(previous.gid,false);
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
    const entry=element('span');entry.title=node.role_meaning;entry.append(dot(node),element('span',node.role_label));el('legend').append(entry);
}
const seedLegend=element('span');seedLegend.append(element('i',undefined,'seed-dot'),element('span','белая рамка — клиент из дела'));el('legend').append(seedLegend);
setText('seed-count',number(DATA.funnel.seeds));setText('node-count',number(DATA.funnel.nodes));setText('top-count',number(DATA.funnel.top));
el('search-form').addEventListener('submit',event=>{event.preventDefault();searchGid(el('gid').value);});
el('back').addEventListener('click',goBack);
el('overview').addEventListener('click',()=>currentMode==='overview' ? openNeighborhood(currentGid) : showOverview());
el('copy-gid').addEventListener('click',async()=>{
    try {await navigator.clipboard.writeText(currentGid);setText('copy-gid','скопировано');}
    catch {const range=document.createRange();range.selectNodeContents(el('detail-gid'));const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);setText('copy-gid','нажмите Ctrl/Cmd+C');}
});
network.on('click',event=>{
    if(!event.nodes.length) return;const gid=event.nodes[0];
    if(currentMode==='overview') {remember(gid,'overview');renderCard(gid);network.selectNodes([gid],false);updateChrome();}
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
            data["hints"] = collect_hints(nodes, edges, top, args.output.parent)
        html = build_html(data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(html, encoding="utf-8")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Ошибка: {error}\n")
    print(f"Готово: {args.output.resolve()} ({perf_counter() - started:.2f} с)")
    print("Откройте HTML в браузере. Поиск работает по всем gid без интернета.")


if __name__ == "__main__":
    main()
