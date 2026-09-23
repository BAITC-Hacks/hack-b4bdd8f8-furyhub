#!/usr/bin/env python3
"""Автономная схема: python viz.py [--gid ID | --cluster ID]."""

import argparse
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import networkx as nx
import pandas as pd
from pyvis.network import Network


ROLES = {
    "coordinator": ("Координатор", "#e15759"),
    "distributor": ("Распределитель", "#f28e2b"),
    "consolidator": ("Консолидатор", "#b279d0"),
    "transit": ("Транзит", "#4e9ee8"),
    "terminal": ("Конечный получатель", "#39b89b"),
    "peripheral": ("Периферия", "#929eaf"),
}


def load_data(nodes_path, edges_path):
    # Нельзя пропускать gid через float или JavaScript Number: он длиннее 53 бит.
    nodes = pd.read_csv(nodes_path, dtype={"gid": str, "cluster_id": str},
                        keep_default_na=False)
    edges = pd.read_parquet(edges_path)
    required = {"gid", "role", "priority_score", "evidence", "is_seed", "cluster_id"}
    if missing := required - set(nodes.columns):
        raise ValueError(f"В nodes_roles.csv отсутствуют колонки: {', '.join(sorted(missing))}")
    if missing := {"src", "dst", "sum_kzt"} - set(edges.columns):
        raise ValueError(f"В edges.parquet отсутствуют колонки: {', '.join(sorted(missing))}")
    for column in ["gid", "cluster_id"]:
        nodes[column] = nodes[column].str.strip()
        if nodes[column].eq("").any():
            raise ValueError(f"Пустой {column} в nodes_roles.csv")
    if nodes.empty or nodes.gid.duplicated().any():
        raise ValueError("Нужен непустой список уникальных gid")
    if not nodes.role.isin(ROLES).all():
        raise ValueError("Неизвестная роль в nodes_roles.csv")
    nodes["priority_score"] = pd.to_numeric(nodes.priority_score, errors="raise")
    if not nodes.priority_score.between(0, 1).all():
        raise ValueError("priority_score должен быть в диапазоне 0..1")
    seeds = nodes.is_seed.astype(str).str.lower().str.strip()
    if not seeds.isin(["true", "false", "1", "0"]).all():
        raise ValueError("is_seed должен содержать true/false или 1/0")
    nodes["is_seed"] = seeds.isin(["true", "1"])
    for column in ["src", "dst"]:
        if edges[column].isna().any() or pd.api.types.is_float_dtype(edges[column]):
            raise ValueError(f"{column}: нужны целые числа или строки без пропусков")
        edges[column] = edges[column].astype(str).str.strip()
    if (set(edges.src) | set(edges.dst)) - set(nodes.gid):
        raise ValueError("В edges есть gid, отсутствующий в nodes_roles.csv")
    edges["sum_kzt"] = pd.to_numeric(edges.sum_kzt, errors="raise")
    if not edges.sum_kzt.map(lambda value: math.isfinite(value) and value >= 0).all():
        raise ValueError("sum_kzt должен содержать конечные неотрицательные суммы")
    # При нескольких строках на пару складываем суммы, встречные потоки не сливаем.
    edges = edges.groupby(["src", "dst"], as_index=False, sort=True).sum_kzt.sum()
    return nodes, edges


def make_graph(nodes, edges):
    graph = nx.Graph()
    graph.add_nodes_from(nodes.gid)
    graph.add_edges_from(zip(edges.src, edges.dst))
    return graph


def select_nodes(nodes, graph, gid=None, cluster=None):
    """Расстояние — по входящим и исходящим связям; рёбра остаются направленными."""
    if gid is not None:
        if gid not in graph:
            raise ValueError(f"gid {gid} не найден")
        return set(nx.single_source_shortest_path_length(graph, gid, cutoff=2)), f"gid {gid} · 2 шага"
    if cluster is not None:
        selected = set(nodes.loc[nodes.cluster_id.eq(cluster), "gid"])
        if not selected:
            raise ValueError(f"Кластер {cluster} не найден")
        return selected, f"Кластер {cluster}"
    top = nodes.sort_values(["priority_score", "gid"], ascending=[False, True]).head(50)
    selected = set(top.gid)
    for node in top.gid:
        selected.update(graph.neighbors(node))
    return selected, f"Топ-{len(top)} + прямые соседи"


PAGE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FuryHub · Граф денежных потоков</title>
<style>{% include 'lib/vis-9.1.2/vis-network.css' %}</style>
<style>
*{box-sizing:border-box}body{margin:0;background:#0c1422;color:#e7eef8;font:14px system-ui,sans-serif}
header{padding:18px 24px;background:#111e30;border-bottom:1px solid #2a3a50}
.top{display:flex;align-items:center;gap:24px;flex-wrap:wrap}.brand{font-size:22px;font-weight:750}
.brand span{color:#78c5ff}.subtitle{color:#99acc5;margin-top:4px;font-size:12px}
form{display:flex;gap:8px;align-items:center;flex:1;min-width:310px}
input{width:280px;max-width:100%;background:#091321;border:1px solid #435571;color:#fff;padding:11px 12px;border-radius:8px;font:15px ui-monospace,monospace}
input:focus{outline:2px solid #78c5ff}button{border:1px solid #435571;border-radius:8px;padding:11px 15px;color:#e7eef8;background:#23344d;cursor:pointer;white-space:nowrap}
button:hover{background:#344b69}button.primary{background:#81c9ff;color:#081522;border-color:#81c9ff;font-weight:700}
#legend{display:flex;flex-wrap:wrap;gap:10px 20px;margin-top:16px;font-size:12px;color:#c4d0df}
.swatch{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.seed{background:#929eaf;border:3px solid #fff;width:15px;height:15px}
.bar{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;padding:12px 24px;border-bottom:1px solid #26354b;color:#a9bcd4}
#mode{color:#e7eef8;font-weight:650}main{height:calc(100vh - 195px);min-height:420px;position:relative}
#network{height:100%;width:100%}#details{position:absolute;top:16px;right:18px;width:300px;max-width:calc(100% - 36px);background:#142238f2;border:1px solid #405673;border-radius:12px;padding:18px;box-shadow:0 8px 30px #0005;pointer-events:auto}
#details[hidden]{display:none}#details h2{font:16px ui-monospace,monospace;overflow-wrap:anywhere;margin:0 0 14px}#details p{line-height:1.6;margin:8px 0}#details .muted{color:#9eafc5;font-size:12px}#details button{width:100%;margin-top:8px}
#status{min-height:20px}#status.error{color:#ff9d9f}footer{padding:8px 24px;font-size:12px;color:#99acc5;border-top:1px solid #26354b}
div.vis-tooltip{max-width:430px;white-space:pre-wrap!important;overflow-wrap:anywhere;background:#142238!important;color:#f1f6ff!important;border:1px solid #647d9c!important;border-radius:8px!important;padding:12px!important;font:13px/1.6 system-ui,sans-serif!important}
@media(max-width:850px){header{padding:12px}.top{gap:12px}form{flex-wrap:wrap}input{flex:1;min-width:150px}main{height:65vh}#details{width:260px}.bar{padding:10px 12px}}
</style>
<script>{% include 'lib/vis-9.1.2/vis-network.min.js' %}</script>
</head>
<body>
<header>
 <div class="top">
  <div><div class="brand"><span>FuryHub</span> / Граф денег</div><div class="subtitle">Направления переводов · роли · приоритет проверки</div></div>
  <form id="search-form"><label for="gid">Поиск gid</label><input id="gid" type="text" inputmode="numeric" autocomplete="off" spellcheck="false" placeholder="Введите полный gid"><button class="primary" type="submit">Найти ↗</button></form>
  <button id="reset" type="button">Исходный вид</button><button id="fit" type="button">Вся схема</button>
 </div>
 <div id="legend">
 {% for role, (label, color) in roles.items() %}<span title="{{ role }}"><i class="swatch" style="background:{{ color }}"></i>{{ label }}</span>{% endfor %}
 <span><i class="swatch seed"></i>Seed</span><span>Размер ∝ priority_score · Толщина ∝ log(1 + сумма)</span>
 </div>
</header>
<div class="bar"><div><span id="mode"></span> <span id="counts"></span></div><div id="status" role="status" aria-live="polite">Поиск доступен по всей сети, включая скрытые узлы.</div></div>
<main><div id="network" aria-label="Интерактивная схема денежных переводов"></div>
<aside id="details" hidden><h2 id="detail-gid"></h2><p id="detail-role"></p><p id="detail-score"></p><p id="detail-evidence"></p><p class="muted" id="detail-cluster"></p><button id="neighborhood">Окружение · 2 шага</button><button id="close-details">Закрыть карточку</button></aside></main>
<footer>Колесо — масштаб · Перетаскивание — перемещение · Клик — карточка · Двойной клик — окружение. Роли — гипотезы для проверки.</footer>
<script>
"use strict";
// Все идентификаторы остаются строками, включая ключи индекса.
const allNodes = {{ nodes|tojson }};
const allEdges = {{ edges|tojson }};
const initialIds = {{ initial_ids|tojson }};
const initialMode = {{ initial_mode|tojson }};
const initialFocus = {{ initial_focus|tojson }};
const byId = new Map(allNodes.map(n => [n.id, n]));
const adjacency = new Map(allNodes.map(n => [n.id, new Set()]));
for (const e of allEdges) { adjacency.get(e.from).add(e.to); adjacency.get(e.to).add(e.from); }
const nodes = new vis.DataSet();
const edges = new vis.DataSet();
const network = new vis.Network(document.getElementById('network'), {nodes, edges}, {{ options|safe }});
const status = document.getElementById('status');
let currentGid = null;
function message(text, error=false) { status.textContent=text; status.classList.toggle('error', error); }
function neighborhood(gid) {
    const selected = new Set([gid]);
    let frontier = [gid];
    for (let step=0; step<2; step++) {
        const next=[];
        for (const id of frontier) for (const peer of adjacency.get(id)) {
            if (!selected.has(peer)) { selected.add(peer); next.push(peer); }
        }
        frontier=next;
    }
    return selected;
}
function nodeForCanvas(node, position) {
    // DOM textContent не позволяет evidence интерпретироваться как HTML.
    const title=document.createElement('div'); title.textContent=node.title;
    return {...node, ...position, title};
}
function radialPositions(ids, focus) {
    // Быстрая детерминированная раскладка двух колец без браузерной физики.
    const positions=new Map([[focus, {x:0,y:0}]]);
    const direct=[...adjacency.get(focus)].filter(id => id!==focus).sort();
    const outer=[...ids].filter(id => id!==focus && !adjacency.get(focus).has(id));
    const innerRadius=Math.max(190, direct.length*12);
    for (const [ring, radius] of [[direct,innerRadius], [outer,Math.max(innerRadius+220,outer.length*10)]]) {
        ring.forEach((id,i) => {const a=2*Math.PI*i/ring.length; positions.set(id,{x:radius*Math.cos(a),y:radius*Math.sin(a)});});
    }
    return positions;
}
function showGraph(ids, mode, focus=null) {
    const selected=new Set(ids);
    const positions=focus ? radialPositions(selected,focus) : null;
    network.unselectAll(); edges.clear(); nodes.clear();
    nodes.add([...selected].map(id => nodeForCanvas(byId.get(id),positions?.get(id))));
    edges.add(allEdges.filter(e => selected.has(e.from) && selected.has(e.to)).map(e => {
        const title=document.createElement('div'); title.textContent=e.title; return {...e,title};
    }));
    document.getElementById('mode').textContent=mode;
    document.getElementById('counts').textContent=` · ${nodes.length} из ${allNodes.length} узлов · ${edges.length} рёбер`;
    document.getElementById('details').hidden=true;
    network.fit({animation:false});
}
function showDetails(gid) {
    const node=byId.get(gid); currentGid=gid;
    document.getElementById('detail-gid').textContent=`gid ${gid}`;
    document.getElementById('detail-role').textContent=`${node.role_label} (${node.role})${node.is_seed ? ' · SEED' : ''}`;
    document.getElementById('detail-score').textContent=`priority_score: ${node.priority_score.toFixed(6)}`;
    document.getElementById('detail-evidence').textContent=node.evidence;
    document.getElementById('detail-cluster').textContent=`Кластер ${node.cluster_id} · Прямых соседей: ${adjacency.get(gid).size}`;
    document.getElementById('details').hidden=false;
}
function focusNode(gid) {
    network.selectNodes([gid], true);
    network.focus(gid, {scale:1.05, animation:{duration:350,easingFunction:'easeInOutQuad'}});
    showDetails(gid);
}
function openNeighborhood(gid) {
    showGraph(neighborhood(gid),`gid ${gid} · 2 шага`,gid);
    focusNode(gid); message(`Показано окружение gid ${gid} на 2 шага.`);
}
document.getElementById('search-form').addEventListener('submit', event => {
    event.preventDefault(); const gid=document.getElementById('gid').value.trim();
    if (!gid) { message('Введите полный gid.',true); return; }
    if (!byId.has(gid)) { message(`gid ${gid} не найден в данных.`,true); return; }
    if (!nodes.get(gid)) { openNeighborhood(gid); }
    else { focusNode(gid); message(`Найден gid ${gid}. Для всех связей нажмите «Окружение · 2 шага».`); }
});
document.getElementById('reset').addEventListener('click',() => {
    showGraph(initialIds,initialMode,initialFocus);
    if(initialFocus) focusNode(initialFocus);
    message('Исходный вид восстановлен.');
});
document.getElementById('fit').addEventListener('click',() => network.fit({animation:{duration:250}}));
document.getElementById('neighborhood').addEventListener('click',() => {if(currentGid) openNeighborhood(currentGid);});
document.getElementById('close-details').addEventListener('click',() => {document.getElementById('details').hidden=true;});
network.on('click',event => {if(event.nodes.length) showDetails(event.nodes[0]);});
network.on('doubleClick',event => {if(event.nodes.length) openNeighborhood(event.nodes[0]);});
showGraph(initialIds,initialMode,initialFocus);
if(initialFocus) focusNode(initialFocus);
</script></body></html>"""


def build_html(nodes, edges, graph, selected, mode, focus=None):
    network = Network(height="100%", width="100%", directed=True,
                      bgcolor="#0c1422", font_color="#cfdded", cdn_resources="in_line")
    # Считаем только начальный подграф, один раз в Python. В браузере physics=False.
    subgraph = graph.subgraph(sorted(selected))
    positions = nx.spring_layout(subgraph, seed=42, iterations=70, weight=None,
                                 scale=max(350, math.sqrt(len(selected))*85))
    max_score = max(float(nodes.priority_score.max()), 1e-12)
    max_log_sum = max(math.log1p(float(edges.sum_kzt.max())) if len(edges) else 0, 1e-12)
    for row in nodes.itertuples(index=False):
        label, color = ROLES[row.role]
        score = float(row.priority_score)
        x, y = positions.get(row.gid, (0, 0))
        network.add_node(
            row.gid, label=row.gid, shape="dot", x=float(x), y=float(y),
            size=8 + 28*score/max_score, borderWidth=4 if row.is_seed else 1,
            borderWidthSelected=6,
            color={"background": color, "border": "#ffffff" if row.is_seed else "#34445c",
                   "highlight": {"background": color, "border": "#ffe175"},
                   "hover": {"background": color, "border": "#ffe175"}},
            title=f"gid: {row.gid}\nРоль: {label} ({row.role})\npriority_score: {score:.6f}\nevidence: {row.evidence}",
            role=row.role, role_label=label, priority_score=score,
            evidence=str(row.evidence), is_seed=bool(row.is_seed), cluster_id=row.cluster_id,
        )
    for index, row in enumerate(edges.itertuples(index=False)):
        amount = float(row.sum_kzt)
        formatted = f"{amount:,.2f}".replace(",", " ")
        network.add_edge(row.src, row.dst, id=f"e{index}", arrows="to",
                         width=0.6 + 5*math.log1p(amount)/max_log_sum,
                         title=f"{row.src} → {row.dst}\nСумма: {formatted} KZT", sum_kzt=amount)
    network.set_options(json.dumps({
        "physics": {"enabled": False},
        "layout": {"improvedLayout": False, "randomSeed": 42},
        "interaction": {"hover": True, "tooltipDelay": 100, "hideEdgesOnDrag": True,
                        "hideEdgesOnZoom": True, "keyboard": {"enabled": True}},
        "nodes": {"font": {"size": 12, "color": "#cfdded", "strokeWidth": 3,
                           "strokeColor": "#0c1422"},
                  "scaling": {"label": {"enabled": True, "drawThreshold": 9}},
                  "chosen": True},
        "edges": {"color": {"color": "#63758e", "highlight": "#ffe175", "opacity": 0.65},
                  "smooth": {"enabled": True, "type": "curvedCW", "roundness": 0.12},
                  "arrows": {"to": {"enabled": True, "scaleFactor": 0.55}}},
    }))
    network.template = network.templateEnv.from_string(PAGE)
    network.template.globals.update(roles=ROLES, initial_ids=sorted(selected),
                                    initial_mode=mode, initial_focus=focus)
    return network.generate_html(notebook=True)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, default=Path("out/nodes_roles.csv"))
    parser.add_argument("--edges", type=Path, default=Path("data/edges.parquet"))
    parser.add_argument("--output", type=Path, default=Path("out/graph.html"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gid", help="Узел и все соседи на два шага в обе стороны")
    group.add_argument("--cluster", help="Полный кластер по cluster_id")
    args = parser.parse_args(argv)
    started = perf_counter()
    try:
        nodes, edges = load_data(args.nodes, args.edges)
        graph = make_graph(nodes, edges)
        selected, mode = select_nodes(nodes, graph, args.gid, args.cluster)
        print(f"{mode}: {len(selected)} из {len(nodes)} узлов. Раскладка...", flush=True)
        html = build_html(nodes, edges, graph, selected, mode, args.gid)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(html, encoding="utf-8")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Ошибка: {error}\n")
    print(f"Готово: {args.output.resolve()} ({perf_counter() - started:.2f} с)")
    print("Откройте HTML в браузере. Поиск работает по всем gid без интернета.")


if __name__ == "__main__":
    main()
