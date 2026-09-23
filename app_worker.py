#!/usr/bin/env python3
"""Изолированный офлайн-процесс одного анализа. Запускается JobRunner."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
from time import perf_counter

import run as pipeline
from starter.starter import load
from viz import build_data, build_html, collect_hints, load_data, load_top


def analyse(data_dir, out_dir, parameters):
    from app import pipeline_version

    if parameters.get("offline") is not True:
        raise ValueError("Рабочее место выполняет только офлайн-анализ.")
    if parameters.get("pipeline_version") != pipeline_version():
        raise ValueError("Код изменился после создания запуска. Создайте новый запуск.")
    for key, attribute in (("thresholds", "THRESHOLDS"), ("priority_weights", "PRIORITY_WEIGHTS"),
                           ("role_weights", "ROLE_PRIORITY_WEIGHTS")):
        values = parameters[key]
        if set(values) != set(getattr(pipeline, attribute)):
            raise ValueError("Снимок правил несовместим с кодом.")
        setattr(pipeline, attribute, values)
    for key, attribute in (("depth_penalty", "DEPTH_PENALTY"), ("random_seed", "RANDOM_SEED"),
                           ("betweenness_samples", "BETWEENNESS_SAMPLES"), ("top_n", "TOP_N")):
        setattr(pipeline, attribute, parameters[key])
    started = perf_counter()
    edges, nodes, tx = load(data_dir)
    pipeline.validate_inputs(edges, nodes, tx)
    print("Расчёт метрик, ролей и кластеров", flush=True)
    result = pipeline.add_priority(pipeline.assign_roles(pipeline.compute_features(edges, nodes)))
    result["evidence"] = [pipeline.evidence(row) for row in result.itertuples(index=False)]
    n_clusters = pipeline.write_outputs(result, edges, out_dir, offline=True)
    print("Подготовка интерактивного графа", flush=True)
    visual_nodes, visual_edges = load_data(out_dir / "nodes_roles.csv", data_dir / "edges.parquet")
    top = load_top(out_dir / "top_nodes.csv", visual_nodes)
    data = build_data(visual_nodes, visual_edges, top)
    data["embedded"] = True
    data["hints"] = collect_hints(visual_nodes, visual_edges, top, out_dir, offline=True)
    (out_dir / "graph.html").write_text(build_html(data), encoding="utf-8")
    packages = {name: importlib.metadata.version(name)
                for name in ("pandas", "networkx", "numpy", "scipy", "pyarrow", "pyvis")}
    metadata_path = data_dir / "metadata.json"
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(),
                "parameters": parameters, "python": sys.version.split()[0], "packages": packages,
                "dataset": json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {},
                "results": {"nodes": len(result), "clusters": n_clusters, "transactions": len(tx)},
                "duration_seconds": round(perf_counter() - started, 3)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Готово: {len(result)} узлов, {n_clusters} кластеров", flush=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    analyse(args.data, args.out, json.loads(args.parameters.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
