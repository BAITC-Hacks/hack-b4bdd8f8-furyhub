#!/usr/bin/env python3
"""Локальное рабочее место аналитика: python app.py (http://127.0.0.1:8765)."""

import argparse
import atexit
import hashlib
import io
import json
from pathlib import Path
import secrets
import shutil
from urllib.parse import urlsplit
import uuid

from flask import Flask, jsonify, render_template, request, send_file
import pandas as pd
from werkzeug.exceptions import HTTPException

from ingestion import ImportValidationError, build_dataset, inspect_upload
import run as pipeline
from workspace import JobRunner, Workspace, WorkspaceLock

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD = 32 * 1024 * 1024
RESULT_FILES = {"nodes_roles.csv", "clusters.csv", "top_nodes.csv", "manifest.json"}
PIPELINE_FILES = ("run.py", "viz.py", "graph_layout.py", "llm_layer.py",
                  "app_worker.py", "starter/starter.py", "constraints.txt")


def pipeline_version():
    digest = hashlib.sha256()
    for name in PIPELINE_FILES:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def run_parameters():
    return {"thresholds": pipeline.THRESHOLDS.copy(),
            "priority_weights": pipeline.PRIORITY_WEIGHTS.copy(),
            "role_weights": pipeline.ROLE_PRIORITY_WEIGHTS.copy(),
            "depth_penalty": pipeline.DEPTH_PENALTY,
            "random_seed": pipeline.RANDOM_SEED,
            "betweenness_samples": pipeline.BETWEENNESS_SAMPLES,
            "top_n": pipeline.TOP_N, "pipeline_version": pipeline_version(), "offline": True}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def public_record(record):
    """Абсолютные пути сервера не являются частью клиентского API."""
    return {key: value for key, value in record.items()
            if key not in {"path", "data_dir", "run_dir", "log_path"}}


def create_app(workspace_root=None, *, testing=False):
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=MAX_UPLOAD,
                      MAX_FORM_MEMORY_SIZE=512 * 1024,
                      TRUSTED_HOSTS=["localhost", "127.0.0.1", "[::1]"],
                      TESTING=testing)
    app.json.ensure_ascii = False
    workspace_root = Path(workspace_root or ROOT / ".workspace").resolve()
    lease = None if testing else WorkspaceLock(workspace_root)
    if lease is not None:
        lease.__enter__()
    try:
        store = Workspace(workspace_root)
        store.recover_interrupted_runs()
    except Exception:
        if lease is not None:
            lease.close()
        raise
    runner = JobRunner(store)
    token = secrets.token_urlsafe(32)
    app.extensions["workspace"] = store
    app.extensions["job_runner"] = runner
    def close():
        try:
            runner.shutdown(wait=True)
        finally:
            if lease is not None:
                lease.close()

    app.extensions["close"] = close
    atexit.register(close)

    @app.before_request
    def local_request():
        if request.remote_addr not in ("127.0.0.1", "::1", None):
            return jsonify(error="Приложение доступно только с этого компьютера."), 403
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            origin = request.headers.get("Origin")
            if origin:
                parsed = urlsplit(origin)
                if parsed.scheme != request.scheme or parsed.netloc != request.host:
                    return jsonify(error="Запрос с другого сайта отклонён."), 403
            if not secrets.compare_digest(request.headers.get("X-FuryHub-Token", ""), token):
                return jsonify(error="Обновите страницу: токен сессии недействителен."), 403

    @app.after_request
    def response_headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self' data:; "
            "connect-src 'self'; frame-src 'self'; frame-ancestors 'self'; "
            "object-src 'none'; base-uri 'none'; form-action 'self'"
        )
        return response

    @app.errorhandler(ImportValidationError)
    def import_error(error):
        return jsonify(error=str(error), errors=error.errors), 400

    @app.errorhandler(ValueError)
    def invalid_value(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(KeyError)
    def not_found(_error):
        return jsonify(error="Запись не найдена."), 404

    @app.errorhandler(HTTPException)
    def http_error(error):
        messages = {400: "Некорректный запрос.", 404: "Страница не найдена.",
                    405: "Метод не поддерживается.", 413: "Файл больше лимита 32 МиБ.",
                    415: "Ожидается JSON-запрос."}
        return jsonify(error=messages.get(error.code, "Запрос отклонён.")), error.code

    def json_body():
        value = request.get_json()
        if not isinstance(value, dict):
            raise ValueError("Ожидается JSON-объект.")
        return value

    def required_run(run_id, *, complete=False):
        result = store.get_run(run_id)
        if not result:
            raise KeyError(run_id)
        if complete and result["status"] != "completed":
            raise ValueError("Результат будет доступен после успешного завершения анализа.")
        return result

    def nodes_for_run(run_id):
        required_run(run_id, complete=True)
        return pd.read_csv(store.run_dir(run_id) / "nodes_roles.csv",
                           dtype={"gid": str, "cluster_id": str}, keep_default_na=False)

    def require_node(run_id, gid):
        frame = nodes_for_run(run_id)
        if not frame.gid.eq(gid).any():
            raise KeyError(gid)

    def schedule(dataset_id):
        if not store.get_dataset(dataset_id):
            raise KeyError(dataset_id)
        result = store.create_run(dataset_id, run_parameters())
        if not result.get("reused"):
            runner.submit(result["id"])
        return public_record(result)

    @app.get("/")
    def index():
        return render_template("workbench.html")

    @app.get("/api/bootstrap")
    def bootstrap():
        return jsonify(csrf_token=token, limits={"upload_mb": 32, "max_rows": 100000,
                       "max_nodes": 10000}, formats=["csv", "parquet"])

    @app.get("/api/template.csv")
    def template():
        content = "src,dst,sum_kzt,date\n1001,1002,15000.50,2026-01-01\n1002,1003,12000,2026-01-02\n"
        return send_file(io.BytesIO(content.encode("utf-8-sig")), mimetype="text/csv",
                         as_attachment=True, download_name="transactions_template.csv")

    @app.post("/api/uploads")
    def upload():
        item = request.files.get("file")
        if item is None or not item.filename:
            raise ValueError("Выберите CSV или Parquet-файл.")
        original_name = item.filename.replace("\\", "/").split("/")[-1][:240]
        extension = Path(original_name).suffix.lower()
        if extension not in {".csv", ".parquet"}:
            raise ValueError("Поддерживаются CSV и Parquet. XLSX пока не поддерживается.")
        upload_id = uuid.uuid4().hex
        directory = workspace_root / "uploads"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (upload_id + extension)
        try:
            item.save(path)
            inspection = inspect_upload(path, original_name)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            store.register_upload(upload_id, original_name, path, inspection, digest)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return jsonify(upload_id=upload_id, filename=original_name, sha256=digest, **inspection), 201

    @app.get("/api/datasets")
    def datasets():
        return jsonify(datasets=[public_record(row) for row in store.list_datasets()])

    @app.post("/api/datasets")
    def dataset_create():
        body = json_body()
        upload_row = store.get_upload(body.get("upload_id", ""))
        if not upload_row:
            raise KeyError("upload")
        metadata = {key: body[key] for key in ("name", "source", "coverage", "currency",
                    "seed_gids", "period_from", "period_to") if key in body}
        mapping = body.get("mapping")
        if not isinstance(mapping, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                                for k, v in mapping.items()):
            raise ValueError("Укажите соответствие колонок.")
        dataset_id = uuid.uuid4().hex
        data_dir = workspace_root / "datasets" / dataset_id
        try:
            summary = build_dataset(Path(upload_row["path"]), mapping, metadata, data_dir)
            document = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
            metadata = {key: document[key] for key in ("name", "source", "coverage", "currency",
                        "seed_gids", "period_from", "period_to")}
            checksum = fingerprint({"sha256": document["source_sha256"], "mapping": document["mapping"],
                                    "metadata": metadata})
            metadata["filename"] = upload_row["original_name"]
            result = store.create_dataset(dataset_id, metadata, summary, data_dir, checksum)
            if result.get("reused"):
                if data_dir.resolve().parent == (workspace_root / "datasets").resolve():
                    shutil.rmtree(data_dir)
        except Exception:
            # Только каталог с созданным нами UUID, никогда путь из пользовательского ввода.
            if data_dir.is_dir() and data_dir.resolve().parent == (workspace_root / "datasets").resolve():
                shutil.rmtree(data_dir)
            raise
        return jsonify(dataset=public_record(result)), 200 if result.get("reused") else 201

    @app.get("/api/runs")
    def runs():
        return jsonify(runs=[public_record(row) for row in store.list_runs()])

    @app.post("/api/runs")
    def run_create():
        return jsonify(run=schedule(json_body().get("dataset_id", ""))), 202

    @app.get("/api/runs/<run_id>")
    def run_detail(run_id):
        return jsonify(run=public_record(required_run(run_id)))

    @app.get("/runs/<run_id>/graph")
    def graph(run_id):
        required_run(run_id, complete=True)
        return send_file(store.run_dir(run_id) / "graph.html", mimetype="text/html")

    @app.get("/api/runs/<run_id>/files/<name>")
    def result_file(run_id, name):
        required_run(run_id, complete=True)
        if name not in RESULT_FILES:
            raise KeyError(name)
        return send_file(store.run_dir(run_id) / name, as_attachment=True, download_name=name)

    @app.get("/api/runs/<run_id>/nodes")
    def nodes_search(run_id):
        frame = nodes_for_run(run_id)
        query = request.args.get("q", "").strip()[:256]
        if query:
            frame = frame.loc[frame.gid.str.contains(query, regex=False)]
        limit = max(1, min(100, request.args.get("limit", 50, type=int)))
        frame = frame.sort_values(["priority_score", "gid"], ascending=[False, True])
        columns = ["gid", "role", "priority_score", "evidence"]
        return jsonify(nodes=frame.head(limit)[columns].to_dict("records"), total=len(frame))

    @app.get("/api/cases")
    def cases():
        run_id = request.args.get("run_id") or None
        if run_id:
            required_run(run_id)
        return jsonify(cases=store.list_cases(run_id))

    @app.route("/api/runs/<run_id>/cases/<path:gid>", methods=["GET", "PUT"])
    def node_case(run_id, gid):
        require_node(run_id, gid)
        if request.method == "PUT":
            body = json_body()
            row = store.save_case(run_id, gid, body.get("status"), body.get("note"))
        else:
            row = store.get_case(run_id, gid)
        return jsonify(case=row, events=store.list_case_events(run_id, gid))

    @app.post("/api/demo")
    def demo():
        source = ROOT / "data"
        names = ("nodes.parquet", "edges.parquet", "transactions.parquet")
        if not all((source / name).is_file() for name in names):
            raise ValueError("Исходный демонстрационный набор отсутствует в data/.")
        checksum = fingerprint({name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                for name in names})
        existing = next((row for row in store.list_datasets() if row["fingerprint"] == checksum), None)
        if existing:
            dataset = existing
        else:
            dataset_id = uuid.uuid4().hex
            data_dir = workspace_root / "datasets" / dataset_id
            data_dir.mkdir(parents=True)
            for name in names:
                shutil.copy2(source / name, data_dir / name)
            nodes, edges, tx = (pd.read_parquet(data_dir / name) for name in names)
            metadata = {"name": "Граф денег · исходный кейс", "source": "Данные хакатона",
                        "currency": "KZT", "coverage": "unknown", "kind": "demo"}
            summary = {"row_count": len(tx), "node_count": len(nodes), "edge_count": len(edges),
                       "seed_count": int(nodes.is_seed.sum()), "total_kzt": float(tx.sum_kzt.sum()),
                       "period_from": str(pd.to_datetime(tx.date).min().date()),
                       "period_to": str(pd.to_datetime(tx.date).max().date()),
                       "currency": "KZT", "coverage": "unknown",
                       "warnings": ["Исходный кейс: входящие seed занижены; обход ограничен глубиной 4."]}
            (data_dir / "metadata.json").write_text(json.dumps({"metadata": metadata, "summary": summary},
                                                  ensure_ascii=False, indent=2), encoding="utf-8")
            dataset = store.create_dataset(dataset_id, metadata, summary, data_dir, checksum)
        return jsonify(dataset=public_record(dataset), run=schedule(dataset["id"])), 202

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workspace", type=Path, default=ROOT / ".workspace")
    args = parser.parse_args()
    try:
        app = create_app(args.workspace)
    except RuntimeError as error:
        parser.exit(1, f"{error}\n")
    try:
        app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
    finally:
        app.extensions["close"]()


if __name__ == "__main__":
    main()
