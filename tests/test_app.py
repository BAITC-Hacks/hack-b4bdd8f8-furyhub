"""Реальный локальный путь CSV -> процесс анализа -> граф/CSV -> история заметок."""

import io
import json
from pathlib import Path
import tempfile
import unittest

from app import create_app, run_parameters


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="furyhub-app-test-")
        self.root = Path(self.directory.name)
        self.app = create_app(self.root, testing=True)
        self.client = self.app.test_client()
        self.store = self.app.extensions["workspace"]
        self.runner = self.app.extensions["job_runner"]
        self.headers = {"X-FuryHub-Token": self.client.get("/api/bootstrap").json["csrf_token"]}

    def tearDown(self):
        self.runner.shutdown(wait=True)
        self.directory.cleanup()

    def upload(self, text=None, filename="payments.csv"):
        text = text or ("sender_id,recipient_id,amount,date,tx_id\n"
                        "000100000000000000001,000100000000000000002,70000.50,2026-01-01,p1\n"
                        "000100000000000000002,000100000000000000003,40000,2026-01-02,p2\n")
        return self.client.post("/api/uploads", headers=self.headers,
                                data={"file": (io.BytesIO(text.encode("utf-8-sig")), filename)})

    def dataset(self, upload=None, **changes):
        upload = upload or self.upload().json
        body = {"upload_id": upload["upload_id"], "name": "Тестовый набор", "source": "Проверка",
                "coverage": "unknown", "currency": "KZT", "seed_gids": [],
                "mapping": upload["suggested_mapping"]}
        body.update(changes)
        return self.client.post("/api/datasets", json=body, headers=self.headers)

    def test_real_pipeline_and_case_survive_restart(self):
        upload = self.upload()
        self.assertEqual(upload.status_code, 201, upload.json)
        self.assertEqual(upload.json["preview"][0]["sender_id"], "000100000000000000001")
        dataset = self.dataset(upload.json)
        self.assertEqual(dataset.status_code, 201, dataset.json)
        data = dataset.json["dataset"]
        self.assertEqual(data["summary"]["node_count"], 3)
        self.assertNotIn("data_dir", data)
        response = self.client.post("/api/runs", headers=self.headers, json={"dataset_id": data["id"]})
        self.assertEqual(response.status_code, 202, response.json)
        run_id = response.json["run"]["id"]
        self.runner.submit(run_id).result(timeout=45)
        run = self.client.get(f"/api/runs/{run_id}").json["run"]
        log = (self.store.run_dir(run_id) / "stderr.log").read_text(encoding="utf-8")
        self.assertEqual(run["status"], "completed", log)
        self.assertEqual(run["progress"], 100)
        graph = self.client.get(f"/runs/{run_id}/graph")
        self.assertEqual(graph.status_code, 200)
        self.assertIn("полнота исходящих неизвестна", graph.get_data(as_text=True))
        self.assertIn("furyhub:node-selected", graph.get_data(as_text=True))
        graph.close()
        for name in ("nodes_roles.csv", "clusters.csv", "top_nodes.csv", "manifest.json"):
            download = self.client.get(f"/api/runs/{run_id}/files/{name}")
            self.assertEqual(download.status_code, 200)
            self.assertIn("attachment", download.headers["Content-Disposition"])
            if name == "manifest.json":
                manifest = json.loads(download.get_data())
                self.assertEqual(manifest["parameters"], run["parameters"])
                self.assertTrue(manifest["parameters"]["offline"])
            download.close()
        nodes = self.client.get(f"/api/runs/{run_id}/nodes?q=000100000000000000003").json
        self.assertEqual(nodes["total"], 1)
        self.assertEqual(nodes["nodes"][0]["role"], "peripheral")
        gid = nodes["nodes"][0]["gid"]
        endpoint = f"/api/runs/{run_id}/cases/{gid}"
        self.assertEqual(self.client.get(endpoint).json["case"]["status"], "new")
        note = "Нужно запросить выписку <script>alert(1)</script>"
        saved = self.client.put(endpoint, headers=self.headers, json={"status": "needs_info", "note": note})
        self.assertEqual(saved.status_code, 200, saved.json)
        self.assertEqual(saved.json["case"]["note"], note)
        self.client.put(endpoint, headers=self.headers, json={"status": "in_progress", "note": "Выписка получена"})
        self.assertEqual(len(self.client.get(endpoint).json["events"]), 2)
        self.assertEqual(self.client.get(f"/api/cases?run_id={run_id}").json["cases"][0]["gid"], gid)
        self.assertEqual(self.client.get(f"/api/runs/{run_id}/cases/absent").status_code, 404)
        self.assertEqual(self.client.get(f"/api/runs/{run_id}/files/parameters.json").status_code, 404)
        self.runner.shutdown(wait=True)
        restarted = create_app(self.root, testing=True)
        try:
            client = restarted.test_client()
            self.assertEqual(client.get(endpoint).json["case"]["status"], "in_progress")
            self.assertEqual(len(client.get(endpoint).json["events"]), 2)
            self.assertEqual(client.get("/api/runs").json["runs"][0]["status"], "completed")
        finally:
            restarted.extensions["job_runner"].shutdown(wait=True)

    def test_duplicate_dataset_reuses_version(self):
        uploaded = self.upload().json
        first, second = self.dataset(uploaded), self.dataset(uploaded)
        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json["dataset"]["id"], second.json["dataset"]["id"])
        self.assertTrue(second.json["dataset"]["reused"])
        self.assertEqual(len(self.client.get("/api/datasets").json["datasets"]), 1)
        self.assertEqual(len(list((self.root / "datasets").iterdir())), 1)

    def test_validation_returns_rows_and_leaves_no_dataset(self):
        uploaded = self.upload("src,dst,sum_kzt,date\na,b,-7,2026-01-01\n").json
        response = self.dataset(uploaded)
        self.assertEqual(response.status_code, 400, response.json)
        self.assertEqual(response.json["errors"][0]["row"], 2)
        self.assertEqual(response.json["errors"][0]["column"], "sum_kzt")
        self.assertEqual(self.store.list_datasets(), [])
        self.assertEqual(list((self.root / "datasets").iterdir()), [])

    def test_local_origin_host_and_token_boundaries(self):
        self.assertEqual(self.client.post("/api/demo").status_code, 403)
        self.assertEqual(self.client.post("/api/demo", headers={**self.headers, "Origin": "https://example.org"}).status_code, 403)
        self.assertEqual(self.client.get("/api/bootstrap", headers={"Host": "evil.example"}).status_code, 400)
        self.assertEqual(self.client.get("/api/bootstrap", environ_overrides={"REMOTE_ADDR": "192.168.1.10"}).status_code, 403)
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/api/template.csv").status_code, 200)

    def test_invalid_file_and_size_limit(self):
        self.assertEqual(self.upload(filename="script.html").status_code, 400)
        self.assertEqual(self.upload("broken", filename="invalid.parquet").status_code, 400)
        self.assertEqual(list((self.root / "uploads").iterdir()), [])
        self.app.config["MAX_CONTENT_LENGTH"] = 64
        self.assertEqual(self.upload().status_code, 413)

    def test_failed_worker_does_not_publish_partial_results(self):
        dataset = self.dataset().json["dataset"]
        params = {**run_parameters(), "pipeline_version": "changed"}
        run = self.store.create_run(dataset["id"], params)
        self.runner.submit(run["id"]).result(timeout=45)
        self.assertEqual(self.store.get_run(run["id"])["status"], "failed")
        self.assertEqual(self.client.get(f"/runs/{run['id']}/graph").status_code, 400)
        self.assertEqual(self.client.get(f"/api/runs/{run['id']}/files/nodes_roles.csv").status_code, 400)

    def test_second_server_cannot_mark_active_runs_interrupted(self):
        first = create_app(self.root / "locked")
        try:
            store = first.extensions["workspace"]
            dataset = store.create_dataset("a" * 32, {"name": "Lease test"}, {},
                                           store.root / "datasets" / ("a" * 32), "lease-test")
            run = store.create_run(dataset["id"], {})
            store.update_run(run["id"], status="running")
            with self.assertRaises(RuntimeError):
                create_app(store.root)
            self.assertEqual(store.get_run(run["id"])["status"], "running")
        finally:
            first.extensions["close"]()
        restarted = create_app(store.root)
        try:
            self.assertEqual(restarted.extensions["workspace"].get_run(run["id"])["status"], "failed")
        finally:
            restarted.extensions["close"]()


if __name__ == "__main__":
    unittest.main()
