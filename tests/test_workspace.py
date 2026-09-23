"""Хранение истории и очередь расчётов без сети и настоящего worker."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sqlite3
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from workspace import JobRunner, Workspace, WorkspaceLock, INTERRUPTED_MESSAGE


class WorkspaceFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "workspace"
        self.store = Workspace(self.root)

    def dataset(self, fingerprint=None):
        dataset_id = uuid4().hex
        data_dir = self.root / "datasets" / dataset_id
        data_dir.mkdir()
        return self.store.create_dataset(dataset_id, {"name": "Моя выборка"},
                                         {"nodes": 12}, data_dir, fingerprint or dataset_id)


class WorkspaceTests(WorkspaceFixture):
    def test_workspace_lock_rejects_second_instance_and_can_be_reacquired(self):
        first, second = WorkspaceLock(self.root), WorkspaceLock(self.root)
        second.close()
        with first:
            with self.assertRaisesRegex(RuntimeError, "уже открыт"):
                second.__enter__()
            # Ошибка второго владельца не должна освобождать первую блокировку.
            second.close()
            with self.assertRaises(RuntimeError):
                WorkspaceLock(self.root).__enter__()
        self.assertTrue((self.root / ".app.lock").exists())
        first.close()
        with second:
            with self.assertRaises(RuntimeError):
                first.__enter__()
        with first:
            pass

    def test_workspace_lock_releases_on_exception_without_deleting_file(self):
        with self.assertRaisesRegex(ValueError, "test failure"):
            with WorkspaceLock(self.root):
                raise ValueError("test failure")
        self.assertTrue((self.root / ".app.lock").exists())
        with WorkspaceLock(self.root):
            pass

    def test_upload_dataset_run_persist_and_snapshots_are_detached(self):
        upload_id = uuid4().hex
        path = self.root / "uploads" / f"{upload_id}.csv"
        path.write_text("gid\n100000000000000001\n", encoding="utf-8")
        inspection = {"columns": ["gid"], "rows": 1}
        upload = self.store.register_upload(upload_id, "Клиенты.csv", path, inspection, "a" * 64)
        inspection["columns"].append("changed")
        dataset = self.dataset()
        parameters = {"thresholds": {"in_deg": 5}, "offline": True}
        run = self.store.create_run(dataset["id"], parameters)
        parameters["thresholds"]["in_deg"] = 100
        reopened = Workspace(self.root)
        self.assertEqual(reopened.get_upload(upload_id), upload)
        self.assertEqual(reopened.get_upload(upload_id)["inspection"]["columns"], ["gid"])
        self.assertEqual(reopened.get_dataset(dataset["id"])["summary"], {"nodes": 12})
        saved = reopened.get_run(run["id"])
        self.assertEqual(saved["parameters"]["thresholds"]["in_deg"], 5)
        self.assertEqual(saved["status"], "queued")
        self.assertEqual(saved["progress"], 0)
        self.assertIsNone(saved["finished_at"])
        self.assertEqual(len(reopened.list_runs()), 1)
        with reopened._connection() as connection:
            self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "wal")
            self.assertEqual(connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)

    def test_identical_dataset_reuses_original_without_replacing_metadata(self):
        original = self.dataset("same-input")
        another_id = uuid4().hex
        repeated = self.store.create_dataset(another_id, {"name": "Другое имя"}, {"nodes": 99},
                                              self.root / "datasets" / another_id, "same-input")
        self.assertFalse(original["reused"])
        self.assertTrue(repeated["reused"])
        self.assertEqual(repeated["id"], original["id"])
        self.assertEqual(repeated["summary"], {"nodes": 12})
        self.assertEqual(len(self.store.list_datasets()), 1)
        self.assertIsNone(self.store.get_dataset(another_id))

    def test_parallel_duplicate_run_creation_is_atomic(self):
        dataset = self.dataset()
        with ThreadPoolExecutor(max_workers=8) as pool:
            runs = list(pool.map(lambda _: self.store.create_run(dataset["id"], {"offline": True}), range(16)))
        self.assertEqual(len({run["id"] for run in runs}), 1)
        self.assertEqual(sum(not run["reused"] for run in runs), 1)
        first = runs[0]
        self.store.update_run(first["id"], status="completed", progress=100, phase="ready")
        following = self.store.create_run(dataset["id"], {"offline": True})
        self.assertFalse(following["reused"])
        self.assertNotEqual(first["id"], following["id"])
        changed = self.store.create_run(dataset["id"], {"offline": True, "thresholds": {"n": 5}})
        self.assertNotEqual(following["id"], changed["id"])
        self.assertEqual(len(self.store.list_runs(dataset["id"])), 3)

    def test_case_history_preserves_exact_gid_and_is_immutable(self):
        dataset = self.dataset()
        run = self.store.create_run(dataset["id"], {})
        gid = "100000000343175100"
        self.assertEqual(self.store.get_case(run["id"], gid)["status"], "new")
        self.assertEqual(self.store.list_cases(), [])
        first = self.store.save_case(run["id"], gid, "in_progress", "Запросить выписку")
        second = self.store.save_case(run["id"], gid, "needs_info", "Нужны данные на начало периода")
        unchanged = self.store.save_case(run["id"], gid, "needs_info", "Нужны данные на начало периода")
        self.assertEqual(second, unchanged)
        self.assertEqual(second["created_at"], first["created_at"])
        reopened = Workspace(self.root)
        self.assertEqual(reopened.get_case(run["id"], gid), second)
        events = reopened.list_case_events(run["id"], gid)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["previous"], {"status": "new", "note": ""})
        self.assertEqual(events[1]["previous"], {"status": "in_progress", "note": "Запросить выписку"})
        self.assertEqual(events[1]["note"], second["note"])
        self.assertTrue(all(event["gid"] == gid for event in events))
        listed = reopened.list_cases(run["id"])
        self.assertEqual(listed[0]["dataset_id"], dataset["id"])
        self.assertEqual(listed[0]["dataset_name"], "Моя выборка")
        self.assertEqual(listed[0]["run_name"], run["name"])
        with reopened._connection(write=True) as connection:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute("UPDATE case_events SET note='modified'")
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute("DELETE FROM case_events")

    def test_notes_scoped_by_run_and_bound_sql_handles_quotes(self):
        dataset = self.dataset()
        first = self.store.create_run(dataset["id"], {"v": 1})
        second = self.store.create_run(dataset["id"], {"v": 2})
        note = "'); DROP TABLE runs; --\nПроверка '<script>'"
        self.store.save_case(first["id"], "100000000000000001", "closed", note)
        self.assertEqual(self.store.get_case(first["id"], "100000000000000001")["note"], note)
        self.assertEqual(self.store.get_case(second["id"], "100000000000000001")["status"], "new")
        self.assertEqual(len(self.store.list_runs()), 2)
        with self.assertRaises(ValueError):
            self.store.save_case(first["id"], "1", "guilty", "")
        with self.assertRaises(ValueError):
            self.store.save_case(first["id"], "1", "new", "x" * 10001)
        self.store.save_case(first["id"], "1", "new", "x" * 10000)

    def test_recovery_marks_only_unfinished_runs(self):
        dataset = self.dataset()
        queued, running, finished = [self.store.create_run(dataset["id"], {"n": n}) for n in range(3)]
        self.store.update_run(running["id"], status="running", progress=10, phase="calculation")
        self.store.update_run(finished["id"], status="completed", progress=100, phase="ready")
        reopened = Workspace(self.root)
        self.assertEqual(reopened.recover_interrupted_runs(), 2)
        for run in (queued, running):
            saved = reopened.get_run(run["id"])
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["error"], INTERRUPTED_MESSAGE)
            self.assertIsNotNone(saved["finished_at"])
        self.assertEqual(reopened.get_run(finished["id"])["status"], "completed")
        self.assertEqual(reopened.recover_interrupted_runs(), 0)

    def test_invalid_identifiers_paths_and_updates_fail_before_writing(self):
        for bad in ("../out", "a" * 31, "A" * 32, 123, "a" * 32 + "/x"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.store.run_dir(bad)
        with self.assertRaises(ValueError):
            self.store.create_dataset(uuid4().hex, {}, {}, self.root.parent, "bad")
        with self.assertRaises(ValueError):
            self.store.register_upload(uuid4().hex, "file.csv", self.root.parent / "outside.csv", {}, "a" * 64)
        with self.assertRaises(ValueError):
            self.store.create_run(uuid4().hex, {})
        dataset = self.dataset()
        run = self.store.create_run(dataset["id"], {})
        for fields in ({"dataset_id": uuid4().hex}, {"status": "unknown"}, {"progress": 101}, {"progress": float("nan")}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.store.update_run(run["id"], **fields)
        with self.assertRaises(ValueError):
            self.store.create_run(dataset["id"], {"invalid": float("nan")})


class JobRunnerTests(WorkspaceFixture):
    def runner(self):
        runner = JobRunner(self.store)
        self.addCleanup(runner.shutdown)
        return runner

    def test_worker_uses_absolute_paths_snapshot_and_logs(self):
        dataset = self.dataset()
        parameters = {"thresholds": {"in_deg": 5}, "offline": True}
        run = self.store.create_run(dataset["id"], parameters)
        seen = []

        def worker(command, **kwargs):
            seen.append(self.store.get_run(run["id"]))
            self.assertEqual(command[:2], [sys.executable, "-B"])
            self.assertTrue(Path(command[2]).is_absolute())
            self.assertEqual(Path(command[2]).name, "app_worker.py")
            self.assertEqual(command[command.index("--data") + 1], dataset["data_dir"])
            snapshot = Path(command[command.index("--parameters") + 1])
            self.assertEqual(json.loads(snapshot.read_text(encoding="utf-8")), parameters)
            self.assertEqual(kwargs["timeout"], 300)
            kwargs["stdout"].write("Расчёт выполнен\n")
            kwargs["stderr"].write("Диагностика\n")
            return SimpleNamespace(returncode=0)

        with patch("workspace.subprocess.run", side_effect=worker) as process:
            runner = self.runner()
            future = runner.submit(run["id"])
            self.assertIs(runner.submit(run["id"]), future)
            runner.shutdown(wait=True)
            future.result()
        process.assert_called_once()
        self.assertEqual(seen[0]["status"], "running")
        self.assertEqual(seen[0]["phase"], "calculation")
        saved = self.store.get_run(run["id"])
        self.assertEqual((saved["status"], saved["progress"], saved["phase"]), ("completed", 100, "ready"))
        self.assertIsNotNone(saved["finished_at"])
        self.assertIn("Расчёт", (self.store.run_dir(run["id"]) / "stdout.log").read_text(encoding="utf-8"))
        with self.assertRaises(RuntimeError):
            runner.submit(run["id"])

    def test_worker_failures_and_timeout_are_persisted(self):
        dataset = self.dataset()
        for index, result in enumerate((SimpleNamespace(returncode=7), subprocess.TimeoutExpired("worker", 300), OSError("secret/path"))):
            with self.subTest(index=index):
                run = self.store.create_run(dataset["id"], {"index": index})
                with patch("workspace.subprocess.run", side_effect=result if isinstance(result, Exception) else None,
                           return_value=result):
                    runner = self.runner()
                    runner.submit(run["id"]).result(timeout=5)
                    runner.shutdown()
                saved = self.store.get_run(run["id"])
                self.assertEqual(saved["status"], "failed")
                self.assertIsNotNone(saved["finished_at"])
                self.assertNotIn("secret", saved["error"])
                if index == 1:
                    self.assertIn("5 минут", saved["error"])

    def test_existing_run_files_are_never_overwritten(self):
        dataset = self.dataset()
        run = self.store.create_run(dataset["id"], {})
        directory = self.store.run_dir(run["id"])
        directory.mkdir()
        marker = directory / "nodes_roles.csv"
        marker.write_text("Сохранённый результат", encoding="utf-8")
        with patch("workspace.subprocess.run") as process:
            self.runner().submit(run["id"]).result(timeout=5)
        process.assert_not_called()
        self.assertEqual(marker.read_text(encoding="utf-8"), "Сохранённый результат")
        self.assertEqual(self.store.get_run(run["id"])["status"], "failed")

    def test_thread_pool_runs_one_job_at_a_time(self):
        dataset = self.dataset()
        first, second = [self.store.create_run(dataset["id"], {"n": n}) for n in range(2)]
        entered, release = threading.Event(), threading.Event()
        counter = 0

        def worker(*args, **kwargs):
            nonlocal counter
            counter += 1
            if counter == 1:
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("Test release signal missing")
            return SimpleNamespace(returncode=0)

        with patch("workspace.subprocess.run", side_effect=worker):
            runner = self.runner()
            try:
                one = runner.submit(first["id"])
                self.assertTrue(entered.wait(5))
                two = runner.submit(second["id"])
                self.assertEqual(self.store.get_run(second["id"])["status"], "queued")
                self.assertEqual(counter, 1)
                release.set()
                one.result(timeout=5)
                two.result(timeout=5)
            finally:
                release.set()
                runner.shutdown()
        self.assertEqual(counter, 2)


if __name__ == "__main__":
    unittest.main()
