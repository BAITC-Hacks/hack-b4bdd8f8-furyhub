"""Локальные наборы данных, история расчётов и заметки одного аналитика."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import threading
from uuid import uuid4


CASE_STATUSES = frozenset({"new", "in_progress", "needs_info", "closed"})
RUN_STATUSES = frozenset({"queued", "running", "completed", "failed"})
MAX_NOTE_LENGTH = 10_000
RUN_TIMEOUT_SECONDS = 300
INTERRUPTED_MESSAGE = "Расчёт прерван при остановке приложения. Запустите его повторно."


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{32}", value):
        raise ValueError("Идентификатор должен содержать 32 строчные шестнадцатеричные цифры")
    return value


def _gid(value):
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError("gid должен быть непустой строкой")
    return value


def _json(value):
    if not isinstance(value, dict):
        raise ValueError("Ожидается словарь параметров")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


class WorkspaceLock:
    """Один работающий экземпляр приложения на каталог, до восстановления задач."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self._handle = None
        self._release = None

    def __enter__(self):
        if self._handle is not None:
            raise RuntimeError("Блокировка рабочего каталога уже получена этим экземпляром")
        self.root.mkdir(parents=True, exist_ok=True)
        path = (self.root / ".app.lock").resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Файл блокировки должен находиться внутри рабочего каталога")
        handle = path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                if handle.seek(0, os.SEEK_END) == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)

                def acquire():
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

                def release():
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                def acquire():
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                def release():
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            try:
                acquire()
            except OSError:
                raise RuntimeError(
                    "Рабочий каталог уже открыт в другом экземпляре приложения. "
                    "Закройте его или выберите другой каталог."
                ) from None
        except BaseException:
            handle.close()
            raise
        self._handle, self._release = handle, release
        return self

    def close(self):
        handle, release = self._handle, self._release
        self._handle = self._release = None
        if handle is not None:
            try:
                release()
            finally:
                # Не удаляем lock-файл: следующий экземпляр блокирует тот же inode.
                handle.close()

    def __exit__(self, *_exc):
        self.close()


class Workspace:
    """SQLite в личном рабочем каталоге; каждый метод открывает своё соединение."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in ("uploads", "datasets", "runs"):
            self._owned_path(self.root / directory).mkdir(parents=True, exist_ok=True)
        self.db_path = self._owned_path(self.root / "workspace.sqlite3")
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS uploads (
                    id TEXT PRIMARY KEY, original_name TEXT NOT NULL,
                    path TEXT NOT NULL, inspection TEXT NOT NULL,
                    sha256 TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, metadata TEXT NOT NULL,
                    summary TEXT NOT NULL, data_dir TEXT NOT NULL,
                    fingerprint TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL REFERENCES datasets(id),
                    name TEXT NOT NULL, parameters TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed')),
                    created_at TEXT NOT NULL, finished_at TEXT,
                    progress REAL NOT NULL DEFAULT 0 CHECK(progress >= 0 AND progress <= 100),
                    phase TEXT NOT NULL DEFAULT 'queued', error TEXT
                );
                CREATE INDEX IF NOT EXISTS runs_dataset ON runs(dataset_id, created_at);
                CREATE TABLE IF NOT EXISTS cases (
                    run_id TEXT NOT NULL REFERENCES runs(id), gid TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('new','in_progress','needs_info','closed')),
                    note TEXT NOT NULL CHECK(length(note) <= 10000),
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY(run_id, gid)
                );
                CREATE TABLE IF NOT EXISTS case_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL, gid TEXT NOT NULL,
                    previous_status TEXT NOT NULL, previous_note TEXT NOT NULL,
                    status TEXT NOT NULL, note TEXT NOT NULL, at TEXT NOT NULL,
                    FOREIGN KEY(run_id, gid) REFERENCES cases(run_id, gid)
                );
                CREATE INDEX IF NOT EXISTS case_events_case ON case_events(run_id, gid, id);
                CREATE TRIGGER IF NOT EXISTS case_events_no_update
                BEFORE UPDATE ON case_events BEGIN
                    SELECT RAISE(ABORT, 'История заметок неизменяема');
                END;
                CREATE TRIGGER IF NOT EXISTS case_events_no_delete
                BEFORE DELETE ON case_events BEGIN
                    SELECT RAISE(ABORT, 'История заметок неизменяема');
                END;
            """)

    def _owned_path(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError("Путь должен находиться внутри рабочего каталога")
        return path

    @contextmanager
    def _connection(self, *, write=False):
        # Повторная проверка не даёт подменить файл базы ссылкой за границу root.
        connection = sqlite3.connect(self._owned_path(self.db_path), timeout=5, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            if write:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            if write:
                connection.commit()
        except BaseException:
            if write:
                connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _decode(row):
        if row is None:
            return None
        result = dict(row)
        for field in ("inspection", "metadata", "summary", "parameters"):
            if field in result:
                result[field] = json.loads(result[field])
        return result

    def register_upload(self, upload_id, original_name, path, inspection, sha256):
        upload_id = _id(upload_id)
        path = self._owned_path(path)
        if not isinstance(original_name, str) or not original_name:
            raise ValueError("Нужно исходное имя файла")
        if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError("Нужен SHA-256 загруженного файла")
        with self._connection(write=True) as connection:
            connection.execute(
                "INSERT INTO uploads VALUES (?, ?, ?, ?, ?, ?)",
                (upload_id, original_name, str(path), _json(inspection), sha256, _now()),
            )
            return self._decode(connection.execute("SELECT * FROM uploads WHERE id=?", (upload_id,)).fetchone())

    def get_upload(self, upload_id):
        with self._connection() as connection:
            return self._decode(connection.execute("SELECT * FROM uploads WHERE id=?", (_id(upload_id),)).fetchone())

    def create_dataset(self, dataset_id, metadata, summary, data_dir, fingerprint):
        dataset_id = _id(dataset_id)
        data_dir = self._owned_path(data_dir)
        metadata_json, summary_json = _json(metadata), _json(summary)
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("Нужен отпечаток набора данных")
        name = str(metadata.get("name") or f"Набор {dataset_id[:8]}")
        with self._connection(write=True) as connection:
            existing = connection.execute("SELECT * FROM datasets WHERE fingerprint=?", (fingerprint,)).fetchone()
            if existing is not None:
                return {**self._decode(existing), "reused": True}
            connection.execute(
                "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?)",
                (dataset_id, name, metadata_json, summary_json, str(data_dir), fingerprint, _now()),
            )
            row = connection.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
            return {**self._decode(row), "reused": False}

    def get_dataset(self, dataset_id):
        with self._connection() as connection:
            return self._decode(connection.execute("SELECT * FROM datasets WHERE id=?", (_id(dataset_id),)).fetchone())

    def list_datasets(self):
        with self._connection() as connection:
            return [self._decode(row) for row in connection.execute("SELECT * FROM datasets ORDER BY created_at DESC, id")]

    def create_run(self, dataset_id, parameters):
        dataset_id, parameters_json = _id(dataset_id), _json(parameters)
        with self._connection(write=True) as connection:
            dataset = connection.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
            if dataset is None:
                raise ValueError("Набор данных не найден")
            existing = connection.execute(
                "SELECT * FROM runs WHERE dataset_id=? AND parameters=? AND status IN ('queued','running') "
                "ORDER BY created_at DESC LIMIT 1", (dataset_id, parameters_json),
            ).fetchone()
            if existing is not None:
                return {**self._decode(existing), "reused": True}
            run_id, created_at = uuid4().hex, _now()
            name = f"{dataset['name']} · {created_at[:19].replace('T', ' ')} UTC"
            connection.execute(
                "INSERT INTO runs (id,dataset_id,name,parameters,status,created_at,progress,phase) "
                "VALUES (?, ?, ?, ?, 'queued', ?, 0, 'queued')",
                (run_id, dataset_id, name, parameters_json, created_at),
            )
            row = connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            return {**self._decode(row), "reused": False}

    def get_run(self, run_id):
        with self._connection() as connection:
            return self._decode(connection.execute("SELECT * FROM runs WHERE id=?", (_id(run_id),)).fetchone())

    def list_runs(self, dataset_id=None):
        sql = "SELECT r.*, d.name AS dataset_name FROM runs r JOIN datasets d ON d.id=r.dataset_id"
        arguments = ()
        if dataset_id is not None:
            sql += " WHERE r.dataset_id=?"
            arguments = (_id(dataset_id),)
        with self._connection() as connection:
            return [self._decode(row) for row in connection.execute(sql + " ORDER BY r.created_at DESC, r.id", arguments)]

    def update_run(self, run_id, **fields):
        run_id = _id(run_id)
        allowed = {"status", "progress", "phase", "error", "finished_at"}
        if fields.keys() - allowed:
            raise ValueError("Нельзя изменить это поле расчёта")
        if "status" in fields and (not isinstance(fields["status"], str) or fields["status"] not in RUN_STATUSES):
            raise ValueError("Неизвестный статус расчёта")
        if "progress" in fields:
            value = fields["progress"]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 100):
                raise ValueError("Прогресс должен быть числом от 0 до 100")
        for field in ("phase", "error", "finished_at"):
            if field in fields and fields[field] is not None and not isinstance(fields[field], str):
                raise ValueError(f"{field} должно быть строкой")
        if "phase" in fields and fields["phase"] is None:
            raise ValueError("phase должно быть строкой")
        if fields.get("status") in {"completed", "failed"}:
            fields.setdefault("finished_at", _now())
        with self._connection(write=True) as connection:
            if fields:
                # Имена колонок берутся только из фиксированного allowlist выше.
                assignments = ", ".join(f"{field}=?" for field in fields)
                connection.execute(f"UPDATE runs SET {assignments} WHERE id=?", (*fields.values(), run_id))
            row = connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise ValueError("Расчёт не найден")
            return self._decode(row)

    def run_dir(self, run_id):
        return self._owned_path(self.root / "runs" / _id(run_id))

    def recover_interrupted_runs(self):
        with self._connection(write=True) as connection:
            cursor = connection.execute(
                "UPDATE runs SET status='failed', phase='interrupted', error=?, finished_at=? "
                "WHERE status IN ('queued','running')", (INTERRUPTED_MESSAGE, _now()),
            )
            return cursor.rowcount

    def save_case(self, run_id, gid, status, note):
        run_id, gid = _id(run_id), _gid(gid)
        if not isinstance(status, str) or status not in CASE_STATUSES:
            raise ValueError("Неизвестный статус проверки")
        if not isinstance(note, str) or len(note) > MAX_NOTE_LENGTH:
            raise ValueError(f"Заметка должна содержать не более {MAX_NOTE_LENGTH} символов")
        with self._connection(write=True) as connection:
            if connection.execute("SELECT 1 FROM runs WHERE id=?", (run_id,)).fetchone() is None:
                raise ValueError("Расчёт не найден")
            previous = connection.execute("SELECT * FROM cases WHERE run_id=? AND gid=?", (run_id, gid)).fetchone()
            if previous is not None and previous["status"] == status and previous["note"] == note:
                return dict(previous)
            now = _now()
            connection.execute(
                "INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(run_id,gid) DO UPDATE SET "
                "status=excluded.status, note=excluded.note, updated_at=excluded.updated_at",
                (run_id, gid, status, note, now, now),
            )
            connection.execute(
                "INSERT INTO case_events (run_id,gid,previous_status,previous_note,status,note,at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, gid, previous["status"] if previous else "new",
                 previous["note"] if previous else "", status, note, now),
            )
            return dict(connection.execute("SELECT * FROM cases WHERE run_id=? AND gid=?", (run_id, gid)).fetchone())

    def get_case(self, run_id, gid):
        run_id, gid = _id(run_id), _gid(gid)
        with self._connection() as connection:
            if connection.execute("SELECT 1 FROM runs WHERE id=?", (run_id,)).fetchone() is None:
                raise ValueError("Расчёт не найден")
            row = connection.execute("SELECT * FROM cases WHERE run_id=? AND gid=?", (run_id, gid)).fetchone()
            return dict(row) if row is not None else {
                "run_id": run_id, "gid": gid, "status": "new", "note": "",
                "created_at": None, "updated_at": None,
            }

    def list_cases(self, run_id=None):
        sql = ("SELECT c.*, r.name AS run_name, r.status AS run_status, r.dataset_id, "
               "d.name AS dataset_name FROM cases c JOIN runs r ON r.id=c.run_id "
               "JOIN datasets d ON d.id=r.dataset_id")
        arguments = ()
        if run_id is not None:
            sql += " WHERE c.run_id=?"
            arguments = (_id(run_id),)
        with self._connection() as connection:
            return [dict(row) for row in connection.execute(sql + " ORDER BY c.updated_at DESC, c.run_id, c.gid", arguments)]

    def list_case_events(self, run_id, gid):
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM case_events WHERE run_id=? AND gid=? ORDER BY id", (_id(run_id), _gid(gid)))
            return [{**dict(row), "created_at": row["at"],
                     "previous": {"status": row["previous_status"], "note": row["previous_note"]}} for row in rows]


class JobRunner:
    """Один расчёт за раз в отдельном процессе; веб-сервер остаётся отзывчивым."""

    def __init__(self, store):
        self.store = store
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="furyhub-run")
        self._lock = threading.Lock()
        self._futures = {}
        self._closed = False

    def submit(self, run_id):
        run_id = _id(run_id)
        with self._lock:
            if self._closed:
                raise RuntimeError("Очередь расчётов остановлена")
            if run_id in self._futures:
                return self._futures[run_id]
            run = self.store.get_run(run_id)
            if run is None:
                raise ValueError("Расчёт не найден")
            if run["status"] != "queued":
                raise ValueError("Можно запускать только новый расчёт из очереди")
            future = self._pool.submit(self._execute, run_id)
            self._futures[run_id] = future
            return future

    def _execute(self, run_id):
        try:
            run = self.store.get_run(run_id)
            if run is None or run["status"] != "queued":
                return
            dataset = self.store.get_dataset(run["dataset_id"])
            data_dir = self.store._owned_path(dataset["data_dir"])
            out_dir = self.store.run_dir(run_id)
            # Повторная отправка задания никогда не перезаписывает старый запуск.
            out_dir.mkdir(parents=True, exist_ok=False)
            self.store.update_run(run_id, status="running", progress=5, phase="startup", error=None)
            (out_dir / "parameters.json").write_text(_json(run["parameters"]), encoding="utf-8")
            script = Path(__file__).resolve().with_name("app_worker.py")
            command = [sys.executable, "-B", str(script), "--data", str(data_dir), "--out", str(out_dir),
                       "--parameters", str(out_dir / "parameters.json")]
            self.store.update_run(run_id, progress=10, phase="calculation")
            with (out_dir / "stdout.log").open("w", encoding="utf-8") as stdout, \
                    (out_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
                result = subprocess.run(command, cwd=script.parent, stdout=stdout, stderr=stderr,
                                        timeout=RUN_TIMEOUT_SECONDS, check=False,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if result.returncode:
                self.store.update_run(run_id, status="failed", phase="failed",
                    error=f"Расчёт завершился с кодом {result.returncode}. Подробности сохранены в журнале запуска.")
                return
            self.store.update_run(run_id, status="completed", phase="ready", progress=100, error=None)
        except subprocess.TimeoutExpired:
            self.store.update_run(run_id, status="failed", phase="failed",
                error="Расчёт превысил лимит 5 минут и был остановлен.")
        except Exception:
            # Пути, содержимое данных и возможные секреты не отдаются в UI через исключения.
            self.store.update_run(run_id, status="failed", phase="failed",
                error="Не удалось выполнить расчёт. Проверьте доступность файлов и журнал запуска.")

    def shutdown(self, wait=True):
        with self._lock:
            self._closed = True
        self._pool.shutdown(wait=wait)
