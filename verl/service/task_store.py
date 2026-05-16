from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import aiosqlite

from .models import TaskInfo, TaskStatus

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id        TEXT PRIMARY KEY,
    status         TEXT NOT NULL DEFAULT 'pending',
    label          TEXT,
    priority       INTEGER NOT NULL DEFAULT 0,
    config_json    TEXT,
    pid            INTEGER,
    return_code    INTEGER,
    error_message  TEXT,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    started_at     TEXT,
    finished_at    TEXT
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_queue
    ON tasks(status, priority DESC, created_at ASC);
"""


class TaskStore:
    """Async SQLite-backed persistence for training tasks."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            from pathlib import Path
            home = Path.home() / ".verl" / "service"
            home.mkdir(parents=True, exist_ok=True)
            db_path = str(home / "tasks.db")
        self._db_path = db_path

    async def initialize(self) -> None:
        async with aiosqlite.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            await conn.executescript(_SCHEMA)
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.commit()

    async def insert_task(self, task: TaskInfo) -> None:
        async with aiosqlite.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            await conn.execute(
                """INSERT INTO tasks (task_id, status, label, priority,
                   config_json, pid, return_code, error_message,
                   created_at, updated_at, started_at, finished_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                self._task_to_row(task),
            )
            await conn.commit()

    async def get_task(self, task_id: str) -> Optional[TaskInfo]:
        async with aiosqlite.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = await conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_task(row)

    async def update_task(self, task: TaskInfo) -> None:
        async with aiosqlite.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = self._task_to_row(task)
            await conn.execute(
                """UPDATE tasks SET status=?, label=?, priority=?,
                   config_json=?, pid=?, return_code=?, error_message=?,
                   created_at=?, updated_at=?, started_at=?, finished_at=?
                   WHERE task_id=?""",
                (*row[1:], row[0]),  # all values + task_id for WHERE
            )
            await conn.commit()

    async def list_tasks(
        self,
        status_filter: Optional[TaskStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TaskInfo]:
        async with aiosqlite.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            if status_filter is not None:
                cursor = await conn.execute(
                    """SELECT * FROM tasks WHERE status = ?
                       ORDER BY priority DESC, created_at ASC
                       LIMIT ? OFFSET ?""",
                    (status_filter.value, limit, offset),
                )
            else:
                cursor = await conn.execute(
                    """SELECT * FROM tasks
                       ORDER BY priority DESC, created_at ASC
                       LIMIT ? OFFSET ?""",
                    (limit, offset),
                )
            rows = await cursor.fetchall()
            return [self._row_to_task(r) for r in rows]

    async def get_pending_tasks(self, limit: int = 10) -> list[TaskInfo]:
        async with aiosqlite.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = await conn.execute(
                """SELECT * FROM tasks WHERE status = 'pending'
                   ORDER BY priority DESC, created_at ASC
                   LIMIT ?""",
                (limit,),
            )
            rows = await cursor.fetchall()
            return [self._row_to_task(r) for r in rows]

    async def count_by_status(self, status: TaskStatus) -> int:
        async with aiosqlite.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = ?",
                (status.value,),
            )
            row = await cursor.fetchone()
            return row[0] if row else 0

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> TaskInfo:
        config_json = row["config_json"]
        config = json.loads(config_json) if config_json else None
        return TaskInfo(
            task_id=row["task_id"],
            status=TaskStatus(row["status"]),
            label=row["label"],
            priority=row["priority"],
            config=config,
            pid=row["pid"],
            return_code=row["return_code"],
            error_message=row["error_message"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
        )

    @staticmethod
    def _task_to_row(task: TaskInfo) -> tuple:
        return (
            task.task_id,
            task.status.value,
            task.label,
            task.priority,
            json.dumps(task.config) if task.config else None,
            task.pid,
            task.return_code,
            task.error_message,
            task.created_at.isoformat() if task.created_at else datetime.now(timezone.utc).isoformat(),
            task.updated_at.isoformat() if task.updated_at else datetime.now(timezone.utc).isoformat(),
            task.started_at.isoformat() if task.started_at else None,
            task.finished_at.isoformat() if task.finished_at else None,
        )
