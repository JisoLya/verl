from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from typing import Optional

from .config_converter import config_dict_to_cli_args
from .models import TaskInfo, TaskStatus, SubmitTaskRequest
from .task_store import TaskStore

logger = logging.getLogger(__name__)

_IS_WINDOWS = os.name == "nt"


class TaskManager:
    """Manages the lifecycle of training tasks: queuing, spawning subprocesses,
    concurrency enforcement, and state persistence.
    """

    def __init__(
        self,
        store: TaskStore,
        max_concurrent_tasks: int = 1,
        python_bin: Optional[str] = None,
    ) -> None:
        self._store = store
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._python_bin = python_bin or sys.executable
        self._max_concurrent = max_concurrent_tasks
        self._active_processes: dict[str, asyncio.subprocess.Process] = {}
        self._drain_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        running_tasks = await self._store.list_tasks(
            status_filter=TaskStatus.RUNNING
        )
        for task in running_tasks:
            task.status = TaskStatus.FAILED
            task.error_message = "Server restarted while task was running."
            task.finished_at = datetime.now(timezone.utc)
            task.return_code = -1
            await self._store.update_task(task)
            logger.warning(
                f"Task {task.task_id} was RUNNING at last shutdown; marked FAILED."
            )

        self._drain_task = asyncio.create_task(self._drain_queue())
        logger.info("TaskManager started (max_concurrent=%d).", self._max_concurrent)

    async def stop(self) -> None:
        if self._drain_task:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass

        for task_id, proc in list(self._active_processes.items()):
            logger.info("Terminating subprocess for task %s (pid=%d)", task_id, proc.pid)
            await _terminate_process(proc)
        logger.info("TaskManager stopped.")

    async def submit_task(self, request: SubmitTaskRequest) -> TaskInfo:
        now = datetime.now(timezone.utc)
        task = TaskInfo(
            status=TaskStatus.PENDING,
            label=request.label,
            priority=request.priority,
            config=request.config,
            created_at=now,
            updated_at=now,
        )
        await self._store.insert_task(task)
        logger.info("Task %s submitted (label=%r, priority=%d).", task.task_id, task.label, task.priority)
        return task

    async def get_task(self, task_id: str) -> Optional[TaskInfo]:
        return await self._store.get_task(task_id)

    async def list_tasks(
        self,
        status_filter: Optional[TaskStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[TaskInfo], int]:
        tasks = await self._store.list_tasks(
            status_filter=status_filter, limit=limit, offset=offset
        )
        if status_filter is not None:
            total = await self._store.count_by_status(status_filter)
        else:
            total = len(tasks)
        return tasks, total

    async def cancel_task(self, task_id: str) -> Optional[TaskInfo]:
        task = await self._store.get_task(task_id)
        if task is None:
            return None

        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            task.finished_at = datetime.now(timezone.utc)
            task.error_message = "Cancelled by user request."
            await self._store.update_task(task)
            return task

        if task.status == TaskStatus.RUNNING:
            task.status = TaskStatus.CANCELLED
            task.finished_at = datetime.now(timezone.utc)
            task.error_message = "Cancelled by user request."
            await self._store.update_task(task)

            proc = self._active_processes.get(task_id)
            if proc is not None:
                await _terminate_process(proc)
                self._active_processes.pop(task_id, None)
            return task

        return task

    async def _drain_queue(self) -> None:
        while True:
            await self._semaphore.acquire()
            try:
                pending = await self._store.get_pending_tasks(limit=1)
                if not pending:
                    self._semaphore.release()
                    await asyncio.sleep(1.0)
                    continue

                task = pending[0]
                await self._run_task(task)
            except Exception:
                logger.exception("Unexpected error in drain loop.")
                self._semaphore.release()

    async def _run_task(self, task: TaskInfo) -> None:
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now(timezone.utc)
        task.updated_at = datetime.now(timezone.utc)
        await self._store.update_task(task)

        try:
            cli_args = config_dict_to_cli_args(task.config if task.config else {})
            cmd = [self._python_bin, "-m", "verl.trainer.main_ppo", *cli_args]
            logger.info("Starting task %s: %s", task.task_id, " ".join(cmd))

            kwargs: dict = {
                "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.PIPE,
            }
            if _IS_WINDOWS:
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["preexec_fn"] = os.setpgrp

            proc = await asyncio.create_subprocess_exec(*cmd, **kwargs)
            self._active_processes[task.task_id] = proc
            task.pid = proc.pid
            await self._store.update_task(task)

            stdout, stderr = await proc.communicate()
            self._active_processes.pop(task.task_id, None)

            if task.status == TaskStatus.CANCELLED:
                return

            if proc.returncode == 0:
                task.status = TaskStatus.COMPLETED
            else:
                task.status = TaskStatus.FAILED
                task.error_message = (
                    stderr.decode("utf-8", errors="replace")[-2000:]
                    if stderr
                    else f"Exit code {proc.returncode}"
                )
            task.return_code = proc.returncode
            task.finished_at = datetime.now(timezone.utc)
        except Exception as e:
            self._active_processes.pop(task.task_id, None)
            if task.status != TaskStatus.CANCELLED:
                task.status = TaskStatus.FAILED
                task.error_message = str(e)[:2000]
            task.finished_at = datetime.now(timezone.utc)
        finally:
            task.updated_at = datetime.now(timezone.utc)
            await self._store.update_task(task)
            self._semaphore.release()


async def _terminate_process(proc: asyncio.subprocess.Process) -> None:
    """Gracefully terminate a subprocess, then force-kill after a timeout."""
    if proc.returncode is not None:
        return
    try:
        if _IS_WINDOWS:
            proc.terminate()
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
    except Exception:
        pass

    try:
        await asyncio.wait_for(proc.wait(), timeout=10.0)
    except asyncio.TimeoutError:
        logger.warning("Process %d did not exit; force-killing.", proc.pid)
        try:
            if _IS_WINDOWS:
                proc.kill()
            else:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        except Exception:
            pass
        await proc.wait()
