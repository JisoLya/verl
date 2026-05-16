from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException

from .models import (
    CancelTaskResponse,
    HealthResponse,
    SubmitTaskRequest,
    SubmitTaskResponse,
    TaskListResponse,
    TaskStatus,
    TaskStatusResponse,
)
from .task_manager import TaskManager

logger = logging.getLogger(__name__)


def create_app(
    task_manager: TaskManager,
    title: str = "verl Training Service",
    version: str = "0.1.0",
) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await task_manager.start()
        logger.info("TaskManager started.")
        yield
        await task_manager.stop()
        logger.info("TaskManager stopped.")

    app = FastAPI(title=title, version=version, lifespan=lifespan)

    @app.post("/api/v1/tasks", response_model=SubmitTaskResponse, status_code=201)
    async def submit_task(request: SubmitTaskRequest):
        task = await task_manager.submit_task(request)
        return SubmitTaskResponse(
            task_id=task.task_id,
            status=task.status,
            message=f"Task {task.task_id} enqueued successfully.",
        )

    @app.get("/api/v1/tasks/{task_id}", response_model=TaskStatusResponse)
    async def get_task(task_id: str):
        task = await task_manager.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found.")
        return TaskStatusResponse(
            task_id=task.task_id,
            status=task.status,
            label=task.label,
            created_at=task.created_at,
            updated_at=task.updated_at,
            started_at=task.started_at,
            finished_at=task.finished_at,
            pid=task.pid,
            return_code=task.return_code,
            error_message=task.error_message,
        )

    @app.get("/api/v1/tasks", response_model=TaskListResponse)
    async def list_tasks(
        status: Optional[TaskStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ):
        tasks, total = await task_manager.list_tasks(
            status_filter=status, limit=limit, offset=offset
        )
        return TaskListResponse(tasks=tasks, total=total)

    @app.delete("/api/v1/tasks/{task_id}", response_model=CancelTaskResponse)
    async def cancel_task(task_id: str):
        task = await task_manager.cancel_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found.")
        return CancelTaskResponse(
            task_id=task.task_id,
            status=task.status,
            message=f"Task {task_id} has been {task.status.value}.",
        )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        pending = await task_manager._store.count_by_status(TaskStatus.PENDING)
        running = await task_manager._store.count_by_status(TaskStatus.RUNNING)
        return HealthResponse(
            pending_tasks=pending,
            running_tasks=running,
            max_concurrent_tasks=task_manager._max_concurrent,
        )

    return app
