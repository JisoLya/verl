from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SubmitTaskRequest(BaseModel):
    config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Full training configuration as a nested JSON object. "
            "Follows the same structure as ppo_trainer.yaml."
        ),
    )
    label: Optional[str] = Field(
        default=None,
        description="Human-readable label for this task.",
    )
    priority: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Task priority. Higher values run sooner (0-100).",
    )


class TaskInfo(BaseModel):
    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: TaskStatus = TaskStatus.PENDING
    label: Optional[str] = None
    priority: int = 0
    config: Optional[dict[str, Any]] = None
    pid: Optional[int] = None
    return_code: Optional[int] = None
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class SubmitTaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    label: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    pid: Optional[int] = None
    return_code: Optional[int] = None
    error_message: Optional[str] = None


class TaskListResponse(BaseModel):
    tasks: list[TaskInfo]
    total: int


class CancelTaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: str


class HealthResponse(BaseModel):
    status: str = "ok"
    pending_tasks: int
    running_tasks: int
    max_concurrent_tasks: int
