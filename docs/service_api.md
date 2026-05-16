# verl Training Service API

Base URL: `http://{host}:{port}`

---

## Health Check

**`GET /health`**

Response `200`:

```json
{
  "status": "ok",
  "pending_tasks": 0,
  "running_tasks": 1,
  "max_concurrent_tasks": 2
}
```

---

## Submit Task

**`POST /api/v1/tasks`**

Request body:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `config` | `object` | no | `{}` | Full training config matching `ppo_trainer.yaml` structure |
| `label` | `string \| null` | no | `null` | Human-readable label |
| `priority` | `int` | no | `0` | Priority 0–100, higher runs first |

Example:

```json
{
  "config": {
    "algorithm": {
      "adv_estimator": "grpo"
    },
    "data": {
      "train_files": ["/data/train.parquet"],
      "train_batch_size": 128
    },
    "trainer": {
      "n_gpus_per_node": 8,
      "nnodes": 1,
      "total_epochs": 10,
      "project_name": "my-project",
      "experiment_name": "exp-v2"
    },
    "actor_rollout_ref": {
      "model": {
        "path": "Qwen/Qwen2.5-7B-Instruct"
      }
    }
  },
  "label": "grpo-qwen-7b",
  "priority": 5
}
```

Response `201`:

```json
{
  "task_id": "a1b2c3d4e5f6",
  "status": "pending",
  "message": "Task a1b2c3d4e5f6 enqueued successfully."
}
```

---

## Get Task Status

**`GET /api/v1/tasks/{task_id}`**

Response `200`:

```json
{
  "task_id": "a1b2c3d4e5f6",
  "status": "running",
  "label": "grpo-qwen-7b",
  "created_at": "2026-05-16T12:00:00.123456+00:00",
  "updated_at": "2026-05-16T12:00:05.654321+00:00",
  "started_at": "2026-05-16T12:00:05.654321+00:00",
  "finished_at": null,
  "pid": 12345,
  "return_code": null,
  "error_message": null
}
```

Response `404`:

```json
{
  "detail": "Task not found."
}
```

### Task Status Values

| Status | Meaning |
|--------|---------|
| `pending` | Waiting in queue |
| `running` | Subprocess executing |
| `completed` | Finished successfully (exit 0) |
| `failed` | Finished with error (exit ≠ 0) |
| `cancelled` | Cancelled by user |

---

## List Tasks

**`GET /api/v1/tasks`**

Query parameters:

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `status` | `string \| null` | no | `null` | Filter by status: `pending` / `running` / `completed` / `failed` / `cancelled` |
| `limit` | `int` | no | `100` | Max results |
| `offset` | `int` | no | `0` | Pagination offset |

Response `200`:

```json
{
  "tasks": [
    {
      "task_id": "a1b2c3d4e5f6",
      "status": "completed",
      "label": "grpo-qwen-7b",
      "priority": 5,
      "created_at": "2026-05-16T12:00:00.123456+00:00",
      "updated_at": "2026-05-16T12:30:00.000000+00:00",
      "started_at": "2026-05-16T12:00:05.654321+00:00",
      "finished_at": "2026-05-16T12:30:00.000000+00:00",
      "pid": 12345,
      "return_code": 0,
      "error_message": null,
      "config": { "...": "..." }
    }
  ],
  "total": 1
}
```

---

## Cancel Task

**`DELETE /api/v1/tasks/{task_id}`**

Response `200`:

```json
{
  "task_id": "a1b2c3d4e5f6",
  "status": "cancelled",
  "message": "Task a1b2c3d4e5f6 has been cancelled."
}
```

Response `404`:

```json
{
  "detail": "Task not found."
}
```

- `pending` tasks are immediately cancelled in the queue.
- `running` tasks are cancelled by terminating the subprocess (SIGTERM → SIGKILL after 10s timeout).
- Terminal tasks (`completed` / `failed` / `cancelled`) are returned unchanged.

---

## State Machine

```
PENDING ──→ RUNNING ──exit 0──→ COMPLETED
   │           │    ──exit≠0──→ FAILED
   │           │    ──DELETE──→ CANCELLED
   └─DELETE──→ CANCELLED
```

Server restart: any task left in `running` state at shutdown is marked `failed`.
