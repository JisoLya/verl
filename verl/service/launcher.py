from __future__ import annotations

import argparse
import asyncio
import logging
import socket
import sys

from verl.service.app import create_app
from verl.service.task_manager import TaskManager
from verl.service.task_store import TaskStore
from verl.workers.rollout.utils import _UvicornServerAutoPort

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="verl Training REST API Service")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the HTTP server to (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port to bind to. 0 means auto-select (default: 0).",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to the SQLite database file (default: ~/.verl/service/tasks.db).",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Maximum number of concurrent training tasks (default: 1).",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python binary for training subprocesses (default: sys.executable).",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level for the service (default: info).",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    store = TaskStore(db_path=args.db_path)
    await store.initialize()
    logger.info("Database initialized at %s.", store._db_path)

    task_manager = TaskManager(
        store=store,
        max_concurrent_tasks=args.max_concurrent,
        python_bin=args.python_bin,
    )

    app = create_app(task_manager)

    import uvicorn
    config = uvicorn.Config(
        app, host=args.host, port=args.port, log_level=args.log_level
    )
    server = _UvicornServerAutoPort(config)
    server_task = asyncio.create_task(server.serve())
    actual_port = await server.get_port()

    if actual_port is None:
        await server_task
        raise RuntimeError("Failed to start HTTP server.")

    hostname = socket.gethostname()
    logger.info("verl Training Service started at http://%s:%d", hostname, actual_port)
    logger.info("Max concurrent tasks: %d", args.max_concurrent)
    logger.info("Health check: http://%s:%d/health", hostname, actual_port)
    logger.info("API docs:    http://%s:%d/docs", hostname, actual_port)

    try:
        await server_task
    except asyncio.CancelledError:
        logger.info("Server shutting down.")


if __name__ == "__main__":
    asyncio.run(main())
