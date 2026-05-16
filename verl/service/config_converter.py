from __future__ import annotations

import json
import re
from typing import Any


def config_dict_to_cli_args(config: dict[str, Any]) -> list[str]:
    """Convert a nested config dict to Hydra CLI override arguments.

    Flattens a nested dictionary into dotted-key ``key=value`` strings
    compatible with Hydra's ``@hydra.main`` interface.

    Example:
        ``{"algorithm": {"adv_estimator": "grpo"}}``
        becomes ``["algorithm.adv_estimator=grpo"]``.
    """

    args: list[str] = []

    def _flatten(d: dict[str, Any], prefix: str) -> None:
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                _flatten(value, full_key)
            else:
                args.append(f"{full_key}={_format_value(value)}")

    _flatten(config, "")
    return args


def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    s = str(value)
    if _needs_quoting(s):
        escaped = s.replace("'", "'\\''")
        return f"'{escaped}'"
    return s


def _needs_quoting(s: str) -> bool:
    return bool(re.search(r"[\s=\[\]{}()$`\"';&|<>*?!]", s))
