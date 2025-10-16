from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def render_progress(current: int, total: int, width: int = 32) -> str:
    if total <= 0:
        return "[未开始]"
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "=" * filled + "." * (width - filled)
    return f"[{bar}] {current}/{total} ({ratio:.0%})"


def _escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def _read_env_int(name: str, default: int) -> int:
    import os

    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _read_env_names(var_name: str) -> List[str]:
    import os

    raw = os.getenv(var_name, "")
    if not raw:
        return []
    parts: List[str] = []
    for token in raw.replace("\n", ",").replace(";", ",").replace(" ", ",").split(","):
        t = token.strip()
        if t:
            parts.append(t)
    return parts


def _filter_patients(dirs: List, include_names: List[str]) -> List:
    if not include_names:
        return dirs
    wanted = set(include_names)
    order = {name: i for i, name in enumerate(include_names)}
    filtered = [d for d in dirs if d.name in wanted]
    filtered.sort(key=lambda d: order.get(d.name, 10_000_000))
    return filtered


def _prioritize_patients(dirs: List, first_names: List[str]) -> List:
    if not first_names:
        return dirs
    first_set = set(first_names)
    order = {name: i for i, name in enumerate(first_names)}
    first = [d for d in dirs if d.name in first_set]
    first.sort(key=lambda d: order.get(d.name, 10_000_000))
    rest = [d for d in dirs if d.name not in first_set]
    return first + rest


def update_usage(target: Dict[str, int], delta: Dict[str, Any]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = delta.get(key)
        if isinstance(value, (int, float)):
            target[key] = target.get(key, 0) + int(value)

