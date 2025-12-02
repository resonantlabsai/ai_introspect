"""ai_introspect bootstrap helpers."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from typing import Optional


DEFAULT_DIRNAME = ".ai_introspect"


@dataclass
class SessionMarker:
    schema_version: str = "1.0"
    session_id: str = ""
    created_at: str = ""
    model: str = ""
    notes: str = ""


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def get_ai_introspect_root(start: Optional[Path] = None) -> Path:
    start = start or Path.cwd()
    current = start.resolve()
    while True:
        candidate = current / DEFAULT_DIRNAME
        if candidate.exists() and candidate.is_dir():
            return candidate
        if current.parent == current:
            return Path.cwd() / DEFAULT_DIRNAME
        current = current.parent


def ensure_ai_introspect_layout(start: Optional[Path] = None) -> Path:
    root = get_ai_introspect_root(start)
    (root).mkdir(parents=True, exist_ok=True)
    for sub in [
        "sessions",
        "logs/conv",
        "logs/runs",
        "repo_index",
        "modules",
        "regex",
        "search",
        "errors",
        "tools",
    ]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    _ensure_session_marker(root)
    return root


def _ensure_session_marker(root: Path) -> Path:
    sessions_dir = root / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    marker_path = sessions_dir / "session_marker.json"
    if marker_path.exists():
        return marker_path
    marker = SessionMarker(
        session_id=f"sess_{_now_iso()}",
        created_at=_now_iso(),
        model="gpt-5.1-thinking",
        notes="Auto-created session marker by ai_introspect bootstrap.",
    )
    with marker_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(marker), f, indent=2, sort_keys=True)
    return marker_path
