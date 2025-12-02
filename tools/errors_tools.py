"""Error logging helpers for .ai_introspect/errors.

These are meant for AI agents to drop structured error snapshots
(tracebacks, failing commands, file/line hints) that can be reused
in later turns without re-streaming big error blobs into context.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .ai_introspect_tools import (
    _get_root,
    _errors_dir,
    _now_iso,
    _write_json,
    _read_json,
    register_artifact,
)


def _errors_index_path(root: Path) -> Path:
    return root / "errors" / "errors_index.json"


def record_error(
    message: str,
    *,
    error_type: str = "runtime",
    file: Optional[str] = None,
    line: Optional[int] = None,
    context: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    cache_root: Optional[str | Path] = None,
) -> Path:
    """Record a structured error entry under .ai_introspect/errors.

    This is meant to be called when a tool run, parse attempt, or
    introspection step fails in a way the AI might want to revisit.
    """
    root = _get_root(cache_root)
    errors_root = _errors_dir(root)
    ts = _now_iso().replace(":", "-")
    safe_type = error_type.replace("/", "_")
    fname = f"{ts}_{safe_type}.json"
    out_path = errors_root / fname

    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "error",
        "created_at": _now_iso(),
        "error_type": error_type,
        "message": message,
        "file": file,
        "line": line,
        "context": context,
        "details": details or {},
    }
    _write_json(out_path, payload)

    # Update errors_index.json
    index_path = _errors_index_path(root)
    if index_path.exists():
        idx = _read_json(index_path)
    else:
        idx = {"schema_version": "1.0", "errors": []}
    idx.setdefault("errors", []).append(
        {
            "created_at": payload["created_at"],
            "error_type": error_type,
            "path": out_path.relative_to(root).as_posix(),
            "file": file,
            "line": line,
        }
    )
    _write_json(index_path, idx)

    rel_path = out_path.relative_to(root).as_posix()
    register_artifact(rel_path, "error", source_path=file, tags={"error_type": error_type}, cache_root=root)
    return out_path


def list_errors(
    *,
    error_type: Optional[str] = None,
    file_contains: Optional[str] = None,
    cache_root: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """Return a list of error index entries, optionally filtered."""
    root = _get_root(cache_root)
    index_path = _errors_index_path(root)
    if not index_path.exists():
        return []
    idx = _read_json(index_path)
    errors = idx.get("errors", [])
    out: List[Dict[str, Any]] = []

    for e in errors:
        if error_type and e.get("error_type") != error_type:
            continue
        if file_contains and file_contains not in (e.get("file") or ""):
            continue
        out.append(e)
    return out
