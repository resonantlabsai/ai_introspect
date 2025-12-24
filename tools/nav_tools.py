from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from .sqlite_index import query_symbols, ensure_schema
from .ai_introspect_tools import _get_root, read_lines

def _default_db(root: Path) -> Path:
    return root / "repo_index" / "code_index.sqlite"

def nav(query: str, *, repo_root: str | None = None, limit: int = 5, index_state: str = "good") -> Dict[str, Any]:
    """
    Lightweight navigation over sqlite symbol index.

    query forms:
      - "symbol:LaneCatalog"
      - "path:blob_lab/runtime/lab_loop.py"
      - plain "LaneCatalog"
    """
    root = _get_root(repo_root)
    db_path = _default_db(root)
    ensure_schema(str(db_path))

    q = query
    for prefix in ("symbol:", "path:", "fq:", "def:"):
        if q.startswith(prefix):
            q = q[len(prefix):].strip()
            break

    matches = query_symbols(str(db_path), q, limit=limit, state=index_state)
    return {"matches": matches, "db_path": str(db_path)}

def show_match(match: Dict[str, Any], *, repo_root: str | None = None, pad: int = 2) -> str:
    root = _get_root(repo_root)
    path = match["path"]
    start = max(1, int(match.get("start_line", 1)) - pad)
    end = int(match.get("end_line", start)) + pad
    # bounded by read_lines itself
    return read_lines(path, start_line=start, end_line=end, repo_root=str(root))
