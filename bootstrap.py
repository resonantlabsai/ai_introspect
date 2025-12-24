"""ai_introspect bootstrap helpers."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from typing import Optional

import zipfile
import shutil
import sys
import importlib
import sqlite3


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


# --------------------------------------------------------------------------------------
# Zip-native bootstrap (AI-friendly)
# --------------------------------------------------------------------------------------

@dataclass
class BootstrapResult:
    ai_introspect_root: str
    repo_root: str
    db_path: str
    fts_enabled: bool
    counts: dict
    smoke: dict


def _extract_zip(zip_path: Path, dest_dir: Path) -> Path:
    """Extract zip into dest_dir. Returns the extraction root.
    If the zip contains a single top-level directory, return that directory; else return dest_dir.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)

    # Detect single top-level folder
    children = [p for p in dest_dir.iterdir() if p.name not in {"__MACOSX"}]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return dest_dir


def _copytree_over(src: Path, dst: Path) -> None:
    """Copy src tree onto dst (overwrite)."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)



def _purge_tools_modules() -> None:
    """Remove any previously-imported 'tools.*' modules to avoid cross-repo collisions."""
    for key in list(sys.modules.keys()):
        if key == 'tools' or key.startswith('tools.'):
            sys.modules.pop(key, None)
    importlib.invalidate_caches()

def install_ai_introspect_into_repo(ai_extracted_root: Path, repo_root: Path) -> Path:
    """Copy the .ai_introspect toolbelt into the repo root.
    Returns the installed .ai_introspect directory path.
    """
    ai_root = get_ai_introspect_root(ai_extracted_root)
    if ai_root is None:
        raise FileNotFoundError(f"Could not locate {DEFAULT_DIRNAME} inside {ai_extracted_root}")

    dst = repo_root / DEFAULT_DIRNAME
    _copytree_over(ai_root, dst)
    return dst


def clear_nav_db(repo_root: Path) -> Path:
    """Delete the repo_nav.sqlite file if present. Returns db_path."""
    db_path = repo_root / DEFAULT_DIRNAME / "repo_nav.sqlite"
    if db_path.exists():
        db_path.unlink()
    return db_path


def reindex_repo(repo_root: Path, *, update_fts: bool = True, max_files: Optional[int] = None, budget_seconds: Optional[float] = None) -> dict:
    """Rebuild the nav sqlite index for repo_root."""
    # Ensure we can import sqlite_indexer from the repo-local .ai_introspect.
    ai_root = repo_root / DEFAULT_DIRNAME
    if str(ai_root) not in sys.path:
        sys.path.insert(0, str(ai_root))

    _purge_tools_modules()

    from tools import sqlite_indexer as _sqlite_indexer  # type: ignore

    db_path = repo_root / DEFAULT_DIRNAME / "repo_nav.sqlite"
    # ensure schema
    _sqlite_indexer.ensure_db(db_path=db_path, enable_fts=update_fts)
    report = _sqlite_indexer.index_repo(
        repo_root=repo_root,
        db_path=db_path,
        update_fts=update_fts,
        max_files=max_files,
        budget_seconds=budget_seconds,
    )
    return report


def _db_counts(db_path: Path) -> dict:
    counts = {}
    if not db_path.exists():
        return counts
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        for table in ["files", "symbols", "imports", "snippets", "refs", "atlas_fts"]:
            try:
                cur.execute(f"SELECT COUNT(1) FROM {table}")
                counts[table] = int(cur.fetchone()[0])
            except Exception:
                # table may not exist (refs/fts optional)
                continue
        # meta flags
        try:
            cur.execute("SELECT value FROM meta WHERE key='fts5_enabled'")
            row = cur.fetchone()
            counts["fts5_enabled"] = (row and row[0] == "1")
        except Exception:
            counts["fts5_enabled"] = False
    return counts


def smoke_test(repo_root: Path) -> dict:
    """Run a few non-destructive tool calls to confirm the belt works."""
    ai_root = repo_root / DEFAULT_DIRNAME
    if str(ai_root) not in sys.path:
        sys.path.insert(0, str(ai_root))

    _purge_tools_modules()

    from tools import ai_introspect_tools as _ait  # type: ignore

    out: dict = {"ok": True, "steps": []}
    try:
        d = _ait.doctor(repo_root=str(repo_root))
        out["steps"].append({"doctor": {"ok": True, "fts": bool(d.get("fts_enabled"))}})
    except Exception as e:
        out["ok"] = False
        out["steps"].append({"doctor": {"ok": False, "error": str(e)}})
        return out

    # Pick one fqname from db to test callers/callees without hardcoding repo specifics
    db_path = Path(_ait._default_db_path(str(repo_root)))  # type: ignore
    fq = None
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.cursor()
            cur.execute("SELECT fqname FROM symbols WHERE fqname IS NOT NULL AND fqname != '' LIMIT 1")
            row = cur.fetchone()
            fq = row[0] if row else None
    except Exception:
        fq = None

    try:
        r = _ait.nav("impact:__init__", repo_root=str(repo_root), explain=True)  # cheap, likely exists somewhere
        out["steps"].append({"nav_impact": {"ok": True, "matches": len(r.get("matches", []))}})
    except Exception as e:
        out["ok"] = False
        out["steps"].append({"nav_impact": {"ok": False, "error": str(e)}})

    if fq:
        try:
            r = _ait.nav(f"fq:{fq}", repo_root=str(repo_root), explain=True)
            out["steps"].append({"nav_fq": {"ok": True, "fq": fq, "matches": len(r.get("matches", []))}})
        except Exception as e:
            out["ok"] = False
            out["steps"].append({"nav_fq": {"ok": False, "error": str(e)}})

        try:
            r = _ait.nav(f"callers:fq:{fq}", repo_root=str(repo_root), explain=True)
            out["steps"].append({"callers": {"ok": True, "matches": len(r.get("matches", []))}})
        except Exception as e:
            out["ok"] = False
            out["steps"].append({"callers": {"ok": False, "error": str(e)}})

        try:
            r = _ait.nav(f"callees:fq:{fq}", repo_root=str(repo_root), explain=True)
            out["steps"].append({"callees": {"ok": True, "matches": len(r.get("matches", []))}})
        except Exception as e:
            out["ok"] = False
            out["steps"].append({"callees": {"ok": False, "error": str(e)}})

    return out


def bootstrap_from_zips(ai_introspect_zip: str, repo_zip: str, *, work_dir: Optional[str] = None, update_fts: bool = True) -> BootstrapResult:
    """End-to-end: extract both zips, install toolbelt into repo, clear+reindex DB, smoke test.
    Returns a BootstrapResult with paths and counts.
    """
    wd = Path(work_dir) if work_dir else Path.cwd() / "_ai_bootstrap"
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True, exist_ok=True)

    ai_root = _extract_zip(Path(ai_introspect_zip), wd / "ai_introspect")
    repo_root = _extract_zip(Path(repo_zip), wd / "repo")

    # If repo zip has a single nested folder (common), keep that as repo_root
    # else repo_root already points at extraction dir.
    installed = install_ai_introspect_into_repo(ai_root, repo_root)

    # Ensure standard layout extras (sessions marker, etc.)
    ensure_ai_introspect_layout(repo_root)

    db_path = clear_nav_db(repo_root)
    report = reindex_repo(repo_root, update_fts=update_fts)

    counts = _db_counts(db_path)
    smoke = smoke_test(repo_root)

    return BootstrapResult(
        ai_introspect_root=str(installed),
        repo_root=str(repo_root),
        db_path=str(db_path),
        fts_enabled=bool(counts.get("fts5_enabled", False)),
        counts=counts,
        smoke=smoke,
    )