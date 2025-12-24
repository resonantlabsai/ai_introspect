from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sqlite3
import ast
from datetime import datetime
import hashlib

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS builds(
        build_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        state TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS files(
        path TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        mtime REAL NOT NULL,
        size INTEGER NOT NULL,
        build_id TEXT NOT NULL,
        PRIMARY KEY(path, build_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS symbols(
        fqname TEXT NOT NULL,
        kind TEXT NOT NULL,
        path TEXT NOT NULL,
        start_line INTEGER NOT NULL,
        end_line INTEGER NOT NULL,
        signature TEXT,
        build_id TEXT NOT NULL,
        PRIMARY KEY(fqname, path, build_id)
    );
    """,
]

def _sha256_text(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def ensure_schema(db_path: str) -> None:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p))
    try:
        cur = con.cursor()
        for stmt in SCHEMA:
            cur.executescript(stmt)
        con.commit()
    finally:
        con.close()

def _iter_py_files(repo_root: Path) -> List[Path]:
    return [p for p in repo_root.rglob("*.py") if p.is_file() and ".ai_introspect" not in str(p)]

def create_build(db_path: str, state: str) -> str:
    ensure_schema(db_path)
    build_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    con = sqlite3.connect(db_path)
    try:
        con.execute("INSERT OR REPLACE INTO builds(build_id, created_at, state) VALUES (?,?,?)",
                    (build_id, datetime.utcnow().isoformat()+"Z", state))
        con.commit()
    finally:
        con.close()
    return build_id

def index_repo(repo_root: str, db_path: str, *, build_id: str, state: str = "good") -> Dict[str, Any]:
    root = Path(repo_root).resolve()
    ensure_schema(db_path)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    files_indexed = 0
    symbols_indexed = 0
    try:
        cur = con.cursor()
        # update build state
        cur.execute("INSERT OR REPLACE INTO builds(build_id, created_at, state) VALUES (?,?,?)",
                    (build_id, datetime.utcnow().isoformat()+"Z", state))
        for p in _iter_py_files(root):
            rel = str(p.relative_to(root))
            b = p.read_bytes()
            sha = _sha256_text(b)
            st = p.stat()
            cur.execute("INSERT OR REPLACE INTO files(path, sha256, mtime, size, build_id) VALUES (?,?,?,?,?)",
                        (rel, sha, st.st_mtime, st.st_size, build_id))
            files_indexed += 1

            try:
                tree = ast.parse(b.decode("utf-8", errors="replace"), filename=rel)
            except Exception:
                continue

            mod_name = rel[:-3].replace("/", ".").replace("\\", ".")
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    fq = f"{mod_name}:{node.name}"
                    start = getattr(node, "lineno", 1)
                    end = getattr(node, "end_lineno", start)
                    sig = f"def {node.name}(...)"
                    cur.execute("INSERT OR REPLACE INTO symbols(fqname, kind, path, start_line, end_line, signature, build_id) VALUES (?,?,?,?,?,?,?)",
                                (fq, "function", rel, start, end, sig, build_id))
                    symbols_indexed += 1
                elif isinstance(node, ast.ClassDef):
                    fq = f"{mod_name}:{node.name}"
                    start = getattr(node, "lineno", 1)
                    end = getattr(node, "end_lineno", start)
                    sig = f"class {node.name}"
                    cur.execute("INSERT OR REPLACE INTO symbols(fqname, kind, path, start_line, end_line, signature, build_id) VALUES (?,?,?,?,?,?,?)",
                                (fq, "class", rel, start, end, sig, build_id))
                    symbols_indexed += 1
        con.commit()
    finally:
        con.close()
    return {"files_indexed": files_indexed, "symbols_indexed": symbols_indexed, "build_id": build_id, "state": state}

def query_symbols(db_path: str, q: str, *, limit: int = 10, state: str = "good") -> List[Dict[str, Any]]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        # pick latest build of state
        row = con.execute("SELECT build_id FROM builds WHERE state=? ORDER BY created_at DESC LIMIT 1", (state,)).fetchone()
        if not row:
            return []
        build_id = row["build_id"]
        like = f"%{q}%"
        rows = con.execute(
            "SELECT fqname, kind, path, start_line, end_line, signature FROM symbols WHERE build_id=? AND (fqname LIKE ? OR path LIKE ?) LIMIT ?",
            (build_id, like, like, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()
