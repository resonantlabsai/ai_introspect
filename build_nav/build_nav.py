#!/usr/bin/env python3
"""
build_nav.py

Drop this file in a repo root and run:

  python -m build_nav
  # or
  python build_nav.py

Default behavior: **build + index** every run (incremental by sha256).
No switches required for normal use.

Auto-healing rules (requested behavior, with a safety twist):
- If multiple `repo_nav.sqlite` are found anywhere under the repo root:
    - Move them into `.repo_nav_backups/<timestamp>/...` (preserving paths)
    - Rebuild a fresh root `repo_nav.sqlite` (clean schema + full re-index)
- If exactly one `repo_nav.sqlite` is found but it's not in repo root:
    - Move it to the repo root as `repo_nav.sqlite`
    - Ensure schema, then incremental index (updates stale rows by sha256)
- If none found:
    - Create root `repo_nav.sqlite` and index

Design notes:
- SQLite schema is "atlas_v1" and is repo/project agnostic.
- Python symbol extraction is Python-specific; non-.py files still go into `files`.
- The sqlite file contains code snippets; **do not commit it to public repos** unless you intend to publish that code.
- Standard library only.

"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import fnmatch
import hashlib
from pathlib import Path
import shutil
import sqlite3
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SCHEMA_VERSION = "atlas_v1"
DB_NAME = "repo_nav.sqlite"

DEFAULT_EXCLUDES = [
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    "node_modules", "dist", "build", ".mypy_cache", ".ruff_cache",
    ".idea", ".vscode",
    # common "big/noisy" folder in your repos; override if you want:
    "data",
]

PY_EXTS = {".py"}


def utc_now_z() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def mtime_utc_z(path: Path) -> Optional[str]:
    try:
        ts = path.stat().st_mtime
    except FileNotFoundError:
        return None
    dt = _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).replace(microsecond=0)
    return dt.isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_norm_text(text: str) -> str:
    norm = "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"))
    return sha256_bytes(norm.encode("utf-8", errors="replace"))


def rel_path(repo_root: Path, file_path: Path) -> str:
    return file_path.resolve().relative_to(repo_root.resolve()).as_posix()


def should_exclude(rel_posix: str, exclude_roots: Sequence[str], exclude_globs: Sequence[str]) -> bool:
    parts = rel_posix.split("/")
    if parts and parts[0] in exclude_roots:
        return True
    for g in exclude_globs:
        if g and fnmatch.fnmatch(rel_posix, g):
            return True
    return False


def iter_repo_files(repo_root: Path, exclude_roots: Sequence[str], exclude_globs: Sequence[str]) -> Iterable[Path]:
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        rp = rel_path(repo_root, p)
        if should_exclude(rp, exclude_roots, exclude_globs):
            continue
        yield p


def ensure_schema(conn: sqlite3.Connection, *, enable_fts: bool = True) -> None:
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT NOT NULL UNIQUE,
        sha256 TEXT NOT NULL,
        sha256_norm TEXT,
        bytes INTEGER NOT NULL,
        line_count INTEGER NOT NULL,
        mtime_utc TEXT,
        last_indexed_utc TEXT NOT NULL,
        language TEXT NOT NULL
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS symbols (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        fqname TEXT NOT NULL,
        kind TEXT NOT NULL,
        parent_id INTEGER,
        start_line INTEGER NOT NULL,
        end_line INTEGER NOT NULL,
        signature TEXT,
        doc_first_line TEXT,
        decorators_json TEXT
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS snippets (
        symbol_id INTEGER PRIMARY KEY,
        content TEXT NOT NULL,
        content_sha256 TEXT NOT NULL,
        truncated INTEGER NOT NULL
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS imports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER NOT NULL,
        imported TEXT NOT NULL,
        alias TEXT,
        is_from INTEGER NOT NULL,
        imported_member TEXT,
        line INTEGER NOT NULL
    );
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_symbols_fqname ON symbols(fqname);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_symbols_file_span ON symbols(file_id, start_line, end_line);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_imports_file ON imports(file_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_imports_imported ON imports(imported);")

    cur.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version', ?)", (SCHEMA_VERSION,))

    # Optional FTS5: enable if the runtime sqlite supports it; otherwise keep working without FTS.
    if enable_fts:
        try:
            cur.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS atlas_fts USING fts5("
                "path, name, fqname, snippet, "
                "tokenize='unicode61 tokenchars ''_./''', "
                "prefix='2 3 4'"
                ");"
            )
            cur.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('fts5_enabled','1')")
        except sqlite3.OperationalError:
            cur.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('fts5_enabled','0')")
    else:
        cur.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('fts5_enabled','0')")
        try:
            cur.execute("DELETE FROM atlas_fts;")
        except Exception:
            pass
    conn.commit()



def rebuild_fts(conn: sqlite3.Connection) -> bool:
    """Rebuild atlas_fts from symbols + files + snippets if FTS is enabled."""
    cur = conn.cursor()
    row = cur.execute("SELECT value FROM meta WHERE key='fts5_enabled'").fetchone()
    if not row or row[0] != "1":
        return False
    try:
        cur.execute("DELETE FROM atlas_fts;")
        cur.execute(
            "INSERT INTO atlas_fts(rowid, path, name, fqname, snippet) "
            "SELECT s.id, f.path, s.name, s.fqname, COALESCE(sn.content,'') "
            "FROM symbols s "
            "JOIN files f ON f.id = s.file_id "
            "LEFT JOIN snippets sn ON sn.symbol_id = s.id"
        )
        cur.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('fts_last_rebuilt_utc', ?)", (utc_now_z(),))
        conn.commit()
        return True
    except sqlite3.OperationalError:
        cur.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('fts5_enabled','0')")
        conn.commit()
        return False

def drop_all(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.executescript("""
    DROP TABLE IF EXISTS imports;
    DROP TABLE IF EXISTS snippets;
    DROP TABLE IF EXISTS symbols;
    DROP TABLE IF EXISTS files;
    DROP TABLE IF EXISTS meta;
    """)
    conn.commit()


def read_existing_files(conn: sqlite3.Connection) -> Dict[str, Tuple[int, str]]:
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, path, sha256 FROM files")
    except sqlite3.OperationalError:
        return {}
    out: Dict[str, Tuple[int, str]] = {}
    for fid, path, sha in cur.fetchall():
        out[path] = (int(fid), str(sha))
    return out


def delete_file_deps(conn: sqlite3.Connection, file_id: int) -> None:
    cur = conn.cursor()
    cur.execute("DELETE FROM imports WHERE file_id = ?", (file_id,))
    cur.execute("SELECT id FROM symbols WHERE file_id = ?", (file_id,))
    sym_ids = [r[0] for r in cur.fetchall()]
    if sym_ids:
        cur.executemany("DELETE FROM snippets WHERE symbol_id = ?", [(sid,) for sid in sym_ids])
    cur.execute("DELETE FROM symbols WHERE file_id = ?", (file_id,))
    conn.commit()


def upsert_file_row(
    conn: sqlite3.Connection,
    path: str,
    sha: str,
    sha_norm: Optional[str],
    nbytes: int,
    line_count: int,
    mtime: Optional[str],
    last_indexed: str,
    language: str,
    existing_id: Optional[int],
) -> int:
    cur = conn.cursor()
    if existing_id is None:
        # Defensive: if our "existing_id" cache missed an already-present row, look it up
        # to avoid UNIQUE(path) failures.
        row = cur.execute("SELECT id FROM files WHERE path=?", (path,)).fetchone()
        if row:
            existing_id = int(row[0])
        cur.execute(
            "INSERT INTO files(path, sha256, sha256_norm, bytes, line_count, mtime_utc, last_indexed_utc, language) VALUES(?,?,?,?,?,?,?,?)",
            (path, sha, sha_norm, nbytes, line_count, mtime, last_indexed, language),
        )
        conn.commit()
        return int(cur.lastrowid)
    cur.execute(
        "UPDATE files SET sha256=?, sha256_norm=?, bytes=?, line_count=?, mtime_utc=?, last_indexed_utc=?, language=? WHERE id=?",
        (sha, sha_norm, nbytes, line_count, mtime, last_indexed, language, existing_id),
    )
    conn.commit()
    return existing_id


def signature_for_function(node: ast.AST) -> Optional[str]:
    try:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            parts: List[str] = []
            for a in getattr(args, "posonlyargs", []):
                parts.append(a.arg)
            for a in args.args:
                parts.append(a.arg)
            if args.vararg:
                parts.append("*" + args.vararg.arg)
            for a in args.kwonlyargs:
                parts.append(a.arg)
            if args.kwarg:
                parts.append("**" + args.kwarg.arg)
            return f"({', '.join(parts)})"
        return None
    except Exception:
        return None


def decorators_json(node: ast.AST) -> Optional[str]:
    import json
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return None
    decs = []
    for d in getattr(node, "decorator_list", []) or []:
        try:
            decs.append(ast.unparse(d))
        except Exception:
            decs.append(d.__class__.__name__)
    if not decs:
        return None
    return json.dumps(decs, ensure_ascii=False)


def doc_first_line(node: ast.AST) -> Optional[str]:
    try:
        doc = ast.get_docstring(node)
        if not doc:
            return None
        return doc.strip().splitlines()[0][:300]
    except Exception:
        return None


def collect_imports(tree: ast.AST) -> List[Tuple[str, Optional[str], int, Optional[str], int]]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.append((a.name, a.asname, 0, None, getattr(node, "lineno", 1)))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for a in node.names:
                out.append((mod, a.asname, 1, a.name, getattr(node, "lineno", 1)))
    return out


def extract_snippet(lines: List[str], start_line: int, end_line: int, max_lines: int, max_chars: int) -> Tuple[str, int]:
    start_i = max(1, start_line) - 1
    end_i = min(len(lines), max(end_line, start_line))
    chunk = lines[start_i:end_i]
    truncated = 0
    if len(chunk) > max_lines:
        chunk = chunk[:max_lines]
        truncated = 1
    content = "\n".join(chunk)
    if len(content) > max_chars:
        content = content[:max_chars]
        truncated = 1
    return content, truncated


def infer_end_lineno(node: ast.AST) -> int:
    end = getattr(node, "end_lineno", None)
    if isinstance(end, int) and end > 0:
        return end
    ln = getattr(node, "lineno", 1)
    return int(ln) + 1


def index_python_file(
    conn: sqlite3.Connection,
    repo_root: Path,
    file_path: Path,
    file_id: int,
    max_snippet_lines: int,
    max_snippet_chars: int,
) -> None:
    rp = rel_path(repo_root, file_path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = file_path.read_text(encoding="utf-8", errors="replace")

    lines = text.splitlines()
    try:
        tree = ast.parse(text, filename=rp)
    except SyntaxError:
        return

    cur = conn.cursor()

    imps = collect_imports(tree)
    if imps:
        cur.executemany(
            "INSERT INTO imports(file_id, imported, alias, is_from, imported_member, line) VALUES(?,?,?,?,?,?)",
            [(file_id, imported, alias, is_from, mem, line) for (imported, alias, is_from, mem, line) in imps],
        )

    def walk(node: ast.AST, parent_sql_id: Optional[int], prefix: str) -> None:
        for child in getattr(node, "body", []) or []:
            if isinstance(child, ast.ClassDef):
                name = child.name
                fq = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
                start = int(getattr(child, "lineno", 1))
                end = infer_end_lineno(child)
                cur.execute(
                    "INSERT INTO symbols(file_id, name, fqname, kind, parent_id, start_line, end_line, signature, doc_first_line, decorators_json) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (file_id, name, fq, "class", parent_sql_id, start, end, None, doc_first_line(child), decorators_json(child)),
                )
                sym_id = int(cur.lastrowid)

                content, trunc = extract_snippet(lines, start, end, max_snippet_lines, max_snippet_chars)
                cur.execute(
                    "INSERT OR REPLACE INTO snippets(symbol_id, content, content_sha256, truncated) VALUES(?,?,?,?)",
                    (sym_id, content, sha256_bytes(content.encode("utf-8", errors="replace")), trunc),
                )
                walk(child, sym_id, fq)

            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = child.name
                fq = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
                start = int(getattr(child, "lineno", 1))
                end = infer_end_lineno(child)
                kind = "async_function" if isinstance(child, ast.AsyncFunctionDef) else "function"
                sig = signature_for_function(child)
                cur.execute(
                    "INSERT INTO symbols(file_id, name, fqname, kind, parent_id, start_line, end_line, signature, doc_first_line, decorators_json) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (file_id, name, fq, kind, parent_sql_id, start, end, sig, doc_first_line(child), decorators_json(child)),
                )
                sym_id = int(cur.lastrowid)

                content, trunc = extract_snippet(lines, start, end, max_snippet_lines, max_snippet_chars)
                cur.execute(
                    "INSERT OR REPLACE INTO snippets(symbol_id, content, content_sha256, truncated) VALUES(?,?,?,?)",
                    (sym_id, content, sha256_bytes(content.encode("utf-8", errors="replace")), trunc),
                )
                walk(child, sym_id, fq)

    walk(tree, None, "")
    conn.commit()


def find_all_dbs(repo_root: Path, exclude_roots: Sequence[str]) -> List[Path]:
    found: List[Path] = []
    for p in repo_root.rglob(DB_NAME):
        if not p.is_file():
            continue
        try:
            rp = rel_path(repo_root, p)
        except Exception:
            continue
        parts = rp.split("/")
        if parts and parts[0] in exclude_roots:
            continue
        found.append(p)
    found.sort(key=lambda x: x.as_posix().lower())
    return found


def backup_and_remove_duplicates(repo_root: Path, db_paths: List[Path]) -> None:
    stamp = utc_now_z().replace(":", "").replace("-", "")
    backup_root = repo_root / ".repo_nav_backups" / stamp
    backup_root.mkdir(parents=True, exist_ok=True)

    for p in db_paths:
        try:
            rp = rel_path(repo_root, p)
        except Exception:
            rp = p.name
        dest = backup_root / rp
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(p), str(dest))
        except Exception:
            try:
                shutil.copy2(str(p), str(dest))
                p.unlink(missing_ok=True)
            except Exception:
                pass


def index_repo(
    repo_root: Path,
    out_path: Path,
    exclude_roots: Sequence[str],
    exclude_globs: Sequence[str],
    *,
    rebuild: bool,
    clean_missing: bool,
    max_snippet_lines: int,
    max_snippet_chars: int,
    enable_fts: bool = True,
) -> Tuple[int, int, int]:
    conn = sqlite3.connect(str(out_path))
    conn.row_factory = sqlite3.Row

    if rebuild:
        drop_all(conn)

    ensure_schema(conn, enable_fts=enable_fts)
    existing = read_existing_files(conn)

    now = utc_now_z()
    total = 0
    changed = 0
    skipped = 0
    seen_paths = set()

    for fp in iter_repo_files(repo_root, exclude_roots, exclude_globs):
        rp = rel_path(repo_root, fp)
        seen_paths.add(rp)
        ext = fp.suffix.lower()

        try:
            data = fp.read_bytes()
        except Exception:
            continue

        sha = sha256_bytes(data)
        sha_norm = None
        line_count = 0

        language = "unknown"
        if ext in PY_EXTS:
            language = "python"
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = data.decode("utf-8", errors="replace")
            sha_norm = sha256_norm_text(text)
            line_count = text.count("\n") + 1 if text else 0
        else:
            if ext in {".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".ini"}:
                language = "text"
                try:
                    s = data.decode("utf-8")
                except UnicodeDecodeError:
                    s = data.decode("utf-8", errors="replace")
                line_count = s.count("\n") + 1 if s else 0
                sha_norm = sha256_norm_text(s)

        nbytes = len(data)
        mt = mtime_utc_z(fp)

        prev = existing.get(rp)
        prev_id = prev[0] if prev else None
        prev_sha = prev[1] if prev else None

        total += 1

        if prev_sha == sha and prev_id is not None and not rebuild:
            upsert_file_row(conn, rp, sha, sha_norm, nbytes, line_count, mt, now, language, prev_id)
            skipped += 1
            continue

        if prev_id is not None:
            delete_file_deps(conn, prev_id)

        file_id = upsert_file_row(conn, rp, sha, sha_norm, nbytes, line_count, mt, now, language, prev_id)

        if ext in PY_EXTS:
            index_python_file(conn, repo_root, fp, file_id, max_snippet_lines, max_snippet_chars)

        changed += 1

    if clean_missing:
        cur = conn.cursor()
        cur.execute("SELECT id, path FROM files")
        for r in cur.fetchall():
            if r["path"] not in seen_paths:
                delete_file_deps(conn, int(r["id"]))
                cur.execute("DELETE FROM files WHERE id = ?", (int(r["id"]),))
        conn.commit()

    # Rebuild FTS index (optional) after indexing structured tables.
    try:
        rebuild_fts(conn)
    except Exception:
        pass

    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('last_indexed_utc', ?)", (now,))
    conn.commit()
    conn.close()
    return total, changed, skipped


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--help", action="help", help="Show help and exit.")
    ap.add_argument("--repo-root", default="", help="Repository root to index (default: inferred from script location).")
    ap.add_argument("--exclude", default=",".join(DEFAULT_EXCLUDES),
                    help="Comma-separated top-level directories to exclude (default: common build/data dirs).")
    ap.add_argument("--exclude-glob", default="",
                    help="Comma-separated glob patterns to exclude (e.g. '*.min.js,docs/_build/*').")
    ap.add_argument("--clean-missing", action="store_true", help="Remove db rows for files removed from disk.")
    ap.add_argument("--max-snippet-lines", type=int, default=40)
    ap.add_argument("--max-snippet-chars", type=int, default=8000)
    ap.add_argument("--no-fts", action="store_true", help="Disable FTS even if SQLite supports it.")
    args = ap.parse_args(argv)

    script_path = Path(__file__).resolve()
    if args.repo_root:
        repo_root = Path(args.repo_root).resolve()
    else:
        # If this script lives at <repo_root>/ai_introspect/.ai_introspect/build_nav/build_nav.py,
        # infer repo_root accordingly. Otherwise keep legacy behavior.
        if script_path.parent.name == 'build_nav' and script_path.parent.parent.name == '.ai_introspect':
            try:
                repo_root = script_path.parents[3]
            except Exception:
                repo_root = script_path.parent
        else:
            repo_root = script_path.parent

    exclude_roots = [s.strip() for s in (args.exclude or "").split(",") if s.strip()]
    exclude_globs = [s.strip() for s in (args.exclude_glob or "").split(",") if s.strip()]

    dbs = find_all_dbs(repo_root, exclude_roots)
    root_db = repo_root / DB_NAME
    rebuild = False

    if len(dbs) > 1:
        backup_and_remove_duplicates(repo_root, dbs)
        rebuild = True
    elif len(dbs) == 1:
        only = dbs[0]
        if only.resolve() != root_db.resolve():
            try:
                shutil.move(str(only), str(root_db))
            except Exception:
                shutil.copy2(str(only), str(root_db))
                try:
                    only.unlink(missing_ok=True)
                except Exception:
                    pass
    else:
        rebuild = True

    total, changed, skipped = index_repo(
        repo_root=repo_root,
        out_path=root_db,
        exclude_roots=exclude_roots,
        exclude_globs=exclude_globs,
        rebuild=rebuild,
        clean_missing=args.clean_missing,
        max_snippet_lines=args.max_snippet_lines,
        max_snippet_chars=args.max_snippet_chars,
        enable_fts=(not args.no_fts),
    )

    print("============================================================")
    print("[REPO NAV BUILT + INDEXED]")
    print("============================================================")
    print(f"repo_root: {repo_root}")
    print(f"db       : {root_db}")
    print(f"schema   : {SCHEMA_VERSION}")
    print("------------------------------------------------------------")
    print(f"dbs found     : {len(dbs)}")
    print(f"rebuild       : {rebuild}")
    print(f"files seen    : {total}")
    print(f"files changed : {changed}")
    print(f"files skipped : {skipped}")
    print("============================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
