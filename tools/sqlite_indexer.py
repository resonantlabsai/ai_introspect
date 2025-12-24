
"""
SQLite Code Atlas Indexer (AI-first navigation)

Schema v1:
- files(id, path, sha256, sha256_norm, bytes, line_count, mtime_utc, last_indexed_utc, language)
- symbols(id, file_id, name, fqname, kind, parent_id, start_line, end_line, signature, doc_first_line, decorators_json)
- snippets(symbol_id, content, content_sha256, truncated)
- imports(id, file_id, imported, alias, is_from, imported_member, line)
- refs(id, file_id, from_symbol_id, ref_kind, ref_text, to_symbol_fqname, line)

This index is designed to help LLMs (and humans) navigate large repos with bounded, deterministic queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, List, Tuple, Dict, Any
import ast
import hashlib
import os
import json
import re
import sqlite3
import time


SCHEMA_VERSION = "atlas_v1"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _norm_path(p: str) -> str:
    return p.replace("\\", "/")


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _sha256_norm_text(text: str) -> str:
    # Normalize CRLF + trim trailing whitespace on each line
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    norm = "\n".join(lines).strip("\n") + "\n"
    return _sha256_bytes(norm.encode("utf-8", errors="replace"))


def ensure_db(db_path: Path, *, enable_fts: bool = True) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE IF NOT EXISTS meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS files (
          id INTEGER PRIMARY KEY,
          path TEXT UNIQUE NOT NULL,
          sha256 TEXT NOT NULL,
          sha256_norm TEXT,
          bytes INTEGER NOT NULL,
          line_count INTEGER NOT NULL,
          mtime_utc TEXT,
          last_indexed_utc TEXT NOT NULL,
          language TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS symbols (
          id INTEGER PRIMARY KEY,
          file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
          name TEXT NOT NULL,
          fqname TEXT NOT NULL,
          kind TEXT NOT NULL,
          parent_id INTEGER REFERENCES symbols(id) ON DELETE SET NULL,
          start_line INTEGER NOT NULL,
          end_line INTEGER NOT NULL,
          signature TEXT,
          doc_first_line TEXT,
          decorators_json TEXT
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_symbols_fqname ON symbols(fqname);
        CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
        CREATE INDEX IF NOT EXISTS idx_symbols_file_span ON symbols(file_id, start_line);

        CREATE TABLE IF NOT EXISTS snippets (
          symbol_id INTEGER PRIMARY KEY REFERENCES symbols(id) ON DELETE CASCADE,
          content TEXT NOT NULL,
          content_sha256 TEXT NOT NULL,
          truncated INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS imports (
          id INTEGER PRIMARY KEY,
          file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
          imported TEXT NOT NULL,
          alias TEXT,
          is_from INTEGER NOT NULL,
          imported_member TEXT,
          line INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_imports_file ON imports(file_id);
        CREATE INDEX IF NOT EXISTS idx_imports_imported ON imports(imported);

        CREATE TABLE IF NOT EXISTS refs (
          id INTEGER PRIMARY KEY,
          file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
          from_symbol_id INTEGER REFERENCES symbols(id) ON DELETE SET NULL,
          ref_kind TEXT NOT NULL,
          ref_text TEXT NOT NULL,
          to_symbol_fqname TEXT,
          line INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_refs_file ON refs(file_id);
        CREATE INDEX IF NOT EXISTS idx_refs_kind ON refs(ref_kind);
        CREATE INDEX IF NOT EXISTS idx_refs_text ON refs(ref_text);
        CREATE INDEX IF NOT EXISTS idx_refs_to_fq ON refs(to_symbol_fqname);
        """
    )
    cur.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", ("schema_version", SCHEMA_VERSION))
    if enable_fts:
        _try_enable_fts(cur)
    else:
        # Keep meta consistent: if caller requests no-FTS indexing, mark FTS disabled even if SQLite supports it.
        _meta_set(cur, "fts5_enabled", "0")
        try:
            cur.execute("DELETE FROM atlas_fts;")
        except Exception:
            pass
    con.commit()
    con.close()



# --- Optional FTS5 support -------------------------------------------------

def _meta_get(cur: sqlite3.Cursor, key: str) -> Optional[str]:
    row = cur.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def _meta_set(cur: sqlite3.Cursor, key: str, value: str) -> None:
    cur.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, value))


def _fts_enabled(cur: sqlite3.Cursor) -> bool:
    return (_meta_get(cur, "fts5_enabled") or "0") == "1"


def _try_enable_fts(cur: sqlite3.Cursor) -> bool:
    """Attempt to enable FTS5. If unavailable, record meta and return False."""
    try:
        cur.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS atlas_fts USING fts5("
            "path, name, fqname, snippet, "
            "tokenize='unicode61 tokenchars ''_./''', "
            "prefix='2 3 4'"
            ");"
        )
        _meta_set(cur, "fts5_enabled", "1")
        return True
    except sqlite3.OperationalError:
        _meta_set(cur, "fts5_enabled", "0")
        return False


def rebuild_fts(con: sqlite3.Connection) -> bool:
    """Rebuild the FTS index from structured tables. Safe to call anytime."""
    cur = con.cursor()
    if not _fts_enabled(cur):
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
        _meta_set(cur, "fts_last_rebuilt_utc", _utc_now())
        return True
    except sqlite3.OperationalError:
        # FTS table missing or unsupported: disable and continue.
        _meta_set(cur, "fts5_enabled", "0")
        return False


def safe_fts_query(user_text: str) -> str:
    """Convert a user query into a safe FTS MATCH expression (AND + prefix)."""
    tokens = re.findall(r"[A-Za-z0-9_./]+", user_text)
    if not tokens:
        return ""
    # Prefix '*' for basic autocomplete behavior.
    return " AND ".join(f"{t}*" for t in tokens)


def _rel_to_repo(repo_root: Path, p: Path) -> str:
    return _norm_path(str(p.relative_to(repo_root)))


def _safe_read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def _line_count(text: str) -> int:
    return len(text.replace("\r\n", "\n").replace("\r", "\n").split("\n"))


def _decorators(node: ast.AST) -> List[str]:
    decs = getattr(node, "decorator_list", None)
    if not decs:
        return []
    out = []
    for d in decs:
        try:
            out.append(ast.unparse(d))
        except Exception:
            out.append("<decorator>")
    return out


def _fn_signature(node: ast.AST) -> Optional[str]:
    args = getattr(node, "args", None)
    if args is None:
        return None
    try:
        return ast.unparse(args)
    except Exception:
        return None


@dataclass
class SymbolRow:
    name: str
    fqname: str
    kind: str
    parent_fqname: Optional[str]
    start_line: int
    end_line: int
    signature: Optional[str]
    doc_first_line: Optional[str]
    decorators: List[str]


def _module_name_from_rel(rel_path: str) -> str:
    if rel_path.endswith(".py"):
        rel_path = rel_path[:-3]
    return rel_path.replace("/", ".")


def extract_imports(tree: ast.AST) -> List[Tuple[str, Optional[str], int, int, Optional[str]]]:
    # returns tuples: (imported, alias, is_from, line, imported_member)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                out.append((n.name, n.asname, 0, getattr(node, "lineno", 1), None))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for n in node.names:
                out.append((mod, n.asname, 1, getattr(node, "lineno", 1), n.name))
    return out


def extract_symbols(tree: ast.AST, module_name: str) -> List[SymbolRow]:
    rows: List[SymbolRow] = []

    class V(ast.NodeVisitor):
        def __init__(self) -> None:
            self.parents: List[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> Any:
            parent = self.parents[-1] if self.parents else None
            fq = f"{parent}.{node.name}" if parent else f"{module_name}.{node.name}"
            doc = ast.get_docstring(node)
            doc1 = doc.splitlines()[0].strip() if doc else None
            end = getattr(node, "end_lineno", node.lineno)
            rows.append(SymbolRow(
                name=node.name,
                fqname=fq,
                kind="class",
                parent_fqname=parent,
                start_line=node.lineno,
                end_line=end,
                signature=None,
                doc_first_line=doc1,
                decorators=_decorators(node),
            ))
            self.parents.append(fq)
            self.generic_visit(node)
            self.parents.pop()

        def _fn(self, node: ast.AST, kind: str) -> Any:
            parent = self.parents[-1] if self.parents else None
            if parent:
                fq = f"{parent}.{node.name}"  # method or nested
            else:
                fq = f"{module_name}:{node.name}"  # top-level
            doc = ast.get_docstring(node)
            doc1 = doc.splitlines()[0].strip() if doc else None
            end = getattr(node, "end_lineno", getattr(node, "lineno", 1))
            rows.append(SymbolRow(
                name=node.name,
                fqname=fq,
                kind=kind,
                parent_fqname=parent,
                start_line=getattr(node, "lineno", 1),
                end_line=end,
                signature=_fn_signature(node),
                doc_first_line=doc1,
                decorators=_decorators(node),
            ))
            self.parents.append(fq)
            self.generic_visit(node)
            self.parents.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
            self._fn(node, "function")

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
            self._fn(node, "async_function")

    V().visit(tree)
    return rows



def _expr_to_ref_text(expr: ast.AST) -> Optional[str]:
    """Best-effort string form for a call target (Name / dotted Attribute chain)."""
    try:
        if isinstance(expr, ast.Name):
            return expr.id
        if isinstance(expr, ast.Attribute):
            parts: List[str] = []
            cur: ast.AST = expr
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            parts = list(reversed(parts))
            if parts:
                return ".".join(parts)
        return None
    except Exception:
        return None

def extract_call_refs(tree: ast.AST) -> List[Tuple[int, str]]:
    """Return (lineno, ref_text) for each call expression in the AST."""
    out: List[Tuple[int, str]] = []

    class V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> Any:
            txt = _expr_to_ref_text(node.func)
            ln = getattr(node, "lineno", None)
            if txt and isinstance(ln, int):
                out.append((ln, txt))
            self.generic_visit(node)

    V().visit(tree)
    return out

_TOKENISH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_./:-]{2,79}$")
_ENVKEY_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,79}$")

def extract_string_refs(tree: ast.AST, *, limit: int = 500) -> List[Tuple[int, str]]:
    """Return (lineno, string) for identifier-like string literals (bounded)."""
    out: List[Tuple[int, str]] = []

    class V(ast.NodeVisitor):
        def visit_Constant(self, node: ast.Constant) -> Any:
            if isinstance(node.value, str):
                s = node.value
                ln = getattr(node, "lineno", None)
                if isinstance(ln, int):
                    if "://" in s:
                        return
                    if _ENVKEY_RE.match(s or ""):
                        # treat env keys as config_key instead (see extract_config_key_refs)
                        return
                    if _TOKENISH_RE.match(s or ""):
                        out.append((ln, s))
                        if len(out) >= limit:
                            return
            # no generic_visit for Constant

        def visit_Str(self, node: ast.Str) -> Any:  # py<3.8 compatibility
            s = node.s
            ln = getattr(node, "lineno", None)
            if isinstance(ln, int):
                if "://" in s:
                    return
                if _ENVKEY_RE.match(s or ""):
                    return
                if _TOKENISH_RE.match(s or ""):
                    out.append((ln, s))
            if len(out) < limit:
                self.generic_visit(node)

    V().visit(tree)
    return out[:limit]


def extract_config_key_refs(tree: ast.AST, *, limit: int = 200) -> List[Tuple[int, str]]:
    """Return (lineno, key) for env/config key references (bounded)."""
    out: List[Tuple[int, str]] = []

    def _add(ln: Any, key: Any) -> None:
        if not isinstance(ln, int):
            return
        if not isinstance(key, str):
            return
        if not key:
            return
        if len(key) > 120:
            return
        out.append((ln, key))

    class V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> Any:
            # os.getenv("KEY") / getenv("KEY") / os.environ.get("KEY")
            fn = node.func
            ln = getattr(node, "lineno", None)

            def _first_str_arg() -> Optional[str]:
                if not node.args:
                    return None
                a0 = node.args[0]
                if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                    return a0.value
                if isinstance(a0, ast.Str):
                    return a0.s
                return None

            key = _first_str_arg()

            # Name('getenv')
            if isinstance(fn, ast.Name) and fn.id in {"getenv"}:
                if key:
                    _add(ln, key)

            # Attribute(...): os.getenv, os.environ.get
            if isinstance(fn, ast.Attribute):
                # os.getenv
                if isinstance(fn.value, ast.Name) and fn.value.id == "os" and fn.attr in {"getenv"}:
                    if key:
                        _add(ln, key)
                # os.environ.get
                if isinstance(fn.value, ast.Attribute):
                    v = fn.value
                    if (
                        isinstance(v.value, ast.Name)
                        and v.value.id == "os"
                        and v.attr == "environ"
                        and fn.attr in {"get"}
                    ):
                        if key:
                            _add(ln, key)

            if len(out) < limit:
                self.generic_visit(node)

        def visit_Subscript(self, node: ast.Subscript) -> Any:
            # os.environ["KEY"]
            ln = getattr(node, "lineno", None)
            # base: os.environ
            base = node.value
            if isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name) and base.value.id == "os" and base.attr == "environ":
                key = None
                sl = node.slice
                if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                    key = sl.value
                elif isinstance(sl, ast.Index):  # py<3.9
                    iv = sl.value
                    if isinstance(iv, ast.Constant) and isinstance(iv.value, str):
                        key = iv.value
                    elif isinstance(iv, ast.Str):
                        key = iv.s
                if key:
                    _add(ln, key)
            if len(out) < limit:
                self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute) -> Any:
            # settings.SOME_KEY or config.SOME_KEY (best-effort)
            ln = getattr(node, "lineno", None)
            if isinstance(node.value, ast.Name) and node.value.id in {"settings", "config", "cfg"}:
                _add(ln, node.attr)
            if len(out) < limit:
                self.generic_visit(node)

    V().visit(tree)
    # keep only env-like or identifier-like keys
    filtered: List[Tuple[int, str]] = []
    for ln, key in out:
        if _ENVKEY_RE.match(key) or _TOKENISH_RE.match(key) or "." in key:
            filtered.append((ln, key))
        if len(filtered) >= limit:
            break
    return filtered[:limit]
def _enclosing_symbol_id(spans: List[Tuple[int, int, int]], line: int) -> Optional[int]:
    """Pick the smallest (tightest) symbol span containing `line`. spans=(start,end,symbol_id)."""
    best: Optional[Tuple[int, int, int]] = None
    for st, en, sid in spans:
        if st <= line <= en:
            if best is None or (en - st) < (best[1] - best[0]):
                best = (st, en, sid)
    return best[2] if best else None

def _slice_lines(text: str, start: int, end: int, cap_chars: int = 65535) -> Tuple[str, int]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    start0 = max(1, start) - 1
    end0 = min(len(lines), max(end, start))  # inclusive end in AST
    chunk = "\n".join(lines[start0:end0]) + "\n"
    if len(chunk) > cap_chars:
        return chunk[:cap_chars], 1
    return chunk, 0


def index_repo(
    repo_root: Path,
    db_path: Path,
    *,
    target_paths: Optional[List[str]] = None,
    include_globs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
    max_files: Optional[int] = None,
    budget_seconds: Optional[float] = None,
    update_fts: bool = True,
) -> Dict[str, Any]:
    """
    Index repo into sqlite db.

    If target_paths is provided (relative paths), only those files are indexed.
    This function is designed to be fast and interruption-tolerant.

    Args:
      max_files: Stop after indexing N files (useful for quick runs).
      budget_seconds: Stop after spending this many seconds (soft budget).
      update_fts: If FTS is enabled, update FTS rows for files indexed in this run.

    Returns a summary dict. If the run stops early, includes "stopped" + "reason".
    """
    ensure_db(db_path)
    repo_root = Path(repo_root)
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA foreign_keys=ON;")
    cur = con.cursor()

    # Exclude directory *names* (pruned during os.walk) and exclude *path prefixes* (e.g. ".ai_introspect/repo_index").
    default_exclude = [
        "__pycache__",
        ".venv",
        "venv",
        "site-packages",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "node_modules",
        "dist",
        "build",
        "data",
        ".ai_introspect",
    ]
    # Keep common internal folders excluded explicitly.
    default_exclude_prefixes = [
        ".ai_introspect/repo_index",
        ".ai_introspect/search",
    ]

    exclude_dirs = exclude_dirs or default_exclude
    exclude_name_set = {d for d in exclude_dirs if "/" not in _norm_path(d)}
    exclude_prefixes = list(default_exclude_prefixes) + [d for d in exclude_dirs if "/" in _norm_path(d)]

    include_globs = include_globs or ["**/*.py"]

    def _skip_by_prefix(rel_norm: str) -> bool:
        for pref in exclude_prefixes:
            pref_n = _norm_path(pref).rstrip("/")
            if rel_norm == pref_n or rel_norm.startswith(pref_n + "/"):
                return True
        return False

    def _iter_files() -> List[Path]:
        """Enumerate candidate files with directory pruning."""
        # If patterns are something other than a basic python glob, fall back to glob.
        simple_py_only = (len(include_globs) == 1 and include_globs[0] == "**/*.py")
        out: List[Path] = []
        if not simple_py_only:
            for g in include_globs:
                for p in repo_root.glob(g):
                    if p.is_file():
                        rel = _rel_to_repo(repo_root, p)
                        if _skip_by_prefix(rel):
                            continue
                        if any(seg in exclude_name_set for seg in p.parts):
                            continue
                        out.append(p)
            return sorted(set(out))

        # Fast path: prune using os.walk
        for dirpath, dirnames, filenames in os.walk(str(repo_root)):
            dirnames[:] = [d for d in dirnames if d not in exclude_name_set]
            base = Path(dirpath)
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                p = base / fn
                try:
                    rel = _rel_to_repo(repo_root, p)
                except Exception:
                    continue
                if _skip_by_prefix(rel):
                    continue
                out.append(p)
        return sorted(set(out))

    # Build list of files
    if target_paths:
        files = []
        for tp in target_paths:
            p = repo_root / Path(tp)
            if p.exists() and p.is_file():
                files.append(p)
        files = sorted(set(files))
    else:
        files = _iter_files()

    # helper: upsert file row, return file_id
    def upsert_file(rel: str, b: bytes, text: str) -> int:
        sha = _sha256_bytes(b)
        sha_norm = _sha256_norm_text(text) if rel.endswith(".py") else None
        bytes_len = len(b)
        lc = _line_count(text)
        now = _utc_now()
        row = cur.execute("SELECT id FROM files WHERE path=?", (rel,)).fetchone()
        if row:
            fid = int(row[0])
            cur.execute(
                "UPDATE files SET sha256=?, sha256_norm=?, bytes=?, line_count=?, mtime_utc=?, last_indexed_utc=?, language=? WHERE id=?",
                (sha, sha_norm, bytes_len, lc, now, now, "python", fid),
            )
            return fid
        cur.execute(
            "INSERT INTO files(path, sha256, sha256_norm, bytes, line_count, mtime_utc, last_indexed_utc, language) VALUES(?,?,?,?,?,?,?,?)",
            (rel, sha, sha_norm, bytes_len, lc, now, now, "python"),
        )
        return int(cur.lastrowid)

    def clear_file(fid: int) -> None:
        cur.execute("DELETE FROM imports WHERE file_id=?", (fid,))
        cur.execute("DELETE FROM symbols WHERE file_id=?", (fid,))
        cur.execute("DELETE FROM refs WHERE file_id=?", (fid,))

    # Optional FTS update (incremental per-file).
    fts_on = bool(update_fts and _fts_enabled(cur))
    if fts_on:
        try:
            cur.execute("SELECT 1 FROM atlas_fts LIMIT 1;")
        except Exception:
            _meta_set(cur, "fts5_enabled", "0")
            fts_on = False

    sym_inserted = 0
    files_indexed = 0
    imports_inserted = 0
    refs_inserted = 0

    t0 = time.time()
    stopped = False
    stop_reason = ""

    for p in files:
        if budget_seconds is not None and (time.time() - t0) >= float(budget_seconds):
            stopped = True
            stop_reason = "budget_seconds"
            break
        if max_files is not None and files_indexed >= int(max_files):
            stopped = True
            stop_reason = "max_files"
            break

        rel = _rel_to_repo(repo_root, p)
        if not rel.endswith(".py"):
            continue

        try:
            b = p.read_bytes()
            text = _safe_read_text(p)
        except Exception:
            continue

        mod = _module_name_from_rel(rel)

        fid = upsert_file(rel, b, text)
        clear_file(fid)
        files_indexed += 1

        if fts_on:
            try:
                cur.execute("DELETE FROM atlas_fts WHERE path=?", (rel,))
            except Exception:
                _meta_set(cur, "fts5_enabled", "0")
                fts_on = False

        try:
            tree = ast.parse(text)
        except SyntaxError:
            cur.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (f"parse_error:{rel}", _utc_now()))
            continue

        pending_import_refs: List[Tuple[int, str, Optional[str]]] = []
        for imported, alias, is_from, line, member in extract_imports(tree):
            cur.execute(
                "INSERT INTO imports(file_id, imported, alias, is_from, imported_member, line) VALUES(?,?,?,?,?,?)",
                (fid, imported, alias, is_from, member, line),
            )
            imports_inserted += 1
            pending_import_refs.append((line, imported, member))

        syms = extract_symbols(tree, mod)
        fq_to_id: Dict[str, int] = {}
        pending: List[Tuple[int, Optional[str]]] = []

        spans: List[Tuple[int, int, int]] = []  # (start_line, end_line, symbol_id)
        top_name_to_fq: Dict[str, str] = {}
        for s in syms:
            cur.execute(
                "INSERT INTO symbols(file_id, name, fqname, kind, parent_id, start_line, end_line, signature, doc_first_line, decorators_json) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (fid, s.name, s.fqname, s.kind, None, s.start_line, s.end_line, s.signature, s.doc_first_line, json.dumps(s.decorators)),
            )
            sid = int(cur.lastrowid)
            fq_to_id[s.fqname] = sid
            spans.append((s.start_line, s.end_line, sid))
            if s.parent_fqname is None:
                top_name_to_fq[s.name] = s.fqname
            pending.append((sid, s.parent_fqname))
            sym_inserted += 1

            chunk, trunc = _slice_lines(text, s.start_line, s.end_line)
            cur.execute(
                "INSERT OR REPLACE INTO snippets(symbol_id, content, content_sha256, truncated) VALUES(?,?,?,?)",
                (sid, chunk, _sha256_bytes(chunk.encode("utf-8", errors="replace")), trunc),
            )

            if fts_on:
                try:
                    cur.execute(
                        "INSERT INTO atlas_fts(rowid, path, name, fqname, snippet) VALUES(?,?,?,?,?)",
                        (sid, rel, s.name, s.fqname, chunk),
                    )
                except Exception:
                    _meta_set(cur, "fts5_enabled", "0")
                    fts_on = False


        # --- refs: imports + call sites (best-effort) ---
        # Insert import refs now that symbols/spans exist (to attach from_symbol_id when possible).
        for imp_line, imp_mod, imp_member in pending_import_refs:
            ref_text = imp_member or imp_mod
            from_sid = _enclosing_symbol_id(spans, int(imp_line)) if spans else None
            cur.execute(
                "INSERT INTO refs(file_id, from_symbol_id, ref_kind, ref_text, to_symbol_fqname, line) VALUES(?,?,?,?,?,?)",
                (fid, from_sid, "import", ref_text, None, int(imp_line)),
            )
            refs_inserted += 1

        # Insert call refs. to_symbol_fqname is resolved only for same-module top-level defs (best-effort).
        for call_line, call_txt in extract_call_refs(tree):
            from_sid = _enclosing_symbol_id(spans, int(call_line)) if spans else None
            to_fq = None
            # Resolve simple name calls to top-level defs in this module (cheap + safe).
            if call_txt in top_name_to_fq:
                to_fq = top_name_to_fq[call_txt]
            cur.execute(
                "INSERT INTO refs(file_id, from_symbol_id, ref_kind, ref_text, to_symbol_fqname, line) VALUES(?,?,?,?,?,?)",
                (fid, from_sid, "call", call_txt, to_fq, int(call_line)),
            )
            

        # Insert identifier-like string refs (bounded).
        for s_line, s_txt in extract_string_refs(tree):
            from_sid = _enclosing_symbol_id(spans, int(s_line)) if spans else None
            cur.execute(
                "INSERT INTO refs(file_id, from_symbol_id, ref_kind, ref_text, to_symbol_fqname, line) VALUES(?,?,?,?,?,?)",
                (fid, from_sid, "string", s_txt, None, int(s_line)),
            )
            refs_inserted += 1

        # Insert env/config key refs (bounded).
        for k_line, k_txt in extract_config_key_refs(tree):
            from_sid = _enclosing_symbol_id(spans, int(k_line)) if spans else None
            cur.execute(
                "INSERT INTO refs(file_id, from_symbol_id, ref_kind, ref_text, to_symbol_fqname, line) VALUES(?,?,?,?,?,?)",
                (fid, from_sid, "config_key", k_txt, None, int(k_line)),
            )
            refs_inserted += 1

        refs_inserted += 1

        for sid, pfq in pending:
            if pfq and pfq in fq_to_id:
                cur.execute("UPDATE symbols SET parent_id=? WHERE id=?", (fq_to_id[pfq], sid))

    cur.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", ("last_indexed_utc", _utc_now()))
    if fts_on:
        _meta_set(cur, "fts_last_updated_utc", _utc_now())

    con.commit()
    con.close()

    out: Dict[str, Any] = {
        "db_path": str(db_path),
        "files_indexed": files_indexed,
        "symbols_indexed": sym_inserted,
        "imports_indexed": imports_inserted,
        "refs_indexed": refs_inserted,
        "schema_version": SCHEMA_VERSION,
        "elapsed_seconds": round(time.time() - t0, 3),
        "fts_updated": bool(fts_on),
    }
    if stopped:
        out["stopped"] = True
        out["reason"] = stop_reason
    return out
