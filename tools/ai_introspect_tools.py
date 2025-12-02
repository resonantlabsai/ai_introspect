"""Core helpers for the .ai_introspect/ workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any, Optional
from collections import defaultdict, deque
from datetime import datetime
import re
import ast
import json
import hashlib


DEFAULT_DIRNAME = ".ai_introspect"


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_root(cache_root: Optional[str | Path] = None) -> Path:
    if cache_root is not None:
        root = Path(cache_root)
        _ensure_dir(root)
        return root
    root = Path.cwd() / DEFAULT_DIRNAME
    _ensure_dir(root)
    return root


def _modules_dir(root: Path) -> Path:
    return _ensure_dir(root / "modules")


def _regex_dir(root: Path) -> Path:
    return _ensure_dir(root / "regex")


def _repo_index_dir(root: Path) -> Path:
    return _ensure_dir(root / "repo_index")


def _logs_dir(root: Path) -> Path:
    return _ensure_dir(root / "logs")


def _logs_conv_dir(root: Path) -> Path:
    return _ensure_dir(root / "logs" / "conv")


def _logs_runs_dir(root: Path) -> Path:
    return _ensure_dir(root / "logs" / "runs")


def _search_dir(root: Path) -> Path:
    return _ensure_dir(root / "search")


def _errors_dir(root: Path) -> Path:
    return _ensure_dir(root / "errors")


def read_lines(path: str | Path, encoding: str = "utf-8") -> List[str]:
    p = Path(path)
    with p.open("r", encoding=encoding) as f:
        return [line.rstrip("\n") for line in f]


def iter_lines_with_numbers(lines: Iterable[str], start: int = 1) -> Iterable[Tuple[int, str]]:
    for i, line in enumerate(lines, start=start):
        yield i, line


def regex_context(
    path: str | Path,
    pattern: str,
    *,
    lines: int = 3,
    direction: str = "both",
    flags: int = re.MULTILINE,
    encoding: str = "utf-8",
) -> List[Tuple[int, List[str]]]:
    if direction not in {"both", "forward", "backward"}:
        raise ValueError("direction must be 'both', 'forward', or 'backward'")
    if lines < 0:
        raise ValueError("lines must be non-negative")
    raw_lines = read_lines(path, encoding=encoding)
    text = "\n".join(raw_lines)
    matches = list(re.finditer(pattern, text, flags))
    if not matches:
        return []
    results: List[Tuple[int, List[str]]] = []
    line_start_offsets: List[int] = []
    offset = 0
    for line in raw_lines:
        line_start_offsets.append(offset)
        offset += len(line) + 1

    def char_index_to_line_no(char_idx: int) -> int:
        line_no = 1
        for i, start in enumerate(line_start_offsets):
            if char_idx < start:
                break
            line_no = i + 1
        return line_no

    for m in matches:
        match_line = char_index_to_line_no(m.start())
        if direction == "both":
            start = max(1, match_line - lines)
            end = min(len(raw_lines), match_line + lines)
        elif direction == "forward":
            start = match_line
            end = min(len(raw_lines), match_line + lines)
        else:
            start = max(1, match_line - lines)
            end = match_line
        block = raw_lines[start - 1 : end]
        results.append((start, block))
    return results

def _leading_spaces(line: str) -> int:
    """Return the number of leading spaces on a line (tabs treated as 4 spaces)."""
    # Keep it simple: just count spaces; most of your code is space-indented.
    count = 0
    for ch in line:
        if ch == " ":
            count += 1
        elif ch == "\t":
            count += 4
        else:
            break
    return count


def print_regex_function(
    path: str | Path,
    pattern: str,
    *,
    flags: int = re.MULTILINE,
    chunk_size: int = 40,
    encoding: str = "utf-8",
) -> None:
    """
    Find and print Python functions whose `def` line matches the given pattern.

    - `pattern` is matched against the *def line* (`def name(...):`).
    - The function body is inferred by indentation:
      from the def line down to the next `def`/`class` at the same or lower indent.
    - Long functions are printed in chunks of `chunk_size` lines to avoid giant outputs.
    """
    path = Path(path)
    if not path.exists():
        print(f"[ai_introspect] File not found: {path}")
        return

    text = path.read_text(encoding=encoding)
    lines = text.splitlines()
    n_lines = len(lines)

    regex = re.compile(pattern, flags)

    # Find all def-line matches
    def_indices: list[int] = []
    for i, line in enumerate(lines):
        # crude but effective: only consider lines that look like a def
        if "def " not in line:
            continue
        if regex.search(line):
            def_indices.append(i)

    if not def_indices:
        print(f"[ai_introspect] No function definitions matching /{pattern}/ in {path}")
        return

    print(
        f"[ai_introspect] Functions in {path} matching /{pattern}/ "
        f"(total: {len(def_indices)})"
    )
    print("-" * 80)

    for idx, start_idx in enumerate(def_indices, 1):
        def_line = lines[start_idx]
        base_indent = _leading_spaces(def_line)

        # Find end of function by scanning forward
        end_idx = n_lines
        for j in range(start_idx + 1, n_lines):
            line = lines[j]
            stripped = line.lstrip()
            if not stripped:
                # blank lines are fine; continue
                continue
            # A new top-level or sibling def/class ends the current function
            if (stripped.startswith("def ") or stripped.startswith("class ")) and \
               _leading_spaces(line) <= base_indent:
                end_idx = j
                break

        func_lines = lines[start_idx:end_idx]
        total = len(func_lines)

        print(f"# Function match {idx}: lines {start_idx + 1}–{end_idx} (len={total})")
        # Chunked printing
        if total <= chunk_size:
            for k, line in enumerate(func_lines, start=start_idx + 1):
                print(f"{k:5d}: {line}")
        else:
            # Print in chunks
            for offset in range(0, total, chunk_size):
                chunk = func_lines[offset:offset + chunk_size]
                first_line_no = start_idx + 1 + offset
                last_line_no = first_line_no + len(chunk) - 1
                print(
                    f"[ai_introspect] Chunk {offset // chunk_size + 1} "
                    f"({first_line_no}–{last_line_no} of {start_idx + 1}–{end_idx})"
                )
                for k, line in enumerate(chunk, start=first_line_no):
                    print(f"{k:5d}: {line}")
                print("-" * 40)

        print("-" * 80)

def print_regex_context(
    path: str | Path,
    pattern: str,
    *,
    lines: int = 3,
    direction: str = "both",
    flags: int = re.MULTILINE,
    encoding: str = "utf-8",
    max_width: int = 120,
) -> None:
    matches = regex_context(path, pattern, lines=lines, direction=direction, flags=flags, encoding=encoding)
    p = Path(path)
    if not matches:
        print(f"[ai_introspect] No matches for /{pattern}/ in {p}")
        return
    print(f"[ai_introspect] Matches for /{pattern}/ in {p}:")
    print("-" * 80)
    for idx, (start_line, block) in enumerate(matches, start=1):
        print(f"# Match {idx} (around line {start_line}):")
        for line_no, raw in iter_lines_with_numbers(block, start=start_line):
            shown = raw if len(raw) <= max_width else raw[: max_width - 3] + "..."
            print(f"{line_no:5d}: {shown}")
        print("-" * 80)


def _call_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts = []
        cur: ast.AST | None = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        parts.reverse()
        return ".".join(parts)
    return None


def print_python_symbols(
    path: str | Path,
    *,
    show_imports: bool = True,
    show_functions: bool = True,
    show_decorators: bool = True,
    show_calls: bool = False,
    min_call_count: int = 1,
    encoding: str = "utf-8",
    max_width: int = 120,
) -> None:
    p = Path(path)
    text = p.read_text(encoding=encoding)
    tree = ast.parse(text, filename=str(p))
    imports: List[Tuple[int, str]] = []
    funcs: List[Tuple[int, str]] = []
    decorators: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    calls: Dict[str, List[int]] = defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            lineno = getattr(node, "lineno", None)
            for alias in node.names:
                name = alias.name
                asname = alias.asname
                disp = f"import {name} as {asname}" if asname else f"import {name}"
                if lineno is not None:
                    imports.append((lineno, disp))
        elif isinstance(node, ast.ImportFrom):
            lineno = getattr(node, "lineno", None)
            module = node.module or ""
            for alias in node.names:
                name = alias.name
                asname = alias.asname
                disp = f"from {module} import {name} as {asname}" if asname else f"from {module} import {name}"
                if lineno is not None:
                    imports.append((lineno, disp))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lineno = getattr(node, "lineno", None)
            func_name = node.name
            if lineno is not None:
                funcs.append((lineno, func_name))
            if show_decorators and node.decorator_list:
                for dec in node.decorator_list:
                    if hasattr(ast, "unparse"):
                        dec_raw = ast.unparse(dec)
                    else:
                        dec_raw = repr(dec)
                    dec_name = _call_name(dec) or dec_raw
                    if lineno is not None:
                        decorators[func_name].append((lineno, dec_name))
        if isinstance(node, ast.Call) and show_calls:
            lineno = getattr(node, "lineno", None)
            name = _call_name(node.func)
            if name and lineno is not None:
                calls[name].append(lineno)

    print(f"[ai_introspect] Python symbols in {p}")
    print("=" * 80)
    if show_imports:
        print("\n[Imports]")
        if not imports:
            print("  (none)")
        else:
            for lineno, disp in sorted(imports, key=lambda t: t[0]):
                if len(disp) > max_width:
                    disp = disp[: max_width - 3] + "..."
                print(f"  L{lineno:4d}: {disp}")
    if show_functions:
        print("\n[Functions]")
        if not funcs:
            print("  (none)")
        else:
            for lineno, name in sorted(funcs, key=lambda t: t[0]):
                print(f"  L{lineno:4d}: def {name}(...):")
    if show_decorators:
        print("\n[Decorators / hooks]")
        if not decorators:
            print("  (none)")
        else:
            for func_name, decs in decorators.items():
                for lineno, dec_name in decs:
                    if len(dec_name) > max_width:
                        dec_name = dec_name[: max_width - 3] + "..."
                    print(f"  L{lineno:4d}: @{dec_name}  ->  {func_name}()")
    if show_calls:
        print("\n[Calls]")
        filtered = {name: sorted(lns) for name, lns in calls.items() if len(lns) >= min_call_count}
        if not filtered:
            print("  (none)")
        else:
            for name, linenos in sorted(filtered.items(), key=lambda t: (-len(t[1]), t[0])):
                sample = ", ".join(f"{ln}" for ln in linenos[:8])
                if len(linenos) > 8:
                    sample += ", ..."
                print(f"  {name}  (count={len(linenos)}): lines {sample}")


def _write_json(path: Path, data: Any) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _manifest_path(root: Path) -> Path:
    return _search_dir(root) / "artifacts_manifest.json"


def _load_manifest(root: Path) -> Dict[str, Any]:
    path = _manifest_path(root)
    if not path.exists():
        return {"schema_version": "1.0", "artifacts": []}
    return _read_json(path)


def register_artifact(
    rel_path: str,
    artifact_type: str,
    *,
    source_path: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    cache_root: Optional[str | Path] = None,
) -> None:
    root = _get_root(cache_root)
    manifest = _load_manifest(root)
    entry_id = rel_path.replace("\\", "/")
    tags = tags or {}
    updated = False
    for art in manifest.get("artifacts", []):
        if art.get("id") == entry_id:
            art.update(
                {
                    "artifact_type": artifact_type,
                    "path": rel_path,
                    "source_path": source_path,
                    "created_at": _now_iso(),
                    "tags": tags,
                }
            )
            updated = True
            break
    if not updated:
        manifest.setdefault("artifacts", []).append(
            {
                "id": entry_id,
                "artifact_type": artifact_type,
                "path": rel_path,
                "source_path": source_path,
                "created_at": _now_iso(),
                "tags": tags,
            }
        )
    _write_json(_manifest_path(root), manifest)


def search_artifacts(
    *,
    artifact_type: Optional[str] = None,
    source_contains: Optional[str] = None,
    tag_equals: Optional[Dict[str, str]] = None,
    latest_only: bool = False,
    cache_root: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    root = _get_root(cache_root)
    manifest = _load_manifest(root)
    arts = manifest.get("artifacts", [])

    def match(art: Dict[str, Any]) -> bool:
        if artifact_type and art.get("artifact_type") != artifact_type:
            return False
        if source_contains and source_contains not in (art.get("source_path") or ""):
            return False
        if tag_equals:
            tags = art.get("tags", {}) or {}
            for k, v in tag_equals.items():
                if tags.get(k) != v:
                    return False
        return True

    filtered = [a for a in arts if match(a)]
    if latest_only and filtered:
        filtered.sort(key=lambda a: a.get("created_at") or "")
        return [filtered[-1]]
    return filtered


def build_python_symbol_index(
    path: str | Path,
    *,
    cache_root: Optional[str | Path] = None,
    encoding: str = "utf-8",
    include_calls: bool = True,
) -> Path:
    root = _get_root(cache_root)
    modules_root = _modules_dir(root)
    p = Path(path).resolve()
    text = p.read_text(encoding=encoding)
    tree = ast.parse(text, filename=str(p))
    imports: List[Dict[str, Any]] = []
    functions: List[Dict[str, Any]] = []
    decorators: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    calls: Dict[str, List[int]] = defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            lineno = getattr(node, "lineno", None)
            for alias in node.names:
                imports.append({"lineno": lineno, "type": "import", "module": alias.name, "asname": alias.asname})
        elif isinstance(node, ast.ImportFrom):
            lineno = getattr(node, "lineno", None)
            module = node.module or ""
            for alias in node.names:
                imports.append({
                    "lineno": lineno,
                    "type": "from",
                    "module": module,
                    "name": alias.name,
                    "asname": alias.asname,
                })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lineno = getattr(node, "lineno", None)
            func_name = node.name
            functions.append({"lineno": lineno, "name": func_name, "async": isinstance(node, ast.AsyncFunctionDef)})
            if node.decorator_list:
                for dec in node.decorator_list:
                    if hasattr(ast, "unparse"):
                        dec_raw = ast.unparse(dec)
                    else:
                        dec_raw = repr(dec)
                    dec_name = _call_name(dec) or dec_raw
                    decorators[func_name].append({"lineno": lineno, "decorator": dec_name})
        if isinstance(node, ast.Call) and include_calls:
            lineno = getattr(node, "lineno", None)
            name = _call_name(node.func)
            if name and lineno is not None:
                calls[name].append(lineno)

    artifact: Dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "symbols",
        "source_path": str(p),
        "created_at": _now_iso(),
        "imports": imports,
        "functions": functions,
        "decorators": decorators,
        "calls": calls,
    }
    try:
        repo_root = Path.cwd().resolve()
        rel = p.relative_to(repo_root)
    except ValueError:
        rel = p.name
    out_path = modules_root / Path(rel).with_suffix(".symbols.json")
    _write_json(out_path, artifact)
    rel_path = out_path.relative_to(root).as_posix()
    register_artifact(rel_path, "symbols", source_path=str(p), cache_root=root)
    return out_path


def build_regex_index(
    path: str | Path,
    pattern: str,
    *,
    lines: int = 3,
    direction: str = "both",
    flags: int = re.MULTILINE,
    encoding: str = "utf-8",
    cache_root: Optional[str | Path] = None,
    pattern_name: Optional[str] = None,
) -> Path:
    root = _get_root(cache_root)
    regex_root = _regex_dir(root)
    p = Path(path).resolve()
    matches = regex_context(p, pattern, lines=lines, direction=direction, flags=flags, encoding=encoding)
    blocks = []
    for start_line, block_lines in matches:
        blocks.append({"start_line": start_line, "lines": block_lines})
    artifact: Dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "regex",
        "source_path": str(p),
        "created_at": _now_iso(),
        "pattern": pattern,
        "pattern_name": pattern_name,
        "lines": lines,
        "direction": direction,
        "flags": int(flags),
        "matches": blocks,
    }
    safe_pat = re.sub(r"[^a-zA-Z0-9_.-]+", "_", pattern_name or pattern)[:40] or "pattern"
    try:
        repo_root = Path.cwd().resolve()
        rel = p.relative_to(repo_root)
    except ValueError:
        rel = p.name
    out_path = regex_root / (Path(rel).stem + f".regex.{safe_pat}.json")
    _write_json(out_path, artifact)
    rel_path = out_path.relative_to(root).as_posix()
    tags = {"pattern_name": pattern_name or "", "raw_pattern": pattern}
    register_artifact(rel_path, "regex", source_path=str(p), tags=tags, cache_root=root)
    return out_path


def _hash_file(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _is_indexable_file(path: Path) -> bool:
    """Check if a file should be indexed."""
    parts = path.parts
    if any(part.startswith('.') for part in parts):
        return False
    return True


def build_file_hash_index(
    root_dir: str | Path = ".",
    *,
    cache_root: Optional[str | Path] = None,
) -> Path:
    ai_root = _get_root(cache_root)
    repo_root_dir = Path(root_dir).resolve()
    repo_index_dir = _repo_index_dir(ai_root)
    files_info: Dict[str, Dict[str, Any]] = {}
    now = _now_iso()
    for file_path in repo_root_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if not _is_indexable_file(file_path):
            continue
        rel_path = file_path.relative_to(repo_root_dir).as_posix()
        files_info[rel_path] = {"hash": _hash_file(file_path), "last_indexed_at": now}
    artifact = {
        "schema_version": "1.0",
        "artifact_type": "file_hash_index",
        "source_path": str(repo_root_dir),
        "created_at": now,
        "files": files_info,
    }
    out_path = repo_index_dir / "file_hashes.json"
    _write_json(out_path, artifact)
    rel_path = out_path.relative_to(ai_root).as_posix()
    register_artifact(rel_path, "file_hash_index", source_path=str(repo_root_dir), cache_root=ai_root)
    return out_path


def list_modified_since_last_index(
    root_dir: str | Path = ".",
    *,
    cache_root: Optional[str | Path] = None,
) -> List[str]:
    ai_root = _get_root(cache_root)
    repo_root_dir = Path(root_dir).resolve()
    repo_index_dir = _repo_index_dir(ai_root)
    hash_path = repo_index_dir / "file_hashes.json"
    current_hashes: Dict[str, str] = {}
    for file_path in repo_root_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if not _is_indexable_file(file_path):
            continue
        rel_path = file_path.relative_to(repo_root_dir).as_posix()
        current_hashes[rel_path] = _hash_file(file_path)
    if not hash_path.exists():
        return sorted(current_hashes.keys())
    data = _read_json(hash_path)
    old_files = data.get("files", {})
    modified: List[str] = []
    for rel_path, h in current_hashes.items():
        old_entry = old_files.get(rel_path)
        if not old_entry or old_entry.get("hash") != h:
            modified.append(rel_path)
    return sorted(modified)


def append_log_entry(
    text: str,
    *,
    role: str,
    turn_id: str,
    tags: Optional[List[str]] = None,
    cache_root: Optional[str | Path] = None,
) -> Path:
    root = _get_root(cache_root)
    logs_root = _logs_dir(root)
    conv_root = _logs_conv_dir(root)
    _logs_runs_dir(root)
    ts = _now_iso().replace(":", "-")
    fname = f"{turn_id}_{role}_{ts}.txt"
    out_path = conv_root / fname
    out_path.write_text(text, encoding="utf-8")
    index_path = logs_root / "logs_index.json"
    if index_path.exists():
        idx = _read_json(index_path)
    else:
        idx = {"schema_version": "1.0", "turns": [], "runs": []}
    idx.setdefault("turns", []).append(
        {
            "id": turn_id,
            "path": out_path.relative_to(root).as_posix(),
            "role": role,
            "created_at": _now_iso(),
            "tags": tags or [],
        }
    )
    _write_json(index_path, idx)
    rel_path = out_path.relative_to(root).as_posix()
    register_artifact(rel_path, "log_turn", source_path=None, tags={"role": role}, cache_root=root)
    return out_path


def record_run_summary(
    summary: Dict[str, Any],
    *,
    run_id: str,
    cache_root: Optional[str | Path] = None,
) -> Path:
    root = _get_root(cache_root)
    logs_root = _logs_dir(root)
    runs_root = _logs_runs_dir(root)
    ts = _now_iso().replace(":", "-")
    fname = f"{run_id}_{ts}.json"
    out_path = runs_root / fname
    payload = {
        "schema_version": "1.0",
        "artifact_type": "run_summary",
        "created_at": _now_iso(),
        "run_id": run_id,
        "summary": summary,
    }
    _write_json(out_path, payload)
    index_path = logs_root / "logs_index.json"
    if index_path.exists():
        idx = _read_json(index_path)
    else:
        idx = {"schema_version": "1.0", "turns": [], "runs": []}
    idx.setdefault("runs", []).append(
        {
            "id": run_id,
            "path": out_path.relative_to(root).as_posix(),
            "created_at": _now_iso(),
            "tags": summary.get("tags", []),
        }
    )
    _write_json(index_path, idx)
    rel_path = out_path.relative_to(root).as_posix()
    register_artifact(rel_path, "run_summary", source_path=None, cache_root=root)
    return out_path


def search_logs(
    pattern: str,
    *,
    roles: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    max_hits: int = 20,
    cache_root: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    root = _get_root(cache_root)
    logs_root = _logs_dir(root)
    conv_root = _logs_conv_dir(root)
    index_path = logs_root / "logs_index.json"
    if not index_path.exists():
        return []
    idx = _read_json(index_path)
    turns = idx.get("turns", [])
    roles_set = set(roles or [])
    tags_set = set(tags or [])
    results: List[Dict[str, Any]] = []
    regex_obj = re.compile(pattern, re.MULTILINE)
    for turn in turns:
        if roles_set and turn.get("role") not in roles_set:
            continue
        if tags_set:
            turn_tags = set(turn.get("tags") or [])
            if not tags_set.intersection(turn_tags):
                continue
        rel_path = turn.get("path")
        if not rel_path:
            continue
        path = root / rel_path
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for m in regex_obj.finditer(text):
            line_no = text.count("\n", 0, m.start()) + 1
            lines = text.splitlines()
            snippet = lines[line_no - 1][:200] if 0 <= line_no - 1 < len(lines) else ""
            results.append(
                {
                    "turn_id": turn.get("id"),
                    "path": rel_path,
                    "role": turn.get("role"),
                    "created_at": turn.get("created_at"),
                    "line_no": line_no,
                    "snippet": snippet,
                }
            )
            if len(results) >= max_hits:
                return results
    return results
    
def _load_symbol_artifacts(cache_root: str | Path | None = None) -> list[dict]:
    """Load all symbol artifacts recorded in the manifest.

    Handles both:
      - search_artifacts(...) -> List[Dict[...]]
      - search_artifacts(...) -> List[str] (paths)

    and attaches:
      - _meta: original manifest entry
      - _artifact_path: absolute path to the JSON on disk
    """
    root = _get_root(cache_root)
    artifacts = search_artifacts(artifact_type="symbols", cache_root=cache_root)
    results: list[dict] = []

    for meta in artifacts:
        # Normalize to (rel_path, meta_dict)
        if isinstance(meta, str):
            rel_path = meta
            meta_dict = {"path": rel_path}
        elif isinstance(meta, dict):
            rel_path = meta.get("path")
            meta_dict = meta
        else:
            # Unknown shape; skip
            continue

        if not rel_path:
            continue

        path = root / rel_path
        if not path.exists():
            continue

        try:
            data = _read_json(path)
            # We only care about dict-like JSON artifacts
            if not isinstance(data, dict):
                continue
            # Attach metadata for convenience
            data.setdefault("_meta", meta_dict)
            data["_artifact_path"] = str(path)
            results.append(data)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[ai_introspect] Failed to read symbol artifact {path}: {exc!r}")

    return results


def scan_symbol_dependencies(
    symbol: str,
    *,
    direction: str = "both",
    max_depth: int = 3,
    cache_root: str | Path | None = None,
) -> dict:
    """
    Trace callers and callees for a given symbol using prebuilt symbol indexes.

    - `symbol`: function/class name to trace (e.g. "run_lab_tick").
    - `direction`: "backward", "forward", or "both".
    - `max_depth`: how far to walk callers/callees.

    Returns a dict:
      {
        "root": <symbol>,
        "backward": { depth -> [names...] },
        "forward":  { depth -> [names...] },
        "function_info": { name -> (source_path, lineno) },
      }
    """
    direction = direction.lower()
    if direction not in {"backward", "forward", "both"}:
        raise ValueError("direction must be 'backward', 'forward', or 'both'")

    artifacts = _load_symbol_artifacts(cache_root=cache_root)

    # Build maps:
    #   func -> (source_path, lineno)
    #   caller -> set(callees)
    #   callee -> set(callers)
    func_info: dict[str, tuple[str, int | None]] = {}
    callees: dict[str, set[str]] = defaultdict(set)
    callers: dict[str, set[str]] = defaultdict(set)

    for art in artifacts:
        source_path = art.get("source_path") or art.get("_meta", {}).get("source_path")
        if not source_path:
            continue

        # Build function info and line ranges
        functions = art.get("functions", [])
        func_ranges: list[tuple[str, int | None]] = []
        
        for fn in functions:
            name = fn.get("name")
            lineno = fn.get("lineno")
            if not name:
                continue
            func_info[name] = (source_path, lineno)
            func_ranges.append((name, lineno))
        
        # Sort functions by line number for range detection
        func_ranges_sorted = sorted(
            [(name, ln) for name, ln in func_ranges if ln is not None],
            key=lambda x: x[1]
        )

        # Process calls: calls is {callee_name: [line_numbers]}
        calls_dict = art.get("calls", {})
        if isinstance(calls_dict, dict):
            for callee, line_numbers in calls_dict.items():
                if not isinstance(line_numbers, list):
                    continue
                
                # For each call location, find the enclosing function (caller)
                for lineno in line_numbers:
                    # Find which function this line belongs to
                    caller = None
                    for i, (func_name, func_line) in enumerate(func_ranges_sorted):
                        if lineno >= func_line:
                            # Check if there's a next function
                            if i + 1 < len(func_ranges_sorted):
                                next_func_line = func_ranges_sorted[i + 1][1]
                                if lineno < next_func_line:
                                    caller = func_name
                                    break
                            else:
                                # Last function in file
                                caller = func_name
                                break
                    
                    if caller:
                        callees[caller].add(callee)
                        callers[callee].add(caller)

    result: dict[str, dict] = {"root": symbol, "backward": {}, "forward": {}}

    # Backward: callers-of-callers
    if direction in {"backward", "both"}:
        visited: set[str] = set()
        q: deque[tuple[str, int]] = deque()
        q.append((symbol, 0))
        layers: dict[int, list[str]] = defaultdict(list)

        while q:
            name, depth = q.popleft()
            if depth > max_depth:
                continue
            if name in visited:
                continue
            visited.add(name)
            layers[depth].append(name)
            for parent in callers.get(name, ()):
                if parent not in visited:
                    q.append((parent, depth + 1))

        result["backward"] = dict(sorted(layers.items()))

    # Forward: callees-of-callees
    if direction in {"forward", "both"}:
        visited_f: set[str] = set()
        qf: deque[tuple[str, int]] = deque()
        qf.append((symbol, 0))
        layers_f: dict[int, list[str]] = defaultdict(list)

        while qf:
            name, depth = qf.popleft()
            if depth > max_depth:
                continue
            if name in visited_f:
                continue
            visited_f.add(name)
            layers_f[depth].append(name)
            for child in callees.get(name, ()):
                if child not in visited_f:
                    qf.append((child, depth + 1))

        result["forward"] = dict(sorted(layers_f.items()))

    # Attach function metadata for convenience
    result["function_info"] = func_info
    return result


def print_symbol_dependencies(
    symbol: str,
    *,
    direction: str = "both",
    max_depth: int = 3,
    cache_root: str | Path | None = None,
) -> None:
    """Pretty-print the dependency stack for a symbol."""
    info = scan_symbol_dependencies(
        symbol,
        direction=direction,
        max_depth=max_depth,
        cache_root=cache_root,
    )

    if not isinstance(info, dict):
        # ultra-defensive guard; shouldn't happen, but avoids 'str.get' crashes
        print(f"[ai_introspect] Unexpected result from scan_symbol_dependencies: {type(info)}")
        print(info)
        return

    raw_func_info = info.get("function_info", {})

    if isinstance(raw_func_info, dict):
        func_info = raw_func_info
    else:
        print(
            f"[ai_introspect] function_info is not a dict; "
            f"got {type(raw_func_info)}. Falling back to empty mapping."
        )
        func_info = {}


    print(
        f"[ai_introspect] Dependency stack for symbol '{symbol}' "
        f"(direction={direction}, max_depth={max_depth})"
    )
    print("-" * 80)

    def _fmt(name: str) -> str:
        meta = func_info.get(name)
        if not meta:
            return f"{name}  [<unknown>]"
        path, lineno = meta
        if lineno is None:
            return f"{name}  [{path}]"
        return f"{name}  [{path}:{lineno}]"

    if direction in {"backward", "both"}:
        print("Backward (callers):")
        back = info.get("backward", {}) or {}
        if not back:
            print("  <no callers found>")
        else:
            for depth, names in sorted(back.items()):
                pretty = ", ".join(_fmt(n) for n in names)
                print(f"  d={depth}: {pretty}")
        print("-" * 80)

    if direction in {"forward", "both"}:
        print("Forward (callees):")
        fwd = info.get("forward", {}) or {}
        if not fwd:
            print("  <no callees found>")
        else:
            for depth, names in sorted(fwd.items()):
                pretty = ", ".join(_fmt(n) for n in names)
                print(f"  d={depth}: {pretty}")
        print("-" * 80)