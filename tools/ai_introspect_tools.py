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

# ---- Repo washer (drift detector) ----
def wash_repo(*args, **kwargs):
    """Compare a local repo tree with an uploaded zip. Optionally drill to def/class/method drift."""
    from .repo_washer import wash_repo as _impl
    return _impl(*args, **kwargs)


# ==============================
# SQLite-backed navigation (Code Atlas)
# ==============================

# ==============================
# SQLite-backed navigation (Code Atlas)
# ==============================

from pathlib import Path as _Path
from typing import Optional as _Optional, List as _List, Dict as _Dict, Any as _Any
import sqlite3 as _sqlite3
import py_compile as _py_compile
import zipfile as _zipfile
import time as _time

from tools.sqlite_indexer import index_repo as _index_repo, ensure_db as _ensure_db


def _default_db_path(repo_root: str) -> str:
    return str(_Path(repo_root) / ".ai_introspect" / "repo_nav.sqlite")


def nav(
    query: str,
    *,
    repo_root: str,
    limit: int = 20,
    db_path: _Optional[str] = None,
    explain: bool = False,
    highlight: bool = False,
) -> _Dict[str, _Any]:
    """
    Query the SQLite code atlas for symbols/files.

    Query formats:
      - "fq:<fqname>"         exact fqname match
      - "symbol:<name>"       name match
      - "path:<pathfrag>"     file path substring
      - "ref:<text>"          find symbols/files that reference text via refs table (imports/calls)
      - "callers:<target>"    find call sites that appear to call <target> (best-effort)
      - "callees:<source>"    find call sites *inside* <source> that call other targets (best-effort)
      - "impact:<target>"     summarize files that reference a target (impact analysis)
      - "fts:<raw>"           raw FTS MATCH query (only if FTS is enabled)
      - otherwise: auto mode (FTS if available; otherwise LIKE fallback)

    Notes:
      - When `highlight=True`, adds `display_*` fields with [[token]] emphasis.
      - When `explain=True`, adds `score`, `why`, and extra metadata about how results were produced.

    Returns a dict with 'matches' list (stable match objects).
    """
    db_path = db_path or _default_db_path(repo_root)
    p = _Path(db_path)
    if not p.exists():
        return {"ok": False, "error": f"sqlite db not found: {db_path}", "matches": []}

    con = _sqlite3.connect(db_path)
    cur = con.cursor()

    q = (query or "").strip()
    mode = "auto"
    val = q
    if ":" in q:
        head, tail = q.split(":", 1)
        if head in {"fq", "symbol", "path", "fts", "ref", "callers", "callees", "impact"}:
            mode, val = head, tail.strip()


    # token list used by explain/highlight (set per-mode)
    toks: _List[str] = []

    def _is_fts_enabled() -> bool:
        try:
            row = cur.execute("SELECT value FROM meta WHERE key=?", ("fts5_enabled",)).fetchone()
            if not row or row[0] != "1":
                return False
            t = cur.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='atlas_fts'"
            ).fetchone()
            return bool(t)
        except Exception:
            return False

    def _tokens(s: str) -> _List[str]:
        toks = re.findall(r"[A-Za-z0-9_./]+", s or "")
        # Keep order, drop duplicates (stable)
        seen = set()
        out = []
        for t in toks:
            tl = t.lower()
            if tl in seen:
                continue
            seen.add(tl)
            out.append(t)
        return out

    def _safe_fts(user_text: str) -> str:
        toks = _tokens(user_text)
        if not toks:
            return ""
        return " AND ".join(f"{t}*" for t in toks)

    def _apply_highlight(s: _Optional[str], toks: _List[str]) -> _Optional[str]:
        if not highlight or not s or not toks:
            return s
        out = s
        # Highlight longest tokens first to reduce nested highlighting
        for t in sorted(toks, key=len, reverse=True):
            if len(t) < 2:
                continue
            # Wrap first occurrence only (readability)
            pat = re.compile(re.escape(t), re.IGNORECASE)
            out, n = pat.subn(lambda m: f"[[{m.group(0)}]]", out, count=1)
        return out

    def _heuristic_score(name: _Optional[str], fqname: _Optional[str], path: _Optional[str], needle: str) -> int:
        """
        Crude but effective ranking for non-FTS fallback results.
        Higher is better.
        """
        n = (needle or "").strip()
        if not n:
            return 0
        nl = n.lower()
        toks = [t.lower() for t in _tokens(n)]
        score = 0

        def _inc_for_field(field: _Optional[str], weight_contains: int, weight_prefix: int) -> None:
            nonlocal score
            if not field:
                return
            fl = field.lower()
            if fl == nl:
                score += 100
                return
            if fl.startswith(nl):
                score += weight_prefix
            for t in toks:
                if t and t in fl:
                    score += weight_contains

        _inc_for_field(name, 18, 45)
        _inc_for_field(fqname, 10, 30)
        _inc_for_field(path, 6, 18)

        # prefer shorter names when otherwise similar
        if name:
            score += max(0, 15 - min(15, len(name)))
        return score

    fts_enabled = _is_fts_enabled()
    used_fts = False
    fts_query_used: _Optional[str] = None
    ranked_by = "heuristic"
    fts_attempted = False
    fts_reason: _Optional[str] = None
    fts_error: _Optional[str] = None

    # Explain why FTS is or isn't used (reported when explain=True)
    if mode == "fts" and not fts_enabled:
        fts_reason = "fts_unavailable"
    elif mode == "auto" and not fts_enabled:
        fts_reason = "fts_disabled"

    rows: _List[_Any] = []

    try:
        if mode == "fq":
            rows = cur.execute(
                "SELECT s.id, s.fqname, s.name, s.kind, f.path, s.start_line, s.end_line "
                "FROM symbols s JOIN files f ON s.file_id=f.id "
                "WHERE s.fqname = ? LIMIT ?",
                (val, limit),
            ).fetchall()
            ranked_by = "exact"
        elif mode == "symbol":
            like = f"%{val}%"
            pre_limit = max(limit * 8, 100)
            rows = cur.execute(
                "SELECT s.id, s.fqname, s.name, s.kind, f.path, s.start_line, s.end_line "
                "FROM symbols s JOIN files f ON s.file_id=f.id "
                "WHERE s.name LIKE ? "
                "LIMIT ?",
                (like, pre_limit),
            ).fetchall()
        elif mode == "path":
            like = f"%{val}%"
            pre_limit = max(limit * 8, 100)
            rows = cur.execute(
                "SELECT NULL, NULL, NULL, 'file', path, 1, line_count FROM files WHERE path LIKE ? LIMIT ?",
                (like, pre_limit),
            ).fetchall()

        elif mode == "impact":
            # Impact analysis: summarize files that reference a target (best-effort).
            # Uses refs table; no FTS is required. Returns file-centric matches.
            has_refs = cur.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='refs' LIMIT 1"
            ).fetchone()
            if not has_refs:
                out: _Dict[str, _Any] = {
                    "ok": True,
                    "db_path": db_path,
                    "fts_enabled": fts_enabled,
                    "used_fts": False,
                    "mode": mode,
                    "ranked_by": "impact_total",
                    "matches": [],
                }
                if explain:
                    out["tokens"] = _tokens(val)
                    out["fts_query_used"] = None
                    out["fts_decision"] = {
                        "enabled": fts_enabled,
                        "used": False,
                        "attempted": False,
                        "reason": "refs_table_missing",
                        "error": None,
                    }
                con.close()
                return out

            imp_val = (val or "").strip()
            imp_toks = _tokens(imp_val)
            target_mode = "ref_text_like"
            params: _List[_Any] = []
            if imp_val.startswith("fq:"):
                target_mode = "to_symbol_fqname"
                fq = imp_val[3:].strip()
                imp_toks = _tokens(fq)
                where = "refs.to_symbol_fqname = ?"
                params = [fq]
            else:
                pat = f"%{imp_val.lower()}%"
                where = "LOWER(refs.ref_text) LIKE ?"
                params = [pat]

            sql = f"""
                SELECT files.path, refs.ref_kind, COUNT(*) as n, MIN(refs.line) as min_line
                FROM refs
                JOIN files ON files.id = refs.file_id
                WHERE {where}
                GROUP BY files.path, refs.ref_kind
            """
            rows = cur.execute(sql, params).fetchall()

            per_file: _Dict[str, _Dict[str, _Any]] = {}
            for path, rk, n, min_line in rows:
                pth = str(path)
                rec = per_file.setdefault(pth, {"path": pth, "total": 0, "by_kind": {}, "sample_line": None})
                nn = int(n)
                rec["total"] += nn
                rec["by_kind"][str(rk)] = rec["by_kind"].get(str(rk), 0) + nn
                if min_line is not None:
                    ml = int(min_line)
                    if rec["sample_line"] is None or ml < int(rec["sample_line"]):
                        rec["sample_line"] = ml

            files_sorted = sorted(per_file.values(), key=lambda r: (-int(r["total"]), r["path"]))[: max(0, int(limit))]

            matches: _List[_Dict[str, _Any]] = []
            for rec in files_sorted:
                line = int(rec.get("sample_line") or 1)
                start = max(1, line - 10)
                end = line + 10
                m: _Dict[str, _Any] = {
                    "match_kind": "impact_file",
                    "path": rec["path"],
                    "start_line": start,
                    "end_line": end,
                    "impact_total": int(rec["total"]),
                    "impact_by_kind": rec.get("by_kind") or {},
                    "sample_line": line,
                    "impact_mode": target_mode,
                    "impact_target": imp_val,
                }
                if explain:
                    m["score"] = float(rec["total"])
                    m["why"] = ["impact", target_mode]
                if highlight:
                    m["display_path"] = _apply_highlight(rec["path"], imp_toks)
                matches.append(m)

            out: _Dict[str, _Any] = {
                "ok": True,
                "db_path": db_path,
                "fts_enabled": fts_enabled,
                "used_fts": False,
                "mode": mode,
                "ranked_by": "impact_total",
                "matches": matches,
            }
            if explain:
                out["tokens"] = imp_toks
                out["fts_query_used"] = None
                out["fts_decision"] = {
                    "enabled": fts_enabled,
                    "used": False,
                    "attempted": False,
                    "reason": "impact_mode",
                    "error": None,
                }
            con.close()
            return out
        elif mode == "ref":
            # Best-effort: find symbols/files that reference text (imports/calls) via refs table.
            has_refs = cur.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='refs' LIMIT 1"
            ).fetchone()
            if not has_refs:
                # No refs table (older DB). Return empty result with a clear decision reason.
                out: _Dict[str, _Any] = {
                    "ok": True,
                    "db_path": db_path,
                    "fts_enabled": fts_enabled,
                    "used_fts": False,
                    "mode": mode,
                    "ranked_by": "ref_score",
                    "matches": [],
                }
                if explain:
                    out["tokens"] = toks
                    out["fts_query_used"] = None
                    out["fts_decision"] = {
                        "enabled": fts_enabled,
                        "used": False,
                        "attempted": False,
                        "reason": "refs_table_missing",
                        "error": None,
                    }
                con.close()
                return out

            like = f"%{val}%"
            pre_limit = max(limit * 12, 200)
            ref_rows = cur.execute(
                "SELECT r.ref_kind, r.ref_text, r.line, r.from_symbol_id, r.to_symbol_fqname, "
                "f.path, s.fqname, s.name, s.kind, s.start_line, s.end_line "
                "FROM refs r "
                "JOIN files f ON r.file_id=f.id "
                "LEFT JOIN symbols s ON r.from_symbol_id=s.id "
                "WHERE (r.ref_text LIKE ? OR COALESCE(r.to_symbol_fqname,'') LIKE ?) "
                "ORDER BY f.path, r.line LIMIT ?",
                (like, like, pre_limit),
            ).fetchall()

            tmp_ref = []
            ql = (val or "").lower()
            toks = _tokens(val)
            toks = _tokens(val)

            for rk, rt, ln, from_id, to_fq, path, sfq, sname, skind, ss, se in ref_rows:
                ln_i = int(ln) if ln is not None else 1

                if from_id is not None:
                    symbol_id = int(from_id)
                    fqname = sfq
                    name = sname
                    kind = skind or "symbol"
                    start_line = int(ss) if ss is not None else max(1, ln_i - 10)
                    end_line = int(se) if se is not None else (start_line + 20)
                else:
                    symbol_id = None
                    fqname = None
                    name = "<module>"
                    kind = "file_ref"
                    start_line = max(1, ln_i - 10)
                    end_line = ln_i + 10

                # Simple scoring: prefer exact matches, then contains, then call/import kind.
                score = 0.0
                why: _List[str] = []

                rt_l = (rt or "").lower()
                to_l = (to_fq or "").lower()

                if ql and (rt_l == ql or to_l == ql):
                    score += 100.0
                    why.append("ref_exact")
                if ql and (ql in rt_l or ql in to_l):
                    score += 30.0
                    why.append("ref_like")

                if rk == "call":
                    score += 10.0
                    why.append("call")
                elif rk == "import":
                    score += 5.0
                    why.append("import")

                if symbol_id is not None:
                    score += 3.0
                    why.append("enclosed_symbol")

                # Small preference for shorter names (when present).
                if name:
                    score += max(0.0, 5.0 - min(5.0, len(str(name)) / 20.0))

                m: _Dict[str, _Any] = {
                    "symbol_id": symbol_id,
                    "fqname": fqname,
                    "name": name,
                    "kind": kind,
                    "path": path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "ref_kind": rk,
                    "ref_text": rt,
                    "ref_line": ln_i,
                    "to_symbol_fqname": to_fq,
                }

                if explain:
                    m["score"] = score
                    m["why"] = why

                if highlight:
                    m["display_name"] = _apply_highlight(name, toks)
                    m["display_fqname"] = _apply_highlight(fqname, toks)
                    m["display_path"] = _apply_highlight(path, toks)
                    m["display_ref_text"] = _apply_highlight(rt, toks)
                    m["display_to_symbol_fqname"] = _apply_highlight(to_fq, toks)

                tmp_ref.append((score, m))

            tmp_ref.sort(key=lambda x: x[0], reverse=True)
            matches = [m for _, m in tmp_ref[:limit]]

            out: _Dict[str, _Any] = {
                "ok": True,
                "db_path": db_path,
                "fts_enabled": fts_enabled,
                "used_fts": False,
                "mode": mode,
                "ranked_by": "ref_score",
                "matches": matches,
            }
            if explain:
                out["fts_query_used"] = None
                out["tokens"] = toks
                out["fts_decision"] = {
                    "enabled": fts_enabled,
                    "used": False,
                    "attempted": False,
                    "reason": "ref_mode",
                    "error": None,
                }

            con.close()
            return out

        elif mode == "callers":
            # Find call sites that appear to call a target (best-effort), using refs table.
            has_refs = cur.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='refs' LIMIT 1"
            ).fetchone()
            if not has_refs:
                out: _Dict[str, _Any] = {
                    "ok": True,
                    "db_path": db_path,
                    "fts_enabled": fts_enabled,
                    "used_fts": False,
                    "mode": mode,
                    "ranked_by": "callers_score",
                    "matches": [],
                }
                if explain:
                    out["fts_query_used"] = None
                    out["tokens"] = _tokens(val)
                    out["fts_decision"] = {
                        "enabled": fts_enabled,
                        "used": False,
                        "attempted": False,
                        "reason": "refs_table_missing",
                        "error": None,
                    }
                con.close()
                return out

            q0 = (val or "").strip()
            target_fq: _Optional[str] = None
            needle = q0
            if q0.startswith("fq:"):
                target_fq = q0[3:].strip()
                row = cur.execute("SELECT name FROM symbols WHERE fqname=? LIMIT 1", (target_fq,)).fetchone()
                if row and row[0]:
                    needle = str(row[0])

            where = ["r.ref_kind='call'"]
            params: _List[_Any] = []
            if target_fq:
                where.append("r.to_symbol_fqname = ?")
                params.append(target_fq)
            elif needle:
                where.append("r.ref_text LIKE ?")
                params.append(f"%{needle}%")

            pre_limit = max(limit * 20, 200)
            sql = (
                "SELECT r.ref_text, r.to_symbol_fqname, r.line, f.path, r.from_symbol_id, "
                "s.fqname, s.name "
                "FROM refs r "
                "JOIN files f ON f.id=r.file_id "
                "LEFT JOIN symbols s ON s.id=r.from_symbol_id "
                f"WHERE {' AND '.join(where)} "
                "ORDER BY f.path, r.line LIMIT ?"
            )
            params.append(int(pre_limit))

            caller_rows = cur.execute(sql, tuple(params)).fetchall()

            toks = _tokens(needle or val)
            ql = (needle or "").lower()
            tmp_call: _List[_Tuple[float, _Dict[str, _Any]]] = []

            for rt, to_fq, ln, path, from_id, from_fq, from_name in caller_rows:
                ln_i = int(ln) if ln is not None else 1

                score = 0.0
                why: _List[str] = []

                rt_l = (rt or "").lower()
                to_l = (to_fq or "").lower()

                if target_fq and (to_fq == target_fq):
                    score += 120.0
                    why.append("to_fq_exact")
                if ql and rt_l == ql:
                    score += 100.0
                    why.append("ref_exact")
                if ql and ql in rt_l:
                    score += 30.0
                    why.append("ref_like")
                if ql and ql in to_l:
                    score += 20.0
                    why.append("to_fq_like")
                if from_id is not None:
                    score += 5.0
                    why.append("has_from_symbol")
                if to_fq:
                    score += 3.0
                    why.append("has_to_fq")

                m: _Dict[str, _Any] = {
                    "symbol_id": None,  # show call-site span (not full snippet)
                    "fqname": from_fq,
                    "name": from_name or "<module>",
                    "kind": "call_site",
                    "path": path,
                    "start_line": max(1, ln_i - 10),
                    "end_line": ln_i + 10,
                    "ref_kind": "call",
                    "ref_text": rt,
                    "ref_line": ln_i,
                    "to_symbol_fqname": to_fq,
                    "from_symbol_id": int(from_id) if from_id is not None else None,
                    "from_symbol_fqname": from_fq,
                }

                if explain:
                    m["score"] = score
                    m["why"] = why

                if highlight:
                    m["display_name"] = _apply_highlight(m.get("name"), toks)
                    m["display_fqname"] = _apply_highlight(from_fq, toks)
                    m["display_path"] = _apply_highlight(path, toks)
                    m["display_ref_text"] = _apply_highlight(rt, toks)
                    m["display_to_symbol_fqname"] = _apply_highlight(to_fq, toks)

                tmp_call.append((score, m))

            tmp_call.sort(key=lambda x: x[0], reverse=True)
            matches = [m for _, m in tmp_call[:limit]]

            out: _Dict[str, _Any] = {
                "ok": True,
                "db_path": db_path,
                "fts_enabled": fts_enabled,
                "used_fts": False,
                "mode": mode,
                "ranked_by": "callers_score",
                "matches": matches,
            }
            if explain:
                out["fts_query_used"] = None
                out["tokens"] = toks
                out["fts_decision"] = {
                    "enabled": fts_enabled,
                    "used": False,
                    "attempted": False,
                    "reason": "callers_mode",
                    "error": None,
                }

            con.close()
            return out

        elif mode == "callees":
            # Find call sites *inside* a given source symbol, using refs table.
            has_refs = cur.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='refs' LIMIT 1"
            ).fetchone()
            if not has_refs:
                out: _Dict[str, _Any] = {
                    "ok": True,
                    "db_path": db_path,
                    "fts_enabled": fts_enabled,
                    "used_fts": False,
                    "mode": mode,
                    "ranked_by": "callees_line",
                    "matches": [],
                }
                if explain:
                    out["fts_query_used"] = None
                    out["tokens"] = _tokens(val)
                    out["fts_decision"] = {
                        "enabled": fts_enabled,
                        "used": False,
                        "attempted": False,
                        "reason": "refs_table_missing",
                        "error": None,
                    }
                con.close()
                return out

            q0 = (val or "").strip()
            source_id: _Optional[int] = None
            source_fq: _Optional[str] = None
            source_name: _Optional[str] = None

            if q0.startswith("fq:"):
                source_fq = q0[3:].strip()
                row = cur.execute("SELECT id, name FROM symbols WHERE fqname=? LIMIT 1", (source_fq,)).fetchone()
                if row:
                    source_id = int(row[0])
                    source_name = str(row[1]) if row[1] else None
            else:
                try:
                    source_id = int(q0)
                except Exception:
                    source_id = None
                if source_id is not None:
                    row = cur.execute("SELECT fqname, name FROM symbols WHERE id=? LIMIT 1", (source_id,)).fetchone()
                    if row:
                        source_fq = str(row[0]) if row[0] else None
                        source_name = str(row[1]) if row[1] else None

            if source_id is None:
                con.close()
                return {"ok": False, "error": "could not resolve callees source; use callees:fq:<fqname> or callees:<symbol_id>", "matches": []}

            pre_limit = max(limit * 20, 200)
            callee_rows = cur.execute(
                "SELECT r.ref_text, r.to_symbol_fqname, r.line, f.path "
                "FROM refs r JOIN files f ON r.file_id=f.id "
                "WHERE r.from_symbol_id=? AND r.ref_kind='call' "
                "ORDER BY r.line LIMIT ?",
                (source_id, pre_limit),
            ).fetchall()

            toks = _tokens(source_fq or q0)
            tmp_call: _List[_Tuple[float, _Dict[str, _Any]]] = []
            for rt, to_fq, ln, path in callee_rows:
                ln_i = int(ln) if ln is not None else 1
                score = 0.0
                why: _List[str] = []
                if to_fq:
                    score += 5.0
                    why.append("has_to_fq")
                if rt:
                    score += 2.0
                    why.append("has_ref_text")

                m: _Dict[str, _Any] = {
                    "symbol_id": None,  # show call-site span
                    "fqname": source_fq,
                    "name": source_name or "<source>",
                    "kind": "call_site",
                    "path": path,
                    "start_line": max(1, ln_i - 10),
                    "end_line": ln_i + 10,
                    "ref_kind": "call",
                    "ref_text": rt,
                    "ref_line": ln_i,
                    "to_symbol_fqname": to_fq,
                    "from_symbol_id": source_id,
                    "from_symbol_fqname": source_fq,
                }

                if explain:
                    m["score"] = score
                    m["why"] = why

                if highlight:
                    m["display_name"] = _apply_highlight(m.get("name"), toks)
                    m["display_fqname"] = _apply_highlight(source_fq, toks)
                    m["display_path"] = _apply_highlight(path, toks)
                    m["display_ref_text"] = _apply_highlight(rt, toks)
                    m["display_to_symbol_fqname"] = _apply_highlight(to_fq, toks)

                tmp_call.append((score, m))

            tmp_call.sort(key=lambda x: x[0], reverse=True)
            matches = [m for _, m in tmp_call[:limit]]

            out: _Dict[str, _Any] = {
                "ok": True,
                "db_path": db_path,
                "fts_enabled": fts_enabled,
                "used_fts": False,
                "mode": mode,
                "ranked_by": "callees_score",
                "matches": matches,
            }
            if explain:
                out["fts_query_used"] = None
                out["tokens"] = toks
                out["fts_decision"] = {
                    "enabled": fts_enabled,
                    "used": False,
                    "attempted": False,
                    "reason": "callees_mode",
                    "error": None,
                }

            con.close()
            return out


        elif mode == "fts" and fts_enabled:
            rows = cur.execute(
                "SELECT s.id, s.fqname, s.name, s.kind, f.path, s.start_line, s.end_line, bm25(atlas_fts) "
                "FROM atlas_fts "
                "JOIN symbols s ON s.id = atlas_fts.rowid "
                "JOIN files f ON f.id = s.file_id "
                "WHERE atlas_fts MATCH ? "
                "ORDER BY bm25(atlas_fts) "
                "LIMIT ?",
                (val, limit),
            ).fetchall()
            fts_attempted = True
            used_fts = bool(rows)
            fts_reason = "explicit" if used_fts else "explicit_no_results"
            fts_query_used = val
            ranked_by = "bm25"
        else:
            # Auto mode: prefer FTS (safe query) when available, but always fallback to LIKE.
            if fts_enabled and mode == "auto":
                fts_q = _safe_fts(val)
                fts_query_used = fts_q or None
                if fts_q:
                    try:
                        rows = cur.execute(
                            "SELECT s.id, s.fqname, s.name, s.kind, f.path, s.start_line, s.end_line, bm25(atlas_fts) "
                            "FROM atlas_fts "
                            "JOIN symbols s ON s.id = atlas_fts.rowid "
                            "JOIN files f ON f.id = s.file_id "
                            "WHERE atlas_fts MATCH ? "
                            "ORDER BY bm25(atlas_fts) "
                            "LIMIT ?",
                            (fts_q, limit),
                        ).fetchall()
                        fts_attempted = True
                        used_fts = bool(rows)
                        if used_fts:
                            fts_reason = "auto"
                            ranked_by = "bm25"
                        else:
                            fts_reason = "fts_no_results"
                    except Exception as _e:
                        fts_attempted = True
                        fts_reason = "fts_error"
                        fts_error = str(_e)
                        rows = []
                else:
                    fts_reason = "no_tokens"

            if not rows:
                # Non-FTS fallback: the "LIKE %val%" approach fails for multi-token queries (e.g. "guardian rotation").
                # Instead, expand into token-wise OR conditions and then rank with heuristic scoring.
                toks_fallback = _tokens(val)
                toks_cap = (toks_fallback or [])[:8]
                pre_limit = max(limit * 10, 150)

                if len(toks_cap) <= 1:
                    needle = (val or (toks_cap[0] if toks_cap else "")).strip()
                    like = f"%{needle}%"
                    rows = cur.execute(
                        "SELECT s.id, s.fqname, s.name, s.kind, f.path, s.start_line, s.end_line "
                        "FROM symbols s JOIN files f ON s.file_id=f.id "
                        "WHERE s.name LIKE ? OR s.fqname LIKE ? OR f.path LIKE ? "
                        "LIMIT ?",
                        (like, like, like, pre_limit),
                    ).fetchall()
                else:
                    clauses = []
                    params = []
                    for t in toks_cap:
                        like = f"%{t}%"
                        clauses.append("(s.name LIKE ? OR s.fqname LIKE ? OR f.path LIKE ?)")
                        params.extend([like, like, like])
                    where = " OR ".join(clauses)
                    sql = (
                        "SELECT s.id, s.fqname, s.name, s.kind, f.path, s.start_line, s.end_line "
                        "FROM symbols s JOIN files f ON s.file_id=f.id "
                        "WHERE " + where + " "
                        "LIMIT ?"
                    )
                    params.append(pre_limit)
                    rows = cur.execute(sql, tuple(params)).fetchall()
    except Exception as e:
        con.close()
        return {"ok": False, "error": f"nav query failed: {e}", "matches": []}

    toks = _tokens(val)

    matches = []
    tmp = []
    for r in rows:
        # FTS rows include bm25 at the end; non-FTS do not.
        if used_fts or (len(r) == 8):
            symbol_id, fqname, name, kind, path, start, end, bm25 = r
            score = -float(bm25) if bm25 is not None else 0.0
            why = ["fts"]
        else:
            symbol_id, fqname, name, kind, path, start, end = r
            score = float(_heuristic_score(name, fqname, path, val))
            why = []
            if mode == "fq":
                why.append("fq_exact")
            elif mode == "path":
                why.append("path_like")
            elif mode == "symbol":
                why.append("name_like")
            else:
                why.append("like_fallback")
        tmp.append((score, symbol_id, fqname, name, kind, path, start, end, why))

    # Sort if not bm25 already ranked by SQLite
    if ranked_by != "bm25":
        tmp.sort(key=lambda t: t[0], reverse=True)
        tmp = tmp[:limit]

    for score, symbol_id, fqname, name, kind, path, start, end, why in tmp:
        m: _Dict[str, _Any] = {
            "symbol_id": symbol_id,
            "fqname": fqname,
            "name": name,
            "kind": kind,
            "path": path,
            "start_line": int(start) if start is not None else None,
            "end_line": int(end) if end is not None else None,
        }
        if explain:
            m["score"] = score
            m["why"] = why

        if highlight:
            m["display_name"] = _apply_highlight(name, toks)
            m["display_fqname"] = _apply_highlight(fqname, toks)
            m["display_path"] = _apply_highlight(path, toks)

        matches.append(m)

    con.close()
    out: _Dict[str, _Any] = {
        "ok": True,
        "query": query,
        "db_path": db_path,
        "fts_enabled": fts_enabled,
        "used_fts": used_fts,
        "mode": mode,
        "ranked_by": ranked_by,
        "matches": matches,
    }
    if explain:
        out["fts_query_used"] = fts_query_used
        out["tokens"] = toks
        out["fts_decision"] = {
            "enabled": fts_enabled,
            "used": used_fts,
            "attempted": fts_attempted,
            "reason": fts_reason,
            "error": fts_error,
        }
    return out


def nav_explain(
    res: _Dict[str, _Any],
    *,
    max_matches: int = 10,
    max_lines: int = 40,
) -> str:
    """Pretty-print a nav() result within a bounded line budget.

    Conservative for CLI/notebook use: truncates output and never prints file bodies
    (use show_match / show_span for source).
    """
    if not res:
        return "(no result)"
    if not res.get("ok", False):
        err = res.get("error", "unknown error")
        return f"nav: ok=false error={err}"

    q = res.get("query", "")
    mode = res.get("mode", "")
    used_fts = bool(res.get("used_fts", False))
    ranked_by = res.get("ranked_by", "")
    dec = res.get("fts_decision") or {}
    reason = dec.get("reason")
    enabled = dec.get("enabled", res.get("fts_enabled"))
    attempted = dec.get("attempted")
    fts_q = res.get("fts_query_used")
    toks = res.get("tokens") or []

    lines: _List[str] = []
    lines.append(f"query={q!r} mode={mode} ranked_by={ranked_by} used_fts={used_fts}")
    lines.append(f"fts_enabled={enabled} attempted={attempted} reason={reason} fts_query={fts_q!r}")
    if toks:
        lines.append("tokens=" + ",".join(toks[:12]) + ("…" if len(toks) > 12 else ""))
    matches = res.get("matches") or []
    lines.append(f"matches={len(matches)} (showing {min(len(matches), max_matches)})")

    for i, m in enumerate(matches[:max_matches], 1):
        kind = m.get("kind") or "?"
        path = m.get("path") or ""
        fq = m.get("display_fqname") or m.get("fqname") or ""
        name = m.get("display_name") or m.get("name") or ""
        span = ""
        if m.get("start_line") and m.get("end_line"):
            span = f":{m['start_line']}-{m['end_line']}"
        label = fq or name or path
        why = m.get("why")
        if isinstance(why, list) and why:
            why_s = ",".join(str(x) for x in why[:4]) + ("…" if len(why) > 4 else "")
        else:
            why_s = ""
        if why_s:
            lines.append(f"{i:>2}. {kind:>8} {path}{span} :: {label}  [{why_s}]")
        else:
            lines.append(f"{i:>2}. {kind:>8} {path}{span} :: {label}")

    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + ["… (truncated)"]

    return "\n".join(lines)


def print_nav_explain(
    res: _Dict[str, _Any],
    *,
    max_matches: int = 10,
    max_lines: int = 40,
) -> None:
    """Print nav_explain() output."""
    print(nav_explain(res, max_matches=max_matches, max_lines=max_lines))


def nav_status(*, repo_root: str, db_path: _Optional[str] = None) -> _Dict[str, _Any]:
    """Return basic diagnostics about the nav SQLite DB (schema, counts, FTS availability)."""
    db_path = db_path or _default_db_path(repo_root)
    p = _Path(db_path)
    if not p.exists():
        return {"ok": False, "error": f"sqlite db not found: {db_path}", "db_path": db_path}

    con = _sqlite3.connect(db_path)
    cur = con.cursor()

    try:
        meta_rows = cur.execute("SELECT key, value FROM meta").fetchall()
        meta = {k: v for (k, v) in meta_rows}
    except Exception:
        meta = {}

    def _count(sql: str) -> int:
        try:
            return int(cur.execute(sql).fetchone()[0])
        except Exception:
            return 0

    files = _count("SELECT COUNT(*) FROM files")
    symbols = _count("SELECT COUNT(*) FROM symbols")
    imports = _count("SELECT COUNT(*) FROM imports")
    snippets = _count("SELECT COUNT(*) FROM snippets")

    refs = _count("SELECT COUNT(*) FROM refs")

    fts_enabled = (meta.get("fts5_enabled") == "1")
    fts_rows = 0
    if fts_enabled:
        try:
            fts_rows = int(cur.execute("SELECT COUNT(*) FROM atlas_fts").fetchone()[0])
        except Exception:
            fts_rows = 0

    con.close()
    return {
        "ok": True,
        "db_path": db_path,
        "schema_version": meta.get("schema_version"),
        "last_indexed_utc": meta.get("last_indexed_utc"),
        "fts_enabled": fts_enabled,
        "fts_rows": fts_rows,
        "counts": {"files": files, "symbols": symbols, "imports": imports, "snippets": snippets, "refs": refs},
    }


def refs(
    query: str,
    *,
    repo_root: str,
    kind: _Optional[str] = None,
    limit: int = 50,
    db_path: _Optional[str] = None,
) -> _Dict[str, _Any]:
    """Search reference rows (imports/calls) captured during indexing."""
    db_path = db_path or _default_db_path(repo_root)
    p = _Path(db_path)
    if not p.exists():
        return {"ok": False, "error": f"sqlite db not found: {db_path}", "db_path": db_path, "refs": []}

    con = _sqlite3.connect(db_path)
    cur = con.cursor()

    q = (query or "").strip()
    params: _List[_Any] = []
    where = []

    # Support fq:<fqname> to resolve symbol name first (more stable than raw text).
    needle = q
    if q.startswith("fq:"):
        fq = q[3:].strip()
        row = cur.execute("SELECT name FROM symbols WHERE fqname=? LIMIT 1", (fq,)).fetchone()
        if row and row[0]:
            needle = str(row[0])

    if needle:
        like = f"%{needle}%"
        where.append("(r.ref_text LIKE ? OR COALESCE(r.to_symbol_fqname,'') LIKE ?)")
        params.extend([like, like])

    if kind:
        where.append("r.ref_kind = ?")
        params.append(kind)

    where_sql = " AND ".join(where) if where else "1=1"
    sql = (
        "SELECT r.ref_kind, r.ref_text, r.to_symbol_fqname, r.line, f.path, "
        "r.from_symbol_id, s.fqname "
        "FROM refs r "
        "JOIN files f ON f.id=r.file_id "
        "LEFT JOIN symbols s ON s.id=r.from_symbol_id "
        f"WHERE {where_sql} "
        "ORDER BY f.path, r.line "
        "LIMIT ?"
    )
    params.append(int(limit))

    rows = []
    try:
        for rk, rt, tfq, ln, path, fsid, ffq in cur.execute(sql, tuple(params)).fetchall():
            rows.append({
                "kind": rk,
                "ref_text": rt,
                "to_symbol_fqname": tfq,
                "line": ln,
                "path": path,
                "from_symbol_id": fsid,
                "from_symbol_fqname": ffq,
            })
    except Exception as e:
        con.close()
        return {"ok": False, "error": str(e), "db_path": db_path, "refs": []}

    con.close()
    return {"ok": True, "db_path": db_path, "query": query, "kind": kind, "refs": rows}

def callers(
    target: str,
    *,
    repo_root: str,
    limit: int = 50,
    db_path: _Optional[str] = None,
) -> _Dict[str, _Any]:
    """Find call refs that appear to target `target` (best-effort)."""
    db_path = db_path or _default_db_path(repo_root)
    con = _sqlite3.connect(db_path) if _Path(db_path).exists() else None
    if con is None:
        return {"ok": False, "error": f"sqlite db not found: {db_path}", "db_path": db_path, "callers": []}
    cur = con.cursor()

    q = (target or "").strip()
    target_fq = None
    needle = q
    if q.startswith("fq:"):
        target_fq = q[3:].strip()
        row = cur.execute("SELECT name FROM symbols WHERE fqname=? LIMIT 1", (target_fq,)).fetchone()
        if row and row[0]:
            needle = str(row[0])

    where = ["r.ref_kind='call'"]
    params: _List[_Any] = []
    if target_fq:
        where.append("r.to_symbol_fqname = ?")
        params.append(target_fq)
    elif needle:
        like = f"%{needle}%"
        where.append("r.ref_text LIKE ?")
        params.append(like)

    sql = (
        "SELECT r.ref_text, r.to_symbol_fqname, r.line, f.path, r.from_symbol_id, s.fqname "
        "FROM refs r "
        "JOIN files f ON f.id=r.file_id "
        "LEFT JOIN symbols s ON s.id=r.from_symbol_id "
        f"WHERE {' AND '.join(where)} "
        "ORDER BY f.path, r.line "
        "LIMIT ?"
    )
    params.append(int(limit))

    out_rows = []
    try:
        for rt, tfq, ln, path, fsid, ffq in cur.execute(sql, tuple(params)).fetchall():
            out_rows.append({
                "ref_text": rt,
                "to_symbol_fqname": tfq,
                "line": ln,
                "path": path,
                "from_symbol_id": fsid,
                "from_symbol_fqname": ffq,
            })
    except Exception as e:
        con.close()
        return {"ok": False, "error": str(e), "db_path": db_path, "callers": []}

    con.close()
    return {"ok": True, "db_path": db_path, "target": target, "callers": out_rows}



def callees(
    source: str,
    *,
    repo_root: str,
    limit: int = 50,
    db_path: _Optional[str] = None,
) -> _Dict[str, _Any]:
    """Find call refs made *from* `source` symbol (best-effort).

    Source formats:
      - "fq:<fqname>" (preferred)
      - "<symbol_id>" (numeric)
    """
    db_path = db_path or _default_db_path(repo_root)
    con = _sqlite3.connect(db_path) if _Path(db_path).exists() else None
    if con is None:
        return {"ok": False, "error": f"sqlite db not found: {db_path}", "db_path": db_path, "callees": []}
    cur = con.cursor()

    q = (source or "").strip()
    symbol_id: _Optional[int] = None

    if q.startswith("fq:"):
        fq = q[3:].strip()
        row = cur.execute("SELECT id FROM symbols WHERE fqname=? LIMIT 1", (fq,)).fetchone()
        if row:
            symbol_id = int(row[0])
    else:
        try:
            symbol_id = int(q)
        except Exception:
            symbol_id = None

    if symbol_id is None:
        con.close()
        return {
            "ok": False,
            "error": "could not resolve source symbol id; use fq:<fqname> or numeric id",
            "db_path": db_path,
            "callees": [],
        }

    has_refs = cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='refs' LIMIT 1"
    ).fetchone()
    if not has_refs:
        con.close()
        return {
            "ok": False,
            "error": "refs table missing; rebuild index",
            "db_path": db_path,
            "symbol_id": symbol_id,
            "callees": [],
        }

    rows = cur.execute(
        "SELECT r.ref_text, r.to_symbol_fqname, r.line, f.path "
        "FROM refs r JOIN files f ON r.file_id=f.id "
        "WHERE r.from_symbol_id=? AND r.ref_kind='call' "
        "ORDER BY r.line LIMIT ?",
        (symbol_id, limit),
    ).fetchall()

    out_rows: _List[_Dict[str, _Any]] = []
    for rt, to_fq, ln, path in rows:
        out_rows.append(
            {
                "ref_text": rt,
                "to_symbol_fqname": to_fq,
                "line": int(ln) if ln is not None else None,
                "path": path,
            }
        )

    con.close()
    return {"ok": True, "db_path": db_path, "source": source, "symbol_id": symbol_id, "callees": out_rows}

def show_match(match: _Dict[str, _Any], *, repo_root: str, db_path: _Optional[str] = None, max_lines: int = 40) -> str:
    """
    Render a match object from nav(). Prefer returning snippet content from sqlite.
    Falls back to reading file lines if snippet not available.
    """
    db_path = db_path or _default_db_path(repo_root)
    symbol_id = match.get("symbol_id")
    path = match.get("path")
    start = match.get("start_line") or 1
    end = match.get("end_line") or start

    # If symbol match, attempt snippet
    if symbol_id:
        p = _Path(db_path)
        if p.exists():
            con = _sqlite3.connect(db_path)
            cur = con.cursor()
            row = cur.execute("SELECT content FROM snippets WHERE symbol_id=?", (int(symbol_id),)).fetchone()
            con.close()
            if row and row[0]:
                snippet = row[0]
                if max_lines is not None and max_lines > 0:
                    s_lines = snippet.splitlines()
                    if len(s_lines) > max_lines:
                        snippet = "\n".join(s_lines[:max_lines]) + "\n"
                return snippet

    # fallback: read file span
    if not path:
        return ""
    abs_path = _Path(repo_root) / path
    if not abs_path.exists():
        return f"[show_match] file not found: {abs_path}"
    lines = abs_path.read_text(encoding="utf-8", errors="replace").splitlines()
    s0 = max(1, int(start)) - 1
    e0 = min(len(lines), int(end))
    if max_lines is not None and max_lines > 0 and (e0 - s0) > max_lines:
        e0 = s0 + max_lines
    return "\n".join(lines[s0:e0]) + "\n"


def preflight_and_package(
    repo_root: str,
    *,
    targets: _Optional[_List[str]] = None,
    use_modified_since_index: bool = True,
    zip_name: str = "update.zip",
    update_sqlite_index: bool = True,
    db_path: _Optional[str] = None,
) -> _Dict[str, _Any]:
    """
    Preflight + package workflow:
      1) Determine target files (explicit targets, else modified since hash index)
      2) AST-parse + py_compile for python targets
      3) Update sqlite navigation index (optional)
      4) Build a patch zip with targets + ALWAYS include the sqlite file (if enabled)

    Returns dict with status + included files.
    """
    root = _Path(repo_root)
    if not root.exists():
        return {"ok": False, "error": f"repo_root not found: {repo_root}"}

    included: _List[str] = []
    errors: _List[str] = []

    # Determine targets
    tpaths: _List[str] = []
    if targets:
        tpaths = [str(_Path(t).as_posix()) for t in targets]
    elif use_modified_since_index:
        try:
            changed = list_modified_since_last_index(repo_root)
            tpaths = [c.replace("\\", "/") for c in changed.get("modified", [])]
        except Exception as e:
            errors.append(f"list_modified_since_last_index failed: {e}")
            tpaths = []
    else:
        tpaths = []

    # Preflight: parse+compile python files in targets
    for rel in tpaths:
        ap = root / rel
        if not ap.exists():
            continue
        included.append(rel)
        if ap.suffix == ".py":
            try:
                src = ap.read_text(encoding="utf-8", errors="replace")
                import ast as _ast
                _ast.parse(src)
                _py_compile.compile(str(ap), doraise=True)
            except Exception as e:
                errors.append(f"{rel}: {e}")

    if errors:
        return {"ok": False, "errors": errors, "included": included, "zip_name": zip_name}

    # SQLite indexing
    db_path = db_path or _default_db_path(repo_root)
    if update_sqlite_index:
        try:
            _ensure_db(_Path(db_path), enable_fts=update_fts)
            # index only python targets; if no targets, do full index
            py_targets = [p for p in tpaths if p.endswith(".py")]
            _index_repo(root, _Path(db_path), target_paths=py_targets if py_targets else None, update_fts=update_fts)
        except Exception as e:
            return {"ok": False, "errors": [f"sqlite index failed: {e}"], "included": included, "zip_name": zip_name}

    # Build zip
    out_path = _Path(zip_name)
    if not out_path.is_absolute():
        out_path = _Path.cwd() / out_path

    zip_files: _List[str] = []
    with _zipfile.ZipFile(str(out_path), "w", compression=_zipfile.ZIP_DEFLATED) as z:
        for rel in included:
            ap = root / rel
            if ap.exists() and ap.is_file():
                z.write(ap, arcname=rel)
                zip_files.append(rel)

        # Always include sqlite DB for alignment
        if update_sqlite_index:
            dbp = _Path(db_path)
            if dbp.exists():
                rel_db = _norm_path(str(dbp.relative_to(root)))
                z.write(dbp, arcname=rel_db)
                zip_files.append(rel_db)

    return {"ok": True, "zip_path": str(out_path), "zip_name": zip_name, "included": zip_files, "db_path": db_path}
def impact(
    target: str,
    *,
    repo_root: str,
    db_path: _Optional[str] = None,
    kind: _Optional[str] = None,
    limit: int = 200,
) -> _Dict[str, _Any]:
    """Summarize what files reference a target (best-effort impact analysis).

    target formats:
      - 'fq:<fqname>' uses refs.to_symbol_fqname when present (highest precision)
      - otherwise, treated as a ref_text pattern (case-insensitive contains)

    kind can filter ref_kind (e.g. 'call', 'import', 'string', 'config_key').
    """
    db_path = db_path or _default_db_path(repo_root)
    p = _Path(db_path)
    if not p.exists():
        return {"ok": False, "error": f"sqlite db not found: {db_path}", "db_path": db_path}

    con = _sqlite3.connect(db_path)
    cur = con.cursor()

    mode = "ref_text_like"
    params: list[_Any] = []
    where = ""
    if target.startswith("fq:"):
        mode = "to_symbol_fqname"
        fq = target[3:]
        where = "refs.to_symbol_fqname = ?"
        params.append(fq)
    else:
        # case-insensitive contains via LIKE on lowered text
        pat = f"%{target.lower()}%"
        where = "LOWER(refs.ref_text) LIKE ?"
        params.append(pat)

    if kind:
        where = f"({where}) AND refs.ref_kind = ?"
        params.append(kind)

    sql = f"""
        SELECT files.path, refs.ref_kind, COUNT(*) as n
        FROM refs
        JOIN files ON files.id = refs.file_id
        WHERE {where}
        GROUP BY files.path, refs.ref_kind
    """
    rows = cur.execute(sql, params).fetchall()
    con.close()

    # Aggregate per file
    per_file: dict[str, dict] = {}
    total = 0
    for path, rk, n in rows:
        n = int(n)
        total += n
        rec = per_file.setdefault(str(path), {"path": str(path), "total": 0, "by_kind": {}})
        rec["total"] += n
        rec["by_kind"][str(rk)] = rec["by_kind"].get(str(rk), 0) + n

    files_sorted = sorted(per_file.values(), key=lambda r: (-int(r["total"]), r["path"]))[: max(0, int(limit))]
    return {"ok": True, "db_path": db_path, "target": target, "mode": mode, "kind": kind, "total_refs": total, "files": files_sorted}


def doctor(*, repo_root: str, db_path: _Optional[str] = None, top: int = 8) -> _Dict[str, _Any]:
    """Deeper diagnostics than nav_status (FTS compile options, indexes, top refs)."""
    base = nav_status(repo_root=repo_root, db_path=db_path)
    if not base.get("ok"):
        return base

    db_path = base["db_path"]
    con = _sqlite3.connect(db_path)
    cur = con.cursor()

    # Compile options (filtered)
    compile_opts: list[str] = []
    try:
        opts = [r[0] for r in cur.execute("PRAGMA compile_options").fetchall()]
        compile_opts = [o for o in opts if ("FTS" in o or "UNICODE" in o)]
    except Exception:
        compile_opts = []

    # Index inventory (filtered)
    indexes: list[str] = []
    try:
        idx_rows = cur.execute("SELECT name FROM sqlite_master WHERE type='index' ORDER BY name").fetchall()
        names = [r[0] for r in idx_rows]
        indexes = [n for n in names if n.startswith("idx_") or n.startswith("sqlite_autoindex")]
    except Exception:
        indexes = []

    # Top refs by kind
    top_refs: dict[str, list[dict]] = {}
    try:
        kinds = ["call", "import", "config_key", "string"]
        for k in kinds:
            rows = cur.execute(
                "SELECT ref_text, COUNT(*) as n FROM refs WHERE ref_kind=? GROUP BY ref_text ORDER BY n DESC, ref_text ASC LIMIT ?",
                (k, int(top)),
            ).fetchall()
            top_refs[k] = [{"ref_text": r[0], "count": int(r[1])} for r in rows]
    except Exception:
        top_refs = {}

    con.close()
    out = dict(base)
    out["compile_options"] = compile_opts
    out["indexes"] = indexes
    out["top_refs"] = top_refs
    return out


def print_doctor(res: _Dict[str, _Any], *, max_lines: int = 40) -> None:
    """Pretty-print doctor() output without exceeding max_lines."""
    lines: list[str] = []
    ok = res.get("ok")
    lines.append(f"ok={ok} db={res.get('db_path')}")
    if not ok:
        lines.append(str(res.get("error")))
    else:
        lines.append(f"schema_version={res.get('schema_version')} last_indexed_utc={res.get('last_indexed_utc')}")
        lines.append(f"fts_enabled={res.get('fts_enabled')} fts_rows={res.get('fts_rows')}")
        counts = res.get("counts", {})
        lines.append("counts=" + json.dumps(counts, sort_keys=True))
        copts = res.get("compile_options", [])
        if copts:
            lines.append("compile_options=" + ", ".join(copts[:6]) + (" ..." if len(copts) > 6 else ""))
        idxs = res.get("indexes", [])
        if idxs:
            lines.append("indexes=" + ", ".join(idxs[:8]) + (" ..." if len(idxs) > 8 else ""))
        top_refs = res.get("top_refs", {})
        for k, arr in top_refs.items():
            if not arr:
                continue
            lines.append(f"top_refs[{k}]=" + ", ".join([f"{d['ref_text']}({d['count']})" for d in arr[:6]]))

    # enforce max_lines
    for ln in lines[: max(1, int(max_lines))]:
        print(ln)

# --------------------------------------------------------------------------------------
# Bootstrap helpers (zip-native)
# --------------------------------------------------------------------------------------

def bootstrap_from_zips(ai_introspect_zip: str, repo_zip: str, *, work_dir: _Optional[str] = None, update_fts: bool = True) -> _Dict[str, _Any]:
    """Extract ai_introspect + repo zips, install toolbelt into repo, rebuild sqlite, and smoke-test tools.
    Returns a dict for convenience.
    """
    from bootstrap import bootstrap_from_zips as _bootstrap  # type: ignore
    res = _bootstrap(ai_introspect_zip, repo_zip, work_dir=work_dir, update_fts=update_fts)
    try:
        from dataclasses import asdict as _asdict
        return _asdict(res)
    except Exception:
        return {
            "ai_introspect_root": getattr(res, "ai_introspect_root", None),
            "repo_root": getattr(res, "repo_root", None),
            "db_path": getattr(res, "db_path", None),
            "fts_enabled": getattr(res, "fts_enabled", None),
            "counts": getattr(res, "counts", None),
            "smoke": getattr(res, "smoke", None),
        }
