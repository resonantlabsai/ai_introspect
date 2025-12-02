"""
Python symbol indexing helpers for ai_introspect.

This module provides two layers:

1. Thin wrappers around the core implementations in `ai_introspect_tools` so
   existing behavior and on-disk symbol indexes continue to work unchanged.
2. A light in-memory `SymbolReport` object and helpers that collapse a single
   Python file's symbols into a structured object or plain dict, suitable for
   "surgery" by AI assistants, plus a chunked `print_python_symbols` that
   avoids giant dumps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import ast
import ai_introspect_tools as _core


# ---------------------------------------------------------------------------
# 1. Thin wrappers around existing core functions (disk-based indexes)
# ---------------------------------------------------------------------------


def build_python_symbol_index(path: str | Path, cache_root: str | Path) -> Path:
    """Wrapper for :func:`ai_introspect_tools.build_python_symbol_index`."""
    return _core.build_python_symbol_index(path=path, cache_root=cache_root)


def scan_symbol_dependencies(
    root_dir: str | Path,
    target_symbol: str,
    *,
    cache_root: str | Path,
    recursive: bool = True,
) -> Dict[str, Any]:
    """Wrapper for :func:`ai_introspect_tools.scan_symbol_dependencies`."""
    return _core.scan_symbol_dependencies(
        root_dir=root_dir,
        target_symbol=target_symbol,
        cache_root=cache_root,
        recursive=recursive,
    )


# ---------------------------------------------------------------------------
# 2. In-memory symbol view for a single Python file
# ---------------------------------------------------------------------------


@dataclass
class SymbolReport:
    """Collapsed view of symbols for a single Python source file."""

    path: Path
    imports: List[Tuple[int, str]]
    functions: List[Tuple[int, str]]
    decorators: Dict[str, List[Tuple[int, str]]]
    calls: Dict[str, List[int]]


def _display_import(node: ast.AST) -> List[Tuple[int, str]]:
    """Turn an import/import-from node into display tuples."""
    results: List[Tuple[int, str]] = []
    if isinstance(node, ast.Import):
        lineno = getattr(node, "lineno", None)
        for alias in node.names:
            name = alias.name
            asname = alias.asname
            disp = f"import {name} as {asname}" if asname else f"import {name}"
            if lineno is not None:
                results.append((lineno, disp))
    elif isinstance(node, ast.ImportFrom):
        lineno = getattr(node, "lineno", None)
        module = node.module or ""
        for alias in node.names:
            name = alias.name
            asname = alias.asname
            disp = (
                f"from {module} import {name} as {asname}"
                if asname
                else f"from {module} import {name}"
            )
            if lineno is not None:
                results.append((lineno, disp))
    return results


def _display_decorator(dec: ast.AST) -> str:
    """Render a decorator expression to a readable string."""
    if hasattr(ast, "unparse"):
        try:
            return ast.unparse(dec)  # type: ignore[attr-defined]
        except Exception:
            pass
    if isinstance(dec, ast.Name):
        return dec.id
    if isinstance(dec, ast.Attribute):
        parts = []
        cur = dec
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        parts.reverse()
        return ".".join(parts)
    return repr(dec)


def _display_call_name(node: ast.Call) -> str | None:
    """Return a simple name for a Call node, or None if not representable."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts = []
        cur = func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        parts.reverse()
        return ".".join(parts)
    return None


def load_python_symbols(path: str | Path) -> SymbolReport:
    """Parse a Python file and return a structured symbol summary."""
    p = Path(path).resolve()
    text = p.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(p))

    imports: List[Tuple[int, str]] = []
    functions: List[Tuple[int, str]] = []
    decorators: Dict[str, List[Tuple[int, str]]] = {}
    calls: Dict[str, List[int]] = {}

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.extend(_display_import(node))

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lineno = getattr(node, "lineno", None)
            func_name = node.name
            if lineno is not None:
                functions.append((lineno, func_name))
            if node.decorator_list:
                for dec in node.decorator_list:
                    dec_name = _display_decorator(dec)
                    decorators.setdefault(func_name, []).append(
                        (getattr(dec, "lineno", lineno or 0), dec_name)
                    )

        if isinstance(node, ast.Call):
            name = _display_call_name(node)
            if name is not None:
                lineno = getattr(node, "lineno", None)
                if lineno is not None:
                    calls.setdefault(name, []).append(lineno)

    imports.sort(key=lambda t: t[0])
    functions.sort(key=lambda t: t[0])
    for fn, decs in decorators.items():
        decs.sort(key=lambda t: t[0])
    for name, lines in calls.items():
        lines.sort()

    return SymbolReport(
        path=p,
        imports=imports,
        functions=functions,
        decorators=decorators,
        calls=calls,
    )


def load_python_symbols_dict(path: str | Path) -> Dict[str, Any]:
    """Return a JSON-friendly dict representation of a file's symbols."""
    report = load_python_symbols(path)
    return {
        "path": str(report.path),
        "imports": report.imports,
        "functions": report.functions,
        "decorators": report.decorators,
        "calls": report.calls,
    }


# ---------------------------------------------------------------------------
# 3. Rendering + chunked printer
# ---------------------------------------------------------------------------


def _render_symbol_report(
    report: SymbolReport,
    *,
    show_imports: bool = True,
    show_functions: bool = True,
    show_decorators: bool = True,
    show_calls: bool = True,
    min_call_count: int = 1,
    max_width: int = 120,
) -> List[str]:
    """Convert a SymbolReport to a list of display lines."""
    lines: List[str] = []
    path = report.path

    lines.append(f"[ai_introspect] Symbols for {path}")
    lines.append("-" * 80)

    if show_imports:
        lines.append("\n[Imports]")
        if not report.imports:
            lines.append("  (none)")
        else:
            for lineno, text in report.imports:
                if len(text) > max_width:
                    text = text[: max_width - 3] + "..."
                lines.append(f"  L{lineno:4d}: {text}")

    if show_functions:
        lines.append("\n[Functions]")
        if not report.functions:
            lines.append("  (none)")
        else:
            for lineno, name in report.functions:
                lines.append(f"  L{lineno:4d}: def {name}(...):")

    if show_decorators:
        lines.append("\n[Decorators / hooks]")
        if not report.decorators:
            lines.append("  (none)")
        else:
            for func_name, decs in report.decorators.items():
                for lineno, dec_name in decs:
                    if len(dec_name) > max_width:
                        dec_name = dec_name[: max_width - 3] + "..."
                    lines.append(f"  L{lineno:4d}: @{dec_name}  ->  {func_name}()")

    if show_calls:
        lines.append("\n[Calls]")
        filtered = {
            name: sorted(lns)
            for name, lns in report.calls.items()
            if len(lns) >= min_call_count
        }
        if not filtered:
            lines.append("  (none)")
        else:
            for name, lns in sorted(filtered.items()):
                disp = ", ".join(str(l) for l in lns)
                lines.append(f"  {name}()  <-  {disp}")

    return lines


def _print_in_chunks(lines: List[str], chunk_size: int = 40) -> None:
    """Print lines in fixed-size chunks to avoid giant dumps."""
    for i in range(0, len(lines), chunk_size):
        chunk = lines[i : i + chunk_size]
        for line in chunk:
            print(line)


def print_python_symbols(
    path: str | Path,
    *,
    show_imports: bool = True,
    show_functions: bool = True,
    show_decorators: bool = True,
    show_calls: bool = True,
    min_call_count: int = 1,
    max_width: int = 120,
    chunk_size: int = 40,
) -> None:
    """Chunked printer built on top of `load_python_symbols`.

    This does not rely on the core `ai_introspect_tools.print_python_symbols`
    implementation; instead it mirrors its display intent while routing
    through the in-memory SymbolReport, which is safer for AI-driven use.
    """
    report = load_python_symbols(path)
    lines = _render_symbol_report(
        report,
        show_imports=show_imports,
        show_functions=show_functions,
        show_decorators=show_decorators,
        show_calls=show_calls,
        min_call_count=min_call_count,
        max_width=max_width,
    )
    _print_in_chunks(lines, chunk_size=chunk_size)
