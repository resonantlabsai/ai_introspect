"""
Regex and log-search helpers for ai_introspect.

This module wraps the regex context and log search utilities from
`ai_introspect_tools.py` to provide a focused import surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from . import ai_introspect_tools as _core


def build_regex_index(
    path: str | Path,
    pattern: str,
    *,
    cache_root: str | Path,
) -> Path:
    """Wrapper for :func:`ai_introspect_tools.build_regex_index`.

    Precomputes a regex index for the given file or directory.
    """
    return _core.build_regex_index(path=path, pattern=pattern, cache_root=cache_root)


def regex_context(
    path: str | Path,
    pattern: str,
    *,
    lines: int = 3,
    direction: str = "both",
    flags: int = 0,
    encoding: str = "utf-8",
    max_width: int = 120,
) -> List[Dict[str, Any]]:
    """Wrapper for :func:`ai_introspect_tools.regex_context`.

    Returns line-numbered context windows around regex matches.
    """
    return _core.regex_context(
        path=path,
        pattern=pattern,
        lines=lines,
        direction=direction,
        flags=flags,
        encoding=encoding,
        max_width=max_width,
    )


def print_regex_context(
    path: str | Path,
    pattern: str,
    *,
    lines: int = 3,
    direction: str = "both",
    flags: int = 0,
    encoding: str = "utf-8",
    max_width: int = 120,
) -> None:
    """Wrapper for :func:`ai_introspect_tools.print_regex_context`.

    Convenience printer for regex_context with the same semantics.
    """
    return _core.print_regex_context(
        path=path,
        pattern=pattern,
        lines=lines,
        direction=direction,
        flags=flags,
        encoding=encoding,
        max_width=max_width,
    )


def search_logs(
    root_dir: str | Path,
    pattern: str,
    *,
    cache_root: str | Path,
    max_results: int = 50,
) -> List[Dict[str, Any]]:
    """Wrapper for :func:`ai_introspect_tools.search_logs`.

    High-level helper to search previously recorded logs for a pattern.
    """
    return _core.search_logs(
        root_dir=root_dir,
        pattern=pattern,
        cache_root=cache_root,
        max_results=max_results,
    )
