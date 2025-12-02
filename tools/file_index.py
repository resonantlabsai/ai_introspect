"""
File index helpers for ai_introspect.

This module provides a thin wrapper around the core implementations in
`ai_introspect_tools.py`, so higher-level code can import from a focused
namespace without changing the existing public API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional

from . import ai_introspect_tools as _core


def build_file_hash_index(root_dir: str | Path, cache_root: str | Path) -> Path:
    """Wrapper for :func:`ai_introspect_tools.build_file_hash_index`.

    See the README for details. This is a convenience re-export so callers
    can import from `.file_index` if they prefer more focused modules.
    """
    return _core.build_file_hash_index(root_dir=root_dir, cache_root=cache_root)


def list_modified_since_last_index(
    root_dir: str | Path,
    cache_root: str | Path,
) -> List[Path]:
    """Wrapper for :func:`ai_introspect_tools.list_modified_since_last_index`.

    Returns the list of modified files since the last hash index was built.
    """
    return _core.list_modified_since_last_index(root_dir=root_dir, cache_root=cache_root)
