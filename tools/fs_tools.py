"""
Filesystem helpers for ai_introspect.

This module is intentionally tiny: it provides a safe `create_file` helper
that AIs (and humans) can use to write or overwrite files without having to
re-implement backup / directory creation logic each time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def create_file(
    path: str | Path,
    content: str,
    *,
    exist_ok: bool = False,
    backup: bool = True,
    encoding: str = "utf-8",
) -> Path:
    """Create or overwrite a text file with optional backup.

    Parameters
    ----------
    path:
        Target file path. Parent directories are created automatically.
    content:
        Text content to write.
    exist_ok:
        If False (default), refuse to overwrite an existing file and raise
        FileExistsError. If True, allow overwriting.
    backup:
        If True and the file already exists, write a `.bak` copy alongside
        the original before overwriting.
    encoding:
        Text encoding used when writing the file.
    """
    p = Path(path).resolve()

    if p.exists():
        if not exist_ok:
            raise FileExistsError(f"Refusing to overwrite existing file: {p}")
        if backup:
            backup_path = p.with_suffix(p.suffix + ".bak")
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            backup_path.write_text(p.read_text(encoding=encoding), encoding=encoding)

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding=encoding)
    return p
