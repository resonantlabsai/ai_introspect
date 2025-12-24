from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import ast
import py_compile

from .ai_introspect_tools import _get_root, list_modified_since_last_index
from .repair_engine import repair_file, try_parse
from .sqlite_index import create_build, index_repo
from .packaging_tools import package_files

def _iter_targets(root: Path, targets: Optional[List[str]], use_modified_since_index: bool) -> List[str]:
    if targets:
        return targets
    if use_modified_since_index:
        mods = list_modified_since_last_index(repo_root=str(root))
        # tools returns dict with key "modified"
        files = mods.get("modified", [])
        # default to python files; if none, still allow
        py = [f for f in files if f.endswith(".py")]
        return py or files
    # fallback: all python files
    return [str(p.relative_to(root)) for p in root.rglob("*.py") if p.is_file() and ".ai_introspect" not in str(p)]

def preflight_and_package(
    repo_root: str | None = None,
    *,
    targets: Optional[List[str]] = None,
    use_modified_since_index: bool = True,
    zip_name: str = "update.zip",
    max_repair_passes: int = 2,
    repair_modes: Optional[List[str]] = None,
    update_sqlite_index: bool = True,
    sqlite_state_on_success: str = "good",
    write_manifest: bool = True,
) -> Dict[str, Any]:
    """
    Parse -> auto-repair (mechanical) -> compile verify -> sqlite update -> zip.
    """
    root = _get_root(repo_root)
    repair_modes = repair_modes or ["misc", "indent", "encoding"]

    chosen = _iter_targets(root, targets, use_modified_since_index)
    errors: List[Dict[str, Any]] = []
    repairs: List[Dict[str, Any]] = []
    repaired_files = 0

    # Stage A: parse + repair loop
    for rel in chosen:
        p = (Path(root) / rel)
        if not p.exists() or not p.is_file():
            continue
        src = p.read_text(encoding="utf-8", errors="replace")
        err = try_parse(src, str(p))
        if err is None:
            continue
        rep = repair_file(str(p), modes=repair_modes, max_passes=max_repair_passes)
        if rep.get("changed"):
            repaired_files += 1
        repairs.append(rep)
        if rep.get("final_error"):
            errors.append({"path": rel, "stage": "parse", "message": rep["final_error"]})

    if errors:
        return {
            "ok": False,
            "zip_path": None,
            "checked_files": len(chosen),
            "repaired_files": repaired_files,
            "repairs": repairs,
            "errors": errors,
            "sqlite_updated": False,
            "sqlite_build_id": None,
        }

    # Stage B: compile (with bounded repair attempts on failure)
    for rel in chosen:
        p = (Path(root) / rel)
        if not p.exists() or not p.is_file() or not rel.endswith(".py"):
            continue
        try:
            py_compile.compile(str(p), doraise=True)
        except Exception as e:
            # attempt repair, then recompile
            rep = repair_file(str(p), modes=repair_modes, max_passes=max_repair_passes)
            repairs.append(rep)
            if rep.get("changed"):
                repaired_files += 1
            try:
                py_compile.compile(str(p), doraise=True)
            except Exception as e2:
                errors.append({"path": rel, "stage": "compile", "message": f"{type(e2).__name__}: {e2}"})

    if errors:
        return {
            "ok": False,
            "zip_path": None,
            "checked_files": len(chosen),
            "repaired_files": repaired_files,
            "repairs": repairs,
            "errors": errors,
            "sqlite_updated": False,
            "sqlite_build_id": None,
        }

    sqlite_updated = False
    build_id = None
    if update_sqlite_index:
        db_path = Path(root) / "repo_index" / "code_index.sqlite"
        build_id = create_build(str(db_path), sqlite_state_on_success)
        index_repo(str(root), str(db_path), build_id=build_id, state=sqlite_state_on_success)
        sqlite_updated = True

    # zip the chosen targets (or the modified list if available)
    zip_path = str(Path(root) / zip_name)
    pkg = package_files(str(root), chosen, zip_path, include_manifest=write_manifest)

    return {
        "ok": True,
        "zip_path": pkg["zip_path"],
        "checked_files": len(chosen),
        "repaired_files": repaired_files,
        "repairs": repairs,
        "errors": [],
        "sqlite_updated": sqlite_updated,
        "sqlite_build_id": build_id,
    }
