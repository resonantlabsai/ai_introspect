from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import ast
import fnmatch
import hashlib
import io
import zipfile

# ----------------------------
# Defaults
# ----------------------------

DEFAULT_INCLUDE_GLOBS = ["**/*.py"]
DEFAULT_EXCLUDE_GLOBS = [
    ".git/**",
    "**/__pycache__/**",
    "**/*.pyc",
    ".venv/**",
    "venv/**",
    ".mypy_cache/**",
    ".pytest_cache/**",
    ".ruff_cache/**",
    "node_modules/**",
    "dist/**",
    "build/**",
    "repo_index/**",
    "**/*.zip",
    "**/*.log",
]

TEXT_EXTS = {".py", ".md", ".txt", ".toml", ".yml", ".yaml", ".json"}


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _normalize_text_bytes(b: bytes) -> bytes:
    # normalize to reduce false drift from line endings / trailing whitespace noise
    try:
        s = b.decode("utf-8", errors="replace")
    except Exception:
        # fall back to raw
        return b
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # strip trailing whitespace
    s = "\n".join([ln.rstrip(" \t") for ln in s.split("\n")])
    # ensure newline EOF if non-empty
    if s and not s.endswith("\n"):
        s += "\n"
    return s.encode("utf-8")


def _match_any(path: str, globs: List[str]) -> bool:
    # fnmatch doesn't understand ** semantics perfectly, but works well enough for our use:
    # - We normalize paths to forward slashes.
    p = path.replace("\\", "/")
    return any(fnmatch.fnmatch(p, g) for g in globs)


def _should_include(path: str, include_globs: List[str], exclude_globs: List[str]) -> bool:
    p = path.replace("\\", "/").lstrip("/")
    if _match_any(p, exclude_globs):
        return False
    return _match_any(p, include_globs)


def _hash_for_path(rel_path: str, b: bytes, hash_mode: str) -> str:
    ext = Path(rel_path).suffix.lower()
    if hash_mode == "raw":
        return _sha256_bytes(b)
    # normalized: normalize text types; keep binary raw
    if ext in TEXT_EXTS:
        return _sha256_bytes(_normalize_text_bytes(b))
    return _sha256_bytes(b)


def build_manifest_from_dir(
    local_repo_root: str,
    *,
    include_globs: Optional[List[str]] = None,
    exclude_globs: Optional[List[str]] = None,
    hash_mode: str = "normalized",
) -> Dict[str, Any]:
    root = Path(local_repo_root).resolve()
    include_globs = include_globs or list(DEFAULT_INCLUDE_GLOBS)
    exclude_globs = exclude_globs or list(DEFAULT_EXCLUDE_GLOBS)

    files: Dict[str, Dict[str, Any]] = {}

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root)).replace("\\", "/")
        if not _should_include(rel, include_globs, exclude_globs):
            continue
        b = p.read_bytes()
        files[rel] = {
            "hash": _hash_for_path(rel, b, hash_mode),
            "bytes": len(b),
        }

    return {
        "kind": "dir",
        "root": str(root),
        "hash_mode": hash_mode,
        "include_globs": include_globs,
        "exclude_globs": exclude_globs,
        "files": files,
    }


def _detect_zip_root(names: List[str]) -> str:
    # If all entries share a single top-level folder, treat that as root prefix.
    # e.g., "orion/..." or "repo/..."
    # Return "" if no single root.
    tops = set()
    for n in names:
        n = n.replace("\\", "/")
        if not n or n.endswith("/"):
            continue
        top = n.split("/", 1)[0]
        if top:
            tops.add(top)
    if len(tops) == 1:
        return next(iter(tops)) + "/"
    return ""


def build_manifest_from_zip(
    uploaded_zip_path: str,
    *,
    include_globs: Optional[List[str]] = None,
    exclude_globs: Optional[List[str]] = None,
    hash_mode: str = "normalized",
) -> Dict[str, Any]:
    zpath = Path(uploaded_zip_path).resolve()
    include_globs = include_globs or list(DEFAULT_INCLUDE_GLOBS)
    exclude_globs = exclude_globs or list(DEFAULT_EXCLUDE_GLOBS)

    files: Dict[str, Dict[str, Any]] = {}
    with zipfile.ZipFile(zpath, "r") as z:
        names = [n for n in z.namelist() if n and not n.endswith("/")]
        prefix = _detect_zip_root(names)
        for n in names:
            nn = n.replace("\\", "/")
            rel = nn[len(prefix):] if prefix and nn.startswith(prefix) else nn
            rel = rel.lstrip("/")
            if not rel:
                continue
            if not _should_include(rel, include_globs, exclude_globs):
                continue
            b = z.read(n)
            files[rel] = {
                "hash": _hash_for_path(rel, b, hash_mode),
                "bytes": len(b),
            }

    return {
        "kind": "zip",
        "zip_path": str(zpath),
        "hash_mode": hash_mode,
        "include_globs": include_globs,
        "exclude_globs": exclude_globs,
        "files": files,
    }


def compare_manifests(
    local_manifest: Dict[str, Any],
    uploaded_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    lf = local_manifest.get("files", {})
    uf = uploaded_manifest.get("files", {})

    local_paths = set(lf.keys())
    up_paths = set(uf.keys())

    only_local = sorted(local_paths - up_paths)
    only_up = sorted(up_paths - local_paths)

    mismatches: List[Dict[str, Any]] = []
    for p in sorted(local_paths & up_paths):
        if lf[p]["hash"] != uf[p]["hash"]:
            mismatches.append({
                "path": p,
                "local_hash": lf[p]["hash"],
                "uploaded_hash": uf[p]["hash"],
                "local_bytes": lf[p]["bytes"],
                "uploaded_bytes": uf[p]["bytes"],
            })

    return {
        "only_in_local": only_local,
        "only_in_uploaded": only_up,
        "hash_mismatch": mismatches,
        "compared": len(local_paths & up_paths),
    }


# ----------------------------
# Symbol-level drift
# ----------------------------

def _module_name_from_relpath(rel: str) -> str:
    rel = rel.replace("\\", "/")
    if rel.endswith(".py"):
        rel = rel[:-3]
    return rel.replace("/", ".")


def _extract_symbols_from_source(rel_path: str, source_text: str) -> Dict[str, Tuple[str, str]]:
    """
    Returns mapping symbol_id -> (kind, hash)
    symbol_id formats:
      - module:func
      - module:Class
      - module:Class.method
    Hash is structural AST hash (ast.dump without attributes).
    """
    mod = _module_name_from_relpath(rel_path)
    out: Dict[str, Tuple[str, str]] = {}
    try:
        tree = ast.parse(source_text, filename=rel_path)
    except Exception:
        return out

    def node_hash(node: ast.AST) -> str:
        dumped = ast.dump(node, include_attributes=False)
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sid = f"{mod}:{node.name}"
            out[sid] = ("function", node_hash(node))
        elif isinstance(node, ast.ClassDef):
            sid = f"{mod}:{node.name}"
            out[sid] = ("class", node_hash(node))
            for b in node.body:
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    mid = f"{mod}:{node.name}.{b.name}"
                    out[mid] = ("method", node_hash(b))
    return out


def symbol_drift_for_file(
    rel_path: str,
    *,
    local_bytes: bytes,
    uploaded_bytes: bytes,
) -> Dict[str, Any]:
    # parse as text (normalized decode)
    ltxt = local_bytes.decode("utf-8", errors="replace")
    utxt = uploaded_bytes.decode("utf-8", errors="replace")

    ls = _extract_symbols_from_source(rel_path, ltxt)
    us = _extract_symbols_from_source(rel_path, utxt)

    lset = set(ls.keys())
    uset = set(us.keys())

    only_local = sorted(lset - uset)
    only_up = sorted(uset - lset)

    changed: List[Dict[str, Any]] = []
    for sid in sorted(lset & uset):
        if ls[sid][1] != us[sid][1]:
            changed.append({
                "symbol": sid,
                "kind": ls[sid][0],
                "local_hash": ls[sid][1],
                "uploaded_hash": us[sid][1],
            })

    return {
        "only_in_local": only_local,
        "only_in_uploaded": only_up,
        "changed": changed,
        "local_symbols": len(ls),
        "uploaded_symbols": len(us),
    }


def wash_repo(
    local_repo_root: str,
    uploaded_zip_path: str,
    *,
    include_globs: Optional[List[str]] = None,
    exclude_globs: Optional[List[str]] = None,
    hash_mode: str = "normalized",
    detail: str = "file",
    max_symbol_files: int = 200,
) -> Dict[str, Any]:
    """
    Compare local repo directory with uploaded zip.

    detail:
      - "file": file-level drift only
      - "symbol": def/class/method-level drift for mismatched .py files

    max_symbol_files: safety cap for symbol drift expansion.
    """
    include_globs = include_globs or list(DEFAULT_INCLUDE_GLOBS)
    exclude_globs = exclude_globs or list(DEFAULT_EXCLUDE_GLOBS)

    local_manifest = build_manifest_from_dir(
        local_repo_root,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
        hash_mode=hash_mode,
    )
    uploaded_manifest = build_manifest_from_zip(
        uploaded_zip_path,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
        hash_mode=hash_mode,
    )

    cmp = compare_manifests(local_manifest, uploaded_manifest)

    report: Dict[str, Any] = {
        "ok": True,
        "local_root": local_manifest["root"],
        "uploaded_zip": uploaded_manifest["zip_path"],
        "hash_mode": hash_mode,
        "include_globs": include_globs,
        "exclude_globs": exclude_globs,
        "summary": {
            "only_in_local": len(cmp["only_in_local"]),
            "only_in_uploaded": len(cmp["only_in_uploaded"]),
            "hash_mismatch": len(cmp["hash_mismatch"]),
            "compared": cmp["compared"],
        },
        "only_in_local": cmp["only_in_local"],
        "only_in_uploaded": cmp["only_in_uploaded"],
        "hash_mismatch": cmp["hash_mismatch"],
    }

    if detail != "symbol":
        return report

    # symbol drift for mismatched python files
    symbol_drift: Dict[str, Any] = {}
    mismatched_py = [m for m in cmp["hash_mismatch"] if m["path"].endswith(".py")]
    # safety cap
    for m in mismatched_py[:max_symbol_files]:
        rel = m["path"]
        # fetch bytes again (local and zip) safely
        lp = Path(local_manifest["root"]) / rel
        if not lp.exists():
            continue
        lb = lp.read_bytes()

        # read from zip by re-opening (bounded list)
        with zipfile.ZipFile(uploaded_manifest["zip_path"], "r") as z:
            # try direct path; also try prefix-rooted
            name = rel
            if name in z.namelist():
                ub = z.read(name)
            else:
                # attempt with detected zip root
                names = [n for n in z.namelist() if n and not n.endswith("/")]
                prefix = _detect_zip_root(names)
                alt = prefix + rel if prefix else rel
                ub = z.read(alt) if alt in z.namelist() else b""
        if not ub:
            continue
        symbol_drift[rel] = symbol_drift_for_file(rel, local_bytes=lb, uploaded_bytes=ub)

    report["symbol_drift"] = symbol_drift
    report["summary"]["symbol_files_analyzed"] = len(symbol_drift)
    return report
