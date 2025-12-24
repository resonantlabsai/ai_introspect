from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional
import zipfile
import hashlib
from datetime import datetime

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def package_files(
    repo_root: str,
    files: Iterable[str],
    zip_path: str,
    *,
    include_manifest: bool = True,
    manifest_name: str = "MANIFEST.json",
) -> Dict[str, Any]:
    root = Path(repo_root).resolve()
    zpath = Path(zip_path)
    zpath.parent.mkdir(parents=True, exist_ok=True)

    file_list: List[Path] = []
    for f in files:
        p = (root / f).resolve() if not str(f).startswith(str(root)) else Path(f).resolve()
        if p.exists() and p.is_file():
            file_list.append(p)

    manifest: Dict[str, Any] = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "repo_root": str(root),
        "files": [],
    }

    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in file_list:
            rel = p.relative_to(root)
            z.write(p, arcname=str(rel))
            if include_manifest:
                manifest["files"].append({
                    "path": str(rel),
                    "bytes": p.stat().st_size,
                    "sha256": _sha256(p),
                })
        if include_manifest:
            z.writestr(manifest_name, __import__("json").dumps(manifest, indent=2, sort_keys=True))

    return {"zip_path": str(zpath), "file_count": len(file_list), "manifest_included": include_manifest}
