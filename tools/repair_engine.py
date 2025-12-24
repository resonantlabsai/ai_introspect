from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional
import ast

from .repair_rules import indentation, encoding, misc_sanitizers

RuleMod = Any

_DEFAULT_RULES: List[Tuple[str, RuleMod]] = [
    ("misc", misc_sanitizers),
    ("indent", indentation),
    ("encoding", encoding),
]

def _pick_rules(modes: List[str]) -> List[Tuple[str, RuleMod]]:
    out: List[Tuple[str, RuleMod]] = []
    mode_set = set(modes or [])
    for name, mod in _DEFAULT_RULES:
        if not mode_set or name in mode_set:
            out.append((name, mod))
    return out

def try_parse(source: str, path: str = "<string>") -> Optional[Exception]:
    try:
        ast.parse(source, filename=path)
        return None
    except Exception as e:
        return e

def repair_source(
    path: str,
    source: str,
    error: Union[Exception, str],
    *,
    modes: List[str] | None = None,
) -> Tuple[str, List[Dict[str, Any]], float]:
    """
    Return (patched_source, actions, confidence).
    Only applies mechanical, low-risk transforms.
    """
    actions: List[Dict[str, Any]] = []
    confidence = 0.0
    patched = source

    rules = _pick_rules(modes or [])
    for rule_name, mod in rules:
        if not getattr(mod, "applies")(error, patched):
            continue
        new_src, acts, conf = getattr(mod, "patch")(patched, error)
        if acts and new_src != patched:
            patched = new_src
            actions.extend(acts)
            confidence = max(confidence, conf)

    return patched, actions, confidence

def repair_file(
    file_path: str,
    *,
    modes: List[str] | None = None,
    max_passes: int = 2,
) -> Dict[str, Any]:
    p = Path(file_path)
    src = p.read_text(encoding="utf-8", errors="replace")
    passes = 0
    all_actions: List[Dict[str, Any]] = []
    last_err: Optional[str] = None
    changed = False

    for _ in range(max_passes):
        err = try_parse(src, str(p))
        if err is None:
            last_err = None
            break
        last_err = f"{type(err).__name__}: {err}"
        patched, actions, conf = repair_source(str(p), src, err, modes=modes)
        if not actions or patched == src:
            break
        src = patched
        all_actions.extend(actions)
        passes += 1
        changed = True

    if changed:
        p.write_text(src, encoding="utf-8")

    # final check
    final_err = try_parse(src, str(p))
    return {
        "path": str(p),
        "changed": changed,
        "passes": passes,
        "actions": all_actions,
        "final_error": None if final_err is None else f"{type(final_err).__name__}: {final_err}",
    }
