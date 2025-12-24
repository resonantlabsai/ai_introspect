from __future__ import annotations
from typing import Tuple, List, Dict, Any, Union

def applies(error: Union[Exception,str], source: str) -> bool:
    msg = str(error)
    return ("IndentationError" in msg) or ("TabError" in msg) or ("unexpected indent" in msg) or ("unindent does not match" in msg)

def patch(source: str, error: Union[Exception,str]) -> Tuple[str, List[Dict[str, Any]], float]:
    # Mechanical: normalize tabs to 4 spaces and strip trailing whitespace.
    lines = source.splitlines(True)
    changed = False
    out = []
    new_lines = []
    for ln in lines:
        new_ln = ln.replace("\t", "    ")
        new_ln2 = new_ln.rstrip(" \t") + ("\n" if new_ln.endswith("\n") else "")
        if new_ln2 != ln:
            changed = True
        new_lines.append(new_ln2)
    if not changed:
        return source, [], 0.0
    out.append({"rule": "indentation.normalize_tabs_and_trailing_ws", "detail": "Replaced tabs with 4 spaces and stripped trailing whitespace."})
    return "".join(new_lines), out, 0.85
