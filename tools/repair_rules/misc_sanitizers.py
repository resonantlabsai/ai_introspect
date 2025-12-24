from __future__ import annotations
from typing import Tuple, List, Dict, Any, Union

def applies(error: Union[Exception,str], source: str) -> bool:
    # Always applicable as a safe pass.
    return True

def patch(source: str, error: Union[Exception,str]) -> Tuple[str, List[Dict[str, Any]], float]:
    # Ensure newline at EOF; normalize CRLF to LF.
    patched = source.replace("\r\n", "\n")
    actions: List[Dict[str, Any]] = []
    if patched != source:
        actions.append({"rule":"misc.normalize_newlines", "detail":"Converted CRLF to LF."})
    if patched and not patched.endswith("\n"):
        patched = patched + "\n"
        actions.append({"rule":"misc.ensure_newline_eof", "detail":"Added newline at end of file."})
    if actions:
        return patched, actions, 0.9
    return source, [], 0.0
