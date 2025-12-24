from __future__ import annotations
from typing import Tuple, List, Dict, Any, Union

BAD_QUOTES = {
    "\u2018": "'", "\u2019": "'",
    "\u201C": '"', "\u201D": '"',
}

def applies(error: Union[Exception,str], source: str) -> bool:
    msg = str(error)
    return ("UnicodeDecodeError" in msg) or ("invalid character" in msg) or any(q in source for q in BAD_QUOTES)

def patch(source: str, error: Union[Exception,str]) -> Tuple[str, List[Dict[str, Any]], float]:
    patched = source
    actions: List[Dict[str, Any]] = []
    for bad, good in BAD_QUOTES.items():
        if bad in patched:
            patched = patched.replace(bad, good)
    if patched != source:
        actions.append({"rule":"encoding.replace_smart_quotes", "detail":"Replaced common smart quotes with ASCII quotes."})
        return patched, actions, 0.75
    return source, [], 0.0
