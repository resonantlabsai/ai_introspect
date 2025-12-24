# ai_introspect — Repo Atlas + Navigation Toolbelt

A lightweight, offline **code navigation + impact-analysis toolbelt** that builds a local SQLite “Repo Atlas” for a Python repo, then gives you fast, structured queries to answer questions like:

* *Where is this symbol defined?*
* *Who calls it? What does it call?*
* *What files are impacted if I change this function?*
* *Where is this config key / string referenced?*

It’s designed to be:

* **Fast** on small/medium repos
* **FTS-aware** (uses SQLite FTS5 when available; falls back cleanly when it isn’t)
* **Patch-friendly** (preflight + package only changed files)
* **Low-noise** (keeps outputs short by default)

---

## What it builds

* `.ai_introspect/repo_nav.sqlite` — the Repo Atlas database

  * files, symbols, imports, references, snippets
  * optional `atlas_fts` table for full-text search (FTS5)

---

## Quick start

### 1) Put `.ai_introspect/` in your repo

This repo is meant to live inside a target repo like:

```
<your_repo>/.ai_introspect/
```

### 2) Build the Repo Atlas

From your repo root:

```bash
python .ai_introspect/build_nav/build_nav.py --repo-root .
```

If you want to force-disable FTS (for environments without FTS5 or to keep behavior consistent):

```bash
python .ai_introspect/build_nav/build_nav.py --repo-root . --no-fts
```

### 3) Use the toolbelt

In Python:

```python
from pathlib import Path
import sys

# Add the .ai_introspect package root to sys.path
sys.path.append(str(Path(".ai_introspect")))

from tools.ai_introspect_tools import nav, show_match, callers, impact, nav_status

print(nav_status(repo_root="."))

res = nav("guardian rotation", repo_root=".")
print(res["matches"][0]["path"])
print(show_match(res["matches"][0], repo_root="."))
```

---

## `nav()` query language (the natural workflow)

Start with **free text**:

* `nav("guardian rotation", repo_root=".")`

Then switch to **lenses** when you know what you’re looking for:

* `path:<text>` — narrow by path
* `symbol:<name>` — jump to a symbol
* `ref:<token>` — find where a token/string/config-key is referenced
* `callers:<symbol>` — find call sites
* `callees:<symbol>` — find callees (best-effort)
* `impact:<symbol>` — estimate blast radius

Typical loop:

1. `nav("free text")`
2. `show_match(top_match)`
3. `callers("target")` / `impact("target")`
4. Make edits
5. Re-index (or use preflight packaging)

---

## FTS behavior (important)

* If your SQLite supports **FTS5**, `nav()` uses `atlas_fts` for multi-token ranking (BM25).
* If FTS is unavailable or disabled, `nav()` falls back to a **tokenized heuristic search**.

You can always check status:

```python
from tools.ai_introspect_tools import nav_status
print(nav_status(repo_root="."))
```

---

## Preflight + package patches

After you change code, you can run a preflight that:

* syntax-checks targets
* optionally refreshes the Repo Atlas
* packages changed files into a zip (preserving paths)

```python
from tools.ai_introspect_tools import preflight_and_package

out = preflight_and_package(
    repo_root=".",
    use_modified_since_index=True,
    update_sqlite_index=True,
    update_fts=True,   # set False to refresh atlas without FTS
    zip_name="patch.zip",
)
print(out)
```

### Hash baseline note

`use_modified_since_index=True` relies on a hash baseline:

```python
from tools.ai_introspect_tools import build_file_hash_index
build_file_hash_index(".")
```

Run that once per repo (or whenever you want to reset your baseline).

---

## Keeping outputs readable

* `show_match(..., max_lines=40)` limits snippet output by default.
* Prefer “deep but narrow” context rather than printing whole files.

---

## Recommended `.gitignore`

The atlas DB and other generated artifacts should not be committed:

```gitignore
# ai_introspect derived DB + backups
.ai_introspect/repo_nav.sqlite
.ai_introspect/*.bak*
.ai_introspect/**/*.bak*

# common python noise
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/

# logs / artifacts
*.log
logs/
*.zip
```

---

## Troubleshooting

### “Free text returns nothing”

* Check `nav_status()` for `fts_enabled` and `fts_rows`.
* If FTS is enabled but empty, rebuild with FTS on:

```bash
python .ai_introspect/build_nav/build_nav.py --repo-root .
```

Or disable FTS explicitly:

```bash
python .ai_introspect/build_nav/build_nav.py --repo-root . --no-fts
```

### “Modified since index includes everything”

* You probably haven’t built a hash baseline yet:

```python
from tools.ai_introspect_tools import build_file_hash_index
build_file_hash_index(".")
```

---

## Contributing

* Keep outputs short and deterministic.
* Prefer schema additions that are backward compatible.
* If you add a public tool, update `TOOLS_API.md`.

---

## License

Add your preferred license here (MIT/Apache-2.0/etc.).
