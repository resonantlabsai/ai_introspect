# AI Introspect – Tooling API Guide (Super Toolbelt)

_Toolbelt release: v2.3.x (FTS-aware indexing + Refs + Impact)_
_Last verified: 2025-12-24_

This file documents the **public tools** exposed by `tools.ai_introspect_tools`.

**Drift guard:** the section **“Full Public API Reference (auto-synced)”** below is generated from live function signatures and docstrings, so it’s the source of truth when in doubt.

---

## Quick Start (recommended workflow)
1. **Bootstrap** a repo (extract + index): `bootstrap_from_zips(...)`
2. Use **`nav()`** to jump to code quickly (FTS when available; heuristic fallback otherwise).
3. Use **`show_match()`** to view the relevant snippet/context.
4. Before edits, use **`callers:` / `impact:`** (or `callers()` / `impact()`) to estimate blast radius.
5. After edits, run **`preflight_and_package(..., update_sqlite_index=True)`** to refresh the DB and ship a patch zip.

---

## Notes on FTS (Full-Text Search)
- FTS is **optional**. If SQLite supports FTS5, the atlas can build `atlas_fts` and `nav()` will use it for free-text ranking.
- If FTS is unavailable (or explicitly disabled), `nav()` falls back to a **tokenized heuristic** search (works for multi-token queries, but rankings are less precise than BM25).
- CLI indexing: `.ai_introspect/build_nav/build_nav.py --repo-root <repo> [--no-fts]`

---

## Full Public API Reference (auto-synced)

### Navigation & Code Atlas

- **`nav(query, *, repo_root, limit=20, db_path=None, explain=False, highlight=False)`** — Query the SQLite code atlas for symbols/files.
- **`show_match(match, *, repo_root, db_path=None, max_lines=40)`** — Render a match object from nav(). Prefer returning snippet content from sqlite.
- **`nav_explain(res, *, max_matches=10, max_lines=40)`** — Pretty-print a nav() result within a bounded line budget.
- **`nav_status(*, repo_root, db_path=None)`** — Return basic diagnostics about the nav SQLite DB (schema, counts, FTS availability).
- **`doctor(*, repo_root, db_path=None, top=8)`** — Deeper diagnostics than nav_status (FTS compile options, indexes, top refs).
- **`refs(query, *, repo_root, kind=None, limit=50, db_path=None)`** — Search reference rows (imports/calls) captured during indexing.
- **`callers(target, *, repo_root, limit=50, db_path=None)`** — Find call refs that appear to target `target` (best-effort).
- **`callees(source, *, repo_root, limit=50, db_path=None)`** — Find call refs made *from* `source` symbol (best-effort).
- **`impact(target, *, repo_root, db_path=None, kind=None, limit=200)`** — Summarize what files reference a target (best-effort impact analysis).

### Bootstrap & Packaging

- **`bootstrap_from_zips(ai_introspect_zip, repo_zip, *, work_dir=None, update_fts=True)`** — Extract ai_introspect + repo zips, install toolbelt into repo, rebuild sqlite, and smoke-test tools.
- **`preflight_and_package(repo_root, *, targets=None, use_modified_since_index=True, zip_name='update.zip', update_sqlite_index=True, db_path=None)`** — Preflight + package workflow:
- **`build_file_hash_index(root_dir='.', *, cache_root=None)`**
- **`list_modified_since_last_index(root_dir='.', *, cache_root=None)`**
- **`wash_repo(*args, **kwargs)`** — Compare a local repo tree with an uploaded zip. Optionally drill to def/class/method drift.

### Regex & Text Context

- **`print_regex_context(path, pattern, *, lines=3, direction='both', flags=re.MULTILINE, encoding='utf-8', max_width=120)`**
- **`regex_context(path, pattern, *, lines=3, direction='both', flags=re.MULTILINE, encoding='utf-8')`**
- **`print_regex_function(path, pattern, *, flags=re.MULTILINE, chunk_size=40, encoding='utf-8')`** — Find and print Python functions whose `def` line matches the given pattern.
- **`build_regex_index(path, pattern, *, lines=3, direction='both', flags=re.MULTILINE, encoding='utf-8', cache_root=None, pattern_name=None)`**
- **`read_lines(path, encoding='utf-8')`**
- **`iter_lines_with_numbers(lines, start=1)`**

### Python Symbols & Dependencies

- **`print_python_symbols(path, *, show_imports=True, show_functions=True, show_decorators=True, show_calls=False, min_call_count=1, encoding='utf-8', max_width=120)`**
- **`build_python_symbol_index(path, *, cache_root=None, encoding='utf-8', include_calls=True)`**
- **`scan_symbol_dependencies(symbol, *, direction='both', max_depth=3, cache_root=None)`** — Trace callers and callees for a given symbol using prebuilt symbol indexes.
- **`print_symbol_dependencies(symbol, *, direction='both', max_depth=3, cache_root=None)`** — Pretty-print the dependency stack for a symbol.

### Artifacts & Logs

- **`register_artifact(rel_path, artifact_type, *, source_path=None, tags=None, cache_root=None)`**
- **`search_artifacts(*, artifact_type=None, source_contains=None, tag_equals=None, latest_only=False, cache_root=None)`**
- **`append_log_entry(text, *, role, turn_id, tags=None, cache_root=None)`**
- **`search_logs(pattern, *, roles=None, tags=None, max_hits=20, cache_root=None)`**
- **`record_run_summary(summary, *, run_id, cache_root=None)`**

### Convenience Printers

- **`print_nav_explain(res, *, max_matches=10, max_lines=40)`** — Print nav_explain() output.
- **`print_doctor(res, *, max_lines=40)`** — Pretty-print doctor() output without exceeding max_lines.

### Other Public Tools


---

## Index Builder Script (CLI)

- **`build_nav.py`**: `.ai_introspect/build_nav/build_nav.py`
  - Primary usage: `python build_nav.py --repo-root <repo_root>`
  - Disable FTS explicitly: `python build_nav.py --repo-root <repo_root> --no-fts`

## Doc correctness checklist
- If you change function signatures, regenerate the “auto-synced” section (or update headings).
- If you add new public tools, ensure they appear under “Full Public API Reference”.

