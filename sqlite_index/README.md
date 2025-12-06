code_index.sqlite – Lightweight Code Registry for AI-Assisted Editing
1. What this file is

code_index.sqlite is a lightweight SQLite database that acts as a code registry for a Python repository.

Its job is to give AI tooling (and humans) a fast, structured way to answer questions like:

“Where is this function defined?”

“Which file is this class in, and what are its line numbers?”

“Did this file change since we last looked at it?”

Instead of repeatedly scanning or re-sending large Python files, the AI can query this small database to locate and manipulate code by ID, by file, or by definition name.

It is read/write, but intentionally narrow in scope:

Tracks only Python source files (*.py) in the repo.

Ignores volatile artefacts (data files, logs, CAMs, etc.).

Focuses on structural metadata: files, defs, line ranges, hashes.

2. Where it lives and lifecycle

The database is designed to live at the root of the repo, right next to .env or other project-level config:

<repo_root>/
├─ code_index.sqlite   ← this file
├─ .env
├─ pyproject.toml
├─ project1/           (example package)
└─ tests/


Recommended lifecycle:

On the AI side (e.g., in a Jupyter tool environment):

The AI generates or updates code_index.sqlite while working.

Every patch zip it produces includes:

The modified .py files

The updated code_index.sqlite

On the user side:

You unzip over your repo.

Your local copy of code_index.sqlite stays in sync with the code.

When you zip the repo to send it back, you include code_index.sqlite again.

This makes the DB a travel companion with the repo:
wherever the repo goes, the code registry comes along.

3. Schema overview

The schema is intentionally simple: two tables – one for files, one for defs.

3.1 files table

Each row = one Python source file.

CREATE TABLE files (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path        TEXT NOT NULL UNIQUE,    -- relative to repo root
    sha256           TEXT NOT NULL,           -- hash of current contents
    line_count       INTEGER NOT NULL,        -- number of lines in file
    last_indexed_utc TEXT NOT NULL            -- ISO 8601 timestamp (UTC)
);


file_path
Path relative to the repo root (e.g. blob_lab/cli/main_menu.py).

sha256
Hash of the entire file contents; can be used to:

Detect drift (file changed since DB was built).

Prove continuity between an AI-edited file and the DB snapshot.

line_count
Total number of lines (lineno_max), useful for sanity checks.

last_indexed_utc
Timestamp of when this file was last (re)indexed.

3.2 defs table

Each row = one top-level definition (function, async function, or class) in a file.

CREATE TABLE defs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id     INTEGER NOT NULL,             -- FK to files.id
    def_name    TEXT NOT NULL,                -- function/class name
    kind        TEXT NOT NULL,                -- 'function' | 'async_function' | 'class'
    lineno      INTEGER NOT NULL,             -- start line
    end_lineno  INTEGER NOT NULL,             -- end line
    FOREIGN KEY (file_id)
        REFERENCES files(id)
        ON DELETE CASCADE
);


file_id
Foreign key into files.id → one file → many defs.

def_name
The symbol’s name ("start_session", "SessionConfig", etc.).

kind
One of: function, async_function, class.

lineno / end_lineno
1-based line numbers for where the definition starts and ends in the file.

These line ranges come from parsing the file’s AST (e.g. via ast.parse), not from regex guesses.

4. What it does / why it helps AI
4.1 Problem: Big files + token limits

When working with AI-assisted tooling, large repos cause problems:

You can’t send huge files repeatedly (token limits + cost).

You don’t want the AI to “guess” where things are based on old context.

You need a fast, structured way to say:

“Show me only this function.”

“Edit just this class without breaking the rest of the file.”

4.2 Solution: Def-centric, ID-based access

The defs table lets the AI treat code as addressable objects:

defs.id = 42 = “that exact function in that exact file.”

No need to ship the entire file just to talk about one region.

Workflow:

AI queries:

SELECT f.file_path, f.sha256, f.line_count,
       d.def_name, d.kind, d.lineno, d.end_lineno
FROM defs d
JOIN files f ON d.file_id = f.id
WHERE d.id = 42;


The environment:

Loads that file from disk.

Slices only lineno..end_lineno (plus a small context halo).

Presents that snippet to the AI for inspection/edit.

This drastically reduces token usage and keeps the AI grounded in the actual file on disk, not a stale memory of it.

4.3 Safer editing with hashes

files.sha256 is a guardrail:

Before editing:

Environment recomputes the file’s hash.

Compares it with sha256 in the DB.

If mismatch:

The file changed since last index (e.g., human edit, git merge).

The AI should re-index that file before trusting the lineno ranges.

This prevents the classic “line numbers shifted and now I’m editing the wrong block” problem.

5. How to build / rebuild the index

A typical indexer does:

Walk all *.py under <repo_root>.

For each file:

Read text.

Compute sha256(text).

Compute line_count = len(text.splitlines()).

Insert or update row in files.

Parse with ast.parse(text).

For each FunctionDef, AsyncFunctionDef, ClassDef:

Insert row in defs with file_id, def_name, kind, lineno, end_lineno.

You can implement this as a small script, for example:

tools/build_code_index.py

Or as part of an ai_introspect helper command.

In AI workflows, the index is usually:

Built once at the start of a “work session”.

Incrementally updated for files that the AI edits.

6. How an AI would use it (practical patterns)
6.1 “Show me this def” (read-only)

User says: “Inspect start_session in the orchestrator.”

AI does:

SELECT d.id, f.file_path, d.lineno, d.end_lineno
FROM defs d
JOIN files f ON d.file_id = f.id
WHERE d.def_name = 'start_session'
ORDER BY f.file_path;


Environment opens file_path, slices lines lineno..end_lineno, shows that snippet.

No giant file dumps; targeted view.

6.2 “Edit def by ID” (safe write)

AI picks a specific defs.id (e.g., 123).

DB lookup gives:

file_path, sha256, lineno, end_lineno.

Environment:

Verifies on-disk hash matches sha256.

Loads file, isolates that range.

Lets the AI propose a new version of the function body.

Environment:

Splices the new text into the file.

Runs ast.parse and py_compile for preflight.

On success:

Writes file.

Recomputes file hash + line count.

Re-indexes that file’s defs (delete old rows in defs for that file_id, reinsert new).

From the AI’s perspective:

It operates on well-defined regions,

With auto-updated line numbers,

And continuity validated by hashes.

6.3 Integration with ai_introspect

code_index.sqlite pairs nicely with ai_introspect:

code_index answers “where?”

Which file,

Which line range,

Hash + length.

ai_introspect answers “what exactly?” and “how does it connect?”

print_regex_context to see nearby lines.

Symbol tools to trace callers/callees or usage.

Hash index tools to see which files changed since last scan.

Recommended pattern:

Use code_index.sqlite to find targets (file + lineno ranges).

Use ai_introspect to zoom in on those targets without printing whole files.

Apply safe edits (with preflight) using the line range from defs.

7. What this enables (now and future)

Right now, code_index.sqlite gives you:

Fast, structured navigation of Python code.

Safe, def-level editing with AST-derived line ranges.

File-level continuity guarantees via hashes.

Reduced token usage in AI conversations (operate by ID, not by raw file dumps).

In the future, this design can be extended with more tables/columns for:

Call graphs (callers / callees).

Type info, decorators, docstrings.

Test coverage mapping (def ↔ tests).

Semantic tags (e.g., “orchestrator,” “guardian,” “lane builder”).

But the core is intentionally small and robust:
a portable, SQLite-based code registry that any AI or toolchain can read and update.

8. Summary

code_index.sqlite is a portable code map for Python repos.

It tracks:

Files (file_path, sha256, line_count, last_indexed_utc)

Defs (def_name, kind, lineno, end_lineno, file_id)

It helps AI:

Locate and edit code precisely,

Avoid hallucinating file structure,

Minimize token usage,

Maintain continuity via hashes.

It’s designed to travel with the repo and play nicely with ai_introspect and patch-zip workflows.