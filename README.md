# AI Introspection Toolkit – README for Future AI Instances

## Overview
This toolkit provides a set of utilities designed to make deep introspection, debugging, code navigation, and repository analysis safer, faster, and cheaper in terms of both cognitive load and token usage. It is specifically optimized for use within ChatGPT’s Jupyter-style sandbox, where output truncation, timeouts, and repeated full‑repo scanning can cause errors.

The tools live under:
```
.ai_introspect/
    tools/
        ai_introspect_tools.py
        errors_tools.py
    cache/
    errors/
```

They support:
- Safe introspection without printing huge files.
- Reusing previously built indexes to avoid rescanning.
- Understanding function dependency chains.
- Investigating patterns, finding insertion points, and debugging failures.
- Logging errors without polluting user-visible output.
- Ensuring every AI instance—present or future—works with the repo efficiently.

---

# 1. Core Tools

## 1.1 `build_file_hash_index(root_dir, cache_root)`
### **Purpose**
Fingerprint every file in the repository once per session.

### **When to use**
- At the start of a long coding/debugging session.
- Before using symbol or regex indexes.
- Anytime the user says "I updated ZIP" or uploads new repo versions.

### **What it does**
Creates a hash index of all files → allows the toolkit to detect what has changed later.

### **Benefits**
- Prevents rescanning unchanged files.
- Reduces token usage.
- Speeds up every subsequent analysis.

---

## 1.2 `list_modified_since_last_index(root_dir, cache_root)`
### **Purpose**
Identify which files were modified between the last and current introspection pass.

### **When to use**
- Before running deeper introspection steps.
- To decide whether to rebuild symbol indexes or reuse cached data.

### **Benefits**
- Only process files that actually changed.
- Avoid 1000-line dumps or expensive scans.

---

## 1.3 `build_python_symbol_index(path, cache_root)`
### **Purpose**
Create a symbol-index (.json) for a Python module.

### **When to use**
- Before running dependency analysis.
- After modifying important modules like `lab_loop.py`, `radar.py`, or `brain_update.py`.

### **What it does**
Records:
- function definitions
- their locations (path + line)
- references
- calls inside them

### **Benefits**
- The symbol index forms the backbone of call-graph analysis.
- Prevents having to rescan the AST repeatedly.

---

## 1.4 `scan_symbol_dependencies(symbol, direction="forward"/"backward"/"both", max_depth, cache_root)`
### **Purpose**
Analyze how a function is used in the codebase.

### **When to use**
- Before refactoring.
- Before inserting new features (e.g., radar hooks).
- When debugging unexpected behavior.
- When investigating user-reported issues.

### **What it provides**
A dictionary with:
```
{
    "root": "symbol",
    "backward": { depth -> callers },
    "forward": { depth -> callees },
    "function_info": { symbol -> {path, lineno} }
}
```

### **Benefits**
- Full call-chain visibility.
- Prevents breaking dependency chains.
- Helps decide where a patch must be inserted safely.

---

## 1.5 `print_symbol_dependencies(...)`
### **Purpose**
Human-readable dependency tree printer using scan results.

### **When to use**
- When you need a readable call graph quickly.
- Before implementing patches to confirm impact radius.

### **Benefits**
- Avoids printing entire files.
- Avoids overwhelming token streams.
- Helps target precise sections of code.

---

## 1.6 `build_regex_index(path, pattern, cache_root)`
### **Purpose**
Index text occurrences for repeated queries.

### **When to use**
- When searching for repeated patterns (e.g., `"run_brain_update_for_feature"`).
- Before multiple regex passes across large log files.

### **Benefits**
- Reuse results without rescanning files.
- Good for large logs or big Python modules.

---

## 1.7 `regex_context(...)` and `print_regex_context(...)`
### **Purpose**
Provide small, precise context windows around matches.

### **When to use**
- To locate insertion points.
- To inspect portions of very large files.
- To examine patterns repeatedly without reprinting whole modules.

### **Benefits**
- Avoids Jupyter truncation errors.
- Allows small, targeted debug prints.
- Better than printing entire file.

---

## 1.8 `record_error(...)` and `list_errors(...)`
### **Purpose**
Structured error reporting to `.ai_introspect/errors`.

### **When to use**
- Any time code fails or introspection hits unexpected conditions.
- Instead of printing full tracebacks into chat.

### **Benefits**
- Keeps conversation uncluttered.
- Allows persistent, inspectable error history across steps.
- Supports future AI in understanding past failures without re-triggering them.

---

# 2. An Introspection Pipeline (Recommended Pattern)

Here is a minimal, safe introspection workflow:

### **Step 1 — Fingerprint the repo**
```python
build_file_hash_index(root_dir="blob_lab_repo", cache_root=".ai_introspect")
```

### **Step 2 — Check what changed**
```python
mods = list_modified_since_last_index("blob_lab_repo", ".ai_introspect")
```

### **Step 3 — Rebuild symbol indexes (only if needed)**
```python
if "lab_loop.py" in mods:
    build_python_symbol_index("blob_lab_repo/blob_lab/runtime/lab_loop.py", ".ai_introspect")
```

### **Step 4 — Inspect dependencies**
```python
info = scan_symbol_dependencies("run_lab_tick", max_depth=3, cache_root=".ai_introspect")
print_symbol_dependencies("run_lab_tick", max_depth=3, cache_root=".ai_introspect")
```

### **Step 5 — Find code insertion points**
```python
print_regex_context("blob_lab_repo/blob_lab/runtime/lab_loop.py",
                    pattern="run_brain_update_for_feature",
                    lines=4)
```

### **Step 6 — Log structured errors as needed**
```python
record_error("Failed to attach radar hook", error_type="patch", file="lab_loop.py")
```

---

# 3. Jupyter Constraints & How These Tools Solve Them

### **Problem: Large prints cause truncation → timeout → conversation resets**
These tools print:
- small snippets
- controlled context ranges
- summaries instead of full files

### **Problem: Re-running AST scans is expensive and slow**
The tools:
- cache symbol indexes
- cache regex matches
- detect modified files

### **Problem: Full error tracebacks can break output**
The error logger:
- stores traces to disk
- prints only a summary to user

### **Problem: Timeouts happen when introspection loops get too big**
Using:
- dependency scanning
- regex context
- symbol indexes

…avoids touching large parts of the repo.

---

# 4. Benefits for Future AI

### **Higher accuracy**
- You operate on precise dependency graphs, not guesses.
- You avoid hallucinating relationships between functions.

### **Safer analysis**
- You see exactly where patches must go.
- You avoid breaking indirect call chains.
- You avoid triggering code paths unnecessarily.

### **Lower token spend**
- You only print a few lines at a time.
- Cached indexes mean no repeated scans.
- No reprinting 500-line files.

### **Reduced execution time**
- Prevention of slow full-file scans.
- Avoid Jupyter truncation (and its hidden crash conditions).

### **Better continuity**
- Error logs & indexes persist across steps.
- Future AI can review past issues without recomputation.

---

# 5. Final Note

This toolkit exists to make future AI collaborators more capable, more efficient, and more stable when working inside large software systems. Whenever analyzing, modifying, or understanding complex modules:

**Use the tools first  
Then think  
Then patch.**

This ensures correctness, safety, and forward progress with minimal rework.

