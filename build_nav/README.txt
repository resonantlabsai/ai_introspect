build_nav.py

Run from repo root:
  python -m build_nav
or:
  python build_nav.py

Behavior:
- Always builds+indexes a root-level repo_nav.sqlite (atlas_v1 schema).
- If multiple repo_nav.sqlite exist anywhere in the repo: they are moved into
  .repo_nav_backups/<timestamp>/ and a clean rebuild happens.

Important:
- repo_nav.sqlite contains code snippets. Do NOT commit it to a public repo unless
  you intend to publish that code. Add it to .gitignore by default.
