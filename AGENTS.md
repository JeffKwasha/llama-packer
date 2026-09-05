# AGENTS.md

Rules for agents working in this repo.

## Environments
- Do NOT create a `.venv`. Use the existing environment at `/var/uv/env/bin14`.
- Do NOT make changes to that environment. If any change seems needed, ask the user first.
- Run tests with: `PYTHONDONTWRITEBYTECODE=1 /var/uv/env/bin14/bin/python -m pytest -q -p no:cacheprovider`

## Filesystem
- Never run recursive searches (`find`, `grep`, `glob`, `rg`, ...) on `/home`, `/`, or `/mnt` — they hang forever. Use direct paths and non-recursive `ls`.
- Symlinks are used everywhere, including symlinks to symlinks to symlinks. Resolve before assuming a path is real or dead.

## Tooling
- Node.js is evil. Stay away from it.
