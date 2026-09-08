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

## Measurement runs (probe-memory, fit-sweep, any llama-server load)
Two legal modes, nothing else:

1. **Fast iterate** (the default): one measurement point at a time —
   one model, one (ctx, parallel) cell — sub-4-minute wall time.
   Process the log + journal IMMEDIATELY, refine code or flags, rerun.
   Every real bug in the measurement pipeline (stall, spill, MTP gate,
   buffering) was found this way in one instrumented run.
2. **Fire-and-collect** (only after mode 1 has survived every code path
   live): launch detached (`setsid`, log to a file), then poll with
   `sleep ≤120` and — this is the part agents get wrong — every poll
   must READ the new journal/log lines and make a decision (continue /
   intervene / kill / fix). Never chain two sleeps. Never sleep without
   having processed the previous result. Never relaunch a batch that
   just failed; diagnose the single failing point first.

Forbidden patterns (all observed 2026-09-08):
- launching a multi-arch × multi-point batch as the FIRST live test of
  unproven code;
- `sleep 290; tail; sleep 290; tail` — blind, decision-free polling;
- rerunning a full batch instead of reproducing one failed point.

Drives: a measurement load reads 6-20 GB from platter; keep strictly
one heavy reader, and keep total batch wall time announced to the user
before starting. The run journal (~/.cache/llama-packer/measure-journal.jsonl)
records per-point duration — read its dur_s trend to catch drift early.
