# AGENTS.md

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
- avoid running commands that will take more than 4 minutes to complete
- when one step of a multi-step action fails, it's best to skip the prior steps.
- if you have to run a slow command, consider running it in a subagent and doing something else in the meantime.

Platter drives read at 100MB/second - estimate time to complete before firing off a command
loading a 22GB model from a platter will almost always take more than 4 minutes.

## References in docs

Docs cite the taxonomy, not the tree's current shape. Line numbers and PR
numbers are both stale-on-arrival: `SPEC 305` pointed at entry ids while the
row it meant had moved, and "PR #4" says nothing to anyone who did not read
that PR.

Legal references, in order of preference:

1. a `SPEC.md` §section or `docs/` page — `SPEC §Knobs (matrix section keys)`;
2. a `requirement-trace.md` row — `requirement #74`;
3. a symbol — `_host_unit_refusals`, `models_AGENTS.md`.

If none of those says it, explain it right there in the sentence. Never
`file.py:123`, never `SPEC 305`, never `PR #12`, never a second copy of the
version number (it lives in `llama_packer/__init__.py`).

## Deleting a file that docs point at

Plan files are records, not contracts: they die when the work lands. Their
references must not die with them — a dangling path is a claim nobody can
check.

1. Find every reference:
   `grep -rn '<basename>' --include='*.md' --include='*.py' --include='*.yaml' .`
2. Resolve each hit. There are only two legal outcomes:
   - **the claim is no longer true** → delete the sentence, not just the path;
   - **the claim still stands** → repoint it at where the information now
     lives (SPEC §section, `docs/` page, requirement-trace row).
3. Do not guess where it moved. Search for the claim in plain words:
   `qmd query "<the claim>"` (hybrid BM25 + vector over the indexed
   collections) or `qmd search "<exact phrase>"` for a literal hit.

Only then delete. The grep for that basename must come back clean. Same rule
for renames.
