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
