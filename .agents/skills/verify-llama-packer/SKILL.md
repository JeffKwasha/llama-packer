---
name: verify-llama-packer
description: Drive llama-packer end to end — pack a fixture model tree, validate the generated config with llama-swap, and prove the served model list. Use when verifying packer, discovery, profile, routing, or entry changes.
---

# Verify llama-packer

The app is the `python -m llama_packer` CLI plus the llama-swap instance that
consumes the config it writes. There is no web UI. A proof packs a real model
tree, checks that llama-swap accepts the config, starts a throwaway instance
on a free port, and reads `/v1/models`.

Never drive the live instance on `:8123`. It belongs to the user and holds
their models.

## Launch

```sh
.agents/skills/verify-llama-packer/scripts/run.sh
```

It creates a scratch tree, packs one fixture model, runs
`llama-swap -validate`, starts llama-swap on a free port, fetches
`/v1/models`, then tears the instance and scratch tree down. Evidence
survives. To keep a run at a known path:

```sh
.agents/skills/verify-llama-packer/scripts/run.sh .verify/my-run
```

Binaries resolve as `$LLAMA_SWAP` / `$LLAMA_SERVER`, else `command -v`, else
`/opt/llama/tools/llama-swap` and `/opt/llama/tools/llama-server`. The packer
finds `llama-fit-params` beside `llama-server`.

## Doctor

```sh
.agents/skills/verify-llama-packer/scripts/doctor.sh
```

Read-only: the `llama_packer` module imports, `gguf` is installed (the fixture
writer needs it), `llama-swap -version` answers, and `llama-server` has a
sibling `llama-fit-params`. Run it first when a drive fails oddly.

## Drive

The harness is the shell plus `curl`. Stable handles:

- model id: the sidecar stem (`m` for `m.md`) unless `model_id:` overrides it;
- entry id: `<model id>[.<task>][.<flags>]` in the generated `config.yaml`;
- served list: `GET http://127.0.0.1:<port>/v1/models`, field `data[].id`;
- config acceptance: `llama-swap -validate -config <file>`.

Pack a tree directly when a feature needs a different shape:

```sh
SCRATCH=$(mktemp -d)
python3 .agents/skills/verify-llama-packer/scripts/make-fixture.py "$SCRATCH"
(cd "$SCRATCH" && PYTHONPATH="$PWD/../../.." python3 -m llama_packer \
  --models-dir models --output "$SCRATCH/config.yaml" \
  --llama-server "${LLAMA_SERVER:-llama-server}" --vram 24576)
```

`make-fixture.py` takes `--layout root` to put the model at the models root,
and `--tasks` / `--engines` to change the sidecar.

## Evidence

`run.sh` writes `$ROOT/.verify/<UTC>/` (or the directory you pass):

| file | proves |
|---|---|
| `doctor.txt` | the toolchain that ran |
| `packer.out`, `packer.err` | what the packer did, including warnings |
| `config.yaml` | the emitted config |
| `validate.txt` | llama-swap accepted it (`config is valid: 1 model(s)`) |
| `models.json` | `/v1/models` lists the entry |
| `swap.log` | the instance that answered |

Proof standards: exercise the CLI as a user does (`--models-dir`, `--profiles`,
`--check`, `--dry-run`), not internal functions. Capture the action and the
result: config plus the served list, not one alone. Check side effects (files
written, exit codes) next to stdout. `--dry-run` must leave no files; prove it
by comparing a tree snapshot, not by trusting the flag.

The fixture GGUF is a header with one zero-filled tensor. `llama-fit-params`
may refuse it, and the entry then carries `estimated: false` at its minimum
context. That is expected; it is not a verification failure.

## Cleanup

`run.sh` traps EXIT, INT, TERM and HUP, kills the llama-swap PID it started,
and removes `.verify/.scratch/<pid>-*`; evidence stays. Stop instances by the
PID you started, never by process name.

A SIGKILLed run cannot trap. Two mechanisms cover it: the server runs under
`timeout ${VERIFY_SERVER_TIMEOUT:-20}`, so it dies on its own even when the
script does not, and the next run wipes `.verify/.scratch/*` unconditionally
before creating its own. Raise `VERIFY_SERVER_TIMEOUT` only for a feature that
loads a real model.

```sh
.agents/skills/verify-llama-packer/scripts/clean.sh         # scratch only
.agents/skills/verify-llama-packer/scripts/clean.sh --all   # evidence too
```

IO budget: the fixture is a header plus one 16×16 tensor (a few KB) with a
seeded `derived:` block, so the packer skips `llama-fit-params`. A full run is
about one second and writes ~10 small text files. `LLAMA_PACKER_CACHE_DIR`
points inside scratch, so any measurement artifact dies with it. Nothing here
copies a real model.

## Helpers

- `scripts/doctor.sh` — read-only readiness.
- `scripts/make-fixture.py <dir> [--layout root] [--tasks ...] [--engines ...]` — writes `models/…/m.gguf` and `m.md`.
- `scripts/run.sh [evidence-dir]` — the full pack → validate → serve → list proof.

## Feature map

[`features/README.md`](features/README.md) indexes the user-facing features
and the journey that proves each one.
