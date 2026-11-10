# Role discovery

A role owns trees with `dir:`. Every model under an owned tree belongs to
that role; a sidecar `role:` overrides it. A model with no role is an error,
not a silent default.

## Sub-features

- A model under `chat/` (owned by the packaged `chat` role) is served.
- A model at the models root with no owning role and no sidecar `role:`
  stops the run with `DiscoveryError`.
- A sidecar `role:` naming an undefined role stops the run.

## How to get to it (user POV)

Put a model in a directory no role lists, or at the models root, and run the
packer. It must refuse by naming the file.

## Driving it with the shell harness

```sh
RUN="$PWD/.verify/discovery-$(date -u +%H%M%S)"; mkdir -p "$RUN/scratch"
python3 .agents/skills/verify-llama-packer/scripts/make-fixture.py \
  "$RUN/scratch" --layout root
(cd "$RUN/scratch" && PYTHONPATH="$PWD/../../.." python3 -m llama_packer \
  --models-dir models --output "$RUN/config.yaml" \
  --llama-server "${LLAMA_SERVER:-llama-server}" --vram 24576 \
  >"$RUN/root.out" 2>"$RUN/root.err"; echo $? >"$RUN/root.rc")
```

Expected: `root.rc` is `1`; `root.err` contains
`no role owns this model`. The owning-tree case is
[`pack-and-validate.md`](pack-and-validate.md), which exits 0.

## Gotchas

- The fixture at the models root has no owning role by design. Do not add a
  role for `.` to "fix" it.
- The failure is a clean `DiscoveryError` message, not a traceback. A
  traceback is the bug.
- A sidecar `role:` override is the way to serve a model outside its tree;
  test that separately rather than moving the fixture.
