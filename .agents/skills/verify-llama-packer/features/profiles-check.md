# Profile check

`--check` loads only the profile: it names every problem it can and exits 0
or 1. No discovery, no measurement, no server.

## Sub-features

- The packaged defaults and `profiles.yaml.example` pass.
- An unknown top-level key is named, with a near-miss hint, and exits 1.
- A retired key (`dirs:`, `device_roles:`, `matrix.sets` placeholders) is
  named as retired.

## How to get to it (user POV)

```sh
llama-packer --profiles profiles.yaml --check
echo $?
```

## Driving it with the shell harness

```sh
RUN=.verify/check-$(date -u +%H%M%S); mkdir -p "$RUN"
python3 -m llama_packer --profiles llama_packer/profiles.yaml --check \
  >"$RUN/ok.txt" 2>&1; echo $? >"$RUN/ok.rc"
python3 -m llama_packer --profiles profiles.yaml.example --check \
  >>"$RUN/ok.txt" 2>&1

printf 'profiles:\n  default: {}\nbadkey: 1\n' >"$RUN/bad.yaml"
python3 -m llama_packer --profiles "$RUN/bad.yaml" --check \
  >"$RUN/bad.txt" 2>&1; echo $? >"$RUN/bad.rc"
```

Expected: `ok.rc` is `0`; `bad.rc` is `1`; `bad.txt` names `unknown key
'badkey'`. Run from the repo root so `python3 -m llama_packer` imports the
checkout.

## Gotchas

- `--check` returns before discovery, so `--models-dir` is irrelevant.
- The live root `profiles.yaml` is machine-local and may legitimately fail if
  it still carries a retired key; do not treat that as a code regression.
- The message text is the observable; assert on the key name, not on the
  whole line.
