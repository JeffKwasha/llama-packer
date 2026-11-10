# Pack and validate

The core journey: a model tree becomes a llama-swap config that the router
accepts and advertises.

## Sub-features

- Discover a model under a role-owned tree.
- Emit an entry with a command, metadata and a served context.
- Validate the config with `llama-swap -validate`.
- Serve a throwaway instance and list the entry at `/v1/models`.

## How to get to it (user POV)

```sh
llama-packer --models-dir ~/models --output ~/llama-swap-config.yaml
llama-swap --config ~/llama-swap-config.yaml
curl -s localhost:8080/v1/models
```

## Driving it with the shell harness

```sh
.agents/skills/verify-llama-packer/scripts/run.sh
```

Expected in `.verify/<run>/`:

- `validate.txt` contains `config is valid: 1 model(s)`.
- `config.yaml` has `models.m`, `routing.router.use: capacity`, and
  `models.m.capacities.default`.
- `models.json` contains `"id":"m"`.

The script picks a free port, starts the instance, fetches the list, then
kills that PID and removes `scratch/`. Evidence stays.

## Gotchas

- The fixture GGUF is a header with one zero tensor. `llama-fit-params` may
  crash on it; the packer then emits the entry as `estimated: false` at its
  minimum context. That is the expected fixture behaviour, not a failure.
- Never reuse the user's live instance on `:8123`; always a free port.
- `run.sh` seeds `LLAMA_PACKER_CACHE_DIR` inside scratch, so fit-params crash
  logs do not accumulate in `~/.cache`.
