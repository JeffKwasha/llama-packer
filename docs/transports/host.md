# Transport: host

Runs a backend's server as a **bare child process** on the host, directly
managed by llama-swap.

- Registry name: the bare engine name (`llama-server`, `vllm`, `sd-server`,
  `whisper-server`, `audio-cpp`).
- No path translation: `PathMap.ref(p) == str(p)`.
- No mounts, no injected container env.
- No lifecycle fields: `stop_cmd` / `unload_timeout` are `None`, so llama-swap
  owns the process (and can stop it by killing the child).
- Availability is the engine's host requirements (`host_requires`), e.g. the
  `llama_bin`, `vllm_bin`, `whisper_bin`, `audio_cpp_bin` resource.

Host is the default and preferred transport: `TRANSPORT_PREFERENCE =
("host", "podman", "docker")`. An engine that cannot run on the host simply
omits `"host"` from `BaseBackend.transports`.

## Runtime config files

A backend whose server needs a per-model config file at launch (today only
audio-cpp's `server.json`) writes it from a shell heredoc inside `cmd`, so
the configuration stays visible in `config.yaml`:

- Runtime-written files live flat under `/tmp/llama-swap/` (created with
  `mkdir -p` in the `cmd` itself); the filename embeds `${PORT}` so parallel
  instances never collide. They are rewritten on every model load.
- Files that must live as long as `config.yaml` go in `configs/` next to it.
- Multi-line `cmd` values are emitted as YAML literal blocks (`|`), never
  folded, so heredoc newlines round-trip exactly (`writer.dump_yaml`).

A future config-file backend follows this convention; no shared helper is
provided until a second backend needs one.

See [`docs/architecture.md`](../architecture.md) and
[`llama_packer/backends/transport.py`](../../llama_packer/backends/transport.py).
