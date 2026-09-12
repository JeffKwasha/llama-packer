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

See [`docs/architecture.md`](../architecture.md) and
[`llama_packer/backends/transport.py`](../../llama_packer/backends/transport.py).
