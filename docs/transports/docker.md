# Transport: docker

Runs a backend's server inside a **docker container**. One implementation,
`ContainerTransport("docker")`, shared with [podman](podman.md) — the two are
CLI-compatible for the `run`/`stop` surface used here.

- Registry name: `<engine>-docker` (e.g. `vllm-docker`).
- Wraps the engine's serve command:
  `docker run --init --rm [args] --name ${MODEL_ID} [mounts] [env] -p ${PORT}:<port> <image> <serve>`.
- **Path translation:** every path the engine emits (model file, chat
  template, speculative draft) is rewritten to a container path by
  `ContainerTransport.path_map`:
  1. under the configured HF cache root → `/root/.cache/huggingface/<rel>`
     (the root is bind-mounted at that path, so HF snapshot blob symlinks
     resolve);
  2. under a configured models dir → `/models`, `/models2`, …;
  3. otherwise a read-only parent bind `-v <parent>:/extN`.
- **Env:** `HF_HOME=/root/.cache/huggingface`, `HF_HUB_OFFLINE=1` — nothing
  downloads; models must be pre-staged.
- **Lifecycle:** `stop_cmd = "docker stop ${MODEL_ID}"`,
  `unload_timeout = 30`. Without `cmdStop`, llama-swap can only kill the
  `docker run` client and leaves the container (and its VRAM) alive.
- **Device flags:** `container_device_flags(vendor, "docker")` — NVIDIA →
  `--runtime=nvidia --gpus all`; AMD → `/dev/kfd` + `/dev/dri` nodes; CPU →
  none. Override the whole set with `container_args` (legacy alias
  `docker_args`).
- **Gating:** requires the image resource (e.g. `vllm_image`) **and** `docker`
  on `PATH` (`avail["docker"]`).

Engines declare `"docker"` in `BaseBackend.transports` to be offered here.
`vllm` is the shipped example (`vllm`, `vllm-podman`, `vllm-docker`).
