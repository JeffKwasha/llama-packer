# Transport: podman

Runs a backend's server inside a **rootless podman container**. Same
implementation as [docker](docker.md) — `ContainerTransport("podman")` — with
podman's device-flag and lifecycle differences.

- Registry name: `<engine>-podman` (e.g. `vllm-podman`).
- **Device flags:** `container_device_flags(vendor, "podman")` — NVIDIA uses
  CDI (`--device nvidia.com/gpu=all`); AMD uses `/dev/kfd` + `/dev/dri` with
  the `video`/`render` groups; CPU adds none. Override with `container_args`
  (legacy alias `podman_args`).
- **Lifecycle:** `stop_cmd = "podman stop ${MODEL_ID}"`,
  `unload_timeout = 30`.
- **Gating:** requires the image resource **and** `podman` on `PATH`
  (`avail["podman"]`).

Podman is preferred over docker in `TRANSPORT_PREFERENCE = ("host", "podman",
"docker")` — a host binary still wins; rootless podman beats docker.

Podman is available everywhere docker is supported; an engine opts in by
listing `"podman"` in `BaseBackend.transports`.
