# llama_packer/backends/transport.py
"""Transport — how a backend's server process is launched.

A *backend* (engine) decides **what** to serve and composes its command line.
A *transport* decides **how** that process runs: as a bare host process or
inside a container runtime (docker/podman).

The two are independent axes.  An engine declares the transports it can run
under (``BaseBackend.transports``) and the registry materialises one *bound*
backend per supported pair.  This keeps container mechanics — host→container
path translation, mounts, env, lifecycle (stop/unload) — in one place instead
of duplicated per engine, and lets one engine be offered on host, docker and
podman without re-implementing its command line.

docker and podman are CLI-compatible for the ``run``/``stop`` surface used
here, so they are one implementation parameterised by runtime plus a small
device-flag hook — not two classes.
"""

from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Mapping, Sequence

if TYPE_CHECKING:
    from llama_packer.model import Model

logger = logging.getLogger(__name__)

# In-container HF cache root (the upstream images' default HOME cache).  The
# host HF_HOME root (the dir containing ``hub/``) is bind-mounted here and
# HF_HUB_OFFLINE forbids downloads — llama-packer never fetches.
CONTAINER_HF_HOME = "/root/.cache/huggingface"

# Preferred transport order within one engine: a host binary needs no runtime;
# podman is rootless by default; docker last.  Overridable via profiles.yaml
# ``backends:`` (which also orders engines).
TRANSPORT_PREFERENCE = ("host", "podman", "docker")


@dataclass
class PathMap:
    """Host→process path translation for one command, plus its mounts."""

    refs: Mapping[Path, str] = field(default_factory=dict)
    mounts: list[str] = field(default_factory=list)

    def ref(self, path: Path) -> str:
        """The launch-side path for *path* (identity when unmapped)."""
        return self.refs.get(path, str(path))


@dataclass(frozen=True)
class Launch:
    """Engine-supplied container launch details (ignored by the host)."""

    image: str | None = None
    port: int | None = None
    args: str = ""
    env: tuple[str, ...] = ()


class Transport(ABC):
    """How a bound backend's process is launched and supervised."""

    name: str = "host"
    #: True for container runtimes (needs the runtime binary + an image).
    container: ClassVar[bool] = False
    #: llama-swap lifecycle fields; only container servers set these.
    stop_cmd: str | None = None
    unload_timeout: int | None = None

    def bind_name(self, engine_name: str) -> str:
        """Registry name for *engine_name* under this transport.

        host keeps the bare engine name; every other transport suffixes it.
        """
        return engine_name if self.name == "host" else f"{engine_name}-{self.name}"

    def is_available(self, avail: dict) -> bool:
        """Whether this transport's own resources (runtime binary, …) exist."""
        return True

    def path_map(self, tvars: dict, extra_paths: Sequence[Path]) -> PathMap:
        """Translate host paths for this transport and collect its mounts."""
        return PathMap({p: str(p) for p in extra_paths})

    def wrap(self, serve_cmd: str, model: "Model", launch: Launch,
             paths: PathMap) -> str:
        """Wrap a finished serve command in the transport's launcher."""
        return serve_cmd


class HostTransport(Transport):
    """Launch the server as a bare child process on the host."""

    name = "host"


def container_device_flags(vendor: str, runtime: str) -> str:
    """Vendor GPU pass-through flags for a container runtime.

    One home for GPU plumbing shared by every containerized engine.  docker
    uses the NVIDIA runtime flags (or AMD device nodes); podman uses CDI for
    NVIDIA and device nodes for AMD.  An unknown/CPU vendor adds nothing.
    """
    if vendor == "nvidia":
        if runtime == "podman":
            return "--device nvidia.com/gpu=all"
        return "--runtime=nvidia --gpus all"
    if vendor == "amd":
        return ("--device /dev/kfd --device /dev/dri "
                "--group-add video --group-add render")
    return ""


class ContainerTransport(Transport):
    """Launch the server inside a container runtime (docker, podman, …)."""

    container = True
    unload_timeout = 30

    def __init__(self, runtime: str):
        self.name = runtime
        # llama-swap must stop the container itself: without cmdStop it can
        # only kill the `docker run`/`podman run` client, leaving the container
        # (and its VRAM) alive.  unloadTimeout covers the stop grace.
        self.stop_cmd = f"{runtime} stop ${{MODEL_ID}}"

    def is_available(self, avail: dict) -> bool:
        # `avail` carries the probed runtime presence (set by __main__);
        # a missing key means "not probed" (programmatic/unit use), where we
        # do not gate on the binary.
        present = avail.get(self.name)
        return True if present is None else bool(present)

    def path_map(self, tvars: dict, extra_paths: Sequence[Path]) -> PathMap:
        models_dirs, hf_cache = _container_roots(tvars)
        refs, ext_mounts = _map_paths_into(extra_paths, models_dirs,
                                           hf_cache or None)
        return PathMap(
            refs=dict(zip(sorted(extra_paths, key=str), refs)),
            mounts=[*_root_mounts(models_dirs, hf_cache), *ext_mounts],
        )

    def wrap(self, serve_cmd: str, model: "Model", launch: Launch,
             paths: PathMap) -> str:
        parts = [f"{self.name} run --init --rm"]
        if launch.args:
            parts.append(launch.args)
        parts.append("--name ${MODEL_ID}")
        parts.extend(paths.mounts)
        parts.extend(f"-e {e}" for e in launch.env)
        if launch.port is not None:
            parts.append(f"-p ${{PORT}}:{launch.port}")
        if launch.image:
            parts.append(launch.image)
        return " ".join(parts) + " " + serve_cmd


def _container_roots(tvars: dict) -> tuple[list[str], str]:
    """(models_dirs, hf_cache) from template vars, normalised."""
    models_dirs = tvars.get("models_dirs") or [tvars.get("models_dir", "")]
    models_dirs = [str(d) for d in models_dirs if d]
    return models_dirs, str(tvars.get("hf_cache") or "")


def _root_mounts(models_dirs: list[str], hf_cache: str) -> list[str]:
    """Bind mounts for the models roots and the shared HF hub."""
    mounts = [f"-v {d}:{'/models' if i == 0 else f'/models{i + 1}'}"
              for i, d in enumerate(models_dirs)]
    if hf_cache:
        mounts.append(f"-v {hf_cache}:{CONTAINER_HF_HOME}")
    return mounts


def _map_paths_into(paths: Sequence[Path], models_dirs,
                    hf_cache: str | None = None) -> tuple[list[str], list[str]]:
    """Map host paths for use inside a container.

    Resolution order per path (``Path.resolve()`` first, so symlinked layouts
    map by their *real* location — an HF snapshot symlinked under ``~/models``
    maps into the HF cache branch):

    1. under ``hf_cache`` (host HF_HOME root — the dir containing ``hub/``) →
       ``/root/.cache/huggingface/<rel>``;
       the whole root is bind-mounted separately, so HF snapshot blob symlinks
       (``file → ../../blobs/<hash>``) resolve
    2. under any of ``models_dirs`` → that dir's container target (``/models``,
       ``/models2``, ...)
    3. else → dedicated read-only parent bind (``-v <parent>:/extN``) and an
       ``/extN/<name>`` ref

    Returns ``(container_refs, docker_mount_flags)``, refs aligned with
    ``sorted(paths, key=str)``.  The hf_cache/roots mounts are the transport's
    own concern (see :meth:`ContainerTransport.path_map`).
    """
    if isinstance(models_dirs, (str, Path)):
        models_dirs = [str(models_dirs)]
    hf_root = Path(hf_cache).resolve() if hf_cache else None
    roots = [
        (Path(d).resolve(), "/models" if i == 0 else f"/models{i + 1}")
        for i, d in enumerate(models_dirs) if d
    ]
    refs: list[str] = []
    mounts: list[str] = []
    parent_targets: dict[Path, str] = {}
    for p in sorted(paths, key=str):
        rp = p.resolve()
        mapped = False
        if hf_root is not None:
            try:
                rel = rp.relative_to(hf_root)
            except ValueError:
                pass
            else:
                refs.append(f"{CONTAINER_HF_HOME}/{rel}")
                mapped = True
        if not mapped:
            for root, target in roots:
                try:
                    rel = rp.relative_to(root)
                except ValueError:
                    continue
                refs.append(f"{target}/{rel}")
                mapped = True
                break
        if mapped:
            continue
        parent = rp.parent
        if parent not in parent_targets:
            idx = len(parent_targets)
            parent_targets[parent] = f"/ext{idx}"
            mounts.append(f"-v {parent}:{parent_targets[parent]}")
        refs.append(f"{parent_targets[parent]}/{rp.name}")
    return refs, mounts
