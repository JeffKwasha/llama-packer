# tests/test_transport.py
"""Transport seam: pair registry, naming, gating, path mapping, wrapping.

The vLLM container tests exercise these paths end-to-end; these unit tests
pin the seam itself so a new engine/transport cannot silently break the
engine x transport model.

Design parts: docs/transports/host.md — the engine x transport seam: pair registry
and naming, runtime gating, path mapping, and the container command shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llama_packer.backends import BACKENDS
from llama_packer.backends.transport import (
    CONTAINER_HF_HOME,
    TRANSPORT_PREFERENCE,
    ContainerTransport,
    HostTransport,
    Launch,
    PathMap,
    container_device_flags,
)


class Dummy:
    """Stand-in for a Model — the transports do not touch it today."""


def test_registry_pairs_are_engine_by_transport():
    # host keeps the bare engine name; every other transport suffixes it.
    # Validates: SPEC.md, 'Backend Selection': one backend per valid (engine, transport) pair; the host transport keeps the bare engine name and llama.cpp is host-only.
    assert BACKENDS["vllm"].transport.name == "host"
    assert BACKENDS["vllm-podman"].engine.name == "vllm"
    assert BACKENDS["vllm-docker"].engine.name == "vllm"
    # llama.cpp is host-only: no container pairs are registered.
    assert "llama-server-podman" not in BACKENDS
    assert "llama-server-docker" not in BACKENDS
    assert BACKENDS["llama-server"].transport.container is False
    assert BACKENDS["vllm-podman"].transport.container is True


def test_transport_preference_is_host_then_podman_then_docker():
    # Validates: docs/transports/podman.md, 'Transport: podman': host first, then podman, then docker.
    assert TRANSPORT_PREFERENCE == ("host", "podman", "docker")
    names = [n for n in BACKENDS if BACKENDS[n].engine.name == "vllm"]
    assert names == ["vllm", "vllm-podman", "vllm-docker"]


@pytest.mark.parametrize("vendor,runtime,expected", [
    ("nvidia", "docker", "--runtime=nvidia --gpus all"),
    ("nvidia", "podman", "--device nvidia.com/gpu=all"),
    ("amd", "docker",
     "--device /dev/kfd --device /dev/dri --group-add video --group-add render"),
    ("amd", "podman",
     "--device /dev/kfd --device /dev/dri --group-add video --group-add render"),
    ("cpu", "docker", ""),
    ("cpu", "podman", ""),
])
def test_container_device_flags(vendor, runtime, expected):
    # Validates: docs/transports/podman.md, 'Transport: podman': the GPU passthrough flags each vendor and runtime needs; cpu needs none.
    assert container_device_flags(vendor, runtime) == expected


def test_host_transport_is_identity():
    # Validates: docs/transports/host.md, 'Transport: host': the host transport maps nothing and wraps nothing.
    t = HostTransport()
    assert t.bind_name("vllm") == "vllm"
    p = Path("/models/a.gguf")
    pm = t.path_map({}, [p])
    assert pm.ref(p) == str(p)
    assert pm.mounts == []
    assert t.wrap("vllm serve", Dummy(), Launch(), pm) == "vllm serve"
    assert t.is_available({}) is True


def test_bind_name_suffixes_non_host():
    # Validates: docs/transports/podman.md, 'Transport: podman': a non-host transport suffixes the engine name.
    assert ContainerTransport("podman").bind_name("vllm") == "vllm-podman"
    assert ContainerTransport("docker").bind_name("vllm") == "vllm-docker"


def test_container_runtime_gating():
    # Validates: docs/transports/podman.md, 'Transport: podman': a container pair is only inferred when its runtime is on PATH
    t = ContainerTransport("podman")
    assert t.is_available({}) is True            # not probed -> do not gate
    assert t.is_available({"podman": True}) is True
    assert t.is_available({"podman": False}) is False
    assert ContainerTransport("docker").is_available({"docker": False}) is False


def test_path_map_prefers_hf_root_over_models_dir(tmp_path):
    # Validates: docs/transports/host.md, 'Runtime config files': a file under the HF cache root maps into the container's HF home, not into /models.
    hf = tmp_path / "hf"
    snap = hf / "hub" / "models--org--m" / "snapshots" / "s1" / "m.safetensors"
    snap.parent.mkdir(parents=True)
    snap.write_bytes(b"x")
    models = tmp_path / "models"
    models.mkdir()
    pm = ContainerTransport("docker").path_map(
        {"models_dirs": [str(models)], "hf_cache": str(hf)}, [snap])
    assert pm.ref(snap) == (
        f"{CONTAINER_HF_HOME}/hub/models--org--m/snapshots/s1/m.safetensors")
    assert f"-v {hf}:{CONTAINER_HF_HOME}" in pm.mounts
    assert f"-v {models}:/models" in pm.mounts


def test_path_map_models_dir_and_outside(tmp_path):
    # Validates: docs/transports/host.md, 'Runtime config files': a file under a models dir maps to /models, anything else to its own /ext mount.
    models = tmp_path / "models"
    inside = models / "chat" / "m.gguf"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(b"x")
    outside = tmp_path / "other" / "draft.safetensors"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"x")
    pm = ContainerTransport("docker").path_map(
        {"models_dirs": [str(models)]}, [inside, outside])
    assert pm.ref(inside) == "/models/chat/m.gguf"
    assert pm.ref(outside) == "/ext0/draft.safetensors"
    assert f"-v {outside.resolve().parent}:/ext0" in pm.mounts


def test_path_map_unmapped_falls_back_to_str():
    # Validates: SPEC.md, 'Container (vllm-podman / vllm-docker)': a path that no mapping covers passes through unchanged
    assert PathMap({}).ref(Path("/x/y")) == "/x/y"


def test_container_wrap_shape():
    # Validates: docs/transports/docker.md, 'Transport: docker': the container command carries the init flag, the devices, the mounts, the env, the port and the image, in that order.
    pm = PathMap({}, mounts=["-v /m:/models"])
    launch = Launch(
        image="img:1", port=8000, args="--device x",
        env=(f"HF_HOME={CONTAINER_HF_HOME}", "HF_HUB_OFFLINE=1"))
    cmd = ContainerTransport("podman").wrap("vllm serve --port 8000", Dummy(),
                                            launch, pm)
    assert cmd.startswith("podman run --init --rm --device x --name ${MODEL_ID}")
    assert "-v /m:/models" in cmd
    assert f"-e HF_HOME={CONTAINER_HF_HOME}" in cmd
    assert "-e HF_HUB_OFFLINE=1" in cmd
    assert "-p ${PORT}:8000" in cmd
    assert cmd.endswith(" img:1 vllm serve --port 8000")


# ── backend-name validation contract ──────────────────────────────────────

def test_validate_backend_names_accepts_registered_pairs():
    # Validates: SPEC.md, 'Backend Selection': every registered engine/transport pair is an accepted name.
    from llama_packer.backends import validate_backend_names

    assert validate_backend_names([]) is None
    assert validate_backend_names(
        ["llama-server", "vllm", "vllm-podman", "vllm-docker",
         "sd-server", "whisper-server", "audio-cpp"]) is None


def test_validate_backend_names_reports_unknown_name():
    # Validates: SPEC.md, 'Backend Selection': an unknown backend name is reported, not ignored
    from llama_packer.backends import validate_backend_names

    err = validate_backend_names(["llama-server", "bogus"])
    assert err is not None
    assert "bogus" in err


def test_validate_backend_names_rejects_removed_kokoro():
    # Validates: SPEC.md, 'Backend Selection': a removed engine stays rejected by name.
    from llama_packer.backends import validate_backend_names

    err = validate_backend_names(["kokoro-podman"])
    assert err is not None and "kokoro-podman" in err

