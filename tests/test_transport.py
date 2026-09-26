# tests/test_transport.py
"""Transport seam: pair registry, naming, gating, path mapping, wrapping.

The vLLM container tests exercise these paths end-to-end; these unit tests
pin the seam itself so a new engine/transport cannot silently break the
engine x transport model.
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
    assert BACKENDS["vllm"].transport.name == "host"
    assert BACKENDS["vllm-podman"].engine.name == "vllm"
    assert BACKENDS["vllm-docker"].engine.name == "vllm"
    # llama.cpp is host-only: no container pairs are registered.
    assert "llama-server-podman" not in BACKENDS
    assert "llama-server-docker" not in BACKENDS
    assert BACKENDS["llama-server"].transport.container is False
    assert BACKENDS["vllm-podman"].transport.container is True


def test_transport_preference_is_host_then_podman_then_docker():
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
    assert container_device_flags(vendor, runtime) == expected


def test_host_transport_is_identity():
    t = HostTransport()
    assert t.bind_name("vllm") == "vllm"
    p = Path("/models/a.gguf")
    pm = t.path_map({}, [p])
    assert pm.ref(p) == str(p)
    assert pm.mounts == []
    assert t.wrap("vllm serve", Dummy(), Launch(), pm) == "vllm serve"
    assert t.is_available({}) is True


def test_bind_name_suffixes_non_host():
    assert ContainerTransport("podman").bind_name("vllm") == "vllm-podman"
    assert ContainerTransport("docker").bind_name("vllm") == "vllm-docker"


def test_container_runtime_gating():
    t = ContainerTransport("podman")
    assert t.is_available({}) is True            # not probed -> do not gate
    assert t.is_available({"podman": True}) is True
    assert t.is_available({"podman": False}) is False
    assert ContainerTransport("docker").is_available({"docker": False}) is False


def test_path_map_prefers_hf_root_over_models_dir(tmp_path):
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
    assert PathMap({}).ref(Path("/x/y")) == "/x/y"


def test_container_wrap_shape():
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
    from llama_packer.backends import validate_backend_names

    assert validate_backend_names([]) is None
    assert validate_backend_names(
        ["llama-server", "vllm", "vllm-podman", "vllm-docker",
         "sd-server", "whisper-server", "audio-cpp"]) is None


def test_validate_backend_names_reports_unknown_name():
    from llama_packer.backends import validate_backend_names

    err = validate_backend_names(["llama-server", "bogus"])
    assert err is not None
    assert "bogus" in err


def test_validate_backend_names_rejects_removed_kokoro():
    from llama_packer.backends import validate_backend_names

    err = validate_backend_names(["kokoro-podman"])
    assert err is not None and "kokoro-podman" in err

