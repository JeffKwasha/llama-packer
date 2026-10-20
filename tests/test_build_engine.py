# tests/test_build_engine.py
"""Tests for extras/build_engine.py — compiling a serving engine from source.

A source build has three ways to go wrong that a download cannot: it can
target the wrong GPU architecture (the binary compiles and sees nothing), it
can produce a tree without the binary in it, and it can fail halfway through
and leave the live symlink pointing at the broken result. Every test here is
about one of those three.

The build itself is never run: ``cmake`` and a compiler are not the thing under
test, and a test that compiles would be slow and would not fail for the
reasons these tests care about. What is exercised is everything around it —
naming, refusal, and promotion — with the compiler stubbed.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

EXTRAS = Path(__file__).resolve().parents[1] / "extras"
sys.path.insert(0, str(EXTRAS))

_spec = importlib.util.spec_from_file_location("build_engine",
                                               EXTRAS / "build_engine.py")
bld = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bld
_spec.loader.exec_module(bld)


# --- helpers ---------------------------------------------------------------


def make_binary(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{text}\n")
    path.chmod(0o755)
    return path


def fake_build_tree(root: Path, *, devices: str = "HIP0: AMD Radeon (gfx1201)",
                    binary: str = "audiocpp_server") -> Path:
    """A source tree with a ``build/bin`` that looks like a finished build."""
    tree = root / "src" / "audio.cpp-0.9.2"
    binroot = tree / "build" / "bin"
    binroot.mkdir(parents=True, exist_ok=True)
    make_binary(binroot / binary, f'if [ "$1" = "--list-devices" ]; then '
                                  f'echo "{devices}"; fi')
    (binroot / "libggml-hip.so").write_bytes(b"\x7fELF fake")
    return tree


def stub_download(monkeypatch, files: dict[str, bytes]):
    """Serve the codeload tarball without touching the network."""
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o755
            tf.addfile(info, io.BytesIO(payload))
    blob = buf.getvalue()
    monkeypatch.setattr(bld.upd, "download",
                        lambda url, dest, **kw: dest.write_bytes(blob))
    return blob


# --- naming ----------------------------------------------------------------


def test_a_built_engine_is_named_apart_from_a_downloaded_one():
    """Same version, different artifact; conflating them breaks rollback.

    ``audiocpp_server-v0.9.2`` is what update.py installs from a release
    archive and ``audiocpp_server-src-v0.9.2-hip-gfx1201`` is what this
    builds. They are different binaries of the same version — one is a
    portable Vulkan build that crashes on this host, the other is a local
    ROCm build — and a directory listing has to be able to say which is which.
    """
    name = bld.versioned_name(bld.AUDIO, "v0.9.2", "hip", "gfx1201")
    assert name == "audiocpp_server-src-v0.9.2-hip-gfx1201"
    assert name != bld.upd.AUDIO_CPP.versioned.format(tag="v0.9.2")


def test_the_arch_is_in_the_name_only_when_there_is_one():
    assert bld.versioned_name(bld.AUDIO, "v0.9.2", "cpu") \
        == "audiocpp_server-src-v0.9.2-cpu"
    assert "gfx1201+gfx1100" in bld.versioned_name(bld.AUDIO, "v0.9.2",
                                                   "hip", "gfx1201;gfx1100")


# --- the plan --------------------------------------------------------------


def test_the_plan_pins_the_arch_and_asks_for_a_deployment_build(tmp_path):
    argv = bld.build_argv(bld.AUDIO, "hip", "gfx1201", 8,
                          Path("/src"), Path("/src/build"))
    joined = " ".join(argv)
    # gfx1201 is the entire point: stock ROCm packages carry no offload
    # binary for it, so the target has to be named rather than detected.
    assert "--gpu-targets gfx1201" in joined
    # A deployment build embeds the model specs and runs outside the checkout,
    # which is what makes the tree movable into tools/ afterwards.
    assert "--deployment-build" in joined
    assert "--backend hip" in joined
    assert "--jobs 8" in joined


def test_the_build_runs_the_upstream_script_not_a_reimplementation(tmp_path):
    """Delegating keeps upstream's ROCm-clang and sticky-cache handling."""
    argv = bld.build_argv(bld.AUDIO, "vulkan", "", 0, Path("/src"),
                          Path("/src/build"))
    assert argv[0].endswith("scripts/build_linux.sh")
    assert "--gpu-targets" not in " ".join(argv)


def test_source_url_is_the_codeload_endpoint():
    url = bld.source_url(bld.AUDIO, "v0.9.2")
    assert url == ("https://codeload.github.com/0xShug0/audio.cpp/"
                   "tar.gz/refs/tags/v0.9.2")
    # codeload, not the archive/ redirect: a redirect that lands on an HTML
    # login page would extract as a source tree with nothing in it.
    assert "github.com" in url and "codeload" in url


# --- GPU detection ---------------------------------------------------------


def test_targets_come_from_amdgpu_arch_when_present(tmp_path, monkeypatch):
    rocm = tmp_path / "rocm"
    (rocm / "bin").mkdir(parents=True)
    make_binary(rocm / "bin" / "amdgpu-arch", "echo gfx1201")
    monkeypatch.setenv("ROCM_PATH", str(rocm))
    assert bld.detect_gpu_targets() == "gfx1201"


def test_rocminfo_is_the_fallback_and_only_full_lines_count(tmp_path,
                                                             monkeypatch):
    rocm = tmp_path / "rocm"
    (rocm / "bin").mkdir(parents=True)
    make_binary(rocm / "bin" / "rocminfo",
                "cat <<'EOF'\n"
                "  Name:                    AMD Ryzen 7 5700X3D\n"
                "  Name:                    gfx1201\n"
                "  ISA Info:\n"
                "    gfx11 fragment in prose that must not be matched\n"
                "EOF")
    monkeypatch.setenv("ROCM_PATH", str(rocm))
    assert bld.detect_gpu_targets() == "gfx1201"


def test_no_device_means_no_target_rather_than_a_guess(tmp_path, monkeypatch):
    rocm = tmp_path / "rocm"
    rocm.mkdir()
    monkeypatch.setenv("ROCM_PATH", str(rocm))
    monkeypatch.setattr(bld.shutil, "which", lambda name: None)
    assert bld.detect_gpu_targets() == ""


# --- the smoke test that makes a build trustworthy --------------------------


def test_probe_reads_the_backends_a_binary_enumerates(tmp_path):
    binary = make_binary(tmp_path / "srv",
                         'echo "Available devices:"; echo "  HIP0: AMD Radeon"')
    assert bld.probe_backends(binary) == {"hip"}


def test_probe_sees_nothing_when_the_engine_finds_no_device(tmp_path):
    binary = make_binary(tmp_path / "srv", 'echo "Available devices:"')
    assert bld.probe_backends(binary) == set()


def test_a_build_for_the_wrong_arch_is_refused(tmp_path):
    """The gfx1201 failure this script exists for.

    A binary compiled without an offload binary for the target arch starts,
    answers /health, and then reports that it has no device — in every
    service request. Promoting that to the live symlink would trade a working
    engine for a broken one on the strength of a successful compile.
    """
    binary = make_binary(tmp_path / "srv", 'echo "  CPU0: CPU"')
    with pytest.raises(bld.BuildError, match="no HIP device"):
        bld.check_backends(binary, "hip", {"cpu"})


def test_a_cpu_build_is_exempt_from_the_device_check(tmp_path):
    """The cpu backend is the portable one, not a mistake."""
    binary = make_binary(tmp_path / "srv", "true")
    bld.check_backends(binary, "cpu", {"cpu"})     # must not raise


def test_vulkan_and_hip_are_not_interchangeable(tmp_path):
    binary = make_binary(tmp_path / "srv", "true")
    with pytest.raises(bld.BuildError, match="no VULKAN device"):
        bld.check_backends(binary, "vulkan", {"hip"})
    bld.check_backends(binary, "hip", {"hip"})


# --- collecting the artifact ------------------------------------------------


def test_a_build_tree_without_the_binary_is_an_error(tmp_path):
    (tmp_path / "build" / "bin").mkdir(parents=True)
    with pytest.raises(bld.BuildError, match="no audiocpp_server"):
        bld.collect_artifact(tmp_path / "build", bld.AUDIO)


def test_the_whole_tree_is_collected_because_libs_must_travel(tmp_path):
    tree = fake_build_tree(tmp_path)
    artifact = bld.collect_artifact(tree / "build", bld.AUDIO)
    assert (artifact / "audiocpp_server").is_file()
    assert (artifact / "libggml-hip.so").is_file()


# --- the install path ------------------------------------------------------


def _stub_everything(monkeypatch, tmp_path, *, devices="HIP0: AMD Radeon",
                     build_rc=0):
    stub_download(monkeypatch, {"audio.cpp-0.9.2/README.md": b"hi\n"})
    monkeypatch.setattr(bld, "fetch_source",
                        lambda engine, ref, dest: fake_build_tree(
                            tmp_path, devices=devices))
    monkeypatch.setattr(bld, "run_build",
                        lambda argv, log, cwd: (log.write_text("log\n"),
                                                build_rc)[1])
    monkeypatch.setattr(bld, "probe_backends", lambda binary: {"hip"})


def test_a_good_build_installs_beside_the_prebuilt_one_and_moves_the_link(
        tmp_path, monkeypatch):
    _stub_everything(monkeypatch, tmp_path)
    install = tmp_path / "tools"
    install.mkdir()

    final = bld.install_built(bld.AUDIO, install, "v0.9.2", "hip", "gfx1201", 8)

    assert final.name == "audiocpp_server-src-v0.9.2-hip-gfx1201"
    assert (final / "audiocpp_server").is_file()
    assert (final / "libggml-hip.so").is_file(), "libraries must travel"
    link = install / "audiocpp_server"
    assert os.readlink(link) == f"./{final.name}/audiocpp_server"
    # and the build log is kept next to it, not thrown away with the staging
    assert (install / f"{final.name}.build.log").is_file()


def test_a_failed_build_leaves_the_live_symlink_alone(tmp_path, monkeypatch):
    _stub_everything(monkeypatch, tmp_path, build_rc=1)
    install = tmp_path / "tools"
    install.mkdir()
    make_binary(install / "audiocpp_server", "the engine that works")

    with pytest.raises(bld.BuildError, match="build failed"):
        bld.install_built(bld.AUDIO, install, "v0.9.2", "hip", "gfx1201", 8)

    assert not (install / "audiocpp_server").is_symlink(), \
        "the live engine must survive a failed build"
    assert (install / "audiocpp_server").read_text().endswith(
        "the engine that works\n")
    assert (install / "audiocpp_server-src-v0.9.2-hip-gfx1201.build.log").is_file()


def test_a_build_that_cannot_see_the_card_is_never_installed(tmp_path,
                                                            monkeypatch):
    """Same guarantee, at the device level rather than the exit-code level."""
    _stub_everything(monkeypatch, tmp_path, devices="CPU0: CPU")
    monkeypatch.setattr(bld, "probe_backends", lambda binary: {"cpu"})
    install = tmp_path / "tools"
    install.mkdir()
    make_binary(install / "audiocpp_server", "the engine that works")

    with pytest.raises(bld.BuildError, match="no HIP device"):
        bld.install_built(bld.AUDIO, install, "v0.9.2", "hip", "gfx1201", 8)

    assert not (install / "audiocpp_server").is_symlink()
    # the build log is kept (someone has to read it); no build tree is left
    assert not [d for d in install.glob("audiocpp_server-src-*") if d.is_dir()]


def test_an_existing_build_is_not_replaced_without_asking(tmp_path,
                                                          monkeypatch):
    _stub_everything(monkeypatch, tmp_path)
    install = tmp_path / "tools"
    install.mkdir()
    name = "audiocpp_server-src-v0.9.2-hip-gfx1201"
    (install / name).mkdir()
    make_binary(install / name / "audiocpp_server", "older build")

    monkeypatch.setattr(bld, "run_build",
                        lambda *a, **k: pytest.fail("must not rebuild"))
    bld.install_built(bld.AUDIO, install, "v0.9.2", "hip", "gfx1201", 8)

    assert "older build" in (install / name / "audiocpp_server").read_text()


def test_rebuild_replaces_the_tree(tmp_path, monkeypatch):
    _stub_everything(monkeypatch, tmp_path)
    install = tmp_path / "tools"
    install.mkdir()
    name = "audiocpp_server-src-v0.9.2-hip-gfx1201"
    (install / name).mkdir()
    make_binary(install / name / "audiocpp_server", "older build")

    bld.install_built(bld.AUDIO, install, "v0.9.2", "hip", "gfx1201", 8,
                      rebuild=True)

    assert "older build" not in (install / name / "audiocpp_server").read_text()


def test_a_real_file_where_the_symlink_belongs_is_not_overwritten(tmp_path,
                                                                  monkeypatch):
    _stub_everything(monkeypatch, tmp_path)
    install = tmp_path / "tools"
    install.mkdir()
    make_binary(install / "audiocpp_server", "someone put a real file here")

    with pytest.raises(bld.BuildError, match="real file"):
        bld.install_built(bld.AUDIO, install, "v0.9.2", "hip", "gfx1201", 8)

    assert "someone put a real file here" in (
        install / "audiocpp_server").read_text()


# --- toolchain gate and the CLI -------------------------------------------


def test_the_toolchain_check_accepts_a_rocm_that_keeps_clang_under_llvm(
        tmp_path, monkeypatch):
    """This box's layout: no $ROCM_HOME/bin/clang++, clang under llvm/.

    A check that only looked for bin/clang++ would refuse a build that works.
    """
    rocm = tmp_path / "rocm"
    (rocm / "llvm" / "bin").mkdir(parents=True)
    make_binary(rocm / "llvm" / "bin" / "clang++", "true")
    monkeypatch.setenv("ROCM_PATH", str(rocm))
    assert bld.rocm_clangxx() == rocm / "llvm" / "bin" / "clang++"
    assert bld.toolchain_problems(bld.AUDIO, "hip") == []


def test_hipconfig_is_consulted_before_giving_up(tmp_path, monkeypatch):
    """Upstream's own resolution order, in the same order."""
    rocm = tmp_path / "rocm"
    libdir = rocm / "lib" / "llvm" / "bin"
    libdir.mkdir(parents=True)
    make_binary(libdir / "clang++", "true")
    (rocm / "bin").mkdir()
    make_binary(rocm / "bin" / "hipconfig", f'echo "{libdir}"')
    monkeypatch.setenv("ROCM_PATH", str(rocm))
    assert bld.rocm_clangxx() == libdir / "clang++"


def test_a_missing_toolchain_is_reported_before_anything_is_fetched(tmp_path,
                                                                     monkeypatch):
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "nowhere"))
    problems = bld.toolchain_problems(bld.AUDIO, "hip")
    assert any("no ROCm" in p for p in problems)


def test_dry_run_prints_the_plan_and_downloads_nothing(tmp_path, monkeypatch,
                                                       capsys):
    monkeypatch.chdir(tmp_path)
    # A stub ROCm home — the plan must be a function of the arguments, not of
    # the host's rocm install (CI has none, and /opt/rocm on this box leaks
    # real clang in otherwise).
    rocm = tmp_path / "rocm"
    (rocm / "bin").mkdir(parents=True)
    (rocm / "bin" / "clang++").write_text("x")
    monkeypatch.setenv("ROCM_PATH", str(rocm))
    monkeypatch.setattr(bld.upd, "download",
                        lambda *a, **k: pytest.fail("must not fetch"))
    monkeypatch.setattr(bld.upd, "load_engines_config", lambda: (None, {}))

    rc = bld.main(["audio", "--ref", "v0.9.2", "--backend", "hip",
                   "--gpu-targets", "gfx1201", "--dir", str(tmp_path / "t"),
                   "--dry-run"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "gfx1201" in out
    assert "build_linux.sh" in out, "the plan must name the real command"
    assert "audiocpp_server-src-v0.9.2-hip-gfx1201" in out
    assert not (tmp_path / "t").exists(), "--dry-run creates no directories"


def test_a_build_that_cannot_succeed_does_not_start(tmp_path, monkeypatch):
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "nowhere"))
    monkeypatch.setattr(bld.upd, "load_engines_config", lambda: (None, {}))
    monkeypatch.setattr(bld, "install_built",
                        lambda *a, **k: pytest.fail("must not build"))

    assert bld.main(["audio", "--ref", "v0.9.2", "--backend", "hip",
                     "--gpu-targets", "gfx1201",
                     "--dir", str(tmp_path / "t")]) == 1

# --- patching the source ---------------------------------------------------
#
# A patched tree is not upstream's tree. The rule these tests enforce is that
# the edits are never invisible: announced, logged, and recorded beside the
# installed engine.


def test_the_clang_diagnostic_patch_is_needed_and_named(tmp_path):
    """The build cannot compile without it, and that is a recorded fact.

    Upstream builds app/server/parallel_runtime.cpp as C++17 with
    -pedantic-errors while the file captures structured bindings in a lambda.
    ROCm's clang++ makes that an error; GCC does not. So a HIP build of v0.9.2
    fails without this patch.
    """
    cmake = tmp_path / "CMakeLists.txt"
    cmake.write_text("project(x)\nset(CMAKE_CXX_STANDARD 17)\n")
    applied = bld.apply_patches(tmp_path)
    assert "-Wno-c++20-extensions" in applied[0]
    patched = cmake.read_text()
    # the standard itself must NOT move: the tree relies on pre-C++20 u8
    # literal semantics, and bumping it trades one error for seven
    assert "set(CMAKE_CXX_STANDARD 17)" in patched
    assert "add_compile_options(" in patched
    assert "Clang" in patched, "clang-only: GCC must be unaffected"
    # the patch must say why, in the record that ships with the engine
    assert "structured bindings" in applied[0]
    assert "u8" in applied[0], "the record must explain why not C++20"


def test_a_patch_that_no_longer_matches_is_reported_not_fatal(tmp_path):
    """Upstream fixing their own tree should make this a no-op, not a stop."""
    (tmp_path / "CMakeLists.txt").write_text("project(x)\n")
    applied = bld.apply_patches(tmp_path)
    assert any("SKIPPED" in line and "no longer matches" in line
               for line in applied)


def test_a_missing_file_is_reported_not_fatal(tmp_path):
    applied = bld.apply_patches(tmp_path)
    assert any("SKIPPED" in line and "is gone" in line for line in applied)


def test_the_edits_are_recorded_next_to_the_installed_engine(tmp_path,
                                                            monkeypatch, capsys):
    _stub_everything(monkeypatch, tmp_path)
    install = tmp_path / "tools"
    install.mkdir()

    # A source tree that both contains the flag to patch and (from the stub)
    # a finished build tree, so the whole install path runs for real.
    real_stub = bld.fetch_source

    def source_with_flag(engine, ref, dest):
        tree = fake_build_tree(tmp_path)
        (tree / "CMakeLists.txt").write_text("set(CMAKE_CXX_STANDARD 17)\n")
        return tree

    monkeypatch.setattr(bld, "fetch_source", source_with_flag)
    final = bld.install_built(bld.AUDIO, install, "v0.9.2", "hip", "gfx1201", 8)

    record = (install / f"{final.name}.patched").read_text()
    assert "-Wno-c++20-extensions" in record
    assert "structured bindings" in record
    assert "Source patches applied: 1/1" in capsys.readouterr().out


def test_a_built_engine_says_it_was_patched_in_its_name(tmp_path):
    """-src- is the flag; the .patched file is the detail."""
    name = bld.versioned_name(bld.AUDIO, "v0.9.2", "hip", "gfx1201")
    assert name.startswith("audiocpp_server-src-")


# --- the device-probe spellings --------------------------------------------


def test_both_device_line_spellings_are_recognised(tmp_path):
    """The check once reported a working HIP build as seeing no device.

    audio.cpp names a HIP device ``ROCm:0 "AMD Radeon AI PRO R9700" [GPU]`` —
    colon *before* the index — while llama-server prints ``Vulkan0: name``.
    Accepting only the second form is how a good build gets refused.
    """
    rocm = make_binary(
        tmp_path / "srv",
        'cat <<\'EOF\'\n'
        'ggml_cuda_init: found 1 ROCm devices (Total VRAM: 32624 MiB):\n'
        '  Device 0: AMD Radeon AI PRO R9700, gfx1201 (0x1201), VMM: no\n'
        'available_devices=2\n'
        'ROCm:0 "AMD Radeon AI PRO R9700" [GPU]\n'
        'CPU:0 "AMD Ryzen 7 5700X3D" [CPU]\n'
        'EOF')
    found = bld.probe_backends(rocm)
    assert "hip" in found, "ROCm is how audio.cpp spells HIP"
    bld.check_backends(rocm, "hip", found)


def test_the_llama_server_spelling_still_parses(tmp_path):
    vulkan = make_binary(
        tmp_path / "srv",
        'echo "Available devices:"; echo '
        '"  Vulkan0: AMD Radeon AI Pro R9700 (RADV GFX1201) (32624 MiB)"')
    found = bld.probe_backends(vulkan)
    assert found == {"vulkan"}
    bld.check_backends(vulkan, "vulkan", found)


def test_an_unknown_output_format_is_an_error_not_an_empty_result(tmp_path):
    """A count with no parseable line means the format moved; say so."""
    binary = make_binary(tmp_path / "srv", 'echo "available_devices=2"; '
                                            'echo "device[0] yes"')
    with pytest.raises(bld.BuildError, match="recognises"):
        bld.probe_backends(binary)


def test_installing_an_existing_build_tree_does_not_recompile(tmp_path,
                                                              monkeypatch):
    """The compile is twenty minutes; re-checking it should not cost that."""
    _stub_everything(monkeypatch, tmp_path)
    monkeypatch.setattr(bld, "fetch_source",
                        lambda *a: pytest.fail("must not fetch"))
    monkeypatch.setattr(bld, "run_build",
                        lambda *a, **k: pytest.fail("must not compile"))
    install = tmp_path / "tools"
    install.mkdir()
    built = fake_build_tree(tmp_path)
    artifact = bld.collect_artifact(built / "build", bld.AUDIO)

    final = bld.install_built(bld.AUDIO, install, "v0.9.2", "hip", "gfx1201", 0,
                             from_build=artifact)

    assert (final / "audiocpp_server").is_file()
    assert (final / "libggml-hip.so").is_file()
    assert "prebuilt tree" in (install / f"{final.name}.patched").read_text()


def test_from_build_still_refuses_a_cardless_binary(tmp_path, monkeypatch):
    _stub_everything(monkeypatch, tmp_path, devices="CPU0: CPU")
    monkeypatch.setattr(bld, "probe_backends", lambda binary: {"cpu"})
    install = tmp_path / "tools"
    install.mkdir()
    built = fake_build_tree(tmp_path)
    artifact = bld.collect_artifact(built / "build", bld.AUDIO)
    with pytest.raises(bld.BuildError, match="no HIP"):
        bld.install_built(bld.AUDIO, install, "v0.9.2", "hip", "gfx1201", 0,
                          from_build=artifact)
    assert not (install / "audiocpp_server").is_symlink()
