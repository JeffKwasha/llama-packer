"""Tests for extras/install.py — the systemd deployer.

Everything here is pure: the discovery helpers run against tmp_path fakes,
the unit is rendered into a string, and the two commands that touch the
system (``systemd-analyze verify``, ``systemctl``) are monkeypatched.  No
file under /opt or /etc is created and no service is started.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

EXTRAS = Path(__file__).resolve().parents[1] / "extras"

_spec = importlib.util.spec_from_file_location("install", EXTRAS / "install.py")
ins = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ins  # dataclass resolves annotations via sys.modules
_spec.loader.exec_module(ins)


def _args(**kw):
    """An argparse.Namespace shaped like main()'s, with the defaults the
    parser would supply."""
    base = dict(user="svc", group="grp", dir=Path("/opt/llama"),
                listen="127.0.0.1:8123", memory_high="32G", memory_max="40G",
                swap_max="1G", spare=None, vram=None, profiles=Path("p.yaml"),
                hf_home=None, agents=False, create_stubs=False,
                dry_run=False)
    base.update(kw)
    return type("Args", (), base)


# --- discovery -------------------------------------------------------------


def test_gpu_nodes_skips_drm_connectors(tmp_path):
    # /dev/dri holds connector symlinks (card1-DP-1, card1-HDMI-A-1) beside
    # the real nodes; a bare glob would hand systemd a path that is not a
    # device handle.
    # Validates: docs/install.md, 'Two things the unit discovers rather than hardcodes': GPU device nodes this host actually has (render + card + kfd). Roots are parameters so the scan is testable against a fake ``/dev``.
    dri = tmp_path / "dri"
    dri.mkdir()
    for name in ("renderD128", "card1", "card1-DP-1", "card1-HDMI-A-1"):
        (dri / name).touch()
    kfd = tmp_path / "kfd"
    kfd.touch()
    nodes = ins.gpu_nodes(dri, kfd)
    assert f"{dri}/renderD128" in nodes
    assert f"{dri}/card1" in nodes
    assert not any("DP-1" in n for n in nodes), nodes
    assert str(kfd) in nodes


def test_gpu_nodes_omits_absent_paths(tmp_path):
    # Only nodes that exist may reach DeviceAllow: systemd refuses to start
    # on one that does not resolve.
    # Validates: docs/install.md, 'Two things the unit discovers rather than hardcodes': GPU device nodes this host actually has (render + card + kfd). Roots are parameters so the scan is testable against a fake ``/dev``.
    dri = tmp_path / "dri"
    dri.mkdir()
    (dri / "renderD128").touch()
    nodes = ins.gpu_nodes(dri, tmp_path / "absent-kfd")
    assert nodes == [f"{dri}/renderD128"]


def test_newest_build_orders_numerically(tmp_path):
    # llama-b9999 must lose to llama-b11658; lexicographic order gets this
    # backwards, and picking the wrong build means deploying a stale binary.
    # Validates: docs/install.md, 'Where an update installs': Newest installed ``llama-bNNNN`` with a llama-server.
    for name in ("llama-b9999", "llama-b11658", "llama-b11539"):
        d = tmp_path / name
        d.mkdir()
        (d / "llama-server").write_text("x")
    (tmp_path / "llama-b12000").mkdir()      # no llama-server: not a build
    assert ins.newest_build(tmp_path).name == "llama-b11658"


def test_newest_build_none_without_builds(tmp_path):
    # Validates: docs/install.md, 'Where an update installs': Newest installed ``llama-bNNNN`` with a llama-server.
    (tmp_path / "llama-b1").mkdir()
    assert ins.newest_build(tmp_path) is None


# --- unit rendering --------------------------------------------------------


def test_render_unit_substitutes_directives_only(monkeypatch):
    # The template header names the placeholders it expects. Substituting
    # those mentions would turn the documentation into a copy of this host.
    # Validates: docs/install.md, 'Two things the unit discovers rather than hardcodes': Fill the unit template for this host. The hardening block comes from the template verbatim; what is.
    monkeypatch.setattr(ins, "gpu_vendor", lambda: "amd")
    monkeypatch.setattr(ins, "gpu_nodes", lambda: ["/dev/dri/renderD128"])
    body = ins.render_unit(_args(), Path("/opt/llama"))
    assert "User=svc" in body and "Group=grp" in body
    assert "WorkingDirectory=/opt/llama" in body
    # No binaries exist under /opt/llama in the test, so the unit names where
    # install.py puts them rather than a path it found.
    assert ("ExecStart=/opt/llama/tools/llama-swap --listen 127.0.0.1:8123"
            in body)
    # No placeholder survives on a directive. (Comments keep their mentions
    # on purpose: the header documents the tokens it expects.)
    directives = [ln for ln in body.splitlines() if not ln.lstrip().startswith("#")]
    assert not [ln for ln in directives if ins._TOKEN_RE.search(ln)]
    assert "@PLACEHOLDER@" in body        # header prose intact


def test_render_unit_gpu_env_follows_hardware(monkeypatch):
    # Validates: docs/install.md, 'Two things the unit discovers rather than hardcodes': Fill the unit template for this host. The hardening block comes from the template verbatim; what is.
    monkeypatch.setattr(ins, "gpu_nodes", lambda: ["/dev/kfd"])
    monkeypatch.setattr(ins, "gpu_vendor", lambda: "amd")
    assert "HIP_VISIBLE_DEVICES=0" in ins.render_unit(_args(), Path("/d"))
    monkeypatch.setattr(ins, "gpu_vendor", lambda: "nvidia")
    body = ins.render_unit(_args(), Path("/d"))
    assert "CUDA_VISIBLE_DEVICES=0" in body
    assert "HIP_VISIBLE_DEVICES" not in body


def test_render_unit_lists_only_present_gpu_nodes(monkeypatch):
    # A pinned /dev/dri/card0 is wrong the moment the card enumerates as
    # card1 — systemd refuses to start on a DeviceAllow for a missing path.
    # Validates: docs/install.md, 'Two things the unit discovers rather than hardcodes': Fill the unit template for this host. The hardening block comes from the template verbatim; what is.
    monkeypatch.setattr(ins, "gpu_vendor", lambda: "amd")
    monkeypatch.setattr(ins, "gpu_nodes",
                        lambda: ["/dev/dri/renderD128", "/dev/dri/card1", "/dev/kfd"])
    body = ins.render_unit(_args(), Path("/d"))
    assert "DeviceAllow=/dev/dri/card1 rw" in body
    assert "card0" not in body


def test_render_unit_lists_only_present_hidden_dirs(monkeypatch):
    # /efi is absent on a box without a separate ESP, and a missing path in
    # InaccessiblePaths= is a fatal namespace error (226/NAMESPACE) — the
    # service never starts. Only what exists may be listed.
    #
    # Validates: docs/install.md, the systemd unit the installer writes.
    # Validates: docs/plans/requirement-trace.md #53: `InaccessiblePaths=` lists only directories present on this host
    monkeypatch.setattr(ins, "gpu_vendor", lambda: "amd")
    monkeypatch.setattr(ins, "gpu_nodes", lambda: ["/dev/kfd"])
    monkeypatch.setattr(ins, "inaccessible_paths",
                        lambda: ["/root", "/home", "/boot"])
    body = ins.render_unit(_args(), Path("/d"))
    assert "InaccessiblePaths=/root /home /boot" in body
    directives = [ln for ln in body.splitlines()
                  if not ln.lstrip().startswith("#")]
    assert not any("/efi" in ln for ln in directives)


def test_inaccessible_paths_omits_absent(tmp_path):
    """Validates: docs/install.md, the systemd unit the installer writes."""
    there = tmp_path / "there"
    there.mkdir()
    gone = tmp_path / "gone"
    assert ins.inaccessible_paths((str(there), str(gone))) == [str(there)]


def test_unit_is_enableable():
    # Without [Install] WantedBy=, `systemctl enable` fails and the service
    # never comes back after a reboot.
    #
    # Validates: docs/install.md, the systemd unit the installer writes.
    # Validates: docs/plans/requirement-trace.md #52: The unit is enableable: it carries `[Install] WantedBy=`
    body = ins.TEMPLATE.read_text()
    assert "[Install]" in body
    assert "WantedBy=multi-user.target" in body


def test_render_unit_rejects_untouched_token(monkeypatch, tmp_path):
    # A placeholder nobody filled is a silent misconfiguration, not a
    # cosmetic issue: the directive would reach systemd verbatim.
    # Validates: docs/install.md, 'Two things the unit discovers rather than hardcodes': Fill the unit template for this host. The hardening block comes from the template verbatim; what is.
    monkeypatch.setattr(ins, "gpu_vendor", lambda: "amd")
    monkeypatch.setattr(ins, "gpu_nodes", lambda: ["/dev/kfd"])
    tmpl = tmp_path / "t.in"
    tmpl.write_text("[Service]\nUser=@USER@\nExecStart=@LISTEN@\nMemoryMax=@NEVER_FILLED@\n")
    monkeypatch.setattr(ins, "TEMPLATE", tmpl)
    with pytest.raises(ins.InstallError, match="unsubstituted"):
        ins.render_unit(_args(), Path("/d"))


def test_no_trailing_comments_on_directives():
    # systemd reads the whole rest of the line as the value, so
    # `MemoryHigh=32G  # cap` is an invalid limit and is dropped without an
    # error. Prose belongs on its own line.
    # Validates: docs/install.md, 'Two things the unit discovers rather than hardcodes': a directive may not carry a trailing comment: systemd drops the whole line.
    for ln in ins.TEMPLATE.read_text().splitlines():
        if ln.lstrip().startswith("#"):
            continue
        assert "#" not in ln, f"inline comment in directive: {ln!r}"


# --- verification ----------------------------------------------------------


def _fake_verify(monkeypatch, stderr, returncode):
    monkeypatch.setattr(ins.subprocess, "run",
                        lambda *a, **k: type("R", (), {"returncode": returncode,
                                                       "stderr": stderr})())


def test_verify_unit_fails_on_our_unit_even_when_exit_zero(monkeypatch, tmp_path):
    # systemd-analyze exits 0 for a directive it merely "ignored" — the
    # exact shape of the trailing-comment bug. The verdict comes from the
    # messages, not the status.
    # Validates: docs/install.md, 'What it does': Run ``systemd-analyze verify`` and fail on any complaint about.
    _fake_verify(monkeypatch,
                 "/etc/systemd/system/llama-swap.service:50: Invalid memory "
                 "limit '32G  # cap', ignoring: Invalid argument\n", 0)
    with pytest.raises(ins.InstallError, match="Invalid memory limit"):
        ins.verify_unit(Path("/etc/systemd/system/llama-swap.service"))


def test_verify_unit_ignores_unrelated_unit_failures(monkeypatch):
    # The command audits the whole system and reports the worst finding
    # anywhere; a broken timer on the host must not fail every install.
    # Validates: docs/install.md, 'What it does': a complaint about another unit is not this install's business.
    _fake_verify(monkeypatch,
                 "/etc/systemd/system/logrotate.timer:7: Failed to parse\n",
                 1)
    ins.verify_unit(Path("/etc/systemd/system/llama-swap.service"))


# --- account ---------------------------------------------------------------


def test_user_supplementary_reports_memberships(monkeypatch):
    # Reading gr_mem off the primary group returns that group's *members*,
    # not this user's memberships, and reports every group as missing —
    # which would re-run usermod on every install.
    # Validates: docs/install.md, 'What it does': Names of the groups *user* belongs to, supplementary or primary.
    import grp as _grp
    monkeypatch.setattr(ins.os, "getgrouplist", lambda u, g: [10, 20, 30])
    monkeypatch.setattr(ins.grp, "getgrgid",
                        lambda g: type("G", (), {"gr_name": {10: "render",
                                                             20: "models",
                                                             30: "svc"}[g]})())
    monkeypatch.setattr(ins.pwd, "getpwnam",
                        lambda u: type("P", (), {"pw_gid": 30})())
    assert ins.user_supplementary("svc") == {"render", "models", "svc"}


def test_dry_run_prints_without_executing(tmp_path, capsys):
    # A dry run that touched the filesystem would be worse than no dry run:
    # the operator's first reaction to the output is to trust it.
    # Validates: docs/install.md, 'What it does': run.
    marker = tmp_path / "side-effect"
    ins.Runner(True).run(["touch", str(marker)])
    assert not marker.exists()
    assert str(marker) in capsys.readouterr().out


# --- binaries and backups --------------------------------------------------


def test_require_binaries_fails_when_config_would_point_nowhere(tmp_path):
    # find_bin_dir resolves relative to the cwd, so a config generated
    # before the binaries land is a config that points nowhere. Better to
    # say so here than as llama-packer's "no llama-b#### directory found".
    # Validates: docs/install.md, 'What it does': Fail early when the config would have nothing to point at. Checked here so the error names the missing piece, instead of surfacing.
    r = ins.Runner(False)
    with pytest.raises(ins.InstallError, match="no llama-b"):
        ins.require_binaries(r, tmp_path)


def test_require_binaries_passes_once_populated(tmp_path):
    # Validates: docs/install.md, 'The updater is the installer for binaries': The llama-swap this deployment actually has.
    build = tmp_path / "llama-b11658"
    build.mkdir()
    (build / "llama-server").write_text("x")
    (tmp_path / "llama-swap").write_text("x")
    ins.require_binaries(ins.Runner(False), tmp_path)


def test_install_binaries_pins_every_engine_to_the_run_dir(tmp_path):
    # Validates: docs/install.md, 'The updater is the installer for binaries': Put ``llama-swap`` and a llama.cpp build under *dir_*. Delegates to ``extras/update.py``, which already owns downloading,.
    """A fresh deployment must not adopt a parent directory's engines.

    update.py updates each engine where it finds it, searching outwards from
    the cwd. Install is the case where that default is wrong: a new run
    directory under an existing tools root would otherwise be handed the
    engines already installed above it, and the new deployment would ship a
    config pointing at somebody else's binaries.
    """
    seen: dict = {}

    class Recorder(ins.Runner):
        def run(self, cmd, **kwargs):
            seen["cmd"] = cmd

    build = ins.tools_dir(tmp_path) / "llama-b11658"
    build.mkdir(parents=True)
    (build / "llama-server").write_text("x")
    make = ins.tools_dir(tmp_path) / "llama-swap"
    make.write_text("x")

    ins.install_binaries(Recorder(False), tmp_path, "update")

    cmd = seen["cmd"]
    for engine in ("cpp", "swap", "audio", "whisper", "sd"):
        assert f"{engine}={ins.tools_dir(tmp_path)}" in cmd, engine


def test_binaries_go_in_the_tools_subdirectory(tmp_path):
    # Validates: docs/install.md, 'The updater is the installer for binaries': The llama-swap this deployment actually has.
    """The deployment layout: config at the top, engines under tools/."""
    build = ins.tools_dir(tmp_path) / "llama-b11714"
    build.mkdir(parents=True)
    (build / "llama-server").write_text("x")
    (ins.tools_dir(tmp_path) / "llama-swap").write_text("x")

    assert ins.require_binaries(ins.Runner(False), tmp_path) is None
    assert ins.newest_build(tmp_path).name == "llama-b11714"
    assert ins.llama_swap_binary(tmp_path) == ins.tools_dir(tmp_path) / "llama-swap"


def test_a_flat_run_directory_still_resolves(tmp_path):
    # Validates: docs/install.md, 'The updater is the installer for binaries': The llama-swap this deployment actually has.
    """Installs made before the tools/ layout must keep working."""
    (build := tmp_path / "llama-b11714").mkdir()
    (build / "llama-server").write_text("x")
    (tmp_path / "llama-swap").write_text("x")

    ins.require_binaries(ins.Runner(False), tmp_path)
    assert ins.newest_build(tmp_path).name == "llama-b11714"
    assert ins.llama_swap_binary(tmp_path) == tmp_path / "llama-swap"


def test_tools_wins_over_a_stale_flat_copy(tmp_path):
    # Validates: docs/install.md, 'The updater is the installer for binaries': The llama-swap this deployment actually has.
    """Both layouts present: the deployment one is the live one."""
    for base in (tmp_path, ins.tools_dir(tmp_path)):
        (build := base / "llama-b11714").mkdir(parents=True)
        (build / "llama-server").write_text("x")
        (base / "llama-swap").write_text("x")

    assert ins.llama_swap_binary(tmp_path) == ins.tools_dir(tmp_path) / "llama-swap"


def test_unit_execstart_names_the_installed_swap(tmp_path):
    # Validates: docs/install.md, 'Two things the unit discovers rather than hardcodes': Fill the unit template for this host. The hardening block comes from the template verbatim; what is.
    """The unit must not point at a file this deployment does not have."""
    import argparse as _argparse

    args = _argparse.Namespace(
        user="llama", group="models", listen="127.0.0.1:8123",
        memory_high="30G", memory_max="32G", swap_max="0")
    (build := ins.tools_dir(tmp_path) / "llama-b11714").mkdir(parents=True)
    (build / "llama-server").write_text("x")
    (ins.tools_dir(tmp_path) / "llama-swap").write_text("x")

    unit = ins.render_unit(args, tmp_path)

    assert f"ExecStart={ins.tools_dir(tmp_path) / 'llama-swap'} " in unit
    assert f"WorkingDirectory={tmp_path}" in unit
    assert "@" not in unit.split("[Service]")[1].replace("@system-service", "")


def test_backup_keeps_the_previous_config(tmp_path):
    # Both are gitignored and machine-local, so an install that overwrites
    # them has no other way back: a regeneration can move every context.
    # Validates: docs/install.md, 'Two orderings that are load-bearing': Copy *path* aside to ``<name>.bak-<n>`` before overwriting it. An install replaces a working config and a working unit in one step.
    live = tmp_path / "config.yaml"
    live.write_text("models: {}\n")
    made = ins.backup(live, dry_run=False)
    assert made.read_text() == "models: {}\n"
    assert live.read_text() == "models: {}\n"      # untouched by the copy
    # A second backup must not clobber the first.
    live.write_text("models: {a: 1}\n")
    again = ins.backup(live, dry_run=False)
    assert again.name == "config.yaml.bak-2"
    assert made.name == "config.yaml.bak-1"


def test_backup_noop_without_file(tmp_path):
    # Validates: docs/install.md, 'Two orderings that are load-bearing': Copy *path* aside to ``<name>.bak-<n>`` before overwriting it. An install replaces a working config and a working unit in one step.
    assert ins.backup(tmp_path / "absent", dry_run=False) is None


def test_config_is_generated_from_the_run_directory(tmp_path, monkeypatch):
    # LLAMA_DIR is derived from wherever find_bin_dir located the binary,
    # so generating from the repo would emit a config pointing back at the
    # repo and leave everything in the run directory unused.
    # Validates: docs/install.md, 'Running llama-packer afterwards': Run llama-packer as the service user to (re)write the config. As the service user rather than root, so the ``derived:`` blocks it.
    seen = {}
    monkeypatch.setattr(ins, "REPO", tmp_path / "repo")
    monkeypatch.setattr(ins, "find_packer", lambda e: "llama-packer")
    (tmp_path / "repo").mkdir()
    opt = tmp_path / "opt"
    opt.mkdir()

    class _R:
        dry_run = False

        def run(self, argv, **kw):
            seen["cwd"] = kw.get("cwd")
            for name in ("config.yaml", "config.env"):
                (kw["cwd"] / name).write_text("x")
            return type("R", (), {"stdout": "", "returncode": 0})()

    args = _args(models_dir=["/models"], profiles=tmp_path / "p.yaml",
                 spare=None, vram=None, hf_home=None)
    ins.generate_config(_R(), "svc", opt, args, "llama-packer")
    assert seen["cwd"] == opt


# --- hf_home resolution ----------------------------------------------------


def test_install_does_not_invent_hf_home(monkeypatch, tmp_path):
    # Under sudo the environment is stripped, so the obvious invention
    # (Path.home()/".cache") resolves to *root's* cache and every hf_repo
    # model silently stops resolving. llama-packer already has a chain for
    # this; install.py must not add a fourth mechanism over it.
    # Validates: docs/install.md, 'Running llama-packer afterwards': Run llama-packer as the service user to (re)write the config. As the service user rather than root, so the ``derived:`` blocks it.
    monkeypatch.setenv("HF_HOME", "/should/not/be/used")
    seen = {}
    monkeypatch.setattr(ins, "REPO", Path("/repo"))
    monkeypatch.setattr(ins, "find_packer", lambda e: "llama-packer")

    class _R:
        dry_run = False

        def run(self, argv, **kw):
            seen["env"] = kw.get("env", {})
            seen["argv"] = argv
            cwd = kw.get("cwd")
            (cwd / "config.yaml").write_text("models: {}\n")
            (cwd / "config.env").write_text(
                "LLAMA_DIR=/opt/llama/llama-b11669\nHF_HOME=/mnt/ai/huggingface\n")
            return type("R", (), {"stdout": "", "returncode": 0})()

    d = tmp_path
    args = _args(models_dir=["/models"], profiles=Path("/p.yaml"), spare=None,
                 vram=None, hf_home=None)
    ins.generate_config(_R(), "svc", d, args, "llama-packer")
    # HF_HOME is neither pinned nor injected into the pack-time environment.
    assert seen["env"]["HF_HOME"] == "/should/not/be/used"  # inherited, not set
    assert "--hf-home" not in seen["argv"]


def test_install_passes_hf_home_through(monkeypatch, tmp_path):
    # Validates: docs/install.md, 'Running llama-packer afterwards': Run llama-packer as the service user to (re)write the config. As the service user rather than root, so the ``derived:`` blocks it.
    seen = {}
    monkeypatch.setattr(ins, "find_packer", lambda e: "llama-packer")

    class _R:
        dry_run = False

        def run(self, argv, **kw):
            seen["argv"] = argv
            cwd = kw.get("cwd")
            (cwd / "config.yaml").write_text("models: {}\n")
            (cwd / "config.env").write_text("HF_HOME=/mnt/ai/huggingface\n")
            return type("R", (), {"stdout": "", "returncode": 0})()

    args = _args(models_dir=["/models"], profiles=Path("/p.yaml"), spare=None,
                 vram=None, hf_home="/mnt/ai/huggingface")
    ins.generate_config(_R(), "svc", tmp_path, args, "llama-packer")
    i = seen["argv"].index("--hf-home")
    assert seen["argv"][i + 1] == "/mnt/ai/huggingface"


def test_report_resolved_warns_on_missing_hf_home(tmp_path, caplog):
    # Validates: docs/install.md, 'Two things the unit discovers rather than hardcodes': Print the path macros the emitted config actually resolved to. Read back from ``config.env`` rather than predicted, because these are.
    (tmp_path / "config.env").write_text(
        "LLAMA_DIR=/opt/llama/llama-b11669\nHF_HOME=/root/.cache/huggingface\n")
    ins.report_resolved(tmp_path)
    warnings = [r.getMessage() for r in caplog.records]
    assert any("HF_HOME" in w and "does not exist" in w for w in warnings), warnings


def test_report_resolved_quiet_when_hf_home_exists(tmp_path, caplog):
    # Validates: docs/plans/requirement-trace.md #51: an unreadable HF_HOME warns instead of crashing the installer
    existing = tmp_path / "hfcache"
    existing.mkdir()
    (tmp_path / "config.env").write_text(f"HF_HOME={existing}\n")
    ins.report_resolved(tmp_path)
    assert not caplog.records, [r.message for r in caplog.records]


def test_report_resolved_warns_when_hf_home_is_unreadable(tmp_path, monkeypatch,
                                                          caplog):
    """EACCES (an HF_HOME under /root, say) must warn, not raise.

    ``Path.is_dir`` re-raises permission errors — it only swallows ENOENT and
    ENOTDIR — so before this the installer crashed on exactly the
    misconfiguration it exists to report. Forced here so the test does not
    depend on the runner's privileges, which is how the bug slipped through
    locally.

    Validates: docs/install.md, 'Pass --hf-home if your models come from an
    HF cache'.
    """
    (tmp_path / "config.env").write_text("HF_HOME=/root/.cache/huggingface\n")

    def boom(self):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(ins.Path, "is_dir", boom)
    ins.report_resolved(tmp_path)          # must not raise
    assert any("does not exist" in r.getMessage() for r in caplog.records)


# --- the unit is verified before it becomes the unit systemd reads --------


def _install_unit(monkeypatch, unit, dir_, verify):
    """Drive install_unit with the parser-shaped args main() supplies."""
    seen = {}
    monkeypatch.setattr(ins, "verify_unit", lambda p: (seen.update(p=p), verify(p))[1])
    args = _args(unit=unit, no_start=True, models_dir=[str(dir_)])
    out = ins.install_unit(ins.Runner(dry_run=False), args, dir_)
    return out, seen.get("p")


def test_rejected_unit_leaves_the_previous_one_installed(tmp_path, monkeypatch):
    """Write-then-verify left a broken unit on disk for the next reload.

    systemd re-reads the unit directory on any daemon-reload or at boot, so
    a unit that failed verification would become the one that starts -- not
    running now, but silently wrong once reloaded.

    Validates: docs/plans/requirement-trace.md #42: the unit is verified before it
    becomes the unit systemd reads.
    """
    unit = tmp_path / "llama-swap.service"
    unit.write_text("[Unit]\nDescription=old-good\n")

    def reject(_path):
        raise ins.InstallError("llama-swap.service: bogus directive")

    with pytest.raises(ins.InstallError):
        _install_unit(monkeypatch, unit, tmp_path, reject)

    assert unit.read_text() == "[Unit]\nDescription=old-good\n", "overwritten"
    assert not list(tmp_path.glob("*candidate*")), "candidate left behind"


def test_candidate_name_is_a_valid_unit_name(tmp_path, monkeypatch):
    """systemd-analyze cannot parse "x.service.tmp".

    It prints "Failed to prepare filename: Invalid argument" and exits 0
    without inspecting anything, which would make the verify step a no-op
    that approves any unit at all.

    Validates: docs/plans/requirement-trace.md #42: the candidate must keep the
    .service suffix, or systemd-analyze exits 0 without reading it.
    """
    unit = tmp_path / "llama-swap.service"
    _, candidate = _install_unit(monkeypatch, unit, tmp_path, lambda _p: None)
    assert candidate.suffix == ".service", candidate
    assert candidate != unit
    assert unit.is_file()


def test_validate_runs_as_the_service_user(tmp_path, monkeypatch):
    """The run directory is group-writable on purpose.

    Unlink+rename there needs only directory write permission, not ownership
    of the entry, so running llama-swap as root out of it lets any member
    of --group swap in a binary and have root execute it on the next
    install.

    Validates: docs/plans/requirement-trace.md #41: llama-swap validates as the
    service user, never root out of a group-writable directory.
    """
    commands = []
    r = ins.Runner(dry_run=False)
    monkeypatch.setattr(r, "run", lambda cmd, **kw: commands.append(cmd) or
                        type("R", (), {"returncode": 0, "stderr": "",
                                      "stdout": ""})())
    (tmp_path / "llama-swap").write_bytes(b"bin")
    (tmp_path / "config.yaml").write_text("models: {}\n")
    monkeypatch.setattr(ins, "probe", lambda *a, **k: None)   # no live server
    ins.start(r, tmp_path / "llama-swap.service", tmp_path, "127.0.0.1:1",
              "lemonade")
    validate = [c for c in commands if "-validate" in c]
    assert validate, commands
    assert validate[0][:4] == ["runuser", "-u", "lemonade", "--"], validate[0]


def test_probe_waits_for_a_slow_start(monkeypatch, capsys):
    """A single request races the socket against llama-swap's bind, so the
    probe must poll — otherwise it reports a dead service for one that came
    up fine, which is what it did on the first successful install.

    Validates: docs/install.md, the /v1/models health probe.
    """
    import urllib.error

    calls = {"n": 0}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"data": [{"id": "m", "name": "M"}]}'

    def fake_urlopen(url, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("Connection refused")
        return _Resp()

    monkeypatch.setattr(ins.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ins.time, "sleep", lambda s: None)
    ins.probe(ins.Runner(dry_run=False), "127.0.0.1:8123")
    assert calls["n"] == 3
    assert "1 models" in capsys.readouterr().out


def test_probe_gives_up_after_the_deadline(monkeypatch):
    """Validates: docs/install.md, the /v1/models health probe."""
    import urllib.error

    clock = {"t": 0.0}

    def fake_urlopen(url, timeout=None):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(ins.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ins.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(ins.time, "sleep",
                        lambda s: clock.__setitem__("t", clock["t"] + 10))
    with pytest.raises(ins.InstallError, match="did not answer"):
        ins.probe(ins.Runner(dry_run=False), "127.0.0.1:8123", timeout_s=5)


def test_repack_does_not_repoint_home(tmp_path, monkeypatch):
    # Validates: docs/install.md, 'Running llama-packer afterwards': Run llama-packer as the service user to (re)write the config. As the service user rather than root, so the ``derived:`` blocks it.
    """HOME=dir_ made the HF cache fallback resolve under the run dir.

    Path.home() honours HOME, so with no --hf-home the fallback became
    /opt/llama/.cache/huggingface, which does not exist: no HF_HOME was
    emitted, nothing warned, and every hf_repo entry silently stopped
    resolving. The exact failure the "invent nothing" note above it
    describes, relocated.
    """
    seen = {}
    r = ins.Runner(dry_run=False)
    monkeypatch.setattr(r, "run", lambda cmd, **kw: seen.update(env=kw.get("env"))
                        or type("R", (), {"returncode": 0, "stdout": ""})())
    (tmp_path / "config.yaml").write_text("models: {}\n")
    (tmp_path / "config.env").write_text("")
    monkeypatch.setattr(ins, "report_resolved", lambda _d: None)
    args = _args(models_dir=[str(tmp_path)],
                 profiles=str(tmp_path / "profiles.yaml"))
    ins.generate_config(r, "lemonade", tmp_path, args, "llama-packer")
    env = seen["env"]
    # Inheriting the operator's real HOME is fine and intended; overriding
    # it to the run directory is the defect.
    assert env.get("HOME") != str(tmp_path), env.get("HOME")
    assert env.get("HOME") == os.environ.get("HOME")


# --- a typo'd group must stop before anything is created -------------------


def test_missing_group_is_fatal_and_creates_nothing(tmp_path, monkeypatch):
    # Validates: docs/install.md, 'What it does': Create the service account if absent; add the GPU + config groups.
    """It used to be a warning.

    prepare_dir chowns the run directory to that group, so the run created
    it as root:root 2775 -- setgid and group-writable by root -- and then
    died on KeyError from grp.getgrnam, i.e. a traceback after the damage
    rather than before it.
    """
    ran = []
    monkeypatch.setattr(ins, "group_exists", lambda g: g != "typo")
    monkeypatch.setattr(ins, "user_exists", lambda u: True)
    r = ins.Runner(dry_run=False)
    monkeypatch.setattr(r, "run", lambda cmd, **kw: ran.append(cmd))
    with pytest.raises(ins.InstallError) as exc:
        ins.ensure_user(r, "svc", "typo")
    assert "does not exist" in str(exc.value)
    assert ran == [], ran


def test_existing_group_proceeds(tmp_path, monkeypatch):
    # Validates: docs/install.md, 'What it does': Create the service account if absent; add the GPU + config groups.
    ran = []
    monkeypatch.setattr(ins, "group_exists", lambda _g: True)
    monkeypatch.setattr(ins, "user_exists", lambda _u: True)
    monkeypatch.setattr(ins, "user_supplementary", lambda _u: {"grp"})
    r = ins.Runner(dry_run=False)
    monkeypatch.setattr(r, "run", lambda cmd, **kw: ran.append(cmd))
    ins.ensure_user(r, "svc", "grp")
    # The GPU groups are genuinely missing here and get added; grp is not.
    added = [c[2] for c in ran if c[:2] == ["usermod", "-aG"]]
    assert "grp" not in added, added
    assert set(added) == set(ins._GPU_GROUPS), added
