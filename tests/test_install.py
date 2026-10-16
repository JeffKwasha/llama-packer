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
    dri = tmp_path / "dri"
    dri.mkdir()
    (dri / "renderD128").touch()
    nodes = ins.gpu_nodes(dri, tmp_path / "absent-kfd")
    assert nodes == [f"{dri}/renderD128"]


def test_newest_build_orders_numerically(tmp_path):
    # llama-b9999 must lose to llama-b11658; lexicographic order gets this
    # backwards, and picking the wrong build means deploying a stale binary.
    for name in ("llama-b9999", "llama-b11658", "llama-b11539"):
        d = tmp_path / name
        d.mkdir()
        (d / "llama-server").write_text("x")
    (tmp_path / "llama-b12000").mkdir()      # no llama-server: not a build
    assert ins.newest_build(tmp_path).name == "llama-b11658"


def test_newest_build_none_without_builds(tmp_path):
    (tmp_path / "llama-b1").mkdir()
    assert ins.newest_build(tmp_path) is None


# --- unit rendering --------------------------------------------------------


def test_render_unit_substitutes_directives_only(monkeypatch):
    # The template header names the placeholders it expects. Substituting
    # those mentions would turn the documentation into a copy of this host.
    monkeypatch.setattr(ins, "gpu_vendor", lambda: "amd")
    monkeypatch.setattr(ins, "gpu_nodes", lambda: ["/dev/dri/renderD128"])
    body = ins.render_unit(_args(), Path("/opt/llama"))
    assert "User=svc" in body and "Group=grp" in body
    assert "WorkingDirectory=/opt/llama" in body
    assert "ExecStart=/opt/llama/llama-swap --listen 127.0.0.1:8123" in body
    # No placeholder survives on a directive. (Comments keep their mentions
    # on purpose: the header documents the tokens it expects.)
    directives = [ln for ln in body.splitlines() if not ln.lstrip().startswith("#")]
    assert not [ln for ln in directives if ins._TOKEN_RE.search(ln)]
    assert "@PLACEHOLDER@" in body        # header prose intact


def test_render_unit_gpu_env_follows_hardware(monkeypatch):
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
    monkeypatch.setattr(ins, "gpu_vendor", lambda: "amd")
    monkeypatch.setattr(ins, "gpu_nodes",
                        lambda: ["/dev/dri/renderD128", "/dev/dri/card1", "/dev/kfd"])
    body = ins.render_unit(_args(), Path("/d"))
    assert "DeviceAllow=/dev/dri/card1 rw" in body
    assert "card0" not in body


def test_render_unit_rejects_untouched_token(monkeypatch, tmp_path):
    # A placeholder nobody filled is a silent misconfiguration, not a
    # cosmetic issue: the directive would reach systemd verbatim.
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
    _fake_verify(monkeypatch,
                 "/etc/systemd/system/llama-swap.service:50: Invalid memory "
                 "limit '32G  # cap', ignoring: Invalid argument\n", 0)
    with pytest.raises(ins.InstallError, match="Invalid memory limit"):
        ins.verify_unit(Path("/etc/systemd/system/llama-swap.service"))


def test_verify_unit_ignores_unrelated_unit_failures(monkeypatch):
    # The command audits the whole system and reports the worst finding
    # anywhere; a broken timer on the host must not fail every install.
    _fake_verify(monkeypatch,
                 "/etc/systemd/system/logrotate.timer:7: Failed to parse\n",
                 1)
    ins.verify_unit(Path("/etc/systemd/system/llama-swap.service"))


# --- account ---------------------------------------------------------------


def test_user_supplementary_reports_memberships(monkeypatch):
    # Reading gr_mem off the primary group returns that group's *members*,
    # not this user's memberships, and reports every group as missing —
    # which would re-run usermod on every install.
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
    marker = tmp_path / "side-effect"
    ins.Runner(True).run(["touch", str(marker)])
    assert not marker.exists()
    assert str(marker) in capsys.readouterr().out


# --- binaries and backups --------------------------------------------------


def test_require_binaries_fails_when_config_would_point_nowhere(tmp_path):
    # find_bin_dir resolves relative to the cwd, so a config generated
    # before the binaries land is a config that points nowhere. Better to
    # say so here than as llama-packer's "no llama-b#### directory found".
    r = ins.Runner(False)
    with pytest.raises(ins.InstallError, match="no llama-b"):
        ins.require_binaries(r, tmp_path)


def test_require_binaries_passes_once_populated(tmp_path):
    build = tmp_path / "llama-b11658"
    build.mkdir()
    (build / "llama-server").write_text("x")
    (tmp_path / "llama-swap").write_text("x")
    ins.require_binaries(ins.Runner(False), tmp_path)


def test_backup_keeps_the_previous_config(tmp_path):
    # Both are gitignored and machine-local, so an install that overwrites
    # them has no other way back: a regeneration can move every context.
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
    assert ins.backup(tmp_path / "absent", dry_run=False) is None


def test_config_is_generated_from_the_run_directory(tmp_path, monkeypatch):
    # LLAMA_DIR is derived from wherever find_bin_dir located the binary,
    # so generating from the repo would emit a config pointing back at the
    # repo and leave everything in the run directory unused.
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
    (tmp_path / "config.env").write_text(
        "LLAMA_DIR=/opt/llama/llama-b11669\nHF_HOME=/root/.cache/huggingface\n")
    ins.report_resolved(tmp_path)
    warnings = [r.getMessage() for r in caplog.records]
    assert any("HF_HOME" in w and "does not exist" in w for w in warnings), warnings


def test_report_resolved_quiet_when_hf_home_exists(tmp_path, caplog):
    existing = tmp_path / "hfcache"
    existing.mkdir()
    (tmp_path / "config.env").write_text(f"HF_HOME={existing}\n")
    ins.report_resolved(tmp_path)
    assert not caplog.records, [r.message for r in caplog.records]


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


def test_repack_does_not_repoint_home(tmp_path, monkeypatch):
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
