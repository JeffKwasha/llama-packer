"""`--dry-run` must touch nothing.

The contract is that a preview creates, modifies and deletes **no** files.
That is easy to assert and easy to violate, because three separate code
paths write independently of the config output:

  1. ``config.yaml`` and its sibling ``config.env``
  2. stub sidecars, created beside orphan model files during discovery
  3. the ``derived:`` block written back into sidecars — both the
     ``file:`` fingerprint that discovery refreshes on sight, and the
     measurement cache filled in afterwards

Every one of those funnels through ``Model.persist_measured`` or the
stub writer, and each gate is tested here directly.

The whole-CLI behaviour is driven in ``tests/test_logging.py``.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pytest

from llama_packer import model as _modelmod
from llama_packer import discover, utils
from llama_packer.model import Model

# module attributes rather than re-exports: the counter is internal state
persisted_sidecars = _modelmod.persisted_sidecars
persisted_sidecars_reset = _modelmod.persisted_sidecars_reset


@pytest.fixture(autouse=True)
def _restore_cache():
    """Never let one test's switch leak into the next."""
    utils.set_dry_run(False)
    persisted_sidecars_reset()
    yield
    utils.set_dry_run(False)
    persisted_sidecars_reset()


def _tree(root: Path) -> dict[str, tuple[int, str]]:
    """Every file under *root* as (size, sha256) - content, not just names."""
    out: dict[str, tuple[int, str]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        data = path.read_bytes()
        out[str(path.relative_to(root))] = (
            len(data), hashlib.sha256(data).hexdigest())
    return out


def _sidecar(dir_: Path, stem: str = "documented") -> Path:
    (dir_ / f"{stem}.gguf").write_bytes(b"x")
    md = dir_ / f"{stem}.md"
    md.write_text(f"---\nname: {stem.capitalize()}\n---\n")
    return md


def _bound_model(dir_: Path) -> Model:
    """A sidecar-*bound* Model.

    ``Model.from_file`` returns the canonical sidecar-less instance, which has
    no ``md_path`` and is not what ``persist_measured`` writes through.
    """
    _sidecar(dir_)
    (model,) = Model.from_dir([dir_], generate_stubs=False)
    assert model.md_path is not None
    return model


# --- 1. stub sidecars ------------------------------------------------------


def test_discovery_creates_no_stub_when_stubs_disabled(tmp_path):
    # The pre-existing escape hatch. Kept: --no-stubs means "never", and must
    # keep working when not dry-running. --dry-run turns it on implicitly.
    # Validates: SPEC.md, 'Model Discovery and Stub Sidecars': stub
    # creation is off, so no sidecar is written beside a lone weight file.
    d = tmp_path / "chat"
    d.mkdir()
    (d / "lonely.gguf").write_bytes(b"x")
    Model.from_dir([tmp_path], generate_stubs=False)
    assert not (d / "lonely.md").exists()


# --- 2. the derived: block -------------------------------------------------


def test_discovery_writes_no_file_fingerprint_when_cache_disabled(tmp_path):
    # Validates: SPEC.md, 'Measured-Block Persistence': a preview does not rewrite the file: fingerprint of a sidecar it merely looked at.
    """The write that fires on *every* pack, measured or not.

    Discovery refreshes the ``derived: file:`` intrinsics (size/mtime/arch)
    through the same writer, so without a gate a preview rewrites every
    sidecar it merely looked at - including models it never measured. Snapshot
    before, because with the cache on, construction itself writes.
    """
    md = _sidecar(tmp_path)
    before = md.read_bytes()

    utils.set_dry_run(True)
    Model.from_dir([tmp_path], generate_stubs=False)

    assert md.read_bytes() == before, "discovery wrote the file: fingerprint"


def test_persist_measured_writes_when_enabled(tmp_path):
    # Guards against over-gating: the cache is the feature, not the bug.
    # Validates: SPEC.md, 'Measured-Block Persistence': the cache is a feature: with it on, the measured block is written to the sidecar.
    m = _bound_model(tmp_path)
    persisted_sidecars_reset()
    m.persist_measured({"model_mib": 4800, "source": "fit-estimate"})
    assert "derived:" in m.md_path.read_text()
    assert "documented" in persisted_sidecars()


def test_persist_measured_skips_the_write_when_cache_disabled(tmp_path):
    # Validates: SPEC.md, 'Measured-Block Persistence': with the cache off the file is untouched, but the value is still in memory for the rest of the run.
    m = _bound_model(tmp_path)
    persisted_sidecars_reset()   # construction writes the file: fingerprint
    before = m.md_path.read_bytes()

    utils.set_dry_run(True)
    m.persist_measured({"model_mib": 4800, "source": "fit-estimate"})

    assert m.md_path.read_bytes() == before, "sidecar was rewritten"
    # ...but the value is still in memory: callers (saved_for,
    # measured_file_valid) must see it within this run.
    assert m.frontmatter["derived"]["model_mib"] == 4800
    assert persisted_sidecars() == []


def test_dry_run_creates_no_stub_via_the_cli_gate(tmp_path):
    # Validates: SPEC.md, 'Measured-Block Persistence': --dry-run is contractually read-only: it creates, modifies and deletes no files
    """Same gate the CLI sets, reached the way the CLI reaches it."""
    d = tmp_path / "chat"
    d.mkdir()
    (d / "lonely.gguf").write_bytes(b"x")
    snapshot = _tree(tmp_path)

    # What main() does for --dry-run.
    from llama_packer.__main__ import parse_args
    assert parse_args(["llama-packer", "--dry-run"]).dry_run is True
    utils.set_dry_run(True)
    Model.from_dir([tmp_path],
                   generate_stubs=not (False or parse_args(
                       ["llama-packer", "--dry-run"]).dry_run))

    assert not (d / "lonely.md").exists()
    assert _tree(tmp_path) == snapshot


# --- 3. the CLI's own switches ---------------------------------------------


def test_dry_run_previews_agents_instead_of_writing_it(tmp_path, caplog):
    # Validates: SPEC.md, 'Measured-Block Persistence': it composes with --agents: those previews report what they would write rather than being refused
    """--agents previews too. "Show me what you would write" is the point of
    a dry run, so the combination is answerable rather than refused."""
    from llama_packer.__main__ import write_agents_md
    models = tmp_path / "models"
    models.mkdir()
    with caplog.at_level(logging.INFO, logger="llama_packer.utils"):
        utils.set_dry_run(True)
        write_agents_md(models)
    reported = [r.getMessage() for r in caplog.records if "would " in r.getMessage()]
    assert any("AGENTS.md guide" in m for m in reported), reported
    assert not (models / "AGENTS.md").exists()


def test_dry_run_parses_and_defaults_off():
    # Validates: SPEC.md, 'Measured-Block Persistence': one flag (utils.DRY_RUN) and one reporting function (utils.note_fs_op) serve all of it
    from llama_packer.__main__ import parse_args
    assert parse_args(["llama-packer", "--dry-run"]).dry_run is True
    assert parse_args(["llama-packer"]).dry_run is False


# --- 4. the preview reports what it withheld -------------------------------


def test_dry_run_reports_each_withheld_write(tmp_path, caplog):
    # Validates: SPEC.md, 'Measured-Block Persistence': every withheld operation is reported, so a preview is evidence rather than silence
    """A preview that says nothing is indistinguishable from a clean run.

    That silence is what hid the three writers in the first place, so the
    report is part of the contract, not a nicety.
    """
    md = _sidecar(tmp_path)
    chat = tmp_path / "chat"
    chat.mkdir()
    (chat / "lonely.gguf").write_bytes(b"x")

    with caplog.at_level(logging.INFO, logger="llama_packer.utils"):
        utils.set_dry_run(True)
        discover._write_stub(chat / "lonely.md")
        model = next(m for m in Model.from_dir([tmp_path], generate_stubs=False)
                     if m.md_path is not None)
        model.persist_measured({"model_mib": 4800})

    reported = [r.getMessage() for r in caplog.records if "would " in r.getMessage()]
    assert any("lonely.md" in m for m in reported), reported
    assert any(str(md) in m for m in reported), reported
    assert not (chat / "lonely.md").exists(), "but nothing was written"
    assert "derived:" not in md.read_text(), "nor the sidecar"


def test_note_fs_op_carries_the_detail(caplog):
    # Validates: SPEC.md, 'Measured-Block Persistence': one reporting function (utils.note_fs_op) serves all of it, so a new writer cannot silently bypass the report
    with caplog.at_level(logging.INFO, logger="llama_packer.utils"):
        utils.note_fs_op("update derived: in", "/m.md", "stem")
    (msg,) = [r.getMessage() for r in caplog.records if "would " in r.getMessage()]
    assert msg == "would update derived: in: /m.md (stem)"


def test_verbosity_ladder_and_floor():
    # Validates: SPEC.md, 'Log Levels': -v lifts the level to INFO, -vv to DEBUG, -vvv to whatever the libraries log. The floor is never lowered: -v adds detail, it never silences a warning.
    from llama_packer.__main__ import setup_logging
    import logging as _logging
    try:
        setup_logging(0)
        assert _logging.getLogger().getEffectiveLevel() == _logging.WARNING
        setup_logging(1)
        assert _logging.getLogger().getEffectiveLevel() == _logging.INFO
        setup_logging(2)
        assert _logging.getLogger().getEffectiveLevel() == _logging.DEBUG
        setup_logging(0, floor=_logging.INFO)
        assert _logging.getLogger().getEffectiveLevel() == _logging.INFO
    finally:
        setup_logging(0)
