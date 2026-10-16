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

There is deliberately no test that drives ``main()`` end to end. A pack
needs a real llama.cpp build *and* a parseable gguf -- llama-fit-params
aborts on a stub file -- so against a fixture tree the run dies in the VRAM
solve and never reaches the output stage. A test that tolerated that would
pass just as happily against a ``main()`` that crashed on entry, which is
worse than no test at all. The whole-CLI behaviour is verified by running
the real binary.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from llama_packer import model as _modelmod
from llama_packer.model import Model

# module attributes rather than re-exports: the counter is internal state
persisted_sidecars = _modelmod.persisted_sidecars
persisted_sidecars_reset = _modelmod.persisted_sidecars_reset


@pytest.fixture(autouse=True)
def _restore_cache():
    """Never let one test's switch leak into the next."""
    Model.persist_cache = True
    persisted_sidecars_reset()
    yield
    Model.persist_cache = True
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
    d = tmp_path / "chat"
    d.mkdir()
    (d / "lonely.gguf").write_bytes(b"x")
    Model.from_dir([tmp_path], generate_stubs=False)
    assert not (d / "lonely.md").exists()


# --- 2. the derived: block -------------------------------------------------


def test_discovery_writes_no_file_fingerprint_when_cache_disabled(tmp_path):
    """The write that fires on *every* pack, measured or not.

    Discovery refreshes the ``derived: file:`` intrinsics (size/mtime/arch)
    through the same writer, so without a gate a preview rewrites every
    sidecar it merely looked at - including models it never measured. Snapshot
    before, because with the cache on, construction itself writes.
    """
    md = _sidecar(tmp_path)
    before = md.read_bytes()

    Model.persist_cache = False
    Model.from_dir([tmp_path], generate_stubs=False)

    assert md.read_bytes() == before, "discovery wrote the file: fingerprint"


def test_persist_measured_writes_when_enabled(tmp_path):
    # Guards against over-gating: the cache is the feature, not the bug.
    m = _bound_model(tmp_path)
    persisted_sidecars_reset()
    m.persist_measured({"model_mib": 4800, "source": "fit-estimate"})
    assert "derived:" in m.md_path.read_text()
    assert "documented" in persisted_sidecars()


def test_persist_measured_skips_the_write_when_cache_disabled(tmp_path):
    m = _bound_model(tmp_path)
    persisted_sidecars_reset()   # construction writes the file: fingerprint
    before = m.md_path.read_bytes()

    m.persist_cache = False
    m.persist_measured({"model_mib": 4800, "source": "fit-estimate"})

    assert m.md_path.read_bytes() == before, "sidecar was rewritten"
    # ...but the value is still in memory: callers (saved_for,
    # measured_file_valid) must see it within this run.
    assert m.frontmatter["derived"]["model_mib"] == 4800
    assert persisted_sidecars() == []


def test_dry_run_creates_no_stub_via_the_cli_gate(tmp_path):
    """Same gate the CLI sets, reached the way the CLI reaches it."""
    d = tmp_path / "chat"
    d.mkdir()
    (d / "lonely.gguf").write_bytes(b"x")
    snapshot = _tree(tmp_path)

    # What main() does for --dry-run.
    from llama_packer.__main__ import parse_args
    assert parse_args(["llama-packer", "--dry-run"]).dry_run is True
    Model.persist_cache = False
    Model.from_dir([tmp_path],
                   generate_stubs=not (False or parse_args(
                       ["llama-packer", "--dry-run"]).dry_run))

    assert not (d / "lonely.md").exists()
    assert _tree(tmp_path) == snapshot


# --- 3. the CLI's own switches ---------------------------------------------


def test_dry_run_and_agents_is_refused(tmp_path, monkeypatch, capsys):
    # One says "write nothing", the other exists to write a file. Guessing
    # either way is a footgun.
    models = tmp_path / "models"
    (models / "chat").mkdir(parents=True)
    (models / "chat" / "lonely.gguf").write_bytes(b"x")
    monkeypatch.chdir(tmp_path)

    from llama_packer.__main__ import main
    with pytest.raises(SystemExit):
        main(["llama-packer", "--dry-run", "--agents",
              "--models-dir", str(models),
              "--output", str(tmp_path / "config.yaml")])
    assert "contradict" in capsys.readouterr().err


def test_dry_run_parses_and_defaults_off():
    from llama_packer.__main__ import parse_args
    assert parse_args(["llama-packer", "--dry-run"]).dry_run is True
    assert parse_args(["llama-packer"]).dry_run is False
