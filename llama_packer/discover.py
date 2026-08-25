# llama_packer/discover.py
"""Depth-first model discovery: one walk, one scope stack, no special cases.

Discovery is a preorder DFS over each models dir.  At every level:

1. push that directory's ``models.yaml`` onto the :class:`ScopeStack`
   (its ``defaults`` and ``overrides``);
2. build the models *of this directory* — authored sidecars first-class,
   orphan GGUF/safetensors via an empty editable stub sidecar;
3. recurse into children; pop.

Every model goes through the identical pipeline, in this order::

    merge scope defaults ← sidecar frontmatter   (layer 1: identity)
    apply override rules                          (layer 2: selection)
    resolve_companions()                          (derived state, once)
    finalize()                                    (backend + file refs)

Because companions resolve only after the frontmatter is final, rules can
freely set ``mmproj`` / ``speculative`` / ``hf_repo`` — no re-resolution
special cases anywhere.

Sidecars are never written into an HF hub ``blobs/`` tree (blob hashes are
not human-readable).  An orphan living in blobs gets its stub beside the
human-named snapshot entry when one resolves, else behind a self-created
symlink in a served category directory.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from llama_packer import utils
from llama_packer.model import Model
from llama_packer.scope import ScopeStack

logger = logging.getLogger(__name__)


def discover(
    models_dirs,
    *,
    stack: ScopeStack | None = None,
    generate_stubs: bool = True,
    extra_dirs: list[str] | None = None,
    dir_roles: dict | None = None,
    hf_home=None,
) -> list[Model]:
    """Walk *models_dirs* depth-first and return deduplicated Models.

    ``stack`` carries global configuration (profiles.yaml overrides); when
    omitted, only directory-scoped ``models.yaml`` applies.  Directory scopes
    are pushed/popped around each level of the walk, inner beating outer.
    """
    if isinstance(models_dirs, (str, os.PathLike)):
        models_dirs = [Path(models_dirs)]
    else:
        models_dirs = [Path(d) for d in models_dirs]

    stack = stack or ScopeStack()
    role_map = utils.dir_role_map(extra_dirs, dir_roles)

    models: list[Model] = []
    skipped: set[str] = set()
    for root in models_dirs:
        if not root.is_dir():
            continue
        ignore = utils.load_model_ignore(root)
        _walk(root, root, None, role_map, ignore, stack,
              generate_stubs, hf_home, models, skipped)

    if skipped:
        logger.info("skipping %s (not in dirs map; extend via profiles.yaml dirs:)",
                    ", ".join(f"{d}/" for d in sorted(skipped)))

    # Deduplicate links to the same model file (first wins): symlinks via
    # realpath, hardlinks via (st_dev, st_ino).
    deduped: list[Model] = []
    seen: set[str] = set()
    for model in models:
        key = str(model.md_path)
        if model.gguf_path:
            try:
                st = os.stat(str(model.gguf_path))
                key = f"{os.path.realpath(str(model.gguf_path))}|{st.st_dev}:{st.st_ino}"
            except OSError:
                key = os.path.realpath(str(model.gguf_path))
        if key in seen:
            logger.info("duplicate model file (symlink/hardlink) skipped: %s", model.stem)
            continue
        seen.add(key)
        deduped.append(model)
    return deduped


def _walk(d: Path, root: Path, role: str | None, role_map: dict[str, str],
          ignore: list[str], stack: ScopeStack, generate_stubs: bool,
          hf_home, out: list[Model], skipped: set[str]) -> None:
    """Preorder DFS: this directory's scope and models first, then children.

    *role* is the inherited directory role (None at a models-dir root and for
    unmapped depth-1 directories, whose subtrees are not served at all).
    """
    stack.push(utils.load_dir_config(d), origin=str(d / utils.DIR_CONFIG_NAME))
    try:
        for p in sorted(d.iterdir()):
            if p.is_dir():
                continue
            rel_parts = p.relative_to(root).parts
            if utils._is_ignored(rel_parts, p.relative_to(root).as_posix(), ignore):
                continue
            suffix = p.suffix.lower()
            if suffix == ".md":
                _model_from_sidecar(p, root, role, stack, hf_home, out)
            elif suffix == ".bin":
                # Whisper GGML weights: served only via an authored same-stem
                # sidecar inside an s2t-mapped directory — never walked as
                # orphans, never stubbed (few models, low churn).
                if role == "s2t" and not p.with_suffix(".md").is_file():
                    logger.info("skipping %s: .bin model without a same-stem "
                                ".md sidecar (write one to serve it)", p.name)
            elif suffix in (".gguf", ".safetensors"):
                if utils.companion_kind(p.stem):
                    continue  # companions join via fuzzy resolution, never walked
                if p.with_suffix(".md").is_file():
                    continue  # authored sidecar owns this file
                # Image role: safetensors orphans are typically companions, not
                # standalone diffusion models. Require a sidecar to serve a safetensors
                # image model (gguf diffusion orphans still auto-stub).
                if role == "image" and suffix == ".safetensors":
                    continue
                _model_from_orphan(p, root, role, stack, generate_stubs,
                                   hf_home, out)
        for child in sorted(d.iterdir()):
            if not child.is_dir():
                continue
            rel_first = child.relative_to(root).parts[0].lower()
            child_role = role or role_map.get(rel_first)
            if role is None and child_role is None:
                # Depth-1 directory outside the role whitelist: not served.
                skipped.add(rel_first)
                continue
            _walk(child, root, child_role, role_map, ignore, stack,
                  generate_stubs, hf_home, out, skipped)
    finally:
        stack.pop()


def _build(path: Path, frontmatter: dict, role: str | None, stack: ScopeStack,
           hf_home, out: list[Model]) -> None:
    """The single model pipeline: merge → construct → rules → companions → finalize."""
    merged = stack.merge_defaults(frontmatter)
    # A model in embed//rerank/image/s2t/t2s inherits its role from the
    # location when its own data (sidecar or defaults) does not declare one.
    if role in ("embeddings", "rerank", "image", "s2t", "t2s") and "role" not in merged:
        merged["role"] = role
    try:
        model = Model(path, merged, hf_home=hf_home)
    except Exception as e:
        logger.warning("failed to load model from %s: %s", path.name, e)
        return
    stack.apply_rules(model)
    model.resolve_companions()
    stack.finalize(model)
    # Guard: keep generative-media weights out of served text roles.  The
    # classification is header-only (GGUF metadata / safetensors names /
    # cached HF model card) and never blocks the run — one error log per
    # offending file, then the model is excluded from the config.
    # The `image` role is exempt — diffusion weights are expected there.
    if model.role in utils.SERVED_ROLES and model.role != "image" and model.gguf_path:
        kind = utils.classify_file(model.gguf_path)
        if kind == "unknown" and model.hf_repo:
            kind = utils.hf_readme_kind(model.hf_repo, hf_home) or "unknown"
        if kind == "image":
            logger.error(
                "%s: diffusion/image weights classified in a served %s role "
                "(header check); excluded. Move under an image dir, set "
                "`ignore: true`, or extend dirs: in profiles.yaml",
                path.name, model.role)
            return
    # Guard: image role should only serve diffusion/image weights — skip
    # textual-inversion embeddings, loras, and other assets that live under
    # img/ but are not diffusion models themselves (they are companions).
    if model.role == "image" and model.gguf_path:
        kind = utils.classify_file(model.gguf_path)
        if kind == "unknown" and model.hf_repo:
            kind = utils.hf_readme_kind(model.hf_repo, hf_home) or "unknown"
        if kind != "image":
            logger.info("skipping %s: not diffusion/image weights in image role (kind=%s)",
                        path.name, kind)
            return
    out.append(model)


def _effective_role(fm: dict, role: str | None) -> str | None:
    """Role for a sidecar: an embeddings/rerank/image/s2t/t2s *directory* wins,
    then the explicit ``role:``, then a ``type:`` field containing embed/rerank/image."""
    if role in ("embeddings", "rerank", "image", "s2t", "t2s"):
        return role
    explicit = str(fm.get("role") or "")
    if explicit:
        return explicit
    typ = str(fm.get("type") or "").lower()
    if "rerank" in typ:
        return "rerank"
    if "embed" in typ:
        return "embeddings"
    if "image" in typ:
        return "image"
    return role


def _model_from_sidecar(md_path: Path, root: Path, role: str | None,
                        stack: ScopeStack, hf_home, out: list[Model]) -> None:
    fm = utils.parse_frontmatter(md_path)
    if not fm and not any(md_path.with_suffix(e).is_file()
                          for e in (".gguf", ".safetensors", ".bin", ".onnx")):
        return  # no data and no model beside it: not a sidecar (README, ...)
    if fm.get("ignore"):
        logger.info("ignore: skipping %s", md_path.name)
        return
    _build(md_path, fm, _effective_role(fm, role), stack, hf_home, out)


def _model_from_orphan(gguf: Path, root: Path, role: str | None,
                       stack: ScopeStack, generate_stubs: bool,
                       hf_home, out: list[Model]) -> None:
    """Serve an orphan model file; materialize its empty editable sidecar.

    Stubs carry no data — identity falls back to the stem, context to the
    built-in default, role to the directory — so a stub and an authored
    sidecar behave identically through the pipeline.
    """
    md_path = _materialize_sidecar(gguf, root, generate_stubs)
    _build(md_path, {}, role, stack, hf_home, out)


def _write_stub(md_path: Path) -> None:
    if md_path.exists():
        return
    try:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        utils.write_stub_md(md_path)
        logger.info("stub: %s", md_path)
    except OSError as e:
        logger.warning("could not write stub sidecar %s (%s); "
                       "serving with defaults only", md_path, e)


def _materialize_sidecar(gguf: Path, root: Path, generate_stubs: bool) -> Path:
    """Return (and create) the editable-sidecar location for an orphan model.

    Normally the sidecar sits beside the model file.  Never inside an HF hub
    ``blobs/`` tree, though — blob hashes are not human-readable.  There we
    prefer the human-named snapshot entry pointing at the blob; if none
    resolves, we create our own symlink under a repo-derived name inside a
    served category directory and put the sidecar next to it.
    """
    if "blobs" not in gguf.parts:
        md = gguf.with_suffix(".md")
        if generate_stubs:
            _write_stub(md)
        return md

    real = os.path.realpath(gguf)
    repo_dir = next((a for a in gguf.parents if a.name.startswith("models--")), None)
    if repo_dir is not None:
        snaps = repo_dir / "snapshots"
        if snaps.is_dir():
            for rev in sorted(snaps.iterdir()):
                if not rev.is_dir():
                    continue
                for entry in sorted(rev.iterdir()):
                    try:
                        hit = os.path.realpath(entry) == real
                    except OSError:
                        continue
                    if hit:
                        md = entry.with_suffix(".md")
                        if generate_stubs:
                            _write_stub(md)
                        return md

        # No snapshot entry resolves: link the blob into a served category
        # directory under a human-readable, repo-derived name.
        rel = gguf.relative_to(root)
        if len(rel.parts) > 1:
            slug = repo_dir.name.removeprefix("models--").replace("--", "-")
            link = root / rel.parts[0] / f"{slug}-{gguf.stem[:8]}{gguf.suffix}"
            try:
                if not link.exists():
                    link.parent.mkdir(parents=True, exist_ok=True)
                    os.symlink(real, link)
                md = link.with_suffix(".md")
                if generate_stubs:
                    _write_stub(md)
                return md
            except OSError as e:
                logger.warning("stub: cannot symlink blob %s into %s (%s)",
                               gguf.name, rel.parts[0], e)

    logger.warning("stub: %s lives in an HF blobs tree with no resolvable "
                   "snapshot entry or category directory; serving without "
                   "an editable sidecar", gguf.name)
    return gguf.with_suffix(".md")
