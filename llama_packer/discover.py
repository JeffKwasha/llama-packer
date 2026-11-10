# llama_packer/discover.py
"""Depth-first model discovery: one walk, deferred orphans, object-owned claims.

Discovery is a preorder DFS over each models dir.  At every level:

1. push that directory's ``models.yaml`` onto the :class:`ScopeStack`
   (its ``defaults`` and ``overrides``);
2. build authored sidecars — every ``.md`` becomes a ``Model`` and registers
   its weight claim in ``Model._by_gguf``;
3. remember candidate weight files (``.gguf``/``.safetensors``) without
   creating stubs (opt-in);
4. recurse into children; pop.

Only after the walk is globally complete are orphans scanned: a remembered
weight is skipped when ``Model.is_claimed(path)`` (realpath + dev:ino, so
an explicit ``model: foo-Q6_K.gguf`` that strips quant/version still claims
the file).  This makes discovery canonically two-phase but a single walk –
the Model owns the claim decision (DRY), discovery only remembers paths.

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

import copy
import logging
import os
from pathlib import Path

from llama_packer import roles, tasks, utils
from llama_packer.endpoints import IMAGES
from llama_packer.endpoints import validate_engines_value
from llama_packer.model import Model
from llama_packer.scope import ScopeStack

logger = logging.getLogger(__name__)


class DiscoveryError(Exception):
    """Models that resolve to no role, or to a role no file defines."""


def discover(
    models_dirs,
    *,
    stack: ScopeStack | None = None,
    generate_stubs: bool = False,
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

    # Global registry: sidecar → weight claims. Cleared per discover() so
    # orphan detection can ask Model.is_claimed(path) after the walk.
    Model.clear_registry()

    models: list[Model] = []
    orphan_candidates: list[tuple[Path, Path, str | None, ScopeStack]] = []
    skipped: set[str] = set()
    problems: list[str] = []
    for root in models_dirs:
        if not root.is_dir():
            continue
        logger.info("scanning models dir: %s", root)
        ignore = utils.load_model_ignore(root)
        _walk(root, root, ignore, stack, hf_home, models,
              orphan_candidates, skipped, problems)

    # Deferred orphan handling: the walk above created sidecars and *remembered*
    # candidate weight files; only now, with all sidecars validated and registered,
    # do we scan for missing sidecars.  Model.is_claimed owns the decision –
    # discovery only remembers paths.
    for gguf, root, role, stack_snapshot in orphan_candidates:
        if Model.is_claimed(gguf):
            continue
        if gguf.with_suffix(".md").is_file():
            continue
        # Defensive: companions should never have been collected, but re-check.
        if utils.companion_kind(gguf.stem):
            continue
        _model_from_orphan(gguf, root, role, stack_snapshot, generate_stubs,
                           hf_home, models, problems)

    if skipped:
        logger.info("skipping %s (no role owns them; give a role that dir:)",
                    ", ".join(f"{d}/" for d in sorted(skipped)))

    if problems:
        raise DiscoveryError("\n".join(problems))

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

    # Enforce unique model_id (template_id) - sidecar stem slug. Duplicates
    # are only allowed when they reference the same sidecar file (reflink/
    # symlink/hardlink to the same inode). Tracks absolute sidecar path.
    # Duplicates of the same physical file are deduplicated (first wins).
    id_to_sidecar: dict[str, str] = {}
    id_to_stat: dict[str, tuple[int, int]] = {}
    filtered: list[Model] = []
    for model in deduped:
        mid = model.template_id
        sidecar_abs = str(model.md_path.resolve())
        try:
            st = os.stat(model.md_path)
            cur_stat = (st.st_dev, st.st_ino)
        except OSError:
            cur_stat = None  # fallback to path string
        prev_path = id_to_sidecar.get(mid)
        if prev_path is None:
            id_to_sidecar[mid] = sidecar_abs
            if cur_stat is not None:
                id_to_stat[mid] = cur_stat
            filtered.append(model)
            continue
        # Same physical sidecar file (reflink/symlink/hardlink) is allowed -
        # deduplicate to first (same as gguf dedup).
        prev_stat = id_to_stat.get(mid)
        if cur_stat is not None and prev_stat is not None and cur_stat == prev_stat:
            logger.info("duplicate model_id %r same sidecar file %s, deduped", mid, sidecar_abs)
            continue
        if sidecar_abs == prev_path:
            logger.info("duplicate model_id %r same sidecar path %s, deduped", mid, sidecar_abs)
            continue
        raise ValueError(
            f"duplicate model_id {mid!r}: sidecar {prev_path!r} collides with "
            f"{sidecar_abs!r} — sidecar stems must be unique (model_id is slug of stem)"
        )
    return filtered


def _walk(d: Path, root: Path, ignore: list[str], stack: ScopeStack,
          hf_home, out: list[Model],
          orphan_candidates: list[tuple[Path, Path, str | None, ScopeStack]],
          skipped: set[str], problems: list[str]) -> None:
    """Preorder DFS: this directory's scope and models first, then children.

    The role is the one whose ``dir:`` owns this directory; a subtree no role
    owns is not descended at all.  This walk *creates sidecars and remembers
    weight files*; orphan stub creation is deferred until all sidecars are
    validated and registered so Model.is_claimed can answer canonically.
    """
    role = roles.owner(d.relative_to(root).parts)
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
                _model_from_sidecar(p, root, role, stack, hf_home, out,
                                    problems)
            elif suffix == ".bin":
                # Whisper GGML weights: served only via an authored same-stem
                # sidecar — never walked as orphans, never stubbed.
                if not p.with_suffix(".md").is_file():
                    logger.info("skipping %s: .bin model without a same-stem "
                                ".md sidecar (write one to serve it)", p.name)
            elif suffix in (".gguf", ".safetensors"):
                if utils.companion_kind(p.stem):
                    continue  # companions join via fuzzy resolution, never walked
                if p.with_suffix(".md").is_file():
                    continue  # exact-stem sidecar owns this file
                # A safetensors weight declares its engine and tasks in a
                # sidecar; guessing from the filename is exactly what this
                # project refuses to do. GGUF orphans still auto-stub.
                if suffix == ".safetensors":
                    continue
                # Remember candidate; Model.is_claimed (global registry) decides later.
                orphan_candidates.append((p, root, role, copy.deepcopy(stack)))
        for child in sorted(d.iterdir()):
            if not child.is_dir():
                continue
            rel = child.relative_to(root)
            if not roles.covers(rel.parts):
                skipped.add(rel.parts[0].lower())
                continue
            _walk(child, root, ignore, stack, hf_home, out,
                  orphan_candidates, skipped, problems)
    finally:
        stack.pop()


def _warn_unsupported_values(model: Model, label: str, catalogue=None) -> None:
    """Warn when a known key carries a value this tool cannot honour.

    Unknown *keys* already warn (``Model.pass_through_metadata``).  An
    unknown *value* used to fall through: ``role: embedder`` only surfaced
    later as "no backend supports", and ``device: gpu`` silently placed the
    model on the device exactly as if the key were absent.
    """
    if "usage" in model.sidecar_keys:
        logger.warning("%s: usage: is not a sidecar key; declare tasks:",
                       label)
    if catalogue is not None:
        for problem in tasks.validate_task_ids(
                model.frontmatter.get("tasks"), catalogue):
            logger.warning("%s: %s", label, problem)
    for problem in validate_engines_value(model.frontmatter.get("engines")):
        logger.warning("%s: %s", label, problem)

    block = model.frontmatter.get("audio_cpp")
    if isinstance(block, dict) and block.get("task"):
        engine_task = str(block["task"])
        implied = {"tts": "t2s", "clon": "t2s", "asr": "s2t"}.get(engine_task)
        claimed = set(model.task_ids) or {model.role}
        if implied is not None and implied not in claimed:
            logger.warning("%s: audio_cpp.task %r is a %s task, but the "
                           "model declares %s", label, engine_task, implied,
                           ", ".join(sorted(claimed)))

    dev = model.frontmatter.get("device")
    if dev is None or model.on_cpu:
        return
    problem = _device_problem(dev)
    if problem:
        logger.warning("%s: device %r is %s; ignored (the model stays on "
                       "the default device)", label, dev, problem)


def _device_problem(dev) -> str | None:
    """Why *dev* is not a usable `device:` value, or None when it is."""
    return utils.parse_device(dev)[1]


def _build(path: Path, frontmatter: dict, role: str | None, stack: ScopeStack,
           hf_home, out: list[Model], problems: list[str],
           stub: bool = False) -> None:
    """The single model pipeline: merge → construct → rules → companions → finalize.

    *stub* marks freshly materialized orphan sidecars: their measured-file
    sync stays memory-only so the stub on disk remains pristine.
    """
    caps = frontmatter.get("capabilities")
    if isinstance(caps, list) and any(
            isinstance(c, str) and c.startswith("-") and len(c) > 1
            for c in caps):
        logger.warning(
            "sidecar %s: capabilities removes inherited entries — sidecars "
            "add capabilities; conditional serving belongs in an mmproj "
            "block", path.name)
    merged = stack.merge_defaults(frontmatter)
    # A model inherits its role from the tree that owns it; a sidecar or
    # directory default may override it.  Anything else is an operator error:
    # there is no default role.
    if role and "role" not in merged:
        merged["role"] = role
    declared = merged.get("role")
    if not isinstance(declared, str) or not declared.strip():
        problems.append(f"{path}: no role owns this model; give a role a "
                        "dir: that covers it or set role: in the sidecar")
        return
    declared = declared.strip()
    if roles.get(declared) is None:
        problems.append(f"{path}: role {declared!r} is not defined in "
                        f"profiles.yaml roles: "
                        f"({', '.join(roles.names()) or 'none'})")
        return
    merged["role"] = declared
    try:
        model = Model(path, merged, hf_home=hf_home)
    except Exception as e:
        logger.error("failed to load model from %s: %s", path.name, e)
        return
    model.stub = stub
    catalogue = stack.tasks
    spans = {t.endpoint for t in ((catalogue.get(tid) if catalogue else None)
                                  for tid in model.task_ids)
             if t is not None}
    if len(spans) > 1:
        problems.append(
            f"{path}: tasks span endpoints ({', '.join(sorted(spans))}); "
            "one serving cannot serve different endpoints")
        return
    # Authored keys only: the directory-derived role and the role finalize
    # derives from tasks: are builder values, not sidecar typos.
    model.sidecar_keys = frozenset(frontmatter)
    stack.apply_rules(model)
    stack.annotate_tasks(model)
    model.resolve_companions()
    model.sync_measured_file(write=not stub)
    stack.finalize(model)
    _warn_unsupported_values(model, path.name, stack.tasks)
    # Guard: keep generative-media weights out of served text roles.  The
    # classification is header-only (GGUF metadata / safetensors names /
    # cached HF model card) and never blocks the run — one error log per
    # offending file, then the model is excluded from the config.
    # The image endpoint is exempt — diffusion weights are expected there.
    if model.file_kind == "image" and not model.serves_endpoint(IMAGES):
        logger.error(
            "%s: diffusion/image weights classified in a role without an "
            "image task (header check); excluded. Move under a role whose "
            "models serve image, set `ignore: true`, or set role: in the "
            "sidecar", path.name)
        return
    # A model serving the image endpoint should carry diffusion/image
    # weights — skip textual-inversion embeddings, loras and other assets.
    if model.serves_endpoint(IMAGES) and model.file_kind != "image":
        logger.info("skipping %s: not diffusion/image weights in an image "
                    "role (kind=%s)", path.name, model.file_kind)
        return
    out.append(model)


def _model_from_sidecar(md_path: Path, root: Path, role: str | None,
                         stack: ScopeStack, hf_home, out: list[Model],
                         problems: list[str]) -> None:
    # Never serve sidecars that live inside the HF hub cache hierarchy
    # (snapshots contain README.md plus any stray .md a user may have dropped).
    try:
        hf_root_path = utils.hf_cache_root(hf_home)
        hf_root = str(hf_root_path) if hf_root_path is not None else None
        if hf_root:
            # Resolve without following mount-skipping semantics - a plain
            # realpath/absolute check is sufficient to recognise HF tree.
            abs_md = str(md_path.resolve())
            if abs_md == hf_root or abs_md.startswith(hf_root + os.sep):
                logger.debug("ignore hf-tree sidecar: %s", md_path)
                return
    except Exception:
        pass
    fm = utils.parse_frontmatter(md_path)
    if not fm and not any(md_path.with_suffix(e).is_file()
                          for e in (".gguf", ".safetensors", ".bin", ".onnx")):
        return  # no data and no model beside it: not a sidecar (README, ...)
    if fm.get("ignore"):
        logger.info("ignore: skipping %s", md_path.name)
        return
    _build(md_path, fm, role, stack, hf_home, out, problems)


def _model_from_orphan(gguf: Path, root: Path, role: str | None,
                       stack: ScopeStack, generate_stubs: bool,
                       hf_home, out: list[Model],
                       problems: list[str]) -> None:
    """Serve an orphan model file; materialize its empty editable sidecar.

    Stubs carry no data — identity falls back to the stem, context to the
    built-in default, role to the directory — so a stub and an authored
    sidecar behave identically through the pipeline.
    """
    md_path = _materialize_sidecar(gguf, root, generate_stubs)
    _build(md_path, {}, role, stack, hf_home, out, problems, stub=True)


def _write_stub(md_path: Path) -> None:
    if md_path.exists():
        return
    if utils.DRY_RUN:
        utils.note_fs_op("create stub sidecar", md_path)
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
