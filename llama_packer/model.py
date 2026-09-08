# llama_packer/model.py
"""Model class representing a language model with companions and VRAM calculations."""

from __future__ import annotations

import copy
import fnmatch
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import yaml

from llama_packer import utils
from llama_packer.consts import (
    _DEFAULT_CONTEXT_LENGTH,
    _DIFFUSION_ARCH_RES,
    _MTP_DRAFT_N_MAX,
)
from llama_packer.backends import DEFAULT_BACKEND

if TYPE_CHECKING:
    from llama_packer.vram import VramBudget

logger = logging.getLogger(__name__)

# Frontmatter keys that are computed/derived rather than declared
_CL_RE = re.compile(r"^context_limit_\d+G$")

# The single dynamically generated sidecar branch. Machine-measured values
# (fit-params VRAM numbers, file intrinsics) live here and only here;
# everything else in the sidecar is authored. ``fit-params`` is the legacy
# name, still read as a fallback but never written.
MEASURED_KEY = "measured"
_LEGACY_MEASURED_KEY = "fit-params"


class WeightFinder:
    """Resolves weight-file references against local dirs + the HF hub cache.

    The small dir-level helper behind :meth:`Model.from_ref`: exact lookups
    in anchor directories, snapshot selection, and snapshot filename
    matching. Snapshot listings are cached per directory mtime, so N models
    sharing one snapshot readdir it once. Tests replace this with a stub —
    callers take an optional ``finder`` parameter defaulting to the global
    instance rather than touching the filesystem or ``utils`` directly.
    """

    def __init__(self) -> None:
        self._snap_files: dict[str, tuple[int, list[str]]] = {}

    def find_local(self, name: str, dirs: list[Path]) -> Path | None:
        """Exact *name* inside the first anchor dir holding it."""
        for d in dirs:
            candidate = d / name
            try:
                if candidate.is_file():
                    return candidate
            except OSError:
                continue
        return None

    def snapshot(self, repo: str, hf_home=None) -> Path | None:
        """Snapshot dir for *repo*, or None when not cached locally."""
        return utils.hf_snapshot_dir(repo, hf_home)

    def snapshot_files(self, repo: str, hf_home=None) -> tuple[Path | None, list[str]]:
        """(snapshot dir, sorted file names), listing cached by dir mtime."""
        snap = self.snapshot(repo, hf_home)
        if snap is None:
            return None, []
        try:
            mtime = os.stat(snap).st_mtime_ns
        except OSError:
            return snap, []
        hit = self._snap_files.get(str(snap))
        if hit is not None and hit[0] == mtime:
            return snap, hit[1]
        try:
            names = sorted(p.name for p in snap.iterdir() if p.is_file())
        except OSError:
            return snap, []
        self._snap_files[str(snap)] = (mtime, names)
        return snap, names

    def snapshot_exact(self, repo: str, name: str, hf_home=None) -> Path | None:
        """Exact *name* inside the repo snapshot, or None."""
        snap, names = self.snapshot_files(repo, hf_home)
        if snap is None or name not in names:
            return None
        candidate = snap / name
        try:
            return candidate if candidate.is_file() else None
        except OSError:
            return None

    def match_snapshot(self, repo: str, pattern: str, hf_home=None) -> list[Path]:
        """Snapshot files matching glob *pattern* (sorted)."""
        snap, names = self.snapshot_files(repo, hf_home)
        if snap is None:
            return []
        return [snap / n for n in names if fnmatch.fnmatchcase(n, pattern)]


_DEFAULT_FINDER: WeightFinder | None = None


def default_finder() -> WeightFinder:
    """Global WeightFinder (overridable per-call via ``finder=``)."""
    global _DEFAULT_FINDER
    if _DEFAULT_FINDER is None:
        _DEFAULT_FINDER = WeightFinder()
    return _DEFAULT_FINDER


class Model:
    """Represents a main model with optional mmproj and MTP companions.

    File identity lives here: every weight file has one canonical
    sidecar-less instance (see :meth:`from_file`, shared by all claimants)
    that parses its header once, lazily, into small scalars. Sidecar-bound
    instances delegate file-derived properties to it via ``_file``.
    """

    md_path: Path | None  # None on canonical file instances (no sidecar)
    _stat_sig: tuple[int, int] | None = None  # (size, mtime_ns) at registration
    _file: Model | None = None  # canonical file delegate (self when sidecar-less)
    _header: object = None  # lazy parsed header (subclass fills on demand)

    # Global registry: claimed weight files → owning Model. The key is the
    # weight's realpath (and dev:ino for hardlinks) so a sidecar with an
    # explicit `model:` that strips quant/version still claims the file.
    # `_by_name` indexes every registered instance by weight basename so
    # `Model["file.gguf"]` (or a glob pattern) finds known models. The
    # registry is cleared at the start of each discover() call; all
    # discovery-time Model construction registers automatically.
    _by_gguf: ClassVar[dict[str, "Model"]] = {}
    _by_gguf_stat: ClassVar[dict[str, "Model"]] = {}
    _by_md: ClassVar[dict[str, "Model"]] = {}
    _by_name: ClassVar[dict[str, list["Model"]]] = {}

    # Weight-file suffix → Model subclass chosen transparently by
    # from_file(). Tests register zero-IO fakes via register_file_class().
    _file_classes: ClassVar[dict[str, type["Model"]]] = {}

    @classmethod
    def clear_registry(cls) -> None:
        cls._by_gguf.clear()
        cls._by_gguf_stat.clear()
        cls._by_md.clear()
        cls._by_name.clear()

    @classmethod
    def _register(cls, m: "Model") -> None:
        if m.gguf_path is not None:
            try:
                real = os.path.realpath(str(m.gguf_path))
                cls._by_gguf.setdefault(real, m)
                try:
                    st = os.stat(str(m.gguf_path))
                    stat_key = f"{real}|{st.st_dev}:{st.st_ino}"
                    cls._by_gguf_stat.setdefault(stat_key, m)
                except OSError:
                    pass
                name = os.path.basename(real)
                bucket = cls._by_name.setdefault(name, [])
                if m not in bucket:
                    bucket.append(m)
            except OSError:
                pass
        if m.md_path is not None:
            try:
                md_real = os.path.realpath(str(m.md_path))
                cls._by_md.setdefault(md_real, m)
            except OSError:
                pass

    @classmethod
    def _register_canonical(cls, m: "Model", real: str, stat_key: str,
                            old: "Model | None" = None) -> None:
        """Register a from_file() instance, replacing *old* when refreshed."""
        if old is not None and old is not m:
            if cls._by_gguf.get(real) is old:
                del cls._by_gguf[real]
            for key, inst in list(cls._by_gguf_stat.items()):
                if inst is old:
                    del cls._by_gguf_stat[key]
            for bucket in cls._by_name.values():
                if old in bucket:
                    bucket.remove(old)
        cls._by_gguf.setdefault(real, m)
        cls._by_gguf_stat[stat_key] = m
        name = os.path.basename(real)
        bucket = cls._by_name.setdefault(name, [])
        if m not in bucket:
            bucket.append(m)

    def __class_getitem__(cls, key: str) -> list["Model"]:
        """Find known models by filename: ``Model["qwen3-8b.gguf"]``.

        An exact basename (or stem) returns every registered instance for
        that file; anything else is treated as a glob matched (fnmatch)
        against basenames, stems, and template ids. Always a list —
        possibly empty. Matching is over *registered* models only; use
        :meth:`from_ref` to resolve (and build) unregistered files.
        """
        if not isinstance(key, str) or not key:
            return []
        seen: list[Model] = []
        seen_files: set[str] = set()

        def _claim(m: Model) -> None:
            if m in seen:
                return
            # One entry per weight file: the sidecar-bound claimant (which
            # registers first) shadows its canonical file instance.
            if m.gguf_path is not None:
                try:
                    ident = os.path.realpath(str(m.gguf_path))
                except OSError:
                    ident = ""
                if ident and ident in seen_files:
                    return
                if ident:
                    seen_files.add(ident)
            seen.append(m)

        bucket = cls._by_name.get(key)
        if bucket:
            for m in bucket:
                _claim(m)
            return seen
        for models in cls._by_name.values():
            for m in models:
                if m.stem == key:
                    _claim(m)
        if seen:
            return seen
        for models in cls._by_name.values():
            for m in models:
                try:
                    tid = m.template_id
                except ValueError:
                    tid = ""
                name = os.path.basename(os.path.realpath(str(m.gguf_path))) \
                    if m.gguf_path else ""
                if (fnmatch.fnmatchcase(name, key)
                        or fnmatch.fnmatchcase(m.stem, key)
                        or (tid and fnmatch.fnmatchcase(tid, key))):
                    _claim(m)
        return seen

    @classmethod
    def register_file_class(cls, suffix: str, sub: type["Model"]) -> None:
        """Choose *sub* for weight files ending in *suffix* (test seam).

        Lets tests present a zero-IO backend: register a fake subclass with
        canned header values and from_file() builds it instead of touching
        disk. Suffixes are matched lowercase, including the dot.
        """
        cls._file_classes[suffix.lower()] = sub

    @classmethod
    def _file_class_for(cls, path: Path) -> type["Model"]:
        """Transparent subclass choice: GGUF vs safetensors vs unknown."""
        sub = cls._file_classes.get(path.suffix.lower())
        if sub is not None:
            return sub
        # Late import: subclasses are defined below Model in this module.
        if path.suffix.lower() == ".gguf":
            return GGUFModel
        if path.suffix.lower() == ".safetensors":
            return SafetensorsModel
        return Model

    @classmethod
    def from_file(cls, path: str | os.PathLike, *,
                  finder: WeightFinder | None = None) -> "Model | None":
        """Canonical Model for a weight file (built once, then shared).

        The registry key is realpath + dev:ino, validated against current
        size+mtime — a changed file transparently replaces a stale instance.
        No header I/O happens here; subclasses parse lazily on first
        property access, so one physical read serves every claimant.
        Returns None when the file does not exist.
        """
        _ = finder  # reserved: finder-scoped registries if ever needed
        p = utils.smart_resolve(Path(path))
        try:
            st = os.stat(p)
        except OSError:
            return None
        real = os.path.realpath(p)
        sig = (st.st_size, st.st_mtime_ns)
        stat_key = f"{real}|{st.st_dev}:{st.st_ino}"
        sub = cls._file_class_for(p)
        hit = cls._by_gguf_stat.get(stat_key)
        if hit is None:
            hit = cls._by_gguf.get(real)
        if (hit is not None and type(hit) is sub
                and getattr(hit, "_stat_sig", None) == sig):
            return hit
        inst = sub._new_from_file(p, sig)
        cls._register_canonical(inst, real, stat_key,
                                old=hit if hit is not None else None)
        return inst

    @classmethod
    def _new_from_file(cls, path: Path, sig: tuple[int, int]) -> "Model":
        """Build a sidecar-less canonical instance (no I/O, no resolution)."""
        inst = cls.__new__(cls)
        inst.md_path = None
        inst.frontmatter = {}
        inst.stem = path.stem
        inst._hf_home = None
        inst.gguf_path = path
        inst._stat_sig = sig
        inst._vram = None
        inst.mmproj = None
        inst.mtp = None
        inst.mmproj_overlay = {}
        inst._view_on = None
        inst._is_view = False
        inst._override_error = None
        inst._file = inst  # canonical: a file model is its own delegate
        inst._header = None  # lazy parsed header (subclass fills on demand)
        return inst

    @classmethod
    def from_ref(cls, ref: str, *, anchors: list[Path] | None = None,
                 hf_repo: str | None = None, hf_home=None,
                 finder: WeightFinder | None = None) -> "Model | None":
        """Resolve a sidecar ``model:``/companion string to a Model.

        Absolute or anchor-relative paths, bare filenames in anchor dirs,
        ``hub:<org>/<repo>:<file>`` forms, snapshot-exact names via
        *hf_repo*, and globs (single match wins; ambiguous warns like the
        legacy resolver). Returns the canonical registered instance, or
        None when unresolvable. Directory scanning goes through *finder*
        (mockable in tests).
        """
        f = finder or default_finder()
        anchors = list(anchors or [])
        repo = hf_repo
        name = ref
        if ref.startswith("hub:"):
            rest = ref[len("hub:"):]
            repo, _, name = rest.rpartition(":")
            if not repo or not name:
                logger.warning("hf: malformed %r (expected hub:org/repo:file)", ref)
                return None
        candidate = Path(name)
        if candidate.is_absolute():
            return cls.from_file(candidate) if candidate.is_file() else None
        hit = f.find_local(name, anchors)
        if hit is not None:
            return cls.from_file(hit)
        if repo:
            if any(ch in name for ch in "*?["):
                matches = f.match_snapshot(repo, name, hf_home)
                if len(matches) == 1:
                    return cls.from_file(matches[0])
                if len(matches) > 1:
                    logger.warning("hf: %s in %s is ambiguous (%d matches): %s",
                                   name, repo, len(matches),
                                   ", ".join(m.name for m in matches))
                return None
            hit = f.snapshot_exact(repo, name, hf_home)
            if hit is not None:
                return cls.from_file(hit)
        return None

    @classmethod
    def is_claimed(cls, weight_path: Path) -> bool:
        """True if any registered Model already claims *weight_path*."""
        import os
        try:
            real = os.path.realpath(str(weight_path))
            if real in cls._by_gguf:
                return True
            try:
                st = os.stat(str(weight_path))
                stat_key = f"{real}|{st.st_dev}:{st.st_ino}"
                if stat_key in cls._by_gguf_stat:
                    return True
            except OSError:
                pass
        except OSError:
            return False
        return False

    @classmethod
    def find_by_gguf(cls, weight_path: Path) -> "Model | None":
        import os
        try:
            real = os.path.realpath(str(weight_path))
            m = cls._by_gguf.get(real)
            if m is not None:
                return m
            try:
                st = os.stat(str(weight_path))
                stat_key = f"{real}|{st.st_dev}:{st.st_ino}"
                return cls._by_gguf_stat.get(stat_key)
            except OSError:
                return None
        except OSError:
            return None

    # Frontmatter keys this Model consumes (not passed through to metadata)
    FIELDS: ClassVar[frozenset[str]] = frozenset({
        "name", "model_id", "id", "context_length", "description", "cli_args", "model",
        "backend", "hf_repo", "chat_template", "chat_template_kwargs", "loras",
        "attention", "kv_cache", "tool_args", "speculative", "mmproj",
        "mtp", "mtp_spec_type", "mtp_draft_n_max", "mtp_draft_p_min",
        "speculative_config",
        "role", "targets", "allow_profiles", "spare", "capabilities",
        "ignore", "device", "concurrency", "measured", "fit-params", "vllm_image",
        "modes", "default_mode", "reasoning-format", "reasoning-preserve",
        "cache_type", "parallel",
        "image_min_tokens", "image_max_tokens",

    })

    # Frontmatter keys a companion block may NOT set: identity, placement,
    # and backend selection stay model-level (planner gates and the matrix
    # solve key off them).  Everything serving-conditional belongs in the
    # block.
    COMPANION_BLOCK_DENIED: ClassVar[frozenset[str]] = frozenset({
        "name", "model", "ignore", "mmproj", "hf_repo", "backend", "role",
    })

    # Known pass-through metadata keys (documented in models_AGENTS.md)
    # Any frontmatter key not in FIELDS and not in this set triggers a warning
    # but still flows through as metadata (elegance over backwards compat).
    KNOWN_METADATA: ClassVar[frozenset[str]] = frozenset({
        "parameters", "quantization", "hf_url", "license", "base_model",
        "architecture", "family", "finetune", "type", "mtp_accuracy",
        "strengths", "weaknesses", "freethought",
        # per-model calibration / quality metrics (kept for now, low signal)
        "quant_layout", "calibration_tokens", "top1_agreement_vs_bf16", "kld_vs_bf16",
    })

    def __init__(self, md_path: Path, frontmatter: dict, hf_home=None,
                 finder: WeightFinder | None = None):
        self.md_path = md_path
        self.frontmatter = frontmatter
        self.stem = md_path.stem
        self._hf_home = hf_home  # HF cache root override for hub snapshot resolution
        self._finder = finder or default_finder()
        self._vram: VramBudget | None = None  # lazy VRAM budget calculator
        self._file: Model | None = None  # canonical file delegate (bound below)

        # Resolve the model file path. hf_repo is a *place* (hub repo)
        # where a file lives, not a file itself. Most sidecars resolve to a
        # concrete file (same-stem, explicit model:, or single non-mmproj file
        # in the snapshot). hf_repo-only (no local file) is allowed for
        # backends that serve directly from a repo id (vLLM safetensors,
        # kokoro-podman which is image-baked). Every other case needs a file.
        self.gguf_path = self._resolve_gguf_path()
        if not self.gguf_path and self.hf_repo is None:
            tried_local = ", ".join(
                f"{self.stem}{ext}" for ext in (".gguf", ".safetensors", ".bin", ".onnx")
            )
            raise ValueError(
                f"sidecar {md_path.name} (no hf_repo): no model file found. "
                f"Tried same-stem {tried_local} beside sidecar ({self.md_path.parent}). "
                f"Fix: add `model: <filename>` (local file or snapshot file) + `hf_repo: org/repo` if in hub, "
                f"or place {self.stem}.gguf (etc.) beside the sidecar, or `ignore: true`."
            )
        if not self.gguf_path and self.frontmatter.get("model"):
            tried_local = ", ".join(
                f"{self.stem}{ext}" for ext in (".gguf", ".safetensors", ".bin", ".onnx")
            )
            snap = None
            snap_listing = ""
            if self.hf_repo:
                try:
                    snap, files = self._finder.snapshot_files(
                        self.hf_repo, self._hf_home)
                    if snap is not None and files:
                        snap_listing = f" – snapshot {snap} contains: {', '.join(files[:12])}"
                        if len(files) > 12:
                            snap_listing += f" (+{len(files)-12} more)"
                    else:
                        snap_listing = f" – snapshot for {self.hf_repo!r} not found (hf download {self.hf_repo})"
                except Exception:
                    pass
            raise ValueError(
                f"sidecar {md_path.name} (hf_repo {self.hf_repo!r}): explicit `model: {self.frontmatter['model']!r}` not found. "
                f"Tried beside sidecar ({self.md_path.parent}) and in hub snapshot{snap_listing}. "
                f"Fix: set `model: <exact filename in snapshot>` (run `ls {snap or '$HF_HOME/hub/models--…/snapshots/...'}`) "
                f"or place same-stem file {tried_local} beside the sidecar."
            )
        if self.gguf_path is not None:
            self._file = Model.from_file(self.gguf_path)

        # Resolve companions later — resolve_companions() is called by
        # discovery after scope defaults and override rules have had their
        # say (frontmatter must be final before companion resolution).
        self.mmproj: Model | None = None
        self.mtp: Model | None = None
        # Overlay keys from an mmproj mapping block (everything but `file`),
        # merged over the frontmatter when the companion is served.
        self.mmproj_overlay: dict = {}
        self._view_on: Model | None = None  # cached companion-on view
        self._is_view: bool = False  # True on copies returned by view_for
        # Register this sidecar↔weight claim globally so orphan detection can
        # ask Model.is_claimed(path) after the walk is complete.
        self.__class__._register(self)

    def resolve_companions(self) -> None:
        """Resolve mmproj and MTP companions from final frontmatter.

        Idempotent: re-running discards previously resolved companions and
        re-derives both from the current frontmatter.  Callers must invoke
        this once frontmatter is final (defaults merged, rules applied) —
        see llama_packer.discover.
        """
        self.mmproj = None
        self.mtp = None
        self.mmproj_overlay = {}
        self._view_on = None
        if not self.gguf_path:
            logger.debug("no gguf_path, skipping companion resolution for %s", self.stem)
            return

        assert self.gguf_path is not None

        # Companion search is anchored to the *model file's* parent directory only
        # (the snapshot or local dir). The sidecar's parent is not searched for
        # fuzzy fallbacks - explicit mmproj:/speculative: may still resolve via
        # _resolve_ref/_resolve_hub_ref (parent, parent.parent, hub).
        search_dirs = [self.gguf_path.parent]

        # --- mmproj ---
        mmproj_val = self.frontmatter.get("mmproj")
        if mmproj_val is False:
            # Explicit disable
            self.mmproj = None
        elif isinstance(mmproj_val, dict):
            # Mapping block: `file` locates the companion; every other key
            # is a conditional overlay merged over the frontmatter when the
            # companion is served (same single merge rule as every layer).
            block = dict(mmproj_val)
            file_val = block.pop("file", None)
            denied = [k for k in block if k in self.COMPANION_BLOCK_DENIED]
            unknown = [k for k in block if k not in self.FIELDS]
            if not isinstance(file_val, str) or not file_val:
                msg = (f"sidecar {self.label}: mmproj block needs "
                       f"a `file:` string")
                logger.error("%s", msg)
                self._override_error = msg
            elif denied:
                msg = (f"sidecar {self.label}: mmproj block may not set "
                       f"{', '.join(sorted(denied))} (model-level keys)")
                logger.error("%s", msg)
                self._override_error = msg
            elif unknown:
                intrinsic = sorted(k for k in unknown
                                   if k in self.KNOWN_METADATA)
                if intrinsic:
                    msg = (f"sidecar {self.label}: mmproj block may "
                           f"not set {', '.join(intrinsic)} (intrinsic model "
                           f"properties belong in the sidecar)")
                else:
                    msg = (f"sidecar {self.label}: unknown mmproj "
                           f"block key(s): {', '.join(sorted(unknown))}")
                logger.error("%s", msg)
                self._override_error = msg
            else:
                companion = Model.from_ref(
                    str(file_val), anchors=search_dirs,
                    hf_repo=self.hf_repo, hf_home=self._hf_home,
                    finder=self._finder)
                if companion is None and self.hf_repo:
                    companion = Model.from_ref(
                        "*mmproj*.gguf", anchors=[],
                        hf_repo=self.hf_repo, hf_home=self._hf_home,
                        finder=self._finder)
                if companion:
                    self.mmproj = companion
                    self.mmproj_overlay = block
                else:
                    logger.warning("mmproj: configured %s missing for %s",
                                   file_val, self.stem)
        elif isinstance(mmproj_val, str):
            msg = (f"sidecar {self.label}: mmproj must be a mapping "
                   f"(`mmproj: {{file: ..., ...}}`) or false — bare filenames "
                   f"carry no purpose; move conditional keys into the block")
            logger.error("%s", msg)
            self._override_error = msg
            self.mmproj = None
        elif mmproj_val:
            msg = (f"sidecar {self.label}: mmproj must be a mapping "
                   f"(`mmproj: {{file: ..., ...}}`) or false, got "
                   f"{mmproj_val!r}")
            logger.error("%s", msg)
            self._override_error = msg
            self.mmproj = None
        else:
            # No explicit mmproj — local models require explicit mmproj: (or
            # mmproj: false to silence). Only HF snapshots auto-discover, and
            # only when the gguf lives inside its own snapshot (every file in
            # that snapshot belongs to the same release). Matching rule:
            # mmproj_stripped (stem without 'mmproj.*' + quant/version) must be
            # prefix of model stem.
            # Rerank/embeddings (and image/s2t/t2s) never auto-attach vision.
            if self.hf_repo and self.role not in utils.NON_CHAT_ROLES:
                snap, _ = self._finder.snapshot_files(
                    self.hf_repo, self._hf_home)
                # Auto only when model is inside its snapshot directory
                if snap is not None and self.gguf_path.parent == snap:
                    def _strip_mmproj(stem: str) -> str:
                        base = re.split(r"[-_.]?mmproj.*", stem, flags=re.I)[0]
                        base = base.rstrip("-_.")
                        # Strip quant/version like _gguf_family but keep base
                        return utils._gguf_family(base).lower() if base else ""

                    def _is_preferred_mmproj(name: str) -> bool:
                        return bool(re.search(r"mmproj-(?:bf|fp|f)16\.gguf$", name, re.I))

                    model_stem_lc = self.gguf_path.stem.lower()
                    candidates: list[Path] = []
                    for f in self._finder.match_snapshot(
                            self.hf_repo, "*mmproj*.gguf", self._hf_home):
                        stripped = _strip_mmproj(f.stem)
                        if not stripped:
                            continue
                        if not model_stem_lc.startswith(stripped):
                            continue
                        candidates.append(f)
                    if candidates:
                        preferred = [c for c in candidates if _is_preferred_mmproj(c.name)]
                        chosen = sorted(preferred or candidates)[0]
                        self.mmproj = Model.from_file(chosen)
                    else:
                        logger.debug("mmproj: no snapshot mmproj prefix for %s in %s",
                                     self.stem, snap)

        # --- MTP (speculative) ---
        # Check frontmatter flags
        has_mtp = self.frontmatter.get("mtp")
        speculative = self.frontmatter.get("speculative")
        if has_mtp or (speculative and "mtp" in Path(speculative).stem.lower()):
            # Baked-in MTP or companion MTP
            if speculative:
                companion = Model.from_ref(
                    str(speculative), anchors=search_dirs,
                    hf_repo=self.hf_repo, hf_home=self._hf_home,
                    finder=self._finder)
                if companion:
                    self.mtp = companion
                else:
                    logger.warning("mtp: companion %s missing for %s", speculative, self.stem)
            else:
                # Baked-in MTP (no separate file)
                self.mtp = None  # baked-in, no separate file

        logger.info("model: %s (gguf=%s, mmproj=%s, mtp=%s)",
                    self.stem, self.gguf_path.name if self.gguf_path else self.hf_repo,
                    self.mmproj.stem if self.mmproj else "none",
                    self.mtp.stem if self.mtp else "none")

    def _resolve_gguf_path(self) -> Path | None:
        """Resolve the main model file (.gguf or .safetensors): frontmatter
        ``model:`` field (local dir, then the HF hub cache via ``hf_repo``),
        then the same-stem convention, then (if ``hf_repo`` is set) a single
        non-mmproj model inside the snapshot.

        ``hf_repo``/``hf_url`` is a *place* (hub repo) that can hold many
        files – it is not itself a model file. The actual model is the GGUF/
        safetensors / .bin resolved here. If the repo holds exactly one
        non-mmproj model file, it is used; if several, the sidecar must
        disambiguate with ``model: <filename>``.
        """
        assert self.md_path is not None  # sidecar-bound resolution only
        parent = self.md_path.parent
        anchors = [parent, parent.parent]

        # 1. Check frontmatter `model` field
        file_ref = self.frontmatter.get("model")
        if file_ref:
            hit = Model.from_ref(str(file_ref), anchors=anchors,
                                 hf_repo=self.hf_repo, hf_home=self._hf_home,
                                 finder=self._finder)
            if hit is not None and hit.gguf_path is not None:
                return hit.gguf_path
            # Explicit model: field but file not found is an error – list how.
            # Keep returning None so __init__ can raise with full guidance;
            # the message there explains the tried locations.
            return None

        # 2. Convention: same stem, .gguf / .safetensors / whisper GGML .bin /
        # kokoro ONNX (.bin and .onnx resolve only for their audio roles —
        # discovery requires the s2t/t2s directory)
        for ext in (".gguf", ".safetensors", ".bin", ".onnx"):
            hit = Model.from_ref(f"{self.stem}{ext}", anchors=[parent],
                                 hf_home=self._hf_home, finder=self._finder)
            if hit is not None and hit.gguf_path is not None:
                return hit.gguf_path

        # 3. No local file – try hf_repo snapshot auto (exactly one non-mmproj
        # model → use it, several → error, none → give up for __init__ error)
        if self.hf_repo:
            snap, names = self._finder.snapshot_files(
                self.hf_repo, self._hf_home)
            if snap is not None:
                # Collect non-mmproj candidates (mmproj/mtp are companions, not
                # main models). Also skip .msgpack etc – only real weight files.
                candidates = [
                    n for n in names
                    if Path(n).suffix.lower() in
                    {".gguf", ".safetensors", ".bin", ".onnx"}
                    and "mmproj" not in Path(n).stem.lower()
                ]
                if len(candidates) == 1:
                    hit = Model.from_file(snap / candidates[0])
                    if hit is not None:
                        return hit.gguf_path
                elif len(candidates) > 1:
                    names_s = ", ".join(sorted(candidates))
                    raise ValueError(
                        f"sidecar {self.label} (hf_repo {self.hf_repo!r}): "
                        f"several models in snapshot {snap}: {names_s} – "
                        f"set `model: <filename>` in the sidecar to choose one "
                        f"(exact file in the snapshot, e.g. `ls {snap}`)"
                    )
                # zero candidates – fall through to __init__ error (no file)

        # 4. No model file found by any method
        return None

    def _resolve_ref(self, ref: str) -> Path | None:
        """Resolve a file reference: sidecar dir, its parent, then HF hub.

        Thin wrapper over :meth:`from_ref` (kept for compatibility).
        Returns an absolute path or None.
        """
        assert self.md_path is not None  # sidecar-bound resolution only
        parent = self.md_path.parent
        hit = Model.from_ref(ref, anchors=[parent, parent.parent],
                             hf_repo=self.hf_repo, hf_home=self._hf_home)
        return hit.gguf_path if hit is not None else None

    def _resolve_hub_ref(self, ref: str,
                         pattern_hint: str | None = None) -> Path | None:
        """Hub-aware resolution of *ref* with *pattern_hint* glob fallback.

        Thin wrapper over :meth:`from_ref` (kept for compatibility).
        """
        hit = Model.from_ref(ref, anchors=[],
                             hf_repo=self.hf_repo, hf_home=self._hf_home)
        if hit is None and pattern_hint and self.hf_repo:
            hit = Model.from_ref(pattern_hint, anchors=[],
                                 hf_repo=self.hf_repo, hf_home=self._hf_home)
        return hit.gguf_path if hit is not None else None

    @staticmethod
    def _get_or_create_companion(path: Path) -> Model:
        """Canonical companion Model for mmproj/mtp files (from_file alias)."""
        inst = Model.from_file(path)
        assert inst is not None  # callers pass existing files
        return inst

    @classmethod
    def from_dir(
        cls,
        models_dirs,
        *,
        generate_stubs: bool = True,
        extra_dirs: list[str] | None = None,
        dir_roles: dict | None = None,
        hf_home=None,
        stack=None,
    ) -> list[Model]:
        """Discover all models across *models_dirs* (thin delegate).

        The real work lives in :func:`llama_packer.discover.discover` — the
        depth-first walk that layers scope defaults, applies override rules,
        materializes empty stub sidecars, and resolves companions once the
        frontmatter is final.  ``stack`` is an optional
        :class:`llama_packer.scope.ScopeStack` carrying global rules; when
        omitted, only directory-scoped configuration applies.
        """
        from llama_packer.discover import discover
        return discover(models_dirs, stack=stack,
                        generate_stubs=generate_stubs, extra_dirs=extra_dirs,
                        dir_roles=dir_roles, hf_home=hf_home)

    @property
    def vram(self) -> VramBudget:
        """Lazy-initialized VRAM budget calculator (see llama_packer.vram)."""
        if self._vram is None:
            from llama_packer.vram import VramBudget
            self._vram = VramBudget(self)
        return self._vram

    def write_md(self, output_path: Path | None = None) -> None:
        """Serialize frontmatter back to .md sidecar.

        Writes ALL frontmatter (both builder-consumed FIELDS and pass-through
        metadata) — not just FIELDS — so agent-written metadata is preserved.
        """
        path = output_path or self.md_path
        assert path is not None  # sidecar-bound only; file models have none
        content = "---\n" + yaml.dump(self.frontmatter, sort_keys=False).rstrip() + "\n---\n"
        path.write_text(content, encoding="utf-8")
        try:
            path.chmod(0o644)
        except OSError:
            pass
        logger.info("wrote sidecar: %s", path.name)

    @property
    def context_length(self) -> int:
        return self.frontmatter.get("context_length", _DEFAULT_CONTEXT_LENGTH)

    @property
    def design_context(self) -> int:
        """Architectural context limit (GGUF) > sidecar context_length > default.

        Single source of truth for a model's effective context ceiling, used by
        the VRAM budget and matrix solver alike.
        """
        arch = self.gguf_context_length
        if arch is not None and arch > 0:
            return arch
        return int(self.frontmatter.get(
            "context_length", _DEFAULT_CONTEXT_LENGTH
        ))

    @property
    def gguf_context_length(self) -> int | None:
        """Architectural context limit from the weight header (GGUF metadata).

        Delegates to the canonical file instance, which parses once per
        process however many sidecars claim the file. None when unavailable
        (safetensors, missing file, parse error).
        """
        if self._file is not None:
            return self._file.max_context
        return None

    # ── file interface (parsed header scalars) ──
    #
    # Sidecar-bound instances forward to the canonical file instance
    # (``_file``), which parses its header once per process however many
    # sidecars claim it. Canonical instances are their own delegate
    # (``_file is self``); subclasses fill values lazily on first access.

    def _file_or_self(self) -> "Model":
        """Canonical file instance, or self when sidecar-less/unknown."""
        if self._file is not None and self._file is not self:
            return self._file
        return self

    @property
    def arch(self) -> str | None:
        """Architecture id from the weight header (e.g. ``qwen3``)."""
        target = self._file_or_self()
        if target is not self:
            return target.arch
        return None

    @property
    def max_context(self) -> int | None:
        """Architectural context limit from the weight header, if any."""
        target = self._file_or_self()
        if target is not self:
            return target.max_context
        return None

    @property
    def kind(self) -> str:
        """Header-only kind: ``"text"``, ``"image"``, or ``"unknown"``."""
        target = self._file_or_self()
        if target is not self:
            return target.kind
        return "unknown"

    def safetensors_numbers(self, cache_type: str = "q8_0") -> tuple[int, float] | None:
        """(weight_mib, kv_bytes_per_token) for safetensors files, else None.

        KV bytes scale with *cache_type*; subclasses memoize per type.
        """
        target = self._file_or_self()
        if target is not self:
            return target.safetensors_numbers(cache_type)
        return None

    @property
    def file_kind(self) -> str:
        """Header kind with HF model-card fallback for repo-backed models."""
        kind = self._file_or_self().kind
        if kind == "unknown" and self.hf_repo:
            kind = utils.hf_readme_kind(self.hf_repo, self._hf_home) or "unknown"
        return kind

    @property
    def size_mb(self) -> int:
        """Weight file size in MB (single stat, no content read)."""
        if self.gguf_path is None:
            return 0
        try:
            return os.stat(self.gguf_path).st_size // (1024 ** 2)
        except OSError:
            return 0

    # ── measured block: the single dynamically generated sidecar branch ──

    def measured_block(self) -> dict | None:
        """Machine-written ``measured:`` block (legacy ``fit-params:`` fallback).

        Holds fit-params VRAM numbers plus a ``file:`` sub-block of header
        intrinsics. Returns None when the sidecar has neither.
        """
        raw = self.frontmatter.get(MEASURED_KEY)
        if isinstance(raw, dict):
            return raw
        legacy = self.frontmatter.get(_LEGACY_MEASURED_KEY)
        return legacy if isinstance(legacy, dict) else None

    def _file_stat_sig(self) -> tuple[int, int] | None:
        """Current (size_bytes, mtime_ns) of the weight file, if statable."""
        if self.gguf_path is None:
            return None
        try:
            st = os.stat(self.gguf_path)
        except OSError:
            return None
        return (st.st_size, st.st_mtime_ns)

    def measured_file_dict(self) -> dict | None:
        """Fresh ``file:`` intrinsics for the weight file (may read once).

        Small scalars only — arch, context, kind, size identity, and (for
        safetensors) the derived weight/KV numbers. Never cached bytes.
        Returns None when there is no local file.
        """
        target = self._file_or_self()
        if target.gguf_path is None:
            return None
        sig = self._file_stat_sig()
        if sig is None:
            return None
        try:
            size_mb = os.stat(target.gguf_path).st_size // (1024 ** 2)
        except OSError:
            return None
        block: dict = {
            "size_mb": size_mb,
            "size_bytes": sig[0],
            "mtime_ns": sig[1],
            "arch": target.arch,
            "context_length": target.max_context,
            "kind": target.kind,
        }
        nums = target.safetensors_numbers()
        if nums is not None:
            block["st_weight_mib"] = nums[0]
            block["st_kv_per_token_mib"] = nums[1]
        return block

    def measured_file_valid(self) -> bool:
        """True when the persisted ``file:`` block matches the file on disk."""
        stored = self.measured_block()
        file_block = stored.get("file") if isinstance(stored, dict) else None
        if not isinstance(file_block, dict):
            return False
        sig = self._file_stat_sig()
        if sig is None:
            return False
        try:
            if int(file_block.get("size_bytes", -1)) != sig[0]:
                return False
            if int(file_block.get("mtime_ns", -1)) != sig[1]:
                return False
        except (TypeError, ValueError):
            return False
        return True

    def sync_measured_file(self, write: bool = True) -> None:
        """Refresh the ``file:`` intrinsics when stale/missing.

        Steady state is a no-op (one stat per run): header I/O happens only
        for new or changed files. The in-memory frontmatter always updates;
        the sidecar file is written only when *write* (discovery passes
        False for freshly materialized stub sidecars, which stay pristine —
        their block is still persisted later if fit-params measures them).
        Sidecar-less instances update memory only.
        """
        if self.gguf_path is None or self.measured_file_valid():
            return
        fresh = self.measured_file_dict()
        if fresh is None:
            return
        block = self.frontmatter.get(MEASURED_KEY)
        if not isinstance(block, dict):
            block = {}
            self.frontmatter[MEASURED_KEY] = block
        block["file"] = fresh
        if write:
            self.persist_measured()

    def persist_measured(self, extra: dict | None = None) -> None:
        """Write the ``measured:`` block to the sidecar, preserving the rest.

        The single writer for the single dynamic branch: round-trips the raw
        .md with ruamel.yaml (comments/formatting survive), merges *extra*
        into ``measured:``, and drops the legacy ``fit-params:`` key once
        migrated. Sidecar-less instances (no ``md_path``) are memory-only.
        """
        block = self.frontmatter.get(MEASURED_KEY)
        if not isinstance(block, dict):
            block = {}
        if extra:
            block.update(extra)
            # Affine blocks supersede the pre-affine schema outright: stale
            # keys would linger beside the new numbers forever otherwise.
            if "kv_per_token_mib" in block:
                block.pop("ctx_factor", None)
                block.pop("parallel", None)
        # In-memory state first: callers (saved_for, measured_file_valid)
        # must see the values even when the file cannot be written.
        self.frontmatter[MEASURED_KEY] = copy.deepcopy(block)
        if self.md_path is None:
            logger.debug("no sidecar for %s; measured block is memory-only",
                         self.stem)
            return
        try:
            content = self.md_path.read_text(encoding="utf-8")
        except (OSError, PermissionError) as e:
            logger.debug("cannot read sidecar for measured persist (%s): %s",
                         self.md_path, e)
            return
        if not content.startswith("---"):
            return
        parts = content.split("---", 2)
        if len(parts) < 3:
            return
        try:
            from ruamel.yaml import YAML
        except ImportError:  # pragma: no cover - ruamel.yaml is a hard dependency
            logger.debug("ruamel.yaml unavailable; skipping measured persist")
            return
        yml = YAML()
        yml.preserve_quotes = True
        try:
            fm = yml.load(parts[1])
        except Exception as e:
            logger.debug("cannot parse sidecar frontmatter for persist: %s", e)
            return
        if fm is None:
            from ruamel.yaml.comments import CommentedMap
            fm = CommentedMap()
        fm[MEASURED_KEY] = copy.deepcopy(block)
        if _LEGACY_MEASURED_KEY in fm:
            del fm[_LEGACY_MEASURED_KEY]
        import io
        buf = io.StringIO()
        yml.dump(fm, buf)
        new_content = "---\n" + buf.getvalue().rstrip("\n") + "\n---" + parts[2]
        try:
            self.md_path.write_text(new_content, encoding="utf-8")
            logger.debug("persisted measured block for %s", self.stem)
        except (OSError, PermissionError) as e:
            logger.debug("cannot write sidecar for measured persist (%s): %s",
                         self.md_path, e)

    @property
    def name(self) -> str:
        return self.frontmatter.get("name", self.stem)

    @property
    def label(self) -> str:
        """Human label for logs/errors: sidecar name, else the file stem."""
        return self.md_path.name if self.md_path is not None else self.stem

    @property
    def backend(self) -> str:
        """Serving engine for this model.

        Resolved from the sidecar / override ``backend:`` setting.  When
        neither declares one, discovery finalization infers the backend from
        the file format (GGUF → llama-server, safetensors/HF-repo → vLLM
        docker; see ``backends.infer_backend`` via ``scope.ScopeStack.finalize``).
        This property's fallback exists only for Models used outside the
        normal pipeline.
        """
        return str(self.frontmatter.get("backend") or DEFAULT_BACKEND)

    @property
    def resolved_chat_template(self) -> Path | None:
        """Absolute path to the resolved chat-template file, or None."""
        return getattr(self, "_resolved_chat_template", None)

    @property
    def resolved_loras(self) -> list:
        """Absolute paths to resolved LoRA adapter files."""
        return getattr(self, "_resolved_loras", [])

    @property
    def chat_template_kwargs(self) -> dict | None:
        """Declared chat-template kwargs exposed to clients (per-request use)."""
        kw = self.frontmatter.get("chat_template_kwargs")
        return dict(kw) if isinstance(kw, dict) else None

    def parallel_for(self, default: int) -> int:
        """Sidecar ``parallel`` when declared, else *default*."""
        return int(self.frontmatter.get("parallel", default))

    def cache_type_for(self, default: str) -> str:
        """Sidecar ``cache_type`` when declared, else *default*."""
        return str(self.frontmatter.get("cache_type", default))

    @property
    def cli_args(self) -> str:
        return self.frontmatter.get("cli_args", "")

    @property
    def reasoning_format(self) -> str | None:
        """llama-server ``--reasoning-format`` value (``reasoning-format:`` key).

        Controls how thought tags are parsed/returned (``none``, ``deepseek``,
        ``deepseek-legacy``, ``auto``).  Validated and gated to reasoning-capable
        chat models by ``writer._filter_supported``.
        """
        v = self.frontmatter.get("reasoning-format")
        return str(v) if v else None

    @property
    def reasoning_preserve(self) -> bool:
        """Whether to emit ``--reasoning-preserve`` (``reasoning-preserve:`` key)."""
        return bool(self.frontmatter.get("reasoning-preserve"))

    @property
    def allow_profiles(self) -> list[str] | None:
        return self.frontmatter.get("allow_profiles")

    @property
    def modes(self) -> dict[str, dict] | None:
        """Sidecar-defined sampling modes: name -> param dict.

        When declared, each mode layers over the same-named resolved profile
        (or the fleet defaults) with the single merge rule, and the layered
        modes replace the global profile sampling overrides for this model.
        ``None`` keeps the legacy global-profile behavior.
        """
        m = self.frontmatter.get("modes")
        if not isinstance(m, dict):
            return None
        return {str(name): dict(params) for name, params in m.items()}

    @property
    def default_mode(self) -> str | None:
        """The mode used as this model's default (bare ``${MODEL_ID}`` key).

        Falls back to the first declared mode. Returns None when no modes
        are declared.
        """
        modes = self.modes
        if not modes:
            return None
        dm = str(self.frontmatter.get("default_mode") or "")
        if dm and dm not in modes:
            logger.warning("modes: %s: default_mode %r not declared; using first mode",
                           self.stem, dm)
            return next(iter(modes))
        if dm:
            return dm
        return next(iter(modes))

    @property
    def role(self) -> str:
        """Role this model plays: ``chat``, ``embeddings``, ``rerank``, or ``image``.

        Defaults to ``chat``. Discovery (``from_dir``) injects the role derived
        from a model's directory (``embed/``/``rerank/``/``img/``) or ``type:`` field
        when the sidecar does not declare one explicitly.
        """
        return str(self.frontmatter.get("role") or "chat")

    @property
    def vllm_image(self) -> str | None:
        """Per-model vLLM docker image override (sidecar `vllm_image:`).

        Overrides the profiles.yaml ``vllm.image`` / ``--vllm-image`` for this
        entry only. Returns None when not declared (uses the global default).
        """
        v = self.frontmatter.get("vllm_image")
        return str(v) if v else None

    @property
    def hf_repo(self) -> str | None:
        """Hugging Face repo id to serve, for backends that load safetensors.

        Resolved from an explicit ``hf_repo`` frontmatter field, else parsed
        out of ``hf_url`` (``https://huggingface.co/{owner}/{repo}``). Returns
        None when neither is declared (llama-server GGUFs then use the local
        file path instead).
        """
        repo = self.frontmatter.get("hf_repo")
        if repo:
            return str(repo)
        url = str(self.frontmatter.get("hf_url", "")).strip()
        if not url:
            return None
        m = re.search(r"huggingface\.co/([^/?#]+/[^/?#]+)", url)
        return m.group(1) if m else None

    @property
    def vram_mb(self) -> int:
        """Model file size in MB (used for 'smallest' selection)."""
        return self.size_mb

    @property
    def on_cpu(self) -> bool:
        """True when the model is CPU-resident (`device: cpu` in frontmatter)."""
        return str(self.frontmatter.get("device", "")).strip().lower() == "cpu"

    @property
    def device(self) -> int | None:
        """Explicit GPU device index (for multi-GPU pinning), if declared.

        Returns None for CPU-resident models (`device: cpu`).
        """
        if self.on_cpu:
            return None
        v = self.frontmatter.get("device")
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    @property
    def concurrency(self) -> int | None:
        """Explicit per-model concurrency limit, if declared."""
        v = self.frontmatter.get("concurrency")
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    @property
    def description(self) -> str | None:
        return self.frontmatter.get("description")

    @property
    def template_id(self) -> str:
        """The id used as the llama-swap model key.

        Defaults to ``slugify(sidecar stem)``; an explicit ``model_id`` (or
        ``id``) frontmatter field overrides it verbatim and is validated to
        ``^[A-Za-z0-9._-]+$`` (error on invalid). ``name`` is display-only
        and does not affect the id.
        """
        raw = self.frontmatter.get("model_id") or self.frontmatter.get("id")
        if raw is not None:
            raw_str = str(raw).strip()
            if not re.match(r"^[A-Za-z0-9._-]+$", raw_str):
                raise ValueError(
                    f"model_id {raw_str!r} for {self.label} contains "
                    f"invalid characters (allowed: A-Za-z0-9._-)"
                )
            return raw_str
        return utils.slugify(str(self.stem))

    @property
    def capabilities(self) -> list[str]:
        """Declared capabilities (explicit; mmproj does not imply image/video)."""
        caps = [str(c) for c in (self.frontmatter.get("capabilities") or [])]
        return caps

    def view_for(self, include_mmproj: bool) -> Model:
        """Frontmatter view for a serving variant.

        The companion-off variant is this model itself: overlay keys live
        only inside the mmproj block, so the base frontmatter already
        describes serving without the companion.  The companion-on variant
        is a cached shallow copy with the block merged over the frontmatter
        (single merge rule); resolved files are shared, the VRAM budget is
        rebuilt lazily so it binds to the view.  Models without a block
        return themselves either way (legacy behavior, untouched).
        """
        if (self._is_view or not include_mmproj
                or not (self.mmproj and self.mmproj.gguf_path)
                or not self.mmproj_overlay):
            return self
        if self._view_on is None:
            view = copy.copy(self)
            view.frontmatter = utils.merge_layer(
                self.frontmatter, self.mmproj_overlay,
                origin=f"mmproj:{self.stem}")
            view._vram = None
            view._view_on = None
            view._is_view = True
            self._view_on = view
        return self._view_on

    @property
    def freethought(self) -> float | None:
        """0.0 = readily refuses 'distasteful' topics; 1.0 = reasons about anything rationally."""
        v = self.frontmatter.get("freethought")
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @property
    def mmproj_size_mb(self) -> int:
        if self.mmproj is not None:
            return self.mmproj.size_mb
        return 0

    def _image_token_value(self, key: str) -> int | None:
        """Validate a positive-integer image token frontmatter key."""
        v = self.frontmatter.get(key)
        if v is None or v == "":
            return None
        try:
            n = int(v)
        except (TypeError, ValueError):
            logger.warning("%s: %s: %r is not an integer; ignoring",
                           self.stem, key, v)
            return None
        if n <= 0:
            logger.warning("%s: %s: %d is not positive; ignoring",
                           self.stem, key, n)
            return None
        return n

    @property
    def image_min_tokens(self) -> int | None:
        """Sidecar ``image_min_tokens``: floor on image tokens per image.

        Dynamic-resolution vision models (Qwen-VL family) upscale small
        images to this token count; ``None`` (unset) means llama-server uses
        the model's own default. Only meaningful with an attached mmproj.
        """
        return self._image_token_value("image_min_tokens")

    @property
    def image_max_tokens(self) -> int | None:
        """Sidecar ``image_max_tokens``: cap on image tokens per image.

        Bounds the KV cost of large images. ``None`` (unset) keeps the model
        default — which can be very large (Qwen2.5-VL tops out at 16384
        tokens/image), so an explicit cap is the normal way to bound VRAM.
        """
        return self._image_token_value("image_max_tokens")

    def pass_through_metadata(self) -> dict:
        """Frontmatter fields exposed to clients, minus builder-consumed keys.

        The result is pass-through-by-default: any new field an agent writes in a
        sidecar flows through automatically. `capabilities` is builder-consumed
        (mapped to llama-swap's native capabilities block) and excluded here.
        Falsy-but-meaningful values (0, 0.0) are kept.
        """
        meta: dict = {}
        for k, v in self.frontmatter.items():
            if k in self.FIELDS or _CL_RE.match(k):
                continue
            if v is None or v == "" or (isinstance(v, (list, dict)) and len(v) == 0):
                continue
            if k not in self.KNOWN_METADATA:
                # difflib hint for likely typo
                import difflib
                close = difflib.get_close_matches(k, sorted(self.FIELDS | self.KNOWN_METADATA), n=1, cutoff=0.7)
                hint = f" (did you mean {close[0]!r}?)" if close else ""
                logger.warning("sidecar %s: unhandled frontmatter key %r%s -> passed through as metadata",
                               self.label, k, hint)
            meta[k] = copy.deepcopy(v)
        return meta

    def _param_counts(self) -> tuple[float, float]:
        """(total_B, active_B) in billions parsed from the `parameters` field."""
        raw = str(self.frontmatter.get("parameters", ""))
        nums = re.findall(r"(\d+(?:\.\d+)?)\s*([BM])", raw)
        if not nums:
            return (0.0, 0.0)

        def to_b(x: str, u: str) -> float:
            return float(x) * (1.0 if u == "B" else 1e-3)

        total = to_b(*nums[0])
        active = to_b(*nums[1]) if len(nums) > 1 else total
        return (total, active)

    def _quant_bits(self) -> float:
        """Approximate bits-per-weight for the `quantization` field."""
        q = str(self.frontmatter.get("quantization", "")).upper()
        q = re.sub(r"^(UD-|I1-|U-)?", "", q)
        table = {
            "Q2_K": 2.75, "Q3_K_S": 3.0, "Q3_K_M": 3.5, "Q3_K_L": 3.75,
            "Q4_0": 4.0, "Q4_1": 4.5, "Q4_K_S": 4.25, "Q4_K_M": 4.5, "Q4_K_XL": 4.5,
            "Q5_0": 5.0, "Q5_1": 5.5, "Q5_K_S": 5.25, "Q5_K_M": 5.5, "Q5_K_XL": 5.5,
            "Q6_K": 6.5, "Q8_0": 8.0, "Q8_K": 8.5,
            "F16": 16.0, "FP16": 16.0, "BF16": 16.0, "FP8": 8.0, "F32": 32.0,
            "IQ1": 1.5, "IQ2": 2.5, "IQ3": 3.5, "IQ4": 4.5,
        }
        for key, bits in table.items():
            if key in q:
                return bits
        return 0.0

    def _mtp_info(self) -> tuple[bool, int]:
        has_mtp = self.frontmatter.get("mtp")
        speculative = self.frontmatter.get("speculative")
        if has_mtp or (speculative and "mtp" in str(speculative).lower()):
            n_max = int(self.frontmatter.get("mtp_draft_n_max", _MTP_DRAFT_N_MAX))
            return True, n_max
        return False, 0

    def throughput_factor(self) -> float | None:
        """Heuristic relative throughput index (higher = faster). Not real tok/s.

        Combines MTP speedup (1 + draft_n * accept_prob) with a relative base from
        active param count and quantization bits. Use only for comparing models.
        """
        active = self._param_counts()[1]
        bits = self._quant_bits()
        if active <= 0 or bits <= 0:
            return None
        base = 54.0 / (active * bits)  # 12B Q4 ~= 1.0
        mtp_on, n_max = self._mtp_info()
        acc = self.frontmatter.get("mtp_accuracy")
        speedup = 1.0
        if mtp_on and acc is not None:
            try:
                speedup = 1.0 + float(n_max) * float(acc)
            except (TypeError, ValueError):
                speedup = 1.0
        return round(base * speedup, 3)


# ── File-type subclasses (chosen transparently by Model.from_file) ─────────
#
# Each parses its header once, lazily, into small scalars held in ``_header``
# — weight-file bytes are never cached. Tests register zero-IO fakes via
# Model.register_file_class() instead of touching disk.


class GGUFModel(Model):
    """Model backed by a GGUF file: arch/context/kind from one header walk."""

    def _load_header(self) -> tuple[str | None, bool, int | None]:
        """(arch, has_context_length, context_length), parsed once."""
        if not isinstance(self._header, tuple):
            path = self.gguf_path
            assert path is not None
            arch, has_ctx = utils.gguf_header_probe(path)
            ctx = utils.read_gguf_context_length(path)
            self._header = (arch, has_ctx, ctx)
        header: tuple[str | None, bool, int | None] = self._header
        return header

    @property
    def arch(self) -> str | None:
        return self._load_header()[0]

    @property
    def max_context(self) -> int | None:
        return self._load_header()[2]

    @property
    def kind(self) -> str:
        arch, has_ctx, _ = self._load_header()
        if arch:
            if any(rx.search(arch) for rx in _DIFFUSION_ARCH_RES):
                return "image"
            if has_ctx:
                return "text"
        return "unknown"


class SafetensorsModel(Model):
    """Model backed by a safetensors file: kind + VRAM numbers, parsed once."""

    def _load_header(self) -> dict:
        """``{"kind": ..., "nums": {cache_type: (mib, kv) or None}}``."""
        header = self._header
        if not isinstance(header, dict):
            path = self.gguf_path
            assert path is not None
            header = {"kind": utils.sniff_safetensors(path), "nums": {},
                      "path": path}
            self._header = header
        return header

    @property
    def kind(self) -> str:
        kind: str = self._load_header()["kind"]
        return kind

    def safetensors_numbers(self, cache_type: str = "q8_0") -> tuple[int, float] | None:
        header = self._load_header()
        nums: dict = header["nums"]
        if cache_type not in nums:
            try:
                nums[cache_type] = utils.estimate_safetensors(
                    header["path"], cache_type)
            except Exception:
                nums[cache_type] = None
        result: tuple[int, float] | None = nums[cache_type]
        return result