# llama_packer/model.py
"""Model class representing a language model with companions and VRAM calculations."""

from __future__ import annotations

import copy
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import yaml

from llama_packer import utils
from llama_packer.backends import DEFAULT_BACKEND

if TYPE_CHECKING:
    from llama_packer.vram import VramBudget

logger = logging.getLogger(__name__)

# Frontmatter keys that are computed/derived rather than declared
_CL_RE = re.compile(r"^context_limit_\d+G$")


class Model:
    """Represents a main model with optional mmproj and MTP companions."""

    # Global registry: claimed weight files → owning Model. The key is the
    # weight's realpath (and dev:ino for hardlinks) so a sidecar with an
    # explicit `model:` that strips quant/version still claims the file.
    # The registry is cleared at the start of each discover() call; all
    # discovery-time Model construction registers automatically.
    _by_gguf: ClassVar[dict[str, "Model"]] = {}
    _by_gguf_stat: ClassVar[dict[str, "Model"]] = {}
    _by_md: ClassVar[dict[str, "Model"]] = {}

    @classmethod
    def clear_registry(cls) -> None:
        cls._by_gguf.clear()
        cls._by_gguf_stat.clear()
        cls._by_md.clear()

    @classmethod
    def _register(cls, m: "Model") -> None:
        import os
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
            except OSError:
                pass
        try:
            md_real = os.path.realpath(str(m.md_path))
            cls._by_md.setdefault(md_real, m)
        except OSError:
            pass

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
        "ignore", "device", "concurrency", "fit-params", "vllm_image",
        "modes", "default_mode", "reasoning-format", "reasoning-preserve",
        "cache_type", "parallel",
        "image_min_tokens", "image_max_tokens",

    })

    # Known pass-through metadata keys (documented in models_AGENTS.md)
    # Any frontmatter key not in FIELDS and not in this set triggers a warning
    # but still flows through as metadata (elegance over backwards compat).
    KNOWN_METADATA: ClassVar[frozenset[str]] = frozenset({
        "parameters", "quantization", "hf_url", "license", "base_model",
        "architecture", "finetune", "type", "mtp_accuracy", "strengths",
        "weaknesses", "freethought",
        # per-model calibration / quality metrics (kept for now, low signal)
        "quant_layout", "calibration_tokens", "top1_agreement_vs_bf16", "kld_vs_bf16",
    })

    def __init__(self, md_path: Path, frontmatter: dict, hf_home=None):
        self.md_path = md_path
        self.frontmatter = frontmatter
        self.stem = md_path.stem
        self._hf_home = hf_home  # HF cache root override for hub snapshot resolution
        self._gguf_ctx_cache: int | None = None  # cached GGUF architectural context
        self._vram: VramBudget | None = None  # lazy VRAM budget calculator

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
                    snap = utils.hf_snapshot_dir(self.hf_repo, self._hf_home)
                    if snap and snap.is_dir():
                        files = sorted(p.name for p in snap.iterdir() if p.is_file())
                        if files:
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

        # Resolve companions later — resolve_companions() is called by
        # discovery after scope defaults and override rules have had their
        # say (frontmatter must be final before companion resolution).
        self.mmproj: Model | None = None
        self.mtp: Model | None = None
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
        elif mmproj_val:
            # Explicit path in frontmatter — search both directories, then the
            # hf_repo snapshot (readable snapshot filenames; globs allowed).
            companion = None
            for d in search_dirs:
                candidate = d / mmproj_val
                if candidate.is_file():
                    companion = candidate
                    break
            if companion is None:
                companion = self._resolve_hub_ref(str(mmproj_val),
                                                  pattern_hint="*mmproj*.gguf")
            if companion:
                self.mmproj = Model._get_or_create_companion(companion)
            else:
                logger.warning("mmproj: configured %s missing for %s", mmproj_val, self.stem)
        else:
            # No explicit mmproj — local models require explicit mmproj: (or
            # mmproj: false to silence). Only HF snapshots auto-discover, and
            # only when the gguf lives inside its own snapshot (every file in
            # that snapshot belongs to the same release). Matching rule:
            # mmproj_stripped (stem without 'mmproj.*' + quant/version) must be
            # prefix of model stem.
            # Rerank/embeddings (and image/s2t/t2s) never auto-attach vision.
            if self.hf_repo and self.role not in utils.NON_CHAT_ROLES:
                snap = utils.hf_snapshot_dir(self.hf_repo, self._hf_home)
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
                    for f in sorted(snap.glob("*mmproj*.gguf")):
                        stripped = _strip_mmproj(f.stem)
                        if not stripped:
                            continue
                        if not model_stem_lc.startswith(stripped):
                            continue
                        candidates.append(f)
                    if candidates:
                        preferred = [c for c in candidates if _is_preferred_mmproj(c.name)]
                        chosen = sorted(preferred or candidates)[0]
                        self.mmproj = Model._get_or_create_companion(chosen)
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
                companion = None
                for d in search_dirs:
                    candidate = d / speculative
                    if candidate.is_file():
                        companion = candidate
                        break
                if companion is None:
                    companion = self._resolve_hub_ref(str(speculative))
                if companion:
                    self.mtp = Model._get_or_create_companion(companion)
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
        parent = self.md_path.parent

        # 1. Check frontmatter `model` field
        file_ref = self.frontmatter.get("model")
        if file_ref:
            hit = self._resolve_ref(str(file_ref))
            if hit is not None:
                return utils.smart_resolve(hit)
            # Explicit model: field but file not found is an error – list how.
            # Keep returning None so __init__ can raise with full guidance;
            # the message there explains the tried locations.
            return None

        # 2. Convention: same stem, .gguf / .safetensors / whisper GGML .bin /
        # kokoro ONNX (.bin and .onnx resolve only for their audio roles —
        # discovery requires the s2t/t2s directory)
        for ext in (".gguf", ".safetensors", ".bin", ".onnx"):
            candidate = parent / f"{self.stem}{ext}"
            if candidate.is_file():
                return utils.smart_resolve(candidate)

        # 3. No local file – try hf_repo snapshot auto (exactly one non-mmproj
        # model → use it, several → error, none → give up for __init__ error)
        if self.hf_repo:
            snap = utils.hf_snapshot_dir(self.hf_repo, self._hf_home)
            if snap is not None and snap.is_dir():
                # Collect non-mmproj candidates (mmproj/mtp are companions, not
                # main models). Also skip .msgpack etc – only real weight files.
                candidates: list[Path] = []
                for p in snap.iterdir():
                    if not p.is_file():
                        continue
                    suf = p.suffix.lower()
                    if suf not in {".gguf", ".safetensors", ".bin", ".onnx"}:
                        continue
                    stem_lc = p.stem.lower()
                    if "mmproj" in stem_lc:
                        continue
                    # mtp/draft companions also contain "mtp" – skip them
                    if "mtp" in stem_lc and "mmproj" not in stem_lc:
                        # Heuristic: draft/MTP files are companions, but a main
                        # model could legitimately contain "mtp" (rare). Be
                        # conservative – only skip if it also looks like a
                        # companion (e.g. draft, speculative). Keep simple for now
                        # and only skip mmproj; MTP main models are uncommon.
                        pass
                    candidates.append(p)
                if len(candidates) == 1:
                    return utils.smart_resolve(candidates[0])
                if len(candidates) > 1:
                    names = ", ".join(sorted(p.name for p in candidates))
                    raise ValueError(
                        f"sidecar {self.md_path.name} (hf_repo {self.hf_repo!r}): "
                        f"several models in snapshot {snap}: {names} – "
                        f"set `model: <filename>` in the sidecar to choose one "
                        f"(exact file in the snapshot, e.g. `ls {snap}`)"
                    )
                # zero candidates – fall through to __init__ error (no file)

        # 4. No model file found by any method
        return None

    def _resolve_ref(self, ref: str) -> Path | None:
        """Resolve a file reference: sidecar dir, its parent, then HF hub.

        ``repo:``-relative references use the sidecar's ``hf_repo`` (snapshot
        filenames are readable — no blob hashes needed) and may be globs
        (``mmproj*.gguf``).  The ``hub:<org>/<repo>:<file>`` form addresses
        any cached repo explicitly.  Returns an absolute path or None.
        """
        parent = self.md_path.parent
        for d in (parent, parent.parent):
            candidate = d / ref
            if candidate.is_file():
                return candidate
        repo = self.hf_repo
        if repo:
            return utils.hf_snapshot_file(repo, ref, self._hf_home)
        return None

    def _resolve_hub_ref(self, ref: str,
                         pattern_hint: str | None = None) -> Path | None:
        """Hub-only resolution of *ref* (companion fallback).

        ``hub:<org>/<repo>:<file-or-glob>`` addresses any cached repo;
        anything else resolves against this sidecar's ``hf_repo``.  With
        *pattern_hint* (e.g. ``*mmproj*.gguf``), an unresolvable exact name
        falls back to a single glob match in the snapshot.
        """
        repo = self.hf_repo
        if ref.startswith("hub:"):
            rest = ref[len("hub:"):]
            repo, _, ref = rest.rpartition(":")
            if not repo:
                logger.warning("hf: malformed %r (expected hub:org/repo:file)", ref)
                return None
        if not repo:
            return None
        hit = utils.hf_snapshot_file(repo, ref, self._hf_home)
        if hit is None and pattern_hint:
            hit = utils.hf_snapshot_file(repo, pattern_hint, self._hf_home)
        return hit

    @staticmethod
    def _get_or_create_companion(path: Path) -> Model:
        """Get or create a lightweight companion Model for mmproj/mtp files."""
        # For companions, we create a minimal Model with just the path
        # They don't have frontmatter or model-file resolution
        companion = Model.__new__(Model)
        companion.md_path = path.with_suffix(".md")
        companion.frontmatter = {}
        companion.stem = path.stem
        companion.gguf_path = utils.smart_resolve(path)
        companion._vram = None
        companion._hf_home = None
        companion.mmproj = None
        companion.mtp = None
        return companion

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
        content = "---\n" + yaml.dump(self.frontmatter, sort_keys=False).rstrip() + "\n---\n"
        path.write_text(content, encoding="utf-8")
        try:
            path.chmod(0o644)
        except OSError:
            pass
        logger.info("wrote sidecar: %s", path.name)

    @property
    def context_length(self) -> int:
        return self.frontmatter.get("context_length", utils._DEFAULT_CONTEXT_LENGTH)

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
            "context_length", utils._DEFAULT_CONTEXT_LENGTH
        ))

    @property
    def gguf_context_length(self) -> int | None:
        """Read the model's architectural context limit from GGUF metadata.

        Returns the value of <architecture>.context_length from the GGUF header,
        or None if unavailable (safetensors, missing file, parse error).
        """
        if self._gguf_ctx_cache is not None:
            return self._gguf_ctx_cache
        if self.gguf_path is None or str(self.gguf_path).endswith(".safetensors"):
            return None
        self._gguf_ctx_cache = utils.read_gguf_context_length(self.gguf_path)
        return self._gguf_ctx_cache

    @property
    def name(self) -> str:
        return self.frontmatter.get("name", self.stem)

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
        """Sidecar-defined sampling modes (full profiles): name -> param dict.

        When declared, these fully replace the global profile sampling
        overrides for this model. ``None`` keeps the legacy global-profile
        behavior.
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
        if not self.gguf_path:
            return 0
        return utils.get_model_size_mb(str(self.gguf_path))

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
                    f"model_id {raw_str!r} for {self.md_path.name} contains "
                    f"invalid characters (allowed: A-Za-z0-9._-)"
                )
            return raw_str
        return utils.slugify(str(self.stem))

    @property
    def capabilities(self) -> list[str]:
        """Declared capabilities (explicit; mmproj does not imply image/video)."""
        caps = [str(c) for c in (self.frontmatter.get("capabilities") or [])]
        return caps

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
        if self.mmproj and self.mmproj.gguf_path:
            return utils.get_model_size_mb(str(self.mmproj.gguf_path))
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
                               self.md_path.name, k, hint)
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
            n_max = int(self.frontmatter.get("mtp_draft_n_max", utils._MTP_DRAFT_N_MAX))
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