# llama_packer/model.py
"""Model class representing a language model with companions and VRAM calculations."""

from __future__ import annotations

import copy
import logging
import os
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

    # Frontmatter keys this Model consumes (not passed through to metadata)
    FIELDS: ClassVar[frozenset[str]] = frozenset({
        "name", "context_length", "description", "cli_args", "model",
        "backend", "hf_repo", "chat_template", "chat_template_kwargs", "loras",
        "attention", "kv_cache", "tool_args", "speculative", "mmproj",
        "mtp", "mtp_spec_type", "mtp_draft_n_max", "mtp_draft_p_min",
        "role", "targets", "allow_profiles", "spare", "capabilities",
        "ignore", "device", "concurrency", "fit-params", "vllm_image",
        "modes", "default_mode", "reasoning-format", "reasoning-preserve",
        "cache_type", "parallel",
    })

    def __init__(self, md_path: Path, frontmatter: dict):
        self.md_path = md_path
        self.frontmatter = frontmatter
        self.stem = md_path.stem
        self._gguf_ctx_cache: int | None = None  # cached GGUF architectural context
        self._vram: VramBudget | None = None  # lazy VRAM budget calculator

        # Resolve the model file path. vLLM backends may have no local file —
        # they are served from an HF repo id (hf_repo / hf_url) instead.
        self.gguf_path = self._resolve_gguf_path()
        if not self.gguf_path and self.hf_repo is None:
            raise ValueError(f"No GGUF/safetensors or hf_repo found for {md_path}")

        # Resolve companions (mmproj, MTP)
        self.mmproj: Model | None = None
        self.mtp: Model | None = None
        self._resolve_companions()

        logger.info("model: %s (gguf=%s, mmproj=%s, mtp=%s)",
                    self.stem, self.gguf_path.name if self.gguf_path else self.hf_repo,
                    self.mmproj.stem if self.mmproj else "none",
                    self.mtp.stem if self.mtp else "none")

    def _resolve_gguf_path(self) -> Path | None:
        """Resolve the main model file (.gguf or .safetensors) from frontmatter or convention."""
        parent = self.md_path.parent

        # 1. Check frontmatter `model` field
        file_ref = self.frontmatter.get("model")
        if file_ref:
            model_file = parent / file_ref
            if not model_file.is_file():
                model_file = parent.parent / file_ref
            if model_file.is_file():
                return utils.smart_resolve(model_file)

        # 2. Convention: same stem, either .gguf or .safetensors
        for ext in (".gguf", ".safetensors"):
            candidate = parent / f"{self.stem}{ext}"
            if candidate.is_file():
                return utils.smart_resolve(candidate)

        # 3. No model file found by stem -> give up
        return None

    def _resolve_companions(self) -> None:
        """Resolve mmproj and MTP companions from frontmatter or fuzzy match."""
        if not self.gguf_path:
            logger.debug("no gguf_path, skipping companion resolution for %s", self.stem)
            return

        assert self.gguf_path is not None

        # Search for companions in both model-file parent and .md parent (models dir)
        # This handles symlinks where .gguf points elsewhere but companions stay in models/
        search_dirs = []
        if self.gguf_path.parent not in search_dirs:
            search_dirs.append(self.gguf_path.parent)
        md_parent = self.md_path.parent
        if md_parent not in search_dirs:
            search_dirs.append(md_parent)

        # --- mmproj ---
        mmproj_val = self.frontmatter.get("mmproj")
        if mmproj_val is False:
            # Explicit disable
            self.mmproj = None
        elif mmproj_val:
            # Explicit path in frontmatter — search both directories
            companion = None
            for d in search_dirs:
                candidate = d / mmproj_val
                if candidate.is_file():
                    companion = candidate
                    break
            if companion:
                self.mmproj = Model._get_or_create_companion(companion)
            else:
                logger.warning("mmproj: configured %s missing for %s", mmproj_val, self.stem)
        else:
            # Fuzzy match: look for *mmproj*.gguf with same family prefix
            family = utils._gguf_family(self.gguf_path.stem).lower()
            for d in search_dirs:
                for f in sorted(d.glob("*mmproj*.gguf")):
                    mmproj_family = utils._gguf_family(f.stem).lower()
                    if mmproj_family.startswith(family) or family.startswith(mmproj_family):
                        self.mmproj = Model._get_or_create_companion(f)
                        break
                if self.mmproj:
                    break

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
                if companion:
                    self.mtp = Model._get_or_create_companion(companion)
                else:
                    logger.warning("mtp: companion %s missing for %s", speculative, self.stem)
            else:
                # Baked-in MTP (no separate file)
                self.mtp = None  # baked-in, no separate file

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
    ) -> list[Model]:
        """Discover all models across *models_dirs* via .md sidecars.

        *models_dirs* is a single directory or a list of them.  Symlinks that
        resolve to the same underlying model file are deduplicated (first wins).
        Returns a list of instantiated Model objects (main models only).
        """
        if isinstance(models_dirs, (str, os.PathLike)):
            models_dirs = [Path(models_dirs)]
        else:
            models_dirs = [Path(d) for d in models_dirs]

        # Single source of truth for role classification (see utils.classify_models).
        classified = utils.classify_models(models_dirs, extra_dirs)
        md_items = {p: k for p, k in classified if p.suffix.lower() == ".md"}
        gguf_items = {p: k for p, k in classified if p.suffix.lower() in (".gguf", ".safetensors")}

        known_md: set[Path] = set()
        models: list[Model] = []

        for md_path, kind in md_items.items():
            if md_path in known_md:
                continue
            fm = utils.parse_frontmatter(md_path)
            if not fm.get("name"):
                continue
            if fm.get("ignore"):
                logger.info("ignore: skipping %s", md_path.name)
                continue
            # Classification drives role: an embed/rerank file whose sidecar
            # does not declare `role:` inherits it from its location/type.
            if kind in ("embeddings", "rerank") and str(fm.get("role") or "") not in ("embeddings", "rerank"):
                fm = {**fm, "role": kind}
            known_md.add(md_path)
            try:
                model = cls(md_path, fm)
                models.append(model)
            except Exception as e:
                logger.warning("failed to load model from %s: %s", md_path, e)

        if generate_stubs:
            for gguf, kind in gguf_items.items():
                # Companions (mmproj/mtp) and embed/rerank models are never
                # main-model (chat) stubs.
                if kind in ("mmproj", "mtp", "embeddings", "rerank"):
                    continue
                md_path = gguf.with_suffix(".md")
                if md_path in md_items:
                    continue
                if not gguf.is_file():
                    continue
                fm = utils.generate_stub_md(md_path, gguf)
                try:
                    model = cls(md_path, fm)
                    models.append(model)
                    logger.info("stub: %s", md_path.name)
                except Exception as e:
                    logger.warning("failed to create stub for %s: %s", gguf, e)

        # Deduplicate symlinks to the same model file (first wins).
        deduped: list[Model] = []
        seen: set[str] = set()
        for model in models:
            key = os.path.realpath(str(model.gguf_path)) if model.gguf_path else str(model.md_path)
            if key in seen:
                logger.info("duplicate model file (symlink) skipped: %s", model.stem)
                continue
            seen.add(key)
            deduped.append(model)
        return deduped

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
        neither declares one, ``apply_overrides`` infers the backend from the
        file format (GGUF → llama-server, safetensors/HF-repo → vLLM docker;
        see ``backends.infer_backend``).  This property's fallback exists only
        for Models used outside the normal pipeline.
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
        chat models by ``apply_overrides``.
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
        """Role this model plays: ``chat``, ``embeddings``, or ``rerank``.

        Defaults to ``chat``. Discovery (``from_dir``) injects the role derived
        from a model's directory (``embed/``/``rerank/``) or ``type:`` field
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
        """The id used as the llama-swap model key (slug of `name`)."""
        return utils.slugify(self.name)

    @property
    def capabilities(self) -> list[str]:
        """Declared capabilities, with `vision` auto-added when a companion mmproj exists."""
        caps = [str(c) for c in (self.frontmatter.get("capabilities") or [])]
        if self.mmproj and "vision" not in [c.lower() for c in caps]:
            caps.append("vision")
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