# model_cfg/model.py
"""Model class representing a language model with companions and VRAM calculations."""

from __future__ import annotations

import copy
import logging
import warnings
from pathlib import Path
from typing import ClassVar

from model_cfg import utils

logger = logging.getLogger(__name__)


class Model:
    """Represents a main model with optional mmproj and MTP companions."""

    _by_stem: ClassVar[dict[str, Model]] = {}
    _by_companion: ClassVar[dict[str, Model]] = {}
    _fit_cache: dict[tuple, tuple[int, int, int]]

    # Frontmatter keys this Model consumes (not passed through to metadata)
    FIELDS: ClassVar[frozenset[str]] = frozenset({
        "name", "template", "context_length", "description", "cli_args", "model",
        "attention", "kv_cache", "tool_args", "speculative", "mmproj",
        "mtp", "mtp_spec_type", "mtp_draft_n_max", "mtp_draft_p_min",
        "targets", "allow_profiles", "reasoning", "spare",
    })

    def __init__(self, md_path: Path, frontmatter: dict):
        self.md_path = md_path
        self.frontmatter = frontmatter
        self.stem = md_path.stem
        self._fit_cache = {}
        self._sf_estimate = None  # cached safetensors header estimate
        self._fit_logged: set[str] = set()  # dedupe fit-params warnings per model

        # Resolve the model file path
        self.gguf_path = self._resolve_gguf_path()
        if not self.gguf_path:
            raise ValueError(f"No GGUF found for {md_path}")

        # Register as main model
        Model._by_stem[self.stem] = self

        # Resolve companions (mmproj, MTP)
        self.mmproj: Model | None = None
        self.mtp: Model | None = None
        self._resolve_companions()

        # Register reverse mapping for companions
        if self.mmproj:
            Model._by_companion[self.mmproj.stem] = self
        if self.mtp:
            Model._by_companion[self.mtp.stem] = self

        logger.info("model: %s (gguf=%s, mmproj=%s, mtp=%s)",
                    self.stem, self.gguf_path.name,
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
        companion._fit_cache = {}
        companion.mmproj = None
        companion.mtp = None
        return companion

    @classmethod
    def from_dir(
        cls,
        models_dir: Path,
        *,
        generate_stubs: bool = True,
        extra_dirs: list[str] | None = None,
    ) -> list[Model]:
        """Discover all models in models_dir via .md sidecars.

        Returns list of instantiated Model objects (main models only).
        """
        cls._by_stem.clear()
        cls._by_companion.clear()

        known_md: set[Path] = set()
        models: list[Model] = []

        for md_path in sorted(models_dir.rglob("*.md")):
            fm = utils.parse_frontmatter(md_path)
            if not fm.get("name"):
                continue
            known_md.add(md_path.absolute())
            try:
                model = cls(md_path, fm)
                models.append(model)
            except Exception as e:
                logger.warning("failed to load model from %s: %s", md_path, e)

        if generate_stubs:
            dirs_to_scan = [models_dir] + [models_dir / d for d in (extra_dirs or [])]
            for scan_dir in dirs_to_scan:
                if not scan_dir.is_dir():
                    continue
                for gguf in sorted(scan_dir.glob("*.gguf")) + sorted(scan_dir.glob("*.safetensors")):
                    if "mmproj" in gguf.stem.lower():
                        continue
                    if utils._is_mtp_companion(gguf.stem):
                        continue
                    md_path = gguf.with_suffix(".md")
                    if md_path not in known_md and gguf.is_file():
                        fm = utils.generate_stub_md(md_path, gguf)
                        try:
                            model = cls(md_path, fm)
                            models.append(model)
                            logger.info("stub: %s", md_path.name)
                        except Exception as e:
                            logger.warning("failed to create stub for %s: %s", gguf, e)

        return models

    @classmethod
    def all(cls) -> list[Model]:
        """Return all registered main models."""
        return list(cls._by_stem.values())

    @classmethod
    def __getitem__(cls, name: str) -> Model:
        """Look up a model by filename stem.

        If the name is a companion (mmproj/mtp), returns the parent main model.
        If no parent found, warns and raises KeyError.
        """
        stem = Path(name).stem
        if stem in cls._by_stem:
            return cls._by_stem[stem]
        if stem in cls._by_companion:
            return cls._by_companion[stem]
        # Looks like a companion file but no parent registered
        if "mmproj" in stem.lower() or utils._is_mtp_companion(stem):
            warnings.warn(f"Not a main model file: {name}")
        raise KeyError(f"Model not found: {name}")

    def fit_params(
        self,
        fit_bin: str,
        fit_ctx: int | None = None,
        fit_target_mib: int | None = None,
        cache_type: str = "q8_0",
        parallel: int = 1,
    ) -> tuple[int, int, int]:
        """Run llama-fit-params and return (model_mib, context_mib, compute_mib).

        Args:
            fit_bin: Path to llama-fit-params binary
            fit_ctx: Design context size (passed as -c)
            fit_target_mib: VRAM target in MiB (passed as --fit-target)
            cache_type: KV cache quantization (e.g., q8_0)
            parallel: Number of parallel slots

        Returns:
            (model_mib, context_mib, compute_mib) or (0, 0, 0) on failure
        """
        cache_key = (fit_ctx, fit_target_mib, cache_type, parallel)
        if cache_key in self._fit_cache:
            return self._fit_cache[cache_key]

        cmd = [
            fit_bin,
            "--fit-print", "on",
            "-m", str(self.gguf_path),
            "--cache-type-k", cache_type,
            "--cache-type-v", cache_type,
        ]
        if fit_target_mib is not None:
            cmd += ["--fit-target", str(fit_target_mib)]
        if fit_ctx is not None:
            cmd += ["-c", str(fit_ctx)]
        if parallel > 1:
            cmd += ["--parallel", str(parallel)]

        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning("fit-params timeout for %s", self.stem)
            return 0, 0, 0
        except FileNotFoundError:
            logger.warning("fit-params binary not found: %s", fit_bin)
            return 0, 0, 0
        except Exception as e:
            logger.warning("fit-params failed for %s: %s", self.stem, e)
            return 0, 0, 0

        for line in out.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) == 4 and parts[0].startswith("Vulkan"):
                try:
                    model_mib = int(parts[1])
                    context_mib = int(parts[2])
                    compute_mib = int(parts[3])
                    self._fit_cache[cache_key] = (model_mib, context_mib, compute_mib)
                    return model_mib, context_mib, compute_mib
                except ValueError:
                    pass

        if out.returncode != 0:
            msg = f"fit-params crashed for {self.stem} (exit {out.returncode})"
            if msg not in self._fit_logged:
                self._fit_logged.add(msg)
                logger.warning(msg)
        else:
            # Parse failed but process exited cleanly — log first 3 lines of output
            preview = "\n".join(out.stdout.splitlines()[:3])
            msg = f"could not parse fit-params output for {self.stem}: {preview}"
            if msg not in self._fit_logged:
                self._fit_logged.add(msg)
                logger.warning(msg)
        return 0, 0, 0

    def calc_ctx(
        self,
        vram_total_mb: int,
        fit_bin: str,
        parallel: int = 1,
        spare_mb: int = 0,
        mmproj_mb: int = 0,
        cache_type: str = "q8_0",
        design_ctx: int | None = None,
    ) -> int:
        """Calculate max context size for given VRAM.

        Uses llama-fit-params --fit-target to find the context that fits.
        """
        available = (
            vram_total_mb
            - utils._RESERVE_SYSTEM
            - utils._RESERVE_VIDEO
            - spare_mb
            - mmproj_mb
        )

        # If companion MTP exists, subtract its size from available VRAM
        if self.mtp and self.mtp.gguf_path:
            mtp_mb = utils.get_model_size_mb(str(self.mtp.gguf_path))
            available -= mtp_mb
            logger.debug("mtp companion %s: %d MB subtracted from VRAM",
                         self.mtp.stem, mtp_mb)

        if available <= 0:
            logger.warning("available VRAM <= 0 for %s (spare=%d, mmproj=%d)",
                           self.stem, spare_mb, mmproj_mb)
            return utils._MIN_CTX_SIZE

        design = design_ctx or self.frontmatter.get("context_length", utils._DEFAULT_CONTEXT_LENGTH)

        # Use --fit-target to let fit-params compute the max ctx for available VRAM
        model_mib, ctx_at_design_mib, compute_mib = self.fit_params(
            fit_bin=fit_bin,
            fit_target_mib=available,
            fit_ctx=design,
            cache_type=cache_type,
            parallel=parallel,
        )

        if model_mib == 0:
            # fit-params failed (binary missing, safetensors unsupported, etc.).
            # For safetensors we can still estimate from the header (tensor shapes),
            # which gives an accurate VRAM + KV-cache figure without benchmarking.
            assert self.gguf_path is not None
            if str(self.gguf_path).endswith(".safetensors"):
                if self._sf_estimate is None:
                    try:
                        self._sf_estimate = utils.estimate_safetensors(
                            self.gguf_path, cache_type
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"fit-params failed and safetensors estimate unavailable "
                            f"for {self.stem}: {e}"
                        ) from e
                    logger.warning(
                        "fit-params failed for %s; estimating VRAM from safetensors header",
                        self.stem,
                    )
                est_model_mib, est_kv_per_token_mib = self._sf_estimate
                model_mib = est_model_mib
                compute_mib = int(0.02 * est_model_mib) + 128
                ctx_at_design_mib = int(est_kv_per_token_mib * design)
            else:
                raise RuntimeError(
                    f"fit-params failed to measure VRAM for {self.stem}; "
                    f"cannot estimate a safe context size. Ensure llama-fit-params "
                    f"is available/built and the model format is supported "
                    f"(safetensors may be unsupported)."
                )

        remaining = available - model_mib - compute_mib
        if remaining <= 0:
            logger.warning("model + compute exceeds available VRAM for %s", self.stem)
            return utils._MIN_CTX_SIZE

        # If ctx_at_design fits, we can use design_ctx
        if ctx_at_design_mib <= remaining:
            return min(design, self.frontmatter.get("context_length", design))

        # Otherwise scale down: remaining / (ctx_at_design / design) = max_ctx
        max_ctx = (remaining * design) // ctx_at_design_mib
        max_ctx = (max_ctx // utils._CTX_ROUND_TO) * utils._CTX_ROUND_TO
        return max(max_ctx, utils._MIN_CTX_SIZE)

    def write_md(self, output_path: Path | None = None) -> None:
        """Serialize frontmatter back to .md sidecar."""
        path = output_path or self.md_path
        fm = {k: v for k, v in self.frontmatter.items() if k in self.FIELDS}
        content = "---\n" + yaml.dump(fm, sort_keys=False).rstrip() + "\n---\n"
        path.write_text(content, encoding="utf-8")
        path.chmod(0o644)
        logger.info("wrote sidecar: %s", path.name)

    @property
    def context_length(self) -> int:
        return self.frontmatter.get("context_length", utils._DEFAULT_CONTEXT_LENGTH)

    @property
    def name(self) -> str:
        return self.frontmatter.get("name", self.stem)

    @property
    def template(self) -> str:
        return self.frontmatter.get("template", "llama-server")

    @property
    def parallel(self) -> int:
        return self.frontmatter.get("parallel", 1)

    @property
    def cache_type(self) -> str:
        return self.frontmatter.get("cache_type", "q8_0")

    @property
    def cli_args(self) -> str:
        return self.frontmatter.get("cli_args", "")

    @property
    def reasoning(self) -> str | None:
        return self.frontmatter.get("reasoning")

    @property
    def allow_profiles(self) -> list[str] | None:
        return self.frontmatter.get("allow_profiles")

    @property
    def targets(self) -> list[str]:
        return self.frontmatter.get("targets", ["llama-server"])

    @property
    def description(self) -> str | None:
        return self.frontmatter.get("description")

    @property
    def mmproj_size_mb(self) -> int:
        if self.mmproj and self.mmproj.gguf_path:
            return utils.get_model_size_mb(str(self.mmproj.gguf_path))
        return 0

    def metadata(self) -> dict:
        """Return remaining frontmatter fields as metadata (not consumed by builder)."""
        return {k: v for k, v in self.frontmatter.items() if k not in self.FIELDS}


# Import yaml here to avoid circular import issues
import yaml
import subprocess