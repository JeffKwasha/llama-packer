"""Config builder and writer for llama-swap.

Pure logic — no CLI, no main(). Import from gen-config.py or use directly.

    from llama_swap import discover_models, build_models, build_config
    from llama_swap import YamlConfigWriter

    models = discover_models(models_dir)
    entries = build_models(models, profiles_cfg, template_vars)
    config  = build_config(entries)
    YamlConfigWriter().write(config, "config.yaml")
"""

from __future__ import annotations

import ast
import copy
import functools
import logging
import os
import re
import subprocess
import textwrap
from pathlib import Path
from abc import ABC, abstractmethod

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required — install with: pip install pyyaml") from None


logger = logging.getLogger(__name__)

# ── Defaults (can be overridden via profiles.yaml `defaults` section or env) ──
_DEFAULT_CONTEXT_LENGTH = 32768
_FALLBACK_VRAM_MB = 24576
_ROCMINFO_TIMEOUT = 10
_CTX_ROUND_TO = 8192
_MIN_CTX_SIZE = 4096

# VRAM reservation breakdown (MB)
_RESERVE_SYSTEM = 1024
_RESERVE_VIDEO = 1024

# MTP defaults (per-model overrides via frontmatter keys below)
_MTP_SPEC_TYPE = "draft-mtp"
_MTP_DRAFT_N_MAX = 2
_MTP_DRAFT_P_MIN = 0.75

_TARGET_TEMPLATES = {
    "llama-server": {
        "bin": "{{llama_bin}}",
        "model": "-m {{model_path}}",
        "ctx": "-c {{ctx_size}}",
        "parallel": "--parallel {{parallel}}",
        "cache_type": "--cache-type-k {{cache_type}} --cache-type-v {{cache_type}}",
        "mtp": "--spec-type {{mtp_spec_type}} --spec-draft-n-max {{mtp_n_max}} --spec-draft-p-min {{mtp_p_min}}",
        "mmproj": "--mmproj {{mmproj_path}}",
        "extra": "{{extra_args}}",
    },
    "vllm": {
        "bin": "vllm serve",
        "model": "{{model_path}}",
        "ctx": "--max-model-len {{ctx_size}}",
        "parallel": "--max-num-batched-tokens {{ctx_size}}",
        "cache_type": "--kv-cache-dtype {{cache_type}}",
        "mtp": "--speculative-model {{model_path}}",
        "mmproj": "--mmproj-path {{mmproj_path}}",
        "extra": "{{extra_args}}",
    }
}

# Frontmatter keys consumed by the builder (not passed through to metadata)
_CONSUMED_KEYS = frozenset({
    "name", "template", "context_length", "description", "cli_args", "model",
    "attention", "kv_cache", "tool_args", "speculative", "mmproj",
    "mtp", "mtp_spec_type", "mtp_draft_n_max", "mtp_draft_p_min",
    "targets", "allow_profiles", "reasoning",
})

SAMPLING_KEYS = frozenset({"temperature", "top_p", "top_k", "min_p", "pres_pen"})

_RE_Q_SUFFIX = re.compile(r"[-_.][iI]?Q\d[_A-Z0-9]*$")
_RE_V_SUFFIX = re.compile(r"[-_][vV]\d.*")
# Matches memory values like 2.3G, 123m, 512k, or bare numbers
_MEM_RE = re.compile(r"^([\d.]+)\s*([kKmMgG]?)$")


def resolve_spare_mb(spare_str: str, vram_mb: int) -> int:
    """Parse --spare value to MB.

    Suffixed values (``2G``, ``512m``, ``64k``) resolve directly.
    Bare numbers auto-detect: if < 3 × VRAM(GB) they're treated as GB,
    otherwise as MB (safe for sub-GB values like ``512``).
    """
    m = _MEM_RE.match(str(spare_str).strip())
    if not m:
        logger.warning("invalid spare value %r, using 0", spare_str)
        return 0
    num = float(m.group(1))
    unit = m.group(2).lower()
    if unit == "g":
        return int(num * 1024)
    if unit == "m":
        return int(num)
    if unit == "k":
        return max(1, int(num // 1024))
    vram_gb = vram_mb / 1024
    if num < 3 * vram_gb:
        return int(num * 1024)
    return int(num)


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text.strip("-")


def parse_context_length(value) -> int:
    if isinstance(value, int):
        return value
    s = str(value).lower().strip()
    m = re.match(r"^(\d+)\s*([km]?)$", s)
    if not m:
        logger.warning("invalid context_length %r, using %d", value, _DEFAULT_CONTEXT_LENGTH)
        return _DEFAULT_CONTEXT_LENGTH
    val = int(m.group(1))
    suffix = m.group(2)
    if suffix == "k":
        return val * 1024
    if suffix == "m":
        return val * 1024 * 1024
    return val


def infer_param_count(stem: str) -> str:
    m = re.search(r"(\d+\.?\d*)\s*[Bb]", stem)
    return (m.group(1) + "B") if m else ""


def infer_quantization(stem: str) -> str:
    m = re.search(r"(Q[2468]_[KMSXL]?|IQ\d_[A-Z0-9]+)", stem)
    return m.group(1) if m else ""


def _gguf_family(stem: str) -> str:
    s = _RE_Q_SUFFIX.sub("", stem)
    s = _RE_V_SUFFIX.sub("", s)
    return s


# ── VRAM detection (cached) ──

def _rocminfo_kb_to_mb(text: str) -> int:
    m = re.search(r"Size:\s*(\d+)\s*\(.*?\)\s*(\w+)", text)
    if not m:
        return 0
    val = int(m.group(1))
    unit = m.group(2).upper()
    if unit == "KB":
        return val // 1024
    if unit == "MB":
        return val
    if unit == "GB":
        return val * 1024
    return 0


@functools.lru_cache(maxsize=1)
def get_vram_mb() -> int:
    try:
        out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=_ROCMINFO_TIMEOUT)
        sections = out.stdout.split("Pool Info:")
        coarse_pools: list[int] = []
        all_pools: list[int] = []

        for section in sections:
            for block in re.split(r"\n\s+Pool\s+\d+\s*\n", section):
                mb = _rocminfo_kb_to_mb(block)
                if mb == 0:
                    continue
                all_pools.append(mb)
                if "COARSE" in block:
                    coarse_pools.append(mb)

        if coarse_pools:
            coarse_pools.sort()
            return coarse_pools[0]
        if all_pools:
            return max(all_pools)
    except Exception:
        logger.debug("rocminfo failed, using fallback VRAM", exc_info=True)
    return _FALLBACK_VRAM_MB


@functools.lru_cache(maxsize=128)
def get_model_size_mb(model_path: str) -> int:
    return Path(model_path).stat().st_size // (1024 ** 2)


# DEPRECATED: Use model_cfg.model.Model.calc_ctx() which uses llama-fit-params
_BYTES_PER_TOKEN = 20  # retained for backward compat only — catastrophically wrong

def calculate_ctx_size(model_size_mb: int, vram_total_mb: int, parallel: int = 1,
                       spare_mb: int = 0, mtp_mb: int = 0, mmproj_mb: int = 0) -> int:
    available = vram_total_mb - _RESERVE_SYSTEM - _RESERVE_VIDEO - spare_mb
    remaining = available - model_size_mb - mtp_mb - mmproj_mb
    if remaining <= 0:
        logger.warning("model %d MB + companions (%d MB mtp, %d MB mmproj) may exceed VRAM %d MB",
                       model_size_mb, mtp_mb, mmproj_mb, available)
        return _MIN_CTX_SIZE
    max_tokens = (remaining * 1024) // _BYTES_PER_TOKEN
    per_slot = max_tokens // parallel
    ctx = (per_slot // _CTX_ROUND_TO) * _CTX_ROUND_TO
    return max(ctx, _MIN_CTX_SIZE)


# ── mmproj detection ──

def detect_mmproj(path: Path, frontmatter: dict | None = None) -> str | None:
    if "mmproj" in path.stem.lower():
        return None

    # 1. Check frontmatter mmproj field first
    if frontmatter and frontmatter.get("mmproj"):
        companion = path.parent / frontmatter["mmproj"]
        if companion.exists():
            return str(companion)

    # 2. Fallback: fuzzy match by model family (exact prefix match)
    family = _gguf_family(path.stem).lower()
    for f in sorted(path.parent.glob("*mmproj*.gguf")):
        name = f.stem.lower()
        mmproj_family = _gguf_family(f.stem).lower()
        # Only match if the mmproj family is a prefix of the model family or vice versa
        if mmproj_family.startswith(family) or family.startswith(mmproj_family):
            return str(f)
    return None


# ── MTP detection ──

def model_has_mtp(model_path: str, frontmatter: dict) -> bool:
    stem = Path(model_path).stem.lower()
    if "mtp" in stem:
        return True
    if frontmatter.get("mtp"):
        return True
    # Check if frontmatter points to an MTP companion file
    speculative = frontmatter.get("speculative", "")
    if speculative and "mtp" in Path(speculative).stem.lower():
        return True
    return False


def _build_mtp_args(model_path: Path, fm: dict, models_dir: Path | None = None) -> tuple[str, bool, int, str]:
    """Returns (flags, enabled, draft_n_max, draft_model_path)."""
    if not model_has_mtp(str(model_path), fm):
        stem = model_path.stem.lower()
        family = _gguf_family(stem)
        if "mtp" not in stem and "mtp" not in family:
            # Only look for MTP variants in the same model family, not unrelated files
            has_mtp_variant = any(
                "mtp" in f.stem.lower() and _gguf_family(f.stem.lower()) == family
                for f in model_path.parent.glob("*.gguf")
            )
            if has_mtp_variant:
                logger.info("mtp: variant available — download MTP GGUF from HF for %s", fm["name"])
        return "", False, 0, ""

    # Determine spec_type: use draft-simple for companion files, draft-mtp for baked-in
    spec_type = fm.get("mtp_spec_type", _MTP_SPEC_TYPE)
    n_max = int(fm.get("mtp_draft_n_max", _MTP_DRAFT_N_MAX))
    p_min = fm.get("mtp_draft_p_min", _MTP_DRAFT_P_MIN)

    # Find the MTP companion file
    draft_model_path = ""
    
    # Check if this is a baked-in MTP model (has mtp: true in frontmatter)
    has_baked_mtp = fm.get("mtp") is True
    
    if not has_baked_mtp and "mtp" in model_path.stem.lower():
        # This IS an MTP companion file - don't add MTP flags to it
        return "", False, 0, ""
    
    # Look for companion via frontmatter 'speculative' field
    speculative = fm.get("speculative", "")
    if speculative:
        # Search in models_dir first, then relative to model_path
        search_dirs = []
        if models_dir:
            search_dirs.append(models_dir)
        search_dirs.append(model_path.parent)
        
        for search_dir in search_dirs:
            companion = search_dir / speculative
            if companion.is_file():
                draft_model_path = str(companion)
                break
            # Also check parent directories of search_dir
            for parent in list(search_dir.parents[:3]):
                companion = parent / speculative
                if companion.is_file():
                    draft_model_path = str(companion)
                    break
            if draft_model_path:
                break
    
    if not draft_model_path:
        # Fallback: look for MTP companion in same directory
        family = _gguf_family(model_path.stem)
        for f in model_path.parent.glob("*.gguf"):
            if "mtp" in f.stem.lower() and _gguf_family(f.stem.lower()) == family:
                draft_model_path = str(f)
                break

    logger.info("mtp: enabled for %s (n_max=%d, draft=%s)",
                fm["name"], n_max, Path(draft_model_path).name if draft_model_path else "N/A")
    if draft_model_path:
        # Companion file: server infers type from --model-draft, no --spec-type needed
        flags = f"--model-draft {draft_model_path} --spec-draft-n-max {n_max}"
    else:
        # Baked-in MTP: need --spec-type draft-mtp
        flags = f"--spec-type {spec_type} --spec-draft-n-max {n_max} --spec-draft-p-min {p_min}"
    return flags, True, n_max, draft_model_path


# ── Version resolver ──

def get_available_versions(base_dir: Path) -> list[int]:
    return sorted(
        int(d.name.removeprefix("llama-b"))
        for d in base_dir.glob("llama-b[0-9]*")
        if d.is_dir()
    )


def find_bin_dir(version: str, base_dir: Path) -> str:
    env = os.environ.get("LLAMA_BIN_DIR")
    if env:
        return env
    if version == "latest":
        versions = get_available_versions(base_dir)
        if not versions:
            raise SystemExit("error: no llama-b#### directory found")
        return f"llama-b{versions[-1]}"
    if (base_dir / f"llama-b{version}").is_dir():
        return f"llama-b{version}"
    avail = " ".join(str(v) for v in get_available_versions(base_dir))
    raise SystemExit(f"error: version {version} not found\n  available: {avail}")


# ── Frontmatter ──

def parse_frontmatter(path: Path) -> dict:
    try:
        content = path.read_text()
    except Exception:
        return {}
    m = re.match(r"^---\s*\n(.*?)\n(?:---|\.\.\.)", content, re.DOTALL)
    if not m:
        return {}
    try:
        return yaml.safe_load(m.group(1)) or {}
    except Exception:
        return {}


_CL_RE = re.compile(r"^context_limit_\d+G$")
# ── Model discovery ──

def generate_stub_md(md_path: Path, model_path: Path) -> dict:
    stem = model_path.stem
    fm = {
        "name": stem,
        "template": "llama-server",
        "parameters": infer_param_count(stem),
        "context_length": _DEFAULT_CONTEXT_LENGTH,
        "quantization": infer_quantization(stem),
    }
    body = textwrap.dedent(f"""\
        ---
        name: {fm['name']!r}
        template: llama-server
        parameters: {fm['parameters']!r}
        context_length: {fm['context_length']}
        quantization: {fm['quantization']!r}
        ---

        # {stem}

        """)
    md_path.write_text(body)
    return fm


def _is_mtp_companion(stem: str) -> bool:
    """Check if a GGUF file is an MTP companion (not a main model)."""
    s = stem.lower()
    return bool(re.search(r'(?:^mtp-|\.mtp$|-mtp$)', s))


def discover_models(
    models_dir: Path,
    *,
    generate_stubs: bool = True,
    extra_dirs: list[str] | None = None,
) -> list[dict]:
    known_md: set[Path] = set()
    models: list[dict] = []

    for md_path in sorted(models_dir.rglob("*.md")):
        fm = parse_frontmatter(md_path)
        if not fm.get("name"):
            continue
        known_md.add(md_path.resolve())
        parent = md_path.parent

        gguf = parent / f"{md_path.stem}.gguf"
        if gguf.is_file():
            # Skip MTP companion files (e.g., model.mtp.gguf) - they're not standalone models
            if _is_mtp_companion(gguf.stem):
                logger.debug("skipping MTP companion: %s", gguf.name)
                continue
            models.append({"frontmatter": fm, "model_path": str(gguf.resolve()), "model_type": "gguf"})
            continue

        if parent != models_dir:
            safetensors = list(parent.glob("*.safetensors"))
            if safetensors:
                models.append({"frontmatter": fm, "model_path": str(parent.resolve()), "model_type": "safetensors"})
                continue

        file_ref = fm.get("model")
        if file_ref:
            model_file = parent / file_ref
            if not model_file.is_file():
                model_file = models_dir / file_ref
            if model_file.is_file():
                models.append({"frontmatter": fm, "model_path": str(model_file.resolve()),
                               "model_type": fm.get("template", "unknown")})
                continue

        logger.debug("no model found for %s", md_path)

    if generate_stubs:
        dirs_to_scan = [models_dir] + [models_dir / d for d in (extra_dirs or [])]
        for scan_dir in dirs_to_scan:
            if not scan_dir.is_dir():
                continue
            for gguf in sorted(scan_dir.glob("*.gguf")):
                if "mmproj" in gguf.stem.lower():
                    continue
                if _is_mtp_companion(gguf.stem):
                    continue
                md_path = gguf.with_suffix(".md").resolve()
                if md_path not in known_md and gguf.is_file():
                    fm = generate_stub_md(md_path, gguf)
                    models.append({"frontmatter": fm, "model_path": str(gguf.resolve()), "model_type": "gguf"})
                    logger.info("stub: %s", md_path.name)

    return models


# ── Template resolution ──

def _eval_expr(expr: str, base_val: float) -> float:
    try:
        tree = ast.parse(expr.strip(), mode="eval")

        def _walk(node):
            if isinstance(node, ast.Expression):
                return _walk(node.body)
            if isinstance(node, ast.BinOp):
                l, r = _walk(node.left), _walk(node.right)
                op = node.op
                if isinstance(op, ast.Mult): return l * r
                if isinstance(op, ast.Add): return l + r
                if isinstance(op, ast.Sub): return l - r
                if isinstance(op, ast.Div): return l / r
                if isinstance(op, ast.FloorDiv): return l // r
                raise ValueError(f"unsupported op: {type(op).__name__}")
            if isinstance(node, ast.UnaryOp):
                val = _walk(node.operand)
                return -val if isinstance(node.op, ast.USub) else +val
            if isinstance(node, ast.Name):
                if node.id == "base":
                    return float(base_val)
                raise ValueError(f"unknown variable: {node.id}")
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                args = [_walk(a) for a in node.args]
                if node.func.id == "max": return max(*args)
                if node.func.id == "min": return min(*args)
            raise ValueError(f"unsupported: {type(node).__name__}")

        return _walk(tree)
    except Exception as e:
        logger.warning("can't evaluate %r: %s", expr, e)
        return base_val


def resolve_params(profile_overrides: dict | None, defaults: dict) -> dict:
    result = dict(defaults)
    if not profile_overrides:
        return result
    for key, val in profile_overrides.items():
        if isinstance(val, str) and "base" in val:
            base_val = defaults.get(key, 1.0)
            try:
                base_val = float(base_val)
            except (TypeError, ValueError):
                base_val = 1.0
            result[key] = _eval_expr(val, base_val)
        else:
            result[key] = val
    return result


def resolve_template(cmd_template: str, variables: dict) -> str:
    def _repl(m):
        val = variables.get(m.group(1))
        return str(val) if val is not None else m.group(0)
    return re.sub(r"\{\{(\w+)\}\}", _repl, cmd_template)


# ── Profile / reasoning helpers ──

_RE_REASONING = re.compile(r"--reasoning\b(?:\s+\S+)?", re.IGNORECASE)


def _resolve_reasoning_cli_args(cli_args: str, reasoning_val) -> list[tuple[str, str]]:
    """Return list of (cli_args, suffix) tuples for each reasoning variant.

    reasoning_val:
      None   — no frontmatter key; use cli_args as-is (legacy --reasoning auto in cli_args)
      True   — ensure --reasoning auto
      False  — remove any --reasoning, add --reasoning off
      "both" — generate two variants: --reasoning auto and --reasoning off
      any other string — treat as the literal value for --reasoning
    """
    if reasoning_val is None:
        return [(cli_args, "")]

    if reasoning_val is True:
        args = _RE_REASONING.sub("--reasoning auto", cli_args).strip()
        if "--reasoning" not in args:
            args = (args + " --reasoning auto").strip()
        return [(args, "")]

    if reasoning_val is False:
        args = _RE_REASONING.sub("--reasoning off", cli_args).strip()
        if "--reasoning" not in args:
            args = (args + " --reasoning off").strip()
        return [(args, ".noreason")]

    if reasoning_val == "both":
        # Variant 1: reasoning auto
        args_on = _RE_REASONING.sub("--reasoning auto", cli_args).strip()
        if "--reasoning" not in args_on:
            args_on = (args_on + " --reasoning auto").strip()
        # Variant 2: reasoning off
        args_off = _RE_REASONING.sub("--reasoning off", cli_args).strip()
        if "--reasoning" not in args_off:
            args_off = (args_off + " --reasoning off").strip()
        return [(args_on, ""), (args_off, ".noreason")]

    # Literal string — use as the value for --reasoning
    args = _RE_REASONING.sub(f"--reasoning {reasoning_val}", cli_args).strip()
    if "--reasoning" not in args:
        args = (args + f" --reasoning {reasoning_val}").strip()
    return [(args, "")]


def _filter_profiles(profile_list: dict, allow_profiles) -> list[tuple[str, dict]]:
    """Filter profiles according to allow_profiles frontmatter value.

    allow_profiles:
      None/True — return all profiles
      False     — return empty list (no profiles)
      str       — treat as regex; only matching profile names included
    """
    if allow_profiles is False:
        return []

    if allow_profiles is None or allow_profiles is True:
        return list(profile_list.items())

    if isinstance(allow_profiles, str):
        try:
            pattern = re.compile(allow_profiles)
        except re.error:
            logger.warning("invalid allow_profiles regex %r, returning all profiles", allow_profiles)
            return list(profile_list.items())
        return [(pname, pover) for pname, pover in profile_list.items() if pattern.search(pname)]

    return list(profile_list.items())


# ── Config builder ──

def _resolve_model(model: dict) -> dict | None:
    fm = copy.deepcopy(model["frontmatter"])
    model_path = Path(model["model_path"])
    model_type = model["model_type"]

    return {
        "frontmatter": fm,
        "model_path": model_path,
        "model_type": model_type,
    }


def _build_entry(
    resolved: dict,
    parallel: int,
    cache_type: str,
    profiles_group: list[tuple[str, dict]],
    profiles_defaults: dict,
    template_vars: dict,
    mmproj_path: str | None,
    mtp_arg_str: str,
    mtp_enabled: bool,
    mtp_n_max: int,
    context_length: int,
    ctx_size: int,
) -> tuple[str, dict]:
    fm = resolved["frontmatter"]
    model_path = resolved["model_path"]
    base_id = slugify(fm.get("name", model_path.stem))

    # Build command string for primary target
    targets = fm.get("targets", ["llama-server"])
    target = targets[0] if targets else "llama-server"
    t_conf = _TARGET_TEMPLATES.get(target, _TARGET_TEMPLATES["llama-server"])
    parts = []
    parts.append(resolve_template(t_conf["bin"], {"llama_bin": template_vars.get("llama_bin", "")}))
    parts.append("--port ${PORT}")
    parts.append(resolve_template(t_conf["model"], {"model_path": str(model_path)}))
    if ctx_size:
        parts.append(resolve_template(t_conf["ctx"], {"ctx_size": str(ctx_size)}))
    if parallel > 1:
        parts.append(resolve_template(t_conf["parallel"], {"parallel": str(parallel)}))
    parts.append(resolve_template(t_conf["cache_type"], {"cache_type": cache_type}))
    if mtp_enabled and mtp_arg_str:
        parts.append(mtp_arg_str)
    if mmproj_path:
        parts.append(resolve_template(t_conf["mmproj"], {"mmproj_path": mmproj_path}))
    extra = fm.get("cli_args", "").strip()
    if extra:
        parts.append(extra)
    cmd_str = " ".join(parts).strip()

    set_params: dict[str, dict] = {}
    for pname, resolved_prof in profiles_group:
        overrides = {}
        for k in SAMPLING_KEYS:
            val, dval = resolved_prof.get(k), profiles_defaults.get(k)
            if val is not None and dval is not None and val != dval:
                overrides[k] = round(val, 6) if isinstance(val, float) else val
        if overrides:
            key = "${MODEL_ID}" if pname == "default" else f"${{MODEL_ID}}:{pname}"
            set_params[key] = overrides

    names = [p[0] for p in profiles_group]
    has_default = "default" in names
    entry_id = base_id if (has_default or len(profiles_group) > 1) else f"{base_id}.{names[0]}"

    metadata = {k: copy.deepcopy(v) for k, v in fm.items()
                if k not in _CONSUMED_KEYS
                and k not in ("parameters", "quantization", "template", "description")
                and not _CL_RE.match(k)
                and v}
    metadata["mtp_enabled"] = mtp_enabled
    if mtp_enabled:
        metadata["mtp_draft_max"] = mtp_n_max

    entry: dict = {"cmd": cmd_str}
    if set_params:
        entry["setParamsByID"] = set_params
    if fm.get("name"):
        entry["name"] = fm["name"]
    if fm.get("description"):
        entry["description"] = fm["description"]
    if metadata:
        entry["metadata"] = metadata

    return entry_id, entry


def build_models(models: list, profiles_cfg: dict, template_vars: dict,
                 spare: str | None = None, max_context: int | None = None,
                 models_dir: Path | None = None) -> dict:
    defaults = profiles_cfg.get("defaults", {})
    profile_list = profiles_cfg.get("profiles", {})
    vram_total = 0
    vram_gb = 0
    spare_mb = 0
    entries: dict[str, dict] = {}

    for model in models:
        resolved = _resolve_model(model)
        if resolved is None:
            continue

        fm = resolved["frontmatter"]
        model_path = resolved["model_path"]
        model_type = resolved["model_type"]
        context_length = fm.get("context_length", _DEFAULT_CONTEXT_LENGTH)
        model_size = get_model_size_mb(str(model_path)) if model_type == "gguf" else 0

        if model_type == "gguf" and vram_total == 0:
            vram_total = get_vram_mb()
            vram_gb = vram_total // 1024
            if spare is not None:
                spare_mb = resolve_spare_mb(spare, vram_total)
                logger.info("spare: %d MB reserved (vram: %d MB = %d GB)", spare_mb, vram_total, vram_gb)

        # Check for VRAM-specific context cap in sidecar
        ctx_override = None
        if vram_gb > 0:
            cl_key = f"context_limit_{vram_gb}G"
            if cl_key in fm:
                override_val = fm[cl_key]
                if override_val and isinstance(override_val, int):
                    ctx_override = override_val
                    logger.info("ctx: using %s=%d for %s", cl_key, ctx_override, fm["name"])

        mmproj_path = None
        mmproj_size_mb = 0
        mtp_enabled = False
        mtp_n_max = _MTP_DRAFT_N_MAX
        mtp_spec_type = _MTP_SPEC_TYPE
        mtp_p_min = _MTP_DRAFT_P_MIN
        mtp_size_mb = 0

        if model_type == "gguf":
            mmproj_path = detect_mmproj(model_path, fm)
            if mmproj_path:
                mmproj_size_mb = get_model_size_mb(mmproj_path)
                logger.info("mmproj: %s (%d MB) for %s", Path(mmproj_path).name, mmproj_size_mb, fm["name"])
            else:
                # Check if mmproj is configured in frontmatter but file is missing
                fm_mmproj = fm.get("mmproj")
                if fm_mmproj:
                    companion = model_path.parent / fm_mmproj
                    if not companion.exists():
                        logger.info("mmproj: configured %s missing for %s", fm_mmproj, fm["name"])

            mtp_arg_str, mtp_enabled, mtp_n_max, mtp_draft_path = _build_mtp_args(model_path, fm, models_dir)

            # Check for configured speculative (MTP companion) file
            speculative = fm.get("speculative")
            if speculative:
                companion = model_path.parent / speculative
                if companion.exists():
                    mtp_size_mb = get_model_size_mb(str(companion))
                    logger.info("mtp: %s (%d MB) for %s", companion.name, mtp_size_mb, fm["name"])
                else:
                    logger.info("mtp: companion %s missing for %s (expected at %s)",
                               speculative, fm["name"], companion)
                    mtp_size_mb = 2048  # Estimate 2GB for baked-in MTP
            elif mtp_enabled:
                mtp_size_mb = 2048  # Estimate 2GB for baked-in MTP

        # Log capped context once per model (before profile groups loop)
        if model_type == "gguf":
            if ctx_override is not None:
                sample_ctx = ctx_override
            else:
                sample_ctx = calculate_ctx_size(model_size, vram_total, 1, spare_mb, mtp_size_mb, mmproj_size_mb)
            sample_ctx = min(sample_ctx, context_length)
            if sample_ctx < context_length and not ctx_override:
                logger.info("ctx: capped %s at %d (design %d)", fm["name"], sample_ctx, context_length)

        # Filter profiles by allow_profiles, resolve reasoning variants
        filtered_profiles = _filter_profiles(profile_list, fm.get("allow_profiles"))
        cli_args_base = fm.get("cli_args", "")
        reasoning_variants = _resolve_reasoning_cli_args(cli_args_base, fm.get("reasoning"))

        for variant_cli_args, variant_suffix in reasoning_variants:
            # Deep-copy frontmatter so each variant has independent cli_args
            variant_fm = copy.deepcopy(resolved["frontmatter"])
            variant_fm["cli_args"] = variant_cli_args

            groups: dict[tuple, list] = {}
            for pname, pover in filtered_profiles:
                resolved_profile = resolve_params(pover, defaults)
                # Frontmatter parallel overrides profile parallel
                parallel_val = variant_fm.get("parallel", resolved_profile.get("parallel", 1))
                gkey = (int(parallel_val), str(resolved_profile.get("cache_type", "q8_0")))
                groups.setdefault(gkey, []).append((pname, resolved_profile))

            # If no profiles matched, generate a single entry with defaults
            if not groups:
                groups = {
                    (int(variant_fm.get("parallel", 1)), str(defaults.get("cache_type", "q8_0"))): [
                        ("default", dict(defaults)),
                    ],
                }

            for (parallel, cache_type), profiles_group in groups.items():
                if model_type == "gguf":
                    if ctx_override is not None:
                        ctx_size = ctx_override
                    else:
                        ctx_size = calculate_ctx_size(model_size, vram_total, parallel, spare_mb, mtp_size_mb, mmproj_size_mb)
                    ctx_size = min(ctx_size, context_length)
                    if max_context is not None:
                        ctx_size = min(ctx_size, max_context)
                else:
                    ctx_size = context_length

                variant_resolved = dict(resolved, frontmatter=variant_fm)
                entry_id, entry = _build_entry(
                    variant_resolved, parallel, cache_type, profiles_group,
                    defaults, template_vars,
                    mmproj_path, mtp_arg_str, mtp_enabled, mtp_n_max,
                    context_length, ctx_size,
                )
                # Append reasoning variant suffix to entry id
                if variant_suffix:
                    entry_id = entry_id + variant_suffix
                entries[entry_id] = entry

    return entries


# ── Config assembly (pure data, no I/O) ──

def build_config(entries: dict, globals_map: dict | None = None) -> dict:
    config: dict = {}
    if globals_map:
        config.update(globals_map)
    config["models"] = {
        eid: entries[eid]
        for eid in sorted(entries, key=lambda e: (e.count("."), e))
    }
    return config
class ConfigWriter(ABC):
    """Abstract base for config output formatters.

    Subclasses must implement ``_serialize(config) -> str``.
    """

    @abstractmethod
    def _serialize(self, config: dict) -> str:
        ...

    def write(self, config: dict, dest: Path | str) -> None:
        text = self._serialize(config)
        with open(dest, "w") as f:
            f.write(text)

    def dumps(self, config: dict) -> str:
        return self._serialize(config)


# ── YAML — llama-swap native format ──

def _normalise_cmd(cmd: str) -> str:
    return cmd.rstrip("\n") + "\n"


class _Dumper(yaml.SafeDumper):
    pass


def _represent_str(dumper, data):
    if "\n" in data:
        data = _normalise_cmd(data)
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_Dumper.add_representer(str, _represent_str)


class YamlConfigWriter(ConfigWriter):
    """Write config dict as llama-swap-compatible YAML.

    - Multiline ``cmd`` values use literal block scalar ``|``
    - ``---`` document-start marker
    - Model entries sorted: base IDs first, then profile variants
    - ``setParamsByID`` keys use ``${MODEL_ID}`` macro patterns
    """

    def _serialize(self, config: dict) -> str:
        lines = ["---\n"]
        yaml_str = yaml.dump(
            config,
            Dumper=_Dumper,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )
        lines.append(yaml_str)
        return "".join(lines)


# ── INI — llama-server router mode ──

class IniConfigWriter(ConfigWriter):
    """Write config dict as llama-server router INI.

    Each model key becomes ``[model_id]`` section header.
    Top-level entry keys are emitted verbatim as ``key = value`` pairs.
    Nested dicts (``metadata``) are flattened one level.
    """

    def _serialize(self, config: dict) -> str:
        buf: list[str] = ["# llama-server router config (auto-generated)\n"]
        models = config.get("models", {})
        for model_id, entry in sorted(models.items(), key=lambda x: (x[0].count("."), x[0])):
            buf.append(f"\n[{model_id}]\n")
            for key, val in entry.items():
                if isinstance(val, dict):
                    for nk, nv in val.items():
                        buf.append(f"{nk} = {nv}\n")
                elif isinstance(val, list):
                    buf.append(f"{key} = {','.join(str(v) for v in val)}\n")
                else:
                    buf.append(f"{key} = {val}\n")
        return "".join(buf)
