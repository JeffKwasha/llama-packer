# model_cfg/utils.py
"""Shared utilities for model-cfg."""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# ── Defaults (can be overridden via profiles.yaml `defaults` section or env) ──
_DEFAULT_CONTEXT_LENGTH = 32768
_CTX_ROUND_TO = 8192
_MIN_CTX_SIZE = 4096

# VRAM reservation breakdown (MB)
_RESERVE_SYSTEM = 1024
_RESERVE_VIDEO = 1024

# MTP defaults (per-model overrides via frontmatter keys)
_MTP_SPEC_TYPE = "draft-mtp"
_MTP_DRAFT_N_MAX = 2
_MTP_DRAFT_P_MIN = 0.75

# Templates for llama-server and vllm
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
    },
}

# Frontmatter keys consumed by the builder (not passed through to metadata)
_CONSUMED_KEYS = frozenset({
    "name", "template", "context_length", "description", "cli_args", "model",
    "attention", "kv_cache", "tool_args", "speculative", "mmproj",
    "mtp", "mtp_spec_type", "mtp_draft_n_max", "mtp_draft_p_min",
    "targets", "allow_profiles", "reasoning", "spare",
})

SAMPLING_KEYS = frozenset({"temperature", "top_p", "top_k", "min_p", "pres_pen"})

_RE_Q_SUFFIX = re.compile(r"[-_.][iI]?Q\d[_A-Z0-9]*$")
_RE_V_SUFFIX = re.compile(r"[-_][vV]\d.*")
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


def parse_context_length(s: str) -> int:
    """Parse context length with k/m suffixes."""
    s = str(s).strip().lower()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1024)
    if s.endswith("m"):
        return int(float(s[:-1]) * 1024 * 1024)
    return int(s)


def infer_param_count(stem: str) -> str | None:
    """Extract parameter count from stem (e.g., '7B', '13B')."""
    m = re.search(r"(\d+(?:\.\d+)?[BbMm])", stem)
    return m.group(1).upper() if m else None


def infer_quantization(stem: str) -> str | None:
    """Extract quantization from stem (e.g., 'Q4_K', 'Q8_0')."""
    m = re.search(r"(?:[-_.](?:i)?(?P<quant>Q\d+(?:_[A-Z0-9]+){0,2}))", stem)
    return m.group("quant") if m else None


def _gguf_family(stem: str) -> str:
    """Extract GGUF family base name (strip quant, version, MTP suffixes)."""
    s = stem
    s = _RE_Q_SUFFIX.sub("", s)
    s = _RE_V_SUFFIX.sub("", s)
    return s


def _detect_vram_amd() -> int | None:
    """Detect VRAM via AMD tools. Returns MiB or None."""
    # amd-smi (structured JSON, most reliable on AMD GPUs)
    try:
        out = subprocess.run(
            ["amd-smi", "metric", "-m", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            data = json.loads(out.stdout)
            total = data["gpu_data"][0]["mem_usage"]["total_vram"]["value"]
            return int(total)
    except Exception:
        pass

    # sysfs (kernel-reported, no userspace tool needed)
    try:
        for p in Path("/sys/class/drm").glob("card*/device/mem_info_vram_total"):
            return int(p.read_text().strip()) // (1024 ** 2)
    except Exception:
        pass

    # rocminfo (ROCm runtime)
    try:
        out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10)
        for line in out.stdout.splitlines():
            if "Global Memory" in line or "VRAM" in line:
                m = re.search(r"(\d+)\s*([GM])B", line)
                if m:
                    val = int(m.group(1))
                    return val * 1024 if m.group(2) == "G" else val
    except Exception:
        pass

    return None


def _detect_vram_nvidia() -> int | None:
    """Detect VRAM via nvidia-smi. Returns MiB or None.

    On unified memory systems (e.g. NVIDIA GB10/Grace), nvidia-smi reports
    'Not Supported' for memory — returns None so caller falls back to system RAM.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            val = out.stdout.strip().splitlines()[0].strip()
            if val in ("N/A", "[N/A]", "Not Supported", ""):
                return None  # unified memory — caller should use system RAM
            return int(val)
    except Exception:
        pass
    return None


def _detect_system_ram_mb() -> int:
    """Get total system RAM in MiB (for unified memory fallback)."""
    try:
        out = subprocess.run(["free", "-m"], capture_output=True, text=True, timeout=5)
        for line in out.stdout.splitlines():
            if line.startswith("Mem:"):
                return int(line.split()[1])
    except Exception:
        pass
    try:
        return int(Path("/proc/meminfo").read_text().split()[1]) // 1024
    except Exception:
        pass
    raise SystemExit("error: could not determine system RAM")


def get_vram_mb() -> int:
    """Detect VRAM budget in MiB.

    Tries GPU-specific tools first. On unified memory systems (NVIDIA GB10,
    Apple Silicon, Intel integrated), falls back to system RAM with a warning.

    Raises SystemExit only if no detection method works at all.
    """
    # Try discrete GPU detection
    vram = _detect_vram_amd()
    if vram is not None:
        logger.info("vram: %d MiB (discrete GPU)", vram)
        return vram

    vram = _detect_vram_nvidia()
    if vram is not None:
        logger.info("vram: %d MiB (nvidia-smi)", vram)
        return vram

    # NVIDIA returned None — likely unified memory. Check if nvidia-smi exists at all.
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and "NVIDIA" in out.stdout:
            # nvidia-smi works but reported N/A — unified memory
            ram = _detect_system_ram_mb()
            available = ram // 2
            logger.warning("unified memory detected (nvidia-smi reports N/A) — "
                           "system RAM %d MiB, using 50%%: %d MiB", ram, available)
            return available
    except Exception:
        pass

    # No GPU tools worked at all — try system RAM as last resort
    # (covers Intel integrated, Apple Silicon, or unknown)
    # Apply 50% discount: conservative for Strix HALO, DGX SPARK (GB10)
    try:
        ram = _detect_system_ram_mb()
        available = ram // 2
        logger.warning("no GPU detection tool available — system RAM %d MiB, using 50%%: %d MiB",
                       ram, available)
        return available
    except Exception:
        pass

    raise SystemExit(
        "error: could not detect VRAM — no GPU tools found (amd-smi, nvidia-smi, rocminfo)\n"
        "use --vram to specify (e.g. --vram 32G)"
    )


@functools.lru_cache(maxsize=128)
def get_model_size_mb(model_path: str) -> int:
    """Get model file size in MB."""
    return Path(model_path).stat().st_size // (1024 ** 2)


def get_available_versions(base_dir: Path) -> list[int]:
    """List available llama-b#### version numbers under base_dir."""
    return sorted(
        int(d.name.removeprefix("llama-b"))
        for d in base_dir.glob("llama-b[0-9]*")
        if d.is_dir()
    )


def find_bin_dir(version: str, base_dir: Path) -> str:
    """Resolve llama-server binary directory."""
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


def parse_frontmatter(md_path: Path) -> dict:
    """Parse YAML frontmatter from .md file."""
    content = md_path.read_text(encoding="utf-8")
    if not content.startswith("---"):
        return {}
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        fm = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as e:
        logger.warning("failed to parse frontmatter in %s: %s", md_path, e)
        fm = {}
    return fm


def generate_stub_md(md_path: Path, model_file: Path) -> dict:
    """Generate stub .md sidecar for an orphan model file (.gguf or .safetensors)."""
    stem = model_file.stem
    fm = {
        "name": stem,
        "parameters": infer_param_count(stem),
        "context_length": _DEFAULT_CONTEXT_LENGTH,
        "quantization": infer_quantization(stem),
        "hf_url": "",
        "targets": ["llama-server"],
    }
    content = "---\n" + yaml.dump(fm, sort_keys=False).rstrip() + "\n---\n\n# " + stem + "\n"
    md_path.write_text(content, encoding="utf-8")
    return fm


def _is_mtp_companion(stem: str) -> bool:
    """Check if a GGUF file is an MTP companion (not a main model)."""
    s = stem.lower()
    return bool(re.search(r"(?:^mtp-|\.mtp$|-mtp$)", s))


def _eval_expr(expr: str, base_val: float) -> float:
    """Safely evaluate expressions like 'base * 0.6'."""
    try:
        return eval(expr, {"__builtins__": {}}, {"base": base_val})
    except Exception:
        logger.warning("failed to eval %r with base=%s, using base", expr, base_val)
        return base_val


def resolve_params(overrides: dict, defaults: dict) -> dict:
    """Resolve profile params with 'base * N' expressions."""
    resolved = dict(defaults)
    for k, v in (overrides or {}).items():
        if isinstance(v, str) and v.startswith("base *") and k in defaults:
            resolved[k] = _eval_expr(v, float(defaults[k]))
        else:
            resolved[k] = v
    return resolved


def _dev(p: str | os.PathLike) -> int | None:
    """Return st_dev of the file/symlink *itself* (does not follow the final symlink)."""
    try:
        return os.lstat(p).st_dev
    except OSError:
        return None


def smart_resolve(path: str | os.PathLike) -> Path:
    """Resolve *path* to an absolute form, but only follow a symlink when doing
    so would cross a filesystem mount boundary.

    Symlinks whose target lives on the same mount as the symlink itself are
    preserved by name: the OS follows them at runtime, so repointing the
    symlink (or its target chain) is reflected automatically. Symlinks that
    escape onto a different mount are expanded, since leaving a transparent
    symlink across a mount boundary would hide a boundary we want explicit.

    Example (``models`` -> /mnt/ai via mergerfs, HF snapshots -> blobs on the
    same mergerfs mount)::

        models/X.gguf                       -> /mnt/ai/models/t2t/X.gguf   (models crossed a mount, expanded)
        models/X.gguf (HF symlink kept)     -> /mnt/ai/models/t2t/X.gguf   (snapshots->blobs same mount, kept)
    """
    path = os.path.abspath(os.fspath(path))
    comps = [c for c in path.split(os.sep) if c]
    result = os.sep
    i = 0
    guard = len(comps) + 8
    while i < len(comps) and guard > 0:
        guard -= 1
        comp = comps[i]
        candidate = os.path.join(result, comp)
        if os.path.islink(candidate):
            target = os.readlink(candidate)
            rt = target if os.path.isabs(target) else os.path.join(result, target)
            rt = os.path.normpath(rt)
            link_dev = _dev(candidate)
            tgt_dev = _dev(rt)
            if link_dev is not None and tgt_dev is not None and link_dev != tgt_dev:
                # Cross-mount symlink: substitute its target and keep descending.
                tparts = [c for c in rt.split(os.sep) if c]
                comps = tparts + comps[i + 1:]
                result = os.sep
                i = 0
                continue
        result = candidate
        i += 1
    return Path(result)


def mount_root(path: str | os.PathLike) -> str:
    """Return the root directory of the filesystem that contains *path*.

    Walks up from *path* until the device (st_dev) changes, i.e. the mount
    point. If *path* does not exist yet, walks up to the nearest existing
    ancestor first.
    """
    p = os.path.abspath(os.fspath(path))
    while not os.path.exists(p) and p not in ("", os.sep):
        p = os.path.dirname(p)
    try:
        dev = os.lstat(p).st_dev
    except OSError:
        return p
    cur = p
    parent = os.path.dirname(cur)
    while parent and parent != cur:
        try:
            if os.lstat(parent).st_dev != dev:
                break
        except OSError:
            break
        cur = parent
        parent = os.path.dirname(cur)
    return cur


def compute_env_prefixes(paths: list[str | os.PathLike], project_hint: str | os.PathLike | None = None):
    """Compute ``${env.*}`` variable names for the longest common path of each
    mount group among *paths*.

    Paths are grouped by the mount they live on; within a group the deepest
    directory shared by all paths becomes the prefix. This yields the shortest
    possible substituted paths while keeping each prefix a real directory on a
    single mount (useful as a docker bind source).

    Returns ``(prefix_to_var, var_to_value)`` where:

        prefix_to_var: {abs_prefix: VAR_NAME}
        var_to_value:  {VAR_NAME: abs_prefix}

    Naming: the group containing *project_hint* (e.g. the llama-server binary)
    is named ``LLAMA_DIR``; remaining groups are ``MODELS_DIR``,
    ``MODELS_DIR_2``, ... in sorted mount order. The number of variables is not
    fixed — one per mount group.
    """
    groups: dict[str, list[str]] = {}
    for p in paths:
        ap = os.path.abspath(os.fspath(p))
        groups.setdefault(mount_root(ap), []).append(ap)

    prefix_to_var: dict[str, str] = {}
    var_to_value: dict[str, str] = {}
    extra = 0
    for mount in sorted(groups):
        ps = groups[mount]
        dirs = [os.path.dirname(p) for p in ps]
        try:
            cp = os.path.commonpath(dirs) if len(dirs) > 1 else dirs[0]
        except ValueError:
            cp = os.path.commonpath([os.path.abspath(p) for p in ps])
        is_project = project_hint is not None and any(
            os.path.abspath(os.fspath(project_hint)) == p for p in ps
        )
        if is_project:
            name = "LLAMA_DIR"
        else:
            extra += 1
            name = "MODELS_DIR" if extra == 1 else f"MODELS_DIR_{extra}"
        prefix_to_var[cp] = name
        var_to_value[name] = cp
    return prefix_to_var, var_to_value


def make_subst(prefix_to_var: dict[str, str]):
    """Return ``sub(path)`` that replaces the longest matching prefix with
    ``${env.VAR}`` (matching llama-swap's environment-variable macro syntax)."""
    prefixes = sorted(prefix_to_var, key=len, reverse=True)

    def sub(path: str | os.PathLike) -> str:
        p = os.fspath(path)
        for pref in prefixes:
            if p == pref or p.startswith(pref + os.sep):
                return "${env." + prefix_to_var[pref] + "}" + p[len(pref):]
        return p

    return sub


def resolve_template(template: str, vars_: dict) -> str:
    """Replace {{variable}} placeholders in template string."""
    out = template
    for k, v in vars_.items():
        out = out.replace(f"{{{{{k}}}}}", str(v))
    return out