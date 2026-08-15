# llama_packer/utils.py
"""Shared utilities for llama-packer."""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# ── Defaults (can be overridden via profiles.yaml `defaults` section or env) ──
_DEFAULT_CONTEXT_LENGTH = 32768
_CTX_ROUND_TO = 8192
_MIN_CTX_SIZE = 4096

# Chat models should be useful beyond this context; mmproj (vision) is dropped
# from the main entry when keeping it would fall below this floor. 128k.
_MIN_USEFUL_CTX = 131072

# VRAM reservation breakdown (MB)
_RESERVE_SYSTEM = 1024
_RESERVE_VIDEO = 1024

# MTP defaults (per-model overrides via frontmatter keys)
_MTP_SPEC_TYPE = "draft-mtp"
_MTP_DRAFT_N_MAX = 2
_MTP_DRAFT_P_MIN = 0.75

# Templates are selected by a model's `role` (chat | embeddings | rerank),
# unless the sidecar declares an explicit `template:` override (e.g.
# ``template: vllm-docker`` to opt a chat model into the vLLM docker backend).
# The llama-server roles differ only in the role-specific extra CLI flags.
_TARGET_TEMPLATES = {
    "chat": {
        "bin": "{{llama_bin}}",
        "model": "-m {{model_path}}",
        "ctx": "-c {{ctx_size}}",
        "parallel": "--parallel {{parallel}}",
        "cache_type": "--cache-type-k {{cache_type}} --cache-type-v {{cache_type}}",
        "mtp": "--spec-type {{mtp_spec_type}} --spec-draft-n-max {{mtp_n_max}} --spec-draft-p-min {{mtp_p_min}}",
        "mmproj": "--mmproj {{mmproj_path}}",
        "extra": "{{extra_args}}",
    },
    # Embedding models: restrict server to /v1/embeddings.
    "embeddings": {
        "bin": "{{llama_bin}}",
        "model": "-m {{model_path}}",
        "ctx": "-c {{ctx_size}}",
        "parallel": "--parallel {{parallel}}",
        "cache_type": "--cache-type-k {{cache_type}} --cache-type-v {{cache_type}}",
        "mtp": "--spec-type {{mtp_spec_type}} --spec-draft-n-max {{mtp_n_max}} --spec-draft-p-min {{mtp_p_min}}",
        "mmproj": "--mmproj {{mmproj_path}}",
        "extra": "--embedding --embd-normalize 2 -b 4096 -ub 4096 {{extra_args}}",
    },
    # Reranking models: expose /v1/rerank.
    "rerank": {
        "bin": "{{llama_bin}}",
        "model": "-m {{model_path}}",
        "ctx": "-c {{ctx_size}}",
        "parallel": "--parallel {{parallel}}",
        "cache_type": "--cache-type-k {{cache_type}} --cache-type-v {{cache_type}}",
        "mtp": "--spec-type {{mtp_spec_type}} --spec-draft-n-max {{mtp_n_max}} --spec-draft-p-min {{mtp_p_min}}",
        "mmproj": "--mmproj {{mmproj_path}}",
        "extra": "--rerank --pooling rank -b 4096 -ub 4096 {{extra_args}}",
    },
    # vLLM served inside a container. Selected via `template: vllm-docker` on a
    # chat sidecar. `${PORT}` is llama-swap's per-model host port macro; vLLM
    # binds the container port ({{container_port}}) which gets published via
    # `-p ${PORT}:{{container_port}}`.
    "vllm-docker": {
        "cmd": "docker run --init --rm {{docker_args}} --name ${MODEL_ID} "
               "-v {{models_dir}}:/models -p ${PORT}:{{container_port}} "
               "{{vllm_image}} "
               "--model {{model_path}} --served-model-name ${MODEL_ID} "
               "--host 0.0.0.0 --port {{container_port}} "
               "--max-model-len {{ctx_size}} --gpu-memory-utilization {{gpu_mem_util}} "
               "{{extra_args}}",
    },
}

# Built-in defaults for the vLLM docker backend. Override via the `vllm:`
# section of profiles.yaml and, for the image only, via --vllm-image.
VLLM_DEFAULT_IMAGE = "vllm/vllm-openai:latest"
VLLM_DEFAULT_CONTAINER_PORT = 8000
VLLM_DEFAULT_DOCKER_ARGS = "--runtime=nvidia --gpus all --shm-size=16g"
VLLM_DEFAULT_GPU_MEM_UTIL = 0.9

# Sidecar/profile sampling parameter names accepted in llama-packer input.
# These are llama.cpp CLI-style names.
SAMPLING_KEYS = frozenset({
    "temperature", "top_p", "top_k", "min_p",
    "pres_pen", "repeat_penalty", "freq_pen",
})

# llama-swap injects setParamsByID values into the OpenAI-compatible request
# body, so the emitted keys must be the request-JSON names llama-server parses.
# (pres_pen/freq_pen are CLI names, not request-body names.)
REQUEST_SAMPLING_KEYS = {
    "pres_pen": "presence_penalty",
    "freq_pen": "frequency_penalty",
}


def request_sampling_key(key: str) -> str:
    """Map a sidecar/profile sampling key to the request-body JSON key."""
    return REQUEST_SAMPLING_KEYS.get(key, key)

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


@functools.lru_cache(maxsize=128)
def get_model_size_mb(model_path: str) -> int:
    """Get model file size in MB."""
    return Path(model_path).stat().st_size // (1024 ** 2)


def read_gguf_context_length(path: str | os.PathLike) -> int | None:
    """Read `<architecture>.context_length` from a GGUF header.

    Minimal dependency-free parser: walks the metadata KV block until it finds a
    `context_length` key and returns its integer value. Returns None for
    safetensors, non-GGUF files, or parse failures. The value is the model's
    architectural context limit as shipped — no RoPE/YaRN extension applied.
    """
    import struct
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            f.read(4)  # version (u32)
            f.read(8)  # tensor_count (u64)
            (n_kv,) = struct.unpack("<Q", f.read(8))  # metadata_kv_count
            # value-type -> byte width (u8/i8/u16/i16/u32/i32/f32/bool/u64/i64/f64)
            widths = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
            for _ in range(n_kv):
                (klen,) = struct.unpack("<Q", f.read(8))
                if klen > 4096:  # sanity: keys are short; guards against misalignment
                    return None
                key = f.read(klen).decode(errors="replace")
                (vtype,) = struct.unpack("<I", f.read(4))
                if vtype == 8:  # string
                    (slen,) = struct.unpack("<Q", f.read(8))
                    f.read(slen)
                elif vtype == 9:  # array
                    (etype,) = struct.unpack("<I", f.read(4))
                    (alen,) = struct.unpack("<Q", f.read(8))
                    esize = widths.get(etype, 4)
                    for _ in range(alen):
                        if etype == 8:
                            (elen,) = struct.unpack("<Q", f.read(8))
                            f.read(elen)
                        else:
                            f.read(esize)
                elif vtype in widths:
                    raw = f.read(widths[vtype])
                    if "context_length" in key:
                        if vtype in (0, 1):
                            return raw[0]
                        if vtype == 2:
                            return struct.unpack("<H", raw)[0]
                        if vtype == 3:
                            return struct.unpack("<h", raw)[0]
                        if vtype == 4:
                            return struct.unpack("<I", raw)[0]
                        if vtype == 5:
                            return struct.unpack("<i", raw)[0]
                        if vtype == 10:
                            return struct.unpack("<Q", raw)[0]
                        if vtype == 11:
                            return struct.unpack("<q", raw)[0]
                else:
                    return None
    except (OSError, ValueError, struct.error):
        return None
    return None


# Bytes per element for safetensors dtypes (used to size model weights).
_SAFETENSORS_DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "F8": 1, "F6E4M3FN": 1, "F6E5M2": 1, "F4": 1, "F6E2M1FN": 0.5,
    "F3": 0.375, "F2": 0.25, "F1": 0.125,
}

# Approx bytes per element for KV-cache quantization (incl. q8_0 block overhead).
_KV_CACHE_BYTES = {
    "q8_0": 1.0625, "q8_1": 1.0625, "q8_k": 1.0625,
    "f16": 2.0, "bf16": 2.0, "f32": 4.0,
    "q4_0": 0.5625, "q4_1": 0.5625, "q4_k": 0.5625,
    "q5_0": 0.6875, "q5_k": 0.6875,
    "q6_0": 0.8125, "q6_k": 0.8125,
}


def estimate_safetensors(
    model_path: str | os.PathLike,
    cache_type: str = "q8_0",
) -> tuple[int, float]:
    """Estimate (model_mib, kv_per_token_mib) from a safetensors header.

    Reads only the JSON header (tensor names, shapes, dtypes) — no weights are
    loaded. Used as a fallback when llama-fit-params cannot measure the model
    (e.g. safetensors input, or an architecture fit-params does not model).

    Raises ValueError if the file is not a parseable safetensors header or if no
    per-layer k/v projection can be found to size the KV cache.
    """
    path = Path(model_path)
    with path.open("rb") as fh:
        magic = fh.read(8)
        if len(magic) < 8:
            raise ValueError("file too small to be safetensors")
        header_len = int.from_bytes(magic[:8], "little")
        header_bytes = fh.read(header_len)
    if not header_bytes:
        raise ValueError("empty safetensors header")
    header = json.loads(header_bytes.decode("utf-8"))

    def dtype_bytes(dt: str) -> float:
        return _SAFETENSORS_DTYPE_BYTES.get(dt, 2.0)  # unknown -> assume 2 (safe)

    total_bytes = 0
    kv_out_dims: dict[int, int] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        shape = meta.get("shape", [])
        n = 1
        for d in shape:
            n *= int(d)
        total_bytes += n * dtype_bytes(meta.get("dtype"))
        low = name.lower()
        if "k_proj" in low and low.endswith("weight"):
            m = re.search(r"\.layers\.(\d+)\.", low) or re.search(r"layers\.(\d+)", low)
            if m and shape:
                kv_out_dims[int(m.group(1))] = int(shape[0])
        elif "v_proj" in low and low.endswith("weight"):
            m = re.search(r"\.layers\.(\d+)\.", low) or re.search(r"layers\.(\d+)", low)
            if m and shape:
                kv_out_dims.setdefault(int(m.group(1)), int(shape[0]))

    if not kv_out_dims:
        raise ValueError("no per-layer k/v projection found; cannot size KV cache")

    cache_bytes = _KV_CACHE_BYTES.get(cache_type, 1.0625)
    kv_per_token_bytes = 2 * sum(kv_out_dims.values()) * cache_bytes
    kv_per_token_mib = kv_per_token_bytes / (1024 * 1024)
    model_mib = int(total_bytes // (1024 * 1024))
    return model_mib, kv_per_token_mib


# Conservative real-world sequential read estimates (MB/s) by device class.
# Not best-case marketing figures; chosen to bound the health-check timeout
# safely. NVMe is set conservatively; SATA SSD/HDD use the operator's figures.
_NVME_READ_MBPS = 1500
_SSD_READ_MBPS = 300
_HDD_READ_MBPS = 100
_UNKNOWN_READ_MBPS = 100


def _detect_drive_speed(model_paths: list[Path]) -> int:
    """Detect the slowest drive speed (MB/s) among the drives holding *model_paths*.

    Classifies each drive by device type (NVMe vs SATA) and the kernel
    rotational flag, then assigns a conservative real-world sequential read
    estimate. No model data is read off disk and no benchmark is run. The
    minimum across all drives is returned so the slowest disk bounds the
    health-check timeout.
    """
    speeds: list[int] = []
    for p in model_paths:
        try:
            mount = mount_root(str(p))
        except Exception:
            mount = str(p)
        try:
            out = subprocess.run(
                ["lsblk", "-dno", "NAME,ROTA", mount],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode != 0 or not out.stdout.strip():
                speeds.append(_UNKNOWN_READ_MBPS)
                continue
            line = out.stdout.strip().splitlines()[0].split()
            dev_name, rota = line[0], int(line[1])
            if dev_name.startswith("nvme"):
                speed, kind = _NVME_READ_MBPS, "NVMe SSD"
            elif rota == 1:
                speed, kind = _HDD_READ_MBPS, "HDD"
            else:
                speed, kind = _SSD_READ_MBPS, "SATA SSD"
            logger.info("drive: %s (%s) estimated %d MB/s", dev_name, kind, speed)
            speeds.append(speed)
        except Exception:
            speeds.append(_UNKNOWN_READ_MBPS)

    if not speeds:
        logger.info("drive: unknown, defaulting to %d MB/s", _UNKNOWN_READ_MBPS)
        return _UNKNOWN_READ_MBPS
    return min(speeds)


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
            raise SystemExit(
                "error: no llama-b#### directory found here\n"
                "  use --llama-server <path> or set LLAMA_BIN_DIR to locate llama-server"
            )
        return f"llama-b{versions[-1]}"
    if (base_dir / f"llama-b{version}").is_dir():
        return f"llama-b{version}"
    avail = " ".join(str(v) for v in get_available_versions(base_dir))
    raise SystemExit(f"error: version {version} not found\n  available: {avail}")


def parse_frontmatter(md_path: Path) -> dict:
    """Parse YAML frontmatter from .md file."""
    try:
        content = md_path.read_text(encoding="utf-8")
    except PermissionError:
        logger.warning("permission denied: %s", md_path)
        return {}
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
    }
    content = "---\n" + yaml.dump(fm, sort_keys=False).rstrip() + "\n---\n\n# " + stem + "\n"
    md_path.write_text(content, encoding="utf-8")
    try:
        md_path.chmod(0o644)
    except OSError:
        # File may be owned by another user on a shared volume; content is
        # already written, so a chmod failure is non-fatal.
        pass
    return fm


def _is_mtp_companion(stem: str) -> bool:
    """Check if a GGUF file is an MTP companion (not a main model)."""
    s = stem.lower()
    return bool(re.search(r"(?:^mtp-|\.mtp$|-mtp$)", s))


# ── Role classification ────────────────────────────────────────────────
#
# A model directory mixes several roles that discovery previously
# distinguished with ad-hoc inline checks ("mmproj" in stem,
# _is_mtp_companion, targets[0]).  classify_models() is the single source of
# truth: it walks the directory and returns (path, kind) tuples keyed by role
# (chat / embeddings / rerank) plus the companion kinds (mmproj / mtp).

MODEL_KINDS = ("chat", "embeddings", "rerank", "mmproj", "mtp")

# Mapping from an ``extra_dir`` name to the role its contents play.  Files
# discovered under ``embed/`` are embedding models, under ``rerank/`` rerankers.
_EXTRA_DIR_ROLE = {"embed": "embeddings", "rerank": "rerank"}


def companion_kind(stem: str) -> str | None:
    """Return 'mmproj' or 'mtp' if ``stem`` names a companion, else None."""
    s = stem.lower()
    if "mmproj" in s:
        return "mmproj"
    if _is_mtp_companion(stem):
        return "mtp"
    return None


def model_kind(path: str | os.PathLike, role: str | None = None) -> str:
    """Classify a single model file by role.

    Companions are detected by filename regardless of any sidecar/companion
    metadata:
      * ``mmproj``  — vision projection (``*mmproj*`` on a .gguf)
      * ``mtp``     — MTP speculative-draft head

    For everything else the role is resolved in priority order:
      1. an explicit ``role:`` field in the ``.md`` sidecar,
      2. ``role`` passed by the caller (the directory a file was found in,
         e.g. ``embed``/``rerank``),
      3. a ``type:`` field of ``embedding``/``rerank``,
      4. default ``chat``.
    """
    p = Path(path)
    if p.suffix.lower() == ".gguf":
        ck = companion_kind(p.stem)
        if ck:
            return ck
    if role in ("embeddings", "rerank"):
        return role
    md = p.with_suffix(".md")
    fm = parse_frontmatter(md) if md.is_file() else {}
    explicit = fm.get("role")
    if explicit:
        return str(explicit)
    typ = str(fm.get("type") or "").lower()
    if "rerank" in typ:
        return "rerank"
    if "embed" in typ:
        return "embeddings"
    return "chat"


def classify_models(
    models_dir: str | os.PathLike,
    extra_dirs: list[str] | None = None,
) -> list[tuple[Path, str]]:
    """Classify model files in ``models_dir`` (and ``extra_dirs``) by role.

    Returns a list of ``(path, kind)`` tuples where ``kind`` is one of
    :data:`MODEL_KINDS` (``chat``, ``embeddings``, ``rerank``, ``mmproj``,
    ``mtp``).  ``model.py`` uses this as the single source of truth for
    discovery, replacing the previously scattered inline heuristics.
    """
    roots: list[tuple[Path, str | None]] = [(Path(models_dir), None)]
    for d in (extra_dirs or []):
        extra = Path(models_dir) / d
        if extra.is_dir():
            roots.append((extra, _EXTRA_DIR_ROLE.get(d)))

    out: list[tuple[Path, str]] = []
    for root, role in roots:
        if not root.is_dir():
            continue
        for pattern in ("*.gguf", "*.safetensors", "*.md"):
            recursive = pattern == "*.md"
            walker = root.rglob(pattern) if recursive else root.glob(pattern)
            for p in walker:
                if p.is_file():
                    out.append((p, model_kind(p, role)))
    return out


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


def compute_env_prefixes(paths: Sequence[str | os.PathLike], project_hint: str | os.PathLike | None = None):
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
    ``${VAR}`` (llama-swap config `macros:` syntax). The resolved values are
    written into the config's ``macros:`` block, so reloads pick them up
    without depending on a stale process environment."""
    prefixes = sorted(prefix_to_var, key=len, reverse=True)

    def sub(path: str | os.PathLike) -> str:
        p = os.fspath(path)
        for pref in prefixes:
            if p == pref or p.startswith(pref + os.sep):
                return "${" + prefix_to_var[pref] + "}" + p[len(pref):]
        return p

    return sub


def resolve_template(template: str, vars_: dict) -> str:
    """Replace {{variable}} placeholders in template string."""
    out = template
    for k, v in vars_.items():
        out = out.replace(f"{{{{{k}}}}}", str(v))
    return out