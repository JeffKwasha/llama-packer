# llama_packer/utils.py
"""Shared utilities for llama-packer."""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


# ── Command-line composition ──────────────────────────────────────────────
# Launch commands are assembled from an ordered flag→value map so that a flag
# can only ever appear once (a dict refuses duplicate keys).  Free-form
# ``cli_args`` are parsed into the same map and merged last, so user-supplied
# args override structured ones without ever duplicating a flag.

def _is_number_token(tok: str) -> bool:
    try:
        float(tok)
        return True
    except (TypeError, ValueError):
        return False


def _pair_flags(tokens: list[str]) -> dict[str, str]:
    """Fold a flat ``[flag, value, flag, value, ...]`` list into an ordered map.

    A ``--flag`` followed by a non-flag token consumes that token as its value;
    otherwise it is valueless.  A token that parses as a number (e.g. ``-1``)
    is treated as a value, not a flag, so ``--temperature -1`` stays intact.
    """
    flags: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("-") and not _is_number_token(tok):
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if nxt is not None and not (nxt.startswith("-") and not _is_number_token(nxt)):
                flags[tok] = nxt
                i += 2
            else:
                flags[tok] = ""
                i += 1
        else:
            flags[tok] = ""
            i += 1
    return flags


def render_command(head: list[str], flag_tokens: list[str], cli_args: str = "") -> str:
    """Compose ``<head> <flags> <cli_args>`` with no duplicate flags.

    Structured ``flag_tokens`` and free-form ``cli_args`` are both reduced to
    an ordered flag→value map; ``cli_args`` merges last (last-write-wins per
    flag), so a flag set both structurally and in ``cli_args`` is emitted once.
    """
    flags = _pair_flags(flag_tokens)
    flags.update(_pair_flags(shlex.split(cli_args)))
    out = list(head)
    for flag, value in flags.items():
        out.append(flag)
        if value != "":
            out.append(value)
    return " ".join(out)


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

# Launch commands are now composed per-backend in ``llama_packer/backends``
# (llama-server, vLLM host/docker).  Each backend owns its own cmd shape, role
# flags and feature flags; the model's `backend:` selection is driven entirely
# by override rules (see ``llama_packer.overrides``).

# Built-in defaults for the vLLM backend. Override via the `vllm:` section of
# profiles.yaml and, for the image only, via --vllm-image.
VLLM_DEFAULT_IMAGE = "vllm/vllm-openai:latest"
VLLM_DEFAULT_BIN = "vllm"
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


def parse_mem_mb(value: str, vram_mb_for_hint: int = 0) -> int:
    """Parse a memory string to MiB.

    Suffixed values (``2G``, ``512m``, ``64k``) resolve directly.
    Bare numbers auto-detect: if < 3 × VRAM(GB) they're treated as GB,
    otherwise as MB (safe for sub-GB values like ``512``).  ``vram_mb_for_hint``
    is only used for that bare-number heuristic.
    """
    m = _MEM_RE.match(str(value).strip())
    if not m:
        logger.warning("invalid memory value %r, using 0", value)
        return 0
    num = float(m.group(1))
    unit = m.group(2).lower()
    if unit == "g":
        return int(num * 1024)
    if unit == "m":
        return int(num)
    if unit == "k":
        return max(1, int(num // 1024))
    vram_gb = vram_mb_for_hint / 1024
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

# Approx bytes per element for KV-cache quantization (incl. block overhead).
# Values are rounded up so any derived memory estimate errs toward reserving
# more (avoid OOM).  This is also the set of cache types llama-packer can size.
_KV_CACHE_BYTES = {
    "q8_0": 1.0625, "q8_1": 1.0625, "q8_k": 1.0625,
    "f16": 2.0, "bf16": 2.0, "f32": 4.0,
    "q4_0": 0.5625, "q4_1": 0.625, "q4_k": 0.5625,
    "q5_0": 0.6875, "q5_1": 0.75, "q5_k": 0.6875,
    "q6_0": 0.8125, "q6_k": 0.8125,
    "iq4_nl": 0.5625,
    # 4-bit E2M1 + FP8 E4M3 block scales per 16 elements ≈ 0.5625 B/elem.
    "nvfp4": 0.5625,
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


def write_stub_md(md_path: Path) -> None:
    """Write an *empty* sidecar for an orphan model file.

    Stubs carry no data: identity falls back to the file stem, context to the
    built-in default, role to the model's directory.  The empty file exists
    purely as the human's editing surface ("drop in a gguf, get a placeholder
    to fill in").  Never called for paths inside an HF blobs tree — see
    ``discover._materialize_sidecar``.
    """
    content = "---\n---\n\n# " + md_path.stem + "\n"
    md_path.write_text(content, encoding="utf-8")
    try:
        md_path.chmod(0o644)
    except OSError:
        # File may be owned by another user on a shared volume; content is
        # already written, so a chmod failure is non-fatal.
        pass


def _is_mtp_companion(stem: str) -> bool:
    """Check if a GGUF file is an MTP companion (not a main model)."""
    s = stem.lower()
    return bool(re.search(r"(?:^mtp-|\.mtp$|-mtp$)", s))


# ── Role classification ────────────────────────────────────────────────
#
# Discovery (llama_packer.discover) owns traversal; these helpers supply the
# pieces of its classification: companion detection by filename
# (companion_kind), the directory-name → role map, and .modelignore parsing.

# Per-models-dir exclusion file: <root>/.modelignore.  One glob per line
# (blank lines and #-comments ignored); a file is skipped when the pattern
# matches its path relative to the root or any single path component — so
# `R3-rerank` excludes that subtree, `*.safetensors` a format, `adetailer*`
# everything named like it.
MODEL_IGNORE_NAME = ".modelignore"


def load_model_ignore(root: Path) -> list[str]:
    """Parse ``<root>/.modelignore`` into a pattern list (empty when absent)."""
    path = Path(root) / MODEL_IGNORE_NAME
    if not path.is_file():
        return []
    patterns: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(line)
    return patterns


def _is_ignored(rel_parts: tuple[str, ...], rel_path: str,
                patterns: list[str]) -> bool:
    import fnmatch
    for pat in patterns:
        if fnmatch.fnmatch(rel_path, pat):
            return True
        if any(fnmatch.fnmatch(part, pat) for part in rel_parts):
            return True
    return False

# Default directory-name → role map for discovery.  The FIRST path component
# of a model file (relative to a models-dir root) selects the role; files at
# the root itself are chat.  Subdirectories absent from this map are not
# served at all (e.g. ``img/`` for stable-diffusion-only models, ``misc/``,
# ``tmp/``).  Extend or override via the profiles.yaml ``dirs:`` mapping;
# ``--extra-dirs`` entries merge in too (backcompat).  Keys are matched
# case-insensitively.
_DEFAULT_DIR_ROLES = {
    "chat": "chat",
    "t2t": "chat",  # legacy name for the chat dir
    "vision": "chat",
    "doc": "chat",
    "ocr": "chat",  # legacy name for the doc dir
    "embed": "embeddings",
    "rerank": "rerank",
}

# Roles a served directory may map to (companions are detected by filename,
# never by directory).  ``s2t`` (whisper.cpp speech-to-text) is opt-in via
# profiles.yaml ``dirs: {s2t: s2t}``, mirroring ``img: image``.
SERVED_ROLES = ("chat", "embeddings", "rerank", "image", "s2t")

# Roles excluded from chat-specific passes: mmproj keep/drop, the shared
# chat+emb+rnk matrix solve, and matrix var collection.
NON_CHAT_ROLES = ("embeddings", "rerank", "image", "s2t")


def validate_dir_roles(dir_roles: dict) -> str | None:
    """Validate a profiles.yaml ``dirs:`` mapping; return an error message or None."""
    for d, r in dir_roles.items():
        if r not in SERVED_ROLES:
            return (f"dirs: {d!r}: unknown role {r!r} "
                    f"(allowed: {', '.join(SERVED_ROLES)})")
    return None


def companion_kind(stem: str) -> str | None:
    """Return 'mmproj' or 'mtp' if ``stem`` names a companion, else None."""
    s = stem.lower()
    if "mmproj" in s:
        return "mmproj"
    if _is_mtp_companion(stem):
        return "mtp"
    return None


# ── Model-kind classification ─────────────────────────────────────────
#
# Header-only classification of a weight file into "text" (LLM family),
# "image" (diffusion/image/video generation weights) or "unknown".  Never
# reads tensor data — GGUF metadata KV walk or the safetensors JSON header.
# Classification drives the discovery guard that keeps diffusion weights
# out of served text roles; the vocabulary below comes from converter
# architecture strings and tensor names, never filenames.

# Prefixes of stable-diffusion.cpp / ComfyUI ``general.architecture``
# values (flux1, sdxl, sd3.5, wan2.1, …).  Architecture names are a
# controlled vocabulary set by the gguf conversion scripts, so prefix
# matching here is reliable — unlike filename matching (e.g. MiniMax H3).
_DIFFUSION_ARCH_RES = tuple(re.compile(p, re.I) for p in (
    r"^flux", r"^sd\d", r"^sdxl$", r"^ssd1", r"^stable-diffusion",
    r"^chroma", r"^wan\d?", r"^hidream", r"^ltxv?$", r"^hunyuan",
    r"^mochi", r"^cosmos", r"^auraflow", r"^pixart", r"^kandinsky",
    r"^sana", r"^ace-?step", r"^omnigen", r"^qwen[-_]?image",
    r"^z-?image", r"^ernie[-_]?image", r"^lumina", r"^sdx?-?l",
))

# Safetensors tensor-name fragments unique to diffusion weights (DiT/UNet/
# VAE blocks).  Text-model transformers never use these block layouts.
_ST_DIFFUSION_MARKERS = (
    ".img_attn.", ".img_mlp.", ".img_mod.", ".txt_attn.", ".txt_mlp.",
    "double_blocks.", "single_blocks.", "input_blocks.", "output_blocks.",
    "middle_blocks.", "model.diffusion_model.", "first_stage_model.",
    "cond_stage_model.", ".up_blocks.", ".down_blocks.", ".mid_block.",
)

# Safetensors tensor-name fragments of autoregressive / pooling text models.
_ST_TEXT_MARKERS = (
    ".layers.", "self_attn", ".attention.", "k_proj", "embed_tokens",
    "lm_head", "encoder.layer", "transformer.h.", ".h.0.",
)

_GGUF_PROBE_CACHE: dict[tuple[str, int], tuple[str | None, bool]] = {}


def gguf_header_probe(path: str | os.PathLike) -> tuple[str | None, bool]:
    """Read ``(general.architecture, has_context_length)`` from a GGUF header.

    Same dependency-free KV walk as :func:`read_gguf_context_length`; returns
    ``(None, False)`` for non-GGUF or unparseable files.  Cached per mtime.
    """
    import struct
    try:
        st = os.stat(path)
    except OSError:
        return None, False
    key = (str(path), st.st_mtime_ns)
    if key in _GGUF_PROBE_CACHE:
        return _GGUF_PROBE_CACHE[key]
    arch: str | None = None
    has_ctx = False
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                raise ValueError
            f.read(12)  # version + tensor_count
            (n_kv,) = struct.unpack("<Q", f.read(8))
            widths = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
                      7: 1, 10: 8, 11: 8, 12: 8}
            for _ in range(n_kv):
                (klen,) = struct.unpack("<Q", f.read(8))
                if klen > 4096:
                    raise ValueError
                key_s = f.read(klen).decode(errors="replace")
                (vtype,) = struct.unpack("<I", f.read(4))
                if vtype == 8:
                    (slen,) = struct.unpack("<Q", f.read(8))
                    raw = f.read(slen)
                    if key_s == "general.architecture":
                        arch = raw.decode(errors="replace")
                        if has_ctx:
                            break
                elif vtype == 9:  # array
                    (etype,) = struct.unpack("<I", f.read(4))
                    (alen,) = struct.unpack("<Q", f.read(8))
                    esize = widths.get(etype, 4)
                    if etype == 8:
                        for _ in range(alen):
                            (elen,) = struct.unpack("<Q", f.read(8))
                            f.seek(elen, 1)
                    else:
                        f.seek(esize * alen, 1)
                elif vtype in widths:
                    f.seek(widths[vtype], 1)
                else:
                    raise ValueError
                if key_s.endswith(".context_length"):
                    has_ctx = True
                    if arch is not None:
                        break
    except (OSError, ValueError, struct.error):
        pass
    _GGUF_PROBE_CACHE[key] = (arch, has_ctx)
    return arch, has_ctx


def sniff_safetensors(path: str | os.PathLike, limit: int = 64) -> str:
    """Classify a safetensors file by its header tensor names.

    Returns ``"image"`` when diffusion DiT/UNet/VAE block names appear,
    ``"text"`` when transformer/pooling names appear, else ``"unknown"``.
    Reads only the JSON header, never tensor data.
    """
    import json
    try:
        with open(path, "rb") as fh:
            magic = fh.read(8)
            if len(magic) < 8:
                return "unknown"
            n = int.from_bytes(magic, "little")
            header = json.loads(fh.read(n).decode("utf-8", errors="replace"))
    except (OSError, ValueError):
        return "unknown"
    names = [k for k in header if k != "__metadata__"][:limit]
    joined = "\n".join(names)
    if any(m in joined.lower() for m in _ST_DIFFUSION_MARKERS):
        return "image"
    if any(m in joined.lower() for m in _ST_TEXT_MARKERS):
        return "text"
    return "unknown"


def classify_file(path: str | os.PathLike) -> str:
    """Header-only kind of a weight file: ``"text"``, ``"image"``, or ``"unknown"``."""
    p = str(path)
    if p.lower().endswith(".gguf"):
        arch, has_ctx = gguf_header_probe(p)
        if arch:
            if any(rx.search(arch) for rx in _DIFFUSION_ARCH_RES):
                return "image"
            if has_ctx:
                return "text"
        return "unknown"
    if p.lower().endswith(".safetensors"):
        return sniff_safetensors(p)
    return "unknown"


def hf_readme_kind(repo_id: str, hf_home=None) -> str | None:
    """Kind implied by the locally cached HF model card's ``pipeline_tag``.

    Reads ``pipeline_tag`` (and falls back to ``tags``) from the snapshot
    ``README.md`` frontmatter — offline, zero network.  Returns ``"image"``
    for image/video generation tags, ``None`` when unresolved or anything
    else.  Online cross-check: ``hf models info <repo>``.
    """
    tag_map = {
        "text-to-image": "image", "image-to-image": "image",
        "unconditional-image-generation": "image", "inpainting": "image",
        "text-to-video": "image", "image-to-video": "image",
    }
    snap = hf_snapshot_dir(repo_id, hf_home)
    if snap is None:
        return None
    rm = snap / "README.md"
    if not rm.is_file():
        return None
    try:
        content = rm.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if not content.startswith("---"):
        return None
    parts = content.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        fm = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return None
    pt = str(fm.get("pipeline_tag") or "")
    hit = tag_map.get(pt.lower())
    if hit:
        return hit
    for t in (fm.get("tags") or []):
        hit = tag_map.get(str(t).lower())
        if hit:
            return hit
    return None


def dir_role_map(extra_dirs: list[str] | None = None,
                 dir_roles: dict | None = None) -> dict[str, str]:
    """Effective directory-name → role map: defaults + extra dirs + profiles.yaml ``dirs:``."""
    role_map = dict(_DEFAULT_DIR_ROLES)
    for d in (extra_dirs or []):
        role_map.setdefault(str(d).lower(), _DEFAULT_DIR_ROLES.get(str(d).lower()) or "chat")
    for d, r in (dir_roles or {}).items():
        role_map[str(d).lower()] = str(r)
    return role_map


# ── Directory-scoped models.yaml ──────────────────────────────────────
#
# Any subdirectory of a models root may carry a ``models.yaml`` that applies
# only to models beneath it: ``defaults`` merge into each sidecar's
# frontmatter (sidecar wins), ``overrides`` are standard override rules whose
# scope is that subtree.  Inner directories beat outer ones beat global.
# Both are folded by the ScopeStack during discovery (llama_packer.discover).

DIR_CONFIG_NAME = "models.yaml"

# Frontmatter keys a directory config may NOT default (identity/skip semantics
# must stay per-model).
_DIR_CONFIG_FORBIDDEN_DEFAULTS = ("name", "model", "ignore")

_dir_config_cache: dict[Path, dict | None] = {}


def load_dir_config(d: Path) -> dict | None:
    """Parse ``models.yaml`` in *d* (cached).  Returns None when absent/empty."""
    d = Path(d)
    if d in _dir_config_cache:
        return _dir_config_cache[d]
    path = d / DIR_CONFIG_NAME
    cfg: dict | None = None
    if path.is_file():
        try:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            raise SystemExit(f"error: {path}: invalid YAML: {e}") from e
        if not isinstance(cfg, dict):
            raise SystemExit(f"error: {path}: must be a mapping")
        defaults = cfg.get("defaults")
        if defaults is not None and not isinstance(defaults, dict):
            raise SystemExit(f"error: {path}: 'defaults' must be a mapping")
        bad = [k for k in (defaults or {}) if k in _DIR_CONFIG_FORBIDDEN_DEFAULTS]
        if bad:
            raise SystemExit(
                f"error: {path}: defaults may not set {', '.join(sorted(bad))} "
                f"(per-model keys)")
    _dir_config_cache[d] = cfg
    return cfg


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


def hf_cache_root(override: str | os.PathLike | None = None) -> str | None:
    """Return the HF cache root, or None when it cannot be determined.

    Resolution order: explicit *override* → ``HF_HOME`` → ``HUGGINGFACE_HUB_CACHE``
    → ``~/.cache/huggingface`` (only when that directory exists).  Used to keep
    HF-cache paths out of the models mount group so they don't widen
    ``${MODELS_DIR}``.
    """
    root = override or os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if root:
        return os.path.abspath(os.fspath(root))
    default = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return default if os.path.isdir(default) else None


def hf_hub_cache(override: str | os.PathLike | None = None) -> Path | None:
    """Return the HF *hub* cache dir (the one holding ``models--org--repo/``).

    Resolution order: explicit *override* (``--hf-home`` / profiles.yaml
    ``hf_home:``, pointing at the HF_HOME-style root, or directly at the hub
    dir) → ``$HF_HOME/hub`` → ``$HUGGINGFACE_HUB_CACHE`` →
    ``~/.cache/huggingface/hub``.
    """
    if override:
        base = Path(override)
        hub = base / "hub"
        return hub if hub.is_dir() else base
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    env = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env:
        return Path(env)
    default = Path.home() / ".cache" / "huggingface" / "hub"
    return default if default.is_dir() else None


def hf_snapshot_dir(repo_id: str, hf_home: str | os.PathLike | None = None) -> Path | None:
    """Locate the local HF hub snapshot directory for ``repo_id``.

    Revision selection: ``refs/main`` when present, else the sole snapshot
    dir, else the newest by mtime (with a warning).  Returns None when the
    repo is not in the hub cache.
    """
    hub = hf_hub_cache(hf_home)
    if hub is None:
        return None
    repo_dir = hub / ("models--" + str(repo_id).replace("/", "--"))
    snaps = repo_dir / "snapshots"
    if not snaps.is_dir():
        return None
    ref = repo_dir / "refs" / "main"
    if ref.is_file():
        rev = ref.read_text(encoding="utf-8").strip()
        if rev and (snaps / rev).is_dir():
            return snaps / rev
    dirs = [d for d in snaps.iterdir() if d.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda d: d.stat().st_mtime)
    snap = dirs[-1]
    if len(dirs) > 1:
        logger.warning("hf: %s has %d snapshots and no refs/main; using newest (%s)",
                       repo_id, len(dirs), snap.name)
    return snap


def hf_snapshot_file(repo_id: str, filename: str,
                     hf_home: str | os.PathLike | None = None) -> Path | None:
    """Resolve ``filename`` inside the local HF hub snapshot of ``repo_id``.

    Lets a sidecar reference a hub-downloaded GGUF (``hf_repo: org/repo`` +
    ``model: file.gguf``) without symlinking it into a models dir — readable
    snapshot filenames, no blob hashes, and it keeps working when sidecars
    move.  ``filename`` may be a glob pattern (``mmproj*.gguf``): an exact
    file wins; otherwise a single glob match resolves and an ambiguous match
    warns and fails.  Returns None when unresolved.
    """
    snap = hf_snapshot_dir(repo_id, hf_home)
    if snap is None:
        return None
    candidate = snap / filename
    if candidate.is_file():
        return candidate
    if any(ch in filename for ch in "*?["):
        matches = sorted(snap.glob(filename))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            logger.warning("hf: %s in %s is ambiguous (%d matches): %s",
                           filename, repo_id, len(matches),
                           ", ".join(m.name for m in matches))
    return None


def compute_env_prefixes(paths: Sequence[str | os.PathLike], project_hint: str | os.PathLike | None = None,
                         hf_home: str | os.PathLike | None = None):
    """Compute ``${VAR}`` macro names for the longest common path of each
    group among *paths*.

    Paths are grouped by the mount they live on; within a group the deepest
    directory shared by all paths becomes the prefix. This yields the shortest
    possible substituted paths while keeping each prefix a real directory on a
    single mount (useful as a docker bind source).

    Paths under the HF cache root are pulled into their own ``HF_HOME`` group
    (prefix = the HF cache root) so they never widen the models group — a
    chat-template symlink into the HF cache otherwise drags ``MODELS_DIR`` up
    to a non-models directory.

    Returns ``(prefix_to_var, var_to_value)`` where:

        prefix_to_var: {abs_prefix: VAR_NAME}
        var_to_value:  {VAR_NAME: abs_prefix}

    Naming: the group containing *project_hint* (e.g. the llama-server binary)
    is named ``LLAMA_DIR``; the HF group is ``HF_HOME``; remaining groups are
    ``MODELS_DIR``, ``MODELS_DIR_2``, ... in sorted mount order.
    """
    hf_root = hf_cache_root(hf_home)

    def _in_hf(p: str) -> bool:
        if not hf_root:
            return False
        return p == hf_root or p.startswith(hf_root + os.sep)

    groups: dict[str, list[str]] = {}
    hf_paths: list[str] = []
    for p in paths:
        ap = os.path.abspath(os.fspath(p))
        if _in_hf(ap):
            hf_paths.append(ap)
        else:
            groups.setdefault(mount_root(ap), []).append(ap)

    prefix_to_var: dict[str, str] = {}
    var_to_value: dict[str, str] = {}

    if hf_paths:
        prefix_to_var[hf_root] = "HF_HOME"
        var_to_value["HF_HOME"] = hf_root

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
