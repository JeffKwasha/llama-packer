# llama_packer/utils.py
"""Shared utilities for llama-packer.

Constants live in :mod:`llama_packer.consts`; this module holds
general-purpose functions with simple input→output semantics.
"""

from __future__ import annotations

import copy
import fnmatch
import functools
import json
import logging
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from typing import Any
from pathlib import Path

import yaml

from llama_packer.consts import (
    _SAFETENSORS_DTYPE_BYTES,
    _KV_CACHE_BYTES,
    _NVME_READ_MBPS,
    _SSD_READ_MBPS,
    _HDD_READ_MBPS,
    _UNKNOWN_READ_MBPS,
    _DEFAULT_DIR_ROLES,
    _DIFFUSION_ARCH_RES,
    _WEIGHT_SUFFIXES,
)

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


def render_command(head: list[str], builtin_flags: list[str], global_args: str = "",
                   role_flags: str = "", cli_args: str = "") -> str:
    """Compose ``<head> <flags>`` from precedence-ordered sources.

    Every source is reduced to one ordered flag→value map (a flag can only
    appear once; a later source overwrites an earlier one — per-flag,
    most-specific-wins).  Precedence, least to most specific:

    1. ``builtin_flags`` — backend built-ins (``-c``, ``--parallel``, …)
    2. ``global_args``  — fleet-wide tuning flags (profiles.yaml ``<section>.args``)
    3. ``role_flags``   — per-role flags (embed/rerank ``-b/-ub`` batch sizes)
    4. ``cli_args``     — per-model sidecar ``cli_args:``

    The free-form sources are shlex-parsed here, so quoting errors surface at
    command-render time.  If a new use case needs a different ordering, open
    an issue for a refactor instead of breaching these argument categories.
    """
    flags = _pair_flags(builtin_flags)
    flags.update(_pair_flags(shlex.split(global_args)))
    flags.update(_pair_flags(shlex.split(role_flags)))
    flags.update(_pair_flags(shlex.split(cli_args)))
    out = list(head)
    for flag, value in flags.items():
        out.append(flag)
        if value != "":
            out.append(value)
    return " ".join(out)


# ── Sampling parameter names ──────────────────────────────────
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
    logger.info("reading GGUF header: %s", path)
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
    logger.info("reading safetensors header: %s", path)
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
            logger.debug("drive: %s (%s) estimated %d MB/s", dev_name, kind, speed)
            speeds.append(speed)
        except Exception:
            speeds.append(_UNKNOWN_READ_MBPS)

    if not speeds:
        logger.debug("drive: unknown, defaulting to %d MB/s", _UNKNOWN_READ_MBPS)
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

# Roles a served directory may map to
# ``img: image``.
SERVED_ROLES = ("chat", "embeddings", "rerank", "image", "s2t", "t2s")

# Roles excluded from chat-specific passes: mmproj keep/drop, the shared
# chat+emb+rnk matrix solve, and matrix var collection.
NON_CHAT_ROLES = ("embeddings", "rerank", "image", "s2t", "t2s")


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

# Diffusion arch prefixes live in llama_packer.consts (_DIFFUSION_ARCH_RES).

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
    logger.info("reading GGUF header: %s", path)
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


_GGUF_MTP_CACHE: dict[tuple[str, int], bool | None] = {}


def gguf_has_mtp_layers(path: str | os.PathLike) -> bool | None:
    """Whether a GGUF carries MTP layers, from header tensor names alone.

    ``nextn`` is the tensor-name marker every GGUF MTP conversion uses
    (deepseek2/GLM/qwen nextn blocks); conversions sometimes strip them
    while the sidecar keeps ``mtp:``.  Reads metadata and tensor names
    only — a few hundred KiB, no tensor data — and caches per mtime.
    Returns True / False, or None when the header cannot be parsed
    (the caller then keeps the declared intent rather than guessing).
    """
    import struct
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (str(path), st.st_mtime_ns)
    if key in _GGUF_MTP_CACHE:
        return _GGUF_MTP_CACHE[key]
    result: bool | None = None
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                raise ValueError
            f.read(4)  # version
            (n_tensors,) = struct.unpack("<Q", f.read(8))
            (n_kv,) = struct.unpack("<Q", f.read(8))
            widths = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                      10: 8, 11: 8, 12: 8}

            def skip_value() -> None:
                (vtype,) = struct.unpack("<I", f.read(4))
                if vtype == 8:
                    (slen,) = struct.unpack("<Q", f.read(8))
                    f.seek(slen, 1)
                elif vtype == 9:
                    (etype,) = struct.unpack("<I", f.read(4))
                    (count,) = struct.unpack("<Q", f.read(8))
                    if etype == 8:
                        for _ in range(count):
                            (slen,) = struct.unpack("<Q", f.read(8))
                            f.seek(slen, 1)
                    elif etype == 9:
                        raise ValueError  # nested arrays: not seen in the wild
                    else:
                        f.seek(widths[etype] * count, 1)
                else:
                    f.seek(widths[vtype], 1)

            for _ in range(n_kv):
                (klen,) = struct.unpack("<Q", f.read(8))
                f.seek(klen, 1)
                skip_value()
            for _ in range(n_tensors):
                (nlen,) = struct.unpack("<Q", f.read(8))
                name = f.read(nlen)
                (ndims,) = struct.unpack("<I", f.read(4))
                f.seek(8 * ndims + 12, 1)  # ne[] + type + offset
                if b"nextn" in name.lower():
                    result = True
                    break
            else:
                result = False
    except (OSError, ValueError, struct.error):
        result = None
    _GGUF_MTP_CACHE[key] = result
    return result


def sniff_safetensors(path: str | os.PathLike, limit: int = 64) -> str:
    """Classify a safetensors file by its header tensor names.

    Returns ``"image"`` when diffusion DiT/UNet/VAE block names appear,
    ``"text"`` when transformer/pooling names appear, else ``"unknown"``.
    Reads only the JSON header, never tensor data.
    """
    import json
    logger.info("reading safetensors header: %s", path)
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


def _merge_list(base: list, overlay: list) -> list:
    """Union-append *overlay* onto *base*: order-preserving, de-duplicated.

    String items starting with ``-`` remove instead (``-image`` drops
    ``image``); removal of an absent item is a no-op.  Returns a new list.
    """
    result = [copy.deepcopy(i) for i in base]
    for item in overlay:
        if isinstance(item, str) and item.startswith("-") and len(item) > 1:
            target = item[1:]
            result = [i for i in result if i != target]
        elif item not in result:
            result.append(copy.deepcopy(item))
    return result


def merge_layer(below: dict | None, above: dict | None, *,
                origin: str = "") -> dict:
    """The single merge rule for every config layer.

    Layers bottom-to-top: ``defaults:`` → profiles → sidecar frontmatter →
    override rules → companion block.  At each boundary the upper layer wins:

    - ``None`` in the upper layer **deletes** the key outright
      (``top_p: None`` removes an inherited ``top_p``).
    - dicts merge recursively, per sub-key, upper winning.
    - lists union-append (order-preserving, de-duplicated); ``-item``
      entries remove instead.
    - scalars (and type changes) are replaced by the upper value.
    - ``"base * N"`` strings on sampling keys evaluate against the merged
      value from the layer below; with no numeric below they warn and are
      skipped (never emitted raw).

    Returns a new dict; inputs are never mutated or aliased.
    """
    result = copy.deepcopy(dict(below or {}))
    for k, v in (above or {}).items():
        if v is None:
            result.pop(k, None)
            continue
        cur = result.get(k)
        if isinstance(v, dict) and isinstance(cur, dict):
            result[k] = merge_layer(cur, v, origin=origin)
        elif isinstance(v, list) and isinstance(cur, list):
            result[k] = _merge_list(cur, v)
        elif isinstance(v, list) and cur is None:
            result[k] = _merge_list([], v)
        else:
            if (isinstance(v, str) and v.startswith("base *")
                    and k in SAMPLING_KEYS):
                if isinstance(cur, (int, float)) and not isinstance(cur, bool):
                    result[k] = _eval_expr(v, float(cur))
                else:
                    logger.warning(
                        "%s: %r has no numeric value below to resolve "
                        "against; skipping expression %r",
                        origin or "merge", k, v)
                continue
            result[k] = copy.deepcopy(v)
    return result


def resolve_params(overrides: dict, defaults: dict) -> dict:
    """Resolve profile params over defaults (one :func:`merge_layer`)."""
    return merge_layer(defaults, overrides, origin="profile")


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


def make_fast_storage_predicate(
    fast_roots: str | list[str] | tuple[str, ...] | None,
    exists: Callable[[str], bool] = os.path.exists,
) -> Callable[[str], bool]:
    """Predicate: is *path* stored on one of the fast-storage branches?

    *fast_roots* name the physical branches behind a merged mount (e.g. the
    SSD branch ``/mnt/@/ssd_ai`` of a mergerfs pool).  A merged view hides
    which branch holds a file — stat(2) reports the merge's own device — so
    the test is structural: the file's path *relative to its mount root*
    must exist under the branch (mergerfs keeps the directory structure
    identical across branches).  ``realpath`` is applied first, so
    HF-hub snapshot symlinks classify by where the bytes (the blob) live.
    Empty/None roots yield an always-False predicate: no tier knowledge,
    behave as before.
    """
    if fast_roots is None:
        roots: list[str] = []
    elif isinstance(fast_roots, str):
        roots = [fast_roots]
    else:
        roots = [str(r) for r in fast_roots]
    branches = [os.path.realpath(r).rstrip("/") for r in roots if r]
    if not branches:
        return lambda path: False

    def is_fast(path: str) -> bool:
        real = os.path.realpath(path)
        for branch in branches:
            if real == branch or real.startswith(branch + "/"):
                return True
            try:
                rel = os.path.relpath(real, mount_root(real))
            except (ValueError, OSError):
                continue
            if exists(os.path.join(branch, rel)):
                return True
        return False

    return is_fast


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


def hf_cache_root(override: str | os.PathLike | None = None) -> Path | None:
    """Return the HF_HOME root — the dir *containing* ``hub/`` — or None.

    ``--hf-home`` / profiles.yaml ``hf_home:`` / ``$HF_HOME`` all name this
    root, **never** the hub itself: the root holds ``hub/`` (plus ``token``,
    ``xet/`` …), while ``models--org--repo/`` lives one level down in
    ``<root>/hub``.  ``$HUGGINGFACE_HUB_CACHE`` is HF's own *hub* variable (not
    a root) and survives only as a last-resort source for path grouping.

    Resolution order: explicit *override* → ``$HF_HOME`` →
    ``$HUGGINGFACE_HUB_CACHE`` → ``~/.cache/huggingface`` (only when that
    directory exists).  Used to keep HF-cache paths out of the models mount
    group so they don't widen ``${MODELS_DIR}``.
    """
    root = override or os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if root:
        return Path(os.path.abspath(os.fspath(root)))
    default = Path.home() / ".cache" / "huggingface"
    return default if default.is_dir() else None


def hf_hub_cache(override: str | os.PathLike | None = None) -> Path | None:
    """Return the HF *hub* cache dir (the one holding ``models--org--repo/``).

    The hub is always ``<HF_HOME-root>/hub``.  An explicit *override*
    (``--hf-home`` / profiles.yaml ``hf_home:``) names the HF_HOME root and is
    **never** interpreted as the hub itself — see :func:`hf_cache_root`.

    Resolution order: explicit *override* → ``$HF_HOME`` →
    ``$HUGGINGFACE_HUB_CACHE`` (already the hub dir) →
    ``~/.cache/huggingface/hub``.
    """
    if override or os.environ.get("HF_HOME"):
        root = hf_cache_root(override)
        return (root / "hub") if root is not None else None
    env = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env:
        return Path(env)
    default = Path.home() / ".cache" / "huggingface" / "hub"
    return default if default.is_dir() else None


def snapshot_weight_paths(snap: Path) -> list[str]:
    """Snapshot-relative weight paths at any depth, sorted.

    The single recursive snapshot listing in the codebase: one ``rglob``
    filtered to :data:`_WEIGHT_SUFFIXES`, returned as posix relative paths
    (``model.gguf``, ``Subdir/model.gguf``). Callers cache per snapshot mtime.
    """
    try:
        files = [p for p in snap.rglob("*") if p.is_file()]
    except OSError:
        return []
    rels = []
    for p in files:
        if p.suffix.lower() not in _WEIGHT_SUFFIXES:
            continue
        try:
            rels.append(p.relative_to(snap).as_posix())
        except ValueError:
            continue
    return sorted(rels)


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
    move.  ``filename`` may be a snapshot-relative path into a subdirectory
    (``Subdir/model.gguf``), a bare basename matched at any depth (a single
    hit wins; several warn and fail), or a glob pattern (an exact file wins;
    otherwise a single glob match resolves and an ambiguous match warns and
    fails).  Returns None when unresolved.
    """
    snap = hf_snapshot_dir(repo_id, hf_home)
    if snap is None:
        return None
    if "/" not in filename and not any(ch in filename for ch in "*?["):
        # Bare basename: always resolve through the index, so a basename
        # present at several depths warns instead of silently winning.
        hits = [r for r in snapshot_weight_paths(snap)
                if Path(r).name == filename]
        if len(hits) == 1:
            return snap / hits[0]
        if len(hits) > 1:
            logger.warning("hf: %s in %s is ambiguous (%d matches): %s",
                           filename, repo_id, len(hits),
                           ", ".join(hits))
        return None
    candidate = snap / filename
    try:
        if candidate.is_file():
            return candidate
    except OSError:
        return None
    rels = snapshot_weight_paths(snap)
    if any(ch in filename for ch in "*?["):
        matches = sorted({r for r in rels
                          if fnmatch.fnmatchcase(Path(r).name, filename)
                          or fnmatch.fnmatchcase(r, filename)})
    else:
        # Explicit relative path that is not an exact file — no basename
        # fallback (a name with a separator names one place, not many).
        return None
    if len(matches) == 1:
        return snap / matches[0]
    if len(matches) > 1:
        logger.warning("hf: %s in %s is ambiguous (%d matches): %s",
                       filename, repo_id, len(matches),
                       ", ".join(matches))
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
    # Prefixes are string keys/values (macros are text), so convert the Path
    # returned by hf_cache_root at this boundary.
    hf_root_path = hf_cache_root(hf_home)
    hf_root = str(hf_root_path) if hf_root_path is not None else None

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

    if hf_paths and hf_root is not None:
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
