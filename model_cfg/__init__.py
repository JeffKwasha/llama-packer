"""model-cfg: Generate llama-swap config from model metadata."""

from model_cfg.model import Model
from model_cfg.hardware import GpuProfile, detect_gpu_env_var
from model_cfg.utils import (
    parse_context_length,
    resolve_spare_mb,
    slugify,
    get_model_size_mb,
    find_bin_dir,
    get_available_versions,
    parse_frontmatter,
    generate_stub_md,
)

# Backward-compat: get_vram_mb is now in hardware.py but old callers may
# import it from model_cfg.  Delegate to hardware.detect_vram_mb.
def get_vram_mb() -> int:
    from model_cfg.hardware import detect_vram_mb
    return detect_vram_mb()

__all__ = [
    "Model",
    "GpuProfile",
    "get_vram_mb",
    "detect_gpu_env_var",
    "parse_context_length",
    "resolve_spare_mb",
    "slugify",
    "get_model_size_mb",
    "find_bin_dir",
    "get_available_versions",
    "parse_frontmatter",
    "generate_stub_md",
]