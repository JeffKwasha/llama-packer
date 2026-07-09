# TODO

1. Add `run_fit_params()` — shell out to `llama-fit-params`, parse output
2. Rewrite `calculate_ctx_size()` — use fit-params output, no more per-model `.md` caps
3. Add `mmproj: false` support in `detect_mmproj()`
4. Update the broken 26B sidecar — drop `context_limit_31G`, set `mmproj: false`
5. Regenerate config and validate

---

## Why

**`_BYTES_PER_TOKEN = 20` is catastrophically wrong** — it underestimates KV cache by ~5500×. The real answer depends on the model's architecture (n_layer, n_head_kv, head_dim, sliding window pattern, cache dtype) and device VRAM.

**`llama-fit-params` already does this accurately.** It loads the GGUF, inspects architecture, queries device memory, accounts for SWA, cache type, and parallel count, and prints the exact per-device breakdown. We just need to call it and parse the output.

**`context_limit_31G: 106496` in `.md` files** works but is manual, brittle, and doesn't scale. Computing context from available VRAM at generation time is automatic, correct, and works for any model.

**`mmproj: false`** lets a multimodal model run text-only when no correct mmproj exists (e.g., stock Gemma 4 26B has no matching mmproj — the 12B one has wrong dimensions → 500 errors).

---

## How

### 1. `run_fit_params()`

```python
import subprocess
import re

_FIT_PARAMS_BIN = "llama-b9874/llama-fit-params"

def run_fit_params(model_path, ctx_size, cache_type="q8_0", parallel=1):
    cmd = [
        _FIT_PARAMS_BIN,
        "--fit-print", "on",
        "-c", str(ctx_size),
        "-m", model_path,
        "--cache-type-k", cache_type,
        "--cache-type-v", cache_type,
    ]
    if parallel > 1:
        cmd += ["--parallel", str(parallel)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    # Parse: "Vulkan0 <model> <context> <compute>"
    for line in out.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].startswith("Vulkan"):
            model_mib = int(parts[1])
            context_mib = int(parts[2])
            compute_mib = int(parts[3])
            return model_mib, context_mib, compute_mib
    raise RuntimeError(f"could not parse fit-params output:\n{out.stdout}")
```

### 2. New `calculate_ctx_size()`

```python
def calculate_ctx_size(model_path: str, vram_total_mb: int, parallel: int = 1,
                       spare_mb: int = 0, mmproj_mb: int = 0,
                       design_ctx: int = 262144, cache_type: str = "q8_0") -> int:
    available = vram_total_mb - reserve_system - reserve_video - spare_mb - mmproj_mb

    model_mib, ctx_at_design_mib, compute_mib = run_fit_params(
        model_path, design_ctx, cache_type, parallel
    )
    remaining = available - model_mib - compute_mib
    if remaining <= 0:
        return _MIN_CTX_SIZE

    if ctx_at_design_mib <= remaining:
        return design_ctx

    # Linear estimate: remaining / (ctx_at_design / design_ctx)
    max_ctx = (remaining * design_ctx) // ctx_at_design_mib
    max_ctx = (max_ctx // _CTX_ROUND_TO) * _CTX_ROUND_TO
    return max(max_ctx, _MIN_CTX_SIZE)
```

For SWA models this slightly under-estimates (conservative) because the first `swa_window` tokens cost more per-token than later tokens. For models where the estimate is tight, the generated config will work but leave a bit more headroom than needed — safe.

### 3. `detect_mmproj()` — `mmproj: false`

```python
def detect_mmproj(path: Path, frontmatter: dict | None = None) -> str | None:
    if "mmproj" in path.stem.lower():
        return None
    if frontmatter:
        mmproj_val = frontmatter.get("mmproj")
        if mmproj_val is False:      # ← explicit disable
            return None
        if mmproj_val:               # ← explicit path
            companion = path.parent / mmproj_val
            if companion.exists():
                return str(companion)
    # ... fallback fuzzy match unchanged
```

### 4. Sidecar fix

Set `mmproj: false` in `gemma-4-26B-A4B-it-UD-Q6_K.md`, remove `context_limit_*G`. The new `calculate_ctx_size` will compute the correct cap from VRAM.

### 5. Regenerate and validate

```bash
python gen-config.py --dry-run
python gen-config.py
```
