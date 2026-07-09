# GGUF Model Analysis Reference

## How to Read GGUF Metadata

GGUF files store model architecture in key-value metadata at the start of the file. The format:

```
Magic: "GGUF" (4 bytes)
Version: uint32
n_tensors: uint64
n_kv: uint64
KV pairs: [key_string, val_type, val_data] × n_kv
```

### Value Types
| Type | ID | Size |
|------|-----|------|
| UINT8 | 0 | 1 byte |
| INT8 | 1 | 1 byte |
| UINT16 | 2 | 2 bytes |
| INT16 | 3 | 2 bytes |
| UINT32 | 4 | 4 bytes |
| INT32 | 5 | 4 bytes |
| FLOAT32 | 6 | 4 bytes |
| BOOL | 7 | 1 byte |
| STRING | 8 | length-prefixed |
| ARRAY | 9 | type + length + elements |

### Key Architecture Fields

| Key Pattern | Description | Example |
|-------------|-------------|---------|
| `{arch}.block_count` | Number of transformer layers | 48 |
| `{arch}.embedding_length` | Hidden dimension (d_model) | 3840 |
| `{arch}.feed_forward_length` | FFN intermediate size | 15360 |
| `{arch}.attention.head_count` | Number of query heads | 16 |
| `{arch}.attention.head_count_kv` | Number of KV heads (GQA) | 4 or [array] |
| `{arch}.context_length` | Max declared context | 262144 |
| `{arch}.expert_count` | MoE expert count (if applicable) | 64 |
| `{arch}.expert_used_count` | Active experts per token | 4 |
| `general.architecture` | Model architecture name | gemma4 |

## KV Cache Memory Formula

```
KV_bytes_per_token = 2 × Σ(head_count_kv[i]) × head_dim × bytes_per_element

Where:
  2 = keys + values
  head_dim = embedding_length / head_count
  bytes_per_element = {f32: 4, f16/bf16: 2, q8_0: 1, q4_0: 0.5}
```

### For models with per-layer KV head counts (like gemma-4)

```python
head_dim = embedding_length / head_count
total_kv_heads = sum(head_count_kv_array)  # sum across all layers
kv_bytes_per_token = 2 * total_kv_heads * head_dim * bytes_per_element
```

### For models with uniform KV heads

```python
head_dim = embedding_length / head_count
kv_bytes_per_token = 2 * block_count * head_count_kv * head_dim * bytes_per_element
```

## VRAM Budget Calculation

```
Available_VRAM = Total_VRAM - Model_Weights - mmproj - MTP - System_Reserve
Max_Context = Available_VRAM / kv_bytes_per_token
Max_Context = min(Max_Context, model_context_length)
```

## Real-World Examples (Measured)

### gemma-4-12B-it-qat-UD-Q4_K_XL (32 GB VRAM)

| Component | Size |
|-----------|------|
| Model weights | 6.3 GB |
| mmproj (BF16) | 0.17 GB |
| MTP companion | 0.24 GB |
| KV cache + overhead | 6.48 GB |
| **Total used** | **13.19 GB** |

Architecture:
- block_count: 48
- embedding_length: 3840
- head_count: 16
- head_count_kv: [8×40 layers, 1×8 layers] (varies per layer)
- head_dim: 240
- context_length: 262144

### Qwen3.6-27B-Q6_K (32 GB VRAM)

| Component | Size |
|-----------|------|
| Model weights | 21 GB |
| mmproj (BF16) | 0.89 GB |
| **Total model** | **21.89 GB** |

Architecture:
- block_count: 64
- embedding_length: 5120
- head_count: 24
- head_count_kv: 4
- head_dim: 213
- context_length: 262144

### GLM-4.7-Flash-UD-Q5_K_XL (32 GB VRAM)

| Component | Size |
|-----------|------|
| Model weights | 21 GB |

Architecture (DeepSeek2):
- block_count: 47
- embedding_length: 2048
- head_count: 20
- head_count_kv: 1
- expert_count: 64
- expert_used_count: 4
- context_length: 202752

## Size Heuristics

| Model Type | Typical GB per B params |
|------------|------------------------|
| Dense (Q4_K_M) | 0.5-0.6 GB/B |
| Dense (Q6_K) | 0.7-0.8 GB/B |
| MoE total (Q4_K_M) | 0.5-0.6 GB/B |
| MoE active (Q4_K_M) | 2-3 GB/B_active |
| MTP companion | 0.2-0.5 GB |
| mmproj (F16/BF16) | 0.1-1.2 GB |

## Companion Detection Rules

1. **mmproj**: Filename contains "mmproj" and matches model family
2. **MTP**: Filename contains "mtp" or frontmatter `speculative` field points to file with "mtp" in name
3. **Model identity**: Main model is the largest GGUF in directory (excluding mmproj/mtp)

## Files

- `llama_swap_config.py` - Uses this analysis for context calculation
- `SPEC.md` - Configuration specification
- `llama-swap` docs: https://raw.githubusercontent.com/mostlygeek/llama-swap/refs/heads/main/docs/configuration.md
