# Model-specific documentation

These documents describe behaviour that depends on the specific model or
projector being served, rather than on llama-packer's own design.

| Document | What it covers |
|---|---|
| [vision-context.md](vision-context.md) | How image tokens enter the LLM context, the VRAM they cost, and why the per-slot context must never drop below `image_max_tokens` when vision is served |
