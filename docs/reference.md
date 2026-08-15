## Documentation References
- [llama-swap configuration docs](https://raw.githubusercontent.com/mostlygeek/llama-swap/refs/heads/main/docs/configuration.md)
- [config.example.yaml](https://github.com/mostlygeek/llama-swap/blob/main/config.example.yaml) — includes a dockerized vLLM entry and matrix `evict_costs` for the vLLM backend
- [SPEC.md](SPEC.md) — model metadata schema (capabilities, freethought, strengths/weaknesses, throughput), the llama-swap metadata channel, and the vLLM docker backend (`template: vllm-docker`)
- [gguf_model_analysis.md](gguf_model_analysis.md)
- [llama-server_help](llama-server_help) — full `llama-server` CLI, including `--override-kv KEY=TYPE:VALUE`
- [plans/vllm-gb10.md](plans/vllm-gb10.md) — vLLM / DGX Spark design and progress
- [vllm-memory-estimator](https://github.com/ashishkamra/vllm-memory-estimator) — CPU-only safetensor memory estimator (planned fit-params analog for vLLM)
