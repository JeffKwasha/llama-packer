# Plan: ComfyUI + stable-diffusion.cpp backends — research findings

Status: **research — not scheduled**. Recorded 2026-08-24.
Related: `docs/plans/vllm-gb10.md`, `docs/plans/gguf-vllm.md`.
Upstream: `mostlygeek/llama-swap` (proxy), `leejet/stable-diffusion.cpp` (sd-server), `yanwk/comfyui-boot` (ComfyUI).

## Findings

llama-swap already acts as a transparent HTTP proxy for **any** OpenAI/API-compatible server. Its feature list explicitly includes `stable-diffusion.cpp`, `audio.cpp`, **ComfyUI**, etc. (README Features, `docs/configuration.md`). Both backends are therefore *already routable* via `models[].cmd`/`models[].proxy` — what llama-packer lacks is first-class discovery, templating, and sizing for them.

### stable-diffusion.cpp (`sd-server`)

* **Unified image:** `ghcr.io/mostlygeek/llama-swap:unified-cuda` ships `sd-server` alongside `llama-server`/`whisper.cpp` (README Docker Install). No extra install when using that image.
* **Server binary:** `sd-server` (from `stable-diffusion.cpp/examples/server`). CLI resembles llama-server but with diffusion args:
  ```yaml
  cmd: >
    sd-server --listen-port ${PORT} --listen-ip 0.0.0.0
      --diffusion-fa --offload-to-cpu --lora-model-dir /tmp
      --diffusion-model /models/flux-2-klein-4b-Q4_0.gguf
      --vae /models/flux2_ae.safetensors
      --llm /models/Qwen3-4B-UD-Q4_K_XL.gguf
  proxy: http://127.0.0.1:${PORT}
  checkEndpoint: /   # NOT /health — sd-server returns 200 on / only (see Discussion #866)
  ```
  Real-world fleet examples (Discussion #866) always set `--diffusion-model` + `--vae` + `--llm`; newer Z-Image Turbo needs the 3-file variant (see `stable-diffusion.cpp/docs/z_image.md`). `--diffusion-fa` and `--offload-to-cpu` are common flags for flash-attn + VRAM relief.
* **Endpoints proxied:** llama-swap advertises `SDAPI via stable-diffusion.cpp's server` (README):
  - `/sdapi/v1/txt2img`, `/sdapi/v1/img2img`, `/sdapi/v1/loras` (requires `model` in body)
  - Compatibility `/v1/images/generations`, `/v1/images/edits` also map through llama-swap — no special config.
  - Web UI at `/sdcpp/v1/...` when sd-server builds with frontend (`SD_SERVER_BUILD_FRONTEND=ON`), but not needed for API use.
* **llama-swap knobs:** Like any model, `proxy: http://127.0.0.1:${PORT}` + `checkEndpoint: /` (or `none` while debugging). Macro `${PORT}` works with `startPort`. `macros.sd-cmd` pattern used in fleet configs to share `--diffusion-fa --offload-to-cpu --lora-model-dir /tmp`.
* **Discussion #866 pitfall:** default `checkEndpoint: /health` never returns 200 for sd-server → model never becomes ready even if listening. The fix is `checkEndpoint: /` (or `none`) and `logToStdout: both` to see sd-server logs. Custom `--listen-port ${PORT}` avoids the stale-port bug where a hard-coded `50001` was exposed separately.

### ComfyUI (`comfyui-boot`)

* **Server:** `yanwk/comfyui-boot:cu130-slim-v2` (Docker) on port `8188`. Requires many volume mounts for models/cache/output (see Issue #1001 comment):
  ```yaml
  models:
    comfyui_auto:  # special name enabling /comfyui/ endpoint
      checkEndpoint: /
      cmdStop: docker stop comfyui-auto
      cmd: >
        docker run --rm --name comfyui-auto --runtime=nvidia
          --gpus '"device=2,3"' -p ${PORT}:8188
          -v /path/to/comfyui/storage-cache/dot-cache:/root/.cache
          -v /path/to/comfyui/storage-models/models:/root/ComfyUI/models
          -v /path/to/comfyui/storage-user/output:/root/ComfyUI/output
          # ... 8-10 mounts for custom_nodes, hf-hub, torch-hub, input, user, etc.
          -e CLI_ARGS=""
          yanwk/comfyui-boot:cu130-slim-v2
  ```
* **llama-swap endpoint:** `/comfyui/` is a **custom endpoint** (README, Issue #1001). It only works when a model is named exactly `comfyui_auto`; llama-swap then injects compatibility workarounds so ComfyUI swaps out smoothly when idle.
* **Generic fallback:** any other model name can proxy ComfyUI via `/upstream/{my-other-comfyui-model}/` plus:
  ```yaml
  upstream:
    ignorePaths:
      - ^\/ws$|^\/api\/jobs$  # prevent swaps while UI is open / polling
  models:
    my-other-comfyui-model:
      checkEndpoint: /
      compat:
        ignoreWebsockets: true  # new in Issue #1001: don't count WS for swap/TTL
  ```
  `compat.ignoreWebsockets: true` (also written `workarounds.ignoreWebSocket` in early comment) tells llama-swap to ignore websocket connections for swap decisions — needed because ComfyUI holds `/ws` open.
* **Discovery note:** ComfyUI models are not GGUF files but a directory of checkpoints/LoRAs under `models/`; llama-packer today ignores `img/`-like dirs. A future `comfy` role will need analogous `dirs:` treatment (`comfy: image` or `comfy: workflow`).

## Implications for llama-packer

* **No new binary needed** — reuse existing proxy/templating path; just add backends that render `sd-server`/`comfyui-boot` commands and validate via `compat`/`checkEndpoint`.
* **Role mapping:** extend `utils.SERVED_ROLES` + `dir_role_map` for `img`/`comfy`→`image` (and decide whether `image` is a new native `capabilities` value or stays under `chat` with `type: image`). Sidecar `role: image` or `type: image` could select the backend, similar to `embed`/`rerank`.
* **VRAM sizing:** sd-server and ComfyUI have no `llama-fit-params` analog. Short-term: honor declared `context_length` / `parameters` or a fixed overhead per model; long-term: measure/load-test or track derived `fit-params` block. Until then, `sizing = declared + companion weights` and budget as fixed overhead (same as CPU-resident path today).
* **Config generation:** `models[].proxy`, `models[].checkEndpoint`, `models[].compat.ignoreWebsockets`, `upstream.ignorePaths`, and macro `${PORT}` must be emitted. The `matrix` feature (see `docs/plans/matrix-categories.md`) should treat image/ComfyUI models as a separate co-located category rather than evictable chat models.

## References

* llama-swap README Features + `docs/configuration.md` (minimal viable `models: model: cmd:`)
* `config.example.yaml` (healthCheckTimeout, macros, upstream.ignorePaths)
* Issue #1001 `compat: improve support for comfyui` — `compat.ignoreWebsockets`, `comfyui_auto` + `/comfyui/` endpoint, generic `/upstream/{id}/` + `ignorePaths` fallback (comments 2026-08-10)
* Discussion #866 `Stable diffusion error on unified-cuda image` — `sd-server` example `sd-cmd` macro, `checkEndpoint: /` pitfall, `${PORT}` vs fixed port, `logToStdout: both`
* `stable-diffusion.cpp` `examples/server/README.md` (sd-server flags: `--diffusion-model`, `--vae`, `--llm`, `--diffusion-fa`, `--offload-to-cpu`, `-v`, `--cfg-scale`, `--listen-port`) and `docs/z_image.md` (3-file models)
* `stable-diffusion.cpp` `examples/server/frontend` (optional embedded WebUI, not required)
