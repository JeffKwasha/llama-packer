# Backend: sd-server

Image (and video) generation via stable-diffusion.cpp's `sd-server`.

- **Engine:** stable-diffusion.cpp · **Transports:** host (`sd-server`).
- **Roles:** `image`.
- **Formats:** `.gguf`, `.safetensors`, `hf_repo`.
- **Proxied:** yes — `proxy` + `checkEndpoint: "/"`.
- **Binary:** `--sd-server` > `profiles.yaml sd.bin` > `$SD_BIN_DIR` >
  `sd-server` on `PATH`.
- **Opt-in:** `dirs: {img: image}` (the `img/` directory is otherwise ignored).

## Command shape

```
<sd_bin> --listen-port ${PORT} --listen-ip 0.0.0.0 --diffusion-model <ref>
  [--diffusion-fa ...]        # sd.args; per-model cli_args wins per flag
```

## profiles.yaml

- `sd: {bin, args}` — `args` are fleet-wide flags (e.g. `--diffusion-fa`).

## VRAM

Fixed overhead — weights + a small runtime buffer (`_SD_COMPUTE_MB`) — and
**excluded from the shared chat matrix**: a 40 GB diffusion model must not
collapse the chat budget. The sidecar `vram_mb:` pin overrides the estimate.

## See also

- [`SPEC.md` → Image Backend (sd-server)](../../SPEC.md#image-backend-sd-server)
- [docs/plans/comfyui-sd.md](../plans/comfyui-sd.md)
