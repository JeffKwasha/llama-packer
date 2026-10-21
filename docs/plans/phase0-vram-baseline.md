# Phase 0 — VRAM baseline measurement (2026-09-26)

Ground truth for validating the demand-driven planner. Machine: AMD ROCm,
32,624 MB visible VRAM.

## Tooling findings

- `amd-smi metric` (no flags) does **not** print VRAM on ROCm 10.0 / amdsmi 27.0.
  Use `amd-smi metric -m` for a one-shot `TOTAL_VRAM / USED_VRAM / FREE_VRAM`
  (plus `TOTAL_GTT`), or `amd-smi monitor -v` to watch it. `--json` is not needed.
- `amd-smi process` reported "No running processes detected" even while a
  `llama-server` was plainly resident. Not usable for attribution here.
  Use the `USED_VRAM` delta instead.
- `hardware.detect_vram_baseline_mb()` returned **7239**, exactly matching
  `amd-smi metric -m` `USED_VRAM` at the same instant. The planner's
  pre-existing-usage baseline and amdsmi agree, so the `--baseline`/spare
  machinery is trustworthy.
- `immich_machine_learning` runs on this box and takes VRAM. It moved the
  idle floor from 1,956 MB to 7,239 MB during this session. Any VRAM
  measurement here needs a baseline taken seconds before the load.

## Measurement 1 — `lfm25-vl-16b-absolute-heresy` (with mmproj)

Contemporaneous baseline taken before the load.

| | MiB |
|---|---|
| idle floor (GUI only, browser closed shortly after) | 1956 |
| with model resident | 8446 |
| **measured delta** | **6490** |
| planner prediction (weights+compute 1802 + KV 4782 @ 128k x 6) | 6584 |
| error | **+94 MiB, planner conservative by 1.4%** |

The emitted tag `[VRAM 6.4GB]` = 6502..6605 MiB (`_fmt_mib` divides by 1024),
consistent with the 6584 prediction. No serve-correction row exists for arch
`lfm2`, so this is raw fit-params + the 4% margin.

## Measurement 2 — same model, same command line

A later load of the identical entry measured a **5873 MiB** delta against the
same 1956 floor. Same cmd line, same model, ~617 MiB apart.

**Conclusion: single-shot VRAM deltas on this box carry ~600 MB (~10%) of
noise** — ROCm allocation/fragmentation plus whatever immich is doing. That is
larger than the 1.4% accuracy the planner is being judged on, so hand-rolled
deltas are not a sufficient validator.

## Recommendation

Use the repo's own supported calibration path instead: `--probe-memory lfm2`
sweeps a (ctx, parallel) grid live and fits the affine law, which separates
weights / KV / slot terms from this noise by construction. It requires a quiet
GPU (no `llama_residents()`, no immich VRAM), so it is a deliberate
foreground task rather than something to run alongside other work.

The useful Phase 0 conclusions are the two tooling facts above plus the single
clean measurement 1, which agrees with the planner to 1.4%.

## Idle-floor campaign (same box, later session, 2026-09-26)

Method: `amd-smi metric -m --json`, 1 Hz for 40 s, no model resident,
`immich_ml` running but idle, KDE Plasma/Wayland with two outputs enabled
(DP-3 3440x1440, DP-4 3840x2160).

| statistic | `USED_VRAM` (MiB) |
|---|---|
| min | 1740 |
| median | 1780 |
| max | 1796 |
| spread | 56 |
| earlier spot samples | 1665–1667 |

`radeontop` at the same time: `vram 1748mb, gtt 334mb` at ~0 % GPU load — the
footprint is resident buffers, not active work.

### The log format

`llama_packer/vram_log.py` writes the campaign's samples as a CSV: **one
header, then a row per sample**, appended rather than rewritten. Rows that
collapse into one (the accumulator compresses samples within a threshold) keep
their count, because **the envelope is a slow sawtooth and a plateau must not
be outvoted by a single spike**.

The report names the floor, the maximum and the spread, and the median used
per tenant. **With no usable sample it says so** rather than printing empty
numbers, and the rows are weighted by their count.

### What drives each part

- **Base (always resident):** driver page tables + connected-display scanout +
  compositor core. Scanout is small: two outputs at 4 B/px, double-buffered is
  only ≈ **101 MiB** (37.8 + 63.3). So `USED_VRAM` is dominantly compositor
  textures, Plasma/Xwayland, and driver overhead — not framebuffers.
- **Slack (variable):** open windows/tabs (the 56 MiB in-session spread; the
  1665→1796 swing between sessions) and, much larger, `immich_ml` model loads
  — the earlier section documents that tenant moving the floor from 1956 to
  7239 MiB. Browsers with many content processes add on top.

### Implication for the reserve

`reserve = _RESERVE_SYSTEM(1024) + max(_RESERVE_VIDEO(1024), baseline_mb)`:
the fixed **2048 MiB covers the observed GUI-only base** (max 1796) with ~250
MiB headroom. It cannot express the slack. That is what `--spare` /
`hardware.spare` is for: pin the expected extra footprint (e.g. immich during
a scan, or a browser wall) so the solve sizes contexts against it. Note
`hardware.baseline_mb: 128m` is **below the 1024 video floor and therefore a
no-op** — it is logged but does not change the reserve.

## Log for days — the envelope is a slow sawtooth

The samples above are all within one session. Background VRAM does **not**
typically move within a minute; it moves on a **timescale of days**, and the
shape is asymmetric:

- **Slow rise.** A model file loads over minutes (platter read); a
  browser/desktop footprint drifts **up over hours**. `immich` a scan, a game
  left open, TTS/whisper loading — all ramp gently.
- **Fast fall.** Release is quick.
- **Long plateau.** Once up, the value *hangs* there for a long time.

So minute-scale sampling is correct — the peaks are long-lived and never
aliased away — and a **5-minute** cadence is ample. The supported tool is the
long-horizon logger:

```sh
python -m llama_packer.vram_log                 # 5-min samples, forever
python -m llama_packer.vram_log --once          # one labeled sample to stdout
python -m llama_packer.vram_log --report        # floor / percentiles / per-tenant medians
```

It writes `~/.cache/llama-packer/vram_baseline.csv`; each row is ISO-8601
time, total/used/free/GTT MiB, a run-length `count`, and counts of the known
tenants (`models` = llama-server/vllm, `audio` = audiocpp/whisper, `image` =
sd-server/comfy, `immich`, `browsers`). Two economies keep it cheap:

- **Run-length counts** (`--threshold`, default 128 MiB): a sample within the
  threshold of the pending row bumps its `count` instead of opening a new
  row, so a flat plateau costs one row however long it lasts.
- **Deferred writes** (`--low-mib`, default 2048): while usage is at or below
  the mark, rows are held in RAM — **no disk writes**, so an idle machine's
  disks are not woken. They flush once usage rises above it, and always at
  shutdown.

After a few days the report's **floor is the base** to give the planner, and
the gap from the floor to each tenant's median/max is the **slack** to hand
`--spare` (each GiB of reserve costs ~25k of qwen3.8 context).

### Natural experiment — TTS (2026-09-27)

While `audiocpp_server` served `kokoro-82m-q8_0`, idle `USED_VRAM` moved from
~1,800 MiB to a steady **~4,222–4,254 MiB**: a **~2.4 GB** variable load. The
emitted entry tag predicts `[VRAM 1.2GB]`, so audio-cpp's fixed-overhead
estimate undercounts the real serving by roughly 1.2 GB. Corroborates why the
variable must be *measured*, not inferred from the fixed reserve.
