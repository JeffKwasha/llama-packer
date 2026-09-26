# Plan: additional audio.cpp roles (vc / vad / music / separation)

Status: **proposal — deferred** (2026-09-12). The shipped `audio-cpp` engine
serves `t2s` (TTS) and `s2t` (ASR) only. This records the other audio.cpp task
families so they can be added without a schema rewrite, and so they are not
silently lost when they come up.

Related: [`audio-cpp.md`](audio-cpp.md) (the engine that ships t2s/s2t),
[`matrix-categories.md`](matrix-categories.md) (the co-load model).

## Families → roles

audio.cpp's `task` values already map onto llama-packer roles; the shipped set
covers the two most useful. The rest:

| audio.cpp tasks | proposed role | `capabilities.in` → `out` |
|---|---|---|
| `vc`, `s2s` | `vc` | audio → audio |
| `sep` | `sep` (or `audio`/`stems`) | audio → audio × tracks |
| `vad`, `align`, `diar` | `s2t` | audio → text (metadata) — no new role needed |
| `music`, `sfx`, `dialogue`, `edit`, `design`, `ctrl` | `t2s` | text → audio — no new role needed |

Only `vc` (audio→audio) is genuinely a new **role**: it differs from `t2s`
(text→audio) in capabilities. `vad`/`align`/`diar` reuse `s2t`;
`music`/`sfx`/`dialogue` reuse `t2s`.

## Why this is cheap when it lands

`capabilities.in/out` are freeform client metadata, so `vc` is a
`roles = {…}` + `_DEFAULT_DIR_ROLES` (`vc/` dir) change plus an entry in the
engine's `roles` set — no schema rewrite. As a matrix category it is just
another `categories:` entry:

```yaml
matrix:
  categories:
    vc: {role: vc}
  sets:
    voice-convert: "__CHAT_VARS__ & vc"
```

Like tts/stt, `vc` is a fixed-overhead resident (weights + family buffer) and
should carry a low `evict_cost`.

## Open questions

- `sep` returns **multiple** tracks — can llama-swap's model/route model carry
  that, or does separation need its own service shape?
- Is `vc` worth a distinct role, or is it a `t2s` capability variant? (Current
  answer: distinct, because `capabilities.in` differs.)
- Directory names: `vc/`, plus whether `vad`/`align`/`diar` get their own dirs
  or share `s2t/`.
- VRAM constants per new family (the `compute_mib` table in the audio-cpp
  backend).

## Non-goals

- Rewriting `whisper-server` in terms of audio.cpp (whisper `.bin` models
  cannot run on it).
- Music/sfx fidelity tuning — those inherit `t2s`.
