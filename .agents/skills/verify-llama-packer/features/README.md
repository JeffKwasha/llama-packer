# llama-packer feature map

Each file names one user-facing feature, how a user reaches it, how to drive
it, and the observable state that proves it works. Start with
[`pack-and-validate.md`](pack-and-validate.md): it is the core journey and
the fastest signal that a change broke the pipeline.

| Feature | User path | Proves |
|---|---|---|
| [Pack and validate](pack-and-validate.md) | `llama-packer --models-dir … --output …` → `llama-swap --config …` | a real tree becomes a config llama-swap accepts and serves |
| [Profile check](profiles-check.md) | `llama-packer --profiles … --check` | the preflight accepts good profiles and names bad ones |
| [Role discovery](role-discovery.md) | pack a tree where a model has no owning role | the run stops instead of guessing a role |
| [Capacity routing](capacity-routing.md) | pack, inspect `routing:` and `capacities:` | entries carry pools and eviction bounds |

Run every journey from the repo root with the harness in
[`../SKILL.md`](../SKILL.md). Evidence goes under `.verify/<run>/`.
