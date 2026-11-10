# Plan: sidecar residency evaluation

Status: **active**. Started 2026-11-08.

Question this answers: *given a fleet and a profiles.yaml, can a chat serving
be resident beside the sidecars a session actually uses — and what does that
cost in chat context?*

It exists because capacity routing replaced matrix sets. Sets used to declare
what lived beside chat, and the shared solve reserved exactly that. Now
co-residency is a property of the emitted numbers alone: two entries coexist
when their summed `capacities` fit every pool. Nothing else declares intent,
so the only way to know whether the plan is sane is to read the plan.

## What a session needs, and when

Ranked by what a miss costs the user:

| sidecar | needed | a miss costs | typical size |
|---|---|---|---|
| embeddings | before generation, every RAG turn | blocks the first token; a reload is seconds of dead air | 0.9–1.2 GiB |
| rerank | before generation, every RAG turn | same — it is on the retrieval critical path | 1.0–3.4 GiB |
| t2s | after each response, for voice output | a pause at the end of a turn, not before it | 1.2–3.1 GiB |
| s2t | only while dictating | a pause when the user starts talking | 2.0–2.9 GiB |
| image | creative sessions only | a long reload; usually too large to share anyway | 6 GiB+ |

That ranking is the reason a reservation policy is not "reserve everything":
the front of the turn earns its keep, the tail does not always.

## The rule the evaluation applies

A set of entries is resident when, for every pool, the sum of their
`capacities` for that pool is at most the pool size. Role pools are size 1,
so a set holds at most one serving per role. The evaluation applies that rule
literally and never second-guesses the planner.

Two invariants matter:

1. **A reservation must reach the solve that sizes chat.** A sidecar reserved
   only in the shared cap but not in the per-model chat solve still ends up
   beside a chat that fills the pool — the emitted numbers then overcommit
   even though the log says "reserved".
2. **The reserve must be large enough.** A reservation using raw measured
   memory under-reserves by the measurement margin the emitted `capacities`
   carry, so the pair sums to just over the pool.

## Strategies

Each strategy is a small transformation of the base profile, so the same
evaluation runs against any fleet:

- **host** — deployed profile. RAG and dictation on the host, voice on GPU
  without a reservation. Chat keeps the pool; voice evicts it.
- **voice** — hold back room for one voice sidecar in every chat solve.
- **rag-hot** — move dictation to GPU so the RAG pair is the only host
  resident. Whether this is possible depends on placement: a sidecar that
  declares `device: cpu` is sticky, and `device_roles:` only forces CPU.
- **rag-voice** — rag-hot plus a voice reservation.
- **all-hot** — every sidecar role on GPU.

## Running it

```sh
python extras/check_sidecar_budget.py config.yaml --require t2s,embeddings,rerank
python extras/sidecar_eval.py --profiles /opt/llama/profiles.yaml
python extras/sidecar_eval.py --profiles P --only host,voice --json
```

`check_sidecar_budget.py` judges one emitted config (accepts a dry-run log
with `I |` prefixes). `sidecar_eval.py` generates the strategy profiles,
runs llama-packer, and prints one row per strategy: largest chat charge,
smallest headroom, entries, median served context, and how many chats can
host each required role.

Exit code is 0 only when every strategy satisfies its own `require` list, so
the pair is usable as a regression gate for a future packer version.

## Measured on the deployed R9700 fleet (32624 MiB, 47 chat entries)

| strategy | largest chat | smallest headroom | entries | median ctx | required set |
|---|---|---|---|---|---|
| host | 29041 | 1279 | 47 | 172032 | — |
| voice | 24953 | 5367 | 37 | 128000 | `t2s` everywhere |
| rag-hot | 25900 | 4420 | 41 | 131072 | `embeddings`, `rerank` everywhere |
| rag-voice | 21805 | 8515 | 31 | 81920 | `embeddings`, `rerank`, `t2s` |
| all-hot | 19786 | 10534 | 25 | 128000 | every sidecar role |

Read it as: holding room for voice costs about 4 GiB of chat context and ten
of the 47 entries; the RAG pair costs a similar amount because on this fleet
`rnk2` (Zerank-2, 3.2 GiB) and the declared `stt` pick are GPU once the role
is no longer forced to the host.

## What a future version must keep true

- Every emitted entry declares the pools it charges, and the sum for any
  admitted set fits those pools.
- Declared GPU categories are charged to the VRAM ledger before chat is
  sized, at the same margin the emitted capacity carries.
- A strategy that claims a sidecar is resident keeps at least one chat entry
  able to host it; if that is no longer true, the strategy's requirement
  fails loudly rather than degrading silently.
