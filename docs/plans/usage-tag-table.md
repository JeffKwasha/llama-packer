# Usage tag assignments (2026-09-26)

Every `usage:` tag currently in the sidecar fleet, with the rule that
produced it. Tags are claims, not guarantees: a model leaves any set whose
co-residents it cannot share the card with while still meeting the demand.

Derived mechanically from what the packer already knows — `role`, `tools`,
and the resolved context — so the whole assignment is reproducible and every
row is arguable. Correct a row by editing the sidecar; nothing else has to
change.

| rule | meaning |
|---|---|
| `role != chat` | untagged. A resident or audio model has no coding/chat demand to express, and the planner only charges chat solves per set. |
| `tools` and `ctx >= 128k` | `coding` — vision shed first, since demanding work rarely needs it |
| vision-capable, `ctx < 128k` | `qa` — vision *is* the point, so it is required |
| `ctx >= 64k` | `chat` |
| `ctx >= 32k` | `qa` |
| below every floor | untagged — nothing to demand |

The per-set overheads these are weighed against, from `matrix: sets:`:
`rag` 1751 MB, `ragz` 6515 MB, `voice` 1898 MB, `talk` 1204 MB. `ragz` is
the binding one — it holds a 5766 MB reranker — and it is what drops
eleven of the thirty chat entries.

## Assignments

| `chatterbox` | t2s | 32768 | 1 | n | n | — | role=t2s — a resident or audio model, not a chat entry |
| `chatterbox-turbo` | t2s | 32768 | 1 | n | n | — | role=t2s — a resident or audio model, not a chat entry |
| `gemma-3-12b-it-heretic-v2` | chat | 131072 | 4 | y | n | coding | tools + ≥128k — coding floor reachable, vision shed first |
| `gemma-4-e4b-it-ultra-uncensored-heretic` | chat | 131072 | 6 | y | n | coding | tools + ≥128k — coding floor reachable, vision shed first |
| `gemma4-12b` | chat | 262144 | 6 | y | y | coding | tools + ≥128k — coding floor reachable, vision shed first |
| `gemma4-26b-a4b` | chat | 262144 | 1 | y | n | coding | tools + ≥128k — coding floor reachable, vision shed first |
| `gemma4-26b-cerebellum` | chat | 262144 | 5 | y | y | coding | tools + ≥128k — coding floor reachable, vision shed first |
| `gemma4-31b-deckard` | chat | 204800 | 1 | y | n | coding | tools + ≥128k — coding floor reachable, vision shed first |
| `gemma4-31b-hauhaucs-balanced` | chat | 196608 | 1 | y | y | coding | tools + ≥128k — coding floor reachable, vision shed first |
| `glimmer-30b` | chat | 131072 | 6 | y | y | coding | tools + ≥128k — coding floor reachable, vision shed first |
| `glm47-flash` | chat | 202752 | 1 | y | n | coding | tools + ≥128k — coding floor reachable, vision shed first |
| `huihui-qwen3-vl-4b-instruct-abliterated` | chat | 262144 | 1 | y | y | coding | tools + ≥128k — coding floor reachable, vision shed first |
| `huihui-qwen3-vl-8b-instruct-abliterated` | chat | 262144 | 1 | y | y | coding | tools + ≥128k — coding floor reachable, vision shed first |
| `jina-v5-small-retrieval` | embeddings | 4096 | 1 | n | n | — | role=embeddings — a resident or audio model, not a chat entry |
| `jina-v5-small-text-matching` | embeddings | 4096 | 1 | n | n | — | role=embeddings — a resident or audio model, not a chat entry |
| `kokoro-82m` | t2s | 32768 | 1 | n | n | — | role=t2s — a resident or audio model, not a chat entry |
| `laguna-xs-21` | chat | 262144 | 1 | y | n | coding | tools + ≥128k — coding floor reachable, vision shed first |
| `lfm25-vl-16b-absolute-heresy` | chat | 128000 | 6 | n | y | qa | vision is the point — kept |
| `minicpm-v-4_6-thinking` | chat | 262144 | 6 | y | y | coding | tools + ≥128k — coding floor reachable, vision shed first |
| `nemotron-3.5-asr-streaming-0.6b` | s2t | 32768 | 1 | n | n | — | role=s2t — a resident or audio model, not a chat entry |
| `nousresearch_nous-hermes-2-vision` | chat | 32768 | 6 | y | n | qa | below coding/chat floors — qa floor only |
| `nuextract3` | chat | 262144 | 5 | n | n | chat | general chat |
| `ornith-15-9b-ad` | chat | 262144 | 4 | y | y | coding | tools + ≥128k — coding floor reachable, vision shed first |
| `parakeet-tdt-0.6b-v3` | s2t | 32768 | 1 | n | n | — | role=s2t — a resident or audio model, not a chat entry |
| `pocket-tts-english` | t2s | 32768 | 1 | n | n | — | role=t2s — a resident or audio model, not a chat entry |
| `qwen3-4b-huihui-v2` | chat | 40960 | 6 | y | n | qa | below coding/chat floors — qa floor only |
| `qwen3-4b-josiefied` | chat | 262144 | 1 | y | n | coding | tools + ≥128k — coding floor reachable, vision shed first |
| `qwen3-asr-0.6b` | s2t | 32768 | 1 | n | n | — | role=s2t — a resident or audio model, not a chat entry |
| `qwen3-embedding-0.6b` | embeddings | 4096 | 1 | n | n | — | role=embeddings — a resident or audio model, not a chat entry |
| `qwen3-reranker-0.6b` | rerank | 4096 | 1 | n | n | — | role=rerank — a resident or audio model, not a chat entry |
| `qwen3-vl-8b-instruct-unredacted-max` | chat | 262144 | 1 | y | y | coding | tools + ≥128k — coding floor reachable, vision shed first |
| `qwen35-4b-uncensored-hauhaucs-aggressive` | chat | 262144 | 5 | y | y | coding | tools + ≥128k — coding floor reachable, vision shed first |
| `qwen36-35b-nail` | chat | 262144 | 1 | y | y | coding | tools + ≥128k — coding floor reachable, vision shed first |
| `qwen38-27b-dirk` | chat | 163840 | 1 | y | y | coding | tools + ≥128k — coding floor reachable, vision shed first |
| `qwen38-27b-unsloth` | chat | 229376 | 1 | y | y | coding | tools + ≥128k — coding floor reachable, vision shed first |
| `r3-rerank-0.6b` | rerank | 4096 | 1 | n | n | — | role=rerank — a resident or audio model, not a chat entry |
| `voyage-4-nano` | embeddings | 4096 | 1 | n | n | — | role=embeddings — a resident or audio model, not a chat entry |
| `zerank-2` | rerank | 4096 | 1 | n | n | — | role=rerank — a resident or audio model, not a chat entry |

## Result

`ragz` went from 30 chat vars to 19. Eleven models cannot hold a 128k
coding window beside Zerank-2, so they serve from `rag` instead. No entry
disappeared and none lost its vision entry.

## Open questions for the operator

1. **`nuextract3` is tagged `chat` on inference alone** — it is 262k with
   no tools, so it fell through to the general-chat rule. If it is really an
   extraction model it wants its own category, or none.
2. **`qwen_3_4b` is tagged `qa` and nothing else in the 4B range is.** It
   resolves to 32k, under the chat floor. Confirm that is intended rather
   than a consequence of a stale `context_length`.
3. **Eleven `coding` models sit in `ragz`.** They clear 128k there, so they
   stay. If Zerank-2 should never co-reside with a large chat model, that is
   a set expression to change, not a tag.
