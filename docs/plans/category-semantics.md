# Category semantics

How a declared category becomes emitted entries. Written down because these
rules are interlocking, and each one was arrived at by getting it wrong
first — the reasons are kept as long as they are load-bearing.

## The category is a user declaration

`usage: [coding]` in a sidecar says what the operator believes the model is
good at. It is not computed, not inferred, not "light" or "heavy", and
nothing about the GPU changes which category a model has. `hf_url` and
`context_length` are the same kind of field: metadata the user wrote.

What the *system* does with a category is decide whether it can be served.

## One entry per category

Each satisfiable category emits one id. A category whose minimums cannot be
met emits nothing — no id claims it. So `usage: [coding, chat]` on a model
that can serve both produces `stem.coding` and `stem.chat`, and the operator
picks by asking for the one they want.

This is why "second entries" needed no separate mechanism: two entries for
one model are two categories, assigned in sidecars. A vision-capable coding
serving is just another declared category with `require: [vision]`.

## Degradation: a category's own sacrifice order

Each category carries an ordered `sacrifice:` list. That order is *the*
answer to "what do we give up first" — per category, not a fleet-wide
threshold. The ladder walks prefixes of that list:

```
sacrifice: [vision]        ->  [] , {vision} , {vision, draft}
sacrifice: [draft, vision] ->  [] , {draft} , {draft, vision}
```

Anything in `require:` is never shed, so an id can never claim coding+vision
while vision was shed. `shade_levels()` is implemented and tested.

`coding` is configured `sacrifice: [vision]`, so it sheds vision to maximise
context, and keeps vision only when VRAM is sufficient for **max design
context + vision** — that is, when vision costs nothing.

## The null category (fallback)

Every model emits a fallback entry, in addition to any category entries. It
has bare minimums only, no category name in the id, and it can itself fail
and emit nothing.

It never degrades: no shed levels at all, so it keeps mmproj and MTP even
when that leaves it only `min_context` (4096). With no goals to protect,
capability is the only thing left to give up. It joins no set — routing is
what categories are for.

`FALLBACK.sacrifice is None` means *never degrade*. An empty tuple means *no
preference stated* and still concedes in default order. Conflating those two
would silently disable degradation for every category that expressed no
preference.

## Ids carry no context

`<stem>[.<category>][.<flags>]` — no `###k`. A client caches the id at
startup, so 182k → 180k renamed the model out from under it. The context
lives in the display name and `metadata.ctx_size`, both of which a client
re-reads. An untagged or unmet entry is just the bare stem.

## Parallel: the same search, a different exponent

The fallback runs the same `(ctx, slots)` value-function solve as a category
— same code path, not a second one — with slot exponent **0**. Exponent 0
makes slots worthless in the score, so the choice is made purely on context:
a 4B model whose VRAM is plentiful holds max context at several slot counts,
and that is exactly the case that should take more of them.

> Note: at exponent 0 a 4B model at 40k max scores identically at every slot
> count that holds 40k, so the tie-break has to favour **more** slots. The
> current loop keeps the *first* maximum (parallel 1). This is the one place
> the "low parallel_power plus plenty of VRAM" description does not fall out
> of the existing tie-break, and it is the next thing to implement.

## MTP is required by coding, not merely preferred

`coding` requires MTP when a draft companion exists and the solved context
stays above the category's `min_context`. That is stricter than vision,
which coding sheds freely. Modelled as `require: [draft]` on the category —
data, not code — so it stays expressible.

## Allocation: residents then remainder

Two tiers, and only two.

**Residents** are pinned for as long as the service is up, because losing
them means losing the KV cache and reloading on every lookup. Embeddings and
rerank are the clear case; STT and TTS are the optional ones. Each resident
serves a *fixed small context* — `embed_context` / `rerank_context`, default
4096. Two reasons, and the second is the important one:

- clients standardise on one RAG context and chunk to it, so the embed size
  is a stable contract rather than a function of what else is loaded;
- **a fixed small context makes the resident tier's VRAM a number you can
  compute up front**, which is what makes the remainder meaningful. A
  context that flexed with load would make the budget move under you.

That cap is not a compromise to be revisited when VRAM is tight — it is the
thing that lets the rest of the card be allocated confidently. If VRAM were
the binding constraint, more tokens in main chat context is worth more than
more in RAG context. When a resident has to grow — needing STT resident as
well as RAG — it grows slightly within the tier rather than by introducing a
second mechanism.

**Everything else loads on demand.** Chat is not pinned. On this box that is
the whole picture: one resident tier, one remainder, no further tiers to
reason about. `matrix:` in its current form also carries `evict_costs:` and
four-way set expressions (`ragz`, `voice`, `talk`) which exist to answer
"which neighbours ride along with chat"; that question disappears when the
resident tier is one list with fixed contexts.

```
resident_vram  = Σ VRAM(resident, at its own fixed context)
chat_budget    = vram_total − reserve − spare − resident_vram
```

Then, per chat model, per declared category *and* the fallback:

```
solve (shed, ctx, parallel) from chat_budget
```

Each member resolves its own context and decides its own mmproj, against
the same number. There is no shared `chat_ctx` and no per-set re-solve —
those were two mechanisms answering one question, and every bug where one
pass read a value the other decided came from that.
