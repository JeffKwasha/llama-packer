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

## Sets

Admission to a routing set is per category and uses the *resolved* serving,
not the model: a model is admitted to the set its emitted category entry can
actually serve alongside that set's residents. A category that cannot be met
alongside RAG is not admitted to that set; the model's fallback entry stays
standalone, and sideload routing goes through the categories.