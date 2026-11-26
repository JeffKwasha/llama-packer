# Taxonomy

The taxonomy links every traced requirement to the doc that owns it and the
test that holds it up. `AGENTS.md` §References in docs defines the legal
reference forms; this page defines the IDs and the gate.

## IDs

`<FAMILY>_<SUBJECT>_<GUARANTEE>`, UPPER_SNAKE_CASE, like a global constant.
The name states the rule; a reader should not need the doc to know what it
means. There are no numbers, sequence suffixes or holes.

| Family | Layer | Authority | Covers |
|---|---|---|---|
| `USER` | demand | `README.md`, `FAQ.md` | Operator-visible promises |
| `ARCH` | design | `docs/architecture.md`, `docs/glossary.md` | Ownership, boundaries, vocabulary |
| `FLOW` | design | `SPEC.md`, `docs/allocation.md`, `docs/memory-and-eviction.md`, `docs/roles-tasks-routing.md`, `docs/profiles-authoring.md` | Algorithms, config semantics, sizing |
| `DEPLOY` | design | `docs/install.md`, `docs/memory-and-eviction.md` §Live VRAM attribution | Install, update, observability |

Examples: `FLOW_TASK_DECLARED_ONLY`, `FLOW_CTX_LENGTH_CEILING`,
`FLOW_FILL_CLAIMS_RESIDUAL`, `DEPLOY_UPDATE_REPORTS_SKIPS`.

A `USER` row states a demand and never cites a design ID. A design row may
cite the `USER` IDs it satisfies in its `Satisfies` column. Rename an ID when
its rule changes meaning; update the owner doc, registry, tests and code in
one change. Delete an ID and every citation when its rule dies.

## Registry and citation

`docs/plans/requirement-trace.md` is the registry. Each row carries `ID`,
`Owner` (a doc path plus section), `Satisfies`, `Requirement` and `Tests`.

Tests named in the registry start their body with:

```
# Validates: FLOW_FILL_CLAIMS_RESIDUAL — docs/allocation.md §The auction
```

Prose may follow, but the ID is first. Other tests may cite a doc section
until they are traced; a test that cites an unknown ID fails the gate.

Code cites an ID or a doc section in a one-line docstring or comment. Long
explanatory docstrings are replaced by the taxonomy reference.

## Gate

`python extras/check_trace.py` checks every registry row:

1. a unique, well-formed ID with a known family prefix and no banned token;
2. an `Owner` file and section that resolve;
3. at least one named test, and every named test exists;
4. every named test cites that exact ID;
5. no test cites an ID absent from the registry;
6. `docs/requirement-index.md` matches the generated index.

`extras/check_trace.py --write` regenerates the index. The gate reports drift;
it never edits product code. `extras/check_mutation.py` remains the separate
mutation gate.
