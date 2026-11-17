# Working on this codebase

Rules that keep the design and the code telling the same story. They are
practice, not history: a rule that stops being true should be deleted, not
softened.

## Design before code

- **Transcribe stable decisions into `docs/` before any code moves.** When a
  conversation settles a design, write the decision and its open questions
  into the owning doc (`docs/architecture.md`, `docs/allocation.md`, …). Code
  that lands first becomes the de facto design, and the conversation that
  produced it is gone.
- **A reversal is an undo, not a new layer.** When a decision is rejected
  after part of it landed, record what the reversal removes in the same doc
  that claimed it. Otherwise green code and green tests read as settled
  design, and the next change builds on a decision nobody holds any more.
- **Map every runtime claim to the emitted surface.** For a config generator,
  write the table of design element → emitted config field *before* building.
  A capability that has no field is either a llama-swap feature to add or a
  design element to drop; "it will work at runtime" is not a mapping.

## Claims

- **Verify a "dead config" claim against the code before repeating it.** A
  claim about a knob, a key, or a function is cheap to check and expensive to
  propagate. When one turns out wrong, grep the *claim*, not only the origin:
  docs copy each other, and example files copy docs.
- **Cite the taxonomy, never a line number.** Section names, symbols and
  requirement ids survive refactors; `file.py:123` and PR numbers do not.

## Tests

- **A mutation anchor must be unique to the rule it names.** The mutation
  checker replaces one match per file. If a test still passes under the
  anchor, the anchor is testing something else, or several code paths cover
  the same rule and the named test is not the one holding it.
- **Each requirement names a test, and each test cites its rule.**
  `extras/check_trace.py` proves both directions against
  `docs/plans/requirement-trace.md`; `extras/check_mutation.py` proves the
  anchors.
- **A test that would pass with the implementation deleted is decoration.**
  `extras/find_decorated_tests.py` finds those by mutation; delete or rewrite
  them rather than keeping them green.

## Tooling

- **A checker must not trust a PATH tool for its own core logic.** `grep`,
  `find` and `sed` are replaced on some hosts by wrappers with different
  behaviour; a checker that shells out to them can report "nothing found" and
  exit zero. Parse with the standard library. Remaining subprocess calls should
  run the thing under test (`pytest`, an engine binary), not inspect files.

## Landing

- **Portion commits with gates.** One portion per commit, each green before
  the next: the suite, `check_trace`, `check_mutation`, and for a config
  change a live `llama-swap -validate`. A portion that cannot pass its gate
  is not finished, however small the diff.
- **Measure before and after a claim about speed or memory.** A number with
  no baseline is an anecdote; `docs/bench.md` owns the benchmark contract.
