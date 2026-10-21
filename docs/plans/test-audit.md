# Test audit: does every test hold up part of the design?

A test that passes whether or not the code is there enforces nothing. This
audit proves the suite against mutation and records what each test file
holds up.

**Verdict: nothing is decoration, and three tests were not holding the rule
they named — all three are now proven.** The work this produced is therefore
explanatory, not subtractive: every test names the rule it holds up, and the
tool that proves it is in the tree.

## Method

`extras/find_decorated_tests.py` does the proving, one test file at a time:

1. Measure which package functions the file actually executes (coverage).
2. Mutate each of them — the body returns a wrong value of the same shape,
   or, for a function that only has side effects, does nothing at all.
3. Run the file's tests against every mutation. A test that fails under at
   least one mutation *detects* it; a test that fails under none of them
   detects nothing and is decoration.

```
python extras/find_decorated_tests.py --test tests/test_writer.py
```

The mutation is reverted after every run, so the tree is untouched.

### What the sweep cannot see

The sweep is a screen, not a proof. It cannot reach:

- **constants** (`_MIN_AGENTIC_CTX`, `FALLBACK`, `KNOWN_CAPABILITIES`, the
  asset definitions in `extras/update.py`),
- **class attributes** (`LlamaServerBackend.roles`),
- **dunder methods** (`Macro.__init__`),
- **file content** (the systemd unit template),
- a **negative** rule ("the CPU backend is exempt", "an unrelated unit's
  failure is not ours"), where doing nothing is also correct,
- code that runs in **another process** — a test that drives the CLI as a
  subprocess is measured at 0% coverage, so the sweep finds only what the
  import did.

Every candidate the sweep produced was therefore checked by hand, with the
mutation that breaks the rule the test claims to hold.

## Per file

| Test file | Tests | Mutants | Detecting a mutation | Candidates | Design parts it holds up |
|---|---|---|---|---|---|
| tests/test_auto_parallel.py | 42 | 159 | 41 | 1 | the (context, slots) value function, the floor cascade, the knobs |
| tests/test_backends.py | 111 | 133 | 109 | 2 | the engine registry, the per-engine formats and roles, the container transports |
| tests/test_bench.py | 38 | 25 | 38 | 0 | the append-only history, the measurement identity, the delta |
| tests/test_build_engine.py | 36 | 31 | 35 | 1 | a source build that cannot see the card, and the live symlink |
| tests/test_companions.py | 9 | 41 | 9 | 0 | the companion VRAM folding, the MTP header gate |
| tests/test_dirconfig.py | 9 | 124 | 9 | 0 | directory defaults and scoped rules, and their precedence |
| tests/test_discovery.py | 33 | 96 | 33 | 0 | multi-directory discovery, role dirs, snapshot refs |
| tests/test_dry_run.py | 10 | 82 | 10 | 0 | --dry-run touches nothing and says what it withheld |
| tests/test_edge_branches.py | 15 | 112 | 15 | 0 | the decision paths the happy path never reaches |
| tests/test_file_refs.py | 20 | 39 | 20 | 0 | the file-ref forms and the pick semantics |
| tests/test_fit_params.py | 21 | 51 | 21 | 0 | what makes a measurement reusable (shape, cache type, source) |
| tests/test_ggufmeta.py | 6 | 6 | 6 | 0 | the on-disk weight footprint |
| tests/test_gpu_state.py | 5 | 9 | 5 | 0 | the measurement guardrails (residents, group kill, lock, journal) |
| tests/test_hardware.py | 36 | 14 | 36 | 0 | the vendor probes, the pool, the GpuProfile precedence |
| tests/test_install.py | 39 | 24 | 36 | 3 | the unit rendering, the atomic install, the tools/ layout |
| tests/test_logging.py | 3 | 3 † | 3 | 0 | the log levels: quiet by default, `-v` lifts it, a preview lifts it |
| tests/test_macros.py | 29 | 21 | 28 | 1 | the observable macro contract |
| tests/test_main.py | 26 | 18 | 25 | 1 | the CLI helpers (health timeout, macro substitution, matrix vars) |
| tests/test_mem_tag.py | 18 | 120 | 18 | 0 | the memory tag grammar and the emitted VRAM number |
| tests/test_memory_probe.py | 13 | 15 | 13 | 0 | the representative probe and the affine fit |
| tests/test_merge.py | 21 | 68 | 21 | 0 | the single merge rule |
| tests/test_model_registry.py | 18 | 68 | 18 | 0 | the canonical instance per weight file, the single writer |
| tests/test_model_resolution.py | 25 | 47 | 25 | 0 | reference resolution, the registry, the accessors |
| tests/test_overrides.py | 24 | 68 | 24 | 0 | the pattern-scoped override rules |
| tests/test_persist.py | 2 | 21 | 2 | 0 | the measured block is the only thing written to a sidecar |
| tests/test_planner.py | 64 | 177 | 64 | 0 | the ladder, the context solve, the matrix solve, the ids |
| tests/test_progress.py | 8 | 35 | 7 | 1 | the bar never fights the log; the measurement is announced |
| tests/test_throughput.py | 5 | 30 | 5 | 0 | the relative speed index and the fields it reads |
| tests/test_transport.py | 18 | 15 | 18 | 0 | the engine x transport seam |
| tests/test_update.py | 137 | 75 | 125 | 12 | the per-backend assets, the cleanup, the prune |
| tests/test_usage.py | 74 | 96 | 71 | 3 | the usage catalogue, the demand resolution, the ladder axes |
| tests/test_utils_parsers.py | 25 | 21 | 25 | 0 | the string grammar, the HF_HOME invariant, the model-kind guard |
| tests/test_utils_paths.py | 15 | 8 | 15 | 0 | the tools/ layout, the build resolution, the drive speed |
| tests/test_vram_budget.py | 54 | 97 | 54 | 0 | the VRAM terms, the context solve, the allocation guard |
| tests/test_vram_log.py | 13 | 13 | 13 | 0 | the GPU probe, the tenant counts, the report, the accumulator |
| tests/test_writer.py | 43 | 185 | 43 | 0 | the entry shape, the capabilities block, the client metadata |
| **total** | **1,065** | **2,147** | **1,040** | **25** | |

† `tests/test_logging.py` drives the CLI as a **subprocess**, so the
sweep's coverage never sees the code it runs: it found only the three
import-time functions, and the three mutants it tried are trivial. Its proof
is the mutation gate instead — three anchors in `extras/check_mutation.py`,
one for each direction the rule can break (the floor always lifted, never
lifted, and the `-v` ladder broken).

The 25 candidates are the sweep's blind spots. Each was broken by hand; each
failed, which is the proof that the test holds its rule.

| Candidate | Broken by | Result |
|---|---|---|
| test_auto_parallel::test_agentic_floor_default_unchanged | `_MIN_AGENTIC_CTX` changed to 65536 | fails |
| test_auto_parallel::test_serving_pin_ignores_the_design_max | `_serving_pin` returns a wrong value | fails |
| test_auto_parallel::test_serving_pin_rejects_non_positive | `_serving_pin` returns a wrong value | fails |
| test_backends::test_basebackend_is_abstract | `BaseBackend` de-abstracted | fails |
| test_backends::test_llama_server_roles_and_formats | `LlamaServerBackend.roles` narrowed | fails |
| test_build_engine::test_a_cpu_build_is_exempt_from_the_device_check | `check_backends` made to always refuse | fails |
| test_install::test_unit_is_enableable | `WantedBy=` removed from the unit template | fails |
| test_install::test_no_trailing_comments_on_directives | a trailing comment added to a directive | fails |
| test_install::test_verify_unit_ignores_unrelated_unit_failures | `verify_unit` made to raise on any message | fails |
| test_macros::test_duplicate_name_replaces_previous | the collision warning removed | fails |
| test_progress::test_progress_noop_under_pytest | both "no bar here" guards removed | fails |
| test_update::test_audio_cpp_is_a_bundle_because_its_libs_must_travel | the audio-cpp asset's `bundle` flag cleared | fails |
| test_update::test_cleanup_old_versions_noop_when_not_owner | the ownership guard removed | fails |
| test_update::test_cpu_pattern_does_not_match_a_vulkan_asset | the cpu pattern widened | fails |
| test_update::test_each_backend_matches_its_own_asset[latest-cpu-…] | the cpu pattern widened | fails |
| test_update::test_prune_swap_keeps_newest_but_never_a_referenced_file | the "still referenced" check inverted | fails |
| test_update::test_sd_is_a_zip_bundle | the sd-server asset's format changed | fails |
| test_update::test_sd_tag_pattern_matches_and_rejects | the sd-server tag pattern widened | fails |
| test_update::test_update_cpp_symlinks_absent_target_is_noop | the missing-target guard removed | fails |
| test_usage::test_fallback_is_a_bare_minimum_category | `_MIN_CTX_SIZE` changed to 8192 | fails |
| test_usage::test_keep_protects_from_shedding_without_gating_existence | `kept` dropped from the protected set | fails |
| test_usage::test_known_capabilities_are_the_ladder_axes | a third ladder axis added | fails |

## The three that were not holding their rule

Two were *weaker* than the rule they named — they could not fail for the
reason they claimed — and both were strengthened rather than deleted. The
third was *racy*: it could fail for a reason that had nothing to do with the
rule, which is the same defect wearing a different hat.

- `test_main::test_every_backend_uses_the_same_chain` asserted that the
  resolver's *source* mentions `run_dir`, `shutil.which` and `env_var`. The
  docstring mentions the tiers and two of the three are parameter names, so
  removing the PATH tier from the code left the test green. The search now
  covers the body only.
- `test_update::test_prune_never_removes_the_bundle_a_symlink_points_at`
  pointed at the *newest* bundle, which the keep count protected anyway, so
  removing the "still referenced" guard left the test green. A third, newer
  bundle now puts the referenced one outside the keep count.
- `test_gpu_state::test_kill_process_group_kills_descendants` asserted that
  the process *group* is gone the instant after the kill. The grandchild is
  an orphan the moment the leader dies, and until init reaps it the zombie
  still belongs to the group — so the assertion raced init's bookkeeping and
  failed about one run in forty (it reddened the CI of a later branch twice).
  The test now asks what the rule is about: the descendant is dead. Gone, or
  a zombie waiting for init — either way not running. It still fails when the
  group kill is broken down to the leader alone.

## What changed

- **Every test names its rule, and names it as a requirement in a design
  doc.** Each test carries a `Validates:` line of the form
  `Validates: <doc>, '<section>': <the rule>` — `SPEC.md, 'Context Tiers'`,
  `docs/plans/requirement-trace.md #21`, `docs/backends/llama-server.md,
  'Command shape'`. No citation points at a function: a function is where a
  rule is implemented, not where it is stated, and a citation that names the
  code cannot be checked against the design.
- **Each test file says in its docstring which part of the design it
  covers**, again by document and section.
- **Three tests were made to hold the rule they name** (above), each with the
  mutation that exposed it.
- **No test was deleted.** None was decoration.
- **The proof is in the tree.** `extras/find_decorated_tests.py` runs the
  sweep for any test file; `extras/check_mutation.py` still proves the
  traced requirements against their own mutations, and now supports
  replacement as well as deletion.

### The rules the design did not state

Pointing the citations at documents surfaced the rules that no document
stated. Those are now written down, because a rule with no home in the design
cannot be traced:

- **`docs/bench.md`** (new) — the benchmark tool's contract: the append-only
  history, what counts as "the same measurement", selecting a row by its
  fields, and the report.
- **`docs/build-engine.md`** (new) — building an engine from source: the
  `src` naming, the target detection, the toolchain check, the patches, and
  what the install does with the tree.
- **`docs/install.md`, "The updater command line"** (new) — the updater's
  positional targets, its flags, the version and architecture rules, the
  asset match, the checksum policy and the pruning rules.
- **`SPEC.md`, "Context Tiers"** — the context-size grammar (a positive
  integer, or a `128k` / `1M` string).
- **`SPEC.md`, "Measured-Block Persistence"** — the `--dry-run` contract:
  it creates, modifies and deletes no files, and every withheld write is
  reported.
- **`docs/plans/phase0-vram-baseline.md`, "The log format"** — the VRAM log's
  CSV shape and the count-weighted report.

Four citations remain without a document: the progress bar's own behaviour
(`tests/test_progress.py`). It is a UI detail with no requirement behind it,
and those tests say so in as many words.

### The four bugs the sweep itself had

The screen was wrong before the screen was right, and each bug hid
candidates:

1. A function with no return value was never mutated, so a test pinning a
   side effect could not be judged.
2. The summary line was cut at the first whitespace, so a parametrised id
   containing a space was recorded under a truncated name.
3. Only `FAILED` lines were collected; a mutant that makes a test raise
   reports an `ERROR`, and an import-time break blanks the whole file.
4. CPython keys a `.pyc` on the source's mtime (whole seconds) and its
   size, so two mutations of the same size inside one second shared a
   cached module — the second mutant ran the first one's code.

## Reproducing

```
python extras/find_decorated_tests.py --test tests/test_writer.py
python extras/check_trace.py
python extras/check_mutation.py
python -m pytest tests -q
```
