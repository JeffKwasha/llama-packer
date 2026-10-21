# The benchmark tool

`extras/bench.py` measures a model with `llama-bench` and compares the
result with the last run of the same configuration. The rules below are the
ones its tests hold up.

## The history is append-only

- One JSON line per measurement, appended to
  `$XDG_CACHE_HOME/llama-packer/bench.jsonl` (else `~/.cache/...`).
- **An existing record is never rewritten.** A benchmark history is evidence,
  and the useful property of evidence is that appending to it cannot rewrite
  what came before.
- **A truncated last line does not break the history.** A crash mid-write
  costs one line, not the file.
- **A missing history reads as empty**, not as an error.

## What counts as the same measurement

- The identity key is the configuration: engine, version, backend, device,
  model, batch, ubatch and metric.
- **The timestamp and the repetition count are not part of it**, and neither
  is the host load: two runs are comparable when they measured the same thing
  on the same configuration.
- **The device identity is the card name and its architecture** — not the
  free-memory counter, which changes every run. A different card is a
  different measurement.
- The comparison is against **the most recent earlier** record with the same
  key, and the delta is relative to that one.

## Reading a row

- **A row is selected by its fields** (`n_prompt`, `n_gen`), never by its
  position: `llama-bench` skips a case rather than omitting it, so selecting
  on position is how a prompt-processing number gets reported as a generation
  number.
- **A missing row is an error, not a zero.**
- **Empty bench output is an error.**
- **The standard deviation travels with the mean** — a run whose spread could
  invert a 5% change has not measured 5%.

## The report

- The measurement and, where there is one, the change since the previous run
  of the same configuration.
- **The first run of a configuration says so**, instead of reporting a
  delta against nothing.
- **With no history, the report says so** rather than printing empty numbers.
- Records are listed **newest last**.
- **The unit follows the metric name**: an RTF record prints as RTF, not as
  tokens per second, and a prompt-throughput record prints its own number.

## Picking what to measure

- **The default is the model the fleet actually runs**, not the newest file
  on disk.
- **A model over the size cap falls back to the largest that fits.**
- **The audio entries come from the config's own ids**, and the audio
  backend is read out of the emitted payload.
- Indirect paths in the config are expanded.

## Versions and builds

- **The version is read from the build number.**
- **A binary that cannot run still yields a version**, so a broken build is
  still comparable.
- Old builds are found so versions can be compared, and the stable name is
  found when there is no build directory.
- Both device-line spellings are recognised, and the `llama-server` spelling
  still parses.
- **An unknown output format is an error**, not an empty result.
