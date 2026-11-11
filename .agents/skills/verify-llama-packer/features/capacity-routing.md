# Capacity routing

Without `matrix.sets:`, llama-swap gets a capacity router: global pools plus
per-entry `capacities` and an `evict_cost`. Any combination that fits may be
resident.

## Sub-features

- `routing.router.use` is `capacity`.
- The pools carry the GPU budget (`default`), the host pool (`ram`), and one
  size-1 pool per role.
- Each entry carries `capacities` and `evict_cost`.

## How to get to it (user POV)

Pack without `matrix.sets:` and inspect the generated config, or ask the live
router for `/v1/models` and watch entries evict by cost.

## Driving it with the shell harness

```sh
.agents/skills/verify-llama-packer/scripts/run.sh .verify/capacity
python3 - <<'PY'
import yaml
c = yaml.safe_load(open('.verify/capacity/config.yaml'))
assert c['routing']['router']['use'] == 'capacity'
pools = c['routing']['router']['settings']['capacity']['pools']
assert pools['default'] == 22528          # 24576 - 2048 reserve, no spare
assert pools['chat'] == 1
entry = c['models']['m']
assert entry['capacities']['default'] == pools['default']
assert entry['evict_cost'] >= 0
print('capacity ok')
PY
```

`--vram 24576` makes `default` deterministic. With a spare, subtract it.

## Gotchas

- The numbers depend on `--vram`, `--spare`, `--baseline`, and the detected
  host pool; assert the relationship, not a single host-derived number.
- `matrix.sets:` present switches the router to `matrix`; that is a different
  feature, not a failure here.
- A CPU-resident entry has no `capacities.default` charge; it charges `ram`.
