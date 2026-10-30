# tests/test_vram_budget.py
"""calc_ctx and solve_matrix_ctx budget math under the affine VRAM law:

    VRAM(C, p) = model_mib + compute_mib + c*C + p*D

with C the total KV pool (= per-slot context X times parallel) and X the
solved, emitted value.
"""

from __future__ import annotations

import json
import os
import struct

import pytest

from _gguf import write_gguf
from llama_packer import gpu_state, vram
from llama_packer.consts import _MIN_CTX_SIZE
from llama_packer.vram import (FitParams, parse_device_buffers,
                               parse_fit_log, parse_host_buffers,
                               parse_spill_mib, solve_matrix_ctx)


# ── serve-shaped log parsing ──────────────────────────────────────────────

def test_parse_device_buffers_sums_device_only():
    # Validates: docs/plans/estimation-functions-facts.md, 'The affine model': Sum device-resident buffer lines of a llama-server ``-lv 5`` log. Returns MiB per component: ``weights`` (max across load passes: the.
    """Real Dirk (-text) numbers: weights max, both contexts' KV/RS/compute,
    host-visible buffers excluded."""
    text = """\
0.00.437 I load_tensors:      Vulkan0 model buffer size =     0.00 MiB
0.24.044 I load_tensors:   CPU_Mapped model buffer size =   994.63 MiB
0.24.044 I load_tensors:      Vulkan0 model buffer size = 18904.68 MiB
0.28.744 I llama_kv_cache:    Vulkan0 KV buffer size =  8704.00 MiB
0.29.025 I llama_kv_cache:    Vulkan0 KV buffer size =  1024.00 MiB
0.28.115 I llama_memory_recurrent:    Vulkan0 RS buffer size =   448.88 MiB
0.28.928 I sched_reserve:    Vulkan0 compute buffer size =  1552.33 MiB
0.28.928 I sched_reserve: Vulkan_Host compute buffer size =  1104.34 MiB
0.894.943 I sched_reserve:    Vulkan0 compute buffer size =  1312.06 MiB
0.00.440 I llama_context: Vulkan_Host  output buffer size =     0.95 MiB
"""
    b = parse_device_buffers(text)
    assert b["weights"] == pytest.approx(18904.68)  # max, not sum; host/CPU out
    assert b["kv"] == pytest.approx(8704.00 + 1024.00)  # main + draft contexts
    assert b["rs"] == pytest.approx(448.88)
    assert b["compute"] == pytest.approx(1552.33 + 1312.06)  # host excluded
    assert b["output"] == 0.0  # host-visible → excluded


def test_parse_device_buffers_empty_log():
    # Validates: docs/plans/estimation-functions-facts.md, 'The affine model': Sum device-resident buffer lines of a llama-server ``-lv 5`` log. Returns MiB per component: ``weights`` (max across load passes: the.
    assert parse_device_buffers("garbage") == {
        "weights": 0.0, "kv": 0.0, "rs": 0.0, "compute": 0.0, "output": 0.0}


# ── final-pass buffer window + host bucket (cpu-residency-budget) ─────────

#: Real Qwen3-Reranker-0.6B CPU-shaped serve log (2026-10-23 probe): the
#: projection pass (`load_mode = none`) prints real compute sizes, so a
#: whole-log sum would double count; each pass ends in a breakdown table.
_CPU_SERVE_LOG = """\
0.00.174.400 I load_tensors:          CPU model buffer size =     0.00 MiB
0.00.176.135 I llama_context:        CPU  output buffer size =     0.58 MiB
0.00.176.234 I llama_kv_cache:        CPU KV buffer size =     0.00 MiB
0.00.182.992 I sched_reserve:        CPU compute buffer size =    36.10 MiB
0.00.183.066 I common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
0.00.183.066 I common_memory_breakdown_print: |   - Host               |                  978 =   466 +     476 +      36                |
0.05.260.351 I load_tensors:   CPU_Mapped model buffer size =     0.00 MiB
0.05.260.351 I load_tensors:   CPU_Mapped model buffer size =   121.50 MiB
0.05.260.351 I load_tensors:   CPU_Mapped model buffer size =   344.78 MiB
0.05.263.659 I llama_context:        CPU  output buffer size =     0.58 MiB
0.05.263.779 I llama_kv_cache:        CPU KV buffer size =   476.00 MiB
0.05.466.825 I sched_reserve:        CPU compute buffer size =    36.10 MiB
"""


def test_parse_host_buffers_reads_final_pass():
    # Validates: docs/plans/cpu-residency-budget.md, 'Requirements (citable by tests)': The parse reads the **final pass** — the buffer lines between the last two breakdown headers — instead of summing duplicated pass lines; and: reports host-side (non-device) sums.
    """Host sums from the final pass only: pass-1's real compute (36.10)
    must not double count, and the host weights equal the breakdown's
    model column (two CPU_Mapped buffers sum, 121.50 + 344.78)."""
    host = parse_host_buffers(_CPU_SERVE_LOG)
    assert host["weights"] == pytest.approx(121.50 + 344.78)
    assert host["kv"] == pytest.approx(476.00)
    assert host["rs"] == 0.0
    assert host["compute"] == pytest.approx(36.10)  # final pass, not 2x
    assert host["output"] == pytest.approx(0.58)


def test_parse_host_buffers_headerless_log_parses_whole():
    # Validates: docs/plans/cpu-residency-budget.md, 'Requirements (citable by tests)': a log with no breakdown header parses whole (fixture/fit-params compatible).
    text = """\
0.05.260.351 I load_tensors:   CPU_Mapped model buffer size =   344.78 MiB
0.05.263.779 I llama_kv_cache:        CPU KV buffer size =   476.00 MiB
"""
    host = parse_host_buffers(text)
    assert host["weights"] == pytest.approx(344.78)
    assert host["kv"] == pytest.approx(476.00)


def test_parse_device_buffers_whole_log_device_path():
    # Validates: docs/plans/cpu-residency-budget.md, 'Requirements (citable by tests)': the device path keeps its whole-log sums — the projection pass of a device run measures 0 — and never touches host sums.
    """Device sums are unchanged by this refactor: host-side values of a
    CPU-shaped run contribute nothing to them (excluded upstream by the
    spill check)."""
    device = parse_device_buffers(_CPU_SERVE_LOG)
    assert device == {
        "weights": 0.0, "kv": 0.0, "rs": 0.0, "compute": 0.0, "output": 0.0}


def test_parse_spill_rejects_vulkanless_fullcpu_launch():
    # Validates: docs/plans/cpu-residency-budget.md, 'Requirements (citable by tests)': a ``-ngl 999`` launch that lost Vulkan still measures nothing — the spill check rejects it exactly as before.
    """A `-ngl 999` run that never got the GPU serves the full-CPU shape;
    the device-shaped spill formula sees host-visible KV and refuses."""
    assert parse_spill_mib(_CPU_SERVE_LOG) == pytest.approx(476.00)


def test_run_measure_server_cpu_shape_returns_host_buffers(
        make_model, tmp_path, guard_env):
    # Validates: docs/plans/cpu-residency-budget.md, 'Requirements (citable by tests)': ``device: cpu`` measures (and shape-stamps) with ``--n-gpu-layers 0`` instead of ``-ngl 999`` … and never runs the device-shaped spill rejection.
    model = make_model("a", device="cpu")
    stub = _stub_server(
        tmp_path,
        "printf '%s\\n' \"$@\" > \"$0.args\"\n"
        + "".join(f"echo '{line}'\n" for line in _CPU_SERVE_LOG.splitlines())
        + "echo listening on http://127.0.0.1:1\nexec sleep 30\n")
    buffers = model.vram._run_measure_server(stub, "q8_0", 8192, 1, "")
    assert buffers is not None
    assert buffers["kv"] == pytest.approx(476.00)
    assert buffers["weights"] == pytest.approx(466.28)
    args = (tmp_path / "stub-measure-server.sh.args").read_text().splitlines()
    assert args.count("--n-gpu-layers") == 1  # the shape adds nothing
    assert args[args.index("--n-gpu-layers") + 1] == "0"
    entry = json.loads((guard_env / "measure-journal.jsonl")
                       .read_text().splitlines()[-1])
    assert entry["outcome"] == "ok"


def test_run_measure_server_gpu_refuses_vulkanless_fullcpu(
        make_model, tmp_path, guard_env):
    # Validates: docs/plans/cpu-residency-budget.md, 'Requirements (citable by tests)': a ``-ngl 999`` launch that lost Vulkan still measures nothing.
    model = make_model("a")
    stub = _stub_server(
        tmp_path,
        "printf '%s\\n' \"$@\" > \"$0.args\"\n"
        + "".join(f"echo '{line}'\n" for line in _CPU_SERVE_LOG.splitlines())
        + "echo listening on http://127.0.0.1:1\nexec sleep 30\n")
    assert model.vram._run_measure_server(stub, "q8_0", 8192, 1, "") is None
    args = (tmp_path / "stub-measure-server.sh.args").read_text().splitlines()
    assert args[args.index("-ngl") + 1] == "999"
    entry = json.loads((guard_env / "measure-journal.jsonl")
                       .read_text().splitlines()[-1])
    assert entry["outcome"] == "spill"


def test_parse_fit_log_host_row_fallback():
    # Validates: docs/plans/cpu-residency-budget.md, 'Requirements (citable by tests)': ``parse_fit_log`` falls back to the ``Host`` report row when no device row exists.
    """Captured CPU-shaped fit-params run: the affine report row is named
    ``Host`` — today the parse rejects the whole run."""
    text = """\
0.04.426.841 D load_tensors:          CPU model buffer size =     0.00 MiB
0.04.428.293 D llama_kv_cache:        CPU KV buffer size =     0.00 MiB
0.04.428.298 D llama_kv_cache: size =  476.00 MiB (  8192 cells,  28 layers,  1/1 seqs), K (q8_0):  238.00 MiB, V (q8_0):  238.00 MiB
0.04.434.024 D sched_reserve:        CPU compute buffer size =   316.33 MiB
0.00.422.303 I common_memory_breakdown_print: |   - Host               |                 1258 =   466 +     476 +     316                |
Host 466 476 316
"""
    parsed = parse_fit_log(text)
    assert parsed is not None
    assert parsed["model"] == 466
    assert parsed["context"] == 476
    assert parsed["compute"] == 316
    assert parsed["kv"] == pytest.approx(476.00)
    assert parsed["rs"] == 0.0 and parsed["rs_cells"] == 0


def test_parse_fit_log_prefers_device_row():
    # Validates: docs/plans/estimation-functions-facts.md: the device report row; a ``Host`` row beside it never wins.
    text = """\
Vulkan0 18904 8704 1552
Host 466 476 316
"""
    parsed = parse_fit_log(text)
    assert parsed is not None
    assert parsed["model"] == 18904
    assert (parsed["context"], parsed["compute"]) == (8704, 1552)


def test_measure_affine_trio_cpu_shape_persists_host_quad(
        make_model, tmp_path):
    # Validates: docs/plans/cpu-residency-budget.md, 'Requirements (citable by tests)': the affine trio (``c`` from ``(C, C/2)``, ``D`` from the p pair) derives CPU quads without measuring anything anew.
    """Stub fit-params replays the three captured CPU-shaped runs
    (8192/1, 8192/2, 4096/1); c = (476−238)/4096, D = 0, weights 466.
    The persisted block carries the CPU device in its shape stamp."""
    stub = _stub_fit_run(tmp_path)
    model = make_model("a", device="cpu", context_length=8192)
    model.vram.remeasure = False
    params = model.vram.fit_params_static(
        stub, cache_type="q8_0", llama_args="--n-gpu-layers 0",
        allow_cpu=True)
    assert params is not None
    assert params.model_mib == 466
    assert params.kv_per_token_mib == pytest.approx(
        (476.0 - 238.0) / (8192 - 4096))
    assert params.slot_mib == 0.0
    assert params.compute_mib == 316
    assert params.source == "fit-estimate"
    assert params.shape == "--n-gpu-layers 0"
    # persisted under the same shape → a later call reuses it unchanged
    again = model.vram.fit_params_static(stub, cache_type="q8_0",
                                         llama_args="--n-gpu-layers 0",
                                         allow_cpu=True)
    assert again == params


def test_host_measurement_does_not_leak_into_vram_pricing(make_model,
                                                         tmp_path):
    # Validates: docs/plans/cpu-residency-budget.md #11/#27 -- a host-RAM
    # measurement (allow_cpu=True, the host ledger) caches the same quad; a
    # VRAM-priced call must still see None, or the CPU resident is charged to
    # the chat solve after all.
    stub = _stub_fit_run(tmp_path)
    model = make_model("a", device="cpu", context_length=8192)
    model.vram.remeasure = False
    assert model.vram.fit_params_static(
        stub, cache_type="q8_0", llama_args="--n-gpu-layers 0",
        allow_cpu=True) is not None
    assert model.vram.fit_params_static(
        stub, cache_type="q8_0", llama_args="--n-gpu-layers 0") is None


#: Captured CPU-shaped llama-fit-params runs (Qwen3-Reranker-0.6B-Q6_K,
#: q8_0, `--n-gpu-layers 0`, 2026-10-23 probe): the affine trio replays.
_P1_RUN = """\
0.04.426.841 D load_tensors:          CPU model buffer size =     0.00 MiB
0.04.428.293 D llama_kv_cache:        CPU KV buffer size =     0.00 MiB
0.04.428.298 D llama_kv_cache: size =  476.00 MiB (  8192 cells,  28 layers,  1/1 seqs), K (q8_0):  238.00 MiB, V (q8_0):  238.00 MiB
0.04.434.024 D sched_reserve:        CPU compute buffer size =   316.33 MiB
0.00.422.303 I common_memory_breakdown_print: |   - Host               |                 1258 =   466 +     476 +     316                |
Host 466 476 316
"""

_P2_RUN = """\
0.00.165.274 D load_tensors:          CPU model buffer size =     0.00 MiB
0.00.166.729 D llama_kv_cache:        CPU KV buffer size =     0.00 MiB
0.00.171.332 D llama_kv_cache: size =  476.00 MiB (  8192 cells,  28 layers,  1/2 seqs), K (q8_0):  238.00 MiB, V (q8_0):  238.00 MiB
0.00.171.728 D sched_reserve:        CPU compute buffer size =   312.33 MiB
0.00.172.375 I common_memory_breakdown_print: |   - Host               |                 1254 =   466 +     476 +     312                |
Host 466 476 312
"""

_R3_RUN = """\
0.00.164.528 D load_tensors:          CPU model buffer size =     0.00 MiB
0.00.165.665 D llama_kv_cache:        CPU KV buffer size =     0.00 MiB
0.00.165.670 D llama_kv_cache: size =  238.00 MiB (  4096 cells,  28 layers,  1/1 seqs), K (q8_0):  119.00 MiB, V (q8_0):  119.00 MiB
0.00.171.438 D sched_reserve:        CPU compute buffer size =   312.33 MiB
0.00.171.438 I common_memory_breakdown_print: |   - Host               |                 1016 =   466 +     238 +     312                |
Host 466 238 312
"""


def _stub_fit_run(tmp_path):
    """Fake llama-fit-params: replays the captured CPU trio, selected by
    the `-c`/`--parallel` value it was invoked with."""
    for name, text in (("p1", _P1_RUN), ("p2", _P2_RUN), ("r3", _R3_RUN)):
        (tmp_path / f"fit-{name}.out").write_text(text)
    stub = tmp_path / "stub-fit-params.sh"
    stub.write_text(
        "#!/bin/sh\n"
        'c=""; p=""; prev=""\n'
        'for arg in "$@"; do\n'
        '  [ "$prev" = "-c" ] && c="$arg"\n'
        '  [ "$prev" = "--parallel" ] && p="$arg"\n'
        '  prev="$arg"\n'
        'done\n'
        f'if [ "$c" = "4096" ]; then exec cat "{tmp_path}/fit-r3.out"; fi\n'
        f'if [ "$p" = "2" ]; then exec cat "{tmp_path}/fit-p2.out"; fi\n'
        f'exec cat "{tmp_path}/fit-p1.out"\n')
    stub.chmod(0o755)
    return str(stub)





# ── spill validity check (2026-09-07 post-mortem) ─────────────────────────

def test_parse_spill_mib_host_kv_is_spill():
    # Validates: docs/plans/auto-parallel.md, 'Findings: the affine VRAM law (2026-09-07, SEAL record)': parse_spill_mib rejects host-KV capacity spills
    """Host KV = the capacity-spill signature; structural CPU-mapped
    weights (<= 2 GiB) and host compute/output staging are not spill."""
    text = """\
0.00.437 I load_tensors:      Vulkan0 model buffer size = 18904.68 MiB
0.24.044 I load_tensors:   CPU_Mapped model buffer size =   994.63 MiB
0.28.744 I llama_kv_cache:    Vulkan0 KV buffer size =  4352.00 MiB
0.28.745 I llama_kv_cache: Vulkan_Host KV buffer size =  4352.00 MiB
0.28.928 I sched_reserve: Vulkan_Host compute buffer size =  1104.34 MiB
0.00.440 I llama_context: Vulkan_Host  output buffer size =     0.95 MiB
"""
    assert parse_spill_mib(text) == pytest.approx(4352.00)


def test_parse_spill_mib_cpu_map_beyond_tolerance():
    # Validates: docs/plans/auto-parallel.md, 'Findings: the affine VRAM law (2026-09-07, SEAL record)': parse_spill_mib tolerates llama.cpp's structural CPU weight placement
    """CPU-mapped weights above the structural headroom = offload failure."""
    text = """\
0.00.437 I load_tensors:      Vulkan0 model buffer size =  8000.00 MiB
0.24.044 I load_tensors:   CPU_Mapped model buffer size =  12000.00 MiB
0.28.744 I llama_kv_cache:    Vulkan0 KV buffer size =  1000.00 MiB
"""
    assert parse_spill_mib(text) == pytest.approx(12000.00 - 2048.0)


def test_parse_spill_mib_clean_log_is_zero():
    # Validates: docs/plans/auto-parallel.md, 'Findings: the affine VRAM law (2026-09-07, SEAL record)': parse_spill_mib rejects host-KV capacity spills while tolerating llama.cpp's structural CPU weight placement
    text = """\
0.00.437 I load_tensors:      Vulkan0 model buffer size = 18904.68 MiB
0.28.744 I llama_kv_cache:    Vulkan0 KV buffer size =  8704.00 MiB
0.28.928 I sched_reserve:    Vulkan0 compute buffer size =  1552.33 MiB
0.28.928 I sched_reserve: Vulkan_Host compute buffer size =  1104.34 MiB
0.00.440 I llama_context: Vulkan_Host  output buffer size =     0.95 MiB
"""
    assert parse_spill_mib(text) == 0.0


# ── guarded serve measurement lifecycle ───────────────────────────────────

def _stub_server(tmp_path, body: str) -> str:
    path = tmp_path / "stub-measure-server.sh"
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return str(path)


_OK_BODY = """\
echo '0.00.437 I load_tensors:      Vulkan0 model buffer size = 18904.68 MiB'
echo '0.28.744 I llama_kv_cache:    Vulkan0 KV buffer size =  8704.00 MiB'
echo '0.28.928 I sched_reserve:    Vulkan0 compute buffer size =  1552.33 MiB'
echo listening on http://127.0.0.1:1
exec sleep 30
"""


@pytest.fixture
def guard_env(tmp_path, monkeypatch):
    """Clear pre-flight, tmp cache dir, fast stall window."""
    monkeypatch.setattr(vram.gpu_state, "llama_residents", lambda: [])
    monkeypatch.setattr(vram, "_MEASURE_STALL_S", 1)
    monkeypatch.setenv("LLAMA_PACKER_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache"


def test_run_measure_server_ok_and_child_dead(make_model, tmp_path, guard_env):
    # Validates: docs/plans/auto-parallel.md, 'Findings: the affine VRAM law (2026-09-07, SEAL record)': One serve-shaped llama-server run → device buffer MiB.
    model = make_model("a")
    buffers = model.vram._run_measure_server(
        _stub_server(tmp_path, _OK_BODY), "q8_0", 8192, 1, "")
    assert buffers is not None
    assert buffers["kv"] == pytest.approx(8704.0)
    entry = json.loads((guard_env / "measure-journal.jsonl")
                       .read_text().splitlines()[-1])
    assert entry["outcome"] == "ok" and entry["pid"]
    with pytest.raises(ProcessLookupError):
        os.killpg(entry["pid"], 0)  # our child died with its measurement


def test_run_measure_server_rejects_spill(make_model, tmp_path, guard_env):
    # Validates: docs/plans/auto-parallel.md, 'Findings: the affine VRAM law (2026-09-07, SEAL record)': One serve-shaped llama-server run → device buffer MiB.
    body = ("echo '0.00 I llama_kv_cache: Vulkan_Host KV buffer size = "
            "8704.00 MiB'\necho listening on http://127.0.0.1:1\nexec sleep 30\n")
    model = make_model("a")
    assert model.vram._run_measure_server(
        _stub_server(tmp_path, body), "q8_0", 8192, 1, "") is None
    entry = json.loads((guard_env / "measure-journal.jsonl")
                       .read_text().splitlines()[-1])
    assert entry["outcome"] == "spill"


def test_run_measure_server_refused_beside_residents(make_model, tmp_path,
                                                     monkeypatch):
    # Validates: docs/plans/auto-parallel.md, 'Findings: the affine VRAM law (2026-09-07, SEAL record)': One serve-shaped llama-server run → device buffer MiB.
    monkeypatch.setattr(vram.gpu_state, "llama_residents",
                        lambda: ["123 jk /x/llama-server -m y"])
    monkeypatch.setenv("LLAMA_PACKER_CACHE_DIR", str(tmp_path / "cache"))
    model = make_model("a")
    # A nonexistent binary proves the refusal happened before any spawn.
    assert model.vram._run_measure_server(
        str(tmp_path / "never-spawned"), "q8_0", 8192, 1, "") is None
    entry = json.loads((tmp_path / "cache" / "measure-journal.jsonl")
                       .read_text().splitlines()[-1])
    assert entry["outcome"] == "refused-residents"


def test_run_measure_server_stall_abandons_quickly(make_model, tmp_path,
                                                   guard_env):
    # Validates: docs/plans/auto-parallel.md, 'Findings: the affine VRAM law (2026-09-07, SEAL record)': One serve-shaped llama-server run → device buffer MiB.
    model = make_model("a")
    stub = _stub_server(tmp_path, "echo starting\nexec sleep 60\n")
    buffers = model.vram._run_measure_server(stub, "q8_0", 8192, 1, "")
    assert buffers is None  # well under _MEASURE_TIMEOUT_S
    entry = json.loads((guard_env / "measure-journal.jsonl")
                       .read_text().splitlines()[-1])
    assert entry["outcome"] == "stall"
    with pytest.raises(ProcessLookupError):
        os.killpg(entry["pid"], 0)


def test_run_measure_server_exit_detected(make_model, tmp_path, guard_env):
    # Validates: docs/plans/auto-parallel.md, 'Findings: the affine VRAM law (2026-09-07, SEAL record)': One serve-shaped llama-server run → device buffer MiB.
    model = make_model("a")
    stub = _stub_server(tmp_path, "exit 1\n")
    assert model.vram._run_measure_server(stub, "q8_0", 8192, 1, "") is None
    entry = json.loads((guard_env / "measure-journal.jsonl")
                       .read_text().splitlines()[-1])
    assert entry["outcome"] == "exited"


# ── calc_ctx ──────────────────────────────────────────────────────────────


def test_calc_ctx_fits_design(make_model, fit_params_block):
    # Validates: SPEC.md, 'Context calculation formula': Calculate max per-slot context size for given VRAM. The solved value is the context available to *each* parallel slot;.
    model = make_model("a", **{"derived": fit_params_block})
    ctx = model.vram.calc_ctx(32768, fit_bin="unused")
    # available = 32768 - 2048 = 30720; remaining = 30720 - 10000 - 1000 = 19720
    # design cost = (0.5*32768 + 0) * 1 = 16384 <= 19720 -> design context
    assert ctx == 32768


def test_calc_ctx_scales_down_and_rounds(make_model):
    # Validates: SPEC.md, 'Context calculation formula': Calculate max per-slot context size for given VRAM. The solved value is the context available to *each* parallel slot;.
    fm = {"model_mib": 25000, "kv_per_token_mib": 0.5, "slot_mib": 0.0,
          "compute_mib": 1000, "cache_type": "q8_0", "source": "llama-server", "shape": ""}
    model = make_model("b", **{"derived": fm})
    ctx = model.vram.calc_ctx(32768, fit_bin="unused")
    # remaining = 30720 - 26000 = 4720; X = 4720/(0.5*1) = 9440 -> rounded 8192
    assert ctx == 8192


def test_calc_ctx_slot_mib_charged_per_slot(make_model):
    # A nonzero D is charged once per slot: at p=2 the KV pool doubles AND
    # the slot cost doubles, so the affordable per-slot context shrinks.
    # Validates: SPEC.md, 'Context calculation formula': Calculate max per-slot context size for given VRAM. The solved value is the context available to *each* parallel slot;.
    fm = {"model_mib": 25000, "kv_per_token_mib": 0.5, "slot_mib": 300.0,
          "compute_mib": 1000, "cache_type": "q8_0", "source": "llama-server", "shape": ""}
    model = make_model("b2", **{"derived": fm})
    p1 = model.vram.calc_ctx(32768, fit_bin="unused", parallel=1)
    p2 = model.vram.calc_ctx(32768, fit_bin="unused", parallel=2)
    # p=1: X = (4720 - 300)/0.5 = 8840 -> 8192
    # p=2: X = (4720 - 600)/(0.5*2) = 4120 -> 4096
    assert p1 == 8192
    assert p2 == 4096

def test_calc_ctx_floors_at_min(make_model, caplog):
    # Validates: SPEC.md, 'Context calculation formula': Calculate max per-slot context size for given VRAM. The solved value is the context available to *each* parallel slot;.
    import logging

    fm = {"model_mib": 30000, "kv_per_token_mib": 0.5, "slot_mib": 0.0,
          "compute_mib": 0, "cache_type": "q8_0",
          "source": "llama-server", "shape": ""}
    model = make_model("c", **{"derived": fm})
    with caplog.at_level(logging.WARNING):
        ctx = model.vram.calc_ctx(32768, fit_bin="unused", spare_mb=1024)
    assert ctx == _MIN_CTX_SIZE
    # The warning must say what the user gets and why: needs vs budget and
    # the verdict (the model cannot load — the planner disables it).
    msg = next(r.message for r in caplog.records
               if "need" in r.message and "budget" in r.message)
    assert "30000" in msg          # needs MiB
    assert "29696" in msg          # budgeted MiB (32768 - 2048 - 1024)
    assert "cannot load" in msg
    # The budget verdict is recorded for the planner's gating check.
    assert model.vram.over_budget_reason is not None
    assert "30000" in model.vram.over_budget_reason
    assert model.vram.vram_squeezed is False


def test_calc_ctx_never_reports_a_context_it_cannot_afford(make_model,
                                                           caplog):
    """The invariant: an emitted entry must never need more VRAM than exists.

    The window this covers is narrow and was a real hole — the model loads,
    so ``remaining > 0`` and no over-budget fires, but the honest context
    left is below the minimum. ``ctx = max(ctx, _MIN_CTX_SIZE)`` clamped
    *upward* over that, and ``_auto_parallel`` infers feasibility from
    ``ctx >= floor``, so the fabricated value was accepted as a real slot
    count. GLM47's c with qwen35's corrected D made that reachable:
    2000 MiB left, 4 slots bought 1863 tokens, reported as 4096.

    Validates: docs/plans/requirement-trace.md #27: no emitted entry may need more
    VRAM than exists.
    """
    import logging
    from llama_packer.vram import _MIN_CTX_SIZE

    fm = {"model_mib": 20471, "kv_per_token_mib": 0.02743133544921875,
          "slot_mib": 448.87, "compute_mib": 1174, "source": "llama-server",
          "cache_type": "q8_0", "shape": ""}
    model = make_model("tight", **{"derived": fm})

    # 24576 - reserve(2048) - weights(20471) - compute(1174) leaves ~883 MiB.
    with caplog.at_level(logging.WARNING):
        ctx = model.vram.calc_ctx(24576, fit_bin="unused", spare_mb=0,
                                  parallel=4, design_ctx=202752)
    assert ctx < _MIN_CTX_SIZE, "reported a servable context"
    assert model.vram.over_budget_reason is not None
    assert "below the" in model.vram.over_budget_reason


def test_calc_ctx_serves_normally_when_the_minimum_is_reachable(make_model):
    # Validates: SPEC.md, 'Context calculation formula': Calculate max per-slot context size for given VRAM. The solved value is the context available to *each* parallel slot;.
    """One MiB more budget and the same model serves -- so the guard above
    is a threshold, not a blanket refusal."""
    fm = {"model_mib": 20471, "kv_per_token_mib": 0.02743133544921875,
          "slot_mib": 448.87, "compute_mib": 1174, "source": "llama-server",
          "cache_type": "q8_0", "shape": ""}
    model = make_model("roomy", **{"derived": fm})
    ctx = model.vram.calc_ctx(24576, fit_bin="unused", spare_mb=0,
                              parallel=1, design_ctx=202752)
    assert ctx >= 4096
    assert model.vram.over_budget_reason is None


def test_calc_ctx_applies_spare(make_model, fit_params_block):
    # Validates: SPEC.md, 'Context calculation formula': Calculate max per-slot context size for given VRAM. The solved value is the context available to *each* parallel slot;.
    model = make_model("d", **{"derived": fit_params_block})
    ctx = model.vram.calc_ctx(32768, fit_bin="unused", spare_mb=3072)
    # available = 32768 - 2048 - 3072 = 27648; remaining = 27648-11000 = 16648
    # design cost = 16384 <= 16648 -> design still fits
    assert ctx == 32768


# ── over-budget / squeeze gating flags ───────────────────────────────────


def test_calc_ctx_over_budget_when_available_nonpositive(make_model, caplog):
    # Validates: SPEC.md, 'Context calculation formula': Calculate max per-slot context size for given VRAM. The solved value is the context available to *each* parallel slot;.
    import logging

    fm = {"model_mib": 1000, "kv_per_token_mib": 0.5, "slot_mib": 0.0,
          "compute_mib": 100, "cache_type": "q8_0",
          "source": "llama-server", "shape": ""}
    model = make_model("nb", **{"derived": fm})
    with caplog.at_level(logging.WARNING):
        ctx = model.vram.calc_ctx(32768, fit_bin="unused", spare_mb=40000)
    assert ctx == _MIN_CTX_SIZE
    assert model.vram.over_budget_reason is not None
    assert "exhausted" in model.vram.over_budget_reason
    assert model.vram.vram_squeezed is False


def test_calc_ctx_squeezed_flag_when_design_does_not_fit(make_model):
    # Weights fit but the full 32768 design context does not: the solve
    # falls back smaller and flags the squeeze for the description note.
    # Validates: SPEC.md, 'Context calculation formula': Calculate max per-slot context size for given VRAM. The solved value is the context available to *each* parallel slot;.
    fm = {"model_mib": 32000, "kv_per_token_mib": 0.5, "slot_mib": 0.0,
          "compute_mib": 100, "cache_type": "q8_0",
          "source": "llama-server", "shape": ""}
    model = make_model("sq", **{"derived": fm})
    ctx = model.vram.calc_ctx(49152, fit_bin="unused", spare_mb=0)
    # available = 47104; remaining = 47104 - 32100 = 15004
    # design cost = 16384 > 15004 -> squeezed; not over budget.
    assert ctx < 32768
    assert model.vram.over_budget_reason is None
    assert model.vram.vram_squeezed is True


def test_calc_ctx_flags_reset_between_calls(make_model, fit_params_block):
    # Validates: SPEC.md, 'Context calculation formula': Calculate max per-slot context size for given VRAM. The solved value is the context available to *each* parallel slot;.
    """Flags are per-solve: a tight call's verdict must not leak into the
    next call's plan (groups may carry different spare)."""
    tight = make_model("r1", **{"derived": {
        "model_mib": 50000, "kv_per_token_mib": 0.5, "slot_mib": 0.0,
        "compute_mib": 100, "cache_type": "q8_0",
        "source": "llama-server", "shape": ""}})
    tight.vram.calc_ctx(49152, fit_bin="unused", spare_mb=0)
    assert tight.vram.over_budget_reason is not None

    # Same budget object, looser effective budget via the fitting block:
    # swap the measured quad for one that fits and the flags must clear.
    tight.vram.effective_static = lambda *a, **k: (
        1000, 0.5, 0.0, 100)
    ctx = tight.vram.calc_ctx(49152, fit_bin="unused", spare_mb=0)
    assert tight.vram.over_budget_reason is None
    assert tight.vram.vram_squeezed is False
    assert ctx >= _MIN_CTX_SIZE


def test_calc_ctx_memory_margin_shrinks_ctx(make_model):
    # Validates: SPEC.md, 'Context calculation formula': Calculate max per-slot context size for given VRAM. The solved value is the context available to *each* parallel slot;.
    fm = {"model_mib": 50000, "kv_per_token_mib": 0.5, "slot_mib": 0.0,
          "compute_mib": 1000, "cache_type": "q8_0", "source": "llama-server", "shape": ""}
    model = make_model("mm", **{"derived": fm})
    plain = model.vram.calc_ctx(65536, fit_bin="unused", memory_margin=0.0)
    margined = model.vram.calc_ctx(65536, fit_bin="unused", memory_margin=0.01)
    # available = 63520 (plain) vs 63520/1.01 = 62891.1 (margined)
    # plain: remaining 12520 -> 25040 -> 24576; margined: 11891 -> 23782 -> 16384
    assert plain == 24576
    assert margined == 16384


def test_calc_ctx_cpu_resident_returns_design(make_model):
    """
    Validates: SPEC.md, 'VRAM Allocation': a CPU-resident model costs no VRAM, so its
    context is its design maximum.
    """
    model = make_model("e", device="cpu", context_length=8192)
    ctx = model.vram.calc_ctx(1024, fit_bin="unused")
    assert ctx == 8192


# ── unestimable fallback (no estimate source works) ──────────────────────


def test_calc_ctx_unestimable_falls_back_to_design(make_model, caplog):
    # Validates: SPEC.md, 'Context calculation formula': Calculate max per-slot context size for given VRAM. The solved value is the context available to *each* parallel slot;.
    """No measurement source -> design ctx + unestimated flag, never raise."""
    import logging

    model = make_model("ue", context_length=32768)
    model.vram.effective_static = lambda *a, **k: None
    with caplog.at_level(logging.WARNING, logger="llama_packer.vram"):
        ctx = model.vram.calc_ctx(32768, fit_bin="unused")
    assert ctx == 32768
    assert model.vram.unestimated_reason is not None
    assert any("VRAM measurement failed" in r.message for r in caplog.records)
    # The warning is deduped: a repeat solve logs it once.
    model.vram.calc_ctx(32768, fit_bin="unused")
    assert sum("VRAM measurement failed" in r.message
               for r in caplog.records) == 1


def test_fit_params_undecodable_output_returns_none(make_model, monkeypatch,
                                                    caplog):
    # Validates: SPEC.md, 'Measurement via `llama-fit-params`': Run llama-fit-params and parse its buffer report. Returns a dict with ``model``/``context``/``compute`` (the report.
    """fit-params logs can carry non-UTF-8 bytes; strict decoding must not
    blow up inside subprocess (LFM2.5-VL regression: UnicodeDecodeError)."""
    import logging
    import subprocess

    model = make_model("uf")
    budget = model.vram

    def fake_run(cmd, **kw):
        raise UnicodeDecodeError("utf-8", b"\xc4", 0, 1,
                                 "invalid continuation byte")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with caplog.at_level(logging.WARNING, logger="llama_packer.vram"):
        assert budget.fit_params("fit-bin", fit_ctx=1024) is None
    assert any("fit-params failed" in r.message for r in caplog.records)


def test_calc_ctx_vllm_no_estimate_returns_design(make_model):
    # vLLM model with no estimator and no local safetensors: graceful fallback.
    # Validates: SPEC.md, 'Context calculation formula': Calculate max per-slot context size for given VRAM. The solved value is the context available to *each* parallel slot;.
    model = make_model("f", backend="vllm", hf_repo="org/model",
                       context_length=65536)
    ctx = model.vram.calc_ctx(32768, fit_bin="unused")
    assert ctx == 65536


# ── affine pair measurement ───────────────────────────────────────────────


def test_fit_params_static_derives_affine_pair(make_model, monkeypatch):
    # Validates: SPEC.md, 'Measured-Block Persistence': Machine-written ``derived:`` block (legacy names fall back). Holds llama-packer-calculated model requirements: the affine.
    model = make_model("ap", **{"derived": None})

    def fake_pair(fit_bin, cache_type, llama_args=""):
        return FitParams(model_mib=18905, kv_per_token_mib=0.0332,
                         slot_mib=448.25, compute_mib=4036,
                         source="fit-estimate", cache_type=cache_type)

    monkeypatch.setattr(model.vram, "_fit_params_serve", fake_pair)
    fp = model.vram.fit_params_static("/bin/fit", cache_type="q8_0")
    assert fp is not None
    assert fp.source == "fit-estimate"
    # v3 block persisted with the calibrated estimate
    block = model.measured_block()
    assert block["kv_per_token_mib"] == 0.0332
    assert block["slot_mib"] == 448.25
    assert block["source"] == "fit-estimate"
    assert "ctx_factor" not in block and "parallel" not in block


def test_fit_params_static_measure_failure_uses_transient_fit_params(
        make_model, monkeypatch):
    # Validates: SPEC.md, 'Measured-Block Persistence': Machine-written ``derived:`` block (legacy names fall back). Holds llama-packer-calculated model requirements: the affine.
    model = make_model("ap2", **{"derived": None})
    c_true, d_true = 0.0332, 150.0

    def fake_fit_params(*, fit_bin, fit_ctx, cache_type, parallel,
                        model_path=None, label=None, llama_args=""):
        # The KV pool line carries the per-slot cost (SWA ring / RS cells)
        # exactly as the real -lv 5 log does.
        return {"model": 18114, "context": 500, "compute": 505,
                "kv": c_true * fit_ctx + d_true * parallel,
                "rs": 0.0, "rs_cells": 0}

    monkeypatch.setattr(model.vram, "_fit_params_serve",
                        lambda *a, **k: None)
    monkeypatch.setattr(model.vram, "fit_params", fake_fit_params)
    fp = model.vram.fit_params_static("/bin/fit", cache_type="q8_0")
    assert fp is not None
    assert fp.source == "fit-params"
    assert fp.kv_per_token_mib == pytest.approx(c_true, abs=2e-3)
    assert fp.slot_mib == pytest.approx(d_true, abs=2e-2)
    # Transient: never persisted — the next run retries the calibrated path.
    assert model.measured_block() is None


def test_fit_params_static_vram_mib_prediction(make_model, fit_params_block):
    # Validates: docs/plans/estimation-functions-facts.md, 'Measurement shape and reuse': Get affine VRAM constants for this model at *cache_type*. Checks.
    """The persisted constants predict memory for any (context, slots)."""
    model = make_model("pred", **{"derived": fit_params_block})
    fp = model.vram.fit_params_static("unused")
    # model + compute + c*pool + p*D
    assert fp.vram_mib(ctx_per_slot=8192, parallel=4) == int(
        10000 + 1000 + 0.5 * (8192 * 4) + 0.0 * 4)


def test_fit_params_static_pair_failure_falls_back(make_model, monkeypatch):
    # Validates: docs/plans/estimation-functions-facts.md, 'Measurement shape and reuse': Get affine VRAM constants for this model at *cache_type*. Checks.
    model = make_model("pf", **{"derived": None})
    monkeypatch.setattr(model.vram, "fit_params", lambda *a, **k: None)
    assert model.vram.fit_params_static("/bin/fit") is None


# ── legacy blocks ─────────────────────────────────────────────────────────


def test_legacy_block_stale_and_rewritten(make_model, fit_params_block):
    # Validates: docs/plans/estimation-functions-facts.md, 'Measurement shape and reuse': Get affine VRAM constants for this model at *cache_type*. Checks.
    legacy = dict(fit_params_block)
    legacy.pop("kv_per_token_mib")
    legacy.pop("slot_mib")
    legacy["ctx_factor"] = 0.5
    legacy["parallel"] = 8
    model = make_model("lg", **{"fit-params": legacy})
    # The pre-affine block cannot be decomposed into (c, D): stale.
    assert model.vram.saved_for("q8_0") is None
    assert model.vram.fit_params_static("unused") is None  # binary unusable


def test_fit_params_static_cache_type_mismatch_remeasures(
        make_model, fit_params_block):
    # Validates: docs/plans/estimation-functions-facts.md, 'Measurement shape and reuse': Get affine VRAM constants for this model at *cache_type*. Checks.
    model = make_model("s", **{"derived": fit_params_block})
    assert model.vram.fit_params_static("unused", cache_type="q8_0") is not None
    # A different cache type has no readable block and no binary: None.
    assert model.vram.fit_params_static("unused", cache_type="f16") is None


# ── image token floor ─────────────────────────────────────────────────────

def _vision(make_model, tmp_path, stem, fit, **extra):
    """Vision model (mmproj companion) with a fit-params block."""
    (tmp_path / f"{stem}-mmproj.gguf").write_bytes(b"mm")
    block = {"file": f"{stem}-mmproj.gguf", "capabilities": ["image"]}
    tokens = {k: v for k, v in extra.items()
              if k in ("image_min_tokens", "image_max_tokens")}
    rest = {k: v for k, v in extra.items() if k not in tokens}
    block.update(tokens)
    fm = {"derived": fit, "mmproj": block}
    fm.update(rest)
    return make_model(stem, **fm)


def test_calc_ctx_image_floor_raises_ctx(make_model, tmp_path):
    # A declared image budget plus the output reserve must fit the solved
    # context: the ctx is raised from the rounded-down value to the floor
    # when affordable.
    # Validates: SPEC.md, 'Companion (mmproj / MTP) VRAM accounting': Frontmatter view for a serving variant. The companion-off variant is this model itself: overlay keys live.
    fit = {"model_mib": 25000, "kv_per_token_mib": 0.5, "slot_mib": 0.0,
           "compute_mib": 1000, "cache_type": "q8_0",
           "source": "llama-server", "shape": ""}
    model = _vision(make_model, tmp_path, "i", fit, image_max_tokens=7500)
    ctx = model.view_for(True).vram.calc_ctx(
        32768, fit_bin="unused", include_mmproj=True)
    # remaining = 30720 - 26000 - mmproj(0 + 150) = 4570; affordable 9140
    # tokens; rounded ctx 8192 -> raised to 7500 + 1024 reserve = 8524
    assert ctx == 8524


def test_calc_ctx_image_floor_unaffordable_warns(make_model, tmp_path, caplog):
    # Validates: SPEC.md, 'Companion (mmproj / MTP) VRAM accounting': Frontmatter view for a serving variant. The companion-off variant is this model itself: overlay keys live.
    import logging

    fit = {"model_mib": 25000, "kv_per_token_mib": 0.5, "slot_mib": 0.0,
           "compute_mib": 1000, "cache_type": "q8_0",
           "source": "llama-server", "shape": ""}
    model = _vision(make_model, tmp_path, "i", fit, image_max_tokens=16384)
    with caplog.at_level(logging.WARNING):
        ctx = model.view_for(True).vram.calc_ctx(
            32768, fit_bin="unused", include_mmproj=True)
    # floor 16384 > affordable 9140: ctx raised to what fits (9140), warning
    assert ctx == 9140
    assert any("large images may still not fit" in r.message
               for r in caplog.records)


def test_calc_ctx_image_floor_native_vision(make_model, tmp_path):
    """Native vision (no mmproj file) must still get the image floor.

    Qwen3-VL / Gemma3 have vision baked into the main weights. image_max_tokens
    is set, image tokens count against context, and the per-slot floor applies.

    Validates: docs/models/vision-context.md.
    """
    fit = {"model_mib": 25000, "kv_per_token_mib": 0.5, "slot_mib": 0.0,
           "compute_mib": 1000, "cache_type": "q8_0",
           "source": "llama-server", "shape": ""}
    model = make_model("i", **{"derived": fit, "image_max_tokens": 9440})
    ctx = model.vram.calc_ctx(32768, fit_bin="unused", include_mmproj=True)
    assert ctx >= 9440


def test_calc_ctx_text_variant_ignores_image_floor(make_model, tmp_path):
    # The -text variant serves no vision, so the floor must not apply.
    # Validates: SPEC.md, 'Context calculation formula': Calculate max per-slot context size for given VRAM. The solved value is the context available to *each* parallel slot;.
    fit = {"model_mib": 25000, "kv_per_token_mib": 0.5, "slot_mib": 0.0,
           "compute_mib": 1000, "cache_type": "q8_0",
           "source": "llama-server", "shape": ""}
    model = _vision(make_model, tmp_path, "i", fit, image_max_tokens=9440)
    ctx = model.vram.calc_ctx(32768, fit_bin="unused", include_mmproj=False)
    assert ctx == 8192


def _write_mmproj(path, *, patch=16, merge=2, projector="qwen3vl_merger"):
    """Write a minimal but valid GGUF carrying clip vision geometry."""
    kv = {"clip.vision.patch_size": patch, "clip.projector_type": projector}
    if merge:
        kv["clip.vision.spatial_merge_size"] = merge
    write_gguf(path, arch="clip", context_length=None, kv=kv)


def _vision_geom(make_model, tmp_path, stem, fit, *, patch=16, merge=2,
                 projector="qwen3vl_merger", **extra):
    """Vision model whose mmproj carries real clip geometry."""
    p = tmp_path / f"{stem}-mmproj.gguf"
    _write_mmproj(p, patch=patch, merge=merge, projector=projector)
    fm = {"derived": fit, "mmproj": {"file": p.name,
                                     "capabilities": ["image"]}}
    fm.update(extra)
    return make_model(stem, **fm)


_FIT = {"model_mib": 25000, "kv_per_token_mib": 0.5, "slot_mib": 0.0,
        "compute_mib": 1000, "cache_type": "q8_0",
        "source": "llama-server", "shape": ""}


def test_min_image_megapixels_converts_through_geometry(make_model, tmp_path):
    """2 megapixels at 1024 pixels/token is 1953 image tokens.

    Validates: docs/models/vision-context.md.
    """
    model = _vision_geom(make_model, tmp_path, "i", _FIT,
                         min_image_megapixels=2)
    assert model.vision_geometry.pixels_per_token == 1024
    assert model.min_image_tokens == 1953
    assert model.image_output_reserve_tokens == 1024


def test_min_image_megapixels_floor_includes_output_reserve(make_model, tmp_path):
    """The floor is image tokens plus room to answer, not the image alone.

    Validates: docs/models/vision-context.md.
    """
    model = _vision_geom(make_model, tmp_path, "i", _FIT,
                         min_image_megapixels=1, image_output_reserve=2048)
    assert model.min_image_tokens == 977
    assert model.view_for(True).vram._image_floor_tokens(True) == 977 + 2048


def test_min_image_megapixels_ignored_on_non_proportional(make_model, tmp_path,
                                                            caplog):
    """A projector that does not scale with pixels cannot honour the knob.

    Validates: docs/models/vision-context.md.
    """
    import logging
    model = _vision_geom(make_model, tmp_path, "i", _FIT, patch=14, merge=1,
                         projector="minicpmv4_6", min_image_megapixels=2)
    assert model.min_image_tokens is None
    with caplog.at_level(logging.WARNING):
        model.image_budget_warnings()
    assert any("does not scale with resolution" in r.message
               for r in caplog.records)


def test_min_image_megapixels_too_large_warns(make_model, tmp_path, caplog):
    """A budget above the declared cap is reported, then clamped.

    Validates: docs/models/vision-context.md.
    """
    import logging
    model = _vision_geom(make_model, tmp_path, "i", _FIT,
                         min_image_megapixels=20, image_max_tokens=16384)
    assert model.min_image_tokens == 19531          # 20 MP at 1024 px/token
    with caplog.at_level(logging.WARNING):
        model.image_budget_warnings()
    assert any("above image_max_tokens" in r.message for r in caplog.records)
    assert model.view_for(True).vram._image_floor_tokens(True) == 16384 + 1024


def test_min_image_megapixels_zero_allows_shedding(make_model, tmp_path, caplog):
    """0 means vision is not required; it may be shed, and that warns.

    Validates: docs/models/vision-context.md.
    """
    import logging
    model = _vision_geom(make_model, tmp_path, "i", _FIT,
                         min_image_megapixels=0)
    assert model.min_image_tokens is None
    with caplog.at_level(logging.WARNING):
        model.image_budget_warnings()
    assert any("vision is not required" in r.message for r in caplog.records)


# ── solve_matrix_ctx ──────────────────────────────────────────────────────


def test_solve_matrix_ctx_basic(make_model):
    # Validates: docs/plans/matrix-categories.md, 'Design sketch': Solve the shared per-slot chat context under the affine VRAM law. The VRAM budget for a co-resident group sharing one per-slot context X:.
    chat = make_model("chat", context_length=32768)
    ctx = solve_matrix_ctx(
        vram_total_mb=32768,
        spare_mb=0,
        chat_models=[(chat, 8000, 0.4, 0.0, 500, 1, 0)],
        embed_params=(500, 0.1, 0.0, 100),
        rerank_params=None,
        embed_ctx=8192,
        rerank_ctx=0,
    )
    # available = 30720; embed overhead = 500+100+0.1*8192 = 1419
    # remaining_for_chat = 29301; chat_budget = 29301-8000-500 = 20801
    # ctx = 20801/0.4 = 52002 -> round 49152 -> min(arch 32768) = 32768
    assert ctx == 32768


def test_solve_matrix_ctx_no_chat_models():
    # Validates: docs/plans/matrix-categories.md, 'Design sketch': Solve the shared per-slot chat context under the affine VRAM law. The VRAM budget for a co-resident group sharing one per-slot context X:.
    ctx = solve_matrix_ctx(
        vram_total_mb=32768, spare_mb=0, chat_models=[],
        embed_params=None, rerank_params=None,
    )
    assert ctx == _MIN_CTX_SIZE


def test_solve_matrix_ctx_exhausted_budget(make_model):
    # Validates: docs/plans/matrix-categories.md, 'Design sketch': Solve the shared per-slot chat context under the affine VRAM law. The VRAM budget for a co-resident group sharing one per-slot context X:.
    chat = make_model("chat", context_length=32768)
    ctx = solve_matrix_ctx(
        vram_total_mb=32768, spare_mb=0,
        chat_models=[(chat, 40000, 0.4, 0.0, 0, 1, 0)],
        embed_params=None, rerank_params=None,
    )
    # chat_budget = 30720 - 40000 < 0 -> skipped -> best_ctx 0 -> MIN_CTX
    assert ctx == _MIN_CTX_SIZE


def test_solve_matrix_ctx_rider_does_not_set_the_bar(make_model):
    # Validates: docs/plans/matrix-categories.md, 'Design sketch': Solve the shared per-slot chat context under the affine VRAM law. The VRAM budget for a co-resident group sharing one per-slot context X:.
    """An unestimable rider (zero-cost placeholder, kv_factor 0) rides the
    measurable models' solve — its native max must not inflate the shared
    chat context (which would suppress tools demotion fleet-wide and serve
    the unverified model at a context nothing bounded)."""
    measurable = make_model("m", context_length=131072)
    rider = make_model("rider", context_length=262144)
    ctx = solve_matrix_ctx(
        vram_total_mb=32768, spare_mb=0,
        chat_models=[
            (measurable, 8000, 0.2, 0.0, 500, 1, 0),
            (rider, 0, 0.0, 0.0, 0, 1, 0),
        ],
        embed_params=None, rerank_params=None,
    )
    # rider excluded: budget = 30720 - 8500 = 22220 -> 111100 -> round to
    # 106496 -> min(arch 131072); NOT the rider's native 262144.
    assert ctx == 106496


def test_solve_matrix_ctx_all_riders_fall_to_min(make_model):
    # Validates: docs/plans/matrix-categories.md, 'Design sketch': Solve the shared per-slot chat context under the affine VRAM law. The VRAM budget for a co-resident group sharing one per-slot context X:.
    rider = make_model("rider", context_length=262144)
    ctx = solve_matrix_ctx(
        vram_total_mb=32768, spare_mb=0,
        chat_models=[(rider, 0, 0.0, 0.0, 0, 1, 0)],
        embed_params=None, rerank_params=None,
    )
    # Nothing measurable bounds the solve -> the floor.
    assert ctx == _MIN_CTX_SIZE


# ── _measure_affine_trio: the (C, C/2) derivation ─────────────────────────


def test_measure_affine_trio_small_design_exact(make_model):
    # Validates: docs/plans/auto-parallel.md, 'Findings: the affine VRAM law (2026-09-07, SEAL record)': VRAM follows a validated affine law: model + kv x ctx x slots + slot + compute
    """The pool difference divides by the ACTUAL ctx delta: designs below
    8192 (the former r3 floor at 4096) previously mis-derived c by ~1.5x
    and persisted it as a trusted fit-estimate block."""
    model = make_model("t", context_length=6144)
    pools = {6144: 400.0, 3072: 220.0}  # 180 MiB linear over 3072 tokens

    def fake_fit_params(*, fit_bin="", fit_ctx=0, cache_type="q8_0",
                        parallel=1, llama_args="", **kw):
        return {"model": 1000, "kv": pools[fit_ctx], "rs": 0.0,
                "compute": 100}

    model.vram.fit_params = fake_fit_params
    fp = model.vram._measure_affine_trio(6144, "q8_0", "", "fit-estimate",
                                         "unused")
    assert fp is not None
    assert fp.kv_per_token_mib == pytest.approx(180 / 3072)
    assert fp.model_mib == 1000 and fp.compute_mib == 100
    assert fp.slot_mib == 0.0 and fp.source == "fit-estimate"


def test_measure_affine_trio_sub4k_design_still_measures(make_model):
    # Validates: docs/plans/auto-parallel.md, 'Findings: the affine VRAM law (2026-09-07, SEAL record)': VRAM follows a validated affine law, measured live over a (ctx, parallel) grid
    """Even a sub-4k design (an error or a testcase for text models) now
    measures exactly instead of returning None — no per-run retry tax."""
    model = make_model("t", context_length=4096)
    pools = {4096: 300.0, 2048: 200.0}  # 100 MiB over 2048 tokens

    def fake_fit_params(*, fit_bin="", fit_ctx=0, cache_type="q8_0",
                        parallel=1, llama_args="", **kw):
        return {"model": 900, "kv": pools[fit_ctx], "rs": 0.0,
                "compute": 50}

    model.vram.fit_params = fake_fit_params
    fp = model.vram._measure_affine_trio(4096, "q8_0", "", "fit-estimate",
                                         "unused")
    assert fp is not None
    assert fp.kv_per_token_mib == pytest.approx(100 / 2048)


def test_design_context_sub4k_text_warns(make_model, caplog):
    # Validates: docs/plans/estimation-functions-facts.md, 'Context ceiling vs design context': the design context is the model's ceiling, and a sub-4k one is warned about
    """A text model with a sub-4k context ceiling is almost certainly a
    mislabeled/corrupt GGUF — one note, not a crash."""
    import logging

    m = make_model("tiny", context_length=2048)
    with caplog.at_level(logging.WARNING):
        assert m.design_context == 2048
        assert m.design_context == 2048  # second read: no repeat
    notes = [r for r in caplog.records if "sub-4k" in r.message]
    assert len(notes) == 1
    # embed/rerank models are legitimate at small contexts: no note
    e = make_model("emb", role="embeddings", context_length=2048)
    with caplog.at_level(logging.WARNING):
        assert e.design_context == 2048
    assert not [r for r in caplog.records if "sub-4k" in r.message
                and "emb" in r.message]


def test_solve_matrix_ctx_fixed_overhead_mb(make_model):
    # Validates: docs/plans/matrix-categories.md, 'Design sketch': Solve the shared per-slot chat context under the affine VRAM law. The VRAM budget for a co-resident group sharing one per-slot context X:.
    chat = make_model("chat", context_length=32768)
    chat_models = [(chat, 8000, 0.2, 0.0, 500, 1, 0)]
    # available = 14336; budget = 5836 -> 29180 -> round 24576
    assert solve_matrix_ctx(
        vram_total_mb=16384, spare_mb=0, chat_models=chat_models,
        embed_params=None, rerank_params=None,
    ) == 24576
    # 2000 MB of co-loads: budget = 3836 -> 19180 -> round 16384
    assert solve_matrix_ctx(
        vram_total_mb=16384, spare_mb=0, chat_models=chat_models,
        embed_params=None, rerank_params=None, fixed_overhead_mb=2000,
    ) == 16384


# ── fixed-overhead backends ───────────────────────────────────────────────


def test_vram_mb_pin_is_authoritative(make_model):
    # Validates: SPEC.md, 'Companion (mmproj / MTP) VRAM accounting': Combined affine VRAM constants for main model plus its companions. Returns (model_mib, kv_per_token_mib, slot_mib, compute_mib) where.
    model = make_model("s", backend="whisper-server", vram_mb=1280,
                       context_length=8192)
    # Pinned: the pin IS the sizing — weights, zero KV terms, zero compute.
    assert model.vram.effective_static("unused") == (1280, 0.0, 0.0, 0)


def test_vram_mb_pin_invalid_ignored(make_model):
    # Validates: SPEC.md, 'Companion (mmproj / MTP) VRAM accounting': Combined affine VRAM constants for main model plus its companions. Returns (model_mib, kv_per_token_mib, slot_mib, compute_mib) where.
    model = make_model("s", backend="whisper-server", vram_mb="big",
                       context_length=8192)
    quad = model.vram.effective_static("unused")
    # Falls back to file size (dummy bytes -> 0 MB) + whisper compute buffer.
    assert quad[0] == 0 and quad[1] == 0.0 and quad[2] == 0.0
    assert quad[3] == 100  # _WHISPER_COMPUTE_MB


def test_whisper_fixed_compute_is_small(make_model):
    # Validates: SPEC.md, 'Companion (mmproj / MTP) VRAM accounting': Combined affine VRAM constants for main model plus its companions. Returns (model_mib, kv_per_token_mib, slot_mib, compute_mib) where.
    model = make_model("s", backend="whisper-server", context_length=8192)
    # Measured on Vulkan (nemo-speech): footprint ~= sum of files, so the
    # per-model buffer is small — not the 512 MB sd-server constant.
    quad = model.vram.effective_static("unused")
    assert quad[3] == 100


def test_sidecar_ceiling_binds_in_the_squeeze_branch(make_model):
    """The operator's pin must bind whether or not the design context fits.

    The ceiling was only applied in the design-fits branch, so the same model
    and sidecar produced two different answers depending on which branch the
    budget happened to select: 32768 when it fit, 98304 when VRAM forced the
    squeeze. SPEC spells out both clamps, and the caller cannot recover it --
    _bounded_ctx passes design_context, which prefers the GGUF header over
    the sidecar.

    Validates: docs/plans/requirement-trace.md #19: context_length is a ceiling in
    both branches, not only the one that fits.
    """
    fm = {"model_mib": 4000, "kv_per_token_mib": 0.10, "slot_mib": 0.0,
          "compute_mib": 500, "source": "llama-server", "cache_type": "q8_0",
          "shape": ""}
    model = make_model("pinned", context_length=32768, **{"derived": fm})

    roomy = model.vram.calc_ctx(24576, fit_bin="unused", parallel=1,
                               design_ctx=131072)
    assert roomy == 32768

    tight = model.vram.calc_ctx(16384, fit_bin="unused", parallel=1,
                               design_ctx=131072)
    assert tight <= 32768, f"squeeze ignored the sidecar pin: {tight}"
    assert model.vram.vram_squeezed is True


def test_sidecar_ceiling_is_a_min_not_a_fixed_cap(make_model):
    """The ceiling must only ever lower the ceiling, never impose one.

    Worth asserting explicitly because make_model seeds every sidecar with
    ``context_length: 32768`` -- so a loose pin has to be able to raise the
    served context above that, or "no pin" and "pinned at 32k" look
    identical in the fixtures.

    Validates: docs/plans/requirement-trace.md #19.
    """
    fm = {"model_mib": 4000, "kv_per_token_mib": 0.10, "slot_mib": 0.0,
          "compute_mib": 500, "source": "llama-server", "cache_type": "q8_0",
          "shape": ""}
    model = make_model("unpinned", context_length=131072, **{"derived": fm})
    ctx = model.vram.calc_ctx(24576, fit_bin="unused", parallel=1,
                              design_ctx=131072)
    assert ctx == 131072
