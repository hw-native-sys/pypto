# Performance Tuning: Measuring Impact

Measure before and after every tuning change. Each tool below answers a
different question — end-to-end wall-clock, compile time, per-kernel cycles,
per-task scheduling, or what a change produced in codegen.

## Benchmark — end-to-end device wall-clock

`pypto.runtime.benchmark` is the headline "how fast is it, really" number:
steady-state on-device wall-clock over many launches.

```python
from pypto.runtime import benchmark

stats = benchmark(compiled, args, rounds=100, warmup=3)
print(stats.device_us_mean(), stats.device_us_median(), stats.device_us_stdev())
```

- **Steady state.** Registers `compiled` **once**, then loops the bound handle —
  each launch re-pays only argument coercion + dispatch, not register/load.
- **What it measures.** `device_wall_us` — on-NPU time between the orchestrator's
  `orch_start` / `orch_end`, unaffected by host-side arg building; `host_wall_us`
  is also recorded. `warmup` launches (default 3) are discarded; `rounds`
  (default 100) are measured.
- **Returns** `BenchmarkStats` — per-round samples (`per_round("device")`) plus
  aggregates (`device_us_mean/median/min/max/stdev`) and `print_mean_tree()`.
- **L2 & L3.** Works for `CompiledProgram` (L2) and `DistributedCompiledProgram`
  (L3, per-round maxima across ranks).

Use benchmark for the headline number; use the tools below to explain *why* it is
that number.

## Compile-Time Profiling

Records **compilation** wall-clock at each pipeline stage — frontend parse,
passes (per-pass), codegen (per-kernel + orchestration), and on-device
execution. Use it to answer *"why is compiling slow"* and *"which pass dominates
compile time"*, not to measure a kernel's on-chip speed (for that, use in-core
profiling and the swimlane below).

**Enable it** (any one):

```python
ir.compile(program, profiling=True)     # writes output_dir/report/pipeline_profile.{txt,json}
```

```bash
PYPTO_COMPILE_PROFILING=1 python3 my_program.py
```

**What each entry point captures:**

| Entry point | Stages recorded |
| ----------- | --------------- |
| `ir.compile(profiling=True)` | `passes` (per-pass) + `codegen` (sub-stages) only |
| `runtime.run(config=RunConfig(compile_profiling=True))` | full hierarchy — also `parse`, `golden_write`, `device_execution` |

**Read the summary** (`pipeline_profile.txt`) — a tree with per-stage seconds and
percent of total:

```text
PyPTO Compile Profile
Total: 2.847s
  parse                    0.023s ( 0.8%)
  passes                   1.204s (42.3%)
    UnrollLoops            0.012s
    ConvertToSSA           0.034s
    AllocateMemoryAddr     0.156s
  codegen                  0.418s (14.7%)
    kernel_codegen:my_kernel   0.312s
    orchestration_codegen      0.106s
  device_execution         1.202s (42.2%)   # only via runtime.run()
```

`pipeline_profile.json` carries the same tree (`total_seconds` + nested `stages`)
for programmatic diffing across runs. Note `device_execution` is a single coarse
total — it does **not** break down per kernel or per task; that resolution comes
from the swimlane and in-core profiling.

## In-Core Profiling (msprof op-simulator)

*TODO:*

- Cycle-accurate per-kernel traces via the Ascend msprof op-simulator.
- Reference the `incore-profiling` skill workflow.

## On-Device Swimlane

The swimlane is a **per-task timeline** captured on real hardware — the fastest
way to see whether tasks overlap or stall across cores. Enable it with
`enable_l2_swimlane`, then read the lanes for gaps and imbalance.

Full reference — enabling, outputs, and how to read it — is in
**[DFX › Swimlane](../dfx/01-swimlane.md)**.

## Codegen Comparison Across Branches

*TODO:*

- Diff `.pto` / pass dumps between `origin/main` and your branch.
- Reference the `compare-codegen` skill workflow.

## See Also

- Developer reference: [`dev/01-compile-profiling.md`](../../../dev/01-compile-profiling.md), [`dev/04-simulator-trace-cleaning.md`](../../../dev/04-simulator-trace-cleaning.md)
