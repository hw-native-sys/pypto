# In-Core Trace

Cycle-accurate, per-pipe, one kernel at a time — what the core did instruction by
instruction.

## Concept

Three views sit at different grains, and this is the finest:

| View | Grain | Page |
| ---- | ----- | ---- |
| Perf hints | What the compiler suspected, statically | [Tools overview](index.md#the-cheap-checks-first) |
| Swimlane | How tasks were scheduled across the chip | [Runtime DFX](04-runtime-dfx.md) |
| **In-core trace** | Which pipe, which instruction, how many cycles | this page |

Reach for it when the swimlane shows one wide bar with no gaps around it — the schedule is
fine and the kernel itself is the cost.

It runs on the **Ascend op simulator** (the cycle-accurate camodel), not on a device and not
in a normal PyPTO run. Two steps: a skill collects, a repo tool cleans.

## Quickstart

Collect, with the `incore-profiling` skill from the `pypto-user` plugin:

```text
/incore-profiling --build-dir build_output/<case> --target a2a3
```

Then clean the raw dump into something readable:

```bash
TRACE="build_output/<case>/kernel_insight_all_funcs_<ts>/funcs/<kernel>/collect/out"
python -m pypto.tools.clean_sim_trace "$TRACE"/OPPROF_* -o trace-out
```

Open `trace-out/trace.clean.json` in the [Perfetto UI](https://ui.perfetto.dev) or
`chrome://tracing`.

The collection script ships with the plugin, not this repository, so there is no in-tree
path to run directly. Register the marketplace before installing from it, or the plugin
name does not resolve:

```bash
claude plugin marketplace add hw-native-sys/pypto-skills
claude plugin install pypto-user@pypto-skills
```

Its prerequisites are real — a built case with `ptoas/` kernels, a TL-capable CANN, and the
`msopprof` worker — and it preflights all three, failing early with a specific message.

## Mechanics

### Why the cleaning step exists

The simulator writes two artifacts per kernel run, and the official one is lossy:

| File | Holds |
| ---- | ----- |
| `trace.json` | The simulator's own Perfetto export — trace events only |
| `visualize_data.bin` | The full MindStudio Insight container: trace events **plus** per-instruction metrics, source mapping, and other blocks |

`clean_sim_trace` reads the binary directly, so it recovers the per-instruction metrics the
official export drops. It also de-clutters: opening `trace.json` raw is unreadable because
`SET_FLAG` / `WAIT_FLAG` slices and scalar address arithmetic bury the actual pipeline.

### CLI

```bash
python -m pypto.tools.clean_sim_trace <path> [-o OUTPUT_DIR] [--keep-scalar] [--raw-metrics] [--no-copy-raw]
```

`<path>` is a `visualize_data.bin` or an `OPPROF_*` directory (the tool finds
`simulator/visualize_data.bin` inside).

| Output | Content |
| ------ | ------- |
| `trace.clean.json` | Rebuilt Chrome Trace Event JSON |
| `instr_metrics.json` | Per-core instruction records — `address`, `pipe`, `cycles`, `vector_utilization_percentage`, … |
| `raw_simulator/` | Copy of the raw dump, so an `-o` target folder is self-contained (`--no-copy-raw` to skip) |

| Flag | Effect |
| ---- | ------ |
| `--keep-scalar` | Keep the `SCALAR` setup lane (dropped by default) |
| `--raw-metrics` | Dump the `API_INSTR` block verbatim instead of reshaping it |
| `--no-copy-raw` | Do not copy the raw binary trace into the output directory |

### What the cleaning does

1. **Lane selection** — keeps the pipeline lanes, drops `CACHEMISS` / `FLOWCTRL` / `ALL`, and `SCALAR` unless asked.
2. **Event filtering** — keeps complete instruction events; drops `SET_FLAG` / `WAIT_FLAG` / `BAR` slices.
3. **Lane ordering** — emits metadata so lanes render in dataflow order: **MTE2 → MTE1 → CUBE → VECTOR → FIXPIPE → MTE3**.
4. **Sub-lane packing** — software-pipelined instructions on one pipe are often several in flight and only partially overlap. Chrome-trace events on one lane must be disjoint or nested, so overlapping instructions are split into `MTE1`, `MTE1#1`, … — one row per concurrently-live instruction. **Without this the visible pipeline depth collapses to about 2**, which reads as "no overlap" when there is plenty.
5. **Sync as arrows** — each `SET_FLAG` → `WAIT_FLAG` pair becomes one flow arrow, re-anchored to the real producing and consuming instructions.
6. **Timestamps** — kept verbatim, so the cleaned trace lines up with the raw `trace.json`.

### Reading it

The per-pipe cycle breakdown is the answer to "what is this kernel actually doing":

| Shape | Means | Go to |
| ----- | ----- | ----- |
| Almost all **MTE2** | Transfer-bound | [Double buffering](../performance/04-incore.md#double-buffering) |
| Almost all **VECTOR**, low utilization | Shaped wrong for the vector unit | [Tuning the InCore function](../performance/04-incore.md) |
| **CUBE = 0** on a matmul kernel | The trace is degenerate, not the matmul free | See the caveat below |

## Edge Cases

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| **Near-empty trace, `CUBE`/`VECTOR` ~0 cycles** | A data-dependent kernel — trip count or work-table size read from an input tensor — profiled under the skill's zeroed inputs | A synthetic-input artifact, not a fast kernel; wire full-size real intermediates per the skill's caveats |
| **Pipeline looks 2 deep on a 4-stage pipeline** | Reading the raw `trace.json` instead of the cleaned one | Use `trace.clean.json`; sub-lane packing is what preserves the depth |
| **No `instr_metrics.json`** | Read from `trace.json`, which does not carry `API_INSTR` | Point the tool at `visualize_data.bin` / the `OPPROF_*` directory |
| **`corrupt block at offset …`** | Truncated or partially written dump | Re-collect; the run likely died mid-write |
| **Skill fails at preflight** | Missing `ptoas/` kernels, a non-TL CANN, or no `msopprof` | The message names which; build the case first |

## See Also

- [Tuning the InCore function](../performance/04-incore.md) — what to change once you know which pipe.
- [Runtime DFX](04-runtime-dfx.md) — the level above; PMU answers a coarser version of the same question without the simulator.
- [Replaying a build](05-replay.md) — hand-edit the kernel, then re-profile it.
- [Simulator Trace Cleaning](../../dev/04-simulator-trace-cleaning.md) — the `visualize_data.bin` block format and the full rebuild rules.
