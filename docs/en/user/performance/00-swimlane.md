# Reading the Swimlane

Per-task timing for the whole chip: when each task was dispatched, when its core actually
started it, and when it ended.

> **Prerequisites:** a kernel that runs. Everything here is measurement, not change.

## Why this comes first

Every later page in this chapter asks a question the swimlane answers directly:

| Page asks | The swimlane shows |
| --------- | ------------------ |
| Are the tasks too small? | Bar widths against the gaps between them |
| Is dispatch the bottleneck? | The `dispatch → start` gap, and idle cores while work is ready |
| Did the graph serialize? | Bars in a staircase where they should have stacked |
| Is the kernel itself slow? | One wide bar with no gaps around it — go to [InCore](04-incore.md) |

Tuning without it is guessing. The compile-time `report/perf_hints.log` tells you what the
compiler suspected; the swimlane tells you what actually happened.

## Capturing it

Two flags, and you want both — the timing and the graph it belongs to:

```python
from pypto.runtime import ChipWorker, RunConfig

cfg = RunConfig(
    platform="a2a3",
    enable_chip_swimlane=4,      # per-task timing, full collection
    enable_dep_gen=True,       # the task DAG the timing is joined against
    save_kernels=True,         # keep the output directory
)
```

Both land under the run's output directory:

```text
<work_dir>/dfx_outputs/
  chip_swimlane_records.json   per-task timing plus scheduler/orchestrator phases (level 4)
  deps.json                    task dependency edges
  merged_swimlane_*.json       onboard only — the joined trace
```

`deps.json` is the reason for the second flag: successor edges are deliberately **not**
recorded in the swimlane record itself, to keep the device hot path clean. The join happens
afterwards, on the host.

### Collection is levelled

The runtime collects at one of four levels, each adding to the one below:

| Level | Adds |
| ----- | ---- |
| 1 `AICORE_TIMING` | AICore per-task start / end |
| 2 `AICPU_TIMING` | + AICPU-stamped dispatch / finish |
| 3 `SCHED_PHASES` | + scheduler main-loop phase records |
| 4 `ORCH_PHASES` | + orchestrator phase records |

Each level is a real guard in the collectors, not a verbosity setting: at level 1 the
dispatch and finish timestamps are **never stamped**, so no post-processing can recover
them.

`RunConfig.enable_chip_swimlane` **is** that level — pass any of `0`-`4` to request it:

```python
cfg = RunConfig(platform="a2a3", enable_chip_swimlane=3,  # sched phases and below
                enable_dep_gen=True, save_kernels=True)
```

`True` is accepted for source compatibility and means level `4` (full) — the same thing the
bare `--enable-chip-swimlane` flag requests; `False` means `0`. Higher levels collect more
and therefore perturb timing more, so drop to the lowest level that answers your question.

### Two things that will mislead you if you do not know them

**Onboard, the flag runs your workload twice.** The converter needs a task graph that only
`deps.json` carries, and collection perturbs timing — so PyPTO takes a first dep_gen pass
for the graph, then a clean pass with dep_gen off for the timing. **Never read wall-clock
from a swimlane-enabled onboard run**; use a separate plain run for that number.

**On the simulator you get the records but not the merged trace.** `*sim` platforms stay
single-pass and emit only `chip_swimlane_records.json` — the simulator does not yet ship the
task metadata the converter needs. Use the simulator to see the *shape* of the schedule,
and an onboard run when the timing itself is the question.

## Opening it

### The IDE plugin

[PyPTO Toolkit](https://github.com/hw-native-sys/pypto-tools) is a VS Code extension that
renders these files directly. Right-click the swimlane JSON in the explorer and choose the
extension's **open-file** action.

> The extension's menus and panels are localised in Chinese; the labels quoted below in
> English are the actions, not the literal strings you will see.

What you get, depending on the collection level:

| View | Shows | Needs level |
| ---- | ----- | ----------- |
| Worker View | Per-core task bars — the main picture | 1 |
| Scheduler View | The scheduler's own timeline | 3 |
| AICPU Scheduler / AICPU Orchestrator | Per-iteration scheduler phase breakdown | 3 / 4 |

The parts worth knowing on day one:

- **Click a task** for its detail panel. With a sibling `deps.json` in the same directory,
  the tasks it depends on are drawn in too; a dependency-depth setting bounds how many
  levels are drawn at once.
- **Selecting an SPMD task highlights all of its blocks**, since they share one
  `func_name` and `task_id`. On a dependency path only the first is drawn, so the view is
  not buried under edges.
- **Search** by `func_name` or `task_id` in the search box.
- **The performance report** (top right) lists the findings; clicking an entry jumps to
  that task on the timeline.
- **Observation lines** — click on the second axis to drop a timestamped ruler, and
  ALT+drag to measure a span by hand.
- The same extension opens a `passes_dump` folder as an **IR trace**, filtered to the
  passes that actually changed something. That is a compile-time view, not a timing one,
  but it is the other half of "what did the compiler do to my kernel".

### Perfetto

The runtime also converts the records into a Chrome Trace Event JSON that loads in
[ui.perfetto.dev](https://ui.perfetto.dev):

```bash
RECORDS="outputs/<run>/chip_swimlane_records.json"
DEPS_JSON="outputs/<run>/deps.json"
python -m simpler_setup.tools.swimlane_converter "$RECORDS" \
    --deps-json "$DEPS_JSON" -o out.json
```

## Reading it

Each task carries four timestamps, and the gaps between them mean different things:

```text
dispatch ──────► start ──────► end ──────► finish
   │               │             │            │
   AICPU wrote     core began    kernel       AICPU observed
   the descriptor  the kernel    done         completion
   └── level 2 ──┘ └──── level 1 ────┘ └── level 2 ──┘

[dispatch, start]  = pickup latency (~0.8 µs per switch)
[start, end]       = the kernel — the only span page 04 can shrink
```

Level 4 — what `RunConfig` gives you — includes all four timestamps plus scheduler and
orchestrator phases. The `[start, end]` interval remains the task's AICore execution time;
the `[dispatch, start]` split requires at least level 2 and is therefore present as well.

**Read the gaps, not the bars.** A chip whose bars are narrow and whose gaps are wide is
not a kernel problem; it is a granularity or dispatch problem, and pages
[01](01-task-granularity.md) and [02](02-runtime-overhead.md) are where it goes.

### Putting a number on the gaps

When you want the gaps quantified rather than eyeballed, the runtime ships an analysis
that answers one specific question — *when is time lost because an idle core had ready
work the scheduler had not placed yet?*

```bash
# $RECORDS and $DEPS_JSON as set above
python -m simpler_setup.tools.sched_overhead_analysis \
    --chip-swimlane-records-json "$RECORDS" --deps-json "$DEPS_JSON"
```

This one needs a **level ≥ 3** capture for its scheduler-loop parts —
`RunConfig(enable_chip_swimlane=3)` or the default full level `4`.

It reports per-engine and system-wide overhead as a share of the makespan, the pickup-cost
distribution, the AICPU scheduler-loop budget, and a critical-path attribution splitting
compute from scheduler-injected microseconds. The same numbers can be overlaid on the
timeline as counter tracks with `swimlane_converter --overhead`.

Two definitions in that report are worth internalising, because they separate a real
problem from a fake one:

- An idle core **with no ready work** is not overhead. Its idleness is mandated by the
  dependency graph — that shows up as low parallelism, and belongs to
  [Managing dependencies](03-dependencies.md).
- An idle core **with ready, undispatched work** is overhead. That is the scheduler not
  keeping up, and belongs to [Runtime overhead](02-runtime-overhead.md).

### Where the makespan actually went

`sched_overhead_analysis` answers one question. The critical-path analysis answers the
broader one — *what is the dependency-limited floor, and which tasks spent the rest?*

```bash
RUN_DIR="outputs/<run>"        # the tree holding the capture
python -m simpler_setup.tools.critical_path "$RUN_DIR"
```

It discovers every directory holding `chip_swimlane_records.json`, `deps.json` and
`name_map*.json` as siblings, and writes one report per rank beside its records file. Point
it at a whole run tree rather than a single rank.

Two paths come out of it, and the difference between them is the finding:

| Path | What it is |
| ---- | ---------- |
| **Static CPM** | the longest duration-weighted chain — the latency floor with unlimited cores |
| **Observed** | the as-executed backward walk from the last task to finish |

Each task's compute plus the stall in front of it tiles the observed makespan exactly, and
the stall is attributed as `data-wait` (an upstream producer is late), `core-wait` (the
assigned core was busy — resource serialization) or `front-gap` (launch delay before any
task ran).

That gives the verdict the rest of this chapter branches on:

| Reading | Verdict | Where it goes |
| ------- | ------- | ------------- |
| Static CPM near the makespan | dependency-bound — more cores cannot help | [03](03-dependencies.md), and granularity in [01](01-task-granularity.md) |
| Static CPM well below, `core-wait` dominant | resource serialization | [01](01-task-granularity.md) |
| Static CPM well below, `front-gap` large | launch and dispatch cost | [02](02-runtime-overhead.md), [06](06-host.md) |
| Compute high, stall low | genuinely compute-bound | [04](04-incore.md) |

> **Check the tiling line before quoting anything.** Each rank prints
> `tiling check: compute+stall = ... vs makespan ...`, and it must read `exact`. A non-zero
> difference means the walk did not tile the makespan and the per-task attribution is
> unsound. Two more that quietly invalidate a report: families named `unknown` or `cid<N>`
> mean the name map did not resolve, so family-level conclusions mean nothing; and with
> multiple rounds the capture covers the **first** round, so the makespan includes warm-up.
>
> **One capture is one sample.** Two captures of the same unchanged workload can differ by
> several points of stall share. Never compare two configurations from one capture each.

The `critical-path-analysis` skill in the `pypto-user` plugin
(`claude plugin install pypto-user@pypto-skills`) drives this end to end — artifact
resolution, the validation list above, and the interpretation. The tool itself is in the
runtime and needs no device, no build and no checkout: it is pure post-processing over a
capture someone else took.

## Next

With the picture in front of you, work down the chapter in order — granularity, then
dispatch overhead, then the graph, and only then the kernel itself.
