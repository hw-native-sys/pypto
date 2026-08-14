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
    enable_l2_swimlane=True,   # per-task timing
    enable_dep_gen=True,       # the task DAG the timing is joined against
    save_kernels=True,         # keep the output directory
)
```

Both land under the run's output directory:

```text
<work_dir>/dfx_outputs/
  l2_swimlane_records.json     per-task start / end / dispatch / finish
  deps.json                    task dependency edges
  merged_swimlane_*.json       onboard only — the joined trace
```

`deps.json` is the reason for the second flag: successor edges are deliberately **not**
recorded in the swimlane record itself, to keep the device hot path clean. The join happens
afterwards, on the host.

### Two things that will mislead you if you do not know them

**Onboard, the flag runs your workload twice.** The converter needs a task graph that only
`deps.json` carries, and collection perturbs timing — so PyPTO takes a first dep_gen pass
for the graph, then a clean pass with dep_gen off for the timing. **Never read wall-clock
from a swimlane-enabled onboard run**; use a separate plain run for that number.

**On the simulator you get the records but not the merged trace.** `*sim` platforms stay
single-pass and emit only `l2_swimlane_records.json` — the simulator does not yet ship the
task metadata the converter needs. Use the simulator to see the *shape* of the schedule,
and an onboard run when the timing itself is the question.

## Opening it

### The IDE plugin

[PyPTO Toolkit](https://github.com/hw-native-sys/pypto-tools) is a VS Code extension that
renders these files directly. Right-click the swimlane JSON in the explorer and choose
**`PyPTO Toolkit: 打开文件`**.

What you get, depending on the collection level:

| View | Shows |
| ---- | ----- |
| Worker View | Per-core task bars — the main picture |
| Scheduler View | The scheduler's own timeline |
| AICPU Scheduler / AICPU Orchestrator | Per-iteration scheduler phase breakdown |

The parts worth knowing on day one:

- **Click a task** for its detail panel. With a sibling `deps.json` in the same directory,
  the tasks it depends on are drawn in too; the *任务连线层级* setting bounds how many
  levels of dependency are drawn at once.
- **Selecting an SPMD task highlights all of its blocks**, since they share one
  `func_name` and `task_id`. On a dependency path only the first is drawn, so the view is
  not buried under edges.
- **Search** by `func_name` or `task_id` in the search box.
- **性能统计** (top right) opens a report; clicking an entry jumps to that task on the
  timeline.
- **Observation lines** — click on the second axis to drop a timestamped ruler, and
  ALT+drag to measure a span by hand.
- The same extension opens a `passes_dump` folder as an **IR trace**, filtered to the
  passes that actually changed something. That is a compile-time view, not a timing one,
  but it is the other half of "what did the compiler do to my kernel".

> **A naming difference to expect.** The plugin's documentation names the file
> `chip_swimlane_records*.json`; the runtime pinned in this repository writes
> `l2_swimlane_records.json`. If the right-click action does not offer your file, that
> difference is the first thing to check.

### Perfetto

The runtime also converts the records into a Chrome Trace Event JSON that loads in
[ui.perfetto.dev](https://ui.perfetto.dev):

```bash
python -m simpler_setup.tools.swimlane_converter <records>.json \
    --deps-json <deps>.json -o out.json
```

## Reading it

Each task carries four timestamps, and the gaps between them mean different things:

```text
dispatch ──────► start ──────► end ──────► finish
   │               │             │            │
   AICPU wrote     core began    kernel       AICPU observed
   the descriptor  the kernel    done         completion

[dispatch, start]  = pickup latency (~0.8 µs per switch)
[start, end]       = the kernel — the only span page 04 can shrink
```

**Read the gaps, not the bars.** A chip whose bars are narrow and whose gaps are wide is
not a kernel problem; it is a granularity or dispatch problem, and pages
[01](01-task-granularity.md) and [02](02-runtime-overhead.md) are where it goes.

### Putting a number on the gaps

When you want the gaps quantified rather than eyeballed, the runtime ships an analysis
that answers one specific question — *when is time lost because an idle core had ready
work the scheduler had not placed yet?*

```bash
python -m simpler_setup.tools.sched_overhead_analysis \
    --l2-swimlane-records-json <records>.json --deps-json <deps>.json
```

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

## Next

With the picture in front of you, work down the chapter in order — granularity, then
dispatch overhead, then the graph, and only then the kernel itself.
