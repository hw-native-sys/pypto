# Tuning the Schedule

Finding out what the runtime did with your graph, and changing it on evidence.

> **Prerequisites:** [Shaping the task graph](04-task-graph.md).

## What you are building

A loop you can re-run on your own kernel: **look at the graph → look at the concurrency →
look at the resource high-water marks → change one thing → measure again.**

Steps 1-4 are `RunConfig` flags that each write a file — nothing there requires
instrumenting your kernel. Steps 5 and 6 are changes to how the work is shaped, not
observations.

## The four observation points

| Question | Flag | Output |
| -------- | ---- | ------ |
| What shape is the graph? | `enable_dep_gen=True` | `<work_dir>/dfx_outputs/deps.json` |
| Did tasks actually overlap? | `enable_chip_swimlane=True` | `<work_dir>/dfx_outputs/chip_swimlane_records.json` |
| Are runtime rings near full? | `enable_scope_stats=True` | `<work_dir>/dfx_outputs/scope_stats/scope_stats.jsonl` |
| Which pipe is the bottleneck? | `enable_pmu=2` | `<work_dir>/dfx_outputs/pmu.csv` |

They compose — one run can capture several. Take them in order; each answers a question the
next one assumes.

## Step 1: is the graph the one you meant?

```python
from pypto.runtime import RunConfig

kernel(a, b, out, config=RunConfig(platform="a2a3sim", enable_dep_gen=True))
```

Then render it:

```bash
python -m simpler_setup.tools.deps_viewer <work_dir>/dfx_outputs/deps.json --format html
```

The viewer defaults to text output, so pass `--format html` for the graph view.

This is the first thing to check, because every later measurement is downstream of it. Two
readings worth having:

- **A chain where you expected a fan-out** — an inferred edge that is not a real dependency
  is serializing you. Go back to [step 3 of the previous page](04-task-graph.md).
- **A fan-out where you expected a chain** — nothing is ordering tasks that must be
  ordered. That is a latent race, and it will not always show up as a wrong answer.

## Step 2: did they actually overlap?

A parallel graph does not guarantee parallel execution. The swimlane shows per-task timing:

```python
kernel(a, b, out, config=RunConfig(platform="a2a3sim", enable_chip_swimlane=True))
```

> **Simulator caveat:** on `*sim` platforms this is single-pass and emits only
> `chip_swimlane_records.json`. The merged `merged_swimlane_*.json` view is intentionally
> skipped, because the simulator does not yet ship the task metadata the converter needs.
> On an onboard platform the same flag runs the workload **twice** — a dep_gen pass to
> capture the graph, then a clean timing pass — since collection perturbs timing.

That second point matters for benchmarking: do not read wall-clock from a swimlane-enabled
onboard run.

## Step 3: are the rings the limit?

The runtime holds in-flight tasks in rings. If one is saturating, adding parallelism to the
graph buys nothing:

```python
kernel(a, b, out, config=RunConfig(platform="a2a3sim", enable_scope_stats=True))
```

```bash
python runtime/tools/scope_stats_plot.py <work_dir>/dfx_outputs/scope_stats/scope_stats.jsonl
```

It reports per-scope peaks for three rings — **task_window** (in-flight task slots),
**heap** (output storage), **tensormap**. A peak at capacity is the signal to raise that
ring; peaks well under capacity mean the ring is not your problem.

## Step 4: change one thing

Each ring has a matching override. They are per-invocation, so you can sweep without
recompiling:

| Knob | Units | Constraint |
| ---- | ----- | ---------- |
| `ring_task_window` | In-flight task slots | Power of two, `>= 4` |
| `ring_heap` | **Bytes** | Power of two, `>= 1024` |
| `ring_dep_pool` | Dependency-edge capacity | `[4, INT32_MAX]` |
| `aicpu_thread_num` | AICPU threads | Defaults to the compile-time `RUNTIME_CONFIG` |

```python
cfg = RunConfig(platform="a2a3sim", ring_task_window=64, ring_heap=1 << 20)
```

Each accepts a scalar (broadcast to all four scope-depth rings) or a list of exactly 4 ints
sizing rings 0..3 independently, where a `0` entry leaves that ring at its default. `None`
— the default — leaves the field unset so the runtime falls back to its compile-time
default. The old process-wide `PTO2_RING_*` env vars are retired and no longer read.

`ring_heap` being in bytes while `ring_task_window` is in slots is the easy mistake: a
`ring_heap=64` is not 64 buffers, it is 64 bytes, and it is rejected for being under 1024.

## Step 5: task granularity

The knobs above resize the machinery. The bigger lever is usually how much work one task
carries:

- **Too fine** — per-task scheduling overhead dominates, and the task_window saturates on
  bookkeeping rather than work.
- **Too coarse** — fewer tasks than cores, so the graph cannot fill the machine no matter
  how the rings are sized.

`sync_start=True` on an SPMD dispatch requires all blocks to launch atomically. It buys a
well-defined start across blocks, and costs the ability to start any block early — so a
`sync_start` task cannot itself be pre-staged block by block, though flagging it
`allow_early_resolve=True` still lets its *consumers* pre-stage.

## Step 6: stop paying setup twice

Worker setup is per-worker, not per-program. Several programs registered against one worker
share it, which removes a full setup from every run after the first — see
`examples/runtime/multi_program_kv_cache.py` for the shape of that (a prefill and a decode
program sharing one KV cache and one worker).

## Debugging a wrong answer, not a slow one

Two flags in the same family, for correctness rather than speed:

- **`enable_dump_args=1`** dumps only the tensors you marked with `pl.dump_tag(t)` (or
  `pl.submit(..., dumps=[...])`), into `<work_dir>/dfx_outputs/args_dump/`. Inspect with
  `python -m simpler_setup.tools.dump_viewer`.
- **`enable_dump_args=2`** dumps every task's inputs and outputs.

> **Fatal pitfall:** a full dump on a large workload can saturate the host-side collector
> (~42 MB/s drain) and get the AICPU killed by a STARS op-execute timeout. Prefer level 1
> plus `pl.dump_tag(t)` on the specific tensors you are chasing.

## The loop

```text
deps.json      →  is the graph right?          →  fix edges  (04-task-graph)
swimlane       →  did it overlap?              →  fix granularity
scope_stats    →  is a ring saturating?        →  raise that ring
pmu.csv        →  which pipe is the limit?     →  fix the kernel (02, 03)
```

Re-measure after each change, and change one thing at a time — the four observations are
not independent, and two simultaneous edits usually leave you unable to attribute the
difference.

## See Also

- [Shaping the task graph](04-task-graph.md) — where the edges come from.
- [Mixed kernels](03-mixed-kernel.md) — the fix when one unit is the bottleneck.
- [Tasks and Ordering](../tasks/index.md) — the reference for the ordering interfaces.
