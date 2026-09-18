# Runtime DFX

Five independent collection flags. What each one records, where it lands, and what opens it.

## Concept

Everything on this page costs a run. The compiler has already told you what it *decided*
(see [the cheap checks](index.md#the-cheap-checks-first)); these flags record what the
hardware *did*.

They are five fully independent toggles on `RunConfig`, combinable in any subset. Enabling
any of them forces `save_kernels=True`, so the output directory survives the run and the
artifacts are still there afterwards.

Every artifact lands under one directory:

```text
<work_dir>/dfx_outputs/
├── chip_swimlane_records.json     enable_chip_swimlane
├── merged_swimlane_*.json         (onboard only — the joined trace)
├── deps.json                      enable_dep_gen
├── args_dump/                     enable_dump_args
├── pmu.csv                        enable_pmu
└── scope_stats/scope_stats.jsonl  enable_scope_stats
```

## Quickstart

The two flags you want first, and you want them together — the timing, and the task graph
it belongs to:

```python
from pypto.runtime import RunConfig

cfg = RunConfig(
    platform="a2a3",
    enable_chip_swimlane=4,     # per-task timing, full collection
    enable_dep_gen=True,        # the task DAG the timing is joined against
    save_kernels=True,          # forced anyway; be explicit about wanting the directory
)
out = kernel(a, b, torch.zeros(...), config=cfg)
```

From pytest, the same two:

```bash
pytest tests/st/runtime/ --platform a2a3 --enable-chip-swimlane --enable-dep-gen
```

## Mechanics

### The flag matrix

| `RunConfig` field | pytest flag | Artifact under `dfx_outputs/` | Opens with |
| ----------------- | ----------- | ----------------------------- | ---------- |
| `enable_chip_swimlane: int` | `--enable-chip-swimlane` (= `4`) / `--chip-swimlane-level N` | `chip_swimlane_records.json` | PyPTO Toolkit; `merged_swimlane_*.json` is produced automatically onboard |
| `enable_dep_gen: bool` | `--enable-dep-gen` | `deps.json` | `python -m simpler_setup.tools.deps_viewer` |
| `enable_dump_args: int` | `--dump-args [LEVEL]` (bare = `1`) | `args_dump/` | `python -m simpler_setup.tools.dump_viewer` |
| `enable_pmu: int` | `--enable-pmu [N]` (bare = `2`) | `pmu.csv` | Any CSV reader |
| `enable_scope_stats: bool` | `--enable-scope-stats` | `scope_stats/scope_stats.jsonl` | `python runtime/simpler_setup/tools/scope_stats_plot.py` |

None of the renderers run automatically. Graphviz layout on a multi-thousand-node graph can
take minutes, and running it on the dispatch hot path has gotten whole job trees SIGKILLed
by outer schedulers. The runner prints the render command instead; you run it when you want
the picture.

### Swimlane: collection is levelled

`enable_chip_swimlane` is a **level**, not a toggle:

| Level | Adds | Unlocks |
| ----- | ---- | ------- |
| `0` / `False` | — | off |
| `1` | AICore per-task start / end, plus the task record buffer | Per-task lanes |
| `2` | + AICPU-stamped dispatch / finish | The `[dispatch, start]` pickup gap |
| `3` | + scheduler main-loop phase records | `sched_overhead_analysis`, the Toolkit's Scheduler View |
| `4` / `True` | + orchestrator phase records | The Toolkit's AICPU Orchestrator view |

**Each level is a real guard in the collectors, not a verbosity setting.** At level 1 the
dispatch and finish timestamps are never stamped, so no post-processing recovers them —
"collect cheap now, analyse deeper later" does not work. Conversely, higher levels perturb
timing more. Pick the lowest level that answers your question.

`True` means level `4`, matching the bare `--enable-chip-swimlane` flag. An out-of-range
level raises `ValueError` from `RunConfig`.

For what to *look for* once it is open, see [Reading the swimlane](../performance/00-swimlane.md).

### Swimlane: onboard runs your workload twice

The converter joins per-task timing against a task graph that **only `deps.json` carries** —
the device hot path no longer records per-task fanout. But dep_gen collection perturbs the
very timing the swimlane measures. So the two captures come from separate runs, and PyPTO
does that split for you:

1. **Graph pass** — dep_gen only, producing `deps.json`. On L2 this runs in a separate
   subprocess, because the runtime does not reliably reclaim the SVM host-register mappings
   the collectors allocate and a second DFX run in the same process hits the registration
   cap. Best-effort: if it fails, a warning is logged and the timing pass still runs, with
   lanes degraded to anonymous `task(rXtY)`.
2. **Timing pass** — swimlane (plus any other timing-sensitive DFX), dep_gen forced off,
   producing the `chip_swimlane_records.json` whose timing is reported.

Consequences worth internalising:

- **Never read wall-clock from a swimlane-enabled onboard run.** Use a separate plain run,
  or [`benchmark`](../performance/index.md#before-you-tune-anything).
- **Both passes execute the program**, and mutable arguments are not restored between them.
- Adding `--enable-dep-gen` explicitly changes nothing about the passes — the graph pass
  already produced `deps.json`. It only makes the run print the `deps_viewer` hint.
- **Simulator platforms (`*sim`) stay single-pass** and emit only
  `chip_swimlane_records.json`; the simulator does not yet ship the task metadata the
  converter needs. Use the simulator for the *shape* of the schedule, onboard when the
  timing itself is the question.

### Args dump: mark the tensors, do not take everything

`enable_dump_args` is a level: `0` off, `1` partial, `2` full.

**Prefer partial.** Level `2` writes every binding of every task; on a large workload that
saturates the host-side dump collector (~42 MB/s drain) and the AICPU gets killed by the
STARS op-execute timeout — a 1 GB KV-cache binding fills the queue far faster than it
drains.

Partial dump captures only what you mark, through two surfaces that mirror the two `deps=`
surfaces exactly:

```python
# Declarative marker — every SUBSEQUENT dispatch consuming that exact value dumps it.
pl.dump_tag(q)
pl.dump_tag(out)
out = self.qk_pv(q, k_cache, out)        # q and out dumped; k_cache filtered out

# Explicit kwarg — this one launch only.
with pl.manual_scope():
    out, tid = pl.submit(self.qk_pv, q, k_cache, out, deps=[prev], dumps=[q, out])
```

The same `dumps=[...]` kwarg also works on a dispatch scope — `pl.at(...)`, `pl.spmd(...)`,
`pl.cluster(...)` or `pl.graph(...)` — and marks the task that scope launches.

Marks are tracked by **Var identity, never by name**, so they ride SSA, inlining, and
codegen — and so a rebound or transformed value is *not* covered:

| You wrote | Covered? |
| --------- | -------- |
| `pl.dump_tag(q)` then `self.k(q, ...)` | Yes |
| `pl.dump_tag(q)` then `q = self.foo(q)` then `self.k(q, ...)` | **No** — the rebound result is a new value; re-tag it |
| `pl.dump_tag(q)` then `q2 = pl.reshape(q)` then `self.k(q2, ...)` | **No** — tag the value the kernel actually receives |
| `dumps=` on a plain `self.kernel(...)` call | **No** — raises `ParserTypeError`; use `pl.dump_tag` or `pl.submit(..., dumps=[...])` |
| `pl.dump_tag` inside an InCore / AIC / AIV body | **No** — raises `ParserSyntaxError`; put it in the enclosing Orchestration or Inline function |

Marks are inert when dump is off and irrelevant under full dump. The full limitation table
is in [Runtime DFX Flags](../../dev/03-runtime-dfx.md#limitations).

### PMU

```python
RunConfig(enable_pmu=2)     # 0 = off; 2 = PIPE_UTILIZATION; 4 = MEMORY
```

One event type per run, into `pmu.csv`. Answers "what were the pipes doing" at a coarser
grain than the [in-core trace](06-incore-trace.md), and without the simulator.

### Scope stats

Per-scope heap / task_window / tensormap ring-fill **peaks**, one JSONL record per scope:

```bash
python runtime/simpler_setup/tools/scope_stats_plot.py \
    <work_dir>/dfx_outputs/scope_stats/scope_stats.jsonl
```

This is the measurement behind ring sizing — read it before touching `ring_task_window`,
`ring_heap`, or `ring_dep_pool`. See [Memory](../performance/05-memory.md).

### Kernel names in the viewers

By default the swimlane and dependency viewers label tasks by numeric id. PyPTO synthesises
`dfx_outputs/name_map_<case>.json` from `kernel_config.py` whenever swimlane or dep_gen is
enabled, and both tools pick it up automatically — so tasks show real kernel names with no
manual step.

### Distributed runs

L3 namespaces the prefix per dispatch, since one chip can receive several dispatches in one
host orchestration and they would otherwise overwrite each other:

```text
<work_dir>/dfx_outputs/rank0/d0/     # rank 0, its 0th dispatch
<work_dir>/dfx_outputs/rank0/d1/
<work_dir>/dfx_outputs/rank1/d0/
```

Each leaf holds the flat artifacts above, so everything on this page applies unchanged
inside one dispatch directory.

## Edge Cases

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| **No `dfx_outputs/` at all** | No flag was enabled, or the run failed before dispatch | Enabling any flag forces `save_kernels`; check the run reached the device |
| **Wall-clock is much worse than expected** | Swimlane onboard ran the workload twice | Measure in a separate plain run |
| **No `merged_swimlane_*.json`** | Simulator platform | Expected — `*sim` emits records only |
| **Lanes are anonymous `task(rXtY)`** | The graph pass failed, so there is no `deps.json` | Diagnose it from the logged warning — adding `--enable-dep-gen` reruns nothing, since that pass already runs on its own |
| **AICPU killed mid-run, STARS timeout** | `enable_dump_args=2` on a large workload | Drop to level `1` and `pl.dump_tag` the tensors you need |
| **A tagged tensor never appears in the dump** | Tag is tracked by Var identity and the value was rebound or transformed | Tag the value the kernel actually receives |
| **`deps_viewer --format html` hangs** | Graphviz `dot` on a huge graph | `--engine sfdp` (O(N log N), scales to 10k+ nodes) |
| **`DeprecationWarning` about `enable_l2_swimlane`** | Former spelling | Rename to `enable_chip_swimlane`; same values, same semantics |

## See Also

- [Reading the swimlane](../performance/00-swimlane.md) — what to look for once it is open.
- [Replaying a build](05-replay.md) — every flag here applies unchanged on the replay path.
- [In-core trace](06-incore-trace.md) — one level down, inside a single kernel.
- [Memory](../performance/05-memory.md) — the rings `scope_stats` measures.
- [Precision](../precision/00-workflow.md) — where `enable_dump_args` fits in the narrowing procedure.
- [Runtime DFX Flags](../../dev/03-runtime-dfx.md) — the `CallConfig` contract, the full dump-tag limitation table, and the deprecated aliases.
