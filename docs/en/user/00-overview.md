# PyPTO Overview

This page is the single entry point to all user documentation. It explains what
PyPTO is, how it is designed, and where to go for each task.

## What is PyPTO?

PyPTO is a Python-based kernel programming framework for Ascend NPUs. You write
compute kernels in Python with the `pypto.language` module (imported as `pl`),
and PyPTO compiles them into optimized device code:

```python
import pypto.language as pl
from pypto import ir
```

You describe *what* to compute — at the whole-tensor level or the tile level —
and express *how* to optimize it with declarative DSL hints (pipelining,
splitting, scopes). The compiler lowers your program through a pass pipeline and
emits code for two device roles: the AI cores that run the compute, and the AI
CPU that schedules it.

If you are brand new, start with **[Getting Started](01-getting_started.md)** and
come back here for the map.

## Design Philosophy

- **Python DSL → PTO → device.** You write kernels in the `pl` DSL. PyPTO builds
  an immutable IR, runs a pass pipeline that progressively lowers tensor-level IR
  to tile-level IR, and generates device code. No hand-written assembly or
  schedules.
- **Two abstraction levels.** *Tensor level* operates on whole tensors (e.g.
  `pl.matmul`, `pl.add`) — you still tile the computation yourself, but the
  compiler automates the data movement and the L0 tiling *inside* matmul. *Tile
  level* gives you explicit control over load / compute / store on tiles when you
  need it. Both coexist in the same program; see the
  [Language Guide](02-language_guide.md).
- **You control the partitioning.** How the computation is carved into InCore
  kernels — the boundary of each InCore scope — is stated *explicitly* by you,
  not chosen by the compiler. You decide what compute lands in each on-core
  function; the compiler lowers and schedules within and across the boundaries
  you draw.
- **Declarative optimization.** Optimizations are expressed as DSL constructs
  (`pl.pipeline`, `pl.split`, `pl.spmd`, scopes) rather than manual scheduling.
  You state intent; the compiler realizes it. The
  [Performance Handbook](handbook/perf/00-workflow.md) covers each knob.

## Execution Model

A compiled PyPTO program runs across three coordinated roles on the device:

| Role | Runs | Produced from |
| ---- | ---- | ------------- |
| **Host** | launches the program, owns host buffers | your Python driver / runtime |
| **AI CPU** | task scheduling & dispatch (*orchestration*) | Orchestration functions → C++ using the PTO2 runtime API |
| **AI Core** | the actual compute (*InCore*) | InCore functions → `.pto` → AICore binaries |

```text
        ┌──────────────────────────────────────────────────────────┐
        │ Host — your Python driver                                 │
        │   launches the program · owns host buffers                │
        └───────────────────────────┬──────────────────────────────┘
                                     │ launch
        ┌───────────────────────────▼──────────────────────────────┐
        │ AI CPU — Orchestration  (task scheduling & dispatch)      │
        │   Orchestration fns → C++ (PTO2 runtime API)              │
        │   builds the task graph · resolves deps · dispatches      │
        └───────────────────────────┬──────────────────────────────┘
                                     │ dispatch InCore kernels
        ┌───────────────────────────▼──────────────────────────────┐
        │ AI Core — InCore  (compute)                               │
        │   InCore fns → .pto → AICore binaries                     │
        │   ┌───────────────────┐   tpush   ┌───────────────────┐   │
        │   │ AIC (cube)        │ ────────▶ │ AIV (vector)      │   │
        │   │ matmul-heavy      │ ◀──────── │ elementwise /     │   │
        │   │                   │   tpop    │ reduction         │   │
        │   └───────────────────┘           └───────────────────┘   │
        └───────────────────────────┬──────────────────────────────┘
                                     │ load / store
        ┌───────────────────────────▼──────────────────────────────┐
        │ DDR — global memory  (tensor parameters)                  │
        └──────────────────────────────────────────────────────────┘
```

- **Orchestration vs InCore.** Orchestration functions describe the task graph
  and dispatch; InCore functions describe per-core compute. The compiler
  separates them automatically as it lowers your program.
- **Cube / Vector cores.** An AI Core has a cube unit (**AIC**, matmul-heavy)
  and a vector unit (**AIV**, elementwise/reduction). Some kernels are *mixed*
  and split work across the two, moving data cross-core.
- **On-chip parallelism vs multi-card distribution.** `pl.spmd` / `pl.cluster`
  dispatch many blocks across the cores of a **single chip** — an on-chip
  performance tool ([Perf › Split & Parallel](handbook/perf/02-split-parallel.md)).
  The `pld.*` family (collectives, remote load/store) spans **multiple cards**
  ([Distributed Guide](distributed/00-guide.md)).

## Memory Hierarchy

PyPTO exposes a layered memory model: off-chip **global memory (DDR)** backs your
tensor parameters, and tiles are staged through progressively smaller/faster
on-chip buffers for compute — the unified **Vec** buffer for vector work, and the
matmul path **Mat** (L1) → **Left**/**Right** (L0A/L0B) → **Acc** (L0C). At tensor
level the compiler inserts the data movement for you; at tile level you move data
explicitly (`pl.load` / `pl.move` / `pl.store`) and steer placement with `pl.Mem`
hints. See
[Perf › Memory Placement](handbook/perf/04-memory-placement.md) and the
[Language Guide](02-language_guide.md#memory-and-data-movement).

## Compilation Pipeline at a Glance

```text
Python DSL  →  IR (immutable tree)  →  Pass pipeline (lower tensor → tile)  →  CodeGen
  @pl.program       @pl.function          inline · SSA · tiling · memory ·         ├─ InCore  → .pto → AICore
  @pl.function                            cross-core split · scheduling            └─ Orch.   → C++ (PTO2 runtime) → AI CPU
```

You trigger the whole pipeline with a single call:

```python
output_dir = ir.compile(MyProgram, backend_type=BackendType.Ascend910B)
```

`ir.compile(..., dump_passes=True)` (the default) writes an IR snapshot after
every pass under `output_dir/passes_dump/` — the backbone of the
[precision-localization](handbook/precision/00-workflow.md) workflow. The
individual passes are documented for compiler developers under
[`dev/passes/`](../dev/passes/00-pass_manager.md); as a user you rarely need
them, but the pass names appear in dumps and diagnostics.

## Feature Map

A one-line capability index; each links to the relevant chapter.

| Capability | Entry point |
| ---------- | ----------- |
| Write your first kernel | [Getting Started](01-getting_started.md) |
| Language / type system | [Language Guide](02-language_guide.md) |
| Look up an operation | [Operation Reference](03-operation_reference.md) · [API Reference](api-reference/index.md) |
| On-chip multi-block parallelism (`pl.spmd`/`pl.cluster`) | [Perf › Split & Parallel](handbook/perf/02-split-parallel.md) |
| Mixed kernel — cube + vector split (`pl.split`, `pl.split_aiv`) | [Perf › Split & Parallel](handbook/perf/02-split-parallel.md) |
| Task graphs & dependency control | [Perf › Dependency & Dispatch](handbook/perf/03-dependency-dispatch.md) |
| Multi-card distribution (`pld.*` collectives) | [Distributed Guide](distributed/00-guide.md) |
| Debug a wrong result | [Precision Localization](handbook/precision/00-workflow.md) |
| Debug a slow kernel | [Performance Tuning](handbook/perf/00-workflow.md) |
| Diagnostics / logging / replay | [DFX Features](handbook/dfx/00-flag-matrix.md) |

## Documentation Map

Where each folder lives and when to read it.

| Folder / file | What's inside | Read it when |
| ------------- | ------------- | ------------ |
| `01-getting_started.md` | Hello World → compile → run on device | Your first kernel |
| `02-language_guide.md` | Types, control flow, memory — conceptual tutorial | Learning the DSL |
| `03-operation_reference.md` | `pl.*` operation lookup tables (→ superseded by `api-reference/`) | Looking up an op |
| `handbook/precision/` | Precision-localization workflow (torch golden, pass-IR bisection, selective dump) | Your result is wrong |
| `handbook/perf/` | Performance-tuning syntax (loop/pipeline, split/parallel, dependency/dispatch, memory placement, measuring) | Your kernel is slow |
| `handbook/dfx/` | DFX diagnostics (flag matrix, HTML renderers, replay, logging) | Inspecting scheduling/deps or capturing logs |
| `distributed/` | Multi-card SPMD authoring (`DistributedTensor`, collectives, remote load/store, signals) | Writing distributed kernels |
| `api-reference/` | Auto-generated `pl.*` reference (every `__all__` symbol) | Looking up an API signature |
| `troubleshooting.md` | Error message → likely cause → chapter | You hit an error |
| `glossary.md` | Term definitions (tile, scope, orchestration, AIV/AIC, TaskId, …) | A term is unfamiliar |

## What's Next

- New to PyPTO? Start with **[Getting Started](01-getting_started.md)**.
- Hit a problem? Jump to the **[Feature Handbook index](handbook/00-index.md)** —
  it maps symptoms to the exact tool.
