# Tuning the InCore Function

Making the bar itself shorter: overlapping transfer with compute, changing the algorithm,
respecting the hardware's granularity, and — when nothing else wins — not writing the
kernel in PyPTO at all.

> **Prerequisites:** the previous pages. If the gaps between bars are still the dominant
> cost, this page is premature.

<!-- doctest: setup -->
```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

NT, TR, TC = 8, 64, 128          # tiles in the loop, tile rows, tile cols
ROWS = NT * TR
CFG = RunConfig(platform="__PLATFORM__")

# Cycle through binary-exact values in a stable range on every host architecture.
indices = torch.arange(ROWS * TC, dtype=torch.int64)
A = (indices % 3 - 1).to(torch.float32).reshape(ROWS, TC)


def check(kernel):
    out = torch.zeros(ROWS, TC, dtype=torch.float32)
    kernel(A, out, config=CFG)
    torch.testing.assert_close(out, torch.exp(A), rtol=1e-4, atol=1e-4)
```

## Double buffering

**When it applies:** a loop inside the kernel alternates load → compute → store, and the
core stalls on the transfer because there is only one buffer to load into.

### With `pl.pipeline`

The compiler-managed form. It replicates the loop body `stage` times per outer iteration so
that iteration `i+1`'s load overlaps iteration `i`'s compute:

<!-- doctest: run -->
```python
@pl.jit
def single_buffer(a: pl.Tensor, out: pl.Out[pl.Tensor]):     # the baseline
    with pl.at(level=pl.Level.CORE_GROUP):
        for i in pl.range(NT):
            tile = pl.load(a, [i * TR, 0], [TR, TC])
            pl.store(pl.exp(tile), [i * TR, 0], out)
    return out


@pl.jit
def pipelined(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        for i in pl.pipeline(NT, stage=2):
            tile = pl.load(a, [i * TR, 0], [TR, TC])
            pl.store(pl.exp(tile), [i * TR, 0], out)
    return out


check(single_buffer)
check(pipelined)
```

The outer loop then advances in strides of `stage * step`, with a tail dispatch covering a
trip count that is not divisible by `stage`. Depths of 2–4 are the usual range.

**Cost:** `stage` copies of every buffer the body stages, live at once. This is the single
most common way to run out of on-chip memory, and the compiler tells you when it happens
rather than silently under-delivering:

```text
[perf_hint PH-MR-001] software pipelining requested depth 4 ... but only 2 of 4 buffers
fit (... B per stage, ... B free) — stages 2 apart share storage and serialize.
```

Read that as: *you asked for 4, you got 2*. The hint then tells you which lever applies —
shrink the per-stage tile to a stated byte budget, or reduce the depth to what fit.

**How to confirm:** the hint is gone from `report/perf_hints.log`, and the
[L0 trace](#the-l0-instruction-trace) shows the MTE2 lane overlapping the compute lanes
instead of alternating with them.

### With explicit slots

The hand-managed *placement*, for when you want the rotation to be exactly what you wrote —
typically because the natural staging does not match what `pl.pipeline` replicates. Note what
this is not: `pl.pipeline` restructures the loop into a schedule, whereas slots only remove
the same-buffer hazard that would prevent an overlap. The loop stays sequential, so confirm
any overlap in the [L0 trace](#the-l0-instruction-trace) rather than assuming the spelling
bought it.
`pl.MemRef("name", slots=N)` reserves `N` equally-sized slots of one allocation, and an
ordinary index expression picks one per iteration:

<!-- doctest: run -->
```python
@pl.jit
def explicit_slots(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        for i in pl.range(NT):
            tile: pl.Tile[[TR, TC], pl.FP32, pl.MemRef("ub", slots=2)[i % 2], pl.Mem.Vec] = pl.load(
                a, [i * TR, 0], [TR, TC], target_memory=pl.Mem.Vec
            )
            pl.store(pl.exp(tile), [i * TR, 0], out)
    return out


check(explicit_slots)
```

Use the inline `pl.MemRef("name", slots=2)` spelling rather than a Python variable holding
the declaration — `@pl.jit` re-parses generated source in a fresh module namespace, where
such a variable is not in scope.

**Cost, and it is planner-dependent:**

| Planner | Slot lowering | Two slots co-live in one iteration |
| ------- | ------------- | ---------------------------------- |
| `PYPTO` (default) | Baked addresses (`alloc_tile`) | Supported |
| `PTOAS` | One `alloc_multi_tile` region + a `multi_tile_get` per use | **Rejected at codegen** |

The PTOAS refusal is deliberate and worth understanding before you design around it: ptoas
guards only the first `multi_tile_get` of an iteration, so a second co-live slot is read
while the next iteration overwrites it. That was measured wrong on device before codegen
started refusing it. **One slot live per iteration** is the shape the region form exists
for, and it is the shape to write if you may switch planners.

## Seeing the on-chip budget

Both forms above spend the same scarce thing: on-chip buffer space. `pypto.tools.memory_map`
renders that allocation as HTML — address across, lifetime down, IR alongside — so you can
see what a deeper pipeline would have to fit into. Its input is a **pass dump**, not a run:

```python
from pypto.ir import PassDumpLevel
from pypto.runtime import RunConfig

compiled = kernel.compile(*args, config=RunConfig(dump_passes=PassDumpLevel.EXPLICIT))
```

```bash
DUMP=path/to/output_dir/passes_dump/NN_after_SomePass.py
python -m pypto.tools.memory_map "$DUMP" -o map.html
```

Read it for two things: tiles alive longer than they need to be, and the headroom that
decides whether another pipeline stage or a deeper cross-core ring will fit.

> Under `memory_planner=PTOAS` the compiler skips `AllocateMemoryAddr` entirely, so the pass
> dump carries no assigned offsets and this tool has nothing to draw. Compare end to end
> instead.

## Algorithmic changes

Some kernels are not transfer-bound or dispatch-bound; they are shaped wrong for the
machine. The canonical example is a matmul whose `M`/`N` are too small to fill the cube
while `K` is long — splitting the reduction gives the parallelism the output dimensions
could not:

```python
for ks in pl.parallel(SPLITS):
    ...   # each split reduces its slice of K; the partials are combined after
```

`examples/advanced/01_split_k.py` is the worked version, and
[the matmul tutorial](../tutorials/02-matmul.md) covers when it pays.

**Cost:** split-K accumulates in a different order, and with atomics the order is not even
fixed between runs. Expect last-place differences, and check the reduction order before you
call them a bug.

## The L0 instruction trace

**When it applies:** the kernel is the bottleneck and you want to know *which pipe*.

The compile-time hints say what the compiler suspected; the L2 swimlane says how tasks were
scheduled. Neither shows what the core did instruction by instruction. The
`incore-profiling` skill (from the `pypto-user` plugin) runs each generated kernel on the
Ascend op simulator and collects a cycle-accurate trace:

Install it (`claude plugin install pypto-user@pypto-skills`) and invoke the skill; it drives
`incore_profile.py` over a built case:

```text
/incore-profiling --build-dir build_output/<case> --target a2a3
```

The script is part of the plugin, not of this repository, so there is no in-tree path to
run directly.

The raw output is cluttered. The repo tool cleans it into a per-pipe, Perfetto-viewable
trace:

```bash
TRACE="<build-dir>/kernel_insight_all_funcs_<ts>/funcs/<kernel>/collect/out"
python -m pypto.tools.clean_sim_trace "$TRACE"/OPPROF_* -o trace-out
```

That writes `trace.clean.json` with the pipeline lanes in dataflow order —
**MTE2 → MTE1 → CUBE → VECTOR → FIXPIPE → MTE3** — plus `instr_metrics.json` with
per-instruction pipe, cycles, and vector utilization.

**How to read it:** the per-pipe cycle breakdown is the answer to "what is this kernel
actually doing". A kernel that is all MTE2 is transfer-bound (double buffer it); one that
is all VECTOR with low utilization is shaped wrong for the vector unit; `CUBE = 0` cycles
on a matmul kernel means the trace is degenerate, not that the matmul is free.

**Prerequisites are real:** a built case with `ptoas/` kernels, a TL-capable CANN, and the
`msopprof` worker. The skill preflights all three and fails early with a specific message.

## Hardware granularity

The compiler checks the most common one for you on every compile. `PH001`
(`TileInnermostDimGranularity`) inspects every `tile.load` / `tile.store` and flags any
whose innermost dimension is smaller than the backend's recommended transfer granularity —
on a2a3 that is the **512 B L2 cache line**:

```text
[perf_hint PH001] TileInnermostDimGranularity: tile.load has innermost dim = 64B
(tile fp32[16], target_memory=Mat); moves 1024B as 16 x 64B rows; recommended >= 512B
for backend a2a3 (L2 cache line = 512B). Consider increasing tile shape on the
innermost axis. at examples/intermediate/05_assemble.py:70:5
```

Two things make this hint usable rather than noise:

- **Rank by the `moves …` clause, not by count.** A `[1024, 64]` weight panel and a
  `[16, 64]` activation panel produce identical-looking hints and differ by 64× in traffic.
  The clause is what separates them.
- **The size of the penalty is real.** The `b_trans` matmul case — whose GM→Mat weight load
  moves 128 B rows against a 512 B recommendation — was measured at a **16–25%** penalty.

**How to fix:** widen the innermost axis of the tile, or transpose so that the contiguous
axis is the one being moved. If the tiling is deliberate and you have measured it, silence
the check with `disabled_diagnostics` rather than living with the noise.

## Escaping to a hand-written kernel

**When it applies:** you already have a tuned AscendC kernel, or a kernel where PyPTO's
codegen is not going to reach what hand-written code does.

`@pl.function(external_source=...)` backs an `AIC` / `AIV` function with a hand-written
C++ `.cpp`. The function's body is a bare `...` — signature only — and the orchestration
calls it exactly like any other kernel; the compiler skips PyPTO codegen for it and
compiles the referenced source instead.

```python
@pl.function(type=pl.FunctionType.AIV, external_source="kernels/my_kernel.cpp")
def my_kernel(x: pl.Tensor[[128, 128], pl.FP16], out: pl.Out[pl.Tensor[[128, 128], pl.FP16]]):
    ...
```

Relative paths resolve against the defining file's directory. See
[Functions § external kernels](../language/01-functions.md) for the full contract.

**Cost:** you leave the compiler behind for that function — no layout inference, no memory
planning, no perf hints, and no protection when the surrounding IR changes shape. The
signature is now a contract you maintain by hand.

## See also

- [Memory](05-memory.md) — where the buffers that double buffering needs come from.
- [Precision](../precision/index.md) — for when an algorithmic change moves the numbers.
