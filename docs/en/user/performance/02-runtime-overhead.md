# Runtime Overhead

Five ways to spend less time per task without changing what the tasks compute.

> **Prerequisites:** [Task granularity](01-task-granularity.md).

## Where this differs from the previous page

Granularity changes *how many* tasks there are. This page changes *what each one costs*:
one dispatch instead of two, one dispatch instead of `N`, no dispatch at all when a runtime
value says the work is unnecessary, a dispatch that starts earlier, or no dispatch at all
where a barrier will do.

| Technique | Removes |
| --------- | ------- |
| [Mixed kernel](#build-a-mixed-kernel) | One of two dispatches, plus the GM round-trip between them |
| [SPMD](#use-spmd) | `N − 1` dispatches for `N` blocks of the same work |
| [Dispatch predicate](#skip-the-task-entirely) | The whole dispatch, when a runtime value says the work is not needed |
| [`allow_early_resolve`](#let-consumers-pre-stage) | The pickup latency on the critical path |
| [In-kernel `syncall`](#synchronize-inside-the-kernel) | An AICPU round-trip per synchronization point |

<!-- doctest: setup -->
```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

BLOCKS, TILE_ROWS, COLS = 4, 64, 128
ROWS = BLOCKS * TILE_ROWS
CFG = RunConfig(platform="__PLATFORM__")

torch.manual_seed(0)
A = torch.randn(ROWS, COLS, dtype=torch.float32)
B = torch.randn(ROWS, COLS, dtype=torch.float32)

def fresh(rows=ROWS):
    return torch.zeros(rows, COLS, dtype=torch.float32)
```

## Build a mixed kernel

**When it applies:** a cube operation feeds a vector operation. Left alone these are two
tasks: the cube task finishes, writes GM, and the vector task is dispatched to read it
back.

**How:** one `pl.at` scope carrying both, with a split mode telling the compiler how to
divide the vector half across the two AIVs that share a cube:

```python
with pl.at(
    level=pl.Level.CORE_GROUP,
    optimizations=[pl.split(pl.SplitMode.UP_DOWN), pl.cross_core_slot(slot_num=2)],
):
    acc = pl.matmul(a, b, out_dtype=pl.FP32)   # cube (AIC)
    out[:] = pl.add(acc, bias)                 # vector (AIV)
```

`examples/advanced/03_mixed_kernel.py` runs this in three modes, and
[the tutorial](../tutorials/03-mixed-kernel.md) walks through it.

**Cost:** `pl.split` halves only the **vector** sub-region; the cube side stays full-size,
so the vector buffers shrink but the cube buffers do not. And the cross-core ring that
carries the intermediate between engines is real memory: it defaults to **8 slots**, which
for a large intermediate is far more than the vector budget can spare. `slot_num=` is
usually not optional — the compiler tells you at compile time when the default does not
fit.

**How to confirm:** the swimlane. Two abutting bars should become one bar with the cube and
vector spans **overlapping**. If they merely became one bar with the same total width, the
engines are still serialized inside the kernel, and the split mode is the thing to revisit.

## Use SPMD

**When it applies:** the same kernel over `N` independent blocks of data. Dispatching them
individually pays `N` dispatches to do one thing.

**How:** one dispatch that the runtime fans out. Each block reads its own index.

```python
# Loop form — the index is bound for you, the body is outlined automatically
for i in pl.spmd(num_blocks):
    tile = pl.load(x, [i * TILE, 0], [TILE, COLS])
    pl.store(pl.exp(tile), [i * TILE, 0], out)

# With a captured TaskId, so later tasks can depend on the whole grid
with pl.spmd(num_blocks, deps=[prev_tid]) as tid:
    ...
```

Size the grid from the device rather than a literal when you can:

| Spelling | For |
| -------- | --- |
| `pl.system.available_cluster_count()` | Mixed or cube-only kernels |
| `pl.system.available_aiv_count()` | Vector-only kernels |

These are the only spellings that stay at full occupancy across devices — which matters on
its own, and is a hard requirement for the barrier below.

`examples/models/09_paged_attention_spmd.py` is the same idea at model scale: each block
takes a subset of the batch through a stride loop, so the batch dimension is parallelised
across hardware blocks by one dispatch.

Both forms below compute the same thing; the first pays `BLOCKS` dispatches, the second one:

<!-- doctest: run -->
```python
@pl.jit
def per_block_tasks(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    for i in pl.unroll(BLOCKS):                     # BLOCKS dispatches
        with pl.at(level=pl.Level.CORE_GROUP):
            ta = pl.load(a, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
            tb = pl.load(b, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
            pl.store(pl.add(ta, tb), [i * TILE_ROWS, 0], c)
    return c

@pl.jit
def spmd_blocks(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    for i in pl.spmd(BLOCKS):                       # one dispatch, BLOCKS blocks
        ta = pl.load(a, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
        tb = pl.load(b, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
        pl.store(pl.add(ta, tb), [i * TILE_ROWS, 0], c)
    return c

for kernel in (per_block_tasks, spmd_blocks):
    c = fresh()
    kernel(A, B, c, config=CFG)
    torch.testing.assert_close(c, A + B, rtol=1e-4, atol=1e-4)
```

**Cost:** every block runs the same program. Divergent work needs a different structure,
and blocks that finish at different times leave their cores idle until the whole grid
retires.

**How to confirm:** `deps.json` collapses `N` nodes into one; the swimlane shows one task
occupying many core lanes at once. The plugin highlights all blocks of an SPMD task
together when you click one.

## Skip the task entirely

**When it applies:** work whose need is only known at run time — an MoE expert whose
router sent it no rows, a refinement step a computed error count makes unnecessary. The
count exists only once an earlier task has run, so the decision cannot be made while
building the graph.

**How:** `predicate=` on `pl.spmd`, `pl.submit` or `pl.spmd_submit`. In a `@pl.jit`
function `pl.spmd` is the form that carries it — `pl.at` has no predicate.

<!-- doctest: run -->
```python
@pl.jit
def gated(a: pl.Tensor, gate: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.manual_scope():
        with pl.spmd(BLOCKS) as base_tid:                    # always dispatched
            i = pl.tile.get_block_idx()
            t = pl.load(a, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
            pl.store(pl.mul(t, 2.0), [i * TILE_ROWS, 0], out)
        # Dispatched only when the scheduler finds gate[0, 0] > 0.
        with pl.spmd(BLOCKS, deps=[base_tid], predicate=(gate[0, 0] > 0)) as _bump:
            i = pl.tile.get_block_idx()
            t = pl.load(out, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
            pl.store(pl.add(t, 1.0), [i * TILE_ROWS, 0], out)
    return out

for gate_value, expected in ((0, A * 2.0), (1, A * 2.0 + 1.0)):
    gate = torch.full((1, 1), gate_value, dtype=torch.int32)
    out = fresh()
    gated(A, gate, out, config=CFG)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
```

The scheduler evaluates the comparison at the **dispatch point**, after the task's
dependencies are satisfied, so it reads a current value without the orchestration ever
waiting on the tensor. When it is false the task is routed to the same queue as a dummy and
retired inline: it never reaches a core, while its fanin and fanout still settle so
consumers unlock exactly as they would have.

**Cost:** the task still exists. You save the dispatch and the core time, not the
bookkeeping — the slot, the edges and the retirement all still happen, so predicating a
task that almost always runs buys nothing and costs a little. Three constraints bite:

- Only `tensor[indices] OP int-literal`, one comparison. No arithmetic, no `and`/`or`. Reduce
  anything richer to a single gate value in a prior kernel.
- The operand must be a **signed** 8/16/32/64-bit integer tensor. The runtime sign-extends
  what it reads, so a `UINT32` count of 3'000'000'000 compares as negative and inverts the
  decision silently.
- The operand's producer must be among this task's `deps=`, or the dispatch-point read can
  see stale data. Above, `gate` is a function parameter with no producer at all, which is
  the trivially safe case.

**How to confirm:** the swimlane shows the predicated task retiring without occupying a core
lane, and the graph keeps its shape — the node is still there, it just did not run.

## Let consumers pre-stage

**When it applies:** a critical path built from many short tasks, where each consumer sits
waiting through its own pickup latency after its producer ends.

**How:** flag the **producer**.

<!-- doctest: run -->
```python
@pl.jit
def early_resolve(a: pl.Tensor, b: pl.Tensor, scratch: pl.Out[pl.Tensor], out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP, allow_early_resolve=True):
        s = pl.add(pl.load(a, [0, 0], [TILE_ROWS, COLS]), pl.load(b, [0, 0], [TILE_ROWS, COLS]))
        pl.store(s, [0, 0], scratch)
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.exp(pl.load(scratch, [0, 0], [TILE_ROWS, COLS])), [0, 0], out)
    return scratch, out

scratch, out = fresh(TILE_ROWS), fresh(TILE_ROWS)
early_resolve(A[:TILE_ROWS], B[:TILE_ROWS], scratch, out, config=CFG)
torch.testing.assert_close(out, torch.exp(A[:TILE_ROWS] + B[:TILE_ROWS]), rtol=1e-4, atol=1e-4)
```

The scheduler may then pre-stage that task's consumers onto idle cores *before* it
completes, releasing them with a doorbell the instant it finishes.

It is available on `pl.at`, `pl.submit`, `pl.spmd`, and `pl.spmd_submit`, and it is a pure
scheduling hint — no effect on results.

**Cost:** effectively none for correctness, but note the rule that decides whether it does
anything: a consumer only pre-stages once **all** of its producers are flagged (or already
complete). Flagging one producer of a three-producer consumer buys nothing. This is why it
tends to be applied along a whole chain — as in `models/qwen3_14b/decode_fwd.py`, where
nearly every task on the decode path carries it.

**How to confirm:** the `[dispatch, start]` gaps on the critical path shrink. Total task
count and graph shape do not change — if they did, something else changed too.

## Synchronize inside the kernel

**When it applies:** blocks of one SPMD launch must meet at a barrier. Expressing that as
two tasks with a dependency sends the synchronization out to the AICPU scheduler and back.

**How:** `pl.system.syncall()` synchronizes the participating cores from inside the kernel.

```python
# Hard barrier (FFTS): no operands, but requires FULL occupancy
with pl.spmd(pl.system.available_aiv_count()):
    ...
    pl.system.syncall(core_type=pl.KernelType.AIV)
    ...
```

Two modes, and the choice is not stylistic:

| Mode | Mechanism | Occupancy | Extra arguments |
| ---- | --------- | --------- | --------------- |
| `pl.SyncAllMode.HARD` (default) | FFTS barrier | **All** physical cores of `core_type` | None |
| `pl.SyncAllMode.SOFT` | GM-polling counter | Any (`used_cores` participants) | `gm_workspace`, `used_cores` |

`mode` and `core_type` are enums (`pl.SyncAllMode`, `pl.KernelType` — `MIX` is the
both-kernel participant set). The strings these keywords used to take — `mode="soft"`,
`core_type="mix"` — still work and emit a `DeprecationWarning`.

Both modes synchronize arrival only: they do not wait for a preceding `TSTORE` or make
business data cache-coherent. For a GM producer-to-consumer handoff that may span multiple
cache lines, conservatively use whole-GM `pl.system.cacheinvalid()` +
`pl.system.fence()` before `syncall`, then call `pl.system.cacheinvalid()` again on the
consumer before its read. The tensor-region overload currently invalidates only the cache
line containing the view's base address.

**Run it:** `python examples/advanced/05_runtime_overhead.py --mode soft_barrier` — it needs
the pto-isa pinned in `runtime/pto_isa.pin`, since the cacheinvalid path emits
`cache_line_t::SINGLE_CACHE_LINE`.

**Cost, and it is a sharp one.** A hard `syncall` under a partial launch **deadlocks on
device** (error 507018). PyPTO rejects that at compile time — the `HardSyncallOccupancy`
verifier — which is why the grid must be sized with `available_aiv_count()` /
`available_cluster_count()` rather than a literal that happens to match today's device. If
you cannot guarantee full occupancy, use `mode=pl.SyncAllMode.SOFT`: it polls a shared GM workspace, so
it works at partial occupancy and costs GM traffic instead.

```python
# Soft barrier: works at partial occupancy
pl.system.syncall(mode=pl.SyncAllMode.SOFT,
                  core_type=pl.KernelType.MIX,
                  gm_workspace=ws,     # exclusive zero-init 16-element INT32 GM tensor
                  used_cores=n)
```

**How to confirm:** the AICPU scheduler lane in the swimlane loses the round-trip that used
to sit between the two halves of the work, and the two tasks become one.

## See also

- [Managing dependencies](03-dependencies.md) — when the cost is not per-task but a
  serialized graph.
- [Tasks and Ordering § refining the graph](../tasks/03-tuning.md) — the reference
  treatment of `allow_early_resolve` and `predicate=`.
