# Runtime Overhead

Four ways to spend less time per task without changing what the tasks compute.

> **Prerequisites:** [Task granularity](01-task-granularity.md).

## Where this differs from the previous page

Granularity changes *how many* tasks there are. This page changes *what each one costs*:
one dispatch instead of two, one dispatch instead of `N`, a dispatch that starts earlier,
or no dispatch at all where a barrier will do.

| Technique | Removes |
| --------- | ------- |
| [Mixed kernel](#build-a-mixed-kernel) | One of two dispatches, plus the GM round-trip between them |
| [SPMD](#use-spmd) | `N − 1` dispatches for `N` blocks of the same work |
| [`allow_early_resolve`](#let-consumers-pre-stage) | The pickup latency on the critical path |
| [In-kernel `syncall`](#synchronize-inside-the-kernel) | An AICPU round-trip per synchronization point |

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

**Cost:** every block runs the same program. Divergent work needs a different structure,
and blocks that finish at different times leave their cores idle until the whole grid
retires.

**How to confirm:** `deps.json` collapses `N` nodes into one; the swimlane shows one task
occupying many core lanes at once. The plugin highlights all blocks of an SPMD task
together when you click one.

## Let consumers pre-stage

**When it applies:** a critical path built from many short tasks, where each consumer sits
waiting through its own pickup latency after its producer ends.

**How:** flag the **producer**.

```python
with pl.at(level=pl.Level.CORE_GROUP, allow_early_resolve=True) as tid:
    ...
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
    pl.system.syncall(core_type="aiv_only")
    ...
```

Two modes, and the choice is not stylistic:

| Mode | Mechanism | Occupancy | Extra arguments |
| ---- | --------- | --------- | --------------- |
| `mode="hard"` (default) | FFTS barrier | **All** physical cores of `core_type` | None |
| `mode="soft"` | GM-polling counter | Any (`used_cores` participants) | `gm_workspace`, `used_cores` |

**Cost, and it is a sharp one.** A hard `syncall` under a partial launch **deadlocks on
device** (error 507018). PyPTO rejects that at compile time — the `HardSyncallOccupancy`
verifier — which is why the grid must be sized with `available_aiv_count()` /
`available_cluster_count()` rather than a literal that happens to match today's device. If
you cannot guarantee full occupancy, use `mode="soft"`: it polls a shared GM workspace, so
it works at partial occupancy and costs GM traffic instead.

```python
# Soft barrier: works at partial occupancy
pl.system.syncall(mode="soft", core_type="mix",
                  gm_workspace=ws,     # shared zero-init INT32 GM tensor,
                  used_cores=n)        # at least used_cores * 8 elements
```

**How to confirm:** the AICPU scheduler lane in the swimlane loses the round-trip that used
to sit between the two halves of the work, and the two tasks become one.

## See also

- [Managing dependencies](03-dependencies.md) — when the cost is not per-task but a
  serialized graph.
- [Tasks and Ordering § refining the graph](../tasks/03-tuning.md) — the reference
  treatment of `allow_early_resolve` and `predicate=`.
