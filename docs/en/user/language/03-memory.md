# Memory and Data Movement

The on-chip spaces, the operations that move data between them, and what happens at the
edges of a tile that is not full.

> **Prerequisites:** [Programming Model § memory hierarchy](../03-programming-model.md#memory-hierarchy).

## Concept

```text
                 off-chip                          AI Core
              ┌────────────┐   ┌──────────────────────────────────────────────┐
              │            │   │                                              │
              │            │   │   Cube unit (AIC)         Vector unit (AIV)  │
              │            │   │  ┌──────────────┐        ┌─────────────────┐ │
              │    DDR     │   │  │ Left  (L0A)  │        │                 │ │
              │            │   │  │ Right (L0B)  │        │   Vec  (UB)     │ │
              │ pl.Tensor  │   │  │ Acc   (L0C)  │        │                 │ │
              │  lives     │   │  │ Bias         │        │                 │ │
              │  here      │   │  └──────▲───────┘        └────────▲────────┘ │
              │            │   │         │ pl.move                 │          │
              │            │   │  ┌──────┴─────────────────────────┴────────┐ │
              │            │   │  │              Mat  (L1)                  │ │
              │            │   │  └──────────────────▲──────────────────────┘ │
              └─────┬──────┘   └─────────────────────┼────────────────────────┘
                    │                                │
                    └────────  pl.load / pl.store  ──┴──►  Vec or Mat only
```

> Placeholder until a proper Ascend 910 floorplan is available.

The six on-chip spaces are **separate buffers, not a nesting**: `Left` is not a region
inside `Mat`, and `Acc` is not inside `Right`. Data moves *between* them, and which moves
are legal is a hardware property, not a compiler policy.

One constraint carries most of the consequences, and the diagram shows it: **a DDR-facing
load can only land in `Vec` or `Mat`.** The matmul operand spaces (`Left`, `Right`) and the
accumulator (`Acc`) are reachable only by a `pl.move` from `Mat` or `Vec`, or by
`pl.matmul` writing `Acc`. That is why the matmul path has an explicit two-step shape and
elementwise code does not.

The second recurring idea is the **valid shape**. A tile has an allocated shape and,
optionally, a smaller region that holds meaningful data. Operations that read past the
valid region see whatever padding says they see — which is why reductions over a partial
tile need the padding value chosen deliberately.

## Quickstart: the two paths

```python
import pypto.language as pl

@pl.jit.incore
def elementwise(x: pl.Tensor[[128, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[128, 128], pl.FP32]]):
    t = pl.load(x, [0, 0], [128, 128])       # DDR -> Vec (default target)
    y = pl.mul(t, t)                          # compute in Vec
    pl.store(y, [0, 0], out)                  # Vec -> DDR
    return out

@pl.jit.incore
def mm(a: pl.Tensor[[32, 32], pl.FP16],
       b: pl.Tensor[[32, 32], pl.FP16],
       out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]):
    a_l1 = pl.load(a, [0, 0], [32, 32], target_memory=pl.Mem.Mat)   # DDR -> Mat
    b_l1 = pl.load(b, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
    a_l0a = pl.move(a_l1, target_memory=pl.Mem.Left)                 # Mat -> Left
    b_l0b = pl.move(b_l1, target_memory=pl.Mem.Right)                # Mat -> Right
    c = pl.matmul(a_l0a, b_l0b)                                      # -> Acc
    pl.store(c, [0, 0], out)                                         # Acc -> DDR
    return out
```

A tensor-level `pl.matmul` lowers into that same chain. Writing it by hand is what buys
control over tiling and residency.

## Mechanics

### The spaces

| Space | Enum | Hardware | Reachable from DDR? |
| ----- | ---- | -------- | ------------------- |
| DDR | `pl.Mem.DDR` | Off-chip global memory | this *is* DDR — where `pl.Tensor` lives |
| Vec | `pl.Mem.Vec` | Unified buffer | **Yes** — the default `pl.load` target |
| Mat | `pl.Mem.Mat` | L1 | **Yes** — `target_memory=pl.Mem.Mat` |
| Left | `pl.Mem.Left` | L0A, matmul left operand | No — `pl.move` only |
| Right | `pl.Mem.Right` | L0B, matmul right operand | No — `pl.move` only |
| Acc | `pl.Mem.Acc` | L0C, matmul accumulator | No — `pl.matmul` writes it |
| Bias | `pl.Mem.Bias` | Bias buffer on the AIC core | No — `pl.move` only |

`pl.MemorySpace` and `pl.Mem` are the same enum under two names.

Dataflow, as opposed to containment — the two operands **converge** on `Acc`, so this is a
graph, not a tree:

```text
     pl.load(target_memory=Mat)     pl.move(Left)
DDR ───────────────────────► Mat ─────────────────► Left ┐
                                                         │ pl.matmul
                                                         ├────────► Acc ────► DDR
DDR ───────────────────────► Mat ─────────────────► Right┘            pl.store
     pl.load(target_memory=Mat)     pl.move(Right)

     pl.load()                elementwise ops             pl.store()
DDR ──────────► Vec ────────────────────────────► Vec ──────────────► DDR
     (default)
```

When a consumer needs `Left` / `Right` / `Acc` / `Bias`, the producer stops at `Mat` (or
`Vec`) and [InferTileMemorySpace](../../dev/passes/18-infer_tile_memory_space.md) inserts
the `tile.move` — you write it explicitly when you want to control where it happens.

### Moving data

| Operation | Direction | Notes |
| --------- | --------- | ----- |
| `pl.load(tensor, offsets, shape, target_memory=...)` | DDR → Vec / Mat | `Vec` by default |
| `pl.store(tile, offsets, tensor)` | any on-chip space → DDR | |
| `pl.move(tile, target_memory=...)` | on-chip → on-chip | The only way into `Left` / `Right` / `Bias` |
| `pl.create_tile(shape, dtype, ...)` | — | Allocate an on-chip buffer |
| `pl.create_l1(...)` | — | Allocate in L1 explicitly |

A placed buffer is described in the IR by a `pl.MemRef` — the space plus the address the
allocator assigned. You rarely write one; it shows up when reading printed IR and in the
memory map.

`offsets` and `shape` are the region of the tensor being moved — the offsets are into the
**tensor**, and the shape is the size of the resulting tile.

### Scalar element access

`pl.read` and `pl.write` reach a single element of a **tensor** by index, with no tile in
between:

```python
n = pl.read(counts, [0])                      # one INT32 out of DDR
pl.write(plan, [row], pl.cast(v, pl.INT32))   # one INT32 into DDR
```

This is a different route to memory than `pl.load` / `pl.store`, not a smaller version of
it:

| Aspect | `pl.load` / `pl.store` | `pl.read` / `pl.write` |
| ------ | ---------------------- | ---------------------- |
| Unit | a tile | one element |
| Route to DDR | DMA, direct | the issuing core's data cache |
| Granularity that reaches DDR | the bytes of the tile | **a whole 64-byte cache line** |
| Safe from several instances at once | yes | only under the rule below |

Use them for control values — counters, offsets, small descriptor tables — not for bulk
data. A loop of `pl.write` moves one element per iteration where one `pl.store` moves a
whole tile.

### Scalar writes from concurrent task instances

A `pl.write` does not reach DDR on its own. It lands in the issuing core's data cache, and
that cache writes back **whole 64-byte lines** when the kernel ends. Nothing keeps
different cores' caches coherent with each other.

So when two cores write different elements that happen to share one 64-byte line, each
writes back its own copy of all 64 bytes — its one fresh element plus the 15 stale ones it
never touched. The last write-back wins the whole line and the other core's store is gone.
There is no error and no warning at runtime: the tensor simply keeps its old value at most
indices, and *which* indices survive changes from run to run.

> **Fatal pitfall:** two instances writing *different* elements of one 64-byte line
> silently lose each other's stores. Disjoint indices are **not** enough — the line is the
> unit that reaches memory, so an instance's write-back also carries the 15 neighbouring
> elements as it saw them, overwriting whatever another instance put there.

This is about **concurrency, not about `pl.spmd`**. Two things run your code as more than
one instance, and either is enough to hit it:

| Construct | Instances |
| --------- | --------- |
| `pl.spmd(n)`, `n > 1` | `n` blocks, one per core |
| `for g in pl.parallel(n):` | `n` task instances the runtime may overlap |

A kernel dispatched from either inherits the multiplicity, so writes inside a
`@pl.function(type=InCore)` callee count too.

**Within one instance there is no hazard.** A single instance runs its body sequentially on
one core, so its stores land in one cache in program order and no line is contended — an
ordinary `pl.range` loop of `pl.write` inside one task is always safe, whatever the indices.

**The rule: each instance must own whole 64-byte lines.** That is 16 elements for
`INT32` / `FP32`, 32 for `FP16` / `BF16`, 8 for `INT64`, 64 for `INT8`.

```python
N = 64          # INT32 -> 16 elements per 64-byte line

# WRONG — grid-stride: blocks 0..15 each land one element in out[0:16], so
# 16 blocks share that first line (and the later lines they also write)
with pl.spmd(24):
    blk = pl.tile.get_block_idx()
    for i in pl.range(pl.cast(blk, pl.INDEX), N, 24):
        pl.write(out, [i], pl.cast(pl.read(src, [i]) + 1, pl.INT32))

# RIGHT — block b owns out[16b : 16b+16], exactly one line
with pl.spmd(N // 16):
    blk = pl.tile.get_block_idx()
    base = pl.cast(blk, pl.INDEX) * 16
    for i in pl.range(base, base + 16):
        pl.write(out, [i], pl.cast(pl.read(src, [i]) + 1, pl.INT32))
```

Both bodies write every index exactly once, from exactly one block. Only the second is
correct.

| If you need | Use |
| ----------- | --- |
| A handful of control values | `pl.spmd(1)` — one instance is correct at any layout |
| Each instance to write a contiguous run | Size *and* align that run to 64 bytes |
| A real scatter from many instances | `pl.store(..., atomic=pl.AtomicType.ADD)` into a zeroed tensor — the DMA path, coherent |
| Per-instance partial results | Write a per-instance scratch row, gather in a later `pl.spmd(1)` |

The compiler warns when it cannot prove the rule holds — see
[`ScalarWriteLineShared`](#scalarwritelineshared) below.

Reads share the cache but not the hazard in practice: an instance only sees a stale element
if another wrote it *during the same task*, which already breaks the independence
`pl.spmd` and `pl.parallel` assert. Across tasks the line is invalidated, so the next one
reads fresh data.

#### `ScalarWriteLineShared`

For every `pl.write` into a tensor that outlives the instance writing it, the compiler
tries to prove that each instance's bytes fall in whole, instance-private 64-byte lines. It
reports what it could not prove, and says which of the two it hit.

When the index is analysable and the layout is genuinely interleaved, it names the measured
stride:

```text
[warning] [ScalarWriteLineShared] pl.write into 'out' from 24 concurrent blocks
  ('fill_spmd') in function 'main': consecutive blocks write 4 bytes apart, so 16 of
  them share each 64-byte cache line and their stores overwrite one another. [...]
  Give each one whole 64-byte lines (16 x INT32), or issue the writes from a single
  instance (pl.spmd(1)).
```

When the index cannot be analysed at all — an index read from another tensor is the usual
reason — it says so rather than guessing:

```text
[warning] [ScalarWriteLineShared] pl.write into 'out' from 24 concurrent blocks
  ('moe_route_gather_spmd') in function 'main': the index is computed at runtime, so
  the compiler cannot tell whether two blocks share a 64-byte cache line. [...]
```

The second form is the common one, and it is a question, not a verdict: code with a
runtime index may well be correct. Check that your instances land on 64-byte boundaries; if
they do, the warning is telling you that correctness rests on a layout invariant nothing
enforces, which is worth a comment at the write site. To silence the check across a build,
put `ScalarWriteLineShared` in the pass context's `disabled_diagnostics`.

Two cases it does not decide precisely, because both need the task dependency graph that
does not exist this early. Two *different* tasks writing one tensor is **not reported** at
all — whether they overlap in time is unknown, so reporting would fire on every ordered
producer/consumer pair. A write guarded by a predicate that pins it to a single instance
(`if blk == 0:`) **is reported**, conservatively: the guard makes it safe, but the check
does not read predicates, so it treats the write as multi-instance.

### Valid shape and padding

An operation may write fewer elements than a tile allocates — the last block of a
dimension that does not divide evenly, for instance. `pl.set_validshape` records that
region; `pl.fillpad` fills the remainder with a chosen value.

The padding value matters for reductions, and the failure is silent:

- A `max` reduction over a tile padded with zeros returns at least 0, even when every
  real element is negative. Pad with `pl.PadValue.min`.
- A `sum` over a tile padded with anything but 0 adds the padding into the result.

```python
t = pl.load(x, [off, 0], [128, 128])
t = pl.set_validshape(t, rows_left, 128)       # only `rows_left` rows are real
m = pl.row_max(t)                              # pad value decides what the tail contributes
```

`pl.fillpad_expand` combines the fill with a broadcast. The reduction and broadcast
families themselves are catalogued in [Operations](../ops/01-catalog.md).

### Keeping data on chip

Reloading the same operand for every tile of a loop is the most common avoidable cost.
Hoist the load out of the loop when the operand is loop-invariant, and prefer `Mat`
residency for a matmul operand reused across the K loop. What the compiler will and will
not do here — buffer reuse, address assignment — is decided by
[MemoryReuse](../../dev/passes/34-memory_reuse.md) and
[AllocateMemoryAddr](../../dev/passes/35-allocate_memory_addr.md); [Memory](../performance/05-memory.md)
covers how to drive them.

## Edge Cases

> **Fatal pitfall:** a `max` reduction over a partially valid tile padded with zeros
> returns 0 for an all-negative row. There is no error and no warning — the number is
> simply wrong for exactly the rows that were short. Choose the pad value from the
> reduction: `PadValue.min` for `max`, `0` for `sum`.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **Tile-level `pl.matmul` rejects its operands** | Operands are not in `Left` / `Right` | `pl.load` to `Mat`, then `pl.move` |
| **`pl.load(..., target_memory=pl.Mem.Left)` rejected** | DDR loads reach only `Vec` / `Mat` | Load to `Mat`, then `pl.move` to `Left` |
| **Reduction result wrong only for the last tile** | Padding participates in the reduction | `pl.set_validshape`, and pick the right `PadValue` |
| **`pl.create_tensor` inside an InCore function fails** | Tensor allocation is control-plane work | Allocate on the control plane, or take a `pl.Out[...]` parameter |
| **Most `pl.write` stores vanish, a different set each run** | Concurrent `pl.spmd` blocks or `pl.parallel` instances write into one 64-byte line | Give each instance whole 64-byte lines, or write from `pl.spmd(1)` |
| **On-chip buffer exhaustion** | Too much resident at once | Shrink tiles, or shrink the cross-core ring with `pl.cross_core_slot(slot_num=N)` |

## Worked examples

| Example | Shows |
| ------- | ----- |
| `examples/intermediate/05_assemble.py` | Writing a tile into a target at an offset, without a GM round-trip |
| `examples/intermediate/01_fused_linear.py` | An intermediate kept on chip across a cube and a vector op |
| `examples/runtime/multi_program_kv_cache.py` | A device-resident buffer shared across programs |

## See Also

- [Types](00-types.md) — `Tensor` versus `Tile`, and what a dtype's `get_byte()` is for.
- [Scopes and Placement](04-scopes.md) — `pl.spmd` blocks, and what independence you are asserting.
- [Scopes and Placement](04-scopes.md) — where the code runs, and cross-core ring depth.
- [Operations](../ops/01-catalog.md) — the movement, reduction, and broadcast families.
- [InferTileMemorySpace](../../dev/passes/18-infer_tile_memory_space.md) — the pass that inserts moves you did not write.
- [MemoryReuse](../../dev/passes/34-memory_reuse.md) — how buffers are shared across lifetimes.
- [Memory Map](../../dev/07-memory-map.md) — visualizing what ended up on chip.
