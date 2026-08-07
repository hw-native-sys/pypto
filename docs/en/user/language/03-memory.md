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
t = pl.set_validshape(t, [rows_left, 128])     # only `rows_left` rows are real
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
[AllocateMemoryAddr](../../dev/passes/35-allocate_memory_addr.md); the performance chapter
covering how to drive them is not written yet.

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
| **On-chip buffer exhaustion** | Too much resident at once | Shrink tiles, or shrink the cross-core ring with `pl.cross_core_slot(slot_num=N)` |

## See Also

- [Types](00-types.md) — `Tensor` versus `Tile`, and what a dtype's `get_byte()` is for.
- [Scopes and Placement](04-scopes.md) — where the code runs, and cross-core ring depth.
- [Operations](../ops/01-catalog.md) — the movement, reduction, and broadcast families.
- [InferTileMemorySpace](../../dev/passes/18-infer_tile_memory_space.md) — the pass that inserts moves you did not write.
- [MemoryReuse](../../dev/passes/34-memory_reuse.md) — how buffers are shared across lifetimes.
- [Memory Map](../../dev/07-memory-map.md) — visualizing what ended up on chip.
