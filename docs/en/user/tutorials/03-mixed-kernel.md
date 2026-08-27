# Mixed Kernels

Cube and vector working at the same time, inside one scope.

> **Prerequisites:** [Tiled matmul](02-matmul.md).
> **Companion file:** `examples/advanced/03_mixed_kernel.py`.

## What you are building

`a @ b + bias` — a cube operation followed by a vector one — written so the two units
overlap instead of taking turns.

## Why bother

A core group pairs one cube unit with vector units. Written the obvious way, the chain
occupies them one after the other:

```python
with pl.at(level=pl.Level.CORE_GROUP, name_hint="cube_only"):
    acc = pl.matmul(a, b, out_dtype=pl.FP32)
with pl.at(level=pl.Level.CORE_GROUP, name_hint="vector_only"):
    out[:] = pl.add(acc, bias)
```

Two scopes, two dispatches. The vector units have nothing to do until the matmul scope has
finished, and the cube unit has nothing to do afterwards. This is the form a mixed kernel
replaces — and it is what `examples/intermediate/01_fused_linear.py`-style "fused" kernels
often still are underneath: fused in name, sequential in execution.

## Step 1: one scope, split

Put both operations in one scope and mark it split:

```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

@pl.jit
def mixed(
    a: pl.Tensor[[128, 256], pl.FP16],
    b: pl.Tensor[[256, 128], pl.FP16],
    bias: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    with pl.at(
        level=pl.Level.CORE_GROUP,
        optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
        name_hint="mixed",
    ):
        acc = pl.matmul(a, b, out_dtype=pl.FP32)                 # cube (AIC)
        out[:] = pl.add(acc, bias)        # vector (AIV)
    return out

torch.manual_seed(0)
a = torch.randn(128, 256, dtype=torch.float16)
b = torch.randn(256, 128, dtype=torch.float16)
bias = torch.randn(128, 128, dtype=torch.float32)
out = torch.zeros(128, 128, dtype=torch.float32)
mixed(a, b, bias, out, config=RunConfig(platform="a2a3sim"))
assert torch.allclose(out, a.float() @ b.float() + bias, rtol=1e-2, atol=1e-2)
```

`pl.split(mode)` marks the scope as mixed, and the mode names the axis along which the
**vector** sub-region is halved. The cube sub-region stays full-sized: the split shards the
vector work across the two AIV lanes, and the compiler inserts the cross-core transfers
(`aiv_shard` at cube→vector boundaries, `aic_gather` on the way back) that carry results
between the units. Overlap comes from cube and vector running concurrently, not from each
taking half of one tile.

| Mode | Halves the vector sub-region along |
| ---- | ---------------------------------- |
| `pl.SplitMode.UP_DOWN` | Rows (height) |
| `pl.SplitMode.LEFT_RIGHT` | Columns (width) |
| `pl.SplitMode.NONE` | No split |

Which one to pick follows from the vector operands' shape: halve the axis that is large
enough to divide across two lanes evenly. Run the companion file with `--mode left_right`
to compare.

## Step 2: the ring spends your vector budget

The transfers the compiler inserted are not free. Every tile crossing the boundary lands in
a ring buffer carved out of the **consuming** core's on-chip memory — UB here, since the
cube feeds the vector units:

| Quantity | Value |
| -------- | ----- |
| Tile crossing the boundary | `[128, 128]` FP32 = 64 KB |
| Default ring depth | 2 slots |
| Ring size | 2 × 64 KB = **128 KB** |
| Vector budget | **184 KB** |

The ring is a queue of whole tiles, so its size scales with the tile that crosses, not with
the work. The default of 2 is the shallowest depth that still double-buffers: the cube can
fill one slot while the vector drains the other.

`pl.cross_core_slot(slot_num=N)` retunes it. Deeper rings buy more overlap — the producer
runs further ahead before it blocks — so raise it when the two units are poorly balanced.
But the budget is tight: at `slot_num=4` this kernel already fails to allocate.

```python
with pl.at(
    level=pl.Level.CORE_GROUP,
    optimizations=[pl.split(pl.SplitMode.UP_DOWN), pl.cross_core_slot(slot_num=4)],
    name_hint="mixed",
):
```

```text
Vec buffer usage (294912 bytes) exceeds platform limit (188416 bytes). The first 262144
bytes of that space are reserved by system.reserve_buffer, so tiles are allocated above
them — this is the cross-core pipe ring. Lower its depth with
optimizations=[pl.cross_core_slot(slot_num=N)] on the enclosing pl.at(...), or shrink the
tile that crosses the cube/vector boundary
```

Two levers when that happens: shrink the tile, or shorten the ring. Pick the largest depth
that fits.

## Step 3: what the compiler inserted

`pl.split` is the automatic path. Underneath, the cross-core dataflow is explicit
operators, and you can write them yourself:

| Operator | Role |
| -------- | ---- |
| `pl.aic_initialize_pipe` / `pl.aiv_initialize_pipe` | Set up the pipe |
| `pl.tpush_to_aiv` / `pl.tpush_to_aic` | Push a tile to the peer core |
| `pl.tpop_from_aic` / `pl.tpop_from_aiv` | Pop a tile the peer pushed |
| `pl.tfree_to_aic` / `pl.tfree_to_aiv` | Release the popped slot back to the producer |
| `pl.aiv_shard` / `pl.aic_gather` | Shard across AIV lanes, gather back on AIC |
| `pl.split_aiv(n, mode=...)` | The explicit region form of the split |

**Every push must be paired with a pop, and every pop with a `tfree`.** A missing `tfree`
does not error — it leaks a ring slot, and the producer stalls once the ring fills.

**The explicit form also makes cross-lane ordering yours.** A boundary operator orders only
the value it carries. Nothing orders a cube-lane write against a vector-lane read of the
same GM buffer. Publish and fence the producer's writes, place a
cross-core `pl.system.syncall` between those phases, then invalidate the consumer's cache
before it reads; the barrier alone only synchronizes arrival. Use the soft form when the
launch may have partial occupancy, and use whole-GM cache maintenance when the buffer may
span multiple cache lines. The `pl.split` path above does not need this sequence — the
compiler inserts the transfers, and the result is checked against torch. See
[Scopes and Placement](../language/04-scopes.md) for the rules.

Reach for the explicit form when `pl.split` cannot express the shape: per-lane addressing,
a gather that only one lane can compute, or a region that mixes split and unsplit work.
`tests/st/codegen/dsl/test_split_aiv_gather_row_codegen.py` is a worked example.
Otherwise stay on `pl.split` — it inserts the same operators and gets the pairing right.

For the machine-level contract see [TPUSH/TPOP](../../reference/pto-isa/01-tpush_tpop.md);
for what the pass does, [ExpandMixedKernel](../../dev/passes/22-expand_mixed_kernel.md).

## Edge Cases

> **Fatal pitfall:** the ring is sized in whole tiles at the cube/vector boundary. A tile
> that grows turns a working kernel into one that cannot be allocated, and the error names
> a byte count rather than the tile — read it as "the crossing tile is too big, or the ring
> too deep".

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **`Vec buffer usage ... exceeds platform limit`** | The ring plus the tiles overrun the on-chip budget | Lower `pl.cross_core_slot(slot_num=N)`, or shrink the crossing tile |
| **No speedup from `pl.split`** | One side dominates, so halving cannot overlap anything | Check the work is genuinely cube-then-vector |
| **The producer stalls after a while** | A popped slot was never `tfree`d | Match every pop with a `tfree` |
| **Split rejected on a scope** | The body mixes split and plain full-width vector ops | Use the explicit `pl.split_aiv` region form |

## The same shape in a real model

`examples/models/qwen3_jit/` is a `@pl.jit` decode path split one file per module, and its
`kernels/projection.py` is this page's pattern at model scale — a matmul and the vector work
that consumes it, inside one scope.

| File | Module |
| ---- | ------ |
| `qwen3_decode.py` | The decode entry that composes the rest |
| `config.py` | Shapes and dtypes the kernels are specialised on |
| `kernels/projection.py` | Mixed cube + vector projection |
| `kernels/attention.py` | Attention |
| `kernels/mlp.py` | MLP |
| `kernels/rmsnorm.py` | RMSNorm |

## Next

[Shaping the task graph](04-task-graph.md) — from inside one kernel to the order between
kernels.
