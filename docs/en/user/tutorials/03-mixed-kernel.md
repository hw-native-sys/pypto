# Mixed Kernels

> **Legacy Python API examples:** `platform=` calls on this page are retained for
> compatibility. They emit `DeprecationWarning`; new code should use
> `backend_type` and `execution_mode`. See the [target mapping](../../dev/08-entry-points.md#legacy-platform-examples).

Cube and vector work inside one scope, with compiler-managed transfers.

> **Prerequisites:** [Tiled matmul](02-matmul.md).
> **Companion file:** `examples/advanced/03_mixed_kernel.py`.

## What you are building

`a @ b + bias` — a cube operation followed by a vector one — written as one mixed
dispatch instead of two separate scopes. This example demonstrates the dataflow;
it does not establish a performance improvement.

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

Put cube work and explicitly sharded vector work in one scope:

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
        optimizations=[pl.cross_core_slot(slot_num=1)],
        name_hint="mixed",
    ):
        acc = pl.matmul(a, b, out_dtype=pl.FP32)                 # cube (AIC)
        for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            half = pl.aiv_shard(acc)
            row = lane * 64
            out[row:row + 64, :] = pl.add(half, bias[row:row + 64, :])
    return out

torch.manual_seed(0)
a = torch.randn(128, 256, dtype=torch.float16)
b = torch.randn(256, 128, dtype=torch.float16)
bias = torch.randn(128, 128, dtype=torch.float32)
out = torch.zeros(128, 128, dtype=torch.float32)
compiled = mixed.compile(a, b, bias, out, config=RunConfig(platform="a2a3sim"))
compiled(a, b, bias, out, config=RunConfig(platform="a2a3sim"))
assert torch.allclose(out, a.float() @ b.float() + bias, rtol=1e-2, atol=1e-2)
```

`pl.split_aiv(mode=...)` binds the AIV lane and supplies the boundary mode.
`pl.aiv_shard(acc)` explicitly partitions the cube result. The author writes
each lane's bias slice and output offset; the compiler does not halve the body.
The cube computation stays full-sized. Mixed function-level `pl.split` AUTO
splitting is no longer supported.

| Mode | Explicit boundary axis |
| ---- | ---------------------- |
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
| Tile crossing the boundary | `[128, 128]` FP32 = 64 KiB |
| Default ring depth | 2 slots |
| Ring depth explicitly chosen by this example | 1 slot = **64 KiB** |
| Bias and output per vector lane | 32 KiB each = **64 KiB** |
| Total with one slot | **128 KiB** |
| Safe vector budget | **184 KiB** |

The ring is a queue of whole tiles, so its size scales with the tile that crosses, not with
the work. The default of 2 is the shallowest depth that still double-buffers: the cube can
fill one slot while the vector drains the other.

`pl.cross_core_slot(slot_num=N)` retunes it. On Ascend910, the load/TPOP safety
constraint requires the loaded bias and addition output to occupy separate buffers.
The default two-slot ring would therefore need 128 + 32 + 32 = **192 KiB**, above
the safe budget. This example explicitly chooses one slot; it demonstrates mixed
dataflow, not double-buffered producer/consumer overlap. Deeper rings can allow more
producer run-ahead when smaller tiles leave enough space, but do not guarantee a speedup.

Changing this example back to two slots fails during allocation:

```python
with pl.at(
    level=pl.Level.CORE_GROUP,
    optimizations=[pl.cross_core_slot(slot_num=2)],
    name_hint="mixed",
):
```

```text
Vec buffer usage (196608 bytes) exceeds platform limit (188416 bytes). The first 131072
bytes of that space are reserved by system.reserve_buffer, so tiles are allocated above
them — this is the cross-core pipe ring. Lower its depth with
optimizations=[pl.cross_core_slot(slot_num=N)] on the enclosing pl.at(...), or shrink the
tile that crosses the cube/vector boundary
```

Two levers when that happens: shrink the tile, or shorten the ring. Pick the largest depth
that fits.

## Step 3: transport lowering

The compiler lowers explicit `aiv_shard`/`aic_gather` to pipe setup,
`tpush`/`tpop`, and the matching `tfree`. Region mode selects the split axis;
`NONE` transfers without changing shape. Users need not handwrite pipe pairing
when using these explicit boundary operators.

A manually authored low-level pipe must pair every push with a pop and every
pop with a free. A boundary orders the value it carries; user-managed GM sharing
and once-only side effects still need the ordering and lane ownership described
in [Scopes and Placement](../language/04-scopes.md).

Ordinary unsplit mixed kernels retain their existing full-width implicit
communication. This change removes implicit halving, not that transport path.
See [TPUSH/TPOP](../../reference/pto-isa/01-tpush_tpop.md) and
[ExpandMixedKernel](../../dev/passes/26-expand_mixed_kernel.md).

## Edge Cases

> **Fatal pitfall:** the ring is sized in whole tiles at the cube/vector boundary. A tile
> that grows turns a working kernel into one that cannot be allocated, and the error names
> a byte count rather than the tile — read it as "the crossing tile is too big, or the ring
> too deep".

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **`Vec buffer usage ... exceeds platform limit`** | The ring plus the tiles overrun the on-chip budget | Lower `pl.cross_core_slot(slot_num=N)`, or shrink the crossing tile |
| **No speedup from explicit splitting** | One side dominates, so halving cannot overlap anything | Check the work is genuinely cube-then-vector |
| **The producer stalls after a while** | A popped slot was never `tfree`d | Match every pop with a `tfree` |
| **Implicit AIV splitting is no longer supported** | Legacy function-level AUTO split | Use `pl.split_aiv`, explicit shard/gather, and per-lane shapes and offsets |

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
