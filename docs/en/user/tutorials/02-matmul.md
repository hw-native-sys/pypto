# Tiled Matmul

The first operator that runs on the cube unit, and the first whose K axis does not fit.

> **Prerequisites:** [Your first operator](00-elementwise.md).
> **Companion files:** `examples/intermediate/04_matmul_acc.py`,
> `examples/advanced/01_split_k.py`, `examples/advanced/02_auto_tile_matmul.py`.

## What you are building

`C = A @ B` where the K axis is larger than one tile, so the product has to be accumulated
across several steps. Then one variant that trades determinism for parallelism.

## Step 1: a matmul that fits

```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

@pl.jit
def matmul_small(
    a: pl.Tensor[[128, 128], pl.FP16],
    b: pl.Tensor[[128, 128], pl.FP16],
    c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="matmul"):
        c[:] = pl.matmul(a, b, out_dtype=pl.FP32)
    return c

torch.manual_seed(0)
a = torch.randn(128, 128, dtype=torch.float16)
b = torch.randn(128, 128, dtype=torch.float16)
c = torch.zeros(128, 128, dtype=torch.float32)
matmul_small(a, b, c, config=RunConfig(platform="a2a3sim"))
assert torch.allclose(c, a.float() @ b.float(), rtol=1e-2, atol=1e-2)
```

Two details that are not stylistic:

**`out_dtype=pl.FP32` with FP16 inputs.** The cube unit multiplies in the input precision
and accumulates in FP32. Asking for an FP16 accumulator loses precision for nothing —
accumulate wide, cast at the end if you must.

**The tolerance is `1e-2`, not `1e-5`.** FP16 inputs carry ~3 decimal digits. Comparing a
FP16 matmul against a FP32 torch reference at `1e-5` fails on correct code; picking the
tolerance to match the input precision is part of writing the test.

## Step 2: the K axis does not fit

`A[128, 512] @ B[512, 128]` cannot stage all of K at once. Split K into blocks, and
accumulate: the first block produces the accumulator, the rest add into it.

```python
K_CHUNK = 128

@pl.jit
def matmul_k_blocked(
    a: pl.Tensor[[128, 512], pl.FP16],
    b: pl.Tensor[[512, 128], pl.FP16],
    c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_blocked"):
        acc = pl.matmul(a[:, 0:K_CHUNK], b[0:K_CHUNK, :], out_dtype=pl.FP32)
        for k in pl.range(1, 512 // K_CHUNK):
            k0 = k * K_CHUNK
            acc = pl.matmul_acc(acc, a[:, k0 : k0 + K_CHUNK], b[k0 : k0 + K_CHUNK, :])
        c[:] = acc
    return c
```

`pl.matmul` **creates** an accumulator; `pl.matmul_acc` **adds into** one. The asymmetry is
why the loop starts at 1 — the zeroth block has nothing to add into yet. Writing
`pl.matmul_acc` for every block, over an accumulator you allocated separately, also works
and costs one extra initialisation.

The accumulator stays on-chip across the whole loop. Only the final store touches
DDR — which is the point of blocking K rather than storing each partial product.

## Step 3: when the compiler does it for you

`AutoTileMatmulL0` re-tiles a matmul that does not fit the cube's L0 buffers, choosing the
M/N/K blocking itself. That is why step 1 worked without you naming a single tile: the
shapes were handed to `pl.matmul` whole and the pass sorted out the staging.

The consequence worth knowing: **a `pl.matmul` on tensor-level operands is not one
instruction.** It is a loop nest the compiler wrote. When you block K by hand as in step 2,
you are overriding that choice for the K axis and leaving M/N to the pass.
`examples/advanced/02_auto_tile_matmul.py` walks the cases where the automatic choice
differs from the obvious one.

## Step 4: split-K, and what it costs

Blocking K keeps one core busy across every block. **Split-K** gives each core its own
slice of K and has them accumulate into the same output with an atomic add:

```python
KS = K // SPLITS                       # each core's slice of K

with pl.at(level=pl.Level.CORE_GROUP, name_hint="zero_init"):
    c[:] = pl.full([M, N], dtype=pl.FP32, value=0.0)
for ks in pl.parallel(SPLITS):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="split_k"):
        k0 = ks * KS
        partial = pl.matmul(a[:, k0 : k0 + KS], b[k0 : k0 + KS, :], out_dtype=pl.FP32)
        c = pl.assemble(c, partial, [0, 0], atomic=pl.AtomicType.Add)
```

Fragment — `M`, `N`, `K` and `SPLITS` come from the enclosing function; the runnable
version is `examples/advanced/01_split_k.py`. Note that unlike step 2, every core writes
the *whole* `[M, N]` output — it is K that is divided, not the output.

| Aspect | Cost |
| ------ | ---- |
| Zero-init | The output must be zeroed first; atomic add has no "first writer" |
| Determinism | **Accumulation order across cores is not fixed**, so repeated runs may differ in the last bits |
| When it pays | K is large and M/N are too small to fill the cores on their own |

That second row is the one to weigh. If a downstream test compares bitwise, or you are
chasing a numerical discrepancy, split-K makes the answer a moving target. Use it when the
parallelism is worth more than reproducibility.

## Edge Cases

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **`allclose` fails at `1e-5` on FP16 inputs** | Tolerance tighter than the input precision | Compare at `1e-2`; keep the accumulator FP32 |
| **Results drift between identical runs** | Split-K's atomic accumulation order is not fixed | Expected — use K-blocking if you need determinism |
| **Split-K output is too large by roughly a factor** | The output was not zeroed before the atomic loop | Zero-init in its own scope first |
| **Accumulator dtype rejected** | `matmul_acc` requires the accumulator's dtype | Create it with `pl.matmul(..., out_dtype=pl.FP32)` |

## Next

[Mixed kernels](03-mixed-kernel.md) — everything so far used the cube *or* the vector
units. Now both, at once.
