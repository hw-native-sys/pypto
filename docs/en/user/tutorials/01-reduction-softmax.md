# Reduction and Softmax

Where one output element depends on a whole row, and the tile vocabulary grows two new
shapes.

> **Prerequisites:** [Your first operator](00-elementwise.md).
> **Companion file:** `examples/intermediate/02_softmax.py`.

## What you are building

A numerically stable softmax over a `[64, 64]` tile. Getting there needs three things the
element-wise track never used: a scratch tile for reductions, a column vector, and a
broadcast back to full width.

## The shape story

Element-wise ops preserve shape. A reduction does not:

```text
[64, 64]  --row_max-->  [64, 1]  --row_expand_sub-->  [64, 64]
```

Everything in this page is a consequence of that `[64, 1]` in the middle. It is a real tile
of its own shape, and you cannot subtract it from a `[64, 64]` with plain `pl.sub` — the
shapes do not match. The `row_expand_*` family exists to close that gap.

## Step 1: a row reduction needs a scratch tile

```python
max_tmp = pl.create_tile([64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
row_max: pl.Tile[[64, 1], pl.FP32] = pl.row_max(tile_a, max_tmp)
```

`pl.row_max` takes a second argument: a **full-width scratch tile**, same shape as the
input, that the reduction uses as working space. It is not an output and you never read it.
`pl.row_sum` has the same signature.

Two consequences worth planning for:

- The scratch is full-width, so a row reduction costs roughly *twice* the buffer of the
  tile it reduces. That, not the reduction itself, is usually what pushes a kernel over the
  vector budget.
- It is created with `target_memory=pl.MemorySpace.Vec` — reductions run on the vector
  units, and the scratch has to live where they can reach it.

## Step 2: broadcasting back

A `[64, 1]` result has to be applied across every column. That is the `row_expand_*`
family — one operator per combining operation, rather than a broadcast rule you can apply
to any op:

| Operator | Computes |
| -------- | -------- |
| `pl.row_expand_sub(t, v)` | `t - v` broadcast along columns |
| `pl.row_expand_div(t, v)` | `t / v` |
| `pl.row_expand_expdif(t, v)` | `exp(t - v)`, fused |

`row_expand_expdif` is the fused form of steps 2 and 3 below. Reach for it once the
unfused version works.

## Step 3: the whole softmax

Softmax is `exp(x) / sum(exp(x))`, but computed that way it overflows: `exp(88)` already
leaves FP32 range. Subtracting the row maximum first is mathematically a no-op — the
`exp(-max)` factors cancel between numerator and denominator — and keeps every exponent
at or below zero.

```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

@pl.jit
def softmax(
    a: pl.Tensor[[64, 64], pl.FP32],
    out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        tile_a = pl.load(a, [0, 0], [64, 64])

        max_tmp = pl.create_tile([64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row_max: pl.Tile[[64, 1], pl.FP32] = pl.row_max(tile_a, max_tmp)

        shifted = pl.row_expand_sub(tile_a, row_max)   # x - max(x)
        exp_shifted = pl.exp(shifted)

        sum_tmp = pl.create_tile([64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row_sum: pl.Tile[[64, 1], pl.FP32] = pl.row_sum(exp_shifted, sum_tmp)

        pl.store(pl.row_expand_div(exp_shifted, row_sum), [0, 0], out)
    return out

torch.manual_seed(0)
a = torch.randn(64, 64)
out = torch.zeros(64, 64)
softmax(a, out, config=RunConfig(platform="a2a3sim"))
assert torch.allclose(out, torch.softmax(a, dim=1), rtol=1e-5, atol=1e-5)
```

Two scratch tiles, because the two reductions are live at different times and each needs
its own working space.

Run it:

```bash
python examples/intermediate/02_softmax.py
```

## Step 4: partial rows change the answer

Real inputs rarely divide evenly into tiles. `pl.load` takes `valid_shape=` to say "only
this sub-region holds real data", and the rest of the tile is padding.

For an element-wise op the padding is harmless — garbage in, garbage out, in lanes nobody
reads. **For a reduction it is not**, because padding participates:

```python
tile = pl.load(a, [0, 0], [64, 64], valid_shape=[64, vlen])
```

With 40 valid columns, `row_max` over that tile sees 24 columns of whatever the padding
holds. If it is zero and your data is all-negative, every row maximum comes back `0.0` —
wrong, and wrong quietly.

`pl.fillpad` sets what the padding contains, and the right value depends on the reduction:

| Reduction | Pad with | Why |
| --------- | -------- | --- |
| `row_max` | `pl.PadValue.min` | The smallest representable value never wins a max |
| `row_sum` | `pl.PadValue.zero` | Zero is the identity of addition |

The rule generalises: **pad with the identity of the operation you are about to apply.**
`PadValue` also offers `max` (for a min reduction) and `null` (no padding).

See [dynamic valid shapes](../language/03-memory.md) and
`examples/intermediate/06_dyn_valid_shape.py` for the loop that walks a tensor whose last
block is partial.

## Edge Cases

> **Fatal pitfall:** a reduction over a padded tile reads the padding. It produces a
> plausible number, not an error. Whenever a tile carries `valid_shape=` and then meets
> `row_max` / `row_sum`, decide what the padding contains.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **Softmax returns NaN or inf** | `exp` applied without subtracting the row max | Subtract `row_max` first — step 3 |
| **Reduction rejected: missing argument** | `row_max` / `row_sum` need a scratch tile | Pass a full-width `pl.create_tile(...)` |
| **Row maxima are all `0.0`** | Padding participated in the reduction | `pl.fillpad(..., pl.PadValue.min)` before `row_max` |
| **Vector buffer exceeded** | Each reduction's scratch is full-width | Reduce the tile, or reuse one scratch across non-overlapping reductions |
| **`row_expand_*` shape mismatch** | Applying a `[N, 1]` vector with plain `pl.sub` | Use the `row_expand_*` member for that operation |

## Next

[Tiled matmul](02-matmul.md) — the first operator that runs on the cube unit, and the first
one whose data path you have to think about.
