# Your First Operator

Building `c = a + b` four times over, each version closer to how real kernels are written.

> **Prerequisites:** [Types](../language/00-types.md) and
> [Functions](../language/01-functions.md).
> **Companion file:** `examples/beginner/02_elementwise.py`.

## What you are building

An element-wise add, checked against torch. It is deliberately the least interesting
arithmetic available, because every step here is about *placement* — where the data is and
who is allowed to touch it — and arithmetic would only get in the way.

Four steps, each runnable:

1. Tensor level — say what to compute, let the compiler place it
2. Tile level — place it yourself: load, compute, store
3. Chunked — a tensor larger than one tile
4. Checked — compare against torch

## Step 1: tensor level

```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

@pl.jit
def add_tensor(
    a: pl.Tensor[[128, 128], pl.FP32],
    b: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        out = pl.assemble(out, pl.add(a, b), [0, 0])
    return out

a = torch.randn(128, 128)
b = torch.randn(128, 128)
out = torch.zeros(128, 128)
add_tensor(a, b, out, config=RunConfig(platform="a2a3sim"))
assert torch.allclose(out, a + b, rtol=1e-5, atol=1e-5)
```

Three things are doing work here.

**`pl.Out[...]` is a direction, not a hint.** It tells the compiler this parameter is
written and not read. That declaration is what the runtime uses to order this task against
others — see [The dependency model](../tasks/00-model.md). Get it wrong and you get a race,
not an error.

**`with pl.at(level=pl.Level.CORE_GROUP):` is where the operators go.** The `@pl.jit` body
itself is the control plane; it dispatches work but cannot contain operators. Writing
`out = pl.add(a, b)` outside the block is an error — the parser says so.

**`pl.assemble` is how you write an output.** This is the step everyone gets wrong:

```python
out = pl.add(a, b)                              # ✗ compiles, writes nothing
out = pl.assemble(out, pl.add(a, b), [0, 0])    # ✓
```

The first line rebinds a local name. It compiles, it runs, and nothing ever writes the
output — you are handed whatever that buffer happened to hold, with no diagnostic from any
stage. On the simulator these examples come back as NaN. `pl.assemble(dst, value, offset)`
is the operator that actually places `value` into `dst` at `offset`.

> **Fatal pitfall:** the wrong form is the one that reads naturally. If a kernel returns
> garbage and nothing errored, check that every write goes through `pl.assemble` or
> `pl.store`.

## Step 2: tile level

Step 1 never said where the data lives. The compiler chose. Doing it yourself means three
explicit operators:

```python
@pl.jit
def add_tile(
    a: pl.Tensor[[128, 128], pl.FP32],
    b: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        pl.store(pl.add(tile_a, tile_b), [0, 0], out)
    return out
```

| Operator | What it does |
| -------- | ------------ |
| `pl.load(t, offset, shape)` | Copies a window of the DDR tensor into an on-chip tile |
| `pl.add(tile, tile)` | Now a *tile* op — same name, operand types decide |
| `pl.store(tile, offset, t)` | Copies the tile back out |

`pl.store` is the tile-level counterpart of `pl.assemble`. Same rule: it is the operator
that writes, so the result has to flow through it.

Both versions compute the same thing. Step 1 is shorter; step 2 is what you need as soon
as the shape stops fitting.

## Step 3: a tensor larger than one tile

A tile is a fixed-size window of on-chip memory, so a `[512, 128]` tensor cannot be loaded
in one go. Loop over the chunks and move the *offset* — the shape argument stays the tile
size:

```python
ROWS, COLS, TILE_ROWS = 512, 128, 128

@pl.jit
def add_chunked(
    a: pl.Tensor[[512, 128], pl.FP32],
    b: pl.Tensor[[512, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        for i in pl.range(ROWS // TILE_ROWS):
            tile_a = pl.load(a, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
            tile_b = pl.load(b, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
            pl.store(pl.add(tile_a, tile_b), [i * TILE_ROWS, 0], out)
    return out
```

`pl.range` is a compile-time loop: it is unrolled into the IR, so `i` is a constant at each
step and the offsets are static. This is the shape of every real kernel — the arithmetic
sits inside a loop nest that walks the tensor a tile at a time.

## Step 4: check it

Nothing so far proves the kernel is right. Compare against torch, and assert:

```python
torch.manual_seed(0)
a = torch.randn(512, 128)
b = torch.randn(512, 128)
out = torch.zeros(512, 128)
add_chunked(a, b, out, config=RunConfig(platform="a2a3sim"))
assert torch.allclose(out, a + b, rtol=1e-5, atol=1e-5)
```

Assert rather than print. A kernel that silently writes nothing leaves an unwritten
buffer, and `allclose` catches that whatever it holds — a glance at printed output may
not.

Run the finished file:

```bash
python examples/beginner/02_elementwise.py
```

## Edge Cases

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **Output is unwritten (NaN or garbage) and nothing errored** | The result was assigned to the `Out` parameter instead of written | Route it through `pl.assemble` / `pl.store` |
| **`Misplaced tensor op`** | Operators sit in the `@pl.jit` body, outside `pl.at` | Move them inside `with pl.at(level=pl.Level.CORE_GROUP):` |
| **Tile shape rejected** | The window exceeds what on-chip memory holds | Chunk it — step 3 |
| **Results differ between runs** | Two tasks touching one buffer with nothing ordering them | See [The dependency model](../tasks/00-model.md) |

## Next

[Reduction and softmax](01-reduction-softmax.md) — where the result of one tile depends on
every element of another, and the tile vocabulary stops being enough.
