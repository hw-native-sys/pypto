# Hello, Rank

Run your first two-rank program: every rank adds its own index to its slice of
the output, and the golden proves each rank touched exactly its own row.

> **Prerequisites:** [05-tutorials](05-tutorials.md) for the tutorial index; the
> [Distributed Programming](../distributed/index.md) chapter for the
> vocabulary. Two devices (or two sim devices). Your first `pld` program needs
> no prior distributed experience — just the [Getting
> Started](../00-getting_started.md) baseline.

**Suggested reading order:** **01** → 02 → 03 → 04 → 05 → 06 — this page is step 01.

## The idea

A distributed program runs the **same source** on every participating device,
but each device needs to know *which one it is* to act on its own slice of the
problem. That identity is the **rank**: a unique index assigned at launch time.

Rank identity flows through three levels. A `@pl.jit.host` function is the
**orchestrator** — it runs on the host CPU and is the only place that knows
about *all* devices. It loops over the world and dispatches a per-device
function once per rank with `device=r`. The per-device function (`@pl.jit`)
runs on the AICPU and forwards to an **InCore** kernel (`@pl.jit.incore`) that
runs on the NPU's AI cores. The rank index is passed down as an ordinary
argument.

## Run it

```bash
# Simulator (CI uses this):
python examples/distributed/01_hello_rank.py -p a2a3sim -d 0,1

# Two-card hardware:
python examples/distributed/01_hello_rank.py -p a2a3 -d 0,1
```

Expected output:

```text
OK
```

`OK` means the golden held: for every rank `r`, `y[r] == x[r] + r`.

## Walkthrough

The kernel — one concept at a time.

```python
N_RANKS = 2

@pl.jit.incore
def add_rank(
    x: pl.Tensor[[1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    rank: pl.Scalar[pl.INT32],
):
    tile = pl.load(x, [0, 0], [1, SIZE])
    rank_f32 = pl.cast(rank, target_type=pl.FP32)
    scaled = pl.mul(tile, rank_f32)
    result = pl.add(scaled, tile)
    y = pl.store(result, [0, 0], y)
    return y
```

- **Tensors first, scalars last.** The signature is `(x, y, rank)` — the scalar
  `rank` comes *after* the tensor arguments. Reversing that order fails at
  runtime with `TaskArgs: cannot add tensor after scalar`.
- **Scalars live on the AICPU.** `rank` arrives as an `INT32` scalar. The
  kernel casts it to `FP32` and folds it into *vector* operations (`x*rank +
  x`). Writing `rank_f32 + 1.0` as scalar arithmetic would be rejected by ptoas
  (`arith.addf explicitly marked illegal`).
- **Cast the parameter, not an expression.** `pl.cast(rank, ...)` on the
  `INT32` parameter is the supported path; casting an index-typed expression
  like `rank + 1` is not (`Cast between float and index types is not
  supported`).

The per-device wrapper and orchestrator:

```python
@pl.jit
def per_rank(x, y, rank):
    return add_rank(x, y, rank)

@pl.jit.host
def rank_program(
    x: pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32]],
):
    for r in pl.range(pld.world_size()):
        per_rank(x[r], y[r], r, device=r)
```

- `x` and `y` carry the **world shape** `[N_RANKS, 1, SIZE]` — rank `r`'s slice
  is `x[r]` / `y[r]`.
- The host loop runs `pld.world_size()` times — once per rank — and slices the
  world tensors per rank. `device=r` pins dispatch `r` to device `r`.
- The host function takes no `rank` argument; it *is* the loop. Rank identity
  is injected at dispatch time.

The harness:

```python
compiled = rank_program.compile(
    x, y,
    config=RunConfig(
        platform=args.platform,
        distributed_config=DistributedConfig(
            device_ids=[0, 1],
            num_sub_workers=0,
        ),
    ),
)
compiled(x, y, config=RunConfig(platform=args.platform))
assert torch.allclose(y, x + torch.arange(N_RANKS).view(N_RANKS, 1, 1), ...)
```

`DistributedConfig(device_ids=[0, 1], num_sub_workers=0)` declares the two
devices and no host sub-workers — the minimal multi-rank setup. The golden
`y == x + r` is checked **with a tolerance** (`allclose`) — the computation is
elementwise, so the tolerance is just headroom for backend floating-point
differences; use exact equality if you need a strict guarantee.

## Edge cases

> **Fatal pitfall — scalar after tensor.** A signature like
> `fn(x, rank, y)` compiles but fails at run time with `TaskArgs: cannot add
> tensor after scalar`. **Fix:** keep every scalar argument after every tensor
> argument: `fn(x, y, rank)`.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| `TaskArgs: cannot add tensor after scalar` | Scalar arg precedes a tensor arg in the child signature | Put all tensors first, scalars last |
| `arith.addf explicitly marked illegal` | Scalar `FP32` arithmetic on the AI core | Fold constants into vector ops (`x*rank + x`) |
| `Cast between float and index types is not supported` | `pl.cast` on an index-typed expression | Cast the `INT32` parameter first, then do float math as vector ops |
| Wrong result only on one rank's row | Rank index not used / wrong device mapping | Check the host loop uses `device=r` and slices `x[r]` |
| Program hangs at dispatch | Device ids not all available | Confirm `-d 0,1` ids exist and are free (`npu-smi info`) |

## See also

- [05-tutorials](05-tutorials.md) — the tutorial index (this step = row 01)
- [00-model](../distributed/00-model.md) — quickstart + model vocabulary
- [03-execution](../distributed/03-execution.md) — `DistributedConfig` and the
  worker lifecycle
- Next step: [07-programming_model](07-programming_model.md) — the three-level
  model, labeled
