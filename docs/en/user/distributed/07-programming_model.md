# The Three-Level Programming Model

Label the three levels of a distributed program — host orchestrator, per-device
orchestration, device kernel — and see which processor runs which.

> **Prerequisites:** [06-hello_rank](06-hello_rank.md). Two devices.

**Suggested reading order:** 01 → **02** → 03 → 04 → 05 → 06 — this page is step 02.

## The idea

A `pld` program is not one function but three, and the interesting part of the
model is *who runs where*:

| Level | Decorator | Runs on | Role |
| ----- | --------- | ------- | ---- |
| Host orchestrator | `@pl.jit.host` | Host CPU | Allocates window buffers, loops over ranks, dispatches |
| Per-device orchestration | `@pl.jit` | AICPU (per device) | One call per device; forwards args, returns results |
| Device kernel | `@pl.jit.incore` | NPU AI cores | The actual compute, on one device |

Step 01 used the same three levels implicitly; this step labels them and shows
how a value flows down the chain. The kernel computes `y[r] = x[r] * (r+1)` —
same shape of compute as before, but now each level is explicitly annotated and
explained in the code.

## Run it

```bash
python examples/distributed/02_programming_model.py -p a2a3sim -d 0,1
```

Expected output:

```text
OK
```

## Walkthrough

The three functions, top to bottom.

```python
@pl.jit.incore
def scale_by_rank(
    x: pl.Tensor[[1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    rank: pl.Scalar[pl.INT32],
):
    tile = pl.load(x, [0, 0], [1, SIZE])
    rank_f32 = pl.cast(rank, target_type=pl.FP32)
    scaled = pl.mul(tile, rank_f32)
    result = pl.add(scaled, tile)   # x * rank + x == x * (rank + 1)
    y = pl.store(result, [0, 0], y)
    return y
```

This is the **device kernel**. It runs on the AI cores of one device and sees
only that device's slice of the problem — `x` here is `[1, SIZE]`, the rank-`r`
slice. Note the same scalar discipline as step 01: `rank` is a scalar `INT32`
that is cast to `FP32` and folded into vector ops.

```python
@pl.jit
def per_rank(x, y, rank):
    return scale_by_rank(x, y, rank)
```

The **per-device orchestration** wrapper. It runs on the AICPU, one instance
per device, and exists so the host dispatches a device-level function rather
than poking the kernel directly. In this example it is a pass-through; in later
steps it is where per-device staging and multi-call sequences live.

```python
@pl.jit.host
def scale_program(
    x: pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32]],
):
    for r in pl.range(pld.world_size()):
        per_rank(x[r], y[r], r, device=r)
```

The **host orchestrator**. It runs on the host CPU, holds the world-shaped
tensors `[N_RANKS, 1, SIZE]`, and is the only function that knows about all
devices. It loops over the world and dispatches `per_rank` once per rank.

The compile/run shape is identical to step 01:

```python
compiled = scale_program.compile(
    x, y,
    config=RunConfig(
        platform=args.platform,
        distributed_config=DistributedConfig(device_ids=[0, 1], num_sub_workers=0),
    ),
)
compiled(x, y, config=RunConfig(platform=args.platform))
assert torch.allclose(y, x * torch.arange(1, N_RANKS + 1).view(N_RANKS, 1, 1), ...)
```

The golden `y == x * (r+1)` checks that each rank scaled *its own* slice by
its own index.

## Edge cases

> **Fatal pitfall — scalars as rank identity in the wrong level.** Only the
> host loop knows the rank index at dispatch time. Trying to read a rank inside
> the host function body (outside the loop) or relying on a module-global
> per-rank value gives every rank the same value. **Fix:** thread `r` down from
> the host loop through the wrapper into the kernel, as above.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| All ranks compute the same slice | Rank index not threaded from the host loop | Pass `r` down through `per_rank(..., r, ...)` |
| Host function has no rank argument | Confusing the orchestrator with the kernel | The host *is* the loop; identity comes from `device=r` |
| `@pl.jit.incore` never runs on host | Forgetting which decorator does what | Kernel = `@pl.jit.incore`; device wrapper = `@pl.jit`; orchestrator = `@pl.jit.host` |
| Value computed twice per rank | Dispatching the kernel, not the wrapper | Host must call the `@pl.jit` wrapper, not the `@pl.jit.incore` directly |

## See also

- [05-tutorials](05-tutorials.md) — the tutorial index (this step = row 02)
- [00-model](../distributed/00-model.md) — model vocabulary, L2 vs L3
- [03-execution](../distributed/03-execution.md) — worker lifecycle per level
- Next step: [08-window_buffer](08-window_buffer.md) — the memory substrate
