# Dynamic Rank Count: One Source, Any P

Make the ring shift from step 06 **rank-count-agnostic**: `NR = pl.dynamic("NR")`
names the world size as a runtime dimension, so the same source compiles and
runs at P=2, P=3, P=4, … — change only `-d`, never the program.

> **Prerequisites:** [11-put_get](11-put_get.md). Any number of devices ≥ 2
> (the examples here use 2, 3 and 4 sim devices).

**Suggested reading order:** 01 → 02 → 03 → 04 → 05 → 06 → **07** — this page is step 07.

## The idea

Every step so far hardcoded `N_RANKS = 2`: the host world tensor was
`[N_RANKS, 1, SIZE]` and the golden was written for two ranks. The kernels,
though, never actually depended on the count — they read it at runtime from
`pld.nranks(ctx)`, looped over it, and computed peers with `% nranks`. Only the
host's world shape was pinned.

`pl.dynamic("NR")` unpins it. `NR` is a **named runtime dimension**: it tells
the compiler "this extent is resolved when the program is invoked, not when it
is written". The host signature becomes `x: pl.Tensor[[NR, 1, SIZE], pl.FP32]`,
and the very same source now compiles for whatever `-d` you pass — the rank
count is gone from the program.

Why this matters now: the later steps compare collective algorithms against
each other, and at two ranks several of those algorithms collapse into the
same exchange — their differences are only observable at four ranks. This
step is the bridge: the same source serving any world size is what those P=4
comparisons build on, with no per-program rank-count edits.

## Run it

```bash
# Two, three, or four ranks — same source, only -d changes:
python examples/distributed/07_dynamic_rank_count.py -p a2a3sim -d 0,1
python examples/distributed/07_dynamic_rank_count.py -p a2a3sim -d 0,1,2
python examples/distributed/07_dynamic_rank_count.py -p a2a3sim -d 0,1,2,3
```

Expected output:

```text
OK
```

## Walkthrough

The only differences from step 06 are the `NR` declaration and the host
signature; the kernels are untouched.

```python
SIZE = 64
NR = pl.dynamic("NR")          # the rank count is a runtime dimension
```

```python
@pl.jit.host
def ring_get(
    x: pl.Tensor[[NR, 1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[NR, 1, SIZE], pl.FP32]],
):
    src_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    dst_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    signal_buf = pld.alloc_window_buffer([1, 1], dtype=pl.INT32)
    for r in pl.range(pld.world_size()):
        src = pld.window(src_buf, [1, SIZE], dtype=pl.FP32)
        dst = pld.window(dst_buf, [1, SIZE], dtype=pl.FP32)
        signal = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
        per_rank_get(x[r], y[r], src, dst, signal, device=r)
```

- **`NR` is symbolic.** `pl.dynamic("NR")` declares that the leading world dim
  is resolved at runtime. `pld.world_size()` already *returns* the runtime
  count; `NR` is the shape-side name for it.
- **The kernels stay as they were.** `get_step` (and `put_step`) bound loops by
  `pld.nranks(ctx)` and compute `peer = (my_rank ± 1) % nranks` — all runtime.
  There is no `N_RANKS` anywhere in the program.
- **`-d` is the only knob.** `main()` takes the device count from
  `-d` (`len(device_ids)`), shapes the world tensors `(P, 1, SIZE)` from it,
  and derives the golden from the actual `P`:

```python
device_ids = [int(d) for d in args.device.split(",")]
x = torch.randn((len(device_ids), 1, SIZE), dtype=torch.float32)
...
expected = expected_ring(x, get_mode)   # y[r] = x[(r+1) % P] (get), x[(r-1) % P] (put)
```

Compile once per invocation (each run's `-d` fixes the concrete `P`), and the
same source serves every world size — this is what the P=4 collective
comparisons build on.

**Cost card:** identical to step 06 — one step, one slice of `SIZE` bytes
exchanged with one peer per rank. The rank count changes *where* the ring
wraps, not the per-rank cost.

## Edge cases

> **Fatal pitfall — leaving a hardcoded rank count in the host shape.**
> `NR = pl.dynamic("NR")` must be in the host annotation. If the world shape
> still says `[N_RANKS, 1, SIZE]` with `N_RANKS = 2`, the program is pinned to
> two ranks again — and a larger `-d` fails with a shape mismatch at compile
> time. **Fix:** replace the constant with `NR` everywhere in the host
> signature.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| Shape mismatch at compile time when P grows | A runtime dim used without `pl.dynamic` | Wrap it: `NR = pl.dynamic("NR")`, use `[NR, ...]` in the host signature |
| Wrong result only at P > 2 | Peer arithmetic with a negative dividend (`(my_rank - 1) % nranks`) | Use `(my_rank + nranks - 1) % nranks` — never negative |
| `get` reads stale data at P > 2 | Handshake targets the wrong rank | Notify the rank that reads you (previous); wait for the rank you read (next) |
| Golden is one step behind | Pull vs push confusion | `get` mode: `y[r] = x[(r+1) % P]`; `put` mode: `y[r] = x[(r-1) % P]` |
| Recompiling for every P feels wasteful | The compiled artifact is P-specific | Re-run with the new `-d`; the *source* never changes |

## See also

- [05-tutorials](05-tutorials.md) — the tutorial index (this step = row 07)
- [11-put_get](11-put_get.md) — the fixed-P=2 version of this ring (step 06)
- [02-primitives](../distributed/02-primitives.md) §System Substrate + §Put and Get — `world_size`/`nranks`, chunking
- [00-getting_started](../00-getting_started.md) — `pl.dynamic(...)` dynamic dims
- Next step: [05-tutorials](05-tutorials.md) — steps 08–16 (the collectives, at P=4) are planned
