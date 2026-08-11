# Putting It Together: Broadcast + AllReduce + AllGather

Three collectives in one kernel — the capstone of the ladder, and the
bridge into real models.

> **Prerequisites:** [17-broadcast](17-broadcast.md) ·
> [18-allgather](18-allgather.md) · [20-all_to_all](20-all_to_all.md). Any
> number of devices ≥ 2 (the examples use 2 and 4 sim devices).

**Suggested reading order:** 01 → … → 15 → **16** — this page is step 16.

## The idea

Every earlier step taught one abstraction in isolation. This step is the
first place a kernel does *more than one* collective — and the first place the
byte count is not the point. Real models do exactly this: weights are
broadcast, activations are allreduced, results are allgathered. The kernel
below is the picotron `model.py` idea in miniature.

The pipeline:

1. **Broadcast** (step 12) — root's weights `w` reach every rank.
2. **Allreduce** (steps 08–11) — every rank ends with `Σ_k x[k]`.
3. **Allgather** (step 13) — every rank ends with `concat(x[0], …, x[P-1])`.
4. **Local compute** — the gathered matrix is scaled by the shared weight `w`
   (a learned per-feature weight over gathered hidden states).

## Run it

```bash
# Two ranks:
python examples/distributed/16_putting_it_together.py -p a2a3sim -d 0,1

# Four ranks — the same source, only -d changes:
python examples/distributed/16_putting_it_together.py -p a2a3sim -d 0,1,2,3
```

Expected output:

```text
OK
```

The golden checks **both** stages: `allred[r] == Σ_k x[k]` on every rank, and
`gathered[r] == concat(x[0], …, x[P-1]) * w` — the allgather result scaled by
the broadcast weight, which also proves the weight reached every rank.

## Walkthrough

The kernel is short — three builtin calls plus a local multiply — because the
ladder did the work:

```python
@pl.function(type=pl.FunctionType.InCore)
def compose_step(self, x, w_in, allred, gathered, w_data, ar_data, ag_data,
                 sig_bcast, sig_ar, sig_ag):
    ctx = pld.get_comm_ctx(w_data)
    my_rank = pld.rank(ctx)

    # 1 — Broadcast: root stages its weights, every rank gets them.
    if my_rank == ROOT_RANK:
        local_w = pl.load(w_in, [0, 0], [1, SIZE])
        w_data = pl.store(local_w, [0, 0], w_data)
    w_data = pld.tensor.broadcast(w_data, sig_bcast, root=ROOT_RANK)
    w = pl.load(w_data, [0, 0], [1, SIZE])

    # 2 — Allreduce: every rank ends with the element-wise sum.
    local_x = pl.load(x, [0, 0], [1, SIZE])
    ar_data = pl.store(local_x, [0, 0], ar_data)
    ar_data = pld.tensor.allreduce(ar_data, sig_ar, op=pld.ReduceOp.Sum, mode="mesh")
    total = pl.load(ar_data, [0, 0], [1, SIZE])
    allred = pl.store(total, [0, 0], allred)

    # 3 — Allgather: every rank ends with all ranks' raw slices.
    ag_data = pld.tensor.allgather(x, ag_data, sig_ag)

    # 4 — Local: scale the gathered matrix by the shared weight.
    for src in pl.range(nr):
        chunk = pl.load(ag_data, [src, 0], [1, SIZE])
        chunk = pl.mul(chunk, w)
        gathered = pl.store(chunk, [0, src * SIZE], gathered)
    return gathered
```

- **One fresh signal per collective.** Signal counters are monotonic and do
  not self-reset, so three back-to-back collectives get three separate
  `[nr, 1]` windows (`sig_bcast`, `sig_ar`, `sig_ag`). Reusing one signal
  across the calls would let the second wait pass early.
- **`mode="mesh"` is explicit** for the allreduce — the step-11 reveal made
  the mode a choice; here it is named so the reader sees the full call.
- **The allgather source is the plain `x` tensor** (step 13's rule), while
  broadcast and allreduce take windows — the three calls show the full
  surface of the `pld.tensor.*` API in one place.
- **The local step is where the collective meets the math.** `chunk * w` is
  an ordinary `pl.mul` on the gathered tile — the same vector op from step 01,
  now acting on data that came from three different ranks.

### The IR diff (the teaching artifact)

The lowered IR is the three hand-rolled schedules you already know, in order:
the broadcast's root-stage + barrier + read (step 12), the allreduce's mesh
barrier + accumulate (step 08), and the allgather's stage + barrier +
per-peer reads (step 13). The three builtins compose because each one's
lowering is self-contained on its own signal window — which is *why* the fresh
signal rule matters.

**Cost card (per rank):** the sum of the pieces — `(P-1)·N` for the
broadcast, `2·(P-1)/P·N` for the mesh allreduce, `(P-1)/P·N` for the
allgather. For the first time, the byte count is not the point: the point is
that three schedules compose into one kernel.

## Edge cases

> **Fatal pitfall — reusing one signal across collectives.** Signal counters
> are monotonic and never reset, so a second collective on the same `[nr, 1]`
> window sees counters that already satisfy `Ge(1)` and its wait passes
> immediately — a silently missing barrier. **Fix:** one fresh signal window
> per collective, as the kernel does.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| Second/third collective passes early | Signal reused across calls | One `[nr, 1]` signal per collective |
| `allred` wrong but `gathered` right | Allreduce source not staged / wrong op | Stage `x` into `ar_data`; `op=Sum`; `mode="mesh"` |
| `gathered` wrong but `allred` right | Broadcast weight not applied, or wrong row | `chunk = pl.mul(chunk, w)` for every row |
| `pld.tensor.allgather` source rejected | Tile passed instead of a tensor | Pass the plain `x` tensor |
| Non-root weights leak into output | Root staging missing | Stage `w_data` only under `if my_rank == ROOT_RANK` |

## See also

- [05-tutorials](05-tutorials.md) — the tutorial index (this step = row 16)
- [01-collectives](../distributed/01-collectives.md) — the whole collective zoo
- [17-broadcast](17-broadcast.md) / [18-allgather](18-allgather.md) /
  [20-all_to_all](20-all_to_all.md) — the pieces this kernel composes
- More advanced applications (not restated here): pypto-lib
  [#869](https://github.com/hw-native-sys/pypto-lib/pull/869) (AllGather-GEMM)
  and the DeepSeek-V4 distributed MoE dispatch/combine — the same patterns at
  model scale
- This is the end of the ladder — the index lists everything, in order.
