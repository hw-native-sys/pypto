# AllToAll: A Different Slice for Every Peer

Every rank sends a *distinct* slice to every peer and receives a distinct
slice from every peer — the most general of the point-to-point patterns — then
the builtin does it in one call.

> **Prerequisites:** [19-reduce_scatter](19-reduce_scatter.md). Any number of
> devices ≥ 2 (the examples use 2 and 4 sim devices).

**Suggested reading order:** 01 → … → 14 → **15** — this page is step 15.

## The idea

All the previous collectives move the *same* shape everywhere: broadcast moves
one slice to all, allgather moves each rank's slice to all, reduce-scatter
reduces. All-to-all is different: rank `r` sends a **different `N/P` slice to
every peer** — the slice for destination `d` is not the slice for destination
`e`. This is the personalized exchange.

| Aspect | AllToAll |
| ------ | -------- |
| Data in | `P` distinct chunks per rank (one per destination) |
| Data out | `P` distinct chunks per rank (one from each source) |
| Pattern | Push each dest chunk (`put`) → barrier → read back |
| Cost | `(P-1)/P · N` bytes received; no two ranks want the same bytes |

This is the pattern behind real dispatch/combine workloads — distributed MoE
sends each token-group to a different expert's rank; AllGather-GEMM pipelines
route per-shard data. The walkthrough's See-also links those applications by
name (without restating them).

## Run it

```bash
# Hand-rolled: put each dest chunk, barrier, read back.
python examples/distributed/15_all_to_all.py -p a2a3sim -d 0,1

# Reveal: pld.tensor.all_to_all in one call.
python examples/distributed/15_all_to_all.py -p a2a3sim -d 0,1 --mode builtin

# The same source at P=4:
python examples/distributed/15_all_to_all.py -p a2a3sim -d 0,1,2,3
python examples/distributed/15_all_to_all.py -p a2a3sim -d 0,1,2,3 --mode builtin
```

Expected output:

```text
OK
```

The golden is constructed so each chunk is unique: `input[r, d, j] =
r*1000 + d*100 + j` — the source, destination, and element are all encoded in
the value. Any routing mistake (wrong source, wrong destination) shows up as
a wrong value, not a subtle shape issue.

## Walkthrough

Both modes share one `[nr, SIZE]` window. Rank `r`'s chunk-for-`d` ends up at
row `r` of destination `d`'s window, then rank `d` reads row `src` of its
own window — in the hand-rolled kernel the caller does the write explicitly;
in the reveal below, the push happens inside the call instead. The
hand-rolled kernel uses step 06's `put`:

```python
@pl.function(type=pl.FunctionType.InCore)
def hand_step(self, x, y, data, signal):
    ctx = pld.get_comm_ctx(data)
    my_rank = pld.rank(ctx)
    nranks = pld.nranks(ctx)

    # Phase 1 — push: write chunk-for-dest into dest's window at our row.
    for dest in pl.range(nranks):
        pld.tensor.put(data, dest, x, [my_rank, 0], [dest, 0], [1, SIZE])

    # Phase 2 — barrier: notify every peer, wait on every peer slot.
    for peer in pl.range(nranks):
        if peer != my_rank:
            pld.system.notify(signal, peer=peer, offsets=[my_rank, 0],
                              value=1, op=pld.NotifyOp.Set)
    for src in pl.range(nranks):
        if src != my_rank:
            pld.system.wait(signal, offsets=[src, 0], expected=1,
                            cmp=pld.WaitCmp.Ge)

    # Phase 3 — read-back: row src of our window holds src's chunk for us.
    for src in pl.range(nranks):
        chunk = pl.load(data, [src, 0], [1, SIZE])
        y = pl.store(chunk, [src, 0], y)
    return y
```

- **The push is a loop over destinations.** `pld.tensor.put(data, dest, x,
  [my_rank, 0], [dest, 0], [1, SIZE])` writes `x`'s row `dest` into peer
  `dest`'s window at *our* row. Each iteration targets a different peer with a
  different source row — the "personalized" part.
- **`Set` not `AtomicAdd` for the notify.** Each rank is the only writer of
  its row (`[my_rank, 0]`), so the barrier here uses `Set`/`Ge(1)` — the
  single-writer form from step 04's walkthrough.
- **Read-back completes the exchange.** After the barrier, row `src` of our
  own window holds the chunk rank `src` intended for us.

The reveal replaces phases 1–2 with one call; phase 3's read-back loop
remains in the caller:

```python
    result = pld.tensor.all_to_all(x, data, signal)
    for src in pl.range(nranks):
        chunk = pl.load(result, [src, 0], [1, SIZE])
        y = pl.store(chunk, [src, 0], y)
    return y
```

- **The source is your local `x` (a plain `pl.Tensor`)** with row `d` = the
  chunk for destination `d` — the same layout the hand-rolled loop used.
- **The window becomes the result** (row `src` = the chunk from rank `src`),
  and `input`/`target` must be **separate** buffers.

### The IR diff (the teaching artifact)

The interesting part isn't which bytes move — both modes move the same chunks
to the same destinations — it's how much of the sequence the builtin writes
for you, and where it stops.

- `--mode hand` lowers to the three phases above: `P` puts (one per
  destination), then the `Set`/`Ge(1)` barrier, then the read-back loop — the
  barrier comes *after* your own pushes but *before* you trust any peer's,
  because your reads would otherwise race their pushes.
- `--mode builtin` expands into the same shape — your per-destination chunks
  pushed via `pld.tile.put`, then the barrier — but it stops there. The
  barrier sits on the same side of the transfer as in the hand-rolled version,
  so it means the same thing: a **completion** barrier confirming that every
  peer's push has landed. What the builtin does *not* do is read the rows back
  for you — the window **is** the result, so the read-back loop stays in your
  code.
- The composite appends a **self-clearing epilogue** after the barrier that
  subtracts this call's credits back out, so the same signal is reusable by a
  later collective ([21-putting_it_together](21-putting_it_together.md)). The
  HOST builtin also chunks large transfers, but inside the generated AIV
  kernel, in fixed-size tiles — not at the orchestration level.

**Cost card (per rank):** every rank sends a *different* `N/P` slice to each
peer — `(P-1)/P · N` bytes received, and no two ranks receive the same bytes.
This is why all-to-all is the most expensive point-to-point pattern to
schedule: every pair exchanges distinct data.

## Edge cases

> **Fatal pitfall — reusing one window for source and result.** All-to-all
> delivers its *result* in place — `target` is rebound as the window-as-result
> — but the *source* is a separate buffer: the API requires `input` and
> `target` to be different — and on the InCore path `x` is a plain
> `pl.Tensor` while `data` is the window, so they are two separate buffers of
> two different kinds, not one window used twice.
> Passing the same window as both lets the push overwrite chunks you still
> need to send. **Fix:** allocate a distinct target window (as the example
> does) — the builtin enforces this too.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| Chunk for dest `d` lands in the wrong peer | `put` peer ≠ destination row | `put(data, dest, x, ..., [dest, 0], ...)` |
| Output has my own chunks, not peers' | Read-back from own rows | Read row `src` after the barrier |
| Source/target aliasing corruption | Same buffer as input and target | Separate `x` and `data` windows |
| One rank hangs | `Set` notify/wait mismatch | Notify writes row `my_rank` of peer; wait reads row `src` of self |
| Wrong values at P=4 only | Missing barrier before read-back | Barrier between the put loop and the read loop |

## See also

- [05-tutorials](05-tutorials.md) — the tutorial index (this step = row 15)
- [01-collectives](../distributed/01-collectives.md) §AllToAll — the full API
- [03-execution](../distributed/03-execution.md) — `DistributedWorker` and
  device-side staging for production all-to-all
- `examples/runtime/distributed_callback.py` — host-side runtime-bound
  callbacks around an L3 distributed program
- [11-put_get](11-put_get.md) — the `put`/`get` substrate this step builds on
- More advanced applications (not restated here): pypto-lib
  [#869](https://github.com/hw-native-sys/pypto-lib/pull/869) (AllGather-GEMM,
  an allgather pattern from step 13) and the DeepSeek-V4 distributed MoE
  dispatch/combine (an all-to-all pattern from this step)
- [04-debugging](04-debugging.md) — the canonical failure catalog for distributed programs
- Next step: [21-putting_it_together](21-putting_it_together.md) — compose
  three collectives in one kernel

## Indexed AllToAllV on A5 URMA

```python
target = pld.tensor.indexed_all_to_all_v(
    input, target, signal, send_counts, recv_counts, index, core_num=2,
)
```

The indexed variant reads original `input[Nmax,H]` rows directly, without a
packed hidden send buffer. `index` is contiguous INT32 `[P,C]`, and
`target` is a distinct receive window `[P*C,H]`. At destination `d`, source
`s` writes `input_s[index_s[d,j]]` into `target_d[s*C+j]` for
`j < send_counts_s[d]`. Repeated indices preserve repeated output slots.
Unwritten tails retain their old contents; consume only `recv_counts[s,0]`
rows of each source partition. MoE token-to-rank deduplication and expert
expansion remain application operations.

This API supports **A5 URMA with PER_PEER queues, L2 only**, with FP32 or
INT8 payloads. L2 accepts
local Tensor input/index and a local INT32 recv_counts `[P,1]` output.
send_counts must be window-bound because peers read it. `signal`, `send_counts`
and `recv_counts` must use pairwise distinct buffers. All views must be packed
and full; shapes are static. HOST, InCore, remote MTE,
and SHARED_POOL are unsupported.

`core_num=B` is a positive exact launch width, subject to the runtime's AIV
sync-start limit. For `B <= P`, AIV `b` handles destinations `b,b+B,...`.
For `B > P`, only AIVs `0..P-1` submit URMA work, one writer per remote rank.
All B AIVs divide local receive rows evenly. Each AIV submits its remote
transfers, copies its local rows, then waits for the final remote completions. A hidden row is never split
between AIVs. All B AIVs participate in the barriers, including empty local
workers; signal is INT32 `[P,S]` with `S >= B`, initially zero. L2 requires a
compile-time core count. All ranks must agree
on geometry, launch width and collective order.

Counts must be in `[0,C]`, and each valid index must be in `[0,Nmax)`.
These are device-data preconditions; invalid values abort execution rather
than being clamped into a different dispatch. Index tails are not read and
may contain `-1`. A distributed run must be terminated as a group on failure.

The kernel builds multi-SGE WRITE SQEs with SIMT: up to twelve complete rows
per SQE, a fixed four-WQEBB slot, and NOPs in unused trailing WQEBBs. Each remote
peer's entire transfer must fit its SQ: `ceil(send_counts[peer] / 12) * 4 <= sqDepth`;
the kernel asserts this before posting. It rings a doorbell after each segment of
at most 64 SQEs without waiting between segments. Only the transfer's final WRITE
requests a strong-order fenced CQE, which is drained once after all segments are
posted. SQ head counts WQEBBs; ISA submittedWqeCount increases once per nonempty
remote peer transfer. All writes complete before the final barrier.
Self uses direct TLOAD/TSTORE with an eight-stage ring of 8 KiB UB slots
(64 KiB per local worker). Consecutive source indices are coalesced; scattered
rows use the same pipeline across tokens. MTE2 waits only before reusing a slot,
allowing loads of later tokens to overlap earlier MTE3 stores; all slots drain
once at the end before publishing completion. It does not use synchronous TPUT
per token. Remote payloads go directly through URMA. The runtime does not
initialize a self-rank SQ/CQ for URMA loopback.

This is a completion-tracked collective task, not an issue-only asynchronous
handle. Producers of input/index/counts must precede it; consumers of target
and recv_counts must follow it. Keep source buffers alive until task completion.
Window reuse must also follow the previous local consumer. Callers must prevent
concurrent exchanges from using the same URMA queues. Reusing the same signal
creates an InOut dependency between calls, even with distinct payload windows;
using independent signals requires an explicit task dependency. This API does
not insert a global queue-ordering dependency.

Synchronization uses SIMT global atomics for both notifications and polling,
with backoff, rather than the scalar TNOTIFY/TWAIT implementation. The same
signal lanes retain the two-round credit protocol. Receive counts use
non-cacheable scalar stores, avoiding whole-cache-line writeback when control
windows share a cache line. Persistent reuse with changed input is covered by
the A5 system test; no full-payload cache sweep is added.
