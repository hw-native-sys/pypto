# Distributed Performance

What changes when there is more than one rank: a new dominant cost, a new way to be
serialized, and metrics that need reading differently.

> **Prerequisites:** [Reading the swimlane](00-swimlane.md) and
> [Distributed Programming](../distributed/index.md). Everything in
> everything in pages [01](01-task-granularity.md)–[06](06-host.md) still applies per
> rank — this page is only
> what is *additional*.

## The three things that are different

1. **Communication is a cost you did not have.** It is often the largest single term, and
   the loop's step 2 will show it as device time that no kernel change touches.
2. **The slowest rank sets the pace.** A rank that starts late delays every collective it
   participates in, so the interesting number is spread across ranks, not the mean.
3. **One mean hides both.** Which is why the benchmark exposes three different groupings.

## Reading the metrics

`BenchmarkStats` groups the same dispatches three ways. Picking the wrong one is how skew
gets missed:

| Method | Returns | Answers |
| ------ | ------- | ------- |
| `per_round(metric)` | One value per round | Did the whole system get faster? |
| `per_rank(metric)` | Per rank → list of values | Is one rank consistently slower? |
| `per_dispatch(metric)` | `{(pid, slot): [round0, ...]}` | Which dispatch slot was the outlier? |

Each takes `metric` = `"device"`, `"host"`, or `"effective"`:

- **`device`** — device wall time for the dispatch.
- **`host`** — host wall time. Compare against `device` to find dispatch overhead.
- **`effective`** — the **union** of the device-domain `orch` and `sched` spans, i.e. the
  runtime's "Effective" figure. Both spans share the invocation's device-clock origin, so
  the union is meaningful within a dispatch. It returns `0.0` on `*sim` platforms and
  non-profiling builds — a zero here means "not collected", not "instant".

`per_dispatch` keys on the dispatch *slot* within a round, not the round — so a rank's
repeated or heterogeneous dispatches stay separate. Pair it with `dispatch_tasks()` to
label the slots. It is **L3 only**: on L2 it returns `{}`.

**Start with `per_rank`.** If the ranks are tight, `per_round` is the number to optimise. If
they are not, no collective tuning matters until the skew is understood — you would be
tuning the wait, not the work.

## Collective algorithm: mesh vs ring

The collectives take `mode=`, defaulting to `"mesh"`:

| `mode=` | Algorithm | Windows |
| ------- | --------- | ------- |
| `"mesh"` (default) | Direct exchange | **O(P)** |
| `"ring"` | Chunked reduce-scatter + allgather | **O(1)** |

That window count is the trade. `"mesh"` exchanges directly and needs a window per peer, so
its buffer footprint grows with the rank count. `"ring"` runs a `2(P−1)`-step schedule and
needs a constant number of windows, paying in steps what it saves in memory.

- **When `"ring"` applies:** many ranks, or window memory is the constraint.
- **Cost:** `2(P−1)` sequential steps — more latency-sensitive, and worse for small payloads
  where the step count dominates the bytes moved.
- **How to enable:** `mode="ring"` at the collective — but that one argument is never the
  whole edit. Signal synthesis (what makes the host form's omitted `signal` work) is
  **mesh only**, and `LowerHostTensorCollectives` additionally requires, for `NR` ranks:
  - an explicit window-bound **INT32** signal of rank 2, shaped `[2*(NR-1) + 1, NR]`;
  - a **statically-known** `src` shape — dynamic host-ring extents are rejected;
  - `numel(src)` an exact multiple of `NR`, since the schedule cuts it into `NR` chunks.

  Each is a compile-time error with a message naming the constraint, so a wrong switch
  fails loudly rather than silently. See [Collectives](../distributed/01-collectives.md).
- **How to confirm:** `per_round("device")` for the total, `per_rank` to check the change
  did not just move the cost onto one rank.

## Overlapping communication with compute

The general shape: the collective is a task like any other, so the graph decides whether
anything runs beside it. If a rank has independent compute available, it should not be
idle inside a collective.

- **How to confirm:** the dependency graph
  ([`enable_dep_gen`](../tutorials/05-scheduling-tuning.md)) shows whether the compute is a
  sibling of the collective or a descendant. A descendant cannot overlap it, however the
  collective is tuned.

## Paying setup once, across ranks

`DistributedWorker` is documented as "prepare once, dispatch many" — it holds an initialized
level-3 worker plus every setup artifact. Two consequences for performance:

- **Reuse the worker.** Several programs registered against one `DistributedWorker` share
  the setup, so only the first run pays it. `examples/runtime/multi_program_kv_cache.py`
  shows the shape.
- **Keep sharded data resident.** `alloc_stacked_tensor(worker_ids=...)` places a shard per
  worker and keeps it there, removing a per-launch scatter of weights that do not change.

*Confirm both in the **host** spans.* Setup and residency never move device time, so a
measurement that only looks at `device` will report no improvement from either.

## Order of attack

1. **`per_rank` first.** Skew invalidates every other number.
2. **Then the graph** — is compute allowed beside the collective at all?
3. **Then the algorithm** — `mesh` vs `ring`, chosen against rank count and window memory.
4. **Then setup and residency** — visible only in host spans.
5. **Then per-rank kernel work** — pages [01](01-task-granularity.md)–[06](06-host.md).

Step 1 before step 3 is the one that saves time. Tuning a collective while one rank arrives
late optimises a queue nobody is waiting in.

## Edge Cases

| Symptom | Likely cause | Where to look |
| ------- | ------------ | ------------- |
| **`effective` is `0.0` everywhere** | `*sim` platform or a non-profiling build | Expected — use `device` |
| **`per_round` improved, users saw nothing** | The cost moved onto one rank | `per_rank` |
| **`ring` slower than `mesh`** | Few ranks, or small payloads — the `2(P−1)` steps dominate | Rank count and payload size |
| **Collective never overlaps compute** | The compute is a descendant of the collective in the graph | `enable_dep_gen` |
| **Adding ranks makes memory fail** | `mesh` needs O(P) windows | Switch to `mode="ring"` |

## See Also

- [Reading the swimlane](00-swimlane.md) — per-task timing, shared with single-card.
- [Task granularity](01-task-granularity.md) onwards — still applies, per rank.
- [Distributed Programming](../distributed/index.md) — the API surface itself.
- [Collectives](../distributed/01-collectives.md) — semantics of each collective.
