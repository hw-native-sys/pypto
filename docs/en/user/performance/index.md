# Performance

Tuning a single-card kernel: making the execution visible first, then working through the
places time actually goes.

> **Prerequisites:** [Tasks and Ordering](../tasks/index.md) and
> [Scopes and Placement](../language/04-scopes.md).

## The shape of the problem

A PyPTO kernel's wall-clock time is spent in three different machines, and they fail in
different ways:

```text
host          orchestration            AICore
 │                  │                     │
 ├─ copies          ├─ task dispatch      ├─ the kernel itself
 │  (06-host)       │  (01, 02, 03)       │  (04-incore)
 │                  │                     │
 └────────────── memory: on-chip buffers + runtime rings (05-memory) ──────┘
```

Most first-time tuning goes straight to the third column — the arithmetic inside the
kernel — and finds that the first two were the problem. The chapter is ordered so you meet
them in the order they usually bite.

## Contents

| Page | Covers |
| ---- | ------ |
| [Reading the swimlane](00-swimlane.md) | Capturing the L2 swimlane and opening it — the one view that shows where time went |
| [Task granularity](01-task-granularity.md) | Dispatch is not free; growing and merging InCore functions without starving the cores |
| [Runtime overhead](02-runtime-overhead.md) | Mixed kernels, SPMD, `allow_early_resolve`, in-kernel `syncall` |
| [Managing dependencies](03-dependencies.md) | Why the runtime serializes work that could overlap, and how to say otherwise |
| [Tuning the InCore function](04-incore.md) | Double buffering, algorithmic splits, L0 instruction traces, hardware granularity, external kernels |
| [Memory](05-memory.md) | The four scope-depth rings, scope placement, and ring sizing |
| [Host](06-host.md) | Keeping resident data resident |
| [Multi-card measurement](07-distributed.md) | Beyond one card: which rank, and which collective |

## How to use it

Every technique below is written with the same four fields, because a speedup with an
unstated cost is not a result:

| Field | What it answers |
| ----- | --------------- |
| **When it applies** | The symptom that makes this the right move |
| **How** | The code change |
| **Cost** | What it spends — memory, generality, or a correctness obligation you now own |
| **How to confirm** | The artifact that shows it worked, and what should change in it |

**There are no speedup numbers in this chapter.** They depend on your shapes, platform, and
toolchain versions, and a stale number is worse than none because you cannot tell it has
gone stale. The confirmation step is the transferable part: run it on your kernel and you
have your own number.

## Before you tune anything

Two cheap checks come before any of this, and both are already done for you:

- **`report/perf_hints.log`** in the build output. The compiler writes what it noticed
  during compilation — undersized transfers, matmuls it could not tile, a pipeline depth
  that did not fit. One summary line goes to stderr on every compile.
- **The benchmark tree.** `run()` reports host and device time separately. If the time is
  on the host, nothing in pages 00–05 will move it — go to [Host](06-host.md).

## See also

- [Tuning the schedule](../tutorials/05-scheduling-tuning.md) — the same ground as a
  hands-on walkthrough.
- [Precision](../precision/index.md) — the equivalent treatment for "the result is wrong".
