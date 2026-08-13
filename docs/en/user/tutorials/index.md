# Tutorials

Six walkthroughs, each ending in something that runs.

> **Prerequisites:** [The Language](../language/index.md) — at least
> [Types](../language/00-types.md), [Functions](../language/01-functions.md) and
> [Scopes and Placement](../language/04-scopes.md).

## How this chapter differs from the rest

[The Language](../language/index.md) and [Operations](../ops/index.md) are organised by
**capability** — one page per feature, so you can look a thing up. This chapter is
organised by **task**: each page builds one artefact from nothing, and every step along
the way is a program you can run.

That means the pages repeat each other's features on purpose. If you want the full surface
of `pl.at`, read [Scopes and Placement](../language/04-scopes.md). If you want to write a
matmul, read [Tiled matmul](02-matmul.md) — it will use `pl.at` without explaining all
of it.

## Two tracks

**Writing operators** (00–03) — how to express the computation.

| Page | You end up with | Reading time |
| ---- | --------------- | ------------ |
| [Your first operator](00-elementwise.md) | A running element-wise kernel, checked against torch | ~20 min |
| [Reduction and softmax](01-reduction-softmax.md) | A numerically stable softmax | ~30 min |
| [Tiled matmul](02-matmul.md) | A K-blocked matmul | ~40 min |
| [Mixed kernels](03-mixed-kernel.md) | Cube and vector working concurrently in one scope | ~40 min |

**Shaping the schedule** (04–05) — how to control what the runtime does with it.

| Page | You end up with | Reading time |
| ---- | --------------- | ------------ |
| [Shaping the task graph](04-task-graph.md) | A multi-task program whose dependency graph you control | ~30 min |
| [Tuning the schedule](05-scheduling-tuning.md) | A measurement loop you can re-run on your own kernel | ~40 min |

## Reading order

```text
00-elementwise ──► 01-reduction-softmax ──► 02-matmul ──► 03-mixed-kernel
      │                                                          │
      └──────────────────────────► 04-task-graph ──► 05-scheduling-tuning
```

The operator track is cumulative — each page assumes the tile vocabulary of the one
before. The scheduling track only needs `00`: it is about the shape of the graph between
kernels, not about what any one kernel computes.

## Which unit runs your operator

A core group pairs one **cube** unit (AIC) with **vector** units (AIV). Which one executes
an operator is not a choice you make per call — it follows from the operator:

| Operator family | Unit | Covered in |
| --------------- | ---- | ---------- |
| `matmul`, `matmul_acc`, `gemv` | Cube (AIC) | [Tiled matmul](02-matmul.md) |
| Element-wise, reduction, broadcast, cast | Vector (AIV) | [00](00-elementwise.md), [01](01-reduction-softmax.md) |
| `tpush_to_aiv`, `tpop_from_aic`, `aiv_shard`, `aic_gather` | Both, by construction | [Mixed kernels](03-mixed-kernel.md) |

A kernel built only from one family occupies one unit and leaves the other idle. That
observation is the whole point of [Mixed kernels](03-mixed-kernel.md); everything before it
writes single-unit kernels. See [Operations](../ops/01-catalog.md) for the full list.

## Running the examples

Every page names a file under `examples/`. `RunConfig.platform` defaults to `"a2a3sim"`, so
none of them needs a device:

```bash
python examples/beginner/02_elementwise.py
python examples/advanced/03_mixed_kernel.py --mode staged
```

Most of the companions hard-code that default. `03_mixed_kernel.py` is the exception: it
takes `--mode` to pick between the split forms, and `--platform` to retarget.

## See Also

- [The Language](../language/index.md) — the same features organised for lookup.
- [Tasks and Ordering](../tasks/index.md) — the reference behind [04](04-task-graph.md).
- [Operations](../ops/index.md) — the operator catalog.
