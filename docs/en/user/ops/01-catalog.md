# Catalog

Every operator family, one line each. Signatures live in the docstrings — see
[Operations](index.md) for why this page does not repeat them.

> **Reading the tables:** the **Reach** column gives the shortest spelling that works.
> `pl.` means the name is available unqualified; `pl.tile.` / `pl.tensor.` mean the
> operator is level-specific. Names marked **(t)** are tile-only operators re-exported at
> top level for convenience — `pl.load` *is* `pl.tile.load`, not a dispatcher.

## Creation

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| `create_tensor` | `pl.` | Allocate a DDR tensor. `manual_dep=True` opts it out of dependency tracking for its lifetime |
| `create_tile` | `pl.` (t) | Allocate an on-chip buffer |
| `create_l1` | `pl.` (t) | Allocate in L1 explicitly |
| `full` | `pl.` | A tensor filled with a constant |
| `arange` | `pl.` | Consecutive integers (`tensor.ci`) |
| `random` | `pl.` | Random-filled tensor |
| `tri` | `pl.` (t) | Lower- or upper-triangular mask tile; `upper=` picks the side, `diagonal=` shifts it, and only the valid region is written |
| `const` | `pl.` | A literal with an explicit dtype — see [Directives](../language/05-directives.md#typed-constants) |
| `array.create` | `pl.array.` | Allocate an on-core array |

## Data movement

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| `load` | `pl.` (t) | DDR → `Vec` (default) or `Mat`, via `target_memory=` |
| `store` | `pl.` (t) | On-chip → DDR |
| `move` | `pl.` (t) | On-chip → on-chip; the only path into `Left` / `Right` / `Bias` |
| `reserve_buffer` | `pl.` | Reserve a cross-core buffer |
| `import_peer_buffer` | `pl.` | Reference a peer core's buffer |

See [Memory and Data Movement](../language/03-memory.md) for which moves are legal.

## Elementwise arithmetic

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| `add` `sub` `mul` `div` | `pl.` | Binary arithmetic; a Python number on the right selects the scalar-operand form |
| `neg` `abs` `recip` | `pl.` | Unary negate, absolute value, reciprocal |
| `rem` `rems` `fmod` `fmods` | `pl.` | Remainder and floating-point modulo, tensor and scalar forms |
| `addc` `subc` `addsc` `subsc` | `pl.` (t) | Three-input add / subtract with carry operand |
| `part_add` `part_mul` `part_max` `part_min` | `pl.` | Partial (segmented) arithmetic |

## Math

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| `exp` `log` | `pl.` | Exponential, natural logarithm |
| `sqrt` `rsqrt` | `pl.` | Square root; reciprocal square root. `high_precision=` is tensor-only and **raises** on a Tile — at tile level precision is selected by passing the scratch tile to `pl.tile.rsqrt(src, tmp)` |
| `sin` `cos` | `pl.` | Trigonometric |

## Comparison and selection

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| `cmp` `cmps` | `pl.` | Compare two operands / operand against a scalar |
| `maximum` `minimum` | `pl.` | Elementwise max / min of two operands |
| `maximums` `minimums` | `pl.` (t) | Elementwise max / min against a scalar |
| `max` `min` | `pl.` (t) | Scalar max / min of two values — **not** a tile reduction. To reduce a tile use `row_max` / `col_max` (and the `min` forms) |
| `sel` `sels` | `pl.` (t) | Select by mask, tensor and scalar forms |

## Activations

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| `relu` | `pl.` (t) | Rectified linear |
| `prelu` `lrelu` | `pl.` (t) | Parametric / leaky rectified linear |

## Bitwise

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| `and_` `or_` `xor` `not_` | `pl.` | Bitwise logic |
| `ands` `ors` `xors` | `pl.` | Bitwise logic against a scalar |
| `shl` `shr` | `pl.` | Shift left / right |
| `shls` `shrs` | `pl.` | Shift by a scalar amount |

## Reductions

Row reductions collapse the last axis; column reductions collapse the first.

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| `row_sum` `row_prod` `row_max` `row_min` | `pl.` | Reduce along a row |
| `col_sum` `col_prod` `col_max` `col_min` | `pl.` | Reduce along a column |
| `row_argmax` `row_argmin` | `pl.` | Index of the row extremum |
| `col_argmax` `col_argmin` | `pl.` | Index of the column extremum |

Several reductions accept a `tmp_tile` argument. Passing one changes the reduction
strategy (binary tree versus sequential), which changes floating-point association — the
results differ within tolerance rather than being wrong. Reductions over a partially valid
tile depend on the pad value; see
[Memory § valid shape](../language/03-memory.md#valid-shape-and-padding).

## Broadcast and expand

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| `row_expand` `col_expand` | `pl.` | Broadcast a reduced axis back to full width |
| `row_expand_add` `row_expand_sub` `row_expand_mul` | `pl.` | Broadcast fused with an arithmetic op |
| `row_expand_div` `row_expand_max` `row_expand_min` | `pl.` | Broadcast fused with division, max, or min |
| `col_expand_add` `col_expand_sub` `col_expand_mul` | `pl.` | Column-wise equivalents |
| `col_expand_div` `col_expand_max` `col_expand_min` | `pl.` | Column-wise division, max, and min |
| `row_expand_expdif` `col_expand_expdif` | `pl.` | Broadcast fused with `exp(x - m)` — the softmax kernel |
| `expand_clone` `expands` | `pl.` | Broadcast a value across a shape |
| `fillpad` `fillpad_expand` | `pl.` | Fill the invalid region; optionally broadcast in the same step |

## Shape and layout

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| `reshape` | `pl.` | Reinterpret dimensions |
| `transpose` | `pl.` | Transpose |
| `slice` | `pl.` | Sub-region; also written `A[0:16, :]` |
| `concat` | `pl.` | Join along an axis |
| `assemble` | `pl.` | Write a sub-region back; also written `dst[i:i+16] = src` |
| `reinterpret_view` | `pl.` | Reinterpret without moving data |
| `set_validshape` | `pl.` | Declare the meaningful region of a tile |
| `cast` | `pl.` | Convert dtype — may expand to a multi-hop chain, see [LegalizeTileCast](../../dev/passes/15-legalize_tile_cast.md) |
| `dim` | `pl.` | A tensor's runtime dimension |
| `read` `write` | `pl.` | Element access |

## Linear algebra

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| `matmul` | `pl.` | Matrix multiply; `a_trans=` / `b_trans=` transpose an operand in place of a DN annotation |
| `matmul_acc` | `pl.` | Multiply-accumulate into an existing `Acc` tile |
| `matmul_bias` | `pl.` (t) | Multiply with a bias operand |
| `batch_matmul` | `pl.` (t) | Batched multiply, **tile operands only**. For tensors call `pl.matmul` — rank > 2 dispatches to `tile.batch_matmul` during lowering |
| `gemv` `gemv_acc` `gemv_bias` | `pl.` (t) | Matrix-vector forms |
| `matmul_mx` `matmul_mx_acc` `matmul_mx_bias` | `pl.` (t) | MX block-scale multiply — each operand is an FP8E4M3FN data tile plus an FP8E8M0 scale tile |

## Gather, scatter, sort

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| `gather` `gather_row` | `pl.` | Gather by index |
| `paged_gather` | `pl.` | Gather across a paged layout |
| `gatherb` | `pl.` (t) | Gather 32-byte blocks by UINT32 byte offset; one offset column expands to `32 / sizeof(output_dtype)` elements |
| `mgather` | `pl.` (t) | Gather rows from a DDR tensor by index tile, with `coalesce=` and `gather_oob=` policies |
| `scatter` `scatter_update` | `pl.` | Scatter by index; update in place |
| `mscatter` | `pl.` (t) | Masked scatter |
| `sort32` `mrgsort` | `pl.` | Sort a 32-element group; merge sorted runs |

## Block identity

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| `get_block_idx` | `pl.` | This block's index under `pl.spmd` |
| `get_block_num` | `pl.` | The block count |
| `get_subblock_idx` | `pl.` | The AIV lane index under `pl.split_aiv` |

## Cross-core transfer

The mixed-kernel surface — AIC and AIV cooperating inside one InCore function.

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| `tpush_to_aiv` `tpush_to_aic` | `pl.` | Push a tile to the peer core |
| `tpop_from_aic` `tpop_from_aiv` | `pl.` | Pop a tile pushed by the peer |
| `tfree_to_aic` `tfree_to_aiv` | `pl.` | Release a popped slot back to the producer |
| `aic_initialize_pipe` `aiv_initialize_pipe` | `pl.` | Set up the cross-core pipe |
| `aiv_shard` `aic_gather` | `pl.` | Shard across AIV lanes / gather back on AIC |
| `AUTO` | `pl.` | Sentinel for compiler-chosen pipe parameters |

Push and pop must be **paired**, and each pop must be matched by a `tfree`. The tutorial
covering this is not written yet; the mechanics are in
[TPUSH/TPOP](../../reference/pto-isa/01-tpush_tpop.md) and
[ExpandMixedKernel](../../dev/passes/22-expand_mixed_kernel.md).

## Tasks and dependencies

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| `submit` `spmd_submit` | `pl.` | Dispatch a kernel and capture its producer TaskId |
| `no_dep` | `pl.` | Exclude one argument of one task from dependency tracking |
| `dump_tag` | `pl.` | Mark a tensor for selective dump |

See [Scopes and Placement](../language/04-scopes.md).

## Arrays

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| `array.create` | `pl.array.` | Allocate |
| `array.get_element` | `pl.array.` | Read; also written `arr[i]` |
| `array.update_element` | `pl.array.` | Functional update; also written `arr[i] = v` |

## Distributed

`pypto.language.distributed` (conventionally `pld`) carries the collectives and remote
primitives. Full reference at [Distributed Programming](../distributed/index.md);
tutorial at [distributed/00-model.md](../distributed/00-model.md).

### Capability Matrix

| Operation | API | Modes | ReduceOp | Atomic | Supported dtypes | Notes |
| --------- | --- | ----- | -------- | ------ | ---------------- | ----- |
| AllReduce | `pld.tensor.allreduce` | `mesh` (InCore + HOST), `ring` (InCore + HOST) | `Sum`, `Max`, `Min`, `Prod` (mesh); `Sum` only (HOST ring) | — | FP16, FP32 (mesh; hard compile-time check); HOST ring: FP32 only (4-byte) | Mesh: O(N) remote traffic per step. Ring: O(N/P) remote traffic per step, 2(P-1) steps. |
| AllGather | `pld.tensor.allgather` | — | — | — | FP32 only (HOST builtin); any GM dtype (InCore) | Push-based. Input and target must be different buffers. |
| ReduceScatter | `pld.tensor.reduce_scatter` | — | `Sum` only | — | FP32 only (HOST builtin); any GM dtype (InCore) | Every rank stages all NR chunks before the call. |
| Broadcast | `pld.tensor.broadcast` | — | — | — | FP32 only (HOST builtin); any GM dtype (InCore) | Root stages data before the call. |
| All-to-All | `pld.tensor.all_to_all` | — | — | — | FP32 only (HOST builtin); any GM dtype (InCore) | Personalized exchange. Input and target must be different buffers. |
| Barrier | `pld.tensor.barrier` | — | — | — | — | Signal is INT32, single-shot per call. |
| Put | `pld.tensor.put` | — | — | `None_` / `Add` | All GM dtypes | `dst` must be window-bound. Supports chunked + pipelined staging. |
| Get | `pld.tensor.get` | — | — | — | All GM dtypes | `src` must be window-bound. Supports chunked + pipelined staging. |
| Notify | `pld.system.notify` | `AtomicAdd` / `Set` | — | — | — | Side-effect-only signal deposit. |
| Wait | `pld.system.wait` | `Eq` / `Ge` | — | — | — | Side-effect-only signal block. |
| Remote Load | `pld.tile.remote_load` | — | — | — | Any (tile) | Tile-level cross-rank load. |
| Remote Store | `pld.tile.remote_store` | — | — | — | Any (tile) | Tile-level cross-rank store. |

## See Also

- [Choosing a Namespace](00-dispatch.md) — which of these spellings to use.
- [IR Operators](../../dev/ir/05-operators.md) — the registry behind these names.
- [PTOAS Operator Status](../../dev/ptoas-op-status.md) — per-backend support.
