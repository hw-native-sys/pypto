# Catalog

Every operator family, one line each. Every name links into the
[API Reference](../../api/index.md) for its signature — see [Operations](index.md) for why
this page does not repeat them.

> **Reading the tables:** the **Reach** column gives the shortest spelling that works.
> `pl.` means the name is available unqualified; `pl.tile.` / `pl.tensor.` mean the
> operator is level-specific. Names marked **(t)** are tile-only operators re-exported at
> top level for convenience — `pl.load` *is* `pl.tile.load`, not a dispatcher.

## Creation

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| [`create_tensor`][pypto.language.tensor.create] | `pl.` | Allocate a DDR tensor. `manual_dep=True` opts it out of dependency tracking for its lifetime |
| [`create_tile`][pypto.language.tile.create] | `pl.` (t) | Allocate an on-chip buffer |
| [`create_l1`][pypto.language.tensor.create_l1] | `pl.` (t) | Allocate in L1 explicitly |
| [`full`][pypto.language.tensor.full] | `pl.` | A tensor filled with a constant |
| [`arange`][pypto.language.tensor.ci] | `pl.` | Consecutive integers (`tensor.ci`) |
| [`random`][pypto.language.tensor.random] | `pl.` | Random-filled tensor |
| [`tri`][pypto.language.tile.tri] | `pl.` (t) | Lower- or upper-triangular mask tile; `upper=` picks the side, `diagonal=` shifts it, and only the valid region is written |
| [`const`][pypto.language.const] | `pl.` | A literal with an explicit dtype — see [Directives](../language/05-directives.md#typed-constants) |
| `array.create` | `pl.array.` | Allocate an on-core array |

## Data movement

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| [`load`][pypto.language.tile.load] | `pl.` (t) | DDR → `Vec` (default) or `Mat`, via `target_memory=` |
| [`store`][pypto.language.tile.store] | `pl.` (t) | On-chip → DDR |
| [`move`][pypto.language.tile.move] | `pl.` (t) | On-chip → on-chip; the only path into `Left` / `Right` / `Bias` |
| [`reserve_buffer`][pypto.language.system.reserve_buffer] | `pl.` | Reserve a cross-core buffer |
| [`import_peer_buffer`][pypto.language.system.import_peer_buffer] | `pl.` | Reference a peer core's buffer |

See [Memory and Data Movement](../language/03-memory.md) for which moves are legal.

## Elementwise arithmetic

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| [`add`][pypto.language.add] [`sub`][pypto.language.sub] [`mul`][pypto.language.mul] [`div`][pypto.language.div] | `pl.` | Binary arithmetic; a Python number on the right selects the scalar-operand form |
| [`neg`][pypto.language.neg] [`abs`][pypto.language.abs] [`recip`][pypto.language.recip] | `pl.` | Unary negate, absolute value, reciprocal. For FP16/FP32 reciprocal, `high_precision=True` selects the slower, higher-precision PTO path on A5 |
| [`rem`][pypto.language.tile.rem] [`rems`][pypto.language.tile.rems] [`fmod`][pypto.language.fmod] [`fmods`][pypto.language.fmods] | `pl.` | Remainder and floating-point modulo, tensor and scalar forms |
| [`addc`][pypto.language.tile.addc] [`subc`][pypto.language.tile.subc] [`addsc`][pypto.language.tile.addsc] [`subsc`][pypto.language.tile.subsc] | `pl.` (t) | Three-input add / subtract with carry operand |
| [`part_add`][pypto.language.part_add] [`part_mul`][pypto.language.part_mul] [`part_max`][pypto.language.part_max] [`part_min`][pypto.language.part_min] | `pl.` | Partial (segmented) arithmetic |

## Math

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| [`exp`][pypto.language.exp] [`log`][pypto.language.log] | `pl.` | Exponential, natural logarithm |
| [`sqrt`][pypto.language.sqrt] [`rsqrt`][pypto.language.rsqrt] | `pl.` | Square root; reciprocal square root. `high_precision=` is tensor-only and **raises** on a Tile — at tile level precision is selected by passing the scratch tile to `pl.tile.rsqrt(src, tmp)` |
| [`sin`][pypto.language.tensor.sin] [`cos`][pypto.language.tensor.cos] | `pl.` | Trigonometric |

## Comparison and selection

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| [`cmp`][pypto.language.cmp] [`cmps`][pypto.language.tile.cmps] | `pl.` | Compare two operands / operand against a scalar |
| [`maximum`][pypto.language.maximum] [`minimum`][pypto.language.minimum] | `pl.` | Elementwise max / min of two operands |
| [`maximums`][pypto.language.tile.maximums] [`minimums`][pypto.language.tile.minimums] | `pl.` (t) | Elementwise max / min against a scalar |
| [`max`][pypto.language.tile.max] [`min`][pypto.language.tile.min] | `pl.` (t) | Scalar max / min of two values — **not** a tile reduction. To reduce a tile use `row_max` / `col_max` (and the `min` forms) |
| [`sel`][pypto.language.tile.sel] [`sels`][pypto.language.tile.sels] | `pl.` (t) | Select by mask, tensor and scalar forms |

## Activations

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| [`relu`][pypto.language.tile.relu] | `pl.` (t) | Rectified linear |
| [`prelu`][pypto.language.tile.prelu] [`lrelu`][pypto.language.tile.lrelu] | `pl.` (t) | Parametric / leaky rectified linear |

## Bitwise

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| [`and_`][pypto.language.and_] [`or_`][pypto.language.or_] [`xor`][pypto.language.xor] [`not_`][pypto.language.not_] | `pl.` | Bitwise logic |
| [`ands`][pypto.language.ands] [`ors`][pypto.language.ors] [`xors`][pypto.language.xors] | `pl.` | Bitwise logic against a scalar |
| [`shl`][pypto.language.shl] [`shr`][pypto.language.shr] | `pl.` | Shift left / right |
| [`shls`][pypto.language.shls] [`shrs`][pypto.language.shrs] | `pl.` | Shift by a scalar amount |

## Reductions

Row reductions collapse the last axis; column reductions collapse the first.

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| [`row_sum`][pypto.language.row_sum] [`row_prod`][pypto.language.row_prod] [`row_max`][pypto.language.row_max] [`row_min`][pypto.language.row_min] | `pl.` | Reduce along a row |
| [`col_sum`][pypto.language.col_sum] [`col_prod`][pypto.language.col_prod] [`col_max`][pypto.language.col_max] [`col_min`][pypto.language.col_min] | `pl.` | Reduce along a column |
| [`row_argmax`][pypto.language.row_argmax] [`row_argmin`][pypto.language.row_argmin] | `pl.` | Index of the row extremum |
| [`col_argmax`][pypto.language.col_argmax] [`col_argmin`][pypto.language.col_argmin] | `pl.` | Index of the column extremum |

Several reductions accept a `tmp_tile` argument. Passing one changes the reduction
strategy (binary tree versus sequential), which changes floating-point association — the
results differ within tolerance rather than being wrong. Reductions over a partially valid
tile depend on the pad value; see
[Memory § valid shape](../language/03-memory.md#valid-shape-and-padding).

## Broadcast and expand

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| [`row_expand`][pypto.language.row_expand] [`col_expand`][pypto.language.col_expand] | `pl.` | Broadcast a reduced axis back to full width |
| [`row_expand_add`][pypto.language.row_expand_add] [`row_expand_sub`][pypto.language.row_expand_sub] [`row_expand_mul`][pypto.language.row_expand_mul] | `pl.` | Broadcast fused with an arithmetic op |
| [`row_expand_div`][pypto.language.row_expand_div] [`row_expand_max`][pypto.language.row_expand_max] [`row_expand_min`][pypto.language.row_expand_min] | `pl.` | Broadcast fused with division, max, or min |
| [`col_expand_add`][pypto.language.col_expand_add] [`col_expand_sub`][pypto.language.col_expand_sub] [`col_expand_mul`][pypto.language.col_expand_mul] | `pl.` | Column-wise equivalents |
| [`col_expand_div`][pypto.language.col_expand_div] [`col_expand_max`][pypto.language.col_expand_max] [`col_expand_min`][pypto.language.col_expand_min] | `pl.` | Column-wise division, max, and min |
| [`row_expand_expdif`][pypto.language.row_expand_expdif] [`col_expand_expdif`][pypto.language.col_expand_expdif] | `pl.` | Broadcast fused with `exp(x - m)` — the softmax kernel |
| [`expand_clone`][pypto.language.tensor.expand_clone] [`expands`][pypto.language.expands] | `pl.` | Broadcast a value across a shape |
| [`fillpad`][pypto.language.fillpad] [`fillpad_expand`][pypto.language.fillpad_expand] | `pl.` | Fill the invalid region; optionally broadcast in the same step |

## Shape and layout

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| [`reshape`][pypto.language.reshape] | `pl.` | Reinterpret dimensions |
| [`transpose`][pypto.language.transpose] | `pl.` | Transpose |
| [`slice`][pypto.language.slice] | `pl.` | Sub-region; also written `A[0:16, :]` |
| [`concat`][pypto.language.concat] | `pl.` | Join along an axis |
| [`assemble`][pypto.language.tensor.assemble] | `pl.` | Write a sub-region back; also written `dst[i:i+16] = src` |
| [`reinterpret_view`][pypto.language.reinterpret_view] | `pl.` | Reinterpret without moving data |
| [`set_validshape`][pypto.language.set_validshape] | `pl.` | Declare the meaningful region of a tile |
| [`cast`][pypto.language.cast] | `pl.` | Convert dtype — may expand to a multi-hop chain, see [LegalizeTileCast](../../dev/passes/15-legalize_tile_cast.md) |
| [`dim`][pypto.language.tensor.dim] | `pl.` | A tensor's runtime dimension |
| [`read`][pypto.language.read] [`write`][pypto.language.write] | `pl.` | Element access |

## Linear algebra

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| [`matmul`][pypto.language.matmul] | `pl.` | Matrix multiply; `a_trans=` / `b_trans=` transpose an operand in place of a DN annotation |
| [`matmul_acc`][pypto.language.matmul_acc] | `pl.` | Multiply-accumulate into an existing `Acc` tile |
| [`matmul_bias`][pypto.language.tile.matmul_bias] | `pl.` (t) | Multiply with a bias operand |
| [`batch_matmul`][pypto.language.batch_matmul] | `pl.` (t) | Batched multiply, **tile operands only**. For tensors call `pl.matmul` — rank > 2 dispatches to `tile.batch_matmul` during lowering |
| [`gemv`][pypto.language.tile.gemv] [`gemv_acc`][pypto.language.tile.gemv_acc] [`gemv_bias`][pypto.language.tile.gemv_bias] | `pl.` (t) | Matrix-vector forms |
| [`matmul_mx`][pypto.language.tile.matmul_mx] [`matmul_mx_acc`][pypto.language.tile.matmul_mx_acc] [`matmul_mx_bias`][pypto.language.tile.matmul_mx_bias] | `pl.` (t) | A5 MX block-scale multiply — data tiles reaching the op must be FP8E4M3FN; the supported FP4-input form is FP4×FP8, with the FP4 lhs explicitly cast to FP8 first; native FP4×FP4 is unsupported |

## Gather, scatter, sort

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| [`gather`][pypto.language.tensor.gather] [`gather_row`][pypto.language.tensor.gather_row] | `pl.` | Gather by index |
| [`paged_gather`][pypto.language.tensor.paged_gather] | `pl.` | Gather across a paged layout |
| [`gatherb`][pypto.language.tile.gatherb] | `pl.` (t) | Gather 32-byte blocks by UINT32 byte offset; one offset column expands to `32 / sizeof(output_dtype)` elements |
| [`mgather`][pypto.language.tile.mgather] | `pl.` (t) | Gather rows from a DDR tensor by index tile, with `coalesce=` and `gather_oob=` policies |
| [`scatter`][pypto.language.tensor.scatter] [`scatter_update`][pypto.language.tensor.scatter_update] | `pl.` | Scatter by index; update in place |
| [`mscatter`][pypto.language.tile.mscatter] | `pl.` (t) | Masked scatter |
| [`sort32`][pypto.language.tensor.sort32] [`mrgsort`][pypto.language.tensor.mrgsort] | `pl.` | Sort a 32-element group; merge sorted runs |

## Block identity

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| [`get_block_idx`][pypto.language.tensor.get_block_idx] | `pl.` | This block's index under `pl.spmd` |
| [`get_block_num`][pypto.language.tensor.get_block_num] | `pl.` | The block count |
| [`get_subblock_idx`][pypto.language.tensor.get_subblock_idx] | `pl.` | The AIV lane index under `pl.split_aiv` |

## Cross-core transfer

The mixed-kernel surface — AIC and AIV cooperating inside one InCore function.

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| [`tpush_to_aiv`][pypto.language.system.tpush_to_aiv] [`tpush_to_aic`][pypto.language.system.tpush_to_aic] | `pl.` | Push a tile to the peer core |
| [`tpop_from_aic`][pypto.language.system.tpop_from_aic] [`tpop_from_aiv`][pypto.language.system.tpop_from_aiv] | `pl.` | Pop a tile pushed by the peer |
| [`tfree_to_aic`][pypto.language.system.tfree_to_aic] [`tfree_to_aiv`][pypto.language.system.tfree_to_aiv] | `pl.` | Release a popped slot back to the producer |
| [`aic_initialize_pipe`][pypto.language.system.aic_initialize_pipe] [`aiv_initialize_pipe`][pypto.language.system.aiv_initialize_pipe] | `pl.` | Set up the cross-core pipe |
| [`aiv_shard`][pypto.language.tile.aiv_shard] [`aic_gather`][pypto.language.tile.aic_gather] | `pl.` | Shard across AIV lanes / gather back on AIC |
| `AUTO` | `pl.` | Sentinel for compiler-chosen pipe parameters |

Push and pop must be **paired**, and each pop must be matched by a `tfree`. The tutorial covering this is
[Mixed kernels](../tutorials/03-mixed-kernel.md); the machine-level mechanics are in
[TPUSH/TPOP](../../reference/pto-isa/01-tpush_tpop.md) and
[ExpandMixedKernel](../../dev/passes/22-expand_mixed_kernel.md).

## Tasks and dependencies

| Operator | Reach | What it does |
| -------- | ----- | ------------ |
| [`submit`][pypto.language.submit] [`spmd_submit`][pypto.language.spmd_submit] | `pl.` | Dispatch a kernel and capture its producer TaskId |
| `deps=` | `pl.at`, captured inline `pl.spmd` | Add strict TaskId dependencies; deferred waiters use this same dependency path |
| [`no_dep`][pypto.language.tensor.no_dep] | `pl.` | Exclude one argument of one task from dependency tracking |
| [`dump_tag`][pypto.language.tensor.dump_tag] | `pl.` | Mark a tensor for selective dump |

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
| Deferred Wait | `pld.system.defer_wait` | `Ge` only | — | — | INT32 signal | Register a monotonic counter condition without spinning the AIV; Simpler keeps the ordinary waiter TaskId incomplete and later work uses ordinary `deps=[wait_tid]`. |
| Remote Load | `pld.tile.remote_load` | — | — | — | Any (tile) | Tile-level cross-rank load. |
| Remote Store | `pld.tile.remote_store` | — | — | — | Any (tile) | Tile-level cross-rank store. |

## Worked examples

One runnable file per family, for when the table entry is not enough:

| Family | Example |
| ------ | ------- |
| Elementwise arithmetic | `examples/beginner/02_elementwise.py` |
| Scalar operands | `examples/beginner/03_scalar_ops.py` |
| Activations | `examples/beginner/04_activation.py` |
| Matmul | `examples/beginner/05_matmul.py` |
| Concatenation / assemble | `examples/beginner/06_concat.py`, `examples/intermediate/05_assemble.py` |
| Reductions | `examples/intermediate/02_softmax.py`, `examples/intermediate/03_normalization.py` |
| Cross-core transfer | `examples/advanced/03_mixed_kernel.py` |
| Tasks and dependencies | `examples/intermediate/07_task_graph.py` |

## See Also

- [Choosing a Namespace](00-dispatch.md) — which of these spellings to use.
- [IR Operators](../../dev/ir/05-operators.md) — the registry behind these names.
- [PTOAS Operator Status](../../dev/ptoas-op-status.md) — per-backend support.
