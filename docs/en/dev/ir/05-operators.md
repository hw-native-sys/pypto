# Operator System

Type-safe operator definitions with automatic type deduction, organized into modular categories (TensorOp, TileOp, SyncOp, CrossCoreOp).

## Operator Categories

| Category | Types | Use Case | File Location |
| -------- | ----- | -------- | ------------- |
| **TensorOp** | TensorType | N-D tensor operations with broadcasting | `src/ir/op/tensor_ops/` |
| **TileOp** | TileType | Hardware-optimized tile operations | `src/ir/op/tile_ops/` |
| **SyncOp** | UnknownType (barriers); ScalarType (task / launch queries) | Pipeline barriers, synchronization, TaskId and SPMD launch-shape queries | `src/ir/op/sync_ops/` |
| **CrossCoreOp** | UnknownType/TileType | AIC↔AIV cross-core communication | `src/ir/op/sync_ops/cross_core.cpp` |
| **PrefetchOp** | Opaque handles | Asynchronous GM→L2 cache prefetch | `src/ir/op/prefetch/prefetch_async.cpp` |

**Key Features**: Fluent API, automatic type deduction, kwargs for metadata, NumPy-style broadcasting, type promotion, dynamic dimensions (`kDynamicDim`)

## Type System

```cpp
// Dynamic dimensions (pypto/core/common.h)
constexpr int64_t kDynamicDim = -1;
auto dynamic_dim = make_int(kDynamicDim);
```

| Type | Dimensions | Use Case | Memory |
| ---- | ---------- | -------- | ------ |
| **TensorType** | N-D | General tensors, function params/returns | DDR (optional MemRef) |
| **TileType** | N-D | Hardware-optimized tiles in unified buffers | Unified buffer (optional MemRef) |
| **ScalarType** | 0D | Scalar values | Register |
| **UnknownType** | N/A | No return value (sync ops) | N/A |

## REGISTER_OP Fluent API

| Method | Purpose | Example |
| ------ | ------- | ------- |
| `set_op_category(str)` | Operator category | `.set_op_category("TensorOp")` |
| `set_description(str)` | Human-readable description | `.set_description("Element-wise add")` |
| `add_argument(name, desc)` | Positional Expr argument | `.add_argument("lhs", "Left tensor")` |
| `no_argument()` | No arguments (sync ops) | `.no_argument()` |
| `set_attr<T>(name)` | Kwarg schema (T: bool, int, DataType, etc.) | `.set_attr<bool>("a_trans")` |
| `f_deduce_type(fn)` | Type deduction function | `.f_deduce_type(DeduceAddType)` |
| `set_core_affinity(a)` | Which core executes the op (**placement**) | `.set_core_affinity(core_affinity::CoreAffinity::VECTOR)` |
| `set_no_duplicate()` | Op must not run on a second core (**replication**) | `.set_no_duplicate()` |
| `set_arg_effect(i, e)` | What the op does to argument `i`'s buffer | `.set_arg_effect(2, ArgEffect::Write)` |
| `set_arg_effect(i, fn)` | Same, when a kwarg decides it | `.set_arg_effect(2, [](const auto& kw) { ... })` |
| `no_arg_writes()` | Classified: writes through no argument | `.no_arg_writes()` |
| `set_write_channel(c)` | Hardware path the op's writes take | `.set_write_channel(WriteChannel::Dma)` |

### Argument effects

> The whole chain that consumes these declarations is laid out in
> [Parameter Direction Inference](08-param-directions.md).

An operator that updates one of its arguments in place must say so. Direction
inference, dependency analysis and the parameter-direction verifier all ask the
registry the same question — *does this call write the buffer this argument
names?* — and an operator that never answered reads as a pure consumer:

```text
tile.mscatter writes output_tensor, but never declared it
  → the parameter it writes keeps direction In
  → no RAW edge is emitted against the kernel that reads it
  → the scheduler is free to run the reader first
  → stale data, or a deadlock waiting on a signal nobody wrote
```

| Effect | Meaning | Examples |
| ------ | ------- | -------- |
| `ArgEffect::Read` | Read, never written. Default for an unnamed argument | `tile.store`'s source tile |
| `ArgEffect::Write` | Overwritten without being read first | `tile.store`'s `output_tensor`, `pld.tile.get`'s `dst` |
| `ArgEffect::ReadWrite` | Read *and* written | `tile.matmul_acc`'s accumulator, an atomic store's destination |

**Partial overwrite is still `Write`.** A store that lands on a sub-region does
not read the untouched remainder — nothing moves *into* the kernel — so its
destination is a pure write. Declaring it `ReadWrite` is not a harmless
approximation: it makes the enclosing parameter `InOut`, which stages the buffer
host→device and, across ranks, invents a dependency between two ranks writing
disjoint rows.

Whether a destination is read is decided per operator, not per family: a
gather or exchange destination is pushed into and never loaded from, so it is
`Write`, while a reduce destination has its running value loaded back and is
`ReadWrite`.

**`ReadWrite` is for operators that genuinely read the slot**: an accumulator
(`out += x` reads the running sum), an atomic store or assemble, or a
destination-passing operator whose untouched positions flow through to its SSA
result (`tile.scatter`, `array.update_element`).

**Kwarg-dependent effects.** When a kwarg decides the answer, pass a resolver
instead of a constant. A kwarg can decide *whether* an argument is written at
all, not only how: `tile.mgather`'s third operand is a written GM scratch tensor
in Mat element mode and a read-only `valid_shape` in Mat row mode. The other
live cases are the `atomic` kwarg on the store family and the `op` kwarg on
`pld.system.notify`, whose default is atomic-add — so an unannotated notify
reads the slot it adds into:

```cpp
REGISTER_OP("tile.store")
    // ... arguments, memory spec ...
    .set_arg_effect(2,
                    [](const std::vector<std::pair<std::string, std::any>>& kwargs) {
                      return GetIntKwarg(kwargs, "atomic", static_cast<int>(AtomicType::kNone)) ==
                                     static_cast<int>(AtomicType::kNone)
                                 ? ArgEffect::Write
                                 : ArgEffect::ReadWrite;
                    })
    .set_write_channel(WriteChannel::Dma)
```

**Declared-read-only is not the same as unclassified.** `HasDeclaredArgEffects()`
distinguishes "a human decided this operator writes nothing" (`no_arg_writes()`,
e.g. `pld.system.wait`) from "nobody has looked at this operator yet". An
analysis that needs the answer can then refuse to guess instead of defaulting an
unclassified writer to read-only.

**Enforcement.** `OpRegistry::ValidateArgEffects()` runs at import and rejects
two shapes, naming every offender and the fix rather than failing on first use:

- an operator declaring `set_output_reuses_input(N)` — its SSA result *is*
  argument N's buffer, so it writes through it — without a verdict about
  argument N specifically. Classification is what is required, not a particular
  answer: an operator whose in-place slot is metadata may declare it read-only.
- an operator declaring a write channel while writing through no argument. A
  channel says *how* an operator writes, so one without a write is either a
  stray declaration or a missing one.

The second rule matters more than it looks. `set_write_channel()` creates the
effect spec as a side effect, so "the spec exists" cannot stand in for "a human
decided" — otherwise an operator that declared a channel and forgot its
`set_arg_effect` would pass the first rule with the argument it updates still
defaulting to `Read`. `no_arg_writes()` records the all-arguments verdict
explicitly, and combining it with `set_arg_effect` is rejected as
contradictory.

`set_write_channel` records whether the writes travel the MTE3/DMA path or the
scalar D-cache path. PyPTO cannot order the two against one GM tensor, so a
function mixing them on one buffer is rejected; the channel lets that diagnostic
read the registry instead of re-listing operators.

Declare it only for an operator whose writes really are one of those two paths,
and leave it unset otherwise — an unset channel keeps the operator out of that
diagnostic, which is where an operator belongs when neither path describes it:

- `pld.system.notify` emits `pto.comm.tnotify`, a distinct comm instruction.
  Claiming either channel would let the diagnostic reject a valid program.
- `system.set_ffts` hands the workspace *pointer* to the FFTS unit rather than
  moving data; the hardware writes that region on its own schedule, which no
  dependency edge models. It declares `no_arg_writes()`.
- A composite collective updates a data window and a signal through different
  mechanisms, and one operator-level channel cannot describe both. Per-argument
  channels would, but no case yet needs the distinction, and a wrong single
  answer is worse than none.

**`set_core_affinity` vs `set_no_duplicate`** — two orthogonal axes, and picking
the wrong one makes a false claim about the ISA:

- `set_core_affinity(...)` answers *which* core runs the op. Declare it only
  when the hardware really constrains the op to one side. When left unset,
  `ClassifyCallAffinity` derives placement from the call (memory spec, then the
  first tile argument's memory space), falling back to `SHARED`.
- `set_no_duplicate()` answers whether the op may run on a *second* core.
  `ExpandMixedKernel` replicates `SHARED` statements onto both the AIC and the
  AIV lane; mark an op for which that copy changes what the program means.

`pld.system.notify` is the canonical case, and the hazard is **premature release
from the wrong lane**, not non-idempotence: the AIC copy can publish the signal
before the AIV lane's TPUT has landed the data that signal releases, so the peer
reads stale bytes. That is why the flag is unconditional — a `NotifyOp::kSet`
fires the same race as an atomic-add, even though only the atomic-add
double-counts.

Its sibling `pld.system.wait` stays **unmarked**, and deliberately so: TWAIT
*blocks*, and its presence on the cube lane is load-bearing. Pinning it to the
vector lane would let the matmul race past the peer data it waits on. Read the
flag as "must not run on a second core", not as "not idempotent".

`IsNoDuplicate()` reads the axis. Its only consumer is `LowerAutoVectorSplit`'s
`pl.split_aiv` region placement stamp (pass 20), which pins exactly the
no-duplicate calls inside a region to the AIV lane. No verifier rejects anything
on this axis.

An op pinned to one lane by `set_core_affinity(...)` is never duplicated in the
first place and needs no flag. The flag exists precisely for the core-agnostic
ops, where no affinity value can express "may run on either core, but must not
run on both". Note that is a claim about *cores*, not about total executions: an
AIV function body still runs on both AIV sub-lanes under `dual_aiv_dispatch`, so
keeping an op off the cube lane does not make it happen once. That part is the
author's, and is documented in
[Scopes → pl.split_aiv](../../user/language/04-scopes.md).

**Type Deduction Signature:**

```cpp
std::function<TypePtr(const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs)>
```

## C++ Registration Examples

### Simple Elementwise Operator

```cpp
// src/ir/op/tensor_ops/elementwise.cpp
REGISTER_OP("tensor.add")
    .set_op_category("TensorOp")
    .add_argument("lhs", "Left tensor")
    .add_argument("rhs", "Right tensor")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 2);
      auto t1 = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());
      auto t2 = std::dynamic_pointer_cast<const TensorType>(args[1]->GetType());
      auto dtype = PromoteDataTypes(t1->dtype_, t2->dtype_);
      auto shape = BroadcastShapes(t1->shape_, t2->shape_);
      return std::make_shared<TensorType>(shape.shape, *dtype);
    });
```

### Operator with Kwargs

```cpp
// src/ir/op/tensor_ops/matmul.cpp
TypePtr DeduceMatMul(const std::vector<ExprPtr>& args,
                     const std::vector<std::pair<std::string, std::any>>& kwargs) {
  auto lhs = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());
  auto rhs = std::dynamic_pointer_cast<const TensorType>(args[1]->GetType());

  auto get = [&](const std::string& k, bool d) {
    for (const auto& [name, val] : kwargs)
      if (name == k) return std::any_cast<bool>(val);
    return d;
  };

  DataType dtype = [&]() {
    for (const auto& [k, v] : kwargs)
      if (k == "out_dtype") return static_cast<DataType>(std::any_cast<int>(v));
    return *PromoteDataTypes(lhs->dtype_, rhs->dtype_);
  }();

  bool a_t = get("a_trans", false), b_t = get("b_trans", false);
  ExprPtr m = a_t ? lhs->shape_[1] : lhs->shape_[0];
  ExprPtr n = b_t ? rhs->shape_[0] : rhs->shape_[1];
  return std::make_shared<TensorType>(std::vector<ExprPtr>{m, n}, dtype);
}

REGISTER_OP("tensor.matmul")
    .set_op_category("TensorOp")
    .add_argument("lhs", "Left matrix")
    .add_argument("rhs", "Right matrix")
    .set_attr<DataType>("out_dtype")
    .set_attr<bool>("a_trans")
    .set_attr<bool>("b_trans")
    .f_deduce_type(DeduceMatMul);
```

For 2D `tile.matmul`, the physical boxed K dimensions must match. PTO derives
the contraction extent from the lhs valid K, so that extent may be smaller than
the rhs valid K but must be contained by it. `tile.matmul_acc` likewise keeps
exact physical M/N/K box compatibility while allowing the accumulator's valid
M/N rectangle and the rhs valid K extent to contain the smaller rectangle PTO
computes from lhs M/K and rhs N.

#### Conditional accumulator initialization (`init_cond`)

`tile.matmul_acc`, `tile.batch_matmul_acc`, `tensor.matmul_acc`, and
`tile.gemv_acc` take an optional fourth operand, `init_cond`: a BOOL scalar that
selects, per execution, whether the accumulator is *overwritten* with
`lhs @ rhs` or accumulated into. It is the split-K `k == 0` idiom, and it removes
the need either to zero the accumulator or to peel the first K step:

```python
acc = pl.tile.create([16, N], pl.INT32, target_memory=pl.Mem.Acc)
for k0 in pl.pipeline(0, K, K_TILE, stage=2):
    ...
    acc = pl.tile.matmul_acc(acc, a_left, b_right, init_cond=(k0 == 0))
```

The predicate's domain is exactly `matmul_acc`'s own: any operand shape that
accumulates without a predicate accumulates with one. A `tensor.matmul_acc` with
an operand of rank > 2 converts to `tile.batch_matmul_acc`, which forwards
`init_cond` verbatim to every 2D `tile.matmul_acc` that `FlattenTileNdTo2D`
unrolls it into — each of those is the sole writer of its own row band of the
accumulator, so the predicate applies band by band. (Only `batch_count == 1`
reaches codegen today; a larger batch is rejected in `FlattenTileNdTo2D` for a
reason unrelated to the predicate — the per-batch accumulator would be a strided
L0C row window, which the MAD cannot address.)

The predicate is a positional operand rather than a registry kwarg because it
may be loop-dependent; kwargs carry only compile-time constants. Registering it
as an operand also means it participates in the use-def chain like any other
SSA value.

Being an operand, it prints positionally at the tile layer —
`pl.tile.matmul_acc(acc, lhs, rhs, k0 == 0)`. Two signatures already spend
positional slot 4 (`a_trans` at the tensor layer, `acc_phase` on GEMV), so there
the printer emits it as a keyword — and `init_cond` is correspondingly
keyword-only in those DSL signatures. Every printed form reparses to the same IR:

`pl.tensor.matmul_acc(acc, lhs, rhs, init_cond=k0 == 0, a_trans=False, b_trans=False)`
`pl.tile.gemv_acc(acc, lhs, rhs, init_cond=k0 == 0, acc_phase='unspecified')`

Lowering depends on whether the predicate is known at compile time:

| `init_cond` | Emitted |
| ----------- | ------- |
| absent, or literal `False` | `pto.tmatmul.acc ins(dst, lhs, rhs) outs(dst)` |
| literal `True` | `pto.tmatmul ins(lhs, rhs) outs(dst)` |
| runtime predicate | `scf.if cond { pto.tmatmul } else { pto.tmatmul.acc }` |

`tile.gemv_acc` lowers through the same emitter with `pto.tgemv.acc` /
`pto.tgemv` substituted — GEMV is a matmul whose M is 1, run on the same cube
MAD, so it carries the same `cmatrixInit` bit. Its `acc_phase` attribute rides on
whichever arm is emitted.

The ISA carries this as bit 63 (`cmatrixInit`) of the MAD's Xt register, so the
hardware needs no branch; `pto.tmatmul` and `pto.tmatmul.acc` are distinct ops
with no init operand, hence the branch. Because `matmul_acc` is in place
(`set_output_reuses_input(0)`), both arms write the same buffer and the `scf.if`
yields no value — no phi is materialized on the Acc tile.

"Literal" covers **both** spellings a constant predicate arrives in: a DSL
`init_cond=True`/`False` reaches the emitter as a BOOL-typed `ConstInt`, while a
predicate an earlier pass folded reaches it as a `ConstBool` — which is what the
generated `ko == 0` becomes when [`LowerPipelineLoops`](../passes/29-lower_pipeline_loops.md)
replicates the K-loop *and* the enclosing loop is eliminated, so each replica's
index is a literal. Both pick an arm outright, and an emitter that folded only
one of the two would double the MADs of every K block it missed.

The fold is therefore conditional on the trip count, not universal: at
`16x512x64` the pipelined loop disappears and the emitted PTO has no `scf.if`,
while at `16x2048x64` the replica indices stay symbolic (`ko`, `ko + 256`) and
two `scf.if`s survive. That is not a regression — the peeled `IfStmt` this
replaced produced the same two branches for those shapes.

The compiler uses the idiom it recommends: `AutoTileMatmulL0` *emits* the
predicated form for the K-loop of a plain `tile.matmul`, so the `tile.create`
seed, the loop-carried value and the loop's `return_var` share one L0C buffer
by construction. `tile.matmul_bias` carries no `init_cond` operand, so it cannot
use the predicated body; its first K block is *head-peeled* out of the loop
instead, applying the bias exactly once and minting the accumulator that the
remaining blocks accumulate into. That reaches the same one-buffer chain without
a predicate, so the pass no longer generates an accumulator phi at all.

One limitation, diagnosed rather than silently dropped:

- **`batch_count > 1` is rejected.** Not because of the predicate — this shape
  fails identically without one. `FlattenTileNdTo2D` gives each batch a
  `tile.slice` of the accumulator, and a row window of a multi-block-column
  L0C tile is strided, which the MAD cannot address (pto-isa#253). Rank > 2 is
  fine as long as the batch dims multiply to 1, which is the grouped-GEMM case
  (`[1, N, K]` weights). For a genuine batch, loop over the batch dimension
  instead.

An oversized *predicated* `tile.matmul_acc` is K-tiled like the unpredicated
one: the caller's predicate is ANDed with the emitted loop's own `ko == 0`, and
the peeled partial tail keeps the unpredicated 3-operand form (it is never the
first K block).

M/N tiling of an accumulate is available only at *loop* level, and equally for
both spellings: a `tile.create([M, N])` / split-K `pl.pipeline` / one-2D-store
triplet is tiled outside its K loop whether the reduction is peeled
(`if ko == 0: matmul else: matmul_acc`) or predicated
(`matmul_acc(acc, lhs, rhs, ko == 0)`). Outside that shape — a standalone
oversized `tile.matmul_acc` on a caller-owned `[M, N]` accumulator, or a
predicated one whose `init_cond` is not a seed test on the loop's induction
variable — slicing the accumulator is unsupported and the pass says so with a
`PH-AT-006` perf hint.

At the tile layer, `tile.batch_matmul` provides batched semantics for
`TileType` operands. It accepts rank >= 2 tiles, broadcasts the leading batch
dimensions, and keeps the same operand-only interface style as `tile.matmul`.
If batch operands need transpose semantics, that can be expressed either with
an explicit `tile.transpose(...)` on the inputs or by a zero-copy
`tile.transpose_view(...)` over a natural `tile.load`. During later lowering to
2D `tile.matmul`, both forms are normalized to the same operand-transpose
semantics.

`tile.batch_matmul_acc(acc, lhs, rhs)` is the accumulating counterpart for
the batched path: `acc = acc + lhs @ rhs` with the same rank>=2 + batch
broadcasting rules as `tile.batch_matmul`. The acc batch shape must match the
broadcast batch shape of `lhs`/`rhs` exactly; the matmul (M, N) dims must
match the trailing dims of acc; and the K dimension must match between lhs
and rhs. The inner accumulator type defaults to FP32 for floating inputs and
INT32 for integer inputs (mirroring `tile.matmul_acc`). At conversion time
`ConvertTensorToTileOps` dispatches `tensor.matmul` / `tensor.matmul_acc` to
this batched path whenever any operand has rank > 2; `FlattenTileNdTo2D`
later unrolls the batched form into per-batch 2D ops.

### MX block-scale matmul (Ascend950)

MX uses dedicated `LeftScale` / `RightScale` memory spaces and the `FP8E8M0`
scale dtype. PyPTO supports host-prequant MXFP8 and an explicit FP4×FP8 conversion path on Ascend950 through
the `matmul_mx` family. `InsertMxScaleAddr` (after `InferTileMemorySpace`)
inserts the internal `tile.tget_scale_addr` bindings once operand memory
spaces are resolved.

| IR / DSL | Notes |
| -------- | ----- |
| `tile.load` of `pl.Tensor[..., pl.MX_A_ZZ \| pl.MX_B_NN]` | The source TensorLayout carries the MX scale GM layout. Dtype is FP8E8M0 or UINT8, `target_memory=Mat` is required, and strided sources are rejected. |
| `tile.move(..., target_memory=LeftScale/RightScale)` | Mat-to-Scale move with hardware-fixed row/row/32 (left) or col/col/32 (right) layout; the source Mat tile and layout overrides must match exactly. |
| `tile.create(..., target_memory=LeftScale/RightScale)` | Not supported; load MX scale data into Mat and then move it into scale memory. |
| `tile.matmul_mx` / `pl.matmul_mx` | `Left, LeftScale, Right, RightScale → Acc`; both data operands reaching the op must be `FP8E4M3FN`, and scale is `FP8E8M0`. The supported FP4-input form is FP4 lhs × FP8 rhs with an explicit `pl.cast(fp4, pl.FP8E4M3FN)` before the op; native FP4×FP4 and FP8×FP4 are rejected. Physical M/K/N, valid K, and scale-group counts use the post-cast FP8 tile extents, never the packed x2 carrier shape. Physical `M % 16 == 0`, `K % 64 == 0`, and `N % 32 == 0`; valid K must satisfy `ceil(validK/32) == ceil(physicalK/32)`. Alignment / scale-group checks run only for constant extents; symbolic dims skip the numeric checks and fall back to the declared scale tile geometry (later PTOAS still verifies). |
| `tile.matmul_mx_acc` / `pl.matmul_mx_acc` | `Acc, Left, LeftScale, Right, RightScale → Acc`; in-place through `set_output_reuses_input(0)`; accumulator physical and valid M/N must match the matmul output. |
| `tile.matmul_mx_bias` / `pl.matmul_mx_bias` | `Left, LeftScale, Right, RightScale, Bias → Acc`; bias is `[1, N]` FP32. |
| `tile.tget_scale_addr` | Compiler-generated A5 binding from `LeftScale↔Left` or `RightScale↔Right`; DPS in-place on `dst_scale`. Users write only the `matmul_mx` family. |

The canonical shape is `M=128, K=64, N=64`, with FP8E4M3FN data,
FP8E8M0 scales shaped `[128,2]` and `[2,64]`, and `mx_a_zz` / `mx_b_nn`
host layouts. The left input may originate as FP4 only when explicitly cast to FP8 first.
Align M↑16, K↑64, and N↑32.

MX tensor subviews are a legacy limitation. `tensor.slice`, `tensor.reshape`,
`tensor.transpose`, `tensor.reinterpret_view`, and `tensor.view` reject
MX-layout sources because the hardware path cannot represent a subview base
offset. `pld.tile.remote_load` also rejects MX layouts until its complete scale
layout contract is implemented.

FP4 Tensor/Tile shapes and `valid_shape` are logical nibble counts; the innermost extent must be positive and even, and slice origins cannot select a byte's second nibble.
Torch/runtime carries `float4_e2m1fn_x2` in a physical x2 shape; JIT/compiled-call conversion avoids a persistent IR `storage_shape`.

An explicit left-side FP4→FP8 tile cast is legalized on A5 as
FP4→BF16→FP32→FP8E4M3FN. Scale values are unchanged because this is a numerical
cast of the data operand. Native packed-FP4 matmul and MXFP4 quantization remain deferred.

#### MX / Ascend950: pto-isa constraints

| Constraint | Detail |
| ---------- | ------ |
| Distinct scale buffers | Cube does not fold scales into Left/Right data. `TileType::ScaleLeft` / `ScaleRight` sidecars map to PyPTO `LeftScale` / `RightScale`. |
| Payload | Scale is `float8_e8m0_t` / `FP8E8M0`; the emitted MX data pair is `FP8E4M3FN × FP8E4M3FN` (rejects `FP8E5M2` and native packed FP4). A logical FP4×FP8 input first casts the FP4 lhs to FP8. Physical K, valid K, and `ceil(K/32)` scale groups are measured on that post-cast FP8 tile, not the packed x2 carrier; physical `K%64==0` and fractal is 32. |
| Layouts | `mx_a_zz` is row-major ZZ; `mx_b_nn` is col-major NN; loads use `TLoadMxCube*` (AZZ2ZZ). |
| `TMov` `CommonCheckMX` | Allows UINT8 Mat → FP8E8M0 ScaleLeft/Right; canonical path: ui8 Mat reshape then ui8→f8 Scale. |
| Bind then fill | Fill **after** `GetScaleAddr(Left/Right)`; writing the provisional alloc address is orphaned once rebound. |
| Alignment | Post-cast FP8 tile extents require physical `M%16==0`, `K%64==0`, and `N%32==0`; `DeduceTileMatMulMxType` enforces these for **constant** extents only. Symbolic dims skip numeric checks. |

#### MX / Ascend950: PTOAS constraints

| Constraint | Detail |
| ---------- | ------ |
| Single `loc=scaling` | PTOAS has no distinct left/right scale locations; EmitC recovers ScaleLeft/Right. |
| FP8E8M0 scaling dtype | UINT8 with `loc=scaling` is treated as Fixpipe scaling, so promote before entering LeftScale/RightScale. |
| No Mat↔Scaling `treshape` | Different locs; reshape stays in Mat (ui8), then `tmov` into scaling. |
| Shape-matched Mat→Scale `tmov` | Flat `[1,G]` must `treshape` to `[M,K/32]` (or B-side shape) first. |
| Order | PyPTO emits Mat→scaling `tmov` in source order; PTOAS `PTOA5NormalizeTMovPass` reorders `tget_scale_addr` before it (ISA bind-then-fill). |
| `#pto.layout` / mx load | `mx_a_zz` / `mx_b_nn` / …; this stage uses **host ZZ/NN** (AZZ2ZZ). |
| Coverage | `pto.tmatmul.mx` / `.acc` / `.bias` + `pto.tget_scale_addr`. |

### Tile-only GEMV family (A2/A3)

The tile-only GEMV family uses logical shape `[1, N]` but follows the Cube
instruction's padded physical contract. Its Acc result has 16 physical rows,
while its physical column count follows the RHS tile (and must satisfy the
target's normal C0 alignment); the bias uses the same physical column count.
Their `valid_shape` retains the logical `[K, N]`, `[1, N]`, and `[1, N]`
regions. The lhs must have exactly one physical and logical row. A single-row Mat load uses `blayout=row_major` and
`slayout=none_box`, selecting PTO-ISA's row-vector extraction path.

The rhs logical K must cover the lhs logical K. Supported dtype triples
are `INT8 x INT8 -> INT32` and same-type `FP16`, `BF16`, or `FP32` inputs to
`FP32`; `gemv_acc` uses that output dtype for `acc`, and `gemv_bias` requires
the same output dtype for `bias`. The bias valid shape must cover the logical
output shape `[1, N]`; its valid N may be wider when the physical N matches.

`tile.gemv`, `tile.gemv_acc`, and `tile.gemv_bias` accept `acc_phase` as
`"unspecified"` (the default), `"partial"`, or `"final"`. Use `"partial"`
while more K chunks remain and `"final"` for the last chunk.

`tile.gemv_acc` additionally takes the optional `init_cond` predicate — see
[Conditional accumulator initialization](#conditional-accumulator-initialization-init_cond).
`tile.gemv_bias` carries none, mirroring `tile.matmul_bias`: a biased GEMV
already mints its accumulator, so it has no initial value to predicate.

The padded Acc contract shapes how a predicated split-K GEMV mints that
accumulator. Because a `[1, N]` result occupies 16 physical rows,
`pl.tile.create([1, N], ...)` is rejected on physical shape and `[16, N]` on
valid shape; create at the physical shape and narrow the valid rectangle:

```python
acc_raw = pl.tile.create([16, N], pl.FP32, target_memory=pl.Mem.Acc)
acc = pl.tile.set_validshape(acc_raw, 1, N)  # then gemv_acc(..., init_cond=(k0 == 0))
```

Before `init_cond`, the peel did this implicitly — a straight-line `pl.tile.gemv`
mints a correctly typed accumulator, at the cost of a phi between the branches.

## Python Usage

```python
from pypto.pypto_core import DataType, ir
from pypto.ir import op

span = ir.Span.unknown()
dim4, dim8 = ir.ConstInt(4, DataType.INT32, span), ir.ConstInt(8, DataType.INT32, span)

# Create tensors
tensor_a = ir.Var("a", ir.TensorType([dim4, dim8], DataType.FP32), span)
tensor_b = ir.Var("b", ir.TensorType([dim8], DataType.FP32), span)

# Simple operators
result = op.tensor.add(tensor_a, tensor_b)  # Broadcasting: [4,8] + [8] → [4,8]

# Operators with kwargs
dim64, dim128 = ir.ConstInt(64, DataType.INT32, span), ir.ConstInt(128, DataType.INT32, span)
a = ir.Var("a", ir.TensorType([dim64, dim128], DataType.FP16), span)
b = ir.Var("b", ir.TensorType([dim128, dim64], DataType.FP16), span)
matmul = op.tensor.matmul(a, b, out_dtype=DataType.FP32, a_trans=True)

# Query registry
assert ir.is_op_registered("tensor.add")
op_instance = ir.get_op("tensor.add")
```

## Kwargs (Keyword Arguments)

Call expressions separate Expr arguments from metadata parameters using kwargs.

### Kwargs vs Args vs Attributes

| - | **Args** | **Kwargs** | **Op Attributes** |
| - | -------- | ---------- | ----------------- |
| **Type** | `ExprPtr` | `std::any` | Type-erased |
| **Scope** | Per-Call | Per-Call | Global |
| **Use** | Tensors, dims, offsets | `out_dtype`, flags, modes | Device, category |
| **Access** | `call.args_` | `call.kwargs_` | `op.get_attr()` |

### C++ - Reading Kwargs

```cpp
TypePtr DeduceCastType(const std::vector<ExprPtr>& args,
                       const std::vector<std::pair<std::string, std::any>>& kwargs) {
  auto input = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());

  // `kwargs` is a vector of pairs, not a map — scan it to look a key up.
  auto find_kwarg = [&kwargs](const std::string& key) {
    return std::find_if(kwargs.begin(), kwargs.end(),
                        [&key](const auto& kv) { return kv.first == key; });
  };

  // Required kwargs — `cast` declares both `target_type` and `mode`, and codegen
  // reads `mode` unconditionally, so a missing one must fail here rather than
  // silently default to round_mode NONE.
  auto it = find_kwarg("target_type");
  CHECK(it != kwargs.end()) << "tensor.cast requires 'target_type'";
  DataType target = static_cast<DataType>(std::any_cast<int>(it->second));

  CHECK(find_kwarg("mode") != kwargs.end()) << "tensor.cast requires 'mode'";

  return std::make_shared<TensorType>(input->shape_, target);
}
```

A genuinely optional kwarg (one codegen reads with a fallback, such as `tile.log`'s
`high_precision`) is read via `Call::GetKwarg<T>(key, default_value)` instead of a
`CHECK` — see `include/pypto/ir/expr.h`.

### Python - Using Kwargs

```python
result = op.tensor.matmul(a, b, out_dtype=DataType.FP32, a_trans=True)
print(result.kwargs)  # {'out_dtype': 51, 'a_trans': True}
```

## Broadcasting and Type Promotion

### NumPy-style Broadcasting

Dimensions aligned right to left:

```text
[4, 8] + [4, 8] → [4, 8]  # Exact match
[4, 8] + [8]    → [4, 8]  # Missing left dimension = 1
[4, 1] + [8]    → [4, 8]  # Size 1 broadcasts
[1, 8] + [4, 8] → [4, 8]  # Size 1 broadcasts
[4, 8] + [5]    → Error   # 8 ≠ 5
```

### Type Promotion

Standard numeric rules: float > int, larger > smaller, signed > unsigned (same size).

```text
INT32 + INT32 → INT32
INT32 + FP32  → FP32   (float precedence)
INT32 + INT64 → INT64  (larger size)
UINT32 + INT32 → INT32 (signed precedence)
```

## TensorOp: N-Dimensional Tensor Operations

**Purpose**: General N-dimensional tensors with full broadcasting
**Type**: `TensorType` (arbitrary dimensions)
**Location**: `src/ir/op/tensor_ops/`
**Python API**: `from pypto.ir.op import tensor`

**Operations:** `tensor.add/sub/mul/div` (element-wise with full N-D broadcasting), `tensor.maximum/minimum` (element-wise max/min; rhs may be tensor or scalar — `ConvertTensorToTileOps` dispatches to `tile.maximum/minimum` or `tile.maximums/minimums` based on the rhs operand type), `tensor.set_validshape` (update valid-shape metadata without data movement; also reachable as `pl.set_validshape`), `tensor.sort32` / `tensor.mrgsort_format1` / `tensor.mrgsort_format2` (sorting; tensor-level counterparts of `tile.sort32` / `tile.mrgsort` — converted to tile ops by `ConvertTensorToTileOps`), `tensor.gather` (per-dim indexing; MVP supports rank-2 inputs with `dim=-1`, lowered by `ConvertTensorToTileOps` with a backend-specific strategy — on A5 (Ascend950) a last-dim gather becomes a single full-tile `tile.gather` over flat element offsets `flat[i, j] = i * src_cols + index[i, j]`, first materializing a strided tile source (e.g. a `tile.slice` view) into a contiguous tile so the flat index addresses it correctly; on A2A3 (Ascend910B) it keeps the legacy per-row `tile.gather` loop where the column index equals the flat index within each 1-row slice), `tensor.gather_mask` (mask-pattern gather; tensor-level counterpart of `tile.gather_mask`, with optional same-bit-width `output_dtype` — see [Mask patterns](#mask-patterns)), `tensor.scatter` (column scatter; the column-wise inverse of `tensor.gather`, MVP supports rank-2 inputs with `dim=-1` — `out[b, index[b, k]] = src[b, k]`, `index` same shape as `src` — and lowers to `tile.scatter` via `ConvertTensorToTileOps`), `tensor.scatter_mask` (mask-pattern row-scatter; tensor-level counterpart of `tile.scatter_mask`, expands a compact `input` tensor into the mask-marked columns of `dst` — see [Mask patterns](#mask-patterns)), `tensor.ci` / `tensor.arange` (contiguous integer sequence generation; lowers to `tile.ci`; also exposed at top level as `pl.arange`), `tensor.and/ands/or/ors/xor/xors/not/shl/shls/shr/shrs` (integer-only bitwise and shift ops. These are the registered *IR* names; the Python spellings for the three whose leaf is a Python keyword carry a trailing underscore -- `tensor.and_`, `tensor.or_`, `tensor.not_` -- and the printer emits that form so IR round-trips as valid Python; tensor-level counterparts of the matching `tile.*` ops. Both operands of a tensor-tensor form must have the same shape — there is no `tile.row_expand_and`, so broadcasting is rejected at type deduction rather than failing later in the pass. `tensor.not` is int16/uint16 only, matching `tile.not`/TNOT. Shifts keep the lhs element type; `and`/`or`/`xor` promote across integer widths, as their tile counterparts do. `ConvertTensorToTileOps` lowers nine of them 1:1, and synthesizes the `pto.txor` scratch operand for `tensor.xor`/`tensor.xors` so tensor-level callers never supply a `tmp`)

`tensor.view` is a metadata-only zero-copy shape/layout reinterpret. It is registered as a `TensorOp` passthrough in `ConvertTensorToTileOps`; PTO in-core codegen lowers it to `pto.make_tensor_view` over the original base pointer. Targets require rank at least 1 (DN requires rank at least 2); orchestration shape reinterpret is ND-only and cannot also change layout. Shape reinterpretation of a partially valid source is limited to either a packed ND leading-dimension collapse to 2D or a contiguous-prefix linear collapse to `[1, product(shape)]`; both require an explicit target `valid_shape`. These forms preserve the source tensor kind and backing metadata.

For plain `TensorType` operands, the supported Tensor-scalar arithmetic
operators (`adds`, `subs`, `muls`, `divs`, `fmods`, and scalar `maximum` or
`minimum`) and bitwise/shift operators (`ands`, `ors`, `shls`, and `shrs`)
create fresh storage but cannot create valid data in padding. Their results
therefore preserve the tensor operand's effective `valid_shape` while dropping
source alias, layout, stride, and padding metadata. This matches the existing
Tile-scalar rule and keeps a ragged tail narrow through Tensor-to-Tile lowering.
Scalar comparison and XOR (`cmp` and `xors`) remain excluded.

The ordinary arithmetic Tensor-tensor operators (`add`, `sub`, `mul`, `div`,
`fmod`, `maximum`, and `minimum`) also preserve the effective `valid_shape`
when both operands have identical physical shapes and their effective valid
regions are provably equal. The same exact-region rule applies to `and`, `or`,
`shl`, and `shr`. It needs no broadcast-axis mapping and agrees with the
corresponding Tile result contract; the result remains fresh storage and
therefore inherits no alias, layout, stride, or padding metadata. Comparison,
XOR, `part_*`, broadcasting, different valid regions, and direct distributed
window operands are not covered by this rule because their current lowering or
combination contracts require separate handling.

`pl.reinterpret_view(data, dtype, *, shape=None)` dispatches to the equivalent `pl.tensor` or `pl.tile` operator and returns the same kind. It is a zero-copy view over exactly the same bytes, so `dtype` must differ and be one of signed/unsigned 8/16/32/64-bit integers, FP16, BF16, or FP32. With no `shape`, ND/row-major scales the last axis and DN/col-major scales the penultimate axis by the source/target byte-width ratio. An explicit shape must be byte-equivalent and fully static unless it is provably identical to the auto-inferred shape; a partial `valid_shape` only permits that auto-equivalent shape. Zero/null padding metadata is preserved, while dtype-dependent max/min padding is cleared. The initial executable path supports packed ND in-core tensors and packed flat (`none_box`) row/col-major tiles; DN tensor inference is available but Tensor-to-Tile lowering rejects it, and orchestration tensors are unsupported.

**Example:**

```python
from pypto.ir.op import tensor

ib = IRBuilder()
with ib.function("tensor_example") as f:
    input_a = f.param("input_a", ir.TensorType([128, 64, 32], DataType.FP32))
    input_b = f.param("input_b", ir.TensorType([128, 64, 32], DataType.FP32))
    f.return_type(ir.TensorType([128, 64, 32], DataType.FP32))
    result = ib.let("result", tensor.add(input_a, input_b))
    ib.return_stmt(result)
```

## TileOp: Hardware-Optimized Tile Operations

**Purpose**: Hardware-optimized tile operations with explicit memory management
**Type**: `TileType` (tiles in unified buffers)
**Location**: `src/ir/op/tile_ops/`
**Python API**: `from pypto.ir.op import tile`

**Design**: Uses `TileType` (not separate `BlockType`) for consistency. Namespace `tile.*` + `TileType` clearly indicates hardware-optimized tile operations.

### Operations

| Category | Operations | Description |
| -------- | ---------- | ----------- |
| **Memory** | `tile.get_block_idx` | Get hardware block index (→ ScalarType(DataType::UINT64)) |
| - | `tile.load` | TensorType → TileType (DDR to unified buffer) |
| - | `tile.store` | TileType → TensorType (unified buffer to DDR) |
| - | `tile.move` | Move a tile between memory spaces (`target_memory`) — see [Result view of tile.move](#result-view-of-tilemove) |
| **Element-wise** | `tile.add/sub/mul/div` | Tile-Tile operations |
| - | `tile.adds/subs/muls/divs` | Tile-Scalar operations. A **constant** scalar operand adopts the tile's element dtype (a bare int literal is otherwise parsed as `index`, which no `pto.t*s` op accepts) — except a float literal on an integer tile, which keeps FP32 so promotion is preserved. An explicit `pl.const(v, dtype)` is a deliberate annotation and is left as-is, as is any non-constant expression; a non-constant `index` scalar (loop var, `pl.dim`) is rejected — convert it with `pl.cast`. Same rule for `tensor.*s`. |
| **Unary** | `tile.sqrt` | Element-wise square root |
| **Transform** | `tile.slice` | Extract a sub-tile with static shape, optional dynamic valid_shape, and optional `drop_dims` (numpy-style rank reduction over static unit axes; result clamped to a 2D minimum) |
| - | `tile.extract` | Extract a sub-tile from `src` at `(index_row, index_col)` — ISA TEXTRACT Variant 1 (Mat→Left/Right, Acc→Mat). The result's layout comes from `target_memory`'s implicit view, except `Left`/`Right`, which take the TEXTRACT-side L0 formats (these differ from `tile.move`'s TMOV-side ones) |
| - | `tile.reshape` | Reshape tile to new dimensions (element count must match). Carries the source's `valid_shape` through without widening it — see [Reshape and the valid region](#reshape-and-the-valid-region) |
| - | `tile.reinterpret_view` | Zero-copy view with a different dtype and the same exact bytes; optional shape uses layout-aware inference (packed flat tiles only) |
| - | `tile.transpose` | Swap two axes of a tile |
| - | `tile.set_validshape` | Update valid-shape metadata without data movement |
| - | `tile.ci` | Generate contiguous integer sequence (start + k / start - k); dtype ∈ {INT16, INT32}; innermost dim != 1 |
| - | `tile.tri` | Generate a lower/upper triangular 0/1 mask with an INT32 diagonal offset; supports an optional partial `valid_shape`; maps to `pto.ttri`. |
| **Reduction** | `tile.row_*` / `tile.col_*` | Direction-specific reduction (`row_sum`/`row_max`/`row_min`/`row_prod` collapse the last axis; `col_*` collapse axis 0). There is no axis-parameterized reduction — the ISA has only direction-specific intrinsics (`pto.trowsum`, `pto.tcolsum`, …) |
| **Gather** | `tile.gatherb` | Gather 32-byte source blocks. Each UINT32 offset selects one block; each offset column expands to `32 / sizeof(output_dtype)` output elements, and valid shape expands identically. `output_dtype` defaults to the source dtype and may select another supported byte interpretation. Offset rows contain a positive multiple of eight entries. A sliced source must have a byte address provably aligned to 32 bytes; dynamic column offsets are rejected, while dynamic row offsets remain valid when the physical row stride preserves alignment. Maps to `pto.tgatherb`. |
| - | `tile.mgather` | Gather from a GM tensor into a fresh Vec or Mat tile. Vec output uses an INT32 index tile (`[1,R]`, or A5 `[R,1]`); Mat output uses ND-layout GM source and INT32 index tensors plus canonical NZ layout, with physical rows aligned to 16 and columns aligned to `C0 = 32 / sizeof(dtype)`. Mat output accepts a smaller 2D `valid_shape` for padded tails. `coalesce="row"` gathers complete rows, while `"elem"` flat-indexes elements and requires a same-dtype, contiguous-ND GM `scratch` tensor with at least as many elements as the physical output. `gather_oob` selects `undefined`, `clamp`, `wrap`, or `zero`. Payload dtypes are I8/U8/I16/U16/I32/U32/FP16/BF16/FP32, plus the A5-only FP8E4M3FN/FP8E5M2/HF8 forms. |
| **Scatter** | `tile.scatter` | Row-scatter `src` into `dst` at per-row indices (`pto.tscatter` index form; DPS — `dst` is in/out, the result aliases `dst`). `src`/`dst` dtype ∈ {I8, I16, I32, FP16, FP32, BF16}; `indexes` dtype ∈ {I16, I32}; element-size matching rule: 4-byte dst ↔ INT32, 2-byte dst ↔ INT16, 1-byte dst ↔ INT16. |
| - | `tile.scatter_mask` | Mask-pattern row-scatter: write each `src` row into the mask-marked columns of `dst` (DPS — `dst` is in/out). A PyPTO codegen form lowered to a `pto.tscatter` mask emission — **not** a distinct pto-isa instruction (unlike `tile.gather_mask`). See [Mask patterns](#mask-patterns). |

`tile.reshape` preserves dtype, element count, and the source's valid region (see below); `tile.reinterpret_view(data, dtype, *, shape=None)` changes dtype while preserving exact byte size. Without `shape`, it scales the physically contiguous axis using the source/target dtype byte widths and tile layout. Under PTOAS memory planning, it lowers to the aliasing PTO `treshape` primitive for both same-shape and width-changing views.

### Result view of `tile.move`

The deduced result `TileView` splits by field:

| Field | Source of the result value |
| ----- | -------------------------- |
| `blayout` / `slayout` | The **destination** space's implicit layout wherever it has one of its own (`Mat`, `Acc`, `Left`, `Right`, `LeftScale`, `RightScale`); for the flat spaces (`Vec`, `Bias`, …) the source tile's effective layout carries over. A `blayout` / `slayout` kwarg overrides either |
| `fractal` | The **destination** space's boxing granularity, never the source's: `Acc` (L0C, NZ-boxed) is 1024, MX scale tiles are 32, everything else 512 |
| `valid_shape` / `pad` | Carried over from the source |
| `stride` / `start_offset` | Dropped — the destination is a dense buffer |

The layout comes from the destination because it describes how that buffer is
boxed; `tile_view_semantics::GetImplicitTileLayout` supplies it. `Right` needs a
local override — L0B requires `blayout=row_major` even for an `[N, 1]` shape,
whose implicit `blayout` is `col_major`.

`tile.move` stamps the destination `memory_space` itself (see the `TileType`
contract in [Types](02-types.md#tiletype)), so a result view matching the
destination's implicit view collapses to `nullopt` — the same per-space view
[`InferTileMemorySpace`](../passes/18-infer_tile_memory_space.md) refreshes a
retyped tile to.

`tile.move` is not in-place safe: within one memory space, its source and result
must resolve to distinct addresses. The PyPTO and DSA-RP planners enforce this
constraint, and baked-address PTO codegen reports an error if an explicit
MemRef binding or hand-built IR still presents a same-address move.

### Reshape and the valid region

A reshape is a zero-copy view, so it cannot invent data: `tensor.reshape` and
`tile.reshape` share one rule that carries the source's `valid_shape` into the
target shape and never widens it. A valid region is an origin-anchored box, so
not every source region survives a repartition — the rule maps what it can:

| Source region | Result |
| ------------- | ------ |
| Fully valid | `new_shape` — canonicalized away, so no view survives and no existing program changes |
| Provably empty | An all-zero box |
| Only full unit axes added / removed | Surviving axes map 1:1; an arbitrary rectangle is preserved exactly |
| A contiguous flat prefix | The rectangle of `new_shape` spanning those same cells, if one exists |
| Anything else | **Rejected** — `valid_shape` cannot describe the reshaped region |

So `[8, 16]` valid `[5, 16]` (a flat prefix of 80 cells) maps to `[16, 8]` valid
`[10, 8]` and to `[128]` valid `[80]`, while `[4, 32]` is rejected — 80 cells is
not a whole number of 32-wide rows. `[1, 8, 16]` valid `[1, 8, 5]` is not a flat
prefix at all, yet `[8, 16]` valid `[8, 5]` is exact, because dropping a full
unit axis keeps rows as rows. `tensor.reshape`'s optional third `valid_shape`
operand may only *narrow* the derived region, never claim data outside it.

An **identity** `tile.reshape` — one whose target shape equals the source's —
additionally keeps the source's layout triple (`blayout` / `slayout` / `fractal`) and its
resolved memory space, instead of re-deriving the layout from the shape. Re-deriving
yields the space-agnostic flat layout, which `NormalizeImplicitTileView` rescues only for
a view that collapses; an Acc box that is narrowed, padded, or declared `compact` never
collapses, so the flat layout would stick and its reader would walk L0C as a plain
row-major buffer (issue #2470).

**Data Flow:** `TensorType (DDR) → tile.load → TileType (Unified Buffer) → tile.{ops} → TileType → tile.store → TensorType (DDR)`

### Mask patterns

`*.gather_mask` / `*.scatter_mask` use a compile-time `MaskPattern` (`pl.tile.MaskPattern`, integer values 1–7, matching the hardware `VREDUCEv2` pattern modes) to mark a per-row subset of columns (names read **right-to-left**, rightmost bit = column 0). The same mark set drives the two ops in opposite directions. **`gather_mask`** *selects & compacts*: it reads the marked columns of a wide input into the leading columns of a narrower output (`out_cols = cols / stride`); this is a real pto-isa instruction (`pto.tgather` mask form), supported on A2/A3 **and A5**. **`scatter_mask`** *places & expands*: it writes a compact input into the marked columns of a wider `dst` (`dst_cols = cols * stride`), leaving unmarked columns at their prior `dst` value (DPS); this is a **PyPTO codegen-level form, not a distinct pto-isa instruction** — there is no `pto.tscatter` mask instruction (unlike gather) — and PyPTO emits it for A2/A3 / CPU-sim style lowering paths. E.g. for `[a0 a1 a2 a3 a4 a5 a6 a7]`: gather `P0101 → [a0 a2 a4 a6]`; scatter of `[s0 s1 s2 s3]` `P0101 → [s0 · s1 · s2 · s3 ·]` (`·` = preserved `dst`).

| Pattern | int | Marks column `c` when | Marked columns | Stride |
| ------- | --- | --------------------- | -------------- | ------ |
| `P0101` | 1 | `c % 2 == 0` | 0, 2, 4, … | 2 |
| `P1010` | 2 | `c % 2 == 1` | 1, 3, 5, … | 2 |
| `P0001` | 3 | `c % 4 == 0` | 0, 4, 8, … | 4 |
| `P0010` | 4 | `c % 4 == 1` | 1, 5, 9, … | 4 |
| `P0100` | 5 | `c % 4 == 2` | 2, 6, 10, … | 4 |
| `P1000` | 6 | `c % 4 == 3` | 3, 7, 11, … | 4 |
| `P1111` | 7 | always | all | 1 |

The last dim must be divisible by the stride. `gather_mask` also accepts an optional same-bit-width `output_dtype` (bit-reinterpret, not a value cast). Reference: gather selection is `MaskSelect` in `pto-isa` `include/pto/cpu/TGather.hpp`; pypto type deduction in `src/ir/op/tile_ops/gather.cpp` (gather) / `src/ir/op/tile_ops/scatter.cpp` (scatter).

### Example Usage

```python
from pypto.ir.op import tile

ib = IRBuilder()
with ib.function("tile_computation") as f:
    input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
    input_b = f.param("input_b", ir.TensorType([128, 128], DataType.FP32))
    output = f.param("output", ir.TensorType([128, 1], DataType.FP32))
    f.return_type(ir.TensorType([128, 1], DataType.FP32))

    # Load, compute, reduce, store
    tile_a = ib.let("tile_a", tile.load(input_a, [0, 0], [32, 128]))
    tile_b = ib.let("tile_b", tile.load(input_b, [0, 0], [32, 128]))
    tile_mul = ib.let("tile_mul", tile.mul(tile_a, tile_b))
    tile_sqrt = ib.let("tile_sqrt", tile.sqrt(tile_mul))
    # row_sum collapses the last axis -> [32, 1]. Its scratch tile must have
    # the same dtype and rank and be at least as large as the input in every dimension.
    tmp_tile = ib.let("tmp_tile", tile.create([32, 128], DataType.FP32))
    tile_sum = ib.let("tile_sum", tile.row_sum(tile_sqrt, tmp_tile))
    result = ib.let("result", tile.store(tile_sum, [0, 0], output))
    ib.return_stmt(result)
```

## SyncOp: Synchronization Operations

**Purpose**: Hardware synchronization and barriers, plus the TaskId and SPMD launch-shape queries sharing the `system.` namespace
**Type**: `UnknownType` (no return, used in `EvalStmt`) for barriers; `ScalarType` for the value-binding query ops (`task_invalid`, `task_is_valid`, `available_cluster_count`, `available_aiv_count`)
**Location**: `src/ir/op/sync_ops/` — `sync.cpp` (barriers), `task.cpp` (TaskId), `launch.cpp` (launch-shape queries)
**Python API**: `from pypto.ir.op import system`

| Operation | Description | Kwargs |
| --------- | ----------- | ------ |
| `system.bar_all` | Global barrier (lowers to `pto.barrier <PIPE_ALL>`) | None |
| `system.bar_v` | Vector barrier (lowers to `pto.barrier <PIPE_V>`) | None |
| `system.bar_m` | Matrix barrier (lowers to `pto.barrier <PIPE_M>`) | None |
| `system.fence` | Memory barrier over global memory (lowers to `pto.fence.barrier_all #pto.fence_scope<gm>`) | None |
| `system.cacheinvalid` | Invalidate the cache line containing a tensor sub-region's base address. Args: `tensor`, `shapes` (N-D), `offsets` (N-D). Every region size — a single element included — lowers to `pto.partition_view` + `pto.cmo.cacheinvalid %payload_view single_cache_line : !pto.partition_tensor_view<...>`; `shapes` does not make it walk every cache line. The no-argument form invalidates all GM. | None |
| `system.syncall` | Cross-core all-participant barrier (`pto::SYNCALL`). Attr `mode` `"hard"` (FFTS, no operands) or `"soft"` (GM-polling, operands) | `core_type` (`"aiv_only"` \| `"aic_only"` \| `"mix"`), `mode` (`"hard"` \| `"soft"`) |
| `system.sync_src` | Set sync flag | `set_pipe`, `wait_pipe`, `event_id` |
| `system.sync_dst` | Wait sync flag | `set_pipe`, `wait_pipe`, `event_id` |
| `system.task_invalid` | Sentinel `TaskId::invalid()` — "no producer" seed for a TaskId carry | None |
| `system.task_is_valid` | Test whether a `TASK_ID` value is a valid (non-sentinel) handle | None; sole positional arg is the TaskId Var |
| `system.available_cluster_count` | This run's MIX cluster (= AIC) count, read from the device. Result `Scalar[INT32]` | None |
| `system.available_aiv_count` | This run's standalone AIV core count, read from the device. Result `Scalar[INT32]` | None |

`system.syncall` has two modes, selected by its `mode` **IR attribute**; the Python surfaces spell them as `pl.SyncAllMode` members instead (see below). The **hard** form (attr `"hard"`, the default) emits an FFTS barrier that waits for **all** physical cores of the selected `core_type`; the kernel must be launched at full occupancy (one block per physical core) **and with `sync_start=True`** (so all blocks are co-resident — a non-sync_start launch may dispatch blocks in waves and deadlock the barrier), or it deadlocks (AICore error 507018). The **soft** form (attr `"soft"`) polls a shared GM workspace and so works at **partial** occupancy. `gm_workspace` is a shared, zero-initialized GM `INT32` tensor containing at least 16 elements (64 bytes). Pass it as a kernel parameter so all blocks share one buffer; it must occupy an exclusive cache line and be zero-initialized before its first use.

The current PTO-ISA uses the same soft operand ABI for every `core_type`: `[gm_workspace]` derives the participant count from the device launch configuration, while `[gm_workspace, used_cores]` supplies it explicitly as a Python integer in the INT32 range or an `INT32` scalar. The high-level DSL requires `used_cores` to make that choice explicit: pass a positive count for the two-operand form, or explicitly pass `0` for the one-operand form. For `mix`, an explicit count is the total number of AIC and AIV participants. Runtimes whose logical grid differs from the device launch registers must use a positive explicit count; this includes the currently pinned Simpler runtime. No UB/L1 scratch tile is required.

Both modes guarantee barrier arrival only. They do not wait for preceding data instructions such as `TSTORE`, and they do not publish or invalidate business-data cache lines. For a cross-core GM handoff that may span multiple cache lines, conservatively publish the producer's writes with whole-GM `system.cacheinvalid()` and `system.fence` before the barrier, then use whole-GM `system.cacheinvalid()` on the consumer before it reads. The tensor-region form invalidates only the cache line containing the view's base address.

The `core_type` and `mode` attributes stay strings **in the IR**, but the Python surfaces are enum-typed: `pl.KernelType` (`AIC` / `AIV` / `MIX`, naming which generated kernel the op belongs to) and `pl.SyncAllMode` (`HARD` / `SOFT`). Only members are accepted: the lowered attr spelling is an output, not an input, so passing one is a `TypeError`, and a member outside an op's own domain is a `ValueError`. The unified `mode=` keyword API is the **DSL** surface (`pl.system.syncall`). The Python IR helpers under `pypto.ir.op.system` are split instead: `syncall(core_type=...)` builds the hard form and `syncall_soft(core_type, gm_workspace, used_cores=None)` builds the soft form.

`system.available_cluster_count` / `system.available_aiv_count` are the SPMD **launch-shape queries**: pass one as `pl.spmd(...)`'s `core_num` so the launch sizes itself on the device the run lands on. Orchestration codegen lowers them to `rt_available_cluster_count()` / `rt_available_aiv_count()`. Use the cluster count for a mixed (AIC+AIV) or cube-only kernel — one block per core-group — and the AIV count for a vector-only kernel. This is the only launch width that stays at full occupancy across devices, which the hard `system.syncall` requires; the `HardSyncallOccupancy` verifier accepts these widths without a count comparison and rejects the query for the *other* core type. Pass the call inline (`pl.spmd(pl.system.available_cluster_count())`) rather than binding it to a name first — a name reaches the outlined `Spmd` wrapper as a variable defined in the caller, which the IR printer cannot re-parse. Source: `src/ir/op/sync_ops/launch.cpp`.

`system.task_invalid` returns [`ScalarType(DataType::TASK_ID)`](02-types.md#scalartype). It is the lowering target of the Python literal `None` when `None` appears in a TaskId position (a `deps=[None]` entry or a TaskId loop iter_arg seed) inside `with pl.manual_scope():` regions. There is no `system.task_id_of` op — producer task ids are obtained from the second tuple element returned by the `pl.submit(...)` parser construct, not from a builtin. Source: `src/ir/op/sync_ops/task.cpp`.

## CrossCoreOp: AIC↔AIV Communication

**Purpose**: Cross-core synchronization, data transfer, and pipe management between AIC (Cube) and AIV (Vector) kernels
**Type**: `UnknownType` (sync/push/init/buffer/free ops) or `TileType` passthrough (pop ops)
**Location**: `src/ir/op/tile_ops/cross_core.cpp` (tpush/tpop) and `src/ir/op/sync_ops/cross_core.cpp` (sync/tfree/pipe init/buffers)
**Python API**: `import pypto.language as pl` (promoted ops) or `from pypto.ir.op import tile, system`

### Explicit Event Synchronization

| Operation | Args | Description | Kwargs |
| --------- | ---- | ----------- | ------ |
| `system.sync_set` | 0 or 1 (`event_id_dyn`) | Emit `pto.sync.set` from one core type | `pipe`, static `event_id`, optional `ffts_mode`, optional `core_type` |
| `system.sync_wait` | 0 or 1 (`event_id_dyn`) | Emit `pto.sync.wait` on the peer core type | `pipe`, static `event_id`, optional `core_type` |
| `system.set_ffts` | 1 (`workspace`) | Declare the A3 FFTS setup required by explicit cross-core events | — |

Use `pl.system.sync_set(event_id, pipe=..., ffts_mode=...)` and `pl.system.sync_wait(event_id, pipe=...)` in explicitly typed AIC/AIV kernels. In a mixed InCore kernel, pass `core_type=pl.KernelType.AIV` or `core_type=pl.KernelType.AIC` to retain each event operation on the intended lane when the kernel is expanded (the IR attr keeps the lowered `"aiv"` / `"aic"` spelling, which is an output of the API, not an accepted input). `pl.KernelType.MIX` is rejected here — an event pins one lane, and both-lane placement is spelled by omitting `core_type`. Both `system.syncall` and the event ops feed the same `KernelType` classification in `ClassifyCallAffinity`; only their IR attr vocabularies differ (`"aic_only"` vs `"aic"`). On A3, call `pl.system.set_ffts(workspace)` in every participating AIC/AIV function before its first explicit event operation; `workspace` must be a one-dimensional `INT64` tensor with at least 256 elements and acts as the PTOAS setup operand. PyPTO's persistent runtime keeps the hardware FFTS control address installed, so generated runtime wrappers do not replace it with this operand. A5 does not require this setup. `event_id` may be an integer in the user-available range 0–13 or a dynamic `pl.Scalar[pl.INDEX]`; IDs 14 and 15 are reserved. `ffts_mode`, when supplied to `sync_set`, must be 0, 1, or 2. The author of a manual cross-core protocol is responsible for pairing event IDs and pipes. PyPTO's normal automatic intra-core dependency insertion remains enabled and uses the separate `set_flag`/`wait_flag` mechanism, so it does not allocate from these explicit cross-core event IDs.

### Data Transfer Operations

| Operation | Args | Description | Kwargs |
| --------- | ---- | ----------- | ------ |
| `tile.tpush_to_aiv` | 1 (tile) | Push tile from Cube to Vector | `split`, optional `id` |
| `tile.tpush_to_aic` | 1 (tile) | Push tile from Vector to Cube | `split`, optional `id` |
| `tile.tpop_from_aic` | 0 | Pop tile from Cube pipe (→ TileType) | `split`, optional `id` |
| `tile.tpop_from_aiv` | 0 | Pop tile from Vector pipe (→ TileType) | `split`, optional `id` |
| `system.tfree_to_aic` | 1 (tile) | Release slot back to Cube producer | optional `id` |
| `system.tfree_to_aiv` | 1 (tile) | Release slot back to Vector producer | optional `id` |

### Pipe Initialization Operations

| Operation | Args | Description | Kwargs |
| --------- | ---- | ----------- | ------ |
| `system.aic_initialize_pipe` | 2 | Init cross-core pipe on Cube side (positional: `c2v_consumer_buf`, `v2c_consumer_buf`, i32 SSA) | `dir_mask`, `slot_size`, optional `slot_num`, optional `local_slot_num`, optional `id` |
| `system.aiv_initialize_pipe` | 2 | Init cross-core pipe on Vector side (positional: `c2v_consumer_buf`, `v2c_consumer_buf`, i32 SSA) | `dir_mask`, `slot_size`, optional `slot_num`, optional `local_slot_num`, optional `id` |

- `slot_num` (when set, must be > 0) pins the GM ring-buffer slot count; omit it to let PTOAS pick its default (8 unidirectional, 4 per direction bidirectional).
- `local_slot_num` (a2/a3 only, must be > 0 and `<= slot_num`) pins the local slot count.
- **Sizing the reserved/imported buffer is your responsibility and is architecture-dependent:** on **a3** use `slot_size * local_slot_num`; on **a5** use `slot_size * slot_num`.

### Buffer Management Operations

| Operation | Args | Description | Kwargs |
| --------- | ---- | ----------- | ------ |
| `system.reserve_buffer` | 0 | Reserve named cross-core buffer (consumer side) | `name`, `size`, `base`* |
| `system.import_peer_buffer` | 0 | Import buffer from peer function (producer side) | `name`, `peer_func` |

\* `base` defaults to `AUTO (-1)` for compiler-assigned address.

### DSL Example (cross-core V2C unidirectional)

`dir_mask=2` enables V2C only, so the C2V buffer operand must be an inactive-direction placeholder (`0`, or `pl.const(0, pl.INT32)`); the active side passes the reserved/imported buffer handle as the first positional operand.

```python
import pypto.language as pl

@pl.program
class CrossCoreExample:
    @pl.function(type=pl.FunctionType.InCore)
    def vector_producer(self, a: pl.Tensor[[16, 16], pl.FP16]):
        peer = pl.import_peer_buffer(name="v2c_buf", peer_func="cube_consumer")
        pl.aiv_initialize_pipe(pl.const(0, pl.INT32), peer, dir_mask=2, slot_size=512)

        tile_a: pl.Tile[[16, 16], pl.FP16] = pl.load(a, [0, 0], [16, 16])
        pl.tpush_to_aic(tile_a, split=0)

    @pl.function(type=pl.FunctionType.InCore)
    def cube_consumer(self, out: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
        buf = pl.reserve_buffer(name="v2c_buf", size=4096, base=0x1000)
        pl.aic_initialize_pipe(pl.const(0, pl.INT32), buf, dir_mask=2, slot_size=512)

        received: pl.Tile[[16, 16], pl.FP16] = pl.tpop_from_aiv(split=0)
        pl.tfree_to_aiv(received)
        result: pl.Tensor[[16, 16], pl.FP32] = pl.store(received, [0, 0], out)
        return result
```

See [TPUSH/TPOP ISA Reference](../../reference/pto-isa/01-tpush_tpop.md) and [Buffer Management](../../reference/pto-isa/02-buffer_management.md) for hardware details.

## PrefetchOp: Asynchronous GM→L2 Prefetch

A latency-hiding cache hint. `async_prefetch` starts an SDMA-backed pull of a
global-memory region into L2 while unrelated compute proceeds; `wait` blocks
until it lands. The prefetch changes no tensor values — a kernel is numerically
identical with or without it, so only performance differs.

Unlike most PTO intrinsics, `TPREFETCH_ASYNC` carries no implicit wait-event
synchronization, so completion is explicit via an event/session pair.

### Operations

| DSL | Operands | Result | PTOAS op |
| --- | -------- | ------ | -------- |
| `pl.prefetch.make_context()` | None | `PrefetchAsyncContextType` | `pto.make_prefetch_async_context` |
| `pl.prefetch.async_prefetch(src, ctx)` | GM Tensor, context | `AsyncEventType` | `pto.tprefetch_async` |
| `pl.prefetch.session(ctx)` | context | `AsyncSessionType` | `pto.get_prefetch_async_session` |
| `pl.prefetch.wait(evt, session)` | event, session | `BOOL` scalar | `pto.comm.wait_async_event` |

The three result types are opaque singleton markers (no shape, no buffer), in
the same family as `CommCtxType`. The SDMA workspace is not a program operand:
the runtime owns it, and codegen injects a hidden pointer into prefetch kernels.

### Constraints

- `src` must be a **flat contiguous logical-1D GM** region: a fully static shape
  whose dimensions are all `1` except the last (`[N]`, `[1, N]`, `[1, 1, N]`).
  This mirrors the PTOAS `TPrefetchAsyncOp::verify()` check, so a shape mistake
  fails at PyPTO IR construction rather than at PTOAS verification.

### Example Usage

```python
@pl.program
class PrefetchExample:
    @pl.function(type=pl.FunctionType.InCore)
    def main(
        self, x: pl.Tensor[[1, 4096], pl.FP32],
        out: pl.Tensor[[1, 128], pl.FP32],
    ) -> pl.Tensor[[1, 128], pl.FP32]:
        ctx = pl.prefetch.make_context()
        evt = pl.prefetch.async_prefetch(x, ctx)     # warms L2, does not block
        session = pl.prefetch.session(ctx)
        # ... unrelated compute overlaps the prefetch ...
        pl.prefetch.wait(evt, session)               # x is now resident in L2
        tile = pl.load(x, [0, 0], [1, 128])
        return pl.store(tile, [0, 0], out)
```

**Core placement**: this family is **AIV-only**. `TPREFETCH_ASYNC` drives its
SDMA `tmpBuf` from a Vec(UB) scratch tile held inside `PrefetchAsyncContext`
(pto-isa static_asserts `ScratchTile::Loc == TileType::Vec`), and UB lives on
the vector core. The ops declare `CoreAffinity::VECTOR`, so in a mixed kernel
`ExpandMixedKernel` keeps them on the vector lane — they are neither placed on
nor duplicated onto the cube lane.

**Runtime ownership and support**: normal one-shot execution reads the generated
artifact's SDMA requirement and automatically constructs an enabled worker. No
workspace appears in the user, orchestration, or runtime tensor signature. For
an explicitly reused L2 worker, opt in when constructing it:

```python
with ChipWorker(
    config=RunConfig(platform="a2a3", device_id=0), enable_sdma=True
):
    compiled(a, out, config=cfg)
```

The current runtime-provisioned execution path is covered only on onboard a2a3.
An enabled worker on simulator, a5, or another runtime without an SDMA provider
fails during runtime initialization. PyPTO does not allocate a fallback
workspace or silently turn a requested prefetch into a no-op. See
`tests/st/runtime/ops/test_prefetch_async.py` for the a2a3 system test.

## File Organization

| Directory/File | Contents |
| -------------- | -------- |
| `src/ir/op/type_inference.cpp` | Shared type inference utilities |
| `tensor_ops/elementwise.cpp` | TensorOp: add, sub, mul, div |
| `tile_ops/matmul.cpp` | TileOp: matmul, gemv |
| `tile_ops/matmul_mx.cpp` | TileOp: matmul_mx, matmul_mx_acc, matmul_mx_bias, internal tget_scale_addr binding |
| `tile_ops/memory.cpp` | TileOp: load, store, read, get_block_idx |
| `tile_ops/elementwise.cpp` | TileOp: add, mul, div, adds, muls, etc. |
| `tile_ops/reduction.cpp` | TileOp: sum (with axis, keepdim) |
| `tile_ops/unary.cpp` | TileOp: sqrt |
| `sync_ops/sync.cpp` | SyncOp: sync_src, sync_dst, barriers |
| `sync_ops/task.cpp` | SyncOp: TaskId sentinel and predicate |
| `sync_ops/launch.cpp` | SyncOp: SPMD launch-shape queries |
| `sync_ops/cross_core.cpp` | CrossCoreOp: tpush, tpop, pipe init, buffers |
| `prefetch/prefetch_async.cpp` | PrefetchOp: make_context, async_prefetch, session, wait |

**Benefits**:

- **Modularity**: Self-contained operator categories
- **Build Performance**: Changes to one category don't rebuild others
- **Maintainability**: Easy to locate and modify operators
- **Scalability**: Straightforward to add new operators

## Adding New Operations

1. **Choose category file**: `src/ir/op/tensor_ops/elementwise.cpp`, `matmul.cpp`, `reduction.cpp`, or `src/ir/op/tile_ops/memory.cpp`, `unary.cpp`

2. **Implement type deduction**:

   ```cpp
   TypePtr DeduceType(const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
     CHECK(args.size() == 2) << "op requires 2 arguments";
     // Validate types, read kwargs, compute output type
     return result_type;
   }
   ```

3. **Register**:

   ```cpp
   REGISTER_OP("tensor.matmul")
       .set_op_category("TensorOp")
       .add_argument("lhs", "Left tensor")
       .add_argument("rhs", "Right tensor")
       .set_attr<DataType>("out_dtype")
       .f_deduce_type(DeduceType);
   ```

4. **Python wrapper** (`python/pypto/ir/op/tensor_ops.py`):

   ```python
   def matmul(lhs: Expr, rhs: Expr, out_dtype=None, a_trans=False) -> Call:
       kwargs = {}
       if out_dtype: kwargs["out_dtype"] = out_dtype.code() if isinstance(out_dtype, DataType) else out_dtype
       if a_trans: kwargs["a_trans"] = a_trans
       return _ir_core.create_op_call("tensor.matmul", [lhs, rhs], kwargs, Span.unknown())
   ```

5. **Add tests** in `tests/ut/ir/` and update `CMakeLists.txt` if needed

## References

Core definitions live in `include/pypto/core/common.h` and `include/pypto/ir/`; registry and type-inference implementations are in `src/ir/`, with operator implementations grouped under `src/ir/op/{tensor_ops,tile_ops,sync_ops}/`.
