# Operator System

> **Legacy Python API examples:** `platform=` calls on this page are retained for
> compatibility. They emit `DeprecationWarning`; new code should use
> `backend_type` and `execution_mode`. See the [target mapping](../08-entry-points.md#legacy-platform-examples).

Type-safe operator definitions with automatic type deduction, organized into modular categories (TensorOp, TileOp, SyncOp, CrossCoreOp).

## Operator Categories

| Category | Types | Use Case | File Location |
| -------- | ----- | -------- | ------------- |
| **TensorOp** | TensorType | N-D tensor operations with broadcasting | `src/ir/op/tensor_ops/` |
| **TileOp** | TileType | Hardware-optimized tile operations | `src/ir/op/tile_ops/` |
| **BufferOp** | BufferType, VoidType | Internal explicit storage and destination writes | `src/ir/op/buffer_ops/` |
| **SyncOp** | UnknownType (barriers); ScalarType (task / launch queries) | Pipeline barriers, synchronization, TaskId and SPMD launch-shape queries | `src/ir/op/sync_ops/` |
| **CrossCoreOp** | UnknownType/TileType | AIC↔AIV cross-core communication | `src/ir/op/sync_ops/cross_core.cpp` |
| **PrefetchOp** | Opaque handles | Asynchronous GM→L2 cache prefetch | `src/ir/op/prefetch/prefetch_async.cpp` |

**Key Features**: Fluent API, automatic type deduction, kwargs for metadata, NumPy-style broadcasting, type promotion, dynamic dimensions (`kDynamicDim`)

The internal Buffer-stage GM and addition operations have no public DSL wrappers:

| Operation | Positional operands | Result |
| --------- | ------------------- | ------ |
| `buffer.load` | GM tensor, offsets tuple, valid extents tuple, destination buffer | Void |
| `buffer.store` | source buffer, offsets tuple, valid extents tuple, GM tensor | Void |
| `buffer.add` | lhs buffer, rhs buffer, destination buffer | Void |
| `buffer.copy` | source buffer, destination buffer (Vec -> Vec, or Mat -> Left/Right) | Void |
| `buffer.extract` | source buffer, row offset, column offset, destination buffer | Void |
| `buffer.matmul` | Left buffer, Right buffer, Acc destination | Void |
| `buffer.matmul_acc` | Left buffer, Right buffer, Acc buffer read and written | Void |
| `buffer.cast` | FP16/FP32 source, destination; explicit rounding/saturation attrs | Void |
| `buffer.cmp`, `buffer.cmps` | FP32 source, tile/scalar comparison operand, packed UINT8 destination | Void |
| `buffer.sel`, `buffer.sels` | packed predicate, value operands, explicit scratch, destination | Void |
| `buffer.row_max`, `buffer.row_sum` | FP32 matrix, explicit scratch, column-vector destination | Void |
| `buffer.row_expand_sub`, `buffer.row_expand_div` | FP32 matrix, column vector, matrix destination | Void |

These use separate data/metadata effects. See [Buffer contracts](02-types.md#buffer-operator-contracts)
for shape, dtype, valid-state, and alias requirements.

Internal `buffer.get_block_idx`, `buffer.get_block_num`, and
`buffer.get_subblock_idx` are zero-argument Buffer-stage queries. They return
an INDEX scalar (`BufferResultBehavior.Value`) with no execution memory access,
using the existing runtime-supplied SPMD kernel parameters. They are not public
DSL operations; lowering creates them from the corresponding Tile queries.

### Typed Buffer elementwise recipes

`backend/common/buffer_elementwise_recipes` is the shared production table for
logical-to-Buffer conversion, explicit operand contracts, and native mnemonics.
`LowerTileToBuffer` selects a typed recipe and creates a real Buffer call. Direct
PTO emission reads that same recipe; it does not reconstruct Tile IR, invoke
legacy callbacks, or choose storage. The table is deliberately separate from
legacy backend callback registration.

| Logical operation | Buffer operation | Native instruction | Inputs |
| ----------------- | ---------------- | ------------------ | ------ |
| `tile.add` | `buffer.add` | `pto.tadd` | 2 |
| `tile.mul` | `buffer.mul` | `pto.tmul` | 2 |
| `tile.sub` | `buffer.sub` | `pto.tsub` | 2 |
| `tile.div` | `buffer.div` | `pto.tdiv` | 2 |
| `tile.maximum` | `buffer.maximum` | `pto.tmax` | 2 |
| `tile.minimum` | `buffer.minimum` | `pto.tmin` | 2 |
| `tile.abs` | `buffer.abs` | `pto.tabs` | 1 |
| `tile.exp` | `buffer.exp` | `pto.texp` | 1 |
| `tile.sqrt` | `buffer.sqrt` | `pto.tsqrt` | 1 |
| `tile.neg` | `buffer.neg` | `pto.tneg` | 1 |
| `tile.relu` | `buffer.relu` | `pto.trelu` | 1 |
| `tile.log` | `buffer.log` | `pto.tlog` | 1 |
| `tile.recip` | `buffer.recip` | `pto.trecip` | 1 |
| `tile.adds` | `buffer.adds` | `pto.tadds` | buffer, scalar |
| `tile.subs` | `buffer.subs` | `pto.tsubs` | buffer, scalar |
| `tile.muls` | `buffer.muls` | `pto.tmuls` | buffer, scalar |
| `tile.divs` | `buffer.divs` | `pto.tdivs` | buffer, scalar |
| `tile.maximums` | `buffer.maximums` | `pto.tmaxs` | buffer, scalar |
| `tile.minimums` | `buffer.minimums` | `pto.tmins` | buffer, scalar |
| `tile.lrelu` | `buffer.lrelu` | `pto.tlrelu` | buffer, scalar |
| `tile.full` | `buffer.full` | `pto.texpands` | scalar |

Every call has its destination as the final operand and returns `Void`. Sources
read data and metadata; destinations write active data and read metadata. This
is not a whole-allocation initialization guarantee. All buffer operands need identical
physical descriptors; broadcasting and partial-combine semantics are separate
recipes. These operators are compiler-internal, not new public DSL functions.

Recipes other than `add` and `mul` require FP32, rank 1 or 2, static
valid extents, dense row-major Vec storage, `none_box`, fractal 512, and no padding
or compact mode. Their dtype contract does not inherit FP16 support from the
native descriptor formatter. `add`/`mul` require FP16/FP32/INT32 operands.
Automatic conversion and ordinary GM transfers support static rank-2
FP16/BF16/FP32/INT32 descriptors; BF16 arithmetic requires a separate native
recipe. Both Ascend910 and Ascend950 use these contracts.

The public `tile.full` wrapper accepts both numeric literals and parsed scalar
constants, so positional and keyword fill values agree. Integer placeholders
use `ConstFloat` for a floating destination; explicitly typed constants keep
their declared dtype until lowering. The Python printer emits `tile.full` fill
constants as `pl.const(value, dtype)` so print-to-parse roundtrips preserve the
fill dtype independently of the tile dtype. Runtime fill values remain invalid.

At the Buffer-call boundary, scalar operands must match the destination element
dtype and have no memory effects. Before constructing that call, lowering converts
supported source scalar expressions using explicit `Cast` expressions for signed integers,
INDEX, FP16, BF16 and FP32; INDEX uses an intermediate INT64 cast. Other
scalar source types remain unsupported. `tile.full(shape, dtype=..., value=...)`
consumes shape/dtype when selecting its destination, then emits
`buffer.full(value_f32, destination)`. Codegen performs no scalar type repair:

```python
# Logical input: result = tile.adds(value, count_i32)
scalar_f32 = ir.Cast(count_i32, DataType.FP32, span)
# Buffer IR diagnostic notation; the internal op has no public DSL wrapper.
buffer.adds(value_buffer, scalar_f32, result_buffer)  # -> Void
```

`div`, `log`, and `recip` preserve the optional boolean `high_precision` kwarg.
The recipe selects the corresponding native precision attribute; false uses
the native default. No implicit workspace, copy, or allocation is introduced.
`recip` requires source and destination storage to be disjoint. Construction
rejects one handle used in both positions. `BufferIR` verification checks placed
ranges across different handles for every table recipe: partial overlap is
rejected, and equal complete windows are accepted only by exact-in-place recipes.
Distinct addressless allocations are disjoint; unknown provenance or placement
requires a later recipe. Constant scalar address definitions are evaluated with
memoization and integer-width checks. Direct emission consumes this verified
contract without performing allocation or alias analysis.

Table recipes accept dense descriptors whose valid extents are static or
runtime (`-1`); the native instructions read each handle's current valid
metadata, which must agree across operands at execution time.

Integer/bitwise operations, `rsqrt` and its optional workspace,
partial combines, broadcasts, reductions, random generators and other layouts
remain separate migration work. The public
`backend.get_buffer_elementwise_recipe_names()` API returns a sorted independent
snapshot from this actual table. A listed name denotes the restricted recipe
above, not every form of that logical operator. Tests exercise public lowering,
binary round trips, BufferIR verification, explicit destination identities, and
native compilation; they do not substitute for numerical device acceptance.

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
| `set_output_arity(N)` | Values produced; `N > 1` means a `TupleType` result — see [Multi-Output Operators](09-multi_output_ops.md) | `.set_output_arity(2)` |
| `set_workspace_arg(i)` | Argument `i` is compiler-supplied scratch, not a result | `.set_workspace_arg(2)` |

### Argument effects

See [Operator effects](10-operator-effects.md) for argument access declarations,
write channels, and core placement/replication contracts. Buffer-stage effects
are described in [Buffer contracts](02-types.md#buffer-operator-contracts).

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

For two ordinary 2D tensors, `tensor.matmul` keeps physical output shape
`[lhs M, rhs N]` and infers `valid_shape=[lhs valid M, rhs valid N]`. Both use
the axis mapping selected by `a_trans`/`b_trans`. An omitted input valid shape
means its full shape; fully valid results retain the canonical view-free type.
Symbolic valid M/N expressions are preserved. The result owns fresh storage,
so input strides, layout, and padding metadata are not copied.

For example, physical `[32,128]` valid `[17,128]` times physical `[128,32]`
valid `[128,24]` gives physical `[32,32]` valid `[17,24]`. It can be assembled
into a `[17,24]` destination without `pl.set_validshape`. This infers metadata;
it does not add padding, clear memory, or extend hardware support for padded K.
Vector, batched, distributed-window, and `tensor.matmul_acc` inference remain unchanged.

For this 2D tensor form and for `tile.matmul`, the physical boxed K dimensions must match. PTO derives
the contraction extent from the lhs valid K, so that extent may be smaller than
the rhs valid K but must be contained by it. `tile.matmul_acc` likewise keeps
exact physical M/N/K box compatibility while allowing the accumulator's valid
M/N rectangle and the rhs valid K extent to contain the smaller rectangle PTO
computes from lhs M/K and rhs N.

#### Cube operand dtype pairs

The element types the Cube multiplies are a hardware table, not a symmetry rule:
`backend::kCubeOperandPairs` (`include/pypto/backend/common/cube_operand_types.h`)
is PyPTO's copy of PTOAS `verifyMatmulTypeTriple` and pto-isa `npu/a5/TMatmul.hpp`.

| lhs x rhs | Accumulator | A2/A3 | A5 | Families |
| --------- | ----------- | ----- | -- | -------- |
| INT8 x INT8 | INT32 | yes | yes | matmul, gemv |
| FP16 x FP16 | FP32 | yes | yes | matmul, gemv |
| BF16 x BF16 | FP32 | yes | yes | matmul, gemv |
| FP32 x FP32 | FP32 | yes | yes | matmul, gemv |
| FP8E4M3FN x FP8E4M3FN | FP32 | no | yes | matmul, gemv, mx |
| FP8E4M3FN x FP8E5M2 | FP32 | no | yes | matmul, gemv |
| FP8E5M2 x FP8E4M3FN | FP32 | no | yes | matmul, gemv |
| FP8E5M2 x FP8E5M2 | FP32 | no | yes | matmul, gemv |
| HF8 x HF8 | FP32 | no | yes | matmul, gemv |

So on Ascend950 `tile.matmul`, `tile.matmul_acc`, `tile.matmul_bias`,
`tile.batch_matmul*`, `tensor.matmul`, and `tensor.matmul_acc` accept **mixed FP8
operands in either order**; the accumulator, the `matmul_acc` input, and the
`matmul_bias` bias stay FP32. `tf32_mode` remains FP32-only, so an FP8 operand
pair rejects it.

The ordinary GEMV family — `tile.gemv`, `tile.gemv_acc`, `tile.gemv_bias` — is the
`M == 1` specialization of that matmul and shares the table: `pto.tgemv`,
`pto.tgemv.acc` and `pto.tgemv.bias` all verify their element types through the
same `verifyMatmulTypeTriple`, and their a5 TileLib templates carry the matmul
`MATMUL_*_DTYPES` signature lists. So the FP8 and HF8 pairs are available there
too, with the same FP32 result, FP32 `gemv_acc` accumulator and FP32 `gemv_bias`
bias. The **MX** GEMV ops are a different instruction with a different table and
are not covered by these rows.

GEMV's dtype check (`ValidateGemvInputDtypes`, `src/ir/op/tile_ops/matmul.cpp`)
is a *closed* allow-list, unlike matmul's: GEMV never had the historical
same-dtype escape hatch, so an unlisted pair is rejected in type deduction rather
than passed down to PTOAS. It writes its pairs out longhand instead of querying
`FindCubeOperandPair`, and a `static_assert` in the same file
(`HandWrittenGemvListMatchesCubeTable`) holds the two descriptions to each other
in both directions, with the accumulator dtype.

That assertion is load-bearing, not tidiness. The `gemv` family bits on the five
FP8 / HF8 rows have one remaining reader — codegen's `CheckMatmulOperandArch`,
which looks the operand pair up in the `kGemv` rows and **returns without
checking when it finds none**. A pair the longhand list accepted but the table
did not carry would therefore lose its "FP8 needs the a5 backend" rejection
silently and reach PTOAS on hardware that cannot multiply it, so that drift
direction is pinned at compile time rather than left to review.

##### A5 GEMV: the static-valid accumulator bridge

An ordinary GEMV on Ascend950 also needs its accumulator **re-typed at the
instruction**, independently of the operand dtype. PTOAS v0.72's
`verifyGemvTileOperandsA5` chains into the *matmul* operand verifier to pick up
the A5 layout rules, and inherits its shape rule — "static matmul tile shapes
lhs[M,K], rhs[K,N], and dst[M,N]". That rule compares `getMatmulLogicalShapeVec`,
which falls back to the *physical* extent whenever the type's valid is dynamic.
PyPTO's generic `pto.alloc_tile` emits `v_row=?, v_col=?` and carries the live
extent in separate `valid_row` / `valid_col` operands, and a GEMV result is the
mandatory 16-row FP32 Acc box holding one logical row — so the rule compared lhs
rows 1 against dst rows 16 and rejected every `pto.tgemv*` for a5, at every
dtype. A2/A3 never reaches that rule: `verifyGemvTileOperandsA2A3` uses
`verifyGemvValidShapes`, which compares *valid* rows only.

`EmitStaticValidGemvAccView` (`src/backend/common/pto_ops_elementwise.cpp`)
therefore emits, on a5 only, a zero-copy `pto.treshape` whose *result type*
carries the accumulator's static valid extents — the same bridge `pto.tquant.mx`
uses — and points the GEMV's `outs()` (and, for `.acc`, its `ins()` accumulator)
at that view. The accumulator's own allocation and the downstream `pto.tstore`
are untouched, and the emitted A2/A3 instruction is byte-for-byte unchanged.

Two cases decline the bridge, both deliberately:

| Case | Why |
| ---- | --- |
| Accumulator valid extents not compile-time constants | There is no static type to offer; PTOAS reports it rather than codegen inventing a constant |
| Accumulator valid N narrower than its physical N | A static dst N turns on the rule's `rhs logical N == dst logical N` clause, and the rhs type's valid stays dynamic, so that clause would read the rhs's *physical* N — a fresh mismatch. A partially-valid N needs every GEMV operand re-typed, not just the accumulator |

Physical K for a GEMV lhs is a function of the operand dtype: the Mat-to-Left
extract (`TExtractToAVector`, pto-isa `npu/a5/TExtractCommon.hpp`) requires
`Cols % (512 / sizeof(dtype)) == 0`, i.e. a whole 512-byte K block — K = 512 for a
1-byte operand, 256 for 2-byte, 128 for 4-byte.

The contract is split across two layers, because the legal set depends on the
compilation target while type deduction does not see one. Type deduction
(`ValidateMatmulOperandDataTypes`, `src/ir/op/type_inference.cpp`) accepts a pair
that is in the table **or** whose operands share one dtype — the second arm
preserves the historical "identical lhs and rhs" contract, so dtypes PTOAS would
reject anyway still reach the backend as they always have. Matmul codegen
(`CheckMatmulOperandArch`, `src/backend/common/pto_ops_elementwise.cpp`) then
rejects a table pair the configured target lacks (an FP8 or HF8 operand on
`a2a3`) before PTOAS sees it; pairs outside the table are left to PTOAS, unchanged.

`tensor.matmul` defaults an omitted `out_dtype` to the accumulator whenever either
operand is FP8 or HF8: dtype promotion would answer FP8, which the unscaled
FIXPIPE writeback cannot produce, and for a mixed pair the answer would depend on
operand order. Wider operand pairs keep their promoted default.

Physical K must stay a multiple of 32 for a 1-byte operand (the Left/Right
fractal packs 32 bytes along K); valid K carries no such restriction.

For rank-2 FP32 operands on Ascend950, `matmul`, `matmul_acc`, and
`tile.matmul_bias` accept keyword-only `tf32_mode="round_even"` or
`tf32_mode="round_away"`. Both round inputs to HF32 (e8m10) and accumulate in
FP32; halfway values round to even or away from zero, respectively. Omitting
the option preserves ordinary FP32 matmul. The setting survives tensor-to-tile
lowering and automatic L0 tiling and applies only to that operation. This
requires PTOAS v0.72 or newer and its PTO/EmitC path; it does not enable VPTO
lowering. Other input dtypes, non-2D operands, and Ascend910 are rejected.
The Torch reference generator rounds both multiplicands using the requested
mode for Tensor, Tile, and Buffer IR; bias and accumulator values stay FP32.

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
`pl.tile.gemv_acc(acc, lhs, rhs, init_cond=k0 == 0, acc_phase=pl.AccPhase.Unspecified)`

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

A literal predicate is a `ConstBool`, whether the user wrote `init_cond=True`/`False`
or a pass decided it — e.g. the generated `ko == 0` once
[`LowerPipelineLoops`](../passes/33-lower_pipeline_loops.md) replicates the K-loop.
The emitter also accepts a BOOL-typed `ConstInt`, the spelling of IR built directly
through the bindings. Either picks an arm outright; an emitter that folded neither
would double the MADs of every K block.

The fold is therefore conditional on the trip count, not universal: at
`16x512x64` the pipelined loop disappears and the emitted PTO has no `scf.if`,
while at `16x2048x64` the main loop survives with replica indices `ko` and
`ko + 256`. [`Simplify`](../passes/06-simplify.md#provable-comparison-in-an-operand)
decides replica 1's predicate (`ko == -256`) false from the loop's static range and
folds it to a `ConstBool`, while replica 0's `ko == 0` is a genuine runtime test —
it seeds the accumulator — so exactly one `scf.if` remains.

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
| `tile.load` of `pl.Tensor[..., pl.MX_A_ZZ \| pl.MX_B_NN]` | The source TensorLayout carries the MX scale GM layout. Dtype is FP8E8M0, and strided sources are rejected. Public `pl.load` defaults an omitted target to `Mat`; raw IR must carry `target_memory=Mat`. |
| `tile.move(..., target_memory=LeftScale/RightScale)` | Mat-to-Scale move with hardware-fixed row/row/32 (left) or col/col/32 (right) layout; the source Mat tile and layout overrides must match exactly. |
| `tile.create(..., target_memory=LeftScale/RightScale)` | Not supported; load MX scale data into Mat and then move it into scale memory. |
| `tile.matmul_mx` / `pl.matmul_mx` | `Left, LeftScale, Right, RightScale → Acc`; operand positions drive automatic placement, including Vec→Mat→LeftScale/RightScale staging for `quant_mx` scales. Both data operands reaching the op must be `FP8E4M3FN`, and scale is `FP8E8M0`; `lhs_scale` and `rhs_scale` must be distinct tiles. The supported packed-input form is either packed FP4 format as lhs × FP8 rhs with an explicit `pl.cast(fp4, pl.FP8E4M3FN)` before the op; native FP4×FP4 and FP8×FP4 are rejected. Physical M/K/N, valid K, and scale-group counts use the post-cast FP8 tile extents, never the packed x2 carrier shape. Physical `M % 16 == 0`, `K % 64 == 0`, and `N % 32 == 0`; valid K must satisfy `ceil(validK/32) == ceil(physicalK/32)`. Alignment / scale-group checks run only for constant extents; symbolic dims skip the numeric checks and fall back to the declared scale tile geometry (later PTOAS still verifies). |
| `tile.matmul_mx_acc` / `pl.matmul_mx_acc` | `Acc, Left, LeftScale, Right, RightScale → Acc`; in-place through `set_output_reuses_input(0)`; accumulator physical and valid M/N must match the matmul output. |
| `tile.matmul_mx_bias` / `pl.matmul_mx_bias` | `Left, LeftScale, Right, RightScale, Bias → Acc`; bias is `[1, N]` FP32. |
| `tile.tget_scale_addr` | Compiler-generated A5 binding from `LeftScale↔Left` or `RightScale↔Right`; DPS in-place on `dst_scale`. Users write only the `matmul_mx` family. |

The canonical shape is `M=128, K=64, N=64`, with FP8E4M3FN data,
FP8E8M0 scales shaped `[128,2]` and `[2,64]`, and `mx_a_zz` / `mx_b_nn`
host layouts. The left input may use either packed FP4 format when explicitly cast to FP8 first.
Align M↑16, K↑64, and N↑32.

MX tensor subviews are a legacy limitation. `tensor.slice`, `tensor.reshape`,
`tensor.transpose`, `tensor.reinterpret_view`, and ordinary MX `tensor.view`
reject MX-layout sources because the hardware path cannot represent a subview
base offset. The one exception is a product-preserving FP8E8M0 shaped alias
between packed ND backing and `MX_A_ZZ` / `MX_B_NN` (used for GM staging).
`pld.tile.remote_load` also rejects MX layouts until its complete scale layout
contract is implemented. `tensor.gather_row` / `tile.gather_row` likewise reject
MX sources.

Packed FP4 Tensor/Tile shapes, `valid_shape`, strides, and offsets count byte
carriers. Torch E2M1 x2 tensors and explicitly annotated UINT8 carriers retain
these dimensions through JIT and compiled calls without unit conversion.

Explicit casts from either packed FP4 format to FP8 are legalized on A5 through
BF16 and FP32. Scale values are unchanged because this numerically converts
only the data operand. Native packed-FP4 matmul remains unsupported.
`pl.quant_mx` supports MXFP8 and E2M1 MXFP4 quantization from FP16/BF16;
only FP8 results can feed `matmul_mx` directly. See [FP4](../fp4.md) for
quantization alignment and memory-planner requirements.

#### MX / Ascend950: pto-isa constraints

| Constraint | Detail |
| ---------- | ------ |
| Distinct scale buffers | Cube does not fold scales into Left/Right data. `TileType::ScaleLeft` / `ScaleRight` sidecars map to PyPTO `LeftScale` / `RightScale`. |
| Payload | Scale is `float8_e8m0_t` / `FP8E8M0`; the emitted MX data pair is `FP8E4M3FN × FP8E4M3FN` (rejects `FP8E5M2` and native packed FP4). A packed-FP4×FP8 input first casts the packed lhs to FP8. Physical K, valid K, and `ceil(K/32)` scale groups are measured on that post-cast FP8 tile, not the packed x2 carrier; physical `K%64==0` and fractal is 32. |
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
| `#pto.layout` / mx load | `mx_a_zz` / `mx_b_nn` / …; codegen emits logical rank-2 `make_tensor_view` (PTOAS v0.60 InferPTOLayout / EmitC map the pack). |
| Coverage | `pto.tmatmul.mx` / `.acc` / `.bias` + `pto.tget_scale_addr`; `pto.tquant.mx` via [LowerCompositeOps](../passes/15-lower_composite_ops.md). |

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

### Accumulator phases (A2/A3)

`tile.matmul`, `tile.matmul_acc`, `tile.gemv`, `tile.gemv_acc`, and
`tile.gemv_bias` accept `acc_phase` as `pl.AccPhase.Unspecified` (the default),
`pl.AccPhase.Partial`, or `pl.AccPhase.Final`. `Unspecified` preserves the
existing behavior. Use `Partial` while more K chunks remain and `Final` for
the last chunk. The Matmul DSL entry points make `acc_phase` keyword-only,
so `pl.tile.matmul_acc(acc, lhs, rhs, init_cond, acc_phase=...)` retains its
existing fourth positional `init_cond`; the IR wrapper likewise preserves
its positional `span`. Both initialization and accumulation codegen arms
carry the selected phase.

With `AccPhase.Final`, `matmul_acc` requires the product's valid M/N rectangle
to provably equal the accumulator's valid rectangle. A smaller update of a
wider accumulator is rejected: its Final store would check untouched blocks
whose unit flags the MAD did not set. Use a matching accumulator window, or
keep the update partial until the whole valid rectangle is finalized.

On a unit-flag-aware path, the final accumulator producer must be paired with
`pl.store(..., st_phase=pl.STPhase.Final)`. The final producer sets the unit flag
and the final store checks and clears it; a plain store intentionally keeps the
default `pl.STPhase.Unspecified` behavior. PyPTO does not expose PTO-ISA's
check-only store phase because that phase requires an ordered multi-consumer
lifecycle. Compilation verifies the supported final pairing in both directions:
bind the final producer's result and store that exact value in the same
straight-line control-flow region. A missing or mismatched pair is rejected
before code generation because it can otherwise stall device execution.
Before its matching store clears the flag, a final result may only be
forwarded through a plain SSA alias; no other operation may consume it. This
also prevents a view or copy from hiding an unsafe accumulator reuse. A complete pair inside a branch or loop body is supported; carrying a
pending final result across a branch, loop, or scope boundary is rejected.
[`AutoTileMatmulL0`](../passes/20-auto_tile_matmul_l0.md) preserves this contract:
intermediate K blocks use `Partial`, and the last block alone uses `Final`.
Explicit phases do not remove other automatically inserted synchronization.

Storage checks follow registered in-place operations and views, including
aliases made before a Final update and straight-line tuple forwarding or
nested tuple projections. A branch selection retains its possible sources
without identifying independent source accumulators with each other. Tuple
values with unresolved provenance across a branch join, loop carry, or opaque
result are rejected when they may contain Acc storage (including unresolved
memory spaces) and the function uses a Final protocol. Unpack those tiles into
separate aliases or tile loop arguments. Known non-Acc tile tuples, scalar
tuples and unphased functions are unaffected. Storage analysis is built only
when a Final producer is encountered; unmatched Final stores are always checked.

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

## Tensor and Tile Operators

See [Tensor and Tile Operators](05-tensor-tile-ops.md) for data-operator APIs,
flat gather, valid-region semantics, tile layouts, and mask patterns.

## SyncOp: Synchronization Operations

**Purpose**: Hardware synchronization and barriers, plus the TaskId and SPMD launch-shape queries sharing the `system.` namespace
**Type**: `UnknownType` (no return, used in `EvalStmt`) for barriers; `ScalarType` for the value-binding query ops (`task_invalid`, `task_is_valid`, `available_cluster_count`, `available_aiv_count`)
**Location**: `src/ir/op/sync_ops/` — `sync.cpp` (barriers), `task.cpp` (TaskId), `launch.cpp` (launch-shape queries)
**Python API**: `from pypto.ir.op import system`

The public Python barrier API is `pl.system.barrier(*, pipe)`, accepting only
`pl.PipeType.V`, `M`, and `ALL`. It constructs the existing zero-operand IR
operators below; there is no new `system.barrier` IR operator or `pipe` IR attr.
The Python IR printer emits `pl.system.barrier(pipe=...)` with the corresponding
enum member. The old `bar_v()`, `bar_m()`, and `bar_all()` Python APIs have been
removed. Barriers use no event slot, and
`ALL` orders all pipelines on the current core, not arrivals across cores.

| Operation | Description | Kwargs |
| --------- | ----------- | ------ |
| `system.bar_all` | Current-core all-pipeline barrier; Python `barrier(pipe=pl.PipeType.ALL)` (lowers to `pto.barrier <PIPE_ALL>`) | None |
| `system.bar_v` | Vector pipeline barrier; Python `barrier(pipe=pl.PipeType.V)` (lowers to `pto.barrier <PIPE_V>`) | None |
| `system.bar_m` | Matrix pipeline barrier; Python `barrier(pipe=pl.PipeType.M)` (lowers to `pto.barrier <PIPE_M>`) | None |
| `system.fence` | Memory barrier over global memory (lowers to `pto.fence.barrier_all #pto.fence_scope<gm>`) | None |
| `system.cacheinvalid` | Invalidate the cache line containing a tensor sub-region's base address. Args: `tensor`, `shapes` (N-D), `offsets` (N-D). Every region size — a single element included — lowers to `pto.partition_view` + `pto.cmo.cacheinvalid %payload_view single_cache_line : !pto.partition_tensor_view<...>`; `shapes` does not make it walk every cache line. The no-argument form invalidates all GM. | None |
| `system.syncall` | Cross-core all-participant barrier (`pto::SYNCALL`). Attr `mode` `"hard"` (FFTS, no operands) or `"soft"` (GM-polling, operands) | `core_type` (`"aiv_only"` \| `"aic_only"` \| `"mix"`), `mode` (`"hard"` \| `"soft"`) |
| `system.sync_src` | Set sync flag | `set_pipe`, `wait_pipe`, `event_id` |
| `system.sync_dst` | Wait sync flag | `set_pipe`, `wait_pipe`, `event_id` |
| `system.set_flag` / `system.wait_flag` | Explicit intra-core pipeline event | `src_pipe`, `dst_pipe`, static `event_id` or one dynamic INDEX operand; optional `core_type` |
| `system.task_invalid` | Sentinel `TaskId::invalid()` — "no producer" seed for a TaskId carry | None |
| `system.task_is_valid` | Test whether a `TASK_ID` value is a valid (non-sentinel) handle | None; sole positional arg is the TaskId Var |
| `system.available_cluster_count` | This run's MIX cluster (= AIC) count, read from the device. Result `Scalar[INT32]` | None |
| `system.available_aiv_count` | This run's standalone AIV core count, read from the device. Result `Scalar[INT32]` | None |

The explicit flag calls have `UnknownType` and are retained as `EvalStmt(Call)`.
`src_pipe` and `dst_pipe` are distinct concrete `PipeType` values (not `ALL`).
Exactly one event form is present: a literal `event_id` in `[0, 7]`, or one
`ScalarType(INDEX)` operand whose runtime value must stay in that range.
`core_type` is optional `"aic"` / `"aiv"` IR metadata for mixed-kernel placement.
Codegen emits `pto.set_flag` / `pto.wait_flag`, or the `_dyn` variants for an
INDEX operand. A dynamic operand folded to a constant remains an operand and
still emits the `_dyn` form; IR printing preserves it with `pl._dump.call`.
The calls do not represent cross-core synchronization.
The `PipelineLoopValid` verifier rejects either call inside a `ForKind::Pipeline`
loop before pipeline scheduling. The scheduling passes also reject this placement
when optional IR verification is disabled. Explicit flags work in ordinary loops.

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
| `system.aic_initialize_pipe` | 2 | Init cross-core pipe on Cube side (positional: `c2v_consumer_buf`, `v2c_consumer_buf`, i32 SSA) | `dir_mask`, `slot_size`, optional `slot_num`, optional `local_slot_num`, optional `id`, optional `c2v_dtype`, optional `pre_relu` |
| `system.aiv_initialize_pipe` | 2 | Init cross-core pipe on Vector side (positional: `c2v_consumer_buf`, `v2c_consumer_buf`, i32 SSA) | `dir_mask`, `slot_size`, optional `slot_num`, optional `local_slot_num`, optional `id`, optional `c2v_dtype`, optional `pre_relu` |

- `slot_num` (when set, must be > 0) pins the GM ring-buffer slot count; omit it to let PTOAS pick its default (8 unidirectional, 4 per direction bidirectional).
- `local_slot_num` (a2/a3 only, must be > 0 and `<= slot_num`) pins the local slot count.
- **Sizing the reserved/imported buffer is your responsibility and is architecture-dependent:** on **a3** use `slot_size * local_slot_num`; on **a5** use `slot_size * slot_num`.

For explicit **A5 C2V-only, non-split** pipes (`dir_mask=1`, `split=0`), set
`c2v_dtype=pl.FP32`, `pl.FP16`, or `pl.BF16` on **both** initializers. The producer
must push an FP32 Acc tile and the consumer must pop the declared payload dtype.
`pre_relu=True` applies ReLU before conversion on every push and requires an explicit
`c2v_dtype`; both peers must agree. FP16/BF16 narrowing uses unscaled RNE, with no
`pre_quant` scale. `slot_size` is measured in payload bytes: a `64 x 128` BF16 tile
uses 16384 bytes, with a 32768-byte reserved buffer for `slot_num=2`. Release the
popped FIFO slot only after its last consumer. Omitting these options preserves
existing pipe behavior. Each AIC/AIV function initializes a given pipe `id` once;
PTOAS rejects repeated initialization of that ID within the same function.
The Torch reference also keeps queues independent by `id`. It applies the
producer's executed initialization to subsequent pushes, including pushes delegated
to inline helpers. When a later Group initializes a default C2V pipe with the same
ID, the reference restores identity transport for that ID.

### Buffer Management Operations

| Operation | Args | Description | Kwargs |
| --------- | ---- | ----------- | ------ |
| `system.reserve_buffer` | 0 | Reserve named cross-core buffer (consumer side) | `name`, `size`, `base`* |
| `system.import_peer_buffer` | 0 | Import buffer from peer function (producer side) | `name`, `peer_func` |

\* `base` defaults to `AUTO (-1)` for compiler-assigned address.

In explicit JIT Groups, symbolic `peer_func` names resolve through inline wrappers
around Group members and inside nested inline initializers. Each specialization of
a helper that imports a peer must belong to one generated AIC/AIV peer pair;
sharing it across different pairs is rejected. Helpers without peer imports may be
shared. Names and call aliases that identify both peer members are rejected as
ambiguous.

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

**Producing more than one value?** Read [Multi-Output Operators](09-multi_output_ops.md) first — the results belong in a `TupleType`, never in the argument list, and the registry rejects at import any argument such an operator writes that was not declared a workspace.

## References

Core definitions live in `include/pypto/core/common.h` and `include/pypto/ir/`; registry and type-inference implementations are in `src/ir/`, with operator implementations grouped under `src/ir/op/{tensor_ops,tile_ops,sync_ops}/`.

## Register operations

The `reg.*` schemas use category `RegOp`, explicit argument effects and
register-region requirements. See [register operations](../language/06-register-operations.md)
for dtype, memory, conversion and ownership contracts.

Ordinary RegOp extensions include unary, integer bitwise/shifts, vector-scalar,
scalar compare, dynamic lane selection and single-lane reductions; see the linked
register operations contract for exact dtype and mask restrictions.

Fused arithmetic (`expdif`, `axpy`, `mula`, `lrelu`, `prelu`), paired results
(`mull`, `addc`, `subc`, `addcs`, `subcs`, `intlv`, `dintlv`) and indexed UB
access (`gather`, `scatter`) extend the initial register set. Paired operations
declare output arity two and return a `TupleType`; scatter returns an alias of
its destination Tile and declares argument 0 as written.

`interpret_cast`, `chist`, `dhist` and `create_group_mask` are pure register
operations with one result. Reinterpretation preserves bit capacity; histogram
results match the UINT16 accumulator rather than the UINT8 source; grouped
masks derive their lane count from static group geometry.

## Scalar SIMT operations

`pypto.ir.op.simt` exposes typed coordinate queries, `lane_id`, and scalar
`load`/`store` on UB Tile or GM Tensor storage. `store` returns `VoidType` and
belongs in `EvalStmt`. See [Scalar SIMT IR](11-simt.md) for the operation,
launch, cache-policy, and validation contracts.

Warp collectives include `vote_all/any/uni`, `vote_ballot`,
`shuffle_idx/up/down/bfly` and `redux_add/min/max`. See the
[collective contracts](11-simt.md#warp-collectives) for active-lane participation,
supported types and shuffle widths.

Atomics `atomic_add/min/max/exch/cas` return the old INT32/FP32 value from
initialized GM/UB storage. `syncthreads`, `threadfence` and `threadfence_block`
return void. See [atomic and synchronization contracts](11-simt.md#atomics-and-synchronization).
