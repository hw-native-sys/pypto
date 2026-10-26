# RFC: Explicit GM Subviews for InCore Loads

## 1. Status and decisions

**Status: Implemented in this worktree; device numerical and physical DMA acceptance remain open.**
Created: 2026-10-22; design updated: 2026-10-26. Related request: [issue #3042](https://github.com/hw-native-sys/pypto/issues/3042).
Source baseline: `acc125cec76c06f3f629bd4274b9a51e102a4b06`.
The English document is authoritative; the Chinese counterpart mirrors it.

For the complete API → IR → pass → codegen example, runnable PTO fixture and
verification status, see the [worked example](gm-subview-worked-example.md).

The agreed direction is to add an explicit `pl.tensor.subview` API. The following
implementation choices define the implemented contract:

1. Reuse `TensorType` and `TensorView`; add an explicit addressing kind to
   `TensorView`, not a new tensor kind. Existing canonical packed views and new
   affine subviews must be distinguishable to verifiers and consumers.
2. Define the public operation as an N-dimensional affine GM view. The first
   executable backend profile is narrower (read-only, function-local, ND, positive
   steps), but that profile must not redefine the operation's semantic contract.
3. Preserve existing `slice` behavior. Explicit GM descriptor construction and
   implicit Tensor-to-Tile materialization remain different operations.
4. Put the address origin in the defining operation and codegen descriptor state,
   not in a new `TensorView` field. Strides and the addressing kind belong in the
   result type.
5. Restrict input/output to ordinary Tensor; require proven strict containment and
   reject unknown bounds, including dynamic geometry without usable range facts.
6. Do not promise arbitrary strided DMA support or silently load a large envelope.
   Supported native loads require backend validation and dedicated evidence.

The public operation is intentionally broader than the first A2/A3 lowering
profile. A backend may reject a semantically valid view when its DMA or layout
contract cannot represent it; it must name the unsupported dimension, dtype or
memory space rather than silently materialize an enclosing span.

## 2. Problem and observed baseline

The requirement is to separate GM address description from data transfer. A view
may span distant rows, while the only on-chip data is the explicitly loaded tile.

The issue's original pattern is:

```python
span = pl.tensor.slice(cache, [2 * delta, 512], [lo, 0])
pitched = pl.tensor.view(span, shape=[2, delta * 512])
pair = pl.load(pitched, [0, 0], [2, 512])
```

At the baseline, `tensor.slice` has a dedicated conversion to `tile.load` for GM
input, or `tile.slice` for Tile input. `tensor.view` passes through only when its
rewritten operand remains tensor-like. The slice/view example fails with:

```text
tensor.view in an InCore function requires a GM Tensor input that remains tensor-like
through ConvertTensorToTileOps; viewing the result of an op lowered to Tile is not supported
```

A separately executed pass-only `slice -> load` reproduction printed the following
body fragment (the output store and signature are omitted):

```python
# Before ConvertTensorToTileOps
span: pl.Tensor[[4, 512], pl.BF16] = pl.tensor.slice(x, [4, 512], [1, 0])
pair: pl.Tile[[2, 512], pl.BF16] = pl.tile.load(span, [0, 0], [2, 512], [2, 512])

# After ConvertTensorToTileOps: observed, invalid load input
span__tile: pl.Tile[[4, 512], pl.BF16, pl.Mem.Vec] = pl.tile.load(
    x, [1, 0], [4, 512], [4, 512], target_memory=pl.Mem.Vec
)
pair: pl.Tile[[2, 512], pl.BF16] = pl.tile.load(span__tile, [0, 0], [2, 512], [2, 512])
```

The second load requires a GM Tensor, but receives a Tile. The pass-only call did
not fail immediately; mutating Call arguments does not rerun the operator deducer.
This is not evidence of two successful DMAs or any final allocation size.
Root `tensor.view -> load` does pass the same conversion test.

Current prescans collect consumer memory-space demands, predicted Tile values,
write safety and some GM allocation lineage. They do not preserve every view chain
backward from an explicit load. `OptimizeOrchTensors` follows this conversion in
the default pipeline. This RFC adds explicit semantics rather than changing that
global materialization policy.

Repository evidence:

- `src/ir/transforms/op_conversion_registry.cpp`: `RegisterMemoryOps` slice rule.
- `src/ir/transforms/convert_tensor_to_tile_ops_pass.cpp`: `TensorToTileMutator`,
  `ConsumerSpaceCollector`, `TensorConversionAnalysis`, `TransformIncoreFunction`.
- `src/ir/transforms/mutator.cpp`: argument substitution for Calls.
- `src/ir/op/tile_ops/memory.cpp`: `DeduceTileLoadType` GM-input contract.
- `python/pypto/ir/pass_manager.py`: outlining, conversion and later pass order.

The worked example links the implementation tests; schematic IR below illustrates the contract.

## 3. Goals and exclusions

Required outcomes:

- A subview is a zero-copy alias, with no GM allocation, UB allocation or payload
  read at construction. Descriptor arithmetic is allowed.
- Device-produced offsets and row steps stay on device and remain runtime values.
- Nested subviews preserve parent pitch, byte origin, valid data and root lifetime.
- A load allocates for its result Tile, never for the subview's address envelope.
- Existing `slice`, computed Tile rejection, packed-view checks and old programs
  retain their semantics.
- Every subview must be proven contained in its immediate parent window; unknown
  bounds are a compilation error, not a caller precondition.

Subject to proven parent-window containment, the semantic operation supports
rank >= 1, positive nonzero steps on every axis,
symbolic integer offsets/steps/shapes, and source layouts whose address map is affine in logical element
coordinates with one explicit stride per axis. Blocked or swizzled layouts such
as NZ/MX are rejected unless their selected logical domain is proven representable
by this mapping; knowing their layout name alone is insufficient. It is read-only in this RFC. Zero steps (broadcasting
one source element), negative steps (reversal), padding/clamp, dropped dimensions,
and writes through aliases are separate operations because they change validity,
alias effects or hardware contracts; they are rejected by this operator rather
than silently assigned a different meaning.

The source and result are ordinary `TensorType` only. Reject `DistributedTensor`
(including its local storage), remote/collective views, host tensors, Tile values
and compiler-owned on-chip allocations. Use an exact ordinary-Tensor type check,
not `AsTensorTypeLike` alone, at deduction and after operand rewriting. A backend profile
may impose stricter rank, layout, dtype, alignment and scope restrictions.

### 3.1 Capability matrix

| Dimension | API semantic contract | Initial executable profile |
| --------- | --------------------- | -------------------------- |
| Rank | Any rank >= 1; `shape`, `offsets`, `steps` have equal rank | Rank 2 ND |
| Offset | Integer scalar per axis; static or proven-bounded device-dynamic | Provably bounded device-dynamic row offset; other axes after probe |
| Step | Provably positive, nonzero integer scalar per axis; composed through nesting | Provably positive and bounded dynamic row step; column step 1 |
| Shape | Provably positive extents; symbolic shape only with proven containment | Tile load shape follows existing allocation rules |
| Layout | Preserves an affine element-stride map; no implicit layout conversion | Packed ND with an explicit outer pitch |
| Dtype | Any dtype with a registered address/load contract | BF16 static and bounded dynamic PTOAS paths verified; other dtypes rejected |
| Source | Ordinary Tensor GM allocation only | Caller-owned Tensor GM parameter |
| Use | Metadata aliases, explicit load, approved read-only consumers | `subview -> load`; targeted `set_validshape` path |
| Scope | Function-local descriptor; no unproven geometry-changing phi | Same; no boundary escape |
| Hardware | Backend decides native DMA support; no semantic fallback is implied | PTOAS v0.67 A2/A3 static and dynamic paths; device DMA evidence pending |

This table separates three answers that must not be conflated: whether the IR
can describe a view, whether a pass can preserve it safely, and whether a target
can execute one native load. Unsupported backend rows are compile-time diagnostics;
they do not authorize a larger implicit transfer.

## 4. Public API and examples

### 4.1 Signature

```python
def subview(
    source: Tensor,
    shape: Sequence[IntLike],
    offsets: Sequence[IntLike],
    *,
    steps: Sequence[IntLike] | None = None,
) -> Tensor:
    """Create a function-local GM alias; do not load or allocate payload data."""
```

Expose only `pl.tensor.subview` initially, plus `pypto.ir.op.tensor.subview`.
Do not add `pl.subview`, a Tile overload or a Tensor method in this release.
Normalize omitted steps to an all-ones sequence with the source rank in the
Python IR builder. Reject booleans, floats, nested sequences, rank mismatches,
zero steps, Tile and DistributedTensor sources. Runtime integer DSL scalars are accepted and widened
before address arithmetic. A negative step is not a subview; use a future reverse
view operation with explicit validity semantics.

| Parameter | Meaning |
| --------- | ------- |
| `source` | Ordinary GM Tensor or its supported function-local descriptor alias |
| `shape` | Number of sampled elements along each result axis |
| `offsets` | First sampled coordinate in the immediate source |
| `steps` | Source-coordinate increment per result-coordinate increment; default 1 |

All geometry is in elements, not bytes. The API does not require any particular
axis to have step one. `shape` may be symbolic; this does not make dynamic UB
allocation legal. The source allocation and its caller-provided logical-to-byte
mapping must already be valid. The initial backend profile below may require
`steps[1] == 1`, but that is a profile restriction, not an API rule.

### 4.2 The two-row request

```python
# Accepted only where proven: rows > 2, delta > 0, lo >= 0, lo + delta < rows.
pair_view = pl.tensor.subview(
    cache, shape=[2, 512], offsets=[lo, 0], steps=[delta, 1]
)
pair = pl.load(pair_view, [0, 0], [2, 512])
```

For a packed `[rows, 512]` BF16 source:

```text
origin = lo * 512 elements from cache's current origin
shape  = [2, 512]
stride = [delta * 512, 1]
pair[0, c] = cache[lo, c]
pair[1, c] = cache[lo + delta, c]
```

There are 2048 logical payload bytes; actual UB size follows existing Tile layout
and alignment. No intermediate buffer scales with `delta`. The old window needed
`lo + 2 * delta <= rows`; the new view needs only `lo + delta < rows`.
For original indices `a,b`, sorting to `lo,delta` sorts output rows. Preserve the
original order in the consumer when `a > b`. Handle `a == b` with a single-row
load and reuse it; zero steps are deliberately outside this release.

### 4.3 Partial loads, nesting and slice

```python
window = pl.tensor.subview(x, [16, 512], [base_row, 0])
tile = pl.load(window, [2, 64], [4, 128])

selected = pl.tensor.subview(window, [2, 128], [3, 64], steps=[4, 1])
pair = pl.load(selected, [0, 0], [2, 128])
# pair rows: x[base_row + 3, 64:192], x[base_row + 7, 64:192]

block = pl.tensor.slice(window, [4, 128], [2, 64])
result = pl.add(block, block)  # existing implicit Tile materialization
```

`subview -> slice -> compute` is supported. `subview -> subview -> load` preserves
GM identity throughout. InCore `slice -> subview` is rejected if the slice has
become a Tile. `slice -> load` is not reinterpreted as a supported descriptor path.
Add a focused diagnostic for a load whose rewritten input is a Tile.

### 4.4 View and other consumers

```python
whole_rows = pl.tensor.subview(x, [4, 512], [base_row, 0])
reshaped = pl.tensor.view(whole_rows, shape=[2, 1024])
pair = pl.load(reshaped, [0, 0], [2, 512])
# Reads original rows base_row and base_row + 2.
```

This succeeds only if the source's selected rows are packed. A `[4,256]` column
window of a `[N,512]` parent keeps stride `[512,1]`; reshaping it to `[2,512]`
must fail. A row-stepped view likewise cannot discard pitch in a reshape.
Symbolic target shapes must prove equal element count in context; the generic
view builder's deferred equality check cannot enlarge a borrowed window.

| Consumer | First-release contract |
| -------- | ---------------------- |
| `dim`, plain SSA alias, nested `subview` | Metadata only; preserve GM identity |
| `view` | Preserve origin; retain existing packed-source/validity restrictions |
| `set_validshape` | May narrow effective validity; must preserve pitch and provenance |
| `slice` / supported Tensor compute | May load at the consumer; never remap the GM view globally |
| `load` | Read the descriptor; return a fresh Tile |
| `store`, `assemble`, scalar write or other writer through an alias | Reject in this release |
| `reshape`, `transpose`, `reinterpret_view` | No new GM guarantee; require explicit load before these operations in this release |
| Return, non-inlined call, descriptor phi with changing geometry | Reject; move descriptor construction to the consumer scope |

Reject descriptor operands of the last two rows before an unrelated converter
silently rewrites them. The allowlist applies to subview provenance, not to all
ordinary Tensor operands. Compute results are fresh values, not read-only aliases.

### 4.5 Reference example and interface decisions

The normative static example selects rows 1 and 4 from an ordinary packed ND
BF16 Tensor [8,512], then writes a packed [2,512] output. Use this same geometry
in builder, pass, codegen and runtime acceptance; the complete proposed user
program is:

```python
import torch
import pypto.language as pl
from pypto.runtime import RunConfig


@pl.jit
def copy_selected_rows(cache: pl.Tensor, output: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        v = pl.tensor.subview(cache, [2, 512], [1, 0], steps=[3, 1])
        pair = pl.load(v, [0, 0], [2, 512])
        pl.store(pair, [0, 0], output)
    return output


if __name__ == "__main__":
    rows = torch.arange(8).reshape(8, 1)
    cols = torch.arange(512).reshape(1, 512)
    cache = (16 * rows + cols % 16).to(torch.bfloat16)
    output = torch.zeros((2, 512), dtype=torch.bfloat16)
    copy_selected_rows(cache, output, config=RunConfig())
    torch.testing.assert_close(output, cache[[1, 4], :], rtol=0, atol=0)
```

The new API is required for these semantics, rather than a new spelling of view:

| Operation on the same source | Meaning | Result / InCore behavior |
| ---------------------------- | ------- | ------------------------ |
| tensor.subview(cache, [2,512], [1,0], steps=[3,1]) | Sample a strictly smaller parent-coordinate window without loading | GM Tensor, origin 512, stride [1536,1] |
| tensor.view(cache, shape=[2,2048]) | Reinterpret the same 4096 elements using a canonical shape/layout | GM Tensor; not the two selected rows |
| tensor.view(cache, shape=[2,512]) | Tries to reinterpret 4096 elements as 1024 | Existing static element-count check rejects |
| tensor.slice(cache, [2,512], [1,0]) | Select consecutive rows 1 and 2 | GM-input slice becomes a Tile load in InCore |
| load(cache, [1,0], [2,512]) | Transfer consecutive rows 1 and 2 explicitly | Fresh Tile |
| load(subview, [0,0], [2,512]) | Transfer sampled rows 1 and 4 explicitly | Fresh Tile; descriptor stays GM |

For this case the row proof is 1+(2-1)*3=4<8, the column proof is
0+(512-1)*1=511<512, and strict reduction is 2<8. All geometry is constant, so
unknown bounds cannot be hidden behind a caller precondition. Offset 5 instead
of 1 must fail because the final row would be 8; an unconstrained dynamic offset
must fail as unknown. View's existing optimistic handling of some symbolic
shape equalities does not apply to subview's required containment proof.

## 5. Coordinate, bounds and validity semantics

For each axis `i`, with source element stride `S[i]` and source origin `B`:

```text
V[j]      = source[offsets + j * steps]
V.shape   = shape
V.stride[i] = S[i] * steps[i]
V.origin  = B + sum(offsets[i] * S[i])
byte_address(V[j]) = root_pointer + sizeof(dtype) *
                     (V.origin + sum(j[i] * V.stride[i]))
```

Offsets are relative to the immediate source, not the root allocation. A nested
subview multiplies steps and accumulates origin using the parent's actual stride.
No address is derived from the result shape alone. Pure alias/view operations
carry origin unchanged; steps never alter dtype.

Required proven predicates on every axis, using the source's effective valid extent
`E[i]`, not just its physical shape:

```text
shape[i] > 0; steps[i] > 0; offsets[i] >= 0
offsets[i] + (shape[i] - 1) * steps[i] < E[i]
```

A selected region is fully valid. This release does not inherit padded bytes as
valid payload and does not expose `valid_shape`/`pad_value`/`clamp` in subview.
Partially valid sources are accepted only when all selected coordinates lie in
their valid prefix. Later `set_validshape` can narrow the result again.

The parent window must **strictly contain** the new view, including its offset and
sampling step. A full-window identity subview is rejected; use the existing plain
alias/view operation when no window is narrowed. Comparing `shape <= source.shape`
or total element counts is insufficient. For example,
source `[8,512]`, result `[2,512]`, row offset 1 and row step 6 fits (last row 7);
row step 7 does not fit (last row 8), even though the result has fewer elements.
For a nested subview prove strict containment against the immediate parent's
effective valid extent before composing its origin and strides. A physically available
root region does not authorize access outside that parent window.

All four predicates above must be **proven true** for every axis, and at least one
axis must prove `shape[i] < E[i]`. Together with positive integer steps and
containment, this makes the sampled coordinate set a proper subset of the parent.
Thus even a strided view with the same bounding endpoints may be a strict subview.
A provably false predicate reports an out-of-bounds/invalid-geometry error. An unknown predicate
reports `tensor.subview cannot prove` with the failed proof obligation and
defining span. There is no unchecked mode, caller-assumed
precondition, implicit runtime assertion or silent clamp. The same prove-or-reject
policy applies to `set_validshape` narrowing and the valid read region of load/slice
consumers of subview-derived descriptors. Strict size reduction is required only
when constructing a subview; a load may read the entire subview and
`set_validshape` may retain its current valid extent. Existing ordinary-Tensor
operations are unchanged.

### 5.1 Proof staging and dynamic geometry

Type deduction performs structural checks and rejects constant violations. It
may construct a provisional symbolic descriptor because the operator registry
has no enclosing branch/loop context. Compilation cannot accept that descriptor
until a contextual verifier succeeds. Add a shared subview-bounds visitor using
`ir::arith::Analyzer` (`include/pypto/ir/arith/analyzer.h`), called before
Tensor-to-Tile conversion and again on final transformed IR before PTO emission.
A pass-only conversion entry must invoke the same check; a codegen-only entry
must not bypass it. Provisional IR is not permission to generate code.

The wrapper widens integer inputs to INDEX before geometry arithmetic. Existing
integer expressions and scalar definitions must also fit their original dtype:
widening after a narrow multiply has overflowed cannot repair it, even if later
cancellation makes the mathematical final offset appear in bounds. Conditions
and loop bounds must also be overflow-free before contributing proof facts.

The visitor binds dominating SSA scalar definitions, proven loop bounds and
branch predicates with scoped `GetConstraintContext`; branch facts are removed
on exit and phi values retain only facts valid on every incoming edge. Reuse
`Analyzer::CanProve` for the four per-axis predicates and the strict-reduction
predicate. Do not treat documentation, a
user assertion without checked semantics, a shape annotation for a scalar, or
unrelated branch conditions as evidence. Use bounded local simplification and
interval queries, not an unbounded global solver; proof-budget exhaustion means
unknown and rejection. Unsupported nonlinear relationships are likewise rejected.

Dynamic does not mean unchecked: constant-folded/specialized values, bounded loop
indices and device scalars within a dominating, recognized guard may pass. An
arbitrary device-loaded `lo` or `delta` without such facts fails. A schematic
rewrite for the issue is:

```python
# rows is a known positive source extent; arithmetic is widened and proven safe.
if rows > 2 and delta >= 1 and delta < rows:
    if lo >= 0 and lo < rows - delta:
        v = pl.tensor.subview(cache, [2, 512], [lo, 0], steps=[delta, 1])
        pair = pl.load(v, [0, 0], [2, 512])
        # Consume pair here; the GM descriptor does not escape the branch.
```

This is the required compiler capability, not evidence that the current compiler
already proves the guard. If its analyzer cannot derive `lo + delta < rows` or
cannot track the defining SSA values, reject this specialization. The issue is
therefore only covered for proven-bounded dynamic inputs; accepting arbitrary
device geometry would contradict the requested containment rule. No fallback
may bypass that proof by recreating an unchecked subview.

Normalize address arithmetic before multiplying/adding. Prove that geometry,
composed strides, origins and maximum selected byte offsets fit the selected
index/pointer arithmetic width; reject constant overflow and unknown dynamic
ranges. Mathematical inequalities cannot justify wrapped machine arithmetic.
The proof uses widened operands, and lowering must preserve that width until a
separately proven backend-width conversion. DMA pitch, alignment and parameter
limits require their own proof in the backend profile; logical containment alone
does not prove DMA legality.

## 6. IR representation and registration

Use the existing Call form with four positional operands:

```text
tensor.subview(source, shape_tuple, offsets_tuple, steps_tuple)
  -> TensorType(shape, source.dtype, TensorView(stride=derived_stride, layout=source.layout, kind=AffineSubview))
```

No `ObjectKind` or `TensorType` subclass is added. `TensorView` gains an explicit
addressing kind, for example `Canonical` versus `AffineSubview`; its existing
`stride` vector carries the affine logical strides. The concrete storage contract
is **no MemRef on a borrowed descriptor**: deduction sets the
subview result's `memref_` to `nullopt`, even if its source carries a MemRef. Allowed
metadata descendants and plain aliases retain this absence through InitMemRef
and address planning. Do not use zero, an envelope size, or the parent's size as
a sentinel in `MemRef.size_`. The caller-owned root keeps its existing storage
representation; it is not reallocated or changed by this feature.

Only caller-owned GM parameter roots and supported function-local aliases of
them are executable sources in this release. Compiler-owned local allocations
are excluded. The defining Call is the sole source of additional relative
offsets. Root identity is recovered through the alias graph; codegen applies
relative origins exactly once against the parameter's current-origin pointer.
Reconstruction after print/serialize round trips uses those same Calls, not
unserialized state. This does not remove MemRefs from ordinary tensors or Tiles.

Proposed registration shape (helper names for deduction are new):

```cpp
REGISTER_OP("tensor.subview")
    .set_op_category("TensorOp")
    .add_argument("source", "Ordinary GM Tensor with an affine address map")
    .add_argument("shape", "Result extents in elements")
    .add_argument("offsets", "Immediate-source starting coordinates")
    .add_argument("steps", "Immediate-source sampling steps")
    .set_arg_effect(0, ArgEffect::Read)
    .set_output_reuses_input(0)
    .set_output_memory_inherit_input()
    .f_deduce_type(DeduceTensorSubviewType);
```

`ArgEffect::Read` is the existing conservative Functional-stage classification;
construction emits no payload read and has no write channel. A real data read
occurs at load/compute. Do not invent a write effect for alias registration.
All geometry operands also receive the normal explicit read/no-write verdicts
required by registry validation. The result-alias query must resolve argument 0.

### 6.1 Canonical and affine strides

Today's `CheckCanonicalView` remains the verifier for `Canonical` views: an ND
view has innermost stride one and the existing layout-specific outer-stride
lower bound. It does not require every ND view to be physically packed.
`AffineSubview` uses a separate verifier: rank and integer stride vectors must
match, every step must be positive, and each result stride must equal the parent
stride multiplied by that step. Its innermost stride may therefore be greater
than one. `TensorViewCanonical` dispatches on the addressing kind instead of
weakening the canonical invariant globally.

`MaterializeTensorStrides` must preserve explicit strides and the addressing kind.
Packed-only operations still require equality with packed strides, while affine
operations may consume a non-packed mapping only when their consumer contract
allows it. A valid affine view is not automatically legal for a DMA or reshape;
those consumers must prove their own layout requirements.

Reflection, equality, hashing, bindings and serialization include the addressing
kind. Old serialized views default to `Canonical`; affine views round-trip their
kind and explicit strides. The kind describes the address map, not a new layout
or allocation. Generic Call serialization carries the new op and operands, with
round-trip tests for symbolic stride expressions, source offsets and old data.
Reparse of a post-pass dump must also enforce source validity.

### 6.2 Builder and concrete IR contract

The DSL wrapper unwraps ordinary Tensor and integer Scalar operands, invokes the
IR builder, and wraps the Call result as a Tensor. After public-argument
normalization, the builder uses the existing tuple helper and registry route:

```python
from pypto.pypto_core import ir as core
from pypto.ir.utils import _to_make_tuple


def subview_ir(source, shape, offsets, steps, span):
    return core.create_op_call(
        "tensor.subview",
        [
            source,
            _to_make_tuple(shape, span),
            _to_make_tuple(offsets, span),
            _to_make_tuple(steps, span),
        ],
        {},
        span,
    )
```

The operator is new and must be registered before this builder can execute.
The reference example constructs the following InCore IR body; this is operation
notation, not current printer output:

```text
func copy_selected_rows_incore(cache: Tensor<[8,512],BF16>,
                              output: Tensor<[2,512],BF16>):
  v    = tensor.subview(cache, (2,512), (1,0), (3,1))
  pair = tile.load(v, (0,0), (2,512), (2,512))
  out  = tile.store(pair, (0,0), output)
  return out
```

| Value | IR type / metadata | Address / allocation |
| ----- | ------------------ | -------------------- |
| cache | TensorType([8,512], BF16), canonical ND stride [512,1] | Caller GM pointer |
| v | TensorType([2,512], BF16), TensorView stride [1536,1], layout ND, addressing kind AffineSubview | No MemRef, no new payload allocation |
| pair | TileType([2,512], BF16), eventually Vec | Fresh result Tile allocation |
| out | Store result for output | Ordinary output GM alias |

The builder/type test asserts the four operands and result metadata above.
The fourth IR operand is always explicit, including omitted-steps normalization.
The result alias resolves to cache, but pair is fresh storage. Source origin 512
is recovered from the Call; it must not be added to MemRef or inferred again
from the smaller shape. The initial load target can be unset; InferTileMemorySpace
later assigns Vec without changing the GM descriptor.

## 7. Compiler changes

### 7.1 Tensor-to-Tile conversion

Add `tensor.subview` to the explicit metadata passthrough set. Check its rewritten
input is still an ordinary GM Tensor and rebuild its deduced descriptor when substitutions
change geometry. Never register it as an op returning a Tile, preload its payload
at entry, or put its result in the global Tensor-to-Tile remapping table.

Consumer-space propagation may pass through it to place a later load, but must
not turn its GM result into a Tile. Consumer-local bridging can load it for
ordinary compute. A view shared by explicit load and compute remains GM, while
each use receives the appropriate representation. Avoid moving a load before a
write to the root or across branch boundaries. Cache declarations must resolve
through subview aliases to the original GM root.

Before/after intent for the issue (schematic):

```python
# Before and after ConvertTensorToTileOps: this descriptor chain stays intact.
v = pl.tensor.subview(cache, [2, 512], [lo, 0], steps=[delta, 1])
t = pl.tile.load(v, [0, 0], [2, 512], [2, 512])
```

The pass may add existing output stores/signature bindings around it. Add an
operand-contract check for passthrough `tile.load` after argument rewriting so
the old `slice -> load` case fails with an actionable source diagnostic instead
of leaking a Tile-input load to later stages. This is a bounded diagnostic repair,
not an automatic reinterpretation of old slice semantics.

`tensor.set_validshape` currently has an unconditional simple conversion to
`tile.set_validshape`. Override that decision only for subview-derived GM aliases:
before conversion lookup/bridging, retain the Tensor op, deduce a Tensor result
with the same shape/stride, no MemRef and narrowed valid extents. The prediction
and preload analyses must apply the identical provenance predicate so they do
not mark that result as Tile. Ordinary Tensor set_validshape keeps its existing
conversion. Check each requested valid extent against the parent's effective
valid extent (both false and unprovable narrowing fail).
Thus `subview -> set_validshape -> subview/load` is explicitly supported.
Register a descriptor-only backend handler that aliases the current pointer and
tensor-view SSA; tile.load already consumes validity from the result type. Emit
no data load, tile.set_validshape or allocation for this metadata operation.

### 7.1.1 Reference-case pass checkpoints

The reference-case body in section 6.2 already contains tile.load and tile.store.
ConvertTensorToTileOps must leave that operation sequence intact. It must not
emit a preload for v or lower v itself to Tile. The required component changes
and before/after assertions are:

| Pass / component | Before -> after for this case | Required adjustment |
| ---------------- | ----------------------------- | ------------------- |
| SSA / outlining | Local descriptor and scalar geometry stay inside the consuming kernel | Do not outline a descriptor into a new function parameter or lose a required guard |
| Bounds verification | Pending geometry -> proven row/column containment and strict reduction | New shared contextual visitor before conversion and again before codegen; false/unknown rejects |
| ConvertTensorToTileOps | tensor.subview + tile.load -> tensor.subview + tile.load | Explicit metadata passthrough; prediction/preload analysis preserves GM identity; validate rewritten load source |
| ConsumerSpaceCollector | Vec demand at load -> demand propagated to appropriate consumer | No global descriptor-to-Tile replacement |
| MaterializeTensorStrides / canonical verifier | Explicit [1536,1], AffineSubview -> same | Preserve affine kind/stride; distinguish canonical and affine rules |
| InitMemRef / memory planning | Borrowed GM descriptor -> no MemRef; result Tile -> normal placement | Skip descriptor allocation and inherited-offset MemRef synthesis |
| Composite / ND/layout passes | Rank-2 ND geometry -> same coordinate map | No packed-stride reconstruction; diagnose unsupported rewrites |
| Final backend check | Proven geometry -> proven target-legal load | Validate dtype, layout, pitch, alignment and integer widths independently |

Use separate Before/Expected structural tests for conversion and placement:
conversion preserves v and its load input identity; after memory-space inference
pair is Vec. Materialization preserves [1536,1] and AffineSubview; after InitMemRef,
v still has no MemRef while pair uses ordinary Tile/Buffer allocation rules.
A zero-sized descriptor MemRef fails this contract. These are independently
verified checkpoints, not one assumed transformation.

### 7.2 Alias identity, lifetime and read-only use

Use the shared result-alias query for nested subviews and allowed metadata
operations. Track subview provenance through aliases, tuples and control-flow
edges once with indexed worklists. Reject writes through any derived alias,
including an ordinary view of the subview; do not rely on Python wrapper types.
Loading does not mark its fresh Tile as an alias of GM.

The view is not a snapshot. Writes through a separate root handle remain legal
under existing ordering/race rules; later loads observe the ordered memory state.
Dependency analysis conservatively attributes subview reads to the root allocation
when exact strided-region overlap cannot be proved. Never infer disjointness from
the descriptor's dense element count. Extend root lifetime through all uses.

Before `InitMemRef`, index the borrowed-descriptor definitions and allowed alias
descendants once. For those Tensor values bypass both `ProcessNormalVar`'s fresh
MemRef creation and `ShareMemRefFrom`'s inherited-offset path; remap type geometry
but leave `memref_` absent. HasMemRefs currently requires MemRefs for Tiles, not
GM Tensors; retain that distinction and add coverage for the descriptor case.
Memory reuse, address allocation and memory-footprint utilities must ignore
borrowed descriptors as allocation candidates and query root alias identity
instead when needed. No descriptor envelope is synthesized. This is distinct
from ordinary tensor.slice's existing view-MemRef behavior. The load result
still receives an ordinary MemRef and physical Tile allocation.

### 7.3 Control flow and scope

Required, subject to successful contextual bounds proofs: construct and load
descriptors inside either branch; construct a
descriptor from scalar values merged by `if`; construct and consume a fresh
descriptor in each loop iteration with changing scalar offsets/steps.
Descriptor values may remain live in their defining scope across ordinary uses.

Do not allow branch/loop results carrying a subview descriptor outside that scope
in the first release. Diagnose even same-shape phis with different origin/stride;
merging only shape is insufficient. Rewrite such code by merging/carrying scalar
geometry first and constructing the view at the load site. Non-inlined call,
return and orchestration boundaries are likewise rejected before codegen.

### 7.4 PTO code generation

Maintain per-SSA descriptor state: root pointer, current origin in elements,
shape and stride. Resolve nested aliases with memoization, never repeated scans.
For a subview, calculate the offset against the immediate source's strides;
emit `pto.addptr` and `pto.make_tensor_view` in the consuming function. Bind both
the current-origin pointer and tensor-view SSA for downstream view/load handlers.
`GetTensorBasePtr` on a subview must mean its current-origin pointer; keep root
identity separately for dependencies. Existing `tensor.view` then preserves the
subview origin rather than accidentally returning to the root pointer.

Reuse the inline addptr/make_tensor_view pattern already present in
`src/backend/common/pto_ops_distributed.cpp`; do not introduce a descriptor-returning
MLIR helper. Backend output below is schematic, not assembled evidence:

```text
row_stride = wide_index(delta) * 512
origin     = wide_index(lo) * 512
p          = pto.addptr cache_pointer, origin
v          = pto.make_tensor_view p, shape=[2,512], strides=[row_stride,1]
window     = pto.partition_view v, offsets=[0,0], sizes=[2,512]
             pto.tload window -> tile[2,512]
```

One tload in generated PTO is not proof of one hardware DMA. No hardware-specific
limit or instruction guarantee is asserted by this RFC without measurement.

### 7.4.1 Lowering stages and final PTO artifacts

Keep `tensor.subview` as an explicit GM descriptor operation until PTO codegen;
there is no conversion to `tile.slice`, `tile.load`, or a new PTO dialect op.
The existing pipeline requires the following contracts:

| Stage | Required handling |
| ----- | ----------------- |
| SSA / simplification / outlining | Bind geometry to SSA definitions; keep descriptors local to the consuming kernel. A guard outside an outlined kernel is not automatically a fact inside it. Preserve an actual dominating guard there or reject. |
| Before Tensor-to-Tile | Prove parent-window containment; preserve subview as ordinary GM Tensor; lower its slice/compute consumers locally to loads. |
| Composite lowering / ND flatten / layout legalization | Preserve the original source-coordinate affine map. The initial rank-2 ND path needs no flattening; reject unsupported rank/layout changes before rewriting, rather than reconstructing packed strides. |
| Stride materialization / InitMemRef / memory planning | Keep origin, explicit stride and addressing kind; do not allocate the borrowed descriptor. Only load results participate as Tiles under the selected memory planner. |
| Final verification / backend eligibility | Re-prove containment after substitutions and control-flow changes; prove dtype/layout, pitch, alignment and width limits. Cached proof success from an earlier IR is insufficient. |
| PTO emission / PTOAS | Emit scalar address arithmetic and standard descriptor ops, then the existing partition/tload path. Assemble the resulting PTO with the installed target toolchain and inspect its lowering. |

In particular, extending a codegen lookup table is insufficient: the new operator
needs a registered backend handler, result-pointer binding and tensor-view binding.
`GetTensorBasePtr(v)` is the pointer at v's current origin; the root allocation key
remains separate. Do not add the origin again in `tile.load`. A load's offsets
stay relative to v and `pto.partition_view` applies them using v's sampled stride.
For load offsets `q`, the accessed start is `B + sum(offset*S) + sum(q*S*step)`.
`set_validshape` changes effective read sizes, not the pointer or pitch. A nested
subview composes the immediate-parent stride exactly once; optional folding may
remove redundant descriptors only after these identities and bounds are proven.

For the static reference case, the complete target PTO module is below. It is
a handwritten backend probe; implementation tests separately compile the real
API output. Constants, allocation, output store and return are all defined:

```text
// RFC #3042 handwritten backend probe; real API output is tested separately.
// Input: packed BF16 [8,512]. Select rows 1 and 4, then copy to output [2,512].
!pair_tile = !pto.tile_buf<loc=vec, dtype=bf16, rows=2, cols=512, v_row=2, v_col=512, blayout=row_major, slayout=none_box, fractal=512, pad=0>
module attributes {pto.target_arch = "a2a3"} {
  func.func @subview_static(%cache: !pto.ptr<bf16>, %out: !pto.ptr<bf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c512 = arith.constant 512 : index
    %c1536 = arith.constant 1536 : index
    %p = pto.addptr %cache, %c512 : !pto.ptr<bf16> -> !pto.ptr<bf16>
    %v = pto.make_tensor_view %p, shape = [%c2, %c512], strides = [%c1536, %c1] {layout = #pto.layout<nd>} : !pto.tensor_view<?x?xbf16>
    %read = pto.partition_view %v, offsets = [%c0, %c0], sizes = [%c2, %c512] : !pto.tensor_view<?x?xbf16> -> !pto.partition_tensor_view<2x512xbf16>
    %pair = pto.alloc_tile : !pair_tile
    pto.tload ins(%read : !pto.partition_tensor_view<2x512xbf16>) outs(%pair : !pair_tile)
    %out_view = pto.make_tensor_view %out, shape = [%c2, %c512], strides = [%c512, %c1] {layout = #pto.layout<nd>} : !pto.tensor_view<?x?xbf16>
    %write = pto.partition_view %out_view, offsets = [%c0, %c0], sizes = [%c2, %c512] : !pto.tensor_view<?x?xbf16> -> !pto.partition_tensor_view<2x512xbf16>
    pto.tstore ins(%pair : !pair_tile) outs(%write : !pto.partition_tensor_view<2x512xbf16>)
    return
  }
}
```

The fixture is maintained in
[gm_subview_static.pto](../../../../tests/fixtures/gm_subview_static.pto).
The handler must call RegisterBasePtr(v, current_pointer), RegisterTensorView
and RegisterVarToMlir for v, so existing view/load handlers resolve the descriptor
rather than recreating a packed root view. The final store addresses output,
not the read-only v. Constants may fold, but origin 512 and row pitch 1536 must
remain semantically exact; no second origin addition is allowed.

The concrete artifact changes are one current-origin pointer, an explicit sampled
pitch, a tensor-view/partition pair, and scalar arithmetic (constant cases may
fold). `AffineSubview` is an internal IR distinction; it is not printed as a new
PTO layout or dialect attribute. The source GM pointer ABI is unchanged; captured
geometry may use ordinary scalar kernel parameters when outlining requires them.
No descriptor buffer is added to the kernel ABI or workspace. The only new
payload allocation attributable to this read is the result Tile: logical
`[2,512]` BF16 data is 2048 bytes before existing physical-layout/alignment rules.
There must be no `[2*delta,512]` GM/UB temporary or envelope `tload`. Descriptor
ops may remain even if the root parameter already has its own tensor view.

The generated partition extents are the requested valid load region, not the
parent window or address envelope. Root cache policy/L2 offset propagation and
existing load attributes still apply; geometry offset and cache-bypass byte
offset must remain distinct. Unrelated kernels keep their existing PTO behavior.
Test nested/partial-load artifacts, exact sampled strides, pointer-origin
composition, descriptor allocation absence and physically allocated Tile sizes.
PTOAS success is a separate test from textual codegen, and a single textual
`tload` remains separate from the final DMA-count/performance acceptance.

### 7.5 Native-load support and fallback policy

The initial executable path targets rank-2 Vec loads with contiguous columns on
A2/A3, validated against the installed PTOAS v0.67. Preserve existing target-memory
selection; reject a subview load requested or inferred into an unvalidated memory
space/backend with a source-level unsupported-path error.

Query architecture rules via registered backend operators / the PassContext
backend handler. They include row-byte alignment, row count, source pitch/gap
range, and index/DMA parameter widths. Do not put architecture constants in the
Python API or a generic IR pass. Establish exact limits from backend/toolchain
evidence before implementing the native path; no guessed numeric gap limit is
part of this design.

For dynamic values, prove the native-path range/alignment constraints using the
same scoped facts as containment; unknown backend constraints reject the
specialization. Test recognized device guards and caller-written two-single-row
loads separately, with valid read bounds on each path. A backend without dynamic
row-pitch support rejects the specialization. Never truncate a dynamic pitch. There is no automatic large-span transfer,
hidden gather kernel or automatic fallback in the first release. Static unsupported
geometry produces a compilation error. The generic API does not promise one DMA;
issue closure additionally requires native one-DMA evidence within documented limits.

## 8. Affected paths and implementation order

Paths below are required implementation targets or audit points, not changes
made by this RFC. Use existing file organization rather than adding global flags.

| Order | Paths | Required work |
| ----- | ----- | ------------- |
| 1 | `src/ir/op/tensor_ops/memory.cpp`; `src/ir/op/tensor_ops/transform.cpp`; `include/pypto/ir/type_inference.h`; `src/ir/op/type_inference.cpp` | Deduction, normalized geometry, validity/overflow checks, set_validshape narrowing and registration |
| 2 | `python/pypto/ir/op/tensor_ops.py`; `python/pypto/language/op/tensor_ops.py` | Builders, exports, normalization, errors and public docstrings |
| 3 | `include/pypto/ir/transforms/utils/tensor_view_semantics.h`; `src/ir/verifier/verify_tensor_view_canonical.cpp`; `src/ir/transforms/materialize_tensor_strides_pass.cpp` | Separate canonical/affine verification; preserve explicit strides and kind |
| 4 | `src/ir/transforms/convert_tensor_to_tile_ops_pass.cpp`; `src/ir/transforms/op_conversion_registry.cpp` | GM passthrough, consumer-local bridging, load operand check, provenance restrictions |
| 5 | `src/ir/transforms/utils/result_alias_utils.cpp`; `src/ir/transforms/utils/op_predicates.cpp`; `src/ir/transforms/utils/return_lineage_utils.cpp`; `src/ir/transforms/init_memref.cpp` | Alias/lifetime/read-only audit; borrowed descriptors with absent MemRef |
| 6 | `include/pypto/ir/transforms/utils/memref_utils.h`; `src/ir/transforms/memory_reuse_pass.cpp`; `src/ir/transforms/allocate_memory_addr_pass.cpp`; `src/ir/transforms/optimize_orch_tensors_pass.cpp` | No borrowed-descriptor allocation; preserve origin/pitch and avoid unsupported hoisting |
| 7 | New `src/ir/verifier/verify_tensor_subview.cpp`; `include/pypto/ir/arith/analyzer.h`; `src/ir/transforms/auto_derive_task_dependencies_pass.cpp`; `src/ir/verifier/type_check_pass.cpp`; `src/ir/verifier/warning_in_param_written.cpp` | Contextual containment verification at conversion/codegen entry; root dependencies and diagnostics |
| 8 | `src/backend/common/pto_ops_memory.cpp`; `src/codegen/pto/pto_codegen.cpp`; `include/pypto/codegen/pto/pto_codegen.h`; `include/pypto/backend/common/backend_handler.h`; backend handler implementations | Origin-aware descriptors, set_validshape alias handler, backend constraints and native Vec load |
| 9 | `python/pypto/debug/torch_codegen.py` | Positive-step Python/Torch slicing as a reference; preserve alias and parent pitch |
| 10 | Existing IR printer/serialization, binding and stub paths | Addressing-kind enum/field binding, reflection, printing and compatible serialization |
| 11 | Tests and paired English/Chinese operator/pass/codegen docs | API, correctness, allocation and hardware acceptance evidence |

The generic `create_op_call` route means a Python operator does not require a
dedicated nanobind function. The new addressing-kind enum and TensorView field
require synchronized updates in `include/pypto/ir/type.h`,
`python/bindings/modules/ir.cpp` and `python/pypto/pypto_core/ir.pyi`, together with
reflection, serializer/deserializer and type-printer handling. The Python printer must retain the fourth
IR operand; the DSL printer may spell it as `steps=`. Generic serialization must
recognize the registered operator without a new node kind.

Keep new analyses O(N+E), or O(N log N) with ordered maps. Index definitions and
alias uses once; process control-flow/alias edges with bounded worklists. Do not
walk the entire alias chain independently at every consumer. Avoid expanding
symbolic arithmetic trees repeatedly; reuse expression DAG nodes.

Implementation sequence begins with **Gate 0: a minimal PTOAS/backend feasibility
probe (static Gate 0a and dynamic/target Gate 0b; see section 9.1)**, kept under tests and using the current worktree/toolchain. Establish a
support matrix for BF16/FP16/FP32, dynamic/static pitch, alignment, row counts,
parameter widths and available simulator/device trace tooling. BF16 proven-bounded dynamic
pitch is mandatory for issue closure; unsupported optional dtype/target entries
must be explicitly rejected and documented. Do not proceed with the native-path
implementation if this probe fails; revise the backend design first.
Then add semantic/negative tests, implement deduction and API, add conversion/
provenance checks, preserve storage identity, implement validated backend
descriptors, run end-to-end tests and update references.
Do not expose a partially implemented public API before the executable path works.

## 9. Validation and acceptance

Proposed test additions and changes:

| Layer/path | Implemented coverage |
| ---------- | -------------------- |
| `tests/ut/ir/operators/test_tensor_subview.py` | Default steps, generic rank/inner steps, parent pitch and validity, invalid geometry/source types, serialization and integer widening |
| `tests/ut/codegen/test_tensor_subview.py` | DSL outlining, standalone conversion/codegen safety gates, guarded dynamic shape/offset/step, scalar phi and loops, nested origins and view aliases, slice consumers, allocation, CPU reference and actual PTOAS A2/A3 compilation |
| `tests/ut/codegen/test_gm_subview_rfc_fixture.py` | Independent handwritten static/dynamic PTOAS fixtures for A2/A3 |
| Existing operator acceptance and backend inventory tests | Operator classification, Buffer IR exclusion and backend registrations |
| `tests/st/runtime/ops/test_tensor_subview.py` | Runtime harness row-selection case for simulator/device; codegen-only validation is available without numerical execution |

Use coordinate-coded inputs so wrong rows or columns cannot match accidentally.
Cover `delta=1`, a large valid gap, last legal selected row, reversed original
indices, equal-row caller fallback, column offset, nested views, both device
branches and multiple loop iterations in the same specialization. Invalid-index
fallbacks must avoid memory access. Include alias writes through view/tuple
aliases, illegal descriptor scope escape, real computed Tile sources and
overflow/alignment boundary tests. Include rejected equal parent/new windows, a smaller
shape whose offset/step still exceeds the parent, unknown scalar bounds, guarded
acceptance versus an unguarded sibling branch, facts lost at phi/outlining, and
rejection after a transformation invalidates a previously established proof.

Compare against direct reference indexing `cache[[lo, lo + delta], :512]`.
Verify UB allocation artifacts for the actual physical `[2,512]` result and
absence of any `delta`-sized allocation/transfer. Instruction/simulator or device
trace inspection must show one strided DMA for supported native inputs; PTO text
alone is insufficient. If that evidence is unavailable, report semantic support
separately and leave the issue's one-DMA acceptance item open.

Run from this worktree using the repository testing skill:

```bash
source .claude/skills/testing/load-env.sh
cmake --build build --parallel "$PYPTO_BUILD_JOBS"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest tests/ut/ -n "$PYPTO_TEST_JOBS" -v
python -m pytest tests/st/runtime/ops/test_tensor_subview.py --codegen-only -v
python -m pytest tests/st/runtime/ops/test_tensor_subview.py -v
```

The runtime test exists at the path above. Numerical execution requires the
pinned runtime and the corresponding simulator/device environment; codegen-only
validation does not establish numerical correctness. Never run the full ST suite locally. Record
exact toolchain/backend versions and unavailable hardware checks. Follow repo
lint/type-check/pre-commit commands for the implementation's changed paths.

Release checklist:

- [ ] Existing slice/view programs retain behavior; invalid Tile-input load fails early.
- [ ] GM subview descriptors survive conversion without payload loads/allocations.
- [ ] Nested origin and pitch remain correct through view and explicit load.
- [ ] Only proven-bounded device-side geometry is accepted; unknown bounds reject without host readback.
- [ ] Branch/loop usage works within scope; unsupported escapes/writers fail clearly.
- [ ] Only result Tile storage is allocated; no envelope materialization.
- [ ] Backend limits are documented with measured PTOAS/target evidence.
- [ ] Native two-row path meets the issue's one-DMA target, or that item remains open.
- [ ] English/Chinese docs, printing, serialization and reference execution agree.

### 9.1 Evidence gates derived from the reference example

Gate 0 has two independently tracked parts. **Gate 0a** is the static BF16
backend fixture: PTOAS 0.67 accepts it for A2 and A3, yielding one [2,512] BF16
Vec Tile, a source row stride of 1536, one TLOAD and one TSTORE. Its 2048-byte
logical payload must not be confused with the 4096-byte selected address envelope.
**Gate 0b**: bounded dynamic BF16 pitch compiles with PTOAS 0.67 for A2/A3.
The implementation restricts native loads to rank-2 ND BF16 Vec, unit inner stride,
at most 4095 physical rows, 32-byte valid/physical row alignment, signed 32-bit
index/address bounds and unsigned 32-bit source-gap bytes. These limits follow
the installed PTO-ISA ND-to-ND load implementation; unsupported or unknown facts
reject. Other dtype/target profiles and final machine DMA evidence remain open.

The existing fixture test is
[test_gm_subview_rfc_fixture.py](../../../../tests/ut/codegen/test_gm_subview_rfc_fixture.py).
It tests handwritten static and dynamic PTO.
[test_tensor_subview.py](../../../../tests/ut/codegen/test_tensor_subview.py)
executes the real API through default passes, checks allocation and generated
loads, and compiles actual static and guarded dynamic PTO for A2/A3 (level3,
because the default allocation pass supplies explicit Tile addresses). Device acceptance compares every output element with cache[[1,4],:].
The complete CPU reference and commands are in the
[worked example](gm-subview-worked-example.md#5-independent-dataaddress-check-and-evidence).
API/pass behavior, CPU reference values, current-worktree build and PTOAS
compilation are covered by implementation tests. Single hardware DMA and device
numerical execution remain unverified; neither follows from one emitted TLOAD.
The 2026-10-23 build/pytest blockers have been resolved in this worktree.

## 10. Alternatives and remaining evidence gates

| Alternative | Reason not selected |
| ----------- | ------------------- |
| `slice(..., gm_view=True)` | One operator switches materialization contracts via an attribute |
| Globally defer slice based on explicit-load users | Changes existing semantics and mixed-user/control-flow decisions; separate project |
| Add arbitrary `strides=` directly to load | Does not provide a composable logical window or bounded parent-coordinate mapping |
| General `as_strided`/new tensor kind | Larger invariant, safety and ABI scope than required for this issue |
| Always load the enclosing span | Transfer and UB use grow with gap; defeats the requirement |

The design fixes API/IR/scope choices; builder discretion is limited to helper
names, placement inside existing files and diagnostic wording. Before closing the original performance request, establish device numerical
correctness and one-DMA evidence in addition to the compiler checks above. A failed toolchain
probe requires revising the backend plan explicitly, not silently weakening the
semantics or substituting a large-span load.
