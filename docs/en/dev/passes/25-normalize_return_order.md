# NormalizeReturnOrder Pass

Reorders the return tuple of every InCore function so that `return[i]`
corresponds to the i-th `Out`/`InOut` parameter in declaration order, and
synchronizes the result `TupleType`, binding `Var`, and `TupleGetItemExpr`
indices in non-InCore callers accordingly. After this pass, positional
consumers see a canonical InCore result tuple, while orchestration codegen can
read the explicit return-position-to-parameter map without tracing through
`tile.store` / `ForStmt` yield chains.

## Overview

User code is free to write `tile.store` calls in any order — `out_b`
before `out_a`, or interleaved with compute. Earlier in the pipeline, the
body order is preserved verbatim, so the InCore `ReturnStmt::value_` may
list its outputs in an order that does not match the declared `Out`/`InOut`
parameter order. Without normalization, orchestration codegen would have
to follow each `return[i]` back through assignments and `tile.store` calls
to discover which parameter it materializes — analysis that belongs in a
pass, not in codegen (see `docs/en/dev/codegen/00-pto_codegen.md`).

This pass canonicalizes the contract so codegen can rely on
`return[k] ↔ out_indices[k]` by position alone:

1. **Step A0 (param-return canonicalization)** — for every `InCore`,
   `Group`, and `Spmd` function, rewrite each tensor return value that is a
   param writeback to reference the parameter directly (pointer identity),
   using the shared `return_lineage` utility. Kernel-allocated outputs (not
   traceable to any param) and scalar returns are exempt and stay unchanged.
2. **Step A (InCore rewrite)** — for every `InCore` function, compute a
   permutation that sorts `ReturnStmt::value_` to match the declared
   `Out`/`InOut` parameter order, then rewrite both the return values and
   `Function::return_types_` accordingly.
3. **Step B (call-site remap)** — for every non-InCore function
   (Orchestration / Group / Spmd / opaque), permute the result `TupleType`
   of each `Call` / `Submit` to a function reordered in Step A, mint a
   matching binding `Var` and remap its uses by identity, then rewrite every
   `TupleGetItemExpr.index_` on that result. The new index is
   `permutation[old_index]`, so observers still see the same SSA values bound
   to the same names. A `Submit` permutes only the callee-return prefix; its
   trailing `Scalar[TASK_ID]` stays fixed. Before applying a permutation, the
   pass verifies that every program-local result of a candidate callee is
   consumed only by direct element projections; whole-tuple aliases,
   control-flow carries, returns, and arguments are rejected rather than
   silently changing their contract.

An identity permutation skips Steps A and B for that callee, but Step A0 may
still replace traceable return aliases with their parameter Vars. A program
with no `InCore`, `Group`, or `Spmd` functions is a complete no-op.

**Pipeline position**: slot #25 in the `Default` strategy — after
`StampTfreeSplit` (#24) and before `SkewCrossCorePipeline` (#26),
`LowerPipelineToSlots` (#27), and `LowerPipelineLoops` (#28). It runs late
enough that all kernel splitting / tile-structural decisions are made on the
original return order, and early enough that downstream tile-level passes
(`CanonicalizeIOOrder`, `InitMemRef`, `MemoryReuse`, `AllocateMemoryAddr`) —
and ultimately PTO orchestration codegen — see the canonical order.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::NormalizeReturnOrder()` | `passes.normalize_return_order()` | Program-level |

```python
from pypto import passes
result = passes.normalize_return_order()(program)
```

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | `SplitIncoreOrch`, `IncoreTileOps` |
| Produced | `ReturnParamsExplicit` |
| Invalidated | — |

`SplitIncoreOrch` guarantees that InCore work has been outlined into its
own functions; `IncoreTileOps` guarantees the body uses tile ops, so the
`tile.store(_, _, out_param)` signal that drives Step A is present. The
pass produces `ReturnParamsExplicit` (verified by
`verify_return_params_explicit.cpp`): every InCore/Group/Spmd tensor
return value that is a param writeback references the param by pointer
identity, so orchestration codegen maps returns to args with a lookup.
It invalidates nothing — SSA form, normalized statement structure, memory
inference, and every other upstream property are preserved.

## Algorithm

### Step A0 — Canonicalize return values to params

For each `InCore` / `Group` / `Spmd` function, `CanonicalizeReturnValues`
calls `return_lineage::ReturnedParamIndices` (which traces var-to-var
aliases, loop carries, tensor `IfStmt` merges whose branch values resolve to
the same parameter, builtin writebacks, `TupleGetItem` of tuple calls, and
Group/Spmd wrapper calls) and replaces every tensor return value that traces
to a param with the param `Var` itself. Untraceable values
(kernel-allocated outputs) and scalars keep their original expression.

**This step is what makes the return→param map readable without an analysis.**
Once it has run, return position `j` writes back param `i` exactly when
`ReturnStmt->value_[j]` *is* `params_[i]` by pointer identity — which is what
`IRProperty::ReturnParamsExplicit` asserts. Consumers at or after this pass
(orchestration codegen, `ClassifyIterArgCarry`) therefore call
`return_lineage::ExplicitReturnedParamIndices(func)`, a local structural read,
rather than re-running the interprocedural tracer. Reserve
`ReturnedParamIndices` for callers that run *before* the property exists
(`ExpandMixedKernel`, the scope outliner), for this pass itself, and for the
property verifier, which must re-derive independently to have anything to check.

Because it is a codegen precondition, a test that hand-builds IR and calls
orchestration codegen directly must run this pass first (see
`tests/ut/codegen/_orchestration_codegen_common.py`), exactly as it must run
`DeriveCallDirections`, `MaterializeRuntimeScopes` and `ClassifyIterArgCarry`.

### Step A — Compute and apply per-function permutations

For each `InCore` function, `BuildReturnToParamMapping` walks the body
once (excluding the trailing `ReturnStmt`) and builds a
`Var* → out_param_index` map by replaying three rules:

| Rule | Pattern | Action |
| ---- | ------- | ------ |
| 1. `tile.store` writes an Out/InOut buffer | `lhs = tile.store(tile, offsets, out_param, ...)` | `lhs → param_index_of(out_param)` |
| 2. Var-to-var alias | `lhs = rhs_var` (and `rhs_var` already mapped) | `lhs → lookup(rhs_var)` |
| 3. `ForStmt` iter-arg yield | `for_stmt.iter_args[i].initValue_` already mapped | `for_stmt.return_vars_[i] → lookup(initValue)` |

Each value of `ReturnStmt::value_` is then resolved by looking up its
`Var` in this map, falling back to direct identity match against
`Function::params_`. Returning `kNoParam` for an entry means "no out-param
linkage detected" — that slot keeps its original index.

`ComputeReturnPermutation` turns the mapping into
`permutation[old_index] = new_index`, where `new_index` is the position
of the matching parameter in `CollectOutIndices(func)`. The function
returns the empty permutation in four cases (each skips Step A while retaining
any Step-A0 canonicalization):

- The body has no non-empty `ReturnStmt` (open IR) or no Out/InOut parameters.
- `out_indices.size() > ret_to_param.size()` — more declared output
  parameters than returned values, so the analysis is incomplete and we
  refuse to construct an out-of-bounds permutation.
- The proposed permutation is not bijective (a duplicate target, a hole, or an
  out-of-range target).
- The computed permutation is the identity (already canonical).

When the permutation is non-empty, `ReorderReturns` builds a fresh
`Function` via `MutableCopy`, replacing the trailing `ReturnStmt` with one
whose `value_[permutation[i]] = old_value_[i]` and permuting
`Function::return_types_` in lockstep so the type list stays aligned with
the values.

### Step B — Synchronize call-site tuple types and projections

Before rewriting, a read-only preflight indexes each candidate call-result
binding and verifies that all of its uses are
`TupleGetItemExpr(binding, index)` in a non-InCore caller. A candidate call
that is nested, appears in an InCore caller, or whose result escapes as a whole
tuple (through an alias, `YieldStmt`/loop carry, `ReturnStmt`, or call argument)
is rejected with a source-located error. Supporting those forms would require
an explicit inverse-permutation tuple adapter; changing their tuple order in
place would be a silent semantic change when element types happen to match.

After that atomic preflight, `TupleIndexPermutationMutator` does a single SSA
pass over each non-InCore function that:

- Rebuilds every `Call(GlobalVar)` / `Submit(GlobalVar)` to a function
  reordered in Step A with its result `TupleType` permuted by the same map.
  For `Submit`, only the first `N = permutation.size()` elements move; the
  final `Scalar[TASK_ID]` is validated and preserved in place, together with
  `deps_`, attrs, kwargs, launch fields, and predicate.
- For an assigned call/submit result, mints a binding `Var` with that new
  tuple type, records `old_var → new_var` in the mutator's identity remap, and
  tracks `new_var → permutation_ref` in `reordered_tuple_vars_`. All later
  uses therefore reference the same, correctly typed tuple definition.
- Clears stale identity-remap and tracking state before recording a new
  definition of the same `Var`. The RHS is visited first, so any RHS use sees
  the preceding complete definition; only then does the new definition replace
  the old tracking state.
- For every `TupleGetItemExpr(tuple_var, k)` whose `tuple_var` is in the
  tracked map, rewrites the index to `permutation[k]`.

Because Step A rewrites function signatures and Step B rewrites call-site
result types, binding Vars, and index access in the same pass invocation, the
program is type-consistent at exit: each tuple element is still bound to the
same physical output buffer, just under a new index.

## Constraints

| Constraint | Reason |
| ---------- | ------ |
| Only `InCore` functions are rewritten in Step A | Other function kinds (`Orchestration` / `Group` / `Spmd` / opaque) follow the user's declared return shape; their callers are remapped in Step B. `Group`/`Spmd` returns are still canonicalized to params in Step A0, but never reordered |
| Step A0 leaves kernel-allocated outputs and scalars untouched | Only param writebacks must be explicit; a return value with no param lineage has no param to reference |
| Skips Step A where `out_indices.size() > ret_to_param.size()` | An incomplete analysis must not produce an out-of-bounds permutation; any Step-A0 param-reference canonicalization is retained |
| Permutation is identity ⇒ no Step-A reorder | Avoids spurious reorder clones while retaining any Step-A0 param-reference canonicalization |
| Reordered call results must be consumed through direct `TupleGetItemExpr` projections in non-InCore callers | Whole-tuple aliases, control-flow carries, returns, arguments, and InCore call sites cannot be remapped locally without an inverse-permutation adapter, so the pass rejects them before changing any function |
| Step B tracks the freshly typed binding `Var`, not the pre-rewrite definition | Identity remapping sends every use to the new tuple definition before projection lookup; each new definition replaces the prior tracking state, preventing stale bindings |
| `Submit` permutes only its callee-return prefix | The trailing `Scalar[TASK_ID]` belongs to task launch semantics rather than the callee return and must retain its final tuple index |

## Example

Two `Out` parameters with the InCore body writing them in the wrong
order. The orchestrator picks `ret[0]` and `ret[1]` assuming those are
`out_a` and `out_b`. After the pass, the InCore return matches the
parameter order and the orchestrator's `TupleGetItemExpr` indices are
remapped so the same SSA values still flow into `a` and `b`.

**Before**:

```python
@pl.program
class Module:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[16], pl.FP32],
               out_a: pl.Out[pl.Tensor[[16], pl.FP32]],
               out_b: pl.Out[pl.Tensor[[16], pl.FP32]]) \
            -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]]:
        x_tile = pl.load(x, [0], [16])
        a_tile = pl.tile.add(x_tile, x_tile)
        b_tile = pl.tile.mul(x_tile, x_tile)
        out_b_store = pl.store(b_tile, [0], out_b)
        out_a_store = pl.store(a_tile, [0], out_a)
        return (out_b_store, out_a_store)        # ← wrong order vs. (out_a, out_b)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x, out_a, out_b):
        ret = self.kernel(x, out_a, out_b)
        a = ret[0]                                # ← currently materializes out_b
        b = ret[1]                                # ← currently materializes out_a
        return (a, b)
```

**After**:

```python
@pl.program
class Module:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x, out_a, out_b):
        x_tile = pl.load(x, [0], [16])
        a_tile = pl.tile.add(x_tile, x_tile)
        b_tile = pl.tile.mul(x_tile, x_tile)
        out_b_store = pl.store(b_tile, [0], out_b)
        out_a_store = pl.store(a_tile, [0], out_a)
        return (out_a_store, out_b_store)        # ReorderReturns: permutation [1, 0]

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x, out_a, out_b):
        ret = self.kernel(x, out_a, out_b)
        a = ret[1]                                # TupleIndexPermutationMutator: 0 → 1
        b = ret[0]                                # TupleIndexPermutationMutator: 1 → 0
        return (a, b)
```

The same SSA assignments keep their original physical outputs: `a` remains
bound to the value produced for `out_b`, and `b` remains bound to the value
produced for `out_a`. Only the paths through the tuple have changed. `InOut`
parameters behave identically.

See `tests/ut/ir/transforms/test_normalize_return_order.py` for the
representative cases:

- `test_swapped_returns_reordered` — the two-Out-param example above
- `test_already_ordered_noop` — Step A skips an identity permutation while
  Step A0 canonicalizes param writebacks
- `test_single_return_noop` — a single Out param needs no permutation
- `test_non_incore_unchanged` — the plain Orchestration-only test program is a no-op
- `test_three_returns_scrambled` — three-way permutation
- `test_2d_tensor_reorder` — 2-D tensors / multi-dim offsets
- `test_inout_param_reorder` — `InOut` participates in reordering

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass NormalizeReturnOrder();
```

**Implementation**: `src/ir/transforms/normalize_return_order_pass.cpp`

- `CanonicalizeReturnValues` — Step A0 rewriter: replaces traceable
  tensor return values with the param `Var` (via
  `return_lineage::ReturnedParamIndices`).
- `BuildReturnToParamMapping` — Step A analysis: walks the function body
  to map each `ReturnStmt` value back to an Out/InOut parameter index.
- `CollectOutIndices` — collects the parameter positions whose
  `ParamDirection` is `Out` or `InOut`.
- `ComputeReturnPermutation` — composes the previous two into the final
  `permutation[old_index] = new_index`; returns empty when no rewrite
  is needed or the analysis is incomplete.
- `ReorderReturns` — builds a `MutableCopy(func)` with the permuted
  `ReturnStmt::value_` and `Function::return_types_`.
- `FindUnsafeReturnPermutations` — preflights every candidate call-result use
  and reports unsupported whole-tuple or InCore-caller forms before Step A.
- `TupleIndexPermutationMutator` — Step B rewriter: permutes `Call` / `Submit`
  result tuple types, remaps their binding Vars by identity, and rewrites
  `TupleGetItemExpr` indices while preserving the `Submit` TASK_ID tail.

**Properties**: `include/pypto/ir/transforms/pass_properties.h`

```cpp
inline const PassProperties kNormalizeReturnOrderProperties{
    .required = {IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps},
    .produced = {IRProperty::ReturnParamsExplicit}};
```

**Python binding**: `python/bindings/modules/passes.cpp`

```cpp
passes.def("normalize_return_order", &pass::NormalizeReturnOrder,
           "Create a return order normalization pass\n\n"
           "Canonicalizes tensor param-writeback returns and reorders InCore return tuples\n"
           "to Out/InOut parameter order. Reordered Call/Submit results must be directly\n"
           "bound and used only through TupleGetItem projections in non-InCore callers.");
```

**Type stub**: `python/pypto/pypto_core/passes.pyi`

```python
def normalize_return_order() -> Pass:
    """Create a return-order normalization pass.

    Reordered Call/Submit results must be directly bound and used only through
    tuple-element projections in non-InCore callers.
    """
```

**Tests**: `tests/ut/ir/transforms/test_normalize_return_order.py`

## Related

- [`OutlineInCoreScopes`](08-outline_incore_scopes.md) — upstream
  producer of the `InCore` functions this pass rewrites
- [`SkewCrossCorePipeline`](26-skew_cross_core_pipeline.md) and
  [`LowerPipelineToSlots`](27-lower_pipeline_to_slots.md) — run in between,
  claiming the pipeline loops each one handles
- [`LowerPipelineLoops`](28-lower_pipeline_loops.md) — consumes the normalized
  returns when expanding the pipeline scopes those two passes left behind
- [`DeriveCallDirections`](37-derive_call_directions.md) — later
  inspects call signatures whose return shape this pass canonicalizes
- [PTO codegen overview](../codegen/00-pto_codegen.md) and
  [orchestration codegen](../codegen/01-orchestration_codegen.md) —
  consumers of the explicit return-position-to-parameter mapping
