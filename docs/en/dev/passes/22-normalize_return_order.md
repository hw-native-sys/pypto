# NormalizeReturnOrder Pass

Reorders the return tuple of every InCore function so that `return[i]`
corresponds to the i-th `Out`/`InOut` parameter in declaration order, and
remaps `TupleGetItemExpr` indices in non-InCore callers accordingly. After
this pass, orchestration codegen can map tuple element indices to output
parameters with a direct `out_indices[i]` lookup, without tracing through
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

1. **Step A (InCore rewrite)** — for every `InCore` function, compute a
   permutation that sorts `ReturnStmt::value_` to match the declared
   `Out`/`InOut` parameter order, then rewrite both the return values and
   `Function::return_types_` accordingly.
2. **Step B (call-site remap)** — for every non-InCore function
   (Orchestration / Group / Spmd / opaque), rewrite every
   `TupleGetItemExpr.index_` whose tuple operand is the result of a call
   to a function reordered in Step A. The new index is
   `permutation[old_index]`, so observers of the call result still see the
   same SSA values bound to the same names.

The pass is a **no-op** for any function whose return order already matches
its `Out`/`InOut` parameter order, and a no-op for any program with no
InCore functions.

**Pipeline position**: slot #20 in the `Default` strategy — after
`SplitVectorKernel` (#19) and before `LowerPipelineLoops` (#21). It runs
late enough that all kernel splitting / tile-structural decisions are made
on the original return order, and early enough that downstream tile-level
passes (`LowerPipelineLoops`, `CanonicalizeIOOrder`, `InitMemRef`,
`MemoryReuse`, `AllocateMemoryAddr`) — and ultimately PTO orchestration
codegen — see the canonical order.

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
| Produced | — |
| Invalidated | — |

`SplitIncoreOrch` guarantees that InCore work has been outlined into its
own functions; `IncoreTileOps` guarantees the body uses tile ops, so the
`tile.store(_, _, out_param)` signal that drives Step A is present. The
pass produces no new property and invalidates none — it preserves SSA
form, normalized statement structure, memory inference, and every other
property already established upstream.

## Algorithm

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
returns the empty permutation in three cases (any of which makes the
pass a no-op for that function):

- The body has no `ReturnStmt` (open IR) or no Out/InOut parameters.
- `out_indices.size() > ret_to_param.size()` — more declared output
  parameters than returned values, so the analysis is incomplete and we
  refuse to construct an out-of-bounds permutation.
- The computed permutation is the identity (already canonical).

When the permutation is non-empty, `ReorderReturns` builds a fresh
`Function` via `MutableCopy`, replacing the trailing `ReturnStmt` with one
whose `value_[permutation[i]] = old_value_[i]` and permuting
`Function::return_types_` in lockstep so the type list stays aligned with
the values.

### Step B — Remap `TupleGetItemExpr` at call sites

For each non-InCore function in the (already-Step-A-rewritten) program,
`TupleIndexPermutationMutator` does a single SSA pass that:

- Tracks every `AssignStmt` whose RHS is a `Call(GlobalVar)` to a
  function reordered in Step A, recording `assign.var → permutation_ref`
  in a `reordered_tuple_vars_` map.
- Removes a tracked entry whenever its `Var` is reassigned (call to a
  non-reordered function, non-call RHS, etc.) so identity-based lookups
  never read a stale binding.
- For every `TupleGetItemExpr(tuple_var, k)` whose `tuple_var` is in the
  tracked map, rewrites the index to `permutation[k]`.

Because Step A rewrites function signatures and Step B rewrites call-site
index access in the same pass invocation, the program is consistent at
exit: each tuple element is still bound to the same physical output
buffer, just under a new index.

## Constraints

| Constraint | Reason |
| ---------- | ------ |
| Only `InCore` functions are rewritten in Step A | Other function kinds (`Orchestration` / `Group` / `Spmd` / opaque) follow the user's declared return shape; their callers are remapped in Step B but their bodies are untouched |
| Skips functions where `out_indices.size() > ret_to_param.size()` | An incomplete analysis must not produce an out-of-bounds permutation — leave the function as-is so the verifier can flag the inconsistency |
| Permutation is identity ⇒ no rewrite | Avoids spurious `Function` clones and keeps the pass idempotent |
| Step B only rewrites `TupleGetItemExpr` whose tuple operand resolves to a tracked `Var` after `VisitExpr` | The mutator preserves `Var` node identity, so the operand pointer stays valid as a key in `reordered_tuple_vars_`; if a future change ever returned a fresh node, looking up the post-visit pointer keeps the check correct |

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

The same SSA assignment (`a = ...`) is still bound to the value produced
by `pl.store(a_tile, ..., out_a)`; only the path through the tuple has
changed. `InOut` parameters behave identically.

See `tests/ut/ir/transforms/test_normalize_return_order.py` for the
full set of cases:

- `test_swapped_returns_reordered` — the two-Out-param example above
- `test_already_ordered_noop` — pass leaves canonical IR untouched
- `test_single_return_noop` — single Out param needs no permutation
- `test_non_incore_unchanged` — programs with no InCore functions are no-ops
- `test_three_returns_scrambled` — three-way permutation
- `test_2d_tensor_reorder` — 2-D tensors / multi-dim offsets
- `test_inout_param_reorder` — `InOut` participates in reordering

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass NormalizeReturnOrder();
```

**Implementation**: `src/ir/transforms/normalize_return_order_pass.cpp`

- `BuildReturnToParamMapping` — Step A analysis: walks the function body
  to map each `ReturnStmt` value back to an Out/InOut parameter index.
- `CollectOutIndices` — collects the parameter positions whose
  `ParamDirection` is `Out` or `InOut`.
- `ComputeReturnPermutation` — composes the previous two into the final
  `permutation[old_index] = new_index`; returns empty when no rewrite
  is needed or the analysis is incomplete.
- `ReorderReturns` — builds a `MutableCopy(func)` with the permuted
  `ReturnStmt::value_` and `Function::return_types_`.
- `TupleIndexPermutationMutator` — Step B rewriter: tracks call-result
  vars and rewrites `TupleGetItemExpr` indices.

**Properties**: `include/pypto/ir/transforms/pass_properties.h`

```cpp
inline const PassProperties kNormalizeReturnOrderProperties{
    .required = {IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps}};
```

**Python binding**: `python/bindings/modules/passes.cpp`

```cpp
passes.def("normalize_return_order", &pass::NormalizeReturnOrder,
           "Create a return order normalization pass\n\n"
           "Reorders return tuple values in InCore functions so that return[i]\n"
           "corresponds to the i-th Out/InOut parameter in declaration order,\n"
           "and updates TupleGetItemExpr indices at call sites accordingly.");
```

**Type stub**: `python/pypto/pypto_core/passes.pyi`

```python
def normalize_return_order() -> Pass:
    """Create a return order normalization pass."""
```

**Tests**: `tests/ut/ir/transforms/test_normalize_return_order.py`

## Related

- [`OutlineInCoreScopes`](10-outline_incore_scopes.md) — upstream
  producer of the `InCore` functions this pass rewrites
- [`LowerPipelineLoops`](23-lower_pipeline_loops.md) — runs immediately
  after; consumes the normalized returns when expanding pipeline scopes
- [`DeriveCallDirections`](32-derive_call_directions.md) — later
  inspects call signatures whose return shape this pass canonicalizes
- [PTO codegen overview](../codegen/00-pto_codegen.md) and
  [orchestration codegen](../codegen/01-orchestration_codegen.md) —
  consumers of the canonical `return[i] ↔ out_indices[i]` mapping
