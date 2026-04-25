# Simplify Pass

Folds arithmetic expressions, type-embedded shape expressions, and scalar constant bindings using algebraic rewrite rules and bound analysis.

## Overview

`Simplify` is a function-level pass that rewrites the IR in place using `arith::Analyzer`. It performs three kinds of work:

1. **Arithmetic folding** at every expression leaf (e.g. `x + 0 → x`, `x * 1 → x`, `min(a, a) → a`, comparisons that the analyzer can decide).
2. **Type rebuild** — re-walks shape expressions embedded in `TensorType`, `TileType`, and `TupleType` so the in-memory IR matches what a fresh parse would produce.
3. **Scalar constant propagation + DCE** — when a scalar `Var` is assigned a constant once, that value is bound in the analyzer and propagated into every downstream use; the now-dead binding is then dropped by a conservative scalar DCE.

The pass runs **twice** in the `Default` strategy of `pass_manager.py`:

- **Post-SSA** (after `ConvertToSSA`, before `FlattenCallExpr`): propagates closure-captured constants such as `CHUNK_K: Scalar[INDEX] = 512` into shape expressions and types so subsequent tile-lowering passes see literals instead of variables.
- **End of tile pipeline** (after `DeriveCallDirections`): final cleanup of folds exposed by memory-space inference, layout resolution, and other late lowering.

**Requires**: nothing.

**Produces**: nothing.

**Invalidates**: nothing.

The empty `PassProperties` contract (`kSimplifyProperties` in `include/pypto/ir/transforms/pass_properties.h`) is intentional: Simplify is conservative enough to preserve every property its callers may have established (`SSAForm`, `NormalizedStmtStructure`, `IncoreTileOps`, ...) — it only rewrites expressions and prunes scalar bindings, never restructures statements.

## When to Use

- After SSA conversion to propagate scalar constants into types/shapes before the tile pipeline inspects them.
- At the end of the tile pipeline as a cleanup pass so that downstream artifacts (printed IR, codegen) are not littered with `K + 0` or `idx * 1` residue.
- Anywhere else a pass produces fresh expressions that may be foldable; Simplify is cheap and idempotent so it is safe to insert defensively.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::Simplify()` | `passes.simplify()` | Function-level |

**Factory function**:

```cpp
Pass Simplify();
```

**Python usage**:

```python
from pypto.pypto_core import passes

simplify_pass = passes.simplify()
program_simplified = simplify_pass(program)
```

## Algorithm

Implemented by `TransformSimplify` in `src/ir/transforms/simplify_pass.cpp` in five phases:

1. **Multi-assign collection** — `MultiAssignCollector` walks the function body and records every scalar `Var` that is either reassigned or assigned inside a nested `IfStmt`/`ForStmt`/`WhileStmt` body. These are excluded from constant binding so a stale initial value cannot propagate past a later reassignment. Under SSA the check is redundant but kept so the pass remains safe on pre-SSA callers.
2. **`SimplifyMutator` traversal** — extends `arith::IRMutatorWithAnalyzer`. The analyzer carries a constraint stack (loop-var bounds, if-branch conditions, scalar bindings). Folding happens at the leaves rather than only at top-level expressions because the analyzer's top-level `Simplify` does not recurse into non-arithmetic containers (`Call`, `MakeTuple`):
   - `VarPtr`: substitute via the var-remap table, then run through the analyzer.
   - `BinaryExpr` / `UnaryExpr`: visit children, then fold the rebuilt node.
   - `CallPtr`: refresh the result `type_` so a Call whose shape arguments folded ends up structurally equal to a freshly parsed Call.
   - `AssignStmt`: bind the LHS `Var` to the simplified RHS when the type is `ScalarType` and the RHS is a `ConstInt`/`ConstFloat`/`ConstBool`, unless the LHS is in `multi_assigned_`.
   - `ForStmt`: rebuild `iter_args_` before visiting the body so body references pick up the remapped identity; if both `start_` and `stop_` fold to `ConstInt` with `stop > start`, bind the loop var to that range while visiting the body and unbind on exit; rebuild `return_vars_` after the body so folds discovered inside are visible in return types.
   - `IfStmt`: enter `Analyzer::GetConstraintContext(cond)` for the then branch and `Not(cond)` for the else branch.
   - `SpmdScopeStmt`: fold `core_num_` (closure arithmetic such as `MAX // TILE` may need one pass of simplification after SSA conversion).
3. **Type rebuild** — `SimplifyType` recurses through `TensorType`, `TileType`, and `TupleType`, calling `SimplifyExpr` on every embedded expression (shape, stride, valid_shape, start_offset, view fields). Identity is preserved when nothing changes so the round-trip identity check stays cheap.
4. **Scalar DCE** — after the mutator finishes, `dce::EliminateDeadScalarAssignments` walks the flattened body and drops scalar `AssignStmt`s whose only uses were folded away. The DCE is conservative: it never removes call-backed assignments because the IR has no purity annotations yet and a `Call` may have observable side effects.
5. **Loop-state repair** — if DCE removed any statements, `loop_repair::MakeBody` reassembles the function body so loop-carried metadata (yield/return mappings) stays consistent.

## Examples

### Algebraic identity

**Before**:

```python
def main(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    a = x + 0
    b = a * 1
    return b
```

**After**:

```python
def main(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return x
```

`x + 0 → x` and `x * 1 → x` apply at every arithmetic leaf. The two scalar bindings are then dropped by the DCE phase and the body collapses to the return.

### Loop-bound aware folding

**Before**:

```python
for i in pl.range(0, 8):
    if i < 16:
        body(i)
```

**After**:

```python
for i in pl.range(0, 8):
    body(i)
```

While visiting the loop body the analyzer is told that `i ∈ [0, 8)`. The condition `i < 16` therefore folds to `True`, the `IfStmt` collapses to its then branch, and the surrounding `for` is preserved unchanged.

### Scalar constant propagation + DCE

**Before** (post-`ConvertToSSA`, closure value `CHUNK_K = 512`):

```python
CHUNK_K__ssa_v0: pl.Scalar[pl.INDEX] = 512
acc: pl.Tile[[CHUNK_K__ssa_v0, 64], pl.FP32] = tile.zeros(...)
for k in pl.range(0, K, CHUNK_K__ssa_v0):
    body(k)
return acc
```

**After**:

```python
acc: pl.Tile[[512, 64], pl.FP32] = tile.zeros(...)
for k in pl.range(0, K, 512):
    body(k)
return acc
```

`CHUNK_K__ssa_v0` is bound to `512` at its `AssignStmt`. Every downstream reference — including the embedded shape inside the `TileType` of `acc` — folds to the literal during the type-rebuild phase. The now-dead binding is dropped by the DCE phase. This is the primary motivation for the post-SSA scheduling point: tile-lowering passes such as `FlattenTileNdTo2D` and `InferTileMemorySpace` see concrete shape literals instead of opaque scalar `Var`s.

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass Simplify();
```

**Properties**: `include/pypto/ir/transforms/pass_properties.h`

```cpp
inline const PassProperties kSimplifyProperties{};
```

**Implementation**: `src/ir/transforms/simplify_pass.cpp`

- `MultiAssignCollector` — IRVisitor that flags scalar `Var`s unsafe to bind as constants.
- `SimplifyMutator` — extends `arith::IRMutatorWithAnalyzer`; folds expressions at leaves and rebuilds `Var` / `IterArg` types when their embedded shape exprs simplify.
- `TransformSimplify` — orchestrates the five phases (collect → mutate → type-rebuild → DCE → repair) and returns a new `Function` only when the body actually changed.

**Underlying analyzer**: `src/ir/arith/analyzer.cpp`, `src/ir/arith/ir_mutator_with_analyzer.cpp`. The analyzer composes a rewrite simplifier, a constant-interval bound analyzer, a transitive comparison analyzer, and a constraint stack.

**Python binding**: `python/bindings/modules/passes.cpp`

```cpp
passes.def(
    "simplify", &pass::Simplify,
    "Create a pass that simplifies expressions and statements using algebraic rules and bound analysis");
```

**Type stub**: `python/pypto/pypto_core/passes.pyi`

```python
def simplify() -> Pass:
    """Create a pass that simplifies expressions and statements using algebraic rules and bound analysis."""
```

**Tests**: `tests/ut/ir/transforms/test_simplify_pass.py`

- Pass metadata (name `"Simplify"`, empty required/produced properties).
- Identity simplifications (`x + 0`, `x * 1`, `min(a, a)`, ...).
- Constant folding through `Call` arguments and embedded shape expressions.
- Loop-bound aware folding via `ForStmt` analyzer binding.
- If-branch constraint propagation via `Analyzer::GetConstraintContext`.
- Scalar constant propagation through SSA-form bindings.
- Conservative scalar DCE — dropped only when every use is foldable.
