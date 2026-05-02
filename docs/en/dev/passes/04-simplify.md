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
   - `ForStmt`: rebuild `iter_args_` before visiting the body so body references pick up the remapped identity; if both `start_` and `stop_` fold to `ConstInt` with `stop > start`, bind the loop var to that range while visiting the body and unbind on exit; rebuild `return_vars_` after the body so folds discovered inside are visible in return types. Pure single-trip and zero-trip loops are also collapsed in-place — see "Control-flow folding" below.
   - `IfStmt`: enter `Analyzer::GetConstraintContext(cond)` for the then branch and `Not(cond)` for the else branch. Conditions the analyzer can prove are also folded — see "Control-flow folding" below.
   - `SpmdScopeStmt`: fold `core_num_` (closure arithmetic such as `MAX // TILE` may need one pass of simplification after SSA conversion).
3. **Type rebuild** — `SimplifyType` recurses through `TensorType`, `TileType`, and `TupleType`, calling `SimplifyExpr` on every embedded expression (shape, stride, valid_shape, start_offset, view fields). Identity is preserved when nothing changes so the round-trip identity check stays cheap.
4. **Scalar DCE** — after the mutator finishes, `dce::EliminateDeadScalarAssignments` walks the flattened body and drops scalar `AssignStmt`s whose only uses were folded away. The DCE is conservative: it never removes call-backed assignments because the IR has no purity annotations yet and a `Call` may have observable side effects.
5. **Loop-state repair** — if DCE removed any statements, `loop_repair::MakeBody` reassembles the function body so loop-carried metadata (yield/return mappings) stays consistent.

### Control-flow folding

Two folds run inside the `SimplifyMutator` traversal so they share the analyzer's constraint stack with the surrounding expression-level work:

- **Fold A — constant-condition `IfStmt` collapse.** After the condition is simplified, query the analyzer with `CanProve(cond)` and `CanProve(Not(cond))`. On a proof of either polarity, drop the dead branch and lift the kept branch into the parent scope. When `return_vars_` is non-empty, the kept branch's trailing `YieldStmt` is stripped and each `return_vars[i]` is bound in `var_remap_` to the corresponding yielded value so subsequent siblings (and the function `ReturnStmt`) read the value directly. Symmetric for true / false; the only edge case is "always-false with no else and empty return_vars," which collapses to an empty body.
- **Fold B — pure single/zero-trip `ForStmt` collapse.** Fires only on *pure* sequential loops: `chunk_config_` empty, `attrs_` empty, `kind_ == ForKind::Sequential`. For these, query the analyzer for the trip count using `CanProveGreaterEqual(step, 1)` plus `CanProve(stop <= start)` (zero trips) or `CanProve(start < stop && stop <= start + step)` (one trip). On zero trips, emit one `AssignStmt(return_vars[i], iter_args[i].initValue_)` per return var and drop the body. On one trip, `DeepClone` the body with `loop_var → start` and `iter_args[i] → init_values[i]` substitutions, re-visit the cloned body so further folds happen in the same pass, then strip the trailing `YieldStmt` and bind each `return_vars[i] → yielded_value[i]` in `var_remap_` (same propagation mechanism as Fold A's lift).

`DeepClone` is used (rather than an in-place `var_remap_` override on the body) because `clone_def_vars=true` is load-bearing here: `MultiAssignCollector` ran on the pre-fold IR and may have flagged scalar `Var`s defined inside the original loop body as multi-assigned (the safety check that protects pre-SSA callers). Producing fresh `Var` clones at every DefField breaks that staleness so subsequent uses of (e.g.) `cur_offset = 0` can be bound and propagated downstream.

The choice to substitute `return_vars` via `var_remap_` rather than emit a literal `AssignStmt(rv, yielded)` is deliberate: the orchestration codegen's role-aware name disambiguation (`role == "out"` etc.) collapses several role-tagged SSA versions to the same C++ identifier, so an `out__rv_v2 = out__co_l0_rv_v3` alias would lower to the ill-formed `auto out = out;`. Substituting at use sites side-steps the disambiguation entirely.

The two folds compose in a single pass: when Fold B substitutes `loop_var → 0` in a body, predicates like `if loop_var == 0` reduce to `if 0 == 0` → `ConstBool(true)`, which Fold A then collapses without a second Simplify run.

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

### Constant-condition branch (Fold A)

**Before**:

```python
for i in pl.range(0, 8, 2):
    if i == -1:
        body_dead(i)
    else:
        body_live(i)
```

**After**:

```python
for i in pl.range(0, 8, 2):
    body_live(i)
```

The analyzer binds `i ∈ [0, 8)` while visiting the loop body. `CanProve(Not(i == -1))` succeeds — the comparison is statically false — so Fold A drops the then branch and lifts the else branch into the surrounding for-body. The same path runs for always-true conditions (drops else, lifts then). When the IfStmt has `return_vars_`, the kept branch's trailing `YieldStmt` is rewritten into `AssignStmt`s on the return vars.

### Single-trip loop collapse (Fold B)

**Before**:

```python
for ko in pl.range(0, 128, 128):
    if ko == 0:
        first_iter(ko)
    else:
        later_iter(ko)
```

**After**:

```python
first_iter(0)
```

The trip count proof `start < stop && stop <= start + step` succeeds for `pl.range(0, 128, 128)`, so Fold B substitutes `ko → 0` (via `DeepClone`) and lifts the body. The substitution turns the inner `if ko == 0` into `if 0 == 0`, which `analyzer_->Simplify` reduces to `ConstBool(true)`. Fold A then drops the dead else branch — both folds compose in the same Simplify pass. The same path handles zero-trip loops by emitting `AssignStmt`s for each `return_vars[i] = iter_args[i].initValue_` and dropping the body entirely.

Loops with `chunk_config_`, `attrs_`, or non-Sequential `kind_` are skipped — those forms participate in execution-model contracts (chunked-loop pairing, Parallel/Unroll/Pipeline scheduling) that downstream passes may depend on observing as a `ForStmt`.

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
