# Simplify Pass

Folds arithmetic expressions, type-embedded shape expressions, and scalar constant bindings using algebraic rewrite rules and bound analysis.

## Overview

`Simplify` is a function-level pass that rewrites the IR in place using `arith::Analyzer`. It performs three kinds of work:

1. **Arithmetic folding** at every expression leaf (e.g. `x + 0 → x`, `x * 1 → x`, `min(a, a) → a`, comparisons that the analyzer can decide).
2. **Type rebuild** — re-walks shape expressions embedded in `TensorType`, `TileType`, and `TupleType` so the in-memory IR matches what a fresh parse would produce.
3. **Scalar binding for folding + DCE** — a scalar `Var` assigned once is registered with the analyzer. A constant assigned at function-body top level is bound fully so its literal propagates into every downstream use; a symbolic value, or a constant inside a loop/branch, contributes only a `ConstIntBound` — enough to fold dead branch guards like `if expr == 0` without inlining the scalar. Bindings left dead are dropped by a conservative scalar DCE.

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

1. **Multi-assign collection** — `MultiAssignCollector` walks the function body and records every scalar `Var` assigned more than once. These are excluded from analyzer binding so a stale value cannot be used past a later reassignment. A `Var` assigned exactly once — even inside a loop body or branch — is safe to bind: `SimplifyMutator` scopes every binding to the region the assignment lives in (see phase 2), unbinding it on region exit. Under SSA every `Var` is single-assigned, so nothing is collected.
2. **`SimplifyMutator` traversal** — extends `arith::IRMutatorWithAnalyzer`. The analyzer carries a constraint stack (loop-var bounds, if-branch conditions, scalar bindings). Folding happens at the leaves rather than only at top-level expressions because the analyzer's top-level `Simplify` does not recurse into non-arithmetic containers (`Call`, `MakeTuple`):
   - `VarPtr`: substitute via the var-remap table, then run through the analyzer.
   - `BinaryExpr` / `UnaryExpr`: visit children, then fold the rebuilt node.
   - `CallPtr`: refresh the result `type_` so a Call whose shape arguments folded ends up structurally equal to a freshly parsed Call.
   - `AssignStmt`: for a scalar LHS `Var` not in `multi_assigned_`, register the simplified RHS with the analyzer. A `ConstInt`/`ConstFloat`/`ConstBool` RHS at function-body top level is bound fully (the literal is substituted into later uses); a symbolic RHS — or a constant inside a loop/branch — contributes only a `ConstIntBound`, so dead branch guards fold without the scalar being inlined. Every binding is logged so the enclosing region's visitor can unbind it on exit.
   - `ForStmt`: rebuild `iter_args_` before visiting the body so body references pick up the remapped identity; if both `start_` and `stop_` fold to `ConstInt` with `stop > start`, bind the loop var to that range while visiting the body and unbind on exit; scalars bound inside the body are unbound after the visit; rebuild `return_vars_` after the body so folds discovered inside are visible in return types. Pure single-trip and zero-trip loops are also collapsed in-place — see "Control-flow folding" below.
   - `IfStmt`: enter `Analyzer::GetConstraintContext(cond)` for the then branch and `Not(cond)` for the else branch; scalars bound inside each branch are unbound after that branch so they do not leak into the other branch or past the `IfStmt`. Conditions the analyzer can prove are also folded — see "Control-flow folding" below.
   - `WhileStmt`: same as `ForStmt` minus the bounds — rebuild `iter_args_` before the condition and body, snapshot `var_remap_` around the body visit, then rebuild `return_vars_` after it, with the same scoped scalar unbinding. Rebuilding `iter_args_` first is required, not cosmetic: an `IterArg` *use* is the same node as its declaration and carries `initValue_`, so the base `IRMutator` mints a fresh `IterArg` at the first use whose init the analyzer rewrote. Seeding `var_remap_` from the header makes every reference resolve to one node; skipping it leaves the header on the stale `IterArg` while all body uses point at an undefined clone (a `UseAfterDef` failure).
   - `SpmdScopeStmt`: visit the body with the same scoped scalar unbinding, and additionally fold `core_num_` (closure arithmetic such as `MAX // TILE` may need one pass of simplification after SSA conversion).
3. **Type rebuild** — `SimplifyType` recurses through `TensorType`, `TileType`, and `TupleType`, calling `SimplifyExpr` on every embedded expression (shape, stride, valid_shape, start_offset, view fields). Identity is preserved when nothing changes so the round-trip identity check stays cheap.
4. **Scalar DCE + dead yield-slot prune** — after the mutator finishes, `dce::EliminateDeadScalarAssignments` walks the flattened body and drops scalar `AssignStmt`s whose only uses were folded away. The DCE is conservative: it never removes call-backed assignments because the IR has no purity annotations yet and a `Call` may have observable side effects. Between the two scalar-DCE runs, `dce::EliminateDeadYieldSlots` drops the yielded slots nobody reads — an `IfStmt` phi whose `return_vars_[i]` has no user, and a `ForStmt` / `WhileStmt` carry whose `iter_args_[i]` (read inside the body) *and* `return_vars_[i]` (read after the loop) are both unused — together with the matching `YieldStmt` slot. Reusing one Python local across two scopes produces exactly that dead carry: SSA seeds the second loop with the first scope's value, every trip overwrites it, and nothing reads either end. Left in place it makes the earlier scope's value live-out, which for a device scope would force a `Scalar` into the outlined kernel's return set — see [08-outline_incore_scopes.md](08-outline_incore_scopes.md).
5. **Loop-state repair** — if DCE removed any statements, `loop_repair::MakeBody` reassembles the function body so loop-carried metadata (yield/return mappings) stays consistent.

### Control-flow folding

Two folds run inside the `SimplifyMutator` traversal so they share the analyzer's constraint stack with the surrounding expression-level work:

- **Fold A — constant-condition `IfStmt` collapse.** After the condition is simplified, query the analyzer with `CanProve(cond)` and `CanProve(Not(cond))`. On a proof of either polarity, drop the dead branch and lift the kept branch into the parent scope. When `return_vars_` is non-empty, the kept branch's trailing `YieldStmt` is stripped and each `return_vars[i]` is bound in `var_remap_` to the corresponding yielded value so subsequent siblings (and the function `ReturnStmt`) read the value directly. Symmetric for true / false; the only edge case is "always-false with no else and empty return_vars," which collapses to an empty body.
- **Fold B — pure single/zero-trip `ForStmt` collapse.** Fires only on *pure* sequential loops: `attrs_` empty, `kind_ == ForKind::Sequential`. For these, query the analyzer for the trip count using `CanProveGreaterEqual(step, 1)` plus `CanProve(stop <= start)` (zero trips) or `CanProve(start < stop && stop <= start + step)` (one trip). On zero trips, emit one `AssignStmt(return_vars[i], iter_args[i].initValue_)` per return var and drop the body. On one trip, `DeepClone` the body with `loop_var → start` and `iter_args[i] → init_values[i]` substitutions, re-visit the cloned body so further folds happen in the same pass, then strip the trailing `YieldStmt` and bind each `return_vars[i] → yielded_value[i]` in `var_remap_` (same propagation mechanism as Fold A's lift).

`DeepClone` with `clone_def_vars=true` is used (rather than an in-place `var_remap_` override on the body) so the unrolled body gets fresh `Var` identities at every DefField, matching `LoopUnrollMutator`. This keeps the lifted copy structurally independent of the original (discarded) loop body and lets the re-visit bind the body's scalars on identities distinct from the surrounding scope.

The choice to substitute `return_vars` via `var_remap_` rather than emit a literal `AssignStmt(rv, yielded)` is deliberate: the orchestration codegen's role-aware name disambiguation (`role == "out"` etc.) collapses several role-tagged SSA versions to the same C++ identifier, so an `out__rv_v2 = out__co_l0_rv_v3` alias would lower to the ill-formed `auto out = out;`. Substituting at use sites side-steps the disambiguation entirely.

#### Escaping return vars

A substitution only reaches uses visited while its `var_remap_` entry is live, and `ForStmt`, `WhileStmt` and `IfStmt` each rebase `var_remap_` to a pre-body baseline on the way out so a body-internal remap cannot rewrite siblings or post-loop code. A use that outlives that restore would keep the original `Var` — which the fold has just stripped the only definition of, a dangling reference `UseAfterDefCheck` reports.

`ReturnVarEscapeIndex` (a pre-pass in `simplify_pass.cpp`) decides this per fold site. It walks the function body once, numbering the restoring scopes in pre-order so a scope owns the contiguous id range `[id, end)` of its subtree; "every use of `v` sits inside scope `S`" is then two integer comparisons. A monotonic tick orders uses against the fold site, so a use *preceding* it inside the same scope counts as escaping too. One walk plus O(1) lookups per fold keeps Simplify within its O(N log N) budget.

Statements the index has never seen answer "does not escape", keeping the substitution. That covers folds nested inside a body Fold B `DeepClone`d, whose `Var` identities are minted after indexing. A clone's Vars are unreachable from outside it, so the only unhandled case is a restore-scope *within* a clone standing between such a fold and a later use of its return var — pre-SSA only, and no worse than the behaviour before this index existed. Re-indexing each clone would close it, but nested single-trip loops would then pay an O(N²) walk.

For an escaping `return_vars[i]`, `LiftBodyToReturnVars` emits `AssignStmt(return_vars[i], yielded_value[i])` at the fold site instead of recording the remap. The assignment stays *inside* the region being lifted — the yielded value may name body-local `Var`s, so it cannot be hoisted past the loop, and in leak-mode semantics the last iteration writing last is exactly what a post-loop read expects.

Nothing escapes in SSA form: a value defined inside a region is never referenced outside it, and every use is dominated by its definition. Since the pipeline runs Simplify only after `ConvertToSSA` (positions 5 and 46), the materializing path never fires there — it exists for callers that run Simplify directly on pre-SSA IR, where the alias-assignment concern above does not apply because SSA conversion still runs afterwards.

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

### Dead branch guard through a scalar bound

**Before**:

```python
for ob in pl.range(0, 68, 2):
    off: pl.Scalar[pl.INDEX] = ob * 256 + 256
    if off == 0:
        first_chunk(off)
    else:
        later_chunk(off)
```

**After**:

```python
for ob in pl.range(0, 68, 2):
    off: pl.Scalar[pl.INDEX] = ob * 256 + 256
    later_chunk(off)
```

The analyzer binds `ob ∈ [0, 68)` while visiting the loop body, so `off`'s `AssignStmt` registers a `ConstIntBound` of `[256, 17408]` for `off`. `CanProve(Not(off == 0))` then succeeds and Fold A drops the dead then branch. `off` is bound for analysis only — it is not substituted — so the surviving `later_chunk(off)` still references the scalar. (If `off` becomes unused after the fold, scalar DCE removes its binding.)

### Where an index bound comes from

`INDEX` is the dtype of every index computation, and it is **signed** — codegen
emits `arith.cmpi slt` and `arith.maxsi` for it. So the dtype alone proves
nothing about a variable's sign, and the analyzer treats an unbound `INDEX` Var
as `[-inf, +inf]`. Non-negativity has to be established, never assumed:

| Source | Bound | Established by |
| ------ | ----- | -------------- |
| Assigned scalar | its RHS's range | `BindScalarBound`, from the value being produced |
| Loop variable | `[start, stop)`, or `[start, +inf)` when `stop` is symbolic | `IRMutatorWithAnalyzer` on `ForStmt`, for a positive step |
| Branch condition | the constraint's range | `EnterConstraint` for the arm's scope |
| Block / subblock builtin | `[0, +inf)`; a block *count* is `[1, +inf)` | the op's own semantics, in `ConstIntBoundAnalyzer` |
| Whole shape / valid-shape dimension | `[0, +inf)` | `DimensionSymbolScope`, around the write-union proofs |
| Runtime scalar parameter | `[-inf, +inf]` | nothing — the caller chooses the value |

The last three rows are the distinction that matters, and none of them is a fact
about `INDEX`. `tile.get_subblock_idx()` is non-negative because of what the op
*returns*. A dimension is non-negative because it is a count of elements.

**That second rule stops at the dimension.** `DimensionSymbolScope` binds a
dimension that *is* a bare symbol, and never descends into one that is a
compound expression. Being a `valid_shape` makes the field an extent; it does not
make every variable that computes it one:

```python
valid = pl.max(-x, 0)    # a legal dynamic extent over a signed runtime scalar
```

Assuming `x >= 0` folds that to a constant `0`, which reads as an empty region
and silently shrinks the result. Offsets are never bound at all — `max(x, 0)` in
an offset is a deliberate clamp whose whole point is that `x` can be negative,
and folding it to `x` would move the region a store writes.

Assuming it everywhere silently changes what a kernel computes:

```python
pos: pl.Scalar[pl.INDEX] = base - 1   # [-1, +inf)
if pos >= 0:                          # live guard, kept
    if pos < 8:
        read_row(pos)
```

Under a blanket `[0, +inf)` default, `pos >= 0` proves statically true and Fold
A drops the outer guard, leaving the upper-bound check standing alone — which a
negative `pos` passes, reaching `read_row` with an index that is then clamped to
row 0 (issue #2500). The same held for a bare `Scalar[pl.INDEX]` parameter: the
caller can pass `-1`.

Everything a pass still needs to know about a symbol's sign now comes from
somewhere that can prove it — the value assigned, the loop, the branch, the op,
or the symbol's use as a whole dimension.

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

Loops with `attrs_` or non-Sequential `kind_` are skipped — those forms participate in execution-model contracts (Parallel/Unroll/Pipeline scheduling) that downstream passes may depend on observing as a `ForStmt`.

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

- `MultiAssignCollector` — IRVisitor that flags scalar `Var`s assigned more than once (unsafe to bind).
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
- Dead branch guards folded via loop-affine scalar `ConstIntBound`s.
- Conservative scalar DCE — dropped only when every use is foldable.
