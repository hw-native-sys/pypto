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

Integer and INDEX rewrites preserve the expression's declared result dtype. If simplification selects an operand of a different integer dtype, it retains the conversion with a cast (or folds it into a typed constant). For example, an INT32 `(index + 1) - 1` becomes `cast(index, INT32)`.

## When to Use

- After SSA conversion to propagate scalar constants into types/shapes before the tile pipeline inspects them.
- At the end of the tile pipeline as a cleanup pass so that downstream artifacts (printed IR, codegen) are not littered with `K + 0` or `idx * 1` residue.
- Anywhere else a pass produces fresh expressions that may be foldable; Simplify is cheap and idempotent so it is safe to insert defensively. The one bounded exception is a chain of more than 16 *nested* single-trip loops, where a second run folds further — see [Substitution-depth cap](#substitution-depth-cap-fold-b). No pipeline input reaches that depth.

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

1. **Multi-assign collection** — `MultiAssignCollector` walks the function body and records every scalar `Var` assigned more than once. These are excluded from analyzer binding so a stale value cannot be used past a later reassignment. Each of their assignments also resets any constant bound the `Var` already carries to unknown: a pre-SSA loop variable reassigned in its body would otherwise keep the header's `[start, stop)`, and `i = i + 100; i = i + 100; i < stop` would fold to `True`. A `Var` assigned exactly once — even inside a loop body or branch — is safe to bind: `SimplifyMutator` scopes every binding to the region the assignment lives in (see phase 2), unbinding it on region exit. Under SSA every `Var` is single-assigned, so nothing is collected.
2. **`SimplifyMutator` traversal** — extends `arith::IRMutatorWithAnalyzer`. The analyzer carries a constraint stack (loop-var bounds, if-branch conditions, scalar bindings). Folding happens at the leaves rather than only at top-level expressions because the analyzer's top-level `Simplify` does not recurse into non-arithmetic containers (`Call`, `MakeTuple`):
   - `VarPtr` / `IterArgPtr`: substitute via the var-remap table, then run through the analyzer. Loop-carried uses are leaves: only the loop header rebuilds their type and initial value. Revisiting a carry's MemRef offset at a use site can otherwise create a fresh, undefined variable, especially across separately scoped branches.
   - `BinaryExpr` / `UnaryExpr`: visit children, then fold the rebuilt node. An integer comparison (`==`, `!=`, `<`, `<=`, `>`, `>=` on signed-integer or `INDEX` operands) whose outcome the analyzer's `ConstIntBound` decides becomes a `ConstBool`, in any expression position including `Call` and `Submit` arguments; see [Provable comparison in an operand](#provable-comparison-in-an-operand).
   - `CallPtr`: refresh the result `type_` so a Call whose shape arguments folded ends up structurally equal to a freshly parsed Call.
   - `AssignStmt`: for a scalar LHS `Var` not in `multi_assigned_`, register the simplified RHS with the analyzer. A `ConstInt`/`ConstFloat`/`ConstBool` RHS at function-body top level is bound fully (the literal is substituted into later uses); a symbolic RHS — or a constant inside a loop/branch — contributes only a `ConstIntBound`, so dead branch guards fold without the scalar being inlined. Every binding is logged so the enclosing region's visitor can unbind it on exit.
   - `ForStmt`: rebuild `iter_args_` before visiting the body so body references pick up the remapped identity; if both `start_` and `stop_` fold to `ConstInt` with `stop > start`, bind the loop var to that range while visiting the body and unbind on exit; scalars bound inside the body are unbound after the visit; rebuild `return_vars_` after the body so folds discovered inside are visible in return types. Pure single-trip and zero-trip loops are also collapsed in-place — see "Control-flow folding" below.
   - `IfStmt`: enter `Analyzer::GetConstraintContext(cond)` for the then branch and `Not(cond)` for the else branch; scalars bound inside each branch are unbound after that branch so they do not leak into the other branch or past the `IfStmt`. Conditions the analyzer can prove are also folded — see "Control-flow folding" below.
   - `WhileStmt`: same as `ForStmt` minus the bounds — rebuild `iter_args_` before the condition and body, snapshot `var_remap_` around the body visit, then rebuild `return_vars_` after it, with the same scoped scalar unbinding. Seeding the remap from the loop header keeps every condition/body use attached to the declared node when its type or initial value changes.
   - `SpmdScopeStmt`: visit the body with the same scoped scalar unbinding, and additionally fold `core_num_` (closure arithmetic such as `MAX // TILE` may need one pass of simplification after SSA conversion).
3. **Type rebuild** — `SimplifyType` recurses through `TensorType`, `TileType`, and `TupleType`, calling `SimplifyExpr` on every embedded expression (shape, stride, valid_shape, start_offset, view fields). Identity is preserved when nothing changes so the round-trip identity check stays cheap.
4. **Scalar DCE + dead yield-slot prune** — after the mutator finishes, `dce::EliminateDeadScalarAssignments` walks the flattened body and drops scalar `AssignStmt`s whose only uses were folded away. The DCE is conservative: it never removes call-backed assignments because the IR has no purity annotations yet and a `Call` may have observable side effects. Between the two scalar-DCE runs, `dce::EliminateDeadYieldSlots` drops the yielded slots nobody reads — an `IfStmt` phi whose `return_vars_[i]` has no user, and a `ForStmt` / `WhileStmt` carry whose `iter_args_[i]` (read inside the body) *and* `return_vars_[i]` (read after the loop) are both unused — together with the matching `YieldStmt` slot. Reusing one Python local across two scopes produces exactly that dead carry: SSA seeds the second loop with the first scope's value, every trip overwrites it, and nothing reads either end. Left in place it makes the earlier scope's value live-out, which for a device scope would force a `Scalar` into the outlined kernel's return set — see [08-outline_incore_scopes.md](09-outline_incore_scopes.md).
5. **Loop-state repair** — if DCE removed any statements, `loop_repair::MakeBody` reassembles the function body so loop-carried metadata (yield/return mappings) stays consistent.

### Control-flow folding

Two folds run inside the `SimplifyMutator` traversal so they share the analyzer's constraint stack with the surrounding expression-level work:

- **Fold A — constant-condition `IfStmt` collapse.** After the condition is simplified, query the analyzer with `CanProve(cond)` and `CanProve(Not(cond))`. On a proof of either polarity, drop the dead branch and lift the kept branch into the parent scope. When `return_vars_` is non-empty, the kept branch's trailing `YieldStmt` is stripped and each `return_vars[i]` is bound in `var_remap_` to the corresponding yielded value so subsequent siblings (and the function `ReturnStmt`) read the value directly. Symmetric for true / false; the only edge case is "always-false with no else and empty return_vars," which collapses to an empty body.
- **Fold B — pure single/zero-trip `ForStmt` collapse.** Fires only on *pure* sequential loops: `attrs_` empty, `kind_ == ForKind::Sequential`. For these, query the analyzer for the trip count using `CanProveGreaterEqual(step, 1)` plus `CanProve(stop <= start)` (zero trips) or `CanProve(start < stop && stop <= start + step)` (one trip). On zero trips, emit one `AssignStmt(return_vars[i], iter_args[i].initValue_)` per return var and drop the body. On one trip, `DeepClone` the body with `loop_var → start` and `iter_args[i] → init_values[i]` substitutions, re-visit the cloned body so further folds happen in the same pass, then strip the trailing `YieldStmt` and bind each `return_vars[i] → yielded_value[i]` in `var_remap_` (same propagation mechanism as Fold A's lift). The one-trip path is capped at `kMaxNestedSingleTripFolds` (16) levels of *nested* single-trip loops per run — see [Substitution-depth cap](#substitution-depth-cap-fold-b).

`DeepClone` with `clone_def_vars=true` is used (rather than an in-place `var_remap_` override on the body) so the unrolled body gets fresh `Var` identities at every DefField, matching `LoopUnrollMutator`. This keeps the lifted copy structurally independent of the original (discarded) loop body and lets the re-visit bind the body's scalars on identities distinct from the surrounding scope.

Only definitions inside the cloned subtree receive fresh identities. A reference to an enclosing loop's `IterArg` must retain that exact object unless explicitly substituted; its defining loop is not being cloned. Local `ForStmt` and `WhileStmt` carries are registered from their `DefField`s before visiting their uses. This prevents folding two sibling single-trip loops from creating two unbound copies of the same outer carry (issue #2955), which would otherwise become invalid SPMD call arguments during outlining.

The choice to substitute `return_vars` via `var_remap_` rather than emit a literal `AssignStmt(rv, yielded)` is deliberate: the orchestration codegen's role-aware name disambiguation (`role == "out"` etc.) collapses several role-tagged SSA versions to the same C++ identifier, so an `out__rv_v2 = out__co_l0_rv_v3` alias would lower to the ill-formed `auto out = out;`. Substituting at use sites side-steps the disambiguation entirely.

#### Escaping return vars

A substitution only reaches uses visited while its `var_remap_` entry is live, and `ForStmt`, `WhileStmt` and `IfStmt` each rebase `var_remap_` to a pre-body baseline on the way out so a body-internal remap cannot rewrite siblings or post-loop code. A use that outlives that restore would keep the original `Var` — which the fold has just stripped the only definition of, a dangling reference `UseAfterDefCheck` reports.

`ReturnVarEscapeIndex` (a pre-pass in `simplify_pass.cpp`) decides this per fold site. It walks the function body once, numbering the restoring scopes in pre-order so a scope owns the contiguous id range `[id, end)` of its subtree; "every use of `v` sits inside scope `S`" is then two integer comparisons. A monotonic tick orders uses against the fold site, so a use *preceding* it inside the same scope counts as escaping too. One walk plus O(1) lookups per fold keeps Simplify within its O(N log N) budget.

Statements the index has never seen answer "does not escape", keeping the substitution. That covers folds nested inside a body Fold B `DeepClone`d, whose `Var` identities are minted after indexing. A clone's Vars are unreachable from outside it, so the only unhandled case is a restore-scope *within* a clone standing between such a fold and a later use of its return var — pre-SSA only, and no worse than the behaviour before this index existed. Re-indexing each clone would close it. It is left open because the gap is pre-SSA-only and the pipeline runs Simplify only after `ConvertToSSA`, so nothing but a direct pre-SSA caller can reach it.

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

### Provable comparison in an operand

**Before**:

```python
for i, (acc,) in pl.range(0, 512, 128, init_values=(seed,)):
    nxt = pl.matmul_acc(acc, a, b, init_cond=(flag and (i + 128 == 0)))
```

**After**:

```python
for i, (acc,) in pl.range(0, 512, 128, init_values=(seed,)):
    nxt = pl.matmul_acc(acc, a, b, init_cond=False)
```

The analyzer's rewrite rules alone only turn `i + 128 == 0` into `i == -128`. The fold decides the comparison by comparing the `ConstIntBound` intervals of its two operands (not the interval of a synthesised `lhs - rhs`, which the program never computes and which can wrap where the operands cannot): with `i ∈ [0, 512)` bound, `i + 128 ∈ [128, 639]` excludes 0, so the comparison becomes `ConstBool(false)`, and `flag and False` then reduces through the ordinary rewrite rules. Only comparison nodes are decided; enclosing `and`/`or`/`not` nodes fold once their operands are literals. The decision deliberately reads `ConstIntBound` alone rather than calling `CanProve` (which Fold A uses): `CanProve` falls back to the transitive-comparison analyzer, whose search rescans every known fact and would make the pass quadratic in the number of bound scalars if run per comparison, and its transitive and int-set facts are not refreshed when a pre-SSA scalar or loop variable is reassigned. Float operands are never decided (`x == x` is false for NaN). A comparison whose bound interval straddles the threshold — `i == 256`, or a comparison on a loop variable nothing bounds, e.g. one whose `stop` is a runtime value — is kept. This is how the `init_cond` of every replica `LowerPipelineLoops` produces after the first (`ko + k*step == 0`) becomes a literal, so codegen emits a single MAD instead of a branch on a compile-time constant.

A bound decides a comparison only when it describes what the hardware actually computes, and several things can make it not:

- **Width.** Codegen evaluates each arithmetic node at that node's own dtype, so the operation wraps there. For `i ∈ [120, 128)`, `pl.cast(i, pl.INT8) + pl.const(1, pl.INT8)` is emitted as `arith.addi : i8` and yields `-128` at `i == 127`, not the mathematical `[121, 128]`. A result the node's dtype cannot hold says nothing about *which* value lands, so the bound widens to that dtype's whole range and the comparison is kept. `INDEX` and `INT64` nodes are never narrowed — their dtype range is the entire int64 domain — so index arithmetic keeps full precision and the `init_cond` fold above is unaffected.
- **Domain edges.** `ConstIntBound` is an interval of int64 *values*, not of mathematical integers, and `kPosInf`/`kNegInf` mark that domain's extremes. A sum or product that leaves int64 wraps just as a narrow one does, so it widens too: `i * 4611686018427387903` for `i ∈ [2, 3]` is negative at `i == 3`. A literal numerically equal to a sentinel (`9223372036854775807`) cannot be told apart from an unbounded operand once stored in a bound, so the analyzer refuses to bound it rather than let it masquerade as `+inf`. `INT64_MIN` is deliberately *not* a sentinel, which is what lets `x - INT64_MIN` cancel exactly and decide `x == INT64_MIN`. Negation widens for the mirror-image reason: an interval whose lower endpoint is `kNegInf` has *no* lower bound and so contains `INT64_MIN`, which `arith.subi 0, x : i64` maps onto itself, leaving `-x` negative. Under `x <= 0`, `-x < 0` is therefore kept rather than folded to `False`; `pl.abs(x)` and `x * -1` reach the same wrap.
- **Rewrites under fixed-width wrap.** Most rewrite rules are identities over the integers, not over a fixed-width type. Moving a value across a non-modular operator (`<`, `<=`, `min`, `max`, `floordiv`, `floormod`, or `==` with a folded constant) is only valid if the sum or product the program emits did not wrap. For INT8 `c` in `[120, 127]`, `c + 1 < 0` is true at `c == 127` (the sum is `-128`), but the integer-identity rewrite `c < -1` is false everywhere, and the bound fold would then delete a branch the program needs. Rules of that kind are therefore **skipped for dtypes narrower than 64 bits** (`WrapsAtNodeWidth` in `src/ir/arith/rewrite_simplify.cpp`); the expression is kept as written and the bound fold still decides any comparison the operand bounds settle. Ring identities (`+`, `-`, `*` reassociation and distribution) are wrap-safe and unaffected. The one exception is dropping the remainder workaround `(x % m + k) % m`, which is applied when the operand bounds prove `x + k` fits the dtype. `INDEX`, `INT64` and `UINT64` keep the index-arithmetic convention of never wrapping, so `i + 1 < n` still normalises. `tests/ut/ir/arith/test_rewrite_wrap_soundness.py` evaluates each rule shape and its simplified form at INT8 over every assignment in a window; a new rule that crosses a non-modular operator should add a template there.
- **Sentinel lower endpoints.** `kNegInf` is stored as `INT64_MIN + 1` but, as a lower endpoint, means *no lower bound*, so a value can reach `INT64_MIN`. `CanProveGreaterEqual(e, k)` therefore proves nothing from a `kNegInf` lower endpoint unless `k == INT64_MIN`, and a cast between signed 64-bit types passes the source interval through unchanged (a genuine `INT64_MIN` endpoint is not "out of range"). Narrower and `UINT64` casts keep their wrap handling.

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

#### Substitution-depth cap (Fold B)

`DeepClone` copies the whole loop body. When that body is itself a pure single-trip loop, the re-visit folds it by cloning *its* body, and so on — clone sizes run N, N-1, …, 1, so a nest of N single-trip loops costs O(N²), above the O(N log N) ceiling `.claude/rules/pass-complexity.md` sets.

`kMaxNestedSingleTripFolds` (16, in `simplify_pass.cpp`) caps how many *nested* single-trip loops one run collapses. `fold_b_depth_` counts the Fold B clones currently on the stack; past the cap the loop falls through to the general path and stays a `ForStmt`. Folds at one depth sit at disjoint positions and so clone at most N nodes between them, which bounds the run's total cloning at O(N).

Declining is sound, not a missed obligation: the survivors are ordinary single-trip `ForStmt`s, and the next Simplify run takes the next 16 levels. No kernel in the pipeline nests *provably* one-trip pure loops anywhere near 16 deep, so the cap never fires on real input.

| Nest depth | Loops left after one run |
| ---------- | ------------------------ |
| ≤ 16 | 0 |
| 19 | 3 |

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
