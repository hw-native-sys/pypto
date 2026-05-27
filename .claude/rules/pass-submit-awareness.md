# Call vs Submit Awareness in Passes

> **Status:** `Submit` is a first-class IR kind sibling to `Call`, representing
> a task launch from `pl.submit(...)` inside a `pl.manual_scope`. If the kind
> has not yet landed in the codebase you are reading, treat this rule as the
> required handling once it does — and treat any current attrs-based encoding
> (e.g. `Call::attrs_["manual_dep_edges"]`, return-type augmentation with
> `Scalar[TASK_ID]`) with the same "submit-aware" diligence in the interim.

## Core Principle

**PyPTO has two call-like Expr kinds: `Call` (plain function call) and `Submit`
(task launch). When you create or modify a pass that inspects calls, you MUST
consider both — never assume `Call` covers all call-like nodes.**

This is the same shape of issue as `Var` / `IterArg` covered in
`ir-kind-traits.md`: a single `ObjectKind` dispatch will silently skip submits,
and the bug only surfaces inside `pl.manual_scope` bodies.

## What Distinguishes Submit from Call

| Aspect | `Call` | `Submit` |
| ------ | ------ | -------- |
| Semantics | Synchronous function call | Asynchronous task launch |
| Where it appears | Anywhere | Inside `manual_scope` bodies |
| Return type | Callee's declared return | `Tuple[<runtime-allocated Out params>..., <callee return>..., TASK_ID]` |
| `args_` vs callee `params_` | Identity, full coverage: `args_.size() == params_.size()` | Identity, *prefix*: `args_.size() <= params_.size()` |
| Has `deps` | No | First-class `deps_` field — TaskId `Var`s / `Array`s |
| Use-def chain | `args_` only | `args_` **and** `deps_` |
| Python syntax | `out = self.foo(...)` | `out, tid = pl.submit(self.foo, ...)` |

## What Pass Authors Must Do

### 1. When walking calls, walk Submit too

```cpp
// ❌ Silently skips submits — bug inside manual_scope
void MyPass::VisitExpr_(const CallPtr& op) {
  for (const auto& arg : op->args()) { Visit(arg); }
}

// ✅ Handle both — visitor dispatches on kind
void MyPass::VisitExpr_(const CallPtr& op)   { VisitCallLike(op); }
void MyPass::VisitExpr_(const SubmitPtr& op) { VisitCallLike(op); }

void MyPass::VisitCallLike(const ExprPtr& op) {
  auto view = AsCallLike(op);              // unified accessor
  for (const auto& arg : view.args()) Visit(arg);
  for (const auto& dep : view.deps()) Visit(dep);  // empty for Call
}
```

### 2. Treat `deps_` as part of the use-def chain

A `Submit::deps_` entry is a TaskId `Var` or `Array` — a real SSA value, not
metadata. Passes that:

- Collect uses → must include `deps_`
- Substitute variables → must rewrite `deps_` too
- Validate SSA dominance → must check `deps_` are defined before the `Submit`
- Run DCE / liveness → TaskId vars are live through `deps_`

```cpp
// ❌ Missing deps_ — TaskId vars look unused, may get DCE'd
std::vector<VarPtr> CollectUses(const SubmitPtr& op) {
  return CollectVarUses(op->args());
}

// ✅ Submit uses include deps_
std::vector<VarPtr> CollectUses(const SubmitPtr& op) {
  auto uses = CollectVarUses(op->args());
  for (const auto& dep : op->deps()) AppendVarUses(&uses, dep);
  return uses;
}
```

### 3. When transforming, preserve Submit-ness

A pass that rewrites a `Submit` must produce another `Submit`, not a plain
`Call` — even if `deps_` becomes empty after the rewrite. The `TASK_ID` return
shape and the structural property "submit appears inside manual_scope" must be
preserved.

```cpp
// ❌ Loses the Submit kind — return type no longer matches the binding LHS
auto new_call = std::make_shared<Call>(op->callee(), new_args, ...);

// ✅ Preserve kind through rewrite
auto new_submit = std::make_shared<Submit>(
    op->callee(), new_args, op->deps(), op->return_type(), op->span());
```

### 4. When examining return types

`Submit`'s return type is **always** augmented with `Scalar[TASK_ID]` at the
tail. If your pass inspects return types of call-like nodes (for tuple
projection, type inference, etc.), strip / account for the trailing `TASK_ID`
before comparing against the callee's declared signature.

### 5. When matching `args_` against callee `params_`

`Submit::args_` is a positional *prefix* of callee `params_`. Mapping is
identity (`args_[i] ↔ params_[i]`) up to `args_.size()`, but Submit may carry
fewer args than the callee declares:

- Indices `[0, args_.size())` are caller-supplied (any direction — In, InOut,
  *or* Out for caller-allocated outputs).
- Indices `[args_.size(), params_.size())` are **runtime-allocated outputs**:
  they must be declared `Out` and the IR builder appends them at the *tail*
  of the callee signature (e.g. `ConvertTensorToTileOps` lowers a local
  `pl.create_tensor` consumed across scopes into an appended `pl.Out` param).
  These materialise as leading return-tuple elements before `TASK_ID`.

For a plain `Call`, args is full coverage: `args_.size() == params_.size()`.

A pass that hard-codes `args_.size() == params_.size()` (typical size guard)
will silently bail on Submit and leave attrs empty. Use a kind-aware bound:

```cpp
// ❌ Wrong for Submit when callee has tail-appended runtime-allocated outputs
if (call->args_.size() != callee->params_.size()) return call;  // bails on Submit

// ✅ Identity mapping; relax the size check for Submit (prefix), keep it
//    exact for Call (full coverage).
const bool size_ok = is_submit ? (call->args_.size() <= callee->params_.size())
                               : (call->args_.size() == callee->params_.size());
if (!size_ok) return call;
for (size_t i = 0; i < call->args_.size(); ++i) {
  auto dir = callee->param_directions_[i];   // identity — always safe
  ...
}
// For Submit, callee params beyond args_.size() are runtime-allocated
// outputs — pass them through the runtime's add_output / TaskOutputTensors
// path (orchestration codegen) rather than args_-indexed code.
```

This is the args-side dual of the return-type asymmetry in rule 4. If your
pass uses a `SubmitToCallView`–style adapter to share code with the Call
path, the adapter does **not** fix this — `args_` is copied through
unchanged, so the size-vs-params guard will still mis-fire.

### 6. Verifier hooks

If a pass produces a new IR property that involves `Submit` (e.g. "all submit
dependencies are dominated by their definitions"), add a `SubmitVerifier` and
register it in the `PropertyVerifierRegistry` — same pattern as other
property verifiers (see `pass-context-config.md` and `documentation.md`).

## Decision Guide: Unified vs Separate Handlers

```text
Does the pass treat Call and Submit identically?
├─ YES → Use AsCallLike() / a shared VisitCallLike helper — one code path
│
└─ NO  → Submit needs special handling (visits deps_, checks scope, etc.)
    └─ Override VisitExpr_(CallPtr) and VisitExpr_(SubmitPtr) separately
```

When in doubt, separate handlers are safer — the unified `CallLike` view is an
optimization for passes that genuinely do not care about deps or task-launch
semantics.

## Audit Checklist

Before merging a new or updated pass that touches calls, verify:

- [ ] If the pass inspects `Call`, it also inspects `Submit` (or uses `CallLike`)
- [ ] If the pass collects variable uses, `Submit::deps_` is included
- [ ] If the pass rewrites or clones calls, Submit-ness is preserved
- [ ] If the pass examines call return types, the `TASK_ID` suffix on `Submit` is accounted for
- [ ] If the pass matches `args_` against callee `params_`, it allows the Submit prefix invariant (size_ok = `args_.size() <= params_.size()`, not `==`)
- [ ] If the pass produces a structural invariant, the verifier covers `Submit` too
- [ ] Tests cover at least one `pl.manual_scope` / `pl.submit` case if the pass is reachable from there

## See Also

- `ir-kind-traits.md` — broader pattern for sibling-kind dispatch (`Var`/`IterArg`, `Call`/`Submit`)
- `pass-complexity.md` — O(N log N) complexity rule still applies
- `pass-context-config.md` — verifier registration via `PassContext`
- `pass-doc-ordering.md` — doc numbering when adding a new pass
