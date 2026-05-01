# IR Kind-Trait Downcasting

## Core Rule

**`As<T>()` matches the exact `ObjectKind` only, NOT subclasses.** When you want to treat a base type and its subclass(es) uniformly, use the corresponding `*Like` helper, not `As<Base>()`.

## Why

PyPTO's IR uses a single `ObjectKind` enum for runtime-type dispatch (see `include/pypto/ir/kind_traits.h`). `As<T>(node)` checks `node->GetKind() == ObjectKind::T` — exact match.

C++ inheritance doesn't help here: `IterArg` is a subclass of `Var`, but `IterArg` has its own `ObjectKind::IterArg`. So `As<Var>(iter_arg_ptr)` returns **null**, even though `iter_arg` IS-A Var.

## The cases that bite

| Have | Want | Correct API | Wrong API |
| ---- | ---- | ----------- | --------- |
| `ExprPtr` that may be `Var` or `IterArg` | Treat both as `Var` | `AsVarLike(expr)` (returns `VarPtr`) | `As<Var>(expr)` — misses `IterArg` |
| Visitor override for both `Var` and `IterArg` | Single handler for both | Override `VisitVarLike_` | Override `VisitExpr_(VarPtr)` only — `IterArg` dispatches separately |

`MemRef` is intentionally **excluded** from `AsVarLike` — `MemRef` has scope/storage semantics that don't fit the Var-bound-name model. Use `As<MemRef>()` directly.

## Examples

```cpp
// ❌ WRONG — As<Var> won't match IterArg
for (const auto& yield_value : yields) {
  if (As<Var>(yield_value)) {     // returns null for IterArg!
    // skip materialization
  }
  // ... materialization runs even for IterArg
}

// ✅ CORRECT — AsVarLike matches Var AND IterArg
for (const auto& yield_value : yields) {
  if (AsVarLike(yield_value)) {   // matches both
    // skip materialization for any already-bound Var
  }
}
```

```cpp
// ❌ WRONG — only catches Var, IterArgs go through default
class MyVisitor : public IRVisitor {
  void VisitExpr_(const VarPtr& op) override { /* ... */ }
};

// ✅ CORRECT — VisitVarLike_ handles both Var and IterArg
class MyVisitor : public IRVisitor {
  void VisitVarLike_(const VarPtr& op) override { /* ... */ }
};
```

## Decision rule

Before writing `As<T>(...)`, ask:

1. Does `T` have subclasses with their own `ObjectKind` values? Check `include/pypto/ir/kind_traits.h` and the class hierarchy.
2. If yes: is there an `As<T>Like()` helper? If yes, use it.
3. If no `*Like` helper exists and you genuinely need union semantics: write one in `kind_traits.h` rather than re-rolling `dynamic_pointer_cast<...>` at the call site.

When in doubt: grep for `AsVarLike` / `VisitVarLike_` in the codebase to confirm the pattern, then mirror it.
