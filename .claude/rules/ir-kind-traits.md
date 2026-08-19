# IR Kind-Trait Downcasting

## Core Rule

**For a concrete node type, `As<T>()` matches that exact `ObjectKind` only, NOT subclasses.** When you want to treat a concrete type and its subclass(es) uniformly, use the corresponding `*Like` helper, not `As<Base>()`.

## Why

PyPTO's IR uses a single `ObjectKind` enum for runtime-type dispatch. `KindTrait<T>` (see `include/pypto/ir/kind_traits.h`) comes in two shapes, and `As<T>()` behaves differently for each:

| `KindTrait<T>` shape | Applies to | `As<T>()` matches |
| -------------------- | ---------- | ----------------- |
| single `kind` member | every concrete node — `Var`, `IterArg`, `MemRef`, `WindowBuffer`, `TensorType`, … | that one kind — exact match, **never** subclasses |
| `kinds[]` array | the seven base types `Expr`, `Stmt`, `Type`, `BinaryExpr`, `UnaryExpr`, `ScopeStmt`, `ShapedType` (the last is also concrete, and lists its own kind) | any kind listed in the array |

The core rule is about the first row, which is where the bugs are: C++ inheritance doesn't help there. `IterArg` is a subclass of `Var`, but `IterArg` has its own `ObjectKind::IterArg`. So `As<Var>(iter_arg_ptr)` returns **null**, even though `iter_arg` IS-A Var.

`As<Expr>()` / `As<Stmt>()` / `As<Type>()` are the second row: they match whole subtrees by design and are correct as written — do **not** rewrite them into `*Like` helpers. Their arrays are hand-maintained, so a new kind must be appended to the base array too; `static_assert`s in `kind_traits.h` catch only the base-covers-derived case, not enum coverage.

## The cases that bite

| Have | Want | Correct API | Wrong API |
| ---- | ---- | ----------- | --------- |
| `ExprPtr` that may be `Var` or `IterArg` | Treat both as `Var` | `AsVarLike(expr)` (returns `VarPtr`) | `As<Var>(expr)` — misses `IterArg` |
| Visitor override for both `Var` and `IterArg` | Single handler for both | Override `VisitVarLike_` | Override `VisitExpr_(VarPtr)` only — `IterArg` dispatches separately |

`MemRef` and `WindowBuffer` are intentionally **excluded** from `AsVarLike` — they carry allocation-source / window-slot semantics that don't fit the Var-bound-name model. Use `As<MemRef>()` / `As<WindowBuffer>()` directly. Both are still listed in `KindTrait<Expr>`, so `As<Expr>()` matches them.

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
