# IR Kind-Trait Downcasting

## Core Rule

**For a concrete node type, `As<T>()` matches that exact `ObjectKind` only, NOT subclasses.** When you want to treat a concrete type and its subclass(es) uniformly, use the corresponding `*Like` helper, not `As<Base>()`.

## Why

PyPTO's IR uses a single `ObjectKind` enum for runtime-type dispatch. `KindTrait<T>` (see `include/pypto/ir/kind_traits.h`) comes in two shapes, and `As<T>()` behaves differently for each:

| `KindTrait<T>` shape | Applies to | `As<T>()` matches |
| -------------------- | ---------- | ----------------- |
| single `kind` member | every concrete node — `Var`, `IterArg`, `MemRef`, `WindowBuffer`, `TensorType`, … | that one kind — exact match, **never** subclasses |
| `kinds[]` array | the seven base types `Expr`, `Stmt`, `Type`, `BinaryExpr`, `UnaryExpr`, `ScopeStmt`, `ShapedType` (the last is also concrete, and lists its own kind) | any kind listed in the array |

The first row is where the bugs are — C++ inheritance doesn't help there. `IterArg` subclasses `Var` and `DistributedTensorType` subclasses `TensorType`, but each has its own `ObjectKind`, so `As<Var>(iter_arg)` and `As<TensorType>(window_type)` both return **null**.

`As<Expr>()` / `As<Stmt>()` / `As<Type>()` are the second row: they match whole subtrees by design and are correct as written — do **not** rewrite them into `*Like` helpers. Their arrays are hand-maintained, so a new kind must be appended to the base array too; `static_assert`s in `kind_traits.h` catch only the base-covers-derived case, not enum coverage.

## The cases that bite

| Have | Want | Correct API | Wrong API |
| ---- | ---- | ----------- | --------- |
| `ExprPtr` that may be `Var` or `IterArg` | Treat both as `Var` | `AsVarLike(expr)` (returns `VarPtr`) | `As<Var>(expr)` — misses `IterArg` |
| Visitor override for both `Var` and `IterArg` | Single handler for both | Override `VisitVarLike_` | Override `VisitExpr_(VarPtr)` only — `IterArg` dispatches separately |
| `TypePtr` of a GM operand that may be a `pld.DistributedTensor` window | Treat a window as this rank's local GM | `AsTensorTypeLike(type)` (returns `TensorTypePtr`) | `As<TensorType>(type)` — misses `DistributedTensorType` |

**The window pair bites hardest**, because a window reaches *every* layer that inspects a
user operand: op type deducers (`src/ir/op/**`), the tensor→tile pass, and orchestration
codegen. Before writing `As<TensorType>(<operand>->GetType())`, ask **"is this asking
*whether* the operand is GM data, or *which kind* of GM it is?"**

- *Whether* → `AsTensorTypeLike`. A window slice is this rank's local GM; ordinary compute
  reads and writes it like any GM tensor. Exact-kind here rejects a legal program, or —
  worse — skips the operand silently and fails later in a pass or emitter.
- *Which kind* → `As<TensorType>` / `As<DistributedTensorType>`, and say why in a comment.
  Legitimate uses: propagating the window kind onto a result (`tensor.slice`, `tensor.assemble`),
  and withholding metadata a window has no lowering contract for (element-wise `valid_shape`).

**Relaxing the deducer alone is not the fix.** Every operand crosses at least two gates —
the type deducer *and* the lowering site (`BridgeInputSpaces` / the entry load in
`ConvertTensorToTileOps`, the op's conversion rule, the emitter). Fix all of them together
and add a lowering test, or the error just moves from a `CHECK` to an `INTERNAL_UNREACHABLE`.

`MemRef` and `WindowBuffer` are intentionally **excluded** from `AsVarLike` — they carry allocation-source / window-slot semantics that don't fit the Var-bound-name model. Use `As<MemRef>()` / `As<WindowBuffer>()` directly. Both are still listed in `KindTrait<Expr>`, so `As<Expr>()` matches them.

## Examples

```cpp
// ❌ WRONG — null for IterArg, so materialization runs for an already-bound Var
if (As<Var>(yield_value)) { /* skip materialization */ }
// ✅ CORRECT — matches Var AND IterArg
if (AsVarLike(yield_value)) { /* skip materialization */ }
```

```cpp
// ❌ WRONG — the window operand is skipped, then fails in a later pass
auto tensor_type = As<TensorType>(args[0]->GetType());

// ✅ CORRECT — a window is GM data here, same as a plain tensor
auto tensor_type = AsTensorTypeLike(args[0]->GetType());
```

The visitor form is the same mistake: override `VisitVarLike_`, not `VisitExpr_(VarPtr)`.

## Decision rule

Before writing `As<T>(...)`: does `T` have subclasses with their own `ObjectKind` (check `include/pypto/ir/kind_traits.h`)? If so, use the `As<T>Like()` helper — or write one there rather than re-rolling `dynamic_pointer_cast<...>` at the call site. Grep `AsVarLike` / `AsTensorTypeLike` and mirror the pattern.
