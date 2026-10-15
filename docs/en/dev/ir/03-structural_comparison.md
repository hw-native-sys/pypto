# PyPTO Structural Comparison

## Overview

PyPTO provides three utility functions for comparing IR nodes by structure rather than pointer identity:

```python
structural_equal(lhs, rhs, enable_auto_mapping=False, *, mode=StructuralComparisonMode.Default) -> bool
structural_hash(node, enable_auto_mapping=False, *, mode=StructuralComparisonMode.Default) -> int
assert_structural_equal(lhs, rhs, enable_auto_mapping=False, *, mode=StructuralComparisonMode.Default) -> None
```

**Use Cases:** CSE, IR optimization, pattern matching, testing

**Key Feature:** All three functions ignore `Span` (source location), focusing only on logical structure.

### Floating-point constants

`ConstFloat` comparison uses its dtype and stored binary64 value. Finite values
use numeric equality (including `+0.0 == -0.0`); infinities compare by sign.
NaNs compare equal only when their binary64 representations match, including
sign and payload. This makes a preserved NaN structurally equal after a dump
round-trip without equating different NaN encodings. Equal constants also have
equal structural hashes. This does not change IEEE runtime comparison semantics
or the separate comparison of floating-point keyword/attribute metadata.

## Reference vs Structural Equality

### Reference Equality (Default `==`)

Compares pointer addresses (O(1), fast):

```python
from pypto import DataType, ir

x1 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
x2 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
assert x1 != x2  # Different pointers
```

### Structural Equality

Compares content and structure:

```python
ir.assert_structural_equal(x1, x2, enable_auto_mapping=True)  # True
```

## Comparison Process

In `Default` mode, `structural_equal` follows these steps:

1. **Fast Path Checks**
   - Null check: two null nodes compare equal; one null node compares unequal
   - Shared nodes still establish and check variable bindings, including shared subtrees
   - Only scalar constants retain the same-pointer shortcut
   - Type check: Compare `TypeName()` - must match exactly

2. **Type Dispatch**
   - Variables get special handling (auto-mapping)
   - Other types use reflection-based field comparison

3. **Field-Based Recursive Comparison**
   - Get field descriptors via `GetFieldDescriptors()`
   - Iterate through all fields using reflection
   - Compare each field based on its type
   - Combine results with AND logic

## Reflection and Field Types

The reflection system defines three field types:

| Field Type | Auto-Mapping | Compared? | Use Case | Effect |
| ---------- | ------------ | --------- | -------- | ------ |
| **IgnoreField** | N/A | ❌ No | Source locations (`Span`), names | Always considered equal |
| **UsualField** | Follows parameter | ✅ Yes | Operands, expressions, types | Compared with current `enable_auto_mapping` |
| **DefField** | ✅ Always enabled | ✅ Yes | Variable definitions, parameters | Always uses auto-mapping |

### Example Field Definitions

```cpp
class IRNode {
  Span span_;
  static constexpr auto GetFieldDescriptors() {
    return std::make_tuple(
      reflection::IgnoreField(&IRNode::span_, "span")
    );
  }
};

class BinaryExpr : public Expr {
  ExprPtr left_;
  ExprPtr right_;
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(
      Expr::GetFieldDescriptors(),
      std::make_tuple(
        reflection::UsualField(&BinaryExpr::left_, "left"),
        reflection::UsualField(&BinaryExpr::right_, "right")
      )
    );
  }
};

class AssignStmt : public Stmt {
  VarPtr var_;     // Definition
  ExprPtr value_;  // Usage
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(
      Stmt::GetFieldDescriptors(),
      std::make_tuple(
        reflection::DefField(&AssignStmt::var_, "var"),
        reflection::UsualField(&AssignStmt::value_, "value")
      )
    );
  }
};
```

### Why DefField Matters

DefFields represent variable definitions. When comparing definitions, we care about structural position, not identity:

```python
# Build: x = y
x1 = ir.Var("x", ir.ScalarType(DataType.INT64), span)
y1 = ir.Var("y", ir.ScalarType(DataType.INT64), span)
stmt1 = ir.AssignStmt(x1, y1, span)

# Build: a = b
a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
b = ir.Var("b", ir.ScalarType(DataType.INT64), span)
stmt2 = ir.AssignStmt(a, b, span)

# var_ is DefField, so x1 and a are mapped automatically
ir.assert_structural_equal(stmt1, stmt2, enable_auto_mapping=True)
```

## structural_equal Function

### Basic Usage

```python
# Same value
c1 = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())
c2 = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())
ir.assert_structural_equal(c1, c2)  # True

# Different types
var = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
const = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
assert not ir.structural_equal(var, const)  # False
```

### Auto-Mapping Behavior

| Scenario | enable_auto_mapping=False | enable_auto_mapping=True |
| -------- | ------------------------- | ------------------------ |
| Same variable pointer | ✅ Equal | ✅ Equal |
| Different variable pointers | ❌ Not equal | ✅ Equal (if type matches) |
| Consistent mapping (`x + x` vs `y + y`) | ❌ Not equal | ✅ Equal |
| Inconsistent mapping (`x + x` vs `y + z`) | ❌ Not equal | ❌ Not equal |

### When to Enable Auto-Mapping

| Use Case | Setting |
| -------- | ------- |
| Pass transform tests (Before/Expected pattern) | `False` (default) — DefFields always auto-map |
| Serialization round-trip tests | `True` — deserialized Vars are never DefFields |
| Pattern matching regardless of variable names | `True` |
| Template matching for optimization rules | `True` |
| Exact matching with same variables | `False` |
| CSE (Common Subexpression Elimination) | `False` |

> **Why pass tests don't need it:** `VisitDefField` always enables auto-mapping
> internally (see `structural_equal.cpp:650`), populating the bidirectional Var
> maps. When the same Vars later appear as UsualField references, they are found
> in the maps — even with `enable_auto_mapping=False`.

## structural_hash Function

### Basic Usage

```python
c1 = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())
c2 = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())
assert ir.structural_hash(c1) == ir.structural_hash(c2)
```

### Determinism

`structural_hash` is deterministic within a single process run. Variable identity is based on monotonic unique IDs (`Var::unique_id_`) assigned at construction, not pointer addresses or `name_hint_` strings, so the same construction sequence always produces the same hashes. The `name_hint_` field is an `IgnoreField` and is excluded from both structural comparison and hashing.

### Hash Consistency Guarantee

**Rule:** For supported inputs, equality implies equal hashes when both operations use the same
`enable_auto_mapping` and `mode` options. A hash match alone does not imply equality.

### Using with Containers

```python
class CSEPass:
    def __init__(self):
        self.expr_cache = {}

    def deduplicate(self, expr):
        hash_val = ir.structural_hash(expr, enable_auto_mapping=False)
        if hash_val in self.expr_cache:
            for cached_expr in self.expr_cache[hash_val]:
                if ir.structural_equal(expr, cached_expr, enable_auto_mapping=False):
                    return cached_expr
            self.expr_cache[hash_val].append(expr)
        else:
            self.expr_cache[hash_val] = [expr]
        return expr
```

## Auto-Mapping Algorithm

The implementation maintains bidirectional maps:

```cpp
class StructuralEqual {
  std::unordered_map<VarPtr, VarPtr> lhs_to_rhs_var_map_;
  std::unordered_map<VarPtr, VarPtr> rhs_to_lhs_var_map_;

  bool EqualVar(const VarPtr& lhs, const VarPtr& rhs) {
    if (!enable_auto_mapping_) {
      auto l = lhs_to_rhs_var_map_.find(lhs);
      auto r = rhs_to_lhs_var_map_.find(rhs);
      if (l != lhs_to_rhs_var_map_.end() || r != rhs_to_lhs_var_map_.end()) {
        return l != lhs_to_rhs_var_map_.end() && r != rhs_to_lhs_var_map_.end()
            && l->second == rhs && r->second == lhs;
      }
      return lhs.get() == rhs.get();  // Unbound free-variable identity
    }

    // Check type equality first
    if (!EqualType(lhs->GetType(), rhs->GetType())) return false;

    // Check existing mapping
    auto it = lhs_to_rhs_var_map_.find(lhs);
    if (it != lhs_to_rhs_var_map_.end()) {
      return it->second == rhs;  // Verify consistent
    }

    // Ensure rhs not already mapped to different lhs
    auto rhs_it = rhs_to_lhs_var_map_.find(rhs);
    if (rhs_it != rhs_to_lhs_var_map_.end() && rhs_it->second != lhs) {
      return false;
    }

    // Create new mapping
    lhs_to_rhs_var_map_[lhs] = rhs;
    rhs_to_lhs_var_map_[rhs] = lhs;
    return true;
  }
};
```

**Key Points:**

- Without auto-mapping: references to definitions follow the binding map; free variables preserve identity
- With auto-mapping: establish and enforce consistent mapping
- Type equality checked before mapping
- Bidirectional maps prevent inconsistent mappings

## Implementation Details

### Hash Combine Algorithm

Uses Boost-inspired algorithm:

```cpp
inline uint64_t hash_combine(uint64_t seed, uint64_t value) {
  return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}
```

### Reflection-Based Field Visitor

Generic traversal without type-specific code:

```cpp
template <typename NodePtr>
bool EqualWithFields(const NodePtr& lhs_op, const NodePtr& rhs_op) {
  using NodeType = typename NodePtr::element_type;
  auto descriptors = NodeType::GetFieldDescriptors();
  return std::apply([&](auto&&... descs) {
    return reflection::FieldIterator<...>::Visit(
      *lhs_op, *rhs_op, *this, descs...);
  }, descriptors);
}
```

### Default Type Comparison Uses Explicit Dispatch

In `Default` mode, `Expr` and `Stmt` use the reflection visitor above; `Type` uses explicit dispatch.
Four independent if/else ladders each re-encode "what fields does this Type
have", and nothing links them:

| Ladder | Location |
| ------ | -------- |
| `EqualType` | `src/ir/transforms/structural_equal.cpp` |
| `HashType` | `src/ir/transforms/structural_hash.cpp` |
| `SerializeType` | `src/ir/serialization/serializer.cpp` |
| `DeserializeType` | `src/ir/serialization/deserializer.cpp` |

**Adding a Type means updating all four and the Strict projection dispatcher.** A Type added to three of them
compiles, passes serialization round-trips, and compares correctly — then hits
`HashType`'s trailing `INTERNAL_CHECK(false)` the first time a user writes
`hash(t)` or puts the Type in a `set`. That breaks the equal-nodes-hash-equally
guarantee above, since `structural_equal` accepts what `structural_hash`
rejects.

Three dispatch hazards to mirror across the ladders:

- **Subclass kinds.** `As<T>()` is a precise `ObjectKind` match, so
  `As<TensorType>(dt)` returns `nullptr` for a `DistributedTensorType` by
  design (see `include/pypto/ir/kind_traits.h`). A shared branch must name both
  kinds explicitly and `static_pointer_cast` — see `.claude/rules/ir-kind-traits.md`.
- **Subclass-only fields.** `DistributedTensorType::window_buffer_` has no
  `TensorType` counterpart; a shared branch has to hash and compare it under a
  kind guard.
- **Nullable fields.** A bare `ExprPtr` field with no non-null invariant —
  `TileView::start_offset`, which the default ctor and `pl.TileView(...)` both
  leave unset — is a *legal, distinct* state, not a construction bug.
  `EqualType` gets that for free by routing it through the null-tolerant
  `Equal(...)`, so `HashType` must hash a presence tag rather than
  `INTERNAL_CHECK` on it. Guarding one ladder and not the other breaks the
  guarantee in the same direction as a missing Type: `structural_equal` reports
  a value equal to itself that `structural_hash` aborts on. Reserve
  `INTERNAL_CHECK` in `HashType` for fields that genuinely cannot be null, such
  as `shape_` elements.

`tests/ut/ir/transforms/test_hash.py::TestHashTypeLadderParity` walks every
Python-constructible Type and fails when one is missing from `HashType`. Add a
factory there for each new Type.

## Strict Comparison for Optimization Candidates

`StructuralComparisonMode.Strict` compares the exact supported IR structure. It is
intended for checking candidates before an optimization such as function deduplication.
The default mode and `Type.__eq__` / `Type.__hash__` retain their compatibility behavior.
The existing positional `enable_auto_mapping` argument remains valid; `mode` is keyword-only
in Python. The Type hash overload also retains the `type=` keyword.

```python
from pypto import DataType, ir

strict = ir.StructuralComparisonMode.Strict
span = ir.Span.unknown()
a = ir.ConstFloat(0.0, DataType.FP32, span)
b = ir.ConstFloat(-0.0, DataType.FP32, span)
assert ir.structural_equal(a, b)  # Default numerical comparison
assert not ir.structural_equal(a, b, mode=strict)  # Exact stored bits
```

```cpp
using namespace pypto::ir;
StructuralComparisonOptions options{StructuralComparisonMode::Strict, false};
bool same = structural_equal(lhs, rhs, options);
uint64_t fingerprint = structural_hash(lhs, options);
// For supported inputs, this throws ValueError exactly when same is false.
assert_structural_equal(lhs, rhs, options);
```

| Property | Strict behavior |
| -------- | --------------- |
| Source spans, comments, function and bound-variable names | Ignored through `IgnoreField` |
| Definitions and uses | Alpha-renamed consistently; alias patterns are preserved |
| Free variables | Identity preserved unless `enable_auto_mapping=True` |
| Scalar dtypes | Exact, including `INDEX` versus `INT64` |
| Floating values | Exact stored bits, including signed zero and NaN payload/sign |
| Operators | Kind and name, independent of pointer identity |
| Attributes | Unique keys sorted before traversal; value types and vector order retained |
| Types and views | Reflected fields, including layout, shape, valid shape and nullable offsets |
| Function metadata | Directions, return types, attrs, function kind and runtime-binding flag |
| Call and Submit | Distinct kinds; all reflected scheduling fields participate |

A Function reserves every parameter before examining its signature, so a tensor shape
may reference a later scalar parameter. Symbols in physical parameter shapes and bare
parameter `valid_shape` slots are additional bindings. Their classification is shared with
`MaterializeValidShapeSymbols`, including symbols nested inside tuple constructors
and tuple-item expressions. Compound `valid_shape` expressions
must refer to these bindings or scalar parameters. Return types, bodies, and attributes
share the same function binding context. Program functions have independent contexts;
program map keys and callee names remain significant. An IterArg initializer is an outer
reference, not an additional definition introduced by the IterArg's `DefField`.

Strict rejects local bindings used before definition or outside their lexical scope,
including uses hidden in types and attributes. Assignments bind after their RHS; loop
initializers use the outer scope; loop and branch results bind after the region finishes.
Loop-body and branch-local definitions do not escape. Shared expressions are checked at
all use sites. True open captures retain free-variable semantics; they do not become
bound merely because another function defines the same object.

These checks do not replace full SSA, closure, effect, or call-graph verification.
An optimization must check those preconditions separately. In particular, an equal
fingerprint does not authorize merging effectful functions.

### Unsupported Inputs

All three Strict operations raise `ir.StructuralComparisonUnsupported` (a `TypeError`)
when the input cannot be represented safely, even when comparing an object to itself.
This outcome is distinct from inequality. The exception exposes `reason` and `field_path`.

Explicit storage is currently unsupported: `MemRef`, `WindowBuffer`, `MemRefType`,
`PtrType`, `WindowBufferType`, `CommCtxType`, and attached storage anywhere in a type,
body, signature, view, or attribute. The traversal also rejects unresolved/unknown types,
unknown attribute payload types, duplicate attribute keys, repeated variable definitions,
shared statement objects, cycles, and node/type nesting beyond 512 levels.
Invalid binding visibility reports `reason="BindingNotVisible"`; `field_path` identifies
the invalid use, including a later scope escape through a shared expression or type.
A nullable view offset is supported and remains distinct from a zero-valued offset.

```python
try:
    fingerprint = ir.structural_hash(candidate, mode=ir.StructuralComparisonMode.Strict)
except ir.StructuralComparisonUnsupported as error:
    reason, field_path = error.reason, error.field_path
    # Skip this optimization candidate; do not turn unsupported into equality.
```

### Shared Projection and Complexity

Strict equality, hashing, and assertion diagnostics use one reflection-based projection
for Expr, Stmt, Function, Type, TensorView and TileView fields. Preparation reserves
binding IDs, then checks lexical visibility. Each binder has a contiguous lifetime;
earliest/latest use positions propagate through expression/type DAG edges in reverse
postorder, checking every use without repeated subtree scans. The final traversal builds
immutable records containing scalar values and child record IDs. Records are interned by exact ordered keys. Equality compares IDs in a shared
arena, never hashes. Hashes combine child hashes rather than arena IDs, so separate calls
agree. Assertions walk differing records to report a field path and exact scalar values.

Memoization distinguishes definitions from uses and belongs to one function context.
Pure expression DAG sharing does not affect equality and does not expand into a tree.
For N distinct visited nodes plus their field/string data, projection uses O(N log N)
time and O(N) space. Hashes are process-local fingerprints, not persistent cache keys.
Default equality memoizes successful node pairs by comparison context. Default hashing
also separates mapping and loop-variable contexts and retains stable variable identities.
Both use reverse dependency edges to invalidate only cached ancestors of a previously
free variable when that variable becomes bound; unrelated bindings preserve cached
shared subtrees. Equality tracks left and right binding roles independently.
Variable hashes include their type traversal in this cache, so repeated auto-mapped
uses of one variable do not rescan its type.
Hashing's loop-variable dtype relaxation matches Default equality.

## Summary

**Key Takeaways:**

1. **Three Field Types**:
   - `IgnoreField`: Never compared (Span, names)
   - `UsualField`: Compared with user's `enable_auto_mapping`
   - `DefField`: Always uses auto-mapping

2. **Auto-Mapping**:
   - Enable for pattern matching
   - Disable for exact CSE
   - Always consistent: maintains bijective variable mapping

3. **Hash Consistency**:
   - Equal nodes → equal hashes (guaranteed)
   - Use same `enable_auto_mapping` for both functions

For IR node types and construction, see [IR Overview](00-overview.md).
