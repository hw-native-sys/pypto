# PyPTO Structural Comparison

## Overview

PyPTO provides two utility functions for comparing IR nodes by structure rather than pointer identity:

```python
structural_equal(lhs, rhs, enable_auto_mapping=False) -> bool
structural_hash(node, enable_auto_mapping=False) -> int
```

**Use Cases:** CSE, IR optimization, pattern matching, testing

**Key Feature:** Both functions ignore `Span` (source location), focusing only on logical structure.

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

The `structural_equal` function follows these steps:

1. **Fast Path Checks**
   - Reference equality: If same pointer, return `true`
   - Null check: If either is null, return `false`
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
|------------|--------------|-----------|----------|--------|
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
|----------|---------------------------|--------------------------|
| Same variable pointer | ✅ Equal | ✅ Equal |
| Different variable pointers | ❌ Not equal | ✅ Equal (if type matches) |
| Consistent mapping (`x + x` vs `y + y`) | ❌ Not equal | ✅ Equal |
| Inconsistent mapping (`x + x` vs `y + z`) | ❌ Not equal | ❌ Not equal |

### When to Enable Auto-Mapping

| Use Case | Setting |
|----------|---------|
| Pattern matching regardless of variable names | `True` |
| Template matching for optimization rules | `True` |
| Exact matching with same variables | `False` |
| CSE (Common Subexpression Elimination) | `False` |

## structural_hash Function

### Basic Usage

```python
c1 = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())
c2 = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())
assert ir.structural_hash(c1) == ir.structural_hash(c2)
```

### Hash Consistency Guarantee

**Rule:** If `structural_equal(a, b, mode)` is `True`, then `structural_hash(a, mode) == structural_hash(b, mode)`

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
      return lhs.get() == rhs.get();  // Strict pointer equality
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
- Without auto-mapping: strict pointer comparison
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

For IR node types and construction, see [IR Overview](00-ir_overview.md).
