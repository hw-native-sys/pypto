# PyPTO Structural Comparison

## Table of Contents

- [Overview](#overview)
- [Reference vs Structural Equality](#reference-vs-structural-equality)
- [The Comparison Process](#the-comparison-process)
- [Reflection and Field Types](#reflection-and-field-types)
- [structural_equal Function](#structural_equal-function)
- [structural_hash Function](#structural_hash-function)
- [Auto-Mapping Deep Dive](#auto-mapping-deep-dive)
- [Implementation Details](#implementation-details)

## Overview

PyPTO provides two utility functions for comparing IR nodes by structure rather than pointer identity:

```python
structural_equal(lhs, rhs, enable_auto_mapping=False) -> bool
structural_hash(node, enable_auto_mapping=False) -> int
```

**Use Cases:**
- **Common Subexpression Elimination (CSE)**: Identifying identical computations
- **IR Optimization**: Finding and replacing equivalent subtrees
- **Pattern Matching**: Applying transformation rules
- **Testing**: Verifying IR transformations

**Key Feature**: Both functions ignore `Span` (source location), focusing only on logical structure.

## Reference vs Structural Equality

### Reference Equality (Default `==`)

The default equality operator compares pointer addresses:

```python
from pypto import DataType, ir

x1 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
x2 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())

assert x1 != x2  # True - different pointers
```

**Why?** Fast O(1) comparison and efficient DAG construction with shared subtrees.

### Structural Equality

Structural equality compares content and structure:

```python
# With auto_mapping, these are structurally equal
assert ir.structural_equal(x1, x2, enable_auto_mapping=True)  # True
```

## The Comparison Process

The `structural_equal` function follows a systematic multi-step process:

### Step 1: Fast Path Checks

```cpp
bool StructuralEqual::Equal(const IRNodePtr& lhs, const IRNodePtr& rhs) {
  // Fast path: reference equality
  if (lhs.get() == rhs.get()) return true;

  // Null check
  if (!lhs || !rhs) return false;

  // Type check: must be same concrete type
  if (lhs->TypeName() != rhs->TypeName()) return false;

  // ... continue to field-based comparison
}
```

**Checks performed:**
1. **Reference equality**: If same pointer, return `true` immediately
2. **Null check**: If either is null, return `false`
3. **Type check**: Compare `TypeName()` - must match exactly (e.g., "Add" == "Add")

### Step 2: Type Dispatch

```cpp
// Dispatch to type-specific handlers
if (auto lhs_var = std::dynamic_pointer_cast<const Var>(lhs)) {
  return EqualVar(lhs_var, std::static_pointer_cast<const Var>(rhs));
}

// Other types use generic field-based comparison
EQUAL_DISPATCH(ConstInt)
EQUAL_DISPATCH(BinaryExpr)
EQUAL_DISPATCH(AssignStmt)
// ... etc
```

**Process:**
- Use `dynamic_pointer_cast` to determine concrete type
- Variables get special handling (auto-mapping)
- Other types use reflection-based field comparison

### Step 3: Field-Based Recursive Comparison

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

**For each field:**
1. Get field descriptors via `GetFieldDescriptors()`
2. Iterate through all fields using reflection
3. Compare each field based on its type (see next section)
4. Combine results with AND logic

## Reflection and Field Types

The reflection system defines three field types that **fundamentally affect** structural comparison:

### 1. IgnoreField

**Purpose**: Fields that should be ignored during comparison

**Example**: `Span` (source location), `name_` (function/variable names)

```cpp
class IRNode {
  Span span_;

  static constexpr auto GetFieldDescriptors() {
    return std::make_tuple(
      reflection::IgnoreField(&IRNode::span_, "span")
    );
  }
};

class Function : public IRNode {
  std::string name_;

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(
      IRNode::GetFieldDescriptors(),
      std::make_tuple(
        reflection::IgnoreField(&Function::name_, "name"),
        // ... other fields
      )
    );
  }
};
```

**Effect on Comparison**:
```cpp
template <typename FVisitOp>
void VisitIgnoreField([[maybe_unused]] FVisitOp&& visit_op) {
  // Do nothing - ignored fields are always considered equal
}
```

**Result**: Two nodes with different `Span` or `name_` values are still considered equal.

### 2. UsualField

**Purpose**: Regular fields that should be compared normally

**Example**: Expression operands, statement bodies

```cpp
class BinaryExpr : public ScalarExpr {
  ExprPtr left_;
  ExprPtr right_;

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(
      ScalarExpr::GetFieldDescriptors(),
      std::make_tuple(
        reflection::UsualField(&BinaryExpr::left_, "left"),
        reflection::UsualField(&BinaryExpr::right_, "right")
      )
    );
  }
};
```

**Effect on Comparison**:
```cpp
template <typename FVisitOp>
void VisitUsualField(FVisitOp&& visit_op) {
  visit_op();  // Execute comparison normally
}
```

**Result**: Fields are compared recursively with current `enable_auto_mapping` setting.

### 3. DefField

**Purpose**: Definition fields that introduce new bindings (variables)

**Example**: Loop variables, assignment targets, function parameters, return variables

```cpp
class AssignStmt : public Stmt {
  VarPtr var_;     // Definition - introduces binding
  ExprPtr value_;  // Usage - regular field

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

class Function : public IRNode {
  std::vector<VarPtr> params_;  // Parameters are definitions

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(
      IRNode::GetFieldDescriptors(),
      std::make_tuple(
        reflection::DefField(&Function::params_, "params"),
        // ... other fields
      )
    );
  }
};
```

**Effect on Comparison**:
```cpp
template <typename FVisitOp>
void VisitDefField(FVisitOp&& visit_op) {
  // Temporarily force enable_auto_mapping=true for DefFields
  bool enable_auto_mapping = true;
  std::swap(enable_auto_mapping, enable_auto_mapping_);
  visit_op();
  std::swap(enable_auto_mapping, enable_auto_mapping_);
}
```

**Result**: DefFields **always use auto-mapping** regardless of the user's `enable_auto_mapping` setting.

### Why DefField Matters

DefFields represent variable definitions, not usages. When comparing definitions, we care about structural position, not identity:

```python
from pypto import DataType, ir

span = ir.Span.unknown()
dtype = DataType.INT64

# Build: x = y
x1 = ir.Var("x", ir.ScalarType(dtype), span)
y1 = ir.Var("y", ir.ScalarType(dtype), span)
stmt1 = ir.AssignStmt(x1, y1, span)

# Build: a = b
a = ir.Var("a", ir.ScalarType(dtype), span)
b = ir.Var("b", ir.ScalarType(dtype), span)
stmt2 = ir.AssignStmt(a, b, span)

# var_ is DefField, so x1 and a are mapped automatically
# value_ is UsualField, follows enable_auto_mapping setting
assert ir.structural_equal(stmt1, stmt2, enable_auto_mapping=True)
```

**Explanation:**
- `var_` (DefField): `x1` → `a` mapping created automatically
- `value_` (UsualField): `y1` vs `b` compared with `enable_auto_mapping=True`

### Comparison Behavior Summary

| Field Type | Auto-Mapping | Compared? | Use Case |
|------------|--------------|-----------|----------|
| **IgnoreField** | N/A | ❌ No | Source locations (`Span`), names |
| **UsualField** | Follows parameter | ✅ Yes | Operands, expressions, types |
| **DefField** | ✅ Always enabled | ✅ Yes | Variable definitions, parameters |

## structural_equal Function

### Basic Usage

```python
from pypto import DataType, ir

# Constants with same value
c1 = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())
c2 = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())
assert ir.structural_equal(c1, c2)  # True

# Different node types
var = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
const = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
assert not ir.structural_equal(var, const)  # False
```

### Auto-Mapping Examples

**Example 1: Without Auto-Mapping (Strict)**

```python
x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
one = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())

expr1 = ir.Add(x, one, DataType.INT64, ir.Span.unknown())  # x + 1
expr2 = ir.Add(y, one, DataType.INT64, ir.Span.unknown())  # y + 1

# Different variable pointers → not equal
assert not ir.structural_equal(expr1, expr2, enable_auto_mapping=False)
```

**Example 2: With Auto-Mapping (Pattern Matching)**

```python
# Same expressions with auto_mapping
assert ir.structural_equal(expr1, expr2, enable_auto_mapping=True)
```

**Example 3: Consistent Mapping**

```python
# Build: x + x
x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
expr1 = ir.Add(x, x, DataType.INT64, ir.Span.unknown())

# Build: y + y
y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
expr2 = ir.Add(y, y, DataType.INT64, ir.Span.unknown())

# x maps to y consistently in both positions
assert ir.structural_equal(expr1, expr2, enable_auto_mapping=True)
```

**Example 4: Inconsistent Mapping Rejected**

```python
# Build: x + x
expr1 = ir.Add(x, x, DataType.INT64, ir.Span.unknown())

# Build: y + z (different variables)
y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
z = ir.Var("z", ir.ScalarType(DataType.INT64), ir.Span.unknown())
expr2 = ir.Add(y, z, DataType.INT64, ir.Span.unknown())

# Cannot map x to both y and z
assert not ir.structural_equal(expr1, expr2, enable_auto_mapping=True)
```

## structural_hash Function

### Basic Usage

```python
from pypto import DataType, ir

c1 = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())
c2 = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())

# Same value → same hash
assert ir.structural_hash(c1) == ir.structural_hash(c2)
```

### Hash Consistency Guarantee

**Rule**: If `structural_equal(a, b, mode)` is `True`, then `structural_hash(a, mode) == structural_hash(b, mode)`

```python
def build_expr():
    x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
    c = ir.ConstInt(5, DataType.INT64, ir.Span.unknown())
    return ir.Add(x, c, DataType.INT64, ir.Span.unknown())

expr1 = build_expr()
expr2 = build_expr()

if ir.structural_equal(expr1, expr2, enable_auto_mapping=True):
    # Guaranteed: equal hashes
    assert ir.structural_hash(expr1, enable_auto_mapping=True) == \
           ir.structural_hash(expr2, enable_auto_mapping=True)
```

### Using with Containers

```python
class CSEPass:
    """Common Subexpression Elimination."""

    def __init__(self):
        self.expr_cache = {}

    def deduplicate(self, expr):
        hash_val = ir.structural_hash(expr, enable_auto_mapping=False)

        if hash_val in self.expr_cache:
            for cached_expr in self.expr_cache[hash_val]:
                if ir.structural_equal(expr, cached_expr, enable_auto_mapping=False):
                    return cached_expr  # Reuse
            self.expr_cache[hash_val].append(expr)
        else:
            self.expr_cache[hash_val] = [expr]

        return expr
```

## Auto-Mapping Deep Dive

### When to Enable Auto-Mapping

**Enable (`True`)** for:
- Pattern matching regardless of variable names
- Template matching for optimization rules
- Finding structurally identical patterns

**Disable (`False`)** for:
- Exact matching with same variables
- CSE (Common Subexpression Elimination)
- Pointer-based identity tracking

### Variable Mapping Algorithm

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
    if (!EqualType(lhs->GetType(), rhs->GetType())) {
      return false;
    }

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
1. Without auto-mapping: strict pointer comparison
2. With auto-mapping: establish and enforce consistent mapping
3. Type equality checked before mapping
4. Bidirectional maps prevent inconsistent mappings

## Implementation Details

### Hash Combine Algorithm

Uses Boost-inspired algorithm for good distribution:

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
    return reflection::FieldIterator<NodeType, StructuralEqual, decltype(descs)...>::Visit(
      *lhs_op, *rhs_op, *this, descs...);
  }, descriptors);
}
```

**Process:**
1. Get field descriptors via `GetFieldDescriptors()`
2. Use `std::apply` to unpack tuple of descriptors
3. `FieldIterator::Visit` iterates through fields
4. Each field calls appropriate visitor method based on type
5. Results combined with AND logic

### Type Dispatch Mechanism

```cpp
#define EQUAL_DISPATCH(Type)                                       \
  if (auto lhs_##Type = std::dynamic_pointer_cast<const Type>(lhs)) { \
    return EqualWithFields(lhs_##Type, std::static_pointer_cast<const Type>(rhs)); \
  }

// Apply to all IR node types
EQUAL_DISPATCH(ConstInt)
EQUAL_DISPATCH(BinaryExpr)
EQUAL_DISPATCH(AssignStmt)
EQUAL_DISPATCH(Function)
// ... etc
```

## Summary

**Key Takeaways:**

1. **Three Field Types**:
   - `IgnoreField`: Never compared (Span, names)
   - `UsualField`: Compared with user's `enable_auto_mapping`
   - `DefField`: Always uses auto-mapping

2. **Comparison Process**:
   - Fast path checks (reference, null, type)
   - Type dispatch to handlers
   - Reflection-based field iteration
   - Recursive comparison with consistent mapping

3. **Auto-Mapping**:
   - Enable for pattern matching
   - Disable for exact CSE
   - Always consistent: maintains bijective variable mapping

4. **Hash Consistency**:
   - Equal nodes → equal hashes (guaranteed)
   - Use same `enable_auto_mapping` for both functions

For IR node types and construction, see [IR Definition](00-ir_definition.md).
