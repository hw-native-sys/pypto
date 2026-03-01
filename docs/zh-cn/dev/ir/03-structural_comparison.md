# PyPTO 结构比较

## 概述

PyPTO 提供两个工具函数，用于按结构而非指针标识比较 IR 节点：

```python
structural_equal(lhs, rhs, enable_auto_mapping=False) -> bool
structural_hash(node, enable_auto_mapping=False) -> int
```

**使用场景：** CSE、IR 优化、模式匹配、测试

**核心特性：** 两个函数都忽略 `Span`（源位置），仅关注逻辑结构。

## 引用相等性与结构相等性

### 引用相等性（默认 `==`）

比较指针地址（O(1)，快速）：

```python
from pypto import DataType, ir

x1 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
x2 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
assert x1 != x2  # Different pointers
```

### 结构相等性

比较内容和结构：

```python
ir.assert_structural_equal(x1, x2, enable_auto_mapping=True)  # True
```

## 比较流程

`structural_equal` 函数遵循以下步骤：

1. **快速路径检查**
   - 引用相等性：如果是同一指针，返回 `true`
   - 空值检查：如果任一为空，返回 `false`
   - 类型检查：比较 `TypeName()` -- 必须完全匹配

2. **类型分发**
   - 变量有特殊处理（自动映射）
   - 其他类型使用基于反射的字段比较

3. **基于字段的递归比较**
   - 通过 `GetFieldDescriptors()` 获取字段描述符
   - 使用反射遍历所有字段
   - 根据类型比较每个字段
   - 用 AND 逻辑合并结果

## 反射与字段类型

反射系统定义三种字段类型：

| 字段类型 | 自动映射 | 是否比较？ | 使用场景 | 效果 |
| -------- | -------- | ---------- | -------- | ---- |
| **IgnoreField** | 不适用 | 否 | 源位置（`Span`）、名称 | 始终视为相等 |
| **UsualField** | 跟随参数设置 | 是 | 操作数、表达式、类型 | 使用当前 `enable_auto_mapping` 进行比较 |
| **DefField** | 始终启用 | 是 | 变量定义、参数 | 始终使用自动映射 |

### 字段定义示例

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

### DefField 的重要性

DefField 表示变量定义。比较定义时，我们关注的是结构位置，而非标识：

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

## structural_equal 函数

### 基本用法

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

### 自动映射行为

| 场景 | enable_auto_mapping=False | enable_auto_mapping=True |
| ---- | ------------------------- | ------------------------ |
| 相同变量指针 | 相等 | 相等 |
| 不同变量指针 | 不相等 | 相等（如果类型匹配） |
| 一致映射（`x + x` vs `y + y`） | 不相等 | 相等 |
| 不一致映射（`x + x` vs `y + z`） | 不相等 | 不相等 |

### 何时启用自动映射

| 使用场景 | 设置 |
| -------- | ---- |
| 不考虑变量名的模式匹配 | `True` |
| 优化规则的模板匹配 | `True` |
| 使用相同变量的精确匹配 | `False` |
| CSE（公共子表达式消除） | `False` |

## structural_hash 函数

### 基本用法

```python
c1 = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())
c2 = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())
assert ir.structural_hash(c1) == ir.structural_hash(c2)
```

### 确定性

`structural_hash` 在单次进程运行中是确定性的。变量标识基于构造时分配的单调递增唯一 ID，而非指针地址，因此相同的构造序列始终产生相同的哈希值。

### 哈希一致性保证

**规则：** 如果 `structural_equal(a, b, mode)` 为 `True`，则 `structural_hash(a, mode) == structural_hash(b, mode)`

### 与容器配合使用

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

## 自动映射算法

实现维护双向映射：

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

**关键要点：**

- 无自动映射时：严格的标识比较（`structural_equal` 使用指针相等性，`structural_hash` 使用唯一 ID）
- 有自动映射时：建立并强制执行一致映射
- 映射前先检查类型相等性
- 双向映射防止不一致的映射

## 实现细节

### 哈希合并算法

使用 Boost 风格的算法：

```cpp
inline uint64_t hash_combine(uint64_t seed, uint64_t value) {
  return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}
```

### 基于反射的字段访问器

无需类型特定代码的通用遍历：

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

## 总结

**关键要点：**

1. **三种字段类型**：
   - `IgnoreField`：从不比较（Span、名称）
   - `UsualField`：使用用户设置的 `enable_auto_mapping` 进行比较
   - `DefField`：始终使用自动映射

2. **自动映射**：
   - 模式匹配时启用
   - 精确 CSE 时禁用
   - 始终保持一致性：维护双射变量映射

3. **哈希一致性**：
   - 相等的节点产生相等的哈希值（有保证）
   - 两个函数使用相同的 `enable_auto_mapping` 设置

关于 IR 节点类型和构造，请参阅 [IR 概述](00-overview.md)。
