# 中间表示 (IR) 序列化指南

## 概述

PyPTO IR 序列化提供了基于 MessagePack 的高效序列化机制，具有以下特性：

- **指针共享保留**：同一对象仅序列化一次，引用关系正确恢复
- **往返一致性**：`deserialize(serialize(node))` 与原始节点结构相等
- **可扩展性**：基于字段访问者模式，易于扩展
- **调试信息**：保留 Span（源码位置）

## 快速入门

### Python API

```python
from pypto import ir, DataType

# Create IR
x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
c = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())
expr = ir.Add(x, c, DataType.INT64, ir.Span.unknown())

# Serialize and deserialize
data = ir.serialize(expr)
restored = ir.deserialize(data)
ir.assert_structural_equal(expr, restored, enable_auto_mapping=True)

# File I/O
ir.serialize_to_file(expr, "expr.msgpack")
restored = ir.deserialize_from_file("expr.msgpack")
```

### C++ API

```cpp
#include "pypto/ir/serialization/serializer.h"
#include "pypto/ir/serialization/deserializer.h"

auto x = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::INT64), Span::unknown());
auto expr = std::make_shared<Add>(x, c, DataType::INT64, Span::unknown());

// Serialize/deserialize
auto data = Serialize(expr);
auto restored = Deserialize(data);

// File I/O
SerializeToFile(expr, "expr.msgpack");
auto restored = DeserializeFromFile("expr.msgpack");
```

## 主要特性

### 指针去重

每个唯一对象仅序列化一次，引用关系得以保留：

```python
x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
expr = ir.Add(x, x, DataType.INT64, ir.Span.unknown())

restored = ir.deserialize(ir.serialize(expr))
assert restored.left is restored.right  # Pointer sharing preserved
```

### Kwargs 保留

带有 kwargs 的 Call 表达式 (Expression) 可正确序列化：

```python
original = ir.op.tensor.matmul(a, b, out_dtype=DataType.FP32, a_trans=True)
restored = ir.deserialize(ir.serialize(original))
assert restored.kwargs["out_dtype"] == DataType.FP32.code()
assert restored.kwargs["a_trans"] == True
```

### 内存信息（内存引用 (MemRef) / TileView）

硬件特定的内存分配详情被完整保留：

```python
# Create MemRef and TileView
memref = ir.MemRef()
memref.memory_space_ = ir.MemorySpace.Left
memref.addr_ = ir.ConstInt(0x1000, DataType.INT64, span)
memref.size_ = 512

tile_view = ir.TileView()
tile_view.valid_shape = [ir.ConstInt(16, DataType.INT64, span)] * 2
tile_view.stride = [ir.ConstInt(1, DataType.INT64, span), ir.ConstInt(16, DataType.INT64, span)]

# Create TileType with memory info
tile_type = ir.TileType(shape, DataType.FP16, memref, tile_view)

# Serialize and deserialize
restored = ir.deserialize(ir.serialize(tile_var))
assert restored.type.memref.memory_space_ == ir.MemorySpace.Left
assert len(restored.type.tile_view.valid_shape) == 2
```

## MessagePack 格式

### 节点结构

```javascript
// Full node (first occurrence)
{
  "id": 123,              // Unique ID
  "type": "Add",          // Node type name
  "fields": {             // Field data
    "left": {...},        // Nested or reference
    "right": {...},
    "dtype": 19,          // DataType code
    "span": {...}
  }
}

// Reference to existing node
{"ref": 123}
```

### 特殊类型 (Type)

| 类型 | 格式 | 示例字段 |
| ---- | ---- | -------- |
| **Span** | Map | `filename`, `begin_line`, `begin_column`, `end_line`, `end_column` |
| **ScalarType** | Map | `type_kind: "ScalarType"`, `dtype: 19` |
| **TensorType** | Map | `type_kind`, `dtype`, `shape`, 可选 `memref` |
| **TileType** | Map | `type_kind`, `dtype`, `shape`, 可选 `memref`, 可选 `tile_view` |
| **Op/GlobalVar** | Map | `name`, `is_global_var` |

### MemRef 和 TileView 格式

```javascript
// MemRef (optional field in TensorType/TileType)
{
  "memref": {
    "memory_space": 3,    // uint8: MemorySpace enum
    "addr": {...},        // Expr node
    "size": 512           // uint64
  }
}

// TileView (optional field in TileType)
{
  "tile_view": {
    "valid_shape": [...], // Array of Expr nodes
    "stride": [...],      // Array of Expr nodes
    "start_offset": {...} // Expr node
  }
}
```

### 带 Kwargs 的 Call

```javascript
{
  "type": "Call",
  "fields": {
    "op": {"name": "tensor.matmul", "is_global_var": false},
    "args": [{...}, {...}],
    "kwargs": {
      "out_dtype": 51,    // int
      "a_trans": true,    // bool
      "scale": 1.5        // double
    }
  }
}
```

支持的 kwarg 类型：`int`、`bool`、`double`、`string`

## 架构

### 组件

| 组件 | 职责 |
| ---- | ---- |
| **IRSerializer** | 将 IR 序列化为 MessagePack，通过 `ptr_to_id_` 映射跟踪指针 |
| **IRDeserializer** | 从 MessagePack 反序列化，维护 `id_to_ptr_` 进行指针重建 |
| **TypeRegistry** | 将类型名称映射到反序列化函数，可扩展以支持新的 IR 节点 |
| **FieldSerializerVisitor** | 与字段访问者模式集成，处理所有字段类型 |

### 流程

```text
Serialization:   IR Node → IRSerializer → FieldVisitor → MessagePack bytes
Deserialization: MessagePack bytes → IRDeserializer → TypeRegistry → IR Node
```

## 扩展系统

添加新的 IR 节点类型：

1. **定义节点类**，包含 `GetFieldDescriptors()`：

```cpp
class MyNewNode : public Expr {
 public:
  ExprPtr field1_;
  int field2_;

  MyNewNode(ExprPtr field1, int field2, TypePtr type, Span span)
      : Expr(std::move(span), std::move(type)),
        field1_(std::move(field1)), field2_(field2) {}

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(
      Expr::GetFieldDescriptors(),
      std::make_tuple(
        reflection::UsualField(&MyNewNode::field1_, "field1"),
        reflection::UsualField(&MyNewNode::field2_, "field2")
      )
    );
  }
};
```

1. **添加反序列化器**，位于 `type_deserializers.cpp`：

```cpp
static IRNodePtr DeserializeMyNewNode(const msgpack::object& fields_obj,
                                       msgpack::zone& zone,
                                       IRDeserializer::Impl& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto type = ctx.DeserializeType(GET_FIELD_OBJ("type"), zone);
  auto field1 = std::static_pointer_cast<const Expr>(
    ctx.DeserializeNode(GET_FIELD_OBJ("field1"), zone));
  int field2 = GET_FIELD(int, field2);
  return std::make_shared<MyNewNode>(field1, field2, type, span);
}
```

1. **注册类型**：

```cpp
static TypeRegistrar _my_new_node_registrar("MyNewNode", DeserializeMyNewNode);
```

序列化器通过字段访问者模式自动处理新类型。

## 性能

在现代硬件上的典型性能：

| 操作 | IR 大小 | 时间 | 吞吐量 |
| ---- | ------- | ---- | ------ |
| 序列化小表达式 | 10 节点 | ~10 μs | 1M 节点/秒 |
| 序列化函数 | 100 节点 | ~50 μs | 2M 节点/秒 |
| 序列化程序 | 1000 节点 | ~500 μs | 2M 节点/秒 |
| 反序列化小表达式 | 10 节点 | ~15 μs | 650K 节点/秒 |
| 反序列化函数 | 100 节点 | ~80 μs | 1.25M 节点/秒 |
| 反序列化程序 | 1000 节点 | ~800 μs | 1.25M 节点/秒 |

**复杂度**：对于 N 个唯一节点为 O(N)。内存开销：约为引用表的 2-3 倍节点大小。

**优化措施：**

- 通过 MessagePack 的零拷贝设计实现最少复制
- 使用哈希映射实现 O(1) 指针查找
- 紧凑的二进制格式，比 JSON 更小

## 错误处理

抛出的异常：

| 错误 | 异常 | 上下文 |
| ---- | ---- | ------ |
| 数据损坏 | `DeserializationError` | 附带错误消息 |
| 未知节点类型 | `TypeError` | 附带类型名称 |
| 无效引用 | `DeserializationError` | 缺失 ID |
| 文件 I/O 错误 | `std::runtime_error` | 附带文件路径 |

```python
try:
    node = ir.deserialize(data)
except Exception as e:
    print(f"Deserialization failed: {e}")
```

## 常见问题

**问：为什么使用 MessagePack 而不是 JSON？**
答：更紧凑（二进制），解析更快，更适合机器间通信。

**问：序列化是否保留指针标识？**
答：是的，在单次序列化内保留。不同的 serialize 调用之间，指针是独立的。

**问：可以序列化部分 IR 图吗？**
答：可以，序列化任何 IR 节点，所有被引用的节点会自动包含在内。

**问：MemRef 和 TileView 是否总是被序列化？**
答：不是，它们是可选的。仅在存在时序列化，保持向后兼容性。

**问：没有 MemRef 的旧序列化 IR 会怎样？**
答：旧 IR 可以正常反序列化。MemRef/TileView 字段将为 `None`。

## 相关文档

- [IR 概述](00-overview.md) - IR 节点结构和语义
- [结构比较](03-structural_comparison.md) - 哈希和相等语义
