# IR Serialization Guide

## Overview

PyPTO IR serialization provides efficient MessagePack-based serialization with:

- **Pointer sharing preservation**: Same object serialized once, references restored correctly
- **Roundtrip equality**: `deserialize(serialize(node))` is structurally equal to original
- **Extensibility**: Field visitor pattern for easy extension
- **Debugging info**: Preserves Span (source location)

## Quick Start

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

## Key Features

### Pointer Deduplication

Each unique object serialized once, references preserved:

```python
x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
expr = ir.Add(x, x, DataType.INT64, ir.Span.unknown())

restored = ir.deserialize(ir.serialize(expr))
assert restored.left is restored.right  # Pointer sharing preserved
```

### Kwargs Preservation

Call expressions with kwargs are serialized correctly:

```python
original = ir.op.tensor.matmul(a, b, out_dtype=DataType.FP32, a_trans=True)
restored = ir.deserialize(ir.serialize(original))
assert restored.kwargs["out_dtype"] == DataType.FP32.code()
assert restored.kwargs["a_trans"] == True
```

### Memory Information (MemRef/TileView)

Hardware-specific memory allocation details are fully preserved:

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

## MessagePack Format

### Node Structure

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

### Special Types

| Type | Format | Example Fields |
| ---- | ------ | -------------- |
| **Span** | Map | `filename`, `begin_line`, `begin_column`, `end_line`, `end_column` |
| **ScalarType** | Map | `type_kind: "ScalarType"`, `dtype: 19` |
| **TensorType** | Map | `type_kind`, `dtype`, `shape`, optional `memref` |
| **TileType** | Map | `type_kind`, `dtype`, `shape`, optional `memref`, optional `tile_view` |
| **Op/GlobalVar** | Map | `name`, `is_global_var` |

### MemRef and TileView Format

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

### Call with Kwargs

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

Supported kwarg types: `int`, `bool`, `double`, `string`

## Architecture

### Components

| Component | Responsibility |
| --------- | -------------- |
| **IRSerializer** | Serializes IR to MessagePack, tracks pointers in `ptr_to_id_` map |
| **IRDeserializer** | Deserializes from MessagePack, maintains `id_to_ptr_` for pointer reconstruction |
| **TypeRegistry** | Maps type names to deserializer functions, extensible for new IR nodes |
| **FieldSerializerVisitor** | Integrates with field visitor pattern, handles all field types |

### Flow

```text
Serialization:   IR Node → IRSerializer → FieldVisitor → MessagePack bytes
Deserialization: MessagePack bytes → IRDeserializer → TypeRegistry → IR Node
```

## Extending the System

To add a new IR node type:

1. **Define the node class** with `GetFieldDescriptors()`:

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

1. **Add deserializer** in `type_deserializers.cpp`:

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

1. **Register the type**:

```cpp
static TypeRegistrar _my_new_node_registrar("MyNewNode", DeserializeMyNewNode);
```

The serializer automatically handles new types via field visitor pattern.

## Performance

Typical performance on modern hardware:

| Operation | IR Size | Time | Throughput |
| --------- | ------- | ---- | ---------- |
| Serialize small expr | 10 nodes | ~10 μs | 1M nodes/sec |
| Serialize function | 100 nodes | ~50 μs | 2M nodes/sec |
| Serialize program | 1000 nodes | ~500 μs | 2M nodes/sec |
| Deserialize small expr | 10 nodes | ~15 μs | 650K nodes/sec |
| Deserialize function | 100 nodes | ~80 μs | 1.25M nodes/sec |
| Deserialize program | 1000 nodes | ~800 μs | 1.25M nodes/sec |

**Complexity:** O(N) for N unique nodes. Memory overhead: ~2-3x nodes for reference tables.

**Optimizations:**

- Minimal copies via MessagePack's zero-copy design
- O(1) pointer lookups using hash maps
- Compact binary format smaller than JSON

## Error Handling

Exceptions thrown for:

| Error | Exception | Context |
| ----- | --------- | ------- |
| Corrupt data | `DeserializationError` | With error message |
| Unknown node type | `TypeError` | With type name |
| Invalid references | `DeserializationError` | Missing IDs |
| File I/O errors | `std::runtime_error` | With file path |

```python
try:
    node = ir.deserialize(data)
except Exception as e:
    print(f"Deserialization failed: {e}")
```

## FAQ

**Q: Why MessagePack instead of JSON?**
A: More compact (binary), faster parsing, better for machine-to-machine communication.

**Q: Does serialization preserve pointer identity?**
A: Yes within a single serialization. Between separate serialize calls, pointers are independent.

**Q: Can I serialize partial IR graphs?**
A: Yes, serialize any IR node. All referenced nodes are included automatically.

**Q: Are MemRef and TileView always serialized?**
A: No, they're optional. Only serialized when present, maintaining backward compatibility.

**Q: What happens to old serialized IR without MemRef?**
A: Old IR deserializes without issues. MemRef/TileView fields will be `None`.

## Related Documentation

- [IR Overview](00-overview.md) - IR node structure and semantics
- [Structural Comparison](03-structural_comparison.md) - Hash and equality semantics
