# 算子系统

类型 (Type) 安全的算子定义，支持自动类型推导，按模块化分类组织（TensorOp、TileOp、SyncOp、CrossCoreOp）。

## 算子分类

| 分类 | 类型 | 用途 | 文件位置 |
| ---- | ---- | ---- | -------- |
| **TensorOp** | TensorType | 支持广播的 N 维张量 (Tensor) 操作 | `src/ir/op/tensor_ops/` |
| **TileOp** | TileType | 硬件优化的 Tile 操作 | `src/ir/op/tile_ops/` |
| **SyncOp** | UnknownType | 流水线屏障和同步 | `src/ir/op/sync_ops/sync.cpp` |
| **CrossCoreOp** | UnknownType/TileType | AIC↔AIV 跨核通信 | `src/ir/op/sync_ops/cross_core.cpp` |

**主要特性**：流式 API、自动类型推导、kwargs 元数据、NumPy 风格广播、类型提升、动态维度（`kDynamicDim`）

## 类型系统

```cpp
// TensorType: N-dimensional tensors
TensorType(DataType::FP32, {dim1, dim2, dim3, ...})

// TileType: Hardware-optimized tiles
TileType(DataType::FP16, {dim1, dim2})

// Dynamic dimensions (pypto/core/common.h)
constexpr int64_t kDynamicDim = -1;
auto dynamic_dim = make_int(kDynamicDim);
```

| 类型 | 维度 | 用途 | 内存 |
| ---- | ---- | ---- | ---- |
| **TensorType** | N 维 | 通用张量、函数参数/返回值 | DDR（可选 MemRef） |
| **TileType** | N 维 | 统一缓冲区中的硬件优化 Tile | 统一缓冲区（可选 MemRef） |
| **ScalarType** | 0 维 | 标量值 | 寄存器 |
| **UnknownType** | 无 | 无返回值（同步操作） | 无 |

## REGISTER_OP 流式 API

| 方法 | 用途 | 示例 |
| ---- | ---- | ---- |
| `set_op_category(str)` | 算子分类 | `.set_op_category("TensorOp")` |
| `set_description(str)` | 人类可读描述 | `.set_description("Element-wise add")` |
| `add_argument(name, desc)` | 位置 Expr 参数 | `.add_argument("lhs", "Left tensor")` |
| `no_argument()` | 无参数（同步操作） | `.no_argument()` |
| `set_attr<T>(name)` | Kwarg 模式（T: bool, int, DataType 等） | `.set_attr<bool>("a_trans")` |
| `f_deduce_type(fn)` | 类型推导函数 | `.f_deduce_type(DeduceAddType)` |

**类型推导签名：**

```cpp
std::function<TypePtr(const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs)>
```

## C++ 注册示例

### 简单逐元素算子

```cpp
// src/ir/op/tensor_ops/elementwise.cpp
REGISTER_OP("tensor.add")
    .set_op_category("TensorOp")
    .add_argument("lhs", "Left tensor")
    .add_argument("rhs", "Right tensor")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 2);
      auto t1 = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());
      auto t2 = std::dynamic_pointer_cast<const TensorType>(args[1]->GetType());
      auto dtype = PromoteDataTypes(t1->dtype_, t2->dtype_);
      auto shape = BroadcastShapes(t1->shape_, t2->shape_);
      return std::make_shared<TensorType>(shape.shape, *dtype);
    });
```

### 带 Kwargs 的算子

```cpp
// src/ir/op/tensor_ops/matmul.cpp
TypePtr DeduceMatMul(const std::vector<ExprPtr>& args,
                     const std::vector<std::pair<std::string, std::any>>& kwargs) {
  auto lhs = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());
  auto rhs = std::dynamic_pointer_cast<const TensorType>(args[1]->GetType());

  auto get = [&](const std::string& k, bool d) {
    for (const auto& [name, val] : kwargs)
      if (name == k) return std::any_cast<bool>(val);
    return d;
  };

  DataType dtype = [&]() {
    for (const auto& [k, v] : kwargs)
      if (k == "out_dtype") return static_cast<DataType>(std::any_cast<int>(v));
    return *PromoteDataTypes(lhs->dtype_, rhs->dtype_);
  }();

  bool a_t = get("a_trans", false), b_t = get("b_trans", false);
  ExprPtr m = a_t ? lhs->shape_[1] : lhs->shape_[0];
  ExprPtr n = b_t ? rhs->shape_[0] : rhs->shape_[1];
  return std::make_shared<TensorType>(std::vector<ExprPtr>{m, n}, dtype);
}

REGISTER_OP("tensor.matmul")
    .set_op_category("TensorOp")
    .add_argument("lhs", "Left matrix")
    .add_argument("rhs", "Right matrix")
    .set_attr<DataType>("out_dtype")
    .set_attr<bool>("a_trans")
    .set_attr<bool>("b_trans")
    .f_deduce_type(DeduceMatMul);
```

在 tile 层，`tile.batch_matmul` 为 `TileType` 操作数提供批量语义。它接受 rank >= 2 的
tile，广播前导批量维度，并保持与 `tile.matmul` 相同的纯操作数接口风格。如果批量操作数
需要转置语义，可以通过两种等价方式表达：在输入上显式使用 `tile.transpose(...)`，或让
输入来自 `tile.load(..., transpose=True)`。在后续降级到 2D `tile.matmul` 时，这两种
写法都会被统一识别为操作数转置语义。

`tile.batch_matmul_acc(acc, lhs, rhs)` 是批量路径上的累加版本：`acc = acc + lhs @ rhs`，
遵循与 `tile.batch_matmul` 一致的 rank>=2 + batch 广播规则。acc 的 batch 形状必须与
lhs/rhs 广播后的 batch 形状完全一致；matmul 的 (M, N) 必须与 acc 的末两维一致；K 维必须
与 lhs/rhs 内层匹配。累加器的内部 dtype 默认为浮点 → FP32、整型 → INT32（与
`tile.matmul_acc` 对齐）。在 conversion 阶段，`ConvertTensorToTileOps` 会把
`tensor.matmul` / `tensor.matmul_acc` 在任一操作数 rank > 2 时分派到该批量路径；后续由
`FlattenTileNdTo2D` 将其展开为逐 batch 的 2D 操作。

## Python 用法

```python
from pypto.pypto_core import DataType, ir
from pypto.ir import op

span = ir.Span.unknown()
dim4, dim8 = ir.ConstInt(4, DataType.INT32, span), ir.ConstInt(8, DataType.INT32, span)

# Create tensors
tensor_a = ir.Var("a", ir.TensorType([dim4, dim8], DataType.FP32), span)
tensor_b = ir.Var("b", ir.TensorType([dim8], DataType.FP32), span)

# Simple operators
result = op.tensor.add(tensor_a, tensor_b)  # Broadcasting: [4,8] + [8] → [4,8]

# Operators with kwargs
dim64, dim128 = ir.ConstInt(64, DataType.INT32, span), ir.ConstInt(128, DataType.INT32, span)
a = ir.Var("a", ir.TensorType([dim64, dim128], DataType.FP16), span)
b = ir.Var("b", ir.TensorType([dim128, dim64], DataType.FP16), span)
matmul = op.tensor.matmul(a, b, out_dtype=DataType.FP32, a_trans=True)

# Query registry
assert ir.is_op_registered("tensor.add")
op_instance = ir.get_op("tensor.add")
```

## Kwargs（关键字参数）

Call 表达式 (Expression) 将 Expr 参数与元数据参数通过 kwargs 分离。

### Kwargs vs Args vs 属性 (Property)

| - | **Args** | **Kwargs** | **Op 属性** |
| - | -------- | ---------- | ----------- |
| **类型** | `ExprPtr` | `std::any` | 类型擦除 |
| **作用域** | 每次调用 | 每次调用 | 全局 |
| **用途** | 张量、维度、偏移 | `out_dtype`、标志、模式 | 设备、分类 |
| **访问方式** | `call.args_` | `call.kwargs_` | `op.get_attr()` |

### C++ - 读取 Kwargs

```cpp
TypePtr DeduceCastType(const std::vector<ExprPtr>& args,
                       const std::vector<std::pair<std::string, std::any>>& kwargs) {
  auto input = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());

  // Required kwarg
  auto it = kwargs.find("target_type");
  CHECK(it != kwargs.end()) << "tensor.cast requires 'target_type'";
  DataType target = static_cast<DataType>(std::any_cast<int>(it->second));

  // Optional with default
  int mode = 0;
  auto mode_it = kwargs.find("mode");
  if (mode_it != kwargs.end()) mode = std::any_cast<int>(mode_it->second);

  return std::make_shared<TensorType>(input->shape_, target);
}
```

### Python - 使用 Kwargs

```python
result = op.tensor.matmul(a, b, out_dtype=DataType.FP32, a_trans=True)
print(result.kwargs)  # {'out_dtype': 51, 'a_trans': True}
```

## 广播与类型提升

### NumPy 风格广播

维度从右向左对齐：

```text
[4, 8] + [4, 8] → [4, 8]  # Exact match
[4, 8] + [8]    → [4, 8]  # Missing left dimension = 1
[4, 1] + [8]    → [4, 8]  # Size 1 broadcasts
[1, 8] + [4, 8] → [4, 8]  # Size 1 broadcasts
[4, 8] + [5]    → Error   # 8 ≠ 5
```

### 类型提升

标准数值规则：浮点 > 整数，大尺寸 > 小尺寸，有符号 > 无符号（相同大小时）。

```text
INT32 + INT32 → INT32
INT32 + FP32  → FP32   (float precedence)
INT32 + INT64 → INT64  (larger size)
UINT32 + INT32 → INT32 (signed precedence)
```

## TensorOp：N 维张量操作

**用途**：支持完整广播的通用 N 维张量
**类型**：`TensorType`（任意维度）
**位置**：`src/ir/op/tensor_ops/`
**Python API**：`from pypto.ir.op import tensor`

**操作：** `tensor.add/sub/mul/div`（逐元素，支持完整 N 维广播），`tensor.maximum/minimum`（逐元素 max/min；rhs 可为 tensor 或 scalar — `ConvertTensorToTileOps` 根据 rhs 类型分发到 `tile.maximum/minimum` 或 `tile.maximums/minimums`），`tensor.set_validshape`（内部 API，更新 valid_shape 元数据，不搬移数据 — 仅供编译器生成代码使用），`tensor.sort32` / `tensor.mrgsort_format1` / `tensor.mrgsort_format2`（排序；分别对应 `tile.sort32` / `tile.mrgsort` 的 tensor 层接口，由 `ConvertTensorToTileOps` 转换为 tile 操作），`tensor.gather`（按维索引；MVP 仅支持 2D 输入 + `dim=-1`，由 `ConvertTensorToTileOps` 按行展开为 `tile.gather` 循环），`tensor.gather_mask`（掩码模式选择；对应 `tile.gather_mask`，支持可选同位宽 `output_dtype`），`tensor.ci` / `tensor.arange`（生成连续整数序列，下层降到 `tile.ci`；同时通过 `pl.arange` 暴露在顶层 namespace）

**示例：**

```python
from pypto.ir.op import tensor

ib = IRBuilder()
with ib.function("tensor_example") as f:
    input_a = f.param("input_a", ir.TensorType([128, 64, 32], DataType.FP32))
    input_b = f.param("input_b", ir.TensorType([128, 64, 32], DataType.FP32))
    f.return_type(ir.TensorType([128, 64, 32], DataType.FP32))
    result = ib.let("result", tensor.add(input_a, input_b))
    ib.return_stmt(result)
```

## TileOp：硬件优化 Tile 操作

**用途**：带有显式内存管理的硬件优化 Tile 操作
**类型**：`TileType`（统一缓冲区中的 Tile）
**位置**：`src/ir/op/tile_ops/`
**Python API**：`from pypto.ir.op import tile`

**设计**：使用 `TileType`（而非单独的 `BlockType`）以保持一致性。命名空间 `tile.*` + `TileType` 清楚地表示硬件优化的 Tile 操作。

### 操作列表

| 分类 | 操作 | 描述 |
| ---- | ---- | ---- |
| **内存** | `tile.get_block_idx` | 获取 block 索引（返回 UINT64 标量） |
| - | `tile.load` | TensorType → TileType（DDR 到统一缓冲区） |
| - | `tile.store` | TileType → TensorType（统一缓冲区到 DDR） |
| **逐元素** | `tile.add/sub/mul/div` | Tile-Tile 操作 |
| - | `tile.adds/subs/muls/divs` | Tile-Scalar 操作 |
| **一元** | `tile.sqrt` | 逐元素平方根 |
| **变换** | `tile.slice` | 提取子 tile，静态 shape，可选动态 valid_shape |
| - | `tile.extract` | 从 `src` 在 `(index_row, index_col)` 处提取子 tile —— ISA TEXTRACT Variant 1（Mat→Left/Right，Acc→Mat） |
| - | `tile.reshape` | 重塑 tile 维度（元素总数须一致） |
| - | `tile.transpose` | 交换 tile 的两个轴 |
| - | `tile.set_validshape` | 更新 valid_shape 元数据，不搬移数据 |
| - | `tile.ci` | 生成连续整数序列（升序 start+k 或降序 start-k）；dtype ∈ {INT16, INT32}；最内维 != 1 |
| **规约** | `tile.sum` | 沿轴规约（axis, keepdim） |

**数据流：** `TensorType (DDR) → tile.load → TileType (Unified Buffer) → tile.{ops} → TileType → tile.store → TensorType (DDR)`

### 使用示例

```python
from pypto.ir.op import tile

ib = IRBuilder()
with ib.function("tile_computation") as f:
    input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
    input_b = f.param("input_b", ir.TensorType([128, 128], DataType.FP32))
    output = f.param("output", ir.TensorType([128, 1], DataType.FP32))
    f.return_type(ir.TensorType([128, 1], DataType.FP32))

    # Load, compute, reduce, store
    tile_a = ib.let("tile_a", tile.load(input_a, [0, 0], [32, 128]))
    tile_b = ib.let("tile_b", tile.load(input_b, [0, 0], [32, 128]))
    tile_mul = ib.let("tile_mul", tile.mul(tile_a, tile_b))
    tile_sqrt = ib.let("tile_sqrt", tile.sqrt(tile_mul))
    tile_sum = ib.let("tile_sum", tile.sum(tile_sqrt, axis=1, keepdim=True))
    result = ib.let("result", tile.store(tile_sum, [0, 0], output))
    ib.return_stmt(result)
```

## SyncOp：同步操作

**用途**：硬件同步与屏障
**类型**：`UnknownType`（无返回值），在 `EvalStmt` 中使用
**位置**：`src/ir/op/sync_ops/sync.cpp`
**Python API**：`from pypto.ir.op import system`

| 操作 | 描述 | Kwargs |
| ---- | ---- | ------ |
| `system.bar_all` | 全局屏障 | 无 |
| `system.bar_v` | 向量屏障 | 无 |
| `system.bar_m` | 矩阵屏障 | 无 |
| `system.sync_src` | 设置同步标志 | `set_pipe`, `wait_pipe`, `event_id` |
| `system.sync_dst` | 等待同步标志 | `set_pipe`, `wait_pipe`, `event_id` |
| `system.task_invalid` | `PTO2TaskId::invalid()` 哨兵——TaskId carry 的 "暂无 producer" 种子（仅由 `DeriveManualScopeDeps` 合成） | 无 |
| `system.task_id_of` | 提取 kernel-Call producer LHS 背后的 `PTO2TaskId`（仅由 `DeriveManualScopeDeps` 合成） | 无；唯一位置参数是 producer Var |

`system.task_invalid` 与 `system.task_id_of` 返回类型均为 [`ScalarType(DataType::TASK_ID)`](02-types.md#scalartype)。它们不能在 DSL 中由用户调用——只在 [DeriveManualScopeDeps pass](../passes/33-derive_manual_scope_deps.md) 降级 `with pl.manual_scope()` 区域时被合成。源码：`src/ir/op/sync_ops/task.cpp`。

**Python 示例：**

```python
from pypto.ir.op import system
ib.emit(system.bar_all())
ib.emit(system.sync_src(set_pipe=2, wait_pipe=4, event_id=0))
```

**C++ 注册 (`src/ir/op/sync_ops/sync.cpp`)：**

```cpp
REGISTER_OP("system.bar_all")
    .set_op_category("SyncOp")
    .no_argument()
    .f_deduce_type(DeduceUnknownType);

REGISTER_OP("system.sync_src")
    .set_op_category("SyncOp")
    .no_argument()
    .set_attr<int>("set_pipe")
    .set_attr<int>("wait_pipe")
    .set_attr<int>("event_id")
    .no_argument()
    .f_deduce_type(DeduceUnknownType);
```

## CrossCoreOp：AIC↔AIV 跨核通信

**用途**：AIC (Cube) 和 AIV (Vector) 内核之间的跨核数据传输和管道管理
**类型**：`UnknownType`（push/init/buffer/free 操作）或 `TileType` 透传（pop 操作）
**位置**：`src/ir/op/tile_ops/cross_core.cpp`（tpush/tpop）和 `src/ir/op/sync_ops/cross_core.cpp`（tfree/管道初始化/缓冲区）
**Python API**：`import pypto.language as pl`（提升的操作）或 `from pypto.ir.op import tile, system`

### 数据传输操作

| 操作 | 参数 | 描述 | Kwargs |
| ---- | ---- | ---- | ------ |
| `tile.tpush_to_aiv` | 1 (tile) | 从 Cube 推送 tile 到 Vector | `split`，可选 `id` |
| `tile.tpush_to_aic` | 1 (tile) | 从 Vector 推送 tile 到 Cube | `split`，可选 `id` |
| `tile.tpop_from_aic` | 0 | 从 Cube 管道弹出 tile（→ TileType） | `split`，可选 `id` |
| `tile.tpop_from_aiv` | 0 | 从 Vector 管道弹出 tile（→ TileType） | `split`，可选 `id` |
| `system.tfree_to_aic` | 1 (tile) | 向 Cube 生产者释放槽位 | 可选 `id` |
| `system.tfree_to_aiv` | 1 (tile) | 向 Vector 生产者释放槽位 | 可选 `id` |

### 管道初始化操作

| 操作 | 参数 | 描述 | Kwargs |
| ---- | ---- | ---- | ------ |
| `system.aic_initialize_pipe` | 2 | 在 Cube 侧初始化跨核管道（位置参数：`c2v_consumer_buf`、`v2c_consumer_buf`，i32 SSA） | `dir_mask`, `slot_size`，可选 `id` |
| `system.aiv_initialize_pipe` | 2 | 在 Vector 侧初始化跨核管道（位置参数：`c2v_consumer_buf`、`v2c_consumer_buf`，i32 SSA） | `dir_mask`, `slot_size`，可选 `id` |

### 缓冲区管理操作

| 操作 | 参数 | 描述 | Kwargs |
| ---- | ---- | ---- | ------ |
| `system.reserve_buffer` | 0 | 预留跨核通信命名缓冲区（消费者侧） | `name`, `size`, `base`* |
| `system.import_peer_buffer` | 0 | 从同组对等函数导入缓冲区（生产者侧） | `name`, `peer_func` |

\* `base` 默认为 `AUTO (-1)`，由编译器自动分配地址。

### DSL 示例（跨核 V2C 单向）

`dir_mask=2` 仅启用 V2C，因此 C2V 侧缓冲区实参需为未使用方向的占位（`0`、`pl.const(0, pl.INT32)`）；启用侧将 `reserve_buffer` / `import_peer_buffer` 的句柄作为第一个位置实参传入。

```python
import pypto.language as pl

@pl.program
class CrossCoreExample:
    @pl.function(type=pl.FunctionType.InCore)
    def vector_producer(self, a: pl.Tensor[[16, 16], pl.FP16]):
        peer = pl.import_peer_buffer(name="v2c_buf", peer_func="cube_consumer")
        pl.aiv_initialize_pipe(pl.const(0, pl.INT32), peer, dir_mask=2, slot_size=512)

        tile_a: pl.Tile[[16, 16], pl.FP16] = pl.load(a, [0, 0], [16, 16])
        pl.tpush_to_aic(tile_a, split=0)

    @pl.function(type=pl.FunctionType.InCore)
    def cube_consumer(self, out: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
        buf = pl.reserve_buffer(name="v2c_buf", size=4096, base=0x1000)
        pl.aic_initialize_pipe(pl.const(0, pl.INT32), buf, dir_mask=2, slot_size=512)

        received: pl.Tile[[16, 16], pl.FP16] = pl.tpop_from_aiv(split=0)
        pl.tfree_to_aiv(received)
        result: pl.Tensor[[16, 16], pl.FP32] = pl.store(received, [0, 0], out)
        return result
```

参阅 [TPUSH/TPOP ISA 参考](../../reference/pto-isa/01-tpush_tpop.md) 和[缓冲区管理](../../reference/pto-isa/02-buffer_management.md)了解硬件细节。

## 文件组织

| 目录/文件 | 内容 |
| --------- | ---- |
| `src/ir/op/type_inference.cpp` | 共享的类型推断工具 |
| `tensor_ops/elementwise.cpp` | TensorOp: add, sub, mul, div |
| `tile_ops/memory.cpp` | TileOp: load, store, read, get_block_idx |
| `tile_ops/elementwise.cpp` | TileOp: add, mul, div, adds, muls 等 |
| `tile_ops/reduction.cpp` | TileOp: sum（含 axis, keepdim） |
| `tile_ops/unary.cpp` | TileOp: sqrt |
| `sync_ops/sync.cpp` | SyncOp: sync_src, sync_dst, barriers |
| `sync_ops/cross_core.cpp` | CrossCoreOp: tpush, tpop, pipe init, buffers |

**优势**：

- **模块化**：自包含的算子分类
- **构建性能**：修改一个分类不会重新构建其他分类
- **可维护性**：易于定位和修改算子
- **可扩展性**：直接添加新算子

## 添加新操作

1. **选择分类文件**：`src/ir/op/tensor_ops/elementwise.cpp`、`matmul.cpp`、`reduction.cpp`，或 `src/ir/op/tile_ops/memory.cpp`、`unary.cpp`

2. **实现类型推导**：

   ```cpp
   TypePtr DeduceType(const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
     CHECK(args.size() == 2) << "op requires 2 arguments";
     // Validate types, read kwargs, compute output type
     return result_type;
   }
   ```

3. **注册**：

   ```cpp
   REGISTER_OP("tensor.matmul")
       .set_op_category("TensorOp")
       .add_argument("lhs", "Left tensor")
       .add_argument("rhs", "Right tensor")
       .set_attr<DataType>("out_dtype")
       .f_deduce_type(DeduceType);
   ```

4. **Python 封装** (`python/pypto/ir/op/tensor_ops.py`)：

   ```python
   def matmul(lhs: Expr, rhs: Expr, out_dtype=None, a_trans=False) -> Call:
       kwargs = {}
       if out_dtype: kwargs["out_dtype"] = out_dtype.code() if isinstance(out_dtype, DataType) else out_dtype
       if a_trans: kwargs["a_trans"] = a_trans
       return _ir_core.create_op_call("tensor.matmul", [lhs, rhs], kwargs, Span.unknown())
   ```

5. **添加测试**，位于 `tests/ut/ir/`，如需要则更新 `CMakeLists.txt`

## 参考

- 常用常量：`include/pypto/core/common.h`
- 类型定义：`include/pypto/ir/type.h`
- 算子注册表：`include/pypto/ir/op_registry.h`
- 类型推断工具：`include/pypto/ir/type_inference.h`
- 类型推断实现：`src/ir/op/type_inference.cpp`
- 算子注册表实现：`src/ir/op_registry.cpp`
- 张量算子实现：`src/ir/op/tensor_ops/`
- Tile 算子实现：`src/ir/op/tile_ops/`
- 同步算子实现：`src/ir/op/sync_ops/`
