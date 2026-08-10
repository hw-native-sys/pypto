# 算子系统

类型 (Type) 安全的算子定义，支持自动类型推导，按模块化分类组织（TensorOp、TileOp、SyncOp、CrossCoreOp）。

## 算子分类

| 分类 | 类型 | 用途 | 文件位置 |
| ---- | ---- | ---- | -------- |
| **TensorOp** | TensorType | 支持广播的 N 维张量 (Tensor) 操作 | `src/ir/op/tensor_ops/` |
| **TileOp** | TileType | 硬件优化的 Tile 操作 | `src/ir/op/tile_ops/` |
| **SyncOp** | UnknownType（屏障）；ScalarType（task / 启动形状查询） | 流水线屏障、同步、TaskId 与 SPMD 启动形状查询 | `src/ir/op/sync_ops/` |
| **CrossCoreOp** | UnknownType/TileType | AIC↔AIV 跨核通信 | `src/ir/op/sync_ops/cross_core.cpp` |
| **PrefetchOp** | 不透明句柄 (opaque handle) | GM→L2 异步预取 | `src/ir/op/prefetch/prefetch_async.cpp` |

**主要特性**：流式 API、自动类型推导、kwargs 元数据、NumPy 风格广播、类型提升、动态维度（`kDynamicDim`）

## 类型系统

```cpp
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

对于二维 `tile.matmul`，物理装箱后的 K 维必须一致。PTO 从 lhs 的有效 K 推导收缩范围，
因此该范围可以小于 rhs 的有效 K，但必须被后者包含。`tile.matmul_acc` 同样要求物理
M/N/K 装箱严格兼容，同时允许累加器的有效 M/N 矩形以及 rhs 的有效 K 包含 PTO 根据
lhs M/K 与 rhs N 实际计算的较小矩形。

在 tile 层，`tile.batch_matmul` 为 `TileType` 操作数提供批量语义。它接受 rank >= 2 的
tile，广播前导批量维度，并保持与 `tile.matmul` 相同的纯操作数接口风格。如果批量操作数
需要转置语义，可以通过两种等价方式表达：在输入上显式使用 `tile.transpose(...)`，或在
自然 `tile.load` 上叠加零拷贝 `tile.transpose_view(...)`。在后续降级到 2D `tile.matmul`
时，这两种写法都会被统一识别为操作数转置语义。

`tile.batch_matmul_acc(acc, lhs, rhs)` 是批量路径上的累加版本：`acc = acc + lhs @ rhs`，
遵循与 `tile.batch_matmul` 一致的 rank>=2 + batch 广播规则。acc 的 batch 形状必须与
lhs/rhs 广播后的 batch 形状完全一致；matmul 的 (M, N) 必须与 acc 的末两维一致；K 维必须
与 lhs/rhs 内层匹配。累加器的内部 dtype 默认为浮点 → FP32、整型 → INT32（与
`tile.matmul_acc` 对齐）。在 conversion 阶段，`ConvertTensorToTileOps` 会把
`tensor.matmul` / `tensor.matmul_acc` 在任一操作数 rank > 2 时分派到该批量路径；后续由
`FlattenTileNdTo2D` 将其展开为逐 batch 的 2D 操作。

### MX block-scale matmul（Ascend950）

MX 使用独立的 `LeftScale` / `RightScale` 内存空间与 `FP8E8M0` scale
dtype。PyPTO 在 Ascend950 上通过 `matmul_mx` 算子族支持 host-prequant MXFP8 路径。
`InsertMxScaleAddr`（在 `InferTileMemorySpace` 之后）在操作数内存空间解析完成后插入内部 `tile.tget_scale_addr` 绑定。

| IR / DSL | 说明 |
| -------- | ---- |
| `tile.load` 读取 `pl.Tensor[..., pl.MX_A_ZZ \| pl.MX_B_NN]` | 源 TensorLayout 携带 MX scale GM layout。dtype 为 FP8E8M0 或 UINT8，必须指定 `target_memory=Mat`，且不支持 strided source。 |
| `tile.move(..., target_memory=LeftScale/RightScale)` | Mat→Scale move；硬件 layout 固定为左侧 row/row/32、右侧 col/col/32，源 Mat tile 与 layout override 必须完全匹配。 |
| `tile.create(..., target_memory=LeftScale/RightScale)` | 不支持；应先把 MX scale 数据加载到 Mat，再 move 到 scale 内存。 |
| `tile.matmul_mx` / `pl.matmul_mx` | `Left, LeftScale, Right, RightScale → Acc`；data 仅支持 `FP8E4M3FN`，scale 为 `FP8E8M0`；physical `M % 16 == 0`、`K % 64 == 0`、`N % 32 == 0`；valid K 必须满足 `ceil(validK/32) == ceil(physicalK/32)`。对齐与 scale-group 数值检查仅作用于常量维；符号维跳过数值校验，回退到声明的 scale tile 几何（后续仍由 PTOAS 验证）。 |
| `tile.matmul_mx_acc` / `pl.matmul_mx_acc` | `Acc, Left, LeftScale, Right, RightScale → Acc`；通过 `set_output_reuses_input(0)` 原地执行；accumulator 的 physical/valid M、N 必须与 matmul 输出一致。 |
| `tile.matmul_mx_bias` / `pl.matmul_mx_bias` | `Left, LeftScale, Right, RightScale, Bias → Acc`；bias 为 `[1, N]` FP32。 |
| `tile.tget_scale_addr` | 编译器生成的 A5 绑定，接受 `LeftScale↔Left` 或 `RightScale↔Right`；对 `dst_scale` 原地 DPS。用户只编写 `matmul_mx` 算子族。 |

规范样例：`M=128,K=64,N=64`，A/B=`FP8E4M3FN`，scale=`FP8E8M0`（`[128,2]` / `[2,64]`），
GM scale layout `mx_a_zz` / `mx_b_nn`（host ZZ/NN pack）；对齐 M↑16、K↑64、N↑32（fp8）。

MX tensor subview 是当前遗留限制。由于硬件路径无法表达 subview base
offset，`tensor.slice`、`tensor.reshape`、`tensor.transpose`、
`tensor.reinterpret_view` 和 `tensor.view` 均拒绝 MX-layout source。
在完整的 scale layout contract 实现前，`pld.tile.remote_load` 也拒绝 MX layout。

#### MX / Ascend950：pto-isa 约束

| 约束 | 要点 |
| ---- | ---- |
| 独立 scale buffer | Cube **不**把 scale 折进 Left/Right data；`TileType::ScaleLeft` / `ScaleRight`（L0A/L0B sidecar）↔ PyPTO `LeftScale` / `RightScale` |
| payload | scale 为 `float8_e8m0_t` / `FP8E8M0`；本阶段 MX data **仅 `FP8E4M3FN`**（**拒绝 `FP8E5M2`**）；physical `K%64==0`，scale 组数 `ceil(K/32)`，fractal=32 |
| layout | `mx_a_zz` → row-major ZZ；`mx_b_nn` → col-major NN；`TLoadMxCube*`（AZZ2ZZ 等） |
| `TMov` `CommonCheckMX` | 允许 `uint8_t` Mat → `float8_e8m0` ScaleLeft/Right；canonical：ui8 Mat reshape 再 ui8→f8 Scale |
| bind-then-fill | **先** `GetScaleAddr(Left/Right)` 再填 sidecar；写 provisional alloc 地址在 rebound 后无效 |
| 对齐 | 与 ISA `tmatmul_mx` 一致：physical `M%16==0`、`K%64==0`、`N%32==0`（fp8）；`DeduceTileMatMulMxType` **仅对常量维**强制；符号维跳过数值检查 |

#### MX / Ascend950：PTOAS 约束

| 约束 | 要点 |
| ---- | ---- |
| 单一 `loc=scaling` | 尚无独立 left/right_scale loc；PyPTO 两侧都降到 `loc=scaling`，EmitC 再选 ScaleLeft/Right |
| dtype 必须 `!pto.f8E8M0` | `ui8`+`scaling` 会错成 Fixpipe `TileType::Scaling`；进 Scale 前需提升为 FP8E8M0 |
| 禁止 Mat↔Scaling `treshape` | 不同 loc；reshape 留在 Mat（ui8），再 `tmov` 进 scaling |
| shape-matched Mat→Scale `tmov` | flat `[1,G]` 须先 `treshape` 到 `[M,K/32]`（或 B 侧 shape） |
| 顺序 | PyPTO 按源序发 Mat→scaling `tmov`；PTOAS `PTOA5NormalizeTMovPass` 把 `tget_scale_addr` 重排到它前面（ISA bind-before-fill） |
| `#pto.layout` / mx load | `mx_a_zz` / `mx_b_nn` / …；本阶段用 **host ZZ/NN**（AZZ2ZZ） |
| 本阶段覆盖 | `pto.tmatmul.mx` / `.acc` / `.bias` + `pto.tget_scale_addr` |

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

  // `kwargs` is a vector of pairs, not a map — scan it to look a key up.
  auto find_kwarg = [&kwargs](const std::string& key) {
    return std::find_if(kwargs.begin(), kwargs.end(),
                        [&key](const auto& kv) { return kv.first == key; });
  };

  // Required kwargs — `cast` declares both `target_type` and `mode`, and codegen
  // reads `mode` unconditionally, so a missing one must fail here rather than
  // silently default to round_mode NONE.
  auto it = find_kwarg("target_type");
  CHECK(it != kwargs.end()) << "tensor.cast requires 'target_type'";
  DataType target = static_cast<DataType>(std::any_cast<int>(it->second));

  CHECK(find_kwarg("mode") != kwargs.end()) << "tensor.cast requires 'mode'";

  return std::make_shared<TensorType>(input->shape_, target);
}
```

真正可选的 kwarg（codegen 读取时带回退值，例如 `tile.log` 的 `high_precision`）应使用
`Call::GetKwarg<T>(key, default_value)` 读取，而不是 `CHECK`——参见 `include/pypto/ir/expr.h`。

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

**操作：** `tensor.add/sub/mul/div`（逐元素，支持完整 N 维广播），`tensor.maximum/minimum`（逐元素 max/min；rhs 可为 tensor 或 scalar — `ConvertTensorToTileOps` 根据 rhs 类型分发到 `tile.maximum/minimum` 或 `tile.maximums/minimums`），`tensor.set_validshape`（内部 API，更新 valid_shape 元数据，不搬移数据 — 仅供编译器生成代码使用），`tensor.sort32` / `tensor.mrgsort_format1` / `tensor.mrgsort_format2`（排序；分别对应 `tile.sort32` / `tile.mrgsort` 的 tensor 层接口，由 `ConvertTensorToTileOps` 转换为 tile 操作），`tensor.gather`（按维索引；MVP 仅支持 2D 输入 + `dim=-1`，由 `ConvertTensorToTileOps` 按后端分策略下降 —— A5（Ascend950）将末维 gather 展开为对扁平元素偏移 `flat[i, j] = i * src_cols + index[i, j]` 的单次整块 `tile.gather`，并在此之前把带 stride 的 tile 源（如 `tile.slice` 视图）物化为连续 tile，使扁平索引能正确寻址；A2A3（Ascend910B）保留 legacy 的按行 `tile.gather` 循环，此时每个单行切片内的列索引即等于扁平索引），`tensor.gather_mask`（掩码模式选择；对应 `tile.gather_mask`，支持可选同位宽 `output_dtype`；见[掩码模式](#掩码模式)），`tensor.scatter`（按列散布；`tensor.gather` 的按列逆操作，MVP 仅支持 2D 输入 + `dim=-1` —— `out[b, index[b, k]] = src[b, k]`，`index` 与 `src` 同形状 —— 由 `ConvertTensorToTileOps` 下降到 `tile.scatter`），`tensor.scatter_mask`（按掩码模式散布；对应 `tile.scatter_mask`，将紧凑 `input` 按掩码扩展到 `dst` 的对应列 —— 见[掩码模式](#掩码模式)），`tensor.ci` / `tensor.arange`（生成连续整数序列，下层降到 `tile.ci`；同时通过 `pl.arange` 暴露在顶层 namespace），`tensor.and/ands/or/ors/xor/xors/not/shl/shls/shr/shrs`（仅整数的位运算与移位。此处列出的是注册的 *IR* 名称；其中名字本身是 Python 关键字的三个，其 Python 拼写带尾部下划线 —— `tensor.and_`、`tensor.or_`、`tensor.not_` —— printer 也按该形式输出，以保证 IR 能往返为合法 Python；对应同名 `tile.*` 操作。张量-张量形式的两个操作数形状必须相同 —— 硬件没有 `tile.row_expand_and`，因此广播在类型推导阶段即被拒绝，而不是延迟到 pass 中失败。`tensor.not` 仅支持 int16/uint16，与 `tile.not`/TNOT 一致。移位保持 lhs 的元素类型；`and`/`or`/`xor` 按整数位宽提升，与其 tile 版本行为一致。`ConvertTensorToTileOps` 将其中九个 1:1 下降，并为 `tensor.xor`/`tensor.xors` 合成 `pto.txor` 所需的临时操作数，使 tensor 层调用者无需提供 `tmp`）

`tensor.view` 是只修改元数据的零拷贝 shape/layout 重新解释操作。它注册为 `TensorOp`，并在 `ConvertTensorToTileOps` 中作为 passthrough 处理；PTO in-core codegen 会将其降级为基于原始 base pointer 的 `pto.make_tensor_view`。目标 rank 至少为 1（DN 至少为 2）；编排层仅支持 ND shape 重新解释，且不能同时改变 layout。对部分有效的源张量进行 shape 重新解释时，仅支持把 packed ND 的 leading dimensions 折叠为 2D，或把连续前缀线性折叠为 `[1, product(shape)]`；两种形式都必须显式提供目标 `valid_shape`，并会保留源张量类型及其底层元数据。

`pl.reinterpret_view(data, dtype, *, shape=None)` 会根据输入分派到等价的 `pl.tensor` 或 `pl.tile` 算子，并保持返回类型种类不变。它是覆盖完全相同字节的零拷贝视图，因此 `dtype` 必须不同，且仅支持有/无符号 8/16/32/64 位整数、FP16、BF16 与 FP32。省略 `shape` 时，ND/row-major 缩放最后一轴，DN/col-major 按源/目标字节宽度比例缩放倒数第二轴。显式 shape 必须字节数相等；除非能证明它与自动推导 shape 等价，否则必须完全静态。部分有效的 `valid_shape` 只能使用与自动推导结果等价的 shape。零值/null padding 元数据会保留，依赖 dtype 的 max/min padding 则会清除。初始可执行路径支持 packed ND in-core tensor 及 packed、flat（`none_box`）row/col-major tile；DN tensor 可做类型推导但 Tensor-to-Tile 下降会拒绝，编排层 tensor 暂不支持。

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
| - | `tile.move` | 在 memory space 之间搬移 tile（`target_memory`）—— 见 [tile.move 的结果 view](#tilemove-的结果-view) |
| **逐元素** | `tile.add/sub/mul/div` | Tile-Tile 操作 |
| - | `tile.adds/subs/muls/divs` | Tile-Scalar 操作。**常量**标量操作数会采用 tile 的元素 dtype（裸整数字面量否则会被解析为 `index`，而任何 `pto.t*s` 算子都不接受它）——但整数 tile 上的浮点字面量仍保持 FP32，以保留类型提升语义。显式的 `pl.const(v, dtype)` 属于用户的有意标注，与任何非常量表达式一样保持不变；非常量的 `index` 标量（循环变量、`pl.dim`）会被拒绝——需用 `pl.cast` 转换。`tensor.*s` 同理。 |
| **一元** | `tile.sqrt` | 逐元素平方根 |
| **变换** | `tile.slice` | 提取子 tile，静态 shape，可选动态 valid_shape |
| - | `tile.extract` | 从 `src` 在 `(index_row, index_col)` 处提取子 tile —— ISA TEXTRACT Variant 1（Mat→Left/Right，Acc→Mat）。结果 layout 取自 `target_memory` 的隐式 view；`Left`/`Right` 例外，使用 TEXTRACT 侧的 L0 格式（与 `tile.move` 的 TMOV 侧不同） |
| - | `tile.reshape` | 重塑 tile 维度（元素总数须一致）。会把源的 `valid_shape` 带到结果上，且绝不扩大 —— 见[reshape 与有效区域（valid region）](#reshape-与有效区域valid-region) |
| - | `tile.reinterpret_view` | 以不同 dtype 对完全相同的字节做零拷贝视图；可选 shape 默认按 layout 推导（仅支持紧密、非分形 tile） |
| - | `tile.transpose` | 交换 tile 的两个轴 |
| - | `tile.set_validshape` | 更新 valid_shape 元数据，不搬移数据 |
| - | `tile.ci` | 生成连续整数序列（升序 start+k 或降序 start-k）；dtype ∈ {INT16, INT32}；最内维 != 1 |
| - | `tile.tri` | 使用 INT32 diagonal offset 生成上三角或下三角 0/1 mask；支持可选的部分 `valid_shape`；映射为 `pto.ttri`。 |
| **规约** | `tile.row_*` / `tile.col_*` | 方向特定的规约（`row_sum`/`row_max`/`row_min`/`row_prod` 折叠最后一轴；`col_*` 折叠第 0 轴）。不存在以 axis 参数化的规约算子 —— ISA 只提供方向特定的指令（`pto.trowsum`、`pto.tcolsum` 等） |
| **聚集** | `tile.gatherb` | 按 32-byte 源块聚集。每个 UINT32 offset 选择一个块；每个 offset 列扩展为 `32 / sizeof(output_dtype)` 个输出元素，valid_shape 同比例扩展。`output_dtype` 默认等于源 dtype，也可选择另一种受支持的字节解释。offset 每行须包含正整数个 8-entry 组。切片源的字节地址必须能被证明为 32-byte 对齐；动态列偏移会被拒绝，而物理行跨度保持对齐时允许动态行偏移。映射为 `pto.tgatherb`。 |
| - | `tile.mgather` | 从 GM tensor 聚集到新 Vec 或 Mat tile。Vec 输出使用 INT32 index tile（`[1,R]`，A5 也支持 `[R,1]`）；Mat 输出使用 ND-layout GM source 与 INT32 index tensor，并采用规范 NZ layout，物理行数按 16 对齐、列数按 `C0 = 32 / sizeof(dtype)` 对齐；可通过较小的二维 `valid_shape` 表达 padding tail。`coalesce="row"` 聚集整行；`"elem"` 按扁平元素索引聚集，且 Mat 输出要求同 dtype、连续 ND、元素数不少于物理输出的 GM `scratch` tensor。`gather_oob` 可选择 `undefined`、`clamp`、`wrap` 或 `zero`。payload dtype 支持 I8/U8/I16/U16/I32/U32/FP16/BF16/FP32，以及仅 A5 支持的 FP8E4M3FN/FP8E5M2/HF8。 |
| **散布** | `tile.scatter` | 按行索引把 `src` 散布到 `dst`（`pto.tscatter` 索引形式；DPS：`dst` 为 in/out，结果别名为 `dst`）。`src` / `dst` dtype ∈ {I8, I16, I32, FP16, FP32, BF16}；`indexes` dtype ∈ {I16, I32}；元素宽度匹配规则：4 字节 dst ↔ INT32，2 字节 dst ↔ INT16，1 字节 dst ↔ INT16。 |
| - | `tile.scatter_mask` | 按掩码模式把 `src` 行写入 `dst` 中由掩码选中的列（DPS：`dst` 为 in/out）。这是 PyPTO codegen 层形式，下降为 `pto.tscatter` 掩码发射 —— **并非**独立的 pto-isa 指令（与 `tile.gather_mask` 不同）。掩码语义见[掩码模式](#掩码模式)。 |

`tile.reshape` 保持 dtype、元素总数以及源的有效区域（见下）；`tile.reinterpret_view(data, dtype, *, shape=None)` 改变 dtype，但要求前后总字节数完全相同。省略 `shape` 时，它会根据源/目标 dtype 字节宽度和 tile layout 缩放物理连续轴。在 PTOAS 内存规划下，无论 shape 是否变化，都会下降为保持别名关系的 PTO `treshape` 原语。

### tile.move 的结果 view

推导出的结果 `TileView` 按字段分别取值：

| 字段 | 结果值的来源 |
| ---- | ------------ |
| `blayout` / `slayout` | 凡目标 space 自带 layout（`Mat`、`Acc`、`Left`、`Right`、`LeftScale`、`RightScale`），取**目标**的 implicit layout；扁平 space（`Vec`、`Bias` 等）则沿用源 tile 的 effective layout。两者都可由 `blayout` / `slayout` kwarg 覆盖 |
| `fractal` | **目标** space 的分块（boxing）粒度，绝不取源的：`Acc`（L0C，NZ 分形）为 1024，MX scale tile 为 32，其余为 512 |
| `valid_shape` / `pad` | 从源带过来 |
| `stride` / `start_offset` | 丢弃 —— 目标是稠密缓冲区 |

layout 来自目标，因为它描述的是目标缓冲区如何分块，由
`tile_view_semantics::GetImplicitTileLayout` 提供。`Right` 仍需就地覆盖：L0B 要求
`blayout=row_major`，而 `[N, 1]` 形状的 implicit `blayout` 是 `col_major`。

`tile.move` 自己把目标 `memory_space` 打到推导出的类型上（参见
[类型](02-types.md#tiletype) 中的 `TileType` 契约），因此当结果 view 与目标 space 的
implicit view 一致时会折叠为 `nullopt` —— 这与
[`InferTileMemorySpace`](../passes/18-infer_tile_memory_space.md) 为重新定型的 tile
刷新的 per-space implicit view 是同一套。

### reshape 与有效区域（valid region）

reshape 是零拷贝视图，无法凭空产生数据：`tensor.reshape` 与 `tile.reshape` 共用
同一条规则，把源的 `valid_shape` 映射到目标 shape，且绝不扩大。有效区域只能表示为
以原点为锚的矩形框，因此并非所有源区域都能在重新切分后保留：

| 源区域 | 结果 |
| ------ | ---- |
| 完全有效 | `new_shape` —— 会被规范化掉，不产生 view，已有程序不受影响 |
| 可证明为空 | 全零矩形框 |
| 仅增删完全有效的单位轴 | 保留的轴按 1:1 映射，可精确保留任意矩形 |
| 连续的扁平前缀 | `new_shape` 中覆盖同一批元素的矩形（若存在） |
| 其他情况 | **拒绝** —— `valid_shape` 无法描述 reshape 后的区域 |

因此 `[8, 16]` valid `[5, 16]`（80 个元素的扁平前缀）可映射为 `[16, 8]` valid
`[10, 8]` 或 `[128]` valid `[80]`，而 `[4, 32]` 会被拒绝 —— 80 个元素不是整数行
（每行 32）。`[1, 8, 16]` valid `[1, 8, 5]` 根本不是扁平前缀，但映射到 `[8, 16]`
valid `[8, 5]` 是精确的，因为丢弃完全有效的单位轴不改变行列关系。
`tensor.reshape` 可选的第三个 `valid_shape` 操作数只能*收窄*推导出的区域，
不能声称拥有该区域之外的数据。

**数据流：** `TensorType (DDR) → tile.load → TileType (Unified Buffer) → tile.{ops} → TileType → tile.store → TensorType (DDR)`

### 掩码模式

`*.gather_mask` / `*.scatter_mask` 使用编译期 `MaskPattern`（`pl.tile.MaskPattern`，整数取值 1–7，与硬件 `VREDUCEv2` 的 pattern mode 一致）按行标记列的一个子集（模式名**从右往左**读，最右位对应列 0）。同一标记集合驱动两个算子做**相反方向**的操作。**`gather_mask`** *选择并紧凑*：从宽输入中读取被标记的列，紧凑写入较窄输出的前若干列（`out_cols = cols / stride`）；这是真实的 pto-isa 指令（`pto.tgather` 掩码形式），A2/A3 **与 A5** 均支持。**`scatter_mask`** *放置并扩展*：把紧凑输入写入更宽 `dst` 的被标记列（`dst_cols = cols * stride`），未标记列保留 `dst` 原值（DPS）；这是 **PyPTO codegen 层形式，并非独立的 pto-isa 指令** —— 不存在 `pto.tscatter` 掩码指令（与 gather 不同）—— PyPTO 为 A2/A3 / CPU-sim 类下降路径发射它。例如对 `[a0 a1 a2 a3 a4 a5 a6 a7]`：gather `P0101 → [a0 a2 a4 a6]`；对 `[s0 s1 s2 s3]` 做 scatter `P0101 → [s0 · s1 · s2 · s3 ·]`（`·` 表示保留的 `dst`）。

| 模式 | 整数 | 标记列 `c` 的条件 | 被标记的列 | 步长 |
| ---- | ---- | ----------------- | ---------- | ---- |
| `P0101` | 1 | `c % 2 == 0` | 0, 2, 4, … | 2 |
| `P1010` | 2 | `c % 2 == 1` | 1, 3, 5, … | 2 |
| `P0001` | 3 | `c % 4 == 0` | 0, 4, 8, … | 4 |
| `P0010` | 4 | `c % 4 == 1` | 1, 5, 9, … | 4 |
| `P0100` | 5 | `c % 4 == 2` | 2, 6, 10, … | 4 |
| `P1000` | 6 | `c % 4 == 3` | 3, 7, 11, … | 4 |
| `P1111` | 7 | 全选 | 全部 | 1 |

最后一维须能被步长整除。`gather_mask` 另接受可选的同位宽 `output_dtype`（按位重解释，而非数值转换）。参考：gather 的选择语义见 `pto-isa` 的 `MaskSelect`（`include/pto/cpu/TGather.hpp`）；pypto 类型推导见 `src/ir/op/tile_ops/gather.cpp`（gather）/ `src/ir/op/tile_ops/scatter.cpp`（scatter）。

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
    # row_sum 折叠最后一轴 -> [32, 1]。scratch tile 必须与输入 dtype 和 rank 相同，
    # 且每一维都不小于输入的对应维度。
    tmp_tile = ib.let("tmp_tile", tile.create([32, 128], DataType.FP32))
    tile_sum = ib.let("tile_sum", tile.row_sum(tile_sqrt, tmp_tile))
    result = ib.let("result", tile.store(tile_sum, [0, 0], output))
    ib.return_stmt(result)
```

## SyncOp：同步操作

**用途**：硬件同步与屏障，以及共用 `system.` 命名空间的 TaskId 与 SPMD 启动形状查询
**类型**：屏障类为 `UnknownType`（无返回值，在 `EvalStmt` 中使用）；查询类为 `ScalarType`，会绑定一个值（`task_invalid`、`task_is_valid`、`available_cluster_count`、`available_aiv_count`）
**位置**：`src/ir/op/sync_ops/` —— `sync.cpp`（屏障）、`task.cpp`（TaskId）、`launch.cpp`（启动形状查询）
**Python API**：`from pypto.ir.op import system`

| 操作 | 描述 | Kwargs |
| ---- | ---- | ------ |
| `system.bar_all` | 全局屏障（下降为 `pto.barrier <PIPE_ALL>`） | 无 |
| `system.bar_v` | 向量屏障（下降为 `pto.barrier <PIPE_V>`） | 无 |
| `system.bar_m` | 矩阵屏障（下降为 `pto.barrier <PIPE_M>`） | 无 |
| `system.fence` | 全局内存屏障（下降为 `pto.fence.barrier_all #pto.fence_scope<gm>`） | 无 |
| `system.cacheinvalid` | 使 tensor 某个子区域对应的 cache line 失效。参数：`tensor`、`shapes`（N 维）、`offsets`（N 维）。任意区域大小（包括单个元素）都下降为 `pto.partition_view` + `pto.cmo.cacheinvalid %payload_view single_cache_line : !pto.partition_tensor_view<...>` | 无 |
| `system.syncall` | 跨核全员屏障（`pto::SYNCALL`）。`mode="hard"`（FFTS，无 operand）或 `mode="soft"`（GM 轮询，带 operand） | `core_type`（`"aiv_only"` \| `"aic_only"` \| `"mix"`）、`mode`（`"hard"` \| `"soft"`） |
| `system.sync_src` | 设置同步标志 | `set_pipe`, `wait_pipe`, `event_id` |
| `system.sync_dst` | 等待同步标志 | `set_pipe`, `wait_pipe`, `event_id` |
| `system.task_invalid` | `PTO2TaskId::invalid()` 哨兵——TaskId carry 的 "暂无 producer" 种子 | 无 |
| `system.task_is_valid` | 测试某个 `TASK_ID` 值是否为有效（非哨兵）handle | 无；唯一位置参数是 TaskId Var |
| `system.available_cluster_count` | 本次运行的 MIX cluster（= AIC）数，由设备读回。结果为 `Scalar[INT32]` | 无 |
| `system.available_aiv_count` | 本次运行的独立 AIV 核数，由设备读回。结果为 `Scalar[INT32]` | 无 |

`system.syncall` 有两种 mode。**hard** 形态（`mode="hard"`，默认）下沉为 FFTS 屏障，等待所选 `core_type` 的**全部**物理核到达；kernel 必须以满占用方式启动（每个物理核一个 block）**且带 `sync_start=True`**（使所有 block 同时驻留——非 sync_start 启动可能分波次派发 block 而使屏障死锁），否则屏障死锁（AICore 错误 507018）。**soft** 形态（`mode="soft"`）轮询一段共享 GM workspace，因此可在**部分**占用下工作。`gm_workspace` 是共享、清零的 GM `INT32` tensor，含 `used_cores * 8` 个 slot（请作为 kernel 参数传入，使所有 block 共享同一缓冲）；暂存 tile 由编译器合成；`used_cores` 是参与核数。soft 形态对每种 `core_type` 都支持，operand 随参与核集合而不同：

- `aiv_only`：`[gm_workspace, ub_scratch, used_cores]` —— 一个 UB（Vec）暂存 tile。
- `aic_only`：`[gm_workspace, l1_scratch, used_cores]` —— 一个扁平 L1（Mat，`slayout=none_box`）暂存 tile。
- `mix`：`[gm_workspace, ub_scratch, l1_scratch, used_cores]` —— UB 与扁平 L1 各一个。该屏障汇合 AIC + AIV 核，故 `used_cores` 是**总**参与数（AIC block 数 + AIV subblock 数）。该 op 会被复制到 cube 与 vector 两条流上，每条流各用自己的 tile（另一个在该流上是死代码），与 pto-isa 的 soft-mix 下沉一致。

扁平 L1 暂存 tile 通过 `pl.tile.create(..., target_memory=pl.Mem.Mat, flat_layout=True)` 创建，保持连续的 `slayout=none_box` 布局（普通的 boxed NZ Mat tile 会错位 8 个 int32 计数槽）。

统一的 `mode=` 关键字 API（`mode="hard"` / `mode="soft"`）是 **DSL** 层接口（`pl.system.syncall`）。`pypto.ir.op.system` 下的 Python IR 辅助函数则是拆开的：`syncall(core_type=...)` 构造 hard 形态，`syncall_soft(core_type, args)` 构造 soft 形态。

`system.available_cluster_count` / `system.available_aiv_count` 是 SPMD **启动形状查询**：把它作为 `pl.spmd(...)` 的 `core_num` 传入，启动宽度即按本次运行落到的设备自适应。Orchestration codegen 分别下沉为 `rt_available_cluster_count()` / `rt_available_aiv_count()`。混合（AIC+AIV）或纯 cube kernel 用 cluster 数（每个 core-group 一个 block），纯 vector kernel 用 AIV 数。这是唯一能跨设备保持满占用的启动宽度，而 hard `system.syncall` 正需要满占用；`HardSyncallOccupancy` verifier 对这类宽度不再做数量比较，并会拒绝用错核类型的查询。请把调用内联传入（`pl.spmd(pl.system.available_cluster_count())`），不要先绑定到变量名——变量名会以「定义在调用方的变量」形式落到外提出的 `Spmd` 包装函数上，IR printer 无法重新解析。源码：`src/ir/op/sync_ops/launch.cpp`。

`system.task_invalid` 返回类型为 [`ScalarType(DataType::TASK_ID)`](02-types.md#scalartype)。当 Python 字面量 `None` 出现在 TaskId 位置（`deps=[None]` 条目或 TaskId 循环 iter_arg 种子）时，它就是 `None` 在 `with pl.manual_scope():` 区域内的下沉目标。不存在 `system.task_id_of` op —— producer task id 由 `pl.submit(...)` parser construct 返回的二元组第二个元素获得，而非来自 builtin。源码：`src/ir/op/sync_ops/task.cpp`。

## CrossCoreOp：AIC↔AIV 跨核通信

**用途**：AIC (Cube) 和 AIV (Vector) 内核之间的跨核同步、数据传输和管道管理
**类型**：`UnknownType`（sync/push/init/buffer/free 操作）或 `TileType` 透传（pop 操作）
**位置**：`src/ir/op/tile_ops/cross_core.cpp`（tpush/tpop）和 `src/ir/op/sync_ops/cross_core.cpp`（sync/tfree/管道初始化/缓冲区）
**Python API**：`import pypto.language as pl`（提升的操作）或 `from pypto.ir.op import tile, system`

### 显式事件同步

| 操作 | 参数 | 描述 | Kwargs |
| ---- | ---- | ---- | ------ |
| `system.sync_set` | 0 或 1（`event_id_dyn`） | 从一种核类型发出 `pto.sync.set` | `pipe`、静态 `event_id`、可选 `ffts_mode`、可选 `core_type` |
| `system.sync_wait` | 0 或 1（`event_id_dyn`） | 在对端核类型发出 `pto.sync.wait` | `pipe`、静态 `event_id`、可选 `core_type` |
| `system.set_ffts` | 1（`workspace`） | 声明 A3 显式跨核事件所需的 FFTS 设置 | — |

在显式指定类型的 AIC/AIV kernel 中使用 `pl.system.sync_set(event_id, pipe=..., ffts_mode=...)` 和 `pl.system.sync_wait(event_id, pipe=...)`。在混合 InCore kernel 中，传入 `core_type="aiv"` 或 `core_type="aic"`，以便 kernel 展开时将各事件操作保留在目标核通道上。在 A3 上，每个参与同步的 AIC/AIV 函数都必须在首次显式事件操作前调用 `pl.system.set_ffts(workspace)`；`workspace` 必须是至少包含 256 个元素的一维 `INT64` 张量，并作为 PTOAS 的设置操作数。PyPTO 的常驻运行时会持续安装硬件 FFTS 控制地址，因此生成的运行时封装不会用该操作数覆盖此地址。A5 不需要该设置。`event_id` 可以是用户可用范围 0–13 内的整数，也可以是动态 `pl.Scalar[pl.INDEX]`；ID 14 和 15 为保留值。`sync_set` 的可选 `ffts_mode` 必须为 0、1 或 2。手写跨核协议的作者负责正确配对事件 ID 和 pipe。PyPTO 的常规核内自动依赖插入仍保持启用，并使用独立的 `set_flag`/`wait_flag` 机制，因此不会占用这些显式跨核事件 ID。

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
| `system.aic_initialize_pipe` | 2 | 在 Cube 侧初始化跨核管道（位置参数：`c2v_consumer_buf`、`v2c_consumer_buf`，i32 SSA） | `dir_mask`, `slot_size`，可选 `slot_num`，可选 `local_slot_num`，可选 `id` |
| `system.aiv_initialize_pipe` | 2 | 在 Vector 侧初始化跨核管道（位置参数：`c2v_consumer_buf`、`v2c_consumer_buf`，i32 SSA） | `dir_mask`, `slot_size`，可选 `slot_num`，可选 `local_slot_num`，可选 `id` |

- `slot_num`（设置时必须 > 0）显式指定 GM 环形缓冲区的槽数量；省略时由 PTOAS 取默认值（单向 8，双向每方向 4）。
- `local_slot_num`（仅 a2/a3，必须 > 0 且 `<= slot_num`）显式指定本地槽数量。
- **预留/导入缓冲区大小需由用户自行设置，且与架构相关**：**a3** 为 `slot_size * local_slot_num`；**a5** 为 `slot_size * slot_num`。

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

## PrefetchOp：GM→L2 异步预取

一种隐藏访存延迟 (latency hiding) 的缓存提示。`async_prefetch` 通过 SDMA 异步地把一段
全局内存 (GM) 拉入 L2 缓存，期间可以并行执行不相关的计算；`wait` 阻塞直到预取完成。
预取不改变任何张量的值——同一个 kernel 加不加预取在数值上完全一致，只影响性能。

与大多数 PTO intrinsic 不同，`TPREFETCH_ASYNC` 不携带隐式的 wait-event 同步，
因此必须通过 event/session 这对句柄显式等待完成。

### 操作

| DSL | 操作数 | 结果 | PTOAS op |
| --- | ------ | ---- | -------- |
| `pl.prefetch.make_context()` | 无 | `PrefetchAsyncContextType` | `pto.make_prefetch_async_context` |
| `pl.prefetch.async_prefetch(src, ctx)` | GM Tensor、context | `AsyncEventType` | `pto.tprefetch_async` |
| `pl.prefetch.session(ctx)` | context | `AsyncSessionType` | `pto.get_prefetch_async_session` |
| `pl.prefetch.wait(evt, session)` | event、session | `BOOL` 标量 | `pto.comm.wait_async_event` |

这三个结果类型都是不透明的单例标记类型 (opaque singleton marker，无 shape、无 buffer)，
与 `CommCtxType` 属于同一族。SDMA workspace 不是程序操作数：runtime 持有它，
codegen 会向 prefetch kernel 注入隐藏指针。

### 约束

- `src` 必须是**扁平连续的逻辑一维 GM** 区域：shape 必须完全静态，且除最后一维外
  所有维度都为 `1`（`[N]`、`[1, N]`、`[1, 1, N]`）。该检查与 PTOAS 的
  `TPrefetchAsyncOp::verify()` 保持一致，因此 shape 写错会在 PyPTO IR 构造阶段就报错，
  而不是拖到 PTOAS 校验阶段。

### 使用示例

```python
@pl.program
class PrefetchExample:
    @pl.function(type=pl.FunctionType.InCore)
    def main(
        self, x: pl.Tensor[[1, 4096], pl.FP32],
        out: pl.Tensor[[1, 128], pl.FP32],
    ) -> pl.Tensor[[1, 128], pl.FP32]:
        ctx = pl.prefetch.make_context()
        evt = pl.prefetch.async_prefetch(x, ctx)     # 预热 L2，不阻塞
        session = pl.prefetch.session(ctx)
        # ... 此处的无关计算与预取重叠执行 ...
        pl.prefetch.wait(evt, session)               # 此时 x 已驻留在 L2
        tile = pl.load(x, [0, 0], [1, 128])
        return pl.store(tile, [0, 0], out)
```

**执行核**：这一族是 **AIV-only**。`TPREFETCH_ASYNC` 的 SDMA `tmpBuf` 来自
`PrefetchAsyncContext` 内部的 Vec(UB) scratch tile（pto-isa 有
`static_assert(ScratchTile::Loc == TileType::Vec)`），而 UB 位于向量核。这些算子
声明了 `CoreAffinity::VECTOR`，因此在混合 kernel 中 `ExpandMixedKernel` 会把它们留在
向量侧——既不会放到 cube 侧，也不会被复制到 cube 侧。

**Runtime 所有权与支持范围**：普通的单次执行 (one-shot execution) 会读取
生成 artifact 中的 SDMA 需求，并自动创建已启用 SDMA 的 worker。user、
orchestration 和 runtime tensor signature 中都不会出现 workspace。显式复用
L2 worker 时，需在构造时启用该能力：

```python
with ChipWorker(
    config=RunConfig(platform="a2a3", device_id=0), enable_sdma=True
):
    compiled(a, out, config=cfg)
```

当前由 runtime 提供 workspace 的执行路径仅在 onboard a2a3 上覆盖。在模拟器、
a5 或不提供 SDMA provider 的 runtime 上，启用该能力的 worker 会在 runtime
初始化时失败。PyPTO 不会分配后备 workspace，也不会把请求的 prefetch
静默降级为 no-op。onboard a2a3 ST 参见 `tests/st/runtime/ops/test_prefetch_async.py`。

## 文件组织

| 目录/文件 | 内容 |
| --------- | ---- |
| `src/ir/op/type_inference.cpp` | 共享的类型推断工具 |
| `tensor_ops/elementwise.cpp` | TensorOp: add, sub, mul, div |
| `tile_ops/matmul.cpp` | TileOp：matmul、gemv |
| `tile_ops/matmul_mx.cpp` | TileOp：matmul_mx、matmul_mx_acc、matmul_mx_bias、内部 tget_scale_addr 绑定 |
| `tile_ops/memory.cpp` | TileOp: load, store, read, get_block_idx |
| `tile_ops/elementwise.cpp` | TileOp: add, mul, div, adds, muls 等 |
| `tile_ops/reduction.cpp` | TileOp: sum（含 axis, keepdim） |
| `tile_ops/unary.cpp` | TileOp: sqrt |
| `sync_ops/sync.cpp` | SyncOp: sync_src, sync_dst, barriers |
| `sync_ops/task.cpp` | SyncOp：TaskId 哨兵与判定 |
| `sync_ops/launch.cpp` | SyncOp：SPMD 启动形状查询 |
| `sync_ops/cross_core.cpp` | CrossCoreOp: tpush, tpop, pipe init, buffers |
| `prefetch/prefetch_async.cpp` | PrefetchOp: make_context, async_prefetch, session, wait |

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

核心定义位于 `include/pypto/core/common.h` 和 `include/pypto/ir/`；注册表与类型推断实现在 `src/ir/`，算子实现按类别位于 `src/ir/op/{tensor_ops,tile_ops,sync_ops}/`。
