# PTO 代码生成 (CodeGen)

PTO 代码生成 (CodeGen) (`PTOCodegen`) 从 PyPTO 中间表示 (IR) 生成 PTO-ISA 方言的 MLIR 代码。它将高层 PyPTO 程序转换为适合加速器执行的低层 PTO 指令。

## 设计原则：严格的 1-to-1 映射

代码生成必须是从 IR 到生成代码的**严格 1-to-1 转换**。每个 IR 节点直接映射到对应的输出代码结构——代码生成层不应执行优化、分析或间接转换。

| 属于代码生成的职责 | 属于前置 Pass 的职责 |
| ------------------ | -------------------- |
| IR 节点 → 输出代码映射 | 数据流分析（如追踪返回值到参数的映射） |
| 类型/格式转换（DataType → MLIR 类型） | IR 重组或规范化 |
| 名称修饰 (Name Mangling) 和 SSA 记录 | 优化或简化 |

**原因：** 嵌入分析逻辑的代码生成会变得脆弱——它重复了 Pass 已有的逻辑，且更难以独立测试。保持代码生成为直接的转换，确保其可预测性和可维护性。

**当发现代码生成中存在分析逻辑时：** 创建跟踪 Issue，在有带宽时将其重构为专用 Pass。[#814](https://github.com/hw-native-sys/pypto/issues/814) 就是一个实例：编排代码生成中的返回值到参数追踪逻辑已重构为 [`NormalizeReturnOrder`](../passes/22-normalize_return_order.md) pass。

## 概述

### 核心特性

- **自动 MLIR 生成**: 将 PyPTO IR 转换为 PTO-ISA MLIR 方言
- **结构化代码生成 (CodeGen)**: 按顺序输出常量、张量 (Tensor) 视图和分配
- **隐式降级**: 从 `tile.load`/`tile.store` 自动生成 `pto.partition_view`
- **基于 Tile 变量的分配**: 为每个 Tile 变量生成带显式 `addr` 的 `pto.alloc_tile` 操作
- **类型 (Type) 感知转换**: 从 TileType 元数据推导 tile_buf/tensor_view 类型
- **PTOAS 类型标注**: 为所有操作生成带类型的 `ins`/`outs` 子句

### 生成顺序

代码生成按以下固定顺序生成 MLIR:

1. **常量**: 索引和浮点值的 `arith.constant`
2. **张量视图**: 所有张量参数的 `pto.make_tensor_view`
3. **分配**: 所有 Tile 变量的 `pto.alloc_tile` (按变量维度, 带 `addr` 属性)
4. **操作**: 包含加载、计算、存储操作的函数体

## 架构

### 类结构

**头文件**: `include/pypto/codegen/pto/pto_codegen.h`

```cpp
namespace pypto::codegen {

class PTOCodegen : public CodegenBase {
 public:
  PTOCodegen();
  explicit PTOCodegen(const backend::Backend* backend);

  std::string Generate(const ir::ProgramPtr& program);

  // CodegenBase interface
  std::string GetCurrentResultTarget() const override;
  void Emit(const std::string& line) override;
  std::string GetExprAsCode(const ir::ExprPtr& expr) override;
  std::string GetTypeString(const DataType& dtype) const override;

  // PTO-specific helpers for operator codegen
  std::string NewTemp();
  std::string GetOrCreateTensorView(const ir::VarPtr& tensor);
  std::string GetOrEmitConstant(int64_t value, DataType dt);   // int/index 重载
  std::string GetOrEmitConstant(double value, DataType dt);    // float 重载
  std::string GetTensorViewTypeString(const ir::TensorType* tensor_type) const;
  std::string GetTileBufTypeString(const ir::MemRef* memref) const;
  std::string GetExprTypeAnnotation(const ir::ExprPtr& expr);
  std::string GetCurrentResultTileBufTypeString() const;
};

}  // namespace codegen
```

### 实现组件

**文件**: `src/codegen/pto/pto_codegen.cpp`

| 组件 | 用途 |
| ---- | ---- |
| `PTOCodegen` | 主访问者类 (继承 `CodegenBase`), 用于 IR 遍历 |
| `MemRefCollectorVisitor` | 收集 MemRef 对象及其关联的 TileType 用于分配 |
| 辅助函数 | `DataTypeToMLIRImpl()`, `MemorySpaceToMLIR()` |

## Python API

### 基本用法

```python
from pypto.ir import compile, OptimizationStrategy
from pypto.backend import BackendType
import pypto.language as pl

@pl.program
class MyKernel:
    @pl.function
    def vector_add(self,
                   a: pl.Tensor[[32, 32], pl.FP32],
                   b: pl.Tensor[[32, 32], pl.FP32]):
        tile_a = pl.load(a, [0, 0], [32, 32])
        tile_b = pl.load(b, [0, 0], [32, 32])
        tile_c = pl.add(tile_a, tile_b)
        pl.store(tile_c, [0, 0], a)

# Compile with PTO backend and DebugTileOptimization (debug only)
output_dir = compile(
    MyKernel,
    strategy=OptimizationStrategy.DebugTileOptimization,
    backend_type=BackendType.Ascend910B,
)
```

`compile()` 函数会自动应用选定的优化策略, 并根据 `backend_type` 调用相应的代码生成器。
正常的 PTO 编译应使用 `Default`；`DebugTileOptimization` 只用于调试 pass 流水线。

### 直接访问代码生成器

```python
from pypto.pypto_core import codegen

# After pass transformations
pto_codegen = codegen.PTOCodegen()
pto_code = pto_codegen.generate(transformed_program)
print(pto_code)
```

## 操作映射

### Tile 操作到 PTO 指令

| PyPTO 操作 | 生成的 PTO-ISA |
| ---------- | -------------- |
| `tile.load(tensor, [row, col], [h, w])` | `pto.partition_view` + `pto.tload` |
| `tile.store(tile, [row, col], tensor)` | `pto.partition_view` + `pto.tstore` |
| `tile.slice(tile, [h, w], [row, col][, valid_shape=...])` | `pto.subview`（零拷贝视图；仅在传入 `valid_shape` 时输出 `valid [...]` 子句） |
| `tile.assemble(target, source, [row, col])` | （可选）`pto.tmov target -> dst` + `pto.subview dst[row, col] sizes [src.rows, src.cols]` + `pto.tmov src -> dst_view` |
| `tile.mul(lhs, rhs)` | `pto.tmul` |
| `tile.add(a, b, c)` | `pto.taddc` (三操作数加法) |
| `tile.adds(tile, scalar)` | `pto.tadds` (Tile + 标量) |

**`tile.slice` / `tile.assemble` 下沉细节。** 两个 op 都通过 `pto.subview`
下沉，它是源 tile 的纯视图别名（不搬数据，也不会额外发 `pto.alloc_tile`）。
`pto.subview` 要求结果 `tile_buf` 与源 `tile_buf` 在 `dtype`、`memory_space`、
`blayout`、`slayout`、`fractal` 和 `pad` 上完全一致，因此
`DeduceTileSliceType` 会将源 `TileView` 的这四个字段透传到结果，使新生成的
`TileType` 天然满足约束。后端 codegen 还会在下沉时执行 `CheckSubviewTileCompat`
做兜底校验：

- 源和结果都必须显式携带 `TileView`。
- `dtype`、`blayout`、`slayout`、`fractal` 与 `pad` 必须严格相等。
- `pad` 必须为 `PadValue::null`——`pto.subview` 是视图而不是 fillpad；如果
  需要 zero/min/max 填充，请在切出来的子 tile 上再调用 `tile.fillpad`。

对 `tile.assemble`，前置的 `pto.tmov target → dst` 仅在缓冲复用未把
`target` 与目标缓冲合并时才会发出，用于保留写入窗口外的数据；末尾的
`pto.tmov src → dst_view` 才是真正写入由 `pto.subview` 切出的子窗口的数据
搬运。

### 跨核操作到 PTO 指令

| PyPTO 操作 | 生成的 PTO-ISA | 描述 |
| ---------- | -------------- | ---- |
| `tile.tpush_to_aiv(tile, split=N)` | `pto.tpush_to_aiv ins(%tile : type) {split = N}` | Cube → Vector 推送 |
| `tile.tpush_to_aic(tile, split=N)` | `pto.tpush_to_aic ins(%tile : type) {split = N}` | Vector → Cube 推送 |
| `tile.tpop_from_aic(split=N)` | `%buf = pto.tpop_from_aic {split = N} -> type` | 从 Cube 管道弹出 |
| `tile.tpop_from_aiv(split=N)` | `%buf = pto.tpop_from_aiv {split = N} -> type` | 从 Vector 管道弹出 |
| `system.tfree_to_aic(tile_from_tpop)` | `pto.tfree_from_aic {split = N}` | 将消费侧槽位释放回 Cube |
| `system.tfree_to_aiv(tile_from_tpop)` | `pto.tfree_from_aiv {split = N}` | 将消费侧槽位释放回 Vector |
| `system.aic_initialize_pipe(...)` | `pto.aic_initialize_pipe {dir_mask = D, slot_size = S} (c2v_consumer_buf = %ssa : i32, v2c_consumer_buf = %ssa : i32)` | Cube 管道初始化 |
| `system.aiv_initialize_pipe(...)` | `pto.aiv_initialize_pipe {dir_mask = D, slot_size = S} (c2v_consumer_buf = %ssa : i32, v2c_consumer_buf = %ssa : i32)` | Vector 管道初始化 |
| `system.reserve_buffer(...)` | `%name = pto.reserve_buffer {name = "N", size = S, location = #pto.address_space<loc>, auto = false, base = B} -> i32` | 预留缓冲区 |
| `system.import_peer_buffer(...)` | `%name = pto.import_reserved_buffer {name = "N", peer_func = @F} -> i32` | 导入对等缓冲区 |

**说明：**

- Push 操作使用带类型 tile buffer 的 `ins()` 子句；前端 Pop 操作生成 SSA 结果，并带 `-> !pto.tile_buf<...>` 结果类型
- 如果被 push 的 tile 通过动态 `valid_row` / `valid_col` operand 分配，或经
  `tile.set_validshape` 更新，`tpush` 会发射已经更新运行时 valid shape 的同一个
  tile handle。对于 split `tpush`，codegen 会临时使用完整的非切分传输维度（上下
  切分使用完整 `cols`，左右切分使用完整 `rows`），随后恢复 producer tile 的逻辑
  valid shape；消费侧动态 tpop operand 仍携带后续计算和 store 使用的逻辑范围。
- 当 tpop 结果的 `TileView.valid_shape` 与物理 tile shape 不一致时，PTO codegen 会生成 PTOAS 前端操作数：`%buf = pto.tpop_from_*(%valid_row, %valid_col) {split = N} -> !pto.tile_buf<..., v_row=?, v_col=?, ...>`。这同时覆盖动态表达式和 `[0, 0]` 这类静态非满形状；operand 携带后续计算和 store 使用的逻辑范围。
- 对于 split consumer，`SplitVectorKernel` 会按 subblock 本地化这些动态
  tpop valid-shape operand（例如 `[16, 16]` tile 做上下切分时，全局
  `[8, 16]` 会变成 `[8, 16]` 和 `[0, 16]`）。
- `system.tfree_*` 的 `split` 来自其 tile 参数，因此前端必须释放由 `tile.tpop_*` 产生的那个确切 SSA 值，即使 PTO 指令本身并不显式接收该 tile 作为操作数
- `ExpandMixedKernel` 现在会在 split 生成的消费侧 `tile.tpop_*` 之后自动补 `system.tfree_*`，保持 `tpop -> direct users -> tfree -> next tpop`
- `reserve_buffer` 和 `import_reserved_buffer` 返回 `i32` SSA 值；`initialize_pipe` 以操作数引用这些值
- `AllocateMemoryAddr` 会在 PTO 输出前解析 `reserve_buffer(base=AUTO)`，因此 PTO 始终输出 `auto = false, base = <value>`
- `reserve_buffer` location 对于 AIC 函数为 `mat`，对于 AIV/InCore 函数为 `vec`
- `import_reserved_buffer` 使用 MLIR 符号语法（`@func_name`）表示 `peer_func`
- 缓冲区名称和 peer_func 字符串由 `CheckSafeIdentifier` 验证（仅允许字母数字和下划线）

### 参数类型处理

| PyPTO 类型 | MLIR 参数类型 | 后处理 |
| ---------- | ------------- | ------ |
| `TensorType` | `!pto.ptr<dtype>` | 生成 `pto.make_tensor_view` |
| `ScalarType` | `dtype` (如 `f32`) | 直接用作 `%argN` |
| `TileType` | 不允许作为参数 | 必须在内部计算 |

## 代码生成细节

### 张量视图生成

对于每个 `TensorType` 参数, 代码生成器会生成:

```mlir
%0 = pto.make_tensor_view %arg0,
     shape = [%c32_index, %c32_index]
     strides = [%c32_index, %c1_index]
     {layout = #pto.layout<nd>}
     : !pto.tensor_view<?x?xf32>
```

**关键要点**:

- 形状来自 `TensorType.shape_`
- 步幅按行主序计算: 二维张量为 `[dim1, 1]`
- 常量 (`%c32_index`, `%c1_index`) 自动生成
- 张量视图类型每个维度使用 `?` (如二维为 `?x?xf32`)

#### 二维张量的 Layout 处理

`make_tensor_view` 上的 `layout` 属性告诉 PTOAS 内存布局约定。代码生成器根据
张量的 IR 类型和形状决定 shape、strides 和 layout:

| 情况 | 输出 Shape | 输出 Strides | Layout | 说明 |
| ---- | ---------- | ------------ | ------ | ---- |
| ND `[R, C]` | `[R, C]` | `[C, 1]` | `nd` | 标准行主序 |
| DN `[R, C]` (均 > 1) | `[C, R]` | `[1, C]` | `dn` | Shape 交换以符合 PTOAS 列主序约定 |
| 列向量 `[M, 1]` | `[M, 1]` | `[1, M]` | `dn` | 自动检测, 无需 DN 标注 |

**列向量自动 DN**: 任何最后一维为编译时常量 `1` 的二维张量 (即形状 `[M, 1]`)
会自动以 `layout = dn` 和步幅 `[1, M]` 输出。这是因为 PTOAS 对于形状/步幅模式
`[M, 1] / [1, 1]` 始终推断为 DN, 使得退化的 ND 表示产生歧义。代码生成器通过始终
使用无歧义的 DN 步幅来解决此问题。用户无需在 DSL 中为 `[M, 1]` 张量标注 `pl.DN`。

列向量 `[16, 1]` 示例 (DSL 中无 DN 标注):

```mlir
%col_view = pto.make_tensor_view %arg1,
    shape = [%c16_index, %c1_index], strides = [%c1_index, %c16_index]
    {layout = #pto.layout<dn>}
    : !pto.tensor_view<?x?xf32>
```

### 分配生成

基于附加到 TileType 变量的 MemRef 对象。代码生成器从关联的 TileType 推导 Tile 维度和数据类型:

```mlir
%mi_tile = pto.alloc_tile addr = %c8320_i64 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                       v_row=16, v_col=1, blayout=col_major,
                       slayout=none_box, fractal=512, pad=0>
%mi_tile_nd = pto.alloc_tile addr = %c8320_i64 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                       v_row=1, v_col=16, blayout=row_major,
                       slayout=none_box, fractal=512, pad=0>
```

**Tile 变量到 alloc_tile 的映射**:

- 内存空间 (`TileType.memory_space_`) 映射到 `loc` 属性 (使用 PTO 地址空间名)
- Tile 数据类型和维度从每个变量自身的 TileType 元数据推导
- 每个 Tile 变量对应一次分配 (不是每个唯一 MemRef)
- `addr` 属性来自 `MemRef.addr_`，输出为 `arith.constant ... : i64`
- 共享同一 MemRef 的变量共享相同的 `addr` SSA 值

### 加载操作转换

**PyPTO IR**:

```python
tile_a = pl.load(tensor_a, [0, 0], [32, 32])
```

**生成的 MLIR** (两个操作):

```mlir
# 1. Create partition view
%3 = pto.partition_view %tensor_view, offsets = [%c0_index, %c0_index],
                 sizes = [%c32_index, %c32_index]
                 : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>

# 2. Load into tile buffer
pto.tload ins(%3 : !pto.partition_tensor_view<32x32xf32>)
          outs(%tile_buf : !pto.tile_buf<loc=vec, ...>)
```

**关键转换**:

- 张量参数通过 tensor_view 查找
- 偏移/大小来自 `tile.load` 参数
- 输出 tile_buf 来自变量的 MemRef, 类型从 TileType 推导

### 存储操作转换

**PyPTO IR**:

```python
pl.store(tile_c, [0, 0], tensor_out)
```

**生成的 MLIR**:

```mlir
# 1. Create partition view for output
%5 = pto.partition_view %output_view, offsets = [%c0_index, %c0_index],
                 sizes = [%c32_index, %c32_index]
                 : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>

# 2. Store from tile buffer
pto.tstore ins(%tile_buf : !pto.tile_buf<loc=vec, ...>)
           outs(%5 : !pto.partition_tensor_view<32x32xf32>)
```

### 计算操作

#### 示例: Tile 乘法

PyPTO:

```python
tile_c = pl.mul(tile_a, tile_b)
```

MLIR:

```mlir
pto.tmul ins(%tile_a_buf : !pto.tile_buf<...>,
             %tile_b_buf : !pto.tile_buf<...>)
         outs(%tile_c_buf : !pto.tile_buf<...>)
```

**结果处理**:

- 结果变量的 MemRef 决定输出 tile_buf
- 输入操作数通过变量名查找解析
- 所有 `ins`/`outs` 子句包含类型标注

## 完整示例

### 输入: PyPTO 程序

```python
import pypto.language as pl

@pl.program
class MulKernel:
    @pl.function
    def mul_kernel_2d(self,
                     a: pl.Tensor[[32, 32], pl.FP32],
                     b: pl.Tensor[[32, 32], pl.FP32],
                     c: pl.Tensor[[32, 32], pl.FP32]):
        # Load tiles
        tile_a = pl.load(a, [0, 0], [32, 32])
        tile_b = pl.load(b, [0, 0], [32, 32])

        # Multiply
        tile_c = pl.mul(tile_a, tile_b)

        # Store result
        pl.store(tile_c, [0, 0], c)
```

### 输出: PTO-ISA MLIR

```mlir
module {
  func.func @mul_kernel_2d(%arg0: !pto.ptr<f32>,
                          %arg1: !pto.ptr<f32>,
                          %arg2: !pto.ptr<f32>) {
    // Constants
    %c32_index = arith.constant 32 : index
    %c1_index = arith.constant 1 : index
    %c0_index = arith.constant 0 : index

    // Tensor views
    %3 = pto.make_tensor_view %arg0, shape = [%c32_index, %c32_index]
         strides = [%c32_index, %c1_index] : !pto.tensor_view<?x?xf32>
    %4 = pto.make_tensor_view %arg1, shape = [%c32_index, %c32_index]
         strides = [%c32_index, %c1_index] : !pto.tensor_view<?x?xf32>
    %5 = pto.make_tensor_view %arg2, shape = [%c32_index, %c32_index]
         strides = [%c32_index, %c1_index] : !pto.tensor_view<?x?xf32>

    // Allocations
    %0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, ...>
    %1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, ...>
    %2 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, ...>

    // Load tile_a
    %6 = pto.partition_view %3, offsets = [%c0_index, %c0_index], sizes = [%c32_index, %c32_index]
         : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    pto.tload ins(%6 : !pto.partition_tensor_view<32x32xf32>)
              outs(%0 : !pto.tile_buf<...>)

    // Load tile_b
    %7 = pto.partition_view %4, offsets = [%c0_index, %c0_index], sizes = [%c32_index, %c32_index]
         : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    pto.tload ins(%7 : !pto.partition_tensor_view<32x32xf32>)
              outs(%1 : !pto.tile_buf<...>)

    // Multiply
    pto.tmul ins(%0 : !pto.tile_buf<...>, %1 : !pto.tile_buf<...>)
             outs(%2 : !pto.tile_buf<...>)

    // Store tile_c
    %8 = pto.partition_view %5, offsets = [%c0_index, %c0_index], sizes = [%c32_index, %c32_index]
         : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    pto.tstore ins(%2 : !pto.tile_buf<...>)
               outs(%8 : !pto.partition_tensor_view<32x32xf32>)

    return
  }
}
```

## 变量映射

### 内部跟踪

代码生成器维护多个映射来跟踪 MLIR 变量名:

| 映射 | 用途 | 示例 |
| ---- | ---- | ---- |
| `var_to_mlir_` | IR 变量到 MLIR 静态单赋值 (SSA) 名 | `"tile_a"` -> `"%0"` |
| `tensor_to_view_` | 参数到 tensor_view | `"a"` -> `"%3"` |
| `memref_to_mlir_` | MemRef 指针到 tile_buf | `memref.get()` -> `"%0"` |
| `memref_to_tile_type_` | MemRef 指针到 TileType | 用于推导 tile_buf 类型 |

**SSA 值命名**:

- 参数: `%arg0`, `%arg1`, `%arg2`, ...
- 常量: `%c0_index`, `%c1_index`, `%c32_index`, `%c0_i64`, `%cst`, ...
- 结果: `%0`, `%1`, `%2`, ...

### 基于 MemRef 的解析

对于 `tile.mul` 等操作:

```python
tile_c = pl.mul(tile_a, tile_b)
```

代码生成器:

1. 通过 `var_to_mlir_` 解析 `tile_a` -> `%0`
2. 通过 `var_to_mlir_` 解析 `tile_b` -> `%1`
3. 从 TileType 获取 `tile_c` 的 MemRef
4. 通过 `memref_to_mlir_` 映射 MemRef -> `%2`
5. 从 `memref_to_tile_type_` 获取 tile_buf 类型
6. 生成: `pto.tmul ins(%0 : !pto.tile_buf<...>, %1 : !pto.tile_buf<...>) outs(%2 : !pto.tile_buf<...>)`

## 类型转换

### 数据类型映射

| PyPTO 数据类型 | MLIR 类型 |
| -------------- | --------- |
| `DataType::FP32` | `f32` |
| `DataType::FP16` | `f16` |
| `DataType::BF16` | `bf16` |
| `DataType::INT32` | `i32` |
| `DataType::INT64` | `i64` |
| `DataType::INT8` | `i8` |
| `DataType::UINT8` | `ui8` |

### 内存空间映射

| PyPTO 内存空间 | PTO 地址空间 |
| -------------- | ------------ |
| `MemorySpace::DDR` | `gm` (全局内存) |
| `MemorySpace::Vec` | `vec` (向量缓冲区) |
| `MemorySpace::Mat` | `mat` (矩阵缓冲区) |
| `MemorySpace::Left` | `left` |
| `MemorySpace::Right` | `right` |
| `MemorySpace::Acc` | `acc` (累加器) |
| `MemorySpace::Bias` | `bias` (偏置缓冲区) |

### Tile 缓冲区属性

生成的 `alloc_tile` 操作从 TileType 元数据推导数据类型和维度, 从关联的 TileView 推导布局/分形/填充 (如有):

```mlir
!pto.tile_buf<
  loc=vec,             // PTO address space (from MemorySpace)
  dtype=f32,           // Element data type (from TileType)
  rows=32,             // Tile height (from TileType shape)
  cols=32,             // Tile width (from TileType shape)
  v_row=32,            // Virtual row size (= rows)
  v_col=32,            // Virtual column size (= cols)
  blayout=row_major,   // Block layout (from TileView, default: row_major)
  slayout=none_box,    // Scatter layout (from TileView, default: none_box)
  fractal=512,         // Fractal size (from TileView, default: 512)
  pad=0                // Pad mode as int (from TileView, default: 0/null)
>
```

**TileView 推导的属性**:

| 属性 | 来源 | 枚举值 | 默认值 |
| ---- | ---- | ------ | ------ |
| `blayout` | `TileView::blayout` | `none_box`, `row_major`, `col_major` | `row_major` |
| `slayout` | `TileView::slayout` | `none_box`, `row_major`, `col_major` | `none_box` |
| `fractal` | `TileView::fractal` | uint64 | `512` |
| `pad` | `TileView::pad` | `null(0)`, `zero(1)`, `max(2)`, `min(3)` | `null(0)` |

当 MemRef 没有关联 TileView 时, 代码生成器使用上表中的默认值。

## 内核包装器生成 (PTO 后端)

通过 `ir.compile()` 使用 PTO 后端编译时, 会自动为每个 InCore 函数生成内核包装器, 以桥接 ptoas 输出到编排调用约定。

### 流水线

```text
InCore Function -> PTOCodegen -> .pto -> ptoas -> .cpp -> kernel_wrapper -> kernels/aiv/<name>.cpp
```

每个 InCore 函数通过 ptoas 独立编译。最终的包装器文件包含:

1. **预处理后的 ptoas 代码** (`__global__ AICORE` 替换为 `static`)
2. **`kernel_entry(__gm__ int64_t* args)`** 包装器, 解包参数数组并转发到 ptoas 函数

### 输出结构

当程序包含编排函数时, PTO 后端生成以下输出结构:

```text
output_dir/
├── passes_dump/                     # IR after each pass
├── ptoas/                           # Intermediates
│   ├── <func_name>.pto              # MLIR from PTOCodegen
│   └── <func_name>.cpp              # C++ from ptoas
├── kernels/aiv/
│   └── <func_name>.cpp              # Final wrapper
├── orchestration/
│   └── <orch_func_name>.cpp         # PTO2 runtime orchestration code
└── kernel_config.py                 # Runtime/orchestration/kernel config
```

编排代码生成使用 PTO2 运行时 API (`pto2_rt_submit_task`, `make_tensor_external` 等) 生成编排 C++ 代码。

### 参数解包

包装器按照标准约定解包 `int64_t* args`:

| 参数类型 | 解包模式 |
| -------- | -------- |
| `TensorType` | `Tensor*` -> `buffer.addr` -> 带类型指针 |
| `ScalarType` | `uint64_t` -> 联合体解码 -> 带类型值 |

### 实现

**模块**: `python/pypto/backend/pto_backend.py`

关键函数:

- `generate()` -- 入口点: 生成所有 PTO 后端文件 (内核 + 编排 + 配置)
- `_preprocess_ptoas_output()` -- 去除重复包含, 将函数设为静态
- `_generate_arg_unpacking()` -- 根据 IR 参数类型生成 C++ 解包代码
- `_generate_kernel_wrapper()` -- 组装完整的包装器文件

## 另请参阅

- [Pass 管理器](../passes/00-pass_manager.md): 了解 Pass 流水线
- [IR 构建器 (Builder)](../ir/06-builder.md): 以编程方式构造 IR
- [操作符组织](../ir/05-operators.md): Tile 操作详情
