# PTO 代码生成 (CodeGen)

PTO 代码生成 (CodeGen) (`PTOCodegen`) 从 PyPTO 中间表示 (IR) 生成 PTO-ISA 方言的 MLIR 代码。它将高层 PyPTO 程序转换为适合加速器执行的低层 PTO 指令。

## 概述

### 核心特性

- **自动 MLIR 生成**: 将 PyPTO IR 转换为 PTO-ISA MLIR 方言
- **结构化代码生成 (CodeGen)**: 按顺序输出常量、张量 (Tensor) 视图和分配
- **隐式降级**: 从 `block.load`/`block.store` 自动生成 `pto.partition_view`
- **基于内存引用 (MemRef) 的分配**: 将 IR MemRef 对象映射到 `pto.alloc_tile` 操作
- **类型 (Type) 感知转换**: 从 TileType 元数据推导 tile_buf/tensor_view 类型
- **PTOAS 类型标注**: 为所有操作生成带类型的 `ins`/`outs` 子句

### 生成顺序

代码生成按以下固定顺序生成 MLIR:

1. **常量**: 索引和浮点值的 `arith.constant`
2. **张量视图**: 所有张量参数的 `pto.make_tensor_view`
3. **分配**: 所有 Tile 缓冲区的 `pto.alloc_tile` (基于 MemRef)
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
  std::string GetIndexConstant(int64_t val);
  std::string GetOrEmitFloatConstant(double value, const std::string& mlir_type = "f32");
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
        pl.store(tile_c, [0, 0], [32, 32], a)

# Compile with PTO backend and PTOAS optimization
output_dir = compile(MyKernel, strategy=OptimizationStrategy.PTOAS, backend_type=BackendType.PTO)
```

`compile()` 函数会自动应用选定的优化策略, 并根据 `backend_type` 调用相应的代码生成器。

### 直接访问代码生成器

```python
from pypto.pypto_core import codegen

# After pass transformations
pto_codegen = codegen.PTOCodegen()
pto_code = pto_codegen.generate(transformed_program)
print(pto_code)
```

## 操作映射

### 块操作到 PTO 指令

| PyPTO 操作 | 生成的 PTO-ISA |
| ---------- | -------------- |
| `block.load(tensor, [row, col], [h, w])` | `pto.partition_view` + `pto.tload` |
| `block.store(tile, [row, col], [h, w], tensor)` | `pto.partition_view` + `pto.tstore` |
| `block.mul(lhs, rhs)` | `pto.tmul` |
| `block.add(a, b, c)` | `pto.taddc` (三操作数加法) |
| `block.adds(tile, scalar)` | `pto.tadds` (Tile + 标量) |

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
     shape = [%c32, %c32]
     strides = [%c32, %c1]
     : !pto.tensor_view<?x?xf32>
```

**关键要点**:

- 形状来自 `TensorType.shape_`
- 步幅按行主序计算: 二维张量为 `[dim1, 1]`
- 常量 (`%c32`, `%c1`) 自动生成
- 张量视图类型每个维度使用 `?` (如二维为 `?x?xf32`)

### 分配生成

基于附加到 TileType 变量的 MemRef 对象。代码生成器从关联的 TileType 推导 Tile 维度和数据类型:

```mlir
%0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                       v_row=32, v_col=32, blayout=row_major,
                       slayout=none_box, fractal=512, pad=0>
```

**MemRef 到 alloc_tile 的映射**:

- 内存空间 (`MemRef.memory_space_`) 映射到 `loc` 属性 (使用 PTO 地址空间名)
- Tile 数据类型和维度从关联的 TileType 元数据推导
- 每个唯一 MemRef 对应一次分配

### 加载操作转换

**PyPTO IR**:

```python
tile_a = pl.load(tensor_a, [0, 0], [32, 32])
```

**生成的 MLIR** (两个操作):

```mlir
# 1. Create partition view
%3 = pto.partition_view %tensor_view, offsets = [%c0, %c0],
                 sizes = [%c32, %c32]
                 : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>

# 2. Load into tile buffer
pto.tload ins(%3 : !pto.partition_tensor_view<32x32xf32>)
          outs(%tile_buf : !pto.tile_buf<loc=vec, ...>)
```

**关键转换**:

- 张量参数通过 tensor_view 查找
- 偏移/大小来自 `block.load` 参数
- 输出 tile_buf 来自变量的 MemRef, 类型从 TileType 推导

### 存储操作转换

**PyPTO IR**:

```python
pl.store(tile_c, [0, 0], [32, 32], tensor_out)
```

**生成的 MLIR**:

```mlir
# 1. Create partition view for output
%5 = pto.partition_view %output_view, offsets = [%c0, %c0],
                 sizes = [%c32, %c32]
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
        pl.store(tile_c, [0, 0], [32, 32], c)
```

### 输出: PTO-ISA MLIR

```mlir
module {
  func.func @mul_kernel_2d(%arg0: !pto.ptr<f32>,
                          %arg1: !pto.ptr<f32>,
                          %arg2: !pto.ptr<f32>) {
    // Constants
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index

    // Tensor views
    %3 = pto.make_tensor_view %arg0, shape = [%c32, %c32]
         strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>
    %4 = pto.make_tensor_view %arg1, shape = [%c32, %c32]
         strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>
    %5 = pto.make_tensor_view %arg2, shape = [%c32, %c32]
         strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>

    // Allocations
    %0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, ...>
    %1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, ...>
    %2 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, ...>

    // Load tile_a
    %6 = pto.partition_view %3, offsets = [%c0, %c0], sizes = [%c32, %c32]
         : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    pto.tload ins(%6 : !pto.partition_tensor_view<32x32xf32>)
              outs(%0 : !pto.tile_buf<...>)

    // Load tile_b
    %7 = pto.partition_view %4, offsets = [%c0, %c0], sizes = [%c32, %c32]
         : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    pto.tload ins(%7 : !pto.partition_tensor_view<32x32xf32>)
              outs(%1 : !pto.tile_buf<...>)

    // Multiply
    pto.tmul ins(%0 : !pto.tile_buf<...>, %1 : !pto.tile_buf<...>)
             outs(%2 : !pto.tile_buf<...>)

    // Store tile_c
    %8 = pto.partition_view %5, offsets = [%c0, %c0], sizes = [%c32, %c32]
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
- 常量: `%c0`, `%c1`, `%c32`, `%cst`, ...
- 结果: `%0`, `%1`, `%2`, ...

### 基于 MemRef 的解析

对于 `block.mul` 等操作:

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

通过 `ir.compile()` 使用 PTO 后端编译时, 会自动为每个 InCore 函数生成内核包装器, 以桥接 ptoas 输出到 CCE/编排调用约定。

### 流水线

```text
InCore Function -> PTOCodegen -> .pto -> ptoas -> .cpp -> kernel_wrapper -> kernels/aiv/<name>.cpp
```

每个 InCore 函数通过 ptoas 独立编译。最终的包装器文件包含:

1. **预处理后的 ptoas 代码** (`__global__ AICORE` 替换为 `static`)
2. **`kernel_entry(__gm__ int64_t* args)`** 包装器, 解包参数数组并转发到 ptoas 函数

### 输出结构

当程序包含编排函数时, PTO 后端生成与 CCE 后端相同的输出结构:

```text
output_dir/
├── passes_dump/                     # IR after each pass
├── ptoas/                           # Intermediates
│   ├── <func_name>.pto              # MLIR from PTOCodegen
│   └── <func_name>.cpp              # C++ from ptoas
├── kernels/aiv/
│   └── <func_name>.cpp              # Final wrapper (CCE-compatible)
├── orchestration/
│   └── <orch_func_name>.cpp         # PTO2 runtime orchestration code
└── kernel_config.py                 # Runtime/orchestration/kernel config
```

编排代码生成与 CCE 共享 -- 两个后端使用 PTO2 运行时 API (`pto2_rt_submit_task`, `make_tensor_external` 等) 生成相同的编排 C++ 代码。

### 参数解包

包装器按照与 CCECodegen 相同的约定解包 `int64_t* args`:

| 参数类型 | 解包模式 |
| -------- | -------- |
| `TensorType` | `Tensor*` -> `buffer.addr` -> 带类型指针 |
| `ScalarType` | `uint64_t` -> 联合体解码 -> 带类型值 |

### 实现

**模块**: `python/pypto/ir/pto_codegen.py`

关键函数:

- `generate()` -- 入口点: 生成所有 PTO 后端文件 (内核 + 编排 + 配置)
- `_preprocess_ptoas_output()` -- 去除重复包含, 将函数设为静态
- `_generate_arg_unpacking()` -- 根据 IR 参数类型生成 C++ 解包代码
- `_generate_kernel_wrapper()` -- 组装完整的包装器文件

## 另请参阅

- [Pass 管理器](../passes/00-pass_manager.md): 了解 Pass 流水线
- [IR 构建器 (Builder)](../ir/06-builder.md): 以编程方式构造 IR
- [操作符组织](../ir/05-operators.md): 块操作详情
