# ConvertTensorToTileOps Pass

将 InCore 函数中的 tensor 操作（张量操作）转换为 tile 操作（块操作），并更新编排函数的调用点。

## 概述

`OutlineIncoreScopes` 将 InCore 作用域提取为独立函数后，这些函数仍使用 `TensorType` 变量和 `tensor.*` 操作。本 pass 将其降级为直接映射到 PTO-ISA 指令的 `TileType` 变量和 `tile.*` 操作。

本 pass 还会更新编排/不透明函数中的调用点：为 InCore 函数新增的每个输出参数，在调用点插入 `tensor.create`。

**前置条件**：

- 输入 IR 必须为 SSA 形式
- InCore 作用域必须已提取（需先运行 `OutlineIncoreScopes`）
- 语句结构必须已规范化

**使用时机**：在 `OutlineClusterScopes` 之后、`OptimizeOrchTensors` 之前运行。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::ConvertTensorToTileOps()` | `passes.convert_tensor_to_tile_ops()` | Program 级 |

**Python 用法**：

```python
from pypto.pypto_core import passes

convert_pass = passes.convert_tensor_to_tile_ops()
program_tiled = convert_pass(program)
```

## 算法

本 pass 在 Program 级别分三阶段执行：

### 阶段一：转换 InCore 函数

对每个 `FunctionType::InCore` 函数：

1. **预扫描 MatmulSlice 模式**：收集被 `tensor.matmul` / `tensor.matmul_acc` 使用的 `tensor.slice` 结果。这些需要生成 Mat 空间的 `tile.load`（自然 load，转置时再叠加零拷贝 `tile.transpose_view`），而非默认的 `tile.load(Vec)`。

2. **插入 tile.load（入口加载）**：为每个被转换 op 直接使用的 `TensorType` 参数，在函数入口插入 `tile.load(param, zeros, shape, shape, target_memory=Vec, transpose=False)`。仅被自加载 op（`tensor.slice`、`tensor.matmul`、`tensor.read`、`tensor.write`、`tensor.assemble`）引用的参数不会生成额外加载。

3. **通过 TensorToTileMutator 转换函数体**：遍历函数体，使用 `OpConversionRegistry` 将每个 `tensor.*` 调用转换为对应的 `tile.*` 调用。Mutator 通过控制流传播类型变更（IterArgs、ForStmt/WhileStmt return_vars、IfStmt return_vars）。

4. **插入 tile.store（出口存储）**：对每个从 `TensorType` 转换为 `TileType` 的返回值，添加 `Out` 参数并插入 `tile.store(tile, zeros, out_param)`。如果返回值来自 `tile.assemble` 循环，则将循环重写为直接使用 `tile.store`（转换时 assemble-loop 重写；与 `OptimizeOrchTensors` 模式 3 不同，该模式处理跨函数优化）。

### 阶段二a：通过 Spmd/Group 包装函数转发新增 Out 参数

`OutlineClusterScopes` 产生的 Spmd/Group 包装函数是对其参数到单个内部 InCore
调用的透明 1:1 转发器。当阶段一为该 InCore 被调用者新增 `Out` 参数时，
包装函数必须在自身签名上镜像这些新增参数并通过内部调用转发给被调用者 ——
否则编排层代码生成的 `BuildWrapperReorderedParams` 不变式（每个内部调用的
`Var` 实参都能解析到某个包装函数参数）会被破坏。

对每个 `FunctionType::Spmd` / `FunctionType::Group` 函数：

1. `ForwardedCallFinder` 查找第一个调用转换后 InCore（阶段一新增了至少一个
   `Out` 参数）的调用点。
2. 若找到，则在包装函数签名末尾追加与 InCore 新增参数类型相同（复用
   `name_hint_`）的 `Out` 参数，并由 `WrapperForwardMutator` 重写该内部调用：
   将新变量追加到实参列表、更新调用返回类型为被调用者新的返回类型。包装
   函数体内部**不会**合成 `tensor.create` —— 分配职责保留在调用者侧。
3. 若未找到转发到转换后 InCore 的调用，则包装函数保持不变。

### 阶段二b：更新编排函数调用点

对每个调用了转换后 InCore 函数或阶段二a 吸收了新增 Out 参数的包装函数的
编排 / 不透明函数：

1. 为每个新增的输出参数插入 `tensor.create`
2. 将创建的张量作为额外参数追加到调用中

InCore、Spmd、Group 函数在本阶段被跳过 —— 它们已在阶段一 / 二a 中被改写。

## MatmulSlice 模式

当 `tensor.slice` 的结果被 `tensor.matmul` 或 `tensor.matmul_acc` 使用时，slice 必须生成 Mat 空间的 tile 而非 Vec 空间。本 pass 预扫描此模式，并根据 matmul kwargs 中的转置标志（LHS 使用 `a_trans`，RHS 使用 `b_trans`）生成 `tile.load(Mat, transpose=...)`。

## Transpose 下沉

`tensor.transpose` 下沉为一个 3-arg 的 **`tile.transpose(input, axis1, axis2)`**。PTO 后端的 `pto.ttrans` 指令需要一个 scratch 工作 tile（与源 tile 同 shape/同 dtype），但该 scratch 纯属 codegen 细节，并非语义操作数。[`FlattenTileNdTo2D`](15-flatten_tile_nd_to_2d.md) 是 scratch 物化的**唯一归属**：它为 2D 以及逐页 >2D 的 transpose 统一产出 codegen-ready 的 4-arg 形态（`tile.create` + `tile.transpose(..., tmp)`），且仍在内存分配器之前（scratch 仍能拿到真实 UB 地址）。把 scratch 从高层 op 中移除后，`tensor.transpose` 与 DSL `pl.tile.transpose(tile, axis1, axis2)` 都与语义操作保持 1:1。

```python
# 转换前
y = tensor.transpose(x, 0, 1)

# 本 pass 转换后
y_tile = pl.tile.transpose(x_tile, 0, 1)   # 3-arg，无 scratch

# 经 FlattenTileNdTo2D 后（scratch 在那里物化）
transpose_tmp = pl.tile.create(x.shape, x.dtype, target_memory=x.memory_space)
y_tile = pl.tile.transpose(x_tile, 0, 1, transpose_tmp)
```

## Scatter Update 下沉

`tensor.scatter_update` / `tile.scatter_update`（整行散射，仅支持 `dim=-2`）下沉为逐元素的 `tile.scatter`（`pto.tscatter`）加上 `tile.sel` 保留混合。硬件 `pto.tscatter` 按扁平目标下标逐元素写入（`dst.flat[idx[k, c]] = src[k, c]`），且其 `dst` 操作数是 **write-only**（未写入的槽位不保留），因此本 pass 自行重建“未命中行保留 `input`”的语义。

整行更新 `input[index.flat[k], :] = src[k, :]` 被表达为扁平下标：

```text
flat_idx[k, c] = index.flat[k] * d + c          # d = 特征宽度（= src 列数）
```

扁平下标的算术**全程在 i32 中计算**，仅在最后把成品 row-major `[n, d]` 下标通过一条 `tile.cast` 窄化到 `pto.tscatter` 要求的宽度（2 字节数据用 i16，4 字节用 i32）。全程 i32 保证每个中间 tile 都是规范的、32 字节对齐的 row-major 布局——更早窄化要么作用在 `col_major [n, 1]` 视图上（`tile.cast` 会错位），要么产生不对齐的 2 字节 `[b, s]` tile（`cols * 2` 字节不满足 32 字节对齐）。

生成的 PTO 算子时序（FP32，`[32, 32]` input、`[2, 8]` index、`[16, 32]` src）：

| # | PTO 算子 | 产出 |
| - | -------- | ---- |
| 1–3 | `pto.tload` ×3 | `input_tile`、`index_tile`、`src_tile` |
| 4 | `pto.tci` | 列 arange `[1, d]` = `0..d-1` |
| 5 | `pto.texpands` | 零模板 `[n, d]` |
| 6 | `pto.tcolexpand` | `col_nd[k, c] = c` |
| 7 | `pto.tmuls` | `row_base[k] = index.flat[k] * d`（index reshape 成 `[n, 1]`） |
| 8 | `pto.trowexpandadd` | `flat_idx = col_nd + row_base` → `[n, d]` |
| 8a | `pto.tcvt` | 把 `flat_idx` 窄化 i32→i16（**仅 2 字节 dtype**） |
| 9 | `pto.texpands` | 置零的散射基底 `[m, d]` |
| 10 | `pto.tscatter` | `scattered` = src 散射进零基底（命中位 = src，未命中 = 0） |
| 11–12 | `pto.texpands` ×2 | mask 零基底 `[m, d]`、ones 源 `[n, d]` |
| 13 | `pto.tscatter` | `mask` = ones 散射进零基底（命中位 = 1，未命中 = 0） |
| 14 | `pto.tcmps` | `pred = (mask != 0)` |
| 15 | `pto.tsel` | `out = sel(pred, scattered, input_tile)` |
| 16 | `pto.tstore` | 把 `out` 写回输出张量 |

用 `tile.sel`（而非 `input * mask`）重建保留混合，使下沉不产生 `pto.tmul`（A2/A3 对 bf16/i8 拒绝 `tmul`）。index 的 `reshape [b, s] → [n, 1]` 是 buffer 视图重命名，不是单独的 PTO 算子。

## Paged Gather 下沉

`tensor.paged_gather(src, indices, block_table, ...)` 把分页 KV 池中分散的行直接聚合到片上 buffer（默认 L1 / `Mem.Mat`，也可 UB / `Mem.Vec`）。硬件 `pto.tgather` 指令只能写 UB，因此“聚合到 L1”**不是**索引 gather 指令，而是 **Cube 核（AIC）** 上一段全标量的逐行 `GM → 片上` DMA 循环。`src`、`indices`、`block_table` 保持为 GM 张量（该算子注册为 self-loading，框架不会把它们预加载成 Vec tile）。

本 pass 直接物化该循环：

```text
rows = tensor.dim(indices, last_axis)                  # 运行期聚合行数
acc  = tile.create([max_indices, size], target_memory=space)   # 静态片上 buffer
for i in [0, rows):                                    # ForStmt，iter_arg = acc
    idx   = tensor.read(indices, [i])                  # 标量读 GM（pto.load_scalar）
    phys  = block_table[idx // block_size] * block_size + idx % block_size   # 标量
    acc   = tile.gather_row(acc, src, [i, 0], [phys, col_off], [1, size])    # GM->片上
    yield acc
```

`tile.gather_row` 是一个 DPS 算子，把一条物理 GM 行直接写入累加器的子区域：`pto.subview`（acc）+ `pto.partition_view`（src）+ `pto.tload`（`GM → 片上`）——**无 `pto.tmov`**。a2a3 上不支持 L1→L1 的 `tmov`（L1 只能经 `GM → L1` 的 `tload` 填充），因此直接把行 load 进累加器子区域,而不是 assemble。

只有小的索引 / 页表元数据是标量读 GM；KV 大数据经 `pto.tload` 直接 `GM → L1`，**全程不经 UB**——消除了 `gather_kv → qk_pv` 流水今天付出的 GM 往返。`is_trans=True`（仅 Mat）按转置加载每行到列偏移 `[0, i]`，得到 matmul B 操作数布局。`max_indices` 静态确定 L1 buffer 大小；运行期 `rows` 驱动循环上界，因此支持动态聚合行数。

**Boxed（NZ）子区域对齐。** L1（`Mem.Mat`）累加器带 matmul 操作数的 NZ 分形布局,`pto.subview` 的 size 必须是内层 box 的整数倍（`M0 = 16` 行；`C0 = fractal_bytes / dtype_bytes / 16` 列）。逐行 gather 只写一行,因此 `tile.gather_row` codegen 发射 **box 对齐的物理 size**（`phys_rows = round_up(1, 16)`,`phys_cols = round_up(size, C0)`），同时只把真实范围标为 valid（`valid = [1, size]`）；`tload` 仅填那一行。UB（`Mem.Vec`,`slayout = none_box`）tile 没有内层 box,使用精确的 `[1, size]` size。聚合后的 L1 tile 由 `tensor.matmul` 直接消费（作为 matmul 操作数的自然用法）。

### 内核驱动的 Gather（`tensor.create_l1` + `tensor.gather_row`）

`tensor.paged_gather` 把每行的源地址写死（`block_table[idx // bs] * bs + idx % bs`）。当内核需要任意的 gather 逻辑——多源选择、无效行 clamp、overlay 池——它可以用两个张量级原语自行构建同样的 L1 累加器，作为 `paged_gather` 的灵活对应版本：

| 算子 | 下降到 | 作用 |
| ---- | ------ | ---- |
| `tensor.create_l1(shape, dtype, transpose=...)` | `tile.create(target_memory=Mat, transpose=...)` | 初始化循环携带的 L1 累加器 |
| `tensor.gather_row(acc, src, dst_off, src_off, shapes, transpose=...)` | `tile.gather_row`（DPS） | 把一条**调用方寻址**的 GM 行 DMA 进 `acc` |

两者都推导出 `TensorType`，因此聚合结果可与张量级 `tensor.matmul` / softmax 组合；两者都注册为 self-loading（`src` 保持为 GM）。调用方自行计算 `src_off` 与 `dst_off` 槽位，在自己的循环里逐行填充累加器。

**转置（ZN）以构造 `b_trans` matmul 操作数。** `transpose=True` 让聚合后的 tile 直接成为转置的 matmul B 操作数，无需 GM 往返：

- `tensor.create_l1(..., transpose=True)` 分配**转置的 Mat（ZN）分形**（`blayout = row_major`,`slayout = col_major`）——即 `b_trans` 操作数所带的布局。
- `tensor.gather_row(..., transpose=True)` 把 GM 行 `[r, c]` 放成 L1 列 `[c, r]`。`pto.tload` 本身不转置,因此 codegen 把 `src` 表示为 **DN 跨步视图**（`pto.make_tensor_view ... {layout = #pto.layout<dn>}`,shape/strides 互换、base ptr 不变）并把该行分区成列——于是 `tload` 执行 `DN → NZ`,这*即是*转置。（`paged_gather` 的 `is_trans=True` 复用同一条 `tile.gather_row` 路径。）直接的 `ND → NZ` `tload` 会打乱分形布局。

## 示例

**转换前**：

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
        return y

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x)
        return y
```

**转换后**：

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.InCore)
    def main_incore_0(
        self, x: pl.Tensor[[64], pl.FP32],
        ret0_out: pl.Out[pl.Tensor[[64], pl.FP32]]
    ) -> pl.Tensor[[64], pl.FP32]:
        x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, (0,), (64,))
        y_tile: pl.Tile[[64], pl.FP32] = pl.tile.add(x_tile, x_tile)
        ret0_store: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, (0,), ret0_out)
        return ret0_store

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        ret0_out: pl.Tensor[[64], pl.FP32] = pl.tensor.create((64,), dtype=pl.FP32)
        y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, ret0_out)
        return y
```

关键变更：

- `pl.add(x, x)` → `pl.tile.add(x_tile, x_tile)`（op 转换）
- 入口插入 `tile.load`，出口插入 `tile.store`
- InCore 函数新增 `Out` 参数 `ret0_out`
- 编排函数调用点插入 `tensor.create`

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

**实现**：`src/ir/transforms/convert_tensor_to_tile_ops_pass.cpp`

**Python 绑定**：`python/bindings/modules/passes.cpp`

**测试**：`tests/ut/ir/transforms/test_convert_tensor_to_tile_ops.py`

## Pass 属性

| 属性 | 值 |
| ---- | -- |
| Required | SSAForm, SplitIncoreOrch, NormalizedStmtStructure |
| Produced | SSAForm, IncoreTileOps, NormalizedStmtStructure |
| Invalidated | — |

## 关键组件

| 组件 | 作用 |
| ---- | ---- |
| `TensorArgsInConvertedOpsCollector` | IRVisitor — 识别需要入口加载的 tensor 参数 |
| `MatmulSlicePatternCollector` | IRVisitor — 查找 slice→matmul 模式以生成 Mat 空间加载 |
| `TypePropagatingMutator` | 基类 IRMutator — 通过控制流传播类型变更 |
| `TensorToTileMutator` | IRMutator — 通过 OpConversionRegistry 将 tensor op 转换为 tile op |
| `ForwardedCallFinder` | IRVisitor — 定位包装函数对转换后 InCore 的调用（阶段二a） |
| `WrapperForwardMutator` | IRMutator — 将新增 Out 参数追加到包装函数的内部调用（阶段二a） |
| `CallSiteUpdateMutator` | IRMutator — 在编排函数调用点插入 tensor.create（阶段二b） |
| `IncoreTileOpsVerifier` | IRVisitor — 验证 InCore 函数中不再包含 TensorType 操作 |

## 作用范围

| 函数类型 | 操作 |
| -------- | ---- |
| InCore | 转换（tensor ops → tile ops）；阶段一可能新增 `Out` 参数 |
| Spmd / Group（转发到转换后 InCore） | 签名镜像 InCore 新增的 `Out` 参数，内部调用转发这些参数（阶段二a） |
| Spmd / Group（未转发到转换后 InCore） | 不变 |
| Orchestration / Opaque | 更新调用点 —— 为每个新增 `Out` 参数插入 `tensor.create`（阶段二b） |
