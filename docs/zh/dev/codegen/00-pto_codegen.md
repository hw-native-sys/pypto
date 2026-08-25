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

**当发现代码生成中存在分析逻辑时：** 创建跟踪 Issue，在有带宽时将其重构为专用 Pass。[#814](https://github.com/hw-native-sys/pypto/issues/814) 就是一个实例：编排代码生成中的返回值到参数追踪逻辑已重构为 [`NormalizeReturnOrder`](../passes/26-normalize_return_order.md) pass。

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

张量视图与分配前缀会**先**渲染到缓冲区、再定稿常量块, 因此只出现在某个 shape 或
stride 表达式里的常量 (例如复合参数维度 `M * 2` 中的 `2`) 也会在使用前被声明到常量块。

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

# Compile with PTO backend
output_dir = compile(
    MyKernel,
    strategy=OptimizationStrategy.Default,
    backend_type=BackendType.Ascend910B,
)
```

`compile()` 函数会自动应用选定的优化策略, 并根据 `backend_type` 调用相应的代码生成器。
`Default` 是唯一的优化策略。

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
| `tile.set_validshape(tile, vr, vc)` | 发 `pto.set_validshape`；操作数是视图时报错（见下） |
| `tile.mul(lhs, rhs)` | `pto.tmul` |
| `tile.add(a, b, c)` | `pto.taddc` (三操作数加法) |
| `tile.adds(tile, scalar)` | `pto.tadds` (Tile + 标量) |
| `tile.fillpad_expand(src, shape)` | `pto.tfillpad_expand ins(%src) outs(%dst)`（`shape` 元组仅用于类型推导；更大的 `dst` 及其 pad 来自结果类型） |

**`tile.slice` / `tile.assemble` 下沉细节。** 两个 op 都通过 `pto.subview`
下沉，它是源 tile 的纯视图别名（不搬数据，也不会额外发 `pto.alloc_tile`）。
`pto.subview` 要求结果 `tile_buf` 与源 `tile_buf` 在 `dtype`、`memory_space`、
`blayout`、`slayout`、`fractal`、`pad` 和 `compact` 上完全一致，因此
`DeduceTileSliceType` 会将源 `TileView` 的这五个字段透传到结果，使新生成的
`TileType` 天然满足约束。后端 codegen 还会在下沉时执行 `CheckSubviewTileCompat`
做兜底校验：

- 源和结果都必须显式携带 `TileView`。
- `dtype`、`blayout`、`slayout`、`fractal`、`pad` 与 `compact` 必须严格相等。
- `pad` 必须为 `PadValue::null`——`pto.subview` 是视图而不是 fillpad；如果
  需要 zero/min/max 填充，请在切出来的子 tile 上再调用 `tile.fillpad`。

对 `tile.assemble`，前置的 `pto.tmov target → dst` 仅在缓冲复用未把
`target` 与目标缓冲合并时才会发出，用于保留写入窗口外的数据；末尾的
`pto.tmov src → dst_view` 才是真正写入由 `pto.subview` 切出的子窗口的数据
搬运。

**`tile.set_validshape` 下沉细节。** `pto.set_validshape` 修改的是操作数的
`valid_row` / `valid_col` 操作数，因此操作数必须是拥有它们的 handle：alloc、
`scf.if` 结果、跨核 pop slot。而**视图**——`tile.slice` 下沉出的 `pto.subview`，
或 `pto.treshape`——把有效范围存在自身类型里，ptoas 会拒绝对它执行这条指令，所以
PyPTO 提前报错，并在信息里指向切片。视图身份在这两个发射点被记录，而不是从渲染出
的维度推断：带运行时 `valid_shape` 的切片渲染成 `v_row=?, v_col=?`，与由 alloc
承载的 handle 完全一样。要收窄视图，请给切片传 `valid_shape=`（它会落到
`pto.subview` 的 `valid [...]` 子句，并且支持运行时范围），或在取视图之前对源
tile 调用 `set_validshape`。

### 跨核操作到 PTO 指令

| PyPTO 操作 | 生成的 PTO-ISA | 描述 |
| ---------- | -------------- | ---- |
| `tile.tpush_to_aiv(tile, split=N[, id=I])` | `pto.tpush_to_aiv ins(%tile : type) {[id = I, ]split = N}` | Cube → Vector 推送 |
| `tile.tpush_to_aic(tile, split=N[, id=I])` | `pto.tpush_to_aic ins(%tile : type) {[id = I, ]split = N}` | Vector → Cube 推送 |
| `tile.tpop_from_aic(split=N[, id=I])` | `%buf = pto.tpop_from_aic {[id = I, ]split = N} -> type` | 从 Cube 管道弹出 |
| `tile.tpop_from_aiv(split=N[, id=I])` | `%buf = pto.tpop_from_aiv {[id = I, ]split = N} -> type` | 从 Vector 管道弹出 |
| `system.tfree_to_aic(tile_from_tpop[, id=I])` | `pto.tfree_from_aic {[id = I, ]split = N}` | 将消费侧槽位释放回 Cube |
| `system.tfree_to_aiv(tile_from_tpop[, id=I])` | `pto.tfree_from_aiv {[id = I, ]split = N}` | 将消费侧槽位释放回 Vector |
| `system.aic_initialize_pipe(...)` | `pto.aic_initialize_pipe {[id = I, ]dir_mask = D, slot_size = S[, slot_num = N][, local_slot_num = L]} (c2v_consumer_buf = %ssa : i32, v2c_consumer_buf = %ssa : i32)` | Cube 管道初始化（仅在显式设置时输出 `slot_num`/`local_slot_num`，否则由 PTOAS 取默认值） |
| `system.aiv_initialize_pipe(...)` | `pto.aiv_initialize_pipe {[id = I, ]dir_mask = D, slot_size = S[, slot_num = N][, local_slot_num = L]} (c2v_consumer_buf = %ssa : i32, v2c_consumer_buf = %ssa : i32)` | Vector 管道初始化（仅在显式设置时输出 `slot_num`/`local_slot_num`，否则由 PTOAS 取默认值） |
| `system.reserve_buffer(...)` | `%name = pto.reserve_buffer {name = "N", size = S, location = #pto.address_space<loc>, auto = false, base = B} -> i32` | 预留缓冲区（`memory_planner=PTOAS` 下发射 `auto = true` 且省略 `base`） |
| `system.import_peer_buffer(...)` | `%name = pto.import_reserved_buffer {name = "N", peer_func = @F} -> i32` | 导入对等缓冲区 |
| `system.syncall(core_type=C)` | `pto.syncall() mode = #pto.sync_all_mode<hard>, core_type = #pto.sync_core_type<C>` | 跨核全员屏障（hard/FFTS 形态） |
| `system.syncall(mode="soft", core_type=C, gm_workspace=ws, used_cores=N)` | `pto.syncall(%gm_ptr[, %used] : !pto.ptr<i32>[, i32]) mode = #pto.sync_all_mode<soft>, core_type = #pto.sync_core_type<C>` | 当前 PTO-ISA 的 soft/GM 轮询屏障（部分占用即可；GM workspace 至少 64 字节；显式 `N=0` 时从设备启动寄存器推导并省略 `%used`） |

**说明：**

- Push 操作使用带类型 tile buffer 的 `ins()` 子句；前端 Pop 操作生成 SSA 结果，并带 `-> !pto.tile_buf<...>` 结果类型
- `id` 是可选属性。省略时 PTOAS 默认使用 frontend pipe id `0`。只有手写多条独立 frontend pipe 时才需要显式 `id`；自动生成的双向 mixed-kernel setup 会保持单条 `dir_mask = 3` pipe。
- 如果被 push 的 tile 通过动态 `valid_row` / `valid_col` operand 分配，或经
  `tile.set_validshape` 更新，`tpush` 会发射已经更新运行时 valid shape 的同一个
  tile handle。对于 split `tpush`，codegen 会临时使用完整物理传输 box，随后恢复
  producer tile 的逻辑 valid shape。
- `split` 就是 pto-isa 的 `TileSplitAxis`，原样打印。`0` = 不切分，`1` / `2` = 上下 /
  左右，`3` / `4` = 同样两个轴、但 extent 为**奇数**：

  | Code | pto-isa | Lane 0 | Lane 1 | Lane 1 的数据段起点 |
  | ---- | ------- | ------ | ------ | ------------------- |
  | 0 | `TILE_NO_SPLIT` | 整块 tile | （单一读者） | — |
  | 1 / 2 | `TILE_UP_DOWN` / `TILE_LEFT_RIGHT` | `e0` | `e1` | `e1 * pitch` |
  | 3 / 4 | `TILE_UP_DOWN_ODD` / `TILE_LEFT_RIGHT_ODD` | `e0` | `e1` | `(e1 + 1) * pitch` |

  `eL` 是 lane `L` 在切分轴上的**运行时** valid extent——ISA 直接从被弹出的 tile 上读取
  （`popVecTileFromGMFiFo`），因此偶数 code 要求 `e0 == e1`，奇数 code 要求
  `e0 == e1 + 1`。这些 extent 由
  [LowerAutoVectorSplit](../passes/21-lower_auto_vector_split.md) 物化，
  [ExpandMixedKernel](../passes/22-expand_mixed_kernel.md) 选择匹配的 code。
- Cube-to-Vector FIFO 搬运的是紧凑矩形：producer 以 `valid_col` 为行间距写入
  `valid_row` x `valid_col` 数据块，每个消费 lane 再以相同间距读回自己的数据段
  （`gmStrideR = valid_col`，左右切分的 code 下加倍）。因此若传输两侧的 valid shape
  不一致，pop 的 stride 就会错位——这会静默破坏*有效*区域的数据，因为 ISA 中对应的
  断言在 release 构建里被编译掉了。所以部分有效的 Acc-to-Vec 传输无论是否切分，
  TPOP 以及 TPUSH 的**列**维度都使用完整物理 box，并在传输两侧立即恢复逻辑 valid
  shape——消费侧通过纯元数据的 `pto.treshape` 恢复（前端 tpop 结果不是 PTOAS 的本地
  绑定 tile，`pto.set_validshape` 无法就地修改它）。第一个例外是切分轴上的 extent：它必须保持
  逐 lane 的值并留在 TPOP 操作数上，因为 ISA 正是靠它定位 lane 1 的数据段起点——也正因如此，
  编译期无法核验的逐 lane extent 绝不能到达那里。`pl.split_aiv` 区域中切分轴 extent 为运行期
  值的边界，会让被弹出的 tile 保留完整 box（`split_axis::WithFullSplitAxisValid`），使偶数
  code 的数据段落在 box 的一半处，而把 lane 自身的 extent 交给消费者携带。
- 第二个例外是**非切分** Acc-to-Vec TPUSH 的**行**维度：它必须保持 producer 写入时
  的值。TPUSH 执行的是 L0C 上的 `TStoreAccNz2nd`，其源 pitch 对 compact tile 为
  `ceil(validRow/16)*16`，否则为 `TileData::Rows`；而 `mad` 是按 L0A 操作数的**有效**
  行数所隐含的 pitch 写出乘积的。在 push 之前把 `validRow` 撑大到物理 box，会让该
  pitch 改按 box 推导，于是 fix-pipe 以 `mad` 从未写入过的 stride 遍历 L0C——64 行的
  box 只有 16 行有效时，fractal `j` 会读到 `4j`（issue #2510）。超出 `validRow` 的行
  会在 slot 中保留旧数据，这正是窄化 `valid_shape` 对其无效区域给出的承诺，同时传输
  量也从整个 box 降到 `validRow` 行。
- **切分**的 Acc-to-Vec 传输走不了这条路：lane 1 从 box 一半处开始读自己的数据段，而
  该数据段只有在 producer 写满整个 box 时才存在——但写满 box 就意味着按物理 pitch 读
  L0C，而那并不是 `mad` 使用的 pitch。二者互斥，因此行窄化的 compact 累加器跨
  `pl.split` / `pl.split_aiv` 边界时会被**拒绝**，并给出指明两种 DSL 替代写法的报错
  （窄化结果而非操作数，或让累加器经 GM 中转），而不是下降成静默错位的数据——在加入该
  拒绝之前，设备上实测 8192 个元素中有 1808 个是错的。该拒绝以 pitch 确实不同为前提，
  因此单个 fractal 行块的累加器（`ceil(validRow/16)*16 == Rows`）仍可照常跨越。
- 当 tpop 结果的 `TileView.valid_shape` 与物理 tile shape 不一致时，PTO codegen 会生成 PTOAS 前端操作数：`%buf = pto.tpop_from_*(%valid_row, %valid_col) {[id = I, ]split = N} -> !pto.tile_buf<..., v_row=?, v_col=?, ...>`。这同时覆盖动态表达式和 `[0, 0]` 这类静态非满形状；operand 携带后续计算和 store 使用的逻辑范围。对于静态形状、非空的部分 pop，上述 Cube-to-Vector 完整 box 传输优先，因为 `pto.treshape` 不带 valid-row/valid-col operand，只能恢复*静态*逻辑范围。
- 对于手写 pop 的 split consumer，`SplitVectorKernel` 会按 subblock 本地化这些动态
  tpop valid-shape operand（例如 `[16, 16]` tile 做上下切分时，全局
  `[8, 16]` 会变成 `[8, 16]` 和 `[0, 16]`）。奇数切分轴走同一条路径——`[17, 128]`
  的 tile 在 `split = 3` 下，lane 0 弹出 `[9, 128]`，lane 1 弹出 `[8, 128]`。
- `system.tfree_*` 的 `split` 来自其 tile 参数，因此前端必须释放由 `tile.tpop_*` 产生的那个确切 SSA 值，即使 PTO 指令本身并不显式接收该 tile 作为操作数
- `ExpandMixedKernel` 现在会在 split 生成的消费侧 `tile.tpop_*` 之后自动补 `system.tfree_*`，保持 `tpop -> direct users -> tfree -> next tpop`
- `reserve_buffer` 和 `import_reserved_buffer` 返回 `i32` SSA 值；`initialize_pipe` 以操作数引用这些值
- `memory_planner=PYPTO` 或 `DSA_RP` 时，`AllocateMemoryAddr` 会在 PTO 输出前
  解析 `reserve_buffer(base=AUTO)`，因此 PTO 输出
  `auto = false, base = <value>`；`memory_planner=PTOAS` 跳过该 pass，PTO 输出
  `auto = true` 且省略 `base`（ptoas 不接受两者同时出现），由 ptoas
  `PlanMemory` 放置该预留区
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
- 常量 (`%c32_index`, `%c1_index`) 自动生成, 包括只出现在复合 shape/stride 表达式里的常量
- 复合维度下沉为算术运算, 例如 `M * 2` → `arith.muli %M, %c2_index`
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

#### 由谁规划内存：`compile(memory_planner=...)`

物理 `addr` 由谁分配，通过 `memory_planner` 选项选择
（`ir.compile(..., memory_planner=passes.MemoryPlanner.PYPTO | DSA_RP | PTOAS)`，
默认 `PYPTO`）。它同时作用于 pass 流水线（经 `PassContext`）与 codegen：

| 模式 | 流水线 | `pto.alloc_tile` | `pto.reserve_buffer` | ptoas |
| ---- | ------ | ---------------- | -------------------- | ----- |
| `PYPTO`（默认） | 运行 `MaterializeSemanticAliases` + `MemoryReuse` + `AllocateMemoryAddr` | 发射 `addr = <const>`（来自 `MemRef.byte_offset_`） | `auto = false, base = <const>` | `--pto-level=level3`（信任已烘焙地址） |
| `DSA_RP` | 运行 `MaterializeSemanticAliases` + `AllocateMemoryAddr`；跳过 `MemoryReuse` | 发射进程内 canonical-greedy DSA-RP 的 `addr = <const>` | `auto = false, base = <const>` | `--pto-level=level3`（信任已烘焙地址） |
| `PTOAS` | 运行 `MaterializeSemanticAliases`；**跳过** `MemoryReuse` + `AllocateMemoryAddr` | 省略 `addr`（`PTOCodegen.generate(emit_tile_addr=False)`） | `auto = true`（不带 `base`） | `--pto-level=level2`（ptoas `PlanMemory` 做复用 + 定址） |

内存规划拆成两个 pass：**`MaterializeSemanticAliases`** 把**语义强制**的别名
（循环累加器、原地算子）归一到同一 MemRef；**`MemoryReuse`** 只做**机会性**的、
基于生命周期的独立 buffer 合并，仅供 `PYPTO` 使用。`DSA_RP` 跳过该合并，
在 `AllocateMemoryAddr` 中按容量与复用惩罚放置独立身份。
`InitMemRef` + `MaterializeSemanticAliases` 三种模式都运行，因此强制别名得以
保留；`PTOAS` 模式由 ptoas `PlanMemory`（level2 强制要求、拒绝任何 `addr`
操作数）完成生命周期复用与地址分配。

> **注意：** `PTOAS` 模式跳过了 `MemoryReuse` 里的 Ascend910B `load + tpop_from_aic`
> 原地写冒险守卫,以及 `AllocateMemoryAddr` 的 reserve-buffer 基址解析,这些交由
> ptoas 处理。`compile()` 会输出告警 —— 相关 kernel 请上机验证。

#### 多槽位声明映射为一块 ptoas 区域（`PTOAS` 模式）

多槽位的声明式分配（`pl.MemRef(slots=N)`，见
[Python 语法](../language/00-python_syntax.md#槽位)）不会被降为 N 条 `alloc_tile`，而是
对应到 ptoas 自己的多缓冲二元组：函数头声明一块区域，每个使用点选一个槽位：

```mlir
%l0c_mb = pto.alloc_multi_tile valid_row = %c64_index valid_col = %c64_index
        : !pto.multi_tile_buf<!pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=64, ...>, count=2>
scf.for %i = %c0_index to %c4_index step %c1_index {
  %0 = arith.remsi %i, %c2_index : index
  %t = pto.multi_tile_get %l0c_mb[%0]
     : !pto.multi_tile_buf<..., count=2> -> !pto.tile_buf<loc=vec, ...>
  ...
}
```

有两点关键：

- **不带 `addr`。** 区域由 ptoas `PlanMemory` 放置，且它被禁止合并这些槽位——这正是把作者
  声明的隔离带进 `level2` 的方式。
- **操作数是槽位下标，而不是** `InitMemRef` 由它算出的字节偏移。ptoas 通过匹配下标的仿射形态
  （`i % 2`）判断哪些访问可能落在同一槽位，这才是轮转能拿到按槽位的（动态）event id 的原因
  ——第 *i* 轮的 load 由此与第 *i-1* 轮的计算重叠。

`PlanMultiBufferRegions` 在遍历函数体之前判定适用性；ptoas 无法描述的形态（各槽位 tile 形状
不一致、各槽位声明的 valid shape 不一致、循环内有两个槽位同时活跃、内存空间不属于
Vec / Mat / Acc、valid shape 是运行期值、某个槽位作为 phi 被带出 `if` 或循环、槽位数不在
ptoas 的 `[2, 16]` 内）会报 `ValueError` 并指明具体形态，因为回退成逐槽位
`alloc_tile` 会让 ptoas 有机会把这些槽位规划到同一块内存上。

**每轮迭代只用一个槽位。** 共活槽位被拒绝，不是因为 ptoas 无法为它*定型*，而是无法为它
*同步*：ptoas 0.54 只为一轮迭代中的**第一个** `multi_tile_get` 推导逐槽位 WAR 保护；有两个时，
第二个 load 前面不会发出任何 `wait_flag`，于是下一轮迭代会在本轮还在读该槽位时覆盖它。真机上
实测算错，因此代码生成直接拒绝该形态并指向 PyPTO planner——那里由固化地址和 PyPTO 自己发射的
同步来处理。直线代码不受影响：没有循环就没有跨迭代复用需要保护。已报
[PTOAS#1118](https://github.com/hw-native-sys/PTOAS/issues/1118)；修好后放宽只需改
`PlanMultiBufferRegions` 里一个条件。

`PYPTO` 模式下则完全不发射区域：在 `--pto-level=level3` 下 ptoas 不会折叠逐槽位的地址展开，
区域形式反而会丢掉它赖以存在的槽位分析
（[PTOAS#1106](https://github.com/hw-native-sys/PTOAS/issues/1106)）。

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

### 源码位置 (`loc`)

每条生成的**操作**都会带上由 IR `Span` 构造的 MLIR 尾随位置, 例如
`pto.tadd ins(...) outs(...) loc("kernels/attn.py":41:9)`。ptoas 会原样把
`loc()` 传递到自己的诊断信息里, 因此校验失败时报告的是用户 `.py` 中的行, 而不是
生成的 `.pto` 中的行 —— 在 `@pl.jit` 下用户根本看不到后者 (该路径上 span 已由
解析器从合成的 `<jit:name>` 文本重映射回真实源文件)。

**使用哪个 span** —— 在两个层级绑定, 后者细化前者:

| 层级 | 绑定位置 | 来源 |
| ---- | -------- | ---- |
| 语句 (主) | `PTOCodegen::VisitStmt` | `Stmt::span_` |
| Call (细化) | `PTOCodegen::VisitExpr_(CallPtr)` | `Call::span_`, 仅当它嵌套在语句 span 内 |

包含性 (containment) 检查是正确性的关键。`Call::span_` 在被保留时精确到列, 但
合成 tile 算子的 pass (`ConvertTensorToTileOps`) 会用所在**函数**的 span 重建
`Call`, 同时保留 `AssignStmt` 自身的 span。这类 span 起始于语句之前, 包含性检查
不通过, 于是被丢弃并回退到语句 span —— 否则大多数操作都会指向 `def` 行。

**不带位置的内容**: 区域花括号、分隔符和基本块标签 (`loc(...)` 只在一条完整操作
的末尾合法, 因此这些行走 `EmitStructural()` 而不是 `Emit()`); 常量段中的
`arith.constant` (在所有使用点之间去重, 没有唯一正确的 span); 以及 span 未知或
没有文件名的节点。

**关闭方式** —— `Generate(program, emit_tile_addr, emit_source_loc)`、
`compile(..., emit_source_loc=...)`, 或环境变量 `PYPTO_EMIT_PTO_LOC=0`。关闭后
输出与不带位置的形式逐字节一致; 由于 ptoas 独立于 PyPTO 发布, 这是应对某个
ptoas 版本解析器拒绝尾随位置时的应急开关。

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

生成的 `alloc_tile` 操作从 TileType 元数据推导数据类型和维度，并从关联的 TileView
推导布局/分形/填充/紧凑模式（如有）：

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
  fractal=512,         // Fractal size in bytes, not elements (from TileView, default: 512)
  pad=0,               // Pad mode as int (from TileView, default: 0/null)
  compact=1            // Optional compact mode (normal=1; null=0 时省略)
>
```

**TileView 推导的属性**:

| 属性 | 来源 | 枚举值 | 默认值 |
| ---- | ---- | ------ | ------ |
| `blayout` | `TileView::blayout` | `none_box`, `row_major`, `col_major` | `row_major` |
| `slayout` | `TileView::slayout` | `none_box`, `row_major`, `col_major` | `none_box` |
| `fractal` | `TileView::fractal` | uint64 | `512` |
| `pad` | `TileView::pad` | `null(0)`, `zero(1)`, `max(2)`, `min(3)` | `null(0)` |
| `compact` | `TileView::compact` | `null(0)`, `normal(1)` | `null(0)` |

当 MemRef 没有关联 TileView 时，代码生成器使用上表中的默认值。默认的 null
`compact` 属性不会输出。有两条路径会自动设置 `normal(1)`：

- 进入 L0A/L0B 的部分 `tile.extract`，使 TEXTRACT 仅传输逻辑 `valid_shape`，
  而不会把 box 对齐填充当作数据。
- **有效行数无法证明等于物理行数的 Acc (L0C) tile**，由 `tile.matmul`、
  `tile.matmul_bias`、`tile.matmul_mx` 的类型推导产生。`mad` 始终以
  `ceil(validRow/16)*16` 的 N-fractal stride 写出乘积，其中 `validRow` 取自 **lhs**
  的有效行数；而所有 Acc 读取方在 tile 非 compact 时都按编译期物理 `Rows` 推导
  stride。缺少该标记时，运行期窄化的累加器读回时使用的 pitch 与写入时不同。只有
  **行** 维度决定这一点 —— ISA 推导的每个 Acc stride 都只是 `validRow` 的函数，
  因此仅窄化列维度时保持非 compact 形式。

  compact 只在**确立累加器布局的那一处**盖章，绝不在它的别名上重新推导。
  `tile.matmul_acc`（以及 `matmul_mx_acc`）**继承**累加器操作数的模式，因为该 op
  原地复用那块 buffer，而 codegen 只有在两者完整 tile 配置一致时才会做别名。
  `tile.set_validshape` 同样是继承：它只改元数据，且可能在 buffer **写入之后**才
  执行，因此读取方必须使用的仍是 `mad` 当初写入时的 pitch —— 若按窄化后的行数重新
  推导，就会以从未重排过的方式重新解释这些字节。

  buffer 也可以在创建时**声明**该模式：`tile.create(..., target_memory=Acc,
  compact=True)`。新建的 L0C buffer 没有既有字节可供重新解释，因此这不是别名上的重新
  推导——`AutoTileMatmulL0` 在切分 K 时正是这样声明它合成的累加器种子：
  `tile.matmul_acc` 会继承该种子的模式，种子若非 compact，就会把整条累加链以及循环之后
  的读取方一起拖回物理 pitch。声明也是唯一能存活的形式：pass 盖在 call 上的类型，会在
  任何后续 pass 重新推导时被丢弃（`InferTileMemorySpace` 就会），而 kwarg 每次都会被
  重新读取。

  `AccCompactValid`（见 [Verifier](../passes/99-verifier.md)）会校验该契约的两半：当 `mad`
  的 pitch 与累加器物理行数不同时，每个 `tile.matmul_acc` 都必须累加进 compact 的 buffer；
  且 Left/Right/Acc 之外的任何 tile 都不得携带 compact 模式。

注意：在 a2a3 与 a5 上，PTO-ISA 的 Acc → L1 读取方（`TExtractAccToMat`、
`TMovCcToCb`）都没有 `CompactMode` 分支，因此运行期窄化的累加器若经
`tile.extract` / `tile.move` 进入 L1，仍会按物理 `Rows` pitch 读取。该缺口需要
PTO-ISA 侧配套修改。

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
├── ptoas_passes/                    # 可选：每个 ptoas Pass 后的 IR
│   └── <kernel-or-group>/            # 由 ptoas/MLIR 管理的转储树
├── ptoas/                           # Intermediates
│   ├── <func_name>.pto              # MLIR from PTOCodegen
│   └── <func_name>.cpp              # C++ from ptoas
├── kernels/aiv/
│   └── <func_name>.cpp              # Final wrapper
├── orchestration/
│   └── <orch_func_name>.cpp         # simpler runtime orchestration code
└── kernel_config.py                 # Runtime/orchestration/kernel config
```

仅当使用 `ir.compile(..., dump_ptoas_passes=True)` 或
`RunConfig(dump_ptoas_passes=True)` 时才会生成 `ptoas_passes/`。

编排代码生成使用 simpler 运行时 API (`rt_submit_task`, `make_tensor_external` 等) 生成编排 C++ 代码。

### 运行时配置 (`kernel_config.py`)

`kernel_config.py` 暴露一个 `RUNTIME_CONFIG` 字典，调用方 (如 `execute_compiled`) 据此派发程序。固定键：

| 键 | 何时写入 | 备注 |
| -- | -------- | ---- |
| `runtime` | 总是 | `"tensormap_and_ringbuffer"`（默认）或 `"host_build_graph"` —— 由 `ir.compile(runtime=...)`（或把调用包在 `PassContext([], runtime=...)` 中）选定的 `RuntimeKind` 所对应的线上名字。 |
| `aicpu_thread_num` | 总是 (`0`) | `0` 选择 runtime 的架构默认值（a2a3：4；a5：5），调用方也可显式覆盖。 |

runtime 由 `PassContext` 以 `ir::RuntimeKind` 携带，而不是仅作为 codegen 参数，
这样需要针对特定 runtime 做合法化的 pass 可以 switch `PassContext::GetRuntime()`
而不是比较字符串。之所以用枚举而非名字：这是一个封闭集合 —— `runtime/src/<arch>/
runtime/` 下每个实现对应一个枚举值 —— 于是拼错是编译错误，而不是一个要到很晚才以
晦涩 CCEC 报错（找不到 include 目录）浮现的取值。

线上名字只在两处跨越 ABI 边界：写 `kernel_config.py` 时用 `ir::RuntimeKindToName`，
读回时用 `ir::RuntimeKindFromName`。两者都以
`passes.runtime_kind_to_name` / `passes.runtime_kind_from_name` 暴露给 Python。

runtime 同时是 `@pl.jit` 缓存键的一个维度：`host_build_graph` 的调用不能复用为
`tensormap_and_ringbuffer` 编译出的产物 —— 后者 `kernel_config.py` 里写的
runtime 没有任何匹配的 worker 会绑定。

### 参数解包

包装器按照标准约定解包 `int64_t* args`:

| 参数类型 | 解包模式 |
| -------- | -------- |
| `TensorType` | `ChipTensor*` -> `buffer.addr` -> 带类型指针 |
| `ScalarType` | `uint64_t` -> 联合体解码 -> 带类型值 |

### SPMD 身份参数

`tile.get_block_idx()`、`tile.get_block_num()` 和 `tile.get_subblock_idx()`
在 codegen 阶段被降阶为合成 `i32` 形参，PTOCodegen 把它们**追加到**
`func.func` 签名末尾，并使用有意义的命名 SSA(`%__pypto_spmd_block_idx`、
`%__pypto_spmd_block_num`、`%__pypto_spmd_subblock_idx`)。这些 op 的 IR
契约不变 -- 合成形参只出现在生成的 MLIR / C++ 中，绝不进入
`Function.params`。追加顺序固定为 `block_idx, block_num, subblock_idx`，
并各自根据函数实际使用的 op 独立决定是否追加。

```mlir
func.func @spmd_kernel(%arg0: !pto.ptr<f32>, %arg1: !pto.ptr<f32>,
                       %__pypto_spmd_block_idx: i32,
                       %__pypto_spmd_block_num: i32)
                       attributes { ... } {
  %0 = arith.index_cast %__pypto_spmd_block_idx : i32 to index
  // ... 把 %0 当作 block 索引使用 ...
}
```

kernel 包装器在 dispatch 时调用 `intrinsic.h::get_block_idx(args)` /
`get_block_num(args)` 一次解析出运行时值，并把它们作为最后两个实参传给
被包装的函数:

```cpp
extern "C" __aicore__ __attribute__((always_inline))
void kernel_entry(__gm__ int64_t* args) {
    // 从运行时 dispatch payload 读取逻辑 SPMD block 身份
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // ... 张量 / 标量 / 动态维参数解包 ...

    // 转发到 ptoas 生成的函数(block 参数追加在末尾)
    spmd_kernel(a, out, __pypto_spmd_block_idx, __pypto_spmd_block_num);
}
```

**subblock_idx(AIV lane)。** `tile.get_subblock_idx()` 走相同的合成形参通道:
wrapper 从 `intrinsic.h::get_sub_block_id(args)`(调度器写入
`GlobalContext.sub_block_id` 的运行时 per-core lane id)解析出值,并把
`__pypto_spmd_subblock_idx` 追加在 block 身份实参之后。它刻意读取运行时
lane id,而非 ccec `get_subblockid()` 寄存器 -- 后者在
`tensormap_and_ringbuffer` 调度下返回过期值。这与 A2A3 dual-AIV wrapper 为
ptoas **内部** pipe-slot 偏移安装的 `get_subblockid()` 宏桥接
(`pypto_runtime_subblock_id`)相互独立、并存。与 block 身份一样,它无条件发射
(无 `__CPU_SIM` 分叉),因为 `GlobalContext.sub_block_id` 在每个平台都由调度器
填充。

**检测范围。** 两层各自基于函数体独立检测 SPMD usage:

- `MemRefCollectorVisitor::UsesSpmdBlockOps` / `UsesSubblockOp`(C++，位于
  `src/codegen/pto/pto_codegen.cpp`)决定 PTOCodegen 是否给该函数签名追加
  block / subblock 形参。
- `_uses_spmd_block_ops` / `_uses_dynamic_subblock_id`(Python，位于
  `python/pypto/backend/pto_backend.py`)决定 wrapper 是否把相应局部变量追加到
  对内函数调用末尾。

对于 SPMD 组内自身不调用 `tile.get_block_*` 的 sibling 函数
(`group_uses_spmd=True` 但函数本身不用 SPMD ops)，wrapper 仍会声明这两个
局部变量供 `_generate_arg_unpacking` 中 `__gm_pipe_buffer` 分片逻辑消费，
但**不会**把它们追加到对内调用 -- 这与该函数 MLIR 签名保持一致。

此设计替换了旧的宏 shadow + `[[block_local]] static` /
`static thread_local` 桥接以及 `#pragma push_macro` / `#undef` /
`pop_macro` 舞步。block 身份现在与张量指针、标量参数、动态维一样
通过调用图正常传递。

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
- [PTOAS Op 状态矩阵](../ptoas-op-status.md): 每个 PTOAS op 的前端 / ST 覆盖状态
