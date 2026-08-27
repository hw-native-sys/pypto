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

2. **插入 tile.load（入口加载）**：为每个被转换 op 直接使用的 `TensorType` 参数，在函数入口插入 `tile.load(param, zeros, shape, shape, target_memory=Vec)`。仅被自加载 op（`tensor.slice`、`tensor.matmul`、`tensor.read`、`tensor.write`、`tensor.assemble`）引用的参数不会生成额外加载。

3. **通过 TensorToTileMutator 转换函数体**：遍历函数体，使用 `OpConversionRegistry` 将每个 `tensor.*` 调用转换为对应的 `tile.*` 调用。Mutator 通过控制流传播类型变更（IterArgs、ForStmt/WhileStmt return_vars、IfStmt return_vars）。

4. **插入 tile.store（出口存储）**：对每个从 `TensorType` 转换为 `TileType` 的返回值，添加 `Out` 参数并插入 `tile.store(tile, zeros, out_param)`。如果返回值来自 `tile.assemble` 循环，则将循环重写为直接使用 `tile.store`（转换时 assemble-loop 重写；与 `OptimizeOrchTensors` 模式 3 不同，该模式处理跨函数优化）。

5. **升级被写入参数的方向（direction）**：通过别名溯源分析（`AnalyzeCallAccess`）把每次读/写归属到其来源参数，再把被写入的 `In` 参数升级为 `Out`（只写）或 `InOut`（既读又写）。**某个算子写哪个实参不再由本 pass 判定**，而是读取该算子在注册表上的声明（`set_arg_effect`，参见 [算子](../ir/05-operators.md#参数效应argument-effects)）；因此 `tile.store`、`tile.mscatter`、`tensor.write`、`tensor.assemble`、`tensor.expand_clone`、`pld.tile.*` / `pld.tensor.*` 推送与拉取家族、`pld.system.notify`、`system.syncall` 以及复合集合通信都经由同一张表进入本分析。被声明为 `Write` 的实参不计为读：只写入子区域的 store 从不读取未触及的部分。由 kwarg 决定的效应按调用逐个解析，因此原子 store 或 `AtomicAdd` 形式的 notify 会把目的操作数标记为读+写，而普通形式不会。用户已显式声明为 `Out` / `InOut` 的参数保持不变。

   从未声明效应的算子仍然按"读取全部实参"处理。该默认值如今只覆盖注册表无话可说的算子（其中绝大多数是纯函数式的），而不再是一张手工维护清单的兜底——新增的写类算子曾经可以悄无声息地从这张清单里漏掉。

### GM 存储一致性限制

同一个 InCore 函数不能对同一底层 GM 张量同时使用批量存储
（`tensor.assemble`，下沉为 `tile.store`）和 `tensor.write`。批量存储走
MTE3，标量存储走 D-cache；在 A2/A3 上，仅靠 barrier 不能保证两条路径间的
cache-line 一致性。因此本 pass 会拒绝该组合，而不是生成可能静默丢失标量
覆盖值或相邻批量写入字节的代码。

在执行该限制检查前，本 pass 会将简单的常量标量填充循环规范化为
`tensor.full` 加 `tensor.assemble`，随后下沉为 `tile.full` 加 `tile.store`。
该循环必须是从零开始、步长为一的串行循环；循环体只能包含一次或多次
`tensor.write`，不能包含其他操作。每次写入都必须使用一个循环不变量常量覆盖
连续区域，且展平后的区域必须满足 MTE3 的 32-byte 行对齐要求。这样，完整
block 的回退填充会统一走 MTE3 路径，而不会被当作混合存储拒绝。动态值、无法
规范化的局部更新、未对齐区域和跨步标量循环仍走 D-cache 路径。

对于先用 `tensor.full` 初始化完整张量、之后只对该张量执行标量更新的模式，本
pass 也会进行片上暂存：标量更新会改写到本地 tile，函数退出前再通过一次
`tile.store` 将完整结果写入 GM。这样可以支持动态稀疏映射构造，同时避免混用
MTE3 与 D-cache 存储路径。该改写要求初始化从零 offset 覆盖完整 shape，且使用
私有的 `tensor.full` 结果；局部初始化、别名、额外 DMA 存储或 GM 目标的其他
用途仍会被拒绝。

检查会跟踪赋值别名以及循环 / 分支携带值。该限制有意保持保守：即使源码中的
offset 看似不相交，只要两种存储路径写入同一个 GM 张量也会被拒绝，因为编译器
目前还不能跨符号 view 与控制流证明 cache-line 分离。写入不同 GM 张量的混合
路径仍然合法。

```python
# 拒绝：局部切片赋值生成 MTE3 TSTORE，随后 pl.write 走 D-cache。
output[0:1, 0:16] = pl.full([1, 16], dtype=pl.INT32, value=-1)
for i in pl.range(4):
    pl.write(output, [0, i], pl.cast(i, pl.INT32))

# 支持：先更新本地片上值，最后只执行一次 GM 存储。
staged = pl.full([1, 32], dtype=pl.INT32, value=-1)
for i in pl.range(4):
    pl.write(staged, [0, i], pl.cast(i, pl.INT32))
output[0:1, 0:32] = staged
```

如果不适合使用单次批量存储，也可以对所有元素统一使用 `tensor.write`。

### 缓存策略声明 → `tile.load` 的 `cache` kwarg

本 pass 是声明式 GM 缓存策略从元数据变成访问本身的地方。
[`OutlineIncoreScopes`](08-outline_incore_scopes.md) 把这些声明留在 InCore 函数的
`cache_policy` attr 上 —— `std::vector<std::pair<int32_t, int>>`（参数索引，
`CachePolicy` 的 int 值）。阶段一在每个函数上把这些索引一次性还原为参数 `Var` 身份，
随后为每条源实参属于列表中参数的 `tile.load` 加上 `{"cache", <policy>}`：包括它合成的
入口 load、consumer-driven 的 Mat load、输入空间桥接（input-space bridge）load，以及
body 中本就存在的任何 `tile.load`（用户手写的，或更早的 pass 产生的）。该 attr 在重建
变换后的函数时被**擦除** —— 下游不允许看到它，因为只要后续 pass 增长参数列表，其中的
参数索引就会失效。

优先级按单次访问判定：load 上已有的显式 `pl.load(..., cache=...)` kwarg 在两个方向上
都优先于作用域声明，因此 `cache=pl.CachePolicy.DEFAULT` 可以在 bypass 作用域内把某一次
读取单独放回缓存。从这里开始，该 kwarg 只是像 `target_memory` 一样随 op 穿过剩余的
pass 抵达 codegen。参见 [GM 缓存访问策略](../language/05-cache-policy.md)。

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

当 `tensor.slice` 的结果被 `tensor.matmul` 或 `tensor.matmul_acc` 使用时，slice 必须生成 Mat 空间的 tile 而非 Vec 空间。本 pass 预扫描此模式，生成自然的 Mat `tile.load`；转置操作数（LHS 用 `a_trans`，RHS 用 `b_trans`）在 matmul 处叠加零拷贝 `tile.transpose_view`。

该需求会**穿过**声明了 `set_output_memory_inherit_input()` 的零拷贝元数据 op 继续向上传播 —— `tensor.slice`、`tensor.view`、`tensor.reshape`、`tensor.reinterpret_view`、`tensor.set_validshape`。因此 `pl.matmul(pl.set_validshape(a[:, :K], rows, K), b)` 这样的操作数仍然直接加载到 Mat。若某个别名输入存储的 op 漏掉该声明，传播链就会断开：操作数被物化到 Vec，再通过 `tile.move` 桥接到 Mat，而这是一个 vector→cube 边界，会把本应是纯 CUBE 的 InCore scope 判定为 `MIXED`，导致 [`ExpandMixedKernel`](22-expand_mixed_kernel.md) 将其拆分为 AIC/AIV 两个函数。

## Transpose 下沉

`tensor.transpose` 下沉为一个 3-arg 的 **`tile.transpose(input, axis1, axis2)`**。PTO 后端的 `pto.ttrans` 指令需要一个 scratch 工作 tile（与源 tile 同 shape/同 dtype），但该 scratch 纯属 codegen 细节，并非语义操作数。[`FlattenTileNdTo2D`](13-flatten_tile_nd_to_2d.md) 是 scratch 物化的**唯一归属**：它为 2D 以及逐页 >2D 的 transpose 统一产出 codegen-ready 的 4-arg 形态（`tile.create` + `tile.transpose(..., tmp)`），且仍在内存分配器之前（scratch 仍能拿到真实 UB 地址）。把 scratch 从高层 op 中移除后，`tensor.transpose` 与 DSL `pl.tile.transpose(tile, axis1, axis2)` 都与语义操作保持 1:1。

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
| `tensor.gather_row(acc, src, dst_off, src_off, shapes, valid_shape=..., transpose=...)` | `tile.gather_row`（DPS） | 把一条**调用方寻址**的 GM 行 DMA 进 `acc` |

两者都推导出 `TensorType`，因此聚合结果可与张量级 `tensor.matmul` / softmax 组合；两者都注册为 self-loading（`src` 保持为 GM）。调用方自行计算 `src_off` 与 `dst_off` 槽位，在自己的循环里逐行填充累加器。

**动态传输长度（`valid_shape`）。** `shapes` 必须是编译期常量：它会成为 `pto.subview` 的 `sizes`，而 PTO 方言把该字段定义为静态的 `I64ArrayAttr`（`PTOOps.td` 中的 `SubViewOp`）。可选的 `valid_shape` 承载*运行期*范围——它填入 subview 的 `valid_row` / `valid_col`（声明为 `Optional<Index>` SSA 操作数），以及 GM 侧 `pto.partition_view` 的 sizes（接受动态 `?` 维）。因此动态行数既不改变内存分配，也不影响下文的 box 对齐：子区域仍然按 `shapes` 静态定尺寸，只有拷贝长度可变。省略 `valid_shape` 即传输整个窗口，与既有行为一致。

这样一来，长度只有运行期才知道的连续行区间只需一次调用，而不必写成带条件的逐行循环：

```python
kv = pl.create_l1([128, HEAD_DIM], pl.BF16)
# r1 是运行期 Scalar[INDEX]——例如页边界的切分点
kv = pl.gather_row(kv, pool, [0, 0],  [b0, 0], [128, HEAD_DIM], valid_shape=[r1, HEAD_DIM])
kv = pl.gather_row(kv, pool, [r1, 0], [b1, 0], [128, HEAD_DIM], valid_shape=[128 - r1, HEAD_DIM])
oi = pl.matmul(q, kv, b_trans=True)
```

**边界约束的是被写区域，而非窗口。** `shapes` 决定静态 `pto.subview` 的尺寸，因此当 `dst_offset` 为运行期值时，声明出的窗口可能越过目标末尾——上例中 run 2 在 128 行的 tile 上声明了 `[r1, r1 + 128)`。这之所以成立，是因为传输长度由 `valid_shape` 而非 `shapes` 界定：实际写入的行是 `[r1, r1 + (128 - r1))`，仍在范围内。因此调用方的义务是逐维满足 `dst_offset + valid_shape <= dst.shape`，而 `dst_offset + shapes` 无需成立。上述两段式写法已由 `test_gather_row_two_run_split` 在设备上覆盖。

`valid_shape` 在 DSL 中是仅关键字参数——第 6 个位置参数早已属于 `transpose`，占用它会静默改变既有 `gather_row(..., shapes, True)` 调用的含义。在 IR 层它是位置操作数而非 attr，正是因为它可能是运行期值：它必须留在 use-def 链上，SSA / 活跃性分析才会保住这个标量。它与 `transpose=True` 互斥（见下文）——那条路径需要在 boxed NZ tile 上给出运行期*列*范围，尚未在设备上验证。类型推导会拒绝任何*可证明*违反 `0 <= valid_shape[i] <= shapes[i]` 的情形；无法判定的符号范围则予以接受，而这正是该操作数存在的意义。

**转置（ZN）以构造 `b_trans` matmul 操作数。** `transpose=True` 让聚合后的 tile 直接成为转置的 matmul B 操作数，无需 GM 往返：

- `tensor.create_l1(..., transpose=True)` 分配**转置的 Mat（ZN）分形**（`blayout = row_major`,`slayout = col_major`）——即 `b_trans` 操作数所带的布局。
- `tensor.gather_row(..., transpose=True)` 把 GM 行 `[r, c]` 放成 L1 列 `[c, r]`。`pto.tload` 本身不转置,因此 codegen 把 `src` 表示为 **DN 跨步视图**（`pto.make_tensor_view ... {layout = #pto.layout<dn>}`,shape/strides 互换、base ptr 不变）并把该行分区成列——于是 `tload` 执行 `DN → NZ`,这*即是*转置。（`paged_gather` 的 `is_trans=True` 复用同一条 `tile.gather_row` 路径。）直接的 `ND → NZ` `tload` 会打乱分形布局。

## AIV 切分边界下降（`tensor.aiv_shard` / `tensor.aic_gather`）

`tensor.aiv_shard` / `tensor.aic_gather` 是 cube↔vector AIV 切分边界的 **`@pl.jit` / `pl.spmd` 作者面向**形式。当 `pl.aiv_shard(x)` / `pl.aic_gather(x)` 的操作数 `x` 是高层 **Tensor**（例如一个 `pl.matmul` 结果）、且位于 `for aiv_id in pl.split_aiv(...)` 区域内时发射：

```python
raw = pl.matmul(q, k, b_trans=True, out_dtype=pl.FP32)   # Tensor，位于区域外
for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
    h = pl.aiv_shard(raw)     # C->V：tensor.aiv_shard — 整块 [M, N] -> 本 lane 半块 [M/2, N]
    s = pl.softmax(h)         # AIV 向量运算作用于半块
    full = pl.aic_gather(s)   # V->C：tensor.aic_gather — 各半块 -> 整块 [M, N]
oi = pl.matmul(full, v, out_dtype=pl.FP32)               # Tensor，位于区域外
```

本 pass 将两者**各自 1:1**下降为对应的 tile 算子（`tensor.aiv_shard` → `tile.aiv_shard`,`tensor.aic_gather` → `tile.aic_gather`）；此后 IR 与 AUTO `pl.split` 路径经 [`LowerAutoVectorSplit`](21-lower_auto_vector_split.md)（pass 20）产出的结果逐字节一致。随后 `ExpandMixedKernel`（pass 21）将两者折叠进跨核 `tpush`/`tpop` 机制。

**约束**（由张量级类型推导器与 DSL 解析器施加,而非本 pass）：

- **仅 2D** —— `UP_DOWN` / `LEFT_RIGHT` 仅在 2D 物理 tile 视图上有良定义；N 维操作数会被拒绝并给出 `pl.reshape` 到 2D 的提示（N 维张量会被展平为 `[product(leading), last]`,因此展平前的按行切分不会匹配下降实际取用的连续半块）。
- **仅区域内** —— `tensor.*` 形式只能经由 `pl.split_aiv` 区域到达（该区域提供切分模式）。已外联的低层 `pl.tile.aiv_shard(t, split=N)` 形式保持 tile-only；在该形式下传入 Tensor 操作数会被拒绝。
- **拒绝分布式** —— `DistributedTensorType` 操作数超出范围（仅支持 AIV/AIC 切分）,在上游即被拒绝。

**转换细节：**

- **split 关键字透传。** `split` 整型属性（`1` = `UP_DOWN`/axis0,`2` = `LEFT_RIGHT`/axis1,即 tpush/tpop 编码）原样透传给 tile 算子,由其对切分轴长度做减半（shard）或加倍（gather）。
- **边界内存。** tile 级切分推导器有意让边界内存空间保持为空（推导定点不得继承输入侧布局）；随后 `OpRegistry::Create` 会用该 tile 算子 `set_output_memory` 的声明填充它,因此本转换器无需自行重新附着。`LowerAutoVectorSplit` 也通过同一个 `Create` 构造 `aiv_shard` / `aic_gather`,这正是两条路径逐字节一致的原因——一处声明,读取一次。该空间是**消费侧 lane** 的：`tile.aiv_shard` → `Vec`（AIV 将半块 pop 进 UB）,`tile.aic_gather` → `Mat`（AIC 将整块 pop 进 L1,即 `ExpandMixedKernel` 构造 V→C tpop 所用的空间）。操作数侧则相反——shard 为 `Acc`,gather 为 `Vec`——它由 `AivSplitValid` 验证器强制,而非声明为输入约束：输入约束一旦被违反,`InferTileMemorySpace` 会*插入一次 move* 去满足它,而不是报告作者的错误。
- **不合成 load。** 现实（仅区域内）操作数在转换器运行时已是片上 tile（其生产者——`aiv_shard` 对应 cube matmul,`aic_gather` 对应 Vec 向量算子——已在本 pass 更早处下降）,因此不注入 `tile.load`；`aiv_shard` / `aic_gather` **本身**即是跨核传输。

**本 pass 之前即被识别。** 由于 `tensor.*` 形式从 `OutlineIncoreScopes` 一直存活到本 pass 运行,更早的阶段已将其视为 AIV 切分边界：`ClassifyCallAffinity` 把 `tensor.*` 与 `tile.*` 的 shard/gather 都归为 `MIXED`（使 cube/vector 外联正确切分）,`SplitAivStructuralVerifier` 要求两种形式都必须位于区域内。

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

## 循环携带值的 valid_shape 修复

`tensor.matmul` 会丢弃操作数的 `valid_shape`，因此只有当本 pass 把它变成作用于被收窄左
操作数的 `tile.matmul` 之后，累加器才会比它所携带的种子更窄：

```python
acc = pl.create_tensor([M, N], dtype=pl.INT32)          # 完整盒
for k0 in pl.pipeline(0, K, K_TILE, stage=2):
    xk = pl.slice(x, [M, K_TILE], [m0, k0], valid_shape=[v, K_TILE])   # 运行期 v
    acc = pl.matmul_acc(acc, xk, wk, b_trans=True)      # 收窄且 compact 的结果
```

循环携带值**只按其初值定型**——`ConvertToSSA` 用种子铸出 `IterArg`，本 pass 再用转换后的
种子重铸一次，两者都会把循环的 `return_var` 拉回同一类型——于是收窄在循环边界上消失。
`mad` 以 `ceil(v/16)*16` 的 N-fractal 步长写 L0C，而相信完整盒高的读者按物理行步长遍历，
第一个之后的每个 N-fractal 都会被打乱（issue #2470）。

因此本 pass 在返回前会对每个函数调用 `narrow_loop_carry::NarrowAccCarries`：由
`tile.create` 播种的 Acc 携带值会按 yield 可证明的范围重新声明——`tile.create(compact=True)`
加 `tile.set_validshape`——并让循环体的 def-use 闭包经由算子自身的 deducer 重新定型。在制造
问题的 pass 里就地修复，才能保持流水线可验证；否则产出的携带值会被 `TypeCheck` 诊断与
`AccCompactValid` 属性验证器拒绝。`FlattenTileNdTo2D` 调用同一个 helper，用于 ND 种子——
它的收窄要等到 `tile.batch_matmul` 展开成 2D matmul 时才出现。

两种情况下携带值保持原样：一是缓冲区的两种读法本来就不会分歧——单 fractal 块的 `[16, N]`
累加器无论有效行是多少都按物理行打包；二是收窄用的表达式只在循环体内计算，重新声明的种子
在那之前根本命名不到它。

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

**实现**：`src/ir/transforms/convert_tensor_to_tile_ops_pass.cpp`

**Python 绑定**：`python/bindings/modules/passes.cpp`

**测试**：`tests/ut/ir/transforms/test_convert_tensor_to_tile_ops.py`、`tests/ut/ir/transforms/test_narrow_loop_carry_valid_shape.py`（携带值修复）

## Pass 属性

| 属性 | 值 |
| ---- | -- |
| Required | SSAForm, SplitIncoreOrch, NormalizedStmtStructure |
| Produced | SSAForm, IncoreTileOps, NormalizedStmtStructure, AivSplitValid |
| Invalidated | AivSplitValid |

`AivSplitValid` 同时被失效并重新产生，从而在此处强制对 split 区域再验证一次。`OutlineIncoreScopes` 建立该属性时，AIV split 边界还是 `tensor.aiv_shard` / `tensor.aic_gather`；TensorType 不携带 memory space，因此验证器的边界内存契约检查在那里必然被跳过。本 Pass 把这些算子改写为 tile 形式并附上声明的边界内存，而这正是该项检查所要检视的内容。

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
