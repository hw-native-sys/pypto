# BlockNzTensorViews Pass

## 概述

`BlockNzTensorViews` 把*逻辑* `pl.NZ` 张量改写成 pto-isa `Layout::NZ`
GlobalTensor 所要求的*分块*形式，并同步改写读取它的 `tile.load`。

`pl.Tensor[[R, C], dtype, pl.NZ]`（或 `[[B, R, C], ...]`）标注是一条**关于 GM 中现有字节的断言**：
这些字节已经按 PTO 原生 NZ 分形序存放。它**不是**"请帮我转换"的请求。DSL 层保持
逻辑 shape 和逻辑切片不变，本 pass 负责补上后端需要的物理描述。

收益是 matmul B 操作数的 `TLOAD` 从 ND→NZ 变成 NZ→NZ，去掉了每次权重载入时的
在线分形转换。

## 分块形式

记 `c0` 为一条 32 字节 C0 线所含的元素个数（`256 / dtype 位宽`，`INT8` 时为 32），分形行数为 16。pto-isa 对 NZ
缓冲的描述为（`pto/common/pto_tile.hpp` 中 `TileShape2D` / `BaseShape2D` 的
`Layout::NZ` 特化）：

```text
shape   = [B, C/c0, R/16, 16, c0]
strides = [C*R, R*c0, 16*c0, c0, 1]
```

从内往外读：`c0` 个连续元素构成一条 32 字节 C0 线，16 行构成一个 `16 x c0` 分形
（512 字节），`R/16` 个分形沿行轴排列，`C/c0` 在列块之间步进。即"列块在外、
行分形在内"——与 tile 侧的 `blayout=col_major, slayout=row_major, fractal=512`
描述的是同一个字节序。

**秩固定为 5，不是"逻辑秩 + 2"。** 前导 batch 槽位 `B` 始终存在，与逻辑张量是否
有前导轴无关。因此逻辑 rank-2 的 `[N, K]` 权重会分块成 `[1, K/c0, N/16, 16, c0]`，
batch 具化为 `1`。PTOAS 直接校验这个 arity——rank-4 的 view 会被
`'pto.make_tensor_view' op user-specified layout=nz requires a rank-5 view` 拒绝，
无论周围 IR 内部多么自洽。

### 为什么 NZ 不需要自己的 stride 规则

对分块 shape 求普通行主序 stride，结果*就是* pto-isa 的 NZ stride：

| 槽位 | 行主序推导 | `BaseShape2D<T, R, C, NZ>` |
| ---- | ---------- | -------------------------- |
| `c0` | `1` | `1` |
| `16` | `c0` | `C0Size` |
| `R/16` | `16*c0` | `FRACTAL_NZ_ROW*C0Size` |
| `C/c0` | `(R/16)*16*c0 = R*c0` | `rows*C0Size` |
| 前导维 | `(C/c0)*R*c0 = C*R` | `cols*rows` |

因此一旦 shape 被分块，NZ 就是行主序家族的普通成员，
`BuildLogicalStridesFromLayout` 通过与 ND 相同的 `BuildRowMajorStrides` 路径处理
它。stride 由 `MaterializeTensorStrides`（pass 33）稍后填充；本 pass 只改写 shape。

这修正了 RFC #1300 中"NZ 没有 logical-stride 表示"的结论——该结论对逻辑 2-D shape
成立，对分块后的 rank-5 shape 不成立。

## 在流水线中的位置

```text
... -> LowerCompositeOps -> FlattenTileNdTo2D -> BlockNzTensorViews -> LegalizeTileCast -> ...
```

三条约束确定了这个位置：

- **在 `ConvertTensorToTileOps` / `LowerCompositeOps` 之后**——阶段 2 要改写的
  `tile.load` 必须已经存在。
- **在 `FlattenTileNdTo2D` 之后**——目标 tile 必须已经是逻辑 2-D 操作数。对仍是
  ND 秩的 tile 做分块，会产生类型标注与参数秩无法同时打印的 `tile.load`，破坏
  printer 往返。
- **在 `MaterializeTensorStrides` 之前**——该 pass 断言每个 NZ view 已分块，然后
  填充其行主序 stride。

`FlattenTileNdTo2D` 对 NZ 源跳过它的 ND2NZ 源窗口塌缩（该塌缩存在是因为 ND→NZ
需要 2-D GlobalTensor，而 NZ→NZ 不需要），所以本 pass 运行时逻辑窗口仍然完整。

## 行为

**阶段 1 —— 对每个 NZ `TensorType` 的 shape 分块。**

```text
# before
w: pl.Tensor[[32, 2048, 4096], pl.INT8, pl.NZ]

# after  (c0 = 32:  4096/32 = 128,  2048/16 = 128)
w: pl.Tensor[[32, 128, 128, 16, 32], pl.INT8, pl.NZ]
```

逻辑 rank-2 的权重落到同样的秩上，batch 由编译器合成：

```text
# before
w: pl.Tensor[[256, 512], pl.INT8, pl.NZ]

# after  (c0 = 32:  512/32 = 16,  256/16 = 16;  batch 具化为 1)
w: pl.Tensor[[1, 16, 16, 16, 32], pl.INT8, pl.NZ]
```

**阶段 2 —— 改写消费它的 `tile.load`。**

```text
# before （从 [E, N, K] 权重中切出 w[1:2, 256:512, 512:1024]）
wt: pl.Tile[[256, 512], pl.INT8, pl.Mem.Mat] =
    pl.tile.load(w, [1, 256, 512], [1, 256, 512], target_memory=pl.Mem.Mat)

# after  （offsets -> [.., k0/c0, n0/16, 0, 0]；sizes -> 分块形式）
wt: pl.Tile[[256, 512], pl.INT8, pl.Mem.Mat] =
    pl.tile.load(w, [1, 16, 16, 0, 0], [1, 16, 16, 16, 32], target_memory=pl.Mem.Mat)
```

### 符号形式的末尾 offset

末尾 offset 不必是常量，但它的对齐必须被**证明**，绝不允许假设。offset 到达本 pass
时只是它被绑定到的那个 SSA 名字，产生它的算术在别处，因此
`IsProvableMultipleOf`（`tensor_view_semantics.h`）会沿着本 pass 一次只读扫描收集到的
两类绑定往回走：

| 绑定形式 | 是轴因子的倍数 | 非负 |
| -------- | -------------- | ---- |
| `AssignStmt`（`n0 = nb * 256`） | 乘积中的某个因子是倍数 | 两个因子都非负 |
| `ForStmt`（`for k0 in pl.pipeline(512, 4096, 512)`） | `start` 与 `step` 同时是倍数 | `start` 与 `step` 同时非负 |
| 符号 start 的 `ForStmt`（`for ob in pl.range(core, TILES, CORES)`） | 向 `start` 与 `step` 递归 | 向 `start` 与 `step` 递归 |
| `Min` / `Max`（`min(800 - o0, 256)`） | 两个操作数都是倍数 | `Min`：两个操作数都非负；`Max`：任一即可 |
| `tile.get_block_idx` / `tile.get_block_num` | —— | lane 编号不可能为负 |
| 对正常量取模 / 整除（`(blk % 2) * 512`） | 由另一个因子承担 | 两者都向下递归：被除数也必须非负 |
| `ConstInt` | 该值本身是倍数 | 该值 `>= 0` |

在此基础上，和与乘积可以组合；差能证明整除性但无法证明符号，因此被拒绝。对正常量的取模与整除，符号是从被除数证明的，而不是从运算的名字：`FloorMod` 降级为 `arith.remsi`、`FloorDiv` 降级为 `arith.divsi`，两者都向零截断，因此负的被除数会得到负的余数。**两列必须
同时成立**——见[为什么符号也要证明](#为什么符号也要证明)。因此催生该特性的
grouped-matmul 权重路径现在可以编译：

```python
for nb in pl.spmd(N // N_TILE):
    n0 = nb * N_TILE                     # -> 行 fractal offset  n0 // 16
    for k0 in pl.pipeline(K_TILE, K, K_TILE, stage=2):
        wt = w[n0 : n0 + N_TILE, k0 : k0 + K_TILE]   # -> c0 block offset  k0 // c0
```

凡是走不通证明的表达式一律拒绝，并在诊断中列出可证明的形式。这种拒绝正是设计本身：
NZ 张量一旦按猜出来的坐标寻址，就会读到错误的 fractal，而下游不会有任何地方发现。

#### 为什么是整体相除，而不是代数重组

被除的只是 offset 的**求值结果**（`FloorDiv(offset, divisor)`）。那个看起来很诱人的
改写——把 `n0 = nb * 256` 变成 `nb * 16`，从而省掉运行期的除法——是不成立的：IR 的
算术在其声明位宽上回绕，而重组后的形式不会在同一处回绕：

```text
x : INT32 = 1 << 24
(x * 256) / 16   ==  0            # x * 256 在 i32 中回绕为 0
x * (256 / 16)   ==  268435456    # 重组后：不回绕，读到错误的 fractal
```

位宽提升并不是元凶，按原位宽重建同样救不回来：对 `a * b = q * 2^W + r`，原表达式得到
`r / d`，而任何重组都得到 `r / d + q * 2^(W - log2 d)`。把 offset 整体相除，才能完整
保留它自己的 dtype 与溢出行为。

证明本身能够在回绕下继续成立，是因为这里的除数一定是 2 的幂、因而整除 `2^W`：一个
`d` 的倍数对 `2^W` 取模之后仍然是 `d` 的倍数。这一前提由 `INTERNAL_CHECK` 固定下来。

另有一处限制是刻意保留的：`IterArg` 永远不会被解析——它的值每次迭代都在变，无论是初值
还是为它记录的任何绑定，都无法描述某一次使用真正看到的值。

#### 为什么符号也要证明

只证明整除性并不足以保证坐标安全。`n0 = -16` 是 16 的整数倍，而 `FloorDiv(n0, 16)`
等于 `-1`；codegen 随后会把 `pto.partition_view` 上的负 offset **钳位**到 0 而不是报错，
于是这次 load 会读到 fractal 0 并返回静默错误的数据——正是 [#2543] 为行索引修掉的那种
形态。

常量路径本来就会拒绝负的字面量，所以如果符号形式只证明整除性，就会出现同一个值写成
字面量被拦下、绑定成名字后却被放行的矛盾。`IsProvableNonNegative` 补上了这个缺口。

[#2543]: https://github.com/hw-native-sys/pypto/pull/2543

目标 `TileType` **原样保留**：GM 分区变成 rank-5，而 tile 保持逻辑 2-D 操作数。
因此该 load 使用显式类型的 `Call` 构造函数重建，而不是 `OpRegistry::Create`——后者
会从分块后的 shapes 参数重新推导出 rank-5 的 tile。

本 pass 之后不再有逻辑 shape 的 NZ `TensorType` 存活，因此下游（包括 codegen）
都不需要知道 NZ 的特殊性——codegen 会分别从 `TensorType::shape_` 推导
`pto.make_tensor_view`、其 `!pto.tensor_view<>` 类型以及 `pto.partition_view` 的秩，
三者必须一致。

## 生成代码

以上文逻辑 rank-2 的 `w: pl.Tensor[[256, 512], pl.INT8, pl.NZ]` 为例——注意前导的
`%c1` / `%c131072` / `1x` 把具化出来的 batch 一路带到三个站点：

```mlir
%w_view = pto.make_tensor_view %arg1,
    shape = [%c1, %c16, %c16, %c16, %c32],
    strides = [%c131072, %c8192, %c512, %c32, %c1]
    {layout = #pto.layout<nz>} : !pto.tensor_view<?x?x?x?x?xi8>
%w_pview = pto.partition_view %w_view,
    offsets = [%c0, %c0, %c0, %c0, %c0], sizes = [%c1, %c16, %c16, %c16, %c32]
    : !pto.tensor_view<?x?x?x?x?xi8> -> !pto.partition_tensor_view<1x16x16x16x32xi8>
pto.tload ins(%w_pview : !pto.partition_tensor_view<1x16x16x16x32xi8>)
          outs(%wt : !pto.tile_buf<loc=mat, dtype=i8, rows=256, cols=512,
                                   blayout=col_major, slayout=row_major, fractal=512, ...>)
```

**PTOAS 0.61 及以后**由此生成真正的 NZ `GlobalTensor`：

```cpp
GlobalTensor<int8_t, pto::Shape<1, 16, 16, 16, 32>,
             pto::Stride<131072, 8192, 512, 32, 1>, pto::Layout::NZ> ...;
```

在 0.60 及更早版本上，同样的 IR 无法汇编——见下文[汇编器版本](#汇编器版本)。

## 范围与拒绝项

里程碑 1 的范围是刻意收窄的。范围之外一律报错并指明修复方式——NZ 张量绝不能被
静默地错误寻址。

| 条件 | 结果 |
| ---- | ---- |
| `shape[-2] % 16 != 0` | 拒绝——不完整的分形没有表示 |
| `shape[-1] % c0 != 0` | 拒绝——不完整的 C0 线没有表示 |
| `shape[-2]` / `shape[-1]` 为动态维 | 拒绝——无法静态证明整除 |
| 切片偏移未对齐到分形边界 | 拒绝——没有分块表示 |
| 末尾切片偏移为符号形式且对齐与符号均可证明 | 映射——见[符号形式的末尾 offset](#符号形式的末尾-offset) |
| 末尾切片偏移为符号形式但对齐无法证明 | 拒绝——绝不基于「大概是对齐的」去做除法 |
| 末尾切片偏移为符号形式但符号无法证明 | 拒绝——负 offset 在 partition view 上会被钳位而不是报错 |
| 逻辑 rank 2 | 分块为 `[1, C/c0, R/16, 16, c0]`——batch 具化 |
| 逻辑 rank 3 | 分块为 `[B, C/c0, R/16, 16, c0]`——前导轴即 batch |
| 逻辑 rank > 3 | 分块时把所有前导轴折叠进 batch 槽位（见下） |
| 逻辑 rank < 2 | 拒绝——末尾两维是分形平面 |
| rank > 3 且前导维为动态 | 拒绝——折叠需要静态 extent 相乘 |
| `target_memory != Mat`（或缺省） | 拒绝——NZ→NZ 是 cube 操作数路径 |
| `tensor.slice` 只收窄前导轴 | 与其后的 load 一样分块（见下） |
| `tensor.slice` 在末尾 `[R, C]` 平面上开窗 | 拒绝——该窗口不连续 |
| `tensor.reshape` 把整块张量展平成 `[N]` | 原样保留——见[展平 NZ 张量](#展平-nz-张量) |
| `tensor.reshape` 成其它形状 | 拒绝——它重新解释了分块形式无法承载的坐标 |
| `tile.load` / `tensor.slice` / 整块展平之外的消费者 | 拒绝——此处 NZ 是只读的 |
| 显式 stride 或部分 `valid_shape` | 拒绝 |
| 动态 `valid_shape[-2]` 可证明是 16 的倍数（不规则的最后一个 tile） | 分块——行 fractal 数变为 `FloorDiv(rows, 16)` |
| 动态 `valid_shape[-2]` 可能止于 fractal 内部 | 拒绝——不完整的 fractal 没有分块形式 |
| 分布式张量 | 拒绝——`remote_load` 没有 NZ 分块 |
| 对 NZ 做 `tensor.view` / `tensor.reinterpret_view` | 在算子构造期拒绝 |
| 多列块 load 的 GM 行间隔超过 65535 个 block | 拒绝——**临时**，见 [GM 行间隔：一道临时防护](#gm-行间隔一道临时防护) |

### GM 行间隔：一道临时防护

**这一行不是设计上的限制。** 它防护的是一个上游缺陷
（[hw-native-sys/pto-isa#317]），该缺陷修复后即删除——连同 pass 里的
`CheckNzGmGapFitsBurstStride` 及其三个单测一并删除。

`TLoadGm2L1Nz2nz` 以 `uint32_t` 计算相邻列块之间的 GM 间隔，再不加范围检查地传给
`TLoadInstrGm2L1` 的 `uint16_t gmGap`。一旦超过 65535 个 32 字节 block 就会回绕，
load 读到错误的 fractal——而且是静默的：PTOAS 能汇编该 view，CCE 编译器接受该
kernel，matmul 在运行期返回错误数值且不报任何错。本 pass 是最后一个还知道是哪个
`pl.NZ` 标注导致问题的层，因此诊断放在这里。

对分块后的 view，该间隔可化简为 load 跳过的行数：

```text
gmGap = (gStride1 - gShape2*gShape3*gShape4) * sizeof(T) / 32
      = (R/16 - TR/16) * 16 * c0 * sizeof(T) / 32
      = R - TR                                       （32 字节 block）
```

因为任何 NZ view 按构造都满足 `c0 * sizeof(T) == 32`——所以该上界与 dtype 无关，
而且降低间隔靠的是**更宽**的行 tile，不是更窄的：

| `R` | 行 tile | 间隔 | 结果 |
| --- | ------- | ---- | ---- |
| 65536 | 16 | 65520 | 接受 |
| 65552 | 16 | **65536** | 拒绝 |
| 65552 | 32 | 65520 | 接受——同一张量，更宽的 tile |

另一种绕过方式是把按层堆叠的权重标注为 rank-3、以堆叠轴作为 batch
（`[LAYERS, K, N]`），其范围随即由 `gStride0` 和一个真正的 `for` 循环承载，而不再
经过 burst 间隔。

这里度量的窗口是 codegen 最终生成 `pto.partition_view` 所用的那个——load 带
`valid_shape` 时取 `valid_shape`，否则取 `shapes`
（`src/backend/common/pto_ops_memory.cpp`）。该 view 就是 pto-isa 的 `gShape`，
因此收窄的 `valid_shape` 载入的行 fractal **更少**，留下的间隔反而**更大**，比只看
`shapes` 得到的值更大。一个 `[65552, 64]` 的 INT8 权重若以 `shapes=[32, 64]`、
`valid_shape=[16, 64]` 读取，生成的是 `partition_tensor_view<1x2x1x16x32>`，
间隔为 65536，而不是 `shapes` 暗示的 65520。载入的行范围为动态时（不规则的最后一个
tile），检查按最坏情况——一行都不载入——计算，因此通过检查的 load 对任意运行期宽度都成立。

**单列块 load 不受此限制。** `TLoadGm2L1Nz2nz` 把 load 自身的列块数作为 `nBurst`
传入，而 DMA 只在从一个 burst 跨到下一个时才使用 `gmGap`，因此只有一个 burst 时那个
被截断的字段根本不会被读取。已在设备上于 pto-isa 自带的 `tload_gm2mat` ST 中确认：
`gShape1 = 1`、间隔为 65536 的 NZ `int16` load 返回逐位正确的数据，而同一个 load 在
`gShape1 = 2` 时会损坏 1837/4096 个元素。

[hw-native-sys/pto-isa#317]: https://github.com/hw-native-sys/pto-isa/issues/317

### 前导轴折叠进唯一的 batch 槽位

pto-isa 的 NZ `GlobalTensor` 只有**一个** batch 槽位，因此逻辑 `[G, E, N, K]` 权重
把两个前导轴折叠进去：分块后的 batch 是 `G*E`，stride 为 `C*R`。折叠是精确的——这些
轴稠密且行主序，折叠拿掉的 stride 正是它乘回去的那些——offset 以同样方式折叠，
`[g, e, 0, 0]` 寻址 batch `g*E + e`（`FoldNzLeadingOffsets`）。

这与[末尾 offset 拒绝的重结合](#为什么是整体相除而不是代数重组)不同：这里没有除法，
也不对对齐做任何假设，因此折叠对任意坐标都精确，而不只对对齐的坐标成立。它就是 ND
路径在地址计算时做的同一套算术，只是提前写进了坐标里。

被折叠的 extent 必须是静态的——动态 extent 无法乘进 batch——诊断会点名这一点。

多卡入口声明的正是 rank-4 参数（`[RANKS, E, R, C]`，dispatch 前按 rank 切片），
因此这个折叠是分布式程序能携带 NZ 权重的前提。

### 对 NZ 张量做切片

按层或按 rank 堆叠的权重是通过 `tensor.slice` 到达 kernel 的，因此切片与其后的
`tile.load` 一样分块：shapes 与 offsets 变成 rank-5；降秩的标量索引（`w[r]`）之后
不再需要 `drop_dims`，因为折叠已经把所有前导轴合并掉了。

只有**前导**轴可以被收窄。在末尾 `[R, C]` 平面上开窗会被拒绝：按 NZ 的字节序，某一层
的行分布在*每一个*分形列块内部，因此 `[layer*R, 0]` 选出的是 `C/c0` 段不连续的数据，
而分块 view 自身没有 stride 来描述它们——`MaterializeTensorStrides` 是从分块 shape
推导行主序 stride 的。请把堆叠轴放成前导轴（`[LAYERS, R, C]`），而不是按行堆叠。

### 展平 NZ 张量

覆盖*全部*元素的 rank-1 view 与 layout 无关：分块形式重排的是索引空间而不是内存，
两种写法走的是同一段连续 GM、顺序也相同。因此这样的 `tensor.reshape` 被原样保留
——不改写坐标，结果是 ND，正好是 `prefetch.async_prefetch` 对源张量的要求。没有
这条规则，给权重标注 `pl.NZ` 就会悄悄让它失去 SDMA L2 预热。

其它目标形状确实重新解释了坐标——`[256, 512] -> [128, 1024]` 是按逻辑行主序把行
两两合并，而在分块形式下这些元素散落在不同的分形块里——所以会被拒绝，而不是当作
ND 去寻址。

NZ 实参到达 orchestration 入口时同样带着*逻辑* shape：调用方就是按逻辑 shape 分配
权重的，只有编译后的形参是分块的。入口处会把它一次性改写成分块形式（纯元数据的
reshape，元素与顺序都不变），这样由它派生的每个 `Tensor::view` 都是按同一个 rank
去做 clamp 的。

sub-byte dtype（INT4 / UINT4 / FP4 / HF4 / BOOL）被拒绝，这是 **PyPTO 里程碑 1 的
范围限制，不是硬件限制**——pto-isa 的 NZ 机制确实处理 FP4（`tload_common.hpp` 中有
显式的 `caps::IsFP4` 分支，并断言 `staticShape[4] == C0_SIZE_BYTE / sizeof(DType)`）。
`c0` 已按位宽推导，因此等 packed-nibble 寻址端到端验证完成后，算术部分即可直接复用。

对齐类诊断是面向用户的（`CHECK_SPAN` → `ValueError`），位于 `BlockNzShape`。
到了下游，*未分块*的 NZ view 属于 pass 顺序不变量（`INTERNAL_CHECK_SPAN`），由
`MaterializeTensorStrides` 中的 `CheckNzViewIsBlocked` 和 `TensorViewCanonical`
验证器强制执行。

## 幂等性

分块**不是**幂等的——对已分块的 shape 再分块是错的——而结构性的
`IsBlockedNzShape` 判断无法区分"已分块的 shape"和"恰好以 `[16, c0]` 结尾的逻辑
shape"。因此本 pass 会在每个改写过的函数上打 `nz_tensor_views_blocked` 标记，
再次进入时直接返回。

## 汇编器版本

本 pass 的输出能否汇编取决于 PTOAS 版本，而仓库当前钉住的版本还不是能工作的那个。

| PTOAS | 行为 |
| ----- | ---- |
| ≤ 0.60 | 通过结构推断 layout。分块 NZ 与 ND 在结构上完全相同（都是行主序），因此推断出 `nd` 并覆盖显式的 `nz` 标注，报 `layout mismatch: user-specified layout=nz but inferred=nd`。任何秩的 NZ view 都无法汇编。 |
| ≥ 0.61 | 把显式的 `ND` / `DN` / `NZ` 标注视为权威并加以校验，因此上面的描述符可以汇编。它同时直接强制 NZ 的 arity：秩不为 5 的 view 会被 `'pto.make_tensor_view' op user-specified layout=nz requires a rank-5 view` 拒绝。 |

`toolchain/versions.env` 钉的是 **v0.61**，所以在钉住的工具链上 `pl.NZ` 已经端到端可用。
`tests/st/runtime/ops/test_matmul_nz.py` 是这一点的保障：ND 激活乘 NZ 权重，配套的 host
packer 负责生成 fractal 字节；此外还有切片用例分别钉住两个偏移轴——行 fractal
(`n0 // 16`) 和 C0 列块 (`k0 // c0`)。整张量 load 的所有偏移都是 0，因此仅靠基线用例
两者都覆盖不到。

0.60 上的失败在两个方向上都是安全而非静默的：它先停在上面的 layout mismatch；
即便越过那一步，pto-isa 的 ND→NZ `TLOAD` 路径要求 `staticShape[0..2] == 1`，
而分块维度违反了它，所以生成的 C++ 会在 `static_assert` 上编译失败，
而不是算出错误结果。

## 相关文档

- [14-flatten_tile_nd_to_2d.md](14-flatten_tile_nd_to_2d.md) —— 对 NZ 源跳过 ND2NZ 窗口塌缩
- [32-materialize_tensor_strides.md](33-materialize_tensor_strides.md) —— 填充分块 NZ stride
- [../ir/02-types.md](../ir/02-types.md) —— `TensorLayout` 与 `TensorView`
