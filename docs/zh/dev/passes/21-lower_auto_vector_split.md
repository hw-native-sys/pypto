# LowerAutoVectorSplit Pass（向量自动拆分下降）

在 `ExpandMixedKernel` **之前**，将带 AUTO `pl.split` 的混合 `InCore` 函数转换为
**显式 `split_aiv` 形态**：在 cube→vector 边界插入 `tile.aiv_shard`，在
vector→cube 边界插入 `tile.aic_gather`，仅对**向量子区域**沿拆分轴折半，注入
`tile.get_subblock_idx()`，并在函数上打 `split` + `split_aiv` 标记。

这是**唯一的自动拆分下降路径**：它始终运行，紧邻 `ExpandMixedKernel` 之前。运行后
每个拆分函数到达 [`SplitVectorKernel`](24-split_vector_kernel.md) 时都已带
`split_aiv` 标记，因此该 pass 只打属性（其 split_aiv 分支）——其旧的逐算子折半驱动
已被删除，折半机制现仅存于 `split_axis_utils`，由本 pass 共享。

本 pass 同时是一等公民区域节点 `SplitAivScopeStmt`（`for aiv_id in
pl.split_aiv(...)`）的**唯一消费者**。该区域作为结构节点存活于 parse → SSA →
`ResolveBackendOpLayouts`；在此处每个区域被就地下降，作用域包装被**擦除**，因此没有
任何 `SplitAivScopeStmt` 会到达 `ExpandMixedKernel`（pass 21）或 codegen。

## 为什么需要本 pass

用 `pl.split` 编写的混合 `InCore` 函数在同一函数体中描述 cube 与 vector 工作，拆分
意图仅由函数级 `split` 模式表达。实现该拆分有两种方式：

1. **`SplitVectorKernel` 中的后期逐算子折半** —— 在 `ExpandMixedKernel` 已经把函数
   体分为带跨核 `tpush`/`tpop` 的 AIC + AIV 之后，再逐算子折半 AIV 函数体。这重复了
   `tile.aiv_shard` / `tile.aic_gather` 已经编码的边界语义。
2. **早期显式下降（本 pass）** —— 在 `ExpandMixedKernel` 之前，把 AUTO `pl.split`
   函数体改写为手写显式核所用的同一 `split_aiv` 形态。随后 `ExpandMixedKernel` 中
   单一的算子驱动边界分支会统一地把 `tile.aiv_shard` / `tile.aic_gather` 折叠为带
   拆分标记的 `tpush`/`tpop`——自动核与手写核走完全相同的下游路径。

方式 2 是当前路径。它与旧的逐算子折半逐字节一致（分阶段收敛期间已验证），因为两者调用
同一套 `split_axis::ProcessStmts` 机制，仅入口与边界处理不同。

## API

| C++ | Python | 层级 |
| --- | ------ | ---- |
| `pass::LowerAutoVectorSplit()` | `passes.lower_auto_vector_split()` | Program 级 |

```python
from pypto import passes
result = passes.lower_auto_vector_split()(program)
```

## Pass 属性

| 属性 | 值 |
| ---- | -- |
| Required | `SSAForm`、`IncoreTileOps`、`SplitIncoreOrch`、`TileOps2D`、`TileMemoryInferred`、`NormalizedStmtStructure`、`AivSplitValid` |
| Produced | `SSAForm`、`IncoreTileOps`、`SplitIncoreOrch`、`TileOps2D`、`TileMemoryInferred`、`NormalizedStmtStructure` |
| Invalidated | `AivSplitValid` |

本 pass 关闭了由 `OutlineIncoreScopes` 打开的 `AivSplitValid` 验证窗口：它消费并擦除第一类 `SplitAivScopeStmt` 区域，此后结构化区域 verifier 无法再运行。因此该属性在入口被要求、在出口被失效。其余属性在 pass 前后保持不变——仍是混合形态的 InCore 函数体被就地改写。

来源：`include/pypto/ir/transforms/pass_properties.h`
（`kLowerAutoVectorSplitProperties`）。

## 作用范围

仅当**全部**满足时改写函数：

- `func_type_ == FunctionType::InCore`，且
- 带函数级拆分模式（`UpDown` / `LeftRight`，`mode != None`），且
- **尚未**为 `split_aiv`（手写显式核保持不动——它们已带显式 shard/gather 形态），且
- 确为**混合（cube↔vector）**：其汇总亲和性为 `MIXED`，与 `ExpandMixedKernel`
  判定 `is_mixed` 所用的 `ClassifyCallAffinity` / `CombineAffinity` 完全一致。

其余一律原样透传。最后一条很关键：**纯向量** `pl.split` 函数（例如把一个逐元素算子
拆到两个 AIV lane，既无 cube 也无 C↔V 边界）没有可收敛的边界，故保持不动——
`ExpandMixedKernel` 会照旧把它转成普通 AIV 函数并剥掉其 `split` 属性，保留其原先
（未拆分）的行为。若在此处对其下降，剥离后它将只带 `split_aiv` 而无 `split` 模式，
`SplitVectorKernel` 会因此报错。

**函数体中仍存在 `ScopeStmt` 的混合函数会被拒绝**，理由与下文的区域路径相同：整函数折半同样
不跨越 scope 边界。汇总亲和性会**穿过** scope 计算（scope 的亲和性即其函数体的亲和性），因此本
pass 能区分「被 scope 包裹的混合函数」与「纯向量函数」，只拒绝前者——纯向量函数体仍按原样透传，
不会报错。该情况通常不可达：本路径所依据的函数级 `split` 属性由 `OutlineIncoreScopes` 写入，而它
在同一步就消费掉了 scope；之所以校验而非假定，是因为另一种失败方式是静默的。在加入该校验之前，
`RollupAffinity` 根本没有 `ScopeStmt` 分支，于是任何被 scope 包裹的函数都汇总为 `SHARED`、被判定
为非混合，从而**完全未拆分地透传且没有任何诊断**。

## 显式 `SplitAivScopeStmt` 区域路径

除上述 AUTO 整函数路径外，函数体仍携带一个或多个 `SplitAivScopeStmt` 区域的
`InCore` 函数走一条独立的**区域路径**（`LowerExplicitRegionFunction`），它在 AUTO
路径**之前**判定。每个区域携带各自的 `split_` 模式，因此可处理单一函数级模式无法表达
的多模式情形。区域局部的 `tile_vars` / `var_replacements` 映射保证折半后的变量不会泄漏
到同级区域或区域外的算子。任何区域**之外**的语句以全宽发出，且永不折半。所有区域下降后，
作用域包装被丢弃，函数被打上 `split_aiv` + `split_aiv_region_validated`（后者通知
[`ExpandMixedKernel`](22-expand_mixed_kernel.md) 跳过其单一函数级模式的转置检查——
改由本 pass 用每个区域正确的拆分轴校验各自的转置风险）。

### 区域外契约（手动模式）

“以全宽发出”描述的是本 pass 对区域外语句所**做**的处理，而不是作者在那里**可以写**什么。
持有**至少一个**区域的函数即进入**手动模式**：区域对向量计算的放置具有决定权，而
[`AivSplitValid`](99-verifier.md) 验证器早在本 pass 之前就会强制这一分工：

| 算子 / 值 | 区域内 | 所有区域之外 |
| --------- | ------ | ------------ |
| 向量计算 | AIV | **拒绝** —— 检查 (e) |
| `tile.load` / `tile.store` | AIV | 允许（编译器物化） |
| cube 计算 | **拒绝** —— 检查 (a) | AIC |
| `aiv_shard` / `aic_gather` | 边界本身 | **拒绝** —— 检查 (c) |
| no-duplicate 算子（`pld.system.notify`） | 钉在 AIV lane 上 | 复制到两条 lane 上（**不会**诊断） |

因此本 pass 在区域外真正会遇到的语句是：cube 计算、`ConvertTensorToTileOps` 从承载其
计算的区域中提升出来的 `tile.load` / `tile.store` 对，以及与核无关的标量 / 控制流语句。
全宽向量计算**不在**其列：若要让某个阶段保持全宽，请把它包进
`for _ in pl.split_aiv(2, mode=pl.SplitMode.NONE):`，其含义恰是“两条 AIV lane 都运行
完整函数体”。因此多模式的目标是*只有区域、每个向量阶段一个区域*。**没有**任何区域的函数
不受手动模式影响。校验器的检查 (f)/(g) 还要求跨越区域边界的 tile 必须用边界算子写明，
因此隐式的 cube↔vector 跨越也不会到达本 pass。

写在所有区域之外的通信算子**不会**被拒绝，只是同样被复制到 cube lane 上（参见
`verify_aiv_split.cpp` 中的 “NOT CHECKED, DELIBERATELY” 注释）。下文的放置标记也并不能让
区域意味着“恰好一次”：AIV 函数带有 `dual_aiv_dispatch`，其函数体会在**两条** AIV 子 lane 上
都运行，把只应发生一次的副作用在两条子 lane 之间分片是作者的职责，`None` 区域 V→C 跨越的
lane 规则同理（见[作用域与放置](../../user/language/04-scopes.md)）。

### 把区域放置信息带过擦除点（`core_placement`）

擦除包装的同时也丢失了唯一记录“作者把语句写在哪里”的信息，而紧随其后的
[`ExpandMixedKernel`](22-expand_mixed_kernel.md) 会把每条 `SHARED` 语句复制到**两条**
lane 上。于是被作者放在区域内、与核无关的算子（`pld.system.notify`：TNOTIFY 未声明任何
core affinity）同样会落到 cube lane 上，而它可能在向量 lane 的 TPUT 把该信号所释放的数据
落盘之前就发布信号。

因此在把区域体拼接出去之前，本 pass 会给即将失去归属的调用打上
`attrs["core_placement"] = "aiv"`；`ClassifyCallAffinity` 把它视为**放置权威**并将这些调用
解析为 `VECTOR`。该属性断言的是一个放置结论，因此只写在“区域确实**决定**其 lane”的调用上：

| 调用的本征亲和性 | 是否打标 | 原因 |
| ---------------- | -------- | ---- |
| `SHARED` **且**带有 `set_no_duplicate()` 标记（`pld.system.notify`） | **是** | 只有区域能决定其放置，`SHARED` 正是 pass 21 会复制的那一类，而复制对它来说是错的 |
| `SHARED` 但*未*标记（`pld.system.wait` 等与核无关的算子） | 否 | 钉住会把它从 cube 通路上**移除**。对阻塞类算子而言这是误编译——matmul 会越过该 wait 本应等待的对端数据 |
| `VECTOR`（普通向量计算） | 否 | 其内存规格已把它放在 AIV lane 上 |
| **自述** lane（`tile.create`、`system.syncall(core_type=…)`） | 否 | 由其自身声明决定，区域不凌驾于声明之上 |
| `MIXED`（`aiv_shard` / `aic_gather`、跨 C/V 的 `tile.move`） | 否 | 它们**就是**那次传输——一侧 tpush、另一侧 tpop |
| `CUBE` | 否 | 区域内的 cube 计算已被检查 (a) 拒绝；覆盖逻辑也拒绝改写它 |

因此一个混合通信 kernel 只会多出一个属性，就打在 notify 上。该标记买到的恰恰只有一件事：
该算子不会被复制到 **cube** lane 上；它对“有多少条 AIV 子 lane 会执行它”只字未言。该遍历会下降进
`for` / `if` / `while` / `seq`，且是幂等的（嵌套区域不会重复打标），并作用在每个区域分支的
**最终**语句上，即在折半机制改写完调用之后。

**生命周期：本 pass → pass 21，到此为止。** `ExpandMixedKernel` 一旦消费完即剥除该属性——
`Call::attrs_` 是反射的 `UsualField`，printer 又以开放世界方式序列化 attrs，未剥除的标记会
出现在后续每一次 pass dump、往返与 `assert_structural_equal` 中，描述一个已不存在的区域。
其生命周期与 `pipeline_stages` 相同（[`LowerPipelineLoops`](29-lower_pipeline_loops.md) →
[`CanonicalizeIOOrder`](30-canonicalize_io_order.md)）。

函数级 AUTO split（`optimizations=[pl.split(mode)]`，包括 `SplitMode.NONE`）与显式
`pl.split_aiv` 区域是**互斥**的；若需在携带区域的作用域上指定自定义跨核槽位数，请使用
`optimizations=[pl.cross_core_slot(slot_num=N)]`：它只决定 pipe 大小，不标注任何拆分。
该检查在更早的 [`OutlineIncoreScopes`](08-outline_incore_scopes.md) 中执行，那里作用域自身
的 `split_` 与其区域都仍可见；否则本区域路径会按区域下降并静默丢弃函数级 split。（提取后
二者会无法区分地合并：**单个**区域会合法地派生出一个函数级代表 `split` 模式。）

按区域的 `split_` 模式处理三种区域体形态：

- **数据并行 · 全宽体**（`UpDown` / `LeftRight`，无显式边界算子）：区域体持有全宽向量计算。
  区域路径注入按区域的 `subblock_idx`，将向量算子路由到共享的 `split_axis::ProcessStmts`
  折半机制（按区域局部进行），并校验按区域的转置风险。这是自动收敛形态产生的范式。
- **数据并行 · 显式边界体**（`UpDown` / `LeftRight`，已存在 `tile.aiv_shard` /
  `tile.aic_gather`）：用户已手动切分 cube tile 并在每 lane 的半块上编写向量计算，故区域体
  **已是**半宽形态。区域路径检测到这一点（`RegionBodyHasExplicitBoundary`）后**原样透传区域
  体**——不再折半，也不注入重复的 `subblock_idx`。若在此处再次折半将导致**双重切分**（下游的
  Acc→Vec move 会被误判为新的 cube→vector 边界并被改写成第二个 `aiv_shard`），从而产生一个
  无任何分配的孤立 Acc memref 并使 PTO codegen 崩溃。`ExpandMixedKernel` 会像处理手写
  split_aiv 核一样，把显式边界折叠为 `tpush`/`tpop`。
- **任务并行体**（`None`）：**没有拆分轴**——两个 AIV lane 都运行**完整**区域体，由作者通过
  区域的 `aiv_id` lane 索引（例如按 `aiv_id` 跨步的循环）分派各自不相交的工作。区域路径
  **原样透传区域体**（不折半、不本地化偏移、不注入 `subblock_idx`；作者的
  `aiv_id = get_subblock_idx()` 绑定已携带 lane 信息）。此处的 `tile.aiv_shard` /
  `tile.aic_gather` 是**被接受**的，并与其余语句一同透传：没有拆分轴时它只跨越 AIC/AIV 边界
  而不切分，其 `split=0` 类型推导原样保留形状，因此没有可折半或可拼合的东西。本模式下会跳过
  `ValidateMixedExplicitRegion`——它拒绝的是「半宽边界算子与全宽向量算子混写」，而这里一切
  都是全宽。该函数仍会被标记 `split_aiv`，因此下游 [`ExpandMixedKernel`](22-expand_mixed_kernel.md) /
  `SplitVectorKernel` 会把它派发到**两个** AIV lane（经由 `dual_aiv_dispatch`），而**非**
  lane-0-only 的非拆分 replay——故从这类区域向外的 V→C 跨越上两个 lane 都会 push，写入同一个
  共享槽位且没有任何仲裁，因此除非作者保证该值 lane-uniform，cube 收到的是二者之一且不确定
  是哪一个。当区域的 tile 无法折半（单位维）或归约必须保持全宽时使用本模式。

### 显式边界区域内允许出现哪些算子

由于该区域体是**原样**拼接的，其中每个 vector 算子都必须已经是 per-lane 的——保持全宽的
算子会在两个 AIV lane 上运行出完全相同的结果。`ValidateMixedExplicitRegion` 负责校验这一
点，并以可操作的错误信息列出违规算子。满足以下任一条件的 tile 生成算子会被接受：

| 接受条件 | 原因 |
| -------- | ---- |
| （传递地）消费了 `tile.aiv_shard` 的结果 | 依构造即处于 half-width 数据流中。 |
| 纯生成算子——`tile.full` / `tile.ci` / `tile.random`（以及 `tile.create`，它归类为 `SHARED`，本就不会被报告） | 其结果仅是自身属性的函数：不读取任何 tile、不读取内存，因此无论作者写的是什么 extent，per-lane 复制都是正确的。 |
| 携带**地址**的算子——`tile.load` / `tile.slice` / `tile.extract` / `tile.gather_row`——且其**读地址**引用了区域的 `aiv_id` | 作者已显式做了 per-lane 定位，例如 `data[base + aiv_id * HALF : ...]`。仅读偏移参数计入（`tile.load` 第 1 个、`tile.slice` 第 2 个、`tile.extract` 第 1–2 个、`tile.gather_row` 第 3 个即 `src_offset`）——出现在 `shape`、`valid_shape` 或**目的**槽位中的 lane 引用并不会移动窗口，因此不予接受。 |

该扫描按**区域**播种，并按程序顺序做一次前向遍历，因此它只能识别在当前扫描区域内定义的边界结果。从别处到达该区域的 `tile.aiv_shard` 结果——在兄弟区域中产生，或经由回边上的循环 `iter_arg` 传入——对它是不可见的，于是消费者会被报成全宽，尽管该值确实是按 lane 的。这两种写法都会被 `AivSplitValid` 验证器提前 12 个 pass 拒绝（检查 (i) 与 (j)，见 [99-verifier.md](99-verifier.md)），这正是让上述误报不会出现在作者面前的原因；因此本扫描实际只会遇到同区域内的数据流，也正是它被设计来处理的范围。

`tile.gather_row` 是其中的 DMA 情形：它是 DPS，因此带有**两个**偏移，而只有 `src_offset`
决定两个 lane 是否在做不同的工作——`src_offset` 由 lane 派生意味着每个 lane 各自拉取属于
自己的散列 GM 行（接受）；若只有 `dst_offset` 由 lane 派生，则两个 lane 会把**相同**的行取到
同一个全宽累加器的不同槽位（仍会被报告）。

其余归类为 `VECTOR` 的算子都会被报告。有两点需要注意：

- 生成算子**仅对其自身**被接受，它不会加入 half-width 数据流。
  `z = pl.full([FULL, N]); y = pl.add(z, z)` 仍会在 `y` 处被拒绝——全宽生成算子不得为其
  消费者背书。
- lane 引用**仅**在携带地址的算子上被信任。在其他算子上，lane 派生的标量只是一个普通操作数，
  并不能说明结果的宽度——因此 `pl.set_validshape(full_width_tile, 1, aiv_id * HALF)`
  无法把一个全宽 tile 洗白进区域。

该校验证明的是**意图**而非**范围**：偏移按 lane 跨步、但 extent 仍为全宽的 load 会被接受，
此时两个 lane 会读到重叠的窗口——这与本 pass 从不检查 `tile.store` 的 lane 相关偏移是同一种
信任。

由于区域经由通用的 `BeginScope`/`EndScope` 构建且不被提取，它可**嵌套**在 `pl.range` /
`pl.pipeline` 循环或 `if` 之内；区域路径会递归进入复合语句以下降每个区域，同时保留外围控制流。

### Per-lane 散列 gather

`pl.gather_row` 是唯一能按任意**运行时**偏移读取 GM 的算子，因此它正是把 paged / top-k 行
集合切分到两个 AIV lane 上的手段：每个 lane 在 UB 中组装 tile 的一半，再由 `pl.aic_gather`
把重组后的 tile 交给 cube。

```python
with pl.at(level=pl.Level.CORE_GROUP, name_hint="sparse_kv",
           allow_early_resolve=True):                           # see the ring note below
    for aiv in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
        ub = pl.full([64, 512], dtype=pl.BF16, value=0.0)       # per-lane HALF extent
        for k in pl.range(64):
            src = pl.cast(pl.read(idx, [aiv * 64 + k]), pl.INDEX)
            ub = pl.gather_row(ub, pool, [k, 0], [src, 0], [1, 512])   # lane-derived src_offset
        kv = pl.aic_gather(ub)                                  # V2C -> [128, 512] in Mat
    out[0:16, 0:128] = pl.matmul(q, kv, b_trans=True, out_dtype=pl.FP32)
```

有两条编写规则使其成立：

- **累加器按半 extent 编写。** `pl.full` 是生成算子，因此无论给它什么 extent 都会被接受，而它
  本身不会加入 half-width 数据流。gather 是凭其由 lane 派生的 `src_offset` 被接受的；该校验
  证明的是**意图**而非**范围**，所以此处若写成全 extent 的累加器，gather 回来就会变成
  `2 x FULL` 并在下游产生形状不匹配。
- **留意跨核 ring 的开销。** V2C ring 会在消费侧核的内存中预留 `slot_size x slot_num` 字节
  （V2C 为 L1，C2V 为 UB），其中 `slot_size` 是消费方弹出的**完整** tile——此处为
  `128 x 512 x 2 = 131072`——而 `slot_num` 默认取 **2**，相当于在 512 KB 的 L1 中占用
  256 KB；每次调用只 push 一次的 kernel 无需超过该深度。`pl.cross_core_slot(slot_num=N)`
  可双向调整该值；若调得过大，`AllocateMemoryAddr` 会报告溢出并指出被预留的字节数。

`pl.aiv_shard` 在这里**不能**替代半 extent 的 `pl.full`：它是 C→V 传输，要求操作数位于
`Acc`（cube 产出），因此无法对 vector lane 自己产生的值做切分。

### 区域必须不被 scope 包裹

区域下降会递归进入 `ForStmt` / `WhileStmt` / `IfStmt` / `SeqStmts`，但**刻意不**进入
`ScopeStmt`：scope 携带提取（outlining）与名字可见性语义，区域局部的折半不应跨越它。因此本
pass 运行时每个区域都必须已不被 scope 包裹——通常由
[`OutlineIncoreScopes`](08-outline_incore_scopes.md)（pass 8）保证，它会把外围的 `InCore`
scope 提取为独立函数。

pass 8 只对 `Opaque` / `Orchestration` 函数提取 scope，因此声明为
`pl.FunctionType.InCore` 的函数内部的 scope 会原封不动到达本 pass。该 scope 必须是
**用户手写的**：解析器不会在那里自行添加——InCore 函数中的顶层区域被裸露地发出，这样打印出来
的 `*_incore_0` 才能重新解析回同样的 IR；而在这类函数中书写区域本身，也会被更早的
[`AivSplitValid`](99-verifier.md) 检查 (h) 拒绝：

```python
@pl.function(type=pl.FunctionType.InCore)   # pass 8 会跳过该函数
def f(self, a: pl.Tensor[[128, 128], pl.FP32],
      c: pl.Out[pl.Tensor[[128, 128], pl.FP32]]) -> pl.Tensor[[128, 128], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP):                   # 不会被提取
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            base = aiv_id * 64
            c = pl.store(pl.exp(pl.load(a, [base, 0], [64, 128])), [base, 0], c)
    return c
```

下降完成后，`LowerExplicitRegionFunction` 会重新扫描函数体，对任何存活下来的区域抛出
`ValueError`，并把源位置指向 `pl.split_aiv` 那一行。修复方式：删掉这层多余的 scope，或改用
普通的 `@pl.function` / `@pl.jit`（Opaque）让 pass 8 提取它。

该重新扫描还会拒绝**其他任何**存活下来的 `ScopeStmt`，以覆盖对称情形：scope 嵌套在区域体
*内部*。此时区域本身已被消费，故上一条检查会通过——但内层遍历（`LowerStmts`、
`CheckNoCubeTileHalved`、`ScanRegionHalfWidth`）会跨过该 scope 而不进入，其中的向量算子会以
全宽被拼接出去，导致两条 AIV lane 都计算整块 tile。该情形从 DSL 不可达（pass 8 会把区域内的
`with pl.at(...)` 提取为独立函数，检查 (h) 又会拒绝在非提取器产生的 InCore 函数中书写区域），
因此它守护的是绕过 pass 8 的 IR——手工构造的，或反序列化的 `.pto`。

该守卫也正是 `split_aiv_region_validated` 标记可信的依据：只有当每个区域都确实被消费后才写入
attrs，因此 [`ExpandMixedKernel`](22-expand_mixed_kernel.md) 凭该标记跳过自身的 func-mode
检查时，背后总有一次真实的逐区域校验。若无此守卫，被 scope 包裹的区域会既未下降、又未校验，
却仍被标记为“已完成区域校验”，问题要到很晚才以 PTO codegen 的内部断言
（`SplitAivScopeStmt reached PTO codegen`）暴露。

## 拆分轴分派

| `SplitMode`（int） | 拆分轴 | 折半的向量子区域 |
| ------------------ | ------ | ---------------- |
| `None`（0） | —（无拆分轴） | 不折半——任务并行；tile 保持全宽，由 `aiv_id` 分派两个 lane |
| `UpDown`（1） | 维 0（高度） | 行 |
| `LeftRight`（2） | 维 1（宽度） | 列 |

`SplitDimension(mode)` 对 `UpDown` 返回 `0`，对 `LeftRight` 返回 `1`
（`split_axis_utils`）；对 `None` **不调用**（区域路径先对 `None` 分支——无轴可推导）。

### split 模式与 pto-isa split code

`SplitMode` 描述的是作者选的轴。真正下发到设备的 `split` 属性含义更窄——它是 pto-isa
的 `TileSplitAxis`，还要说明两个 lane 的**运行时** extent 之间的关系，因为消费侧正是
靠它在 FIFO 槽位中定位 lane 1 的数据段（`popVecTileFromGMFiFo`）：

| Code | pto-isa | Lane 1 的数据段起点 | 要求 |
| ---- | ------- | ------------------- | ---- |
| 0 | `TILE_NO_SPLIT` | — | 单一读者 |
| 1 / 2 | `TILE_UP_DOWN` / `TILE_LEFT_RIGHT` | `e1 * pitch` | `e0 == e1` |
| 3 / 4 | `TILE_UP_DOWN_ODD` / `TILE_LEFT_RIGHT_ODD` | `(e1 + 1) * pitch` | `e0 == e1 + 1` |

这两套词汇分别落在不同的算子上：

- **`tile.aiv_shard` / `tile.aic_gather` 携带 MODE**（`0` / `1` / `2`）。本 pass 只
  盖 `int(mode)`——作者选的轴，仅此而已。
- **`tile.tpush_*` / `tile.tpop_*` / `system.tfree_*` 携带 CODE**（`0`..`4`），由
  [ExpandMixedKernel](22-expand_mixed_kernel.md) 在把边界折叠成传输对时，根据该 mode
  与全宽 tile 的 extent 推导（`split_axis::ShardSplitCode`）。PTO codegen 原样打印为
  `{split = N}`。

奇数切分轴有两种来源，折半逻辑对二者完全一致：`ComputeHalfDimSize` 让**两个** lane 都
拿到 **ceil** 折半作为物理 box，参差不齐的部分由逐 lane 的 valid extent 承载。

| 边界 tile | Lane box | Lane extent | Code |
| --------- | -------- | ----------- | ---- |
| `[17, 128]`，全有效 | `[9, 128]` | 9 / 8 | 3 |
| `[16, 128]`，valid `[15, 128]` | `[8, 128]` | 8 / 7 | 3 |
| `[16, 128]`，全有效 | `[8, 128]` | 8 / 8 | 1 |

## 跨边界的部分有效（partially-valid）操作数

`valid_shape` 不足物理 box 的跨界值，正是 kernel 的 ragged 尾部。Cube→Vector FIFO
把传输的**列** extent 钉死为物理值，而**行** extent 是自由的（推导与 ISA 依据见
[PTO codegen](../codegen/00-pto_codegen.md)）。**切分轴**上的收窄会让 extent 变成逐
lane 的——lane `L` 持有 `clamp(V - L*half, 0, half)`——因此哪种模式 ragged，就决定了
由哪个字段来承载：

下表是 **shard**（Cube→Vector）的契约；`aic_gather` 遵循本节末尾的几何规则。

| 收窄的轴 | 模式 | extent 性质 | 载体 | 状态 |
| -------- | ---- | ----------- | ---- | ---- |
| 行（切分轴） | `UpDown` | 逐 lane | TPOP `valid_row` 操作数 | 两个 lane 可摆放时**支持**（见下） |
| 列（非切分轴） | `UpDown` | 两 lane 相同、静态 | 整 box 传输 + 静态 `pto.treshape` | 支持 |
| 行（非切分轴） | `LeftRight` | 两 lane 相同、静态 | TPOP `valid_row` 操作数 | 支持 |
| 列（切分轴） | `LeftRight` | 逐 lane | 无 | **拒绝** |
| 列为运行期值 | 任意 | 两 lane 相同、动态 | 无（`treshape` 不带操作数） | **拒绝** |
| 行为运行期值 | `UpDown` | 逐 lane、动态 | 边界算子保留整 box + 第一个消费者的 `valid_shape` | 支持（见下方说明） |
| 行逐 lane **且**列收窄 | `UpDown` | 两者 | 无（`treshape` 会同时重写两个轴） | **拒绝** |

`ReshapeSplitAxis` 只能对切分轴做 ceil 折半，因为 lane 索引不属于 op 的类型函数。
`LocalizeExplicitBoundaryValid` 在本 pass 修正该猜测——区域自身的
`aiv_id = tile.get_subblock_idx()` 在作用域内——并把逐 lane extent 传给那些原样透传
`valid_shape` 的消费者；若消费者会改变逻辑矩形（reduction、slice），则连同 span 一并
报错。AUTO 分支通过 `LocalizeShardValidForLane` 施加同样的 *extent* 修正，但不含下面的
store 保护——它的消费者由折半遍历重建，而非本遍历。

- **两个 lane 的 extent 必须可摆放。** 切分轴上的 extent 并不是自由字段：pto-isa 根据
  被弹出 tile 自身的 valid extent 推导 lane 1 的数据段起点——`TILE_UP_DOWN` 下是 `e1`，
  `_ODD` 模式下是 `e1 + 1`。因此可摆放的形态只有三种：`e0 == e1`（偶数 code）、
  `e0 == e1 + 1`（奇数 code），以及 `e1 == 0`（lane 1 不弹出任何数据，其数据段永远不会
  被解引用，偶数 code 依然精确）。box 分区对 ragged 边界并不保证这一点——16 行 box 上
  `V = 13` 会得到 8 与 5——这正是下文均分分区要解决的问题；均分不适用时，
  `ShardSplitCode` 会报错并给出可行的取值。
- **运行期的切分轴 valid extent 弹出完整 box。** split code 是编译期属性，而两个 lane
  需要哪一个取决于它们的**运行期** extent：16 行的轴上 valid 12 会让两 lane 变成 8 与 4，
  valid 16 则是 8 与 8，没有哪个 code 对两者都正确。因此边界算子干脆不携带逐 lane
  extent——`LocalizeExplicitBoundaryValid` 给它完整 box
  （`split_axis::WithFullSplitAxisValid`），并把 lane 的 extent 放到第一个消费者上。
  这与偶数 code 恰好配套：生产者搬运的是完整物理 box，lane 1 的数据段就落在 box 的一半处，
  偶数 code 正好指向那里，与运行期 extent 无关。已在 a2a3 上对 16 行边界的 1..16 全部
  extent 验证。[pto-isa 的 pop](https://github.com/hw-native-sys/pto-isa/issues/263)
  确实按被弹出 tile 自身的 extent 放置 lane 1，与其源码读法一致——此前得出相反结论的实测，
  其探针两个操作数都是常量，乘积的每一行每一列都相同，落位错误因而无法分辨。
- **手写 `tile.tpop_from_aic` 上的同类行 extent 仍会落位错误。** `SplitVectorKernel` 的
  折半会把用户声明的 `valid_shape` 局部化到 pop 自身，而 pto-isa 正是从这里读取数据段偏移。
  仅加宽 pop 并不能修复该路径：其消费者继承作者的声明，随后从完整来源写入部分目标，实测
  结果更差。`tests/st/runtime/cross_core/test_cross_core_split_parity.py` 中标记 xfail
  的参数记录了受影响的取值范围：`UP_DOWN` 下 `half < V < box`。
- **被收窄的列 extent 在所有路径上一律拒绝。** 它根本没有承载者——槽位按生产者的物理列
  间距写入，而 pop 依据 tile 自身的 `validCol` 重建读取几何——并且在 `LeftRight` 下它就是
  切分轴，必须逐 lane 取值。该契约由 `CheckSplitBoundaryCarriesValid`
  （`src/ir/op/tile_ops/cross_core.cpp`）统一持有：它既在边界算子的类型推导中运行，也由
  `ShardSplitCode` 调用，因此手写的 `tile.tpush_to_aiv` / `tile.tpop_from_aic` 同样受其约束。
- **空 lane 的 store 被保护。** ragged extent 覆盖不到的 lane，其 extent 为 `0`，而
  零行 `TSTORE` 超出 pto-isa 契约（`TSTORE_IMPL` 断言 `GetValidRow() > 0`）。store
  被加上运行时 `extent > 0` 判断；`tpop` 与 `tfree` 保持**无条件**——两个 lane 都占用
  槽位且都必须释放。
- **gather 的限制源自几何而非 DMA。** V2C 的 pop 落到 NZ Mat tile
  （`TLoadGm2L1Nd2nz`），不读取任何 valid extent。真正限制 `aic_gather` 的是摆放：
  lane `l` 位于偏移 `l*half`，拼合后的数据是 `[0, v0) ∪ [half, half + v1)`——只有两段
  相邻时才是矩形。该规则在**本 pass** 而非类型推导中执行：推导发生在逐 lane extent 存在
  之前，只能依据自己的 ceil 猜测来判断。由 localized shard 供给的 gather 会被精确定型
  ——两段必然相邻，拼合 extent 即 shard 前的 `V`——只有两个 lane 共享的部分 extent 才会
  被拒绝。同样的几何原因决定了 gather **没有奇数形态**：`GatherSplitCode` 始终返回偶数
  code，并拒绝两个 lane extent 不等的边界（请改为补齐该轴，再在 cube 侧收窄结果）。

## 把 ragged 边界在两个 lane 之间均分

切分会把切分轴分给两个 lane：lane `L` 拥有 `[L*S, L*S + S)`。可选的分区有两种，只有
边界 tile 是 ragged 时二者才不同：

| 分区 | `S` | `[16, …]` box、valid 为 13 时的 lane extent | 可摆放？ |
| ---- | --- | ------------------------------------------- | -------- |
| box（默认） | `ceil(box / 2)` = 8 | 8 与 5 | 否 |
| **valid**（均分） | `ceil(V / 2)` = 7 | 7 与 6 | 是——`TILE_UP_DOWN_ODD` |

box 分区是通用的：无论 tile 的 valid extent 如何都成立，所以它是默认选择。它做不到的
是让两个 lane 相差不超过 1，而这恰恰是传输的硬要求（见上文可摆放规则）。改为均分
**valid** 区域后，`e0 - e1 ∈ {0, 1}` 由构造保证——并且真实工作量也被平均分配，而不是把
余数全丢给 lane 1。

`split_axis::ResolveLaneStride` 在改写函数体之前扫描一遍，只有在既必要又安全时才返回
均分步长：

- 至少有一个 Cube→Vector 边界，且所有边界在切分轴上的静态 `(box, valid)` 一致，并且
  `ceil(V / 2) < ceil(box / 2)`（否则 box 分区本身已经是均分的）；
- 没有 Vector→Cube 边界（gather 按各自的 extent 位置拼接两个 lane，只有 extent 相等时
  两段才相邻）；
- 函数体内其他切分轴上的 tile 全都派生自该边界——独立来源的 tile（`tile.load`、
  生成类算子）跨越**整个** box，而均分分区覆盖不到。

其余情况一律保持 box 分区。选定的步长会贯穿整个折半流程——偏移（`AdjustOffsets`）、
逐 lane 的 valid extent（`LocalizeValidDimForSplit`）、shard 自身的类型
（`LocalizeShardValidForLane`）——而物理 box 仍是 `ceil(box / 2)`，因此缓冲区和槽位大小
不变。步长以 `lane_stride=S` 盖在边界算子上，供
[ExpandMixedKernel](22-expand_mixed_kernel.md) 用同一分区推导传输 code。在均分的函数体里
若出现迁移切分轴的 `tile.reshape`，会被拒绝：迁移后的轴带着自己的 half，无法表达按另一
条轴的 valid 做的均分。

```python
# 16 行 box 里有 13 行有效，UP_DOWN。lane 0 取第 0-6 行，lane 1 取第 7-12 行。
popped: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec,
                pl.TileView(valid_shape=[pl.min(pl.max(13, aiv_id * 7) - aiv_id * 7, 7), 128])
               ] = pl.tile.aiv_shard(qk, split=1, lane_stride=7)
out_store = pl.tile.store(popped, [0 + aiv_id * 7, 0], out_0)
```

显式 `pl.split_aiv` 区域永远不做均分：那里的逐 lane 偏移是作者自己写的
（`out[aiv_id * HALF : ...]`），编译器不能用与区域索引方式不同的分区去切数据。

## 算法

`LowerFunction` 改写一个混合 `InCore` 函数：

```text
1. split_dim = SplitDimension(mode)；边界算子携带 int(mode)。
2. InjectSubblockIdx(func, is_aiv=true) 在函数体顶部插入
       subblock_idx = tile.get_subblock_idx()
   （若 'subblock_idx' 已占用则取新名）。
2a. ResolveLaneStride(body, split_dim) 选择分区：对于只有一个 ragged 跨界的
    函数体取均分的 ceil(V / 2)，否则为 null（box 分区）。它会贯穿下面每一处
    偏移与逐 lane extent。
3. LowerStmts 遍历扁平函数体：

   边界 tile.move（ClassifyMoveDirection）：
     CUBE_TO_VECTOR —— 将 move 替换为
         tile.aiv_shard(full_cube_tile, split=int(mode)[, lane_stride=S])  -> 半
       推导出的半类型已经带有消费侧 lane 内存（Vec）：切分推导器让 memory_space
       保持为空，由 OpRegistry::Create 用 tile.aiv_shard 的 set_output_memory
       声明填充，与显式形式同源。将结果 var 连同其半尺寸种入 tile_vars，并记录
       旧->新 var 重绑。cube 源（matmul / Acc 结果）保持全尺寸。
     VECTOR_TO_CUBE —— 插入
         tile.aic_gather(half_vector_tile, split=int(mode))  -> 全
       将源解析到其折半后的 var 使 gather 把 半 -> 全 翻倍，随后保留对折叠后全尺寸
       tile 的原 cube 放置 move（命名为 "<dest>_mat"，以便 ExpandMixedKernel 的
       V->C 边界据此命名其合成的 tpop）。

   亲和性门控（ClassifyCallAffinity）：
     VECTOR 亲和叶子 —— 将单条语句送入
       split_axis::ProcessStmts({stmt}, ..., is_aiv=true)：与已删除的
       SplitVectorKernel 驱动所用的同一机制。沿 split_dim 折半 tile.load /
       tile.store / tile.slice / tile.reshape / 计算结果，按 subblock 本地化偏移，
       在 tile_vars 中跟踪折半 var。
     CUBE 亲和叶子 —— 全尺寸透传，绝不折半。

   ForStmt / IfStmt —— 递归进入函数体处理向量内容。

4. CheckNoCubeTileHalved 重新遍历改写后的函数体，断言没有 CUBE 亲和算子消费或产生
   tile_vars 中的 tile（亲和性门控绝不能把折半 tile 漏入 cube 操作数）——失败时
   INTERNAL_CHECK。
5. transform_utils::Substitute 应用 var_replacements；DeepClone 脱离共享子树。
6. WithSplitAivAttrs 打 split + split_aiv（丢弃任何先前的 split / split_aiv /
   dual_aiv_dispatch 条目）。
```

逐算子向量折半（沿拆分轴折半形状、按 `subblock_idx * half` 本地化偏移、`tile.slice`
静态形状参数与结果类型同步折半、rank-1 load 的 reshape 按 lane 切片、拒绝在拆分轴上
归约、保留单元素拆分维、循环 `iter_arg`/`return_var` 跟踪）全部由
`split_axis::ProcessStmts` / `ProcessStmt` 产生；同样的事实由
`tests/ut/ir/transforms/test_lower_auto_vector_split.py` 验证。

当拆分维不是单元素维时，自动折半会拒绝根生成器 `tile.ci` 和 `tile.random`。它们的
生成值依赖位置，因此仅修改结果类型并不充分；正确改写还需要按 lane 调整 shape 与生成器
状态，而本 pass 不会合成这些调整。请将算子移到会自动折半的拆分区域之外。单元素拆分维与
已有显式边界的半宽区域仍保持不变。

## 亲和性门控

仅折半**向量**工作，cube 工作保持全尺寸。亲和性由
`core_affinity::ClassifyCallAffinity`（按内存空间）决定：产生或消费 `Vec` tile 的算子
为 `VECTOR`；matmul 操作数与 Acc/Mat cube 结果为 `CUBE`。C→V 边界 `tile.aiv_shard`
是接缝：全尺寸 cube tile 是其输入，半尺寸向量 tile 是其输出。`CheckNoCubeTileHalved`
是兜底——若 cube 操作数被缩小则触发。

## 示例 —— cube→vector 边界，向量区域折半（UpDown）

混合核：cube tile（`Mat`）跨入 `Vec`，向量 `add` 在其上运行，结果被存储。

**之前**（InferTileMemorySpace 之后的混合 `InCore`）：

```python
@pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
def split_auto(qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
               out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]]):
    popped: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec] = pl.tile.move(qk, target_memory=pl.Mem.Vec)
    y: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec] = pl.add(popped, popped)
    return pl.store(y, [0, 0], out_0)
```

**之后**：

```python
@pl.function(type=pl.FunctionType.InCore,
             attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True})
def split_auto(qk, out_0):
    subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
    popped: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.aiv_shard(qk, split=1)  # C->V, 半
    y: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.add(popped, popped)
    return pl.store(y, [0 + subblock_idx * 64, 0], out_0)
```

cube 操作数 `qk` 保持 `[128, 128]`；向量子区域折半为 `[64, 128]`，store 偏移按
subblock 本地化。

## 示例 —— vector→cube 边界保持全尺寸（UpDown）

V→C `tile.move` 变为 `tile.aic_gather`；对折叠后 tile 的 cube 放置 move 保持全尺寸
`[128, 128]` `Mat`——cube 侧绝不会看到折半 tile：

```python
# `v` 是 affinity gate 产出的每 lane 折半 tile，例如 [64, 128]。
gathered_mat: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat] = pl.tile.aic_gather(v, split=1)
gathered:     pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat] = pl.tile.move(gathered_mat,
                                                                      target_memory=pl.Mem.Mat)
```

**操作数必须是每 lane 的折半 tile。** `tile.aic_gather` 声明为 HALF → FULL，因此
gather 把 `[64, 128]` 加倍为 `[128, 128]`，恰好等于 cube 放置 move 所保留的全尺寸
结果类型。这种一致性是**前置条件**而非保证：向量值可能以未折半的形式到达边界——
例如直接使用的 `Vec` 参数，或 split 维为 1 而被 affinity gate 特意保留的 tile。对
这类操作数做加倍会得到 `[256, 128]` 的 gather，而其后的 move 仍是 `[128, 128]`，
这与 `tile.move` 的保形契约矛盾，且产生的 IR 无法通过 print→parse 往返。既然没有
折半就不存在正确的 gather，本 pass 会以可操作的 `ValueError` **拒绝**它：

```text
LowerAutoVectorSplit: the V->C boundary tile.move here carries a full-width
vector operand 'vec'. tile.aic_gather reassembles the two AIV lanes' per-lane
halves into the full tile the cube expects, so its operand must be a value the
split halving produced (a tile.load / tile.slice / elementwise result inside the
vector sub-region). An un-halved value has no half to gather — either derive the
per-lane half first (load or slice the value inside the split function) and move
that to the cube side, or, if the split axis is a singleton that cannot be
halved, keep the value on the vector side.
```

**gather 沿操作数自身的 split 轴，而非函数的 split 轴。** `tile.reshape` 可能迁移
split 轴——rms_norm 的 `[N, 1] ↔ [1, N]` 列 reshape 会把它从 dim 0 移到 dim 1——
而 `TileInfo::split_dim` 记录了它最终所在的位置。gather 使用**该维度**对应的
`split` 编码发射（`dim 0 → 1`，`dim 1 → 2`），因此在 `UpDown` 函数中，lane 本地
的 `[1, 8]` 操作数经由 `split=2` gather 为 `[1, 16]`，与 move 一致；若改为对函数
轴加倍则会得到 `[2, 8]`。若追踪到的 split 维不在 `{0, 1}` 内，则无法表示为 2D
gather 的 `split` 属性，会被拒绝。

gather 结果是 `Mat` 而非 `Vec`：边界算子声明的类型指的是**消费侧** lane 的空间，
而 AIC 会把 V→C 传输 pop 进 L1。（`Vec` 指的是*生产侧* lane，与镜像算子
`tile.aiv_shard` 相矛盾——后者为其 cube 产出的操作数声明向量侧的 `Vec`。）随后的
cube 放置 move 才把 tile 放到最终的操作数空间——matmul 操作数为 `Mat → Left`；
此处的 `Mat → Mat` 是空操作，仅因本 pass 保留了作者原有的 move 而存在。

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

```cpp
Pass LowerAutoVectorSplit();
```

**实现**：`src/ir/transforms/lower_auto_vector_split_pass.cpp`

- `LowerFunction` / `LowerStmts` —— 边界改写 + 亲和性门控折半。
- `MakeReshapeOpCall` —— 构造 `tile.aiv_shard` / `tile.aic_gather` 调用。
- `CheckNoCubeTileHalved` —— cube 操作数完整性兜底。
- `WithSplitAivAttrs` —— 打 `split` + `split_aiv`。

**共享机制**：`src/ir/transforms/utils/split_axis_utils.cpp`
（`ProcessStmts`、`InjectSubblockIdx`、`SplitDimension`、`IsReduceOnSplitAxis`）
—— 逐算子向量折半，与 `SplitVectorKernel` 的独立拆分分支
（`ProcessStandaloneSplitFunction`）以及 `AivSplitValid` 校验器
（`SplitDimension` / `IsReduceOnSplitAxis`）共享。

**Python 绑定**：`python/bindings/modules/passes.cpp`

```cpp
passes.def("lower_auto_vector_split", &pass::LowerAutoVectorSplit, ...);
```

**测试**：`tests/ut/ir/transforms/test_lower_auto_vector_split.py` 以及
`tests/st/codegen/torch/test_torch_codegen_cross_core.py` 中端到端 `pl.split`
golden 场景（`test_lower_auto_vector_split_golden`）。

## 相关

- [`ResolveBackendOpLayouts`](20-resolve_backend_op_layouts.md) —— 紧邻其前运行。
- [`ExpandMixedKernel`](22-expand_mixed_kernel.md) —— 紧邻其后运行；把
  `tile.aiv_shard` / `tile.aic_gather` 折叠为带拆分标记的 `tpush`/`tpop`。
- [`SplitVectorKernel`](24-split_vector_kernel.md) —— 下游；仅为本 pass 产生的
  `split_aiv` 函数打属性，外加无拆分 dual-AIV 路径。
