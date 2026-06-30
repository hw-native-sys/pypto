# LowerAutoVectorSplit Pass（向量自动拆分下降）

在 `ExpandMixedKernel` **之前**，将带 AUTO `pl.split` 的混合 `InCore` 函数转换为
**显式 `split_aiv` 形态**：在 cube→vector 边界插入 `tile.aiv_shard`，在
vector→cube 边界插入 `tile.aic_gather`，仅对**向量子区域**沿拆分轴折半，注入
`tile.get_subblock_idx()`，并在函数上打 `split` + `split_aiv` 标记。

这是**唯一的自动拆分下降路径**：它始终运行，紧邻 `ExpandMixedKernel` 之前。运行后
每个拆分函数到达 [`SplitVectorKernel`](21-split_vector_kernel.md) 时都已带
`split_aiv` 标记，因此该 pass 只打属性（其 split_aiv 分支）——其旧的逐算子折半驱动
已被删除，折半机制现仅存于 `split_axis_utils`，由本 pass 共享。

本 pass 同时是一等公民区域节点 `SplitAivScopeStmt`（`for aiv_id in
pl.split_aiv(...)`）的**唯一消费者**。该区域作为结构节点存活于 parse → SSA →
`ResolveBackendOpLayouts`；在此处每个区域被就地下降，作用域包装被**擦除**，因此没有
任何 `SplitAivScopeStmt` 会到达 `ExpandMixedKernel`（pass 22）或 codegen。

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
| Required | `SSAForm` |
| Produced | `SSAForm` |
| Invalidated | — |

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

## 显式 `SplitAivScopeStmt` 区域路径

除上述 AUTO 整函数路径外，函数体仍携带一个或多个 `SplitAivScopeStmt` 区域的
`InCore` 函数走一条独立的**区域路径**（`LowerExplicitRegionFunction`），它在 AUTO
路径**之前**判定。每个区域携带各自的 `split_` 模式，因此可处理单一函数级模式无法表达
的多模式情形。区域局部的 `tile_vars` / `var_replacements` 映射保证折半后的变量不会泄漏
到同级区域或区域外的全宽算子。任何区域**之外**的语句以全宽发出。所有区域下降后，作用域
包装被丢弃，函数被打上 `split_aiv` + `split_aiv_region_validated`（后者通知
[`ExpandMixedKernel`](21-expand_mixed_kernel.md) 跳过其单一函数级模式的转置检查——
改由 pass 21 用每个区域正确的拆分轴校验各自的转置风险）。

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
  `aiv_id = get_subblock_idx()` 绑定已携带 lane 信息）。`None` 区域内的 `tile.aiv_shard` /
  `tile.aic_gather` 会被拒绝（无拆分轴可切分）——由 `AivSplitValid` 校验器与此处的常开保护
  共同拦截。该函数仍会被标记 `split_aiv`，因此下游 [`ExpandMixedKernel`](21-expand_mixed_kernel.md) /
  `SplitVectorKernel` 会把它派发到**两个** AIV lane（经由 `dual_aiv_dispatch`），而**非**
  lane-0-only 的非拆分 replay（后者只针对非 `split_aiv` 核）——故两个 lane 都运行完整函数体。
  当区域的 tile 无法折半（单位维）或归约必须保持全宽时使用本模式。

由于区域经由通用的 `BeginScope`/`EndScope` 构建且不被提取，它可**嵌套**在 `pl.range` /
`pl.pipeline` 循环或 `if` 之内；区域路径会递归进入复合语句，找到并下降每个区域，同时保留
外围控制流。

## 拆分轴分派

| `SplitMode`（int） | 拆分轴 | 折半的向量子区域 |
| ------------------ | ------ | ---------------- |
| `None`（0） | —（无拆分轴） | 不折半——任务并行；tile 保持全宽，由 `aiv_id` 分派两个 lane |
| `UpDown`（1） | 维 0（高度） | 行 |
| `LeftRight`（2） | 维 1（宽度） | 列 |

`SplitDimension(mode)` 对 `UpDown` 返回 `0`，对 `LeftRight` 返回 `1`
（`split_axis_utils`）；对 `None` **不调用**（区域路径先对 `None` 分支——无轴可推导）。

## 算法

`LowerFunction` 改写一个混合 `InCore` 函数：

```text
1. split_dim = SplitDimension(mode); split_int = int(mode)。
2. InjectSubblockIdx(func, is_aiv=true) 在函数体顶部插入
       subblock_idx = tile.get_subblock_idx()
   （若 'subblock_idx' 已占用则取新名）。
3. LowerStmts 遍历扁平函数体：

   边界 tile.move（ClassifyMoveDirection）：
     CUBE_TO_VECTOR —— 将 move 替换为
         tile.aiv_shard(full_cube_tile, split=int(mode))   -> 半
       把 move 的目标内存（Vec）重新附加到推导出的半类型上，将结果 var 连同其半
       尺寸种入 tile_vars，并记录 旧->新 var 重绑。cube 源（matmul / Acc 结果）
       保持全尺寸。
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
gathered_mat: pl.Tile[[..], pl.FP32, pl.Mem.Vec]  = pl.tile.aic_gather(vec, split=1)
gathered:     pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat] = pl.tile.move(gathered_mat,
                                                                      target_memory=pl.Mem.Mat)
```

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

- [`ResolveBackendOpLayouts`](17-resolve_backend_op_layouts.md) —— 紧邻其前运行。
- [`ExpandMixedKernel`](19-expand_mixed_kernel.md) —— 紧邻其后运行；把
  `tile.aiv_shard` / `tile.aic_gather` 折叠为带拆分标记的 `tpush`/`tpop`。
- [`SplitVectorKernel`](21-split_vector_kernel.md) —— 下游；仅为本 pass 产生的
  `split_aiv` 函数打属性，外加无拆分 dual-AIV 路径。
