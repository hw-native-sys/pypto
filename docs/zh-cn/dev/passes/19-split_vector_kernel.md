# SplitVectorKernel Pass

将 vector kernel 沿一个 tile 轴拆分，使两个 AIV lane 各自承担一半工作；该
Pass 会把 per-lane 的 tile 形状减半，并重写 `tile.load`、`tile.store`、
`tile.tpop_from_aic` 以指向各自的那一半。在 Ascend910B 上，本 Pass 还
负责 **no-split 双 AIV 派发** 路径：当 `ExpandMixedKernel` 判断混合
kernel 不可拆分时，会给 AIV 函数打上 `dual_aiv_dispatch=True` 标记，本
Pass 据此把函数体包装为 `if subblock_idx == 0 ... else`，让 AIC↔AIV 跨核
握手在两条 lane 上仍然对称（即使只有 lane 0 做真实计算）。

## 概述

这两类重写共用一个 Pass，因为它们都依赖 `subblock_idx`，并且都需要维
护跨核 `tpush`/`tpop` 的对称性：

1. **拆分模式（split mode）** —— 由
   `Function::attrs["split"]`（`SplitMode::UpDown` 或
   `SplitMode::LeftRight`）或函数体内任意
   `tile.tpush_*` / `tile.tpop_*` 调用上的 `split=` kwarg 触发。AIC 侧
   只需要将所有跨核 op 上的 `split=` 同步；AIV 侧才会真正改写 shape：
   tile 在 split 轴上减半，`tile.load` / `tile.store` 的偏移量加上
   `subblock_idx * half_dim`，让每个 lane 取自己那一半，并把
   `tile.tpop_from_aic` 结果在 split 轴上减半。

2. **No-split 双 AIV 派发** —— 仅在后端
   `BackendHandler::RequiresNoSplitDualAivDispatch()` 返回 `true`
   （目前只有 Ascend910B）且 AIV 函数被
   `ExpandMixedKernel` 打上 `dual_aiv_dispatch=True` 标记时启用，参见
   [`ExpandMixedKernel`](17-expand_mixed_kernel.md) 中的「no function
   split mode」段落。本 Pass 注入 `subblock_idx`，把
   `reserve_buffer`、`import_peer_buffer`、`aic_initialize_pipe`、
   `aiv_initialize_pipe` 这些共享 pipe-setup 调用从分支前缀中外提到分
   支之外，并发出一个 `IfStmt`：then 分支保留原始函数体，else 分支
   通过 replay 保留所有跨核 `tpush`/`tpop`/`tfree`，但让所有产生 tile
   的 op 把结果 `valid_shape` 强制为 `[0, 0]`，并丢弃用户可见的
   `tile.store` 写回。

`ResolveSplitMode` 决定走哪一条：

- 若 `attrs["split"]` 为非 `None`，以它为准（函数体中跨核 op 的
  `split=` 必须一致，否则 `ValueError`）。
- 否则由 `CrossCoreSplitCollector` 扫描函数体，使用唯一非零的
  `split=` 作为推断模式。
- 函数体内不同跨核 op 上的 `split=` 取值不一致时抛 `ValueError`。
- 若函数为 AIV 且 `dual_aiv_dispatch=True`，且解析得到的拆分模式为
  `None`，则改走 no-split 双 AIV 派发路径。

### 拆分轴

| `SplitMode`（int） | 拆分轴 | 减半维度 | `tile.load` / `tile.store` 偏移修正 |
| ------------------ | ------ | -------- | ----------------------------------- |
| `None`（0） | — | — | 该函数视为 no-op |
| `UpDown`（1） | dim 0（高度） | 行数 | `[orig + subblock_idx * H/2, orig]` |
| `LeftRight`（2） | dim 1（宽度） | 列数 | `[orig, orig + subblock_idx * W/2]` |

`subblock_idx` 由 `pl.tile.get_subblock_idx()` 物化，由
`InjectSubblockIdx` 注入为重写后 AIV 函数体的第一条语句。若
`subblock_idx` 与已有 param/局部变量重名，由
`auto_name::GenerateFreshNameLike` 生成新的名字以避免冲突。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::SplitVectorKernel()` | `passes.split_vector_kernel()` | 程序级 |

```python
from pypto import passes
result = passes.split_vector_kernel()(program)
```

## Pass 属性

| 属性 | 取值 |
| ---- | ---- |
| 前置（Required） | `SSAForm`、`MixedKernelExpanded` |
| 产出（Produced） | `SSAForm`、`VectorKernelSplit`、`NormalizedStmtStructure` |
| 失效（Invalidated） | — |

`MixedKernelExpanded` 来自上游契约，保证不再有 `FunctionType::InCore`
函数同时混合 Cube 和 Vector 操作，并且 AIC↔AIV 跨核 op 已就位。
`VectorKernelSplit` 表示已对 `attrs["split"]` 非 `None` 的 AIV 函数完
成 tile 形状、`tile.tpop_from_aic` 结果、`tile.load`/`tile.store` 偏
移的 per-lane 调整。来源：`include/pypto/ir/transforms/pass_properties.h`、
`include/pypto/ir/transforms/ir_property.h`。

## 算法 —— 拆分模式

`ProcessFunction` 重写 `ResolveSplitMode` 解析为
`UpDown` 或 `LeftRight` 的单个 AIC 或 AIV 函数：

```text
1. 解析拆分模式与拆分轴：
   split_dim = (mode == UpDown) ? 0 : 1
2. 克隆 params（保持原 name 与 type），登记到 var_replacements，使重写
   后的函数体仍持有同一份 param 身份。
3. （仅 AIV）InjectSubblockIdx 在函数体头部插入：
       subblock_idx = tile.get_subblock_idx()
   若 'subblock_idx' 已被占用则改用一个新名字。
4. 通过 ProcessStmt 遍历每条语句：

   tile.tpush_to_aiv / tile.tpush_to_aic / tile.tpop_from_aiv：
     RebuildCallWithSplit —— 仅同步 `split=` kwarg。AIC 保留完整操作
     数 tile（cube 仍然消费整张 matmul 输出）。

   tile.tpop_from_aic（仅 AIV）：
     RebuildTpopWithHalvedShape —— 在 split_dim 上将结果 shape 减半，
     按 subblock 局部化 TileView.valid_shape，并同步 `split=`。

   tile.load（仅 AIV，≥4 个参数）：
     若结果 tile 在 split 轴上是 singleton（如 UpDown 下的 [1, 128]）
     保持原状；
     若 tile 是低秩（rank < split_dim+1）保持原状；
     否则减半结果 shape、减半静态 shape 参数、局部化 valid_shape，
     并把 split 轴偏移加 `subblock_idx * H/2`。

   tile.store（仅 AIV，≥3 个参数）：
     若源 tile 在 tile_vars 中（早先被减半），把它的 split 轴偏移加
     `subblock_idx * H/2`。

   产生 TileType 的其他 tile.* op（仅 AIV）：
     在 split_dim 上减半结果 shape；tile.full / tile.create 还会减半
     静态 shape 参数。命中 split 轴的 reduce op 由 IsReduceOnSplitAxis
     拦截抛 ValueError（单 subblock 内的部分 reduce 语义不正确）。

   ForStmt：
     若 iter_args 的 initValue 是被追踪的 halved tile，则重建 iter_arg
     使其类型也减半；同时按 iter_arg 类型重建 return_vars。递归处理函
     数体；loop-carried state 由 loop_repair::RebuildForStmt 修复。

   IfStmt / SeqStmts：
     递归处理分支与语句序列。

5. 重写完成后，transform_utils::Substitute 应用 var_replacements，确
   保所有引用（param、iter_arg、return_var、tpop 结果）看到的是重写
   后的 Var。
6. 调用 DeepClone，避免与共享 IR 子树纠缠。
7. WithSplitAttr 把解析得到的 SplitMode 写回 Function::attrs（覆盖原
   有 `split` 项）。
```

`tile_vars` 是 Pass 内部的映射，记录哪些 `Var` 携带 halved tile 以及
对应的 `half_dim_size`。它让在循环外发出的 `tile.store` 也能识别到
循环内 `tile.load` 已经把源 tile 减半这一事实。

## 算法 —— No-split 双 AIV 派发

`ProcessNoSplitDualAivFunction` 仅在
`RequiresNoSplitDualAivSync(func)` 为 true 时触发，即后端为
Ascend910B（或任意
`BackendHandler::RequiresNoSplitDualAivDispatch()` 返回 true 的后端）、
函数为 AIV、`attrs["dual_aiv_dispatch"]` 为 true。它**取代**
`ProcessFunction`（同一函数永远不会同时走两条路径）。

```text
1. 克隆 params 到 param_replacements（与拆分模式一致）。
2. InjectSubblockIdx —— 注入 `subblock_idx = tile.get_subblock_idx()`。
3. 剥离 body 头部的 subblock_idx 赋值后，再切出共享 pipe-setup 前缀：
     SplitNoSplitSharedPipeSetupPrefix 取最长前缀，仅包含
     reserve_buffer / import_peer_buffer / aic_initialize_pipe /
     aiv_initialize_pipe（参见 IsNoSplitSharedPipeSetupCall），让它们
     在两条 lane 之外保持原位执行。
4. Lane 0 = 剩余的分支语句（保持原状）。
5. Lane 1 = BuildNoSplitLane1ReplayStmts(分支语句)：
     - tile.store：EvalStmt 形式整体丢弃；AssignStmt 形式让 LHS 直
       接 passthrough 第三个参数（目标 tensor），让 SSA 后续仍能看
       到一个值，但不再发生写入。
     - 任意产生 TileType 的 call：通过
       RebuildLane1CallWithZeroValidShape 改写 —— `tile.load` →
       `tile.create`，结果类型携带 `valid_shape=[0, 0]`；
       `tile.slice`、`tile.set_validshape` 的 valid_shape 参数清零；
       其他 op 的结果类型 `valid_shape` 全部清空。
     - 跨核 tile.tpush_* / tile.tpop_* / system.tfree_* 全部保留，
       使 AIC↔AIV 握手保持平衡。
     - For/While/If 用 fork 后的 replacements 递归处理，避免分支内
       的 SSA 重命名串到兄弟语句。
6. 用以下结构包装 lane 0 与 lane 1：
       if subblock_idx == 0:
           <lane 0>
       else:
           <lane 1>
7. 新 body =
     subblock_idx 赋值
     <外提的共享 pipe-setup>
     <分支 IfStmt>
8. Substitute / DeepClone；attrs 不变（`dual_aiv_dispatch=True` 仍然
   保留 —— 后续 lowering 会读它）。
```

## 约束

| 约束 | 原因 |
| ---- | ---- |
| 拆分轴必须是偶数（动态维度走 `// 2`） | `ComputeHalfDimSize`：`ConstInt` 为奇数时直接抛错；动态维度发出 `MakeFloorDiv(dim, 2)` |
| 函数级与跨核 op `split=` 模式冲突 | `ResolveSplitMode` 抛 `ValueError` |
| 函数体内多个跨核 op 的 `split=` 不一致 | `CrossCoreSplitCollector` 抛 `ValueError` |
| split 轴上的 reduce 被禁止 | `IsReduceOnSplitAxis` 抛错 —— 单 subblock 内的部分 reduce 语义不正确 |
| split 轴 singleton 维度保持原状 | UpDown 下的 `[1, 128]` 或 LeftRight 下的 `[16, 1]` 等广播 tile |
| 低秩 tile 跳过拆分改写 | LeftRight（split dim=1）下的 rank-1 `tile.load` 保持原样 |
| AIC 上的 `tile.tpop_from_aiv` 保持完整 shape | cube 仍需消费整张 matmul 操作数；只同步 `split=` |
| No-split lane 1 不可产生可见写回 | `tile.store` 写入被丢弃；产生 tile 的 op 强制 `valid_shape=[0, 0]`，PTO op 以空 tile 形式执行 |

## 示例

### 示例 1 —— UpDown：tpop 减半 + store 偏移调整

提炼自 `tests/ut/ir/transforms/test_split_vector_kernel.py` 中的
`test_tpop_shape_halved_and_store_offset_adjusted`。AIC 函数体保持
不变，仅同步 `split=`；AIV 完整经历减半与偏移修正。

**Before**:

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
    def main_aic(self, x: pl.Tensor[[16, 128], pl.BF16], y: pl.Tensor[[128, 128], pl.BF16]):
        x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
        x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
        y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
        y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
        z_tile = pl.matmul(x_left, y_right)
        pl.tpush_to_aiv(z_tile, split=0)        # split=0 表示 "None" 哨兵值

    @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
    def main_aiv(self, out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]):
        z_vec: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView()] = pl.tpop_from_aic(split=0)
        return pl.store(z_vec, [0, 0], out_0)
```

**After**:

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
    def main_aic(self, x, y):
        # ... cube ops 不变 ...
        pl.tpush_to_aiv(z_tile, split=1)        # 仅同步 split kwarg

    @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
    def main_aiv(self, out_0):
        subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
        z_vec: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=1)
        return pl.store(z_vec, [0 + subblock_idx * 8, 0], out_0)
```

### 示例 2 —— LeftRight：宽度减半，dim-1 偏移调整

提炼自 `test_load_shape_halved_left_right`。AIV 同时包含
`tile.load` 与 `tile.tpop_from_aic`；两者最终都通过
`subblock_idx * 64` 各自落到源数据的右半区域。

**Before**:

```python
@pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.LEFT_RIGHT})
def main_aiv(self, data: pl.Tensor[[16, 128], pl.FP32],
             out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]):
    prev = pl.load(data, [0, 0], [16, 128], target_memory=pl.Mem.Vec)
    pop_tile: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView()] = pl.tpop_from_aic(split=0)
    result = pl.add(prev, pop_tile)
    return pl.store(result, [0, 0], out_0)
```

**After**:

```python
@pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.LEFT_RIGHT})
def main_aiv(self, data, out_0):
    subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
    prev: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.load(
        data, [0, 0 + subblock_idx * 64], [16, 64], target_memory=pl.Mem.Vec)
    pop_tile: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=2)
    result: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.add(prev, pop_tile)
    return pl.store(result, [0, 0 + subblock_idx * 64], out_0)
```

### 示例 3 —— Ascend910B no-split 双 AIV 派发

提炼自
`test_no_split_dual_dispatch_producer_replays_compute_and_tpush_on_lane1`。
AIV 函数携带 `dual_aiv_dispatch=True`（由 `ExpandMixedKernel` 为
no-split 混合 kernel 打上），且无 `split` 属性。本 Pass 让 lane 0 做
真正的工作，lane 1 重建为空 tile replay，使
`tpush_to_aic` 握手仍然发生两次。

**Before**:

```python
@pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
def main_aiv(self, a, b, out):
    slot_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=4096, base=0x1000)
    pl.aiv_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=slot_buf)
    a_tile = pl.load(a, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
    b_tile = pl.load(b, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
    summed = pl.add(a_tile, b_tile)
    pl.tpush_to_aic(summed, split=0)
    return out
```

**After**（保持 shape 不变 —— lane 1 通过 `valid_shape=[0, 0]` 携带空
tile）:

```python
@pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
def main_aiv(self, a, b, out):
    subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
    slot_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=4096, base=0x1000)
    pl.aiv_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=slot_buf)
    if subblock_idx == 0:
        a_tile = pl.load(a, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
        b_tile = pl.load(b, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
        summed = pl.add(a_tile, b_tile)
        pl.tpush_to_aic(summed, split=0)
    else:
        # tile.load 改写为 tile.create，valid_shape=[0, 0]
        a_tile: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = \
            pl.tile.create([16, 16], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        b_tile: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = \
            pl.tile.create([16, 16], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        summed: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = \
            pl.add(a_tile, b_tile)
        pl.tpush_to_aic(summed, split=0)        # 握手仍然触发
    return out
```

`reserve_buffer` 与 `aiv_initialize_pipe` 被外提到 `if`/`else` 之外，
两条 lane 共享同一份 buffer 状态。

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

```cpp
Pass SplitVectorKernel();
```

**实现**：`src/ir/transforms/split_vector_kernel_pass.cpp`

- `ResolveSplitMode` —— 在函数级 attr 与函数体推断之间挑选拆分模式。
- `ProcessFunction` / `ProcessStmt` / `ProcessStmts` —— 拆分模式重写。
- `RebuildCallWithSplit` / `RebuildTpopWithHalvedShape` —— 跨核 call
  改写器。
- `HalveTileShape` / `ApplyTrackedTileShape` /
  `LocalizeValidDimForSplit` —— TileType 改写器。
- `AdjustOffsets` —— `tile.load`/`tile.store` 在 split 轴上的偏移修正。
- `IsReduceOnSplitAxis` —— 部分 reduce 错误的守卫。
- `RequiresNoSplitDualAivSync` / `ProcessNoSplitDualAivFunction` /
  `BuildNoSplitLane1ReplayStmts` / `RebuildLane1CallWithZeroValidShape` /
  `IsNoSplitSharedPipeSetupCall` —— Ascend910B no-split 路径。

**Pass 属性**：`include/pypto/ir/transforms/pass_properties.h`

```cpp
inline const PassProperties kSplitVectorKernelProperties{
    .required = {IRProperty::SSAForm, IRProperty::MixedKernelExpanded},
    .produced = {IRProperty::SSAForm, IRProperty::VectorKernelSplit,
                 IRProperty::NormalizedStmtStructure}};
```

**Python 绑定**：`python/bindings/modules/passes.cpp`

```cpp
passes.def("split_vector_kernel", &pass::SplitVectorKernel,
           "Create a pass that splits vector kernels based on SplitMode "
           "(adjusts tpush/tpop split, halves tpop shapes, adjusts store offsets)");
```

**类型 stub**：`python/pypto/pypto_core/passes.pyi`

```python
def split_vector_kernel() -> Pass:
    """Create a pass that splits vector kernels based on SplitMode."""
```

**测试**：`tests/ut/ir/transforms/test_split_vector_kernel.py`
（`TestSplitVectorKernelUpDown`、`TestSplitVectorKernelLeftRight`、
`TestSplitVectorKernelNoSplitA2A3`）。

## 相关文档

- [`ExpandMixedKernel`](17-expand_mixed_kernel.md) —— 上游 Pass，产生
  AIC/AIV 函数和 `dual_aiv_dispatch` 标记。
- [`InjectGMPipeBuffer`](18-inject_gm_pipe_buffer.md) —— 紧邻上游；本
  Pass 依赖它布置好的 backend-gated GM pipe buffer。
- [`NormalizeReturnOrder`](20-normalize_return_order.md) —— 紧邻下游；
  会观察到本 Pass 产出的 per-lane tile 形状。
