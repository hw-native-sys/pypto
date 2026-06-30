# SplitVectorKernel Pass

经过分阶段收敛重构后，`SplitVectorKernel` 只剩两项窄职责，**不再折半任何函数体**：

1. **`split_aiv` 属性打标** —— 本 pass 唯一的拆分路径。`split_aiv` 核（手写，或由上游
   [`LowerAutoVectorSplit`](18-lower_auto_vector_split.md) 产生）已经把其显式
   `tile.aiv_shard` / `tile.aic_gather` 下降为带拆分标记的 `tpush`/`tpop`（由
   `ExpandMixedKernel` 折叠），并携带已折半的计算 tile 与自己的
   `tile.get_subblock_idx()`。本 pass 不动函数体，只在函数属性上打 `split`（对 AIV
   函数另加 `dual_aiv_dispatch`）。

2. **无拆分双 AIV 派发** —— 在 Ascend910B（任何
   `BackendHandler::RequiresNoSplitDualAivDispatch()` 返回 `true` 的后端）上，当
   `ExpandMixedKernel` 判断混合核不可拆分时，会给 AIV 函数打 `dual_aiv_dispatch=True`。
   本 pass 据此把函数体包装为按 lane 的 `if subblock_idx == 0 ... else` 重放，使
   AIC↔AIV 跨核握手在两条 lane 上仍对称（即使只有 lane 0 做真实计算）。

> **历史说明。** 本 pass 曾驱动逐算子 AIV 折半（`ProcessFunction` /
> `ResolveSplitMode` / `CrossCoreSplitCollector`）。该驱动在
> `LowerAutoVectorSplit` 成为在用自动拆分下降路径后被删除：在其运行后，每个拆分函数
> 到达本 pass 时都已带 `split_aiv` 标记，此处再折半会把已折半的函数体二次折半。折半
> 机制本身（形状折半、偏移本地化、`tile.slice` 参数折半、拒绝拆分轴归约、循环跟踪）
> 现位于 `split_axis_utils`，由 `LowerAutoVectorSplit` 调用——逐算子改写规则见该 pass
> 文档。

## API

| C++ | Python | 层级 |
| --- | ------ | ---- |
| `pass::SplitVectorKernel()` | `passes.split_vector_kernel()` | Program 级 |

```python
from pypto import passes
result = passes.split_vector_kernel()(program)
```

## Pass 属性

| 属性 | 值 |
| ---- | -- |
| Required | `SSAForm`、`MixedKernelExpanded` |
| Produced | `SSAForm`、`VectorKernelSplit`、`NormalizedStmtStructure` |
| Invalidated | — |

`MixedKernelExpanded` 是上游契约：没有 `FunctionType::InCore` 函数仍混用 Cube 与
Vector 算子，且 AIC↔AIV 跨核算子已就位。`VectorKernelSplit` 表明拆分 AIV 函数已为
per-lane 形态（由上游 `LowerAutoVectorSplit` + `ExpandMixedKernel` 达成；本 pass 经
属性打标分支予以确认）。来源：`include/pypto/ir/transforms/pass_properties.h`、
`include/pypto/ir/transforms/ir_property.h`。

### 退出时的函数属性不变式

本 pass 以 `attrs["dual_aiv_dispatch"]` 作为双 AIV 派发决策的唯一来源，并维护：

```text
带非 None 拆分模式的 split_aiv 函数  ⇒  attrs["dual_aiv_dispatch"] == true（AIV）
```

带 `split_aiv` 但无函数级拆分模式的函数是 `INTERNAL_CHECK` 失败
（`OutlineIncoreScopes` / `LowerAutoVectorSplit` 必须把 `split` 与 `split_aiv`
一同传播）。编排 codegen（`src/codegen/orchestration/orchestration_codegen.cpp` 中的
`RequiresDualAivDispatch`）只读该属性，绝不从 `SplitMode` 重新推导。

## 分派

```text
对每个函数：
  if (AIV 或 AIC) and attrs["split_aiv"]:
      # 唯一拆分路径 —— 打属性，函数体原样透传。
      assert 函数级拆分模式已设且非 None  (INTERNAL_CHECK)
      attrs = WithSplitAttrs(func, mode, is_aiv)            # split（AIV 另加 dual_aiv_dispatch）
  elif RequiresNoSplitDualAivSync(func):
      # Ascend910B 无拆分双 AIV 派发（正交路径）。
      ProcessNoSplitDualAivFunction(func)
  else:
      原样透传
```

## 算法 —— 无拆分双 AIV 派发

`ProcessNoSplitDualAivFunction` 仅当 `RequiresNoSplitDualAivSync(func)` 为真时触发
——后端为 Ascend910B（或任何 `BackendHandler::RequiresNoSplitDualAivDispatch()` 返回
真的后端），函数为 AIV，且 `attrs["dual_aiv_dispatch"]` 为真。

```text
1. 克隆参数到 param_replacements。
2. InjectSubblockIdx —— 前置 `subblock_idx = tile.get_subblock_idx()`。
3. 去掉开头的 subblock_idx 赋值，再切出共享 pipe-setup 前缀：
   SplitNoSplitSharedPipeSetupPrefix 取 reserve_buffer / import_peer_buffer /
   aic_initialize_pipe / aiv_initialize_pipe 语句的最大前缀，使其在原位置于两条
   lane 上都运行。
4. Lane 0 函数体 = 原分支语句（不变）。
5. Lane 1 函数体 = BuildNoSplitLane1ReplayStmts(分支语句)：
     - tile.store：丢弃 EvalStmt 形式；对 AssignStmt 形式透传目标 tensor，使 SSA
       使用者仍见到值，但不发生写入。
     - 其他产生 TileType 的调用：经 RebuildLane1CallWithZeroValidShape 重写——
       tile.load 变为 valid_shape=[0, 0] 的 tile.create；tile.slice /
       tile.set_validshape 的 valid_shape 置零；其余清空结果 valid_shape。
     - 跨核 tile.tpush_* / tile.tpop_* / system.tfree_* 保留，使 AIC↔AIV 握手对称。
     - For/While/If 以分叉的 replacements 映射递归。
6. 将 lane 0 / lane 1 包装为 `if subblock_idx == 0: <lane 0> else: <lane 1>`。
7. 新函数体 = subblock_idx 赋值；提升的共享 pipe-setup；分支 IfStmt。
8. Substitute / DeepClone；属性不变（dual_aiv_dispatch=True 保留）。
```

### Codegen 传输：整列 box、保留行

lane 1 重放将其 tile `valid_shape` 置零以不产生可见写入，但其保留的 AIC↔AIV `tpush`
仍通过共享 GM FIFO 槽移动数据，由单一 cube 消费者整列弹出。在 codegen 侧
（`EmitSplitTpushTransportValidShape`，`pto_ops_common.cpp`），用 `set_validshape`
收窄了 `valid_shape` 的无拆分双 AIV 生产者必须传输**整列 box**，否则消费者会读到
`valid_col` 之后的陈旧槽列。无拆分路径**仅加宽列并保留行 `valid_shape`**：subblock 0
的真实推送携带整列 box，而 subblock 1 的 `valid_shape=[0, 0]` 重放**完全不传输**
（静态 0 行推送不移动数据），因此它保持真正的 0 行无操作，而非把垃圾行赛入 subblock 0
的槽。检测开关是 `PTOCodegen::IsDualAivDispatchFunction()`，读取本 pass 的
`dual_aiv_dispatch` 属性。

## 示例

### 示例 1 —— split_aiv 属性打标（唯一拆分路径）

一对 `split_aiv` AIC/AIV 函数已为显式 per-lane 形态（折半的 `[8, 128]` 计算 tile，
一个 手写/`LowerAutoVectorSplit` 写入的 `get_subblock_idx`）。本 pass 打属性，函数体
不动。

```python
@pl.function(type=pl.FunctionType.AIV,
             attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True})
def main_aiv(self, out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]):
    subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
    z_vec: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=1)
    return pl.store(z_vec, [0 + subblock_idx * 8, 0], out_0)
```

pass 之后，`attrs` 增加 `dual_aiv_dispatch=True`；`z_vec` 保持 `[8, 128]`（不再折半），
且恰有一个 `get_subblock_idx`。

### 示例 2 —— Ascend910B 无拆分双 AIV 派发

提炼自 `test_no_split_dual_dispatch_producer_replays_compute_and_tpush_on_lane1`。
AIV 函数带 `dual_aiv_dispatch=True`（由 `ExpandMixedKernel` 为无拆分混合核设置），无
`split` 属性。

**之前**：

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

**之后**（lane 1 以 `valid_shape=[0, 0]` 携带空 tile）：

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
        a_tile: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = \
            pl.tile.create([16, 16], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        b_tile: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = \
            pl.tile.create([16, 16], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        summed: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = \
            pl.add(a_tile, b_tile)
        pl.tpush_to_aic(summed, split=0)        # 握手仍触发
    return out
```

`reserve_buffer` 与 `aiv_initialize_pipe` 被提升到 `if`/`else` 之上，使两条 lane 共享
相同的缓冲状态。

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

```cpp
Pass SplitVectorKernel();
```

**实现**：`src/ir/transforms/split_vector_kernel_pass.cpp`

- `WithSplitAttrs` —— split_aiv 分支上打 `split`（AIV 另加 `dual_aiv_dispatch`）。
- `RequiresNoSplitDualAivSync` / `ProcessNoSplitDualAivFunction` /
  `BuildNoSplitLane1ReplayStmts` / `RebuildLane1CallWithZeroValidShape` /
  `IsNoSplitSharedPipeSetupCall` —— Ascend910B 无拆分路径。

**属性**：`include/pypto/ir/transforms/pass_properties.h`

```cpp
inline const PassProperties kSplitVectorKernelProperties{
    .required = {IRProperty::SSAForm, IRProperty::MixedKernelExpanded},
    .produced = {IRProperty::SSAForm, IRProperty::VectorKernelSplit,
                 IRProperty::NormalizedStmtStructure}};
```

**Python 绑定**：`python/bindings/modules/passes.cpp`

```cpp
passes.def("split_vector_kernel", &pass::SplitVectorKernel, ...);
```

**测试**：`tests/ut/ir/transforms/test_split_vector_kernel.py`
（`TestSplitVectorKernelExplicitSplitAivBypass`、`TestSplitVectorKernelNoSplitA2A3`、
`TestSplitVectorKernelNoSplitPassthrough`）。逐算子折半测试已迁移至
`tests/ut/ir/transforms/test_lower_auto_vector_split.py`。

## 相关

- [`LowerAutoVectorSplit`](18-lower_auto_vector_split.md) —— 在用自动拆分下降路径；
  产生本 pass 打标的 `split_aiv` 函数。逐算子向量折半规则位于该处及
  `split_axis_utils`。
- [`ExpandMixedKernel`](19-expand_mixed_kernel.md) —— AIC/AIV 函数与
  `dual_aiv_dispatch` 标记的上游生产者。
- [`InjectGMPipeBuffer`](20-inject_gm_pipe_buffer.md) —— 紧邻其前运行；本 pass 依赖
  的后端门控 GM pipe 缓冲布线。
- [`NormalizeReturnOrder`](23-normalize_return_order.md) —— 紧邻其后运行。
