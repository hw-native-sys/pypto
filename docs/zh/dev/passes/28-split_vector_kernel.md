# SplitVectorKernel Pass

显式 `split_aiv` kernel 已包含用户编写的每 lane 计算和降低后的 `tpush/tpop`
边界。本 pass 仅标记 dispatch 属性，不再次拆半。`LowerAutoVectorSplit` 不再生成 AUTO 分片。

携带 split 元数据的独立 AIV/AIC 函数仍使用 `ProcessStandaloneSplitFunction`
和共享 `split_axis::ProcessStmts` 改写尺寸与偏移；这条路径不属于 mix AUTO 拆半。

未拆分 mix 保留 backend 所需的双 AIV 握手重放：lane 0 计算，lane 1 使用空有效尺寸
执行必要通信协议。

Lane-1 的 `IfStmt` tile 结果保留原合流结果的布局和填充元数据，并将有效形状设为零；
后续使用引用重建后的结果变量。分支 tile 可以具有不同的填充元数据，
因此重放不能要求完整的结构相等，也不能将某个分支的填充属性用作合流结果。
当两分支的重放 yield 类型一致时，张量结果采用共同类型，保留原 backing view。
否则非 tile 结果保留原合流类型；对于形状不同的目标张量，
通过仅修改元数据的 `tensor.view` 别名恢复该契约，不写入目标。
复用合流类型之前，shape 和 view 表达式会映射到回放 lane 的标量定义，
避免克隆时错误绑定到 lane 0 的局部变量。
插入的别名保留在末尾的 runtime 或 AIV-split scope 内，
保留 scope 属性及 yield 使用的值定义。

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

### 分派契约

带 `split_aiv` 标记的 AIV 函数始终获得 `dual_aiv_dispatch=True`，包括没有统一
函数级 split mode 的 NONE 和多模式区域。存在统一非 NONE 模式时予以保留；
通信算子上逐次传输的 split mode 仍是权威数据。

```text
explicit split_aiv AIV/AIC -> stamp attrs, preserve body
standalone AIV/AIC with split metadata -> lower standalone split body
unsplit AIV requiring dual dispatch -> build lane-1 handshake replay
other functions -> unchanged
```

## 算法 —— 无拆分双 AIV 派发

`ProcessNoSplitDualAivFunction` 仅当 `RequiresNoSplitDualAivSync(func)` 为真时触发
——后端为 Ascend910（或任何 `BackendHandler::RequiresNoSplitDualAivDispatch()` 返回
真的后端），函数为 AIV，且 `attrs["dual_aiv_dispatch"]` 为真。

```text
1. 保留未改变的签名及参数身份，包括 shape 符号。
2. InjectSubblockIdx —— 前置 `subblock_idx = tile.get_subblock_idx()`。
3. 去掉开头的 subblock_idx 赋值，再切出共享 pipe-setup 前缀：
   SplitNoSplitSharedPipeSetupPrefix 取 reserve_buffer / import_peer_buffer /
   aic_initialize_pipe / aiv_initialize_pipe 语句的最大前缀，使其在原位置于两条
   lane 上都运行。
4. Lane 0 函数体 = 原分支语句（不变）。
5. Lane 1 函数体 = BuildNoSplitLane1ReplayStmts(分支语句)：
     - tile.store：丢弃 EvalStmt 形式；对 AssignStmt 形式透传目标 tensor，使 SSA
       使用者仍见到值，但不发生写入。别名类型使用目标 tensor 的类型及 shape 符号。
     - 别名：继承替换值的类型，并传播到后续使用处。
     - 其他产生 TileType 的调用：经 RebuildLane1CallWithZeroValidShape 重写——
       tile.load 变为 valid_shape=[0, 0] 的 tile.create；tile.slice / tile.transpose /
       tile.fmods / tile.rems / tile.row_max / tile.row_min / tile.row_sum /
       tile.row_prod 也变为空 tile.create，因为它们无法安全处理零有效范围。
       tile.set_validshape 的 valid_shape 置零；其余清空结果 valid_shape。
     - 跨核 tile.tpush_* / tile.tpop_* / system.tfree_* 保留，使 AIC↔AIV 握手对称。
     - For/While/If 以分叉的 replacements 映射递归。
6. 将 lane 0 / lane 1 包装为 `if subblock_idx == 0: <lane 0> else: <lane 1>`。
7. 新函数体 = subblock_idx 赋值；提升的共享 pipe-setup；分支 IfStmt。
8. 对函数体执行 DeepClone；属性不变（dual_aiv_dispatch=True 保留）。
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

### 示例 1 —— split_aiv 属性打标

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

### 示例 2 —— Ascend910 无拆分双 AIV 派发

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
  `IsNoSplitSharedPipeSetupCall` —— Ascend910 无拆分路径。

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

- [`LowerAutoVectorSplit`](25-lower_auto_vector_split.md) —— 显式区域降低路径；
  产生本 pass 打标的 `split_aiv` 函数。逐算子向量折半规则位于该处及
  `split_axis_utils`。
- [`ExpandMixedKernel`](26-expand_mixed_kernel.md) —— AIC/AIV 函数与
  `dual_aiv_dispatch` 标记的上游生产者。
- [`InjectGMPipeBuffer`](27-inject_gm_pipe_buffer.md) —— 紧邻其前运行；本 pass 依赖
  的后端门控 GM pipe 缓冲布线。
- [`NormalizeReturnOrder`](30-normalize_return_order.md) —— 紧邻其后运行。
