# LowerAutoVectorSplit Pass

保留公开 pass 名称以兼容现有调用。该 pass 现在为 `ExpandMixedKernel` 准备
**显式 AIV 区域**，不再隐式拆半 mix 计算，也不自动生成 shard/gather。

## 编写契约

保留 `pl.split_aiv(2, mode=...)`，用于指定 AIV 执行区域并绑定 lane ID。
AIC→AIV 边界显式写 `pl.aiv_shard`，AIV→AIC 边界显式写 `pl.aic_gather`。
它们继承区域的 mode；高层算子自身没有 `mode` 参数。

区域 body 已是每 lane 的计算。编译器不再隐式改写计算尺寸、load/store 偏移、
循环携带值或辅助输入切片。没有跨核数据的区域无需添加虚假的 shard/gather，
但仍由用户明确每 lane 的尺寸和地址。

| 模式 | 显式边界 | 计算 body |
| ---- | -------- | --------- |
| `UP_DOWN` | 按行分片/聚合 | 用户编写每 lane 计算 |
| `LEFT_RIGHT` | 按列分片/聚合 | 用户编写每 lane 计算 |
| `NONE` | 跨核传输，形状不变 | 完整尺寸；用户用 lane ID 安排不同工作 |

以 `[128, 128]` 的 accumulator、bias 和 output 为例：

```python
with pl.at(level=pl.Level.CORE_GROUP):
    acc = pl.matmul(a, b, out_dtype=pl.FP32)
    for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
        half = pl.aiv_shard(acc)
        row = lane * 64
        out[row:row + 64, :] = pl.add(half, bias[row:row + 64, :])
```

mixed InCore 函数中旧的 `optimizations=[pl.split(pl.SplitMode.UP_DOWN)]`
或 `LEFT_RIGHT` AUTO 路径会报迁移诊断。仅删除该标记不等同于显式分片。

**未拆分 mix 的全幅隐式跨核通信保持不变。** 本次删除隐式拆半，不删除所有
自动通信。独立 AIV/AIC 函数继续使用 `SplitVectorKernel` 的独立处理路径。

## 降低过程 (Lowering)

1. 递归查找循环和条件中的显式区域。
2. 对分片模式保留 transpose hazard 校验，利用用户绑定的 lane 修正显式边界有效尺寸。
3. 保留计算 body 和区域包装，递归处理子区域。
4. 克隆重建后的函数并标记 `split_aiv`；由 `ExpandMixedKernel` 消费区域并生成 `tpush/tpop`。
5. 拒绝 mixed 函数级 AUTO 拆半，其余函数透传；适用时保留原函数级 transpose 校验。

普通区域不再注入额外 lane index。ragged 显式 shard 仍可能需要修正每 lane 的
valid extent；这属于边界元数据，不是隐式改写计算或地址。现有传输可表示性约束
仍适用；已删除的 AUTO 路径不再替用户重新平衡 ragged body 的分片。

非 region scope 应已完成 outlining。隐藏在未展开 scope 后的区域，或区域中仍存活
的 scope，会收到明确诊断。

## API 与属性

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::LowerAutoVectorSplit()` | `passes.lower_auto_vector_split()` | Program |

Default pipeline 中紧邻 `ExpandMixedKernel` 之前运行。

| 属性 | 契约 |
| ---- | ---- |
| Required | `SSAForm`、`IncoreTileOps`、`SplitIncoreOrch`、`TileOps2D`、`TileMemoryInferred`、`NormalizedStmtStructure`、`AivSplitValid` |
| Produced | 相同结构属性，以及 `AivSplitLoweredValid` |
| Invalidated | `AivSplitValid` |

源级 verifier 继续检查 region placement 和显式 crossing；降低后的 verifier 接受保留
区域及受支持的扁平显式 IR。`SplitVectorKernel` 为显式函数标记 dispatch，不再次拆半。

## 实现与测试

- `src/ir/transforms/lower_auto_vector_split_pass.cpp`
- `src/ir/transforms/utils/split_axis_utils.cpp`：显式边界和独立 kernel 的共享工具，并非全部属于 AUTO。
- `tests/ut/ir/transforms/test_lower_auto_vector_split.py`
- `tests/ut/ir/transforms/test_expand_mixed_kernel_split_aiv.py`
- `tests/st/codegen/torch/test_torch_codegen_cross_core.py`

参见 [ExpandMixedKernel](26-expand_mixed_kernel.md)、
[SplitVectorKernel](28-split_vector_kernel.md) 和
[作用域与放置](../../user/language/04-scopes.md)。
