# MaterializeInplaceAliases Pass

为 PTOAS 内存规划器物化安全的操作边界输入/输出别名。

## 概述

PTOAS 可以复用生命周期不重叠的句柄，但一条指令的输入与输出生命周期
会在该指令处相接。PTO 模块原本没有表达这条指令是否允许 `dst == src`，
因此 PTOAS 的保守规划会保留两个独立句柄。

`MaterializeInplaceAliases` 只编码这一操作级事实。对于 tile 赋值，当某个
直接输入的有效最后一次使用恰好是输出定义点时，该 Pass 可以把输出
MemRef 重定向到该输入。独立生命周期的复用、地址选择、对齐和容量校验
仍完全由 PTOAS 负责。

该 Pass 在默认流水线中位于
[`MaterializeSemanticAliases`](29-materialize_semantic_aliases.md) 之后、
[`MemoryReuse`](31-memory_reuse.md) 之前。只有活动 `PassContext` 选择
`MemoryPlanner.PTOAS` 时才会执行别名物化；在 `MemoryPlanner.PYPTO`
下为空操作。

## 安全规则

候选输入必须：

- 与输出具有相同的内存空间、数据类型、物理分配大小、块布局、scatter
  布局、fractal 大小和 padding 语义；
- 根据循环与 phi 感知的有效生命周期分析，恰好在输出定义点死亡（循环后
  的 return wrapper 通过其物理分配解析）；
- 对应算子在注册表中标记为可原地执行；
- 不受 `forbid_output_alias` 保护；
- 不跨越冲突的软件流水阶段；
- 不违反 Ascend910B split-AIV 的 load/tpop 风险约束；
- 不参与 phi family。

带有 `set_output_reuses_input(k)` 的操作已由
`MaterializeSemanticAliases` / InitMemRef 处理，本 Pass 不再修改。
在 PTOAS 路径上完整的 `MemoryReuse` 不会运行，因此本 Pass 还会消费并
移除 `pipeline_membership`。

实现复用了 `memory_reuse_pass.cpp` 中的生命周期、硬件风险与注册表
no-alias 分析。分析索引只构建一次，每个输出只做一次有序候选决策，
不会执行全局缓冲区装箱。

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::MaterializeInplaceAliases()` | `passes.materialize_inplace_aliases()` | 函数级 |

```python
from pypto.pypto_core import passes

with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS):
    result = passes.materialize_inplace_aliases()(program)
```

## 测试

覆盖位于：

- `tests/ut/ir/transforms/test_memory_reuse.py`
- `tests/ut/codegen/test_memory_planner_switch.py`
