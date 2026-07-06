# LowerHostTensorCollectives Pass

## 概览

`LowerHostTensorCollectives` 将 host orchestrator 中的
`pld.tensor.allreduce`、`pld.tensor.barrier`、`pld.tensor.broadcast`、
`pld.tensor.reduce_scatter` 和 `pld.tensor.allgather` 调用改写为编译器内部的
builtin chip dispatch。它在 [`MaterializeCommDomainScopes`](38-materialize_comm_domain_scopes.md) 之后运行，
因此 window 绑定的 data tensor 和用户显式传入或编译器合成的 signal tensor 已经带有
`WindowBuffer` 反向引用，并属于推断出的通信域。

该 pass 不修改非 host 函数。InCore allreduce 仍然走
[`LowerCompositeOps`](12-lower_composite_ops.md)。

## Pipeline 位置

```text
... -> SynthesizeAllReduceSignals -> MaterializeCommDomainScopes -> LowerHostTensorCollectives -> Simplify（最终） -> MaterializeRuntimeScopes
```

最终的 `Simplify` 位于本 pass 之后，用于继续折叠生成的循环边界或常量表达式，
随后再插入 runtime scopes。

## 行为

对于 host orchestrator 中的调用：

```python
data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
signal = pld.tensor.barrier(signal)
data = pld.tensor.broadcast(data, signal, root=0)
data = pld.tensor.reduce_scatter(data, signal, op=pld.ReduceOp.Sum)
data = pld.tensor.allgather(data, signal)
```

本 pass 会为每个参与设备生成对应的 `builtin.tensor.*` 调用（如
`builtin.tensor.allreduce`、`builtin.tensor.barrier`、
`builtin.tensor.broadcast`、`builtin.tensor.reduce_scatter`、
`builtin.tensor.allgather`）。若外层
comm-domain scope 带有显式 device 列表，则生成 `SeqStmts`；否则生成顺序
`for r in pld.system.world_size()` 循环。

每个生成的 builtin call 携带来源 `pld.tensor.*` 调用中 collective 特定的
参数和 kwarg 属性。窗口绑定的 INOUT tensor 原样传递；标量 kwarg 值
（`op`、`root`、`dtype`）转发给 builtin。

若用户代码使用赋值形式，pass 会在生成的 builtin 调用之后追加
`<result> = <original expr>`，保留 public API 的 rebind 语义。

## 检查

该 pass 要求两个参数都是已经 materialize 的 `DistributedTensorType` view，并且位于同一个
`CommDomainScopeStmt` 中。当前 host builtin 路径仅支持 FP32 data 上的
`ReduceOp.Sum`，并要求 signal 是 INT32 tensor，形状可以是 rank-1 `[world_size]`
或 rank-2 `[world_size, 1]`；当参与设备数静态可知时，signal 的静态容量必须足够。

## Pass 属性

| 字段 | 取值 |
| ---- | ---- |
| `required` | `{IRProperty::CommDomainScopesMaterialized}` |
| `produced` | `{IRProperty::CommDomainScopesMaterialized}` |
| `invalidated` | `{}` |

## 参考

- 实现：[src/ir/transforms/lower_host_tensor_collectives_pass.cpp](../../../../src/ir/transforms/lower_host_tensor_collectives_pass.cpp)
- 头文件：[include/pypto/ir/transforms/passes.h](../../../../include/pypto/ir/transforms/passes.h)
- 测试：[tests/ut/ir/transforms/test_lower_host_tensor_collectives.py](../../../../tests/ut/ir/transforms/test_lower_host_tensor_collectives.py)
