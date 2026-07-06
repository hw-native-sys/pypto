# SynthesizeAllReduceSignals Pass

## 概览

`SynthesizeAllReduceSignals` 将 host 层
`pld.tensor.allreduce(data, op=...)` 归一化为内部显式 signal IR。这样用户层
host DSL 可以省略 signal，而下游仍然只需要处理已有的内部形态：

```python
data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
```

这个 pass 只处理 host orchestrator 函数。InCore allreduce 仍然显式接收
signal，并继续由 [`LowerCompositeOps`](12-lower_composite_ops.md) lower。

## Pipeline 位置

```text
... -> ExpandManualPhaseFence -> SynthesizeAllReduceSignals -> MaterializeCommDomainScopes -> LowerHostTensorCollectives -> Simplify（最终）
```

它运行在 [`MaterializeCommDomainScopes`](38-materialize_comm_domain_scopes.md)
之前，此时 host 侧 `alloc_window_buffer` / `window` / dispatch 链路仍然可见。
随后 comm-domain materialization 会把合成的 signal buffer 当成普通 window
allocation 处理，并放入 allreduce data buffer 所属的通信域。

## 算法

对每个 host-orchestration 函数：

1. 收集当前 program 中已有变量名。
2. 访问直接出现在 `AssignStmt`、`EvalStmt`、`ReturnStmt` 中的
   `pld.tensor.allreduce`。
3. 已经传入显式 signal 的调用保持不变。
4. 对只传入 target tensor 的调用，生成 fresh 的私有 signal buffer 和
   signal view 名字。
5. 在 allreduce 之前插入普通 statement-level binding：

```python
__allreduce_signal_world_size_0 = pld.system.world_size()
__allreduce_signal_buf_0: pl.Ptr = pld.tensor.alloc_window_buffer(__allreduce_signal_world_size_0 * 4)
__allreduce_signal_0 = pld.tensor.window(
    __allreduce_signal_buf_0,
    [__allreduce_signal_world_size_0, 1],
    dtype=pl.INT32,
)
data = pld.tensor.allreduce(data, __allreduce_signal_0, op=pld.ReduceOp.Sum)
```

合成 signal 使用 rank-2 `[world_size, 1]`。这个形态与 InCore allreduce 的
signal 索引模型一致，也让 host lowering 面向同一种 canonical signal 表示。

## Print / Parse Round Trip

合成的 buffer allocation 会打印成普通赋值语句。IR 内部 call 可以携带
`name` kwarg 供 consumer 使用，但 Python printer 会省略这个 kwarg，并依赖赋值
左侧变量名。打印出来的源码再次 parse 时，parser 会像处理用户手写
`pld.tensor.alloc_window_buffer` 一样，从 LHS 恢复 buffer name。

因此 dump / reparse 流程看到的是普通 DSL 语句，重新 parse 后仍然保留同样的
alloc / window / allreduce 链路。

## 检查

以下情况会抛出 `pypto::ValueError`：

- allreduce 位置参数数量不是 `target` 或 `target, signal`；
- allreduce 出现在 `for` 或 `while` 循环中；
- allreduce 作为嵌套表达式出现，而不是直接赋值、表达式语句或 return value。

循环内 allreduce 会被拒绝，因为当前 signal 协议是 single-use；编译器不能在
动态多次调用之间复用同一个 signal buffer。

## Pass 属性

| 字段 | 值 |
| ---- | -- |
| `required` | `{}` |
| `produced` | `{}` |
| `invalidated` | `{}` |

## 参考

- 实现：[src/ir/transforms/synthesize_allreduce_signals_pass.cpp](../../../../src/ir/transforms/synthesize_allreduce_signals_pass.cpp)
- 头文件：[include/pypto/ir/transforms/passes.h](../../../../include/pypto/ir/transforms/passes.h)
- 测试：[tests/ut/ir/transforms/test_materialize_comm_domain_scopes.py](../../../../tests/ut/ir/transforms/test_materialize_comm_domain_scopes.py)
