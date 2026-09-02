# SynthesizeAllReduceSignals Pass

## 概览

`SynthesizeAllReduceSignals` 将 host 层
`pld.tensor.allreduce(data, op=...)` 归一化为内部显式 signal IR。这样用户层
host DSL 可以省略 signal，而下游仍然只需要处理已有的内部形态：

```python
data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
```

这个 pass 只处理 host orchestrator 函数。InCore allreduce 仍然显式接收
signal，并继续由 [`LowerCompositeOps`](13-lower_composite_ops.md) lower。

## Pipeline 位置

```text
... -> ExpandManualPhaseFence -> SynthesizeAllReduceSignals -> MaterializeCommDomainScopes -> LowerHostTensorCollectives -> Simplify（最终）
```

它运行在 [`MaterializeCommDomainScopes`](43-materialize_comm_domain_scopes.md)
之前，此时 host 侧 `alloc_window_buffer` / `window` / dispatch 链路仍然可见。
随后 comm-domain materialization 会把合成的 signal buffer 当成普通 window
allocation 处理，并放入 allreduce data buffer 所属的通信域。

## 算法

对每个 host-orchestration 函数：

1. 收集当前 program 中已有变量名。
2. 预扫描函数体：是否携带 `pld.tensor.allreduce` 调用，其中是否有省略
   signal 参数的调用。
3. 若函数存在隐式 signal（单参数）调用，则按数据 buffer 的血缘（lineage，
   沿 `pld.tensor.window` 回溯到 `pld.tensor.alloc_window_buffer` 的 LHS）将
   调用分组，并为每个分组把一个共享 signal binding（world_size /
   alloc_window_buffer / window）提升到函数体顶部：

    ```python
    __allreduce_signal_world_size_0 = pld.system.world_size()
    __allreduce_signal_buf_0: pl.Ptr = pld.tensor.alloc_window_buffer(__allreduce_signal_world_size_0 * core_num * pl.INT32.get_byte())
    __allreduce_signal_0 = pld.tensor.window(
        __allreduce_signal_buf_0,
        [__allreduce_signal_world_size_0, core_num],
        dtype=pl.INT32,
    )
    ```

4. 把每个隐式 signal 的 `pld.tensor.allreduce` 调用 —— 包括 `for` / `while`
   循环内的调用 —— 改写为使用该共享 signal：

    ```python
    data = pld.tensor.allreduce(data, __allreduce_signal_0, op=pld.ReduceOp.Sum)
    ```

5. 已经传入显式 signal 的调用保持不变；return 位置的调用仍然提升为
   赋值语句，以便 host lowering 派发。

合成 signal 使用 rank-2 `[world_size, core_num]`，其中 `core_num` 是该血缘
分组内所有隐式 signal 调用请求的最大 lane 数（默认 `core_num=1` 保留原有的
单 lane 表示），buffer 字节数也使用相同的 lane 数。每个数据 buffer 血缘
（设备覆盖范围）分组共享一个 binding，并被该 buffer 上所有隐式 signal 调用
复用 —— 这是正确的，因为 host builtin kernel 会在每次调用后自清理屏障 cell，
因此共享 signal 在连续调用与循环迭代之间都可以安全复用。不同的数据 buffer
会得到不同的 signal，因此一个函数内针对不同设备子集的隐式 allreduce 不会被
合并进同一个 comm-domain scope。

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
- allreduce 作为嵌套表达式出现，而不是直接赋值、表达式语句或 return value。

`for` / `while` 循环内的隐式 signal 调用会被接受：该调用所属数据 buffer 血缘
的共享 signal 会在每次迭代中被复用，这是正确的，因为 host builtin kernel
（`builtin.tensor.allreduce` / `builtin.tensor.allreduce_ring`，由
`LowerHostTensorCollectives` lower）通过信用屏障尾声在每次调用后自清理屏障
cell。由 [`LowerCompositeOps`](13-lower_composite_ops.md#屏障-信号协议) lower
的 InCore 组合算子同样具备循环安全性 —— 该 pass 会发出自清理尾声。

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
