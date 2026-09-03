# LowerL2TensorCollectives Pass

## 概述

`LowerL2TensorCollectives` 是托管集合通信（managed collective）的 CHIP/L2 通路。
它把写在 **CHIP orchestration 函数体**里的 `pld.tensor.*` 集合通信改写成对编译器
合成的 AIV kernel 的调用，使该集合通信成为调用方自身 pipeline 里的一个普通 task。

目前支持 `pld.tensor.all_to_all_v`，且要求 `core_num=1`。

HOST 通路（[`LowerHostTensorCollectives`](44-lower_host_tensor_collectives.md)）
在上一层解决同一问题，做法不同：它把集合通信按设备扇出成**每个设备一次**
`builtin.tensor.*` chip dispatch。每次这样的 dispatch 都是一个完整的 L2
orchestration task，而它唯一的工作就是提交一个 AIV kernel，因此
`compute -> collective -> consume` 序列每张卡要花三次 L3 -> L2 往返。在本通路上
只需一次。

```text
HOST 通路                                 CHIP 通路（本 pass）
─────────                                ─────────────────────
L3 -> L2  stage task                     L3 -> L2  chip_pipeline
L3 -> L2  builtin collective dispatch              ├── stage         (AIV task)
          └── rt_submit_aiv_task                   ├── collective    (AIV task)
L3 -> L2  consume task                             └── consume       (AIV task)
```

## 在流水线中的位置

```text
... -> FuseCreateAssembleToSlice -> LowerL2TensorCollectives -> DeriveCallDirections -> AutoDeriveTaskDependencies -> ...
```

这个位置是必要条件而非偏好。改写后的调用必须像任何其他 kernel 调用一样经过
[`DeriveCallDirections`](39-derive_call_directions.md) 和
[`AutoDeriveTaskDependencies`](40-auto_derive_task_dependencies.md)：正是这两个
pass 把合成 kernel 的参数方向转换成排序 `compute -> collective -> consume` 的
TensorMap 依赖边。放在它们之后改写，会让该 collective task 失去顺序约束。

它同样运行在 [`MaterializeDistTensorCtx`](45-materialize_dist_tensor_ctx.md)
之前 —— 后者会补上 kernel 需要的 `CommCtx` 实参（见下文 *ABI*）。

## 行为

对于一个 CHIP orchestration 函数体：

```python
@pl.function(type=pl.FunctionType.Orchestration)
def chip_pipeline(self, inp, out, stage, data, signal, counts, recv):
    stage, counts = self.stage_step(inp, stage, counts)
    data = pld.tensor.all_to_all_v(stage, data, signal, counts, recv, core_num=1)
    return self.consume_step(data, recv, out)
```

该集合通信被改写为：

```python
data = self.__builtin_all_to_all_v__fp32(stage, data, signal, counts, recv)
```

其中 `__builtin_all_to_all_v__fp32` 是新增到 program 中的合成
`FunctionType.AIV` 函数：

| 方面 | 取值 |
| ---- | ---- |
| 参数 | `input, target, signal, send_counts, recv_counts` —— 类型取自调用点 |
| 方向 | `In, InOut, InOut, In, InOut` |
| 函数体 | `return target` —— 单条 `ReturnStmt`，永不参与代码生成 |
| Attrs | `builtin_template_dir`、`builtin_template_vars` |

每个 variant 只合成一个函数，被该 variant 的所有调用点共享。

### 为什么函数体是 `ReturnStmt` 而不是空声明

该 kernel 的实现是手写 builtin 源码，函数体不会被代码生成 —— 对 backend 来说空
声明就够了。之所以写成返回 `target` 参数的真实 `ReturnStmt`，是因为后续仍会读取
该函数的 pass 需要它：`ReturnParamsExplicit` 成立，且 `MaterializeDistTensorCtx`
能把返回的 `DistributedTensor` 解析回它写入的那个参数。返回 `target` 也与 public
op 的 window-as-result 语义一致，使调用点保持为一次普通 rebind。

## Kernel 源码：一份实现，两条通路

合成函数并不像
[external kernel](../language/04-external-kernels.md) 那样直接给出 `.cpp` 路径，
而是指向 builtin **模板包** —— 与 `builtin.tensor.all_to_all_v` 通过
`set_template_dir` 声明的是同一个 —— 外加渲染所需的替换项：

```text
builtin_template_dir  = ":pypto.runtime.builtins.collectives.all_to_all_v"
builtin_template_vars = "dtype_cpp=float"
```

PTO backend 把 `templates/kernel.cpp.in` 渲染到 chip 子构建的
`kernels/aiv/<name>.cpp`，并写入生成的 `kernel_config.py` —— 与 PyPTO 生成的
kernel 走同一条路径，区别只在于文本来自模板而非 ptoas。

`dtype_cpp` 是两条通路**唯一**的替换项，且取值相同，因此两边渲染出的 kernel
**逐字节相同**。ST 通过 diff 两条通路的渲染产物端到端断言这一点。

## ABI

两条通路以**相同**的实参布局到达 kernel：

| 槽位 | HOST 通路 | CHIP 通路（本 pass） |
| ---- | --------- | -------------------- |
| `args[0..4]` | `input, target, signal, send_counts, recv_counts` | 相同 |
| `args[5]` | `CommContext*` | `CommContext*` |
| `args[6..]` | — | `args[5]` 的未读副本 |

两条通路都不传 rank 数标量。kernel 读 `CommContext::rankNum`，它与 HOST dispatch
过去传的 `domain_size` 是同一个数：`comm_derive_context` **按 comm domain** 建
context，其 `rankNum` 就是该域的 rank 数。去掉这个标量的代价是入口处多一次 GM
读取，换来的是一份共享源码。这在 CHIP 通路上也是唯一可行方案 —— 它根本无法计算
rank 数，`pld.system.nranks` 只有 InCore 代码生成，没有 orchestration 代码生成。

`args[6..]` 的副本是 `MaterializeDistTensorCtx` **为每个 `DistributedTensor`
参数各追加一个 `CommCtx` 参数**的产物；它们全部解析到同一个 `device_ctx`，因为
一次调用的五个操作数必属同一个 comm domain。`input` 允许是普通 `pl.Tensor`，这会
让该尾部少一项 —— 第一个 ctx 仍落在 `args[5]`，因为尾部总是跟在全部五个 tensor
参数之后。

## 约束与诊断

| 条件 | 诊断 |
| ---- | ---- |
| `core_num != 1` | 拒绝 —— 多 AIV 启动尚未实现 |
| `dtype != FP32` | 拒绝 —— 与 HOST 通路声明的单 dtype 支持一致 |
| 集合通信残留在非 HOST 的 orchestration 函数体中 | 被本 pass 自身的后置条件检查拒绝 |

残留检查覆盖除 HOST orchestrator 之外的每个 orchestration 函数体（HOST 交由自己的
通路处理，在五个 pass 之后）。InCore 函数体不检查：它们归 composite 通路
（[`LowerCompositeOps`](12-lower_composite_ops.md)）所有，且该 pass 早在 26 个
pass 之前就已运行，在这里重复报告会指向错误的 pass。

## 本 pass 不做的事

- 不创建 `CommDomain`，也不分配任何集合通信 staging buffer。L3 仍负责创建域、交换
  window 地址并绑定 window；L2 只消费本地视图和已经建好的 context。
- 不做按设备扇出。`device=` 派发仍留在 HOST orchestrator，每张卡一次
  `chip_pipeline`。
- 不产生嵌套的 L2 -> L2 dispatch。该集合通信是调用方 pipeline 的一个 AIV task，
  不是另一个 chip callable。

## 当前限制

- **`core_num > 1`**。请求的 block 上限会随 op 传递，但此处只接受 `1`。
  `L -> B` 映射、原子 gang 准入和 per-lane 同步协议属于独立工作项。
- **运行期操作数校验**。buffer 契约中静态可证的违规（带洞的 stride view、
  `input` 与 `target` 别名）由 `pld.tensor.all_to_all_v` 的类型推导拒绝；在提交
  AIV task 前没有运行期复检 —— `B` 固定为 1 时，需要复检的项（signal stride
  `>= B`）本身是平凡成立的。
- **rank 数**。kernel 读取 `CommContext::rankNum`，因此当显式设备子集小于 context
  的 rank 数时，其行为与传入 comm domain `domain_size` 的 HOST 通路不同。
- **「同属一个通信域」是未经检查的前置条件**。kernel 通过单个 `CommContext`
  （`args[5]`）解析所有对端地址，因此绑定到不同域的操作数会寻址到错误的远端窗口。
  HOST 通路用 `FindScopeForBuffers` 强制了等价约束——它能直接看到 window buffer；
  本通路做不到：comm domain 在 `MaterializeCommDomainScopes`（pass 43）与
  `MaterializeDistTensorCtx`（pass 45）之前没有 IR 表示，而这两者都在本 pass
  之后运行；到那时该集合通信的操作数已是外层 pipeline 的参数，把它们追溯回绑定它们
  的 host window 需要目前不存在的跨函数分析。改为比较追加的 `CommCtx` 实参也不成立：
  每个 `DistributedTensor` 参数各自生成一个，单域调用本就携带多个互不相同的 SSA 值。

## 测试

- `tests/ut/ir/transforms/test_lower_l2_tensor_collectives.py` —— 改写后的形态、
  合成签名与方向、模板 attrs、variant 共享、InCore 透传、`core_num > 1` 拒绝。
- `tests/ut/ir/transforms/test_lower_composite_ops.py` —— composite 通路把
  CHIP orchestration 中的集合通信交给本 pass，并在 InCore 函数体中拒绝
  `core_num != 1`。
- `tests/st/distributed/collectives/test_l2_tensor_all_to_all_v.py` —— P=2/4 的
  硬件正确性；InCore 与 HOST 通路同样会跑的 0 / 1 / 容量 / 超容量 / 负数计数矩阵，
  它把三条通路约束到同一份链路 golden；外加三条结构断言：不产生 builtin chip
  dispatch、builtin kernel 被渲染进 pipeline 自己的子构建、HOST 通路渲染出逐字节
  相同的 kernel 源码。
