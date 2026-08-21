# MaterializeDistTensorCtx Pass

本 pass 为每个 `DistributedTensorType` 函数参数显式物化一个对应的
`CommCtxType` 参数和实参。

## 概览

DistributedTensor 的通信上下文需要沿完整调用链传递：host orchestration
提供每个 rank 的 `device_ctx`，L2 orchestration 通过 task args 转发，L1 PTO
codegen 再用它降低 `pld.system.rank`、`pld.system.nranks`、`notify`、`wait`、
`put` 和 remote memory ops。

旧路径在多个 codegen 站点分别合成 ctx，容易漏站点或顺序漂移。本 pass 把这条
数据流放进 IR：

1. 对每个带 `DistributedTensorType` 参数的函数，按分布式 tensor 参数顺序在签名
   末尾追加 `CommCtxType` 参数，方向为 `ParamDirection::In`。
2. 对每个调用点追加对应 ctx 实参。若 distributed tensor 是调用者自己的参数或其
   SSA alias，则转发调用者已有的 ctx 参数。返回位置按被调函数实际返回的参数
   建立对应关系，因此混合类型或重排后的返回 tuple 不会退化为尾部位置猜测。
   `Submit` 同理：它的结果位置就是被调函数的返回位置（末尾的
   `Scalar[TASK_ID]` 没有对应 ctx）。把新 SSA 变量绑定到「已经存在的
   DistributedTensor」的 builtin op 会转发该值的 ctx，包括两类：输出侧写回
   （由 op 自己通过 `set_output_reuses_input(idx)` 声明，例如 `tile.store`
   -> `args[2]`、`tensor.assemble` -> `args[0]`），以及零拷贝 buffer-aliasing
   view（`tensor.view`、`tile.slice`、`tensor.reshape` 等），后者的结果类型
   直接从 `args[0]` 传播 `DistributedTensorType::window_buffer_`。
   `ForStmt` / `WhileStmt` 携带的 tensor alias 也会被追踪。仅对 host
   orchestration，在无法解析 lineage 时才会在调用前插入
   `pld.system.get_comm_ctx(dist)` 绑定并传递该结果。chip orchestration 和
   device 函数必须解析出显式 ctx；如果无法解析则直接报错，不会生成
   device-side 查询。
3. 在 chip orchestration 和 device 函数中，本 pass 会把所有
   `pld.system.get_comm_ctx(dist)` 替换为已经解析出的显式 `CommCtxType` SSA
   值。host orchestration 保留该 op，由 host codegen 从每个 rank 的运行时
   window context 中解析。
4. 若调用点已有 `arg_directions`，同步追加 `ArgDirection::Scalar`，让后续
   codegen 把 ctx 当作普通 scalar task payload 处理。

本 pass 不会向 `IfStmt` 的 return variable 或分支 yield 添加 `CommCtxType`。
DistributedTensor 的 if lowering 继续保持 then/else 必须引用同一 backing/context
的现有约束；动态 context merge（issue #2027）不在本次修改范围内。

循环 carry 遵循同一条「单一 context」规则。carry 先用 init value 播种（这样
`data = self.comm(data)` 这类自携带才能解析），遍历完循环体后再用 yield 回来的
值校验该播种值：如果循环体内把 carry 重新绑定到**另一个** DistributedTensor，
会直接报错，而不是继续沿用 init value 的 context。若 yield 的 lineage 完全无法
追踪，则保留播种值。

本 pass 产出 `IRProperty::DistTensorCtxMaterialized`：host orchestration 之外
不再存在 `pld.system.get_comm_ctx`。pass 对它改写过的每个函数都保证了这一点，
property verifier 再独立校验一遍——这也覆盖了「程序里没有任何函数带
`DistributedTensorType` 参数、pass 原样返回」的情况。该属性已登记在
`GetVerifiedProperties()` 中，因此默认 verification level 下流水线就会校验它，
而不是只在测试手工安装 `VerificationInstrument` 时才生效。

本 pass 要求 `IRProperty::ReturnParamsExplicit`：返回位置映射来自
`return_lineage::ExplicitReturnedParamIndices`，它是对 `ReturnStmt` 的指针恒等
读取，只有在 `NormalizeReturnOrder` 规范化之后才有意义。

该 pass 位于 `LowerHostTensorCollectives` 之后、最终 `Simplify` 之前。此时
host window buffer 已由 `MaterializeCommDomainScopes` 填好，host tensor
collective 也已降低完成，同时后续仍有一次 simplify 可清理转发别名。

## 与 dynamic dim 的区别

dynamic dim 可以在 wrapper 边界从 tensor descriptor 本地恢复；CommCtx 不行。
CommCtx 是真实的跨层数据流，必须从 host 到 orchestration、task payload、kernel
signature 一路传递。把它放进 IR 可以避免多个 codegen 站点各自维护隐式规则。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::MaterializeDistTensorCtx()` | `passes.materialize_dist_tensor_ctx()` | Program-level |

```python
from pypto.pypto_core import passes

program = passes.materialize_dist_tensor_ctx()(program)
```

## 示例

Before:

```python
def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
    return self.kernel(data)

def host_orch(self):
    data = pld.window(buf, [256], dtype=pl.FP32)
    self.chip_orch(data, device=r)
```

After:

```python
def chip_orch(self, data, data_ctx: pld.CommCtxType):
    return self.kernel(data, data_ctx)

def host_orch(self):
    data = pld.window(buf, [256], dtype=pl.FP32)
    data_ctx = pld.system.get_comm_ctx(data)
    self.chip_orch(data, data_ctx, device=r)
```

kernel body 不需要修改。device 函数 body 中已有的
`pld.system.get_comm_ctx(data)` 会在本 pass 中改写为显式 ctx 参数；host
orchestration 中的用法仍保留为运行时查询。
