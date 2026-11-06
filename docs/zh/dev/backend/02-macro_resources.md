# ISA 宏同步资源

ISA 宏内部可能占用带编号的 event/flag。后端提供一张稀疏信号占用表，与
[硬件资源目录](01-sync_resources.md) 中的可用资源池和合法 ID 配合使用。
只有存在信号占用的宏才有条目；仅执行屏障（barrier）的宏无需登记。

本组件提供声明与查询，不绑定实际 IR 调用、不分配 ID，也不插入同步。
现有自动插同步流程继续使用 PTOAS。

## 查询声明

`Backend.find_isa_macro_resources(name)` 按限定的 ISA 宏名查询所属后端，
`Backend.get_isa_macro_names()` 返回排序后的已登记名称。
命名空间属于宏身份：`pto::TGATHER` 与 `pto::comm::TGATHER` 是不同的宏。
每条返回的声明都包含 `name`、`target` 和非空 `claims`。

```python
from pypto.pypto_core import backend, ir, sync

target = backend.get_backend_instance(backend.BackendType.Ascend950)
transfer = target.find_isa_macro_resources("pto::comm::TPUT")
assert transfer is not None
assert {(claim.pool.src, claim.pool.dst) for claim in transfer.claims} == {
    (ir.PipeType.MTE2, ir.PipeType.MTE3),
    (ir.PipeType.MTE3, ir.PipeType.MTE2),
}
assert all(claim.source == sync.IsaIdSource.Fixed for claim in transfer.claims)
assert all(claim.ids == [0, 1] for claim in transfer.claims)
```

对应的 C++ 查询返回拥有自身数据的可选值：

```cpp
const auto* target = pypto::backend::GetBackendInstance(pypto::backend::BackendType::Ascend950);
const auto transfer = target->FindIsaMacroResources("pto::comm::TPUT");
const auto names = target->GetIsaMacroNames();
```

未登记名称返回 `None` / `std::nullopt`，表示没有为该名称登记信号占用，
不能据此认定未知外部实现不占用信号。查询结果是独立快照；Python 对象只读，
嵌套键和集合返回副本。查询使用显式后端，不改变全局后端选择。

## 占用字段

| 字段 | 含义 |
| ---- | ---- |
| `pool` | 资源种类、核作用域、有向流水线对和资源实例 |
| `source` | `Fixed`、`Parameter` 或 `CompilerAllocated` |
| `ids` | 固定 ID，或编译器 token 的候选 ID |
| `parameter`、`offsets` | 参数角色，以及绑定基址 ID 上需要叠加的偏移 |
| `lifetime` | `Invocation` 或 `Owner` |
| `owner` | 共享所有者角色，使用方需要绑定其实际身份 |
| `modes` | 适用的 `Manual`、`Auto` 或 `FlagTest` 构建模式 |

`Fixed` 记录 ISA 实现已经选定的 ID。例如 TPUT/TGET 保守地包含 MTE2/MTE3
两个方向的 ID 0 和 1，因为只有一个显式暂存 tile 时，内部仍可能采用 ping-pong 分片。

`Parameter` 记录 `pipe.FlagID` 或 `session.syncId` 这样的值。
绑定基址加上每个偏移，都必须符合资源池的合法 ID 范围。`Owner` 保持跨调用共享的绑定：
pipe 的 ready/free flags 从初始化到最终排空都必须与该 pipe 关联，
一次 push/pop 返回不能释放这些 ID。`Invocation` 覆盖单次调用内部的信号使用。

`CompilerAllocated` 表示 ISA helper 向其编译器请求一个 token。
候选列表不表示同时占用所有 ID；这份声明也不会选择或控制返回的 token。
接入时需要与实际生成该 token 的实现协调分配。候选集合遵循硬件合法范围，
可以包含 PyPTO 分配策略排除的 ID。

`modes` 区分普通手动代码、`__PTO_AUTO__` 和 `PTO_FLAG_TEST`；
两者同时定义时，`__PTO_AUTO__` 优先。例如 `PtoSetWaitFlag` 在手动模式使用固定 ID 0，
在 flag-test 模式请求编译器 token，在自动模式不生成 flag。

## 后端接入边界

`src/backend/common/isa_macro_resources*.cpp` 中的原生规则汇总各实现变体的占用，
相关 C++ 注释说明硬件适用条件和共享 helper 行为。后续后端使用方需要根据实际降低
（lowering）结果及操作数确定适用的实现变体，再绑定其占用、参数值与所有者生命周期。
内部/外部 session、单向/双向 pipe 等互斥变体不能同时绑定。在适用变体内部，
无法证明可以缩小范围时，应保留保守并集。声明本身不保证操作数合法性、内存效果或整个 kernel 已完成。

首次查询会构建目标表，并在发布前验证信号占用非空、资源池有效、ID 和偏移合法、
模式与所有者生命周期有效。
错误的原生规则抛出 `pypto::InternalError`。即使编译器策略禁止新分配使用某些 ID，
ISA 已固定使用的这些 ID 仍可属于硬件合法范围。
