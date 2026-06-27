# MaterializeCommDomainScopes Pass

## 概览

`MaterializeCommDomainScopes` 扫描每个 host-orchestration 函数，组装出分布式 runtime
为分配 / 填充 per-rank 通信窗口所需要的 host 侧元数据。它与
[`InitMemRef`](31-init_memref.md) 在结构上完全同构：追溯一次分配到所有
消费点，构造反向引用对象，再把该对象挂到 IR 类型上，让下游 codegen 能 O(1)
访问。

| 比较项 | `MemRef` 一侧 | `WindowBuffer` 一侧 |
| ------ | ------------- | ------------------- |
| 分配 op | `tile.alloc(memory_space, size_in_bytes)` | `pld.tensor.alloc_window_buffer(size_in_bytes)` |
| Parse 时赋值语句 LHS | `Var(PtrType)` | `Var(PtrType)`（同一个 singleton） |
| 包装 Var 子类 | `MemRef` | `WindowBuffer` |
| 包装类的 SSA-edge 类型 | `MemRefType`（singleton） | `WindowBufferType`（singleton） |
| 构造者 | `InitMemRef` | **`MaterializeCommDomainScopes`**（本 pass） |
| 回填到 | `TensorType.memref_` | `DistributedTensorType.window_buffer_` |
| Program 级注册表 | `Program.functions_`（alloc 语句） | `CommDomainScopeStmt wrappers in each host_orch body` |

## 流水线位置

```text
... -> DeriveCallDirections -> AutoDeriveTaskDependencies -> ExpandManualPhaseFence -> MaterializeCommDomainScopes -> LowerHostTensorCollectives -> Simplify（最终）
```

本 pass 跑在默认 pipeline 的末尾阶段，位于
[`LowerHostTensorCollectives`](40-lower_host_tensor_collectives.md) 和最后一次
`Simplify` 之前。从 `InlineFunctions` 到这里之间的所有 pass 都不会触碰
host_orch 的 alloc / window / dispatch 链：host_orch 本身不会被 tile lower，
L2（chip 级）orchestration 也永远不会被 inline 进 L3，所以本 pass 需要的
alloc / view / dispatch 点在此时仍然可见。放在较晚阶段还能让产生 IR 在描述符
分析之前先被充分规范化，最后的 `Simplify` 也能统一对收集到的 size 表达式做常量折叠。

## 算法

对每个 host-orchestration 函数（`Function::level_ == Level::HOST` 且
`Function::role_ == Role::Orchestrator`，不强求 `func_type_`）：

1. **收集 alloc**。找到所有 RHS 是 `pld.tensor.alloc_window_buffer(size, *, name)`
   的 `AssignStmt`。记录 `(ptr_var, size_expr, name, span, call)`。

2. **收集 view**。所有 RHS 是 `pld.tensor.window(ptr_var, [shape], *, dtype)`、且
   引用已记录 `ptr_var` 的 `AssignStmt`，记录 `view_var → alloc` 绑定。

3. **扫描 dispatch**。带着 `ForStmt` 栈遍历函数体。对每个 `op_` 是
   `GlobalVar` 且解析到 chip-level orchestration 的 Call，读 `attrs["device"]`，
   在当前 for 循环上下文中推导 **device 描述符**：

   | `device=` 形态 | 描述符 |
   | -------------- | ------ |
   | `ConstInt(N)` | `subset = {N}` |
   | `for r in pl.range(pld.system.world_size())` 的 IterArg | `kAll` |
   | `for r in pl.range(ConstInt(N))` 的 IterArg | `subset = {0, …, N − 1}` |
   | 其它 | `pypto::ValueError` |

   每个位置参数若是已记录的 view Var，就向对应 alloc 追加该描述符。

4. **合并描述符**。对每个 alloc 折叠记录到的所有描述符：任何 `kAll` ⇒ `kAll`；
   否则取 subset 并集。

5. **构造 `WindowBuffer`**。对每个 alloc 构造
   `WindowBuffer(base = ptr_var, size = size_expr, load_from_host = false,
   store_to_host = false)`；`Var::name_hint_` 自动继承自 `ptr_var->name_hint_`。
   （host-staging 标志位是 N4+ 的占位字段。）

6. **改写 view 类型**（仅 host_orch）。对每个 view 绑定，mint 一个同
   `name_hint_` 的新 `Var`，类型为
   `DistributedTensorType(shape, dtype, memref, tensor_view, wb)`；用
   `Substitute` 把所有对旧 view Var 的引用替换为新 Var。同一 alloc 被 N 次
   `pld.tensor.window` 物化的多个 view 共享同一 `shared_ptr<const WindowBuffer>`。
   chip-orch / InCore 形参类型不动。

7. **聚类成 group**。按源代码顺序遍历 alloc 列表，匹配描述符已存在的
   comm-domain scope 则追加 slot，否则新开一个。`CommDomainScopeStmt wrappers in each host_orch body` 最终
   填充该列表。

## Sanity 校验

下列情况抛 `pypto::ValueError`（携带 alloc 的 span）：

- 某 alloc 没有任何 `pld.tensor.window` 物化（dead alloc）。
- 某 alloc 有 view 但没有 chip-orch dispatch 消费它。
- dispatch 的 `device=` 既不是 `ConstInt`、也不是已识别的 `pl.range`
  归纳变量。
- 同一 comm-domain scope 内 `name_hint_` 重名（parser 已在程序范围内做了唯一性
  校验，本 pass 再次断言）。

## 输出不变量

pass 运行之后：

- `CommDomainScopeStmt wrappers in each host_orch body` 已填（程序不分配 window buffer 时为空）。
- 每个 `pld.tensor.window` 结果 Var 的类型是 `DistributedTensorType`，
  `window_buffer_` 字段指向对应的 `WindowBuffer`。
- 同一 alloc 的多个 view 共享同一 `shared_ptr<const WindowBuffer>`——指针
  相等是下游 codegen 的关键不变量。
- chip-orchestration 与 InCore 的形参类型 `window_buffer_` 仍是 `nullopt`。
  N7 codegen 在 *host_orch* 的 dispatch 处读取反向引用、再为 chip-orch 显式
  下发对应的 `CommContext` 指针。

## Pass 属性

| 字段 | 取值 |
| ---- | ---- |
| `required` | `{}` |
| `produced` | `{IRProperty::CommDomainScopesMaterialized}` |
| `invalidated` | `{}` |

## 参考

- 实现：[src/ir/transforms/materialize_comm_domain_scopes_pass.cpp](../../../../src/ir/transforms/materialize_comm_domain_scopes_pass.cpp)
- 头文件：[include/pypto/ir/transforms/passes.h](../../../../include/pypto/ir/transforms/passes.h)
- Schema：[include/pypto/ir/program.h](../../../../include/pypto/ir/program.h)
  定义了 `WindowBuffer` 与 comm-domain scope。
- DSL：[`pld.tensor.alloc_window_buffer`](../../../../python/pypto/language/distributed/op/tensor_ops.py)、
  [`pld.tensor.window`](../../../../python/pypto/language/distributed/op/tensor_ops.py)、
  [`pld.system.world_size`](../../../../python/pypto/language/distributed/op/system_ops.py)。
