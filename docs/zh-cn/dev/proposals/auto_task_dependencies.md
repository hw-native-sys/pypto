# 自动 Task 依赖推导

## 状态

这是 `auto-deps` 分支的设计提案。本文描述的 pass 还没有接入默认
pipeline。

## 目标

在 pass 层推导 task 之间的依赖边，让用户大多数情况下不需要手写
`pl.submit(..., deps=[...])`。第一阶段目标是保证 `with pl.manual_scope():`
的正确性；普通 auto scope 仍保留 runtime TensorMap 作为 fallback，等分析成熟
后再考虑接管。

下沉路径复用现有机制：

```text
Call.attrs["manual_dep_edges"] -> orchestration codegen -> Arg::set_dependencies(...)
```

P0 不需要修改 runtime。

## 当前代码触点

| 区域 | 当前行为 | auto-deps 含义 |
| ---- | -------- | -------------- |
| `manual_dep_edges` (`include/pypto/ir/expr.h`) | codegen 消费的 TaskId 依赖 Var 列表 | 复用存储格式，但要区分用户边和编译器边 |
| Orchestration codegen | 为每个 call 发出一个栈上 `PTO2TaskId[]` 和 `set_dependencies` | compiler deps 写入后继续复用该 lowering |
| `DeriveCallDirections` | 只解析 `arg_directions`，manual deps 目前由 parser 写入 | auto-deps 应在 direction 稳定后运行，而不是塞进 direction 推导 |
| `BufferRootCollector` | 为 direction/codegen 需求把 `tensor.slice` 当作新 root | 不能直接当作 storage alias analysis 复用 |
| `OptimizeOrchTensors` | 显式化静态 out window，并证明部分 loop disjointness | 后续可复用 affine/window 推理来删掉保守依赖 |

## 分析模型

每个 task 生成一个访问摘要：

```text
TaskAccess {
  task_id_var,
  accesses: [
    { storage_root, region, direction }
  ]
}
```

- `storage_root`：依赖分析使用的 allocation identity。`tensor.slice` 和
  view 类操作继承 parent storage root。
- `region`：相对 storage root 的 offset、shape、stride/layout 和符号 loop
  表达式。
- `direction`：读、写或读写。

依赖判断：

```text
NoAlias                 -> 不建边
MustDisjoint            -> 不建边
MayOverlap/MustOverlap  -> 根据 RAW/WAR/WAW hazard 建边
```

Hazard 规则：

| 当前访问 | 历史访问 | 建边? |
| -------- | -------- | ----- |
| read | write | 是 |
| write | read | 是 |
| write | write | 是 |
| read | read | 否 |

## 关键设计约束

1. 除非后续明确引入 pass-owned mode，否则 `manual_dep_edges` 和
   `arg_directions` 是追加关系。P0 不尝试抵消 runtime direction 语义。
2. 用户手写 deps 保持最高优先级。编译器自动边只做补充，最终 merge 前应能
   追踪 provenance。
3. 不修改现有 `BufferRootCollector`。auto-deps 需要新的
   `StorageRootAnalysis`，因为 storage 语义和 direction/codegen root 语义不同。
4. dynamic fan-in 只有在能编码成现有 `Scalar[TASK_ID]` 或定长
   `Array[N, TASK_ID]` carry 时才支持。`manual_scope` 下无法表达的情况应给出
   明确诊断，而不是泛化的 internal check。

## P0：Manual-Scope 正确性

范围：

- 只分析 `with pl.manual_scope():`。
- 只在能静态表达依赖集合时自动生成 deps。
- 保留用户手写 `deps=[...]`，去重后追加 compiler deps。
- 普通 auto scope 不变。

实现清单：

1. 新增 `AutoDeriveTaskDependencies` program pass，放在
   `DeriveCallDirections` 之后、final `Simplify` 之前。
2. 新增内部 `StorageRootAnalysis`，保守跟踪 assignment、tuple get、yield、
   loop、`tensor.slice`、`tensor.assemble` 和 callsite formal-to-actual
   substitution。
3. 为可能作为依赖 producer 的 call 生成或保留 producer TaskId 变量。
4. 维护每个 scope 内的历史 read/write access 集合，并为 RAW/WAR/WAW hazard
   发出 compiler dependency edges。
5. 增加测试：overlap、暂时保守处理的 disjoint window、用户 deps 加 compiler
   deps、unsupported dynamic fan-in 诊断。

## P1：稳定 Storage Lineage

P1 扩展分析能力，但不改变 runtime contract：

- 完整处理嵌套 loop、if/yield、tuple return 和 callsite formal-to-actual
  substitution 的 storage lineage。
- 在存在 MemRef 时接入 `MemRef::MayAlias`：相同 `base_` 且 byte range overlap
  视作 may alias；符号 offset 保守处理。
- 覆盖 Group/Spmd effective directions，避免 access summary 错读原始
  `param_directions_`。

## P2：删除保守依赖

P2 用来恢复并行度：

- 复用或抽出 `OptimizeOrchTensors` 中 affine out-window disjointness 推理。
- 将更多 `MayOverlap` 提升为 `MustDisjoint`。
- 避免把静态 `pl.parallel` 中写 disjoint window 的分支串行化。

## P3：静态完整性与 Runtime Fallback

P3 用来补齐单个 traced storage root 无法表达的正确性缺口，并为静态依赖推导
无法可靠编码依赖集合的情况定义安全 fallback。

本阶段实现目标：为 `IfStmt`、loop 和 while 的 return var 增加有限 root-set
lineage；当某个必需依赖无法编码成固定 TaskId deps 时，把整个 scope fallback 到
runtime tracking。

优先级：

1. 为分支 yield 不同 storage root 的 `IfStmt` 结果增加 root-set lineage。例如：

   ```python
   if cond:
       selected = pl.yield_(a)
   else:
       selected = pl.yield_(b)

   out, _ = pl.submit(self.consume, selected)
   ```

   `selected` 可能 alias `a` 或 `b`；依赖发射必须同时考虑两个 root 的历史
   producer。如果所有 producer TaskId 都能静态取得，就为完整的有限 root set
   发出 deps。

2. 增加 loop 和 while 的 body-yield lineage。循环 return var 不能只从
   `initValue` 推导；循环体末尾的 `pl.yield_()` 可能改变 carried storage root：

   ```python
   selected = a
   for i, selected in pl.range(0, 4, init_values=[selected]):
       selected = pl.yield_(produced_b)

   out, _ = pl.submit(self.consume, selected)
   ```

   如果能证明循环至少执行一次，并且 yield root 可追踪，return var 可以继承
   yield root。如果循环可能执行零次，或 init/yield root 不同，return location
   应拓宽为有限 root set，例如 `{a, produced_b}`。在没有 root-set 支持前，这类
   场景必须保守处理，不能任意选择其中一个 root。

3. 对无法识别、或静态编码容易出错的情况，增加整个 `manual_scope` 回退到原始
   runtime TensorMap/OverlapMap 的能力。典型场景包括：动态 fan-in 且 producer
   TaskId 数量不受静态上界约束、动态 gather/scatter 类 alias、root-set 爆炸、缺少
   producer TaskId、或混合控制流需要的 deps 不是固定列表。fallback 应优先作用于
   整个 `manual_scope`，而不是单个 call，避免静态 deps 与 runtime TensorMap 状态在
   分段边界不一致。

## 开放问题

- compiler-derived edges 是否应使用新 attr，例如
  `compiler_manual_dep_edges`，只在 codegen 合并；还是复用
  `manual_dep_edges` 并在别处保存 provenance？
- 对非 `pl.submit` 的普通 orchestration call，producer TaskId 变量应该在哪一层生成？
- `manual_scope` 下哪些情况应该成为用户可见错误，哪些情况应该退化成保守依赖？
