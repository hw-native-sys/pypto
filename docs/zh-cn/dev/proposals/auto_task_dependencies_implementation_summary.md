# 自动任务依赖实现修改说明

## 涉及背景

`manual_scope` 过去依赖用户为每个需要排序的任务手写 `deps=[...]`。这种方式在
完全写对时很精确，但一旦 orchestration 代码里出现 tensor alias、slice 或控制流，
就很容易漏掉 producer-consumer 依赖。

本次修改把 `manual_scope` 从“完全手写依赖”推进为“编译器辅助依赖”：用户手写
deps 仍然是权威来源；编译器在能静态表达依赖时自动补充固定 TaskId 边；对于无法
安全表达的场景，回退到 runtime TensorMap/OverlapMap 跟踪。

## 总体设计

### 依赖来源分层

用户依赖和编译器依赖分开存储：

```text
manual_dep_edges              # 用户手写 deps
compiler_manual_dep_edges     # AutoDeriveTaskDependencies 推导的 deps
```

这样 IR dump 可以保留依赖来源。codegen 在下沉到
`Arg::set_dependencies(...)` 前统一合并并去重。

### Storage Access 抽象

每个 task argument 被归纳为一个访问描述：

```text
{ storage_root, region, direction }
```

- `storage_root` 表示依赖分析使用的底层存储身份。`tensor.slice` 这类 view-like
  value 继承父 tensor 的 root。
- `region` 表示相对 root 的访问区域。常量矩形 slice 会精确记录；symbolic 或
  unknown region 保守处理。
- `direction` 来自已经解析完成的 `arg_directions`，因此 pass 放在
  `DeriveCallDirections` 之后运行。

这里没有复用 `BufferRootCollector`。原因是 dependency root 需要表达 alias 语义，
而 buffer root 主要服务 direction/codegen，两者语义不同。

### Hazard 建边规则

pass 按源码顺序扫描每个 `RuntimeScopeStmt(manual=true)`，只在当前 manual scope
内维护 prior access，并将当前 task access 与历史访问比较。

| 当前访问 | 历史访问 | 是否建边 |
| -------- | -------- | -------- |
| read | write | 是 |
| write | read | 是 |
| write | write | 是 |
| read | read | 否 |

如果用户已经在 `manual_dep_edges` 里写了对应依赖，编译器不会再把同一条边重复写入
`compiler_manual_dep_edges`。

### Region 与 Alias 精度

静态 slice region 用来避免不必要的串行化。如果两个常量 slice window 能证明
disjoint，则不建边；如果 region overlap、unknown，或 offset/shape 含 symbolic
表达式，则按 may-overlap 保守处理。

对于带 MemRef 的 tensor type，pass 还会调用 `MemRef::MayAlias`。因此即使两个
tensor Var 不同，只要底层存储范围可能 alias，也会参与依赖判断。

### 控制流 Lineage

storage lineage 会穿过常见 IR 结构：

- assignment 和直接 alias；
- `pl.submit` 返回 tuple 的 tuple get；
- 通过 `Out` / `InOut` 参数建立的 function output 替换；
- `tensor.slice` 和 `tensor.assemble`；
- `IfStmt` 分支末尾的 `pl.yield_()`；
- loop / while 的 iter args 和 body 末尾 `pl.yield_()`。

控制流结果可能对应多个 root。本次实现不再把这类结果压成单一 root，而是保留有限
root-set。比如 if 两个分支分别 yield 不同 tensor，或者 loop init 与 body yield
来自不同 root，后续依赖发射会展开到所有可能 root。

### 回退机制

静态依赖发射的前提是：依赖两端都能表达成有界 root 集合和固定 TaskId 列表。当前
实现会在检测到以下情况时回退：

- 必须建边的 hazard 依赖了历史 producer，但该 producer 没有静态绑定、可编码的
  TaskId；
- loop 内 producer 形成 dynamic fan-in：单个 scalar TaskId binding 不能代表所有
  运行时 producer instance；但如果 consumer 已经位于 loop 之后，并且显式依赖了
  loop-carried `TaskId`（代表最后一个 producer），则仍可保留 manual mode；
- dynamic gather/scatter 类访问：实际访问的 root 或 region 依赖运行时 index，
  无法归纳为有界静态 access；
- 控制流 join 后需要的依赖集合不是固定列表，例如分支/循环混合动态 producer set；
- root-set 增长超过 pass 允许的静态 alternatives 上限；
- dependency-relevant tensor argument 无法通过当前 lineage analysis 解析 storage
  location。

并不是所有保守场景都需要回退。比如 symbolic slice 如果 root 已知、producer TaskId
也能静态取得，可以直接发一条保守依赖；`IfStmt` 或 loop 的有限 root-set 如果所有
可能 producer 都有可编码 TaskId，也可以继续走静态 deps。

回退选择 scope-wide，而不是单 call fallback。这样可以避免部分 compiler deps 与
runtime TensorMap 状态在 manual/auto 边界处混用，降低状态不一致风险。

## 代码修改

### IR Attr 与 Codegen

- 在 `include/pypto/ir/expr.h` 新增 `kAttrCompilerManualDepEdges`。
- 保留 `kAttrManualDepEdges` 作为用户手写依赖列表。
- orchestration codegen 同时读取两类 attr，合并、去重后分配固定
  `PTO2TaskId[]` 依赖数组，并对 invalid TaskId 做 guard，最后调用
  `set_dependencies(...)`。
- Python printer 支持打印 compiler-derived deps，便于 IR dump 中观察编译器补边。

### Pass 注册与管线接入

- 新增 `AutoDeriveTaskDependencies` program pass，主体实现位于
  `src/ir/transforms/auto_derive_task_dependencies_pass.cpp`。
- 在默认 pass manager 中将该 pass 放在 `DeriveCallDirections` 之后。
- 补齐 C++ pass declaration、Python binding、Python type stub。
- 更新 pass index 和单独 pass 文档，说明 pass 的 required/produced property 与
  pipeline 位置。

### StorageRootAnalysis

- 新增 orchestration body 内部的 storage-lineage 分析。
- 将 tensor 参数初始化为 full-root region。
- 通过 assignment、tuple get、call output、`tensor.slice`、`tensor.assemble`
  传播 storage location。
- 将常量 slice 记录为 root-relative box region。
- 对 unknown 或 symbolic slice region 拓宽为保守 overlap。
- 为 root alternatives 记录 MemRef 信息，用于跨 Var 的 alias 判断。
- 对 `IfStmt`、`ForStmt`、`WhileStmt` 的 return var 合并有限 root alternatives。

### 依赖生成 Mutator

- 新增只在 `RuntimeScopeStmt(manual=true)` 内生效的 mutator。
- 每个 manual scope 维护独立 prior read/write access 集合。
- 对每个非 builtin call，基于已解析 `arg_directions` 生成访问摘要。
- 当 storage root may-alias 且 region may-overlap，并且满足 RAW/WAR/WAW hazard
  时，写入 `compiler_manual_dep_edges`。
- 对 read-read 和静态 disjoint region 不建边。
- 保留用户手写 deps，并避免 compiler deps 重复。比较重复边时会 canonicalize
  `TaskId` alias，所以用户依赖 submit-return alias 时，不会再额外生成等价的
  compiler dep。
- 如果 required dependency 无法表达成有界 roots 加固定 TaskId deps，则将整个
  scope fallback 到 auto mode。

### TaskId 收集与保留

- 从 `pl.submit` 返回 tuple 中收集 producer TaskId Var。
- 跟踪简单 TaskId alias，包括 scalar assignment 与 loop-carried TaskId yield，
  让依赖比较基于 producer identity，而不是 callsite 处使用的具体变量写法。
- 修复相关 scalar DCE 路径，避免 submit TaskId 被过早删除后无法作为依赖 producer。
- 依赖表达仍然对齐现有 scalar TaskId 与固定 TaskId array 的 codegen 模型。

### 测试与文档

- 新增/更新单测覆盖 RAW 自动补边、read-read 不补边、auto scope 不受影响、用户边与
  compiler 边分离、静态 disjoint slice、overlap slice、symbolic slice 保守处理、
  IfStmt root-set、loop yield root-set、MemRef may-alias、dynamic gather fallback、
  loop dynamic fan-in fallback、root-set cap fallback、missing TaskId fallback。
- 同步更新 pass、codegen、IR hierarchy、pass manager 和 proposal 文档，并保持
  English / zh-CN 两套文档对齐。

## 执行效果

常见 `manual_scope` producer-consumer 依赖现在可以不由用户手写，编译器会自动生成
显式 `set_dependencies(...)` 边。

对于能静态证明 disjoint 的 slice 访问，pass 不会额外串行化；对于 symbolic
region、MemRef may-alias、控制流来源不确定等场景，则保守建边或扩大 root-set，
优先保证正确性。

当静态依赖推导无法安全编码固定 TaskId 集合时，实现会把整个 scope 回退到 runtime
TensorMap/OverlapMap 跟踪。最终策略是：信息足够时生成显式 deps；信息不足时使用
已有 runtime 依赖机制兜底正确性。
