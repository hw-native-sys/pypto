# Loop-Carried Compiler Dependency 压缩

更新：2026-07-01

## 问题

`AutoDeriveTaskDependencies` 可以通过 `compiler_manual_dep_edges` 给
AUTO scope 中的调用附加编译器推导出的依赖边。这样 codegen 可以生成显式的
`set_dependencies(...)` 调用，并在参数被改写成 `NoDep` 或 `OutputExisting`
时减少 TensorMap 查询。

qwen14 prefill 暴露了另一个成本：一个 loop-carried tensor 版本可能解析成很大的
TaskId 数组，而同一个数组又会在后面的 loop 里被每个 consumer 反复展开。

例如，`down_proj_residual` 依赖：

- 来自前面 loop 的 80 槽 `resid1_tile` TaskId 数组；
- 当前 iteration 的 scalar `down_acc` TaskId。

修复前，40 次 consumer loop 每轮都会发出 `80 + 1` 个依赖：

```text
40 * 81 = 3240 dependency entries
```

这些依赖在语义上是真实的，但表达方式过于重复。

## 实现

第一版实现在 orchestration codegen 中：

- `src/codegen/orchestration/orchestration_codegen.cpp`
- `tests/ut/codegen/test_phase_fence_dep_compression.py`

在生成静态 `ForStmt` 之前，codegen 会扫描 loop body 里的
`compiler_manual_dep_edges`。如果某条 edge 能解析成 loop 外产生的 TaskId 数组，
并且在 loop 内反复展开它的成本高于增加一个 summary barrier，codegen 会在 loop
前生成一个只负责依赖的 dummy task。

然后它会把这条 edge 的 TaskId 绑定改写成 barrier 的 scalar TaskId，让普通的
`EmitManualDeps(...)` 在 loop 内发出更小的依赖数组。

安全条件保持保守：

- loop trip count 必须是静态值，并且大于 1；
- edge 必须解析成 TaskId 数组；
- edge 不能定义在 loop body 内；
- 估算收益必须为正：

```text
producer_count * consumer_count - (producer_count + consumer_count) > 0
```

## 生成形状

修复前：

```cpp
PTO2TaskId params_t21_deps[81];
// 80 resid1_tile deps + 1 current down_acc dep
params_t21.set_dependencies(params_t21_deps, params_t21_deps_count);
```

修复后：

```cpp
L0TaskArgs params_phase_fence_barrier_0;
PTO2TaskId params_phase_fence_barrier_0_deps[80];
params_phase_fence_barrier_0.set_dependencies(...);
TaskOutputTensors phase_fence_barrier_0_outs =
    rt_submit_dummy_task(params_phase_fence_barrier_0);
PTO2TaskId phase_fence_barrier_0_tid =
    phase_fence_barrier_0_outs.task_id();

PTO2TaskId params_t21_deps[2];
// barrier TaskId + current down_acc TaskId
params_t21.set_dependencies(params_t21_deps, params_t21_deps_count);
```

## 验证

远端 focused test：

```text
tests/ut/codegen/test_phase_fence_dep_compression.py
21 passed
```

qwen14 prefill golden 在 no-auto-deps 和 optimized 两种模式下均通过。

生成的 orchestration 从：

```text
task dep arrays before: [1, 1, 20, 4, 1, 80, 1, 1, 2, 136, 81]
```

变成：

```text
task dep arrays after:  [1, 1, 20, 4, 1, 80, 1, 1, 2, 1, 2]
barrier dep arrays:    [80, 136]
```

关键 loop 的估算变为：

```text
before: 40 * (136 + 81) = 8680
after:  136 + 80 + 40 * (1 + 2) = 336
```

五次 qwen14 prefill STRACE 聚合：

```text
no-auto-deps orch_us: 2360.365
optimized orch_us:   2335.500
delta:               -24.865 us
speedup:             +1.053%
```
