# Loop-Carried Compiler Dependency Compression

Updated: 2026-07-01

## Problem

`AutoDeriveTaskDependencies` can attach compiler-derived dependency edges to
AUTO-scope calls through `compiler_manual_dep_edges`. This lets codegen emit
explicit `set_dependencies(...)` calls and can reduce TensorMap lookups when an
argument is rewritten to `NoDep` or `OutputExisting`.

The qwen14 prefill case exposed a second cost: a loop-carried tensor version can
resolve to a large TaskId array, and the same array may be expanded for every
consumer in a later loop.

For example, `down_proj_residual` depends on:

- an 80-slot `resid1_tile` TaskId array from an earlier loop;
- the current iteration's scalar `down_acc` TaskId.

Before this fix, the 40-trip consumer loop emitted `80 + 1` deps per iteration:

```text
40 * 81 = 3240 dependency entries
```

Those dependencies are semantically real, but the representation is too
repetitive.

## Implementation

The first implementation is in orchestration codegen:

- `src/codegen/orchestration/orchestration_codegen.cpp`
- `tests/ut/codegen/test_phase_fence_dep_compression.py`

Before emitting a static `ForStmt`, codegen scans the loop body for
`compiler_manual_dep_edges`. If an edge resolves to a TaskId array produced
outside the loop, and repeatedly expanding it inside the loop would cost more
than adding a summary barrier, codegen emits a dependency-only dummy task before
the loop.

Then it rewrites the TaskId binding for that edge to the barrier's scalar
TaskId, so normal `EmitManualDeps(...)` emits a small dependency array inside
the loop.

The safety conditions are intentionally conservative:

- the loop trip count must be static and greater than 1;
- the edge must resolve to a TaskId array;
- the edge must not be defined inside the loop body;
- estimated saving must be positive:

```text
producer_count * consumer_count - (producer_count + consumer_count) > 0
```

## Generated Shape

Before:

```cpp
PTO2TaskId params_t21_deps[81];
// 80 resid1_tile deps + 1 current down_acc dep
params_t21.set_dependencies(params_t21_deps, params_t21_deps_count);
```

After:

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

## Validation

Remote focused test:

```text
tests/ut/codegen/test_phase_fence_dep_compression.py
21 passed
```

qwen14 prefill golden passed for both no-auto-deps and optimized modes.

Generated orchestration changed from:

```text
task dep arrays before: [1, 1, 20, 4, 1, 80, 1, 1, 2, 136, 81]
```

to:

```text
task dep arrays after:  [1, 1, 20, 4, 1, 80, 1, 1, 2, 1, 2]
barrier dep arrays:    [80, 136]
```

The key loop estimate becomes:

```text
before: 40 * (136 + 81) = 8680
after:  136 + 80 + 40 * (1 + 2) = 336
```

Five-repeat qwen14 prefill STRACE aggregation:

```text
no-auto-deps orch_us: 2360.365
optimized orch_us:   2335.500
delta:               -24.865 us
speedup:             +1.053%
```
