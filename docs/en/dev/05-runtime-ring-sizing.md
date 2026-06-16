# Per-Task Ring Sizing

PyPTO exposes Simpler's per-task ring sizing as three optional overrides on
[`RunConfig`](../../../python/pypto/runtime/runner.py). They let you size the
runtime's per-task ring resources for a single `run()` invocation without
touching the compiled artifact or any global state.

The runtime keeps its task-launch resources in *ring buffers*. Each override
maps 1:1 to a field on Simpler's `CallConfig.runtime_env` and is sized **per L2
task submission** — i.e. per `run()` call (or per chip in an L3 dispatch), so
different submissions of the same kernel can use different ring sizes.

## Field matrix

| `RunConfig` field | `CallConfig.runtime_env` member | Controls | Constraint |
| ----------------- | ------------------------------- | -------- | ---------- |
| `ring_task_window: int \| None` | `ring_task_window` | Number of in-flight task slots in the task ring | power of 2, `>= 4` |
| `ring_heap: int \| None` | `ring_heap` | Bytes of the per-ring task-output heap | power of 2, `>= 1024` |
| `ring_dep_pool: int \| None` | `ring_dep_pool` | Dependency-edge pool capacity | `[4, INT32_MAX]` |

`None` (the default) leaves the field **unset** (`0` on `CallConfig`). PyPTO
writes the value into `CallConfig.runtime_env` only when it is not `None`, so an
unset field defers entirely to the runtime.

## Precedence

For each value the runtime resolves the effective size as:

```text
per-task CallConfig.runtime_env value   (RunConfig override — highest priority)
  └─ falls back to → PTO2_RING_* env var (process-wide)
       └─ falls back to → compile-time default (lowest priority)
```

So a `RunConfig` override wins over the `PTO2_RING_TASK_WINDOW` /
`PTO2_RING_HEAP` / `PTO2_RING_DEP_POOL` environment variables, which in turn win
over the runtime's built-in defaults.

## Validation

`RunConfig` validates the overrides at construction (in `__post_init__`),
mirroring the runtime's `RuntimeEnv::validate()`. This surfaces a clear
`ValueError` at the call site rather than a deep failure inside the runtime's
own `CallConfig::validate()` at dispatch time:

```python
RunConfig(platform="a2a3", ring_heap=1000)
# ValueError: ring_heap must be a power of 2 >= 1024 (bytes per ring), got 1000
```

Non-integers (including `bool`) are rejected with the same `ValueError` rather
than raising an opaque `TypeError` from the power-of-two check.

## Usage

```python
from pypto.runtime import run, RunConfig

compiled = run(
    MyProgram,
    a, b, c,
    config=RunConfig(
        platform="a2a3",
        ring_task_window=128,        # 128 in-flight task slots
        ring_heap=8 * 1024 * 1024,   # 8 MiB output heap per ring
        ring_dep_pool=256,           # 256 dependency-edge entries
    ),
)
```

Set only the fields you want to override; the rest stay unset and fall back per
the precedence above.

## Related

- Simpler's runtime-side implementation: the `tensormap_and_ringbuffer` runtime
  under `runtime/src/<arch>/runtime/tensormap_and_ringbuffer/` and the
  `RuntimeEnv` / `CallConfig` definitions in
  `runtime/src/common/task_interface/call_config.h`.
- Worker-API examples: `runtime/examples/workers/{l2,l3}/per_task_runtime_env/`.
- Orthogonal runtime diagnostics on the same `RunConfig`:
  [03-runtime-dfx.md](03-runtime-dfx.md).
