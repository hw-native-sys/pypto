# Public JIT `lower()` API Design

## Status

Approved direction:

- Keep post-pass IR inspection as a public user capability.
- Add `JITFunction.lower()` as the single-purpose API.
- Delete `compile_for_test()` immediately; do not retain an alias or deprecation period.
- Migrate all repository callers, examples, and documentation.

## Problem

`compile_for_test()` currently combines three different behaviors:

1. It specializes a `@pl.jit` function and runs the default pass pipeline.
2. It attempts a full `_compile()` to populate the `CompiledProgram` cache.
3. It suppresses every exception from that full compile and returns a separately
   lowered `ir.Program`.

The name suggests a complete compile with execution omitted, but success only
guarantees that the separately executed pass pipeline succeeded. The method can
therefore generate artifacts, mutate the compile cache, and hide code-generation
failures while presenting itself as a pass-only test helper.

## Goals

- Give users an explicit way to inspect the post-pass IR of a `@pl.jit` function.
- Make `lower()` and `compile()` use the same specialization, pass-affecting
  configuration, verification settings, and diagnostic settings.
- Guarantee that `lower()` performs no code generation, ptoas invocation, device
  dispatch, artifact creation, or `CompiledProgram` cache mutation.
- Propagate all argument-binding, parsing, and pass failures.
- Remove every version-controlled `compile_for_test` reference.

## Non-goals

- Expose arbitrary intermediate pass stages.
- Add a `stop_before_codegen` mode to `compile()`.
- Cache lowered `ir.Program` objects.
- Make runtime-only `RunConfig` fields affect lowering.
- Add a public `ir.lower()` API for already-parsed `ir.Program` objects.

## Public API

### Before

```python
program = kernel.compile_for_test(a, b, out)
```

This returns an `ir.Program`, but may first call `_compile()`, generate code, and
populate `kernel._cache`. Code-generation failures are swallowed.

### After

```python
program = kernel.lower(a, b, out)
program = kernel.lower(a, b, out, config=RunConfig(
    strategy=OptimizationStrategy.DebugTileOptimization,
))
```

Proposed signature in `python/pypto/jit/decorator.py`:

```python
def lower(self, *args: Any, **kwargs: Any) -> ir.Program:
    """Specialize the JIT function, run configured passes, and return IR."""
```

The `config` keyword is consumed by JIT machinery exactly as it is for
`compile()` and is not bound to the decorated kernel's parameters.

### Behavior matrix

| Entry point | Return value | Passes | Codegen | Cache | Device |
| --- | --- | --- | --- | --- | --- |
| `kernel.lower(...)` | `ir.Program` | Yes | No | No read/write | No |
| `kernel.compile(...)` | `CompiledProgram` | Yes | Yes | Read/write | No |
| `kernel(...)` | Kernel result | Yes on miss | Yes on miss | Read/write | Yes |

## Configuration Contract

`lower()` accepts `config=RunConfig(...)` so users can inspect the IR associated
with the same lowering choices as `compile()`.

Pass-affecting inputs are honored:

- `strategy`
- `analyze_auto_scopes_for_deps`
- `memory_planner`
- `platform`/backend selection where passes consult backend capabilities
- an active `PassContext`, including its memory planner and
  `enable_pypto_l0c_double_buffer` setting

Pass-time validation and diagnostic inputs are also honored even though they do
not change successful IR:

- `RunConfig.diagnostic_phase`
- `RunConfig.disabled_diagnostics`
- an active `PassContext`'s verification level, diagnostic phase, disabled
  diagnostics, and pass instruments

Their precedence exactly matches `ir.compile()`:

- Verification level: active `PassContext`, otherwise the global default;
  `RunConfig` has no verification-level override.
- Diagnostic phase: a non-`None` `RunConfig.diagnostic_phase` conflicts with an
  active `PassContext`; otherwise use the non-`None` `RunConfig` value, then the
  active context value, then the global default.
- Disabled diagnostics: a non-`None` `RunConfig.disabled_diagnostics` value,
  then the active context value, then the default disabled-check set. This does
  not have the diagnostic-phase conflict rule.

Runtime-only fields such as `device_id`, tolerances, and DFX controls are ignored,
as they already are by `compile()`. Artifact-only fields such as
`save_kernels_dir`, `dump_passes`, and compile profiling do not cause `lower()` to
write output: the no-artifact contract takes precedence.

If an explicit `RunConfig.memory_planner` conflicts with an active
`PassContext`, `lower()` raises the same error as `compile()` instead of silently
choosing one.

PyPTO itself creates no artifact directories during `lower()`. Explicit
user-provided `PassContext` instruments are still invoked for parity with
`compile()` and may perform their own documented side effects; those
user-controlled effects are outside the no-artifact guarantee.

## Internal Design

### 1. Share specialization without sharing the compiled cache

Refactor the binding/config extraction currently embedded in
`JITFunction._resolve_compiled()` into a focused helper used by both paths:

```python
specialization, run_config = self._resolve_specialization(
    args,
    kwargs,
    allow_signature_mode=True,
)
```

`compile()` continues from that result into cache lookup and `_compile()`.
`lower()` continues into `_compile_to_program()` and pass execution. This keeps:

- positional and keyword argument binding identical;
- tensor shape/dtype and scalar specialization identical;
- dynamic-dimension handling identical;
- fully annotated, no-sample-tensor signature mode available to both APIs.

### 2. Share pass-context construction with `ir.compile()`

Extract the pass-only portion of `python/pypto/ir/compile.py` into an internal
helper. Conceptually:

```python
def _run_pass_pipeline(
    program: ir.Program,
    *,
    strategy: OptimizationStrategy,
    platform: str | None,
    memory_planner: MemoryPlanner | None,
    verification_level: VerificationLevel | None,
    diagnostic_phase: DiagnosticPhase | None,
    disabled_diagnostics: DiagnosticCheckSet | None,
    analyze_auto_scopes_for_deps: bool,
    emit_artifacts: bool,
) -> ir.Program:
    ...
```

`ir.compile()` calls this helper with its existing dump/report behavior, then
runs codegen. `JITFunction.lower()` calls it with `emit_artifacts=False` and
returns its result. In that mode the helper does not create an output directory,
install `ReportInstrument`, dump passes, or start file-backed compile profiling.
It does preserve explicit outer `PassContext` instruments. Centralizing
context/backend setup prevents `lower()` from drifting away from the actual
compile pipeline.

### 3. Delete the old behavior

Delete `JITFunction.compile_for_test()` in full, including:

```python
try:
    self._cache[key] = self._compile(...)
except Exception:
    pass
```

No compatibility alias remains. Calling `compile_for_test` after this change
raises normal `AttributeError`.

## Errors and Side Effects

`lower()` propagates:

- signature and argument-binding errors;
- source specialization and parser errors, with the existing JIT source-name
  rewriting;
- pass verification and transformation errors;
- invalid/conflicting lowering configuration errors.

After either success or failure:

- `JITFunction._cache` is unchanged;
- no PyPTO-owned build directory, report, or pass dump is created;
- no code-generation failure can be hidden because codegen is never entered.

## Migration

Most callers change mechanically:

```python
# Before
post_pass = kernel.compile_for_test(a, b, out)

# After
post_pass = kernel.lower(a, b, out)
```

Cache tests are not mechanically renamed. Tests that used
`compile_for_test()` specifically to populate `_cache` must call `compile()` or
mock `_compile()` explicitly, because cache mutation is outside `lower()`'s
contract.

## Affected Files

Core implementation:

- `python/pypto/jit/decorator.py` — add `lower`, share specialization binding,
  delete `compile_for_test`.
- `python/pypto/ir/compile.py` — extract reusable pass-only execution so
  `lower` and `compile` share pass context and backend selection.
- `python/pypto/runtime/runner.py` — document which `RunConfig` fields affect
  `lower`.

User documentation, kept synchronized in English and Chinese:

- `docs/en/user/01-installation.md`
- `docs/en/user/02-quickstart.md`
- `docs/en/user/03-programming-model.md`
- `docs/zh/user/01-installation.md`
- `docs/zh/user/02-quickstart.md`
- `docs/zh/user/03-programming-model.md`

Examples:

- `examples/kernels/08_assemble.py`
- `examples/kernels/09_dyn_valid_shape.py`
- `examples/models/qwen3_jit/qwen3_decode.py`
- `examples/utils/cross_function_calls.py`

Unit tests:

- `tests/ut/codegen/test_orchestration_returned_param_map.py`
- `tests/ut/ir/transforms/test_auto_tile_matmul_l0.py`
- `tests/ut/jit/test_allow_early_resolve.py`
- `tests/ut/jit/test_decorator.py`
- `tests/ut/jit/test_dump_tag_outline.py`
- `tests/ut/jit/test_jit_compile_extraction.py`
- `tests/ut/jit/test_qwen3_decode.py`
- `tests/ut/jit/test_roundtrip.py`
- `tests/ut/jit/test_split_k.py`
- `tests/ut/jit/test_spmd.py`

System tests:

- `tests/st/codegen/dsl/test_add_mul_orch_codegen.py`
- `tests/st/codegen/dsl/test_dyn_valid_shape_loop.py`
- `tests/st/codegen/dsl/test_dynamic_valid_shape_if_else.py`
- `tests/st/codegen/dsl/test_flatten_dynamic_tile_3d.py`
- `tests/st/codegen/torch/test_torch_codegen_cross_core.py`

Nested consumer documentation:

- `runtime/examples/a2a3/tensormap_and_ringbuffer/qwen3_14b_decode/README.md`
  in the `runtime` submodule.

## Test Strategy

Add focused JIT tests that verify:

1. `lower()` returns a post-pass `ir.Program`.
2. `lower()` does not invoke `_compile()` or codegen.
3. `lower()` leaves `_cache` unchanged on success and failure.
4. Positional, keyword, dynamic-dimension, scalar, and signature-only
   specialization match `compile()`.
5. `RunConfig.strategy`, memory planner, dependency analysis, backend, and an
   active `PassContext` select the expected pipeline.
6. `RunConfig` and active-`PassContext` verification/diagnostic settings have
   the same precedence and conflict behavior as `compile()`.
7. With `save_kernels_dir`, `dump_passes=True`, and compile profiling set,
   successful and failing `lower()` calls create no build directory, report, or
   pass dump; an explicit outer instrument is still invoked.
8. Binding, parse, pass, and configuration errors are propagated.
9. `compile()` still reuses `CompiledProgram` cache entries.
10. A focused parity test compares `lower()` with the shared internal pass
    helper applied to the same pre-pass program and configuration. It does not
    compare against `CompiledProgram.program`, whose public contract remains the
    original pre-pass IR.
11. `JITFunction` no longer exposes `compile_for_test`.

Migrate existing structural tests to `lower()` and existing cache tests to
`compile()`/explicit mocks. Run the focused JIT and pass tests, then the full unit
suite and repository lint/type checks. Finish with a tracked-file search that
returns no `compile_for_test` references.

## Compatibility

This is an intentional breaking API removal. The migration is a direct rename
only for callers that used the method as documented to obtain post-pass IR.
Callers that depended on its undocumented cache warming must switch to
`compile()`.
