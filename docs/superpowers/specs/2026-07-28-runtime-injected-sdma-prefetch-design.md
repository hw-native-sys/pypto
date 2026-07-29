# Runtime-Injected SDMA Prefetch Design

## Goal

Adapt PR #2089's `pl.prefetch.*` frontend to the SDMA workspace contract that
the pinned simpler runtime already implements. Users express prefetch intent in
the DSL, while the runtime owns, provisions, injects, and releases the SDMA
workspace.

## Current Problem

PR #2089 models the SDMA workspace as a user-owned tensor:

```python
ctx = pl.prefetch.make_context(workspace)
```

Its system test asks `pypto.runtime.ChipWorker` for a device address, wraps that
address in `DeviceTensor`, and passes it as a program argument. The wrapper then
calls a `simpler.Worker.sdma_prefetch_workspace_addr()` method that does not
exist.

The pinned runtime commit `9922afdb` already contains simpler commit
`717d703f` (runtime PR #1406). That implementation deliberately exposes no host
getter. A `simpler.Worker` constructed with `enable_sdma=True` provisions the
workspace during initialization and injects its address into each kernel's
`GlobalContext`. The kernel reads it with
`get_dma_workspace(args, DMA_WORKSPACE_SDMA)`.

Updating the runtime submodule cannot repair the mismatch. PyPTO must consume
the injected resource instead of threading it through the public program
signature.

## Public API

The prefetch context constructor becomes parameterless:

```python
ctx = pl.prefetch.make_context()
event = pl.prefetch.async_prefetch(source, ctx)
session = pl.prefetch.session(ctx)
pl.prefetch.wait(event, session)
```

`prefetch.make_context` takes no positional or keyword arguments and returns
`PrefetchAsyncContextType`. Passing a workspace is rejected by normal DSL/IR
arity validation. No public operation exposes a raw SDMA workspace address.

`pypto.runtime.ChipWorker` gains a keyword-only `enable_sdma: bool = False`
option for callers that explicitly manage a reusable worker. Ordinary workers
retain the current default and do not allocate SDMA streams.

## Kernel Code Generation

An InCore function containing `prefetch.make_context` requires one hidden SDMA
workspace pointer. This pointer is not an IR function parameter and does not
appear in orchestration or runtime tensor signatures.

PTOCodegen detects the prefetch operation and appends a synthetic
`!pto.ptr<i8>` argument to the generated PTO function after all user tensor and
scalar arguments and before the existing synthetic SPMD identity arguments.
The `prefetch.make_context` emitter consumes this well-known synthetic SSA and
emits:

```text
%ctx = pto.make_prefetch_async_context(%sdma_workspace : !pto.ptr<i8>)
    -> !pto.prefetch_async_context
```

The generated kernel wrapper includes `intrinsic.h`, obtains the runtime-owned
pointer once, and forwards it in the same synthetic-argument position:

```cpp
__gm__ int8_t *sdma_workspace =
    reinterpret_cast<__gm__ int8_t *>(get_dma_workspace(args, DMA_WORKSPACE_SDMA));
generated_kernel(user_args..., sdma_workspace, spmd_args...);
```

Runtime initialization is the fail-fast boundary. A worker that requests SDMA
on an unsupported platform fails during `init`; the wrapper does not create a
fallback allocation and never substitutes an arbitrary non-null buffer.

## Runtime Capability Propagation

The backend records `"enable_sdma": True` in generated `RUNTIME_CONFIG` when
any emitted kernel requires the hidden SDMA workspace. Artifacts without
prefetch omit the key or record `False`, preserving current behavior.

The L2 one-shot execution path propagates that value into
`simpler.Worker(..., enable_sdma=True)`. The lower-level
`execute_on_device` API accepts an `enable_sdma` keyword so artifact runners,
replay, dependency-capture, and test-harness paths can preserve the capability.

An explicit PyPTO `ChipWorker(enable_sdma=True)` passes the flag to the simpler
worker constructor. Reuse is capability-aware: an SDMA-enabled worker may run
ordinary programs, but an ordinary worker cannot run an SDMA-requiring program.
The latter case raises a clear error instructing the caller to construct the
worker with `enable_sdma=True`, instead of falling through to a second worker on
the same device.

Distributed assembly reads the same `RUNTIME_CONFIG` from every chip artifact.
If any prepared program requires SDMA, its shared L3 simpler worker is created
with `enable_sdma=True`; runtime PR #1406 then provisions the workspace in each
chip child. Runtime names must continue to agree across all prepared programs.

## Platform Behavior

Compilation remains target-independent. Actual execution follows the runtime
contract: the current simpler implementation supports SDMA provisioning on
onboard a2a3 and fails fast for simulator, a5, or runtimes without the SDMA
provider. PyPTO does not silently downgrade a requested prefetch to an ordinary
copy and does not report a skipped ST as coverage of the SDMA path.

The device system test is therefore restricted to onboard a2a3. Its successful
completion, bit-exact output, and runtime SDMA provisioning log together prove
that the real path ran.

## Documentation Changes

English and Chinese operator documentation will describe the parameterless
context API, runtime ownership, `ChipWorker(enable_sdma=True)` for explicit
worker reuse, and the current onboard a2a3 execution requirement. References to
`sdma_prefetch_workspace_addr`, `DeviceTensor` workspace wrapping, simulator
no-op behavior, and user-managed workspace sizes will be removed.

## Testing Strategy

Implementation follows red-green-refactor cycles:

1. DSL and IR tests require zero-argument `make_context` and reject the old
   workspace form.
2. PTO codegen tests require a hidden pointer argument, wrapper-side
   `get_dma_workspace`, stable synthetic argument ordering, and no workspace in
   user signatures.
3. Backend manifest tests require `RUNTIME_CONFIG["enable_sdma"]` only for
   programs containing prefetch.
4. Runtime unit tests require L2 one-shot propagation, explicit-worker
   propagation, capability-aware reuse, replay/harness propagation, and L3
   aggregation.
5. Existing non-prefetch runtime tests verify that `enable_sdma` remains off by
   default.
6. The focused operator/codegen/runtime unit suites run after rebuilding the
   worktree.
7. The a2a3 system test runs through `task-submit` after the required
   architecture precheck. It takes only input and output tensors and must pass
   without a host workspace getter.

## Non-Goals

- No runtime submodule update.
- No public API for reading runtime-owned DMA workspace addresses.
- No fallback allocation or simulated SDMA implementation.
- No unrelated refactoring of worker lifecycle or PTO code generation.
