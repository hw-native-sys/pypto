# BackendHandler: principled backend dispatch

> Tracking: [issue #948](https://github.com/hw-native-sys/pypto/issues/948)

## Why

Earlier revisions of PyPTO branched on `backend::BackendType` directly inside
passes and codegen:

```cpp
if (backend::GetBackendType() != backend::BackendType::Ascend910B) { ... }
```

Every new backend therefore required hunting for these scattered conditionals
and adding another arm to each. `BackendHandler` replaces every such branch
with a single virtual call so that adding a new backend is a self-contained
change.

## What

`BackendHandler` (`include/pypto/backend/common/backend_handler.h`) is an
abstract interface that names every behavioural difference between backends.
Each `Backend` subclass owns a singleton `BackendHandler` subclass and exposes
it through the new pure-virtual `Backend::GetHandler()`.

```text
                       ┌────────────────────────────┐
        Pass / Codegen │  PassContext::Current()    │
                       │     ->GetBackendHandler()  │
                       └─────────────┬──────────────┘
                                     │
                                     ▼
                       ┌────────────────────────────┐
                       │  BackendConfig::GetBackend │
                       │     ->GetHandler()         │
                       └─────────────┬──────────────┘
                                     │
                                     ▼
                ┌──────────────────────────────────────┐
                │  Backend910B / Backend950 / ...      │
                │     -> Ascend910BHandler::Instance() │
                │     -> Ascend950Handler::Instance()  │
                └──────────────────────────────────────┘
```

The `PassContext` accessor satisfies the `pass-context-config` rule: passes
look up backend behaviour via the active `PassContext`, not by reaching out
to global state.

## Interface

| Method | Purpose | Ascend910B | Ascend950 |
| ------ | ------- | ---------- | --------- |
| `GetPtoTargetArch()` | `module attributes {pto.target_arch = "..."}` | `"a2a3"` | `"a5"` |
| `GetLaunchSpecCoreCountMethod()` | runtime API name on `launch_spec` | `"set_block_num"` | `"set_core_num"` |
| `GetDefaultSimPlatform()` | default simulator platform | `"a2a3sim"` | `"a5sim"` |
| `GetExtraPtoasFlags()` | extra ptoas flags | `[]` | `["--pto-arch", "a5"]` |
| `RequiresGMPipeBuffer()` | inject GM-backed pipe slot in `ExpandMixedKernel` | `true` | `false` |
| `RequiresSplitLoadTpopWorkaround()` | split-load tpop hazard fix in `LegalizePtoBufferReuse` | `true` | `false` |
| `RequiresVtoCFractalAdapt()` | AIV-side V-to-C fractal adapter `tile.move` | `false` | `true` |
| `RequiresRuntimeSubblockBridge()` | split AIV wrappers source subblock id from runtime | `true` | `false` |
| `RequiresNoSplitDualAivDispatch()` | `no_split` mixed kernels still dispatch on both AIV lanes | `true` | `false` |
| `BuildCrossCoreTransferView(dest, view)` | layout at cross-core transfer boundary | NZ for Mat/Left/Right; preserve for Vec | NZ for Mat/Left/Right; preserve for Vec (a5 hardware also requires fractal at the boundary) |

## Adding a new backend

1. Create `Backend<Arch>` in `src/backend/<arch>/backend_<arch>.cpp` (subclass
   `Backend`). Add the source to `CMakeLists.txt`.
2. Implement `Backend<Arch>Handler` in
   `src/backend/<arch>/backend_<arch>_handler.cpp` (subclass `BackendHandler`).
3. Override `Backend<Arch>::GetHandler()` to return your handler singleton.
4. Append the new backend to `BackendType` and to the factory switch in
   `src/backend/common/backend.cpp` (`GetBackendInstance` and
   `BackendTypeToString`). These are the only places that fan out by enum.

No pass or codegen file needs to change. Validate with
`tests/ut/backend/test_backend_handler.py` plus the regression tests in
`tests/ut/ir/transforms/` and `tests/ut/codegen/`.

## Python access

The handler is also exposed through the Python bindings:

```python
from pypto.pypto_core import backend as _backend_core

# When global config has been set:
handler = _backend_core.get_handler()

# When the caller already knows the desired backend:
handler = _backend_core.get_backend_instance(BackendType.Ascend950).get_handler()

handler.get_pto_target_arch()             # "a2a3" or "a5"
handler.requires_runtime_subblock_bridge()  # bool
handler.get_extra_ptoas_flags()           # list[str]
```

The runtime layer (`pypto.runtime.runner`, `pypto.ir.compiled_program`,
`pypto.backend.pto_backend`) consumes these accessors instead of branching on
`BackendType`, mirroring the C++ refactor.
