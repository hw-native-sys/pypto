# BackendHandler: principled backend dispatch

> Tracking: [issue #948](https://github.com/hw-native-sys/pypto/issues/948)

## Why

Earlier revisions of PyPTO branched on `backend::BackendType` directly inside
passes and codegen:

```cpp
if (backend::GetBackendType() != backend::BackendType::Ascend910) { ... }
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

| Method | Purpose | Ascend910 | Ascend950 |
| ------ | ------- | --------- | --------- |
| `GetPtoTargetArch()` | `module attributes {pto.target_arch = "..."}` | `"a2a3"` | `"a5"` |
| `GetSyncResourceCatalog()` | Immutable [synchronization resource facts](01-sync_resources.md) owned by this backend | A2/A3 pools | A5 pools |
| `GetLaunchSpecCoreCountMethod()` | runtime API name for the logical block count on `launch_spec` | `"set_block_num"` | `"set_block_num"` |
| `GetDefaultSimPlatform()` | default simulator platform | `"a2a3sim"` | `"a5sim"` |
| `GetExtraPtoasFlags()` | extra ptoas flags | `[]` | `["--pto-arch", "a5"]` |
| `HonorsHighPrecisionAlgorithm()` | PTO-ISA selects a distinct algorithm from the `precisionType` attribute that `high_precision=True` emits; when false, both PTO emitters (Tile op callbacks and Buffer IR) warn that the request is dropped | `false` | `true` |
| `RequiresGMPipeBuffer()` | inject GM-backed pipe slot in `ExpandMixedKernel` | `true` | `false` |
| `SupportsRuntimeBoundLocalSoftwarePipeline()` | enable local Vec software-pipeline lowering for runtime trip counts | `true` | `false` |
| `SupportsLocalPipeDepth()` | local consumer depth can differ from the cross-core transport ring depth | `true` | `false` |
| `RequiresSplitLoadTpopWorkaround()` | MemoryReuse load + tpop_from_aic in-place hazard guard | `true` | `false` |
| `RequiresLevel3TmpScratch()` | InitMemRef materialize level-3 explicit tmp; codegen static-view bridges | `true` | `false` |
| `RequiresVtoCFractalAdapt()` | AIV-side V-to-C fractal adapter `tile.move` | `false` | `true` |
| `RequiresRuntimeSubblockBridge()` | split AIV wrappers source subblock id from runtime | `true` | `false` |
| `RequiresNoSplitDualAivDispatch()` | `no_split` mixed kernels still dispatch on both AIV lanes | `true` | `false` |
| `SplitsCubeToVectorTransportInHardware()` | a split Cube-to-Vector push hands each AIV lane half of the transported extent, so that extent is the lane partition | `false` | `true` |
| `GetL0cMAlignment(dtype)` | physical M-row alignment shared by L0C tiling, dbC capacity, and allocation | 32 for INT32; otherwise 16 | 16 |
| `BuildCrossCoreTransferView(dest, view)` | layout at cross-core transfer boundary | NZ for Mat/Left/Right; preserve for Vec | NZ for Mat/Left/Right; preserve for Vec (a5 hardware also requires fractal at the boundary) |

`SupportsRuntimeBoundLocalSoftwarePipeline()` defaults to `false`, preserving
the existing loop fallback until a backend enables this lowering. It is
independent of cross-core transport and does not describe hardware event support.

## Registered operator rules

`tile.sel` and `tile.sels` register their scratch planners in
`BackendOpInfo::scratch_plan_func`; TSELS also registers its existing dtype and
MemRef alias checks in `validate_func`. `PTOCodegen` invokes the validator before
Tile PTO emission. Both backends share `pto_ops_select.cpp`, with target-local
rules bound once at registration. The four select-specific handler virtuals
(`GetTselScratchSpec`, `GetTselsScratchSpec`, `SupportsTselDataType`, and
`SupportsTselsDataType`) have been removed.

Passes use `PassContext::GetBackend()` to query operator rules. This accessor
retains the existing global target resolution; it does not introduce per-context
target ownership. Standalone select lowering with no configured backend retains
its historical A5 scratch geometry and ungated dtype behavior.

```python
from pypto import DataType, backend

target = backend.get_backend_instance(backend.BackendType.Ascend910)
plan = target.plan_op_scratch("tile.sels", DataType.FP16, source_cols=128)
# Required: FP16 [1, 128], one physical source row.
```

`Required` carries a read-only `TileScratchSpec` in element units; `NotRequired`
explicitly means no scratch. `Unknown` means necessary input information is
missing, and `Unsupported` rejects a candidate. A registered op without a
planner returns `None`; an unregistered op raises an error. No query reads the
global target, creates IR, or allocates memory.

Synthesis and explicit operands remain distinct: caller-owned scratch is passed
through unchanged, including A5 placeholders with other dtypes. This migration
preserves existing compiler checks; it does not add exact scratch-geometry
validation or change PTOAS checks. Memory allocation and the existing IR
workspace/alias metadata remain responsible for lifetime and reuse decisions.

### Cast scratch rules

`tile.cast` registers its scratch planner with `f_plan_scratch` alongside codegen
in `pto_ops_cast.cpp`. The target is captured during registration; no cast-specific
handler virtual is needed. `InitMemRef` and codegen both call
`Backend::PlanOpScratch("tile.cast", request)`.

Cast queries take `source_tile` (the immutable physical and valid geometry),
`target_dtype`, and the effective `saturating` flag. A separately supplied
`source_dtype` must agree with the tile. Missing inputs or dynamic geometry
needed for sizing return `Unknown`; invalid rank returns `Unsupported`.
Consumers report errors at the call span. A5, native conversion pairs, and
saturating conversions return `NotRequired` without requiring static geometry.

```python
from pypto import ir

source = ir.TileType([4, 128], DataType.FP32, tile_view=ir.TileView(valid_shape=[3, 96]))
plan = target.plan_op_scratch(
    "tile.cast", source_tile=source, target_dtype=DataType.INT16, saturating=False
)
# Required: INT8 [1, 1152], including the physical row pitch for partial validity.
```

PyPTO and DSA-RP materialize the returned spec in `InitMemRef`; PTOAS retains
ownership of implicit scratch. Codegen supplies static-valid views for the
scratch-backed casts. Caller-provided tmp operands are preserved, rather than
checked against the compiler's synthesis recipe.

## Memory capacity source

`BackendHandler::GetCapacityBytes(memory_space)` reads `Backend::GetMemSize()`
on the handler's owning backend. The SoC is the single source for capacities;
handlers do not repeat numeric limits or multiply them by core/die counts.
The query returns one matching memory component's byte capacity as `uint64_t`,
or `0` when the SoC does not model that space. For a space present on multiple
cores, it returns the first matching component's capacity. An unmodeled space
is not evidence that the hardware lacks that memory.

Pass `MemorySpace::Left`, `Right`, `Acc`, `Bias`, or `Mat` for L0A, L0B, L0C,
Bias, or L1, respectively. The same API also queries `Vec` and the scale memories.
Python exposes `handler.get_capacity_bytes(memory_space)`. This unified query
replaces the five separate L0A/L0B/L0C/Bias/Mat capacity methods in both languages.
Explicit backend queries work even when no global target is configured, or when
a different target is active. Alignment rules, minimum tile dimensions, and
cost-model parameters remain separate from capacity.

## Adding a new backend

1. Create `Backend<Arch>` in `src/backend/<arch>/backend_<arch>.cpp` (subclass
   `Backend`). Add the source to `CMakeLists.txt`.
2. Implement `Backend<Arch>Handler` in
   `src/backend/<arch>/backend_<arch>_handler.cpp` (subclass `BackendHandler`).
   Initialize its base with the owning backend reference, which must outlive the
   handler. Built-in handlers use their backend singleton; backend constructors
   must not request the handler while that singleton is being initialized.
3. Override `Backend<Arch>::GetHandler()` to return your handler singleton.
4. Append the new backend to `BackendType` and to the factory switch in
   `src/backend/common/backend.cpp` (`GetBackendInstance` and
   `BackendTypeToString`). Also add it to the backend list used by
   `FindResourceCatalog` in `src/backend/common/sync_resources.cpp`, so target-string
   queries reach the new handler's synchronization catalog.

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
handler.supports_local_pipe_depth()       # True on A2/A3, False on A5
handler.get_extra_ptoas_flags()           # list[str]
```

The runtime layer (`pypto.runtime.runner`, `pypto.ir.compiled_program`,
`pypto.backend.pto_backend`) consumes these accessors instead of branching on
`BackendType`, mirroring the C++ refactor.
