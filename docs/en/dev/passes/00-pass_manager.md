# Pass, PassContext, PassPipeline, and PassManager

Framework for organizing and executing IR transformation passes on Programs with property tracking, instrumentation, and strategy-based optimization pipelines.

## Overview

| Component | Description |
| --------- | ----------- |
| **Pass (C++)** | Standalone class for Program → Program transformations with property declarations |
| **IRProperty / IRPropertySet** | Enum + bitset for verifiable IR properties (SSAForm, HasMemRefs, etc.) |
| **PassInstrument / PassContext** | Instrument callbacks (before/after pass) with thread-local context stack |
| **PassPipeline (C++)** | Ordered sequence of passes executed in order |
| **PassManager (Python)** | High-level manager using PassPipeline, with strategy-based optimization |

### Key Features

- **Property Tracking**: Passes declare required, produced, and invalidated properties
- **Instrumentation**: PassContext holds PassInstruments that run before/after each pass
- **Runtime Verification**: VerificationInstrument checks properties against actual IR
- **Strategy-based Pipelines**: Pre-configured optimization levels (`Default`)
- **Immutable Transformations**: Return new IR nodes, don't modify in place

## IRProperty System

### IRProperty Enum

**Header**: `include/pypto/ir/transforms/ir_property.h`

| Property | Description |
| -------- | ----------- |
| `SSAForm` | IR is in SSA form |
| `TypeChecked` | IR has passed type checking |
| `NoNestedCalls` | No nested call expressions |
| `NormalizedStmtStructure` | Statement structure normalized |
| `NoRedundantBlocks` | No single-child or nested SeqStmts |
| `SplitIncoreOrch` | InCore scopes outlined into separate functions |
| `HasMemRefs` | MemRef objects initialized on variables |
| `IncoreTileOps` | InCore functions use tile ops (tile types, load/store) |
| `AllocatedMemoryAddr` | All MemRefs have valid addresses within buffer limits |
| `MixedKernelExpanded` | Mixed InCore functions split into AIC + AIV + Group |
| `ClusterOutlined` | Cluster scopes outlined into Group functions |
| `TileOps2D` | All tile ops in InCore functions use ≤2D tiles |
| `TileMemoryInferred` | `TileType::memory_space_` populated in InCore functions |
| `BreakContinueValid` | Break/continue only in sequential/while loops |
| `UseAfterDef` | All variable uses are dominated by a definition |
| `HierarchyOutlined` | Hierarchy scopes outlined into level/role functions |
| `StructuredCtrlFlow` | No BreakStmt/ContinueStmt — only structured control flow |
| `VectorKernelSplit` | AIV functions with a split mode have tpop shapes and store offsets adjusted |
| `OutParamNotShadowed` | Out/InOut params are not reassigned with tensor-creating ops |
| `NoNestedInCore` | No nested InCore scopes (ScopeStmt inside ScopeStmt) |
| `InOutUseValid` | No reads of InOut/Out-passed variables after the call (RFC #1026) |
| `PipelineLoopValid` | Bidirectional invariant: `ForStmt.kind_ == Pipeline` ⇔ has a `pipeline_stages` attr |
| `PipelineResolved` | No `ForKind::Pipeline` survives; produced by CanonicalizeIOOrder |
| `CallDirectionsResolved` | Every non-builtin Call has explicit `attrs['arg_directions']` |
| `TileTypeCoherence` | Every TileType has canonical tile_view (implicit views stored as nullopt) |
| `InlineFunctionsEliminated` | No `FunctionType::Inline` functions or Calls to them remain |
| `OrchestrationReferencesResolved` | Every non-builtin Call inside a `FunctionType::Orchestration` function targets a Function in the surrounding Program |
| `TensorViewCanonical` | TensorView canonicality verified (weak: empty stride ok; strict: requires materialization, RFC #1300 §2.2) |
| `ArrayNotEscaped` | ArrayType never appears as a function parameter or return type |
| `CommDomainScopesMaterialized` | Host_orch bodies wrapped in CommDomainScopeStmts, and `pld.tensor.window` result types carry `DistributedTensorType::window_buffer_` back-references |
| `DistTensorCtxMaterialized` | No `pld.system.get_comm_ctx` survives outside host orchestration; every chip-orchestration / device communication context is an explicit CommCtxType SSA value traceable to a parameter |
| `RuntimeScopesMaterialized` | Orchestration functions carry explicit RuntimeScopeStmt nodes, so codegen emits no implicit `SIMPLER_SCOPE()` wrappers |
| `AssignTypeSymmetry` | Every AssignStmt has `structural_equal(var->GetType(), value->GetType())` (memref excluded as an allocation detail) |
| `ManualDepsOnSubmitOnly` | No plain cross-function Call carries `attrs["manual_dep_edges"]` — manual edges live in `Submit::deps_` |
| `ReturnParamsExplicit` | InCore/Group/Spmd tensor returns reference function params by pointer identity (#1702) |
| `UnrollResolved` | No `ForKind::Unroll` survives; produced by UnrollLoops |
| `AivSplitValid` | SplitAivScopeStmt regions are structurally valid: no cube compute or split-axis reduce inside a region, boundary ops only inside one |
| `HardSyncallOccupancyValid` | Every hard (FFTS) `system.syncall` is launched at full occupancy — a partial or over launch deadlocks on device (507018) |
| `IterArgCarryClassified` | Every Orchestration ForStmt with iter_args carries its `iter_arg_rebind_<i>` carry plan, so codegen reads it instead of re-deriving it |
| `AccToGmStoreValid` | Every `tile.store` from an Acc-resident tile targets a GM dtype the backend's fix-pipe can narrow into |
| `AtomicAddDtypeValid` | Every atomic-add write into GM targets a destination dtype the backend's store pipe can combine |

### IRPropertySet

Efficient bitset-backed set with `Insert`, `Remove`, `Contains`, `ContainsAll`, `Union`, `Difference`, `ToString`.

### Declaring a New Property

An enumerator is spelled out in four places, and nothing in the build links them:

| Layer | File | Form |
| ----- | ---- | ---- |
| Enum | `include/pypto/ir/transforms/ir_property.h` | `MyProperty,` with a `///<` description |
| Name | `src/ir/transforms/ir_property.cpp` | `case IRProperty::MyProperty: return "MyProperty";` |
| Binding | `python/bindings/modules/passes.cpp` | `.value("MyProperty", IRProperty::MyProperty, "<doc>")` |
| Stub | `python/pypto/pypto_core/passes.pyi` | `MyProperty = ...` |

Add it to all four, in the enum's declaration order. A property missing from the binding still
compiles and still prints correctly from `str(IRPropertySet)` — the switch above renders the name —
but `IRPropertySet.to_list()` then raises `ValueError: <n> is not a valid IRProperty` for any set
containing it, and every Python caller of that set fails with it.
`tests/lint/check_ir_property_parity.py` (a pre-commit hook) holds the four lists together.

### PassProperties

```cpp
struct PassProperties {
  IRPropertySet required;      // Preconditions
  IRPropertySet produced;      // New properties guaranteed after running
  IRPropertySet invalidated;   // Properties this pass breaks
};
```

## Per-Pass Property Declarations

| Pass | Required | Produced | Invalidated |
| ---- | -------- | -------- | ----------- |
| InlineFunctions | — | InlineFunctionsEliminated | — |
| UnrollLoops | — | UnrollResolved | — |
| CtrlFlowTransform | — | StructuredCtrlFlow | — |
| ConvertToSSA | — | SSAForm | NormalizedStmtStructure |
| Simplify | — | — | — |
| NormalizeStmtStructure | — | NormalizedStmtStructure | — |
| FlattenCallExpr | SSAForm, NormalizedStmtStructure | SSAForm, NoNestedCalls, NormalizedStmtStructure | — |
| OutlineHierarchyScopes | SSAForm | SSAForm, HierarchyOutlined, OrchestrationReferencesResolved | — |
| OutlineIncoreScopes | SSAForm | SSAForm, SplitIncoreOrch, AivSplitValid | — |
| OutlineClusterScopes | SSAForm | SSAForm, ClusterOutlined | — |
| ConvertTensorToTileOps | SSAForm, SplitIncoreOrch, NormalizedStmtStructure | SSAForm, IncoreTileOps, NormalizedStmtStructure, AivSplitValid | AivSplitValid |
| OptimizeOrchTensors | SplitIncoreOrch, IncoreTileOps | SplitIncoreOrch, IncoreTileOps | — |
| LowerCompositeOps | — | — | — |
| FlattenTileNdTo2D | SSAForm, IncoreTileOps, NormalizedStmtStructure | SSAForm, TileOps2D, NormalizedStmtStructure | — |
| LegalizeTileCast | — | — | — |
| AutoTileMatmulL0 | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, NormalizedStmtStructure | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, NormalizedStmtStructure | — |
| CanonicalizeTileSlice | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, NormalizedStmtStructure | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, NormalizedStmtStructure | — |
| InferTileMemorySpace | SSAForm, IncoreTileOps, SplitIncoreOrch, NormalizedStmtStructure | SSAForm, TileMemoryInferred, NormalizedStmtStructure, AivSplitValid, AccToGmStoreValid | AivSplitValid |
| InsertMxScaleAddr | SSAForm, IncoreTileOps, SplitIncoreOrch, NormalizedStmtStructure, TileMemoryInferred | SSAForm, IncoreTileOps, SplitIncoreOrch, NormalizedStmtStructure, TileMemoryInferred | — |
| ResolveBackendOpLayouts | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D, NormalizedStmtStructure | — |
| LowerAutoVectorSplit | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D, TileMemoryInferred, NormalizedStmtStructure, AivSplitValid | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D, TileMemoryInferred, NormalizedStmtStructure | AivSplitValid |
| ExpandMixedKernel | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D, TileMemoryInferred, NormalizedStmtStructure | SSAForm, MixedKernelExpanded, NormalizedStmtStructure, HardSyncallOccupancyValid | — |
| InjectGMPipeBuffer | SSAForm, MixedKernelExpanded, NormalizedStmtStructure | SSAForm, MixedKernelExpanded, NormalizedStmtStructure | — |
| SplitVectorKernel | SSAForm, MixedKernelExpanded | SSAForm, VectorKernelSplit, NormalizedStmtStructure | — |
| StampTfreeSplit | SplitIncoreOrch | — | — |
| NormalizeReturnOrder | SplitIncoreOrch, IncoreTileOps | ReturnParamsExplicit | — |
| SkewCrossCorePipeline | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure | — |
| LowerPipelineToSlots | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure | — |
| LowerPipelineLoops | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure | — |
| CanonicalizeIOOrder | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure, PipelineResolved | — |
| MaterializeTensorStrides | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure, TensorViewCanonical | — |
| InitMemRef | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred | HasMemRefs, NormalizedStmtStructure | SSAForm |
| MaterializeSemanticAliases | SplitIncoreOrch, IncoreTileOps, HasMemRefs, TileOps2D, NormalizedStmtStructure | NormalizedStmtStructure | — |
| MemoryReuse | SplitIncoreOrch, IncoreTileOps, HasMemRefs, TileOps2D, NormalizedStmtStructure | NormalizedStmtStructure | — |
| AllocateMemoryAddr | SplitIncoreOrch, IncoreTileOps, HasMemRefs, TileOps2D | AllocatedMemoryAddr | — |
| FoldNoOpReshape | SplitIncoreOrch, IncoreTileOps, HasMemRefs, TileOps2D | — | — |
| FuseCreateAssembleToSlice | SplitIncoreOrch | — | — |
| DeriveCallDirections | SplitIncoreOrch | CallDirectionsResolved | — |
| AutoDeriveTaskDependencies | SplitIncoreOrch, CallDirectionsResolved | CallDirectionsResolved | — |
| ExpandManualPhaseFence | NoNestedCalls, NormalizedStmtStructure, CallDirectionsResolved | NoNestedCalls, NormalizedStmtStructure, CallDirectionsResolved | — |
| SynthesizeAllReduceSignals | — | — | — |
| MaterializeCommDomainScopes | — | CommDomainScopesMaterialized | — |
| LowerHostTensorCollectives | CommDomainScopesMaterialized | CommDomainScopesMaterialized | — |
| MaterializeDistTensorCtx | CommDomainScopesMaterialized, ReturnParamsExplicit | CommDomainScopesMaterialized, DistTensorCtxMaterialized | — |
| MaterializeRuntimeScopes | SplitIncoreOrch, CallDirectionsResolved | RuntimeScopesMaterialized | — |
| ClassifyIterArgCarry | CallDirectionsResolved, RuntimeScopesMaterialized | IterArgCarryClassified, RuntimeScopesMaterialized | — |
| InsertCommFence | SplitIncoreOrch | — | — |
| MaterializeValidShapeSymbols | — | — | — |

The table lists every registered pass, in `Default`-strategy execution order. Update the row
here whenever you add a pass or change its declaration.

Most `PassProperties` constants live in `include/pypto/ir/transforms/pass_properties.h`, but that
header is not the whole list: `kFuseCreateAssembleToSliceProperties` is declared locally in
`src/ir/transforms/fuse_create_assemble_to_slice_pass.cpp`. What authoritatively binds a pass
*name* to its properties is the `CreateFunctionPass` / `CreateProgramPass` call site, since that
is where `Pass::GetName()` and the constant meet. Regenerate a row from the call sites rather
than from the header alone, or a pass-local declaration is silently missed.

> **Note**: VerifySSA and TypeCheck are **PropertyVerifiers** (verification rules), not Passes. They run via `VerificationInstrument` or the `run_verifier()` utility — see [Verifier](99-verifier.md). That is why no pass declares `TypeChecked`: it is a *structural* property (`GetStructuralProperties()`), verified once on the pipeline's input IR rather than established by any pass.

## C++ Pass Infrastructure

### Pass Class

```cpp
class Pass {
  ProgramPtr operator()(const ProgramPtr& program) const;  // checks PassContext
  std::string GetName() const;
  IRPropertySet GetRequiredProperties() const;
  IRPropertySet GetProducedProperties() const;
  IRPropertySet GetInvalidatedProperties() const;
};
```

`Pass::operator()` checks `PassContext::Current()` and runs instruments before/after the actual transform.

### Creating Passes with Properties

```cpp
namespace pass {
Pass YourPass() {
  return CreateFunctionPass(TransformFunc, "YourPass",
      {.required = {IRProperty::SSAForm},
       .produced = {IRProperty::SomeProperty},
       .invalidated = {IRProperty::AnotherProperty}});
}
}
```

## PassContext and Instruments

**Header**: `include/pypto/ir/transforms/pass_context.h`

### PassInstrument

Abstract base class for pass instrumentation callbacks:

```cpp
class PassInstrument {
  virtual void RunBeforePass(const Pass& pass, const ProgramPtr& program) = 0;
  virtual void RunAfterPass(const Pass& pass, const ProgramPtr& program) = 0;
  virtual std::string GetName() const = 0;
};
```

### VerificationInstrument

Concrete instrument that uses `PropertyVerifierRegistry` to verify properties:

```cpp
class VerificationInstrument : public PassInstrument {
  explicit VerificationInstrument(VerificationMode mode);
  // BEFORE: verify required properties before pass
  // AFTER: verify produced properties after pass
  // BEFORE_AND_AFTER: both
};
```

### CallbackInstrument

Lightweight instrument that invokes user-provided callbacks, useful for ad-hoc instrumentation (IR dumping, logging, profiling) without subclassing `PassInstrument`:

```cpp
class CallbackInstrument : public PassInstrument {
  using Callback = std::function<void(const Pass&, const ProgramPtr&)>;
  explicit CallbackInstrument(Callback before_pass = nullptr,
                              Callback after_pass = nullptr,
                              std::string name = "CallbackInstrument");
};
```

```python
# Python: dump IR after each pass
def after_pass(p, program):
    print(f"After {p.get_name()}")

with passes.PassContext([passes.CallbackInstrument(after_pass=after_pass)]):
    pipeline.run(program)
```

`run_passes(dump_ir=True)` uses `CallbackInstrument` internally to dump IR after each pass, delegating verification to the C++ pipeline. When invoked inside an existing `PassContext`, dump mode preserves the outer context's instruments (e.g., user-provided `VerificationInstrument`) and verification level, appending the dump instrument to the combined list.

**Dump verbosity (`PassDumpLevel`).** The `dump_passes` knob (on `ir.compile`, `RunConfig`, and `run_passes`' `dump_ir`) accepts a `PassDumpLevel` enum — or a `bool` for backwards compatibility (`True` → `CONCISE`, `False` → `NONE`):

| Level | Meaning |
| ----- | ------- |
| `NONE` | No per-pass dumps. |
| `CONCISE` | Concise canonical IR (the default); best for diffing passes. |
| `EXPLICIT` | Fully-resolved dump — self-describing for layouts (issue #2088). |

By default (`CONCISE`) a dumped `pl.Tile` annotation omits its `blayout`/`slayout`/`fractal` whenever they equal the memory-space *implicit* view, and canonical IR stores an implicit view as `nullopt` — so a tile can print with no `TileView` at all even though its real layout is non-trivial (e.g. a `pl.Mem.Acc` tile is really `blayout=col_major, slayout=row_major, fractal=1024`). `EXPLICIT` makes every dumped tile print its fully-resolved layout from `GetEffectiveTileView`, and surfaces the `window_buffer` back-reference that a `pld.DistributedTensor` carries but the concise form drops — so a layout/aliasing bug is decidable from the printed IR alone. `EXPLICIT` dumps still reparse to identical IR: the tile layout canonicalizes back to `nullopt` (an explicit view matching the implicit one), and the window-buffer marker is an informational trailing string the parser strips on reload (the real reference re-derives from `pld.tensor.window`). This keeps `compiled.validate_ir()` — which reloads every dump — working. Programmatically, pass `explicit_layout=True` to `python_print(...)`.

```python
from pypto.ir import PassDumpLevel
from pypto.runtime import RunConfig

RunConfig(dump_passes=PassDumpLevel.EXPLICIT)   # fully-resolved dumps
RunConfig(dump_passes=True)                     # == PassDumpLevel.CONCISE
```

### ReportInstrument

Carries the directory that on-disk pipeline artifacts are written to. It observes no pass itself — `DiagnosticInstrument` reads its `output_dir` to decide where to append `perf_hints.log`:

```cpp
class ReportInstrument : public PassInstrument {
  explicit ReportInstrument(std::string output_dir);
  const std::string& GetOutputDir() const;
};
```

```python
instrument = passes.ReportInstrument("/path/to/report")

with passes.PassContext([instrument]):
    pipeline.run(program)
```

`compile()` creates one pointing at `build_output/<name>/report/`.

Memory usage is no longer reported here. It is rendered from a pass dump by `python -m pypto.tools.memory_map` — see [Memory Map](../07-memory-map.md).

### RoundtripInstrument

Print→parse roundtrip verification instrument. After every pass, it:

1. Prints the resulting IR to Python DSL text via `python_print()`
2. Parses the text back to an IR `Program` via `parse()`
3. Asserts `structural_equal(original, reparsed)` — a failure means the printer or parser cannot faithfully represent the IR produced by that pass

```python
from pypto.pypto_core import passes
from pypto.ir.instruments import make_roundtrip_instrument

with passes.PassContext([make_roundtrip_instrument()]):
    result = passes.convert_to_ssa()(program)
```

**Known non-fatal cases** (instrument skips the check without failing):

| Case | Behaviour | Reason |
| ---- | --------- | ------ |
| Printer `InternalError` (e.g. `ForKind::Unroll` + SSA `iter_args`) | `UserWarning`, roundtrip skipped | No valid DSL syntax for this transitional state |
| `UnknownType` in original IR (manually built via `ir.Call(ir.Op(...))`) | Silent skip | Parsing infers a concrete type; this is a type improvement, not a bug |
| `tensor.add(x, scalar)` → `tensor.adds` after roundtrip | Silent skip | Python API dispatches scalar RHS to `tensor.adds`; manual construction used wrong op name |
| `tile.load` 3-arg → 4-arg after roundtrip | Silent skip | C++ requires 4 args; manually constructed IR with 3 args is normalised by the printer |
| Variable pointer mismatch (dynamic-shape Vars in return types) | Silent skip | `structural_equal` without `enable_auto_mapping` cannot track Vars outside the function body |

**Enabled by default in unit tests** via `tests/ut/conftest.py` (see [Test Fixture](#test-fixture) below). Disable with `PYPTO_VERIFY_LEVEL=basic` or `PYPTO_VERIFY_LEVEL=none`.

### PassContext

Thread-local context stack with `with`-style nesting. Holds both instruments and pass configuration (e.g., verification level):

```cpp
class PassContext {
  explicit PassContext(std::vector<PassInstrumentPtr> instruments,
                       VerificationLevel verification_level = VerificationLevel::Basic);
  void EnterContext();      // push onto thread-local stack
  void ExitContext();       // pop from stack
  VerificationLevel GetVerificationLevel() const;
  static PassContext* Current();  // get active context
};
```

**All pass-related configuration belongs in PassContext** — see `.claude/rules/pass-context-config.md`.

### Python Usage

```python
from pypto.pypto_core import passes

# Enable verification for a block of code
with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.AFTER)]):
    result = passes.convert_to_ssa()(program)  # instruments fire automatically

# Disable automatic verification for a block
with passes.PassContext([], passes.VerificationLevel.NONE):
    result = pipeline.run(program)  # no automatic verification

# Nesting: inner context overrides outer
with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.AFTER)]):
    with passes.PassContext([]):  # disable instruments for this block
        result = some_pass(program)  # no verification
```

### Test Fixture

All unit tests automatically run with property verification **and roundtrip verification** via `tests/ut/conftest.py`. Roundtrip is the default for tests so that printer/parser asymmetries are caught automatically.

```python
@pytest.fixture(autouse=True)
def pass_verification_context():
    level_str = os.environ.get("PYPTO_VERIFY_LEVEL", "roundtrip").lower()
    instruments = []
    if level_str != "none":
        instruments.append(passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER))
    if level_str == "roundtrip":
        from pypto.ir.instruments import make_roundtrip_instrument
        instruments.append(make_roundtrip_instrument())
    with passes.PassContext(instruments):
        yield
```

Override via environment variable:

| `PYPTO_VERIFY_LEVEL` | Property verification | Roundtrip |
| -------------------- | --------------------- | --------- |
| `roundtrip` (default for tests) | ✅ BEFORE_AND_AFTER | ✅ |
| `basic` | ✅ BEFORE_AND_AFTER | ❌ |
| `none` | ❌ | ❌ |

### PassPipeline (C++)

```cpp
class PassPipeline {
  void AddPass(Pass pass);
  ProgramPtr Run(const ProgramPtr& program) const;  // executes passes in order
  std::vector<std::string> GetPassNames() const;
  std::vector<Pass> GetPasses() const;
};
```

`PassPipeline` is the single source of truth for ordered pass objects and their names. `GetPasses()` returns lightweight copies of the pass handles for inspection or composing another pipeline. Each pass's `operator()` checks the active `PassContext` for instruments.

### Automatic Verification

When `VerificationLevel` is `Basic` (the default), the pipeline automatically verifies the **lightweight properties** listed by `GetVerifiedProperties()` (`src/ir/transforms/ir_property.cpp`), each one exactly once per time it is produced. This catches common IR errors without requiring manual `PassContext` setup.

The members of that set — and of `GetStructuralProperties()` and `GetDefaultVerifyProperties()` — are restated in prose three times: the `Returns {...}` clause on each declaration in `ir_property.h`, and a summary row in each language's [Verifier](99-verifier.md) doc. `tests/lint/check_property_set_doc_parity.py` (a pre-commit hook) holds those copies to the C++ initializers, since a property added to an initializer alone compiles and passes CI while every list a developer reads stays short by one.

**How it works**:

1. At pipeline input, verify `GetStructuralProperties() ∩ GetVerifiedProperties()` — the invariants that hold on the user's own IR before any pass runs
2. After each pass, verify the properties it *produces* that are in `GetVerifiedProperties()` and not already verified
3. When a pass *invalidates* such a property, drop it from the verified set so a later producer re-verifies it
4. Throw `VerificationError` on errors

**With the `Default` strategy** (23 checks; the two sets are declared in `ir_property.cpp`, so this schedule follows from them and from the per-pass table above):

| Verification point | Properties verified |
| ------------------ | ------------------- |
| pipeline input | TypeChecked, BreakContinueValid, NoRedundantBlocks, InOutUseValid, ManualDepsOnSubmitOnly, AtomicAddDtypeValid |
| ConvertToSSA | SSAForm |
| OutlineIncoreScopes | AivSplitValid |
| ConvertTensorToTileOps | AivSplitValid *(re-verified — the pass invalidates it, see [10](10-convert_tensor_to_tile_ops.md))* |
| InferTileMemorySpace | AivSplitValid *(re-verified)*, TileMemoryInferred, AccToGmStoreValid, AccCompactValid |
| ExpandMixedKernel | MixedKernelExpanded, HardSyncallOccupancyValid, AccCompactValid *(re-verified)* |
| NormalizeReturnOrder | ReturnParamsExplicit |
| AllocateMemoryAddr | AllocatedMemoryAddr |
| DeriveCallDirections | CallDirectionsResolved |
| MaterializeDistTensorCtx | DistTensorCtxMaterialized |
| LegalizeGraphBoundary | GraphBoundaryLegalized |
| MaterializeRuntimeScopes | RuntimeScopesMaterialized |
| ClassifyIterArgCarry | IterArgCarryClassified |

A pass that under-declares `produced` therefore does not just mis-document itself — it silently removes a verification from this schedule.

**Control via `PassContext`**:

```python
from pypto import ir
from pypto.pypto_core import passes

# Disable automatic verification via PassContext
with passes.PassContext([], passes.VerificationLevel.NONE):
    pipeline.run(program)

# Or per-compilation
ir.compile(program, verification_level=ir.VerificationLevel.NONE)

# Environment variable (default when no PassContext): PYPTO_VERIFY_LEVEL=none|basic|roundtrip
```

**How the level is determined**:

1. If `PassContext` is active → use its `verification_level` (default: Basic)
2. If no `PassContext` → use `GetDefaultVerificationLevel()` (reads `PYPTO_VERIFY_LEVEL` env var, default: Basic)

## Python PassManager

**File**: `python/pypto/ir/pass_manager.py`

### API

| Method | Description |
| ------ | ----------- |
| `get_strategy(strategy)` | Get PassManager configured for strategy |
| `run_passes(program, dump_ir, output_dir, prefix)` | Execute passes via PassPipeline |
| `get_pass_names()` | Get names of all passes |
| `passes` / `pass_names` | Read-only snapshots derived from the underlying PassPipeline |

### Usage

```python
from pypto import ir
from pypto.pypto_core import passes

# Default usage
pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.Default)
result = pm.run_passes(program)

# With verification via PassContext
with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.AFTER)]):
    result = pm.run_passes(program)
```

### Strategy Notes

The PTO-oriented tile stage of `Default` is:

1. [`LowerCompositeOps`](12-lower_composite_ops.md)
2. [`FlattenTileNdTo2D`](13-flatten_tile_nd_to_2d.md)
3. [`LegalizeTileCast`](15-legalize_tile_cast.md) (expands `tile.cast` pairs the target ISA cannot emit as one `pto.tcvt`)
4. [`AutoTileMatmulL0`](16-auto_tile_matmul_l0.md)
5. [`CanonicalizeTileSlice`](17-canonicalize_tile_slice.md)
6. `InferTileMemorySpace`
7. [`InsertMxScaleAddr`](19-insert_mx_scale_addr.md) (Ascend950 MX path; inserts internal scale-address bindings after memory spaces are resolved)
8. [`ResolveBackendOpLayouts`](20-resolve_backend_op_layouts.md) (self-normalizes statement structure internally)
9. [`LowerAutoVectorSplit`](21-lower_auto_vector_split.md) (live auto-split lowering path; converts AUTO `pl.split` mixed InCore functions into the explicit `split_aiv` form before ExpandMixedKernel)
10. `ExpandMixedKernel`
11. [`InjectGMPipeBuffer`](23-inject_gm_pipe_buffer.md)
12. [`SplitVectorKernel`](24-split_vector_kernel.md) (only stamps attrs for split_aiv functions + handles the no-split dual-AIV path)
13. [`StampTfreeSplit`](25-stamp_tfree_split.md) (copies each cross-core tpop's split/pipe-id onto its matching tfree op)
14. `NormalizeReturnOrder`
15. [`SkewCrossCorePipeline`](27-skew_cross_core_pipeline.md) (cross-core cube/vector software-pipeline skew; runs immediately before LowerPipelineLoops)
16. [`LowerPipelineToSlots`](28-lower_pipeline_to_slots.md) (rotates an eligible `pl.pipeline` body through the slots of one allocation instead of replicating it; self-gated on `memory_planner=PTOAS`, and every loop it declines is left for `LowerPipelineLoops`)
17. [`LowerPipelineLoops`](29-lower_pipeline_loops.md)
18. [`CanonicalizeIOOrder`](30-canonicalize_io_order.md)
19. [`MaterializeTensorStrides`](31-materialize_tensor_strides.md) — wired into the default pipeline starting from RFC #1300 P6
20. `InitMemRef`
21. [`MaterializeSemanticAliases`](33-materialize_semantic_aliases.md) (semantics-required must-alias: loop-carry / in-place; always runs)
22. `MemoryReuse`
23. `AllocateMemoryAddr`
24. [`FoldNoOpReshape`](36-fold_no_op_reshape.md)
25. [`FuseCreateAssembleToSlice`](37-fuse_create_assemble_to_slice.md)
26. [`DeriveCallDirections`](38-derive_call_directions.md)
27. [`AutoDeriveTaskDependencies`](39-auto_derive_task_dependencies.md) (compiler deps for runtime scopes; AUTO-scope analysis is opt-in)
28. [`ExpandManualPhaseFence`](40-expand_manual_phase_fence.md) (manual-scope phase-fence TaskId dep compression)
29. [`SynthesizeAllReduceSignals`](41-synthesize_allreduce_signals.md) (distributed: host allreduce optional signal -> explicit internal signal IR)
30. [`MaterializeCommDomainScopes`](42-materialize_comm_domain_scopes.md) (distributed: WindowBuffer + CommDomainScopeStmt wrappers in each host_orch body; no-op for comm-less programs)
31. [`LowerHostTensorCollectives`](43-lower_host_tensor_collectives.md) (host-level tensor collectives -> internal builtin chip dispatches)
32. [`MaterializeDistTensorCtx`](44-materialize_dist_tensor_ctx.md) (explicit CommCtx params/args for DistributedTensor params)
33. `Simplify`
34. [`LegalizeGraphBoundary`](45-legalize_graph_boundary.md) (hoists values a Graph body derives from its boundary scalars to the call sites, and rejects the boundaries the host_build_graph runtime cannot record; no-op for programs with no Graph function)
35. [`MaterializeRuntimeScopes`](46-materialize_runtime_scopes.md) (inserts AUTO RuntimeScopeStmt so orchestration codegen emits SIMPLER_SCOPE 1:1)
36. [`ClassifyIterArgCarry`](47-classify_iter_arg_carry.md) (stamps each ForStmt iter_arg as trivial alias / rebind carry, and sizes manual-scope TaskId fence arrays)
37. [`InsertCommFence`](48-insert_comm_fence.md) (inserts a whole-tensor system.cacheinvalid + GM system.fence between each publishing write and the pld.system.notify that releases it; runs dead last so the inserted ops stay adjacent to their notify through codegen)

[`ResolveBackendOpLayouts`](20-resolve_backend_op_layouts.md) repairs
backend-constrained elementwise tile ops using registered layout metadata.
For the current PTO row-major elementwise ops, it rewrites `[N, 1]` vector
operands into `[1, N] row_major` `tile.reshape` operations at the
constrained use site, where row-major is inferred from the target shape.
It then reshapes the result back to the original vector shape when
needed.

[`NormalizeReturnOrder`](26-normalize_return_order.md) reorders `ReturnStmt::value_` in InCore functions so that
`return[i]` corresponds to the i-th `Out`/`InOut` parameter in declaration order,
and updates `TupleGetItemExpr` indices at call sites accordingly. This lets
orchestration codegen map tuple element indices to output parameters with a
direct `out_indices[i]` lookup, without tracing through `tile.store`/yield
chains. The pass is placed before `InitMemRef` so it runs after all kernel
splitting but before memory allocation.

`Simplify` folds arithmetic identities (`x + 0 → x`, `x * 1 → x`), evaluates
constant-only expressions, runs range-aware rewrites using loop-variable
bounds and if-branch constraints, and propagates scalar constants through
single-assignment bindings. As a final step it runs a **conservative scalar
DCE**: any `AssignStmt` whose LHS is a scalar `Var` and whose RHS contains
no `Call` anywhere is removed once its LHS has no remaining uses. Any
expression that contains a `Call` — at the top level or nested inside an
arithmetic tree — is preserved because the IR has no purity annotation yet,
so the call might have observable side effects. The DCE step recurses into
`ForStmt`/`IfStmt`/`WhileStmt`/`ScopeStmt` bodies so nested dead scalars
are cleaned up as well.

### Using PassPipeline Directly

```python
from pypto.pypto_core import passes

pipeline = passes.PassPipeline()
pipeline.add_pass(passes.convert_to_ssa())
pipeline.add_pass(passes.init_mem_ref())
pipeline.add_pass(passes.memory_reuse())

# Execute
result = pipeline.run(program)

# Inspect pass properties
p = passes.convert_to_ssa()
print(p.get_name())                  # "ConvertToSSA"
print(p.get_produced_properties())   # {SSAForm}
```

## Adding New Passes

1. **Declare** in `passes.h`: `Pass YourNewPass();`
2. **Implement** in `src/ir/transforms/` with `PassProperties`
3. **Python binding** in `python/bindings/modules/passes.cpp`
4. **Property declarations**: Set required/produced/invalidated in factory
5. **Type stub** in `python/pypto/pypto_core/passes.pyi`
6. **Register** in PassManager if part of a strategy
7. **Test** in `tests/ut/ir/transforms/`

## Testing

- `tests/ut/ir/transforms/test_ir_property.py` — IRProperty/IRPropertySet tests
- `tests/ut/ir/transforms/test_pass_pipeline.py` — Pipeline, PassContext, instruments, and automatic verification tests
- `tests/ut/ir/transforms/test_pass_manager.py` — PassManager backward compatibility
- `tests/ut/conftest.py` — Autouse fixture enabling AFTER verification for all tests
