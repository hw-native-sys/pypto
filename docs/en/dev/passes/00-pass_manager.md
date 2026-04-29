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
- **Strategy-based Pipelines**: Pre-configured optimization levels (`Default`, `DebugTileOptimization`)
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
| `ClusterOutlined` | Cluster scopes outlined into Group functions |
| `HasMemRefs` | MemRef objects initialized on variables |
| `IncoreTileOps` | InCore functions use tile ops |
| `MixedKernelExpanded` | Mixed InCore functions split into AIC + AIV + Group |
| `AllocatedMemoryAddr` | All MemRefs have valid addresses within buffer limits |

### IRPropertySet

Efficient bitset-backed set with `Insert`, `Remove`, `Contains`, `ContainsAll`, `Union`, `Difference`, `ToString`.

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
| UnrollLoops | TypeChecked | TypeChecked | — |
| CtrlFlowTransform | TypeChecked | TypeChecked, StructuredCtrlFlow | — |
| ConvertToSSA | TypeChecked | TypeChecked, SSAForm | NormalizedStmtStructure |
| FlattenCallExpr | SSAForm | SSAForm, NoNestedCalls | NormalizedStmtStructure |
| SplitChunkedLoops | TypeChecked, SSAForm | TypeChecked, SSAForm | — |
| InterchangeChunkLoops | TypeChecked, SSAForm | TypeChecked, SSAForm | — |
| NormalizeStmtStructure | TypeChecked | TypeChecked, NormalizedStmtStructure | — |
| OutlineIncoreScopes | TypeChecked, SSAForm | SplitIncoreOrch | — |
| OutlineClusterScopes | TypeChecked, SSAForm | ClusterOutlined | — |
| ConvertTensorToTileOps | SplitIncoreOrch | IncoreTileOps | — |
| FlattenTileNdTo2D | SSAForm, IncoreTileOps | SSAForm, TileOps2D | — |
| ResolveBackendOpLayouts | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D | NormalizedStmtStructure |
| ExpandMixedKernel | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D | SSAForm, MixedKernelExpanded | — |
| NormalizeReturnOrder | SplitIncoreOrch, IncoreTileOps | — | — |
| InitMemRef | TypeChecked, SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D | HasMemRefs | SSAForm |
| MemoryReuse | TypeChecked, SplitIncoreOrch, IncoreTileOps, HasMemRefs, TileOps2D | — | — |
| AllocateMemoryAddr | TypeChecked, SplitIncoreOrch, IncoreTileOps, HasMemRefs, TileOps2D | AllocatedMemoryAddr | — |
| FuseCreateAssembleToSlice | — | — | — |
| DeriveCallDirections | SplitIncoreOrch | CallDirectionsResolved | — |
| Simplify | — | — | — |

> **Note**: VerifySSA and TypeCheck are **PropertyVerifiers** (verification rules), not Passes. They run via `VerificationInstrument` or the `run_verifier()` utility — see [Verifier](99-verifier.md).

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

### ReportInstrument

Instrument that generates reports to files after specified passes. Uses `ReportGeneratorRegistry` to dispatch report generation:

```cpp
class ReportInstrument : public PassInstrument {
  explicit ReportInstrument(std::string output_dir);
  void EnableReport(ReportType type, std::string trigger_pass);
};
```

```python
# Python: generate memory report after AllocateMemoryAddr
instrument = passes.ReportInstrument("/path/to/report")
instrument.enable_report(passes.ReportType.Memory, "AllocateMemoryAddr")

with passes.PassContext([instrument]):
    pipeline.run(program)
```

`compile()` automatically creates a `ReportInstrument` that generates memory reports to `build_output/<name>/report/`.

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
};
```

`PassPipeline` is a simple ordered list of passes. Each pass's `operator()` checks the active `PassContext` for instruments.

### Automatic Verification

When `VerificationLevel` is `Basic` (the default), the pipeline automatically verifies a small set of **lightweight properties** exactly once each. This catches common IR errors without requiring manual `PassContext` setup.

**Verified properties**: `{SSAForm, TypeChecked, AllocatedMemoryAddr}`

**How it works**:

1. After each pass, check if it produced any verified properties not yet checked
2. Verify those properties using `PropertyVerifierRegistry`
3. Throw `VerificationError` on errors
4. Track verified properties to avoid re-checking

**With the `Default` strategy**:

| After Pass | Properties Verified | Cumulative |
| ---------- | ------------------- | ---------- |
| ConvertToSSA | SSAForm, TypeChecked | 2 |
| FlattenCallExpr | *(TypeChecked already verified — skipped)* | 2 |
| AllocateMemoryAddr | AllocatedMemoryAddr | 3 |

**Total: 3 property checks** (each property verified exactly once).

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

The PTO-oriented tile stage shared by `Default` and `DebugTileOptimization` is:

1. `FlattenTileNdTo2D`
2. `InferTileMemorySpace`
3. `ResolveTransposeLayout`
4. [`ResolveBackendOpLayouts`](16-resolve_backend_op_layouts.md)
5. `NormalizeStmtStructure`
6. `ExpandMixedKernel`
7. [`InjectGMPipeBuffer`](18-inject_gm_pipe_buffer.md)
8. `SplitVectorKernel`
9. `NormalizeReturnOrder`
10. [`LowerPipelineLoops`](21-lower_pipeline_loops.md)
11. [`CanonicalizeIOOrder`](22-canonicalize_io_order.md)
12. `InitMemRef`
13. `MemoryReuse`
14. [`LegalizePTOBufferReuse`](25-legalize_pto_buffer_reuse.md)
15. `AllocateMemoryAddr`
16. `FuseCreateAssembleToSlice`
17. [`DeriveCallDirections`](28-derive_call_directions.md)
18. `Simplify`

`DebugTileOptimization` is a debug-only strategy for inspecting this tile stage
without the tensor-only prefix passes. Use `Default` for normal compilation and
for non-strategy-specific tests so the maintained pipeline stays covered.

[`ResolveBackendOpLayouts`](16-resolve_backend_op_layouts.md) repairs
backend-constrained elementwise tile ops using registered layout metadata.
For the current PTO row-major elementwise ops, it rewrites `[N, 1]` vector
operands into `[1, N] row_major` `tile.reshape` operations at the
constrained use site, where row-major is inferred from the target shape.
It then reshapes the result back to the original vector shape when
needed.

`NormalizeReturnOrder` reorders `ReturnStmt::value_` in InCore functions so that
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
