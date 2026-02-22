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
- **Strategy-based Pipelines**: Pre-configured optimization levels (Default/PTOAS)
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
| `FlattenedSingleStmt` | Single-statement blocks flattened |
| `SplitIncoreOrch` | InCore scopes outlined into separate functions |
| `HasMemRefs` | MemRef objects initialized on variables |

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
| ConvertToSSA | TypeChecked | SSAForm | NormalizedStmtStructure, FlattenedSingleStmt |
| FlattenCallExpr | TypeChecked | NoNestedCalls | NormalizedStmtStructure, FlattenedSingleStmt |
| NormalizeStmtStructure | TypeChecked | NormalizedStmtStructure | FlattenedSingleStmt |
| FlattenSingleStmt | TypeChecked | FlattenedSingleStmt | NormalizedStmtStructure |
| OutlineIncoreScopes | SSAForm | SplitIncoreOrch | — |
| InitMemRef | SSAForm | HasMemRefs | — |
| BasicMemoryReuse | HasMemRefs | — | — |
| InsertSync | HasMemRefs | — | — |
| AddAlloc | HasMemRefs | — | — |
| RunVerifier | — | — | — |

> **Note**: VerifySSA and TypeCheck are **PropertyVerifiers** (verification rules), not Passes. They run via `RunVerifier` or `VerificationInstrument` — see [Verifier](01-verifier.md).

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

### PassContext

Thread-local context stack with `with`-style nesting:

```cpp
class PassContext {
  explicit PassContext(std::vector<PassInstrumentPtr> instruments);
  void EnterContext();      // push onto thread-local stack
  void ExitContext();       // pop from stack
  static PassContext* Current();  // get active context
};
```

### Python Usage

```python
from pypto.pypto_core import passes

# Enable verification for a block of code
with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.AFTER)]):
    result = passes.convert_to_ssa()(program)  # instruments fire automatically

# Nesting: inner context overrides outer
with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.AFTER)]):
    with passes.PassContext([]):  # disable instruments for this block
        result = some_pass(program)  # no verification
```

### Test Fixture

All unit tests automatically run with AFTER verification via `tests/ut/conftest.py`:

```python
@pytest.fixture(autouse=True)
def pass_verification_context():
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.AFTER)]):
        yield
```

### PassPipeline (C++)

```cpp
class PassPipeline {
  void AddPass(Pass pass);
  ProgramPtr Run(const ProgramPtr& program) const;  // executes passes in order
  std::vector<std::string> GetPassNames() const;
};
```

`PassPipeline` is a simple ordered list of passes. Each pass's `operator()` checks the active `PassContext` for instruments.

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

### Using PassPipeline Directly

```python
from pypto.pypto_core import passes

pipeline = passes.PassPipeline()
pipeline.add_pass(passes.convert_to_ssa())
pipeline.add_pass(passes.init_mem_ref())
pipeline.add_pass(passes.basic_memory_reuse())

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
- `tests/ut/ir/transforms/test_pass_pipeline.py` — Pipeline, PassContext, and instrument tests
- `tests/ut/ir/transforms/test_pass_manager.py` — PassManager backward compatibility
- `tests/ut/conftest.py` — Autouse fixture enabling AFTER verification for all tests
