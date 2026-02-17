# Pass, PassPipeline, and PassManager

Framework for organizing and executing IR transformation passes on Programs with property tracking, requirement verification, and strategy-based optimization pipelines.

## Overview

| Component | Description |
| --------- | ----------- |
| **Pass (C++)** | Standalone class for Program → Program transformations with property declarations |
| **IRProperty / IRPropertySet** | Enum + bitset for verifiable IR properties (SSAForm, HasMemRefs, etc.) |
| **PassPipeline (C++)** | Ordered sequence of passes with property tracking and optional verification |
| **PassManager (Python)** | High-level manager using PassPipeline, with strategy-based optimization |

### Key Features

- **Property Tracking**: Passes declare required, produced, and invalidated properties
- **Static Validation**: `PassPipeline.Validate()` checks property flow without executing
- **Runtime Verification**: Optional property verifiers before/after each pass
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

**Property update rule**: `new_props = (current - invalidated) | produced`. Required properties are auto-preserved.

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

> **Note**: VerifySSA and TypeCheck are **PropertyVerifiers** (verification rules), not Passes. They run via `RunVerifier` or `PassPipeline` verification modes — see [Verifier](01-verifier.md).

## C++ Pass Infrastructure

### Pass Class

```cpp
class Pass {
  ProgramPtr operator()(const ProgramPtr& program) const;
  std::string GetName() const;
  IRPropertySet GetRequiredProperties() const;
  IRPropertySet GetProducedProperties() const;
  IRPropertySet GetInvalidatedProperties() const;
};
```

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

### PassPipeline (C++)

```cpp
enum class VerificationMode { None, Before, After, BeforeAndAfter };

class PassPipeline {
  void AddPass(Pass pass);
  void SetVerificationMode(VerificationMode mode);
  void SetInitialProperties(const IRPropertySet& properties);
  ProgramPtr Run(const ProgramPtr& program) const;  // throws on unmet requirements
  std::vector<std::string> Validate() const;         // static check without executing
  std::vector<std::string> GetPassNames() const;
};
```

`Run()` tracks `current_props`, checks requirements before each pass, and updates properties after. `Validate()` simulates the flow without executing.

## Python PassManager

**File**: `python/pypto/ir/pass_manager.py`

### API

| Method | Description |
| ------ | ----------- |
| `get_strategy(strategy, verification_mode)` | Get PassManager configured for strategy |
| `run_passes(program, dump_ir, output_dir, prefix)` | Execute passes via PassPipeline |
| `validate()` | Static property flow validation |
| `get_pass_names()` | Get names of all passes |

### Usage

```python
from pypto import ir

# Default usage
pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.Default)
result = pm.run_passes(program)

# With verification
pm = ir.PassManager.get_strategy(
    ir.OptimizationStrategy.Default,
    verification_mode=ir.VerificationMode.BEFORE_AND_AFTER,
)
result = pm.run_passes(program)

# Static validation
errors = pm.validate()  # [] if valid
```

### Using PassPipeline Directly

```python
from pypto.pypto_core import passes

pipeline = passes.PassPipeline()
pipeline.add_pass(passes.convert_to_ssa())
pipeline.add_pass(passes.init_mem_ref())
pipeline.add_pass(passes.basic_memory_reuse())

# Static validation
errors = pipeline.validate()

# Execute with property tracking
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
- `tests/ut/ir/transforms/test_pass_pipeline.py` — Pipeline validation and execution
- `tests/ut/ir/transforms/test_pass_manager.py` — PassManager backward compatibility
