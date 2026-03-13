# IR Verifier

Extensible verification system for validating PyPTO IR correctness through pluggable property verifiers with diagnostic reporting and Pass integration.

## Overview

| Component | Description |
| --------- | ----------- |
| **PropertyVerifier (C++)** | Base class for verification rules |
| **PropertyVerifierRegistry (C++)** | Singleton mapping IRProperty → PropertyVerifier factories with verify/report API |
| **Diagnostic** | Structured error/warning report with severity, location, and message |
| **VerificationError** | Exception thrown when verification fails |

### Key Features

- **Pluggable Rule System**: Extend with custom verification rules
- **Property-Based Verification**: Opt-in property sets — verify exactly what you need
- **Structural Properties**: TypeChecked and BreakContinueValid are verified once at pipeline start, not per-pass
- **Dual Verification Modes**: Collect diagnostics or throw on first error
- **Pass Integration**: Use as a Pass in optimization pipelines
- **Comprehensive Diagnostics**: Collect all issues with source locations

## Architecture

### Structural vs Pipeline Properties

| Category | Examples | Behavior |
| -------- | -------- | -------- |
| **Structural** | TypeChecked, BreakContinueValid | Always true. Verified at pipeline start. Never in PassProperties. |
| **Pipeline** | SSAForm, NoNestedCalls, HasMemRefs, ... | Produced/invalidated by passes. Verified per pass-declared contracts. |

`GetStructuralProperties()` returns `{TypeChecked, BreakContinueValid}`. These are verified **once at pipeline start** by `PassPipeline::Run()`. Since no pass declares them in `required`/`produced`/`invalidated`, they stay verified throughout.

### Verification Rule System

The verifier uses a **plugin architecture** where each `PropertyVerifier` subclass is an independent rule:

- Rules run in registration order across all functions
- Each rule operates independently — one rule's failure doesn't affect others
- Rules receive `ProgramPtr` and internally decide whether to iterate over functions or check program-level properties
- Rules can be selectively included via `IRPropertySet`

### Diagnostic System

| Field | Type | Purpose |
| ----- | ---- | ------- |
| `severity` | `DiagnosticSeverity` | Error or Warning |
| `rule_name` | `string` | Which rule detected the issue |
| `error_code` | `int` | Numeric error identifier |
| `message` | `string` | Human-readable description |
| `span` | `Span` | Source location information |

### Integration with Pass System

1. **Automatic property verification**: `PassPipeline` uses `PropertyVerifierRegistry` to check produced properties after each pass (controlled by `VerificationLevel` in `PassContext`). Structural properties are checked at pipeline start. See [Pass Manager](00-pass_manager.md).
2. **`VerificationInstrument`**: A `PassInstrument` that verifies required/produced properties before/after passes via `PassContext`.

The `run_verifier()` utility creates a standalone `Pass` for ad-hoc use in custom pipelines, but it is **not** part of the default optimization strategies.

## Built-in Rules

| Rule Name | IRProperty | Purpose |
| --------- | ---------- | ------- |
| **SSAVerify** | SSAForm | No multiple assignment, no name shadowing, no missing yield |
| **TypeCheck** | TypeChecked | Type kind/dtype/shape/size consistency |
| **NoNestedCall** | NoNestedCalls | No nested call expressions in args, conditions, ranges |
| **BreakContinueCheck** | BreakContinueValid | Break/continue only in sequential/while loops |
| **NormalizedStmtStructure** | NormalizedStmtStructure | Bodies are SeqStmts, consecutive assigns wrapped in OpStmts |
| **FlattenedSingleStmt** | FlattenedSingleStmt | No single-element SeqStmts/OpStmts |
| **SplitIncoreOrch** | SplitIncoreOrch | No InCore ScopeStmts remain in Opaque functions |
| **IncoreTileOps** | IncoreTileOps | InCore functions use tile ops (no tensor-level ops remain) |
| **HasMemRefs** | HasMemRefs | All TileType variables have MemRef initialized |
| **AllocatedMemoryAddr** | AllocatedMemoryAddr | All MemRefs have valid addresses within buffer limits |

### SSAVerify

**Error types** (`ssa::ErrorType`):

| Error Code | Name | Description |
| ---------- | ---- | ----------- |
| 1 | `MULTIPLE_ASSIGNMENT` | Variable assigned more than once in the same scope |
| 2 | `NAME_SHADOWING` | Variable name shadows an outer scope variable |
| 3 | `MISSING_YIELD` | ForStmt or IfStmt missing required YieldStmt |

### TypeCheck

**Error types** (`typecheck::ErrorType`):

| Error Code | Name | Description |
| ---------- | ---- | ----------- |
| 101 | `TYPE_KIND_MISMATCH` | Type kind mismatch (e.g., ScalarType vs TensorType) |
| 102 | `DTYPE_MISMATCH` | Data type mismatch |
| 103 | `SHAPE_DIMENSION_MISMATCH` | Shape dimension count mismatch |
| 104 | `SHAPE_VALUE_MISMATCH` | Shape dimension value mismatch |
| 105 | `SIZE_MISMATCH` | Vector size mismatch in control flow |
| 106 | `IF_CONDITION_MUST_BE_SCALAR` | IfStmt condition must be ScalarType |
| 107 | `FOR_RANGE_MUST_BE_SCALAR` | ForStmt range must be ScalarType |

### NoNestedCall

| Name | Description |
| ---- | ----------- |
| `CALL_IN_CALL_ARGS` | Call expression nested in another call's arguments |
| `CALL_IN_IF_CONDITION` | Call expression in if-statement condition |
| `CALL_IN_FOR_RANGE` | Call expression in for-loop range |
| `CALL_IN_BINARY_EXPR` | Call expression in binary expression |
| `CALL_IN_UNARY_EXPR` | Call expression in unary expression |

## PropertyVerifierRegistry

**Header**: `include/pypto/ir/verifier/property_verifier_registry.h`

Singleton registry mapping `IRProperty` values to `PropertyVerifier` factories. Used by `PassPipeline` to automatically verify properties before/after passes.

| Method | Description |
| ------ | ----------- |
| `GetInstance()` | Get singleton instance |
| `Register(prop, factory)` | Register a verifier factory for a property |
| `GetVerifier(prop)` | Create a verifier instance (nullptr if none registered) |
| `HasVerifier(prop)` | Check if a verifier is registered |
| `VerifyProperties(properties, program)` | Verify a set of properties, return diagnostics |
| `VerifyOrThrow(properties, program)` | Verify and throw VerificationError on errors |
| `GenerateReport(diagnostics)` | Static — format diagnostics into readable report |

## C++ API Reference

### PropertyVerifier Interface

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `GetName()` | `std::string GetName() const` | Return unique rule identifier |
| `Verify()` | `void Verify(const ProgramPtr&, std::vector<Diagnostic>&)` | Check program and append diagnostics |

### Structural and Default Properties

| Function | Returns | Description |
| -------- | ------- | ----------- |
| `GetStructuralProperties()` | `{TypeChecked, BreakContinueValid}` | Invariants verified at pipeline start |
| `GetDefaultVerifyProperties()` | `{SSAForm, TypeChecked, NoNestedCalls, BreakContinueValid}` | Default set for `run_verifier()` |
| `GetVerifiedProperties()` | `{SSAForm, TypeChecked, AllocatedMemoryAddr, BreakContinueValid}` | Lightweight set for `PassPipeline` auto-verify |

### RunVerifier Pass Factory

```cpp
Pass RunVerifier(const IRPropertySet& properties);
```

Creates a `Pass` that verifies the given properties using `PropertyVerifierRegistry`.

## Python API Reference

**Module**: `pypto.pypto_core.passes`

### PropertyVerifierRegistry

| Method | Parameter | Returns | Description |
| ------ | --------- | ------- | ----------- |
| `verify(properties, program)` | `IRPropertySet, Program` | `list[Diagnostic]` | Collect diagnostics |
| `verify_or_throw(properties, program)` | `IRPropertySet, Program` | `None` | Throw on error |
| `generate_report(diagnostics)` | `list[Diagnostic]` | `str` | Format diagnostics |

### Helper Functions

| Function | Returns | Description |
| -------- | ------- | ----------- |
| `get_default_verify_properties()` | `IRPropertySet` | Default properties for `run_verifier()` |
| `get_structural_properties()` | `IRPropertySet` | Structural invariant properties |

### run_verifier Function

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `properties` | `IRPropertySet \| None` | `None` | Properties to verify (None → default set) |
| **Returns** | `Pass` | - | Verifier Pass object |

## Usage Examples

### Basic Verification

```python
from pypto.pypto_core import passes

# Verify default properties
props = passes.get_default_verify_properties()
diagnostics = passes.PropertyVerifierRegistry.verify(props, program)

if diagnostics:
    report = passes.PropertyVerifierRegistry.generate_report(diagnostics)
    print(report)
```

### Selective Verification

```python
# Verify only specific properties
props = passes.IRPropertySet()
props.insert(passes.IRProperty.SSAForm)
props.insert(passes.IRProperty.TypeChecked)
diagnostics = passes.PropertyVerifierRegistry.verify(props, program)
```

### Disabling Checks

```python
# Start from default set and remove what you don't want
props = passes.get_default_verify_properties()
props.remove(passes.IRProperty.SSAForm)
diagnostics = passes.PropertyVerifierRegistry.verify(props, program)
```

### Error Handling with Exceptions

```python
props = passes.get_default_verify_properties()
try:
    passes.PropertyVerifierRegistry.verify_or_throw(props, program)
    print("Program is valid")
except Exception as e:
    print(f"Verification failed: {e}")
```

### Using in a Custom Pipeline

```python
# Create verifier pass (defaults to get_default_verify_properties())
verify_pass = passes.run_verifier()
result = verify_pass(program)

# Or with custom properties
props = passes.get_default_verify_properties()
props.remove(passes.IRProperty.SSAForm)
verify_pass = passes.run_verifier(properties=props)
result = verify_pass(program)
```

## Adding Custom Rules

### Implementation Steps

1. Inherit from `PropertyVerifier`, implement `GetName()` and `Verify()`
2. Create a factory function returning `PropertyVerifierPtr`
3. Register with `PropertyVerifierRegistry` in the constructor
4. Add Python binding and type stub (optional)

### Guidelines

- Use `IRVisitor` to traverse IR nodes systematically
- Keep rules focused — one rule checks one category of issues
- Avoid side effects — only read IR and write diagnostics
- Create descriptive diagnostics with severity, rule name, error code, message, and span

## Related Components

- **Pass System** (`00-pass_manager.md`): Verifier integrates as a Pass, PropertyVerifierRegistry used by PassPipeline
- **IRBuilder** (`../ir/06-builder.md`): Construct IR that verifier validates
- **Type System** (`../ir/02-types.md`): TypeCheck rule validates against type system
- **Error Handling** (`include/pypto/core/error.h`): Diagnostic and VerificationError definitions

## Testing

Test coverage in `tests/ut/ir/transforms/test_verifier.py`: valid/invalid program verification, property-based selection, exception vs. diagnostic modes, pass integration, diagnostic field access, report generation, structural/default property sets.
