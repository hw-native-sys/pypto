# IR Verifier

Extensible verification system for validating PyPTO IR correctness through pluggable rules with diagnostic reporting and Pass integration.

## Overview

| Component | Description |
| --------- | ----------- |
| **PropertyVerifier (C++)** | Base class for verification rules |
| **IRVerifier (C++)** | Manages rule collection and executes verification on Programs |
| **PropertyVerifierRegistry (C++)** | Singleton mapping IRProperty → PropertyVerifier factories |
| **Diagnostic** | Structured error/warning report with severity, location, and message |
| **VerificationError** | Exception thrown when verification fails in throw mode |

### Key Features

- **Pluggable Rule System**: Extend with custom verification rules
- **Selective Verification**: Enable/disable rules individually per use case
- **Dual Verification Modes**: Collect diagnostics or throw on first error
- **Pass Integration**: Use as a Pass in optimization pipelines
- **Comprehensive Diagnostics**: Collect all issues with source locations
- **Property-Based Verification**: Registry maps IRProperty values to verifiers for automatic pipeline checks

## Architecture

### Verification Rule System

The verifier uses a **plugin architecture** where each `PropertyVerifier` subclass is an independent rule:

- Rules run in registration order across all functions
- Each rule operates independently — one rule's failure doesn't affect others
- Rules receive `ProgramPtr` and internally decide whether to iterate over functions or check program-level properties
- Rules can be selectively enabled/disabled without removing them

### Verification Modes

| Mode | Method | Behavior | Use When |
| ---- | ------ | -------- | -------- |
| **Diagnostic Collection** | `Verify()` | Collects all errors/warnings, returns vector | Need complete error list, building tools, reporting |
| **Fail-Fast** | `VerifyOrThrow()` | Throws VerificationError on first error | Pipeline validation, testing, development |

**Mode selection guide**:

- Use `Verify()` for IDE/tool integration - users want to see all issues
- Use `VerifyOrThrow()` in pipelines - fail immediately on invalid IR
- Use `VerifyOrThrow()` in tests - clear pass/fail with exception handling

### Diagnostic System

**Diagnostic structure**:

| Field | Type | Purpose |
| ----- | ---- | ------- |
| `severity` | `DiagnosticSeverity` | Error or Warning |
| `rule_name` | `string` | Which rule detected the issue |
| `error_code` | `int` | Numeric error identifier |
| `message` | `string` | Human-readable description |
| `span` | `Span` | Source location information |

**Severity levels**:

- `Error`: IR is invalid, must be fixed
- `Warning`: IR is valid but potentially problematic

**Report generation**: `GenerateReport()` formats diagnostics into a human-readable report with counts, grouping, and location details.

### Integration with Pass System

The verifier integrates into Pass pipelines via `run_verifier()`:

- **Returns**: A `Pass` object (Program → Program transformation)
- **Behavior**: Validates program, logs diagnostics, throws on error
- **Configuration**: Accepts `disabled_rules` parameter
- **Pipeline position**: Typically inserted after transformations to validate output

**Design consideration**: The verifier Pass is **transparent** - it returns the input program unchanged if valid, making it safe to insert anywhere in a pipeline.

## Built-in Rules

| Rule Name | IRProperty | Purpose |
| --------- | ---------- | ------- |
| **SSAVerify** | SSAForm | No multiple assignment, no name shadowing, no missing yield |
| **TypeCheck** | TypeChecked | Type kind/dtype/shape/size consistency |
| **NoNestedCallVerify** | NoNestedCalls | No nested call expressions in args, conditions, ranges |
| **NormalizedStmtStructure** | NormalizedStmtStructure | Bodies are SeqStmts, consecutive assigns wrapped in OpStmts |
| **FlattenedSingleStmt** | FlattenedSingleStmt | No single-element SeqStmts/OpStmts |
| **SplitIncoreOrch** | SplitIncoreOrch | No InCore ScopeStmts remain in Opaque functions |
| **IncoreBlockOps** | IncoreBlockOps | InCore functions use block ops (no tensor-level ops remain) |
| **HasMemRefs** | HasMemRefs | All TileType variables have MemRef initialized |
| **AllocatedMemoryAddr** | AllocatedMemoryAddr | All MemRefs have valid addresses within buffer limits |

### SSAVerify

**Design goal**: Enforce SSA invariants that PyPTO IR depends on for correctness.

**Error types** (`ssa::ErrorType`):

| Error Code | Name | Description |
| ---------- | ---- | ----------- |
| 1 | `MULTIPLE_ASSIGNMENT` | Variable assigned more than once in the same scope |
| 2 | `NAME_SHADOWING` | Variable name shadows an outer scope variable |
| 3 | `MISSING_YIELD` | ForStmt or IfStmt missing required YieldStmt |

**Detection details**:

- **MULTIPLE_ASSIGNMENT**: Tracks all variable declarations per scope. Reports error if a variable name appears in multiple AssignStmt nodes within the same scope.
- **NAME_SHADOWING**: Maintains scope stack. Reports error when entering a nested scope (ForStmt, IfStmt) if any new variable name matches a name from an outer scope.
- **MISSING_YIELD**: Validates that loop and conditional blocks contain at least one yield statement where semantically required by IR structure.

**Why it matters**: SSA form enables optimization passes to make assumptions about variable lifetimes and dependencies. Violations can cause incorrect transformations.

### TypeCheck

**Design goal**: Catch type mismatches that would cause runtime errors or generate invalid code.

**Error types** (`typecheck::ErrorType`):

| Error Code | Name | Description |
| ---------- | ---- | ----------- |
| 101 | `TYPE_KIND_MISMATCH` | Type kind mismatch (e.g., ScalarType vs TensorType) |
| 102 | `DTYPE_MISMATCH` | Data type mismatch (e.g., INT64 vs FLOAT32) |
| 103 | `SHAPE_DIMENSION_MISMATCH` | Shape dimension count doesn't match |
| 104 | `SHAPE_VALUE_MISMATCH` | Shape dimension value mismatch |
| 105 | `SIZE_MISMATCH` | Vector size mismatch in control flow branches |

**Detection details**:

- **TYPE_KIND_MISMATCH**: Checks that operations receive the correct category of type (scalar, tensor, tuple, etc.).
- **DTYPE_MISMATCH**: Validates data type consistency across operations (e.g., all operands to an Add must have the same dtype).
- **SHAPE_DIMENSION_MISMATCH**: Ensures tensor operations receive inputs with compatible dimension counts.
- **SHAPE_VALUE_MISMATCH**: Validates specific dimension sizes match where required (e.g., matrix multiplication dimensions).
- **SIZE_MISMATCH**: In control flow (if/else, loops), ensures variable vectors have consistent sizes across branches.

### NoNestedCallVerify

**Error types** (`NestedCallErrorType`):

| Name | Description |
| ---- | ----------- |
| `CALL_IN_CALL_ARGS` | Call expression nested in another call's arguments |
| `CALL_IN_IF_CONDITION` | Call expression in if-statement condition |
| `CALL_IN_FOR_RANGE` | Call expression in for-loop range |
| `CALL_IN_BINARY_EXPR` | Call expression in binary expression |
| `CALL_IN_UNARY_EXPR` | Call expression in unary expression |

## PropertyVerifierRegistry

**Header**: `include/pypto/ir/transforms/property_verifier_registry.h`

Singleton registry mapping `IRProperty` values to `PropertyVerifier` factories. Used by `PassPipeline` to automatically verify properties before/after passes. Each verifier is co-located with its corresponding pass (e.g., `CreateSplitIncoreOrchPropertyVerifier` lives in `outline_incore_scopes_pass.cpp`), while the registry wires them together at startup. Factory declarations are in `verifier.h`.

| Method | Description |
| ------ | ----------- |
| `GetInstance()` | Get singleton instance |
| `Register(prop, factory)` | Register a verifier factory for a property |
| `GetVerifier(prop)` | Create a verifier instance (nullptr if none registered) |
| `HasVerifier(prop)` | Check if a verifier is registered |
| `VerifyProperties(properties, program)` | Verify a set of properties, return diagnostics |

All 9 built-in properties are pre-registered in the constructor.

## C++ API Reference

**Header**: `include/pypto/ir/transforms/verifier.h`

### PropertyVerifier Interface

Base class for implementing custom verification rules.

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `GetName()` | `std::string GetName() const` | Return unique rule identifier |
| `Verify()` | `void Verify(const ProgramPtr&, std::vector<Diagnostic>&)` | Check program and append diagnostics |

Each verifier receives a `ProgramPtr` and internally decides whether to iterate over functions or check program-level properties. Verifiers should append to diagnostics, not throw exceptions.

### IRVerifier Class

Manages verification rules and executes verification.

#### Construction and Configuration

| Method | Description |
| ------ | ----------- |
| `IRVerifier()` | Construct empty verifier with no rules |
| `static IRVerifier CreateDefault()` | Factory method - returns verifier with SSAVerify and TypeCheck rules |
| `void AddRule(PropertyVerifierPtr rule)` | Register a verification rule (ignored if duplicate name) |

#### Rule Management

| Method | Description |
| ------ | ----------- |
| `void EnableRule(const std::string& name)` | Enable previously disabled rule (no-op if not found) |
| `void DisableRule(const std::string& name)` | Disable rule by name - it will be skipped during verification |
| `bool IsRuleEnabled(const std::string& name) const` | Check if rule is currently enabled |

#### Verification Execution

| Method | Return | Throws | Description |
| ------ | ------ | ------ | ----------- |
| `Verify(const ProgramPtr&)` | `std::vector<Diagnostic>` | No | Run all enabled rules, collect all diagnostics |
| `VerifyOrThrow(const ProgramPtr&)` | `void` | `VerificationError` | Run verification, throw if any errors found |

#### Reporting

| Method | Description |
| ------ | ----------- |
| `static std::string GenerateReport(const std::vector<Diagnostic>&)` | Format diagnostics into readable report with counts and details |

**Report format**: Summary line with error/warning counts, followed by detailed listing of each diagnostic with rule name, severity, location, and message.

## Python API Reference

**Module**: `pypto.pypto_core.passes`

### IRVerifier Class

Python binding of C++ IRVerifier with snake_case naming.

#### Factory and Construction

| Method | Description |
| ------ | ----------- |
| `IRVerifier()` | Create empty verifier (usually not used directly) |
| `IRVerifier.create_default()` | Static method - returns verifier with default rules enabled |

#### Rule Management

| Method | Parameter | Description |
| ------ | --------- | ----------- |
| `enable_rule(name)` | `name: str` | Enable a disabled rule |
| `disable_rule(name)` | `name: str` | Disable a rule by name |
| `is_rule_enabled(name)` | `name: str` | Check if rule is enabled (returns `bool`) |

#### Verification

| Method | Parameter | Returns | Throws | Description |
| ------ | --------- | ------- | ------ | ----------- |
| `verify(program)` | `program: Program` | `list[Diagnostic]` | No | Collect all diagnostics |
| `verify_or_throw(program)` | `program: Program` | `None` | Exception | Throw on error |

#### Reporting

| Method | Parameter | Returns | Description |
| ------ | --------- | ------- | ----------- |
| `generate_report(diagnostics)` | `diagnostics: list[Diagnostic]` | `str` | Static method - format diagnostics |

### run_verifier Function

Factory function creating a verifier Pass for use in PassManager.

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `disabled_rules` | `list[str] \| None` | `None` | List of rule names to disable |
| **Returns** | `Pass` | - | Verifier Pass object |

**Usage**: `verify_pass = passes.run_verifier(disabled_rules=["TypeCheck"])`

### Diagnostic Type

Read-only structure representing a single verification issue.

| Field | Type | Description |
| ----- | ---- | ----------- |
| `severity` | `DiagnosticSeverity` | `Error` or `Warning` |
| `rule_name` | `str` | Name of rule that detected issue |
| `error_code` | `int` | Numeric identifier |
| `message` | `str` | Human-readable description |
| `span` | `Span` | Source code location |

### DiagnosticSeverity Enum

| Value | Meaning |
| ----- | ------- |
| `DiagnosticSeverity.Error` | IR is invalid |
| `DiagnosticSeverity.Warning` | Potentially problematic but valid |

## Usage Examples

### Basic Verification

```python
from pypto import ir
from pypto.pypto_core import passes

# Build program (assume 'program' is constructed)
verifier = passes.IRVerifier.create_default()
diagnostics = verifier.verify(program)

if diagnostics:
    report = passes.IRVerifier.generate_report(diagnostics)
    print(report)
```

### Disabling Rules

```python
# Create verifier and disable specific rules
verifier = passes.IRVerifier.create_default()
verifier.disable_rule("TypeCheck")  # Skip type checking

# Only SSAVerify will run
diagnostics = verifier.verify(program)
```

### Error Handling with Exceptions

```python
verifier = passes.IRVerifier.create_default()

try:
    verifier.verify_or_throw(program)
    print("Program is valid")
except Exception as e:
    print(f"Verification failed: {e}")
```

### Using as a Pass

```python
from pypto.ir import PassManager, OptimizationStrategy

# Verifier automatically included in Default strategy
pm = PassManager.get_strategy(OptimizationStrategy.Default)
result = pm.run_passes(program)  # Verifier runs after ConvertToSSA
```

### Custom Pass Configuration

```python
from pypto.pypto_core import passes

# Create verifier pass with specific rules disabled
verify_pass = passes.run_verifier(disabled_rules=["SSAVerify"])

# Use in custom pipeline
result = verify_pass(program)
```

## Adding Custom Rules

To extend the verifier with domain-specific checks, implement a custom PropertyVerifier.

### Implementation Steps

**1. Create Rule Class** (C++)

Inherit from `PropertyVerifier` and implement required methods:

```cpp
#include "pypto/ir/transforms/verifier.h"

class MyCustomRule : public PropertyVerifier {
 public:
  std::string GetName() const override { return "MyCustom"; }

  void Verify(const ProgramPtr& program,
              std::vector<Diagnostic>& diagnostics) override {
    for (const auto& [gv, func] : program->functions_) {
      // Verification logic per function
    }
  }
};
```

#### 2. Create Factory Function

```cpp
PropertyVerifierPtr CreateMyCustomRule() {
  return std::make_shared<MyCustomRule>();
}
```

#### 3. Register Rule

```cpp
// Add to default verifier in verifier.cpp CreateDefault():
verifier.AddRule(CreateMyCustomRule());

// Or register with PropertyVerifierRegistry for pipeline integration:
PropertyVerifierRegistry::GetInstance().Register(IRProperty::MyProp, CreateMyCustomRule);
```

**4. Python Binding** (optional)

Add to `python/bindings/modules/passes.cpp`:

```cpp
passes.def("create_my_custom_rule", &CreateMyCustomRule,
           "Create MyCustom verification rule");
```

**5. Type Stub** (optional)

Add to `python/pypto/pypto_core/passes.pyi`:

```python
def create_my_custom_rule() -> PropertyVerifier: ...
```

### Guidelines

- Use `IRVisitor` to traverse IR nodes systematically
- Keep rules focused — one rule checks one category of issues
- Avoid side effects — only read IR and write diagnostics
- Create descriptive diagnostics with severity, rule name, error code, message, and span

### Integration Points

| Location | Purpose |
| -------- | ------- |
| `src/ir/transforms/your_rule.cpp` | Implementation |
| `include/pypto/ir/transforms/passes.h` | Factory declaration (if exposing) |
| `src/ir/transforms/verifier.cpp` | Add to `CreateDefault()` |
| `python/bindings/modules/passes.cpp` | Python binding |
| `tests/ut/ir/transforms/test_verifier.py` | Test cases |

## Related Components

- **Pass System** (`00-pass_manager.md`): Verifier integrates as a Pass, PropertyVerifierRegistry used by PassPipeline
- **IRBuilder** (`../ir/06-builder.md`): Construct IR that verifier validates
- **Type System** (`../ir/02-types.md`): TypeCheck rule validates against type system
- **Error Handling** (`include/pypto/core/error.h`): Diagnostic and VerificationError definitions

## Testing

Test coverage in `tests/ut/ir/transforms/test_verifier.py`: valid/invalid program verification, rule enable/disable, exception vs. diagnostic modes, pass integration, diagnostic field access, report generation.
