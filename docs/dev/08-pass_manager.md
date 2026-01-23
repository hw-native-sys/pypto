# Pass and PassManager

The Pass and PassManager system provides a framework for organizing and executing IR transformation passes on Functions and Programs. This system enables optimization pipelines with different strategies (Default/Custom1/Custom2) and supports both Function-level and Program-level transformations.

## Overview

The Pass system consists of three main components:

1. **Pass (C++)** - Base class for IR transformations that operate on Functions
2. **PassManager (Python)** - Manages sequences of passes and execution strategies
3. **Concrete Passes** - Specific transformation implementations (e.g., IdentityPass)

### Key Features

- **Strategy-based Pipeline**: Pre-configured optimization levels (Default/Custom1/Custom2)
- **Immutable Transformations**: Passes return new IR nodes rather than modifying in place
- **Function and Program Support**: Can transform individual Functions or entire Programs
- **Pipeline Composition**: Multiple passes execute sequentially, with each pass's output feeding into the next

## C++ Pass Infrastructure

### Pass Base Class

The `Pass` class is the abstract base for all IR transformations. It extends `IRMutator` and defines the interface for function-level transformations.

**Header**: `include/pypto/ir/transform/base/pass.h`

```cpp
namespace pypto {
namespace ir {

/**
 * @brief Base class for IR transformation passes
 *
 * Pass is an abstract base class that extends IRMutator to provide function-level transformations.
 * Each pass operates on a Function and returns a transformed Function.
 * Passes maintain immutability - they return new FunctionPtr instances rather than modifying in place.
 */
class Pass : public IRMutator {
 public:
  ~Pass() override = default;

  /**
   * @brief Execute the pass on a function
   *
   * This is the main entry point for pass execution. Subclasses must implement this method
   * to define their transformation logic.
   *
   * @param func Input function to transform
   * @return Transformed function (may be the same pointer if no changes were made)
   */
  virtual FunctionPtr Run(const FunctionPtr& func) = 0;
};

}  // namespace ir
}  // namespace pypto
```

### IdentityPass Example

The `IdentityPass` is a simple concrete pass implementation used primarily for testing. It demonstrates the pass interface by appending `"_identity"` to function names.

**Header**: `include/pypto/ir/transform/identity_pass.h`

```cpp
namespace pypto {
namespace ir {

/**
 * @brief Identity pass that appends a suffix to function name
 *
 * This pass appends "_identity" to the function name for testing purposes.
 * This allows tests to verify that the pass was actually executed.
 */
class IdentityPass : public Pass {
 public:
  /**
   * @brief Execute the identity pass
   *
   * @param func Input function
   * @return New function with modified name
   */
  FunctionPtr Run(const FunctionPtr& func) override;
};

}  // namespace ir
}  // namespace pypto
```

**Implementation**: `src/ir/transform/identity_pass.cpp`

```cpp
FunctionPtr IdentityPass::Run(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "IdentityPass cannot run on null function";

  // Append "_identity" suffix to the function name to mark that this pass was applied
  std::string new_name = func->name_ + "_identity";

  // Create a new function with the modified name
  return std::make_shared<const Function>(new_name, func->params_, func->return_types_,
                                          func->body_, func->span_);
}
```

### Python Bindings

Passes are exposed to Python through nanobind bindings.

**File**: `python/bindings/modules/pass.cpp`

```cpp
void BindPass(nb::module_& m) {
  // Create a new 'passes' submodule (using 'passes' instead of 'pass' to avoid Python keyword)
  nb::module_ passes = m.def_submodule("passes", "IR transformation passes");

  // Pass base class for IR transformations
  nb::class_<Pass>(passes, "Pass", "Base class for IR transformation passes")
      .def("run", &Pass::Run, nb::arg("func"), "Execute the pass on a function");

  // IdentityPass - a pass that appends a suffix to function name
  nb::class_<IdentityPass, Pass>(passes, "IdentityPass",
                                 "A pass that appends '_identity' suffix to function name for testing")
      .def(nb::init<>(), "Create an identity pass");
}
```

The bindings create a `pypto.pypto_core.passes` module with:
- `Pass` base class with a `run(func)` method
- `IdentityPass` concrete implementation

## Python PassManager

The `PassManager` class provides a high-level API for managing and executing pass pipelines with different optimization strategies.

**File**: `python/pypto/ir/pass_manager.py`

### Optimization Strategies

```python
class OptimizationStrategy(Enum):
    """Enumeration of optimization strategies."""

    Default = "Default"      # No optimization
    Custom1 = "Custom1"      # Custom optimization strategy 1
    Custom2 = "Custom2"      # Custom optimization strategy 2
```

### PassManager Class

```python
class PassManager:
    """Manager for organizing and executing IR transformation passes.

    PassManager maintains a sequence of Pass instances for different optimization
    strategies and executes them in order on a given Function or Program. It uses
    a pipeline model where each pass's output becomes the input to the next pass.
    """
```

#### Key Methods

**1. Getting a Configured Strategy**

```python
@classmethod
def get_strategy(cls, strategy: OptimizationStrategy = OptimizationStrategy.Default) -> "PassManager":
    """Get a PassManager configured for the specified strategy.

    Args:
        strategy: The optimization strategy to use (default: Default)

    Returns:
        A PassManager instance configured with the appropriate passes
    """
```

**2. Running Passes**

```python
def run_passes(self, input_ir: Union[core_ir.Function, core_ir.Program]):
    """Execute all passes in sequence on a Function or Program.

    Each pass's output becomes the input to the next pass.
    For Program inputs, all passes are applied to each function in the program.

    Args:
        input_ir: Input Function or Program to transform

    Returns:
        Transformed Function or Program after all passes have been applied
    """
```

**3. Getting Pass Names**

```python
def get_pass_names(self) -> List[str]:
    """Get the names of all passes in this manager.

    Returns:
        List of pass names assigned during registration
    """
```

### Strategy Configuration

Strategies are configured in the `_register_passes` class method:

```python
@classmethod
def _register_passes(cls):
    """Register all strategy Pass configurations.

    This method defines the static Pass pipeline for each optimization strategy.
    Each pass is registered with a unique name and a factory function.
    To add a new strategy or modify existing ones, edit this method.
    """
    cls._strategy_passes = {
        OptimizationStrategy.Default: [
            # No passes for Default (no optimization)
        ],
        OptimizationStrategy.Custom1: [
            # Custom optimization strategy 1
            ("IdentityPass_1", lambda: passes.IdentityPass()),
        ],
        OptimizationStrategy.Custom2: [
            # Custom optimization strategy 2
            ("IdentityPass_1", lambda: passes.IdentityPass()),
            ("IdentityPass_2", lambda: passes.IdentityPass()),
        ],
    }
```

## Usage Examples

### Function-Level Transformation

```python
from pypto import ir, DataType

# Create a simple function
span = ir.Span.unknown()
dtype = DataType.INT64
x = ir.Var("x", ir.ScalarType(dtype), span)
y = ir.Var("y", ir.ScalarType(dtype), span)
assign = ir.AssignStmt(x, y, span)
func = ir.Function("my_func", [x], [ir.ScalarType(dtype)], assign, span)

# Get a PassManager with Custom2 optimization strategy
pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.Custom2)

# Run passes on the function
result = pm.run_passes(func)

# Custom2 has 2 IdentityPasses, so the name becomes "my_func_identity_identity"
print(result.name)  # Output: my_func_identity_identity
```

### Program-Level Transformation

```python
from pypto import ir, DataType

span = ir.Span.unknown()
dtype = DataType.INT64

# Create first function
x1 = ir.Var("x", ir.ScalarType(dtype), span)
y1 = ir.Var("y", ir.ScalarType(dtype), span)
assign1 = ir.AssignStmt(x1, y1, span)
func1 = ir.Function("func1", [x1], [ir.ScalarType(dtype)], assign1, span)

# Create second function
x2 = ir.Var("x", ir.ScalarType(dtype), span)
y2 = ir.Var("y", ir.ScalarType(dtype), span)
assign2 = ir.AssignStmt(x2, y2, span)
func2 = ir.Function("func2", [x2], [ir.ScalarType(dtype)], assign2, span)

# Create program with both functions
program = ir.Program([func1, func2], "test_program", span)

# Get a PassManager with Custom1 optimization strategy
pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.Custom1)

# Run passes on the entire program
result = pm.run_passes(program)

# All functions in the program are transformed
assert isinstance(result, ir.Program)
assert result.name == "test_program"

# Get function names from result
func_names = [func.name for func in result.functions.values()]
print(func_names)  # Output: ['func1_identity', 'func2_identity']
```

### Shorthand Usage

```python
# One-liner execution
result = ir.PassManager.get_strategy(ir.OptimizationStrategy.Custom2).run_passes(func)
```

## Implementation Details

### Function Transformation Flow

When `run_passes` is called with a Function:

1. Initialize `current` to the input function
2. For each pass in the pipeline:
   - Call `pass.run(current)`
   - Assign the result back to `current`
3. Return the final transformed function

```python
# For Function input, apply passes in sequence
current = input_ir
for pass_instance in self.passes:
    current = pass_instance.run(current)
return current
```

### Program Transformation Flow

When `run_passes` is called with a Program:

1. Iterate over all functions in the program
2. For each function:
   - Apply all passes sequentially (same as Function flow)
   - Collect the transformed function
3. Create a new Program with the transformed functions

```python
if isinstance(input_ir, core_ir.Program):
    # Apply passes to each function in the program
    transformed_functions = []
    for global_var, func in input_ir.functions.items():
        transformed_func = func
        for pass_instance in self.passes:
            transformed_func = pass_instance.run(transformed_func)
        transformed_functions.append(transformed_func)

    # Create a new Program with the transformed functions
    return core_ir.Program(transformed_functions, input_ir.name, input_ir.span)
```

### Pass Registration Pattern

The PassManager uses a factory pattern for pass instantiation:

- Each strategy maps to a list of `(name, factory)` tuples
- Factories are lambda functions that create fresh pass instances
- This ensures each PassManager instance gets its own pass objects
- Multiple PassManager instances can coexist independently

## Testing

### Test Organization

Tests are located in `tests/ut/ir/transforms/test_pass_manager.py` and organized into classes:

1. **TestOptimizationStrategy** - Tests strategy enum values
2. **TestPassManagerBasics** - Tests PassManager creation and configuration
3. **TestPassManagerExecution** - Tests pass execution on Functions
4. **TestPassManagerMultipleInstances** - Tests multiple PassManager instances
5. **TestPassManagerWithProgram** - Tests pass execution on Programs

### Example Test: Custom2 Strategy on Function

```python
def test_run_with_custom2_strategy(self):
    """Test running PassManager with Custom2 strategy and verify pass execution."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)
    y = ir.Var("y", ir.ScalarType(dtype), span)
    assign = ir.AssignStmt(x, y, span)
    func = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)

    pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.Custom2)
    result = pm.run_passes(func)

    # Custom2 has 2 IdentityPasses, should append "_identity" twice
    assert result is not func
    assert result.name == "test_func_identity_identity"
```

### Example Test: Custom2 Strategy on Program

```python
def test_run_passes_on_program_with_custom2_strategy(self):
    """Test running PassManager with Custom2 strategy on a Program."""
    span = ir.Span.unknown()
    dtype = DataType.INT64

    # Create two functions
    x1 = ir.Var("x", ir.ScalarType(dtype), span)
    y1 = ir.Var("y", ir.ScalarType(dtype), span)
    assign1 = ir.AssignStmt(x1, y1, span)
    func1 = ir.Function("func1", [x1], [ir.ScalarType(dtype)], assign1, span)

    x2 = ir.Var("x", ir.ScalarType(dtype), span)
    y2 = ir.Var("y", ir.ScalarType(dtype), span)
    assign2 = ir.AssignStmt(x2, y2, span)
    func2 = ir.Function("func2", [x2], [ir.ScalarType(dtype)], assign2, span)

    # Create program
    program = ir.Program([func1, func2], "test_program", span)

    pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.Custom2)
    result = pm.run_passes(program)

    # Custom2 has 2 IdentityPasses, should append "_identity" twice to each function
    assert isinstance(result, ir.Program)
    assert result.name == "test_program"
    assert len(result.functions) == 2

    func_names = [func.name for func in result.functions.values()]
    assert "func1_identity_identity" in func_names
    assert "func2_identity_identity" in func_names
```

## Adding New Passes

To add a new pass to the system:

### 1. Implement the C++ Pass

Create header file in `include/pypto/passes/`:

```cpp
// your_pass.h
#include "pypto/ir/transform/base/pass.h"

namespace pypto {
namespace ir {

class YourPass : public Pass {
 public:
  FunctionPtr Run(const FunctionPtr& func) override;
};

}  // namespace ir
}  // namespace pypto
```

Create implementation in `src/passes/`:

```cpp
// your_pass.cpp
#include "pypto/passes/your_pass.h"

namespace pypto {
namespace ir {

FunctionPtr YourPass::Run(const FunctionPtr& func) {
  // Your transformation logic here
  return transformed_func;
}

}  // namespace ir
}  // namespace pypto
```

### 2. Add Python Bindings

Update `python/bindings/modules/pass.cpp`:

```cpp
#include "pypto/passes/your_pass.h"

void BindPass(nb::module_& m) {
  // ... existing bindings ...

  nb::class_<YourPass, Pass>(passes, "YourPass", "Description of your pass")
      .def(nb::init<>(), "Create your pass");
}
```

### 3. Register in PassManager

Update `python/pypto/ir/pass_manager.py`:

```python
@classmethod
def _register_passes(cls):
    cls._strategy_passes = {
        # ... existing strategies ...
        OptimizationStrategy.Custom2: [
            ("IdentityPass_1", lambda: passes.IdentityPass()),
            ("IdentityPass_2", lambda: passes.IdentityPass()),
            ("YourPass", lambda: passes.YourPass()),  # Add your pass
        ],
    }
```

### 4. Add Type Stubs

Update `python/pypto/pypto_core/passes.pyi`:

```python
class YourPass(Pass):
    """Description of your pass."""

    def __init__(self) -> None:
        """Create your pass."""
```

### 5. Add Tests

Add tests in `tests/ut/ir/transforms/test_pass_manager.py` or create a new test file for your specific pass.

## Design Rationale

### Why Immutable Transformations?

Passes return new IR nodes rather than modifying existing ones:
- **Thread Safety**: Multiple passes can analyze the same IR concurrently
- **Debugging**: Original IR is preserved for comparison
- **Undo/Rollback**: Easy to revert transformations
- **Functional Style**: Aligns with functional programming principles

### Why Strategy-Based Configuration?

Pre-configured optimization levels provide:
- **Ease of Use**: Users don't need to manually configure pass sequences
- **Consistency**: Same optimization level produces same pass pipeline
- **Maintainability**: Centralized configuration makes it easy to update strategies
- **Extensibility**: New strategies can be added without changing existing code

### Why Support Both Function and Program?

- **Function-Level**: Fine-grained control for individual function optimization
- **Program-Level**: Batch processing for entire programs, enables inter-procedural optimizations in the future
- **Unified API**: Single `run_passes` method handles both cases transparently

## Commit History

This Pass and PassManager system was implemented in three commits on the `WIP_pass_mngr` branch:

### 1. Initial Implementation (5a5b905)

**Commit**: `add PassManager`

Added the complete Pass infrastructure:
- C++ Pass base class and IdentityPass implementation
- Python bindings for passes
- PassManager with strategy-based configuration
- Comprehensive test suite for Function-level transformations

**Files Added**:
- `include/pypto/ir/transform/base/pass.h` - Pass base class
- `include/pypto/ir/transform/passes/identity_pass.h` - IdentityPass header
- `src/ir/transform/passes/identity_pass.cpp` - IdentityPass implementation
- `python/bindings/modules/pass.cpp` - Python bindings
- `python/pypto/ir/pass_manager.py` - PassManager implementation
- `tests/ut/ir/test_pass_manager.py` - Test suite

### 2. Refactoring and Cleanup (dc2e416)

**Commit**: `remove unused method and pre-commit update`

Refined the implementation:
- Removed unused methods from Pass base class
- Added type stubs (`passes.pyi`) for better IDE support
- Moved tests to dedicated `tests/ut/ir/transforms/` directory
- Updated Python exports and imports

**Key Changes**:
- Created `python/pypto/pypto_core/passes.pyi` for type hints
- Moved `tests/ut/ir/test_pass_manager.py` → `tests/ut/ir/transforms/test_pass_manager.py`
- Cleaned up Pass interface

### 3. Program Support (e2fd396)

**Commit**: `PassManager support Program as input`

Extended PassManager to handle Program transformations:
- Renamed `run()` method to `run_passes()` for clarity
- Added Program support to `run_passes()` method
- Updated all documentation and examples
- Added comprehensive tests for Program transformations

**Key Changes**:
- Modified `run_passes()` to accept `Union[Function, Program]`
- Implemented Program transformation logic (applies passes to all functions)
- Added 5 new test cases for Program-level transformations
- Updated all existing tests to use `run_passes()` instead of `run()`

## Future Enhancements

Potential improvements to the Pass system:

1. **Pass Dependencies**: Declare dependencies between passes
2. **Pass Analysis**: Add analysis passes that don't transform IR but collect information
3. **Pass Metrics**: Track execution time and transformation statistics
4. **Pass Verification**: Optional verification passes to check IR validity
5. **Inter-procedural Passes**: Passes that optimize across function boundaries
6. **Program-Level Passes**: Dedicated passes that operate on entire Programs (not just per-function)
7. **Pass Configuration**: Allow passes to accept configuration parameters
8. **Parallel Execution**: Run independent passes in parallel

## Summary

The Pass and PassManager system provides:
- ✅ **Extensible Framework**: Easy to add new transformation passes
- ✅ **Strategy-Based Optimization**: Pre-configured optimization levels (Default/Custom1/Custom2)
- ✅ **Dual-Level Support**: Works with both Functions and Programs
- ✅ **Clean API**: Simple Python interface with type hints
- ✅ **Well-Tested**: Comprehensive test coverage for all features
- ✅ **Immutable Transformations**: Safe, functional-style IR transformations

This infrastructure provides the foundation for building sophisticated optimization pipelines in PyPTO.
