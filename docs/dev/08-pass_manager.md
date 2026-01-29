# Pass and PassManager

The Pass and PassManager system provides a framework for organizing and executing IR transformation passes on Programs. This system enables optimization pipelines with different strategies (Default/Custom1/Custom2/XPlatform) and supports Program-level transformations.

## Overview

The Pass system consists of three main components:

1. **Pass (C++)** - Standalone class for IR transformations that operate on Programs
2. **PassManager (Python)** - Manages sequences of passes and execution strategies
3. **Factory Functions** - Functions that create specific passes (e.g., `pass::Identity()`, `pass::InitMemRef()`)

### Key Features

- **Strategy-based Pipeline**: Pre-configured optimization levels (Default/Custom1/Custom2/XPlatform)
- **Immutable Transformations**: Passes return new IR nodes rather than modifying in place
- **Program-Only Interface**: All passes operate at the Program level (Program → Program transformations)
- **Pipeline Composition**: Multiple passes execute sequentially, with each pass's output feeding into the next
- **Factory Functions**: Passes are created via factory functions (e.g., `pass::Identity()`, `pass::InitMemRef()`)
- **Opaque Pass Objects**: Pass implementation details are hidden; only factory functions and execution methods are exposed
- **Unified Header**: All pass declarations and factory functions are in a single header file (`passes.h`)

## C++ Pass Infrastructure

### Pass Base Class

The `Pass` class is a standalone class (not inheriting from IRMutator) for all IR transformations. It uses a pimpl pattern to hide implementation details. **All pass declarations and factory functions are in a single header file** - there are no standalone header files for individual passes.

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
namespace pypto {
namespace ir {

/**
 * @brief Base class for IR transformation passes
 *
 * Pass is a standalone class that provides Program-level transformations.
 * Each pass operates on a Program and returns a transformed Program.
 * All passes transform Program → Program (not Function → Function).
 * Passes maintain immutability - they return new ProgramPtr instances rather than modifying in place.
 * Uses pimpl pattern to hide implementation details.
 */
class Pass {
 public:
  Pass();
  explicit Pass(std::shared_ptr<PassImpl> impl);
  ~Pass();

  /**
   * @brief Execute the pass on a program (primary API)
   *
   * This is the main entry point for pass execution using function call operator.
   *
   * @param program Input program to transform
   * @return Transformed program (may be the same pointer if no changes were made)
   */
  ProgramPtr operator()(const ProgramPtr& program) const;
};

// Factory functions for built-in passes
namespace pass {

/**
 * @brief Create an identity pass for testing
 *
 * Appends "_identity" to function names to verify pass execution.
 */
Pass Identity();

/**
 * @brief Create an init memref pass
 *
 * Initializes MemRef for all variables in functions.
 */
Pass InitMemRef();

/**
 * @brief Create a basic memory reuse pass
 *
 * Uses dependency analysis to identify memory reuse opportunities.
 */
Pass BasicMemoryReuse();

/**
 * @brief Create an insert sync pass
 *
 * Analyzes data dependencies and inserts synchronization operations.
 */
Pass InsertSync();

/**
 * @brief Create an add alloc pass
 *
 * Creates alloc operations for each unique MemRef.
 */
Pass AddAlloc();

}  // namespace pass
}  // namespace ir
}  // namespace pypto
```

### Pass Implementation Structure

Passes are implemented using the pimpl pattern with internal `PassImpl` classes. **There are no standalone header files for individual passes** - all pass declarations are in `passes.h`, and implementations are in `src/ir/transforms/`.

**Implementation Pattern**:

1. **Factory Function** (in `passes.h`): Public API for creating the pass
2. **PassImpl Class** (in `.cpp` file): Internal implementation inheriting from `PassImpl`
3. **Operator()** (in `PassImpl`): Transforms Program → Program

**Example: Identity Pass** (in `src/ir/transforms/identity_pass.cpp`)

```cpp
namespace pypto {
namespace ir {

// Internal implementation class
class IdentityPassImpl : public PassImpl {
 public:
  ProgramPtr operator()(const ProgramPtr& program) override {
    // Transform each function in the program
    std::vector<FunctionPtr> transformed_functions;
    for (const auto& [name, func] : program->functions_) {
      // Append "_identity" to function name for testing
      std::string new_name = func->name_ + "_identity";
      auto new_func = std::make_shared<const Function>(
          new_name, func->params_, func->return_types_, func->body_, func->span_);
      transformed_functions.push_back(new_func);
    }
    return std::make_shared<const Program>(
        transformed_functions, program->name_, program->span_);
  }
};

// Factory function exposed in passes.h
namespace pass {
Pass Identity() {
  return Pass(std::make_shared<IdentityPassImpl>());
}
}  // namespace pass

}  // namespace ir
}  // namespace pypto
```

**Key Points**:
- All passes operate on **Program → Program** (not Function → Function)
- Implementation details are hidden in `.cpp` files
- Only factory functions are exposed in `passes.h`
- Passes transform entire programs by applying transformations to all functions

### Python Bindings

Passes are exposed to Python through nanobind bindings as opaque objects with factory functions.

**File**: `python/bindings/modules/passes.cpp`

```cpp
void BindPass(nb::module_& m) {
  // Create a new 'passes' submodule
  nb::module_ passes = m.def_submodule("passes", "IR transformation passes");

  // Pass class - opaque to Python, only expose call operator
  nb::class_<Pass>(passes, "Pass", "Opaque pass object. Do not instantiate directly - use factory functions.")
      .def("__call__", &Pass::operator(), nb::arg("program"), "Execute pass on program");

  // Factory functions with snake_case names
  passes.def("identity", &pass::Identity, "Create an identity pass for testing");
  passes.def("init_mem_ref", &pass::InitMemRef, "Create an init memref pass");
  passes.def("basic_memory_reuse", &pass::BasicMemoryReuse, "Create a basic memory reuse pass");
  passes.def("insert_sync", &pass::InsertSync, "Create an insert sync pass");
  passes.def("add_alloc", &pass::AddAlloc, "Create an add alloc pass");
}
```

The bindings create a `pypto.pypto_core.passes` module with:
- `Pass` class with a `__call__(program)` method for execution
- Factory functions: `identity()`, `init_mem_ref()`, `basic_memory_reuse()`, `insert_sync()`, `add_alloc()`
- All passes operate on Program → Program transformations

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
    strategies and executes them in order on a given Program. It uses
    a pipeline model where each pass's output becomes the input to the next passes.
    All passes operate on Program → Program transformations.
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
def run_passes(self, program: core_ir.Program) -> core_ir.Program:
    """Execute all passes in sequence on a Program.

    Each pass's output becomes the input to the next passes.
    All passes transform Program → Program.

    Args:
        program: Input Program to transform

    Returns:
        Transformed Program after all passes have been applied
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
            ("IdentityPass_1", lambda: passes.identity()),
        ],
        OptimizationStrategy.Custom2: [
            # Custom optimization strategy 2
            ("IdentityPass_1", lambda: passes.identity()),
            ("IdentityPass_2", lambda: passes.identity()),
        ],
        OptimizationStrategy.XPlatform: [
            ("InitMemRef", lambda: passes.init_mem_ref()),
            ("MemoryReuse", lambda: passes.basic_memory_reuse()),
            ("AddAlloc", lambda: passes.add_alloc()),
        ],
    }
```

## Usage Examples

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
result = ir.PassManager.get_strategy(ir.OptimizationStrategy.Custom2).run_passes(program)
```

## Implementation Details

### Program Transformation Flow

All passes operate on **Program → Program** transformations. When `run_passes` is called:

1. Initialize `current` to the input program
2. For each pass in the pipeline:
   - Call `pass(current)` - the pass transforms the entire program
   - Assign the result back to `current`
3. Return the final transformed program

```python
def run_passes(self, program: core_ir.Program) -> core_ir.Program:
    """Execute all passes in sequence on a Program."""
    current = program
    for pass_instance in self.passes:
        current = pass_instance(current)  # Program → Program transformation
    return current
```

**Key Points**:
- Each pass receives a `Program` and returns a transformed `Program`
- Passes internally apply transformations to all functions in the program
- The pipeline composes passes sequentially: `Pass3(Pass2(Pass1(program)))`
- All transformations maintain immutability - new IR nodes are created

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
3. **TestPassManagerExecution** - Tests pass execution on Programs
4. **TestPassManagerMultipleInstances** - Tests multiple PassManager instances

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

### 1. Declare Factory Function in Header

Update `include/pypto/ir/transforms/passes.h` to add your factory function declaration:

```cpp
namespace pypto {
namespace ir {
namespace pass {

// ... existing factory functions ...

/**
 * @brief Create your new pass
 *
 * Description of what your pass does.
 */
Pass YourNewPass();

}  // namespace pass
}  // namespace ir
}  // namespace pypto
```

### 2. Implement the Pass

Create implementation in `src/ir/transforms/your_new_pass.cpp`:

```cpp
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/pass_impl.h"

namespace pypto {
namespace ir {

// Internal implementation class
class YourNewPassImpl : public PassImpl {
 public:
  ProgramPtr operator()(const ProgramPtr& program) override {
    // Transform each function in the program
    std::vector<FunctionPtr> transformed_functions;
    for (const auto& [name, func] : program->functions_) {
      // Your transformation logic here
      auto transformed_func = TransformFunction(func);
      transformed_functions.push_back(transformed_func);
    }
    return std::make_shared<const Program>(
        transformed_functions, program->name_, program->span_);
  }

  std::string GetName() const override { return "YourNewPass"; }

 private:
  FunctionPtr TransformFunction(const FunctionPtr& func) {
    // Implementation details
    return func;  // Replace with actual transformation
  }
};

// Factory function
namespace pass {
Pass YourNewPass() {
  return Pass(std::make_shared<YourNewPassImpl>());
}
}  // namespace pass

}  // namespace ir
}  // namespace pypto
```

### 3. Add Python Bindings

Update `python/bindings/modules/passes.cpp`:

```cpp
void BindPass(nb::module_& m) {
  // ... existing bindings ...

  passes.def("your_new_pass", &pass::YourNewPass,
             "Create your new pass\n\n"
             "Description of what your pass does.");
}
```

### 4. Register in PassManager

Update `python/pypto/ir/pass_manager.py`:

```python
@classmethod
def _register_passes(cls):
    cls._strategy_passes = {
        # ... existing strategies ...
        OptimizationStrategy.Custom2: [
            ("IdentityPass_1", lambda: passes.identity()),
            ("IdentityPass_2", lambda: passes.identity()),
            ("YourNewPass", lambda: passes.your_new_pass()),  # Add your pass
        ],
    }
```

### 5. Add Type Stubs

Update `python/pypto/pypto_core/passes.pyi`:

```python
def your_new_pass() -> Pass:
    """Create your new pass.

    Description of what your pass does.
    """
```

### 6. Add Tests

Add tests in `tests/ut/ir/transforms/test_your_new_pass.py`:

```python
def test_your_new_pass():
    """Test your new pass."""
    # Create a program
    program = create_test_program()

    # Create and run the pass
    pass_obj = passes.your_new_pass()
    result = pass_obj(program)

    # Verify the transformation
    assert isinstance(result, ir.Program)
    # Add specific assertions
```

**Important Notes**:
- **No standalone header files** - all declarations go in `passes.h`
- All passes must be **Program → Program** transformations
- Use `PassImpl` base class for implementation
- Expose only factory functions to Python, not implementation classes

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

### Why Program-Only Interface?

All passes operate on Program → Program transformations:
- **Consistency**: Uniform interface for all passes simplifies the API
- **Flexibility**: Passes can apply per-function transformations internally or program-wide transformations
- **Future-Proof**: Enables inter-procedural optimizations and whole-program analysis
- **Simpler Mental Model**: One transformation type to understand

### Why No Standalone Headers for Each Pass?

All pass declarations are in a single `passes.h` file:
- **Reduced Header Bloat**: Single header for all pass declarations
- **Cleaner Organization**: Factory functions clearly show what passes are available
- **Opaque Implementation**: Implementation details hidden in `.cpp` files via pimpl pattern
- **Easier Discovery**: Users can see all available passes in one place

## Commit History

This Pass and PassManager system was implemented through multiple iterations on the `pass_refactor` branch:

### Key Refactoring Changes

**Current State** (as of `pass_refactor` branch):

**Architecture Changes**:
- All pass declarations unified in single header `include/pypto/ir/transforms/passes.h`
- No standalone header files for individual passes
- All passes operate on Program → Program transformations (not Function → Function)
- Factory functions (e.g., `pass::Identity()`, `pass::InitMemRef()`) create passes
- Opaque Pass objects exposed to Python via `__call__` operator

**Key Files**:
- `include/pypto/ir/transforms/passes.h` - Unified pass declarations and factory functions
- `src/ir/transforms/pass_impl.h` - Internal PassImpl base class
- `src/ir/transforms/*.cpp` - Individual pass implementations
- `python/bindings/modules/passes.cpp` - Python bindings with factory functions
- `python/pypto/ir/pass_manager.py` - PassManager implementation
- `python/pypto/pypto_core/passes.pyi` - Type stubs for Python

**Pass Implementations**:
- Identity pass (for testing)
- InitMemRef pass (memory space initialization)
- BasicMemoryReuse pass (lifetime-based memory reuse)
- InsertSync pass (synchronization insertion)
- AddAlloc pass (allocation operation insertion)

**Design Principles**:
- Pimpl pattern to hide implementation details
- Immutable transformations (return new IR nodes)
- Strategy-based pipeline configuration
- Unified Program-level interface

## Future Enhancements

Potential improvements to the Pass system:

1. **Pass Dependencies**: Declare dependencies between passes for automatic ordering
2. **Pass Analysis**: Add analysis passes that don't transform IR but collect information (e.g., liveness analysis)
3. **Pass Metrics**: Track execution time and transformation statistics
4. **Pass Verification**: Optional verification passes to check IR validity after transformations
5. **Inter-procedural Passes**: Passes that optimize across function boundaries (e.g., inlining, global value numbering)
6. **Pass Configuration**: Allow passes to accept configuration parameters
7. **Parallel Execution**: Run independent passes in parallel when safe
8. **Pass Caching**: Cache pass results to avoid redundant computation

## Summary

The Pass and PassManager system provides:
- ✅ **Extensible Framework**: Easy to add new transformation passes via factory functions
- ✅ **Strategy-Based Optimization**: Pre-configured optimization levels (Default/Custom1/Custom2/XPlatform)
- ✅ **Unified Interface**: All passes operate on Program → Program transformations
- ✅ **Clean API**: Simple Python interface with opaque pass objects and factory functions
- ✅ **Well-Tested**: Comprehensive test coverage for all features
- ✅ **Immutable Transformations**: Safe, functional-style IR transformations
- ✅ **Organized Structure**: Single header file (`passes.h`) with all pass declarations

This infrastructure provides the foundation for building sophisticated optimization pipelines in PyPTO.
