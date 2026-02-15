# Pass and PassManager

Framework for organizing and executing IR transformation passes on Programs with strategy-based optimization pipelines (Default/PTOAS).

## Overview

| Component | Description |
| --------- | ----------- |
| **Pass (C++)** | Standalone class for Program → Program transformations |
| **PassManager (Python)** | Manages pass sequences and execution strategies |
| **Factory Functions** | Create passes (e.g., `pass::InitMemRef()`, `pass::BasicMemoryReuse()`) |

### Key Features

- **Program-Only Interface**: All passes transform Program → Program
- **Immutable Transformations**: Return new IR nodes, don't modify in place
- **Strategy-based Pipelines**: Pre-configured optimization levels
- **Factory Pattern**: Passes created via factory functions, implementation details hidden
- **Unified Header**: All declarations in `include/pypto/ir/transforms/passes.h`

## C++ Pass Infrastructure

### Pass Base Class

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
class Pass {
 public:
  ProgramPtr operator()(const ProgramPtr& program) const;  // Execute pass
};

// Factory functions for built-in passes
namespace pass {
  Pass ConvertToSSA();           // Convert to SSA form
  Pass FlattenCallExpr();        // Flatten nested call expressions
  Pass RunVerifier(...);         // Run IR verification
  Pass InitMemRef();             // Initializes MemRef for variables
  Pass BasicMemoryReuse();       // Dependency-based memory reuse
  Pass InsertSync();             // Inserts synchronization operations
  Pass AddAlloc();               // Creates alloc operations for MemRefs
  Pass OutlineIncoreScopes();    // Outline InCore scopes to functions
  Pass NormalizeStmtStructure(); // Normalize statement structure
  Pass FlattenSingleStmt();      // Flatten single-statement blocks
}
```

**Key points**: Pimpl pattern hides implementation; all declarations in single header; Program → Program transformations only.

### Pass Implementation Patterns

| Pattern | Use When | Implementation |
| ------- | -------- | -------------- |
| **Simple Function-Level** (90% of cases) | Per-function transformations | Use `CreateFunctionPass()` helper |
| **Complex Custom** | State, helpers, or program-level analysis | Inherit from `PassImpl` |

**Pattern 1: Simple** (e.g. `src/ir/transforms/init_memref.cpp` uses helpers; most passes use `CreateFunctionPass` with a lambda.)

**Pattern 2: Complex** (with state)

```cpp
namespace {
class ComplexPassImpl : public PassImpl {
 public:
  ProgramPtr operator()(const ProgramPtr& program) override {
    for (const auto& [name, func] : program->functions_)
      state_ += ComputeSomething(func);
    return program;
  }
  std::string GetName() const override { return "ComplexPass"; }
 private:
  int state_ = 0;
};
}
namespace pass {
Pass ComplexPass() { return Pass(std::make_shared<ComplexPassImpl>()); }
}
```

### Python Bindings

**File**: `python/bindings/modules/passes.cpp`

```cpp
void BindPass(nb::module_& m) {
  nb::module_ passes = m.def_submodule("passes", "IR transformation passes");

  // Opaque pass object
  nb::class_<Pass>(passes, "Pass")
      .def("__call__", &Pass::operator(), nb::arg("program"));

  // Factory functions (snake_case)
  passes.def("convert_to_ssa", &pass::ConvertToSSA);
  passes.def("flatten_call_expr", &pass::FlattenCallExpr);
  passes.def("run_verifier", &pass::RunVerifier);
  passes.def("init_mem_ref", &pass::InitMemRef);
  passes.def("basic_memory_reuse", &pass::BasicMemoryReuse);
  passes.def("insert_sync", &pass::InsertSync);
  passes.def("add_alloc", &pass::AddAlloc);
  passes.def("outline_incore_scopes", &pass::OutlineIncoreScopes);
  passes.def("normalize_stmt_structure", &pass::NormalizeStmtStructure);
  passes.def("flatten_single_stmt", &pass::FlattenSingleStmt);
}
```

Creates `pypto.pypto_core.passes` module with opaque `Pass` class and factory functions.

## Python PassManager

**File**: `python/pypto/ir/pass_manager.py`

### Optimization Strategies

```python
class OptimizationStrategy(Enum):
    Default = "Default"      # Full pipeline with SSA conversion, verification, and all optimizations
    PTOAS = "PTOAS"         # PTO assembly: Memory management only (no SSA, scheduling, or sync)
```

### PassManager API

| Method | Description |
| ------ | ----------- |
| `get_strategy(strategy)` | Get PassManager configured for strategy |
| `run_passes(program, dump_ir=False, output_dir=None, prefix='pl')` | Execute all passes sequentially on Program; optionally dump IR |
| `get_pass_names()` | Get names of all passes in manager |

### Strategy Configuration

Strategies configured in `_register_passes`:

```python
@classmethod
def _register_passes(cls):
    cls._strategy_passes = {
        OptimizationStrategy.Default: [
            ("ConvertToSSA", lambda: passes.convert_to_ssa()),
            ("FlattenCallExpr", lambda: passes.flatten_call_expr()),
            ("RunVerifier", lambda: passes.run_verifier()),
            ("InitMemRef", lambda: passes.init_mem_ref()),
            ("MemoryReuse", lambda: passes.basic_memory_reuse()),
            ("InsertSync", lambda: passes.insert_sync()),
            ("AddAlloc", lambda: passes.add_alloc()),
        ],
        OptimizationStrategy.PTOAS: [
            ("InitMemRef", lambda: passes.init_mem_ref()),
            ("MemoryReuse", lambda: passes.basic_memory_reuse()),
            ("AddAlloc", lambda: passes.add_alloc()),
        ],
    }
```

## Usage Examples

```python
from pypto import ir, DataType

# Create program with multiple functions
span = ir.Span.unknown()
dtype = DataType.INT64
x1, y1 = ir.Var("x", ir.ScalarType(dtype), span), ir.Var("y", ir.ScalarType(dtype), span)
func1 = ir.Function("func1", [x1], [ir.ScalarType(dtype)], ir.AssignStmt(x1, y1, span), span)
x2, y2 = ir.Var("x", ir.ScalarType(dtype), span), ir.Var("y", ir.ScalarType(dtype), span)
func2 = ir.Function("func2", [x2], [ir.ScalarType(dtype)], ir.AssignStmt(x2, y2, span), span)
program = ir.Program([func1, func2], "test_program", span)

# Run passes with PTOAS strategy
pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.PTOAS)
result = pm.run_passes(program)
# Result has same function names; passes apply InitMemRef, MemoryReuse, AddAlloc

# One-liner shorthand
result = ir.PassManager.get_strategy(ir.OptimizationStrategy.PTOAS).run_passes(program)
```

## Implementation Details

### Program Transformation Flow

```python
def run_passes(
    self,
    input_ir: core_ir.Program,
    dump_ir: bool = False,
    output_dir: Optional[str] = None,
    prefix: str = "pl",
) -> core_ir.Program:
    current = input_ir
    for pass_instance in self.passes:
        current = pass_instance(current)  # Program → Program
        if dump_ir:
            # Optionally dump IR after each pass to output_dir
            dump_to_file(current, output_dir, prefix)
    return current
```

**Parameters**:

- `input_ir`: Input Program to transform
- `dump_ir`: Whether to dump IR after each pass (default: False)
- `output_dir`: Directory to dump IR files (required when dump_ir=True)
- `prefix`: Module prefix for python_print (default: 'pl')

Pipeline composition: `Pass3(Pass2(Pass1(program)))` - each pass receives and returns a Program.

### Pass Registration Pattern

- Each strategy maps to `(name, factory)` tuples
- Factories are lambdas creating fresh pass instances
- Enables independent PassManager instances

## Testing

**Location**: `tests/ut/ir/transforms/test_pass_manager.py`

**Example**: PTOAS runs InitMemRef, MemoryReuse, AddAlloc; function names unchanged:

```python
def test_run_passes_on_program_with_ptoa_strategy(self):
    program = ir.Program([func1, func2], "test_program", span)
    pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.PTOAS)
    result = pm.run_passes(program)
    func_names = [func.name for func in result.functions.values()]
    assert "func1" in func_names
    assert "func2" in func_names
```

## Adding New Passes

1. **Declare in `passes.h`**: `Pass YourNewPass();`

2. **Implement** (`src/ir/transforms/your_new_pass.cpp`):

   ```cpp
   // Simple (recommended)
   namespace pass {
   Pass YourNewPass() {
     return CreateFunctionPass([](const FunctionPtr& func) {
       // Transform function
       return func;
     }, "YourNewPass");
   }
   }

   // Complex with state
   namespace {
   class YourNewPassImpl : public PassImpl {
    public:
     ProgramPtr operator()(const ProgramPtr& program) override { /* ... */ }
     std::string GetName() const override { return "YourNewPass"; }
    private:
     int state_ = 0;
   };
   }
   namespace pass {
   Pass YourNewPass() { return Pass(std::make_shared<YourNewPassImpl>()); }
   }
   ```

3. **Python binding** (`python/bindings/modules/passes.cpp`):

   ```cpp
   passes.def("your_new_pass", &pass::YourNewPass, "Description");
   ```

4. **Register in PassManager** (`python/pypto/ir/pass_manager.py`):

   ```python
   ("YourNewPass", lambda: passes.your_new_pass()),
   ```

5. **Type stub** (`python/pypto/pypto_core/passes.pyi`):

   ```python
   def your_new_pass() -> Pass: """Description."""
   ```

6. **Test** (`tests/ut/ir/transforms/test_your_new_pass.py`)

## Design Rationale

| Design Choice | Rationale |
| ------------- | --------- |
| **Immutable Transformations** | Thread safety, debugging (preserve original IR), easy rollback, functional style |
| **Strategy-Based Config** | Ease of use, consistency, centralized maintenance, extensibility |
| **Program-Only Interface** | Uniform API, enables inter-procedural optimizations, simpler mental model |
| **Single Header (`passes.h`)** | Reduced bloat, clear discovery, opaque implementation via pimpl |

## Summary

The Pass and PassManager system provides:

- **Extensible Framework**: Easy to add passes via factory functions
- **Strategy-Based Optimization**: Pre-configured levels (Default/PTOAS)
- **Unified Interface**: All passes transform Program → Program
- **Clean API**: Opaque pass objects with factory functions
- **Well-Tested**: Comprehensive test coverage
- **Immutable Transformations**: Safe, functional-style IR transformations
- **Organized Structure**: Single header file with all declarations

This infrastructure provides the foundation for building sophisticated optimization pipelines in PyPTO.
