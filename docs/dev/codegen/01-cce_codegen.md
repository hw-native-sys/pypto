# PyPTO Code Generation Module

## Overview

The PyPTO code generation (codegen) module converts optimized PyPTO IR into executable C++ code using the pto-isa instruction set.

**Pipeline:** `IR → PassManager → CCECodegen → Compiler`

**Key Design Principles:**
- **Standalone Component**: Not a Pass. Passes transform IR→IR, codegen transforms IR→String
- **Visitor-Based**: Extends `IRVisitor` for IR tree traversal
- **Immutable**: Input IR never modified
- **Modular**: Separate concerns (emission, mapping, type conversion)

## Architecture

### Component Structure

| Component | Purpose | Location |
|-----------|---------|----------|
| `CCECodegen` | Main orchestrator, extends IRVisitor | [cce_codegen.h](../../include/pypto/codegen/cce_codegen.h) |
| `CodeEmitter` | Structured output with indentation | [code_emitter.h](../../include/pypto/codegen/code_emitter.h) |
| `CodeContext` | Variable name mapping and pointer tracking | [code_context.h](../../include/pypto/codegen/code_context.h) |
| `TypeConverter` | IR types → pto-isa C++ types | [type_converter.h](../../include/pypto/codegen/type_converter.h) |
| `ISAMapper` | IR operations → pto-isa instructions | [isa_mapper.h](../../include/pypto/codegen/isa_mapper.h) |

## Core Components

### CodeEmitter

Manages structured code output with proper indentation.

**Key Methods:**
- `EmitLine(line)` - Emit line with indentation
- `IncreaseIndent()` / `DecreaseIndent()` - Manage indent level
- `GetCode()` - Retrieve accumulated code

### CodeContext

Tracks variable name mappings and pointer associations.

**Key Features:**
- Maps IR variables to C++ names via `RegisterVar(var, cpp_name)`
- Sanitizes IR names for C++ compatibility via `SanitizeName(var)`
- Tracks tensor→pointer mappings via `RegisterPointer(tensor_var, ptr_name)`
- Enforces one-time registration (prevents duplicate declarations)

**Naming Convention:**
- Function parameters: `input_a` → `input_aGlobal` (GlobalTensor), `input_a` (raw pointer)
- Tile variables: Sanitized IR name on first assignment
- Regular variables: Sanitized name on first assignment

**Pointer Tracking:**
GlobalTensor variables wrap raw pointers. For address arithmetic (e.g., `output + offset`), we need the raw pointer name. CodeContext maintains this mapping and supports pointer inheritance through ForStmt iter_args and IfStmt return_vars.

### TypeConverter

Converts PyPTO IR types to pto-isa C++ type strings.

**Conversion Tables:**

| PyPTO DataType | C++ Type | PyPTO MemorySpace | Annotation |
|----------------|----------|-------------------|------------|
| FP32 | `float` | DDR | `__gm__` |
| FP16 | `half` | UB/L0A/L0B/L0C | (none) |
| INT32 | `int32_t` | | |
| INT64 | `int64_t` | | |
| BOOL | `bool` | | |
| BF16 | `bfloat16` | | |

**Shape/Stride:** Padded to 5D with leading 1s, row-major layout.

### ISAMapper

Maps PyPTO IR operations to pto-isa instructions.

| IR Operation | pto-isa | Category | Notes |
|--------------|---------|----------|-------|
| `block.load` | `TLOAD` | Memory | DDR→UB |
| `block.store` | `TSTORE` | Memory | UB→DDR |
| `block.add` / `sub` / `mul` / `div` | `TADD` / `TSUB` / `TMUL` / `TDIV` | Binary | Tile+Tile |
| `block.adds` / `subs` / `muls` / `divs` | `TADDS` / `TSUBS` / `TMULS` / `TDIVS` | Binary | Tile+Scalar |
| `block.sqrt` | `TSQRT` | Unary | Element-wise |
| `block.sum` (axis=0/1) | `TCOLSUM` / `TROWSUM` | Reduction | Axis-dependent |
| `system.sync_src` | `set_flag` | Sync | Set flag |
| `system.sync_dst` | `wait_flag` | Sync | Wait flag |
| `system.bar_v/m/all` | `pipe_barrier` | Sync | Barrier |

### CCECodegen

Main class orchestrating all components. Extends `IRVisitor`.

**Entry Point:** `std::string Generate(const FunctionPtr& func)`

## Code Generation Flow

### Three-Phase Generation

**Phase 1: Prologue**
1. Function signature with `__aicore__` and `__attribute__((always_inline))`
2. Argument unpacking from `int64_t* args` array
3. GlobalTensor type definitions and instances
4. Tile type definitions with TASSIGN memory allocation (if MemRef present)

**TileCollector** traverses function body to discover tile-typed variables from AssignStmt nodes. IfStmt return_vars are NOT collected; they're declared before the if statement.

**Phase 2: Body**
- Block operations (TLOAD, TADD, TSTORE, etc.)
- Synchronization (set_flag, wait_flag, pipe_barrier)
- Control flow (loops, conditionals)
- Variable assignments

**Phase 3: Epilogue**
- Closing brace
- Optional cleanup

### Visitor Methods

```cpp
void VisitExpr_(const CallPtr& op);         // Operations
void VisitStmt_(const AssignStmtPtr& op);   // Assignments
void VisitStmt_(const EvalStmtPtr& op);     // Sync operations
void VisitStmt_(const SeqStmtsPtr& op);     // Statement sequences
void VisitStmt_(const ForStmtPtr& op);      // Loops
void VisitStmt_(const IfStmtPtr& op);       // Conditionals
void VisitStmt_(const YieldStmtPtr& op);    // Yield values
```

## Usage Example

**Python API** (unified in codegen module: `codegen.PTOCodegen()`, `codegen.CCECodegen()`):
```python
from pypto.pypto_core import codegen
cg = codegen.CCECodegen()
cpp_code = cg.Generate(func)
```

**C++ API:**
```cpp
#include "pypto/codegen/cce_codegen.h"

FunctionPtr func = /* from IR */;
codegen::CCECodegen generator;
std::string cpp_code = generator.Generate(func);
```

**Input IR (conceptual):**
```python
def simple_add(x: Tensor([128, 64], FP32), y: Tensor([128, 64], FP32)):
    tile_x = block.load(x, [0, 0], [128, 64])
    tile_y = block.load(y, [0, 0], [128, 64])
    system.sync_src(PIPE_MTE2, PIPE_V, EVENT_ID0)
    system.sync_dst(PIPE_MTE2, PIPE_V, EVENT_ID0)
    tile_z = block.add(tile_x, tile_y)
    system.sync_src(PIPE_V, PIPE_MTE3, EVENT_ID0)
    system.sync_dst(PIPE_V, PIPE_MTE3, EVENT_ID0)
    result = block.store(tile_z, [0, 0], [128, 64], output)
```

**Generated C++ (simplified):**
```cpp
__aicore__ __attribute__((always_inline)) void runSimpleAdd(__gm__ int64_t* args) {
    // Unpack arguments
    __gm__ float* x = reinterpret_cast<__gm__ float*>(args[0]);
    __gm__ float* y = reinterpret_cast<__gm__ float*>(args[1]);
    __gm__ float* output = reinterpret_cast<__gm__ float*>(args[2]);

    // GlobalTensor declarations (types omitted for brevity)
    xGlobalType xGlobal(x);
    yGlobalType yGlobal(y);
    outputGlobalType outputGlobal(output);

    // Tile declarations
    tile_xType tile_x(128, 64);
    TASSIGN(tile_x, 0x0);
    tile_yType tile_y(128, 64);
    TASSIGN(tile_y, 0x10000);
    tile_zType tile_z(128, 64);
    TASSIGN(tile_z, 0x20000);

    // Function body
    TLOAD(tile_x, xGlobal);
    TLOAD(tile_y, yGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADD(tile_z, tile_x, tile_y);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(outputGlobal, tile_z);
}
```

## Implementation Details

### Memory Address Management

UB memory addresses come from IR metadata via TileType's MemRef field:
- Transformation passes set `TileType::memref_::addr_` (ConstInt expressions)
- Codegen extracts addresses and formats as hex (e.g., `0x0`, `0x10000`)
- TASSIGN instructions bind tiles to specific UB addresses
- If no MemRef, TASSIGN is skipped (future AllocOp handles allocation)

**Pointer Tracking:** CodeContext maintains tensor→pointer mappings for correct address arithmetic in TASSIGN instructions. Supports inheritance through control flow.

### Dual-Mode Expression Pattern

Expression visitors operate in two modes:

**Mode 1: Statement-Emitting (Call Expressions)**
- Input: `current_target_var_` contains assignment target
- Behavior: Emit complete instruction statements
- Output: Clear `current_expr_value_`
- Example: `tile_z = block.add(tile_x, tile_y)` → `TADD(tile_z, tile_x, tile_y);`

**Mode 2: Value-Returning (Scalar Expressions)**
- Input: Expression tree
- Behavior: Generate inline C++ code
- Output: Set `current_expr_value_` with inline code
- Example: `i * 128 + j` → `"(i * 128 + j)"`

### Synchronization Strategy

Synchronization is explicit in the IR:
- Transformation passes insert `system.sync_src/dst` operations
- Codegen translates directly to `set_flag/wait_flag`
- No automatic synchronization inference

**Typical Pattern:**
```
Load:  TLOAD → set_flag → wait_flag
Compute: TADD → set_flag → wait_flag
Store: TSTORE
```

## Control Flow Generation

### ForStmt (Loops)

**Simple loop:**
```cpp
for (int64_t i = start; i < stop; i += step) {
    // body
}
```

**Loop with iter_args (loop-carried values):**
```cpp
sum = init_value;  // Initialize
for (int64_t i = start; i < stop; i += step) {
    // body updates sum via yield
    sum = yielded_value;
}
// return_var registered as "sum" (no separate assignment)
```

**Features:**
- Loop variables scoped via automatic registration
- YieldStmt updates iter_args with new values
- Return variables share C++ names with final iter_arg state

### IfStmt (Conditionals)

**Basic if/if-else:**
```cpp
if (condition) { /* then */ }
if (condition) { /* then */ } else { /* else */ }
```

**If-else with return values:**
```cpp
// Declare return variables BEFORE if statement
output_finalType output_final(128, 64);
TASSIGN(output_final, 0x20000);  // If memref present

if (has_tail) {
    // ... compute output_with_tail ...
    output_final = output_with_tail;  // Assign after then_body
} else {
    output_final = output_updated;    // Assign after else_body
}
```

**Features:**
- Return variables declared before if statement with full type definitions
- TileType includes TASSIGN if memref present
- GlobalTensor declared with shape/stride types
- Each branch assigns return values independently
- Pointer mappings inherited for GlobalTensor assignments

### YieldStmt

Passes values from statement bodies to containing control structures:
- **ForStmt**: Updates iter_args for next iteration
- **IfStmt**: Assigns to return_vars after branch completes

Implementation stores values in `yield_buffer_` during traversal.

## Error Handling

Uses PyPTO error conventions:
- `CHECK` for user input validation (raises `pypto::ValueError`)
- `INTERNAL_CHECK` for internal invariants
- Never uses native C++ exceptions

## Testing

**Location:** [tests/ut/codegen/](../../tests/ut/codegen/)

**Test Files:**
- `test_type_converter.py` - DataType, Shape, Stride conversions
- `test_isa_mapper.py` - Operation mapping
- `test_cce_codegen.py` - Integration tests

**Run Tests:**
```bash
pytest tests/ut/codegen/          # All tests
pytest tests/ut/codegen/test_type_converter.py  # Specific file
pytest -v tests/ut/codegen/       # Verbose
```

**Coverage:**
- ✅ Type conversion (DataType, Shape, Stride)
- ✅ Operation mapping (20+ operations)
- ✅ Function generation (signature, prologue, body, epilogue)
- ✅ GlobalTensor and Tile generation
- ✅ Block operations (TLOAD, TSTORE, TADD, TMUL, etc.)
- ✅ Scalar operations (TADDS, TSUBS, TMULS, TDIVS)
- ✅ Reduction operations (sum with axis)
- ✅ Synchronization (set_flag, wait_flag, barriers)
- ✅ Control flow (ForStmt, IfStmt, YieldStmt, nesting)

## Future Enhancements

**Planned:**
1. Dynamic shapes (runtime shape parameters)
2. Enhanced expression handling (nested expressions, constant folding)
3. Optimization (dead code elimination, CSE, instruction scheduling)
4. Debugging support (print statements, profiling, source tracking)

**Extensibility:**
- Add operations: Update `ISAMapper::InitializeMappings()` + optional handling in CCECodegen
- Add types: Update `TypeConverter::ConvertDataType()`
- Add visitor methods: Override in CCECodegen + test

## References

- [IR Overview](../ir/00-overview.md)
- [IR Hierarchy](../ir/01-hierarchy.md)
- [Visitor Pattern](../../include/pypto/ir/transform/base/visitor.h)
- [Pass System](../passes/00-pass_manager.md)
- [pto-isa Documentation](https://gitcode.com/cann/pto-isa)

## Summary

PyPTO codegen provides a clean, modular system for IR→C++ code conversion:

**Design:**
- Standalone architecture (not a Pass)
- Visitor-based traversal
- Modular components with single responsibilities
- Extensible (easy to add operations/types)

**Implemented:**
- ✅ Function generation with `__aicore__` attribute
- ✅ Argument unpacking and GlobalTensor definitions
- ✅ Tile type definitions and TASSIGN allocation
- ✅ 20+ block/sync/barrier operations
- ✅ Control flow (loops, conditionals, yield)
- ✅ Variable name management and pointer tracking
- ✅ Comprehensive test coverage (31 tests)

The foundation is solid and ready for dynamic shapes, optimization, and enhanced expression handling.
