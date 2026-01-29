# PyPTO Code Generation Module

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Core Components](#core-components)
- [Code Generation Flow](#code-generation-flow)
- [Operation Mapping](#operation-mapping)
- [Usage Examples](#usage-examples)
- [Implementation Details](#implementation-details)
- [Future Enhancements](#future-enhancements)

## Overview

The PyPTO code generation (codegen) module converts optimized PyPTO IR into executable C++ code using the pto-isa instruction set. After all IR transformation passes complete, the codegen module produces compilable C++ kernel functions that can run on the PTO (Parallel Tensor Operations) hardware architecture.

**Pipeline:**
```
IR → PassManager (IR-to-IR transforms) → CodeGenerator (IR-to-C++ code) → Compiler (C++ to binary)
```

**Key Design Decisions:**
- **Standalone Component**: Codegen is NOT a Pass. Passes transform IR → IR, while codegen transforms IR → String
- **Visitor-Based Traversal**: Extends `IRVisitor` to traverse the IR tree
- **Immutability**: Input IR is never modified during code generation
- **Modular Design**: Separate concerns (emission, mapping, type conversion)

## Architecture

### Component Structure

```
include/pypto/codegen/
├── code_generator.h      # Main generator extending IRVisitor
├── code_emitter.h        # Structured code output with indentation
├── isa_mapper.h          # IR operation → pto-isa instruction mapping
├── type_converter.h      # IR types → pto-isa C++ types
└── code_context.h        # State tracking during generation

src/codegen/
├── code_generator.cpp
├── code_emitter.cpp
├── isa_mapper.cpp
├── type_converter.cpp
└── code_context.cpp
```

### Design Philosophy

1. **Separation of Concerns**: Each component has a single, well-defined responsibility
2. **Extensibility**: Easy to add new operations or type conversions
3. **Testability**: Each component can be tested independently
4. **Readability**: Generated code includes comments and proper indentation

## Core Components

### 1. CodeEmitter

**Purpose**: Manages structured code output with proper indentation.

**Location**: [code_emitter.h](../../include/pypto/codegen/code_emitter.h), [code_emitter.cpp](../../src/codegen/code_emitter.cpp)

**Key Methods:**
```cpp
class CodeEmitter {
  void EmitLine(const std::string& line);    // Emit a single line with indentation
  void EmitBlock(const std::string& code);   // Emit a code block as-is
  void IncreaseIndent();                     // Increase indentation level
  void DecreaseIndent();                     // Decrease indentation level
  std::string GetCode() const;               // Get accumulated code
};
```

**Example:**
```cpp
CodeEmitter emitter;
emitter.EmitLine("void myFunction() {");
emitter.IncreaseIndent();
emitter.EmitLine("int x = 42;");
emitter.DecreaseIndent();
emitter.EmitLine("}");
// Result:
// void myFunction() {
//     int x = 42;
// }
```

### 2. CodeContext

**Purpose**: Tracks variable name mappings during code generation, managing IR variable to C++ variable name associations.

**Location**: [code_context.h](../../include/pypto/codegen/code_context.h), [code_context.cpp](../../src/codegen/code_context.cpp)

**Key Features:**
- Maps IR variables to C++ variable names with explicit registration
- Sanitizes IR names for C++ compatibility
- Enforces one-time registration to prevent duplicate variable declarations

**Key Methods:**
```cpp
class CodeContext {
  std::string SanitizeName(const VarPtr& var) const;  // Convert IR name to valid C++ identifier
  std::string GetVarName(const VarPtr& var);          // Get registered C++ name (throws if not found)
  void RegisterVar(const VarPtr& var, const std::string& cpp_name);  // Register IR var → C++ name mapping
  void RegisterPointer(const std::string& tensor_var_name, const std::string& ptr_name);  // Register tensor → raw pointer mapping
  std::string GetPointer(const std::string& tensor_var_name) const;  // Get raw pointer name for tensor variable
  void Clear();                                         // Clear all state
};
```

**Usage Pattern:**
```cpp
// First sanitize the IR name to get a valid C++ identifier
std::string cpp_name = context_.SanitizeName(ir_var);

// Then register the variable for later lookup
context_.RegisterVar(ir_var, cpp_name);

// Later, retrieve the registered name
std::string name = context_.GetVarName(ir_var);  // Returns cpp_name
```

**Important:** `RegisterVar()` enforces one-time registration. Attempting to register the same variable twice will raise an error. `GetVarName()` requires the variable to be already registered, otherwise it will raise an error.

**Naming Convention:**
- **Function parameters**: Registered with "Global" suffix (e.g., `input_a` → `input_aGlobal`)
  - Raw pointer unpacking uses base name: `__gm__ float* input_a`
  - GlobalTensor instance uses "Global" suffix: `input_aGlobal`
- **Tile variables**: Registered with sanitized IR name when first assigned
- **Regular variables**: Registered on first assignment with sanitized name

**Pointer Tracking:**

CodeContext maintains a separate mapping from tensor variables to their underlying raw pointers. This is essential for correct address computation in TASSIGN instructions used by block.load/store operations.

**Why Pointer Tracking is Needed:**
- GlobalTensor variables like `outputGlobal` wrap raw pointers like `output`
- For address arithmetic (e.g., `output + offset`), we need the raw pointer name, not the GlobalTensor name
- Iteration variables and IfStmt return values may alias different tensor variables, requiring pointer inheritance

**Example:**
```cpp
// Function prologue generates:
__gm__ float* output = reinterpret_cast<__gm__ float*>(args[2]);  // Raw pointer
outputGlobalType outputGlobal(output);  // GlobalTensor wrapper

// CodeContext tracks: outputGlobal → output
context_.RegisterPointer("outputGlobal", "output");

// Later in block.store, we need the raw pointer:
std::string ptr = context_.GetPointer("outputGlobal");  // Returns "output"
emitter_.EmitLine("TASSIGN(outputGlobal, " + ptr + " + offset);");
// Result: TASSIGN(outputGlobal, output + tile_idx * 128 + 0);
```

**Pointer Inheritance:**
- **ForStmt iter_args**: When initializing from a tensor variable, inherits its pointer mapping
- **IfStmt return_vars**: When assigned from yielded tensor values, inherits their pointer mappings
- This ensures correct pointer names throughout control flow structures

**Example of Pointer Inheritance:**
```cpp
// ForStmt iteration argument initialization
for (auto& iter_arg : op->iter_args_) {
  // iter_arg initialized from outputGlobal
  auto init_var = std::dynamic_pointer_cast<const ir::Var>(iter_arg->initValue_);
  std::string init_var_name = context_.GetVarName(init_var);  // "outputGlobal"
  std::string init_ptr = context_.GetPointer(init_var_name);  // "output"

  // Register iter_arg with inherited pointer
  context_.RegisterPointer(iter_arg_name, init_ptr);  // output_iter → output
}

// IfStmt return variable assignment
if (...) {
  // yielded_value might be "output_iter" pointing to "output"
  std::string yielded_ptr = context_.GetPointer(yielded_value);  // "output"
  context_.RegisterPointer(return_var_name, yielded_ptr);  // output_final → output
}
```


### 3. TypeConverter

**Purpose**: Converts PyPTO IR types to pto-isa C++ type strings.

**Location**: [type_converter.h](../../include/pypto/codegen/type_converter.h), [type_converter.cpp](../../src/codegen/type_converter.cpp)

**Conversion Tables:**

**DataType → C++ Type:**
| PyPTO DataType | pto-isa C++ Type |
|----------------|------------------|
| FP32           | `float`          |
| FP16           | `half`           |
| INT32          | `int32_t`        |
| INT64          | `int64_t`        |
| BOOL           | `bool`           |
| BF16           | `bfloat16`       |

**MemorySpace → Annotation:**
| PyPTO MemorySpace | pto-isa Annotation |
|-------------------|-------------------|
| DDR               | `__gm__`          |
| UB                | (none)            |
| L0A/L0B/L0C       | (none)            |

**Shape/Stride Generation:**
- Shapes are padded to 5 dimensions with leading 1s
- Strides are calculated for row-major layout

**Examples:**
```cpp
TypeConverter converter;

// DataType conversion
converter.ConvertDataType(DataType::FP32);  // → "float"

// Shape generation (5D with padding)
converter.GenerateShapeType({128, 64});     // → "Shape<1, 1, 1, 128, 64>"

// Stride generation (row-major)
converter.GenerateStrideType({128, 64});    // → "Stride<1, 1, 1, 64, 1>"
```

### 4. ISAMapper

**Purpose**: Maps PyPTO IR operation names to pto-isa instruction names.

**Location**: [isa_mapper.h](../../include/pypto/codegen/isa_mapper.h), [isa_mapper.cpp](../../src/codegen/isa_mapper.cpp)

**Operation Mapping Table:**

| IR Operation    | pto-isa Instruction | Description |
|-----------------|---------------------|-------------|
| `block.load`    | `TLOAD`             | Load from global to tile |
| `block.store`   | `TSTORE`            | Store from tile to global |
| `block.add`     | `TADD`              | Tile + Tile addition |
| `block.sub`     | `TSUB`              | Tile - Tile subtraction |
| `block.mul`     | `TMUL`              | Tile * Tile multiplication |
| `block.div`     | `TDIV`              | Tile / Tile division |
| `block.adds`    | `TADDS`             | Tile + Scalar addition |
| `block.subs`    | `TSUBS`             | Tile - Scalar subtraction |
| `block.muls`    | `TMULS`             | Tile * Scalar multiplication |
| `block.divs`    | `TDIVS`             | Tile / Scalar division |
| `block.sqrt`    | `TSQRT`             | Element-wise square root |
| `block.sum`     | `TROWSUM`/`TCOLSUM` | Reduction (axis-dependent) |
| `system.sync_src` | `set_flag`        | Set synchronization flag |
| `system.sync_dst` | `wait_flag`       | Wait for synchronization flag |
| `system.bar_v`  | `pipe_barrier`      | Vector unit barrier |
| `system.bar_m`  | `pipe_barrier`      | Matrix unit barrier |
| `system.bar_all`| `pipe_barrier`      | Global barrier |

**Special Handling: block.sum**

The `block.sum` operation requires an `axis` attribute to determine the instruction:
- `axis=0` → `TCOLSUM` (sum across rows, collapse dimension 0)
- `axis=1` → `TROWSUM` (sum across columns, collapse dimension 1)

### 5. CodeGenerator

**Purpose**: Main code generation class that orchestrates all components.

**Location**: [code_generator.h](../../include/pypto/codegen/code_generator.h), [code_generator.cpp](../../src/codegen/code_generator.cpp)

**Extends**: `IRVisitor` for IR tree traversal

**Main Entry Point:**
```cpp
class CodeGenerator : public IRVisitor {
  std::string Generate(const FunctionPtr& func);  // Generate C++ code from IR function
};
```

## Code Generation Flow

### Three-Phase Generation

#### Phase 1: Prologue

Generates:
1. Function signature with `__aicore__` and `__attribute__((always_inline))`
2. Argument unpacking from `int64_t* args` array
3. GlobalTensor type definitions and instances
4. Tile type definitions with TASSIGN memory allocation

The prologue uses **TileCollector**, a visitor that traverses the function body to discover tile-typed variables from `AssignStmt` nodes. These variables are declared in the prologue with their full type definitions, TASSIGN memory allocation, and pointer tracking.

**Note:** IfStmt return_vars are NOT collected by TileCollector. Instead, they are declared immediately before the if statement (see IfStmt section below).

**Example Output:**
```cpp
__aicore__ __attribute__((always_inline)) void runSimpleAdd(__gm__ int64_t* args)
{
    // Unpack arguments
    __gm__ float* x = reinterpret_cast<__gm__ float*>(args[0]);
    __gm__ float* y = reinterpret_cast<__gm__ float*>(args[1]);
    __gm__ float* output = reinterpret_cast<__gm__ float*>(args[2]);

    // Global tensor declarations
    using xShapeDim5 = Shape<1, 1, 1, 128, 64>;
    using xStrideDim5 = Stride<1, 1, 1, 64, 1>;
    using xGlobalType = GlobalTensor<float, xShapeDim5, xStrideDim5>;
    xGlobalType xGlobal(x);

    using yShapeDim5 = Shape<1, 1, 1, 128, 64>;
    using yStrideDim5 = Stride<1, 1, 1, 64, 1>;
    using yGlobalType = GlobalTensor<float, yShapeDim5, yStrideDim5>;
    yGlobalType yGlobal(y);

    using outputShapeDim5 = Shape<1, 1, 1, 128, 64>;
    using outputStrideDim5 = Stride<1, 1, 1, 64, 1>;
    using outputGlobalType = GlobalTensor<float, outputShapeDim5, outputStrideDim5>;
    outputGlobalType outputGlobal(output);

    // Tile type definitions and allocations
    using tile_xType = Tile<TileType::Vec, float, 128, 64, BLayout::RowMajor, -1, -1>;
    tile_xType tile_x(128, 64);
    TASSIGN(tile_x, 0x0);

    using tile_yType = Tile<TileType::Vec, float, 128, 64, BLayout::RowMajor, -1, -1>;
    tile_yType tile_y(128, 64);
    TASSIGN(tile_y, 0x10000);

    using tile_zType = Tile<TileType::Vec, float, 128, 64, BLayout::RowMajor, -1, -1>;
    tile_zType tile_z(128, 64);
    TASSIGN(tile_z, 0x20000);
```

#### Phase 2: Body

Generates:
- Block operation instructions (TLOAD, TADD, TSTORE, etc.)
- Synchronization operations (set_flag, wait_flag)
- Control flow structures (loops, conditionals - future)
- Variable assignments

**Example Output:**
```cpp
    // Function body
    TLOAD(tile_x, xGlobal);
    TLOAD(tile_y, yGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADD(tile_z, tile_x, tile_y);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(outputGlobal, tile_z);
```

#### Phase 3: Epilogue

Generates:
- Closing brace
- Optional cleanup code (currently minimal)

**Example Output:**
```cpp
}
```

### Visitor Pattern Usage

The `CodeGenerator` overrides key visitor methods to generate code:

```cpp
// Called for Call expressions (operations)
void VisitExpr_(const CallPtr& op) override;

// Called for assignment statements
void VisitStmt_(const AssignStmtPtr& op) override;

// Called for evaluation statements (sync operations, barriers)
void VisitStmt_(const EvalStmtPtr& op) override;

// Called for statement sequences
void VisitStmt_(const SeqStmtsPtr& op) override;

// Called for return statements
void VisitStmt_(const ReturnStmtPtr& op) override;
```

## Operation Mapping

The ISAMapper component maps PyPTO IR operations to pto-isa instructions. Below is the complete list of supported operations:

### Complete Mapping Table

| IR Operation | pto-isa Instruction | Category | Notes |
|--------------|---------------------|----------|-------|
| **Memory Operations** |
| `block.load` | `TLOAD` | BlockOp | Load from DDR to UB tile |
| `block.store` | `TSTORE` | BlockOp | Store from UB tile to DDR |
| **Element-wise Binary (Tile + Tile)** |
| `block.add` | `TADD` | BlockOp | Element-wise addition |
| `block.sub` | `TSUB` | BlockOp | Element-wise subtraction |
| `block.mul` | `TMUL` | BlockOp | Element-wise multiplication |
| `block.div` | `TDIV` | BlockOp | Element-wise division |
| **Element-wise Binary (Tile + Scalar)** |
| `block.adds` | `TADDS` | BlockOp | Tile + scalar addition |
| `block.subs` | `TSUBS` | BlockOp | Tile - scalar subtraction |
| `block.muls` | `TMULS` | BlockOp | Tile * scalar multiplication |
| `block.divs` | `TDIVS` | BlockOp | Tile / scalar division |
| **Unary Operations** |
| `block.sqrt` | `TSQRT` | BlockOp | Element-wise square root |
| **Reduction Operations** |
| `block.sum` (axis=0) | `TCOLSUM` | BlockOp | Sum across rows (collapse dim 0) |
| `block.sum` (axis=1) | `TROWSUM` | BlockOp | Sum across columns (collapse dim 1) |
| **Synchronization Operations** |
| `system.sync_src` | `set_flag` | SyncOp | Set synchronization flag |
| `system.sync_dst` | `wait_flag` | SyncOp | Wait for synchronization flag |
| `system.bar_v` | `pipe_barrier` | SyncOp | Vector unit barrier |
| `system.bar_m` | `pipe_barrier` | SyncOp | Matrix unit barrier |
| `system.bar_all` | `pipe_barrier` | SyncOp | Global barrier |

### Block Operations

#### Memory Operations

**block.load: Load from global memory to tile**
```cpp
// IR: tile = block.load(tensor, row_offset, col_offset, height, width)
// Generated:
//   Compute offset: row_offset * stride + col_offset
//   TASSIGN(tensorGlobal, tensor + offset);  // Set start address
//   TLOAD(tile, tensorGlobal);               // Load from computed address
```

Example with concrete values:
```cpp
// IR: tile_x = block.load(input, 0, 0, 32, 128)
// Generated:
//   TASSIGN(inputGlobal, input + 0 * 128 + 0);
//   TLOAD(tile_x, inputGlobal);
```

**block.store: Store from tile to global memory**
```cpp
// IR: result = block.store(tile, row_offset, col_offset, height, width, output)
// Generated:
//   Compute offset: row_offset * stride + col_offset
//   TASSIGN(outputGlobal, output + offset);  // Set start address
//   TSTORE(outputGlobal, tile);              // Store to computed address
```

Example with concrete values:
```cpp
// IR: output_new = block.store(tile_z, 0, 0, 32, 128, output)
// Generated:
//   TASSIGN(outputGlobal, output + 0 * 128 + 0);
//   TSTORE(outputGlobal, tile_z);
```

#### Element-wise Operations

**block.add: Element-wise addition**
```cpp
// IR: tile_z = block.add(tile_x, tile_y)
// Generated: TADD(tile_z, tile_x, tile_y);
```

**block.mul: Element-wise multiplication**
```cpp
// IR: tile_z = block.mul(tile_x, tile_y)
// Generated: TMUL(tile_z, tile_x, tile_y);
```

#### Unary Operations

**block.sqrt: Element-wise square root**
```cpp
// IR: tile_y = block.sqrt(tile_x)
// Generated: TSQRT(tile_y, tile_x);
```

### Synchronization Operations

**system.sync_src: Set synchronization flag**
```cpp
// IR: system.sync_src(set_pipe, wait_pipe, event_id)
// Generated: set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
```

**system.sync_dst: Wait for synchronization flag**
```cpp
// IR: system.sync_dst(set_pipe, wait_pipe, event_id)
// Generated: wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
```

**system.bar_v, system.bar_m, system.bar_all: Barrier synchronization**
```cpp
// IR: system.bar_v()
// Generated: pipe_barrier(PIPE_V);

// IR: system.bar_m()
// Generated: pipe_barrier(PIPE_M);

// IR: system.bar_all()
// Generated: pipe_barrier(PIPE_ALL);
```

**Note:** Synchronization operations are inserted by transformation passes into the IR. Codegen directly translates them without inferring where synchronization is needed.

## Usage Examples

### Basic Usage (C++)

```cpp
#include "pypto/codegen/code_generator.h"

// Assuming you have a FunctionPtr from IR construction or parsing
FunctionPtr func = /* ... */;

// Create generator
codegen::CodeGenerator generator;

// Generate C++ code
std::string cpp_code = generator.Generate(func);

// Write to file or compile
std::ofstream out("generated_kernel.cpp");
out << cpp_code;
out.close();
```

### Complete Example: Simple Add

**Input IR (conceptual):**
```python
def simple_add(x: Tensor([128, 64], FP32),
               y: Tensor([128, 64], FP32)) -> Tensor([128, 64], FP32):
    tile_x = block.load(x, 0, 0, 128, 64)
    tile_y = block.load(y, 0, 0, 128, 64)
    sync.set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0)
    sync.wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0)
    tile_z = block.add(tile_x, tile_y)
    sync.set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0)
    sync.wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0)
    result = block.store(tile_z, 0, 0, 128, 64, output)
    return result
```

**Generated C++ Code:**
```cpp
__aicore__ __attribute__((always_inline)) void runSimpleAdd(__gm__ int64_t* args)
{
    // Unpack arguments
    __gm__ float* x = reinterpret_cast<__gm__ float*>(args[0]);
    __gm__ float* y = reinterpret_cast<__gm__ float*>(args[1]);
    __gm__ float* output = reinterpret_cast<__gm__ float*>(args[2]);

    // Global tensor declarations
    // (type definitions would go here)

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

**Design Decision:** UB (Unified Buffer) memory addresses come from IR metadata via TileType's MemRef field. GlobalTensor raw pointer tracking ensures correct address computation.

**Implementation:**
- Transformation passes set the MemRef field in TileType with memory addresses
- Codegen extracts addresses from `TileType::memref_::addr_` (expecting ConstInt expressions)
- TASSIGN instructions bind tiles to specific UB memory addresses
- Addresses are formatted as hexadecimal (e.g., `0x0`, `0x10000`, `0x20000`)
- **Pointer Tracking**: CodeContext maintains mappings from tensor variables to their raw pointers
  - Enables correct address computation: `TASSIGN(tensorGlobal, raw_ptr + offset)`
  - Supports pointer inheritance through control flow (ForStmt iter_args, IfStmt return_vars)

**Example:**
```cpp
// IR: TileType with memref_.addr_ = ConstInt(0x10000)
// Generated code:
using tile_aType = Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor, -1, -1>;
tile_aType tile_a(64, 64);
TASSIGN(tile_a, 0x10000);
```

**Note:** If TileType does not have a MemRef (memref_ is nullopt), the TASSIGN instruction is skipped. AllocOp will handle memory allocation in a future pass (see Pass 4: AllocOpInsertionPass).

### Dual-Mode Expression Pattern

**Design Decision:** Expression visitors operate in two distinct modes depending on expression context.

The `CodeGenerator` uses two member variables to implement a dual-mode pattern for expression handling:

```cpp
std::string current_target_var_;  // INPUT: Assignment target variable name (for Call expressions)
std::string current_expr_value_;  // OUTPUT: Inline C++ value for scalar expressions
```

**Mode 1: Statement-Emitting Mode (Call Expressions)**

Used for block operations that emit complete instruction statements.

- **Input**: `current_target_var_` contains the assignment target variable name
- **Behavior**: Expression visitor emits complete instruction statements directly to the code emitter
- **Output**: Clears `current_expr_value_` to indicate no inline value
- **Example**: `tile_z = block.add(tile_x, tile_y)`

```cpp
// Before visiting Call expression:
current_target_var_ = "tile_z";  // Set by AssignStmt visitor

// During VisitExpr_(const CallPtr& op) for block.add:
emitter_.EmitLine("TADD(tile_z, tile_x, tile_y);");  // Emit instruction
current_expr_value_ = "";  // Clear to indicate statement was emitted

// Result: TADD instruction emitted directly
```

**Mode 2: Value-Returning Mode (Scalar Expressions)**

Used for scalar expressions that produce inline C++ code.

- **Input**: No specific input needed (operates on expression tree)
- **Behavior**: Expression visitor generates inline C++ code representing the expression
- **Output**: Sets `current_expr_value_` with the inline code string
- **Example**: `offset = i * 128 + j`

```cpp
// Visiting scalar expressions:
VisitExpr_(const VarPtr& op) {
  current_expr_value_ = context_.GetVarName(op);  // "i"
}

VisitExpr_(const ConstIntPtr& op) {
  current_expr_value_ = std::to_string(op->value_);  // "128"
}

VisitExpr_(const MulPtr& op) {
  VisitExpr(op->lhs_);  // Gets "i"
  std::string lhs = current_expr_value_;
  VisitExpr(op->rhs_);  // Gets "128"
  std::string rhs = current_expr_value_;
  current_expr_value_ = "(" + lhs + " * " + rhs + ")";  // "(i * 128)"
}

// Result: current_expr_value_ = "(i * 128 + j)" as inline code
```

**Pattern Usage:**

1. **Block Operations**: Use Mode 1 to emit pto-isa instructions
2. **Scalar Computations**: Use Mode 2 to generate inline C++ expressions
3. **Mixed Contexts**: Switch modes based on expression type and context

**Implementation Note:**

The dual-mode pattern allows the same visitor infrastructure to handle both:
- High-level tile operations that map to single pto-isa instructions
- Low-level scalar computations that become inline C++ expressions

This design keeps the visitor pattern clean while supporting both code emission styles without type-specific branching in every visitor method.

### Synchronization Strategy

**Design Decision:** Synchronization is explicit in the IR.

- Transformation passes insert `system.sync_src` and `system.sync_dst` operations
- Codegen directly translates these operations to C++ function calls
- No automatic synchronization inference in codegen

**Typical Synchronization Pattern:**
```
Load Operations (MTE2 → V):
  TLOAD(...)
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0)
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0)

Compute Operations (V → MTE3):
  TADD/TMUL/etc(...)
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0)
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0)

Store Operations:
  TSTORE(...)
```

### Type Assumptions

**Current Implementation:**
- Only immediate (constant) shapes are supported
- Dynamic shapes are not yet implemented
- All tensor shapes must be known at compile time

### Error Handling

The codegen module uses PyPTO error conventions:
- `CHECK` for user input validation (raises `pypto::ValueError`)
- `INTERNAL_CHECK` for internal invariants
- Never uses native C++ exceptions (`std::runtime_error`, etc.)

## Control Flow Code Generation

### ForStmt (Loop)

Generates C++ for loops with optional loop-carried values (iteration arguments).

**Simple for loop (no iter_args):**
```cpp
for (int64_t i = start; i < stop; i += step) {
    // loop body
}
```

**For loop with iteration arguments (loop-carried values):**
```cpp
// Initialize iteration arguments
sum = init_value;

for (int64_t i = start; i < stop; i += step) {
    // loop body (may update iter_args via yield)
    sum = yielded_value;
}

// return_var is registered with name "sum" (same as iter_arg)
// No assignment needed - they represent the same value
```

**Key Features:**
- Loop variables are scoped within the loop via automatic registration
- Iteration arguments enable SSA-style value threading across loop iterations
- YieldStmt updates iteration arguments with new values from each iteration
- Return variables are registered with the same C++ names as their corresponding iter_args
- No code emission needed for return variables - they directly reference the final iter_arg state

### IfStmt (Conditional)

Generates C++ if and if-else statements for conditional execution.

**If statement without else:**
```cpp
if (condition) {
    // then_body
}
```

**If-else statement:**
```cpp
if (condition) {
    // then_body
} else {
    // else_body
}
```

**If-else with return values (SSA-style):**

When IfStmt has return_vars (variables capturing results from branches), the code generator:
1. **Declares return variables BEFORE the if statement** (not in prologue)
   - For TileType: generates type alias, instance, and TASSIGN if memref is present
   - For TensorType (GlobalTensor): generates shape/stride types, type alias, and uninitialized instance
   - For ScalarType: generates basic C++ type declaration
2. Each branch yields its result value
3. Immediately after each branch completes, assigns the yielded value to the return variable and inherits pointer mappings

```cpp
// Return variables declared BEFORE if statement
using output_finalType = Tile<TileType::Vec, float, 128, 64, BLayout::RowMajor, -1, -1>;
output_finalType output_final(128, 64);
TASSIGN(output_final, 0x20000);  // If memref is present in TileType

// Generated if-else
if (has_tail) {
    // ... compute output_with_tail ...
    output_final = output_with_tail;  // Assigned after then_body
    // Inherit pointer mapping if GlobalTensor
} else {
    output_final = output_updated;    // Assigned after else_body
    // Inherit pointer mapping if GlobalTensor
}
// output_final now available for use
```

**Key Features:**
- Condition can be any boolean expression (comparison, logical ops, variables)
- Both branches support arbitrary statements (including nested control flow)
- Optional else branch (nullopt means no else)
- Return variables are declared BEFORE the if statement with proper type definitions
  - TileType variables include TASSIGN if memref is present
  - GlobalTensor variables are declared with full shape/stride types
- Return variables capture results from branches via yield statements
- Pointer mappings are inherited when assigning GlobalTensor return values
- Each branch assigns its return values independently (prevents overwriting)

### YieldStmt

Used to pass values from statement bodies to their containing control flow structures.

**Usage in ForStmt (loop-carried values):**
```cpp
for (...) {
    // compute new values
    computed_value_1 = ...;
    computed_value_2 = ...;

    // yield new values for iter_args
    yield(computed_value_1, computed_value_2);
}
```

**Usage in IfStmt (branch return values):**
```cpp
if (condition) {
    result = compute_then_value();
    yield(result);  // Passes result to return_var
} else {
    result = compute_else_value();
    yield(result);  // Passes result to return_var
}
```

**Implementation:**
- Evaluates yielded expressions during body traversal
- Stores values in temporary buffer (`yield_buffer_`)
- ForStmt: assigns yielded values to iteration arguments for next iteration
- IfStmt: assigns yielded values to return variables after each branch completes

## Future Enhancements

### Completed Features

1. **GlobalTensor Generation** ✅
   - Generate full GlobalTensor type definitions with Shape and Stride types
   - Generate GlobalTensor instances with proper initialization from argument pointers
   - Support constant shapes (immediate values extracted from IR)

2. **Tile Management** ✅
   - Generate Tile type definitions with TileType::Vec (UB storage)
   - Generate Tile instances with dimensions from IR
   - Emit TASSIGN with addresses from MemRef
   - Support row-major layout with dynamic valid regions

3. **Synchronization Operations** ✅
   - Handle system.sync_src and system.sync_dst operations via EvalStmt
   - Translate to set_flag/wait_flag with pipe and event parameters
   - Support barrier operations (vector, matrix, all)

4. **Control Flow Support** ✅
   - Generate C++ loops from ForStmt (simple and nested)
   - Generate C++ conditionals from IfStmt (with and without else branch)
   - Handle YieldStmt for loop-carried values and branch return values
   - Support nested control structures (for loops within for loops, if within for, etc.)
   - Assign return values from yield statements in each branch

### Planned Features

1. **Dynamic Shapes**
   - Support runtime shape parameters
   - Generate shape computations in prologue

2. **Expression Handling**
   - Support more expression types in GetExprName
   - Handle nested expressions
   - Constant folding in generated code

3. **Optimization**
   - Dead code elimination
   - Common subexpression elimination
   - Instruction scheduling hints

4. **Debugging Support**
   - Insert debug print statements
   - Generate profiling instrumentation
   - Source location tracking in comments

### Extensibility Points

**Adding New Operations:**
1. Add mapping to `ISAMapper::InitializeMappings()`
2. (Optional) Add special handling in `CodeGenerator::VisitStmt_(AssignStmtPtr)`
3. Update documentation

**Adding New Types:**
1. Add conversion in `TypeConverter::ConvertDataType()`
2. Update documentation

**Adding New Visitor Methods:**
1. Override in `CodeGenerator`
2. Emit appropriate code
3. Test thoroughly

## Testing

### Unit Tests

Tests are located in [tests/ut/codegen/](../../tests/ut/codegen/):

- `test_type_converter.py` - Tests for DataType, Shape, Stride conversions
- `test_isa_mapper.py` - Tests for operation mapping
- `test_code_generator.py` - Integration tests for code generation

### Running Tests

```bash
# Run all codegen tests
pytest tests/ut/codegen/

# Run specific test file
pytest tests/ut/codegen/test_type_converter.py

# Run with verbose output
pytest -v tests/ut/codegen/
```

### Test Coverage

Current test coverage includes:
- ✅ DataType conversion
- ✅ Shape/Stride generation (5D padding)
- ✅ Operation mapping (block ops, sync ops, barrier ops)
- ✅ Function signature generation with `__aicore__` attribute
- ✅ Argument unpacking and GlobalTensor generation
- ✅ Tile type definitions and TASSIGN memory allocation
- ✅ Block operation code generation (TLOAD, TSTORE, TADD, TMUL, TSQRT, etc.)
- ✅ Scalar operation support (TADDS, TSUBS, TMULS, TDIVS)
- ✅ Reduction operations (sum with axis attribute)
- ✅ Synchronization operations (set_flag, wait_flag)
- ✅ Barrier operations (vector, matrix, all)
- ✅ Control flow generation (ForStmt, IfStmt, YieldStmt)
- ✅ Nested for loops
- ✅ If-else statements

## References

- **PyPTO IR Definition**: [00-ir_definition.md](00-ir_definition.md)
- **Visitor Pattern**: [include/pypto/ir/transform/base/visitor.h](../../include/pypto/ir/transform/base/visitor.h)
- **Pass System**: [08-pass_manager.md](08-pass_manager.md)
- **pto-isa Documentation**: `/data/q00953770/workspace/github/pto-isa/docs/`
- **Codegen Requirements**: [docs/_build/codegen.md](../_build/codegen.md)

## Summary

The PyPTO codegen module provides a clean, modular, and extensible system for converting IR to executable C++ code:

- **Standalone Architecture**: Separate from the Pass system
- **Visitor-Based**: Leverages existing IRVisitor infrastructure
- **Modular Components**: Each with a single responsibility
- **Extensible**: Easy to add new operations and types
- **Testable**: Comprehensive unit test coverage (31 tests passing)
- **Well-Documented**: Inline comments and external documentation

**Implemented Features:**
- ✅ Function signature generation with `__aicore__` attribute
- ✅ Argument unpacking from `int64_t* args`
- ✅ GlobalTensor type definitions (Shape, Stride, GlobalTensor)
- ✅ GlobalTensor instance creation with proper initialization
- ✅ Tile type definitions and instances (TileType::Vec)
- ✅ Tile memory allocation via TASSIGN with addresses from MemRef
- ✅ Block operation code generation (TLOAD, TSTORE, TADD, TMUL, etc.)
- ✅ Scalar operation support (TADDS, TSUBS, TMULS, TDIVS)
- ✅ Synchronization operation translation (set_flag, wait_flag)
- ✅ Barrier operation support (vector, matrix, all)
- ✅ ISA instruction mapping for 20+ operations
- ✅ Type conversion utilities (DataType, MemorySpace, Shape, Stride)
- ✅ Variable name sanitization and registration via CodeContext

The foundation is solid and ready for future enhancements including control flow, dynamic shapes, and optimization features.
