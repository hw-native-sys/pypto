# PTO Codegen

The PTO Codegen (`PTOCodegen`) generates MLIR code in PTO-ISA dialect from PyPTO IR. It transforms high-level PyPTO programs into low-level PTO instructions suitable for accelerator execution.

## Overview

### Key Features

- **Automatic MLIR Generation**: Converts PyPTO IR to PTO-ISA MLIR dialect
- **Structured Code Generation**: Outputs constants, tensor views, allocations in order
- **Implicit Lowering**: Automatically generates `pto.subview` from `block.load`/`block.store`
- **MemRef-based Allocation**: Maps IR MemRef objects to `pto.alloc_tile` operations
- **Type-aware Conversion**: Handles TensorType, TileType, ScalarType appropriately

### Generation Order

The codegen generates MLIR in the following fixed order:

1. **Constants**: `arith.constant` for index and float values
2. **Tensor Views**: `pto.make_tensor_view` for all tensor parameters
3. **Allocations**: `pto.alloc_tile` for all tile buffers (based on MemRef)
4. **Operations**: Function body with load, compute, store operations

## Architecture

### Class Structure

**Header**: `include/pypto/codegen/pto_codegen.h`

```cpp
namespace pypto::codegen {

class PTOCodegen {
 public:
  PTOCodegen() = default;
  ~PTOCodegen() = default;

  // Generate PTO-ISA MLIR from program
  std::string Generate(const ir::ProgramPtr& program);
};

}  // namespace codegen
```

### Implementation Components

**File**: `src/codegen/pto_codegen.cpp`

| Component | Purpose |
|-----------|---------|
| `PTOMLIRCodegen` | Main visitor class for IR traversal |
| `MemRefCollectorVisitor` | Collects all MemRef objects for allocation |
| Helper functions | `DataTypeToMLIR()`, `MemorySpaceToMLIR()` |

## Python API

### Basic Usage

```python
from pypto.ir import compile, OptimizationStrategy
from pypto.backend import BackendType
import pypto.language as pl

@pl.program
class MyKernel:
    @pl.function
    def vector_add(self,
                   a: pl.Tensor[[32, 32], pl.FP32],
                   b: pl.Tensor[[32, 32], pl.FP32]):
        tile_a = pl.load(a, [0, 0], [32, 32])
        tile_b = pl.load(b, [0, 0], [32, 32])
        tile_c = pl.add(tile_a, tile_b)
        pl.store(tile_c, [0, 0], [32, 32], a)

# Compile with PTO backend and PTOAS optimization
output_dir = compile(MyKernel, strategy=OptimizationStrategy.PTOAS, backend_type=BackendType.PTO)
```

The `compile()` function automatically applies the selected optimization strategy and invokes the appropriate codegen based on `backend_type`.

### Direct Codegen Access

```python
from pypto.pypto_core import codegen

# After pass transformations
pto_codegen = codegen.PTOCodegen()
pto_code = pto_codegen.generate(transformed_program)
print(pto_code)
```

## Operator Mappings

### Block Operations → PTO Instructions

| PyPTO Operation | Generated PTO-ISA |
|----------------|-------------------|
| `block.load(tensor, [row, col], [h, w])` | `pto.subview` + `pto.tload` |
| `block.store(tile, [row, col], [h, w], tensor)` | `pto.subview` + `pto.tstore` |
| `block.mul(lhs, rhs)` | `pto.tmul` |
| `block.add(a, b, c)` | `pto.taddc` (3-operand add) |
| `block.adds(tile, scalar)` | `pto.tadds` (tile + scalar) |

### Parameter Type Handling

| PyPTO Type | MLIR Parameter Type | Post-processing |
|------------|---------------------|-----------------|
| `TensorType` | `!pto.ptr<dtype>` | Generate `pto.make_tensor_view` |
| `ScalarType` | `dtype` (e.g., `f32`) | Direct usage as `%argN` |
| `TileType` | Not allowed as parameter | Must be computed internally |

## Code Generation Details

### Tensor View Generation

For each `TensorType` parameter, the codegen generates:

```mlir
%0 = pto.make_tensor_view %arg0,
     shape = [%c32, %c32]
     strides = [%c32, %c1]
     : !pto.tensor_view<2xf32>
```

**Key aspects**:
- Shape from `TensorType.shape_`
- Strides computed as row-major: `[dim1, 1]` for 2D tensors
- Constants (`%c32`, `%c1`) auto-generated

### Allocation Generation

Based on MemRef objects attached to TileType variables:

```mlir
%0 = pto.alloc_tile : <loc=ub, dtype=f32, rows=32, cols=32,
                       v_row=32, v_col=32, blayout=row_major,
                       slayout=none_box, fractal=512, pad=0>
```

**MemRef → alloc_tile mapping**:
- Memory space (`MemRef.memory_space_`) → `loc` attribute
- Tile dimensions inferred from usage context
- One allocation per unique MemRef

### Load Operation Transformation

**PyPTO IR**:
```python
tile_a = pl.load(tensor_a, [0, 0], [32, 32])
```

**Generated MLIR** (two operations):
```mlir
# 1. Create tile view
%3 = pto.subview %tensor_view, offsets = [%c0, %c0],
                 sizes = [%c32, %c32]
                 : !pto.tensor_view<2xf32> -> !pto.tile_view<32x32xf32>

# 2. Load into tile buffer
pto.tload ins(%3 : !pto.tile_view<32x32xf32>)
          outs(%tile_buf : !pto.tile_buf<loc=ub, ...>)
```

**Key transformations**:
- Tensor parameter → tensor_view lookup
- Offsets/sizes from `block.load` arguments
- Output tile_buf from variable's MemRef

### Store Operation Transformation

**PyPTO IR**:
```python
pl.store(tile_c, [0, 0], [32, 32], tensor_out)
```

**Generated MLIR**:
```mlir
# 1. Create tile view for output
%5 = pto.subview %output_view, offsets = [%c0, %c0],
                 sizes = [%c32, %c32]
                 : !pto.tensor_view<2xf32> -> !pto.tile_view<32x32xf32>

# 2. Store from tile buffer
pto.tstore ins(%tile_buf : !pto.tile_buf<loc=ub, ...>)
           outs(%5 : !pto.tile_view<32x32xf32>)
```

### Compute Operations

**Example: Tile Multiplication**

PyPTO:
```python
tile_c = pl.mul(tile_a, tile_b)
```

MLIR:
```mlir
pto.tmul ins(%tile_a_buf : !pto.tile_buf<...>,
             %tile_b_buf : !pto.tile_buf<...>)
         outs(%tile_c_buf : !pto.tile_buf<...>)
```

**Result handling**:
- Result variable's MemRef determines output tile_buf
- Input operands resolved through variable name lookup

## Complete Example

### Input: PyPTO Program

```python
import pypto.language as pl

@pl.program
class MulKernel:
    @pl.function
    def mul_kernel_2d(self,
                     a: pl.Tensor[[32, 32], pl.FP32],
                     b: pl.Tensor[[32, 32], pl.FP32],
                     c: pl.Tensor[[32, 32], pl.FP32]):
        # Load tiles
        tile_a = pl.load(a, [0, 0], [32, 32])
        tile_b = pl.load(b, [0, 0], [32, 32])

        # Multiply
        tile_c = pl.mul(tile_a, tile_b)

        # Store result
        pl.store(tile_c, [0, 0], [32, 32], c)
```

### Output: PTO-ISA MLIR

```mlir
module {
  func.func @mul_kernel_2d(%arg0: !pto.ptr<f32>,
                          %arg1: !pto.ptr<f32>,
                          %arg2: !pto.ptr<f32>) {
    // Constants
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index

    // Tensor views
    %3 = pto.make_tensor_view %arg0, shape = [%c32, %c32]
         strides = [%c32, %c1] : !pto.tensor_view<2xf32>
    %4 = pto.make_tensor_view %arg1, shape = [%c32, %c32]
         strides = [%c32, %c1] : !pto.tensor_view<2xf32>
    %5 = pto.make_tensor_view %arg2, shape = [%c32, %c32]
         strides = [%c32, %c1] : !pto.tensor_view<2xf32>

    // Allocations
    %0 = pto.alloc_tile : <loc=ub, dtype=f32, rows=32, cols=32, ...>
    %1 = pto.alloc_tile : <loc=ub, dtype=f32, rows=32, cols=32, ...>
    %2 = pto.alloc_tile : <loc=ub, dtype=f32, rows=32, cols=32, ...>

    // Load tile_a
    %6 = pto.subview %3, offsets = [%c0, %c0], sizes = [%c32, %c32]
         : !pto.tensor_view<2xf32> -> !pto.tile_view<32x32xf32>
    pto.tload ins(%6 : !pto.tile_view<32x32xf32>)
              outs(%0 : !pto.tile_buf<...>)

    // Load tile_b
    %7 = pto.subview %4, offsets = [%c0, %c0], sizes = [%c32, %c32]
         : !pto.tensor_view<2xf32> -> !pto.tile_view<32x32xf32>
    pto.tload ins(%7 : !pto.tile_view<32x32xf32>)
              outs(%1 : !pto.tile_buf<...>)

    // Multiply
    pto.tmul ins(%0 : !pto.tile_buf<...>, %1 : !pto.tile_buf<...>)
             outs(%2 : !pto.tile_buf<...>)

    // Store tile_c
    %8 = pto.subview %5, offsets = [%c0, %c0], sizes = [%c32, %c32]
         : !pto.tensor_view<2xf32> -> !pto.tile_view<32x32xf32>
    pto.tstore ins(%2 : !pto.tile_buf<...>)
               outs(%8 : !pto.tile_view<32x32xf32>)

    return
  }
}
```

## Variable Mapping

### Internal Tracking

The codegen maintains several mappings to track MLIR variable names:

| Mapping | Purpose | Example |
|---------|---------|---------|
| `var_to_mlir_` | IR variable → MLIR SSA name | `"tile_a"` → `"%0"` |
| `tensor_to_view_` | Parameter → tensor_view | `"a"` → `"%3"` |
| `memref_to_mlir_` | MemRef pointer → tile_buf | `memref.get()` → `"%0"` |

**SSA value naming**:
- Parameters: `%arg0`, `%arg1`, `%arg2`, ...
- Constants: `%c0`, `%c1`, `%c32`, `%cst`, ...
- Results: `%0`, `%1`, `%2`, ...

### MemRef-based Resolution

For operations like `block.mul`:

```python
tile_c = pl.mul(tile_a, tile_b)
```

The codegen:
1. Resolves `tile_a` → `%0` via `var_to_mlir_`
2. Resolves `tile_b` → `%1` via `var_to_mlir_`
3. Gets `tile_c`'s MemRef from its TileType
4. Maps MemRef → `%2` via `memref_to_mlir_`
5. Generates: `pto.tmul ins(%0, %1) outs(%2)`

## Type Conversions

### DataType Mapping

| PyPTO DataType | MLIR Type |
|----------------|-----------|
| `DataType::FP32` | `f32` |
| `DataType::FP16` | `f16` |
| `DataType::BF16` | `bf16` |
| `DataType::INT32` | `i32` |
| `DataType::INT64` | `i64` |
| `DataType::INT8` | `i8` |
| `DataType::UINT8` | `ui8` |

### Memory Space Mapping

| PyPTO MemorySpace | PTO-ISA loc |
|-------------------|-------------|
| `MemorySpace::DDR` | `ddr` |
| `MemorySpace::UB` | `ub` (unified buffer) |
| `MemorySpace::L1` | `l1` |
| `MemorySpace::L0A` | `l0a` |
| `MemorySpace::L0B` | `l0b` |
| `MemorySpace::L0C` | `l0c` |

### Tile Buffer Attributes

Generated `alloc_tile` operations include:

```mlir
!pto.tile_buf<
  loc=ub,              // Memory space
  dtype=f32,           // Element data type
  rows=32,             // Tile height
  cols=32,             // Tile width
  v_row=32,            // Virtual row size
  v_col=32,            // Virtual column size
  blayout=row_major,   // Block layout
  slayout=none_box,    // Sub-layout
  fractal=512,         // Fractal parameter
  pad=0                // Padding
>
```

## Limitations and Future Work

### Current Limitations

1. **Fixed Tile Attributes**: `rows`, `cols`, `blayout` etc. use default values (32x32, row_major)
2. **2D Tensors Only**: Shape/stride generation assumes 2D tensors
3. **Single Memory Space**: All allocations use `ub` (unified buffer) by default
4. **Limited Operations**: Only basic block operations supported

## See Also

- [Pass Manager](10-pass_manager.md): Understanding pass pipeline
- [IR Builder](08-ir_builder.md): Constructing IR programmatically
- [Operator Organization](06-operator_organization.md): Block operation details
