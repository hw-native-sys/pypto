# PTO Codegen

The PTO Codegen (`PTOCodegen`) generates MLIR code in PTO-ISA dialect from PyPTO IR. It transforms high-level PyPTO programs into low-level PTO instructions suitable for accelerator execution.

## Overview

### Key Features

- **Automatic MLIR Generation**: Converts PyPTO IR to PTO-ISA MLIR dialect
- **Structured Code Generation**: Outputs constants, tensor views, allocations in order
- **Implicit Lowering**: Automatically generates `pto.partition_view` from `block.load`/`block.store`
- **MemRef-based Allocation**: Maps IR MemRef objects to `pto.alloc_tile` operations
- **Type-aware Conversion**: Derives tile_buf/tensor_view types from TileType metadata
- **PTOAS Type Annotations**: Emits typed `ins`/`outs` clauses for all operations

### Generation Order

The codegen generates MLIR in the following fixed order:

1. **Constants**: `arith.constant` for index and float values
2. **Tensor Views**: `pto.make_tensor_view` for all tensor parameters
3. **Allocations**: `pto.alloc_tile` for all tile buffers (based on MemRef)
4. **Operations**: Function body with load, compute, store operations

## Architecture

### Class Structure

**Header**: `include/pypto/codegen/pto/pto_codegen.h`

```cpp
namespace pypto::codegen {

class PTOCodegen : public CodegenBase {
 public:
  PTOCodegen();
  explicit PTOCodegen(const backend::Backend* backend);

  std::string Generate(const ir::ProgramPtr& program);

  // CodegenBase interface
  std::string GetCurrentResultTarget() const override;
  void Emit(const std::string& line) override;
  std::string GetExprAsCode(const ir::ExprPtr& expr) override;
  std::string GetTypeString(const DataType& dtype) const override;

  // PTO-specific helpers for operator codegen
  std::string NewTemp();
  std::string GetOrCreateTensorView(const ir::VarPtr& tensor);
  std::string GetIndexConstant(int64_t val);
  std::string GetOrEmitFloatConstant(double value, const std::string& mlir_type = "f32");
  std::string GetTensorViewTypeString(const ir::TensorType* tensor_type) const;
  std::string GetTileBufTypeString(const ir::MemRef* memref) const;
  std::string GetExprTypeAnnotation(const ir::ExprPtr& expr);
  std::string GetCurrentResultTileBufTypeString() const;
};

}  // namespace codegen
```

### Implementation Components

**File**: `src/codegen/pto/pto_codegen.cpp`

| Component | Purpose |
| --------- | ------- |
| `PTOCodegen` | Main visitor class (inherits `CodegenBase`) for IR traversal |
| `MemRefCollectorVisitor` | Collects MemRef objects and their associated TileType for allocation |
| Helper functions | `DataTypeToMLIRImpl()`, `MemorySpaceToMLIR()` |

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
| --------------- | ----------------- |
| `block.load(tensor, [row, col], [h, w])` | `pto.partition_view` + `pto.tload` |
| `block.store(tile, [row, col], [h, w], tensor)` | `pto.partition_view` + `pto.tstore` |
| `block.mul(lhs, rhs)` | `pto.tmul` |
| `block.add(a, b, c)` | `pto.taddc` (3-operand add) |
| `block.adds(tile, scalar)` | `pto.tadds` (tile + scalar) |

### Parameter Type Handling

| PyPTO Type | MLIR Parameter Type | Post-processing |
| ---------- | ------------------- | --------------- |
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
     : !pto.tensor_view<?x?xf32>
```

**Key aspects**:

- Shape from `TensorType.shape_`
- Strides computed as row-major: `[dim1, 1]` for 2D tensors
- Constants (`%c32`, `%c1`) auto-generated
- Tensor view type uses `?` for each dimension (e.g., `?x?xf32` for 2D)

### Allocation Generation

Based on MemRef objects attached to TileType variables. The codegen derives tile dimensions and dtype from the associated TileType:

```mlir
%0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                       v_row=32, v_col=32, blayout=row_major,
                       slayout=none_box, fractal=512, pad=0>
```

**MemRef → alloc_tile mapping**:

- Memory space (`MemRef.memory_space_`) → `loc` attribute (using PTO address space names)
- Tile dtype and dimensions derived from associated TileType metadata
- One allocation per unique MemRef

### Load Operation Transformation

**PyPTO IR**:

```python
tile_a = pl.load(tensor_a, [0, 0], [32, 32])
```

**Generated MLIR** (two operations):

```mlir
# 1. Create partition view
%3 = pto.partition_view %tensor_view, offsets = [%c0, %c0],
                 sizes = [%c32, %c32]
                 : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>

# 2. Load into tile buffer
pto.tload ins(%3 : !pto.partition_tensor_view<32x32xf32>)
          outs(%tile_buf : !pto.tile_buf<loc=vec, ...>)
```

**Key transformations**:

- Tensor parameter → tensor_view lookup
- Offsets/sizes from `block.load` arguments
- Output tile_buf from variable's MemRef with type derived from TileType

### Store Operation Transformation

**PyPTO IR**:

```python
pl.store(tile_c, [0, 0], [32, 32], tensor_out)
```

**Generated MLIR**:

```mlir
# 1. Create partition view for output
%5 = pto.partition_view %output_view, offsets = [%c0, %c0],
                 sizes = [%c32, %c32]
                 : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>

# 2. Store from tile buffer
pto.tstore ins(%tile_buf : !pto.tile_buf<loc=vec, ...>)
           outs(%5 : !pto.partition_tensor_view<32x32xf32>)
```

### Compute Operations

#### Example: Tile Multiplication

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
- All `ins`/`outs` clauses include type annotations

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
         strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>
    %4 = pto.make_tensor_view %arg1, shape = [%c32, %c32]
         strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>
    %5 = pto.make_tensor_view %arg2, shape = [%c32, %c32]
         strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>

    // Allocations
    %0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, ...>
    %1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, ...>
    %2 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, ...>

    // Load tile_a
    %6 = pto.partition_view %3, offsets = [%c0, %c0], sizes = [%c32, %c32]
         : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    pto.tload ins(%6 : !pto.partition_tensor_view<32x32xf32>)
              outs(%0 : !pto.tile_buf<...>)

    // Load tile_b
    %7 = pto.partition_view %4, offsets = [%c0, %c0], sizes = [%c32, %c32]
         : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    pto.tload ins(%7 : !pto.partition_tensor_view<32x32xf32>)
              outs(%1 : !pto.tile_buf<...>)

    // Multiply
    pto.tmul ins(%0 : !pto.tile_buf<...>, %1 : !pto.tile_buf<...>)
             outs(%2 : !pto.tile_buf<...>)

    // Store tile_c
    %8 = pto.partition_view %5, offsets = [%c0, %c0], sizes = [%c32, %c32]
         : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    pto.tstore ins(%2 : !pto.tile_buf<...>)
               outs(%8 : !pto.partition_tensor_view<32x32xf32>)

    return
  }
}
```

## Variable Mapping

### Internal Tracking

The codegen maintains several mappings to track MLIR variable names:

| Mapping | Purpose | Example |
| ------- | ------- | ------- |
| `var_to_mlir_` | IR variable → MLIR SSA name | `"tile_a"` → `"%0"` |
| `tensor_to_view_` | Parameter → tensor_view | `"a"` → `"%3"` |
| `memref_to_mlir_` | MemRef pointer → tile_buf | `memref.get()` → `"%0"` |
| `memref_to_tile_type_` | MemRef pointer → TileType | Used for deriving tile_buf types |

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
5. Gets tile_buf type from `memref_to_tile_type_`
6. Generates: `pto.tmul ins(%0 : !pto.tile_buf<...>, %1 : !pto.tile_buf<...>) outs(%2 : !pto.tile_buf<...>)`

## Type Conversions

### DataType Mapping

| PyPTO DataType | MLIR Type |
| -------------- | --------- |
| `DataType::FP32` | `f32` |
| `DataType::FP16` | `f16` |
| `DataType::BF16` | `bf16` |
| `DataType::INT32` | `i32` |
| `DataType::INT64` | `i64` |
| `DataType::INT8` | `i8` |
| `DataType::UINT8` | `ui8` |

### Memory Space Mapping

| PyPTO MemorySpace | PTO Address Space |
| ----------------- | ----------------- |
| `MemorySpace::DDR` | `gm` (global memory) |
| `MemorySpace::UB` | `vec` (vector buffer) |
| `MemorySpace::L1` | `mat` (matrix buffer) |
| `MemorySpace::L0A` | `left` |
| `MemorySpace::L0B` | `right` |
| `MemorySpace::L0C` | `acc` (accumulator) |

### Tile Buffer Attributes

Generated `alloc_tile` operations derive dtype and dimensions from TileType metadata, and layout/fractal/pad from the associated TileView (when available):

```mlir
!pto.tile_buf<
  loc=vec,             // PTO address space (from MemorySpace)
  dtype=f32,           // Element data type (from TileType)
  rows=32,             // Tile height (from TileType shape)
  cols=32,             // Tile width (from TileType shape)
  v_row=32,            // Virtual row size (= rows)
  v_col=32,            // Virtual column size (= cols)
  blayout=row_major,   // Block layout (from TileView, default: row_major)
  slayout=none_box,    // Scatter layout (from TileView, default: none_box)
  fractal=512,         // Fractal size (from TileView, default: 512)
  pad=0                // Pad mode as int (from TileView, default: 0/null)
>
```

**TileView-derived attributes**:

| Attribute | Source | Enum Values | Default |
| --------- | ------ | ----------- | ------- |
| `blayout` | `TileView::blayout` | `none_box`, `row_major`, `col_major` | `row_major` |
| `slayout` | `TileView::slayout` | `none_box`, `row_major`, `col_major` | `none_box` |
| `fractal` | `TileView::fractal` | uint64 | `512` |
| `pad` | `TileView::pad` | `null(0)`, `zero(1)`, `max(2)`, `min(3)` | `null(0)` |

When no TileView is associated with the MemRef, the codegen falls back to the default values listed above.

## See Also

- [Pass Manager](../passes/00-pass_manager.md): Understanding pass pipeline
- [IR Builder](../ir/06-builder.md): Constructing IR programmatically
- [Operator Organization](../ir/05-operators.md): Block operation details
