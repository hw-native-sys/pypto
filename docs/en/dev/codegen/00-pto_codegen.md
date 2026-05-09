# PTO Codegen

The PTO Codegen (`PTOCodegen`) generates MLIR code in PTO-ISA dialect from PyPTO IR. It transforms high-level PyPTO programs into low-level PTO instructions suitable for accelerator execution.

## Design Principle: Strict 1-to-1 Mapping

Codegen must be a **strict 1-to-1 translation** from IR to generated code. Each IR node maps directly to its corresponding output construct — no optimization, analysis, or indirection transformation should occur in the codegen layer.

| Belongs in codegen | Belongs in an earlier pass |
| ------------------ | -------------------------- |
| IR node → output code mapping | Data-flow analysis (e.g., tracing return values to parameters) |
| Type/format conversion (DataType → MLIR type) | IR restructuring or canonicalization |
| Name mangling and SSA bookkeeping | Optimization or simplification |

**Why:** Codegen that embeds analysis becomes fragile — it duplicates logic that passes already handle, and it's harder to test in isolation. Keeping codegen a straightforward translation ensures it stays predictable and maintainable.

**When analysis is found in codegen:** File a tracking issue and refactor it into a dedicated pass when bandwidth allows. [#814](https://github.com/hw-native-sys/pypto/issues/814) was an example: return-to-parameter tracing in orchestration codegen has been refactored into the [`NormalizeReturnOrder`](../passes/23-normalize_return_order.md) pass.

## Overview

### Key Features

- **Automatic MLIR Generation**: Converts PyPTO IR to PTO-ISA MLIR dialect
- **Structured Code Generation**: Outputs constants, tensor views, allocations in order
- **Implicit Lowering**: Automatically generates `pto.partition_view` from `tile.load`/`tile.store`
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
  std::string GetOrEmitConstant(int64_t value, DataType dt);   // int/index overload
  std::string GetOrEmitConstant(double value, DataType dt);    // float overload
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
        pl.store(tile_c, [0, 0], a)

# Compile with PTO backend and DebugTileOptimization (debug only)
output_dir = compile(
    MyKernel,
    strategy=OptimizationStrategy.DebugTileOptimization,
    backend_type=BackendType.Ascend910B,
)
```

The `compile()` function automatically applies the selected optimization strategy and invokes the appropriate codegen based on `backend_type`.
Use `Default` for normal PTO compilation; `DebugTileOptimization` is intended only for pass-pipeline debugging.

### Direct Codegen Access

```python
from pypto.pypto_core import codegen

# After pass transformations
pto_codegen = codegen.PTOCodegen()
pto_code = pto_codegen.generate(transformed_program)
print(pto_code)
```

## Operator Mappings

### Tile Operations → PTO Instructions

| PyPTO Operation | Generated PTO-ISA |
| --------------- | ----------------- |
| `tile.load(tensor, [row, col], [h, w])` | `pto.partition_view` + `pto.tload` |
| `tile.store(tile, [row, col], tensor)` | `pto.partition_view` + `pto.tstore` |
| `tile.slice(tile, [h, w], [row, col][, valid_shape=...])` | `pto.subview` (zero-copy view; `valid [...]` clause emitted only when `valid_shape` is supplied) |
| `tile.assemble(target, source, [row, col])` | (optional) `pto.tmov target -> dst` + `pto.subview dst[row, col] sizes [src.rows, src.cols]` + `pto.tmov src -> dst_view` |
| `tile.mul(lhs, rhs)` | `pto.tmul` |
| `tile.add(a, b, c)` | `pto.taddc` (3-operand add) |
| `tile.adds(tile, scalar)` | `pto.tadds` (tile + scalar) |

**`tile.slice` / `tile.assemble` lowering details.**  Both ops are lowered
through `pto.subview`, which is a pure view alias of the source tile (no
data movement, no extra `pto.alloc_tile`).  `pto.subview` requires the
result `tile_buf` to share `dtype`, `memory_space`, `blayout`, `slayout`,
`fractal`, and `pad` with the source — `DeduceTileSliceType` propagates
those four `TileView` fields from the source so the produced `TileType`
satisfies the constraints by construction.  Backend codegen also runs a
`CheckSubviewTileCompat` guard at lowering time:

- Source and result must both carry an explicit `TileView`.
- `dtype`, `blayout`, `slayout`, `fractal`, and `pad` must match exactly.
- `pad` must be `PadValue::null` — `pto.subview` is a view, not a fillpad,
  so use `tile.fillpad` on the slice result if zero/min/max padding is
  required.

For `tile.assemble`, the leading `pto.tmov target → dst` is only emitted
when buffer reuse did not collapse `target` and the destination buffer; in
that case it preserves any data outside the insertion window.  The
trailing `pto.tmov src → dst_view` is the actual data write into the
sub-window carved out by `pto.subview`.

### Cross-Core Operations → PTO Instructions

| PyPTO Operation | Generated PTO-ISA | Description |
| --------------- | ----------------- | ----------- |
| `tile.tpush_to_aiv(tile, split=N[, id=I])` | `pto.tpush_to_aiv ins(%tile : type) {[id = I, ]split = N}` | Cube → Vector push |
| `tile.tpush_to_aic(tile, split=N[, id=I])` | `pto.tpush_to_aic ins(%tile : type) {[id = I, ]split = N}` | Vector → Cube push |
| `tile.tpop_from_aic(split=N[, id=I])` | `%buf = pto.tpop_from_aic {[id = I, ]split = N} -> type` | Pop from Cube pipe |
| `tile.tpop_from_aiv(split=N[, id=I])` | `%buf = pto.tpop_from_aiv {[id = I, ]split = N} -> type` | Pop from Vector pipe |
| `system.tfree_to_aic(tile_from_tpop[, id=I])` | `pto.tfree_from_aic {[id = I, ]split = N}` | Release a consumer slot back to Cube |
| `system.tfree_to_aiv(tile_from_tpop[, id=I])` | `pto.tfree_from_aiv {[id = I, ]split = N}` | Release a consumer slot back to Vector |
| `system.aic_initialize_pipe(...)` | `pto.aic_initialize_pipe {[id = I, ]dir_mask = D, slot_size = S} (c2v_consumer_buf = %ssa : i32, v2c_consumer_buf = %ssa : i32)` | Cube pipe init |
| `system.aiv_initialize_pipe(...)` | `pto.aiv_initialize_pipe {[id = I, ]dir_mask = D, slot_size = S} (c2v_consumer_buf = %ssa : i32, v2c_consumer_buf = %ssa : i32)` | Vector pipe init |
| `system.reserve_buffer(...)` | `%name = pto.reserve_buffer {name = "N", size = S, location = #pto.address_space<loc>, auto = false, base = B} -> i32` | Reserve buffer |
| `system.import_peer_buffer(...)` | `%name = pto.import_reserved_buffer {name = "N", peer_func = @F} -> i32` | Import peer buffer |

**Notes:**

- Push ops use an `ins()` clause with a typed tile buffer; frontend pop ops produce an SSA result with a `-> !pto.tile_buf<...>` result type
- `id` is optional. When omitted, PTOAS defaults to frontend pipe id `0`. Use explicit ids only when authoring multiple independent frontend pipes; automatic bidirectional mixed-kernel setup keeps a single `dir_mask = 3` pipe.
- If the pushed tile was allocated with dynamic `valid_row` / `valid_col` operands or updated by
  `tile.set_validshape`, `tpush` emits the same tile handle after its runtime valid shape has been
  updated. For split `tpush`, codegen temporarily uses a full non-split transport dimension (`cols`
  for up/down, `rows` for left/right), then restores the producer tile's logical valid shape;
  consumer-side dynamic tpop operands carry the logical extents used by compute and store.
- When a tpop result `TileView.valid_shape` differs from the physical tile shape, PTO codegen emits PTOAS frontend operands as `%buf = pto.tpop_from_*(%valid_row, %valid_col) {[id = I, ]split = N} -> !pto.tile_buf<..., v_row=?, v_col=?, ...>`. This covers dynamic expressions and static non-full shapes such as `[0, 0]`; the operands carry the logical extents used by compute and store.
- For split consumers, `SplitVectorKernel` localizes those dynamic tpop
  valid-shape operands per subblock (for example global `[8, 16]` becomes
  `[8, 16]` then `[0, 16]` under up/down split of a `[16, 16]` tile).
- `system.tfree_*` derives `split` from its tile argument, so the frontend must free the exact SSA value produced by `tile.tpop_*`, even though the PTO instruction itself does not take the tile as an explicit operand
- `ExpandMixedKernel` now auto-generates consumer-side `system.tfree_*` after split-generated `tile.tpop_*`, preserving `tpop -> direct users -> tfree -> next tpop`
- `reserve_buffer` and `import_reserved_buffer` return `i32` SSA values; `initialize_pipe` references them as operands
- `AllocateMemoryAddr` resolves `reserve_buffer(base=AUTO)` before PTO emission, so PTO always emits `auto = false, base = <value>`
- `reserve_buffer` location is `mat` for AIC functions, `vec` for AIV/InCore functions
- `import_reserved_buffer` uses MLIR symbol syntax (`@func_name`) for `peer_func`
- Buffer name and peer_func strings are validated by `CheckSafeIdentifier` (alphanumeric + underscore only)

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
     shape = [%c32_index, %c32_index]
     strides = [%c32_index, %c1_index]
     {layout = #pto.layout<nd>}
     : !pto.tensor_view<?x?xf32>
```

**Key aspects**:

- Shape from `TensorType.shape_`
- Strides computed as row-major: `[dim1, 1]` for 2D tensors
- Constants (`%c32_index`, `%c1_index`) auto-generated
- Tensor view type uses `?` for each dimension (e.g., `?x?xf32` for 2D)

#### Layout Handling for 2D Tensors

The `layout` attribute on `make_tensor_view` tells PTOAS the memory layout
convention. The codegen determines shape, strides, and layout based on the
tensor's IR type and shape:

| Case | Shape emitted | Strides emitted | Layout | Notes |
| ---- | ------------- | --------------- | ------ | ----- |
| ND `[R, C]` | `[R, C]` | `[C, 1]` | `nd` | Standard row-major |
| DN `[R, C]` (both > 1) | `[C, R]` | `[1, C]` | `dn` | Shape swapped for PTOAS column-major convention |
| Column vector `[M, 1]` | `[M, 1]` | `[1, M]` | `dn` | Auto-detected, no DN annotation needed |

**Column vector auto-DN**: Any 2D tensor whose last dimension is a compile-time
constant `1` (i.e., shape `[M, 1]`) is automatically emitted with `layout = dn`
and strides `[1, M]`. This is required because PTOAS always infers DN for the
shape/stride pattern `[M, 1] / [1, 1]`, making the degenerate ND representation
ambiguous. The codegen resolves this by always using unambiguous DN strides.
Users do not need to annotate `[M, 1]` tensors with `pl.DN` in the DSL.

Example for a `[16, 1]` column vector (no DN annotation in DSL):

```mlir
%col_view = pto.make_tensor_view %arg1,
    shape = [%c16_index, %c1_index], strides = [%c1_index, %c16_index]
    {layout = #pto.layout<dn>}
    : !pto.tensor_view<?x?xf32>
```

### Allocation Generation

Based on TileType variables collected from the function body. Each tile variable gets its own `pto.alloc_tile` instruction with an explicit `addr` attribute derived from the variable's MemRef. Variables sharing the same MemRef share the same address:

```mlir
%mi_tile = pto.alloc_tile addr = %c8320_i64 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major,
                      slayout=none_box, fractal=512, pad=0>
%mi_tile_nd = pto.alloc_tile addr = %c8320_i64 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>
```

**Tile variable → alloc_tile mapping**:

- Memory space (`TileType.memory_space_`) → `loc` attribute (using PTO address space names)
- Tile dtype and dimensions derived from each variable's own TileType metadata
- One allocation per tile variable (not per unique MemRef)
- `addr` attribute from `MemRef.addr_`, emitted as `arith.constant ... : i64`
- Variables sharing the same MemRef produce the same `addr` SSA value

### Load Operation Transformation

**PyPTO IR**:

```python
tile_a = pl.load(tensor_a, [0, 0], [32, 32])
```

**Generated MLIR** (two operations):

```mlir
# 1. Create partition view
%3 = pto.partition_view %tensor_view, offsets = [%c0_index, %c0_index],
                 sizes = [%c32_index, %c32_index]
                 : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>

# 2. Load into tile buffer
pto.tload ins(%3 : !pto.partition_tensor_view<32x32xf32>)
          outs(%tile_buf : !pto.tile_buf<loc=vec, ...>)
```

**Key transformations**:

- Tensor parameter → tensor_view lookup
- Offsets/sizes from `tile.load` arguments
- Output tile_buf from variable's MemRef with type derived from TileType

### Store Operation Transformation

**PyPTO IR**:

```python
pl.store(tile_c, [0, 0], tensor_out)
```

**Generated MLIR**:

```mlir
# 1. Create partition view for output
%5 = pto.partition_view %output_view, offsets = [%c0_index, %c0_index],
                 sizes = [%c32_index, %c32_index]
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
        pl.store(tile_c, [0, 0], c)
```

### Output: PTO-ISA MLIR

```mlir
module {
  func.func @mul_kernel_2d(%arg0: !pto.ptr<f32>,
                          %arg1: !pto.ptr<f32>,
                          %arg2: !pto.ptr<f32>) {
    // Constants
    %c32_index = arith.constant 32 : index
    %c1_index = arith.constant 1 : index
    %c0_index = arith.constant 0 : index

    // Tensor views
    %3 = pto.make_tensor_view %arg0, shape = [%c32_index, %c32_index]
         strides = [%c32_index, %c1_index] : !pto.tensor_view<?x?xf32>
    %4 = pto.make_tensor_view %arg1, shape = [%c32_index, %c32_index]
         strides = [%c32_index, %c1_index] : !pto.tensor_view<?x?xf32>
    %5 = pto.make_tensor_view %arg2, shape = [%c32_index, %c32_index]
         strides = [%c32_index, %c1_index] : !pto.tensor_view<?x?xf32>

    // Allocations
    %0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, ...>
    %1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, ...>
    %2 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, ...>

    // Load tile_a
    %6 = pto.partition_view %3, offsets = [%c0_index, %c0_index], sizes = [%c32_index, %c32_index]
         : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    pto.tload ins(%6 : !pto.partition_tensor_view<32x32xf32>)
              outs(%0 : !pto.tile_buf<...>)

    // Load tile_b
    %7 = pto.partition_view %4, offsets = [%c0_index, %c0_index], sizes = [%c32_index, %c32_index]
         : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    pto.tload ins(%7 : !pto.partition_tensor_view<32x32xf32>)
              outs(%1 : !pto.tile_buf<...>)

    // Multiply
    pto.tmul ins(%0 : !pto.tile_buf<...>, %1 : !pto.tile_buf<...>)
             outs(%2 : !pto.tile_buf<...>)

    // Store tile_c
    %8 = pto.partition_view %5, offsets = [%c0_index, %c0_index], sizes = [%c32_index, %c32_index]
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
- Constants: `%c0_index`, `%c1_index`, `%c32_index`, `%c0_i64`, `%cst`, ...
- Results: `%0`, `%1`, `%2`, ...

### MemRef-based Resolution

For operations like `tile.mul`:

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
| `MemorySpace::Vec` | `vec` (vector buffer) |
| `MemorySpace::Mat` | `mat` (matrix buffer) |
| `MemorySpace::Left` | `left` |
| `MemorySpace::Right` | `right` |
| `MemorySpace::Acc` | `acc` (accumulator) |
| `MemorySpace::Bias` | `bias` (bias buffer) |

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

## Kernel Wrapper Generation (PTO Backend)

When compiling with the PTO backend via `ir.compile()`, a kernel wrapper is automatically generated for each InCore function to bridge the ptoas output to the orchestration calling convention.

### Pipeline

```text
InCore Function → PTOCodegen → .pto → ptoas → .cpp → kernel_wrapper → kernels/aiv/<name>.cpp
```

Each InCore function is compiled independently through ptoas. The final wrapper file combines:

1. **Preprocessed ptoas code** (with `__global__ AICORE` → `static`)
2. **`kernel_entry(__gm__ int64_t* args)`** wrapper that unpacks the args array and forwards to the ptoas function

### Output Structure

When the program contains an Orchestration function, the PTO backend generates the following output structure:

```text
output_dir/
├── passes_dump/                     # IR after each pass
├── ptoas/                           # Intermediates
│   ├── <func_name>.pto              # MLIR from PTOCodegen
│   └── <func_name>.cpp              # C++ from ptoas
├── kernels/aiv/
│   └── <func_name>.cpp              # Final wrapper
├── orchestration/
│   └── <orch_func_name>.cpp         # PTO2 runtime orchestration code
└── kernel_config.py                 # Runtime/orchestration/kernel config
```

The orchestration codegen generates identical orchestration C++ code using the PTO2 runtime API (`rt_submit_task`, `make_tensor_external`, etc.).

### Argument Unpacking

The wrapper unpacks `int64_t* args` following the standard convention:

| Parameter Type | Unpacking Pattern |
| -------------- | ----------------- |
| `TensorType` | `Tensor*` → `buffer.addr` → typed pointer |
| `ScalarType` | `uint64_t` → union decode → typed value |

### Implementation

**Module**: `python/pypto/backend/pto_backend.py`

Key functions:

- `generate()` — entry point: produces all PTO backend files (kernels + orchestration + config)
- `_preprocess_ptoas_output()` — strips duplicate includes, makes functions static
- `_generate_arg_unpacking()` — generates C++ unpacking code from IR parameter types
- `_generate_kernel_wrapper()` — assembles the complete wrapper file

## See Also

- [Pass Manager](../passes/00-pass_manager.md): Understanding pass pipeline
- [IR Builder](../ir/06-builder.md): Constructing IR programmatically
- [Operator Organization](../ir/05-operators.md): Block operation details
