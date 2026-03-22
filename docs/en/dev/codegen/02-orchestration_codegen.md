# Orchestration Code Generation

## Overview

The orchestration codegen generates PTO2 runtime C++ code that manages task-graph execution on Ascend hardware. While [CCE codegen](01-cce_codegen.md) produces InCore kernel code (tile-level compute), orchestration codegen produces the host-side code that:

- Wraps device memory pointers into `Tensor` objects
- Builds `PTOParam` arrays classifying each tensor as input/output/inout
- Submits tasks to AIC (CUBE) or AIV (VECTOR) cores via `pto2_rt_submit_*_task`
- Handles control flow (loops, conditionals) with `PTO2_SCOPE`

**Pipeline:** `IR (Orchestration function) → OrchestrationCodegen → C++ (PTO2 runtime API)`

**Location:** `src/codegen/orchestration/orchestration_codegen.cpp`

## Architecture

### Component Structure

| Component | Purpose | Location |
| --------- | ------- | -------- |
| `OrchestrationInfoCollector` | IR visitor collecting metadata (tuple maps, tensor assignments) | orchestration_codegen.cpp |
| `OrchestrationStmtCodegen` | Statement-level C++ code generator (extends CodegenBase) | orchestration_codegen.cpp |
| `OrchestrationOpRegistry` | Singleton registry for tensor operation codegen handlers | orchestration_op_registry.h |
| `GenerateOrchestration()` | Main entry point combining all generation phases | orchestration_codegen.cpp |
| `VarLineageCollector` | Traces body variables back to function params via VarPtr identity | orchestration_codegen.cpp |
| `GetSSABaseName()` | Strips SSA/pass-pipeline suffixes for C++ name emission (not identity) | orchestration_codegen.cpp |

### OrchestrationInfoCollector

An IR visitor that pre-scans the function body to collect:

- **Tuple element maps** — tracks which variables come from tuple decomposition
- **Call-to-tuple keys** — unique keys (`_tc_N`) preventing cross-call collisions
- **Output tensor assignments** — maps variable names to their assignment statements

### OrchestrationStmtCodegen

The main code generator. Visits each IR statement and emits corresponding C++:

- **AssignStmt** → tensor operations, function calls, or alias generation
- **ForStmt** → `PTO2_SCOPE` with iter_arg initialization and yield updates
- **IfStmt** → conditional blocks with return variable handling
- **YieldStmt** → variable reassignment for loop-carried values

### Operation Registry

Tensor operations are registered via `REGISTER_ORCHESTRATION_OP` macro:

```cpp
REGISTER_ORCHESTRATION_OP("tensor.create", TensorCreateHandler);
REGISTER_ORCHESTRATION_OP("tensor.read", TensorReadHandler);
REGISTER_ORCHESTRATION_OP("tensor.slice", TensorSliceHandler);
```

This allows extensible operation codegen without modifying the core visitor.

## Code Generation Flow

`GenerateOrchestration()` produces C++ in 10 phases:

### Phase 1–3: Boilerplate

```cpp
// Phase 1: Includes
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "pto_orchestration_api.h"

// Phase 2: ARG defines (one per tensor parameter)
#define ARG_PTR_A 0
#define ARG_PTR_B 1
#define ARG_PTR_OUTPUT 2

// Phase 3: Helper functions
static uint64_t float_to_u64(float f) { ... }
static inline Tensor make_tensor_external_2d_dn(...) { ... }
static inline Tensor make_tensor_2d_dn(...) { ... }
```

### Phase 4–5: Entry Points

```cpp
// Phase 4: Config function — returns expected argument count
PTO2OrchestrationConfig aicpu_orchestration_config(uint64_t* args, int arg_count) {
    return PTO2OrchestrationConfig{ .expected_arg_count = 3 };
}

// Phase 5: Entry function signature
void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args,
    int arg_count, int orch_thread_num, int orch_thread_index) {
```

### Phase 6–8: Tensor Setup

```cpp
// Phase 6: Extract device pointers from args array
void* arg_a_ptr = reinterpret_cast<void*>(args[ARG_PTR_A]);
void* arg_b_ptr = reinterpret_cast<void*>(args[ARG_PTR_B]);
void* arg_output_ptr = reinterpret_cast<void*>(args[ARG_PTR_OUTPUT]);

// Phase 7: External tensors (from function parameters)
uint64_t a_shapes[2] = {16, 16};
Tensor ext_a = make_tensor_external(arg_a_ptr, a_shapes, 2, DataType::FLOAT32);

// Phase 8: Internal tensors (from pl.create_tensor — intermediates only)
uint64_t tmp_shapes[2] = {16, 16};
Tensor tmp = make_tensor(tmp_shapes, 2, DataType::FLOAT32);
```

### Phase 9–10: Task Submission and Control Flow

```cpp
// Phase 9: Task submission
PTOParam params_t0[] = {
    make_input_param(ext_a),
    make_input_param(ext_b),
    make_output_param(ext_output),
};
pto2_rt_submit_aiv_task(rt, 0, params_t0, 3);

// Phase 10: Control flow (ForStmt example)
PTO2_SCOPE {
    for (int64_t i = start; i < stop; i += step) {
        // task submissions inside loop
    }
}
```

## Key Concepts

### External vs Internal Tensors

| Type | Source | C++ Function | Naming |
| ---- | ------ | ------------ | ------ |
| External | Function parameters (`In`/`Out`/`InOut`) | `make_tensor_external(ptr, shapes, ndims, dtype)` | `ext_<name>` |
| Internal | `pl.create_tensor(...)` in function body | `make_tensor(shapes, ndims, dtype)` | `<name>` (no prefix) |

External tensors wrap device memory pointers passed from the host. Internal tensors are temporary workspace allocated by the runtime.

### Parameter Direction

The `ParamDirection` of each function parameter determines how it appears in task submission:

| Direction | Python Annotation | C++ Task Param | Semantics |
| --------- | ----------------- | -------------- | --------- |
| `In` | `pl.Tensor[...]` (default) | `make_input_param(ext_x)` | Read-only |
| `Out` | `pl.Out[pl.Tensor[...]]` | `make_output_param(ext_x)` | Write-only |
| `InOut` | `pl.InOut[pl.Tensor[...]]` | `make_inout_param(ext_x)` | Read-write |
| Scalar | `pl.Scalar[...]` | `make_scalar_param(value)` | Scalar constant |

### Alias Generation

When an InCore call returns a value with a different name than the `Out`/`InOut` argument, the codegen emits a C++ reference alias:

```python
# Python IR
result = self.kernel_add(a, b, output)  # result ≠ output
```

```cpp
// Generated C++
PTOParam params_t0[] = { ... make_output_param(ext_output) ... };
pto2_rt_submit_aiv_task(rt, 0, params_t0, 3);
Tensor& result = ext_output;  // alias — result refers to ext_output
```

If the return name matches the `Out`/`InOut` arg name, no alias is needed.

### Core Type Inference

The codegen determines whether to submit to AIC (CUBE) or AIV (VECTOR) based on the callee's `MemorySpace`:

| MemorySpace | Core Type | Submit Function |
| ----------- | --------- | --------------- |
| `Left`, `Right`, `Acc` | CUBE (AIC) | `pto2_rt_submit_aic_task` |
| `Vec`, `Mat` (default) | VECTOR (AIV) | `pto2_rt_submit_aiv_task` |

### Tuple Handling

Tuple-returning calls use unique keys (`_tc_N`) to track elements:

```python
# Python IR
pij, mij, lij = self.kernel_softmax(sij, scale, pij, mij, lij)
```

```cpp
// Generated C++ — each element maps to its Out/InOut arg
PTOParam params_t0[] = {
    make_input_param(ext_sij),
    make_scalar_param(float_to_u64(scale)),
    make_output_param(ext_pij),
    make_output_param(ext_mij),
    make_output_param(ext_lij),
};
pto2_rt_submit_aiv_task(rt, 0, params_t0, 5);
```

### Group Functions (Mixed Kernels)

When a kernel uses both AIC and AIV cores (mixed kernel), the codegen generates `MixedKernels` submission:

```cpp
// Group: mixed_kernel (AIC + AIV)
PTOParam params_t0[] = { ... };
MixedKernels mixed_0 = {aic_id, aiv_id, INVALID_KERNEL_ID};
pto2_rt_submit_task(rt, mixed_0, params_t0, param_count);
```

## Operation Mappings

| IR Operation | C++ Codegen | Description |
| ------------ | ----------- | ----------- |
| `tensor.create` | `make_tensor(shapes, ndims, dtype)` | Allocate internal tensor |
| `tensor.read` | `*reinterpret_cast<T*>(arg_ptr + offset)` | Read scalar from host tensor |
| `tensor.slice` | `make_tensor_external(ptr + byte_offset, ...)` | Create view into existing tensor |
| `tensor.dim` | `int64_t d0 = <compile_time_value>` | Extract dimension (compile-time) |

## Complete Example

### Input: PyPTO Orchestration Function

```python
@pl.function(type=pl.FunctionType.Orchestration)
def orch_basic(
    self,
    a: pl.Tensor[[16, 16], pl.FP32],
    b: pl.Tensor[[16, 16], pl.FP32],
    d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
) -> pl.Tensor[[16, 16], pl.FP32]:
    c: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
    c = self.kernel_add(a, b, c)       # c is internal (intermediate)
    d = self.kernel_add(c, b, d)       # d is external (Out param)
    return d
```

### Output: Generated C++

```cpp
// Orchestration Function: orch_basic
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "pto_orchestration_api.h"

#define ARG_PTR_A 0
#define ARG_PTR_B 1
#define ARG_PTR_D 2

static uint64_t float_to_u64(float f) { /* ... */ }

extern "C" {

PTO2OrchestrationConfig aicpu_orchestration_config(uint64_t* args, int arg_count) {
    return PTO2OrchestrationConfig{ .expected_arg_count = 3 };
}

void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args,
    int arg_count, int orch_thread_num, int orch_thread_index) {

    // Extract device pointers
    void* arg_a_ptr = reinterpret_cast<void*>(args[ARG_PTR_A]);
    void* arg_b_ptr = reinterpret_cast<void*>(args[ARG_PTR_B]);
    void* arg_d_ptr = reinterpret_cast<void*>(args[ARG_PTR_D]);

    // External tensors (from params)
    uint64_t a_shapes[2] = {16, 16};
    Tensor ext_a = make_tensor_external(arg_a_ptr, a_shapes, 2, DataType::FLOAT32);
    uint64_t b_shapes[2] = {16, 16};
    Tensor ext_b = make_tensor_external(arg_b_ptr, b_shapes, 2, DataType::FLOAT32);
    uint64_t d_shapes[2] = {16, 16};
    Tensor ext_d = make_tensor_external(arg_d_ptr, d_shapes, 2, DataType::FLOAT32);

    // Internal tensor (intermediate)
    uint64_t c_shapes[2] = {16, 16};
    Tensor c = make_tensor(c_shapes, 2, DataType::FLOAT32);

    // Task 0: kernel_add (a + b → c)
    PTOParam params_t0[] = {
        make_input_param(ext_a),
        make_input_param(ext_b),
        make_output_param(c),
    };
    pto2_rt_submit_aiv_task(rt, 0, params_t0, 3);

    // Task 1: kernel_add (c + b → d)
    PTOParam params_t1[] = {
        make_input_param(c),
        make_input_param(ext_b),
        make_output_param(ext_d),
    };
    pto2_rt_submit_aiv_task(rt, 0, params_t1, 3);
}

}  // extern "C"
```

## Variable Naming

### Variable Identity via VarPtr Lineage

Variable identity decisions (is this var a param? are two vars the same tensor?) use
`VarPtr`-based pointer identity, not string matching. `VarLineageCollector` walks the
function body before codegen and traces each body `Var*` back to its originating
function parameter `Var*` through ForStmt iter_arg/return_var chains and simple
Var-to-Var assignments. This avoids the name-collision bugs where suffix stripping
(e.g., `out_0` → `out`) would merge distinct variables.

`GetSSABaseName()` is still used for C++ code emission (generating clean variable
names in the output), but never for identity decisions.

### Naming Conventions

| Entity | Pattern | Example |
| ------ | ------- | ------- |
| ARG define | `ARG_PTR_<UPPER>` | `ARG_PTR_A` |
| Device pointer | `arg_<name>_ptr` | `arg_a_ptr` |
| External tensor | `ext_<name>` | `ext_a` |
| Internal tensor | `<name>` (no prefix) | `c` |
| Task params | `params_t<N>` | `params_t0` |

## Control Flow Generation

### ForStmt

```python
# Python IR
for i in pl.range(0, 4):
    acc = self.kernel_add(a, acc, acc)
```

```cpp
// Generated C++
Tensor acc = ext_acc;  // iter_arg initialization
PTO2_SCOPE {
    for (int64_t i = 0; i < 4; i += 1) {
        PTOParam params_t0[] = { ... };
        pto2_rt_submit_aiv_task(rt, 0, params_t0, 3);
    }
}
```

Iter_args are initialized before the loop. `YieldStmt` updates are emitted at the end of each iteration.

### IfStmt

```python
# Python IR
if condition:
    c = self.kernel_a(a, b, c)
else:
    c = self.kernel_b(a, b, c)
```

```cpp
// Generated C++
if (condition) {
    PTOParam params_t0[] = { ... };
    pto2_rt_submit_aiv_task(rt, 0, params_t0, 3);
} else {
    PTOParam params_t1[] = { ... };
    pto2_rt_submit_aiv_task(rt, 1, params_t1, 3);
}
```

## Python API

```python
from pypto import codegen, backend

backend.set_backend_type(backend.BackendType.Ascend910B_CCE)
generator = codegen.CCECodegen()
files = generator.generate(MyProgram)

# Access generated orchestration code
orch_code = files["orchestration/orch_func_name.cpp"]
```

The orchestration file is named `orchestration/<func_name>.cpp` in the generated file map.

## See Also

- [PTO Codegen](00-pto_codegen.md) — MLIR generation for PTO backend
- [CCE Codegen](01-cce_codegen.md) — C++ code generation for InCore kernels
- [Pass Manager](../passes/00-pass_manager.md) — IR optimization passes applied before codegen
