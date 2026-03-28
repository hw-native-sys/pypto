# Orchestration Code Generation

## Overview

The orchestration codegen generates PTO2 runtime C++ code that manages task-graph execution on Ascend hardware. While [PTO codegen](00-pto_codegen.md) produces InCore kernel code (tile-level compute), orchestration codegen produces the host-side code that:

- Wraps device memory pointers (via `TaskArg`) into `Tensor` objects
- Builds `PTOParam` objects and calls `add_input`/`add_output`/`add_inout`/`add_scalar` to classify parameters
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
- **ForStmt** → `for` loop with iter_arg initialization and yield updates
- **IfStmt** → conditional blocks with `PTO2_SCOPE` per branch and return variable handling
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

`GenerateOrchestration()` produces C++ in 9 phases:

### Phase 1–2: Boilerplate

```cpp
// Phase 1: Includes
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "pto_orchestration_api.h"

// Phase 2: Helper functions
static uint64_t float_to_u64(float f) { ... }
static inline Tensor make_tensor_external_2d_dn(...) { ... }
static inline Tensor make_tensor_2d_dn(...) { ... }
```

### Phase 3–4: Entry Points

```cpp
// Phase 3: Config function — returns expected argument count
PTO2OrchestrationConfig aicpu_orchestration_config(TaskArg* orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{ .expected_arg_count = 3 };
}

// Phase 4: Entry function signature
// A2/A3:
void aicpu_orchestration_entry(TaskArg* orch,
    int arg_count, int orch_thread_num, int orch_thread_index) {
// A5 (Ascend950): PTO2Runtime* rt prepended
// void aicpu_orchestration_entry(PTO2Runtime* rt, TaskArg* orch, ...)
```

### Phase 5–6: Tensor Setup

```cpp
// Phase 5: External tensors — ND layout via from_task_arg()
Tensor ext_a = from_task_arg(orch[0]);
Tensor ext_b = from_task_arg(orch[1]);

// DN layout: pass logical shape — make_tensor_external_2d_dn handles axis transposition internally
uint32_t dn_shapes[2] = {orch[2].tensor.shapes[0], orch[2].tensor.shapes[1]};
Tensor ext_dn = make_tensor_external_2d_dn(orch[2].data<void>(), dn_shapes, 2, DataType::FLOAT32);

// Phase 6: Internal tensors (from pl.create_tensor — intermediates only)
uint32_t tmp_shapes[2] = {16, 16};
Tensor tmp = make_tensor(tmp_shapes, 2, DataType::FLOAT32);
```

### Phase 7–9: Task Submission and Control Flow

All task submission is wrapped in a top-level `PTO2_SCOPE()`:

```cpp
// Phase 7–9: Top-level PTO2_SCOPE wraps all task submissions
// A2/A3: PTO2_SCOPE()   A5: PTO2_SCOPE(rt)
PTO2_SCOPE() {
    PTOParam params_t0;
    params_t0.add_input(ext_a);
    params_t0.add_input(ext_b);
    params_t0.add_output(ext_output);
    // A2/A3: pto2_rt_submit_aiv_task(0, params_t0)
    // A5:    pto2_rt_submit_aiv_task(rt, 0, params_t0)
    pto2_rt_submit_aiv_task(0, params_t0);

    // ForStmt example — plain for loop, no nested PTO2_SCOPE
    for (int64_t i = start; i < stop; i += step) {
        // task submissions
    }
}
```

## Key Concepts

### External vs Internal Tensors

| Type | Source | C++ Construction | Naming |
| ---- | ------ | ---------------- | ------ |
| External (ND) | Function parameters (`In`/`Out`/`InOut`) | `from_task_arg(orch[N])` | `ext_<name>` |
| External (DN) | Function parameters, DN layout | `make_tensor_external_2d_dn(orch[N].data<void>(), {orch[N].tensor.shapes[0], orch[N].tensor.shapes[1]}, ...)` — axis ordering handled internally | `ext_<name>` |
| Internal | `pl.create_tensor(...)` in function body | `make_tensor(shapes, ndims, dtype)` | `<name>` (no prefix) |

External tensors wrap device memory pointers passed from the host via `TaskArg`. Internal tensors are temporary workspace allocated by the runtime.

### Parameter Direction

The `ParamDirection` of each function parameter determines how it appears in task submission:

| Direction | Python Annotation | C++ Task Param | Semantics |
| --------- | ----------------- | -------------- | --------- |
| `In` | `pl.Tensor[...]` (default) | `params.add_input(ext_x)` | Read-only |
| `Out` | `pl.Out[pl.Tensor[...]]` | `params.add_output(ext_x)` | Write-only |
| `InOut` | `pl.InOut[pl.Tensor[...]]` | `params.add_inout(ext_x)` | Read-write |
| Scalar | `pl.Scalar[...]` | `params.add_scalar(value)` | Scalar constant (after all tensors) |

### Alias Generation

When an InCore call returns a value with a different name than the `Out` argument, the codegen emits a C++ reference alias:

```python
# Python IR
result = self.kernel_add(a, b, output)  # result ≠ output
```

```cpp
// Generated C++
PTOParam params_t0;
params_t0.add_output(ext_output);
pto2_rt_submit_aiv_task(0, params_t0);
Tensor& result = ext_output;  // alias — result refers to ext_output
```

If the return name matches the `Out`/`InOut` arg name, no alias is needed. `InOut` params are never aliased — they are already the external tensor.

### Backend Differences (A5 / Ascend950)

When targeting Ascend950 (`BackendType::Ascend950`), the generated C++ differs in three ways:

| Element | A2/A3 | A5 |
| ------- | ----- | -- |
| Entry function | `aicpu_orchestration_entry(TaskArg* orch, ...)` | `aicpu_orchestration_entry(PTO2Runtime* rt, TaskArg* orch, ...)` |
| Scope macro | `PTO2_SCOPE()` | `PTO2_SCOPE(rt)` |
| Submit calls | `pto2_rt_submit_aiv_task(id, params)` | `pto2_rt_submit_aiv_task(rt, id, params)` |

The `rt` parameter is an explicit `PTO2Runtime*` pointer that A5 passes through the call chain instead of relying on thread-local storage.

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
// Generated C++ — tensors first, then scalars
PTOParam params_t0;
params_t0.add_input(ext_sij);
params_t0.add_output(ext_pij);
params_t0.add_output(ext_mij);
params_t0.add_output(ext_lij);
params_t0.add_scalar(float_to_u64(scale));  // scalar after all tensors
pto2_rt_submit_aiv_task(0, params_t0);
```

### Group Functions (Mixed Kernels)

When a kernel uses both AIC and AIV cores (mixed kernel), the codegen generates `MixedKernels` submission:

```cpp
// Group: mixed_kernel (AIC + AIV)
PTOParam params_t0;
// ... add_input / add_output / add_scalar calls ...
MixedKernels mixed_0 = {aic_id, aiv_id, INVALID_KERNEL_ID};
// A2/A3: pto2_rt_submit_task(mixed_0, params_t0)
// A5:    pto2_rt_submit_task(rt, mixed_0, params_t0)
pto2_rt_submit_task(mixed_0, params_t0);
```

## Operation Mappings

| IR Operation | C++ Codegen | Description |
| ------------ | ----------- | ----------- |
| `tensor.create` | `make_tensor(shapes, ndims, dtype)` | Allocate internal tensor |
| `tensor.read` | `*reinterpret_cast<T*>(arg_ptr + offset)` | Read scalar from host tensor |
| `tensor.slice` | `make_tensor_external(ptr + byte_offset, ...)` | Create view into existing tensor |
| `tensor.dim` (static) | `int64_t d0 = 16` | Constant dimension value |
| `tensor.dim` (dynamic) | `int64_t d0 = (int64_t)orch[N].tensor.shapes[axis]` | Runtime dimension from TaskArg |

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

static uint64_t float_to_u64(float f) { /* ... */ }

extern "C" {

PTO2OrchestrationConfig aicpu_orchestration_config(TaskArg* orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{ .expected_arg_count = 3 };
}

void aicpu_orchestration_entry(TaskArg* orch,
    int arg_count, int orch_thread_num, int orch_thread_index) {
    // Note: A5 adds PTO2Runtime* rt as the first parameter
    (void)arg_count;
    (void)orch_thread_num;
    (void)orch_thread_index;

    // External tensors (from TaskArg)
    Tensor ext_a = from_task_arg(orch[0]);
    Tensor ext_b = from_task_arg(orch[1]);
    Tensor ext_d = from_task_arg(orch[2]);

    // Internal tensor (intermediate)
    uint32_t c_shapes[2] = {16, 16};
    Tensor c = make_tensor(c_shapes, 2, DataType::FLOAT32);

    // A2/A3: PTO2_SCOPE()   A5: PTO2_SCOPE(rt)
    PTO2_SCOPE() {
        // Task 0: kernel_add (a + b → c)
        PTOParam params_t0;
        params_t0.add_input(ext_a);
        params_t0.add_input(ext_b);
        params_t0.add_output(c);
        // A2/A3: pto2_rt_submit_aiv_task(0, params_t0)
        // A5:    pto2_rt_submit_aiv_task(rt, 0, params_t0)
        pto2_rt_submit_aiv_task(0, params_t0);

        // Task 1: kernel_add (c + b → d)
        PTOParam params_t1;
        params_t1.add_input(c);
        params_t1.add_input(ext_b);
        params_t1.add_output(ext_d);
        pto2_rt_submit_aiv_task(1, params_t1);
    }
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
| External tensor | `ext_<name>` | `ext_a` |
| Internal tensor | `<name>` (no prefix) | `c` |
| Task params | `params_t<N>` | `params_t0` |
| TaskArg index | `orch[N]` (N-th tensor parameter) | `orch[0]` |

## Control Flow Generation

### ForStmt

```python
# Python IR
for i in pl.range(0, 4):
    acc = self.kernel_add(a, acc, acc)
```

```cpp
// Generated C++ (inside top-level PTO2_SCOPE)
Tensor acc = ext_acc;  // iter_arg initialization
for (int64_t i = 0; i < 4; i += 1) {
    PTOParam params_t0;
    // ... add_input / add_output calls ...
    pto2_rt_submit_aiv_task(0, params_t0);
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
    // A2/A3: PTO2_SCOPE()   A5: PTO2_SCOPE(rt)
    PTO2_SCOPE() {
        PTOParam params_t0;
        // ... add_input / add_output calls ...
        pto2_rt_submit_aiv_task(0, params_t0);
    }
} else {
    PTO2_SCOPE() {
        PTOParam params_t1;
        // ... add_input / add_output calls ...
        pto2_rt_submit_aiv_task(1, params_t1);
    }
}
```

## Python API

```python
from pypto import codegen, backend

backend.set_backend_type(backend.BackendType.Ascend910B)
result = codegen.generate_orchestration(MyProgram, orch_func)
code = result.code

# Access generated orchestration code
orch_code = files["orchestration/orch_func_name.cpp"]
```

The orchestration file is named `orchestration/<func_name>.cpp` in the generated file map.

## See Also

- [PTO Codegen](00-pto_codegen.md) — MLIR generation for PTO backend
- [Pass Manager](../passes/00-pass_manager.md) — IR optimization passes applied before codegen
