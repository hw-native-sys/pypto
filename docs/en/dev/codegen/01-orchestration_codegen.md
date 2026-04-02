# Orchestration Code Generation

## Design Principle: Strict 1-to-1 Mapping

Orchestration codegen follows the same principle as [PTO codegen](00-pto_codegen.md#design-principle-strict-1-to-1-mapping): a **strict 1-to-1 translation** from IR to generated C++ code. The codegen should not perform optimization, analysis, or indirection — such work belongs in earlier passes.

For example, return-to-parameter tracing (mapping callee return values back to `Out` parameters) is analysis that should be resolved by a pass before codegen sees the IR. The `NormalizeReturnOrder` pass ([passes](../passes/00-pass_manager.md)) now canonicalizes this before codegen, so orchestration codegen maps `return[i]` directly to `out_indices[i]` without tracing through `tile.store`/yield chains.

## Overview

The orchestration codegen generates PTO2 runtime C++ code that manages task-graph execution on Ascend hardware. While [PTO codegen](00-pto_codegen.md) produces InCore kernel code (tile-level compute), orchestration codegen produces the host-side code that:

- Wraps device memory pointers (via `ChipStorageTaskArgs`) into `Tensor` objects
- Builds `Arg` objects and calls `add_input`/`add_output`/`add_inout`/`add_scalar` to classify parameters
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
PTO2OrchestrationConfig aicpu_orchestration_config(const ChipStorageTaskArgs& orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{ .expected_arg_count = 3 };
}

// Phase 4: Entry function signature
void aicpu_orchestration_entry(const ChipStorageTaskArgs& orch_args,
    int orch_thread_num, int orch_thread_index) {
```

### Phase 5–6: Tensor Setup

```cpp
// Phase 5: External tensors — ND layout via from_tensor_arg()
Tensor ext_a = from_tensor_arg(orch_args.tensor(0));
Tensor ext_b = from_tensor_arg(orch_args.tensor(1));

// DN layout: pass logical shape — make_tensor_external_2d_dn handles axis transposition internally
uint32_t dn_shapes[2] = {orch_args.tensor(2).shapes[0], orch_args.tensor(2).shapes[1]};
Tensor ext_dn = make_tensor_external_2d_dn(orch_args.tensor(2).data_as<void>(), dn_shapes, 2, DataType::FLOAT32);

// Phase 6: Internal tensors (from pl.create_tensor — intermediates only)
// Only TensorCreateInfo is emitted here. The Tensor variable is bound at the submit site:
//   non-DN: const Tensor& tmp = outs_t0.get_ref(0);   (declared at submit)
//   DN:     Tensor tmp = make_tensor_2d_dn(...);        (declared here, copy-assigned at submit)
uint32_t tmp_ci_shapes[2] = {16, 16};
TensorCreateInfo tmp_ci(tmp_ci_shapes, 2, DataType::FLOAT32);
```

### Phase 7–9: Task Submission and Control Flow

All task submission is wrapped in a top-level `PTO2_SCOPE()`:

```cpp
PTO2_SCOPE() {
    Arg params_t0;
    params_t0.add_input(ext_a);
    params_t0.add_input(ext_b);
    params_t0.add_output(tmp_ci);            // add_output takes TensorCreateInfo for internal tensors
    TaskOutputTensors outs_t0 = pto2_rt_submit_aiv_task(0, params_t0);
    const Tensor& tmp = outs_t0.get_ref(0); // non-DN: bind const ref at submit site

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
| External (ND) | Function parameters (`In`/`Out`/`InOut`) | `from_tensor_arg(orch_args.tensor(N))` | `ext_<name>` |
| External (DN) | Function parameters, DN layout | `make_tensor_external_2d_dn(orch_args.tensor(N).data_as<void>(), {...}, ...)` | `ext_<name>` |
| Internal (non-DN) | `pl.create_tensor(...)` in function body | `TensorCreateInfo var_ci(...)` + `const Tensor& var` bound at submit | `<name>` (no prefix) |
| Internal (DN) | `pl.create_tensor(...)` with DN layout | `TensorCreateInfo var_ci(...)` + `Tensor var = make_tensor_2d_dn(...)` | `<name>` (no prefix) |

External tensors wrap device memory pointers passed from the host via `ChipStorageTaskArgs`. Internal tensors are allocated by the runtime from a ring buffer when passed to `add_output(var_ci)`.

### Parameter Direction

The `ParamDirection` of each function parameter determines how it appears in task submission:

| Direction | Python Annotation | C++ Task Param | Semantics |
| --------- | ----------------- | -------------- | --------- |
| `In` | `pl.Tensor[...]` (default) | `params.add_input(ext_x)` | Read-only; if the tensor is from `pl.create_tensor`, uses `add_output` for first-time allocation |
| `Out` (external) | `pl.Out[pl.Tensor[...]]` (param) | `params.add_inout(ext_x)` | Pre-allocated buffer |
| `Out` (internal) | `pl.Out[pl.Tensor[...]]` (tensor.create) | `params.add_output(x_ci)` + `const Tensor& x = outs.get_ref(i)` (non-DN) or `x = outs.get_ref(i)` (DN) | Runtime ring-buffer alloc |
| `InOut` | `pl.InOut[pl.Tensor[...]]` | `params.add_inout(ext_x)` | Read-write |
| Scalar | `pl.Scalar[...]` | `params.add_scalar(value)` | Scalar constant (separate scalar slot) |

When `add_output` is used, the submit call returns `TaskOutputTensors`. For non-DN tensors,
a `const Tensor& var = outs.get_ref(i)` binding is declared at the submit site. For DN tensors,
a `Tensor var = make_tensor_2d_dn(...)` placeholder (null data pointer, with logical view) is
pre-declared at the `tensor.create` site and copy-assigned after submit.

### Scalar Parameter Encoding

Scalar params occupy `ChipStorageTaskArgs` scalar slots (0-indexed, separate from tensor slots).
Float scalars use `float_to_u64(f)` (bit-cast). Other integer/bool scalars are cast to `uint64_t`.
At the receiving end, union-based type punning is used to reinterpret the `uint64_t` as the target C type:

```cpp
union { uint64_t u64; float val; } scale_conv;
scale_conv.u64 = orch_args.scalar(0);
float scale = scale_conv.val;
```

### Alias Generation

When an InCore call returns a value with a different name than the `Out` argument, the codegen emits a C++ reference alias:

```python
# Python IR
result = self.kernel_add(a, b, output)  # result ≠ output
```

```cpp
// Generated C++
Arg params_t0;
params_t0.add_inout(ext_output);
pto2_rt_submit_aiv_task(0, params_t0);
const Tensor& result = ext_output;  // alias — result refers to ext_output
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
// Generated C++ — tensors first, then scalars
Arg params_t0;
params_t0.add_input(ext_sij);
params_t0.add_inout(ext_pij);
params_t0.add_inout(ext_mij);
params_t0.add_inout(ext_lij);
params_t0.add_scalar(float_to_u64(scale));  // scalar after all tensors
pto2_rt_submit_aiv_task(0, params_t0);
```

### Group Functions (Mixed Kernels)

When a kernel uses both AIC and AIV cores (mixed kernel), the codegen generates `MixedKernels` submission:

```cpp
// Group: mixed_kernel (AIC + AIV)
Arg params_t0;
// ... add_input / add_inout / add_scalar calls ...
MixedKernels mixed_0 = {aic_id, aiv_id, INVALID_KERNEL_ID};
pto2_rt_submit_task(mixed_0, params_t0);
```

## Operation Mappings

| IR Operation | C++ Codegen | Description |
| ------------ | ----------- | ----------- |
| `tensor.create` | `TensorCreateInfo var_ci(...)` | Ring-buffer alloc info; non-DN: `const Tensor& var` bound at submit; DN: `Tensor var = make_tensor_2d_dn(...)` declared before submit |
| `tensor.read` | `*reinterpret_cast<T*>(arg_ptr + offset)` | Read scalar from host tensor |
| `tensor.slice` | `make_tensor_external(ptr + byte_offset, ...)` | Create view into existing tensor |
| `tensor.dim` (static) | `int64_t d0 = 16` | Constant dimension value |
| `tensor.dim` (dynamic) | `int64_t d0 = (int64_t)orch_args.tensor(N).shapes[axis]` | Runtime dimension from ChipStorageTaskArgs |

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

PTO2OrchestrationConfig aicpu_orchestration_config(const ChipStorageTaskArgs& orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{ .expected_arg_count = 3 };
}

void aicpu_orchestration_entry(const ChipStorageTaskArgs& orch_args,
    int orch_thread_num, int orch_thread_index) {
    (void)orch_thread_num;
    (void)orch_thread_index;

    // External tensors (from ChipStorageTaskArgs)
    Tensor ext_a = from_tensor_arg(orch_args.tensor(0));
    Tensor ext_b = from_tensor_arg(orch_args.tensor(1));
    Tensor ext_d = from_tensor_arg(orch_args.tensor(2));

    // Internal tensor (intermediate) — only TensorCreateInfo declared here
    uint32_t c_ci_shapes[2] = {16, 16};
    TensorCreateInfo c_ci(c_ci_shapes, 2, DataType::FLOAT32);

    PTO2_SCOPE() {
        // Task 0: kernel_add (a + b → c)
        Arg params_t0;
        params_t0.add_input(ext_a);
        params_t0.add_input(ext_b);
        params_t0.add_output(c_ci);
        TaskOutputTensors outs_t0 = pto2_rt_submit_aiv_task(0, params_t0);
        const Tensor& c = outs_t0.get_ref(0);  // non-DN: bind const ref at submit site

        // Task 1: kernel_add (c + b → d)
        Arg params_t1;
        params_t1.add_input(c);
        params_t1.add_input(ext_b);
        params_t1.add_inout(ext_d);
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
| Internal TensorCreateInfo | `<name>_ci` | `c_ci` |
| Task params | `params_t<N>` | `params_t0` |
| TaskOutputTensors | `outs_t<N>` | `outs_t0` |
| Tensor arg index | `orch_args.tensor(N)` | `orch_args.tensor(0)` |
| Scalar arg index | `orch_args.scalar(N)` | `orch_args.scalar(0)` |

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
    Arg params_t0;
    // ... add_input / add_inout calls ...
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
    PTO2_SCOPE() {
        Arg params_t0;
        // ... add_input / add_inout calls ...
        pto2_rt_submit_aiv_task(0, params_t0);
    }
} else {
    PTO2_SCOPE() {
        Arg params_t1;
        // ... add_input / add_inout calls ...
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
