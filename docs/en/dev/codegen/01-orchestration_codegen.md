# Orchestration Code Generation

## Design Principle: Strict 1-to-1 Mapping

Orchestration codegen follows the same principle as [PTO codegen](00-pto_codegen.md#design-principle-strict-1-to-1-mapping): a **strict 1-to-1 translation** from IR to generated C++ code. The codegen should not perform optimization, analysis, or indirection — such work belongs in earlier passes.

For example, return-to-parameter tracing (mapping callee return values back to `Out` parameters) is analysis that should be resolved by a pass before codegen sees the IR. The [`NormalizeReturnOrder`](../passes/28-normalize_return_order.md) pass now canonicalizes this before codegen, so orchestration codegen maps `return[i]` directly to `out_indices[i]` without tracing through `tile.store`/yield chains.

Likewise, deciding whether a `ForStmt` iter_arg needs a materialised carry variable used to require an alias-equivalence fixpoint over the loop body. The [`ClassifyIterArgCarry`](../passes/49-classify_iter_arg_carry.md) pass now stamps that decision (and the TaskId fence-array extent) onto `ForStmt::attrs_`, so codegen reads `iter_arg_rebind_<i>` / `iter_arg_array_size_<i>` instead of deriving them.

## Overview

The orchestration codegen generates simpler runtime C++ code that manages task-graph execution on Ascend hardware. While [PTO codegen](00-pto_codegen.md) produces InCore kernel code (tile-level compute), orchestration codegen produces the host-side code that:

- Borrows device-memory descriptors from `ChipTaskArgs` as `ChipTensor` references
- Builds `CoreTaskArgs` objects and calls `add_input`/`add_output`/`add_inout`/`add_scalar` to classify parameters (manual-scope dep edges are emitted separately via a `set_dependencies` stack array — see [Manual Scope and TaskId Lowering](#manual-scope-and-taskid-lowering))
- Submits tasks to AIC (CUBE) or AIV (VECTOR) cores via `rt_submit_*_task`
- Handles control flow (loops, conditionals) with `SIMPLER_SCOPE`

**Pipeline:** `IR (Orchestration function) → OrchestrationCodegen → C++ (simpler runtime API)`

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
- **IfStmt** → conditional blocks with `SIMPLER_SCOPE` per branch and return variable handling
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

### Phase 1: Boilerplate

```cpp
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "orchestration_api.h"
```

### Phase 2–3: Entry Points

```cpp
// Phase 2: Config function — returns expected argument count
OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs& orch_args) {
    (void)orch_args;
    return OrchestrationConfig{ .expected_arg_count = 3 };
}

// Phase 3: Entry function signature
void aicpu_orchestration_entry(const ChipTaskArgs& orch_args) {
```

### Phase 4–5: Tensor Setup

```cpp
// Phase 4: External tensors — borrow the chip-resident descriptors
const ChipTensor& ext_a = orch_args.tensor(0).ref();
const ChipTensor& ext_b = orch_args.tensor(1).ref();
const ChipTensor& ext_dn = orch_args.tensor(2).ref();

// Phase 5: Internal tensors (from pl.create_tensor — intermediates only)
// All tensor.create in the same scope are batched into a single alloc_tensors call.
uint32_t tmp_ci_shapes[2] = {16, 16};
TensorCreateInfo tmp_ci(tmp_ci_shapes, 2, DataType::FLOAT32);
TaskOutputTensors alloc_0 = alloc_tensors(tmp_ci);
const ChipTensor& tmp = alloc_0.get_ref(0);
```

### Phase 6–8: Task Submission and Control Flow

All task submission is wrapped in a top-level `SIMPLER_SCOPE()`. Codegen no longer
decides scope placement from the `for` / `if` structure: the
[MaterializeRuntimeScopes](../passes/48-materialize_runtime_scopes.md) pass
inserts explicit AUTO `RuntimeScopeStmt` nodes (the function body and each
`for` / `if` body) into the IR, and codegen emits `SIMPLER_SCOPE` 1:1 from those
nodes (manual scopes lower to `SIMPLER_SCOPE(ScopeMode::MANUAL)`):

```cpp
SIMPLER_SCOPE() {
    CoreTaskArgs params_t0;
    params_t0.add_input(ext_a);
    params_t0.add_input(ext_b);
    params_t0.add_output(tmp);               // pre-allocated tensor uses add_output(const ChipTensor&)
    rt_submit_aiv_task(0, params_t0);

    // ForStmt example — plain for loop, no nested SIMPLER_SCOPE
    for (int64_t i = start; i < stop; i += step) {
        // task submissions
    }
}
```

## Key Concepts

### External vs Internal Tensors

| Type | Source | C++ Construction | Naming |
| ---- | ------ | ---------------- | ------ |
| External (ND/DN) | Function parameters | `orch_args.tensor(N).ref()` | `ext_<name>` |
| Internal | `pl.create_tensor(...)` in function body | `TensorCreateInfo var_ci(...)` + `alloc_tensors(...)` at scope entry | `<name>` (no prefix) |

External tensors borrow chip-resident descriptors passed through `ChipTaskArgs`. Internal tensors are pre-allocated at scope entry via `alloc_tensors()` — all `tensor.create` calls within the same scope (function body, for body, if body) are batched into a single `alloc_tensors` invocation. Pre-allocated tensors are then passed to kernels via `add_output(const ChipTensor&)` (OUTPUT_EXISTING overload).

A create is only hoisted to scope entry when its size is *entry-valid*. A size that reads a value the body itself defines — a plain local, or an `if` / `for` / `while` return_var, whose C++ declaration is emitted where that statement sits — keeps the create in body order instead, so the emitted C++ never uses a name before its declaration. The same rule covers the `__gm_pipe_buffer` placeholder, whose real size is `slot_bytes * core_num` rather than its IR shape.

### Parameter Direction

The `ParamDirection` of each function parameter determines how it appears in task submission:

| Direction | Python Annotation | C++ Task Param | Semantics |
| --------- | ----------------- | -------------- | --------- |
| `In` | `pl.Tensor[...]` (default) | `params.add_input(var)` | Read-only |
| `Out` (external) | `pl.Out[pl.Tensor[...]]` (param) | `params.add_output(ext_x)` | Write-only pre-allocated buffer |
| `Out` (internal) | `pl.Out[pl.Tensor[...]]` (tensor.create) | `params.add_output(x)` | Pre-allocated via `alloc_tensors`, uses OUTPUT_EXISTING overload |
| `InOut` | `pl.InOut[pl.Tensor[...]]` | `params.add_inout(ext_x)` | Read-write |
| Scalar | `pl.Scalar[...]` | `params.add_scalar(value)` | Scalar constant (separate scalar slot) |

Internal tensors from `tensor.create` are pre-allocated at scope entry via `alloc_tensors()`. When passed to kernels, they use `add_output(const ChipTensor&)` which triggers the OUTPUT_EXISTING overload — the runtime reuses the pre-allocated buffer instead of allocating a new one.

### Scalar Parameter Encoding

Scalar params occupy `ChipTaskArgs` scalar slots (0-indexed, separate from tensor slots).
Float scalars use `to_u64(f)` (bit-cast). Other integer/bool scalars are cast to `(uint64_t)`.
At the receiving end, union-based type punning is used to reinterpret the `uint64_t` as the target C type:

```cpp
union { uint64_t u64; float val; } scale_conv;
scale_conv.u64 = orch_args.scalar(0);
float scale = scale_conv.val;
```

### Output Aliasing (emit-name remap)

A kernel/submit output is the in-place `Out`/`InOut` arg it writes — the *same
physical tensor*. So when the result Var has a different name than that arg, the
codegen does **not** mint a `const ChipTensor& result = ext_output;` rename; it
remaps the result Var's emit name to the source, and every downstream reference
resolves directly to the source name. (This is the same strategy
`tensor.assemble` uses, applied uniformly.)

```python
# Python IR
result = self.kernel_add(a, b, output)  # result ≠ output
consumer = self.kernel_use(result)
```

```cpp
// Generated C++ — `result` is remapped to ext_output; the consumer reads it directly
CoreTaskArgs params_t0;
params_t0.add_output(ext_output);
rt_submit_aiv_task(0, params_t0);

CoreTaskArgs params_t1;
params_t1.add_input(ext_output);  // `result` -> ext_output (no alias decl)
```

Which `Out`/`InOut` param a result aliases is a lookup, not a heuristic — and
not an analysis either. `ReturnParamsExplicit`
([`NormalizeReturnOrder`](../passes/28-normalize_return_order.md)) guarantees
that every tensor param-writeback return value *is* the param, by pointer
identity. Codegen therefore reads the return-position → param-index map straight
off the callee's `ReturnStmt` via `ir::return_lineage::ExplicitReturnedParamIndices`;
no SSA walk, no callee recursion, no `Program`. The interprocedural lineage
tracer (`ReturnedParamIndices`) stays behind in the IR layer for the passes that
run *before* the property is established.

The property is thus a codegen precondition. When a return position resolves to
no param, single-return aliasing falls back to the sole `Out`/`InOut` param only
when the callee has exactly one — a multi-output callee whose `ReturnStmt` does
not reference a param directly is an internal error, never a guess.

Excluded from remap: a phi/loop-carry reassignment (it rebinds an lvalue the
enclosing `if`/loop owns) keeps its `<name> = <src>;` form; and a tensor whose
source is not valid in the reader's C++ scope (a manual-scope-local source —
see *Cross-scope tensors and `manual_scope`* below) keeps the decl path. A
runtime-allocated output bound to `task_<n>_outs.get_ref(k)` likewise keeps its
`const ChipTensor&` binding.

### Core Type Inference

The codegen determines whether to submit to AIC (CUBE) or AIV (VECTOR) based on the callee's `MemorySpace`:

| MemorySpace | Core Type | Submit Function |
| ----------- | --------- | --------------- |
| `Left`, `Right`, `Acc`, `Mat` | CUBE (AIC) | `rt_submit_aic_task` |
| `Vec` (default) | VECTOR (AIV) | `rt_submit_aiv_task` |

**Exception — dual-AIV kernels.** An AIV kernel stamped `dual_aiv_dispatch` (any
`split_aiv` kernel; see [`SplitVectorKernel`](../passes/26-split_vector_kernel.md))
must run on **both** vector lanes of a cluster — a `pl.split_aiv` region hands each
lane disjoint work selected by `aiv_id`. `rt_submit_aiv_task` fills only the AIV0
slot, so the runtime schedules an *AIV-shape* task — one AIV core per block: the second
lane never launches, and the lone lane that does reads a `get_sub_block_id()` that the
runtime documents as **meaningless** for a single-AIV task. The scheduler seeds that value
per core from the core's fixed position in its cluster, so `aiv_id` becomes that position
rather than a per-block lane id. Such a kernel is therefore submitted as a two-lane
`MixedKernels` + `rt_submit_task` instead (see [Group Functions (Mixed Kernels)](#group-functions-mixed-kernels)),
whatever the dispatch path (direct call, `Spmd` wrapper, or AIV-only `Group`). Two active
AIV slots make it a MIX-shape task, so the scheduler places both lanes of one cluster under
the same `block_idx` and gives them `sub_block_id` 0 and 1; the cluster's AIC core is simply
unused. A plain (non-`dual_aiv_dispatch`) vector kernel keeps `rt_submit_aiv_task`, which
dispatches across independent AIV cores.

### Tuple Handling

Tuple-returning calls use unique keys (`_tc_N`) to track elements:

```python
# Python IR
pij, mij, lij = self.kernel_softmax(sij, scale, pij, mij, lij)
```

```cpp
// Generated C++ — tensors first, then scalars
CoreTaskArgs params_t0;
params_t0.add_input(ext_sij);
params_t0.add_inout(ext_pij);
params_t0.add_inout(ext_mij);
params_t0.add_inout(ext_lij);
params_t0.add_scalar(to_u64(scale));  // scalar after all tensors
rt_submit_aiv_task(0, params_t0);
```

### Group Functions (Mixed Kernels)

When a kernel uses both AIC and AIV cores (mixed kernel), the codegen generates `MixedKernels` submission:

```cpp
// Group: mixed_kernel (AIC + AIV)
CoreTaskArgs params_t0;
// ... add_input / add_inout / add_scalar calls ...
MixedKernels mixed_0 = {aic_id, aiv_id, INVALID_KERNEL_ID};
rt_submit_task(mixed_0, params_t0);
```

The three slots are `{aic_kernel_id, aiv0_kernel_id, aiv1_kernel_id}`; `INVALID_KERNEL_ID`
marks a slot inactive. The AIV1 slot repeats the AIV kernel id when the AIV function
carries `dual_aiv_dispatch` — so both vector lanes run:

| Kernel | `MixedKernels` |
| ------ | -------------- |
| Mixed, single AIV lane | `{aic_id, aiv_id, INVALID_KERNEL_ID}` |
| Mixed, `dual_aiv_dispatch` | `{aic_id, aiv_id, aiv_id}` |
| Vector-only, `dual_aiv_dispatch` | `{INVALID_KERNEL_ID, aiv_id, aiv_id}` |

## Operation Mappings

| IR Operation | C++ Codegen | Description |
| ------------ | ----------- | ----------- |
| `tensor.create` | `TensorCreateInfo var_ci(...)` + `alloc_tensors(...)` | Scope-level batched alloc; `const ChipTensor& var = alloc_N.get_ref(i)` |
| `tensor.read` | `*reinterpret_cast<T*>(arg_ptr + offset)` | Read scalar from host tensor |
| `tensor.slice` | `ChipTensor xs = ext_x.view(shapes, offsets)` | Create a metadata view into an existing tensor |
| `tensor.transpose` | `ChipTensor xt = ext_x.transpose(axis1, axis2)` | Zero-copy metadata swap of two axes (lowers to runtime `ChipTensor::transpose`) |
| `tensor.dim` (static) | `int64_t d0 = 16` | Constant dimension value |
| `tensor.dim` (dynamic) | `int64_t d0 = (int64_t)orch_args.tensor(N).ref().shapes[axis]` | Runtime dimension from ChipTaskArgs. In an Orchestration body the parser folds it onto the declared extent instead — see below |

### Dynamic-dim symbols

A `pl.dynamic("M")` symbol names the runtime extent of whatever tensor argument
declares it. In a kernel it is a type-level placeholder, but an Orchestration body
may use it as a **value** — a loop bound, a `pl.create_tensor` extent, or a folded
`pl.tensor.dim` — so each symbol the body references is defined once at entry, read
from the descriptor of the first parameter declaring it:

```cpp
    // Dynamic-dim symbols (extent of the declaring argument)
    int64_t M = (int64_t)orch_args.tensor(0).ref().shapes[0];
```

Only symbols the emitted body mentions get a definition; a symbol appearing solely
in a parameter's type produces none (external tensor shapes are never printed).

Because the symbol **is** the extent, the parser folds `pl.tensor.dim(x, i)` in an
Orchestration body onto the extent `x`'s type already names — one runtime extent,
one IR name. Reading it back would mint a second scalar that no analyzer can prove
equal to the symbol, and every shape built from that copy would then disagree
structurally with shapes built from the symbol. The fold is **Orchestration-only**:
an Inline/InCore callee may be reached with a differently-shaped actual, so there
`tensor.dim` stays a genuine runtime read.

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
#include "orchestration_api.h"

extern "C" {

OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs& orch_args) {
    (void)orch_args;
    return OrchestrationConfig{ .expected_arg_count = 3 };
}

void aicpu_orchestration_entry(const ChipTaskArgs& orch_args) {
    // External tensors (from ChipTaskArgs)
    const ChipTensor& ext_a = orch_args.tensor(0).ref();
    const ChipTensor& ext_b = orch_args.tensor(1).ref();
    const ChipTensor& ext_d = orch_args.tensor(2).ref();

    SIMPLER_SCOPE() {
        // Internal tensor — pre-allocated via alloc_tensors at scope entry
        uint32_t c_ci_shapes[2] = {16, 16};
        TensorCreateInfo c_ci(c_ci_shapes, 2, DataType::FLOAT32);
        TaskOutputTensors alloc_0 = alloc_tensors(c_ci);
        const ChipTensor& c = alloc_0.get_ref(0);

        // Task 0: kernel_add (a + b → c)
        CoreTaskArgs params_t0;
        params_t0.add_input(ext_a);
        params_t0.add_input(ext_b);
        params_t0.add_output(c);
        rt_submit_aiv_task(0, params_t0);

        // Task 1: kernel_add (c + b → d)
        CoreTaskArgs params_t1;
        params_t1.add_input(c);
        params_t1.add_input(ext_b);
        params_t1.add_output(ext_d);
        rt_submit_aiv_task(1, params_t1);
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
| Alloc result | `alloc_<N>` | `alloc_0` |
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
// Generated C++ (inside top-level SIMPLER_SCOPE)
ChipTensor acc = ext_acc;  // iter_arg initialization
for (int64_t i = 0; i < 4; i += 1) {
    CoreTaskArgs params_t0;
    // ... add_input / add_inout calls ...
    rt_submit_aiv_task(0, params_t0);
}
```

Iter_args are initialized before the loop. `YieldStmt` updates are emitted at the end of each iteration.

An `IterArg::initValue_` is usually an SSA `Var`, but any expression is legal —
a scalar carry seeded by a constant (`acc: pl.Scalar[pl.INT64] = 0` before the
loop, which `Simplify` propagates into the loop, or an explicit
`pl.range(..., init_values=(0,))`) arrives as a `ConstInt`. Codegen emits the
init expression directly (`int64_t acc__rv_v1 = 0;`); a trivial carry — one the
body never rebinds — aliases both names straight to that expression instead. The
one path that still requires the init to *name* something is the `ArrayType`
carry copy-in, which indexes it slot-by-slot.

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
    SIMPLER_SCOPE() {
        CoreTaskArgs params_t0;
        // ... add_input / add_inout calls ...
        rt_submit_aiv_task(0, params_t0);
    }
} else {
    SIMPLER_SCOPE() {
        CoreTaskArgs params_t1;
        // ... add_input / add_inout calls ...
        rt_submit_aiv_task(1, params_t1);
    }
}
```

#### `ArrayType` return_vars (array phis)

Writing one element of a `pl.array` under an `if` whose condition survives to
codegen makes the array a branch result. `arr[i] = v` is SSA-functional — it
desugars to `arr = pl.array.update_element(arr, i, v)` — so `ConvertToSSA` sees
`arr` diverge between the branches and synthesizes an `IfStmt` return_var for
it:

```python
if i < n:                       # runtime-valued n — the guard survives
    tids[i] = tid               # tids diverges => ArrayType phi
```

Such a phi is **not** declared. An `ArrayType` SSA value is a *reference* to one
backing C-stack array rather than a copyable value: every `array.update_element`
aliases its result onto its input's emit name, so both branches mutate the same
storage and the merge is a no-op. A raw C array could not be declared from its
type nor assigned anyway — asking `GetCppType` for one is an internal error.

Instead, each branch's `YieldStmt` records the backing array it resolves to, and
the phi's emit name is bound onto it once both branches are emitted, so reads
after the `if` resolve straight to that array:

```cpp
TaskId tids[8];             // the one backing array
...
if ((i < static_cast<int64_t>(n))) {
    tids[i] = p_tid;            // in-place; no phi variable, no copy
} else {
}
```

Two ordering constraints shape this:

- **Resolution happens at each branch's yield**, not by pre-scanning the branch.
  The yielded value is not always an `array.update_element` result — when the
  array is updated under *nested* control flow, the branch yields the inner
  `if` / `for`'s own ArrayType return_var. By yield time that nested statement
  has already been bound, so one lookup resolves every nesting depth. This also
  keeps the work O(1) per phi: no branch subtree is traversed twice.
- **Installation happens after both branches**, at the enclosing level, because
  every generated `SIMPLER_SCOPE` snapshots and restores `array_carry_vars_` (see
  [Manual Scope and TaskId Lowering](#manual-scope-and-taskid-lowering)) — a
  registration made inside a branch body would be discarded at its closing
  brace. The PTO backend captures and binds its in-place return_vars the same
  way.

Both branches must name the same backing array. Creating a *different* array in
each branch leaves nothing single to bind the phi onto, and is rejected with a
user-facing `CHECK` — create the array before the `if` and write its elements
inside the branches instead.

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

## Manual Scope and TaskId Lowering

`with pl.manual_scope():` regions lower to a `SIMPLER_SCOPE(ScopeMode::MANUAL)`
block where the runtime's auto OverlapMap is disabled. Per-task params are
always declared as a plain `CoreTaskArgs <task_var>;`. The orchestration codegen
materialises the required dependency edges as a fixed-size stack array plus
a single `set_dependencies` call:

```cpp
CoreTaskArgs params_t1;
params_t1.add_input(...);
// ...
TaskId params_t1_deps[K];          // K = exact dep-edge count
uint32_t params_t1_deps_count = 0;
params_t1_deps[params_t1_deps_count++] = tid;                          // fresh producer — unguarded
if (carry.is_valid()) params_t1_deps[params_t1_deps_count++] = carry;  // loop carry — may be invalid
params_t1.set_dependencies(params_t1_deps, params_t1_deps_count);
```

A dep slot is guarded with `if (task_id.is_valid())` only when the TaskId may
legitimately hold the `TaskId::invalid()` sentinel, because an invalid id
must never reach `set_dependencies`; a **fresh direct-producer** TaskId is
statically always-valid and is emitted unguarded (issue #1966). See
[TaskId sourcing](#taskid-sourcing) for the full case list.

There is no `params.add_dep(...)` call and no 16-dep cap — the runtime
`CoreTaskArgs::set_dependencies` primitive has no upper bound, and the stack array is
sized to the exact count. User edges come from the parser: it writes the user's
`pl.submit(..., deps=[tid1, tid2])` kwarg into the typed `Submit::deps_` field;
codegen reads them through the transient `SubmitToCallView`, which surfaces
`deps_` as a synthesised `attrs["manual_dep_edges"]` entry. Plain `Call`
carriers of `manual_dep_edges` no longer exist — the
ManualDepsOnSubmitOnly structural property verifies that no cross-function
`Call` carries it; only the `system.task_dummy` barrier op keeps the attr as
its fanin contract. Compiler-derived edges come from
[`AutoDeriveTaskDependencies`](../passes/41-auto_derive_task_dependencies.md)
in `Call.attrs["compiler_manual_dep_edges"]` (a separate key, allowed on plain
calls). That pass never analyzes a user-written MANUAL scope — inside
`pl.manual_scope()` the explicit `deps=[...]` list stays the only source of
dependency edges. It analyzes AUTO regions only, and only under the compile-time
`analyze_auto_scopes_for_deps` switch. A default-mode (`auto_scope=True`) region
becomes a *compiler-owned* MANUAL scope when fully covered, and otherwise stays
AUTO with its representable edges emitted on top of runtime auto-tracking. A
hand-placed `pl.scope()` always keeps `manual=false`, and has its partial edges
stripped if analysis falls back. Codegen merges the two lists in that order and
deduplicates by Var identity before emitting the stack array, tagging each entry
with its provenance (`DepEdge::user_written`).

The provenance decides what happens when an edge fails to resolve to a live
TaskId binding — see [Unresolvable dep edges](#unresolvable-dep-edges).

### TaskId sourcing

Every kernel task launch inside a manual scope is a `Submit`. Each dep entry —
`manual_dep_edges` synthesised from `deps_` by `SubmitToCallView`, or
`compiler_manual_dep_edges` on a plain call — is a TaskId `VarPtr`
(`Scalar[TASK_ID]` or `Array[N, TASK_ID]`) and resolves at codegen time through
`manual_task_id_map_` to one of the following forms:

| Producer kind | C++ source emitted by codegen |
| ------------- | ----------------------------- |
| `pl.submit` producer TaskId (the augmented Call's TaskId tuple element) | `TaskId <tid_name> = task_<n>_outs.task_id();` where `task_<n>_outs` is the `TaskOutputTensors` captured from the submit |
| `None` seed (the literal in a `deps=[None]` entry or a TaskId iter_arg init) | `TaskId::invalid()` |
| Loop-carry iter_arg (TaskId companion threaded through a loop) | A named variable threaded through the for-loop, either scalar or array — see below |
| Array-slot read (`prev = tids[k]` — `array.get_element` on an `Array[TASK_ID]`) | `TaskId <name> = <arr>[k];` — a scalar snapshot local; the dep references this local, not a re-read of the slot, so a later `tids[k] = ...` overwrite does not change it |

The kernel-result tuple elements of a `pl.submit` call alias the kernel's
`Out`/`InOut` args exactly like an ordinary multi-output kernel call.

A dep array-fill entry is wrapped in `if (<task_id>.is_valid())` when the id may
hold the `TaskId::invalid()` sentinel — a first-iteration iter_arg carry, an
unwritten array slot, an array-slot read, or a `None` seed. A **fresh
`pl.submit`-producer** TaskId is statically always-valid, so `EmitManualDeps`
emits its insert unguarded (issue #1966); every other scalar (string-backed)
TaskId keeps the guard. Array-carry iter_args fill one guarded slot per element.

**Lexical-scope lifetime.** TaskId bindings name C++ locals (`TaskId tid
= ...`) declared inside the generated `SIMPLER_SCOPE { ... }` block they are
produced in. Each `SIMPLER_SCOPE` (AUTO or MANUAL) snapshots `manual_task_id_map_`
and `array_carry_vars_` on entry and restores them on exit, so a binding
produced inside a scope does not leak to an enclosing scope where its identifier
would be out of C++ scope. Loop / branch carries are declared *before* their
body's `SIMPLER_SCOPE`, so they correctly survive the block.

### Unresolvable dep edges

A consequence of that lifetime rule: a dep edge naming a TaskId produced in a
scope that has **already closed** cannot resolve — the binding was restored away
on scope exit. Codegen's response depends on the edge's provenance:

| Edge provenance | Behavior when unresolvable |
| --------------- | -------------------------- |
| Compiler-derived (`compiler_manual_dep_edges`) | Silently skipped. These are a best-effort hazard patch; `PrepareCrossScopeTaskIdHoists` already LCA-hoists the ones it can, and dropping the rest is safe because the pass only ever *adds* ordering |
| User-written (`deps=[...]`) | **Hard error** — `CHECK_SPAN` raises a `pypto::ValueError` naming the TaskId and the DSL source line |

Provenance is the attr key. Note that `attrs["dummy_task"]` is *not* an
authorship marker: the parser stamps it on a user-written
`pl.system.task_dummy(deps=[...])` exactly as
[`ExpandManualPhaseFence`](../passes/42-expand_manual_phase_fence.md) does on the
barriers it synthesises, so every `manual_dep_edges` carrier is enforced. The
synthesised barrier only ever names a TaskId live in the manual scope it
rewrites, so its fanin always resolves.

Dropping a user edge would leave the consumer unordered against its producer and
surface at runtime as a silent stale read, so codegen refuses to emit rather than
emit wrong code. `ResolveDepEdgeBinding` is the single resolve-and-validate site
both `CountManualDeps` (array sizing) and `EmitManualDeps` (array fill) route
through, so the two can never disagree on which edges survive.

The scope that closed need not be user-written. `MaterializeRuntimeScopes` wraps
every `ForStmt` body and every `IfStmt` branch body in its own AUTO scope, so
this fires on ordinary orchestration code with no `pl.scope()` / `pl.manual_scope()`
in sight — e.g. a TaskId captured inside a `pl.range` body and depended on after
the loop. The fix is to keep the consumer in the producer's scope, or hoist the
producer outward.

One exception applies to the `array_carry_vars_` restore on a `MANUAL` scope: an
array carry registered *inside* the scope whose backing array was declared in
the *enclosing* scope must survive the restore. This is the loop-carry of a
`manual_scope`-produced TaskId into an `Array[TASK_ID]` (issue #1811) — e.g. a
`pl.parallel` array carry threaded from an outer `pl.range` loop's backing
store, where each iteration writes one slot (`carry[n] = prod_tid`). The
enclosing loop's `YieldStmt`, emitted *after* the `SIMPLER_SCOPE(MANUAL)` block,
references that carry; wiping it would drop the loop-carried TaskIds and trip
the *scalar yield to array carry* `INTERNAL_CHECK`. On exit codegen therefore
preserves array carries whose backing storage is enclosing-scope-valid (named by
an identifier not in the scope's local set), and reverts only the scope-local
ones.

**Cross-scope tensors and `manual_scope`.** A `manual_scope` is a *scheduling*
region, not a storage/value scope: a tensor it touches flows transparently to
tasks placed *after* the `SIMPLER_SCOPE(MANUAL) { ... }` block. So nothing an
after-scope reader names may be a manual-scope-local C++ identifier — otherwise
it dies at the closing brace and the reader's `add_input(...)` references an
out-of-scope name (the `.cpp` then fails to C++-compile, issue #1697). Two
mechanisms enforce this, both gated on whether a name is *enclosing-scope-valid*
(reserved before the block, or a hoisted in-scope buffer — i.e. not scope-local):

- **Output remap.** A caller-allocated kernel/submit output that aliases an
  enclosing-scope source is *not* given its own `const ChipTensor&` decl — its emit
  name is remapped to the source, so every reference (in-scope and after-scope)
  resolves to the enclosing name directly. This is the strategy `tensor.assemble`
  already uses, and since the output is the same physical tensor as its source
  (an in-place write), a shared name is exactly correct. A phi/loop-carry
  reassignment is excluded — it rebinds an lvalue the enclosing `if`/loop owns.

- **Allocation hoisting.** A buffer *created* inside the block
  (`pl.create_tensor` → `alloc_tensors`) is a storage reservation with no
  scheduling dependency, so its declaration is hoisted to the enclosing scope.
  Codegen buffers each `SIMPLER_SCOPE(MANUAL)` body and flushes the hoisted
  `alloc_tensors` decls ahead of the block header. The batch is enclosing-scope-
  valid by construction (a create whose shape references a scope-local value is
  excluded and stays put).

Together these make a tensor created before *or* inside the scope and read after
it resolve to a single enclosing-scope `const ChipTensor& buf = ...;` — the
after-scope task simply does `add_input(buf)`, with no per-SSA-version alias.

### Array carry for `pl.parallel` TaskId iter_args

A `pl.parallel(N)` ForStmt whose iter_arg threads a TaskId companion is
lowered as an **array carry of size N**, not a scalar last-write-wins
variable. The pass leaves the iter_arg as `Scalar[TASK_ID]` in the IR; the
codegen detects this shape (Parallel kind + TaskId iter_arg) and:

1. Allocates a fixed-size backing store at the iter_arg's declaration site:
   `TaskId arr[N];`, initialised by broadcasting the loop's init value
   (scalar) or by slot-by-slot copy (when init is itself an array — e.g. the
   inner `pl.parallel` of `case1` reads its init from the outer
   `pl.range`'s array carry).
2. In each parallel iteration's body, writes the freshly produced task id
   into one slot: `arr[(loop_var - start) / step] = <task_id>;`. The slot
   expression peephole-simplifies to `arr[loop_var]` when `start == 0` and
   `step == 1` (the common form).
3. On every downstream consumer `Submit` whose `deps_` references this
   iter_arg, fills N guarded slots into the task's dep stack array,
   one per slot:

   ```cpp
   for each k in [0..N):
       if (arr[k].is_valid()) { params_deps[params_deps_count++] = arr[k]; }
   ```

A `pl.range` (Sequential) loop whose yield value is the rv of an inner
`pl.parallel` array-carry inherits the same array size: its own iter_arg
becomes an array carry of the same N, slot-by-slot copied on outer yield.
This propagation is the structural source of the multi-iter fence semantics
in topologies like case1 (outer SEQ × inner PARALLEL).

### Phase-fence dummy barriers

After `DeriveCallDirections`, the `ExpandManualPhaseFence` pass may compress a
profitable stable full-array manual dependency by rewriting selected consumer
`Submit`s from `deps_=[tids]` to `deps_=[barrier_tid]`. It inserts a marked
`system.task_dummy` call whose own `manual_dep_edges` attr still references
the original TaskId array (the sanctioned op-call carrier of the attr —
plain cross-function `Call`s never carry it). Orchestration codegen lowers that marked
call to `rt_submit_dummy_task(...)`, then emits ordinary scalar dependency
lowering for the rewritten consumers.

**Empty-deps vs dep-carrying dummies.** Codegen submits a dep-carrying dummy
under an `if (deps_count > 0)` runtime guard, because its deps are appended
under per-edge `is_valid()` guards and may all resolve to an invalid sentinel
(fencing nothing). A statically empty-deps dummy — which can only come from a
user-written `pl.system.task_dummy(deps=[])`, since `ExpandManualPhaseFence`
never inserts an empty barrier — is instead submitted **unconditionally**. A
no-predecessor barrier is still a real, ready-immediately task whose valid id
must join each consumer's fanin; having no predecessors affects neither its
submission nor the edges to its successors, so guarding it on `deps_count > 0`
would statically elide it and silently drop those edges.

This preserves the phase boundary while avoiding repeated all-to-all fanout:

```text
tids[N] -> dummy barrier -> consumers[M]
```

Shapes that are not clearly safe or profitable stay on the direct
`Submit::deps_` lowering path. In particular, `manual_scope` treats explicit
deps as authoritative: a `pl.parallel` body that reads `deps=[tids]` and then
updates `tids[branch]` is a same-carrier dependency chain, not a snapshot
source for pre-loop compression. Users who want layer-parallel snapshot
semantics should write a separate `tids_next` carrier and carry it back after
the parallel body via loop-carried `init_values` / `pl.yield_`. We do not spell
this as plain `tids = tids_next` here because the current codegen path does not
support an ordinary `AssignStmt` on `ArrayType`.

**Constraints checked at codegen entry (with user-facing CHECK messages):**

- The `pl.parallel` trip count must be a Python literal (statically known).
  A dynamic trip count is rejected at codegen with a "statically-known trip
  count" message.

The dep stack array is sized to the exact dep count (for an array carry,
`N` slots), so trip counts larger than 16 are not capped — the runtime
primitive `CoreTaskArgs::set_dependencies(ptr, count)` has no upper bound either.

### Example

Source DSL (case1 shape):

```python
with pl.manual_scope():
    prev_tid = None
    for phase in pl.range(N_PHASES):
        for branch in pl.parallel(N_BRANCHES):
            row = (phase * N_BRANCHES + branch) * TILE_M
            out, prev_tid = pl.submit(self.kernel_stripe, data, row, 1.0, out, deps=[prev_tid])
```

Generated C++ (skeleton):

```cpp
SIMPLER_SCOPE(ScopeMode::MANUAL) {
    TaskId out__rv_v2__tid[N_BRANCHES];                    // outer rv = array
    for (int64_t i = 0; i < N_BRANCHES; ++i)
        out__rv_v2__tid[i] = TaskId::invalid();            // broadcast None seed
    for (int64_t phase = 0; phase < N_PHASES; phase += 1) {
        TaskId out__rv_v4__tid[N_BRANCHES];                // inner rv = array
        for (int64_t i = 0; i < N_BRANCHES; ++i)
            out__rv_v4__tid[i] = out__rv_v2__tid[i];           // copy slot-by-slot
        for (int64_t branch = 0; branch < N_BRANCHES; branch += 1) {
            int64_t row = ...;
            CoreTaskArgs params_t0; /* ... */
            TaskId params_t0_deps[N_BRANCHES];             // sized to array-carry N
            uint32_t params_t0_deps_count = 0;
            for (int64_t k = 0; k < N_BRANCHES; ++k) {         // multi-deps fanout
                if (out__rv_v2__tid[k].is_valid())
                    params_t0_deps[params_t0_deps_count++] = out__rv_v2__tid[k];
            }
            params_t0.set_dependencies(params_t0_deps, params_t0_deps_count);
            TaskOutputTensors task_0_outs = rt_submit_aiv_task(0, params_t0);
            TaskId out__ssa_v5__tid = task_0_outs.task_id();
            out__rv_v4__tid[branch] = out__ssa_v5__tid;        // slot yield
        }
        for (int64_t i = 0; i < N_BRANCHES; ++i)
            out__rv_v2__tid[i] = out__rv_v4__tid[i];           // outer yield (copy)
    }
}
```

Every task in phase `N+1` waits for **all** `N_BRANCHES` tasks of phase `N`.

## See Also

- [PTO Codegen](00-pto_codegen.md) — MLIR generation for PTO backend
- [Pass Manager](../passes/00-pass_manager.md) — IR optimization passes applied before codegen
- [Python syntax: manual dependency primitives](../language/02-manual_dependencies.md#manual-dependency-primitives) — the user-facing surface form
