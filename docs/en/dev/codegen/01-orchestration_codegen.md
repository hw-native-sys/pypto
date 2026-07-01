# Orchestration Code Generation

## Design Principle: Strict 1-to-1 Mapping

Orchestration codegen follows the same principle as [PTO codegen](00-pto_codegen.md#design-principle-strict-1-to-1-mapping): a **strict 1-to-1 translation** from IR to generated C++ code. The codegen should not perform optimization, analysis, or indirection — such work belongs in earlier passes.

For example, return-to-parameter tracing (mapping callee return values back to `Out` parameters) is analysis that should be resolved by a pass before codegen sees the IR. The [`NormalizeReturnOrder`](../passes/23-normalize_return_order.md) pass now canonicalizes this before codegen, so orchestration codegen maps `return[i]` directly to `out_indices[i]` without tracing through `tile.store`/yield chains.

## Overview

The orchestration codegen generates PTO2 runtime C++ code that manages task-graph execution on Ascend hardware. While [PTO codegen](00-pto_codegen.md) produces InCore kernel code (tile-level compute), orchestration codegen produces the host-side code that:

- Wraps device memory pointers (via `ChipStorageTaskArgs`) into `Tensor` objects
- Builds `Arg` objects and calls `add_input`/`add_output`/`add_inout`/`add_scalar` to classify parameters (manual-scope dep edges are emitted separately via a `set_dependencies` stack array — see [Manual Scope and TaskId Lowering](#manual-scope-and-taskid-lowering))
- Submits tasks to AIC (CUBE) or AIV (VECTOR) cores via `rt_submit_*_task`
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

### Phase 1: Boilerplate

```cpp
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "pto_orchestration_api.h"
```

### Phase 2–3: Entry Points

```cpp
// Phase 2: Config function — returns expected argument count
PTO2OrchestrationConfig aicpu_orchestration_config(const ChipStorageTaskArgs& orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{ .expected_arg_count = 3 };
}

// Phase 3: Entry function signature
void aicpu_orchestration_entry(const ChipStorageTaskArgs& orch_args) {
```

### Phase 4–5: Tensor Setup

```cpp
// Phase 4: External tensors — all layouts via from_tensor_arg()
Tensor ext_a = from_tensor_arg(orch_args.tensor(0));
Tensor ext_b = from_tensor_arg(orch_args.tensor(1));
Tensor ext_dn = from_tensor_arg(orch_args.tensor(2));

// Phase 5: Internal tensors (from pl.create_tensor — intermediates only)
// All tensor.create in the same scope are batched into a single alloc_tensors call.
uint32_t tmp_ci_shapes[2] = {16, 16};
TensorCreateInfo tmp_ci(tmp_ci_shapes, 2, DataType::FLOAT32);
TaskOutputTensors alloc_0 = alloc_tensors(tmp_ci);
const Tensor& tmp = alloc_0.get_ref(0);
```

### Phase 6–8: Task Submission and Control Flow

All task submission is wrapped in a top-level `PTO2_SCOPE()`. Codegen no longer
decides scope placement from the `for` / `if` structure: the
[MaterializeRuntimeScopes](../passes/39-materialize_runtime_scopes.md) pass
inserts explicit AUTO `RuntimeScopeStmt` nodes (the function body and each
`for` / `if` body) into the IR, and codegen emits `PTO2_SCOPE` 1:1 from those
nodes (manual scopes lower to `PTO2_SCOPE(PTO2ScopeMode::MANUAL)`):

```cpp
PTO2_SCOPE() {
    Arg params_t0;
    params_t0.add_input(ext_a);
    params_t0.add_input(ext_b);
    params_t0.add_output(tmp);               // pre-allocated tensor uses add_output(const Tensor&)
    rt_submit_aiv_task(0, params_t0);

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
| External (ND/DN) | Function parameters | `from_tensor_arg(orch_args.tensor(N))` | `ext_<name>` |
| Internal | `pl.create_tensor(...)` in function body | `TensorCreateInfo var_ci(...)` + `alloc_tensors(...)` at scope entry | `<name>` (no prefix) |

External tensors wrap device memory pointers passed from the host via `ChipStorageTaskArgs`. Internal tensors are pre-allocated at scope entry via `alloc_tensors()` — all `tensor.create` calls within the same scope (function body, for body, if body) are batched into a single `alloc_tensors` invocation. Pre-allocated tensors are then passed to kernels via `add_output(const Tensor&)` (OUTPUT_EXISTING overload).

### Parameter Direction

The `ParamDirection` of each function parameter determines how it appears in task submission:

| Direction | Python Annotation | C++ Task Param | Semantics |
| --------- | ----------------- | -------------- | --------- |
| `In` | `pl.Tensor[...]` (default) | `params.add_input(var)` | Read-only |
| `Out` (external) | `pl.Out[pl.Tensor[...]]` (param) | `params.add_output(ext_x)` | Write-only pre-allocated buffer |
| `Out` (internal) | `pl.Out[pl.Tensor[...]]` (tensor.create) | `params.add_output(x)` | Pre-allocated via `alloc_tensors`, uses OUTPUT_EXISTING overload |
| `InOut` | `pl.InOut[pl.Tensor[...]]` | `params.add_inout(ext_x)` | Read-write |
| Scalar | `pl.Scalar[...]` | `params.add_scalar(value)` | Scalar constant (separate scalar slot) |

Internal tensors from `tensor.create` are pre-allocated at scope entry via `alloc_tensors()`. When passed to kernels, they use `add_output(const Tensor&)` which triggers the OUTPUT_EXISTING overload — the runtime reuses the pre-allocated buffer instead of allocating a new one.

### Orchestration Signature (copy-back control)

`OrchestrationResult::orchestration_signature` is the entry's per-tensor runtime
`ArgDirection` name list (`IN` / `OUT` / `INOUT`), in `orch_args` tensor order
(scalars excluded). The config generator emits it as the `ORCHESTRATION`
`"signature"`, and the runtime (`bind_callable_to_runtime_impl`) uses
`signature[i]` for `orch_args.tensor(i)` to decide device staging and copy-back:
a tensor marked `IN` **skips the D2H copy-back** (read-only input) and a pure
`OUT` tensor takes the on-device memset fast path.

**The signature is built from the entry's declared `ParamDirection`s, so declared
directions are authoritative and outputs must be marked by hand:** an entry
parameter that a kernel writes **must** be declared `pl.Out[...]` (write-only) or
`pl.InOut[...]` (read-write). Left as a plain `pl.Tensor` (`ParamDirection::In`)
it is marked `IN`, the runtime skips its copy-back, and its result silently reads
back as all-zeros on the host.

As a lightweight guard, `GenerateOrchestration` logs a **non-fatal** warning
(`LOG_ERROR`) when the entry declares no `pl.Out` / `pl.InOut` parameter at all —
commonly a forgotten output annotation. Compilation still proceeds: returning a
runtime-allocated output (a tensor produced by a kernel and returned, rather than
written into a caller-supplied buffer) is a legitimate pattern that needs no
output parameter, so this stays a warning rather than a hard error.

### Scalar Parameter Encoding

Scalar params occupy `ChipStorageTaskArgs` scalar slots (0-indexed, separate from tensor slots).
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
codegen does **not** mint a `const Tensor& result = ext_output;` rename; it
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
Arg params_t0;
params_t0.add_output(ext_output);
rt_submit_aiv_task(0, params_t0);

Arg params_t1;
params_t1.add_input(ext_output);  // `result` -> ext_output (no alias decl)
```

Which `Out`/`InOut` param a result aliases is a lookup, not a heuristic:
pipeline IR satisfies the `ReturnParamsExplicit` property
([`NormalizeReturnOrder`](../passes/23-normalize_return_order.md)),
so `FindReturnedParamIndex` resolves each return value to a param by pointer
identity via `ir::return_lineage`. The legacy lineage tracing (var-to-var
aliases, loop carries, builtin writebacks, `TupleGetItem` of tuple calls,
Group/Spmd wrappers) remains only for hand-parsed IR. When no param can be
traced, single-return aliasing falls back to the sole `Out`/`InOut` param only
when the callee has exactly one — multiple outputs with an untraceable return
are an internal error, never a guess.

Excluded from remap: a phi/loop-carry reassignment (it rebinds an lvalue the
enclosing `if`/loop owns) keeps its `<name> = <src>;` form; and a tensor whose
source is not valid in the reader's C++ scope (a manual-scope-local source —
see *Cross-scope tensors and `manual_scope`* below) keeps the decl path. A
runtime-allocated output bound to `task_<n>_outs.get_ref(k)` likewise keeps its
`const Tensor&` binding.

### Core Type Inference

The codegen determines whether to submit to AIC (CUBE) or AIV (VECTOR) based on the callee's `MemorySpace`:

| MemorySpace | Core Type | Submit Function |
| ----------- | --------- | --------------- |
| `Left`, `Right`, `Acc`, `Mat` | CUBE (AIC) | `rt_submit_aic_task` |
| `Vec` (default) | VECTOR (AIV) | `rt_submit_aiv_task` |

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
params_t0.add_scalar(to_u64(scale));  // scalar after all tensors
rt_submit_aiv_task(0, params_t0);
```

### Group Functions (Mixed Kernels)

When a kernel uses both AIC and AIV cores (mixed kernel), the codegen generates `MixedKernels` submission:

```cpp
// Group: mixed_kernel (AIC + AIV)
Arg params_t0;
// ... add_input / add_inout / add_scalar calls ...
MixedKernels mixed_0 = {aic_id, aiv_id, INVALID_KERNEL_ID};
rt_submit_task(mixed_0, params_t0);
```

## Operation Mappings

| IR Operation | C++ Codegen | Description |
| ------------ | ----------- | ----------- |
| `tensor.create` | `TensorCreateInfo var_ci(...)` + `alloc_tensors(...)` | Scope-level batched alloc; `const Tensor& var = alloc_N.get_ref(i)` |
| `tensor.read` | `*reinterpret_cast<T*>(arg_ptr + offset)` | Read scalar from host tensor |
| `tensor.slice` | `make_tensor_external(ptr + byte_offset, ...)` | Create view into existing tensor |
| `tensor.transpose` | `Tensor xt = ext_x.transpose(axis1, axis2)` | Zero-copy metadata swap of two axes (lowers to runtime `Tensor::transpose`) |
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

extern "C" {

PTO2OrchestrationConfig aicpu_orchestration_config(const ChipStorageTaskArgs& orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{ .expected_arg_count = 3 };
}

void aicpu_orchestration_entry(const ChipStorageTaskArgs& orch_args) {
    // External tensors (from ChipStorageTaskArgs)
    Tensor ext_a = from_tensor_arg(orch_args.tensor(0));
    Tensor ext_b = from_tensor_arg(orch_args.tensor(1));
    Tensor ext_d = from_tensor_arg(orch_args.tensor(2));

    PTO2_SCOPE() {
        // Internal tensor — pre-allocated via alloc_tensors at scope entry
        uint32_t c_ci_shapes[2] = {16, 16};
        TensorCreateInfo c_ci(c_ci_shapes, 2, DataType::FLOAT32);
        TaskOutputTensors alloc_0 = alloc_tensors(c_ci);
        const Tensor& c = alloc_0.get_ref(0);

        // Task 0: kernel_add (a + b → c)
        Arg params_t0;
        params_t0.add_input(ext_a);
        params_t0.add_input(ext_b);
        params_t0.add_output(c);
        rt_submit_aiv_task(0, params_t0);

        // Task 1: kernel_add (c + b → d)
        Arg params_t1;
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
// Generated C++ (inside top-level PTO2_SCOPE)
Tensor acc = ext_acc;  // iter_arg initialization
for (int64_t i = 0; i < 4; i += 1) {
    Arg params_t0;
    // ... add_input / add_inout calls ...
    rt_submit_aiv_task(0, params_t0);
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
        rt_submit_aiv_task(0, params_t0);
    }
} else {
    PTO2_SCOPE() {
        Arg params_t1;
        // ... add_input / add_inout calls ...
        rt_submit_aiv_task(1, params_t1);
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

## Manual Scope and TaskId Lowering

`with pl.manual_scope():` regions lower to a `PTO2_SCOPE(PTO2ScopeMode::MANUAL)`
block where the runtime's auto OverlapMap is disabled. Per-task params are
always declared as a plain `Arg <task_var>;`. The orchestration codegen
materialises the required dependency edges as a fixed-size stack array plus
a single `set_dependencies` call:

```cpp
Arg params_t1;
params_t1.add_input(...);
// ...
PTO2TaskId params_t1_deps[K];          // K = exact dep-edge count
uint32_t params_t1_deps_count = 0;
if (tid.is_valid()) params_t1_deps[params_t1_deps_count++] = tid;      // every entry is is_valid()-guarded
if (carry.is_valid()) params_t1_deps[params_t1_deps_count++] = carry;
params_t1.set_dependencies(params_t1_deps, params_t1_deps_count);
```

Every dep slot is wrapped in `if (task_id.is_valid())`: any TaskId may
legitimately hold the `PTO2TaskId::invalid()` sentinel — a `None` loop-carry
seed, an early loop iteration's iter_arg carry, or an unwritten array slot —
and an invalid id must never reach `set_dependencies`. The guard is a cheap
always-true branch for ids known valid.

There is no `params.add_dep(...)` call any more, and there is no 16-dep cap
— the runtime `Arg::set_dependencies` primitive has no upper bound, and the
stack array is sized to the exact count. User edges come from the parser:
it writes the user's `pl.submit(..., deps=[tid1, tid2])` kwarg into the typed
`Submit::deps_` field; codegen reads them through the transient
`SubmitToCallView`, which surfaces `deps_` as a synthesised
`attrs["manual_dep_edges"]` entry (a `vector<VarPtr>` of `Scalar[TASK_ID]`
variables). Plain `Call` carriers of `manual_dep_edges` no longer exist — the
ManualDepsOnSubmitOnly structural property verifies that no cross-function
`Call` carries it; only the `system.task_dummy` barrier op keeps the attr as
its fanin contract. Compiler-derived manual-scope edges come from
[`AutoDeriveTaskDependencies`](../passes/34-auto_derive_task_dependencies.md)
in `Call.attrs["compiler_manual_dep_edges"]` (a separate key, allowed on plain
calls). Codegen merges the two lists in that order and deduplicates by Var
identity before emitting the stack array.

### TaskId sourcing

Every kernel task launch inside a manual scope is a `Submit`; the
`SubmitToCallView` adapter presents its `deps_` to codegen as
`manual_dep_edges` — a `vector<VarPtr>` of `Scalar[TASK_ID]` variables.
Compiler-derived edges arrive as `attrs["compiler_manual_dep_edges"]` on
plain calls. Each entry resolves at codegen time through
`manual_task_id_map_` to one of the following forms:

| Producer kind | C++ source emitted by codegen |
| ------------- | ----------------------------- |
| `pl.submit` producer TaskId (the augmented Call's TaskId tuple element) | `PTO2TaskId <tid_name> = task_<n>_outs.task_id();` where `task_<n>_outs` is the `TaskOutputTensors` captured from the submit |
| `None` seed (the literal in a `deps=[None]` entry or a TaskId iter_arg init) | `PTO2TaskId::invalid()` |
| Loop-carry iter_arg (TaskId companion threaded through a loop) | A named variable threaded through the for-loop, either scalar or array — see below |
| Array-slot read (`prev = tids[k]` — `array.get_element` on an `Array[TASK_ID]`) | `PTO2TaskId <name> = <arr>[k];` — a scalar snapshot local; the dep references this local, not a re-read of the slot, so a later `tids[k] = ...` overwrite does not change it |

The kernel-result tuple elements of a `pl.submit` call alias the kernel's
`Out`/`InOut` args exactly like an ordinary multi-output kernel call.

Every dep array-fill entry is wrapped in `if (<task_id>.is_valid())` —
including direct `pl.submit`-producer TaskId bindings. `EmitManualDeps` guards
every scalar (string-backed) TaskId uniformly because any TaskId may hold the
`PTO2TaskId::invalid()` sentinel (a first-iteration iter_arg carry, an unwritten
array slot, an array-slot read, or a `None` seed). Array-carry iter_args fill
one guarded slot per element.

**Lexical-scope lifetime.** TaskId bindings name C++ locals (`PTO2TaskId tid
= ...`) declared inside the generated `PTO2_SCOPE { ... }` block they are
produced in. Each `PTO2_SCOPE` (AUTO or MANUAL) snapshots `manual_task_id_map_`
and `array_carry_vars_` on entry and restores them on exit, so a binding
produced inside a scope does not leak to an enclosing scope where its identifier
would be out of C++ scope. Loop / branch carries are declared *before* their
body's `PTO2_SCOPE`, so they correctly survive the block.

One exception applies to the `array_carry_vars_` restore on a `MANUAL` scope: an
array carry registered *inside* the scope whose backing array was declared in
the *enclosing* scope must survive the restore. This is the loop-carry of a
`manual_scope`-produced TaskId into an `Array[TASK_ID]` (issue #1811) — e.g. a
`pl.parallel` array carry threaded from an outer `pl.range` loop's backing
store, where each iteration writes one slot (`carry[n] = prod_tid`). The
enclosing loop's `YieldStmt`, emitted *after* the `PTO2_SCOPE(MANUAL)` block,
references that carry; wiping it would drop the loop-carried TaskIds and trip
the *scalar yield to array carry* `INTERNAL_CHECK`. On exit codegen therefore
preserves array carries whose backing storage is enclosing-scope-valid (named by
an identifier not in the scope's local set), and reverts only the scope-local
ones.

When a later sibling or parent scope references a producer TaskId created in an
earlier nested or sibling scope through `compiler_manual_dep_edges`, codegen
hoists only that TaskId binding: it declares a
`PTO2TaskId <name> = PTO2TaskId::invalid();` sentinel before the producer
`PTO2_SCOPE`, assigns `<name> = task_<n>_outs.task_id();` inside the block, and
then emits the later guarded `set_dependencies(...)` entry from the enclosing
C++ scope. Ordinary scope-local TaskIds still stay local.

**Cross-scope tensors and `manual_scope`.** A `manual_scope` is a *scheduling*
region, not a storage/value scope: a tensor it touches flows transparently to
tasks placed *after* the `PTO2_SCOPE(MANUAL) { ... }` block. So nothing an
after-scope reader names may be a manual-scope-local C++ identifier — otherwise
it dies at the closing brace and the reader's `add_input(...)` references an
out-of-scope name (the `.cpp` then fails to C++-compile, issue #1697). Two
mechanisms enforce this, both gated on whether a name is *enclosing-scope-valid*
(reserved before the block, or a hoisted in-scope buffer — i.e. not scope-local):

- **Output remap.** A caller-allocated kernel/submit output that aliases an
  enclosing-scope source is *not* given its own `const Tensor&` decl — its emit
  name is remapped to the source, so every reference (in-scope and after-scope)
  resolves to the enclosing name directly. This is the strategy `tensor.assemble`
  already uses, and since the output is the same physical tensor as its source
  (an in-place write), a shared name is exactly correct. A phi/loop-carry
  reassignment is excluded — it rebinds an lvalue the enclosing `if`/loop owns.

- **Allocation hoisting.** A buffer *created* inside the block
  (`pl.create_tensor` → `alloc_tensors`) is a storage reservation with no
  scheduling dependency, so its declaration is hoisted to the enclosing scope.
  Codegen buffers each `PTO2_SCOPE(MANUAL)` body and flushes the hoisted
  `alloc_tensors` decls ahead of the block header. The batch is enclosing-scope-
  valid by construction (a create whose shape references a scope-local value is
  excluded and stays put).

Together these make a tensor created before *or* inside the scope and read after
it resolve to a single enclosing-scope `const Tensor& buf = ...;` — the
after-scope task simply does `add_input(buf)`, with no per-SSA-version alias.

### Array carry for `pl.parallel` TaskId iter_args

A `pl.parallel(N)` ForStmt whose iter_arg threads a TaskId companion is
lowered as an **array carry of size N**, not a scalar last-write-wins
variable. The pass leaves the iter_arg as `Scalar[TASK_ID]` in the IR; the
codegen detects this shape (Parallel kind + TaskId iter_arg) and:

1. Allocates a fixed-size backing store at the iter_arg's declaration site:
   `PTO2TaskId arr[N];`, initialised by broadcasting the loop's init value
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
primitive `Arg::set_dependencies(ptr, count)` has no upper bound either.

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
PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
    PTO2TaskId out__rv_v2__tid[N_BRANCHES];                    // outer rv = array
    for (int64_t i = 0; i < N_BRANCHES; ++i)
        out__rv_v2__tid[i] = PTO2TaskId::invalid();            // broadcast None seed
    for (int64_t phase = 0; phase < N_PHASES; phase += 1) {
        PTO2TaskId out__rv_v4__tid[N_BRANCHES];                // inner rv = array
        for (int64_t i = 0; i < N_BRANCHES; ++i)
            out__rv_v4__tid[i] = out__rv_v2__tid[i];           // copy slot-by-slot
        for (int64_t branch = 0; branch < N_BRANCHES; branch += 1) {
            int64_t row = ...;
            Arg params_t0; /* ... */
            PTO2TaskId params_t0_deps[N_BRANCHES];             // sized to array-carry N
            uint32_t params_t0_deps_count = 0;
            for (int64_t k = 0; k < N_BRANCHES; ++k) {         // multi-deps fanout
                if (out__rv_v2__tid[k].is_valid())
                    params_t0_deps[params_t0_deps_count++] = out__rv_v2__tid[k];
            }
            params_t0.set_dependencies(params_t0_deps, params_t0_deps_count);
            TaskOutputTensors task_0_outs = rt_submit_aiv_task(0, params_t0);
            PTO2TaskId out__ssa_v5__tid = task_0_outs.task_id();
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
- [Python syntax: manual dependency primitives](../language/00-python_syntax.md#manual-dependency-primitives) — the user-facing surface form
