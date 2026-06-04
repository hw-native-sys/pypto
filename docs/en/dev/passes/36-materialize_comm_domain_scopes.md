# MaterializeCommDomainScopes Pass

## Overview

`MaterializeCommDomainScopes` walks each host-orchestration function and assembles the
host-side metadata that the distributed runtime needs in order to size and
populate per-rank communication windows. It is the structural analogue of
[`InitMemRef`](28-init_memref.md): it traces an allocation through to its
consumption points, constructs a back-reference object, and threads it onto
the IR types so downstream codegen has O(1) access.

| Aspect | `MemRef` side | `WindowBuffer` side |
| ------ | ------------- | ------------------- |
| Allocation op | `tile.alloc(memory_space, size_in_bytes)` | `pld.tensor.alloc_window_buffer(size_in_bytes)` |
| Assignment LHS at parse time | `Var(PtrType)` | `Var(PtrType)` (same singleton) |
| Wrapper Var subclass | `MemRef` | `WindowBuffer` |
| Wrapper's SSA-edge type | `MemRefType` (singleton) | `WindowBufferType` (singleton) |
| Built by | `InitMemRef` | **`MaterializeCommDomainScopes`** (this pass) |
| Threaded back onto | `TensorType.memref_` | `DistributedTensorType.window_buffer_` |
| Program-level registry | `Program.functions_` (alloc stmts) | `Program.comm_groups_` |

## Position in the pipeline

```text
…  →  DeriveCallDirections  →  MaterializeCommDomainScopes  →  Simplify (final)
```

The pass runs at the very end of the default pipeline, immediately before the
final `Simplify`. None of the intervening passes between `InlineFunctions`
and here touches the host_orch alloc/window/dispatch chain — host_orch is
never tile-lowered, and L2 (chip-level) orchestrations are never inlined
into L3 — so the alloc / view / dispatch sites this pass needs are still
discoverable. Running last keeps the producing IR fully canonicalised before
the descriptor analysis kicks in, and any constant folding that the trailing
`Simplify` does on the collected sizes is applied uniformly.

## Algorithm

For every host-orchestration function (`Function::level_ == Level::HOST` and
`Function::role_ == Role::Orchestrator`, regardless of `func_type_`):

1. **Collect allocations.** Find every `AssignStmt` whose RHS is a
   `pld.tensor.alloc_window_buffer(size, *, name)` Call. Record `(ptr_var, size_expr,
   name, span, call)`.

2. **Collect views.** For every `AssignStmt` whose RHS is a
   `pld.tensor.window(ptr_var, [shape], *, dtype)` Call referencing a recorded
   `ptr_var`, record the binding `view_var → alloc`.

3. **Scan dispatches.** Walk the body with a stack of enclosing `ForStmt`s.
   For every Call whose `op_` is a `GlobalVar` resolving to a chip-level
   orchestration, read `attrs["device"]` and infer a **device descriptor**
   from the device expression in the current loop context:

   | `device=` shape | Descriptor |
   | --------------- | ---------- |
   | `ConstInt(N)` | `subset = {N}` |
   | `IterArg of for r in pl.range(pld.system.world_size())` | `kAll` |
   | `IterArg of for r in pl.range(ConstInt(N))` | `subset = {0, …, N − 1}` |
   | other | `pypto::ValueError` |

   Every positional dispatch arg that is a recorded view Var contributes that
   descriptor to its underlying allocation.

4. **Merge descriptors.** Per allocation, fold every recorded descriptor:
   any `kAll` ⇒ `kAll`; otherwise union the subsets.

5. **Materialise `WindowBuffer`s.** For each allocation construct
   `WindowBuffer(base = ptr_var, size = size_expr, load_from_host = false,
   store_to_host = false)`. The `Var::name_hint_` is inherited from
   `ptr_var->name_hint_`. (Host-staging flags are placeholders for N4+.)

6. **Rewrite view types** *(host_orch only)*. For every view binding, mint a
   fresh `Var` of the same `name_hint_` whose type is
   `DistributedTensorType(shape, dtype, memref, tensor_view, wb)` and run
   `Substitute` to swap every reference to the old view Var with the fresh
   one. Two `pld.tensor.window` views over the same allocation share the same
   `shared_ptr<const WindowBuffer>`. Chip-orch / InCore parameter types are
   not touched.

7. **Cluster into groups.** Walk the allocation list in source order; append
   to the first existing `CommGroup` whose descriptor matches, or open a new
   one. `Program.comm_groups_` ends up populated with the resulting list.

## Sanity checks

The pass raises `pypto::ValueError` (carrying the alloc's span) if:

- An allocation has no `pld.tensor.window` materialisation (dead alloc).
- An allocation has at least one view but no chip-orch dispatch consumes it.
- The `device=` expression on a dispatch is something other than `ConstInt`
  or a recognised `pl.range` induction var.
- Two allocations within the same `CommGroup` share a `name_hint_` (the
  parser already enforces global uniqueness; the pass re-asserts).

## Output invariants

After the pass:

- `Program.comm_groups_` is populated (possibly empty if the program does no
  window-buffer allocation).
- Every `pld.tensor.window` result Var's type is a `DistributedTensorType` whose
  `window_buffer_` field points to the corresponding `WindowBuffer`.
- `pld.tensor.window` views over the same allocation share the same
  `shared_ptr<const WindowBuffer>` — pointer-equality is a load-bearing
  invariant for downstream codegen.
- Chip-orchestration and InCore parameter types remain `nullopt` on
  `window_buffer_`. N7 codegen reads the back-reference at the *host_orch*
  dispatch site and threads the matching `CommContext` pointer explicitly.

## Pass properties

| Field | Value |
| ----- | ----- |
| `required` | `{}` |
| `produced` | `{IRProperty::CommGroupsCollected}` |
| `invalidated` | `{}` |

## Reference

- Source: [src/ir/transforms/materialize_comm_domain_scopes_pass.cpp](../../../../src/ir/transforms/materialize_comm_domain_scopes_pass.cpp)
- Header: [include/pypto/ir/transforms/passes.h](../../../../include/pypto/ir/transforms/passes.h)
- Schema: [include/pypto/ir/program.h](../../../../include/pypto/ir/program.h)
  defines `WindowBuffer` and `CommGroup`.
- DSL: [`pld.tensor.alloc_window_buffer`](../../../../python/pypto/language/distributed/op/tensor_ops.py),
  [`pld.tensor.window`](../../../../python/pypto/language/distributed/op/tensor_ops.py),
  [`pld.system.world_size`](../../../../python/pypto/language/distributed/op/system_ops.py).
