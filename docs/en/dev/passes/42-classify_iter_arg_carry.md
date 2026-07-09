# ClassifyIterArgCarry Pass

Classifies every `ForStmt` iter_arg in an Orchestration function as a **trivial
alias** or a **rebind carry**, and sizes `Scalar[TASK_ID]` fence arrays inside a
`pl.manual_scope`. The plan is stamped onto `ForStmt.attrs_` so the orchestration
codegen reads it instead of re-deriving it.

## Overview

An orchestration loop carry lowers one of two ways in the generated C++:

| Classification | Emitted C++ | Why |
| -------------- | ----------- | --- |
| **trivial** | iter_arg and return_var both alias the init value's emit name | The runtime dependency tracker keys off `Tensor*` identity, and `OUTPUT_EXISTING` / `INOUT` params record the address of the `Tensor` lvalue passed in. Materialising a fresh `Tensor` for the carry would break dep chains — kernel reads/writes would see a different `&tensor` than the producer. |
| **rebind** | a mutable carry variable is declared, and the `YieldStmt` assigns back to it | The yield value is a *different* buffer (e.g. a tensor freshly created inside the body). Without the carry, a Python rebind like `current = next` would never propagate to the next iteration or to code after the loop (issue #1286). |

An iter_arg is **trivial** exactly when its yield value lies in the iter_arg's
*alias class* — the set of Vars naming the same backing buffer. A
`Scalar[TASK_ID]` carry is never trivial: the runtime hands back a fresh
`PTO2TaskId` each iteration.

Inside a `pl.manual_scope`, a `Scalar[TASK_ID]` rebind carry additionally lowers
to a fixed-extent `PTO2TaskId[N]` fence array. `N` is the constant trip count of
the `pl.parallel` loop that owns the array; a `Sequential` loop threading that
array through an inner `pl.parallel` inherits the inner extent.

**When to use**: last pass in the `Default` and `DebugTileOptimization`
strategies, immediately after
[`MaterializeRuntimeScopes`](41-materialize_runtime_scopes.md). Running last
means the classified IR is exactly the IR codegen lowers.

## Alias classes

Four rules put a Var into an iter_arg's alias class. Each names *one* alias
source, so the edges form a forest and the class query is a memoized chain walk
(O(N)) rather than a fixpoint:

| Rule | Edge |
| ---- | ---- |
| `tensor.assemble` | the result aliases `args[0]` (the write target) |
| Out / InOut call | the result aliases the Out/InOut arg the callee actually *returns* (traced via `return_lineage`, so a GM-scratch Out param never captures the alias) |
| `TupleGetItemExpr` | `ret_tuple[i]` aliases the i-th output-side arg of the tuple-producing `Call` / `Submit` |
| nested `ForStmt` | a carry threaded through a nested loop re-emerges as that loop's `return_var`, which aliases the nested loop's init value |

The assemble rule and the Out/InOut rule can never both fire on one assignment:
`tensor.assemble` is a builtin op, and `DeriveCallDirections` stamps
`arg_directions` on non-builtin calls only.

`ArrayType` iter_args are **excluded** from the nested-loop rule. Unlike a
`TensorType` (a pointer-to-buffer alias), an `ArrayType` iter_arg owns a *fresh*
C-stack array at each level. Treating the inner return_var as an alias of the
outer iter_arg would mark the outer slot trivial and silently drop the outer
yield-back copy — the very mechanism that propagates state across phases in a
`SEQ x PARALLEL` phase fence.

## Stamped attributes

`ForStmt::attrs_` is a flat `string → scalar` map, so the plan uses
index-suffixed keys:

| Key | Type | Meaning |
| --- | ---- | ------- |
| `iter_arg_rebind_<i>` | `bool` | `True` = materialised carry, `False` = trivial alias. Stamped for **every** slot, so its presence proves the pass ran. |
| `iter_arg_array_size_<i>` | `int` | `PTO2TaskId[N]` fence-array extent. Stamped only when positive; absence means the scalar / tensor / `ArrayType` carry path. |

Read them with `ir::transform_utils::IterArgIsRebind()` /
`IterArgArraySize()` (`include/pypto/ir/transforms/utils/transform_utils.h`) —
never by string-matching the keys.

## Example

```python
@pl.function(type=pl.FunctionType.Orchestration)
def main(self, x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    for _i, (acc,) in pl.range(0, 4, init_values=(out,)):
        acc2 = self.accumulate(x, acc)   # writes `acc` in place, returns it
        (out,) = pl.yield_(acc2)
    return out
```

After the pass the loop carries `attrs={"iter_arg_rebind_0": False}`: `acc2`
aliases `acc` through the InOut writeback rule, so codegen routes both `acc` and
`out` to the parameter's emit name and skips the yield self-assign.

Swapping the body for a `pl.create_tensor` result yields
`attrs={"iter_arg_rebind_0": True}` instead, and codegen declares
`Tensor <carry> = <init>;` plus a yield-time assignment.

Inside a manual scope, a TaskId carry on `pl.parallel(4)` yields
`attrs={"iter_arg_rebind_0": True, "iter_arg_array_size_0": 4}`, lowering to
`PTO2TaskId arr[4];`.

## Errors

A `pl.parallel` loop carrying a manual-scope dependency (`deps=[...]`) must have
a statically-known trip count — the runtime fence needs a `PTO2TaskId[N]` array
of fixed `N`. A dynamic trip count raises a user-facing error:

```text
manual_scope: pl.parallel loops carrying a manual_scope dep (via ``deps=[...]``)
must have a statically-known trip count. ...
```

The diagnostic surfaces during this pass, before codegen runs.

## Pass properties

| - | Properties |
| - | ---------- |
| Required | `CallDirectionsResolved`, `RuntimeScopesMaterialized` |
| Produced | `IterArgCarryClassified`, `RuntimeScopesMaterialized` |
| Invalidated | — |

`IterArgCarryClassified` is a codegen precondition (see
`VerifyOrchestrationCodegenPreconditions`) and has a registered property
verifier: a `ForStmt` with iter_args and no `iter_arg_rebind_<i>` attr means the
pass never ran, and codegen would silently lower every carry as a trivial alias.

## Compiler-derived dependency carries

The orchestration codegen overlays two further flags on the stamped plan for
iter_args that collect *compiler-derived* task dependencies
(`attrs["compiler_manual_dep_edges"]`, produced by `AutoDeriveTaskDependencies`).
Those depend on program-wide dependency edges rather than on the loop's own
structure, so they stay in codegen: the carry is forced to `rebind`, sized from
the outer loop's const trip count, and falls back to a `std::vector<PTO2TaskId>`
collection when that trip count is dynamic.

## See also

- [MaterializeRuntimeScopes](41-materialize_runtime_scopes.md) — the pass that runs immediately before
- [Orchestration codegen](../codegen/01-orchestration_codegen.md) — the consumer of the stamped plan
- [Pass manager](00-pass_manager.md)
