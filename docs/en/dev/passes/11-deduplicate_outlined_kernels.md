# DeduplicateOutlinedKernels

Runs after `OutlineClusterScopes` and before `ConvertTensorToTileOps` in the tensor
pipeline. It shares definitions inside one Program and preserves every Call and
Submit. Two equivalent kernels become one definition with two executions.

```python
# Before
x1 = self.step(x0)
x2 = self.step_copy(x1)
# After
x1 = self.step(x0)
x2 = self.step(x1)
```

## Eligibility and equality

Candidates are referenced, closed InCore functions or exclusive Group bundles.
Unreferenced functions are preserved as potentially exposed entries. Orchestration,
Graph and Spmd identities are retained. Group members are indexed before ordinary
kernels: an externally called or shared member prevents the whole bundle from
merging, including its members. A bundle includes the Group bridge and all direct
InCore/AIC/AIV members, in first-use order. Internal callees and known
`system.import_peer_buffer(peer_func=...)` names are normalized together.

The pass uses exact structural hash/equality with `enable_auto_mapping=False`.
Bound parameters and signature symbols are alpha-renamed; semantic attrs, directions,
result relationships, dtypes, layouts and constant bits remain significant.
Aliasing additionally requires exact signature types with shape/view symbol
identities preserved, for the root and every Group member. This restriction is
separate from structural equality: tensor-to-tile lowering allocates outputs from
the callee signature without mapping its symbols into each call's coordinates.
Thus `[N]` and `[M]` kernels remain separate even when alpha-equivalent; kernels
sharing the same symbols can still merge. Keeping the original call result type
alone does not protect downstream output allocation.
Binding verification and closure inspection include types, shapes, views and attrs.
Free captures, unknown types, explicit storage/communication resources, external or
builtin kernels, runtime bindings and unsupported helper graphs are skipped.
Ordinary memory effects are allowed: all executions and actual arguments remain.
An unsupported payload that prevents complete reference indexing conservatively
preserves the Program. Debug logs identify unsupported projection diagnostics.

## Rewriting and complexity

Call result types remain in caller coordinates. Submit keeps its deps, TaskId,
core count, synchronization, predicate and early-resolve fields. No task is removed.
Group aliases are applied atomically, followed by a dangling-reference check.

Units are considered in name order. Each hash bucket has at most eight representatives;
hash collisions still require exact equality. A comparison is charged both units'
projection weights against a budget of eight times the new unit's weight. Exhaustion
retains the original definition. The private projection inspection uses the same
DAG memoization and reflected fields as equality, including payload bytes. Overall
work is O(N log N) with O(N) storage; this is not a promise to find every duplicate.

Required and preserved properties: `SSAForm`, `SplitIncoreOrch`, `ClusterOutlined`,
`NormalizedStmtStructure`. There is no “all functions unique” property.

## Code and registration identity

Codegen exports `OrchestrationResult.kernel_registrations`, replacing the old
`func_name_to_id`, `func_name_to_core_type` and `func_name_to_signature` maps.
Each `KernelRegistration` has `func_id`, `code_name`, `core_type` and `signature`.
IDs are assigned after final task-parameter ordering, using code, core type and
tensor signature. Scalars are omitted; NoDep maps to runtime INOUT while its task
still uses `add_no_dep`. Equal signatures reuse an ID; different signatures can
reference the same code with distinct IDs.

Generated configuration carries one KERNELS row per registration. The runtime
compiles each generated code/compilation contract once, then builds each callable
with its own signature. External, builtin-template and legacy configurations retain
separate compilation identities. Generated cache keys include source/header content;
binary-context invalidation still applies. Prebuilt schema
1 permits several registrations to reference one validated binary file; duplicate
function IDs remain invalid. Binary reuse does not imply shared device load addresses.
