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
Ordinary InCore kernels with different physical-shape symbol identities can
merge when every affected Call/Submit binds the canonical signature consistently
and reproduces the original caller-coordinate result type. This includes calls
inside Spmd wrappers, scalar parameters also used as dimensions, repeated symbols,
and same-structure expressions such as `N * 128`. Bindings come from supplied
arguments, never from a missing output. Shape-distinct aliases must supply every
parameter, including Submit outputs: runtime-synthesized outputs still read the
callee signature directly, without these call-site bindings. Identical-signature
merges retain their existing omitted-output behavior. No algebraic inversion or runtime shape
guard is introduced. Return aliases and void memory effects remain unchanged.

The initial contract accepts scalar parameters and ordinary Tensor types without
storage, optionally carrying a layout-only TensorView (including NZ weights).
Strides, valid-shape metadata, distributed storage, unsupported expressions and
Group bundles retain exact signature-symbol identity requirements. Signature
parameter/return counts and unique shape-expression nodes are capped at 256 for
the new proof; larger signatures retain the old path. A failed proof retains the
entire candidate, even if another call could be rewritten safely.

Binding verification and closure inspection include types, shapes, views and attrs.
Free captures, unresolved value types, explicit storage/communication resources, external or
builtin kernels, runtime bindings and unsupported helper graphs are skipped.
Ordinary memory effects are allowed: all executions and actual arguments remain.
An unsupported payload that prevents complete reference indexing conservatively
preserves the Program. Debug logs identify unsupported projection diagnostics.

Two narrowly checked exceptions admit additional kernels:

- Legacy statement-only calls may carry `UnknownType` only as a direct
  `EvalStmt` result. The allowlist is `pld.tensor.put`, `pld.tile.remote_store`,
  `pld.tensor.remote_store`, `pld.system.notify`, `pld.system.wait`,
  `pld.system.defer_wait`, `system.set_ffts`, `system.sync_set` and
  `system.sync_wait`. Unknown values in signatures, operands, returns or metadata
  remain ineligible.
- Prefetch contexts, events and sessions must be produced and consumed locally
  by the corresponding `prefetch` operations. A wait must use the event's
  originating context. Handles cannot escape through signatures, metadata,
  aliases or arbitrary consumers. Each invocation still creates its own context.

Admission checks every incoming expression edge, including shared DAG nodes,
and nested type payloads. These exceptions do not relax structural equality or
the applicable signature proof.

## Rewriting and complexity

Call result types remain in caller coordinates. Submit keeps its deps, TaskId,
core count, synchronization, predicate and early-resolve fields. No task is removed.
Group aliases are applied atomically, followed by a dangling-reference check.

Units are considered in name order. Each hash bucket has at most eight representatives;
hash collisions still require exact equality. A comparison is charged both units'
projection weights against a budget of eight times the new unit's weight. Exhaustion
retains the original definition. Call-shape proof additionally consumes a global
budget equal to the inspected Program weight, with at most 65,536 matching steps
per site; shared expression pairs are memoized. The private projection inspection uses the same
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
