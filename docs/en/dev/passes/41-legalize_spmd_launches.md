# LegalizeSpmdLaunches

Documentation number **41**, after LowerL2TensorCollectives and before
DeriveCallDirections and AutoDeriveTaskDependencies. Counting the utility
NormalizeStmtStructure invocation, this is execution slot **42** in the default
pipeline; the documentation numbering omits that utility pass.

A zero-block SPMD compute launch can raise an AICPU exception. This pass uses
`arith::Analyzer` to prove `core_num >= 1`, including pure scalar SSA bindings,
loop bounds and enclosing branch constraints. Proven-positive launches are
unchanged. Otherwise, it emits `if core_num > 0`; the non-positive path skips
the compute launch and preserves the caller's existing output tensors.
Literal non-positive counts are still rejected by the DSL parser.

## Output contract

Every Tensor Out/InOut parameter must have a caller-provided argument, even
when it is unused or absent from the return list. A missing output produces a
source-located error naming the launch, tensor and zero-based parameter index.
The diagnostic lists all missing outputs. Preallocate those tensors and pass
them to the launch. A kernel with only Out parameters is supported.

Return slots map to parameters by Var identity using the canonical returns
established by NormalizeReturnOrder, then to actual caller arguments. Multiple
outputs, reordered returns and views retain their own mapping. An escaping
return that cannot be mapped to an existing Tensor is rejected. The empty path
does not initialize output bytes: a consumer sees the buffer's previous contents.
A launch already proven positive can still use runtime-allocated outputs.

## SSA and dependencies

Conceptual IR (the printed spelling of temporaries may differ):

```python
if n > 0:
    (a_then, b_then), tid_then = pl.spmd_submit(
        self.kernel, a0, b0, core_num=n, deps=[prior]
    )
    a, b, tid = pl.yield_(a_then, b_then, tid_then)
else:
    tid_empty = pl.system.task_dummy(deps=[prior])
    a, b, tid = pl.yield_(a0, b0, tid_empty)
result, _ = pl.submit(self.consumer, a, b, deps=[tid])
```

The IfStmt defines the merged results once, outside both branches. Tuple
projections are expanded inside the positive branch and consumers refer to
outer results. Generated Tensor results use the mapped caller buffer type,
so dynamic shape variables belong to the caller rather than the callee. The dummy task preserves explicit predecessor ordering, even
when no compute task executes. Launch metadata remains on the positive path.

AutoDeriveTaskDependencies analyzes branches against the same incoming storage
history, lifts yielded TaskIds to If results, and retains incoming producers
for the empty path. It falls back to runtime TensorMap tracking when a local
producer cannot escape safely; it never exports a branch-local dependency.
Explicit manual scopes retain their authored dependencies.

Compiler-injected GM pipe scratch used by a single launch moves into its
positive branch so allocation sizing stays adjacent to dispatch. User output
allocations stay outside the guard. Shared compiler scratch is rejected.
Dynamic launches inside Graph functions remain subject to Graph validation.

## API and properties

```python
from pypto import passes
result = passes.legalize_spmd_launches()(program)
```

Requires SSAForm, NoNestedCalls and ReturnParamsExplicit; preserves SSAForm
and NoNestedCalls and produces NormalizedStmtStructure. It invalidates call
directions, runtime scope materialization and loop carry classification, which
must run after the control-flow change. The pass is idempotent and needs no
configuration flag. Traversals and cached callee/tuple lookups are linear in
the IR and call argument sizes.

Codegen repeats the arithmetic check as a preflight and rejects a potentially
non-positive launch if legalization was omitted or a later pass introduced one.
