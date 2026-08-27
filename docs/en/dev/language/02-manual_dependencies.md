# Manual dependency primitives

By default the runtime auto-derives task→task dependencies from buffer
read/write overlap (the `OverlapMap`). The DSL exposes **two orthogonal
mechanisms** the user can combine:

> **The two mechanisms are independent.** Opting a buffer / region / arg
> out of auto-tracking does **not** require declaring explicit edges, and
> declaring explicit edges does **not** require turning auto-tracking off.
> The final task fanin is **`auto-tracked deps ∪ explicit deps`** — they
> compose, they don't substitute for each other.

## Mechanism A — opt out of auto-dep tracking (3 granularities)

All three granularities are independent of each other. Pick the smallest
unit that fits your use case; combine if needed.

| Surface | Granularity | Effect |
| ------- | ----------- | ------ |
| `with pl.manual_scope():` | per-region | Lowers to `SIMPLER_SCOPE(ScopeMode::MANUAL)`. Inside, the runtime never auto-tracks; the user must declare every required ordering edge explicitly (see Mechanism B). |
| `pl.create_tensor([...], dtype=..., manual_dep=True)` | per-tensor lifetime | Every task that reads or writes this tensor skips `OverlapMap` lookup and insert for its **entire lifetime**, regardless of scope. Useful for scratch buffers that are managed entirely by explicit edges. |
| `pl.no_dep(arg)` | per-call argument | At a kernel call site, the wrapped argument's `ArgDirection` becomes `NoDep` — auto-tracking ignores that slot **for this submission only**. Legal regardless of whether the callee declares the slot as `In`, `Out`, or `InOut`: the user asserts out-of-band that there is no RaW / WaW / WaR conflict on this slot (e.g. paged-attention writes whose offset is data-dependent but disjoint by allocation protocol). No effect inside `pl.manual_scope` (the scope already disables auto-tracking). |
| `with pl.at(..., no_dep_args=[t1, t2]):` | per-arg, on a `pl.at`-block | The `pl.at`-block analogue of `pl.no_dep(arg)`. The outliner makes the listed tensors arguments of the synthesised kernel call; `DeriveCallDirections` then forces those arg slots to `NoDep` — same effect as wrapping the tensors with `pl.no_dep(...)` at an explicit call site. Each entry must be a bare tensor name visible to the enclosing scope. Same In / Out / InOut applicability as `pl.no_dep(arg)`: a captured tensor that the scope body mutates via `pl.assemble` becomes `InOut` on the synthesised kernel, and `no_dep_args=` overrides it to `NoDep` just as it overrides `In`. Note: `no_dep_args=` takes **tensors**, while `deps=` takes **TaskIds** — same word "dep", different layer. |

## Mechanism B — declare explicit task→task edges (`deps=`)

These surfaces all produce `set_dependencies` codegen; choose by producer
shape (single kernel call, outlined `pl.at` region, or dependency-only fan-in).

| Surface | Producer shape | Notes |
| ------- | -------------- | ----- |
| `result, tid = pl.submit(kernel, *args, deps=[...], allow_early_resolve=False)` | single kernel call | The trailing `tid` is the producer `pl.Scalar[pl.TASK_ID]`. A parser construct (like `pl.range`), not a runtime function. `allow_early_resolve=True` opts this task in as a speculative early-dispatch producer (lets the scheduler pre-stage its consumers; lowers to `Arg::set_allow_early_resolve(true)`). Also accepts `predicate=(t[i] > 0)` — a dispatch predicate the scheduler evaluates at the dispatch point (see [Dispatch predicate](#dispatch-predicate-predicate)). |
| `result, tid = pl.spmd_submit(kernel, *args, core_num=N, sync_start=False, deps=[...])` | single SPMD task launch | The SPMD sibling of `pl.submit`: dispatches the kernel across `N` blocks (one orchestration task → one `tid`). `core_num` is a required keyword (positive int expr); `sync_start=True` forces atomic launch of all blocks. Callee may be InCore / AIC / AIV / Group. Records the launch spec on `Submit.core_num` / `Submit.sync_start`. Also accepts `allow_early_resolve=True` (same early-dispatch opt-in as `pl.submit`) and `predicate=(t[i] > 0)` (see [Dispatch predicate](#dispatch-predicate-predicate)). |
| `with pl.at(level=pl.Level.CORE_GROUP, deps=[...]) as tid:` | outlined `pl.at`-block | The whole block is outlined into an `InCore` kernel + `Submit`; `tid` captures the synthesized Submit's TaskId, usable as a dep for later `pl.submit` / `pl.at` sites. Without `as tid` the outliner synthesizes an unused TaskId Var — deps always travel on `Submit::deps_`. Also accepts `allow_early_resolve=True` (same early-dispatch opt-in as `pl.submit`); it forces the `Submit` shape even without `as tid` and lowers to `Arg::set_allow_early_resolve(true)`. |
| `with pl.spmd(N, deps=[...]) as tid:` | outlined SPMD dispatch | The SPMD sibling of the `pl.at ... as tid` form. The inline body is auto-outlined into an `InCore` kernel and dispatched across `N` blocks; `tid` captures the grid-wide producer TaskId. `deps=` accepted only with `as tid`. `core_num` / `sync_start` ride on the lowered `Submit`'s own `core_num` / `sync_start` fields (the launch spec belongs to the launch site, not the outlined callee); codegen reads them from there. Also accepts `allow_early_resolve=True` (same early-dispatch opt-in as `pl.submit` / `pl.at`; valid on all three `pl.spmd` forms, forcing the `Submit` shape even without `as tid`) and `predicate=(t[i] > 0)` (see [Dispatch predicate](#dispatch-predicate-predicate); also valid on all three forms and also forces the `Submit` shape). Cannot nest inside `pl.cluster()`. |
| `barrier = pl.system.task_dummy(deps=[...])` | dependency-only barrier | Submits no kernel. The returned TaskId is a compact fan-in point for later `deps=[barrier]`. |
| `None` (Python literal) | seed / dep entry | The "no producer yet" sentinel. `prev_tid = None` seeds a TaskId loop iter_arg; `None` in `deps=[None]` is dropped (contributes no edge). Lowers to `system.task_invalid` → `TaskId::invalid()`. |

**These surfaces work regardless of Mechanism A state.** Use explicit deps in
plain auto-tracked orchestration, inside `pl.manual_scope()`, or with a
`manual_dep=True` tensor; explicit edges are added on top of auto-tracking.
The earlier "`deps=` only inside `pl.manual_scope`" restriction no longer applies.

Plain `out = self.kernel(...)` is **fire-and-forget**: it returns no task
id, and `deps=` is rejected on it (the parser raises, hinting "use
`pl.submit`"). Each `deps=[...]` entry must be a TaskId value: a `tid`
bound by a prior `pl.submit(...)` / `pl.at(..., deps=) as tid`, the result of
`pl.system.task_dummy(deps=[...])`, a TaskId loop iter_arg carry, a
`Scalar[TASK_ID]` read from a TaskId array slot (`prev = tids[k]`), an
`Array[N, TASK_ID]` from `pl.array.create(N, pl.TASK_ID)`, or the literal
`None`. Tensors are **not** accepted in `deps=[...]`.

```python
# Example 1 — both mechanisms together: scope-wide opt-out + explicit edge.
@pl.function(type=pl.FunctionType.Orchestration)
def main(self, x: pl.Tensor[[64], pl.FP32],
         scratch: pl.Out[pl.Tensor[[64], pl.FP32]],
         out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
    with pl.manual_scope():                                           # Mechanism A: scope-wide
        scratch, stage1_tid = pl.submit(self.stage1, x, scratch)
        out, _ = pl.submit(self.stage2, scratch, out, deps=[stage1_tid])  # Mechanism B
    return out
```

```python
# Example 2 — Mechanism B alone, NO manual_scope. Auto-tracking stays on
# for everything else; the explicit edge is *added on top* of whatever
# auto-tracking decided. Note the absence of `with pl.manual_scope():`.
@pl.function(type=pl.FunctionType.Orchestration)
def main(self, x: pl.Tensor[[64], pl.FP32],
         out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
    tmp, prep_tid = pl.submit(self.preprocess, x)
    out, _ = pl.submit(self.consume, tmp, out, deps=[prep_tid])
    return out
```

```python
# Example 3 — pl.at-block as the producer, with deps= on a downstream block.
# `as tid` captures the synthesized outlined-Call's TaskId.
@pl.function(type=pl.FunctionType.Orchestration)
def main(self, x: pl.Tensor[[64], pl.FP32],
         out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP) as tid_a:
        # body becomes an outlined InCore kernel
        ...
    with pl.at(level=pl.Level.CORE_GROUP, deps=[tid_a]) as tid_b:
        # explicit edge — runs strictly after the `tid_a` block
        ...
    return out
```

```python
# Example 4 — Mechanism A tensor-lifetime: scratch buffer opted out for its
# whole lifetime; explicit edge still pins the ordering between producer
# and consumer.
scratch = pl.create_tensor([N], dtype=pl.FP32, manual_dep=True)
scratch, prod_tid = pl.submit(self.fill, x, scratch)
out, _ = pl.submit(self.consume, scratch, out, deps=[prod_tid])
```

`pl.submit` desugars to a single `ir.Submit` whose return type is the flat
augmented `TupleType([*<kernel return types>, ScalarType(TASK_ID)])` —
elements `0..N-1` are the kernel results, element `N` is the producer
TaskId. The parser writes each `deps=[...]` list directly into the typed
`Submit::deps_` field (no plain `Call` ever carries `manual_dep_edges` —
the ManualDepsOnSubmitOnly invariant). `pl.at(..., deps=)` follows the same
path: the outliner reads `attrs["task_id_var"]` and `attrs["manual_dep_edges"]`
on the `ScopeStmt` and lifts them onto a synthesized `Submit` (a scope with
deps but no `as tid` gets a synthetic unused TaskId Var so the dispatch is
still a Submit). Codegen fills a fixed-size stack array sized to the
exact dep count and emits one `params.set_dependencies(arr, count);`
call per task. The runtime's `Arg::set_dependencies(ptr, count)` accepts a
caller-owned array of arbitrary size, so there is no per-call edge cap.
For explicit fan-in, write `barrier = pl.system.task_dummy(deps=[tids])` and
then `pl.submit(..., deps=[barrier])`; it uses the same dependency parser,
lowers to `rt_submit_dummy_task(...)`, returns invalid without submitting when
all deps are invalid, and coexists with automatic `ExpandManualPhaseFence`
barriers for profitable full-array phase fences.

`pl.no_dep(arg)` is an auto-scope primitive; inside `pl.manual_scope` it
has no effect (the whole region already skips auto-tracking).

## Dispatch predicate (`predicate=`)

`pl.submit` / `pl.spmd_submit` accept an optional
`predicate=(tensor[indices] <op> target)`. The scheduler evaluates the
comparison **at the dispatch point** — after the task's dependencies are
satisfied, so the value is current without an orchestration-time
`wait_for_tensor_ready` stall. When the comparison is **false** the task is
**retired inline** (never dispatched to a core) while still settling
fanin/fanout, so downstream consumers still unlock — it does not vanish from the
task graph. When **true** it dispatches normally.

The canonical use is MoE "skip empty experts": submit every expert statically,
each carrying `predicate=(row_count[e] > 0)` and depending on the gather
producer — the scheduler dispatches only the non-empty experts, without stalling
orchestration to read the per-expert count.

> **The comparison is parsed as an ordinary expression, but never evaluated.**
> `rc[0, 0]` is the usual sugar for `pl.read`, so the kwarg lowers to plain IR —
> `Gt(Cast(tensor.read(rc, [0, 0])), 0)` — reusing the IR's existing comparison
> nodes rather than any bespoke encoding. It is stored on `Submit.predicate`,
> never in a statement position, so the `tensor.read` is **not** executed in
> orchestration: doing so would stall on `wait_for_tensor_ready`, exactly what
> the predicate exists to avoid. Orchestration codegen decomposes the Expr into
> the runtime's `operand OP target` triple, so only the shape below is accepted.

| Part | Meaning | Constraint |
| ---- | ------- | ---------- |
| `tensor` | operand tensor read at the dispatch point | must be a named tensor (a parameter or a variable bound to one), subscripted to one element |
| `indices` | element locator into `tensor` | each index an integer scalar (`ConstInt` or an int/index `Var`); one index per `tensor` dimension |
| `<op>` | comparison | one of `==` `!=` `>` `<` `>=` `<=` (a single, unchained comparison) |
| `target` | right-hand side | an **integer literal** (may be negative) |

The mirrored order is accepted — `0 < rc[e]` means the same as `rc[e] > 0`. The
IR keeps the comparison as written; orchestration codegen flips the operator so
the tensor is always the runtime's operand.

Lowers to the runtime `CoreTaskPredicate` + `Arg::set_predicate(...)` in
orchestration codegen (operand → its `ext_<name>` reference, `op` →
`PredicateOp::*`, `target` verbatim; `elem_size` is derived by the runtime from
the tensor dtype).

**Contract:** the predicate operand tensor's producer **must** be one of the
submit's `deps=`, so the dispatch-point read observes the current value. Omitting
it lets the scheduler evaluate the predicate before the producer has written the
tensor, deciding from stale data.

The parser makes a **best-effort spot check**, not a guarantee: it tracks the
result variables a `pl.submit(...)` binds via tuple unpacking, and rejects a
predicate whose operand is one of them when the producing TaskId is absent from
`deps=`. Treat a clean parse as "no obvious mistake found", not as proof.

It does **not** see through, and therefore silently accepts:

| Not covered | Why |
| ----------- | --- |
| `rc2 = rc` then `rc2[0, 0]` | the alias is a fresh variable with no recorded producer |
| a tensor passed as an `pl.Out` argument and rebound under a new name | only the returned binding is tracked, not the argument alias |
| `rc3 = self.helper(rc)` | any intervening call launders the association |
| `res = pl.spmd_submit(...)` (single-target form) | the single-target path records nothing |
| any `deps=` list containing an `Array[N, TASK_ID]` entry — including the common `deps=[tids[i]]` | array entries do not name their producers individually, so the whole check is skipped for that submit |
| a producer written **later** in the source, e.g. a loop-carried `rc` written by the previous iteration | the lookup happens while parsing the predicate, so producers that follow it are not yet recorded |

Getting `deps=` right therefore remains the author's responsibility.

**Expressiveness** is fixed to `tensor[indices] OP const` — one comparison,
matching the runtime's single-comparison `DispatchPredicate`. Chained
comparisons (`0 < t[i] < 8`), arithmetic (`t[i] % 8 == 0`), boolean combination
(`a[0] > 0 and b[0] > 0`), and a non-literal right-hand side (`t[i] > u[i]`) are
all rejected at parse time; reduce anything richer to a single gate value in a
prior kernel and predicate on that.

```python
with pl.manual_scope():
    rc, g_tid = pl.spmd_submit(self.gate, rc, core_num=1)       # producer of rc
    out, _ = pl.spmd_submit(
        self.expert, x, out, core_num=1,
        deps=[g_tid],                                           # producer is a dep
        predicate=(rc[0, 0] > 0),
    )
```

**Scope:** `predicate=` is accepted on `pl.submit` / `pl.spmd_submit` (the
direct-`Submit` forms) and on the `with pl.spmd(...)` scope form — all three
spmd spellings (plain `with`, `with ... as tid`, and `for i in pl.spmd(...)`).
It is not accepted on `pl.at(...)`.

### Scope form

The scope form takes the same expression and the same validation; it differs
only in how the predicate reaches the IR. It rides on `SpmdScopeStmt.attrs`
until the scope is outlined, at which point it moves onto `Submit.predicate` —
so the lowering, the codegen output, and the contract are identical.

```python
with pl.spmd(1) as g_tid:                                       # producer of rc
    rc = self.gate(rc)

with pl.spmd(4, deps=[g_tid], predicate=(rc[0, 0] > 0)) as tid:  # producer is a dep
    out = self.expert(x, out)
```

Two things follow from that route:

- **`deps=` needs the `as tid` form.** `deps=` is only accepted on
  `with pl.spmd(...) as tid:`. A predicate over a tensor produced elsewhere in
  the same function therefore needs that form — the plain `with` and `for`
  forms can only carry a predicate over a tensor with no in-function producer
  (typically a function parameter), which the contract check permits.
- **No `as tid` is required otherwise.** Like `allow_early_resolve=True`, a
  predicate forces the scope to lower to a `Submit`; when the scope has no
  `as tid` the outliner synthesises an unused TaskId Var.

A cluster-nested `pl.spmd` is unwrapped into the Group function and never
produces a `Submit`, so `predicate=` (like `allow_early_resolve=`) is rejected
there at parse time rather than silently dropped.

The contract check covers scope producers too: a tensor assigned inside a
`with pl.spmd(...) as tid:` body is recorded as produced by that scope, so
omitting it from a later `deps=` is rejected. The same best-effort limits in the
table above still apply (aliases, intervening calls, `Array[N, TASK_ID]` deps).

## `pl.parallel` under manual scope: array-carry fence

When a manual-dep edge is carried through a `pl.parallel` loop (i.e. the
loop's iter_arg holds the TaskId being depended on), the orchestration codegen
treats the corresponding TaskId iter_arg as **an array of size equal to the
parallel loop's trip count**. Each parallel iteration writes its own slot,
and downstream consumers depend on **every** slot (not just the
last-dispatched task). This guarantees the user-declared fence semantics
even when iters finish out of dispatch order.

Requirements for the array-carry path:

- The `pl.parallel` trip count must be a Python literal (statically known).
  A dynamic trip count under `pl.parallel` carrying a manual dep is rejected
  at codegen with a "statically-known trip count" message.

```python
with pl.manual_scope():
    prev_tid = None                                      # seed: no producer yet
    for phase in pl.range(N_PHASES):
        for branch in pl.parallel(N_BRANCHES):           # const trip count
            row = (phase * N_BRANCHES + branch) * TILE_M
            out, prev_tid = pl.submit(self.kernel_stripe, data, row, 1.0, out, deps=[prev_tid])
```

`prev_tid` is rebound inside `pl.parallel`, so codegen lowers the carry as
a `TaskId[N_BRANCHES]` array. Each task in phase `N+1` waits for all
`N_BRANCHES` tasks of phase `N`, not just the last-dispatched one.

## See Also

- [Statements and Control Flow](01-statements.md) — the scope context managers these build on
- [Orchestration Codegen](../codegen/01-orchestration_codegen.md) — how these lower
- [AutoDeriveTaskDependencies](../passes/39-auto_derive_task_dependencies.md) — the pass that consumes them
