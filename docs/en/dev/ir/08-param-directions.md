# Parameter Direction Inference

`Function::param_directions_` says, for each parameter, whether the function
reads it, writes it, or both — `In`, `Out`, `InOut`. Almost nothing downstream
works without it: dependency analysis emits RAW edges from it, the distributed
codegen tags each per-rank chip dispatch from it, and the host ABI decides from
it which buffers the caller must allocate.

Getting it wrong is quiet. A parameter that reads `In` while the body writes it
produces no error and no wrong number at compile time — the dependency edge it
needed is simply never emitted, and the program races or deadlocks on device.

This page is the whole chain in one place. Each stage has its own page; this one
says what the stages are, which question each answers, and where the answers
come from.

## The single source of truth

Every stage below asks the same question — *does this call write the buffer this
argument names?* — and every stage reads the answer from the same place: the
operator's own registry declaration.

```cpp
REGISTER_OP("tile.mscatter")
    .set_arg_effect(2, ArgEffect::Write)        // fixed
    .set_write_channel(WriteChannel::Dma);

REGISTER_OP("tile.mgather")
    .set_arg_effect(2, [](const auto& kwargs) { // kwarg-dependent: `scratch` is
      ...                                       // written only in Mat elem mode
      return mat_output && elem_mode ? ArgEffect::Write : ArgEffect::Read;
    });

REGISTER_OP("system.set_ffts")
    .no_arg_writes();                           // "writes nothing" must be said
```

See [Operator System — Argument effects](05-operators.md#argument-effects) for
the full declaration surface. The query side is
`GetArgEffect(i, kwargs)`, `HasDeclaredArgEffect(i)`, `WritesAnyArg()` and
`GetOutputReusesInputArg()`.

This was not always so. The write set used to be a hand-kept list of operator
names inside each analysis, and the lists disagreed — `pld.system.notify`
reached production writing a signal no analysis knew about (#2391), and
`tile.mscatter` was in the same state until its effects were declared.

### What the registry gate does and does not cover

`OpRegistry::ValidateArgEffects()` runs at registration and rejects two shapes:

- an operator declaring `set_output_reuses_input(N)` without classifying
  argument `N` — a reuse contract implies a verdict about that argument; and
- an operator declaring a write channel while writing through no argument — a
  channel describes *how* an operator writes, so one without a write is either a
  stray declaration or a missing one.

An operator with **neither** — no reuse contract, no write channel — trips
neither gate. Omitting `set_arg_effect` there is still silent. That is the
original failure class, and it is not closed; see
[Known limits](#known-limits).

## The chain

| Stage | Pass | Question it answers |
| ----- | ---- | ------------------- |
| [Outlining](#1-outlining-passes-79) | 7 / 8 / 9 | What directions does the function I am *creating* have? |
| [Caller propagation](#2-caller-propagation-pass-10) | 10 | A callee writes this argument — does my own parameter behind it become `Out`? |
| [Wrapper recovery + call sites](#3-wrapper-recovery-and-call-sites-pass-37) | 37 | What is each wrapper's *effective* signature, and what direction does each call-site argument have? |
| [Consistency warning](#4-consistency-warning-postpipeline) | PostPipeline | Does any parameter still read `In` while its body writes it? |

### 1. Outlining (passes 7–9)

`ScopeOutliner::InferParamDirections` gives a freshly outlined scope function its
signature. Four steps, each a *lower* bound on the accesses — no step may
overwrite what another saw:

| Step | Evidence |
| ---- | -------- |
| 0 | `ParamReadCollector` walks the body for reads |
| 1a | exported store targets are writes |
| 1b | `CallWriteTargets` — writes the registry declares |
| 2 | inner callees' declared slots |

Two subtleties carry most of the correctness.

**Step 0 skips slots the callee purely overwrites.** Passing a capture to a
write-only slot moves no data into the scope, so it is not a read. Two
declarations say which slots those are, one per kind of callee: a builtin's
`ArgEffect::Write` and a user function's `ParamDirection::Out`.

**Step 2 does not merge along `In < Out < InOut`.** `In` is the seeded *no
evidence yet* floor, so it cannot also mean "somebody read this" — reading it
that way would promote every write-only capture to `InOut`, the false read that
turns disjoint per-rank slices of one `pl.Out` tensor into a cross-rank
dependency (issue #2415). The callee slots are accumulated as two independent
flags — `In`/`InOut` marks a read, `Out`/`InOut` marks a write — and the
direction is derived once at the end.

See [Outline InCore Scopes](../passes/08-outline_incore_scopes.md).

### 2. Caller propagation (pass 10)

`ConvertTensorToTileOps` phase 3 lifts a caller's parameter to `Out`/`InOut`
when the caller forwards it into a callee slot the callee writes.

The argument is rarely the parameter itself. A loop-carried tensor arrives as an
`IterArg` whose value is the parameter's buffer:

```python
acc = dst
for _ in pl.range(4):
    acc = self.kernel(x, acc)     # kernel(x, out: pl.Out[...])
```

So the argument is resolved to the buffer it owns — via `BufferRootCollector` —
before the parameter lookup. An argument that *is* the parameter resolves to
itself, so this generalises the identity lookup rather than replacing it.

### 3. Wrapper recovery and call sites (pass 37)

`DeriveCallDirections` runs in two parts.

**Phase 0 — write each Group/Spmd wrapper's effective directions back into its
own signature.** A wrapper forwards its parameters 1:1 to an inner kernel, but
its own `param_directions_` can still read `In` for a parameter that kernel
writes: the outliners infer from the body they extracted, and
`ExpandMixedKernel` / `SplitVectorKernel` later rebuild wrapper bodies around
freshly split callees without revisiting the signature.

Recovering this used to be recomputed on the fly in four places — this pass,
`AutoDeriveTaskDependencies`, the `CallDirectionsResolved` verifier, and
orchestration codegen. Doing it once and storing it in the IR gives every
consumer one source of truth: `callee->param_directions_`.

**Then the call sites.** Each argument gets an `ArgDirection` — `Input`,
`Output`, `OutputExisting`, `InOut`, `NoDep`, `Scalar` — which is what
dependency analysis and codegen actually consume.

See [Derive Call Directions](../passes/38-derive_call_directions.md).

### 4. Consistency warning (PostPipeline)

`DiagnosticCheck::InParamWritten` reports a parameter that still reads `In` while
its own body writes it. It reads the same two declarations the inference reads —
registry effects and a callee's `param_directions_` — and reports where they
contradict the parameter.

It is a **warning, not an `IRProperty`**, and that is forced rather than chosen:
the check must run after `DeriveCallDirections` (pass 37), and `InitMemRef`
(pass 31) invalidates `SSAForm` with nothing re-establishing it, so no pipeline
position is both after pass 37 and in SSA form. Its buffer lineage is therefore
best-effort across control flow.

See [Verifier — InParamWritten](../passes/99-verifier.md#inparamwritten).

## Shared analyses

Three helpers are read by more than one stage, which is what keeps the stages
from drifting apart:

| Helper | Answers |
| ------ | ------- |
| `BufferRootCollector` | which buffer does this variable own? |
| `ResultAliasedArgIndex` | which argument's buffer does this call's result name? |
| `op_predicates::IsBufferAliasingViewOp` | is this a zero-copy view of its input? (`OutputMemoryInheritsInput() && IsInplaceSafe()`) |

The third excludes `tile.transpose`: it inherits the memory *space* but permutes
into a fresh buffer (`pto.ttrans` is registered `not_inplace_safe()`), so its
output does not alias its input.

## Known limits

Two gaps are documented rather than closed. The first only under-reports; the
second errs in **both** directions, which is worth stating plainly because an
earlier version of this page claimed otherwise.

**A missing declaration is invisible.** Every stage reads the registry, so an
operator that never declared its effects contributes no write anywhere, and the
consistency warning cannot see it either — it is reading the very declaration
that is absent. `ValidateArgEffects` narrows this but does not close it, as
above.

**The consistency warning's lineage is not per-access.** It runs on non-SSA IR
(see above), where one environment per name is not exact. `BufferRootCollector`
scans the whole body up front, so a rebound name carries a single *final*
mapping that is applied to earlier writes too:

```python
t = buf1
t = pl.tile.assemble(t, patch, [0, 0])   # writes buf1
t = buf2                                  # ... but the map says t -> buf2
```

`buf2` is reported although nothing writes it, and `buf1` is missed although it
is written — a false positive and a false negative from the same cause. Pinned
as a strict `xfail` in `tests/ut/ir/verifier/test_in_param_written.py`.

A view built inside a branch is *not* an instance of this. Its lineage does
survive the join, but the write after the join really does reach that buffer on
the taken path, so naming it is the correct may-write answer; that case is
pinned as a passing test.

## See Also

- [Operator System](05-operators.md) — the declaration surface, in full.
- [Outline InCore Scopes](../passes/08-outline_incore_scopes.md) — stage 1.
- [Convert Tensor to Tile Ops](../passes/10-convert_tensor_to_tile_ops.md) — stage 2.
- [Derive Call Directions](../passes/38-derive_call_directions.md) — stage 3.
- [IR Verifier](../passes/99-verifier.md) — stage 4.
