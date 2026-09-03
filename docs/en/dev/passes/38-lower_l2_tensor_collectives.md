# LowerL2TensorCollectives Pass

## Overview

`LowerL2TensorCollectives` is the CHIP/L2 rail for managed collectives. It
rewrites a `pld.tensor.*` collective written in a **CHIP orchestration body**
into a call to a compiler-synthesized AIV kernel, so the collective becomes one
ordinary task inside the caller's own pipeline.

Today it handles `pld.tensor.all_to_all_v` with `core_num=1`.

The HOST rail ([`LowerHostTensorCollectives`](44-lower_host_tensor_collectives.md))
solves the same problem one level up, and differently: it fans the collective
out into one `builtin.tensor.*` chip dispatch *per device*. Each such dispatch is
a whole extra L2 orchestration task whose only job is to submit one AIV kernel,
so a `compute -> collective -> consume` sequence costs three L3 -> L2 round
trips per rank. On this rail it costs one.

```text
HOST rail                                CHIP rail (this pass)
─────────                                ─────────────────────
L3 -> L2  stage task                     L3 -> L2  chip_pipeline
L3 -> L2  builtin collective dispatch              ├── stage         (AIV task)
          └── rt_submit_aiv_task                   ├── collective    (AIV task)
L3 -> L2  consume task                             └── consume       (AIV task)
```

## Position in the pipeline

```text
... -> FuseCreateAssembleToSlice -> LowerL2TensorCollectives -> DeriveCallDirections -> AutoDeriveTaskDependencies -> ...
```

The position is load-bearing. The emitted call must reach
[`DeriveCallDirections`](39-derive_call_directions.md) and
[`AutoDeriveTaskDependencies`](40-auto_derive_task_dependencies.md) like any
other kernel call: those two passes are what turn the synthesized kernel's
parameter directions into the TensorMap edges that order
`compute -> collective -> consume`. Running the rewrite after them would leave
the collective task unordered.

It also runs before
[`MaterializeDistTensorCtx`](45-materialize_dist_tensor_ctx.md), which appends
the `CommCtx` arguments the kernel needs (see *ABI* below).

## Behavior

For a CHIP orchestration body:

```python
@pl.function(type=pl.FunctionType.Orchestration)
def chip_pipeline(self, inp, out, stage, data, signal, counts, recv):
    stage, counts = self.stage_step(inp, stage, counts)
    data = pld.tensor.all_to_all_v(stage, data, signal, counts, recv, core_num=1)
    return self.consume_step(data, recv, out)
```

the collective becomes:

```python
data = self.__builtin_all_to_all_v__fp32(stage, data, signal, counts, recv)
```

where `__builtin_all_to_all_v__fp32` is a synthesized `FunctionType.AIV`
function added to the program:

| Aspect | Value |
| ------ | ----- |
| Parameters | `input, target, signal, send_counts, recv_counts` — types taken from the call site |
| Directions | `In, InOut, InOut, In, InOut` |
| Body | `return target` — one `ReturnStmt`, never compiled |
| Attrs | `builtin_template_dir`, `builtin_template_vars` |

One function is synthesized per variant and shared by every call site of that
variant.

### Why the body is a `ReturnStmt` and not an empty header

The kernel's implementation is the hand-written builtin source, so the body is
never codegen'd — an empty header would do for the backend. It is a real
`ReturnStmt` returning the `target` parameter because the passes that still read
the function need it: `ReturnParamsExplicit` holds, and
`MaterializeDistTensorCtx` can resolve the returned `DistributedTensor` back to
the parameter it writes. Returning `target` also matches the public op's
window-as-result contract, which keeps the call site a plain rebind.

## Kernel source: one implementation, two rails

The synthesized function does not name a `.cpp` path the way an
[external kernel](../language/04-external-kernels.md) does. It names the builtin
*template package* — the same one `builtin.tensor.all_to_all_v` declares via
`set_template_dir` — plus the substitutions to render it with:

```text
builtin_template_dir  = ":pypto.runtime.builtins.collectives.all_to_all_v"
builtin_template_vars = "dtype_cpp=float"
```

The PTO backend renders `templates/kernel.cpp.in` into
`kernels/aiv/<name>.cpp` of the chip sub-build and lists it in the generated
`kernel_config.py` — the same path a PyPTO-generated kernel takes, except the
text comes from the template instead of from ptoas.

`dtype_cpp` is the *only* substitution either rail makes, and both give it the
same value, so the two rendered kernels are **byte-identical**. The ST asserts
this end to end by diffing the two rails' rendered sources.

## ABI

Both rails reach the kernel with the same argument layout:

| Slot | HOST rail | CHIP rail (this pass) |
| ---- | --------- | --------------------- |
| `args[0..4]` | `input, target, signal, send_counts, recv_counts` | same |
| `args[5]` | `CommContext*` | `CommContext*` |
| `args[6..]` | — | unread duplicates of `args[5]` |

Neither rail passes a rank-count scalar. The kernel reads
`CommContext::rankNum`, which is the same number the HOST dispatch used to pass
as `domain_size`: `comm_derive_context` builds a context **per comm domain**, so
its `rankNum` is that domain's rank count. Dropping the scalar costs one GM load
at kernel entry and buys a single shared source. It is also the only option on
the CHIP rail, which cannot compute a rank count at all —
`pld.system.nranks` has an InCore codegen but no orchestration codegen.

The `args[6..]` duplicates are an artifact of `MaterializeDistTensorCtx`
appending **one `CommCtx` parameter per `DistributedTensor` parameter**; all of
them resolve to the same `device_ctx`, since every operand of one collective
belongs to one comm domain. A plain `pl.Tensor` `input` is allowed and shortens
that tail by one — the first ctx still lands at `args[5]`, because the tail
always follows all five tensor parameters.

## Constraints and diagnostics

| Condition | Diagnostic |
| --------- | ---------- |
| `core_num != 1` | rejected — the multi-AIV launch is not implemented yet |
| `dtype != FP32` | rejected — the same single-dtype support the HOST rail declares |
| collective left in a non-HOST orchestration body | rejected by the pass's own postcondition check |

The residual check runs over every orchestration body except a HOST
orchestrator (which defers to its own rail, five passes later). InCore bodies
are not checked: the composite rail
([`LowerCompositeOps`](12-lower_composite_ops.md)) owns those and already ran 26
passes earlier, so re-reporting them here would blame the wrong pass.

## What this pass does *not* do

- It does not create a `CommDomain` or allocate any collective staging buffer.
  L3 still creates the domain, exchanges window addresses and binds the
  windows; L2 consumes local views and an already-built context.
- It does not fan out per device. The `device=` dispatch stays in the HOST
  orchestrator, one `chip_pipeline` per rank.
- It does not emit a nested L2 -> L2 dispatch. The collective is an AIV task of
  the caller's pipeline, not another chip callable.

## Current limitations

- **`core_num > 1`.** The requested block limit is carried through the op but
  only `1` is accepted here. The `L -> B` mapping, atomic gang admission and
  per-lane synchronization protocol are separate work.
- **Runtime operand validation.** Statically provable violations of the buffer
  contract (a strided view, `input` aliasing `target`) are rejected by the
  `pld.tensor.all_to_all_v` type deducer. There is no runtime re-check before
  the AIV task is submitted; with `B` fixed at 1 the checks that would need one
  (signal stride `>= B`) are vacuous.
- **Rank count.** The kernel reads `CommContext::rankNum`, so an explicit device
  subset smaller than the context's rank count would be handled differently from
  the HOST rail, which passes the comm domain's `domain_size`.
- **One communication domain is an unchecked precondition.** The kernel resolves
  every peer address through a single `CommContext` (`args[5]`), so operands
  bound to different domains would address the wrong remote windows. The HOST
  rail enforces the equivalent through `FindScopeForBuffers`, which sees the
  window buffers directly. This rail cannot: a comm domain has no IR
  representation until `MaterializeCommDomainScopes` (pass 43) and
  `MaterializeDistTensorCtx` (pass 45), both of which run *after* this pass, and
  by then the collective's operands are the enclosing pipeline's parameters —
  relating them back to the host windows that bind them needs interprocedural
  tracing that does not exist today. Comparing the appended `CommCtx` arguments
  instead does not work either: one is minted per `DistributedTensor` parameter,
  so a single-domain call already carries several distinct SSA values.

## Tests

- `tests/ut/ir/transforms/test_lower_l2_tensor_collectives.py` — lowered shape,
  synthesized signature and directions, template attrs, variant sharing,
  InCore pass-through, `core_num > 1` rejection.
- `tests/ut/ir/transforms/test_lower_composite_ops.py` — the composite rail
  defers a CHIP-orchestration collective to this pass and rejects
  `core_num != 1` in an InCore body.
- `tests/st/distributed/collectives/test_l2_tensor_all_to_all_v.py` — hardware
  correctness for P=2/4; the 0 / 1 / capacity / over-capacity / negative count
  matrix the InCore and HOST rails also run, which holds all three rails to one
  wire golden; and the structural assertions that no builtin chip dispatch is
  emitted, that the builtin kernel is rendered into the pipeline's own
  sub-build, and that the HOST rail renders a byte-identical kernel source.
