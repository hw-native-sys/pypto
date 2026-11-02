# LowerL2TensorCollectives Pass

## Overview

`LowerL2TensorCollectives` is the CHIP/L2 rail for managed collectives. It
rewrites a `pld.tensor.*` collective written in a **CHIP orchestration body**
into a call to a compiler-synthesized AIV kernel, so the collective becomes one
ordinary task inside the caller's own pipeline.

Today it handles `pld.tensor.all_to_all_v` with a compile-time `core_num`.

The HOST rail ([`LowerHostTensorCollectives`](48-lower_host_tensor_collectives.md))
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
... -> FuseCreateAssembleToSlice -> LowerL2TensorCollectives -> LegalizeSpmdLaunches -> DeriveCallDirections -> AutoDeriveTaskDependencies -> ...
```

The position is load-bearing. The emitted call must reach
[`DeriveCallDirections`](43-derive_call_directions.md) and
[`AutoDeriveTaskDependencies`](44-auto_derive_task_dependencies.md) like any
other kernel call: those two passes are what turn the synthesized kernel's
parameter directions into the TensorMap edges that order
`compute -> collective -> consume`. Running the rewrite after them would leave
the collective task unordered.

It also runs before
[`MaterializeDistTensorCtx`](49-materialize_dist_tensor_ctx.md), which appends
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
# INT8 (RFC #2521 A1 canonical payload) synthesizes __builtin_all_to_all_v__int8
# with builtin_template_vars = "dtype_cpp=int8_t"
```

where `__builtin_all_to_all_v__{fp32,int8}` is a synthesized `FunctionType.AIV`
function added to the program:

| Aspect | Value |
| ------ | ----- |
| Parameters | `input, target, signal, send_counts, recv_counts` — canonical types, **not** the call site's: `input` / `send_counts` are declared plain `Tensor`, the other three `DistributedTensor` |
| Directions | `In, InOut, InOut, In, InOut` |
| Body | `return target` — one `ReturnStmt`, never compiled |
| Attrs | `builtin_template_dir`, `builtin_template_vars`, `require_contiguous_operands` |

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
| `args[6..7]` | — | unread duplicates of `args[5]` |

Neither rail passes a rank-count scalar. The kernel reads
`CommContext::rankNum`, which is the same number the HOST dispatch used to pass
as `domain_size`: `comm_derive_context` builds a context **per comm domain**, so
its `rankNum` is that domain's rank count. Dropping the scalar costs one GM load
at kernel entry and buys a single shared source. It is also the only option on
the CHIP rail, which cannot compute a rank count at all —
`pld.system.nranks` has an InCore codegen but no orchestration codegen.

The `args[6..7]` duplicates are an artifact of `MaterializeDistTensorCtx`
appending **one `CommCtx` parameter per `DistributedTensor` parameter**. The
synthesized signature declares exactly three of those — `target`, `signal`,
`recv_counts` — because `input` and `send_counts` are canonically plain
`Tensor` whichever kind the call site passes. The tail is therefore always
three slots wide, and the first ctx always lands at `args[5]` because the tail
follows all five tensor parameters. All three resolve to the same `device_ctx`,
since every operand of one collective belongs to one comm domain.

## Launch width

`core_num` is the requested block limit `L`. It must be a compile-time constant on this rail, so the pass folds
the admitted block count `B = CalAllToAllVBlocks(P, L)` itself: `L` while `L < P`, otherwise the largest
multiple of `P` not exceeding `L`. `P` is the static `NR` of the `[NR, S]` signal.

- `B == 1` leaves the kernel call untouched.
- `B > 1` attaches `core_num=B` and `sync_start=True` to the kernel call. The block-aware barrier needs every
  admitted block resident together, and the runtime rejects an over-capacity sync-start launch instead of
  part-starting it.

The launch spec rides the call's attrs, which are not kernel parameters, so the five-operand ABI above is
unchanged. `B` — never `L` — is what is validated: `signal` must satisfy `S >= B`, so `L=10, P=8` (which
launches 8 blocks) accepts `S=8`. The HOST rail computes the same `B` in its generated entry; both sites are
pinned to RFC #2521's worked-example table. `NR` must equal the runtime comm-domain size, because the kernel
reads the rank count from the `CommContext`.

## Constraints and diagnostics

| Condition | Diagnostic |
| --------- | ---------- |
| `core_num` not a compile-time constant | rejected — a runtime `core_num` is not supported on this rail yet |
| `core_num` outside `[1, INT32_MAX]` | rejected — the same bound the HOST rail's entry enforces at runtime |
| `signal` stride `S < B` | rejected — `B` is the folded block count (see *Launch width*), not the raw `core_num` |
| `dtype != FP32 && dtype != INT8` | rejected — FP32 and INT8 are supported (same allowlist as the HOST rail) |
| collective left in a non-HOST orchestration body | rejected by the pass's own postcondition check |

The residual check runs over every orchestration body except a HOST
orchestrator (which defers to its own rail, five passes later). InCore bodies
are not checked: the composite rail
([`LowerCompositeOps`](14-lower_composite_ops.md)) owns those and already ran 26
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

- **A runtime `core_num`.** Only a compile-time `core_num` is accepted. A runtime value would need `B`
  computed in the emitted orchestration and a runtime `S >= B` check, and orchestration has no surface for
  raising that fatal today. The HOST rail accepts a runtime value because its entry runs on the host.
- **Operand validation is static apart from contiguity, and aliasing is only partly covered.** The
  `pld.tensor.all_to_all_v` type deducer rejects what the operand types alone
  prove: a non-ND layout, a stride vector that is not the packed one, a
  `valid_shape` narrower than the shape, and `input` being the *same
  expression* as `target`. It does **not** reject two distinct `pld.window()`
  views of one allocation — deduction runs when the Call is built, and
  `DistributedTensorType::window_buffer_` is not bound until
  [`MaterializeCommDomainScopes`](47-materialize_comm_domain_scopes.md)
  (pass 46). Whole-allocation distinctness is a **HOST-rail** guarantee:
  `LowerHostTensorCollectives` resolves each operand back to its `WindowBuffer`
  within the same `host_orch` body and runs `CheckPairwiseDistinctWindows` over
  all five. This rail — like the InCore composite rail — sees the operands as
  enclosing-pipeline parameters and has no such provenance, so pairwise-distinct
  windows are the caller's obligation here. **Contiguity is decided at compile
  time here, with a runtime backstop.** The type deducer cannot see a slice's
  strides (a sliced type carries none), and a runtime verdict would be per rank:
  suppressing one rank's submit while its peers launch leaves them in the
  kernel's `ExchangeBarrier`, a hang rather than a rejection. So this pass
  traces every operand of the call as written through every value that can
  reach it (AssignStmt, loop-carried and `if`/`for`/`while` results — a loop
  result also traces its initializer, which a zero-trip loop returns) as an
  **allow-list**: params, windows, `tensor.create` and an Out a `pl.submit`
  leaves to the runtime to allocate are row-major roots; `reshape`, `view`,
  `set_validshape`, `assemble` (its target), a collective's result (its target),
  a kernel's or `pl.submit`'s result or tuple element (the argument it returns)
  and a slice that narrows only the leading dimension of a statically shaped
  tensor are followed to their source; anything else, including
  `tensor.transpose` and any other slice, is rejected, as is any value on the
  chain with a non-ND layout or an explicit stride other than the packed one
  (the steps assume a packed source and a slice drops its parent's stride: a
  leading-dimension slice of a DN or padded tensor is not contiguous, and an ND
  `view` of a DN tensor lowers to a runtime transpose). Some
  safe programs are rejected too — a `tensor.write` result, a `[1, N]` slice of
  a `[1, W]` tensor, a slice whose offsets are not a literal tuple, a result its
  callee's ReturnStmt does not name as a parameter — since a missed acceptance
  is safe and a wrong one is not. The synthesized kernel also carries
  `require_contiguous_operands`, and orchestration codegen emits one
  `is_contiguous` check per tensor operand ahead of the submit, latching
  `SIMPLER_ERROR_INVALID_ARGS` on the first violation, as a backstop for
  caller-supplied operands: argument packing rejects a non-contiguous host torch
  tensor, but a prebuilt runtime `Tensor` is checked only there. The other check
  that would need a runtime form (signal stride `>= B`) is still enforced
  statically at lowering, where both `S` and the folded `B` are compile-time
  values.
- **Rank count.** The kernel reads `CommContext::rankNum`, so an explicit device
  subset smaller than the context's rank count would be handled differently from
  the HOST rail, which passes the comm domain's `domain_size`.
- **One communication domain is an unchecked precondition.** The kernel resolves
  every peer address through a single `CommContext` (`args[5]`), so operands
  bound to different domains would address the wrong remote windows. The HOST
  rail enforces the equivalent through `FindScopeForBuffers`, which sees the
  window buffers directly. This rail cannot: a comm domain has no IR
  representation until `MaterializeCommDomainScopes` (pass 46) and
  `MaterializeDistTensorCtx` (pass 48), both of which run *after* this pass, and
  by then the collective's operands are the enclosing pipeline's parameters —
  relating them back to the host windows that bind them needs interprocedural
  tracing that does not exist today. Comparing the appended `CommCtx` arguments
  instead does not work either: one is minted per `DistributedTensor` parameter,
  so a single-domain call already carries several distinct SSA values.

## Tests

- `tests/ut/ir/transforms/test_lower_l2_tensor_collectives.py` — lowered shape,
  synthesized signature and directions, template attrs, variant sharing,
  InCore pass-through, launch-spec attachment and block-count fold, capacity check, runtime `core_num`
  rejection, INT8 variant.
- `tests/ut/codegen/distributed/test_builtin_collective_kernel_source.py` —
  HOST and CHIP rails render a byte-identical kernel for FP32 and INT8.
- `tests/ut/ir/transforms/test_lower_composite_ops.py` — the composite rail
  defers a CHIP-orchestration collective to this pass and rejects
  `core_num != 1` in an InCore body.
- `tests/ut/ir/transforms/test_lower_host_tensor_collectives.py` — pins the
  window-aliasing rejection as a **HOST-rail** guarantee
  (`CheckPairwiseDistinctWindows`), which is the contrast the *Current
  limitations* note above is defined against.
- `tests/st/distributed/collectives/test_l2_tensor_all_to_all_v.py` — hardware
  correctness for P=2/4; the 0 / 1 / capacity / over-capacity / negative count
  matrix the InCore and HOST rails also run, which holds all three rails to one
  wire golden; and the structural assertions that no builtin chip dispatch is
  emitted, that the builtin kernel is rendered into the pipeline's own
  sub-build, and that the HOST rail renders a byte-identical kernel source.

## Indexed A5 URMA variant

`pld.tensor.indexed_all_to_all_v(input, target, signal, send_counts,
recv_counts, index, core_num=L)` uses the managed L2 task pipeline and its own
kernel template. Six tensor operands precede CommContext at slot 6. Input,
index and recv_counts may be local tensors; send_counts must be a window.
All six operands, `index` included, get the same contiguity check and backstop
as the plain op (see [Current limitations](#current-limitations)).
`L` is a positive compile-time exact launch width. All AIVs share local copies
between remote submission and CQ waits; AIVs beyond P only help local copies.
Only A5 URMA PER_PEER transport is supported, with one writer per destination.
Callers must order exchanges sharing queues; reusing the same InOut signal
creates that dependency even when payload windows differ. HOST and InCore
calls are unsupported. See the [user contract](../../user/distributed/20-all_to_all.md#indexed-alltoallv-on-a5-urma).
