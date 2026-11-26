# Integrating Hand-Written C++ Kernels

PyPTO can call an **existing hand-written C++ InCore kernel** from a
PyPTO-authored orchestration, without going through PyPTO's tile codegen. You
declare a *signature-only function header* in the DSL; the orchestration calls
it like any InCore kernel, but the compiler skips codegen for its body and
compiles the referenced `.cpp` as the kernel instead.

Use this when you already have a tuned AICore kernel (e.g. a bespoke attention
kernel) and want to drive it from a PyPTO orchestration, reusing PyPTO's
task scheduling, dependency analysis, and runtime dispatch.

## Contract

An external kernel is a normal InCore kernel from the runtime's point of view,
so the hand-written source must meet the same ABI PyPTO-generated kernels do:

- Export a single `extern "C" void kernel_entry(__gm__ int64_t* args)` entry
  (one entry `.cpp` per kernel; the runtime dispatches by `func_id`, the symbol
  is fixed). This is the same entry PyPTO's own generated kernels export.
- The declared **parameter order and directions** (`pl.Out` / `pl.InOut`) must
  match how the kernel reads its arguments — the orchestration builds the task
  payload (`add_input` / `add_inout` / `add_output`) from the declaration.

The declaration carries only the signature; its body is a bare `...`.

### Multi-file kernels

`external_source` names the single entry `.cpp` (the one exporting
`kernel_entry`). That file may `#include` any number of sibling files —
PyPTO **references it at its original path** (it does not copy it), so
relative includes resolve against the original tree. A kernel laid out as

```text
my_kernel/
  aic/entry.cpp          # external_source; #include "../kernel/impl.cce"
  kernel/impl.cce        #                  #include "../tiling/params.h"
  tiling/params.h
```

works unchanged — point `external_source` at `aic/entry.cpp` and the whole
`../kernel/` / `../tiling/` include chain is picked up at compile time.
Headers on the runtime include path (e.g. `tensor.h`, `intrinsic.h`,
`pto/pto-inst.hpp`) resolve as usual.

## `@pl.program` route

Declare the kernel with `@pl.function(type=AIC/AIV, external_source=...)` and an
empty body. A mixed **AIC + AIV** kernel (one `MixedKernels` submit) is a
`pl.FunctionType.Group` of one AIC member and one AIV member:

```python
from pathlib import Path
import pypto.language as pl

KDIR = Path(__file__).parent / "kernels"

@pl.program
class PagedAttention:
    @pl.function(type=pl.FunctionType.AIC, external_source=KDIR / "aic/pa.cpp")
    def PA_AIC(self, query: pl.Tensor[[B, H, D], pl.FP16], ...,
               out: pl.Out[pl.Tensor[[B, H, D], pl.FP16]], ...
               ) -> pl.Tensor[[B, H, D], pl.FP16]:
        ...                                   # body lives in aic/pa.cpp

    @pl.function(type=pl.FunctionType.AIV, external_source=KDIR / "aic/pa.cpp")
    def PA_AIV(self, ...same signature...) -> ...:
        ...

    @pl.function(type=pl.FunctionType.Group)
    def PA(self, ...same signature...) -> ...:
        r = self.PA_AIC(...)                  # defines the group members
        self.PA_AIV(...)
        return r

    @pl.function(type=pl.FunctionType.Orchestration)
    def entry(self, query, ..., out):
        # build tiling / workspace tensors here (host-side), then dispatch:
        out = self.PA(query, ..., out)        # -> MixedKernels submit
        return out
```

`external_source` accepts an absolute path or one relative to the file that
defines the program. The AIC and AIV members may point at the **same** source
(compiled once per core) or different files.

For a single-core kernel, declare just one `AIC` or `AIV` function and call it
directly from the orchestration (no group).

## `@pl.jit.extern` route

Under `@pl.jit`, declare the kernel with `@pl.jit.extern`. A `core_type="mixed"`
kernel auto-expands to the AIC + AIV + Group form above:

```python
@pl.jit.extern(core_type="mixed",
               aic_source="kernels/aic/pa.cpp",
               aiv_source="kernels/aiv/pa.cpp",
               include_dirs=["kernels/include", "third_party/attention/include"],
               dual_aiv_dispatch=True)
def pa(query: pl.Tensor[[B, H, D], pl.FP16], ...,
       out: pl.Out[pl.Tensor[[B, H, D], pl.FP16]], ...
       ) -> pl.Tensor[[B, H, D], pl.FP16]: ...

@pl.jit
def decode(query: pl.Tensor, ..., out: pl.Out[pl.Tensor]):
    out = pa(query, ..., out)                 # dep discovered automatically
    return out
```

Single-core form: `@pl.jit.extern(core_type="aic"|"aiv", source="k.cpp")`.

For multiple outputs, declare a tuple with the complete shape and dtype of each
result. The JIT preserves this return annotation for AIC, AIV, and mixed
externs, resolving tensor aliases and folding module, closure, or call-bound
`pl.constexpr` values. Evaluated annotations also retain factory locals that
appear only in the signature; their values participate in the memory and disk
cache keys. Keep the stub body as `...`; no Python `return` statement is needed:

```python
@pl.jit.extern(core_type="aiv", source="kernels/pair.cpp")
def pair(a: pl.Tensor, y: pl.Out[pl.Tensor], z: pl.Out[pl.Tensor]) -> tuple[
    pl.Tensor[[16, 16], pl.INT8], pl.Tensor[[16, 1], pl.FP32]
]:
    ...

@pl.jit
def entry(a: pl.Tensor, y: pl.Out[pl.Tensor], z: pl.Out[pl.Tensor]):
    y, z = pair(a, y, z)
    return y, z
```

The unpacked outputs can also feed later JIT kernels. A bare
`tuple[pl.Tensor, pl.Tensor]` does not supply complete return types for a
signature-only declaration. Existing single-output stubs can still infer their
return type from the first `pl.Out` parameter when no complete return annotation
is supplied.

For a signature-only stub, complete explicit return annotations must match **all**
`pl.Out` and `pl.InOut` parameters in declaration order, including each output's
specialized shape and dtype. Types alone cannot identify a reordered or partial
output list. A stub with a single return of parameter names instead validates
the annotation against those parameters in return order, including readonly
aliases (see below). Incompatible declarations fail during specialization.
Use Python `tuple[...]` (or `typing.Tuple[...]`) for multiple results.
A singleton declaration such as `tuple[pl.Tensor[[16, 32], pl.FP32]]` returns
one flat Tensor value, which can be passed directly to another kernel.
`pl.Tuple[...]` denotes a single tuple-valued result and is rejected for externs,
because it cannot alias the individual tensor output buffers.
An explicit `tuple[()]` (or `typing.Tuple[()]`) declares zero results and is
valid in a signature-only stub only without `pl.Out`/`pl.InOut` parameters.
An explicit `return ()` can declare zero aliases even with output parameters.
`pl.Tuple[()]` and legacy
`pl.Tuple([...])` still denote a single tuple-valued result and are rejected.
This validation does not apply to legacy inference from missing or bare return
annotations. Kernels that write several buffers and are called only for their
side effects can continue to omit the return annotation.

Use `include_dirs=[...]` for headers that are not reachable through quoted
includes relative to the entry source. Like `source`, each directory may be
absolute or relative to the Python file defining the `@pl.jit.extern` stub.
The resolved directories are passed to CCEC in order. Included header contents
and the resolved include-path metadata participate in the JIT cache key.

By default, a mixed external kernel dispatches one AIC lane and one AIV lane.
Set `dual_aiv_dispatch=True` when the hand-written kernel requires the AIC lane
and **both** AIV sub-lanes. The AIV entry is then launched twice for every
logical block and must use its sub-block id to partition work; an entry that
does not distinguish the lanes can race or duplicate writes. This option is
valid only with `core_type="mixed"`. On A2/A3 external wrappers, read the
runtime-provided lane id with `get_sub_block_id(args)` from `intrinsic.h`;
the standalone CCE lane accessor is not wired to PyPTO's logical task payload
and may report lane 0 for both dispatches.

For a direct `@pl.program` declaration, put the same launch contract on the
AIV member:

```python
@pl.function(type=pl.FunctionType.AIV, external_source="kernels/aiv/pa.cpp",
             attrs={"dual_aiv_dispatch": True})
def pa_aiv(self, query: pl.Tensor, out: pl.Out[pl.Tensor]): ...
```

Paths resolve relative to the file defining the kernel. Editing the referenced
`.cpp` changes the JIT cache key, so a kernel change triggers recompilation even
though the Python stub is unchanged.

## Returned tensor identities

An extern result aliases a tensor parameter supplied to the native kernel.
Declare a subset or reordered results with a single return of parameter names:

```python
@pl.jit.extern(core_type="aiv", source="kernels/producer.cpp")
def producer(
    first: pl.Out[pl.Tensor],
    second: pl.Out[pl.Tensor],
    third: pl.Out[pl.Tensor],
) -> tuple[pl.Tensor, pl.Tensor]:
    return second, third
```

This return declares aliases; Python does not execute it as kernel computation.
The JIT preserves the names in the `external_return_params` function attribute,
so dropping an unused result or outlining a SPMD wrapper cannot shift later
consumers onto a different output buffer. Repeated parameter returns are also
preserved. Caller-side shape, dtype, and layout propagation follows the same
aliases when specializing downstream kernels, including readonly returns from
externs with writable scratch parameters. Return expressions that compute a
value, return a local variable, or select a result through control flow are rejected.

In raw `@pl.function` IR, declare the same mapping with a prologue such as
`pl.func_attr({"external_return_params": [second, third]})`, followed by `...`.
Each reference must name a parameter of that function and match the corresponding
return's tensor dtype and logical shape. References survive SSA renaming,
memory initialization, printing and serialization.

Without this attribute, a raw extern declaration returns **all** `Out`/`InOut`
parameters in declaration order, with matching result count and tensor types.
An ambiguous signature is rejected instead of binding results by a heuristic.
The JIT's existing first-`Out` default for a signature-only single-result stub
is recorded explicitly; use an explicit return to declare another result.

## What the compiler does

- **Passes**: the declaration has no executable tile body. `ReturnParamsExplicit`
  validates its returned-parameter contract, and `NormalizeReturnOrder` uses it
  to make Group/SPMD wrapper returns explicit parameter references.
- **Orchestration codegen**: assigns the kernel a `func_id` and emits the submit
  exactly as for a DSL kernel — a single AIC/AIV `rt_submit_*_task`, or a
  `MixedKernels{aic_id, aiv_id, ...}` + `rt_submit_task` for a group.
- **Backend**: skips ptoas for the external kernel and lists it in the
  generated `kernel_config.py` manifest like a generated kernel (`func_id`,
  `core_type`, `signature`), but with `source` pointing at the original
  hand-written `.cpp` (referenced in place, so its sibling files stay
  reachable) rather than a copy under the artifact dir.

## Restrictions

- `external_source` is only valid on `FunctionType.AIC` / `FunctionType.AIV`.
- A raw `@pl.function` body must be `...` (signature only), optionally preceded
  by `pl.func_attr`. A `@pl.jit.extern` may additionally declare the single
  parameter-return statement described above.
- A `Group` must be all-external or all-DSL — mixing external and DSL members in
  one group is rejected.

See [Internal external-kernel contract](07-external-kernel-contract.md) for the provider/bridge interface baseline.
