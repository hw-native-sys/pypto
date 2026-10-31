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

## What the compiler does

- **Passes**: the header-only function survives the pass pipeline unchanged
  (its empty body is a no-op for tile passes); it is exempt only from the
  `ReturnParamsExplicit` property, which requires a `ReturnStmt` a header cannot
  have.
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
- The body must be a bare `...` (signature only).
- A `Group` must be all-external or all-DSL — mixing external and DSL members in
  one group is rejected.

## Calling a Python DSL kernel directly

A Python DSL can implement the lazy `__pypto_jit__()` protocol so callers keep
only their operator expressions. With the companion CANNBotDSL device-export
integration installed, the A5 example is:

```python
import os

import cannbotdsl as cb
import pypto.language as pl
import torch
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig


@cb.kernel
def add_one(x: cb.Tensor, out: cb.Tensor):
    for i in cb.range(x.shape[1]):
        out[0, i] = x[0, i] + 1.0


@pl.jit
def program(x: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    add_one(x, out)
    return out


def main():
    # Torch is used only for CPU buffers and result validation.
    x = torch.arange(32, dtype=torch.float32, device="cpu").reshape(1, 32)
    out = torch.empty_like(x)
    config = RunConfig(
        backend_type=BackendType.Ascend950,
        execution_mode=ExecutionMode.ONBOARD,
        device_id=int(os.environ.get("TASK_DEVICE", "0")),
    )

    program(x, out, config=config)

    torch.testing.assert_close(out, x + 1.0, rtol=0, atol=0)
    print("PASS: program -> CANNBotDSL add_one")
    print(out)


if __name__ == "__main__":
    main()
```

This is a complete **program mode** example: the call to
`program(x, out, config=config)` compiles and executes on A5, and the assertion
checks the copied-back CPU output. Torch is used only for CPU buffers and result
validation. No user-written `TensorSpec`, export call, ABI wrapper or `jit.extern`
declaration is required. Install both companion changes and activate the A5
environment before running the script. It defaults to device 0 and uses
`TASK_DEVICE` when launched through the device queue. This does not add Torch
kernel mode or allow a device kernel to submit another kernel from an InCore scope.

The producer hook returns a stable `JITFunction(func, func_type="extern",
external_builder=...)`. The builder receives a request-local `SpecializeContext`
with inferred tensor/scalar metadata and returns an `ExternalKernelSpec` from
`pypto.jit.foreign`. It supplies the source path, include directories, core type,
tensor access directions and optional backend/runtime requirements. It must
publish complete source files that remain available for compilation and caching.
PyPTO has no mandatory dependency on the producer's package.

A bare call is rewritten internally to capture each written tensor, preserving
SSA and loop-carried dependencies. Multiple outputs and keyword arguments work;
output arguments must be distinct tensor variables. The Python callable remains
void: using its call as a value is rejected. All call sites of one producer in
one program must have the same inferred tensor signature. Foreign calls are
supported from chip orchestration and inline orchestration helpers.

Builders run before cache lookup; source content, local includes and access/core
metadata participate in both cache identities. Generated source is reused within
the request. Producers may use content-addressed storage; no captured tensors,
addresses or mutable per-request source paths are stored on the JIT dependency.
The CANNBotDSL integration currently supports A5/TMR, ND tensors with a contiguous
innermost axis, explicit runtime scalars, AIV/AIC/MIX and basic dynamic shapes.
The CANN 9.3.0 SDK has a missing CPU-debug-only include which can make the optional
persistent cache bypass conservatively; the default in-process cache still works.
