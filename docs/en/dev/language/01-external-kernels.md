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
  (one kernel per `.cpp`; the runtime dispatches by `func_id`, the symbol is
  fixed). This is the same entry PyPTO's own generated kernels export.
- The declared **parameter order and directions** (`pl.Out` / `pl.InOut`) must
  match how the kernel reads its arguments — the orchestration builds the task
  payload (`add_input` / `add_inout` / `add_output`) from the declaration.

The declaration carries only the signature; its body is a bare `...`.

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
               aiv_source="kernels/aic/pa.cpp")
def pa(query: pl.Tensor[[B, H, D], pl.FP16], ...,
       out: pl.Out[pl.Tensor[[B, H, D], pl.FP16]], ...
       ) -> pl.Tensor[[B, H, D], pl.FP16]: ...

@pl.jit
def decode(query: pl.Tensor, ..., out: pl.Out[pl.Tensor]):
    out = pa(query, ..., out)                 # dep discovered automatically
    return out
```

Single-core form: `@pl.jit.extern(core_type="aic"|"aiv", source="k.cpp")`.

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
- **Backend**: skips ptoas for the external kernel and copies its `.cpp` to
  `kernels/<aic|aiv>/<name>.cpp`, so the generated `kernel_config.py` manifest
  lists it exactly like a generated kernel (`func_id`, `source`, `core_type`,
  `signature`, `arg_index`).

## Restrictions

- `external_source` is only valid on `FunctionType.AIC` / `FunctionType.AIV`.
- The body must be a bare `...` (signature only).
- A `Group` must be all-external or all-DSL — mixing external and DSL members in
  one group is rejected.
