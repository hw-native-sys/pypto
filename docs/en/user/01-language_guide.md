# Compiling a Program

Turning a program or a `@pl.jit` kernel into device kernels and host orchestration.

> **The language material that used to live on this page has moved** to the
> [Language Guide](language/index.md) chapter — types, functions, control flow, memory,
> scopes and tasks, and directives each have their own page there, and the operator
> surface is in [Operations](ops/index.md). What remains here is compilation, pending its
> own chapter.

## `ir.compile()`

```python
from pypto import ir
from pypto.backend import BackendType

output_dir = ir.compile(
    program,
    output_dir=None,                           # auto-generated if None
    strategy=ir.OptimizationStrategy.Default,  # the only optimization strategy
    dump_passes=True,                          # dump IR snapshots under output_dir/passes_dump/
    backend_type=BackendType.Ascend910B,
)
```

| Parameter | Options | Description |
| --------- | ------- | ----------- |
| `program` | `ir.Program` | Required program object (from `@pl.program` or equivalent) |
| `strategy` | `OptimizationStrategy.Default` | `Default` = full tensor-oriented pipeline (the only strategy) |
| `backend_type` | `BackendType.Ascend910B`, `BackendType.Ascend950` | Target hardware for passes and codegen (import `BackendType` from `pypto.backend`) |
| `dump_passes` | `PassDumpLevel` | Write IR snapshots under `<output_dir>/passes_dump/` after each pass |
| `skip_ptoas` | `True` / `False` | Skip the ptoas step; emit raw `.pto` (MLIR) instead of compiled C++ wrappers (default `False`) |
| `output_dir` | path or `None` | If `None`, uses `<base>/<program_name>_<timestamp>`, where `<base>` is `PYPTO_PROG_BUILD_DIR` or `build_output`; created as needed |
| `verification_level` | `None`, `ir.VerificationLevel.NONE`, `BASIC` | `None` = use the default (`BASIC`, or `PYPTO_VERIFY_LEVEL`) |

`ir.compile()` takes more parameters than the ones above — diagnostics, memory planner
selection, profiling, platform, and distributed configuration among them. The full table
belongs to the execution chapter, which is not written yet; until then read the signature
in `python/pypto/ir/compile.py`.

## `JITFunction.compile()`

`@pl.jit` kernels normally fuse specialize + compile + dispatch into one `kernel(*args)`
call. `compile(*sample_args)` stops after compilation and returns the `CompiledProgram`:

```python
@pl.jit
def my_kernel(x, w, out): ...

compiled = my_kernel.compile(sample_x, sample_w, sample_out)
print("artifacts in:", compiled.output_dir)

from pypto.runtime import ChipWorker, RunConfig

worker = ChipWorker(config=RunConfig(platform="a2a3sim"))
w_dev = worker.alloc_tensor(sample_w.shape, sample_w.dtype, init=sample_w)
handle = worker.register(compiled)
for batch in stream:
    handle(batch.x, w_dev, batch.out)
worker.free_tensor(w_dev)
worker.close()
```

- `compile()` honours `config=RunConfig(...)` the same way `__call__` does: compile-side
  knobs (`strategy`, `dump_passes`, diagnostics, …) are forwarded to `ir.compile()`.
  Runtime-side fields (`device_id`, DFX flags) affect dispatch, not the artifact.
- The returned `CompiledProgram` is the object the JIT cache holds, so a later call with
  the same specialization key returns the identical instance.
- It exposes the full extraction surface — `chip_callable`, `runtime_name`,
  `runtime_config`, `build_orch_args`, `build_call_config`, `output_dir`, `platform`,
  `output_indices` — so a harness driving the runtime directly can do so without writing a
  `@pl.program` wrapper.

**`compile()`'s positional arguments are the kernel's own arguments, not compile
options.** `compile(skip_ptoas=True)` is treated as a kernel argument and silently
ignored; compile options go through `config=RunConfig(...)`.

## Inspecting the result

`node.as_python()` prints the IR of a function or program; `concise=True` omits
intermediate type annotations. A `JITFunction` has no `as_python()` of its own — the IR
exists once a specialization does, so print `compiled.program.as_python()`.

`dump_passes=` writes a snapshot after every pass under `passes_dump/`, which is how you
find the pass that changed something you did not expect.

## See Also

- [Language Guide](language/index.md) — the language surface this page used to cover.
- [Operations](ops/index.md) — the operator catalog.
- [Functions and Programs](language/01-functions.md) — the decorators being compiled.
- [Pass Manager](../dev/passes/00-pass_manager.md) — the pipeline `strategy` selects.
- [Torch Codegen Debug Guide](03-torch_codegen_debug.md) — comparing the result against PyTorch.
