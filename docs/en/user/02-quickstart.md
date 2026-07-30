# Quickstart

Write, compile, and inspect your first PyPTO kernels with `@pl.jit`.

> **Prerequisites:** PyPTO installed and importable — see [Installation](01-installation.md).
> The compile-and-inspect examples need only the install. The one example that *runs* a
> kernel on hardware is marked as such and needs the runtime plus a device or simulator.

## Concept

A PyPTO kernel is Python source that is *parsed*, not executed. `@pl.jit` reads the
decorated function's body and specializes it into PyPTO IR; nothing runs until you
compile that IR and dispatch it.

`@pl.jit` marks a **chip-level entry point** — an Orchestration function, which is
control-plane code. Control-plane code cannot touch on-chip memory, so a jit body reaches
the execution plane in one of two ways:

- `with pl.at(level=pl.Level.CORE_GROUP):` — open an on-chip scope inline. This is the
  short form, and what most single-kernel examples use.
- Call a `@pl.jit.incore` sub-function — the compiler discovers it from the entry's body
  and outlines it into its own device kernel.

Putting `pl.load` directly in a jit body, with neither of those, fails at compile time.
That is not a quirk: it is the control-plane / execution-plane split showing up as an
error message, and it is the single most important structural idea in PyPTO.

```python
import pypto.language as pl
import torch
```

## Quickstart: element-wise add

The smallest complete kernel — the same one as `examples/hello_world.py`.

```python
import pypto.language as pl
import torch

@pl.jit
def tile_add(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        tile_c = pl.add(tile_a, tile_b)
        pl.store(tile_c, [0, 0], c)
    return c

a = torch.full((128, 128), 2.0, dtype=torch.float32)
b = torch.full((128, 128), 3.0, dtype=torch.float32)
c = torch.zeros((128, 128), dtype=torch.float32)

# Runs the full pass pipeline. No device, no ptoas.
tile_add.compile_for_test(a, b, c)
print("compiles")
```

| Line | What it does |
| ---- | ------------ |
| `@pl.jit` | Specializes the body into an Orchestration entry point on first compile |
| `a: pl.Tensor` | A DDR tensor. No shape given — it is read from the torch tensor you pass |
| `c: pl.Out[pl.Tensor]` | **Direction**: this parameter is written, not read |
| `with pl.at(level=pl.Level.CORE_GROUP)` | Opens an on-chip scope; tile operations are only legal inside one |
| `pl.load(a, [0, 0], [128, 128])` | DDR → on-chip tile. `[0, 0]` is the offset, `[128, 128]` the shape |
| `pl.store(tile_c, [0, 0], c)` | On-chip tile → DDR, into the `Out` parameter |
| `return c` | Returns the written tensor |

`pl.Out[...]` is load-bearing rather than decorative: it tells the compiler this buffer is
written, which decides whether the runtime uploads it before the call and downloads it
after. Every tensor parameter carries a direction — `In` by default, or an explicit
`pl.Out[...]` / `pl.InOut[...]`.

`compile_for_test(...)` runs the whole pass pipeline and stops before code generation, so
it is the cheapest way to check that a kernel is well-formed. It needs neither ptoas nor a
device, which is why the examples on this page use it.

### Running it on hardware

> **Needs the runtime and a device or simulator platform.** Everything above this point
> does not.

```python
from pypto.runtime import RunConfig

tile_add(a, b, c, config=RunConfig())          # compiles, caches, dispatches
assert torch.allclose(c, a + b, rtol=1e-5, atol=1e-5)
```

Calling a `@pl.jit` function directly does the whole thing: specialize on the argument
shapes and dtypes, compile, cache the result, dispatch. Later calls with the same shapes
reuse the cached compilation.

## Mechanics

### Shapes: from the arguments, or from the signature

`a: pl.Tensor` leaves the shape to the call site. Annotate it fully instead, and the
contract lives in the signature — which lets you compile with no sample tensors at all:

```python
@pl.jit
def tile_add_128(
    a: pl.Tensor[[128, 128], pl.FP32],
    b: pl.Tensor[[128, 128], pl.FP32],
    c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.add(pl.load(a, [0, 0], [128, 128]),
                        pl.load(b, [0, 0], [128, 128])), [0, 0], c)
    return c

compiled = tile_add_128.compile(skip_ptoas=True)   # no torch.empty(...) needed
```

Use the bare form while exploring; use the annotated form for kernels with large
signatures, where a list of throwaway `torch.empty(...)` buffers is worse than a
signature that states the contract once.

### Loops inside the on-chip scope

`pl.range()` builds a loop in the IR. It goes **inside** `pl.at` — it is a loop the device
kernel runs, not something the host iterates:

```python
@pl.jit
def double_thrice(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        acc = pl.load(x, [0, 0], [128, 128])
        for i in pl.range(3):
            acc = pl.add(acc, acc)      # rebinding is fine — the parser renames
        pl.store(acc, [0, 0], y)
    return y
```

Rebinding `acc` looks like mutation but is not: the IR is SSA, and the parser gives each
iteration's value its own name while threading it through the loop as a carried value.
Reading `acc` after the loop reads the last iteration's result.

Loop forms:

```python
for i in pl.range(10):        # 0 .. 9
for i in pl.range(0, 100, 2): # 0, 2, 4, ... 98
```

### Splitting work across functions

For anything beyond one kernel, put the compute in a `@pl.jit.incore` sub-function and let
the `@pl.jit` entry dispatch it. The entry does not need `pl.at` — the sub-function *is*
the execution plane.

```python
@pl.jit.incore
def add_kernel(
    a: pl.Tensor[[128, 128], pl.FP32],
    b: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    ta = pl.load(a, [0, 0], [128, 128])
    tb = pl.load(b, [0, 0], [128, 128])
    pl.store(pl.add(ta, tb), [0, 0], out)
    return out

@pl.jit
def add_program(
    a: pl.Tensor[[128, 128], pl.FP32],
    b: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    return add_kernel(a, b, out)      # discovered automatically — no registration
```

```text
add_program   (@pl.jit, Orchestration)  — control plane: dispatches
  └── add_kernel (@pl.jit.incore)       — execution plane: loads, computes, stores
```

The `@pl.jit` family, one decorator per IR function kind:

| Decorator | Becomes | Use for |
| --------- | ------- | ------- |
| `@pl.jit` | Orchestration | The chip-level entry point |
| `@pl.jit.incore` | InCore | A device kernel, outlined into its own file |
| `@pl.jit.inline` | Inline | A helper spliced into every call site |
| `@pl.jit.opaque` | Opaque | A separate IR function that may wrap loops and `pl.at` scopes |
| `@pl.jit.host` | `level=HOST, role=Orchestrator` | The HOST entry of a distributed (multi-card) program |

Sub-functions are discovered from the entry's body, so you just call them by name. One
deliberate exception: a plain `@pl.jit` entry does **not** discover other `@pl.jit`
entries — only `.host` reaches across the chip boundary, which keeps two unrelated
top-level kernels from silently folding into one program.

### Compiling, and reading what came out

```python
compiled = add_program.compile(skip_ptoas=True)
print(f"Generated code in: {compiled.output_dir}")
```

`compile()` returns a **`CompiledProgram`** — not a path. `compiled.output_dir` is a
`pathlib.Path` holding:

```text
kernels/       generated device kernels, one per InCore function
orchestration/ generated host-side C++
report/        compile-time reports, including perf hints
debug/         a runnable `run.py` harness
passes_dump/   per-pass IR snapshots (only when dump_passes is on)
```

`skip_ptoas=True` stops after emitting `.pto` (MLIR). Drop it to get compiled C++ kernel
wrappers — that step invokes **ptoas**, which is distributed separately from the Python
package. `compile()` accepts the same options as `ir.compile()`, which takes 15
parameters in total; the ones you will reach for first:

| Parameter | Default | What it does |
| --------- | ------- | ------------ |
| `output_dir` | `None` → `<base>/<name>_<timestamp>` | Where output lands. `<base>` is `$PYPTO_PROG_BUILD_DIR`, or `build_output` |
| `strategy` | `OptimizationStrategy.Default` | Pass pipeline preset. `DebugTileOptimization` is a debugging shortcut — prefer `Default` |
| `dump_passes` | `True` | `bool`, or a `PassDumpLevel` (`NONE` / `CONCISE` / `EXPLICIT`) |
| `backend_type` | `BackendType.Ascend910B` | Target architecture — `Ascend910B` or `Ascend950` |
| `skip_ptoas` | `False` | Stop at `.pto` instead of invoking ptoas |

**To read the IR, go through the compiled program.** A `JITFunction` has no
`as_python()` — only `compile` and `compile_for_test` — so the IR becomes readable once
one of those has produced it:

```python
print(compiled.program.as_python())
```

What comes back is the specialized `@pl.program` class your jit functions turned into,
which is also the clearest way to see what `@pl.jit` actually does:

```python
@pl.program
class _jit_add_program:
    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def add_kernel(a: pl.Tensor[[128, 128], pl.FP32], ...):
        ta: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(a, [0, 0], [128, 128], ...)
        ...
```

Note what the parser filled in: `pl.load` resolved to `pl.tile.load`, tiles picked up
`pl.Mem.Vec`, and the sub-function was assigned a level and role. Reading this is the
fastest way to check what the compiler actually built when a kernel misbehaves.

## Edge Cases

> **Fatal pitfall:** `@pl.jit` **parses** the body — it does not run it. A `print()` or
> `assert` inside the body never executes at runtime, and stepping through it in a
> debugger shows you the parse, not the computation. Debug by reading
> `compiled.program.as_python()`.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **Compile fails with an orchestration codegen error** | Tile operations directly in a `@pl.jit` body | Wrap them in `with pl.at(level=pl.Level.CORE_GROUP):`, or move them into a `@pl.jit.incore` sub-function |
| **`missing a required argument`** | `compile()` / `compile_for_test()` called without samples on bare `pl.Tensor` parameters | Pass sample tensors, or annotate the parameters fully |
| **Output tensor comes back unchanged** | Result written to a parameter not declared `pl.Out[...]` | Add the direction, and write through `pl.store` |
| **ptoas failures on a machine without the toolchain** | Codegen ran the assembler | Pass `skip_ptoas=True`, or use `compile_for_test()` |
| **`AttributeError: as_python`** | `as_python()` called on the jit function | It lives on the IR: `compiled.program.as_python()` |

`PYPTO_PROG_BUILD_DIR` is a **runtime environment variable** —
`PYPTO_PROG_BUILD_DIR=/tmp/out python kernel.py` relocates every compile output.
Distinguish it from `SIMPLER_HOST_STRACE` and `SIMPLER_DFX`, which are **compile-time
macros** of the runtime (`-DXXX=1` at build time); setting those in the shell has no
effect.

## See Also

- [Installation](01-installation.md) — getting to the point where these examples import.
- [Programming Model](03-programming-model.md) — the abstractions behind `pl.at`, the two planes, and the memory hierarchy.
- [Language Guide](01-language_guide.md) — the full surface, including the `@pl.function` / `@pl.program` class form that `@pl.jit` specializes into.
- [Operation Reference](02-operation_reference.md) — the operator surface across `pl.*`, `pl.tensor.*`, and `pl.tile.*`.
- [Running on Device](00-getting_started.md) — resident device tensors, explicit dispatch, benchmarking, distributed execution.
- `examples/kernels/` — every kernel there is written in this idiom.
