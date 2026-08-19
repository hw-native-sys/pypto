# Functions and Programs

How a Python function becomes an IR function, which decorator to reach for, and how
functions call each other.

> **Prerequisites:** [Types](00-types.md).

## Concept

A decorator does not wrap your function — it **parses its source**. The body never
executes as Python. That single fact explains most of what follows: why closure variables
behave the way they do, why `pl.yield_` is meaningful only inside one, and why an error in a kernel body is reported at parse time with a line number
rather than at call time with a traceback.

**`@pl.jit` is how you write PyPTO kernels.** Types come from the arguments at the first
call, the function specializes, and sub-functions are discovered automatically — you call
them by name and the decorator finds them. It is what `examples/` uses and what the rest
of this manual uses.

You will also meet `@pl.function` inside `@pl.program`: a class where each method is one
IR function and calls between them are written `self.other(...)`. That form is a
one-to-one transcription of the IR, and it exists mainly for writing compiler test cases,
where a test needs to state a program's exact shape without ever running it. As a user you
do not need it — [the section below](#plfunction-and-plprogram) is there for when you
read a compiler test or a piece of printed IR.

## Quickstart: an entry point and a device kernel

```python
import pypto.language as pl

@pl.jit.incore
def add_kernel(
    a: pl.Tensor[[128, 128], pl.FP32],
    b: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    out[:] = pl.add(a, b)
    return out

@pl.jit
def entry(
    a: pl.Tensor[[128, 128], pl.FP32],
    b: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    out = add_kernel(a, b, out)      # sub-function discovered automatically
    return out
```

| Line | What it does |
| ---- | ------------ |
| `@pl.jit.incore` | Marks a device kernel — the execution plane, where operators live |
| `@pl.jit` | Marks the chip-level entry point — the control plane, which dispatches |
| `add_kernel(a, b, out)` | A plain call; the decorator discovers the callee and wires it in |
| `out: pl.Out[...]` | Declares the direction, which is how the compiler orders this task |

To read the IR this becomes, call `entry.lower(*args)` for the post-pass `ir.Program`, or
`entry.compile(*args)` and print `compiled.program.as_python()`.

## Mechanics

### The `@pl.jit` family

Five variants, one per IR function kind, so a single program can span host, chip, and
core levels:

| Decorator | IR target | Use for |
| --------- | --------- | ------- |
| `@pl.jit` | `Orchestration` | Chip-level entry point that dispatches InCore work |
| `@pl.jit.host` | `level=HOST, role=Orchestrator` | HOST entry — allocates window buffers, dispatches chip orchestrators per rank |
| `@pl.jit.incore` | `InCore` | A device kernel (accepts `level=` to target a specific hierarchy level) |
| `@pl.jit.inline` | `Inline` | Helper spliced into every call site by `InlineFunctions` |
| `@pl.jit.opaque` | `Opaque` | A separate IR function that may hold orchestration loops and `pl.at` scopes |

Sub-function dependencies (`.incore` / `.inline` / `.opaque`) are auto-discovered from
the entry's body — call them by name. A `@pl.jit.host` entry additionally discovers
`@pl.jit` chip-orchestration dependencies, so a full distributed program needs no
`@pl.program` class.

The fragment below shows only the discovery structure — the kernel bodies are elided, and
the distributed types it names are covered in the distributed chapter, which is not
written yet:

```python
import pypto.language.distributed as pld

@pl.jit.inline
def reduce_step(local, peer, out): ...

@pl.jit
def chip_orch(inp: pl.Tensor, out: pl.Out[pl.Tensor],
              data: pl.InOut[pld.DistributedTensor], peer: pl.Scalar[pl.INT32]):
    return reduce_step(inp, peer, out)      # auto-discovered sub-function

@pl.jit.host
def host_orch(
    inputs: pl.Tensor[[2, 1, 256], pl.FP32],
    outputs: pl.Out[pl.Tensor[[2, 1, 256], pl.FP32]],
):
    data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
    for r in pl.range(pld.world_size()):
        data = pld.window(data_buf, [1, 256], dtype=pl.FP32)
        chip_orch(inputs[r], outputs[r], data, (r + 1) % pld.world_size(), device=r)
    return outputs
```

Plain `@pl.jit` entries do **not** discover other `@pl.jit` entries — only `.host`
reaches across the chip boundary. That keeps two unrelated top-level kernels from
silently folding into one program.

`@pl.jit.host` rejects `level=` (HOST is implicit).

### Three constraints that decide whether a jit kernel compiles

These are the failures new `@pl.jit` code hits, in the order it hits them.

**1. A `@pl.jit` entry body cannot hold operators.** It is an Orchestration function —
the control plane. Put the operators inside `with pl.at(level=pl.Level.CORE_GROUP):`, or
move them into a `@pl.jit.incore` sub-function.

```python
@pl.jit
def bad(x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    out[:] = pl.add(x, x)        # ✗ Misplaced tensor op ... should be inside InCore block
    return out

@pl.jit
def good(x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        out[:] = pl.add(x, x)    # ✓
    return out
```

**2. `JITFunction` has no `as_python()`.** The IR does not exist until a specialization
does. Call `lower(*args)` for the post-pass `ir.Program`, or `compile(*args)` and read
`compiled.program.as_python()` for the specialized, pre-pass IR.

**3. `compile()` takes the kernel's own arguments, not compile options.** Compile-time
knobs go through `config=RunConfig(...)`. A stray `compile(skip_ptoas=True)` is bound
against the kernel's signature and raises `TypeError: got an unexpected keyword argument`.
`@pl.jit` detects whether `ptoas` is available on its own, so `skip_ptoas` is not something
you need to pass.

### `@pl.function` and `@pl.program`

You reach for this form when writing a compiler test case, not when writing a kernel. It
describes the IR one-to-one: the class is the program, each method is a function, and the
call graph is written out rather than discovered. `@pl.jit` specializes into exactly this
shape — printing a compiled program shows you `@pl.program` source.

```python
@pl.program
class Adder:
    @pl.function(type=pl.FunctionType.InCore)
    def add_kernel(self, a, b, out): ...

    @pl.function(type=pl.FunctionType.Orchestration)
    def entry(self, a, b, out):
        out = self.add_kernel(a, b, out)     # explicit cross-function call
        return out
```

Every method takes `self` (it is stripped from the IR), and `Adder` becomes an
`ir.Program` — not a Python class you can instantiate. `Adder.as_python()` prints it.

`type=` names the plane each function belongs to:

| Function type | Plane | Typical use |
| ------------- | ----- | ----------- |
| `Opaque` (default) | none yet | Standalone building block; takes its plane from where it is used |
| `InCore` | Execution | Load / compute / store kernel |
| `Orchestration` | Control | Creates tensors, dispatches InCore tasks |
| `Inline` | none | Spliced at every call site; leaves no function behind |

A standalone `@pl.function` called from inside a `@pl.program` is added to the program as
a separate function. `@pl.inline` (and `@pl.jit.inline`) instead expand at the call site
and leave no function behind.

```python
@pl.inline
def normalize(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return pl.mul(x, 2.0)
```

The decorated object is a `pl.InlineFunction` — a template the parser splices, not a
function you can call from Python.

### Function attributes: `pl.func_attr`

Metadata about the function as a whole is declared with `pl.func_attr({...})` as the
**first statement** of the body:

```python
@pl.program
class Kernels:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 64], pl.FP32], w: pl.Tensor[[64, 64], pl.FP32],
               out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
        pl.func_attr({"stationary": w, "split": pl.SplitMode.UP_DOWN})
        ...
```

It reads oddly for a *function*-level declaration to sit inside the body, so it is worth
saying why. A decorator is evaluated before the signature binds any name, so
`@pl.function(attrs={"stationary": w})` cannot be written at all — `w` does not exist
yet. Body position places the declaration after the parameters are bound, which is what
makes an attribute that *references a parameter* expressible. The alternatives are a
positional index (`{"stationary_param": 1}`, which breaks the moment a pass reorders
parameters) or a naming convention nothing enforces.

Rules worth knowing:

| Rule | Why |
| ---- | --- |
| Must precede every other statement | An attribute describes the whole function; it must not appear to start applying partway down a body. This also bounds what it can reference to the parameters. |
| A bare name is always a parameter | `pl.func_attr({"n": k})` records the parameter `k`, never a same-named Python variable from the enclosing scope. Write Python constants as literals. |
| Multiple calls merge | A key declared twice is an error naming the key, so which value wins is never a matter of parse order. |
| `auto_scope=` and `external_source=` stay on the decorator | The parser reads them *before* it walks the body, so a body-position declaration would arrive too late to take effect. |

`@pl.function(attrs={...})` is **deprecated** and emits a `DeprecationWarning`. It still
parses and behaves identically, but it can only ever carry values that reference nothing.
Printed IR always uses a non-deprecated spelling — the `pl.func_attr` prologue, or the
dedicated `auto_scope=` / `external_source=` keywords — so reparsing compiler output never
warns.

### Splitting compile from dispatch

`@pl.jit` kernels normally fuse specialize + compile + dispatch into one `kernel(*args)`
call. `JITFunction.compile(*sample_args)` stops after compilation and hands back the
`CompiledProgram` — for driving `ChipWorker` yourself, inspecting artifacts under
`compiled.output_dir`, or validating codegen ahead of time.

```python
compiled = my_kernel.compile(sample_x, sample_w, sample_out)
print("artifacts in:", compiled.output_dir)
```

The returned object is the same one the JIT cache holds, so a later call with the same
specialization key returns the identical instance.

`lower(*sample_args)` stops one stage earlier: it runs the passes and returns the
post-pass `ir.Program`, with no code generation, no `ptoas`, no artifacts, and no cache
write. Use it to read lowered IR; use `compile()` when codegen itself is what you want to
check. Both accept `config=RunConfig(...)`, but `lower()` ignores the runtime and artifact
fields. Compile options are in [Compiling](../execution/00-compile.md) and the runtime surface in
[Running](../execution/01-run.md).

### External C++ kernels

A hand-written C++ kernel can be called like any other function. See
[Integrating Hand-Written C++ Kernels](../../dev/language/04-external-kernels.md).

## Edge Cases

> **Fatal pitfall:** verify a new `@pl.jit` example with a full `compile()`, never with
> `lower()` alone. `lower()` stops after the passes, so the "operators in an Orchestration
> body" error above never fires — the kernel appears to pass and fails only when someone
> runs it for real.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **`Misplaced tensor op ... should be inside InCore block`** | Operators directly in a `@pl.jit` body | Wrap in `with pl.at(level=pl.Level.CORE_GROUP):` or move to `@pl.jit.incore` |
| **`AttributeError: 'JITFunction' object has no attribute 'as_python'`** | Printing IR that does not exist yet | `f.lower(*args)`, or `f.compile(*args)` then `compiled.program.as_python()` |
| **`lower()` passes but `compile()` fails** | `lower()` runs no code generation | Expected — use `compile()` to check codegen |
| **`TypeError: got an unexpected keyword argument`** | A compile option was passed to `compile()`, which binds against the kernel's signature | Pass `config=RunConfig(...)` |
| **A second top-level kernel is missing from the program** | Plain `@pl.jit` does not discover other `@pl.jit` entries | Use `@pl.jit.host`, or make the callee `.incore` / `.opaque` |
| **`auto_scope=False` rejected** | Used on `.incore` / `.opaque` | Put it on the entry or on an `.inline` helper |
| **`self` missing from a `@pl.program` method** | Every method needs it | Add `self`; it is stripped from the IR |

## Worked example

`examples/utils/cross_function_calls.py` — `@pl.jit.inline` helpers auto-discovered as
deps of a `@pl.jit` entry and spliced at the call site.

## See Also

- [Control Flow](02-control-flow.md) — loops and conditionals inside these bodies.
- [Scopes and Placement](04-scopes.md) — `pl.at` and the other placement scopes.
- [Quickstart](../02-quickstart.md) — the same decorators in a worked example.
- [InlineFunctions](../../dev/passes/01-inline_functions.md) — how `Inline` bodies are spliced.
- [Integrating Hand-Written C++ Kernels](../../dev/language/04-external-kernels.md) — calling external kernels.
