# Functions and Programs

How a Python function becomes an IR function, which decorator to reach for, and how
functions call each other.

> **Prerequisites:** [Types](00-types.md).

## Concept

A decorator does not wrap your function — it **parses its source**. The body never
executes as Python. That single fact explains most of what follows: why closure variables
behave the way they do, why `pl.submit` raises if you call it outside a decorated
function, and why an error in a kernel body is reported at parse time with a line number
rather than at call time with a traceback.

There are two authoring styles, and they produce the same IR.

**`@pl.jit`** writes kernels as plain functions. Types come from the arguments at the
first call, the function specializes, and sub-functions are discovered automatically.
This is the style `examples/` uses and the style this manual uses everywhere except this
page.

**`@pl.function` inside `@pl.program`** declares everything up front: the class is the
program, each method is a function, and calls between them are written `self.other(...)`.
Reach for it when you want the program's shape stated explicitly, or when a tool needs
the IR without ever calling the kernel.

Which one you pick does not change what the compiler sees. `@pl.jit` specializes into
`@pl.program` source — you can print it.

## Quickstart: the same program, both ways

```python
import pypto.language as pl

# --- jit style -------------------------------------------------------------
@pl.jit.incore
def add_kernel(
    a: pl.Tensor[[128, 128], pl.FP32],
    b: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    out = pl.add(a, b)
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

```python
# --- program style ---------------------------------------------------------
@pl.program
class Adder:
    @pl.function(type=pl.FunctionType.InCore)
    def add_kernel(self, a, b, out): ...

    @pl.function(type=pl.FunctionType.Orchestration)
    def entry(self, a, b, out):
        out = self.add_kernel(a, b, out)     # explicit cross-function call
        return out
```

| Aspect | `@pl.jit` | `@pl.program` |
| ------ | --------- | ------------- |
| Function kind | From the decorator variant | From `type=` |
| Sub-function wiring | Discovered from the body | Written as `self.method(...)` |
| Types | Specialized from the first call's arguments | Declared in annotations |
| Getting the IR | `entry.lower(*args)`, or `entry.compile(*args)` then `compiled.program.as_python()` | `Adder.as_python()` |

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
    out = pl.add(x, x)        # ✗ Misplaced tensor op ... should be inside InCore block
    return out

@pl.jit
def good(x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        out = pl.add(x, x)    # ✓
    return out
```

**2. `JITFunction` has no `as_python()`.** The IR does not exist until a specialization
does. Call `lower(*args)` for the post-pass `ir.Program`, or `compile(*args)` and read
`compiled.program.as_python()` for the specialized, pre-pass IR.

**3. `compile()` takes the kernel's own arguments, not compile options.** Compile-time
knobs go through `config=RunConfig(...)`. A stray `compile(skip_ptoas=True)` is bound
against the kernel's signature and raises `TypeError: got an unexpected keyword argument`. `@pl.jit` detects whether `ptoas` is available on its own, so
`skip_ptoas` is not something you need to pass.

### `@pl.function` and `@pl.program`

`@pl.function` parses one function; `type=` names its plane:

| Function type | Plane | Typical use |
| ------------- | ----- | ----------- |
| `Opaque` (default) | none yet | Standalone building block; takes its plane from where it is used |
| `InCore` | Execution | Load / compute / store kernel |
| `Orchestration` | Control | Creates tensors, dispatches InCore tasks |
| `Inline` | none | Spliced at every call site; leaves no function behind |

`@pl.program` groups functions into a compilable program. Every method takes `self` (it is
stripped from the IR), cross-function calls are `self.method(...)`, and the decorated
class becomes an `ir.Program` — not a Python class you can instantiate.

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

### Runtime scope placement

By default the compiler inserts AUTO runtime scopes (`PTO2_SCOPE`) for you. Pass
`auto_scope=False` to place them by hand with `with pl.scope():` — accepted on `@pl.jit`,
`@pl.jit.host` and `@pl.jit.inline`, rejected on `.incore` / `.opaque` (they outline into
separate kernels). Inline bodies are spliced into the caller, so their hand-placed scopes
land there. See [Scopes and Tasks](04-scopes-and-tasks.md) and
[MaterializeRuntimeScopes](../../dev/passes/42-materialize_runtime_scopes.md).

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
fields. Details of the compile options and the runtime surface belong to the execution
chapter, which is not written yet — for now see
[Compiling a Program](../01-language_guide.md).

### External C++ kernels

A hand-written C++ kernel can be called like any other function. See
[Integrating Hand-Written C++ Kernels](../../dev/language/01-external-kernels.md).

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

## See Also

- [Control Flow](02-control-flow.md) — loops and conditionals inside these bodies.
- [Scopes and Tasks](04-scopes-and-tasks.md) — `pl.at`, runtime scopes, and dispatching tasks.
- [Quickstart](../02-quickstart.md) — the same decorators in a worked example.
- [InlineFunctions](../../dev/passes/01-inline_functions.md) — how `Inline` bodies are spliced.
- [Integrating Hand-Written C++ Kernels](../../dev/language/01-external-kernels.md) — calling external kernels.
