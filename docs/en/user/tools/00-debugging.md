# Debugging

Error types, log levels, and reading the IR the compiler produced.

## Concept

PyPTO separates two kinds of failure, and the distinction is the first thing to read off an
error. A **user error** means the input was invalid — a shape that cannot work, an operator
that does not support the dtype — and the message names what to change. An **internal
error** means an invariant the compiler maintains was violated, which is a compiler bug
whatever your input was.

Both arrive as Python exceptions. What separates them is the type and the wording.

## Quickstart: read the IR the compiler produced

<!-- doctest: setup -->
```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

CFG = RunConfig(platform="__PLATFORM__")
torch.manual_seed(0)
A = torch.randn(64, 128, dtype=torch.float32)
```

<!-- doctest: run -->
```python
@pl.jit
def scale(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        out[:] = pl.mul(a, 2.0)
    return out


prog = scale.lower(A, torch.zeros(64, 128))   # passes only: no codegen, no artifacts
src = prog.as_python()                        # the lowered IR, as DSL source

# What comes back is post-pass IR, not what you wrote: the pl.at scope has been
# outlined into its own InCore function.
assert "@pl.program" in src
assert "def scale_incore_0(" in src
assert "pl.at" not in src

print(prog.as_python(concise=True)[:200])     # concise= drops intermediate type annotations
```

`lower()` is the cheap form — it runs the passes and returns the `Program` without writing
anything. `as_python()` is available on a `Program` or a single `Function`; a `JITFunction`
has none of its own, because the IR does not exist until a specialization does.

**Expect the IR not to look like your source.** The assertions above are the point: by the
time the passes have run, `pl.at` is gone — the region became `scale_incore_0`, an `AIV`
function with a `SubWorker` role, and its tiles carry explicit `pl.MemRef` allocations.
Reading a dump means reading that form, not yours.

## Mechanics

### Error types

| Type | Means | What to do |
| ---- | ----- | ---------- |
| `pypto.Error` | Base of the PyPTO hierarchy | — |
| `ValueError` / `TypeError` / `IndexError` | A user error raised by a `CHECK` | Fix the input; the message states what was expected and what arrived |
| `pypto.InternalError` | An invariant broke — a compiler bug | File it, with the IR that reproduces it |
| `PartialCodegenError` | Codegen produced some kernels and failed on others | The report names which; usually a ptoas rejection |

An internal error says so in its text (`Internal error: ...`) and carries the source
location of the check that failed. That location is inside the compiler, not inside your
kernel — the DSL span, when there is one, is printed alongside it.

### Log levels

```python
import pypto

pypto.set_log_level(pypto.LogLevel.DEBUG)
print(pypto.get_log_level())
```

`NONE` / `FATAL` / `ERROR` / `WARN` / `INFO` / `EVENT` / `DEBUG`, increasing in volume.
`INFO` is the default.

### Environment variables

| Variable | Controls |
| -------- | -------- |
| `PYPTO_VERIFY_LEVEL` | Default IR verification level when no `PassContext` sets one |
| `PYPTO_WARNING_LEVEL` | Default diagnostic phase gate |
| `PYPTO_PROG_BUILD_DIR` | Base directory for generated artifacts (default `build_output`) |
| `PYPTO_EMIT_PTO_LOC` | Carry DSL source locations into the emitted `.pto` |
| `PYPTO_COMPILE_PROFILING` | Per-stage compile timing |
| `PYPTO_EMIT_DEBUG_RUNNER` | Emit the standalone debug runner beside the artifacts |

Each is a *default* only: an explicit argument or an active `PassContext` overrides it.

### Pass dumps

`dump_passes=PassDumpLevel.EXPLICIT` writes `passes_dump/NN_after_<PassName>.py` in
execution order. Two pages read them: this one, for "which pass changed my IR", and the
[memory map](02-memory-map.md), for what the allocation looks like.

Diffing two adjacent dumps is the mechanical way to attribute a change to a pass.
`CompiledProgram.validate_ir` automates the semantic half of that comparison — see
[Precision](../precision/00-workflow.md), including the note about its tolerances.

### Round-tripping IR through text

A program can be written out as DSL source and read back:

| Direction | Call |
| --------- | ---- |
| IR → text | `program.as_python()` |
| text → IR | `pl.parse_program(code)` |
| file → IR | `pl.loads_program(path)` |

`examples/utils/parse_from_text.py` is the worked version. The round trip is also what
`VerificationLevel.ROUNDTRIP` exercises on every pass, which is why it is slow enough to be
opt-in.

## Edge Cases

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **`JITFunction` has no `as_python`** | The IR does not exist until specialization | `kernel.lower(*args).as_python()` |
| **Error names a compiler file, not your kernel** | It is an `InternalError` | File a bug; do not work around it |
| **`ptoas compilation failed:` with an empty message** | The ptoas binary crashed | Point `PTOAS_ROOT` at a working version |
| **No `passes_dump/`** | `lower()` writes no artifacts | Use `compile()` with `dump_passes=` |

> **An internal error is not yours to route around.** Silencing it with a shape change or
> a different spelling hides an invariant violation that will resurface somewhere less
> convenient.

## See Also

- [Torch codegen](01-torch-codegen.md) — running the IR's semantics on the host.
- [Memory map](02-memory-map.md) — the other reader of pass dumps.
- [Precision](../precision/00-workflow.md) — the workflow these tools plug into.
- [Error handling](../../dev/02-error-handling.md) — the `CHECK` / `INTERNAL_CHECK` contract behind the types above.
