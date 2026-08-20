# PyPTO

A high-performance programming framework for tile-centric computing.

PyPTO lets you write NPU kernels and their orchestration in Python, compiles them
through a multi-level IR, and hands the resulting task graph to a runtime that
executes it on device.

## Where to start

| You are... | Start here |
| ---------- | ---------- |
| New to PyPTO | [Getting Started](user/00-getting_started.md) — install, first program, compile and run |
| Writing kernels | [Language Guide](user/language/index.md) — types, functions, control flow, memory, scopes |
| Looking for an operator | [Operations](user/ops/index.md) — the `pl.*` / `pl.tensor.*` / `pl.tile.*` surface |
| Looking up a signature | [API Reference](api/index.md) — every `pl.*` name, generated from the docstrings |
| Chasing a wrong result | [Torch Codegen Debug Guide](user/tools/01-torch-codegen.md) — run the IR through PyTorch and compare |
| Working on the compiler | [Developer documentation](dev/index.md) — IR, passes, code generation |
| Reading generated code | [PTO ISA reference](reference/index.md) — cluster architecture and instruction semantics |

## PyPTO and the runtime

PyPTO is the **compiler and programming language**. The **runtime** that schedules
and executes what it produces is a separate project,
[`hw-native-sys/simpler`](https://github.com/hw-native-sys/simpler), included here
as the `runtime/` submodule.

The split decides what you will find where:

| Concern | Owner | Documentation |
| ------- | ----- | ------------- |
| The `pl.*` language, `ir.compile()`, IR and passes | PyPTO | This site |
| `pypto.runtime` — `ChipWorker`, `DeviceTensor`, `RunConfig`, `benchmark` | PyPTO (its own Python layer) | This site |
| Scheduler internals, graph building, message queue, tensormap and ring buffers | simpler | <https://hw-native-sys.github.io/simpler/> |

`pypto.runtime` is PyPTO's own Python wrapper, not a simpler API — it is documented
here. What is linked out is simpler's *internals*.

## About this documentation

- **English is authoritative.** `docs/zh/` mirrors it page for page; use the
  language selector in the header to switch.
- The markdown under `docs/` is the single source of truth and is readable directly
  on GitHub. This site is a build artifact and is never committed.
- The user manual is being expanded from the four guides above into a full
  chaptered manual (tutorials, distributed programming, performance, accuracy
  debugging). See [issue #2120](https://github.com/hw-native-sys/pypto/issues/2120)
  for the plan.
