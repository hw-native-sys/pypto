# Programming Model

The abstractions behind a PyPTO program: three levels of description, a compilation
pipeline that lowers between them, and the memory hierarchy they are all describing.

> **Prerequisites:** you have compiled the tensor-level examples in
> [Quickstart](02-quickstart.md). This page explains what they were doing, and introduces
> the tile level the quickstart deliberately left out.

## Concept

PyPTO asks you to describe a computation at whatever level of control you actually need,
and lowers everything else for you. The same program can name whole arrays and let the
compiler place them, or name individual on-chip buffers and move data by hand — usually
it does both, in different functions.

That flexibility rests on a separation that runs through the whole system: **what to
compute** is described in your Python source, **where it runs** is described by a
function's type and level, and **when it runs** is decided by the runtime from a task
graph the compiler derives. Confusing these three is the root of most early
misunderstandings, so it is worth reading the rest of this page with the distinction in
mind.

Nothing in a PyPTO program executes when Python runs it. Decorators parse source into
IR; passes rewrite that IR; code generation emits device kernels and host orchestration;
the runtime schedules them. Python is the authoring language, not the execution engine.

## Quickstart: the three levels in one program

Below, the same computation — `x * x` — is written twice: once at tensor level, once at
tile level. Both are `@pl.jit.incore` device kernels, and one orchestration entry
dispatches both.

```python
import pypto.language as pl

@pl.jit.incore
def square_tensor(
    x: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    # Tensor level: name the whole array. Placement and movement are the compiler's.
    out = pl.mul(x, x)
    return out

@pl.jit.incore
def square_tile(
    x: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    # Tile level: name the on-chip buffer, and move the data yourself.
    t = pl.load(x, [0, 0], [128, 128])
    y = pl.mul(t, t)
    pl.store(y, [0, 0], out)
    return out

@pl.jit
def levels(
    x: pl.Tensor[[128, 128], pl.FP32],
    out_t: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    out_k: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    # Control plane: no computation, just dispatch.
    out_t = square_tensor(x, out_t)
    out_k = square_tile(x, out_k)
    return out_t, out_k
```

The two kernels are interchangeable to the caller and produce the same numbers. They
differ only in how much you said out loud:

| What differs | `square_tensor` | `square_tile` |
| ------------ | --------------- | ------------- |
| What you name | The whole array `x` | The on-chip buffer `t` |
| Data movement | Compiler inserts it | You write `pl.load` / `pl.store` |
| Region | Implicit — the whole tensor | Explicit — offsets `[0, 0]`, shape `[128, 128]` |
| Memory space | Compiler chooses | You may pass `target_memory=` |
| Lines of code | 1 | 3 |

`ConvertTensorToTileOps` turns the first into something very close to the second — compare
the pass dumps to see it happen. So tile level is not a different language; it is the same
program with the choices spelled out. You descend when a choice matters: which region, when
it lands on chip, which buffer it lands in, how it is reused.

Compiling `levels` produces **two** device kernels, one per `.incore` function:

```text
kernels/aiv/square_tensor.cpp
kernels/aiv/square_tile.cpp
```

The three levels, and where each appears above:

| Level | What you name | In this example | Who decides placement |
| ----- | ------------- | --------------- | --------------------- |
| **Tensor** | Whole arrays in DDR | `square_tensor`; also `levels`, which only passes arrays around | The compiler |
| **Tile** | On-chip buffers | `square_tile` — `pl.load`, `pl.mul`, `pl.store` | You |
| **Block** | Cores and their coordination | Not used here. `pl.at(level=...)` names one core group; `pl.spmd` and `pl.cluster` go further | You, explicitly |

`pl.at` is the Block-level knob you meet first, and `@pl.jit.incore` is why this example
does not need it: an `.incore` function is already placed on a core. A single-function
kernel has no sub-function to carry that placement, so it opens the scope inline with
`with pl.at(level=pl.Level.CORE_GROUP):` — see [Quickstart](02-quickstart.md).

The quickstart stays entirely at tensor level — `out = pl.add(a, b)`, with no `pl.load` in
sight — so `square_tensor` above is the shape you already know. `square_tile` is the step
down. Block level is for saying which core does what: multi-block dispatch, cluster
scopes, mixed AIC/AIV kernels.

## Mechanics

### Control plane and execution plane

Execution is split across two planes. `Orchestration` sits on the control plane, and the
InCore family (`InCore`, and the `AIC` / `AIV` / `Group` / `Spmd` forms the compiler
derives from it) sits on the execution plane:

```text
HOST / Orchestration          control plane
  │  creates tensors, dispatches tasks, carries loop state
  │  never touches tile memory
  ▼
InCore (AIC / AIV)            execution plane
     loads, computes, stores
     never allocates tensors or dispatches work
```

`Opaque` and `Inline` are the two values that carry **no** plane, and for opposite
reasons: `Opaque` has not committed to one yet, and `Inline` never reaches code
generation as a function at all.

| Value | Plane | Meaning |
| ----- | ----- | ------- |
| `Orchestration` | Control | Host-side coordinator — allocates tensors, dispatches kernels |
| `InCore` | Execution | Compute kernel on an AICore |
| `AIC` / `AIV` / `Group` / `Spmd` | Execution | Produced by the compiler when it splits and outlines your code — you rarely write these by hand |
| `Opaque` | none (yet) | Default. No specific execution context; a building block that takes its plane from where it is used |
| `Inline` | none | Spliced into every call site by the first pass; leaves no function behind, so it never has a plane of its own |

A function's `level` and `role` refine this further — `pl.Level.HOST` with
`pl.Role.Orchestrator` marks the host orchestrator of a distributed program. (There is
no `FunctionType.Host`; the level/role pair is how host-ness is expressed.)

### The compilation pipeline

```text
Python DSL          @pl.jit / @pl.program parse source into IR
     │
     ▼
IR                  immutable tree, shared across the whole compilation
     │
     ▼
Pass pipeline       the default strategy, in order: inline, SSA, outline scopes, tensor->tile,
     │              layout, memory planning, task dependencies, ...
     ▼
CodeGen             device kernels (.pto -> C++) + host orchestration C++
```

Each stage is observable. `compiled.program.as_python()` prints the IR that came out of
the pipeline; `dump_passes=` writes a snapshot after every pass; the passes themselves are
documented individually in [Passes](../dev/passes/index.md), numbered in execution order.
`lower()` specializes the JIT function, runs the configured pass pipeline, and returns
the post-pass `ir.Program`. It performs no code generation and does not populate the
compiled-program cache. Use `compile()` to verify code generation. (`@pl.jit` functions
have no `as_python()` of their own; call `program.as_python()` on the result of `lower()`,
or `compiled.program.as_python()` after `compile()`.)

Two properties of the IR matter to you as a user:

- **It is SSA.** Every binding is written once. Rebinding a name in Python source is
  fine — the parser renames, and threads a value rebound inside a loop through that loop
  as a carried value. That is why `acc = pl.add(acc, ...)` inside `pl.range` works despite
  the IR having no mutation.
- **It is immutable.** Passes build new IR rather than mutating it, which is what makes
  per-pass snapshots meaningful for debugging.

### Memory hierarchy

Tile-level code is explicit about memory because the hardware is. `pl.load` and
`pl.move` take a `target_memory=` argument naming where the data should land:

The on-chip spaces are **six distinct buffers, not a nesting**. `Left` is not a region
inside `Mat`, and `Acc` is not inside `Right` — they are separate hardware buffers that
data moves *between*.

| Space | Enum | Hardware | Reachable directly from DDR? |
| ----- | ---- | -------- | ---------------------------- |
| DDR | `pl.Mem.DDR` | Off-chip global memory | — this *is* DDR; `pl.Tensor` parameters live here |
| Vec | `pl.Mem.Vec` | Unified buffer | **Yes** — the default `pl.load` target |
| Mat | `pl.Mem.Mat` | L1 | **Yes** — `pl.load(..., target_memory=pl.Mem.Mat)` |
| Left | `pl.Mem.Left` | L0A, matmul left operand | No — only via `pl.move` from `Mat` / `Vec` |
| Right | `pl.Mem.Right` | L0B, matmul right operand | No — only via `pl.move` |
| Acc | `pl.Mem.Acc` | L0C, matmul accumulator | No — `pl.matmul` writes it |
| Bias | `pl.Mem.Bias` | Bias buffer on the AIC core | No — only via `pl.move` |

`pl.MemorySpace` and `pl.Mem` are the same enum under two names.

That last column is the load-bearing constraint: **a DDR-facing load can only land in `Vec`
or `Mat`.** When a consumer needs `Left` / `Right` / `Acc` / `Bias`, the producer stops at
`Mat` (or `Vec`) and `InferTileMemorySpace` inserts a `tile.move` to reach the specialized
space — which is why the matmul path below has an explicit `pl.move` step.

Dataflow, as opposed to containment. The two matmul operands **converge** on `Acc`, so this
is a graph, not a tree:

```text
       pl.load(target_memory=Mat)      pl.move(Left)
  DDR ────────────────────────► Mat ─────────────────► Left ┐
                                                            │  pl.matmul
                                                            ├──────────► Acc ──────► DDR
  DDR ────────────────────────► Mat ─────────────────► Right┘                pl.store
       pl.load(target_memory=Mat)      pl.move(Right)

       pl.load()                  elementwise ops              pl.store()
  DDR ───────────► Vec ─────────────────────────────► Vec ───────────────► DDR
       (default)
```

The matmul path is the reason these spaces are exposed rather than hidden: operands have to
reach L0A/L0B through L1, and the result accumulates in L0C.

```python
@pl.jit.incore
def mm(
    a: pl.Tensor[[32, 32], pl.FP16],
    b: pl.Tensor[[32, 32], pl.FP16],
    out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
):
    a_l1 = pl.load(a, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
    b_l1 = pl.load(b, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
    a_l0a = pl.move(a_l1, target_memory=pl.Mem.Left)
    b_l0b = pl.move(b_l1, target_memory=pl.Mem.Right)
    c_acc = pl.matmul(a_l0a, b_l0b)      # lands in Acc
    pl.store(c_acc, [0, 0], out)         # Acc -> DDR
    return out
```

You do not always have to write this out — a tensor-level `pl.matmul` is lowered into
this chain for you. Writing it by hand is what buys control over tiling and residency.

### The execution model

Compiled output is not a single binary that runs top to bottom. It is a set of device
kernels plus host orchestration that **submits tasks** to the runtime, which schedules
them against a dependency graph.

```text
compiled program
 ├── orchestration/   host C++: submits tasks, carries loop state
 └── kernels/         device kernels, one per InCore function
                          │
                    runtime scheduler
                          │  derives / consumes the task dependency graph
                          ▼
                    AICore execution
```

The consequence for how you write code: **source order is not an ordering guarantee.**
The runtime orders two tasks only when something in the program establishes that they are
ordered — a dependency the compiler derived from an overlapping buffer, or one you stated
explicitly. Writing one dispatch after another expresses nothing on its own. Where a
sequence is required, express it; do not infer it from statement placement.

The hardware those tasks land on is organized in clusters: **1 Cube core and 2 buddy
Vector cores** sharing a flag-based synchronization mechanism. That shape is why mixed
kernels and cross-core pipelines exist as concepts; see
[Cluster Architecture](../reference/pto-isa/00-cluster_architecture.md).

## Edge Cases

> **Fatal pitfall:** statement order in an Orchestration function does not constrain
> execution order. If two dispatches must run in sequence, that sequence has to be
> expressed — through a dependency, or through a buffer relationship the compiler can
> see. Relying on source order alone leaves the runtime free to overlap them, and the
> result is a race that reproduces intermittently and disappears under a debugger.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **Results change between runs** | Two tasks that must be ordered have nothing expressing that order | State the dependency explicitly; source order alone does not order them |
| **`pl.load` directly in a `@pl.jit` body fails** | Tile operations used on the control plane | Wrap them in `with pl.at(level=...)`, or move them into a `@pl.jit.incore` sub-function |
| **`pl.create_tensor` inside a `@pl.jit.incore` function fails** | Tensor allocation used on the execution plane | Allocate on the control plane, or take the buffer as a `pl.Out[...]` parameter |
| **A value written in a loop is empty afterwards** | The carried value never leaves the loop | Rebind it each iteration (`acc = pl.add(acc, ...)`) and read it after the loop |
| **`pl.matmul` rejects its operands** | Operands not in `Left` / `Right` | `pl.load` to `Mat`, then `pl.move` to `Left` / `Right` |

## See Also

- [Quickstart](02-quickstart.md) — the examples this page explains.
- [Language Guide](01-language_guide.md) — the full surface: types, control flow, scopes, compilation.
- [Passes](../dev/passes/index.md) — every pass in the pipeline, in execution order.
- [IR Overview](../dev/ir/00-overview.md) — the IR's structure and design principles.
- [Cluster Architecture](../reference/pto-isa/00-cluster_architecture.md) — the Cube + Vector cluster the execution model targets.
