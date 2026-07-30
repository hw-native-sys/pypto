# Programming Model

The abstractions behind a PyPTO program: three levels of description, a compilation
pipeline that lowers between them, and the memory hierarchy they are all describing.

> **Prerequisites:** you have run the examples in [Quickstart](02-quickstart.md). This
> page explains what those examples were doing.

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

```python
import pypto.language as pl

@pl.jit.incore
def scale(
    x: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    # Tile level: on-chip buffers, moved explicitly
    t = pl.load(x, [0, 0], [128, 128])
    y = pl.mul(t, t)
    pl.store(y, [0, 0], out)
    return out

@pl.jit
def levels(
    x: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    # Tensor level: whole arrays; placement is the compiler's problem
    return scale(x, out)
```

| Level | What you name | Where it appears above | Who decides placement |
| ----- | ------------- | ---------------------- | --------------------- |
| **Tensor** | Whole arrays in DDR | `levels` — passing `x` and `out` around | The compiler |
| **Tile** | On-chip buffers | `scale` — `pl.load`, `pl.mul`, `pl.store` | You |
| **Block** | Cores and their coordination | `pl.at(level=...)` names one; `pl.spmd` and `pl.cluster` go further | You, explicitly |

`pl.at` is the Block-level knob you meet first. `@pl.jit.incore` above already places the
compute on a core, so this example does not need it; a single-function kernel says
`with pl.at(level=pl.Level.CORE_GROUP):` instead — see
[Quickstart](02-quickstart.md#quickstart-element-wise-add).

Most programs stop at the first two. Block level is for when you need to say which core
does what — multi-block dispatch, cluster scopes, mixed AIC/AIV kernels.

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
Pass pipeline       44 passes in the default strategy: inline, SSA, outline scopes, tensor->tile,
     │              layout, memory planning, task dependencies, ...
     ▼
CodeGen             device kernels (.pto -> C++) + host orchestration C++
```

Each stage is observable. `compiled.program.as_python()` prints the IR that came out of
the pipeline; `dump_passes=` writes a snapshot after every pass; the passes themselves are
documented individually in [Passes](../dev/passes/index.md), numbered in execution order.
(`@pl.jit` functions have no `as_python()` of their own — the IR exists once `compile()` or
`compile_for_test()` has produced it.)

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

```text
DDR (off-chip)
 ├── Vec    unified buffer        <- pl.load()                      default
 └── Mat    L1 matrix buffer      <- pl.load(..., target_memory=pl.Mem.Mat)
      ├── Left   L0A              <- pl.move(..., target_memory=pl.Mem.Left)
      └── Right  L0B              <- pl.move(..., target_memory=pl.Mem.Right)
           └── Acc  L0C           <- pl.matmul() writes here
                └── DDR           <- pl.store()
```

| Space | Enum | Role |
| ----- | ---- | ---- |
| DDR | `pl.Mem.DDR` | Off-chip global memory; where `pl.Tensor` parameters live |
| Vec | `pl.Mem.Vec` | Unified vector buffer — the default target of `pl.load` |
| Mat | `pl.Mem.Mat` | L1 matrix buffer |
| Left | `pl.Mem.Left` | L0A — matmul left operand |
| Right | `pl.Mem.Right` | L0B — matmul right operand |
| Acc | `pl.Mem.Acc` | L0C — matmul accumulator |
| Bias | `pl.Mem.Bias` | Bias buffer on the AIC core |

`pl.MemorySpace` and `pl.Mem` are the same enum under two names.

The matmul path is the reason this hierarchy is exposed rather than hidden: operands
have to reach L0A/L0B through L1, and the result accumulates in L0C.

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
