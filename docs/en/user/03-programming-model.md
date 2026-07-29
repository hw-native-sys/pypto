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

@pl.program
class Levels:
    @pl.function(type=pl.FunctionType.InCore)
    def scale(
        self,
        x: pl.Tensor[[128, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        # Tile level: an on-chip buffer, moved explicitly
        t: pl.Tile[[128, 128], pl.FP32] = pl.load(x, [0, 0], [128, 128])
        y: pl.Tile[[128, 128], pl.FP32] = pl.mul(t, t)
        return pl.store(y, [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
        # Tensor level: whole arrays; placement is the compiler's problem
        buf: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], dtype=pl.FP32)
        return self.scale(x, buf)
```

| Level | What you name | Where it appears above | Who decides placement |
| ----- | ------------- | ---------------------- | --------------------- |
| **Tensor** | Whole arrays in DDR | `main` — `pl.create_tensor`, passing `x` around | The compiler |
| **Tile** | On-chip buffers | `scale` — `pl.load`, `pl.mul`, `pl.store` | You |
| **Block** | Cores and their coordination | Not used here; `pl.spmd`, `pl.cluster`, `pl.at` | You, explicitly |

Most programs stop at the first two. Block level is for when you need to say which core
does what — multi-block dispatch, cluster scopes, mixed AIC/AIV kernels.

## Mechanics

### Control plane and execution plane

A program is split across two planes, and every function belongs to exactly one:

```text
HOST / Orchestration          control plane
  │  creates tensors, dispatches tasks, carries loop state
  │  never touches tile memory
  ▼
InCore (AIC / AIV)            execution plane
     loads, computes, stores
     never allocates tensors or dispatches work
```

`FunctionType` records which plane a function is on:

| Value | Plane | Meaning |
| ----- | ----- | ------- |
| `Opaque` | — | Default. No specific execution context; usable as a building block |
| `Orchestration` | Control | Host-side coordinator — allocates tensors, dispatches kernels |
| `InCore` | Execution | Compute kernel on an AICore |
| `Inline` | — | Spliced into every call site by the first pass; leaves no function behind |
| `AIC` / `AIV` / `Group` / `Spmd` | Execution | Produced by the compiler when it splits and outlines your code — you rarely write these by hand |

A function's `level` and `role` refine this further — `pl.Level.HOST` with
`pl.Role.Orchestrator` marks the host orchestrator of a distributed program. (There is
no `FunctionType.Host`; the level/role pair is how host-ness is expressed.)

### The compilation pipeline

```text
Python DSL          @pl.program / @pl.function parse source into IR
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

Each stage is observable. `as_python()` prints the IR at any point;
`dump_passes=` writes a snapshot after every pass; the passes themselves are documented
individually in [Passes](../dev/passes/index.md), numbered in execution order.

Two properties of the IR matter to you as a user:

- **It is SSA.** Every binding is written once. Rebinding a name in Python source is
  fine — the parser renames — but values that cross a loop boundary must be carried
  explicitly with `pl.yield_`, which is why accumulators look the way they do.
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
a_l1 = pl.load(a, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
b_l1 = pl.load(b, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
a_l0a = pl.move(a_l1, target_memory=pl.Mem.Left)
b_l0b = pl.move(b_l1, target_memory=pl.Mem.Right)
c_acc = pl.matmul(a_l0a, b_l0b)          # lands in Acc
out = pl.store(c_acc, [0, 0], output)    # Acc -> DDR
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

The consequence for how you write code: **the order of statements in an orchestration
function is not the order of execution.** The compiler derives dependencies from buffer
overlap, and the runtime runs anything unordered concurrently. Two dispatches that touch
disjoint buffers may overlap even though one is written after the other. Where you need
a specific order, you say so — with dependencies rather than statement placement.

The hardware those tasks land on is organized in clusters: **1 Cube core and 2 buddy
Vector cores** sharing a flag-based synchronization mechanism. That shape is why mixed
kernels and cross-core pipelines exist as concepts; see
[Cluster Architecture](../reference/pto-isa/00-cluster_architecture.md).

## Edge Cases

> **Fatal pitfall:** statement order in an Orchestration function does not constrain
> execution order. If two dispatches must be ordered but touch different buffers,
> nothing in the source expresses that — the runtime is free to overlap them, and the
> result is a race that reproduces intermittently and disappears under a debugger.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **Results change between runs** | Two tasks race because no dependency links them | Make the dependency explicit rather than relying on statement order |
| **`pl.load` inside an Orchestration function fails** | Tile operations used on the control plane | Move the tile code into an `InCore` function and dispatch it |
| **`pl.create_tensor` inside an InCore function fails** | Tensor allocation used on the execution plane | Allocate in the Orchestration function; pass the buffer in as `pl.Out[...]` |
| **A value written in a loop is empty afterwards** | Loop-carried value not yielded | Carry it with `init_values=` + `pl.yield_` |
| **`pl.matmul` rejects its operands** | Operands not in `Left` / `Right` | `pl.load` to `Mat`, then `pl.move` to `Left` / `Right` |

## See Also

- [Quickstart](02-quickstart.md) — the examples this page explains.
- [Language Guide](01-language_guide.md) — the full surface: types, control flow, scopes, compilation.
- [Passes](../dev/passes/index.md) — every pass in the pipeline, in execution order.
- [IR Overview](../dev/ir/00-overview.md) — the IR's structure and design principles.
- [Cluster Architecture](../reference/pto-isa/00-cluster_architecture.md) — the Cube + Vector cluster the execution model targets.
