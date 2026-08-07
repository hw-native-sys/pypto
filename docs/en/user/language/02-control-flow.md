# Control Flow

Loops and conditionals, the values they carry across iterations, and the SSA rule that
shapes both.

> **Prerequisites:** [Functions and Programs](01-functions.md).

## Concept

PyPTO's IR is **SSA**: every binding is written exactly once. That is invisible in
straight-line code — the parser renames `x` for you as often as you rebind it — but at a
loop boundary or a branch merge it becomes visible, because there is no single place for
the compiler to put the second write.

`pl.yield_` is the answer. It names the value that leaves a scope:

- In a loop, the yielded value becomes the next iteration's input and, after the last
  iteration, the loop's result.
- In an `if` / `else`, both branches yield, and the merge produces one result variable
  (a phi node).

You do not always write `yield_` yourself. Rebinding a name inside a loop is the ordinary
way to accumulate, and the parser turns it into a carried value for you — `init_values=`
plus `yield_` is the same thing spelled out, and it is mandatory only for `pl.while_`. A
branch that produces a value, on the other hand, must always yield from both arms.
Everything else on this page is a variation on those rules.

Four loop constructs share one syntax and differ only in what the compiler is told:

```text
pl.range     sequential — the default
pl.parallel  iterations are independent and may overlap
pl.unroll    fully unrolled at compile time; bounds must be literals
pl.pipeline  body replicated `stage` times for ping-pong buffering
```

## Quickstart: a loop that accumulates

```python
import pypto.language as pl

@pl.jit
def accumulate(
    a: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.add(a, a)
        for i in pl.range(3):
            t = pl.add(t, a)      # carried across iterations
        out = pl.mul(t, t)
    return out
```

| Line | What it does |
| ---- | ------------ |
| `t = pl.add(a, a)` | Establishes the value before the loop |
| `for i in pl.range(3)` | Three iterations; `i` is the loop variable |
| `t = pl.add(t, a)` | Looks like mutation; is a carried value |
| `out = pl.mul(t, t)` | Reading `t` after the loop reads the last iteration's result |

Rebinding `t` inside the loop is the idiomatic way to carry a value. The IR is SSA, so
the parser gives each iteration's value its own name and threads it through the loop as a
carried value — which is exactly what the explicit form below spells out by hand.

## Mechanics

### The four loop forms

All four take `(stop)`, `(start, stop)`, or `(start, stop, step)`. Arguments may be `int`
literals or `pl.Scalar` values, except where noted.

| Form | Emits | Carried values | Bounds |
| ---- | ----- | -------------- | ------ |
| `pl.range(...)` | `ForKind.Sequential` | Yes | int or `Scalar` |
| `pl.parallel(...)` | `ForKind.Parallel` | Yes | int or `Scalar` |
| `pl.unroll(...)` | `ForKind.Unroll` | **No** | literals only |
| `pl.pipeline(..., stage=N)` | Sequential + pipelining | Yes | int or `Scalar` |

```python
for i in pl.range(10): ...            # 0..9
for i in pl.range(2, 10): ...         # 2..9
for i in pl.range(0, 100, 4): ...     # 0, 4, ..., 96
for i in pl.parallel(0, nblocks): ...
for i in pl.unroll(4): ...            # no init_values here
```

`pl.parallel` is an assertion, not a request: you are telling the compiler the iterations
are independent. If they are not, the result is a race.

`pl.pipeline(N, stage=F)` replicates the body `F` times per outer iteration so buffers can
ping-pong. The outer loop advances in strides of `stage * step` and a tail dispatch covers
the remainder when the trip count does not divide evenly. `stage` is required and must be
a positive integer (typically 2–4). It is lowered at tile level by
[LowerPipelineLoops](../../dev/passes/29-lower_pipeline_loops.md) — or, under
`memory_planner=PTOAS`, by [LowerPipelineToSlots](../../dev/passes/28-lower_pipeline_to_slots.md),
which rotates one un-replicated body through the slots of a single allocation instead.

```python
for i in pl.pipeline(64, stage=4):
    t = pl.load(a, [i * 64, 0], [64, 64])
    pl.store(t, [i * 64, 0], out)
```

### Naming the carry explicitly

`init_values=` plus `pl.yield_` is the explicit spelling of what rebinding does
implicitly. It is **required** for `pl.while_` (below), and it is the form printed IR
uses, so you will read it even when you do not write it. Up to five carried values are
supported by the typed overloads.

The fragment below assumes `init_max` / `init_sum` are values of the same type as the
loop body produces:

```python
for i, (acc_max, acc_sum) in pl.range(4, init_values=(init_max, init_sum)):
    out_max, out_sum = pl.yield_(pl.maximum(acc_max, row_i), pl.add(acc_sum, row_i))
```

Two rules govern it: the loop is read **after** it under the name `yield_` bound (not the
name in `init_values`), and every carried value must keep one type across the whole loop.
Mixing levels is where this bites — a value seeded from a tensor-level expression and
then combined with a tile-level one inside an InCore scope does not type-check, and the
error names the operator rather than the carry.

### While loops

`pl.while_` **always** requires `init_values`, and the condition is set by `pl.cond()` as
the **first statement** of the body:

```python
for (x,) in pl.while_(init_values=(0,)):
    pl.cond(x < 10)            # continue while true
    x_out = pl.yield_(x + 1)
```

`pl.cond` is purely syntactic — the parser lifts it onto the `WhileStmt`. It is not a
runtime call and cannot appear anywhere else.

### Conditionals

A branch that produces a value must yield from **both** arms, with the same number and
types of values:

```python
for i, (prev,) in pl.range(4, init_values=(init,)):
    if i == 0:
        result = pl.yield_(a)
    else:
        result = pl.yield_(pl.add(prev, delta))
    out = pl.yield_(result)
```

If one branch yields, the other must too. A branch with no value to produce yields
nothing in either arm.

### SSA, and when you have to think about it

Write ordinary Python — rebind names freely. The parser renames:

```python
result = pl.mul(x, 2.0)
result = pl.add(result, 1.0)      # fine; the parser produces two bindings
```

`@pl.function(strict_ssa=True)` makes the parser reject a rebind instead, which is
occasionally useful for catching unintended shadowing. It also disables the
`dst[...] = src` subscript-write sugar, which works by rebinding — see
[Directives § subscript sugar](06-syntax.md#subscript-sugar).

The pipeline runs [ConvertToSSA](../../dev/passes/04-convert_to_ssa.md) early, so
non-SSA source is normal input, not a compatibility mode.

## Edge Cases

> **Fatal pitfall:** reading the `init_values` name after a loop reads the *initial*
> value, not the accumulated one. The result is a tensor of whatever the initializer held
> — frequently zeros, which looks like a compute bug rather than a naming one. Read the
> name bound by `pl.yield_`.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **A value written in a loop is empty afterwards** | Reading the `init_values` name instead of the `yield_` name | Read the `yield_` binding |
| **`tensor.add requires ... but got TileType`** | A carried value changed level mid-loop — inside an InCore scope `pl.slice` yields a Tile | Keep one level across the carry; use `pl.load` / `pl.store` explicitly inside the scope |
| **`while_() requires init_values to be specified`** | `pl.while_()` with no carried state | Add `init_values=(...)`, even if it is a single counter |
| **Parser error on `pl.cond`** | Not the first statement of the `while_` body | Move it to the first line |
| **Branch merge rejected** | Only one arm yields, or arities differ | Yield the same count and types from both arms |
| **`unroll()` rejects `init_values`** | Unrolled loops carry no state | Use `pl.range`, or restructure so no value is carried |
| **`unroll()` rejects a bound** | Bounds must be compile-time literals | Use `pl.range` for a `Scalar` bound |
| **`pipeline()` stage error** | `stage` missing or not a positive int | Pass `stage=N`, N ≥ 1 |
| **Race under `pl.parallel`** | Iterations are not actually independent | Use `pl.range`, or remove the cross-iteration dependency |
| **`SSAViolationError`** | Rebinding under `strict_ssa=True` | Use distinct names, or drop `strict_ssa` |

## See Also

- [Types](00-types.md) — what the carried values are.
- [Scopes and Placement](04-scopes.md) — the placement scopes these loops sit inside.
- [ConvertToSSA](../../dev/passes/04-convert_to_ssa.md) — the conversion this page's rules come from.
- [UnrollLoops](../../dev/passes/02-unroll_loops.md) — what `pl.unroll` becomes.
- [LowerPipelineToSlots](../../dev/passes/28-lower_pipeline_to_slots.md) — what `pl.pipeline` becomes under `memory_planner=PTOAS`.
- [LowerPipelineLoops](../../dev/passes/29-lower_pipeline_loops.md) — what `pl.pipeline` becomes otherwise.
