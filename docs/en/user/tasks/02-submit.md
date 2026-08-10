# Declaring an Edge

Naming a dispatch so a later one can wait on it.

> **Prerequisites:** [The dependency model](00-model.md) and [Runtime scopes](01-scopes.md).

## Concept

An ordinary kernel call becomes a task, but you never get a handle to it. To declare an edge
the inference cannot reach, you need a **TaskId** — a handle naming one dispatch — and a way
to pass it to a later task as a dependency.

There are three spellings, and which ones you can use is decided by how the function is
written, not by preference:

| Spelling | Available in | Names the task |
| -------- | ------------ | -------------- |
| `with pl.at(level=..., deps=[...]) as tid:` | `@pl.jit` and `@pl.function` | An inline region |
| `with pl.spmd(n, deps=[...]) as tid:` | `@pl.jit` and `@pl.function` | An inline multi-block region |
| `result, tid = pl.submit(self.kernel, ...)` | `@pl.program` classes only | A pre-declared kernel |

The two scope forms bind the TaskId with `as`; `pl.submit` returns it. `pl.submit` requires
its callee to be written as `self.<kernel>` — a method of the enclosing `@pl.program` class
— so it is not reachable from a `@pl.jit` function, where kernels are plain module-level
functions. All three feed the same `deps=` machinery.

**Explicit edges are not tied to manual scope.** `deps=` composes with automatic tracking —
the final wait set is the union of both — so an explicit edge is perfectly at home in an
ordinary auto scope. Use it there as a precision tool for the one edge the inference could
not reach; use [`pl.manual_scope`](01-scopes.md) only when you want the whole graph.

## Quickstart: an explicit edge in an auto scope

```python
import pypto.language as pl

@pl.jit
def two_stage(
    x: pl.Tensor[[256, 128], pl.FP32],
    scratch: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
    out: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP) as first:
        scratch = pl.add(x, x)
    with pl.at(level=pl.Level.CORE_GROUP, deps=[first]) as second:
        out = pl.add(scratch, scratch)
    return scratch, out
```

| Element | Meaning |
| ------- | ------- |
| `as first` | Binds the TaskId of the region — the handle a later task names |
| `deps=[first]` | This region waits on that producer, whatever the OverlapMap did or did not infer |

Note there is no `manual_scope` here. The `deps=` is added on top of normal tracking, not
instead of it.

## Mechanics

### `pl.at` as a task boundary

Every `pl.at` region is one dispatch, so binding it with `as tid` gives you its TaskId.
The dependency keywords are:

| Keyword | Purpose |
| ------- | ------- |
| `deps=[...]` | TaskId scalars / arrays this region must wait on |
| `no_dep_args=[...]` | Tensors to opt out of tracking — see [Refining the graph](03-tuning.md) |
| `dumps=[...]` | Tensors to mark for selective dump |
| `allow_early_resolve=True` | Scheduling hint — see [Refining the graph](03-tuning.md) |

`pl.at` has no `predicate=`. For a predicated region use `pl.spmd` or a submit form.

### `pl.spmd` as a task boundary

`pl.spmd(n)` — covered as a placement construct in
[Scopes and Placement](../language/04-scopes.md) — is also one dispatch, so it names a task
the same way:

```python
with pl.spmd(4, name_hint="stage1") as first:
    ...
with pl.spmd(4, name_hint="stage2", deps=[first]) as second:
    ...
```

It accepts `deps=`, `predicate=` and `allow_early_resolve=`, which makes it the one inline
form that carries a dispatch predicate.

**`deps=` requires the `as` form.** Binding with `as` is what gives you the TaskId; the bare
`with pl.spmd(4):` and `for i in pl.spmd(4):` forms run the same work without naming it, and
passing `deps=` to either is rejected with `pl.spmd() does not accept 'deps=' here`.

### `pl.submit`

For programs written as a `@pl.program` class with pre-declared kernels:

```text
result, tid = pl.submit(self.kernel, *kernel_args, deps=[...], dumps=[...],
                        allow_early_resolve=False, predicate=(...))
```

The positional slots after the callee are the kernel's own arguments; everything else is an
optional keyword. The callee **must** be written `self.<kernel>` — any other expression is a
parse error.

The unpacking shapes that work:

```python
a, tid = pl.submit(self.k1, x)          # result and TaskId
res    = pl.submit(self.k1, x)          # whole flat tuple in one name
```

A TaskId reached by indexing that flat tuple cannot feed `deps=` — the dependency must be a
TaskId variable or a TASK_ID array element, so bind it by name.

### `pl.spmd_submit`

The SPMD sibling: one orchestration task the runtime fans out across `core_num` logical
blocks, each kernel reading its own index via `pl.tile.get_block_idx()`. It still returns a
single producer TaskId, so the whole fan-out is named as one dependency.

```python
a, tid = pl.spmd_submit(self.k1, x, core_num=8)
```

`core_num` is a **required keyword** — the positional slots belong to the kernel.
`sync_start` (default `False`) requires all blocks to launch atomically. `deps=`,
`allow_early_resolve=` and `predicate=` behave exactly as on `pl.submit`.

### Fan-in through a TaskId array

One TaskId names one task. To wait on a *set* of tasks — a loop's worth of producers —
collect them in a `pl.array` of `pl.TASK_ID` and pass the array itself as a dependency:

```python
tids = pl.array.create(branches, pl.TASK_ID)
for branch in pl.parallel(branches):
    out, tid = pl.submit(self.producer, data, branch, out)
    tids[branch] = tid
out, _ = pl.submit(self.consumer, data, out, deps=[tids])
```

`deps=` accepts arrays as well as scalars, so the consumer waits on every producer the loop
created. Remember that an array update rebinds — inside a loop the array is a carried value
like any other. See [Control Flow](../language/02-control-flow.md).

## Edge Cases

> **Fatal pitfall:** `deps=` is the *only* thing an explicit edge gives you. It does not
> imply a manual scope, and a manual scope does not imply edges. Mixing the two ideas up —
> assuming `manual_scope` orders the statements inside it — produces a region where nothing
> is ordered at all.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **`pl.submit(...) first argument must be a self.<kernel> method reference`** | `pl.submit` used in a `@pl.jit` function | Use `with pl.at(..., deps=[...]) as tid:` — or `with pl.spmd(n, deps=[...]) as tid:` for a multi-block dispatch |
| **`pl.submit is a DSL parser construct and cannot be called directly`** | Used outside a decorated function body | Move it inside the decorated body |
| **`unpacks 1 result value(s) but kernel returns 0`** | The kernel writes an `Out` parameter and declares no return type | Unpack only what the kernel returns, or give it a return type |
| **`deps= entries must be a TaskId variable`** | A TaskId reached by indexing a flat submit result | Bind the TaskId to its own name and pass that |
| **`pl.spmd() does not accept 'deps=' here`** | `deps=` was passed to the bare `with` or the `for` form | Bind the region with `as tid:` — only that form takes `deps=` |
| **`core_num` missing** | It is a required keyword on `pl.spmd_submit` | Pass `core_num=N`; positional slots are the kernel's arguments |
| **A consumer only waits on the last producer of a loop** | One TaskId was reused instead of collected | Collect into a `pl.array` of `pl.TASK_ID` and pass the array |
| **Explicit edge seems ignored in auto scope** | It is not — the wait set is the union | Look for a *missing* edge elsewhere, not a discarded one |

## See Also

- [Runtime scopes](01-scopes.md) — where auto tracking is on or off.
- [Refining the graph](03-tuning.md) — `predicate=` and `allow_early_resolve=`.
- [Control Flow](../language/02-control-flow.md) — carried values, which a TaskId array is.
- [Compile-Time Directives](../language/05-directives.md) — the dump marks `dumps=` feeds.
