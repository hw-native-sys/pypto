# Directives

Constructs that shape compilation or observation rather than computation: compile-time
statements, dump marks, arrays, and the syntax sugar the parser recognizes.

> **Prerequisites:** [Functions and Programs](01-functions.md).

## Concept

Everything on this page shares one property: it does not run when you think it does.

`pl.static_print` and `pl.static_assert` run at **parse time** and are gone from the IR.
`pl.dump_tag` runs at neither parse nor kernel time — it records an intent that the
runtime acts on, and only when dumping is enabled. `pl.const` is a typed literal, not a
call. The subscript sugar is rewriting, not indexing.

Knowing which phase a construct belongs to is the difference between a debugging aid and a
puzzle. A `static_print` that shows nothing is not broken — the function was never parsed.

## Quickstart: seeing what the parser sees

```python
import pypto.language as pl

@pl.jit.incore
def probe(x: pl.Tensor[[64, 128], pl.FP32],
          out: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
    pl.static_print("x =", x)                      # prints at parse time
    pl.static_assert(x.shape[1] == 128, "expected 128 columns")
    out = pl.mul(x, 2.0)
    return out
```

Both statements vanish from the IR. The `static_print` output appears when the decorator
parses the source — that is, when the module is imported for `@pl.function`, or at the
first specializing call for `@pl.jit`.

## Mechanics

### Compile-time statements

| Construct | When it runs | Failure mode |
| --------- | ------------ | ------------ |
| `pl.static_print(*args)` | Parse time | none — pure output |
| `pl.static_assert(cond, msg)` | Parse time | `ParserError` if false |

`static_assert` is **statement-only** — it cannot appear inside an expression — and its
`msg` must be a **string literal** at the call site. Passing a variable raises
`ParserSyntaxError`. The condition must be compile-time evaluable; it is never checked at
execution time.

### Typed constants

`pl.const(value, dtype)` builds a constant with an explicit dtype rather than the default
one inferred from the literal. It exists so the printer can round-trip non-default
constant types, and it is what you want when a literal's width matters:

```python
step = pl.const(1, pl.INT32)
```

### Selective tensor dump

Dumping every binding saturates the host-side collector on large workloads, so the runtime
supports partial dump with per-tensor marks. Two spellings feed the same `dump_vars` set:

| Form | Scope of the mark |
| ---- | ----------------- |
| `pl.dump_tag(t)` as a standalone statement | Every *subsequent* dispatch consuming that value |
| `dumps=[t, ...]` on `pl.submit(...)` / `pl.at(...)` | That one dispatch |

Use `dump_tag` when one declaration should stick across every consumer; use `dumps=` when
you want the targets listed at a single launch.

```python
pl.dump_tag(q)                                    # sticks to later consumers
out, tid = pl.submit(self.attn, q, k, out, dumps=[k])   # this launch only
```

The marks only take effect under **partial** dump (`RunConfig.enable_dump_args == 1`).
They are a no-op when dump is off (`0`), and irrelevant under full dump (`2`), which
captures everything regardless. Under full dump on a large workload the collector — which
drains at roughly 42 MB/s — falls behind, and the AICPU side eventually takes a STARS
op-timeout kill. Partial dump plus tags is the way to keep dump usable there.

### Arrays

`pl.array` is a small on-core array, used for indexed scalar state — a TaskId table, a set
of per-block offsets. Arrays do not cross function boundaries, so they are created rather
than annotated.

| Call | Meaning |
| ---- | ------- |
| `pl.array.create(extent, dtype)` | Allocate |
| `arr[i]` | `array.get_element` |
| `arr[i] = v` | `array.update_element` — functional, rebinds `arr` |

The update is functional: it produces a new array value and rebinds the name. That is
consistent with SSA, and it means an array assignment inside a loop is a carried value
like any other. Under `pl.parallel`, an array carry acts as a barrier — see
[Scopes and Tasks](04-scopes-and-tasks.md).

A common use is collecting TaskIds for a fan-in:

```python
tids = pl.array.create(4, pl.TASK_ID)
for i in pl.range(4):
    _, tid = pl.submit(self.stage, x, out)
    tids[i] = tid
```

### Subscript sugar

The parser rewrites subscripts on `Tensor` and `Tile` values:

| Written | Becomes |
| ------- | ------- |
| `A[0:16, :]` | `pl.slice(A, [16, N], [0, 0])` |
| `A[i, j]` | `pl.tensor.read(A, [i, j])` / `pl.tile.read(A, [i, j])` |
| `A[0:16, 0:32]` | `pl.slice(A, [16, 32], [0, 0])` |
| `dst[i:i+16, j:j+32] = src` | `dst = pl.assemble(dst, src, [i, j])` |

The write form rebinds `dst`, which is incompatible with strict SSA. Under
`@pl.function(strict_ssa=True)` — or any post-SSA context — call `pl.assemble(...)`
explicitly instead.

### Python operators

Standard operators map to IR operations on `Tensor`, `Tile`, and `Scalar` values:

| Python | Operation |
| ------ | --------- |
| `a + b` / `a - b` / `a * b` / `a / b` | `add` / `sub` / `mul` / `div` |
| `a == b` / `a != b` | `eq` / `ne` |
| `a < b` / `a > b` | `lt` / `gt` |

A scalar on either side is detected and dispatched to the scalar-operand form
(`pl.add(a, 1.0)` → `adds`).

### Closure capture

The decorator parses source, so a name from the enclosing Python scope is resolved by the
parser, not captured by a closure at call time. Integer constants and constant arithmetic
fold into the IR. A captured value that cannot be folded is an error at parse time, not a
surprise at run time.

One consequence worth knowing: an expression like `pl.system.available_cluster_count()`
should be written **inline** at the call site rather than bound to a name first. Binding it
compiles and lowers correctly, but the printed IR of the outlined wrapper then references a
variable defined in the caller and cannot be re-parsed.

## Edge Cases

> **Fatal pitfall:** `dumps=` and `pl.dump_tag` are silently inert unless
> `RunConfig.enable_dump_args == 1`. With dump off you get no files and no warning; with
> full dump you get every binding and the collector backs up. Check the setting before
> concluding a tag did not work.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **`static_print` prints nothing** | The function was never parsed | For `@pl.jit`, parsing happens at the first specializing call |
| **`ParserSyntaxError` on `static_assert`** | `msg` is not a string literal, or it was used in an expression | Pass a literal; use it as a standalone statement |
| **`static_assert` did not catch a runtime value** | It is parse-time only | Validate runtime values in host code |
| **Dump produced nothing** | `enable_dump_args` is `0` | Set it to `1` for partial dump |
| **Dump killed the run (STARS op-timeout)** | Full dump (`2`) saturated the collector | Use partial dump plus `dump_tag` / `dumps=` |
| **Array update did not stick** | `arr[i] = v` rebinds; the old name still holds the old value | Use the rebound name, or carry it through the loop |
| **`dst[...] = src` rejected** | `strict_ssa=True` forbids the rebind | Call `pl.assemble(dst, src, [...])` |
| **Printed IR will not re-parse** | A device-geometry query was bound to a name before use | Write the call inline at the use site |

## See Also

- [Control Flow](02-control-flow.md) — carried values, which array updates participate in.
- [Scopes and Tasks](04-scopes-and-tasks.md) — `dumps=` on a submit, and TaskId arrays.
- [Operations](../ops/00-dispatch.md) — where the operators behind the sugar live.
- [Python IR Syntax Specification](../../dev/language/00-python_syntax.md) — the parser's full surface.
- [Runtime DFX](../../dev/03-runtime-dfx.md) — the dump pipeline these marks feed.
