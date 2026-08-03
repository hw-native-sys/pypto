# Language Syntax

What ordinary Python syntax means inside a kernel: subscripts, operators, and names taken
from the enclosing scope.

> **Prerequisites:** [Types](00-types.md).

## Concept

A decorator parses your source; it never runs it as Python. So the familiar syntax in a
kernel body is **rewritten into IR operations**, not executed:

- A subscript is not an index operation — it becomes a slice, a read, or an assemble.
- An operator is not Python's `+` — it becomes an IR op chosen from the operand types.
- A name from the enclosing scope is resolved by the parser at parse time, not captured by
  a closure at call time.

The practical consequence is that these constructs fail — or refuse to compile — at parse
time with a line number, rather than surprising you at run time.

## Quickstart: the same slice, two ways

```python
import pypto.language as pl

@pl.jit.incore
def head(x: pl.Tensor[[128, 64], pl.FP32],
         out: pl.Out[pl.Tensor[[16, 64], pl.FP32]]):
    top = x[0:16, :]              # sugar for pl.slice(x, [16, 64], [0, 0])
    out = pl.mul(top, 2.0)        # sugar-free: an explicit operator call
    return out
```

Both lines are parsed into IR calls. The first shows the rewrite; the second is what the
rewrite produces everywhere else.

## Mechanics

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

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **`dst[...] = src` rejected** | `strict_ssa=True` forbids the rebind | Call `pl.assemble(dst, src, [...])` |
| **A subscript produced a read where a slice was wanted** | `A[i, j]` reads one element; `A[i:i+1, :]` slices | Use slice syntax when you want a region |
| **Printed IR will not re-parse** | A device-geometry query was bound to a name before use | Write the call inline at the use site |
| **A captured value is rejected at parse time** | It cannot be folded into the IR | Pass it as a parameter instead of capturing it |

## See Also

- [Types](00-types.md) — what the operands of this syntax are, including `pl.Array`.
- [Compile-Time Directives](05-directives.md) — the other parse-time constructs.
- [Operations](../ops/00-dispatch.md) — where the operators behind the sugar live.
- [Python IR Syntax Specification](../../dev/language/00-python_syntax.md) — the parser's full surface.
