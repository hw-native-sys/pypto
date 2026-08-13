# Compile-Time Directives

Statements that run while the decorator parses your source, and then vanish from the IR.

> **Prerequisites:** [Functions and Programs](01-functions.md).

## Concept

A decorator parses your function's source rather than executing it. Three constructs live
in that parse step and nowhere else:

`pl.static_print` and `pl.static_assert` run **at parse time** and leave no trace in the
IR. `pl.const` is a typed literal — a way to pin a constant's dtype instead of accepting
the one inferred from the Python literal.

Knowing that these belong to parse time is the difference between a debugging aid and a
puzzle. A `static_print` that shows nothing is not broken — the function was never parsed.

## Quickstart: seeing what the parser sees

```python
import pypto.language as pl

@pl.jit.incore
def probe(x: pl.Tensor[[64, 128], pl.FP32],
          out: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
    pl.static_print("x =", x)                      # prints at parse time
    pl.static_assert(x.shape[1] == 128, "expected 128 columns")
    out = pl.assemble(out, pl.mul(x, 2.0), [0, 0])
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

Use `static_print` to inspect what the parser inferred — the type and shape it gave a
value — which is often faster than reading printed IR when you only need one fact.

### Typed constants

`pl.const(value, dtype)` builds a constant with an explicit dtype rather than the default
one inferred from the literal. It exists so the printer can round-trip non-default
constant types, and it is what you want when a literal's width matters:

```python
step = pl.const(1, pl.INT32)
```

## Edge Cases

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **`static_print` prints nothing** | The function was never parsed | For `@pl.jit`, parsing happens at the first specializing call |
| **`ParserSyntaxError` on `static_assert`** | `msg` is not a string literal, or it was used in an expression | Pass a literal; use it as a standalone statement |
| **`static_assert` did not catch a runtime value** | It is parse-time only | Validate runtime values in host code |
| **A constant came out with the wrong width** | The dtype was inferred from the Python literal | Pin it with `pl.const(value, dtype)` |

## See Also

- [Language Syntax](06-syntax.md) — the other parse-time behaviour: subscript sugar, operators, closure capture.
- [Functions and Programs](01-functions.md) — when each decorator parses.
- [Python IR Syntax Specification](../../dev/language/00-python_syntax.md) — the parser's full surface.
