# Operations

Which namespace an operator lives in, how to choose between them, and what the operator
surface contains.

> **Prerequisites:** [Types](../language/00-types.md).

## What this chapter is

Two pages, deliberately thin:

| Page | What it covers |
| ---- | -------------- |
| [Choosing a Namespace](00-dispatch.md) | The rule for picking `pl.*` versus `pl.tensor.*` versus `pl.tile.*`, and when the unified form cannot be used |
| [Catalog](01-catalog.md) | Every operator family, one line each, with the namespace it is reachable from |

**Signatures are not duplicated here.** Nearly every operator in `pl.__all__` carries a
docstring, and the docstring is the reference — a hand-maintained signature table would
drift from it within a release. The catalog exists to answer "what is there, and what is
it called", not "what are its arguments".

For "how do I use this operator to do something useful", the answer is a worked example,
not a table. The tutorials chapter is not written yet; until it lands, `examples/beginner/`
and `examples/intermediate/` in the repository are the closest thing.

## See Also

- [Language Guide](../language/index.md) — the constructs these operators appear inside.
- [Memory and Data Movement](../language/03-memory.md) — the movement operators in context.
- [IR Operators](../../dev/ir/05-operators.md) — the operator registry these names resolve to.
- [PTOAS Operator Status](../../dev/ptoas-op-status.md) — backend support per operator.
