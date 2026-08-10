# Choosing a Namespace

Three namespaces expose overlapping operators. The rule for picking one is short; the
reasons it sometimes does not apply are the useful part.

> **Prerequisites:** [Types](../language/00-types.md).

## Concept

The same arithmetic exists at two levels — on whole DDR arrays and on on-chip buffers —
because those are two different machines. `pl.tensor.add` names an operation the compiler
will place and schedule for you. `pl.tile.add` names an operation on a buffer you already
placed.

`pl.*` is neither. It is a **dispatcher**: it inspects the argument types and forwards to
the tensor or tile implementation. It is not a third level, and it adds no semantics of
its own.

That gives the rule:

```text
Writing an operator?
├─ Does an unqualified `pl.` version exist?  → use it
└─ Otherwise → the operator is level-specific; name the level
```

**Default to `pl.*`.** It keeps a kernel readable when it is written at tensor level and
still correct if the surrounding code later descends to tile level.

## Mechanics

### The three namespaces

| Namespace | Operates on | Who chooses placement |
| --------- | ----------- | --------------------- |
| `pl.*` | `Tensor` or `Tile` — dispatched by argument type | Depends on what it dispatched to |
| `pl.tensor.*` | `Tensor` (DDR) | The compiler |
| `pl.tile.*` | `Tile` (on-chip) | You |

Two further namespaces are not levels but families:

| Namespace | Contents |
| --------- | -------- |
| `pl.system.*` | Cross-core transfer, pipes, device geometry queries, synchronization |
| `pl.array.*` | On-core array create / read / update |

```python
c = pl.add(a, b)          # dispatches on the type of a and b
c = pl.mul(a, 2.0)        # scalar rhs detected -> the *s form
t = pl.tile.load(x, [0, 0], [64, 64])
t = pl.tile.adds(t, 1.0)  # tile-specific spelling
```

### Path-specific keyword arguments raise

A few operators take a keyword only one level can honour. A tensor has no layout, so
`matmul` transposes through `a_trans=` / `b_trans=`, while at tile level transposition is a
type property expressed with `pl.tile.transpose_view(...)`. Conversely a tile's scratch
buffer is caller-supplied, where the tensor path has the compiler allocate it.

Passing such a keyword with a non-default value to the wrong path raises `TypeError`,
naming the offending keyword and the level-appropriate alternative — it is not dropped.
Passing the documented default (`b_trans=False`, say) is a no-op and stays accepted.

### When the unified form does not exist

An operator has no `pl.*` spelling when it is meaningful at only one level. Reaching for
the namespace directly is correct in those cases, not a fallback:

| Only at tensor level | Why |
| -------------------- | --- |
| `pl.create_tensor`, `pl.full`, `pl.assemble` | Whole-array creation and placement |
| `pl.dim` | A tensor's runtime dimension |

| Only at tile level | Why |
| ------------------ | --- |
| `pl.tile.load` / `store` / `move` | Movement between memory spaces |
| `pl.tile.transpose_view` | A view over a placed buffer |
| `pl.tile.get_block_idx` / `get_subblock_idx` | The executing block's identity |

Several tile-only operators are re-exported at top level for convenience — `pl.load`,
`pl.store`, `pl.move`, `pl.create_tile`, `pl.matmul_acc` and others are the same functions
as their `pl.tile.*` originals. `pl.load` is not a dispatcher; it is `pl.tile.load` under
a shorter name.

### Scalar-operand forms

Many operators have a companion taking a scalar on the right, named with a trailing `s`:
`adds`, `muls`, `maximums`, `ands`, `shls`. You rarely spell these — passing a Python
number to the unified form selects them:

```python
c = pl.add(a, b)        # tensor/tile + tensor/tile
c = pl.add(a, 1.0)      # -> the scalar-operand form
```

Where an operator takes a scalar in a position the dispatcher cannot infer, the explicit
`pl.tile.*` spelling is the way to say it.

## Edge Cases

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **`AttributeError` on `pl.<name>`** | The operator is level-specific | Qualify it: `pl.tile.<name>` or `pl.tensor.<name>` |
| **A tile operator rejected in a `@pl.jit` body** | Tile work on the control plane | Move it into `pl.at(level=...)` or a `@pl.jit.incore` function |
| **A tensor operator rejected inside an InCore function** | Tensor creation is control-plane work | Allocate on the control plane; pass in a `pl.Out[...]` parameter |
| **The dispatcher picked the level you did not want** | It dispatches on argument type | Pass the operand at the level you meant, or name the namespace |
| **An operator exists but the backend rejects it** | Not all operators are supported on every backend | Check [PTOAS Operator Status](../../dev/ptoas-op-status.md) |

## See Also

- [Catalog](01-catalog.md) — what exists in each namespace.
- [Memory and Data Movement](../language/03-memory.md) — why the tile-only movement operators are shaped as they are.
- [Programming Model](../03-programming-model.md#quickstart-the-three-levels-in-one-program) — the same computation written at both levels.
- [ConvertTensorToTileOps](../../dev/passes/11-convert_tensor_to_tile_ops.md) — the pass that turns the first into the second.
