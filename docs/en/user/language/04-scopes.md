# Scopes and Placement

Where work runs: marking a region as device work, grouping co-scheduled cores, and
spreading a kernel across blocks.

> **Prerequisites:** [Functions and Programs](01-functions.md) and
> [Programming Model § execution model](../03-programming-model.md#the-execution-model).

## Concept

Placement answers one question: **which piece of hardware runs this code.**

There are four constructs, all written with `with` (or `for`), and they compose:

| Construct | Places work on |
| --------- | -------------- |
| `pl.at` | A core group — marks a region as device work |
| `pl.cluster` | One physical cluster — co-schedules a Cube and a Vector kernel |
| `pl.spmd` | `n` blocks — the same kernel, once per block |
| `pl.split_aiv` | Two AIV lanes — splits one region across both |

The alternative to `pl.at` is writing a separate `@pl.jit.incore` function and calling it.
They produce the same thing: `pl.at` is outlined into exactly such a function during
compilation. Use the scope when the region is short and belongs where it is written; use a
separate function when it deserves a name or is called from more than one place.

Placement is not the same as **ordering** — what must finish before a task starts. The
runtime derives that from the parameter directions in [Types](00-types.md) and the buffers
each task touches. That machinery, and the interfaces for steering it by hand, get their
own chapter; this page is only about where code lands.

## Quickstart: mark a region as device work

```python
import pypto.language as pl

@pl.jit
def scale(
    x: pl.Tensor[[256, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        out = pl.mul(x, 2.0)
    return out
```

| Element | What it does |
| ------- | ------------ |
| `@pl.jit` | The entry point — control plane, which cannot hold operators itself |
| `with pl.at(level=pl.Level.CORE_GROUP)` | Marks the region as device work, giving the operators somewhere legal to live |
| `pl.mul(x, 2.0)` | Runs on the core group |

Without the `pl.at`, this kernel fails to compile with `Misplaced tensor op ... should be
inside InCore block` — see [Functions and Programs](01-functions.md).

## Mechanics

### `pl.at`

`level=` picks the hierarchy level. `pl.Level.CORE_GROUP` is the one that produces an
InCore scope, and the region becomes its own kernel function during compilation.

Two optional keywords shape that outlined kernel:

| Keyword | Meaning |
| ------- | ------- |
| `optimizations=[pl.split(mode)]` | Cross-core split mode for the outlined kernel |
| `optimizations=[pl.cross_core_slot(slot_num=N)]` | Ring depth of the automatic cross-core pipeline |
| `name_hint="..."` | Name for the outlined function |

Entries in `optimizations=` must be written inline at the call site — the parser reads the
AST, so a list built up in a variable is not accepted. `pl.split` and `pl.cross_core_slot`
are orthogonal and combine freely: one splits the work, the other sizes the channel.

```python
with pl.at(level=pl.Level.CORE_GROUP,
           optimizations=[pl.split(pl.SplitMode.UP_DOWN),
                          pl.cross_core_slot(slot_num=4)]):
    ...
```

Omitting `cross_core_slot` keeps the default ring depth: 8 slots when one direction is
active, 4 per direction when both are.

### SPMD

`pl.spmd(n)` runs the same kernel on `n` blocks. Two forms, differing in whether the body
reads the block index:

```python
# Dispatch form — the body launches a kernel defined elsewhere.
with pl.spmd(4):
    out = self.kernel(a, b, out)

# Loop form — the body is auto-outlined and `i` binds the block index.
for i in pl.spmd(4):
    off = i * 128
    out = pl.store(pl.add(pl.load(a, [off, 0], [128, 128]),
                          pl.load(b, [off, 0], [128, 128])), [off, 0], out)
```

A `with pl.spmd(n):` body that neither reads the block index nor dispatches a kernel is
rejected — every block would be doing identical work.

When a hard `pl.system.syncall` is involved, size the launch from the device rather than
from a literal: pass `pl.system.available_cluster_count()` (mixed or cube-only kernels) or
`pl.system.available_aiv_count()` (vector-only), written inline at the call site.

### Clusters and AIV lanes

`with pl.cluster():` groups AIC and AIV kernels so they are co-scheduled on the same
physical cluster, producing a `Group` function.

`for aiv_id in pl.split_aiv(2, mode=...):` splits one region across the two AIV lanes. It
belongs to mixed-kernel programming — AIC and AIV cooperating inside one function — which
the tutorials chapter covers end to end.

## Edge Cases

> **Fatal pitfall:** `pl.spmd` is an assertion, not a request. You are telling the compiler
> the blocks are independent. If they are not, the result is a race — not a diagnostic.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **`Misplaced tensor op ... should be inside InCore block`** | Operators sit directly in the `@pl.jit` body | Wrap them in `with pl.at(level=pl.Level.CORE_GROUP):` |
| **`with pl.spmd(n):` body rejected** | It neither reads the block index nor dispatches a kernel | Read `pl.tile.get_block_idx()`, or call a kernel |
| **`optimizations=` rejected** | Built up in a variable — the parser reads the AST | Write the list inline at the call site |
| **Printed IR cannot be reparsed** | A device-size query was bound to a name before use | Write the call inline where it is used |

## See Also

- [Functions and Programs](01-functions.md) — the alternative to `pl.at`: a separate `@pl.jit.incore` function.
- [Control Flow](02-control-flow.md) — the loops these scopes sit inside.
- [Memory and Data Movement](03-memory.md) — what the placed code does with buffers.
- [OutlineIncoreScopes](../../dev/passes/09-outline_incore_scopes.md) — how `pl.at` becomes a function.
- [ExpandMixedKernel](../../dev/passes/22-expand_mixed_kernel.md) — what `pl.split` drives.
