# StampTfreeSplit Pass

## Overview

`StampTfreeSplit` copies each cross-core tpop's `split` (and pipe `id`) onto its
matching `tfree` op, so PTO codegen can read those attributes directly from the
`tfree` op instead of recovering them from a codegen-side lookup table.

A `system.tfree_to_aic` / `system.tfree_to_aiv` carries no split of its own — the
split mode lives on the originating `tile.tpop_from_aic` / `tile.tpop_from_aiv`
call. This pass makes that data explicit on the IR node.

## Position in the pipeline

```text
... -> SplitVectorKernel -> StampTfreeSplit -> NormalizeReturnOrder -> SkewCrossCorePipeline -> ...
```

It runs immediately after `SplitVectorKernel` finalizes the `split` value on
tpops, and before `SkewCrossCorePipeline` clones tpop/tfree pairs into the
software-pipeline prologue/epilogue — so each clone carries the already-stamped
`split`, and the lookup still sees every `tfree` arg as its direct tpop result
var (not a pipelined loop carry).

## Behavior

For every function (AIC and AIV bodies included), the pass builds a map from each
tpop result `Var` to its `{split, id}`, then for each `tfree` whose tile argument
is a known tpop result it stamps `split` (and `id` when the tpop carried one) onto
the `tfree` call:

```text
# before
t = tile.tpop_from_aic(split=1, id=2)
...
system.tfree_to_aic(t)

# after
t = tile.tpop_from_aic(split=1, id=2)
...
system.tfree_to_aic(t, split=1, id=2)
```

If a `tfree` already carries an explicit `id` that disagrees with its tpop's id,
the pass raises (a user error). It also rejects a `tfree` whose direction does not
match its originating tpop (e.g. `tfree_to_aiv` over a `tpop_from_aic` result),
and a `tfree` whose tile has no traceable originating tpop. These are the same
consistency checks codegen used to perform.

## Why a single late pass

Two authoring paths produce tfrees, and both must be covered:

- **Mixed-kernel** (`pl.at(..., split=...)`): tfrees are created by
  `FinalizeTpopTfrees` inside [`ExpandMixedKernel`](19-expand_mixed_kernel.md),
  which only processes InCore functions.
- **Explicit** `@pl.function(type=AIC/AIV)`: the user writes `pl.tfree_to_aic`
  directly; these bypass the finalizer entirely.

Running one pass over every function after split is finalized covers both
uniformly, so neither `FinalizeTpopTfrees` nor the explicit-function lowering
needs split-stamping logic of its own.

## Consumers

PTO codegen for `system.tfree_to_ai{c,v}` reads `split` (and `id`) from the op via
`op->GetKwarg<int>("split", 0)`. There is no codegen-side tpop tracking table.
