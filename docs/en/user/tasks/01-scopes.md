# Runtime Scopes

The boundary the dependency inference runs inside, and the switch that turns it off.

> **Prerequisites:** [The dependency model](00-model.md).

## Concept

A **runtime scope** (`SIMPLER_SCOPE`) is two things at once:

- the region the OverlapMap tracks dependencies within, and
- a heap tier, so nested scopes reclaim their memory independently.

The runtime provides an implicit top-level scope, which is why you can write whole programs
without ever mentioning one. **Writing scopes is tuning and control, never a correctness
requirement.**

By default the compiler places AUTO scopes for you — around a function body, and around
each `for` and `if` body. You take over only when you want the second thing a scope
carries: its **mode**.

| Mode | What it means |
| ---- | ------------- |
| `pl.scope()` / `ScopeMode.AUTO` | OverlapMap auto dependency tracking is on |
| `pl.scope(mode=pl.ScopeMode.MANUAL)`, alias `pl.manual_scope()` | Auto tracking is off — every edge is yours to declare |

`manual_scope` is the coarsest of the runtime's opt-outs. Before reaching for it, note that
you usually do not have to: `deps=` already works in an auto scope, so patching one missing
edge does not require giving up the inference for a whole region. Finer-grained opt-outs
are in [Refining the graph](03-tuning.md).

## Quickstart: taking over the edges for one region

```python
with pl.manual_scope():
    scratch, tid = pl.submit(self.stage1, x, scratch)
    out, _       = pl.submit(self.stage2, scratch, out, deps=[tid])
```

Inside this block the runtime skips the OverlapMap lookup and insert for every submit, so
the `scratch` overlap that would have been inferred is *not* — the `deps=[tid]` is now the
only thing ordering the two stages. Drop it and they may overlap.

> This is a fragment: `self.stage1` / `self.stage2` are methods of the enclosing
> `@pl.program`, and the block sits in an Orchestration function body.

## Mechanics

### Where a scope may appear

| Rule | Detail |
| ---- | ------ |
| Orchestration only | A scope belongs to the control plane; it is not valid in an InCore function |
| `mode=AUTO` needs `auto_scope=False` | Under the default `@pl.function(auto_scope=True)` the compiler owns AUTO placement, so writing one yourself is rejected |
| `mode=MANUAL` is always allowed | It is a dependency-semantics choice, not ring tuning |
| AUTO may not nest inside MANUAL | The runtime forbids it |
| `manual_scope` may not nest inside `manual_scope` | The runtime forbids it |

### Which decorators accept `auto_scope=False`

`@pl.jit`, `@pl.jit.host` and `@pl.jit.inline` accept it. `.incore` and `.opaque` reject
it — they are outlined into standalone kernels, so there is no orchestration body for a
scope to live in. An inline body is spliced into its caller, so a scope written inside one
lands in the caller.

### What you give up in a manual scope

Everything the OverlapMap would have inferred, for every submit in the region — including
the edges you were not thinking about. That is the point of the construct, and also its
risk: a missing `deps=` in a manual scope is not a diagnostic, it is a race.

Reach for it when you genuinely want to own the graph — a hand-tuned pipeline whose shape
you already know — rather than as a way to fix one inference you disagreed with.

## Edge Cases

> **Fatal pitfall:** `manual_scope` does not warn about edges you forgot. The inference
> that would have caught the omission is exactly what you switched off.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **Scope rejected in a kernel** | Scopes are Orchestration-only | Move it to the orchestration function that dispatches the kernel |
| **`mode=AUTO` rejected** | The default `auto_scope=True` reserves AUTO placement for the compiler | Set `@pl.function(auto_scope=False)`, or use MANUAL |
| **`auto_scope=False` rejected** | Used on `.incore` / `.opaque` | Put it on the entry or an `.inline` helper |
| **Nested scope rejected** | AUTO inside MANUAL, or `manual_scope` inside `manual_scope` | Flatten — the runtime forbids both |
| **A race appeared after adding `manual_scope`** | Edges that used to be inferred are now absent | Declare them with `deps=`, or drop the manual scope and patch the one edge you needed |

## See Also

- [The dependency model](00-model.md) — what the tracking infers when it is on.
- [Declaring an edge](02-submit.md) — `deps=`, which works in either mode.
- [Refining the graph](03-tuning.md) — the finer-grained opt-outs that do not cost a whole region.
- [MaterializeRuntimeScopes](../../dev/passes/46-materialize_runtime_scopes.md) — how AUTO scopes are placed.
