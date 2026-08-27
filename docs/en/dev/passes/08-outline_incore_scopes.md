# OutlineIncoreScopes Pass

Outlines InCore scopes into separate functions.

## Overview

This pass transforms `InCoreScopeStmt` nodes into separate `Function(InCore)` definitions and replaces the scope with a Call to the outlined function.

**Requirements**:

- Input IR must be in SSA form (run ConvertToSSA first); SSAForm is preserved (produced) by this pass
- Processes Opaque and Orchestration functions (InCore functions are left
  unchanged). An Orchestration function carries InCore scopes when the parser
  desugars a high-level construct such as `for i in pl.spmd(...)`; an Opaque
  parent that outlines at least one scope is promoted to Orchestration

**When to use**: Run after ConvertToSSA when you need to extract InCore computation regions into separate callable functions.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::OutlineIncoreScopes()` | `passes.outline_incore_scopes()` | Program-level |

**Factory function**:

```cpp
Pass OutlineIncoreScopes();
```

**Python usage**:

```python
from pypto.pypto_core import passes

outline_pass = passes.outline_incore_scopes()
program_outlined = outline_pass(program)
```

## Algorithm

1. **Scan for InCore Scopes**: Find all `InCoreScopeStmt` nodes in Opaque and
   Orchestration functions
2. **Analyze Inputs**: Collect the scope's *live-in* set — variables the body reads
   before it (re)defines them, so their incoming value comes from the caller
3. **Analyze Outputs**: Determine internal definitions used after scope (variables defined inside, used outside)
4. **Create Function**: Extract scope body into new `Function(scope_type=InCore)` with:
   - Parameters = input variables
   - Returns = output variables
   - Body = scope body
5. **Replace Scope**: Replace `InCoreScopeStmt` with:
   - Call to outlined function with input arguments
   - AssignStmt for each output variable
6. **Add to Program**: Add outlined function to program's function list
7. **Promote the parent**: an Opaque parent that outlined at least one scope becomes
   `Orchestration` — and its param dyn-dim reads are folded first (below)

**Param dyn-dim reads fold on promotion**: a tensor's declared extent *is* its
runtime extent, so `pl.tensor.dim(a, 0)` on a param whose axis is a `pl.dynamic`
symbol mints a *second* IR name for one quantity, and shapes built from the copy
no longer compare equal to shapes built from the symbol. The DSL parser folds
that read onto the symbol (`ASTParser._fold_tensor_dim`), but only in an
Orchestration body — that is where Orchestration codegen defines the symbol from
the param's task-arg descriptor, and where the fold is therefore sound. A body
written as `Opaque` keeps the read, so this pass folds it at the moment it
promotes the function, *before* outlining:

```python
# Opaque parent, as written                # after promotion
m = pl.tensor.dim(a, 0)                    # (binding folded away)
with pl.spmd(m // 16):                     with pl.spmd(M_DYN // 16):
    ...                                        ...
```

Folding before the outliner runs means the promoted body reaches it in the same
shape the parser hands an already-Orchestration function, so both paths produce
identical IR. Without it the pass emits IR that no longer parses back to itself
(the printed `tensor.dim` binding vanishes on reparse), breaking print→parse
round-trip verification. Reads the parser would not fold are left alone: a
constant extent, a runtime axis, or a symbol the signature does not declare.

**Live-in, not `uses \ defs`**: the input set is computed flow-sensitively
(`UpwardExposedUseCollector`). A plain set difference is wrong for a captured
tensor that the body reads *and* rebinds under the same name — the shape the
parser emits for a `pl.Out` param before `ConvertToSSA` splits it:

```python
with pl.at(level=pl.Level.CORE_GROUP):
    c = pl.store(t, [0, 0], c)   # one Var: read as the store target, then rebound
```

`c` is in both `var_uses` and `var_defs`, so the difference drops it from the
parameter list and leaves the use dangling in the outlined body. Treating it as
live-in makes it a write parameter, and the `tile.store` result binds a
distinct Var (`c__store`) so the outlined body never rebinds its own parameter:

```python
def main_incore_0(a: Tensor[[128, 128], FP32], c: Out[Tensor[[128, 128], FP32]]):
    c__store = pl.tile.store(..., c)
    return c                       # the param — store writes through it in place
```

On SSA input — the pass's declared `IRProperty::SSAForm` precondition — live-in
and `uses \ defs` are identical, so this only changes behaviour for IR that
reaches the pass without that precondition holding. A captured variable rebound
by anything *other* than a `tile.store` cannot be expressed without real SSA
construction and is rejected with an internal error naming `ConvertToSSA`.

The distinct result Var is the InCore/Cluster/Spmd outcome. Hierarchy scopes
skip the store-target export entirely (the buffer is already visible to the
caller through its write parameter), so their body keeps the original rebind —
the capture still becomes a parameter, which is the part that was broken.

**Which operators write** is not decided here. Each operator declares the
effect it has on every argument (`set_arg_effect`, see
[Operators](../ir/05-operators.md#argument-effects)), and `InferParamDirections`
reads that. Before, this pass recognised exactly two writers — `tile.store` and
`tensor.assemble` — so a scope whose only write to a captured tensor went
through `tensor.write`, `tensor.expand_clone`, `pld.system.notify`,
`pld.tile.put` or any other writer left that tensor looking untouched: the
parameter stayed `In`, the caller got no dependency on the write, and the two
passes that later re-derive directions disagreed with this one about the same
call.

**Write direction: `Out` unless the body reads**: a captured tensor the scope
writes is lifted off `In` by `InferParamDirections`. Which write direction it
earns is decided by whether the body also *reads* it. An argument the operator
declares `Write` updates a sub-region of the destination **in place**: the
untouched region is neither loaded nor re-stored, so appearing in that
destination slot moves no data into the scope and is not a read. A parameter
whose only uses are such slots is therefore `Out`; anything else — feeding a
`tensor.slice`, a compute op, or a callee's `In`/`InOut` param — makes it
`InOut`. An argument declared `ReadWrite` stays on the read path, which is how
an atomic store or assemble (`out += x` reads the accumulator) and an
`AtomicAdd` notify keep their destination `InOut` while the plain forms do not —
one rule, stated per operator, rather than a carve-out per pass. Under SSA the
post-write state binds to a fresh Var, so reading *that* alias counts too: the
alias names the same buffer, and a read over a region the scope never wrote does
need the incoming contents. Unrecognised uses count as reads, so the inference
can only err towards `InOut`.

Two keys are excluded: `dump_vars` and `arg_direction_overrides_vars` name a
tensor as bookkeeping (dump marking, `NoDep` opt-out) rather than accessing it.

Each source of evidence — the read scan, the store-target set, the body's
declared writes, and each inner callee's declared slot — is a *lower* bound on
the accesses, so no source may overwrite another. The body-side sources merge
along `In < Out < InOut`.

The callee slots do **not**, and this is the one place the ordering does not
apply. `In` is the seeded *no evidence yet* floor, so it cannot also stand for
"somebody read this" — reading it that way would promote every write-only
capture to `InOut`, the false read of issue #2415. Folding the callee directions
one call at a time therefore lost information: a capture handed to one callee's
`In` slot and another's `Out` slot merged to `In`, then to `Out`, dropping the
read. The callees' evidence is instead accumulated as two independent flags —
`In`/`InOut` marks a read, `Out`/`InOut` marks a write — and the direction is
derived once at the end, so such a capture comes out `InOut` while a capture
only ever written still comes out `Out`.

The read half of that verdict draws on the body scan as well as the callee
slots, because a capture can be read by the body and overwritten by a callee:

```python
with pl.cluster():
    value = pl.load(shared, [0, 0], [16, 128])  # this body reads it
    self.overwrite(shared)                      # and a callee overwrites it
```

`shared` is `InOut`. Consulting only the callee slots would call it `Out` and
tell the wrapper it need not stage the very contents `pl.load` consumes. The
body scan can be trusted here because it skips the arguments a callee declares
`Out` — the user-function counterpart of a builtin's declared write slot — so
handing a capture to a write-only slot is not itself counted as a read.

That skip covers `pl.submit` as well as a plain call. The base visitor does not
forward `Submit` to the `Call` handler, so a launch needs its own rule or every
launch argument counts as a read and a capture handed only to an `Out` slot
comes back `InOut`. `Submit` maps `args_[i]` to `params_[i]` over a prefix
(`args_.size() <= params_.size()`; the omitted tail is runtime-allocated), and
the trailing `CommCtx` params that would break that identity are materialised by
pass 43, long after any outliner runs. Its `deps_` are always read — they are
TaskId values the launch consumes, never a write destination.

**Hierarchy scopes are an exception.** `OutlineScope` deliberately leaves
`store_output_set` empty for `ScopeKind::Hierarchy` (the buffer is already
visible to the caller without an explicit returned output), so a `tile.store`
target captured by a Hierarchy scope never reaches the rule above and its
parameter stays `In`. That predates the write-direction rule and is unchanged
here.

Claiming `InOut` for a parameter the body never reads is not a safe
approximation. The direction propagates into
`DistributedCodegen::EmitCallToWorker`, which tags each per-rank chip dispatch
argument from the *callee's* direction, so a false `InOut` turns disjoint
per-rank slices of one `pl.Out` tensor into a cross-rank write dependency
(issue #2415). Ordering a write-only parameter genuinely needs is not lost:
[`DeriveCallDirections`](38-derive_call_directions.md) re-derives the
*call-site* direction and promotes a callee `Out` back to `InOut` under a
sequential ancestor, behind a prior writer of the same root, or when the root is
an enclosing `InOut` parameter.

**Param-explicit returns**: the outlined function returns its
own parameters, not SSA result vars, whenever a tensor output writes through
a parameter — store-target outputs return the param directly, other outputs
are traced via the shared `return_lineage` utility. Kernel-allocated outputs
keep their SSA value. This makes the return→param mapping a pointer-identity
lookup for orchestration codegen (`ReturnParamsExplicit` invariant).

**Naming**:

- Default: `{original_func}_incore_{counter}` (e.g., `main_incore_0`, `main_incore_1`)
- User-provided: when `InCoreScopeStmt.name_hint` is non-empty, that name is used directly
  - `with pl.at(level=pl.Level.CORE_GROUP, name_hint="fused_add"):` → function named `fused_add`

**Name collisions** (`name_hint` is a *hint*, not a unique identifier — outlined
functions share one program-wide namespace, so collisions are resolved
automatically):

- **In-function** — two scopes in the same function sharing a `name_hint` get a
  numeric suffix: `my_kernel`, `my_kernel_0`.
- **Cross-function** — two *different* functions outlining scopes with the same
  `name_hint` (typically a reused `@pl.jit.inline` helper composed into a host
  program) are disambiguated by namespacing the collision under the originating
  function. The first function keeps the bare hint (stable, matching its
  standalone compilation); the later one is prefixed:
  - `single_a` → `dup_scope`, `single_b` → `single_b_dup_scope`

  This lets independently-runnable child kernels be composed into one
  `@pl.jit.host` program without manually renaming shared helper internals. The
  same rule applies to the sibling `OutlineHierarchyScopes` and
  `OutlineClusterScopes` passes (which share the outlining utility).

**Cache-policy declarations become param indices**: a
`pl.set_cache_policy(t, pl.CachePolicy.BYPASS)` statement in the scope body is
hoisted by the parser onto the scope's `cache_policy_vars` attr
(`std::vector<std::pair<VarPtr, int>>`, keyed by Var identity). This pass
resolves each Var through the same captured-input index map the `no_dep_args`
translation uses and re-emits the list as the outlined function's `cache_policy`
attr — `std::vector<std::pair<int32_t, int>>` (param index, `CachePolicy` as
int), sorted by index so declaration order and capture order cannot change the
IR. The scope attr is **consumed here and never propagated**: from this point the
function attr is the single carrier, until
[`ConvertTensorToTileOps`](10-convert_tensor_to_tile_ops.md) turns it into a
`cache` kwarg on each `tile.load` and erases it. Param indices are only valid
across that window — later passes both append to
([`InjectGMPipeBuffer`](23-inject_gm_pipe_buffer.md),
[`MaterializeDistTensorCtx`](44-materialize_dist_tensor_ctx.md)) and prepend onto
([`MaterializeValidShapeSymbols`](49-materialize_valid_shape_symbols.md)) param
lists. Two user errors are rejected here with `CHECK_SPAN`: a declaration naming
a tensor the scope body does not capture (it is neither read nor written, so no parameter
carries the policy), and `BYPASS` on a parameter `InferParamDirections` resolved
to `Out` / `InOut` (a bypassing read of bytes the same kernel writes is a
coherency bug). The translation lives in the shared outlining utility, so the
sibling `OutlineHierarchyScopes` path stamps the attr the same way. See
[GM Cache-Access Policy](../language/05-cache-policy.md).

## Example

### Basic Outlining

**Before**:

```python
@pl.program
class Before:
    @pl.function  # Opaque function
    def main(self, x: Tensor[[64], FP32]) -> Tensor[[64], FP32]:
        y = x + 1

        with pl.at(level=pl.Level.CORE_GROUP):  # InCore scope
            tile = pl.load(y, [0], [64])
            tile_sq = pl.mul(tile, tile)
            result_tile = tile_sq + 1
            result = pl.store(result_tile, [0], x)

        z = result + 2
        return z
```

**After**:

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.Orchestration)  # promoted from Opaque
    def main(self, x: Tensor[[64], FP32]) -> Tensor[[64], FP32]:
        y = x + 1

        # Scope replaced with call + assignments
        result = self.main_incore_0(y, x)  # Call outlined function

        z = result + 2
        return z

    @pl.function(scope_type=InCore)  # Outlined InCore function
    def main_incore_0(self, y: Tensor[[64], FP32], x: Tensor[[64], FP32]) -> Tensor[[64], FP32]:
        # Scope body moved here
        tile = pl.load(y, [0], [64])
        tile_sq = pl.mul(tile, tile)
        result_tile = tile_sq + 1
        result = pl.store(result_tile, [0], x)
        return x  # store target: returns the param, not `result`
```

### Multiple Outputs

**Before**:

```python
with pl.at(level=pl.Level.CORE_GROUP):
    a_tile = pl.load(a, [0], [64])
    b_tile = pl.load(b, [0], [64])
    c_tile = pl.add(a_tile, b_tile)
    out_a = pl.store(c_tile, [0], out)
    out_b = pl.mul(c_tile, 2.0)
# Both out_a and out_b used after scope
x = out_a + out_b
```

**After**:

```python
out_a, out_b = self.main_incore_0(a, b, out)  # Multiple outputs
x = out_a + out_b

# Outlined function:
def main_incore_0(self, a, b, out):
    a_tile = pl.load(a, [0], [64])
    b_tile = pl.load(b, [0], [64])
    c_tile = pl.add(a_tile, b_tile)
    out_a = pl.store(c_tile, [0], out)
    out_b = pl.mul(c_tile, 2.0)
    return (out, out_b)  # out_a → param `out`; out_b is kernel-local, kept as-is
```

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass OutlineIncoreScopes();
```

**Implementation**: `src/ir/transforms/outline_incore_scopes.cpp`

- Uses SSA analysis to determine inputs/outputs
- Creates new Function nodes with InCore scope type
- Replaces InCoreScopeStmt with Call + AssignStmt
- Manages function naming and counters

**Python binding**: `python/bindings/modules/passes.cpp`

```cpp
passes.def("outline_incore_scopes", &pass::OutlineIncoreScopes, "Outline InCore scopes");
```

**Tests**: `tests/ut/ir/transforms/test_outline_incore_scopes.py`

- Tests basic scope outlining
- Tests input/output analysis
- Tests multiple scopes in same function
- Tests nested scopes
- Tests SSA preservation

## Requirements

**SSA form required**: The pass relies on SSA properties:

- Single assignment ensures clear input/output analysis
- No variable shadowing simplifies scope analysis
- YieldStmt in control flow handled correctly

**Run ConvertToSSA first** if IR is not in SSA form.

**Mutually exclusive AIV-split mechanisms**: a function-level AUTO split
(`optimizations=[pl.split(mode)]`, carried as the scope's own `split_`) and
explicit `pl.split_aiv` regions (`SplitAivScopeStmt`) cannot coexist on one
scope (the outliner bridges a single region's mode into a function-level
representative `split`, which would silently collide with the user's
`pl.split`). See [`LowerAutoVectorSplit`](21-lower_auto_vector_split.md) for how
the surviving mechanism is lowered.

**Any** `pl.split(...)` is rejected, `SplitMode.NONE` included (RFC #1820). NONE
carries no split of its own, but writing it on a scope that also holds regions
still reads as "auto and manual split mixed on one scope". The exemption that
used to allow it existed only because the cross-core slot count had no carrier
other than `pl.split(..., slot_num=N)`; it now has one —
`optimizations=[pl.cross_core_slot(slot_num=N)]`, which is orthogonal to
splitting and coexists with regions freely.

**Where the rejection fires.** `InCoreScopeStmt::split_` has a single encoding of
"no split" (`SplitMode::None`), so a literal `pl.split(pl.SplitMode.NONE)` is
invisible to this pass — it looks identical to writing no `pl.split` at all. The
**parser** therefore owns the rejection: it is the only layer that sees the
literal the user wrote, and it rejects every mode, NONE included. This pass keeps
the check for `split_ != SplitMode::None` as the backstop for IR that never went
through the parser (deserialized `.pto`, programmatically built scopes).

The three states read distinctly:

| Annotation | Meaning |
| ---------- | ------- |
| `optimizations=[pl.split(MODE)]` | AUTO split — the compiler partitions the vector work |
| `for aiv_id in pl.split_aiv(2, mode=...)` | Manual split — the author partitions it per region |
| `optimizations=[pl.cross_core_slot(slot_num=N)]` | Neither — just sizes the cross-core pipe |

**The function-level `split` attr has one encoding of "no split": an absent key.**
When the outlined body holds `pl.split_aiv` regions, this pass bridges their mode
onto the function only when all regions agree *and* that mode is a real split:

| Regions in the body | Attrs stamped on the outlined function |
| ------------------- | -------------------------------------- |
| All `mode=UP_DOWN` (or all `LEFT_RIGHT`) | `{"split_aiv": True, "split": pl.SplitMode.UP_DOWN}` |
| All `mode=NONE` | `{"split_aiv": True}` — no `split` key |
| Differing modes | `{"split_aiv": True}` — no representative mode |

`Function::GetSplitMode()` maps a stored `0` to `nullopt` exactly as it does an
absent key, so a `split=SplitMode.NONE` entry was invisible to every consumer —
and the parser drops it, which made print → parse lossy (`Kwargs size mismatch`).
The authoritative per-region mode always rides `SplitAivScopeStmt::split_`, which
[`LowerAutoVectorSplit`](21-lower_auto_vector_split.md) consumes. The printer
applies the same rule as a backstop: it omits a `split` attr of `SplitMode.NONE`
so IR that bypassed this pass (a pre-existing `.pto` blob, a programmatically
built `Function`) still prints in the canonical, re-parsable form.

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | SSAForm |
| Produced | SSAForm, SplitIncoreOrch, AivSplitValid |
| Invalidated | — |

`AivSplitValid` opens here. The pass preserves the first-class `SplitAivScopeStmt` regions inside
each outlined InCore function, so the structural region verifier can run from this point until
[`LowerAutoVectorSplit`](21-lower_auto_vector_split.md) erases the node and invalidates the
property. `ConvertTensorToTileOps` and `InferTileMemorySpace` re-verify it in between, once the
boundary's memory side becomes observable.
