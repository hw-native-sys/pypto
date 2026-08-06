# OutlineIncoreScopes Pass

Outlines InCore scopes into separate functions.

## Overview

This pass transforms `InCoreScopeStmt` nodes into separate `Function(InCore)` definitions and replaces the scope with a Call to the outlined function.

**Requirements**:

- Input IR must be in SSA form (run ConvertToSSA first); SSAForm is preserved (produced) by this pass
- Only processes Opaque functions (InCore functions are left unchanged)

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

1. **Scan for InCore Scopes**: Find all `InCoreScopeStmt` nodes in Opaque functions
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

**Live-in, not `uses \ defs`**: the input set is computed flow-sensitively
(`UpwardExposedUseCollector`). A plain set difference is wrong for a captured
tensor that the body reads *and* rebinds under the same name — the shape the
parser emits for a `pl.Out` param before `ConvertToSSA` splits it:

```python
with pl.at(level=pl.Level.CORE_GROUP):
    c = pl.store(t, [0, 0], c)   # one Var: read as the store target, then rebound
```

`c` is in both `var_uses` and `var_defs`, so the difference drops it from the
parameter list and leaves the read dangling in the outlined body. Treating it as
live-in makes it an `InOut` parameter, and the `tile.store` result binds a
distinct Var (`c__store`) so the outlined body never rebinds its own parameter:

```python
def main_incore_0(a: Tensor[[128, 128], FP32], c: InOut[Tensor[[128, 128], FP32]]):
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
caller through its InOut parameter), so their body keeps the original rebind —
the capture still becomes a parameter, which is the part that was broken.

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
    @pl.function  # Opaque function
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
`pl.split`). See [`LowerAutoVectorSplit`](20-lower_auto_vector_split.md) for how
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
[`LowerAutoVectorSplit`](20-lower_auto_vector_split.md) consumes. The printer
applies the same rule as a backstop: it omits a `split` attr of `SplitMode.NONE`
so IR that bypassed this pass (a pre-existing `.pto` blob, a programmatically
built `Function`) still prints in the canonical, re-parsable form.
