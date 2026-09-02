# OutlineGraphScopes Pass

Outlines `pl.graph` regions into `FunctionType::Graph` functions, making the
scope form and `@pl.jit.graph` converge into one representation.

## Overview

This pass transforms `GraphScopeStmt` nodes — produced by
`with pl.graph("name"):` — into `Function(Graph)` definitions and replaces each
scope with a Call to the outlined function.

Its entire purpose is convergence. After this pass, a region the user marked in
place has the same shape as a function the user decorated with `@pl.jit.graph` —
a `Function(Graph)` with orchestration metadata plus a `Call` — so
`LegalizeGraphBoundary`, the Graph verifier and orchestration codegen need no
knowledge of which surface was written. The two surfaces exist because the choice
is ergonomic, not semantic: the decorator suits a layer that is already its own
function; the scope suits a slice of a larger orchestration body the user would
rather not split up.

The one thing that does **not** carry over is parameter *order*: the outliner
appends parameters in capture order, whereas the decorator form uses the
signature the user declared. The two boundaries are permutations of each other,
which nothing downstream depends on — every consumer reads the parameters through
`param_directions_`, not by position.

**Requirements**:

- Input IR must be in SSA form (run `ConvertToSSA` first); SSAForm is preserved
  (produced) by this pass
- `InlineFunctions` must have run (`InlineFunctionsEliminated`). The parser
  deliberately permits `pl.graph` inside an `Inline` body, on the understanding
  that the body is spliced into its orchestration caller before this pass sees
  it; a pipeline that orders the two the other way is rejected rather than
  silently leaving the region un-outlined
- Processes `Opaque` and orchestration-like (`Orchestration` / `Graph`)
  functions; device-side kernel types carry no Graph region to outline
- Runs **immediately before** `OutlineIncoreScopes`

**Why that position.** The InCore scopes inside a marked region must be outlined
*after* the region becomes a function, so that `OutlineIncoreScopes` sees the
same input it sees for a hand-written `@pl.jit.graph` function — an
orchestration body carrying `pl.at` scopes — and produces the same output. Doing
it the other way, or later in the pipeline, would leave a `GraphScopeStmt` alive
across dozens of passes that would each need to learn about it; the RFC (#2399)
calls this out as the main cost of a scope-shaped carrier, and running the
outliner early is what avoids paying it.

**Parent function type is preserved.** Unlike `OutlineIncoreScopes`, this pass
does **not** promote an `Opaque` parent to `Orchestration`. Carrying a Graph
region says nothing about what the enclosing function is, and promoting would
make any Opaque helper that happens to contain one eligible to be picked as the
compiled entry — the backend takes the *first* Orchestration function, so an
unrelated edit could silently change which function a program compiles to.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::OutlineGraphScopes()` | `passes.outline_graph_scopes()` | Program-level |

**Factory function**:

```cpp
Pass OutlineGraphScopes();
```

**Python usage**:

```python
from pypto.pypto_core import passes

outline_pass = passes.outline_graph_scopes()
program_outlined = outline_pass(program)
```

## Algorithm

1. **Scan each function once** for Graph regions, rejecting a `GraphScopeStmt`
   nested inside another one (a compile error — see "Nesting" below).
2. **Emit Graph-free functions unchanged.** Almost no program contains a
   `pl.graph` region, and one that does not pays a single linear scan.

   This fast path is a saving, not the complexity bound. `ScopeOutliner`
   computes a position's used-after set by walking every following statement's
   subtree, which would be O(M²) over a block of M statements if it ran
   everywhere; it runs only at the positions that are — or contain — a scope of
   the kind being outlined, since those are the only ones that can read the
   answer. A region among ordinary statements is therefore linear, and the pass
   stays inside the O(N log N) bound in `.claude/rules/pass-complexity.md` for
   programs that *do* use `pl.graph`, not merely for those that don't.
3. **Reject a Graph region in a function this pass does not outline.** Skipping
   it would leave the `GraphScopeStmt` in place while the pass still advertises
   `GraphOutlined`; because `required` is checked only when verification is
   enabled, that false property would otherwise reach codegen unnoticed.
4. **Outline**: run the shared `outline_utils::ScopeOutliner` with
   `ScopeKind::Graph` / `FunctionType::Graph` / suffix `_graph_`. Captured values
   become parameters, values used after the region become returns, and the scope
   is replaced by a Call.

The outlined function needs no level/role fixup: `Function`'s constructor
derives `Level::CHIP` + `Role::Orchestrator` for any orchestration-like type, so
a Graph minted here carries the same metadata the parser gives a
`@pl.jit.graph` function.

## Naming and the graph key

The region name is **required** — `pl.graph()` with no argument is a parse error,
and `IRBuilder::EndScope` rejects an empty `name_hint` for this kind. Every other
scope kind treats `name_hint` as an optional hint, so this is worth stating
plainly:

`name` becomes the outlined function's name; codegen derives the emitted C++
symbol from that name; the runtime keys its cached `GraphDefinition` on the
symbol's address. An auto-generated name would therefore change the recorded
graph's identity whenever an unrelated region was added earlier in the file. The
user owns the name because the user owns that stability.

**Duplicate names are disambiguated, not merged.** Two regions asking for the
same name get suffixed distinct names via the program-wide reserved-name set
(#1711). This is the safe direction: sharing one name would give two different
topologies one Definition, and the second call would replay the first's recorded
graph.

## Nesting

A Graph region inside another Graph region is rejected at two levels:

| Level | Catches | Message quality |
| ----- | ------- | --------------- |
| Parser (`_parse_graph_scope`) | textually nested `with pl.graph(...)` | points at the offending source line |
| Pass (`NestedGraphScopeChecker`) | any nested `GraphScopeStmt`, however built | names both regions and the function |

The runtime treats a `graph_begin` inside a recording as unsupported and falls
the whole region back to ordinary submits. That fallback is **silent** — the
program still computes the right answer and simply loses the speedup — so a
compile error is the only way the user finds out. The pass-level check is the
invariant; the parser check exists to give the better diagnostic while the
source-level nesting is still visible.

The parser additionally rejects a Graph region nested inside `pl.at`,
`pl.cluster` or `pl.spmd`: those become a single device task, whereas a Graph
region records a *topology* of tasks, so it must sit around the dispatches
rather than inside one.

## Example

**Before** (`with pl.graph(...)` in an orchestration body):

```python
@pl.jit
def entry(w: pl.Tensor, acc: pl.InOut[pl.Tensor]):
    for i in pl.range(LAYERS):
        with pl.graph("accumulate_band"):
            base = i * ROWS
            with pl.at(level=pl.Level.CORE_GROUP):
                band = pl.load(w, [base, 0], [ROWS, COLS])
                cur = pl.load(acc, [0, 0], [ROWS, COLS])
                pl.store(pl.add(cur, band), [0, 0], acc)
    return acc
```

**After** (region lifted; the Call is what the runtime records once and replays):

```python
@pl.function(type=pl.FunctionType.Graph, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
def accumulate_band(self, i, w, acc, base):
    with pl.at(level=pl.Level.CORE_GROUP):
        band = pl.load(w, [base, 0], [ROWS, COLS])
        cur = pl.load(acc, [0, 0], [ROWS, COLS])
        pl.store(pl.add(cur, band), [0, 0], acc)
    return acc


@pl.function(type=pl.FunctionType.Orchestration)
def entry(self, w, acc):
    for i in pl.range(LAYERS):
        acc = self.accumulate_band(i, w, acc, i * ROWS)
    return acc
```

`base` reaching the call site is `LegalizeGraphBoundary`'s Step A, not this pass:
a scalar the region *derives* has no argument slot, so the runtime would freeze
the first call's value into the recording. This pass only lifts the region; the
boundary contract is enforced downstream.

## Properties

| Property | Role |
| -------- | ---- |
| `SSAForm` | required and produced |
| `InlineFunctionsEliminated` | required |
| `GraphOutlined` | produced |

`GraphOutlined` asserts that **no** function retains a `GraphScopeStmt` — not
even a Graph function, since nested regions are rejected outright.

## Related

- [09-outline_incore_scopes.md](09-outline_incore_scopes.md) — runs next; keeps a
  Graph function `Graph` while outlining the InCore scopes in its body
- [46-legalize_graph_boundary.md](46-legalize_graph_boundary.md) — enforces the
  runtime's boundary contract on the outlined function
- [00-pass_manager.md](00-pass_manager.md) — the `GraphOutlined` IRProperty in the
  property table
