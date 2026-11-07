# OutlineSimtScopes

`OutlineSimtScopes` turns each transient `SimtScopeStmt` in an explicit AIV
function into a void `FunctionType.SIMT` helper and a `SimtLaunchStmt`. It runs
before `InlineFunctions`, control-flow normalization, SSA conversion and scope
flattening. Code generation rejects unoutlined scopes.

## Captures and results

Captures follow first lexical use order and use Var identity. The helper gets
fresh parameters; its body references those parameters instead of caller Vars.
Runtime scalar captures are uniform inputs. Scalar reassignments create private
bindings and do not update the caller. Thread-private values cannot escape.

The transient node records reassigned outer scalar bindings in `private_scalars`.
These are uses of caller bindings, not caller definitions. Cloning preserves
their initial value and prevents region assignments from changing later caller
uses. This also covers assignments executed on only one branch.

A store makes its captured storage parameter `InOut`, even inside a conditional.
Each such parameter has exactly one launch result, in parameter order. Before
SSA, that result defines the captured caller binding; SSA then gives the updated
handle a fresh version. Existing Tile output materialization preserves old Tile
aliases and unwritten elements. Tensor aliases continue to reference GM.

Outer `In` permissions survive storage copies/views. Captures must already be
initialized. A complete, unmasked `tile.store` can initialize an `Out` Tensor
before capture; a partial or conditional write is not proof at a later join.
Initialization inside a branch is usable by a later scope in that same branch.
Initialization follows the reaching storage value, separately from conservative
alias permissions. A full store proves only its definite destination and aliases
initialized; a fresh Tile load replaces an earlier uninitialized value without
initializing old aliases. Branch-local proofs and changed storage identities are
not assumed after a join. A loop capture must also be initialized on later
iterations; a fresh load before the capture supplies a new proof. JIT shape folding and scalar renaming inside a SIMT
scope do not affect subsequent caller code. Conditionally rebound private scalar
assignments are retained inside the region, including shape-derived assignments.
A straight-line reassignment retires the old fold after evaluating its right-hand side. An incoming
shape constant read before a private write is materialized as a scalar, so
conditional writes preserve both the original value and subsequent rebindings. Both sides of an `if`
start from the incoming bindings; their changed bindings remain invalid at the
join. Explicit loop carries retain initialized seeds, and captures that depend
on them require initialized yielded values on every iteration. Loop outputs are
initialized only when both the seed and the backedge values are initialized.
Seed and yield expressions are checked before call flattening, including nested
Tile arithmetic. All backedges are evaluated before requirements are validated,
so dependencies between carries cannot evade checks through parameter order.
Shape unpacking inside SIMT expands into scalars even when conditional writes
require runtime bindings.

Storage returned by a function call or projected from a rebound tuple is rejected
when its provenance cannot be proved. Storage projections from tuples also do not
supply initialization proofs in loop seeds or yields, including when nested in
Tile arithmetic: a tuple may retain an older handle after its source is rebound.
Capture or carry the original named storage instead.

## Legality and resources

Only explicit AIV functions may contain scopes initially. Nested SIMT regions,
active register regions, storage allocation/rebinding inside helpers, VMI
register captures, and control flow crossing the region boundary are rejected.
The canonical SIMT verifier validates the generated helper bodies.

Dimensions are exactly three positive compile-time integers. The pass uses the
same target resource validation as explicit launches; later specialization
supplies thread/register limits. Users cannot provide resource overrides.

The pass indexes region definitions and storage permissions once per function.
Capture collection traverses disjoint region bodies. Alias analysis uses union
by rank/path compression and names use an ordered set: O(N log N) overall.

See [SIMT authoring and IR](../ir/11-simt.md) for the public syntax and
[pass manager](00-pass_manager.md) for pipeline integration.
