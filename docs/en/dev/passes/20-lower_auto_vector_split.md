# LowerAutoVectorSplit Pass

Converts an AUTO `pl.split` mixed `InCore` function into the **explicit
`split_aiv` form** *before* `ExpandMixedKernel`. It inserts `tile.aiv_shard`
at cube→vector boundaries and `tile.aic_gather` at vector→cube boundaries,
halves only the **vector sub-region** along the split axis, injects
`tile.get_subblock_idx()`, and stamps `split` + `split_aiv` on the function.

This is the **live auto-split lowering path**: it always runs, immediately
before `ExpandMixedKernel`. After it runs, every split function reaches
[`SplitVectorKernel`](23-split_vector_kernel.md) already `split_aiv`-marked,
so that pass only stamps attributes (its split_aiv arm) — its former per-op
halving driver was deleted, and the halving machinery now lives solely in
`split_axis_utils`, shared by this pass.

This pass is also the **sole consumer of the first-class `SplitAivScopeStmt`
region node** (`for aiv_id in pl.split_aiv(...)`). The region survives parse →
SSA → `ResolveBackendOpLayouts` as a structural node; here each region is
lowered in place and the scope wrapper is **erased**, so no `SplitAivScopeStmt`
reaches `ExpandMixedKernel` (pass 21) or codegen.

## Why this pass exists

A mixed `InCore` function written with `pl.split` describes cube and vector work
in one body, with the split intent expressed only by the function-level `split`
mode. Rather than halve the AIV body op-by-op *after* `ExpandMixedKernel` has
already split it (the old `SplitVectorKernel` path, which duplicated the boundary
semantics `tile.aiv_shard` / `tile.aic_gather` already encode), this pass
rewrites the AUTO body into the same explicit `split_aiv` shape a hand-authored
kernel uses, *before* that split. `ExpandMixedKernel`'s single op-driven boundary
arm then folds shard/gather into split-stamped `tpush`/`tpop` for auto and
hand-written kernels alike — one downstream path. The result is byte-identical to
the old halving (proved during the staged convergence): both call the same
`split_axis::ProcessStmts` machinery, and only the entry point differs.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::LowerAutoVectorSplit()` | `passes.lower_auto_vector_split()` | Program-level |

```python
from pypto import passes
result = passes.lower_auto_vector_split()(program)
```

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | `SSAForm` |
| Produced | `SSAForm` |
| Invalidated | — |

Source: `include/pypto/ir/transforms/pass_properties.h`
(`kLowerAutoVectorSplitProperties`).

## Scope

A function is rewritten iff **all** of:

- `func_type_ == FunctionType::InCore`, and
- it carries a function-level split mode (`UpDown` / `LeftRight`,
  `mode != None`), and
- it is **not already** `split_aiv` (hand-authored explicit kernels are left
  untouched — they already carry the explicit shard/gather form), and
- it is **genuinely mixed** (cube↔vector): its rolled-up affinity is `MIXED`,
  the same `ClassifyCallAffinity` / `CombineAffinity` decision `ExpandMixedKernel`
  uses for `is_mixed`.

Everything else is passed through unchanged. The last condition matters: a
**pure-vector** `pl.split` function (an elementwise op split across the two AIV
lanes, with no cube and no C↔V boundary) has nothing to converge, so it is left
untouched and `ExpandMixedKernel` converts it to a plain AIV function and strips
its `split` attr exactly as before. Were it lowered here, it would carry
`split_aiv` without a `split` mode after that strip, and `SplitVectorKernel` would
reject it.

## Explicit `SplitAivScopeStmt` region path

In addition to the AUTO whole-function path above, an `InCore` function whose body
still carries one or more `SplitAivScopeStmt` regions takes a separate **region
path** (`LowerExplicitRegionFunction`), checked **before** the AUTO path. Each
region carries its own `split_` mode, so this handles the multi-mode case the
single function-level mode cannot. Region-local `tile_vars` / `var_replacements`
maps keep a halved var from leaking into a sibling region or an out-of-region op;
statements **outside** any region are emitted full-width. After all regions are
lowered, the wrappers are dropped and the function is stamped `split_aiv` +
`split_aiv_region_validated` (the latter signals
[`ExpandMixedKernel`](21-expand_mixed_kernel.md) to skip its single-func-mode
transpose check — this pass validates each region's transpose hazard with the
correct per-region split axis instead).

### The out-of-region contract (manual mode)

"Emitted full-width" describes what this pass *does* with an out-of-region
statement, not what an author may *write* there. A function opening **at least
one** region enters **manual mode**: the regions own vector placement, and the
[`AivSplitValid`](99-verifier.md) verifier enforces that division well before
this pass runs:

| op / value | inside a region | outside every region |
| ---------- | --------------- | -------------------- |
| vector compute | AIV | **rejected** — check (e) |
| `tile.load` / `tile.store` | AIV | allowed (compiler-materialised) |
| cube compute | **rejected** — check (a) | AIC |
| `aiv_shard` / `aic_gather` | the boundary | **rejected** — check (c) |
| `pld.system.notify` | pinned to AIV | duplicated onto both (**not** diagnosed) |

So outside a region this pass sees only cube work, the `tile.load` /
`tile.store` pairs `ConvertTensorToTileOps` hoists out, and core-agnostic
scalar / control-flow statements — never full-width vector compute, which must be
wrapped in `for _ in pl.split_aiv(2, mode=pl.SplitMode.NONE):`. The multi-mode
goal is *regions only, one per vector phase*; a function with **no** region is
untouched. Checks (f)/(g) add that a tile crossing a region edge must name the
crossing, so no implicit cube↔vector crossing reaches this pass.

The last row is **documented, not enforced**, and the stamp below does not make a
region mean "exactly once": sharding a once-only side effect across the AIV
sub-lanes is the author's job, as is the lane rule for a `None`-region V→C
crossing ([Scopes and Placement](../../user/language/04-scopes.md)).

### Carrying region placement past the erasure (`core_placement`)

Erasing the wrappers loses the record of *where the author put a statement*, and
[`ExpandMixedKernel`](21-expand_mixed_kernel.md) duplicates every `SHARED`
statement onto **both** lanes. A core-agnostic op in a region
(`pld.system.notify`, whose TNOTIFY declares no affinity) would land on the cube
lane too, where it can publish a signal before the vector lane's TPUT has landed
the data that signal releases.

So before splicing a region body out, this pass stamps
`attrs["core_placement"] = "aiv"` on the calls it is about to orphan, and
`ClassifyCallAffinity` reads it as the **placement authority**, resolving them to
`VECTOR`. The attr asserts a placement, so it is written only where the region is
what *decides* one:

| intrinsic affinity | stamped? | why |
| ------------------ | -------- | --- |
| `SHARED` **and** `set_no_duplicate()` (`pld.system.notify`) | **yes** | only the region places it, and duplication is wrong for it |
| `SHARED` but *not* marked (`pld.system.wait`) | no | pinning **removes** it from the cube lane — for a blocking op that is a miscompile |
| `VECTOR` | no | already the AIV lane, by its own memory spec |
| a **stated** lane (`tile.create`, `system.syncall(core_type=…)`) | no | placed by its own declaration, which a region does not outrank |
| `MIXED` (`aiv_shard` / `aic_gather`, C/V `tile.move`) | no | these *are* the transfer — tpush on one lane, tpop on the other |
| `CUBE` | no | rejected in a region by check (a) |

A mixed comm kernel therefore gains exactly one attr, on the notify. The stamp
buys one thing — the op is not copied onto the **cube** lane; it says nothing
about how many AIV sub-lanes run it. The walk descends into compound statements,
is idempotent, and runs on each arm's **final** statements.

**Lifetime: this pass → pass 21, no further.** `ExpandMixedKernel` strips the
attr once consumed; `Call::attrs_` is a reflection `UsualField` and the printer
serialises attrs open-world, so an un-stripped stamp would surface in every later
pass dump and `assert_structural_equal`. Same lifecycle as `pipeline_stages`.

A function-level AUTO split and explicit `pl.split_aiv` regions are **mutually
exclusive**, enforced at [`OutlineIncoreScopes`](08-outline_incore_scopes.md)
while the scope's `split_` and its regions are both still visible; use
`optimizations=[pl.cross_core_slot(slot_num=N)]` to size the pipe without
annotating a split.

Three region body shapes are handled, selected by the region's `split_` mode:

- **Data-parallel, full-width body** (`UpDown` / `LeftRight`, no explicit boundary
  op): the region body holds full-width vector compute. The region path injects a
  per-region `subblock_idx`, routes the vector ops through the shared
  `split_axis::ProcessStmts` halving machinery (region-scoped), and validates the
  per-region transpose hazard. This is the paradigm the auto-converged form
  produces.
- **Data-parallel, explicit boundary body** (`UpDown` / `LeftRight` with
  `tile.aiv_shard` / `tile.aic_gather` already present): the user manually sharded
  the cube tile and wrote the vector compute on the per-lane half, so the body is
  **already** in half-width form. The region path detects this
  (`RegionBodyHasExplicitBoundary`) and **splices the body through unchanged**.
  Re-halving would double-shard — a downstream Acc→Vec move misread as a fresh
  cube→vector boundary and rewritten to a second `aiv_shard` — orphaning a halved
  Acc memref and crashing PTO codegen. `ExpandMixedKernel` folds the boundary into
  `tpush`/`tpop` as for a hand-authored split_aiv kernel.
- **Task-parallel body** (`None`): **no split axis** — both AIV lanes run the
  **full** body for disjoint work the author dispatches via `aiv_id`. The body is
  **spliced through unchanged** (no halving, no offset localization, no injected
  `subblock_idx`; the author's `aiv_id = get_subblock_idx()` already carries the
  lane). `tile.aiv_shard` / `tile.aic_gather` are **accepted** here: with no split
  axis they cross the boundary without splitting, and `split=0` preserves the
  shape. `ValidateMixedExplicitRegion` is skipped — everything is full width. The
  function is still stamped `split_aiv`, so `SplitVectorKernel` dispatches it to
  **both** AIV lanes (`dual_aiv_dispatch`) rather than the lane-0-only replay;
  both therefore push on a V→C crossing, into one shared slot with no
  arbitration, so the cube receives an unspecified one of the two values unless
  the author keeps it lane-uniform. Use this mode when the tiles cannot be halved
  or a reduction must stay full width.

### What may appear inside an explicit-boundary region

Because that body is spliced through **unchanged**, every vector op in it must
already be per-lane — an op left at full width would run identically on both AIV
lanes. `ValidateMixedExplicitRegion` enforces this and rejects the region with an
actionable error naming the offending ops. A tile-producing op is accepted when
any of the following holds:

| Accepted | Why |
| -------- | --- |
| Consumes a `tile.aiv_shard` result (transitively) | It is in the half-width dataflow by construction. |
| A pure generator — `tile.full` / `tile.ci` / `tile.random` (and `tile.create`, which classifies `SHARED` and so was never reportable anyway) | Its result is a function of its attributes only: it reads no tile and no memory, so per-lane replication is correct at whatever extent the author wrote. |
| An address-carrying op — `tile.load` / `tile.slice` / `tile.extract` / `tile.gather_row` — whose **read address** references the region's `aiv_id` | The author localized it explicitly, e.g. `data[base + aiv_id * HALF : ...]`. Only the read-offset args count (`tile.load` arg 1, `tile.slice` arg 2, `tile.extract` args 1–2, `tile.gather_row` arg 3 = `src_offset`) — a lane reference in a `shape`, a `valid_shape`, or a *destination* slot does not move the window, so it does not admit. |

`tile.gather_row` is the DMA case: being DPS it carries **two** offsets, and only
`src_offset` decides whether the lanes do different work — a lane-derived
`src_offset` means each lane pulls its own scattered GM rows (admitted), while a
lane-derived `dst_offset` over a lane-invariant `src_offset` means both lanes fetch
the *same* rows into different slots of a full-width accumulator (still reported).

Anything else that classifies `VECTOR` is reported. A generator is accepted for
**itself only** — `z = pl.full([FULL, N]); y = pl.add(z, z)` still rejects on `y`,
because a full-width generator must not vouch for its consumers. And the lane
reference is trusted **only** on an addressing op, so
`pl.set_validshape(full_width_tile, 1, aiv_id * HALF)` cannot launder a full tile
into the region.

The guard proves *intent*, not *extent*: a load at a lane-strided offset but a
full-width extent is accepted, and the two lanes then read overlapping windows —
the same trust already extended to `tile.store`, whose lane-dependent offset the
pass never checks.

Because the region is built via the generic `BeginScope`/`EndScope` and is
non-outlined, it can be **nested** inside a `pl.range` / `pl.pipeline` loop or an
`if`; the region path recurses into compound statements to lower every region while
preserving the surrounding control flow.

### Per-lane scattered gather

`pl.gather_row` is the only op that reads GM at an arbitrary **runtime** offset, so
it is how a paged/top-k row set is sharded across the two AIV lanes: each lane
assembles half the tile in UB and `pl.aic_gather` hands the reassembled tile to the
cube.

```python
with pl.at(level=pl.Level.CORE_GROUP, name_hint="sparse_kv", allow_early_resolve=True,
           optimizations=[pl.cross_core_slot(slot_num=2)]):     # see the ring note below
    for aiv in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
        ub = pl.full([64, 512], dtype=pl.BF16, value=0.0)       # per-lane HALF extent
        for k in pl.range(64):
            src = pl.cast(pl.read(idx, [aiv * 64 + k]), pl.INDEX)
            ub = pl.gather_row(ub, pool, [k, 0], [src, 0], [1, 512])   # lane-derived src_offset
        kv = pl.aic_gather(ub)                                  # V2C -> [128, 512] in Mat
    out[0:16, 0:128] = pl.matmul(q, kv, b_trans=True, out_dtype=pl.FP32)
```

Two authoring rules make this work:

- **Write the accumulator at the half extent.** `pl.full` is a generator, accepted
  at whatever extent you give it and never joining the half-width dataflow on its
  own. The gather is admitted on its lane-derived `src_offset`, and the guard proves
  *intent*, not *extent* — so a full-extent accumulator would be gathered back to
  `2 x FULL` and mismatch downstream.
- **Size the cross-core ring.** The V2C ring reserves `slot_size x slot_num` bytes
  of the consuming core's memory (L1 for V2C, UB for C2V), where `slot_size` is the
  **full** tile the consumer pops (`128 x 512 x 2 = 131072` here) and `slot_num`
  defaults to **8** — 1 MB of a 512 KB L1. `pl.cross_core_slot(slot_num=N)` lowers
  it; a kernel pushing once per invocation needs no more than 2. Omit it and
  `AllocateMemoryAddr` reports the overflow.

`pl.aiv_shard` is **not** an alternative to the half-extent `pl.full` here: it is
the C→V transfer and needs an `Acc` operand, so it cannot shard a value the
vector lane produced itself.

### Regions must be scope-free

Region lowering recurses into `ForStmt` / `WhileStmt` / `IfStmt` / `SeqStmts` but
deliberately **not** into a `ScopeStmt`: a scope carries outlining and
name-visibility semantics that region-local halving must not reach through. Every
region must therefore already be scope-free when this pass runs — normally
guaranteed by [`OutlineIncoreScopes`](08-outline_incore_scopes.md) (pass 8), which
lifts the enclosing `InCore` scope into its own function.

That guarantee has a hole, so the pass enforces it: pass 7 only outlines scopes
out of `Opaque` / `Orchestration` functions, while the parser wraps a top-level
`for aiv_id in pl.split_aiv(...)` in an `InCore` scope whatever the enclosing
function's type — so declaring the function `pl.FunctionType.InCore` delivers a
scope-wrapped region here:

```python
@pl.function(type=pl.FunctionType.InCore)   # pass 8 skips this function
def f(self, a: pl.Tensor[[128, 128], pl.FP32],
      c: pl.Out[pl.Tensor[[128, 128], pl.FP32]]) -> pl.Tensor[[128, 128], pl.FP32]:
    for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):   # wrapped in an InCore scope
        base = aiv_id * 64
        c = pl.store(pl.exp(pl.load(a, [base, 0], [64, 128])), [base, 0], c)
    return c
```

After lowering, `LowerExplicitRegionFunction` re-scans the body and rejects any
surviving region with a `ValueError` pointing at the `pl.split_aiv` line. Use
plain `@pl.function` / `@pl.jit` (Opaque) so pass 8 outlines the scope, or move
the region out of the enclosing scope.

The guard is also what makes the `split_aiv_region_validated` stamp trustworthy:
the attrs are written only once every region has actually been consumed, so
[`ExpandMixedKernel`](21-expand_mixed_kernel.md) skipping its own func-mode check
on the strength of that stamp is always backed by a real per-region validation.
Without it a scope-nested region passed through unlowered *and* un-validated
while still being stamped "region validated", and the failure surfaced much later
as an internal assertion in PTO codegen (`SplitAivScopeStmt reached PTO codegen`).

## Split-axis dispatch

| `SplitMode` (int) | Split axis | Vector sub-region halved on |
| ----------------- | ---------- | --------------------------- |
| `None` (0) | — (no split axis) | nothing — task-parallel; tiles stay FULL, `aiv_id` dispatches both lanes |
| `UpDown` (1) | dim 0 (height) | rows |
| `LeftRight` (2) | dim 1 (width) | cols |

`SplitDimension(mode)` returns `0` for `UpDown`, `1` for `LeftRight`
(`split_axis_utils`); it is **not** called for `None` (the region path branches on
`None` first — there is no axis to derive).

## Partially-valid operands across the boundary

A crossing value whose `valid_shape` is short of its physical box is a kernel's
ragged tail. The Cube→Vector FIFO pins the transported **column** extent to the
physical one and leaves the **row** extent free (derivation and ISA references:
[PTO codegen](../codegen/00-pto_codegen.md)). A narrowing on the *split axis* is
what makes an extent per-lane — lane `L` holds `clamp(V - L*half, 0, half)` — so
which mode is ragged decides which field has to carry it:

The table is the **shard's** (Cube→Vector) contract; `aic_gather` follows the
geometric rule at the end of this section instead.

| Ragged axis | Mode | Extent is | Carrier | Status |
| ----------- | ---- | --------- | ------- | ------ |
| rows (split axis) | `UpDown` | per-lane | TPOP `valid_row` operand (free field) | **supported** |
| cols (non-split) | `UpDown` | shared, static | full-box transport + static `pto.treshape` | supported |
| rows (non-split) | `LeftRight` | shared, static | TPOP `valid_row` operand | supported |
| cols (split axis) | `LeftRight` | per-lane | none | **rejected** |
| cols, runtime-valued | either | shared, dynamic | none (`treshape` takes no operands) | **rejected** |
| rows per-lane **and** cols narrowed | `UpDown` | both | none (`treshape` rewrites both axes) | **rejected** |

`ReshapeSplitAxis` can only ceil-halve the split-axis extent (the lane index is
not part of an op's type function). `LocalizeExplicitBoundaryValid` repairs that
guess here, where the region's `aiv_id` is in scope, and carries the per-lane
extent to consumers that pass `valid_shape` through; one that reshapes the
logical rectangle is rejected with its span. The AUTO arm applies the same *extent* repair through
`LocalizeShardValidForLane`, but not the store guard below — its consumers are
rebuilt by the halving walk rather than by this one.

- **An empty lane's store is guarded.** A lane the ragged extent does not reach
  has extent `0`, and a zero-row `TSTORE` is outside pto-isa's contract
  (`TSTORE_IMPL` asserts `GetValidRow() > 0`). The store gets a runtime
  `extent > 0` guard; `tpop` and `tfree` stay **unconditional** — both lanes
  occupy a slot and both must release it.
- **The gather is limited by geometry, not the DMA.** A V2C pop lands in an NZ
  Mat tile (`TLoadGm2L1Nd2nz`), which reads no valid extent at all. What limits
  `aic_gather` is placement: lane `l` sits at offset `l*half`, so the joined data
  `[0, v0) ∪ [half, half + v1)` is a rectangle only when the bands abut. That
  rule is enforced **in this pass**, not in the deducer, which runs before the
  per-lane extents exist and could only judge the join on its own ceil-div guess.
  A gather fed by a localized shard is typed exactly — the bands always abut, so
  the joined extent is the pre-shard `V` — and only a partial that both lanes
  share is rejected.

## Algorithm

`LowerFunction` rewrites one mixed `InCore` function:

```text
1. split_dim = SplitDimension(mode); split_int = int(mode).
2. InjectSubblockIdx(func, is_aiv=true) prepends
       subblock_idx = tile.get_subblock_idx()
   to the body (fresh name if 'subblock_idx' is taken).
3. LowerStmts walks the flat body:

   Boundary tile.move (ClassifyMoveDirection):
     CUBE_TO_VECTOR — replace the move with
         tile.aiv_shard(full_cube_tile, split=int(mode))   -> HALF
       The deduced HALF type already carries the consuming-lane memory
       (Vec): the split deducer leaves memory_space null and
       OpRegistry::Create fills it from tile.aiv_shard's set_output_memory
       declaration, shared with the explicit form. Seed it into tile_vars (its half
       extent) and record the old->new var rebind. The cube source (the
       matmul / Acc result) stays FULL.
     VECTOR_TO_CUBE — insert
         tile.aic_gather(half_vector_tile, split=int(mode))  -> FULL
       resolving the source to its halved var so the gather doubles
       HALF -> FULL, then keep the original cube-placement move on the
       gathered FULL tile (named "<dest>_mat" so ExpandMixedKernel's V->C
       boundary names its synthesized tpop after it).

   Affinity gate (ClassifyCallAffinity):
     VECTOR-affine leaf — route the single statement through
       split_axis::ProcessStmts({stmt}, ..., is_aiv=true): the SAME machinery
       the deleted SplitVectorKernel driver used. Halves tile.load /
       tile.store / tile.slice / tile.reshape / compute results on split_dim,
       localizes offsets per subblock, tracks halved vars in tile_vars.
     CUBE-affine leaf — passed through FULL, never halved.

   ForStmt / IfStmt — recurse into the body for vector content.

4. CheckNoCubeTileHalved re-walks the rebuilt body and asserts no CUBE-affine
   op consumes or produces a tile in tile_vars (the affinity gate must never
   leak a halved tile into a cube operand) — INTERNAL_CHECK on failure.
5. transform_utils::Substitute applies var_replacements; DeepClone detaches
   shared sub-trees.
6. WithSplitAivAttrs stamps split + split_aiv (dropping any prior split /
   split_aiv / dual_aiv_dispatch entries).
```

The per-op vector halving (shape halved on the split axis, offset localized by
`subblock_idx * half`, `tile.slice` static-shape-arg halving in lockstep with
the result type, rank-1-load reshape sliced per lane, reduce-on-split-axis
rejected, singleton split-dim preserved, loop `iter_arg`/`return_var`
tracking) is all produced by `split_axis::ProcessStmts` / `ProcessStmt` —
documented in detail in the shared machinery; the same facts are exercised by
`tests/ut/ir/transforms/test_lower_auto_vector_split.py`.

Automatic halving rejects the root generators `tile.ci` and `tile.random` when
their split dimension is non-singleton. Their generated values depend on
position, so changing only the result type is insufficient: a correct rewrite
also needs lane-specific shape and generator state, which this pass does not
synthesize. Move the operation outside the automatically-halved split region.
Singleton split dimensions and already-half-width explicit-boundary regions
remain unchanged.

## The affinity gate

Only **vector** work is halved; cube work stays full. Affinity is decided by
`core_affinity::ClassifyCallAffinity` (memory-space driven): an op producing or
consuming a `Vec` tile is `VECTOR`; matmul operands and the Acc/Mat cube result
are `CUBE`. `tile.aiv_shard` is the seam — FULL cube tile in, HALF vector tile
out — and `CheckNoCubeTileHalved` is the backstop.

## Example — cube→vector boundary, vector region halved (UpDown)

A mixed kernel: a cube tile (`Mat`) crosses to `Vec`, a vector `add` runs on
it, the result is stored.

**Before** (post-InferTileMemorySpace mixed `InCore`):

```python
@pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
def split_auto(qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
               out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]]):
    popped: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec] = pl.tile.move(qk, target_memory=pl.Mem.Vec)
    y: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec] = pl.add(popped, popped)
    return pl.store(y, [0, 0], out_0)
```

**After**:

```python
@pl.function(type=pl.FunctionType.InCore,
             attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True})
def split_auto(qk, out_0):
    subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
    popped: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.aiv_shard(qk, split=1)  # C->V, HALF
    y: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.add(popped, popped)
    return pl.store(y, [0 + subblock_idx * 64, 0], out_0)
```

The cube operand `qk` stays `[128, 128]`; the vector sub-region is halved to
`[64, 128]` and the store offset is localized per subblock.

## Example — vector→cube boundary stays full (UpDown)

A V→C `tile.move` becomes `tile.aic_gather`; the cube placement move on the
gathered tile keeps the FULL `[128, 128]` `Mat` shape — the cube side never
sees a halved tile:

```python
# `v` is the per-lane HALF the affinity gate produced, e.g. [64, 128].
gathered_mat: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat] = pl.tile.aic_gather(v, split=1)
gathered:     pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat] = pl.tile.move(gathered_mat,
                                                                      target_memory=pl.Mem.Mat)
```

**The operand must be a per-lane half.** `tile.aic_gather` is declared
HALF → FULL, so the gather doubles `[64, 128] → [128, 128]`, which is exactly the
FULL result type the cube placement move keeps. That agreement is a
*precondition*, not a guarantee: a VECTOR value can reach the boundary un-halved
— a `Vec` parameter used directly, or a tile whose split dim is a singleton the
affinity gate deliberately preserves. Doubling such an operand would produce a
`[256, 128]` gather feeding a move still typed `[128, 128]`, contradicting
`tile.move`'s shape-preserving contract and yielding IR that does not survive
print→parse. There is no correct gather for a value with no half, so the pass
**rejects** it with an actionable `ValueError`:

```text
LowerAutoVectorSplit: the V->C boundary tile.move here carries a full-width
vector operand 'vec'. tile.aic_gather reassembles the two AIV lanes' per-lane
halves into the full tile the cube expects, so its operand must be a value the
split halving produced (a tile.load / tile.slice / elementwise result inside the
vector sub-region). An un-halved value has no half to gather — either derive the
per-lane half first (load or slice the value inside the split function) and move
that to the cube side, or, if the split axis is a singleton that cannot be
halved, keep the value on the vector side.
```

**The gather follows the operand's split axis, not the function's.** A
`tile.reshape` can migrate the split axis — the rms_norm `[N, 1] ↔ [1, N]` column
reshape moves it from dim 0 to dim 1 — and `TileInfo::split_dim` tracks where it
ended up. The gather is emitted with the `split` encoding of *that* dim
(`dim 0 → 1`, `dim 1 → 2`), so a `[1, 8]` lane-local operand under an `UpDown`
function gathers to `[1, 16]` via `split=2`, matching the move. Doubling the
function axis instead would yield `[2, 8]`. A tracked split dim outside `{0, 1}`
cannot be expressed as a `split` attr on a 2D gather and is rejected.

The gather result is `Mat`, not `Vec`: the declared type of a boundary op names
the **consuming** lane's space, and AIC pops a V→C transfer into L1. (`Vec` would
name the *producing* lane, contradicting the mirror op `tile.aiv_shard`, which
declares the vector-side `Vec` for its cube-produced operand.) The cube placement
move that follows is what puts the tile in its final operand space — `Mat → Left`
for a matmul operand; the `Mat → Mat` shown here is a no-op that survives only
because the pass preserves the author's original move.

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass LowerAutoVectorSplit();
```

**Implementation**: `src/ir/transforms/lower_auto_vector_split_pass.cpp`

- `LowerFunction` / `LowerStmts` — boundary rewrite + affinity-gated halving.
- `MakeReshapeOpCall` — builds `tile.aiv_shard` / `tile.aic_gather` calls.
- `CheckNoCubeTileHalved` — cube-operand integrity backstop.
- `WithSplitAivAttrs` — stamps `split` + `split_aiv`.

**Shared machinery**: `src/ir/transforms/utils/split_axis_utils.cpp`
(`ProcessStmts`, `InjectSubblockIdx`, `SplitDimension`, `IsReduceOnSplitAxis`) —
the per-op vector halving, shared with `SplitVectorKernel`'s
`ProcessStandaloneSplitFunction` and the `AivSplitValid` verifier.

**Python binding**: `python/bindings/modules/passes.cpp`

```cpp
passes.def("lower_auto_vector_split", &pass::LowerAutoVectorSplit, ...);
```

**Tests**: `tests/ut/ir/transforms/test_lower_auto_vector_split.py`, plus the
end-to-end `pl.split` golden scenarios in
`tests/st/codegen/torch/test_torch_codegen_cross_core.py`.

## Related

- [`ResolveBackendOpLayouts`](19-resolve_backend_op_layouts.md) — runs
  immediately before.
- [`ExpandMixedKernel`](21-expand_mixed_kernel.md) — runs immediately after;
  folds `tile.aiv_shard` / `tile.aic_gather` into split-stamped `tpush`/`tpop`.
- [`SplitVectorKernel`](23-split_vector_kernel.md) — downstream; only stamps
  attrs for the `split_aiv` functions this pass produces, plus the no-split
  dual-AIV path.
