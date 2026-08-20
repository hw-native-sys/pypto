# IR Verifier

Extensible verification system for validating PyPTO IR correctness through pluggable property verifiers with diagnostic reporting and Pass integration.

## Overview

| Component | Description |
| --------- | ----------- |
| **PropertyVerifier (C++)** | Base class for verification rules |
| **PropertyVerifierRegistry (C++)** | Singleton mapping IRProperty → PropertyVerifier factories with verify/report API |
| **Diagnostic** | Structured error/warning report with severity, location, and message |
| **VerificationError** | Exception thrown when verification fails |

### Key Features

- **Pluggable Rule System**: Extend with custom verification rules
- **Property-Based Verification**: Opt-in property sets — verify exactly what you need
- **Structural Properties**: TypeChecked, BreakContinueValid, NoRedundantBlocks, UseAfterDef, OutParamNotShadowed, NoNestedInCore, InOutUseValid, PipelineLoopValid, ArrayNotEscaped, ManualDepsOnSubmitOnly, and AtomicAddDtypeValid are verified before/after each pass by `VerificationInstrument`; at pipeline start `PassPipeline` verifies only the lightweight subset shared with `GetVerifiedProperties()`
- **Dual Verification Modes**: Collect diagnostics or throw on first error
- **Pass Integration**: Use as a Pass in optimization pipelines
- **Comprehensive Diagnostics**: Collect all issues with source locations

## Architecture

### Structural vs Pipeline Properties

| Category | Examples | Behavior |
| -------- | -------- | -------- |
| **Structural** | TypeChecked, BreakContinueValid, NoRedundantBlocks, UseAfterDef, OutParamNotShadowed, NoNestedInCore, InOutUseValid, PipelineLoopValid, ArrayNotEscaped, ManualDepsOnSubmitOnly, AtomicAddDtypeValid | Always true. Verified before/after each pass by `VerificationInstrument`; the subset shared with `GetVerifiedProperties()` is also checked at pipeline start. Never in PassProperties. |
| **Pipeline** | SSAForm, NoNestedCalls, HasMemRefs, ... | Produced/invalidated by passes. Verified per pass-declared contracts. |

`GetStructuralProperties()` returns `{TypeChecked, BreakContinueValid, NoRedundantBlocks, UseAfterDef, OutParamNotShadowed, NoNestedInCore, InOutUseValid, PipelineLoopValid, ArrayNotEscaped, ManualDepsOnSubmitOnly, AtomicAddDtypeValid}`. These are verified **before/after each pass** by `VerificationInstrument`. At **pipeline start**, `PassPipeline::Run()` verifies only the lightweight subset shared with `GetVerifiedProperties()` (`GetStructuralProperties().Intersection(GetVerifiedProperties())`) — so e.g. `ArrayNotEscaped` is checked before/after each pass but not at pipeline start. Since no pass declares them in `required`/`produced`/`invalidated`, `VerificationInstrument` unions them with the pass's declared properties to ensure no pass breaks these fundamental invariants.

### Verification Rule System

The verifier uses a **plugin architecture** where each `PropertyVerifier` subclass is an independent rule:

- Rules run in registration order across all functions
- Each rule operates independently — one rule's failure doesn't affect others
- Rules receive `ProgramPtr` and internally decide whether to iterate over functions or check program-level properties
- Rules can be selectively included via `IRPropertySet`

### Diagnostic System

| Field | Type | Purpose |
| ----- | ---- | ------- |
| `severity` | `DiagnosticSeverity` | Error or Warning |
| `rule_name` | `string` | Which rule detected the issue |
| `error_code` | `int` | Numeric error identifier |
| `message` | `string` | Human-readable description |
| `span` | `Span` | Source location information |

### Integration with Pass System

1. **Automatic property verification**: `PassPipeline` uses `PropertyVerifierRegistry` to check produced properties after each pass (controlled by `VerificationLevel` in `PassContext`). The lightweight subset of structural properties shared with `GetVerifiedProperties()` is checked at pipeline start. See [Pass Manager](00-pass_manager.md).
2. **`VerificationInstrument`**: A `PassInstrument` that verifies properties via `PassContext`. Before each pass, it checks the pass's declared `required` properties. After each pass, it checks the pass's declared `produced` properties **plus all structural properties** — ensuring no pass breaks fundamental IR invariants.

The `run_verifier()` utility creates a standalone `Pass` for ad-hoc use in custom pipelines, but it is **not** part of the default optimization strategies.

## Built-in Rules

| Rule Name | IRProperty | Purpose |
| --------- | ---------- | ------- |
| **SSAVerify** | SSAForm | No multiple assignment, no name shadowing, no missing yield, scope violations, cardinality checks |
| **TypeCheck** | TypeChecked | Type kind/dtype/shape/size consistency |
| **NoNestedCall** | NoNestedCalls | No nested call expressions in args, conditions, ranges |
| **BreakContinueCheck** | BreakContinueValid | Break/continue only in sequential/while loops |
| **UseAfterDefCheck** | UseAfterDef | Every Var use dominated by a definition (param, AssignStmt, loop var, iter_arg, return_var) |
| **NormalizedStmtStructure** | NormalizedStmtStructure | Nested `SeqStmts` flattened and single-child `SeqStmts` unwrapped |
| **NoRedundantBlocks** | NoRedundantBlocks | No single-child or nested `SeqStmts` |
| **SplitIncoreOrch** | SplitIncoreOrch | No `InCoreScopeStmt` nodes remain in Opaque functions |
| **IncoreTileOps** | IncoreTileOps | InCore functions use tile ops (no tensor-level ops remain) |
| **HasMemRefs** | HasMemRefs | All TileType variables have MemRef initialized |
| **AllocatedMemoryAddr** | AllocatedMemoryAddr | All MemRefs have valid addresses within buffer limits |
| **OutParamNotShadowed** | OutParamNotShadowed | Out/InOut params not reassigned with tensor-creating ops |
| **NoNestedInCore** | NoNestedInCore | No nested InCore scopes (`InCoreScopeStmt` inside `InCoreScopeStmt`) |
| **InOutUseValid** | InOutUseValid | Variables passed as InOut/Out to user-function calls are not read after the call (RFC #1026). Group-typed function bodies are skipped pending follow-up. |
| **PipelineLoopValid** | PipelineLoopValid | Bidirectional invariant on every `ForStmt`: `kind_ == ForKind::Pipeline` ⇔ `pipeline_stages` attr present. Either direction failing means the pipeline loop is malformed. |
| **ArrayNotEscaped** | ArrayNotEscaped | No `ArrayType` appears as a function parameter or return type (checked transitively through `TupleType`). `ArrayType` is on-core scalar-register-file / C-stack storage owned by the enclosing function — letting it cross a function boundary would leak a stack pointer, so it must be created and used locally inside the function body. |
| **ManualDepsOnSubmitOnly** | ManualDepsOnSubmitOnly | No plain cross-function `Call` (GlobalVar callee) carries `attrs["manual_dep_edges"]` — manual dependency edges live in the typed `Submit::deps_` field. Op calls (`system.task_dummy`) keep the attr as their codegen fanin contract and are exempt. |
| **OrchestrationReferencesResolved** | OrchestrationReferencesResolved | Every non-builtin Call inside a `FunctionType::Orchestration` function targets a Function that exists in the surrounding Program. Replaces the codegen-side `ValidateOrchestrationReferences` walk that used to throw at codegen time. |
| **RuntimeScopesMaterialized** | RuntimeScopesMaterialized | Every `FunctionType::Orchestration` function has `attrs_["auto_scope"] == false`, the marker stamped by `MaterializeRuntimeScopes` once explicit `RuntimeScopeStmt` nodes are in place (or set by user `@pl.function(auto_scope=False)`). Orchestration codegen emits `SIMPLER_SCOPE()` only from those nodes; skipping the pass leaves `auto_scope=True` and would silently omit scopes. **Produced by** `MaterializeRuntimeScopes` and listed in `GetVerifiedProperties()`, so `PassPipeline` auto-verifies it after that pass. |
| **AssignTypeSymmetry** | AssignTypeSymmetry | Every `AssignStmt(var, value)` satisfies `structural_equal(var.type, value.type)`. Covers dtype, shape, and tile_view/tensor_view; additionally TileType `memory_space` (TensorType has no `memory_space`) and DistributedTensorType `window_buffer`; tuple-typed assignments compare every element recursively. `memref_` is **excluded** — `structural_equal` treats it as an allocation detail bound to the Var, governed by `HasMemRefs` / `AllocatedMemoryAddr`. Catches passes that mutate one side of an assignment without the other (e.g. #1262 TileType memory_space, #1278 tile_view). Registered in `PropertyVerifierRegistry` but not yet in `GetStructuralProperties()` — run on demand via `PropertyVerifierRegistry::verify` or by adding the property to a `VerificationInstrument`. |
| **AivSplitValid** | AivSplitValid | Structural checks on the first-class `SplitAivScopeStmt` region, keyed on the node itself (so nested / multi-mode functions are checked region by region). A function holding at least one region opts into **manual mode**, where the regions are authoritative for vector placement. **(a)** No cube compute inside **any** region — for two independent reasons, which is why it fires whatever the mode: a data-parallel region cannot vector-split a `matmul` (each AIV lane holds only half the tile), and every region, task-parallel included, *is* the AIV lane's body, where cube work does not belong. **(b)** No vector reduce (`tile.row_*` / `tile.col_*` / `tile.sum` / `tile.max` / `tile.min`) that collapses the **split axis** — it yields a partial per-lane result (a miscompile). Unlike (a), this reasoning is split-axis-specific, so (b) stays gated on the data-parallel modes. **(c)** `aiv_shard` / `aic_gather` must appear inside a region (both the `tile.*` and author-facing `tensor.*` forms). Inside a task-parallel `mode=NONE` region they are **accepted**: with no split axis they still carry the meaning checks (f)/(g) demand — this value crosses the AIC/AIV boundary — and their `split=0` deduction preserves the shape instead of halving or re-joining it. **(d)** The **boundary memory contract**: `tile.aiv_shard` is `Acc → Vec` and `tile.aic_gather` is `Vec → Mat`. Both ops *are* the cross-core transfer, so the operand must live on the producing lane and the result on the consuming one; the memory sides are skipped until resolved, so the same check is safe across the whole window. Mode-independent — a task-parallel crossing spans the same two lanes as a data-parallel one — so it runs in every region, `NONE` included. **(e)** No VECTOR-affine op **outside** every region in a manual-mode function — with the region authoritative for placement, such an op is neither pinned to the AIV lane nor cube work, so it has no defined home. Three carve-outs: `tile.load` / `tile.store` are the compiler's own out-of-region output, since `ConvertTensorToTileOps` hoists the load/store pair for a tensor-level op out of the region holding the compute it feeds; and an op that **declares** its lane via `set_core_affinity` (`system.syncall(core_type="aiv_only")`, `system.sync_set/sync_wait(core_type="aiv")`, `pld.tile.put` / `get`) was never inferred in the first place, so a region cannot disambiguate it. A function with **no** region is unaffected. **(f)** V->C: a value defined inside a region and read on the **cube** lane outside it must be a `tile.aic_gather` result. **(g)** C->V: a cube-produced value defined outside every region and read on the **vector** lane inside one must arrive through a `tile.aiv_shard`. Both directions already lower without the op — the compiler emits a `split=0` tpush/tpop pair either way — which is exactly why they are checked: manual mode exists so the author, not the compiler, places the AIC/AIV boundary, and a boundary nobody wrote is one nobody chose. (f)/(g) share (e)'s `tile.load` / `tile.store` carve-out on **both** the definer and the consumer side, for the same reason: the compiler hoists that pair out of the region itself. A cross-C/V `tile.move` counts as a consumer on the side it delivers to, so the checks stay live at the `InferTileMemorySpace` verification point, where an implicit crossing has become such a move. **Not checked, deliberately:** the lane rule for a V->C crossing out of a `mode=NONE` region — the ISA requires both AIV sub-lanes to take part in a no-split handshake and they share one destination slot with no per-lane offset, and nothing arbitrates between them, so when the lanes hold different values the cube receives an unspecified one of the two; keeping the gathered value lane-uniform is the author's job, and is documented rather than synthesized. Also lane-sharding of a once-only side effect (`pld.system.notify`). A region cannot mean "exactly once" — the AIV function carries `dual_aiv_dispatch`, so its body runs on both AIV sub-lanes — and the correct authoring form (sharded by `aiv_id`) and the incorrect one (both lanes notifying the same peer) are structurally identical IR. Both rules are stated for authors in [Scopes and Placement](../../user/language/04-scopes.md) instead. **(h)** **Placement** — `pl.split_aiv` opens a CORE_GROUP-level region, so it must be authored inside a CORE_GROUP scope (`pl.at(level=pl.Level.CORE_GROUP)`) or at the top of an Opaque function (which the parser wraps into exactly such a scope) — never inside a function already declared `pl.FunctionType.InCore`. Keyed on **provenance**: a region reaches an InCore function legitimately only when `OutlineIncoreScopes` lifted the enclosing CORE_GROUP scope into it, which the outliner records by stamping `split_aiv` on the function it mints (`scope_outline_utils.h`; `LowerAutoVectorSplit` re-stamps it). So the check is "InCore function carrying a region, but not one the outliner produced". Provenance rather than shape is what makes it survive the parser emitting a top-level region **bare** in an InCore function — which it must, so that printing an outlined function and reparsing it rebuilds the same IR. A shape-based test ("region nested in a surviving InCore scope") worked only while the parser wrapped every top-level region, and would have silently stopped rejecting anything once it stopped. `LowerAutoVectorSplit` keeps its own post-lowering guard as the backstop for scope kinds this check does not walk; reporting here puts the diagnostic 12 passes closer to the source. **(i)** **No boundary result across a loop back-edge** — a `pl.aiv_shard` / `pl.aic_gather` result must not be the init value of a loop `iter_arg`, nor the value yielded back into one. A boundary result is per-lane (half-width), and neither `LowerAutoVectorSplit`'s half-width scan nor `ExpandMixedKernel`'s boundary folding follows a value through a loop phi: the scan sees the `IterArg`, not the shard that defined it, and reports an already-halved tile as full width; and `FixupIterArgInitValues` (`loop_state_repair.cpp`), which runs before the `DeepClone` that substitutes the boundary tpop, reads the still-original init var as undefined and resurrects the very `tile.aiv_shard` the pass just folded away — surfacing as an INTERNAL error naming SSA the author never wrote. BOTH ends are checked because the loop type-checks either way: a body seeded with a half-shaped `tile.full` and fed by a shard from iteration 1 onwards defeats the same passes with no init-side signal. **(j)** **A boundary result belongs to the region that produced it** — a `pl.aiv_shard` / `pl.aic_gather` result read inside a DATA-PARALLEL (`UP_DOWN` / `LEFT_RIGHT`) region other than its defining one is rejected. Such a region halves every tile it computes along the split axis and localizes every store offset to the lane, so an already-per-lane value is halved twice and offset twice — the offset corruption is silent, and the shape corruption escapes pypto entirely, surfacing from ptoas as `'pto.tcvt' op expects src and dst to have compatible shapes`. Gated on the data-parallel modes for the same reason as (b): a `mode=NONE` region has no split axis, rewrites nothing, and MAY consume a boundary result produced elsewhere — that is the cross-core comm-kernel shape, and the carve-out is what keeps it legal. A boundary op is not judged as a consumer (it *is* the crossing, same as in (f)/(g)); the `tile.load` / `tile.store` carve-out deliberately does NOT apply, since it exists for ops `ConvertTensorToTileOps` hoists OUT of a region and (j) only ever looks at a consumer INSIDE one. **(k)** **One function, one cross-core pipe — so one transport class** — a function must not hold both a no-split crossing (`split=0`, from a `mode=NONE` region) and a split one. Every `pl.aiv_shard` / `pl.aic_gather` in one function rides a SINGLE logical pipe: `BuildAutomaticPipeSetup` (`cross_core_pipe.cpp`) emits one `reserve_buffer` + `initialize_pipe` pair per side with a **combined** `dir_mask`, so C→V and V→C share it too. pto-isa carries no-split as a parameter of the pipe **type** (`TPipe<FlagID, Dir, SlotSize, SlotNum, LocalSlotNum, IsNoSplit>`), which selects a different handshake protocol (`ShouldNoSplitC2VConsumerLaneParticipate` gates on `Pipe::is_no_split`), so one pipe runs the no-split protocol or the split one for its whole lifetime. PTOAS enforces the same thing in `PTOInferValidatePipeInitPass`, which infers one `nosplit` bool per pipe from its users and rejects a disagreement as `'pto.initialize_l2g2l_pipe' op cannot mix 'split = 0' with 'split = 1', 'split = 2', 'split = 3', or 'split = 4' on the same logical pipe` — an internal op the author never wrote, from a backend the DSL does not mention. The split **axis** is genuinely per-transfer (`TALLOC` / `TPUSH` / `TPOP` take `TileSplitAxis` as their own template argument), so `UP_DOWN` beside `LEFT_RIGHT` shares one pipe and is **accepted**: the rule is drawn between `SplitMode::None` and everything else, exactly where PTOAS draws it, not between "different modes". Keyed on each boundary op's own `split` kwarg rather than on the enclosing region's mode, which is what makes it right in three places at once: the parser stamps that kwarg from the *innermost* `pl.split_aiv` (so nesting resolves for free), the outlined `pl.tile.aiv_shard(t, split=N)` form carries it with no region at all, and a region holding **no** crossing contributes no transport and may keep any mode — the comm-kernel shape, where a `mode=NONE` region exists only to pin `pld.system.notify` to the vector lane. Reported once per function, on the later of the two offending ops, naming the other one's location. **Produced by** `OutlineIncoreScopes`, then invalidated-and-re-produced by `ConvertTensorToTileOps` and `InferTileMemorySpace`, and finally invalidated by `LowerAutoVectorSplit` (which erases the region node). `PassPipeline` only verifies *produced* properties (`passes.cpp`), so those re-productions are what give checks (d) and (e) a live verification point — at `OutlineIncoreScopes` the boundary is still the space-less `tensor.*` form (so (d) is inert) and tensor-level compute classifies `SHARED`, which leaves (e), (f) and (g) with no VECTOR or CUBE op to find. Checked ops are plain `Call`s with a non-null `op_`; `Submit`s are correctly skipped. **Fix**: (a) move the cube op outside the region; (b) reduce the non-split axis, or `tile.aic_gather` the lanes back first; (c) move the boundary op inside the region it belongs to; (d) shard only a cube-produced value — a vector-produced one (`pl.load` / `pl.full`) is already on the AIV lane, so drop the `pl.aiv_shard` and let the implicit affinity-gated split halve it; (e) wrap that phase in its own region — `for _ in pl.split_aiv(2, mode=pl.SplitMode.NONE):` runs the full body on both lanes and is the task-parallel form for full-width compute; (f) gather the value inside the region (`x = pl.aic_gather(x)`) and read the gathered one outside; (g) read it as `pl.aiv_shard(x)` at the top of the region; (i) move the `pl.aiv_shard` call into the loop body next to its consumer, or carry the FULL-width `pl.matmul` accumulator across the loop and shard it at the point of use (which keeps a second set of L0C accumulators live and may exceed the Acc budget); (j) move the ops that read the boundary result into the region that produced it, or move the `pl.aiv_shard` call into the consuming region; (k) make every crossing in the function agree on split-vs-no-split (give the `mode=NONE` phase a split mode, or the split phase `pl.SplitMode.NONE`), or drop the crossing from one of the regions, or put the two phases in separate CORE_GROUP scopes so each outlines into its own function and gets its own pipe. |
| **HardSyncallOccupancy** | HardSyncallOccupancyValid | A hard (FFTS) `system.syncall` waits for **every** physical core of its `core_type` to reach the barrier, so the enclosing SPMD launch must satisfy two independent guarantees: (1) **full occupancy** — fill all those cores exactly (`N != required` → error, covering both partial and over-occupancy); and (2) **`sync_start=True`** — all blocks co-resident at once, since a non-sync_start launch may dispatch blocks in waves and deadlock the barrier even at full occupancy. Either gap deadlocks on device (AICore timeout 507018) and leaves it needing a reset. **Produced by** `ExpandMixedKernel` (which resolves each launched kernel's `FunctionType` — the precondition the check needs) and listed in `GetVerifiedProperties()`, so `PassPipeline` auto-verifies it right after that pass. Covers every SPMD launch site whose block count it can classify — a `FunctionType::Spmd` function (scope `pl.spmd`), a `FunctionType::Group` function with a `core_num` attr (`pl.cluster()`-nested `pl.spmd`), and a `Submit` with `core_num` (`pl.spmd_submit`). A width is either a compile-time literal (checked against the SoC's physical counts) or a **launch-shape query** — `pl.system.available_cluster_count()` / `pl.system.available_aiv_count()`, directly or through the Var it binds. A query resolves on device to the run's own geometry, so it fills its core type by construction and needs no count comparison; it is also the portable spelling, since a literal matching this SoC is wrong on any run whose device reports a different usable count — too few blocks when more cores are available, too many when fewer are (which the runtime rejects outright). Using the query for the *other* core type (an AIV-only launch sized by the cluster count, or a mixed launch sized by the AIV count) is rejected. Per launch site and its direct callee: standalone **AIV** kernel + `aiv_only` → `required = GetCoreCount(VECTOR)`; standalone **AIC** kernel + `aic_only` → `required = GetCoreCount(CUBE)`; a standalone AIV/AIC kernel with a *mismatched* `core_type` (incl. the default `mix`) is rejected outright — the other core type has zero participants in a single-core launch, so the barrier can never complete; **Group** (mixed kernel) → `required = GetCoreCount(CUBE)` core-groups (one AIC per group, filling all cores) for any barrier `core_type`, checking the hard syncalls in its AIC/AIV sub-kernels (duplicated `mix` reported once). Skipped when the backend is unconfigured, `core_num` is neither a literal nor a launch-shape query, or no SPMD site launches the kernel (occupancy is a launch-time property). **Fix**: launch at full occupancy with a matching `core_type` and `sync_start=True` — preferably by sizing the launch with the matching launch-shape query — or use `pl.system.syncall(mode=pl.SyncAllMode.SOFT, ...)` (GM-polling) for partial occupancy. |
| **AccToGmStoreValid** | AccToGmStoreValid | Every `tile.store` whose source tile is **Acc-resident** targets a GM tensor whose dtype the cube fix-pipe can narrow into: `INT32/FP32/FP16/BF16` on both Ascend910B and Ascend950 (`BackendHandler::SupportsAccToGmDtype`, mirroring pto-isa's non-quant `CheckAcc2gm` whitelist and ptoas' `pto.tstore` "acc tstore dst element type" rule; the hook stays per-backend because the two sets are independent facts about each pinned target). An `INT8`/`INT16` destination is rejected. **Produced by** `InferTileMemorySpace` and listed in `GetVerifiedProperties()`, so `PassPipeline` auto-verifies it right after that pass. That placement is load-bearing: legality is a property of the tile's **memory space**, not of the user-visible dtypes, so it cannot be decided earlier — the identical DSL program is legal when its matmul result routes through Vec (an explicit `pl.cast` narrows in the vector unit) and illegal when it stays in Acc. Without this check the program reaches ptoas, which rejects it at `pto.tstore` verification against a line in a generated `.pto`; one such store also tends to trip a second, unrelated-looking op (an int8 zero-init lowers to a `pto.texpands` with the same illegal dtype), so the late diagnostic names two symptoms and never the cause. Skipped when the backend is unconfigured (nothing to verify against) or the tile's memory space is unresolved. **Fix**: narrow through the vector unit — `pl.cast` the matmul result to the destination dtype and store that — or accumulate into an `INT32`/`FP32` tensor and convert afterwards. |
| **AccCompactValid** | AccCompactValid | Two halves of the L0C compact-mode contract. **(a)** Every `tile.matmul_acc` / `tile.matmul_mx_acc` whose lhs valid rows make `mad`'s pitch differ from the accumulator's physical row count accumulates into a `CompactMode::normal` buffer. `mad` takes M from the L0A operand's valid rows and lays the product out with an N-fractal stride of `ceil(M/16)*16` (pto-isa `TMatmul.hpp`), while every reader derives that stride from the compile-time physical `Rows` unless the tile is compact, in which case it recomputes `ceil(validRow/16)*16` (`tstore_common.hpp`). An accumulator that stays non-compact is read back at a pitch it was never written at, scrambling every N-fractal above the first — the store path of #2470 and the Cube→Vector push path of #2510, both of which shipped as wrong numbers on device with no diagnostic at any layer. The accumulate op is where both halves of the comparison are in hand (the lhs `mad` takes M from, and the buffer the result aliases in place) and it is the one op that *inherits* compact rather than deriving it, so it is exactly where a chain seeded by a non-compact buffer loses the mode. Checking the **readers** instead would be wrong: a `tile.store` cannot tell an accumulator `mad` wrote from an Acc tile a `tile.load` filled at the physical pitch. Only a pitch that *provably* differs is rejected — `ceil(validRow/16)*16 == Rows` holds when the valid rows fill the box and, for a single-fractal-block box, for every extent it can hold (a `[16, N]` gemv accumulator valid to one row still packs to 16), and `AccPitchesCoincide` is shared with the stamper so the two cannot drift apart. **(b)** A compact accumulator's packed pitch survives its aliases: `tile.set_validshape` is metadata-only and deliberately keeps `compact`, so re-declaring the valid rows across a fractal boundary (`mad` writes 17 rows at pitch 32, the alias then says 16) leaves the bytes packed at 32 while every compact reader now derives 16. Such an alias is rejected — narrow the result after it leaves L0C instead. A *fresh* full-box source is exempt, since `AutoTileMatmulL0` declares its seed as `tile.create(compact=True)` and narrows it immediately, and a buffer nothing has written re-interprets nothing; an undecidable pair of extents is left alone rather than rejected. **(c)** No tile outside the fractal spaces (`Left` / `Right` / `Acc`) carries a compact mode at all: compact *is* an N-fractal pitch, a UB tile has none, and no pto-isa Vec path reads `TileData::Compact` — so marking the Vec side of a C2V pop compact is inert where it is harmless and a silent layout change on the ops that do consult the flag (`TMov`, `TFillPad`). Checked on Var-like definitions and params, not on call result types, since `AssignStmt` binds the same type to both sides. **Produced by** `InferTileMemorySpace` (the first point where memory spaces are resolved) and re-produced by `ExpandMixedKernel`, which rebuilds the boundary `tile.move` as a tpush/tpop pair with a freshly built consumer type; both are listed in `GetVerifiedProperties()`. **Fix**: a failure is a compiler bug, not an authoring error — `tile.matmul` derives the mode via `StampCompactForNarrowedAccRows` and a synthesized accumulator seed declares it with `tile.create(..., compact=True)`; a chain that lost it has a seed or an alias that never carried it. |
| **TileMemoryInferred** | TileMemoryInferred | Every `TileType` variable **bound by an `AssignStmt`** carries a resolved `memory_space_`, **and** every constrained argument of a `Call` (in an `AssignStmt` or `EvalStmt`) sits in a space the op's registered `input_constraints` allow (`set_input_memory`). The visitor walks those two statement kinds only, so `ForStmt` `iter_args_` / `return_vars_` and `IfStmt` `return_vars_` annotations are **not** checked. The second half is the load-bearing one: an op whose declared input space is violated has no legal lowering, and the symptom always surfaces far downstream. `tile.cast` requires `Vec`; fed an `Acc` operand it leaves the cube→vector cut with no boundary `tile.move`, so `ExpandMixedKernel` splits the kernel with the cast referencing a var defined only on the cube half — the failure then lands 11+ passes later as an illegal `Acc->Acc tile.move` in `MemoryReuse` or as `no MLIR mapping for MemRef base` in PTO codegen, neither of which names the offending op. The verifier reads the `TileType` annotation directly rather than the analyzer's `var_memory_` map, so it stays honest about spaces the analyzer failed to record. **Produced by** `InferTileMemorySpace` and listed in `GetVerifiedProperties()`, so `PassPipeline` auto-verifies it right after that pass. **Fix**: the pass inserts the required `tile.move` itself; a failure here is a compiler bug in `InferTileMemorySpace`, not an authoring error. |
| **AtomicAddDtypeValid** | AtomicAddDtypeValid | Every atomic-add write into global memory targets a destination dtype the backend's store pipe can combine. Covers every atomic site in one place: `tile.store`, `tensor.assemble`, `pld.tensor.put`, `pld.tile.put`, `pld.tensor.remote_store` and `pld.tile.remote_store`. Only `bf16` varies by backend — pto-isa lowers it to `SetAtomicAdd<bfloat16_t>` -> `set_atomic_bf16`, honoured on Ascend910B (A2/A3) and not on Ascend950 (A5) (`BackendHandler::SupportsBf16AtomicAdd`); the remaining hardware atomic-add dtypes (`FP32/FP16/INT32/INT16/INT8`) are accepted everywhere and gated backend-neutrally in the op deducers. The remote-put path is the *same* mechanism as the local store, not a parallel one: pto-isa's comm `TPut` streams the transfer through its VEC staging tile and lands each chunk with `TSTORE_IMPL<..., AtomicAdd>`, and `remote_store` emits a `pto.tstore` directly, so one predicate governs every site — and ptoas carries no atomic dtype rule of its own (`TPutOp::verify` checks element-type agreement and shapes only), so without this check the program reaches a pto-isa `static_assert` in generated code the user never wrote. Listed in **`GetStructuralProperties()`**, not produced by any pass: nothing here depends on lowering (the atomic kwarg and the destination dtype are present in the user's own IR), so `PassPipeline` verifies it at `pipeline_input` and the error carries the original `Span`. Skipped when the backend is unconfigured (nothing to verify against). **Fix**: accumulate into an `FP32` tensor and cast to `bf16` after the reduction. |

### InParamWritten

> This is the last stage of the chain described in
> [Parameter Direction Inference](../ir/08-param-directions.md).

**Warning**: `DiagnosticCheck::InParamWritten` — a parameter declared `In` is
written by its own function body.

This is a **warning, not an `IRProperty`**, and the distinction is load-bearing
rather than cosmetic. See "Why it is not a property" below.

**What it proves, and what it does not.** Every pass that derives directions
builds a set of "which argument does this call write", and it reads that set
from each operator's registry declaration (`set_arg_effect`, see
[Operators](../ir/05-operators.md#argument-effects)) plus each callee's own
`param_directions_`. This check reads the *same two declarations* and reports
where they contradict a parameter's own `In`. That makes it a consistency check
over the declared write semantics — not an independent discovery of them.

The distinction matters, because the failure that motivated the work is the
*missing* declaration: an operator that never declared its effects reads as a
pure consumer, its write disappears, the parameter stays `In`, no RAW edge is
emitted against it, and the symptom surfaces on device as a race or a scheduler
deadlock rather than at compile time. `pld.system.notify` shipped that way
(#2391) and `tile.mscatter` was still in that state when this check was written.
**This verifier cannot catch that class.** For an operator with no declared
effect, `CallWriteTargets` returns nothing and the check is silent — it is
reading the very declaration that is absent.

Nor is the gap closed elsewhere, and it is worth being exact about how far the
registry gate reaches. `ValidateArgEffects` fires on two shapes only:

- an operator that declares `set_output_reuses_input(N)` without classifying
  argument `N`, and
- an operator that declares a write channel while writing through no argument.

An operator with **neither** — no reuse contract, no write channel — trips
neither gate. `pld.system.notify` is exactly that shape: drop its
`set_arg_effect` and it passes registration *and* this verifier, silently, just
as it did in #2391. The original production failure class is therefore **not**
universally closed by these two checks together; what they cover is an operator
that has already said something about itself and said it incompletely.

What this check does buy: once `tile.mscatter` and `pld.system.notify` carry
declarations, it stops a caller from re-declaring their destination `In`, and it
covers every cross-function call, where the callee's own signature is the
declaration and no registry entry is involved.

**How it runs.** Registered as `DiagnosticCheck::InParamWritten`, a **warning**
at `DiagnosticPhase::PostPipeline`. It is reachable only through the diagnostic
registry, which is what makes it a warning:

```python
checks = passes.DiagnosticCheckSet()
checks.insert(passes.DiagnosticCheck.InParamWritten)
diagnostics = passes.DiagnosticCheckRegistry.run_checks(
    checks, passes.DiagnosticPhase.POST_PIPELINE, program
)
```

**Why it is not a property.** A property is a claim the compiler can stand
behind, and this one cannot be. The check has to run after `DeriveCallDirections`
(pass 37) — a wrapper's signature legitimately reads `In` until then — and
`InitMemRef` (pass 31) declares `.invalidated = {IRProperty::SSAForm}` with
nothing re-establishing it. **No pipeline position is both after pass 37 and in
SSA form.** The buffer lineage below has no merging at a join, which is exact
only when each name has one definition, so on the IR it actually receives it can
both miss a write and attribute one to a buffer the write reaches on no path:

- a view built inside a branch leaks its lineage past the join, so a write after
  the branch may be blamed on a buffer only the taken path names; and
- `BufferRootCollector` scans the whole body up front, so a rebound name carries
  one final mapping that is applied to earlier writes too.

Both shapes are pinned in `tests/ut/ir/verifier/test_in_param_written.py`, the
first as a strict `xfail` so that fixing it fails the test. A report is a signal
to go and look; silence proves nothing. Making this sound means a real
control-flow dataflow analysis with join merging and candidate-set lineage —
separate work, and a fourth alias model in a tree that already has three.

Three choices are load-bearing:

- **On the finished program, not after one pass.** A Group/Spmd wrapper forwards
  its parameters to an inner kernel, and its own signature legitimately reads
  `In` for a parameter that kernel writes until `DeriveCallDirections` phase 0
  materialises the effective directions back into the IR. The invariant only
  holds once the pipeline is done.
- **Orchestration functions are skipped.** Their directions are the user's
  declaration and their parameters are the host ABI — a pure `Out` parameter is
  auto-allocated by the host in return-style calls, so flipping one is a
  migration the user makes, not an inference the compiler completes.
- **A warning, not an error.** It never invents a write, but it does report
  programs that compile and run today; the promotion path is the `IRProperty`
  above, once the report is empty.

**Zero-copy views are followed.** A write through a view of a parameter is a
write to the parameter:

```python
view = pl.tile.slice(acc, [8, 128], [0, 0])   # acc declared In
view = pl.tile.assemble(view, src, [0, 0])    # writes acc's buffer
```

Two shared declarations decide what a value names, and neither is a list kept
here: `ResultAliasedArgIndex` (the operator returns the argument it updated —
`tensor.assemble`, `tensor.write`, the collectives) and
`op_predicates::IsBufferAliasingViewOp`, which reads
`OutputMemoryInheritsInput() && IsInplaceSafe()` — the zero-copy views, which
update nothing and so declare no reuse contract. `tile.transpose` falls out of
the second by its own `not_inplace_safe()` registration: it permutes into a fresh
buffer, so its output is not an alias of its input, and any future inherit-input
op registered the same way is excluded without an edit here.

`tensor.slice` is one of those views, so a store into a slice of a parameter *is*
reported — even though `BufferRootCollector` deliberately maps it to a fresh
root. The chain is resolved inside the verifier rather than in that collector,
which three other passes share; widening what counts as an alias for all of them
is a separate change.

**SSA is a precondition.** The lineage is a single environment with no merging
at a join, which is sound exactly when each name has one definition — and
`PostPipeline` guarantees that. On pre-SSA input it is unsound in both
directions: a branch that re-points a name leaks its lineage past the join
(blaming a buffer the write does not reach when the branch is not taken), and a
plain `t = buf1; ...; t = buf2` gives `BufferRootCollector` one final mapping for
two different buffers. A direct caller must convert first.

Lineage is *not* carried across a phi (`return_vars_` / `iter_args_`), so a view
write after a branch or loop under-reports — the safe direction. The source is
resolved when the binding is recorded, so a lookup is a single map read and the
walk stays linear in the body.

**Fix**: declare the parameter `pl.Out` (written, never read) or `pl.InOut`
(read and written). Adding `.set_arg_effect(...)` is *not* the fix for a report
from this check — a builtin appears here only because its effect is already
declared, and a cross-function writer is a user function with no `REGISTER_OP`
block. A missing effect is the registry gap described above, which this check
cannot see.

### SSAVerify

**Error types** (`ssa::ErrorType`):

| Error Code | Name | Description |
| ---------- | ---- | ----------- |
| 1 | `MULTIPLE_ASSIGNMENT` | Variable assigned more than once in the same scope |
| 2 | `NAME_SHADOWING` | Variable name shadows an outer scope variable |
| 3 | `MISSING_YIELD` | ForStmt or IfStmt missing required YieldStmt |
| 4 | `ITER_ARGS_RETURN_VARS_MISMATCH` | iter_args count != return_vars count in ForStmt/WhileStmt |
| 5 | `YIELD_COUNT_MISMATCH` | YieldStmt value count != iter_args/return_vars count |
| 6 | `SCOPE_VIOLATION` | Variable used outside its defining scope |
| 7 | `MISPLACED_YIELD` | YieldStmt appears before the trailing position in its scope |

### TypeCheck

**Error types** (`typecheck::ErrorType`):

| Error Code | Name | Description |
| ---------- | ---- | ----------- |
| 101 | `TYPE_KIND_MISMATCH` | Type kind mismatch (e.g., ScalarType vs TensorType) |
| 102 | `DTYPE_MISMATCH` | Data type mismatch |
| 103 | `SHAPE_DIMENSION_MISMATCH` | Shape dimension count mismatch |
| 104 | `SHAPE_VALUE_MISMATCH` | Shape dimension value mismatch |
| 105 | `SIZE_MISMATCH` | Vector size mismatch in control flow |
| 106 | `IF_CONDITION_MUST_BE_SCALAR` | IfStmt/WhileStmt condition must be ScalarType |
| 107 | `FOR_RANGE_MUST_BE_SCALAR` | ForStmt range must be ScalarType |
| 108 | `CONDITION_MUST_BE_BOOL` | IfStmt/WhileStmt condition dtype must be BOOL |
| 109 | `TENSOR_PADDING_MISMATCH` | Tensor pad metadata mismatch |
| 110 | `DISTRIBUTED_WINDOW_IDENTITY_MISMATCH` | Distributed tensors refer to different window buffers |
| 111 | `TILE_VIEW_MISMATCH` | Effective TileView metadata mismatch |

### NoNestedCall

| Name | Description |
| ---- | ----------- |
| `CALL_IN_CALL_ARGS` | Call expression nested in another call's arguments |
| `CALL_IN_IF_CONDITION` | Call expression in if-statement condition |
| `CALL_IN_FOR_RANGE` | Call expression in for-loop range |
| `CALL_IN_BINARY_EXPR` | Call expression in binary expression |
| `CALL_IN_UNARY_EXPR` | Call expression in unary expression |

### UseAfterDefCheck

**Error types** (`use_after_def::ErrorType`):

| Error Code | Name | Description |
| ---------- | ---- | ----------- |
| 401 | `USE_BEFORE_DEF` | Variable referenced before any definition in the current scope |

**Scoping rules:**

- Function parameters are in scope for the entire function body
- `AssignStmt`: LHS variable enters scope after RHS is evaluated
- `ForStmt`: `loop_var` and `iter_args` are in scope inside the loop body only; `return_vars` enter the enclosing scope after the loop
- `WhileStmt`: `iter_args` are in scope for the condition and body; `return_vars` enter the enclosing scope after the loop
- `IfStmt`:
  - **SSA / phi-node form (`return_vars_` present)**: definitions inside then/else branches do **not** propagate to the outer scope; only `return_vars` enter the enclosing scope after the `if`
  - **Non-SSA "leak" form (`return_vars_` absent)**: branch-local definitions may be visible after the `if`; `ConvertToSSA` and `SSAVerify` are responsible for validating the resulting form

## PropertyVerifierRegistry

**Header**: `include/pypto/ir/verifier/property_verifier_registry.h`

Singleton registry mapping `IRProperty` values to `PropertyVerifier` factories. Used by `PassPipeline` to automatically verify properties before/after passes.

| Method | Description |
| ------ | ----------- |
| `GetInstance()` | Get singleton instance |
| `Register(prop, factory)` | Register a verifier factory for a property |
| `GetVerifier(prop)` | Create a verifier instance (nullptr if none registered) |
| `HasVerifier(prop)` | Check if a verifier is registered |
| `VerifyProperties(properties, program)` | Verify a set of properties, return diagnostics |
| `VerifyOrThrow(properties, program)` | Verify and throw VerificationError on errors |
| `GenerateReport(diagnostics)` | Static — format diagnostics into readable report |

## C++ API Reference

### PropertyVerifier Interface

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `GetName()` | `std::string GetName() const` | Return unique rule identifier |
| `Verify()` | `void Verify(const ProgramPtr&, std::vector<Diagnostic>&)` | Check program and append diagnostics |

### Structural and Default Properties

| Function | Returns | Description |
| -------- | ------- | ----------- |
| `GetStructuralProperties()` | `{TypeChecked, BreakContinueValid, NoRedundantBlocks, UseAfterDef, OutParamNotShadowed, NoNestedInCore, InOutUseValid, PipelineLoopValid, ArrayNotEscaped, ManualDepsOnSubmitOnly, AtomicAddDtypeValid}` | Invariants verified before/after each pass by `VerificationInstrument` (the subset shared with `GetVerifiedProperties()` is also checked at pipeline start) |
| `GetDefaultVerifyProperties()` | `{SSAForm, TypeChecked, NoNestedCalls, BreakContinueValid, NoRedundantBlocks, UseAfterDef, OutParamNotShadowed, NoNestedInCore, TileTypeCoherence, ArrayNotEscaped}` | Default set for `run_verifier()` |
| `GetVerifiedProperties()` | `{SSAForm, TypeChecked, MixedKernelExpanded, AllocatedMemoryAddr, BreakContinueValid, NoRedundantBlocks, InOutUseValid, CallDirectionsResolved, ManualDepsOnSubmitOnly, ReturnParamsExplicit, AivSplitValid, TileMemoryInferred, HardSyncallOccupancyValid, IterArgCarryClassified, RuntimeScopesMaterialized, DistTensorCtxMaterialized, GraphBoundaryLegalized, AccToGmStoreValid, AccCompactValid, AtomicAddDtypeValid}` | Lightweight set for `PassPipeline` auto-verify |

### RunVerifier Pass Factory

```cpp
Pass RunVerifier(const IRPropertySet& properties);
```

Creates a `Pass` that verifies the given properties using `PropertyVerifierRegistry`.

## Python API Reference

**Module**: `pypto.pypto_core.passes`

### PropertyVerifierRegistry

| Method | Parameter | Returns | Description |
| ------ | --------- | ------- | ----------- |
| `verify(properties, program)` | `IRPropertySet, Program` | `list[Diagnostic]` | Collect diagnostics |
| `verify_or_throw(properties, program)` | `IRPropertySet, Program` | `None` | Throw on error |
| `generate_report(diagnostics)` | `list[Diagnostic]` | `str` | Format diagnostics |

### Helper Functions

| Function | Returns | Description |
| -------- | ------- | ----------- |
| `get_default_verify_properties()` | `IRPropertySet` | Default properties for `run_verifier()` |
| `get_structural_properties()` | `IRPropertySet` | Structural invariant properties |

### run_verifier Function

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `properties` | `IRPropertySet \| None` | `None` | Properties to verify (None → default set) |
| **Returns** | `Pass` | - | Verifier Pass object |

## Usage Examples

### Basic Verification

```python
from pypto.pypto_core import passes

# Verify default properties
props = passes.get_default_verify_properties()
diagnostics = passes.PropertyVerifierRegistry.verify(props, program)

if diagnostics:
    report = passes.PropertyVerifierRegistry.generate_report(diagnostics)
    print(report)
```

### Selective Verification

```python
# Verify only specific properties
props = passes.IRPropertySet()
props.insert(passes.IRProperty.SSAForm)
props.insert(passes.IRProperty.TypeChecked)
diagnostics = passes.PropertyVerifierRegistry.verify(props, program)
```

### Disabling Checks

```python
# Start from default set and remove what you don't want
props = passes.get_default_verify_properties()
props.remove(passes.IRProperty.SSAForm)
diagnostics = passes.PropertyVerifierRegistry.verify(props, program)
```

### Error Handling with Exceptions

```python
props = passes.get_default_verify_properties()
try:
    passes.PropertyVerifierRegistry.verify_or_throw(props, program)
    print("Program is valid")
except Exception as e:
    print(f"Verification failed: {e}")
```

### Using in a Custom Pipeline

```python
# Create verifier pass (defaults to get_default_verify_properties())
verify_pass = passes.run_verifier()
result = verify_pass(program)

# Or with custom properties
props = passes.get_default_verify_properties()
props.remove(passes.IRProperty.SSAForm)
verify_pass = passes.run_verifier(properties=props)
result = verify_pass(program)
```

## Adding Custom Rules

### Implementation Steps

1. Inherit from `PropertyVerifier`, implement `GetName()` and `Verify()`
2. Create a factory function returning `PropertyVerifierPtr`
3. Register with `PropertyVerifierRegistry` in the constructor
4. Add Python binding and type stub (optional)

### Guidelines

- Use `IRVisitor` to traverse IR nodes systematically
- Keep rules focused — one rule checks one category of issues
- Avoid side effects — only read IR and write diagnostics
- Create descriptive diagnostics with severity, rule name, error code, message, and span

## Related Components

- **Pass System** (`00-pass_manager.md`): Verifier integrates as a Pass, PropertyVerifierRegistry used by PassPipeline
- **IRBuilder** (`../ir/06-builder.md`): Construct IR that verifier validates
- **Type System** (`../ir/02-types.md`): TypeCheck rule validates against type system
- **Error Handling** (`../02-error-handling.md`): Exception hierarchy, assertion macros (`CHECK`, `INTERNAL_CHECK_SPAN`), and `Diagnostic` / `VerificationError` definitions

## Testing

Test coverage in `tests/ut/ir/transforms/test_verifier.py`: valid/invalid program verification, property-based selection, exception vs. diagnostic modes, pass integration, diagnostic field access, report generation, structural/default property sets.

UseAfterDef-specific coverage in `tests/ut/ir/transforms/test_verify_use_after_def.py`: valid programs (params, chained assigns, for loop body, return_var after loop), invalid programs (use-before-def, loop var out of scope, branch def not visible outside), error code/rule name verification, structural property membership.
