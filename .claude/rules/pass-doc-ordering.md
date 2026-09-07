# Pass Documentation Ordering

## Rule

Pass documentation files in `docs/en/dev/passes/` (and `docs/zh/dev/passes/`) must be numbered to match the pass execution order in the pass manager (`python/pypto/ir/pass_manager.py`).

## Why

Developers read pass docs sequentially to understand the compilation pipeline. If numbering doesn't match execution order, the reading experience is confusing.

## Current Order

| Number | File | Pass Manager Position |
| ------ | ---- | --------------------- |
| 00 | `00-pass_manager.md` | Overview (not a pass) |
| 01 | `01-inline_functions.md` | 01st pass |
| 02 | `02-unroll_loops.md` | 02nd pass |
| 03 | `03-ctrl_flow_transform.md` | 03rd pass |
| 04 | `04-convert_to_ssa.md` | 04th pass |
| 05 | `05-simplify.md` | 05th pass (also runs as the last pass of the tile pipeline) |
| 06 | `06-flatten_call_expr.md` | 06th pass |
| 07 | `07-outline_hierarchy_scopes.md` | 07th pass |
| 08 | `08-outline_graph_scopes.md` | Outlines `pl.graph` regions into `FunctionType::Graph` functions; runs immediately before `OutlineIncoreScopes` so the InCore scopes inside a region are outlined on the same terms as those in a hand-written `@pl.jit.graph` function |
| 09 | `09-outline_incore_scopes.md` | 09th pass |
| 10 | `10-outline_cluster_scopes.md` | 10th pass |
| 11 | `11-convert_tensor_to_tile_ops.md` | 11th pass |
| 12 | `12-optimize_orch_tensors.md` | 12th pass |
| 13 | `13-lower_composite_ops.md` | 13th pass (first tile_pto pass) |
| 14 | `14-flatten_tile_nd_to_2d.md` | 14th pass |
| 15 | `15-block_nz_tensor_views.md` | Rewrites a logical `pl.NZ` tensor into pto-isa's blocked rank-(r+2) shape `[..., C/c0, R/16, 16, c0]` and retargets its `tile.load` coordinates, keeping the destination tile logical 2D. Runs immediately after `FlattenTileNdTo2D` (which skips its ND2NZ window collapse for NZ sources) and before `MaterializeTensorStrides`, whose plain row-major rule then yields pto-isa's NZ strides |
| 16 | `16-block_mx_scale_tensor_views.md` | Physicalizes logical MX scale tensor views into A5's packed rank-5 SFractal form and retargets their `tile.load` coordinates; runs independently after `BlockNzTensorViews` and before `LegalizeTileCast` |
| 17 | `17-legalize_tile_cast.md` | Expands `tile.cast` pairs the target ISA cannot emit as one `pto.tcvt` into the shortest chain of native casts (A5 `INT32->FP16` becomes `INT32->FP32->FP16`); runs between `BlockMxScaleTensorViews` and `AutoTileMatmulL0` |
| 18 | `18-auto_tile_matmul_l0.md` | 18th pass |
| 19 | `19-canonicalize_tile_slice.md` | Runs immediately after `AutoTileMatmulL0` (lowers Mat/Vec `tile.slice` → `tile.extract`) |
| 20 | `20-infer_tile_memory_space.md` | 20th pass |
| 21 | `21-insert_mx_scale_addr.md` | Inserts `tile.tget_scale_addr` before MX matmul consumers after InferTileMemorySpace resolves their memory spaces |
| 22 | `22-resolve_backend_op_layouts.md` | 22nd pass |
| 23 | `23-lower_auto_vector_split.md` | Live auto-split lowering path; converts AUTO `pl.split` mixed InCore functions into the explicit `split_aiv` form (aiv_shard/aic_gather + halved vector sub-region). ALSO the sole consumer of the first-class `SplitAivScopeStmt` region node (`pl.split_aiv`, nestable/multi-mode): lowers each region in place (region-scoped halving; explicit-boundary bodies passed through unchanged) and erases the scope wrapper. Runs immediately before `ExpandMixedKernel` |
| 24 | `24-expand_mixed_kernel.md` | 24th pass (no `SplitAivScopeStmt` survives to here; its single-func-mode transpose check is skipped for functions stamped `split_aiv_region_validated` by pass 23) |
| 25 | `25-inject_gm_pipe_buffer.md` | Runs immediately after `ExpandMixedKernel` (backend-gated, Ascend910B) |
| 26 | `26-split_vector_kernel.md` | 26th pass (after the convergence refactor: only stamps attrs for split_aiv functions + handles the no-split dual-AIV path; the per-op halving driver was deleted — moved to LowerAutoVectorSplit + split_axis_utils. Single-func-mode assertion relaxed for multi-mode `split_aiv` functions: stamps the mode-agnostic `dual_aiv_dispatch` and trusts the per-op `split` ints from pass 23) |
| 27 | `27-stamp_tfree_split.md` | 27th pass (copies each cross-core tpop's split/pipe-id onto its matching tfree op; runs right after SplitVectorKernel finalizes split, before SkewCrossCorePipeline clones tpop/tfree pairs) |
| 28 | `28-normalize_return_order.md` | 28th pass |
| 29 | `29-skew_cross_core_pipeline.md` | 29th pass (cross-core cube/vector software-pipeline skew; runs immediately before LowerPipelineLoops) |
| 30 | `30-lower_pipeline_to_slots.md` | Rotates an eligible `pl.pipeline` body through the slots of one allocation instead of replicating it; self-gated on `memory_planner=PTOAS`, and every loop it declines is left for LowerPipelineLoops |
| 31 | `31-lower_pipeline_loops.md` | 31st pass |
| 32 | `32-canonicalize_io_order.md` | 32nd pass |
| 33 | `33-materialize_tensor_strides.md` | 33rd pass (RFC #1300 P3 — wired into Default starting from P6) |
| 34 | `34-init_memref.md` | 34th pass |
| 35 | `35-materialize_semantic_aliases.md` | Semantics-required must-alias (loop-carry / in-place); split out of MemoryReuse (its former "Step 0"); always runs, even when MemoryReuse is skipped under `memory_planner=PTOAS` |
| 36 | `36-memory_reuse.md` | Opportunistic lifetime reuse (also enforces the Ascend910B load + tpop_from_aic in-place hazard guard); skippable under `memory_planner=PTOAS` |
| 37 | `37-allocate_memory_addr.md` | 37th pass (skippable under `memory_planner=PTOAS`) |
| 38 | `38-fold_no_op_reshape.md` | 38th pass |
| 39 | `39-fuse_create_assemble_to_slice.md` | 39th pass |
| 40 | `40-derive_call_directions.md` | 40th pass (two-phase: arg directions + manual-scope lowering) |
| 41 | `41-auto_derive_task_dependencies.md` | 41st pass (manual-scope compiler deps; opt-in AUTO-scope analysis/emission via compile-time switch; default behavior unchanged) |
| 42 | `42-expand_manual_phase_fence.md` | 42nd pass (manual-scope phase-fence TaskId dep compression; runs after AutoDeriveTaskDependencies) |
| 43 | `43-synthesize_allreduce_signals.md` | 43rd pass (distributed: host allreduce optional signal -> explicit internal signal IR) |
| 44 | `44-materialize_comm_domain_scopes.md` | 44th pass (distributed: WindowBuffer + CommDomainScopeStmt wrappers in each host_orch body; runs immediately before LowerHostTensorCollectives) |
| 45 | `45-lower_host_tensor_collectives.md` | 45th pass (host-level tensor collectives -> internal builtin chip dispatches; runs after comm-domain scopes) |
| 46 | `46-materialize_dist_tensor_ctx.md` | 46th pass (materializes explicit CommCtx params/args for DistributedTensor params; runs before the final Simplify) |
| 47 | `47-legalize_graph_boundary.md` | Hoists the boundary scalars a `FunctionType::Graph` body derives out to its call sites (a derived scalar has no runtime argument slot, so replay would freeze the first call's value) and rejects boundaries the host_build_graph runtime could not record; runs after the final Simplify, before MaterializeRuntimeScopes |
| 48 | `48-materialize_runtime_scopes.md` | Runs after the final Simplify; inserts AUTO RuntimeScopeStmt so orchestration codegen emits SIMPLER_SCOPE 1:1 |
| 49 | `49-classify_iter_arg_carry.md` | Classifies each Orchestration ForStmt iter_arg (trivial alias vs materialised rebind carry) and sizes manual-scope TaskId array carries; runs after MaterializeRuntimeScopes |
| 50 | `50-insert_comm_fence.md` | Last pass (distributed: inserts a whole-tensor system.cacheinvalid + GM system.fence between each publishing write and the pld.system.notify that releases it; runs after all statement-reordering passes so the inserted ops stay adjacent to notify through codegen) |
| 51 | `51-materialize_valid_shape_symbols.md` | Runs dead last; turns each device-kernel `valid_shape` symbol the kernel cannot bind (not a physical dim, not a scalar param) into a leading `Scalar[INDEX]` param fed from the call site's actual valid extent |
| 91 | `91-utility_passes.md` | Not in Default strategy |
| 99 | `99-verifier.md` | Infrastructure (not a pipeline pass) |

**Gaps**: When a pass has no documentation yet, reserve its number and note it in the table. This keeps subsequent numbering aligned with execution order.

## Numbering scope: pipeline passes only

The main `01-89` sequence numbers **pipeline passes** — those that appear once in the `Default` strategy and have a dedicated per-pass doc. Two categories are intentionally excluded from the main sequence:

- **Utility passes** that may run at multiple positions in the pipeline (e.g. `NormalizeStmtStructure`, which runs both as the 5th and 18th entry in `pass_manager.py`). Giving them a single slot in the main sequence would misrepresent execution order; reserving every invocation would make the sequence harder to read. They are documented together in `91-utility_passes.md`.
- **Infrastructure** that is not a pipeline pass at all (e.g. the verifier registry in `99-verifier.md`).

The `90+` range is reserved for these excluded categories. Pipeline passes always live in `01-89`.

## When Adding a New Pass

1. Check where the pass appears in `pass_manager.py` default strategy
2. Assign the doc file number matching that execution position
3. Renumber subsequent files if needed (use `git mv` with temp names to avoid collisions)
4. Update both `docs/en/dev/passes/` and `docs/zh/dev/passes/`
5. Update any cross-references in other docs

## When Reordering Passes

If the pass manager execution order changes, renumber the doc files to match.
