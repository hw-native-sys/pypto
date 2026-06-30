# Pass Documentation Ordering

## Rule

Pass documentation files in `docs/en/dev/passes/` (and `docs/zh-cn/dev/passes/`) must be numbered to match the pass execution order in the pass manager (`python/pypto/ir/pass_manager.py`).

## Why

Developers read pass docs sequentially to understand the compilation pipeline. If numbering doesn't match execution order, the reading experience is confusing.

## Current Order

| Number | File | Pass Manager Position |
| ------ | ---- | --------------------- |
| 00 | `00-pass_manager.md` | Overview (not a pass) |
| 01 | `01-inline_functions.md` | 1st pass |
| 02 | `02-unroll_loops.md` | 2nd pass |
| 03 | `03-ctrl_flow_transform.md` | 3rd pass |
| 04 | `04-convert_to_ssa.md` | 4th pass |
| 05 | `05-simplify.md` | 5th pass (also runs as the last pass of the tile pipeline) |
| 06 | `06-flatten_call_expr.md` | 6th pass |
| 07 | `07-outline_hierarchy_scopes.md` | 7th pass |
| 08 | `08-outline_incore_scopes.md` | 8th pass |
| 09 | `09-outline_cluster_scopes.md` | 9th pass |
| 10 | `10-convert_tensor_to_tile_ops.md` | 10th pass |
| 11 | `11-optimize_orch_tensors.md` | 11th pass |
| 12 | `12-lower_composite_ops.md` | 12th pass (first tile_pto pass) |
| 13 | `13-flatten_tile_nd_to_2d.md` | 13th pass |
| 14 | `14-auto_tile_matmul_l0.md` | 14th pass |
| 15 | `15-canonicalize_tile_slice.md` | Runs immediately after `AutoTileMatmulL0` (lowers Mat/Vec `tile.slice` → `tile.extract`) |
| 16 | `16-infer_tile_memory_space.md` | 16th pass |
| 17 | `17-resolve_backend_op_layouts.md` | 17th pass |
| 18 | `18-lower_auto_vector_split.md` | Live auto-split lowering path; converts AUTO `pl.split` mixed InCore functions into the explicit `split_aiv` form (aiv_shard/aic_gather + halved vector sub-region). Runs immediately before `ExpandMixedKernel` |
| 19 | `19-expand_mixed_kernel.md` | 19th pass |
| 20 | `20-inject_gm_pipe_buffer.md` | Runs immediately after `ExpandMixedKernel` (backend-gated, Ascend910B) |
| 21 | `21-split_vector_kernel.md` | 21st pass (after the convergence refactor: only stamps attrs for split_aiv functions + handles the no-split dual-AIV path; the per-op halving driver was deleted — moved to LowerAutoVectorSplit + split_axis_utils) |
| 22 | `22-stamp_tfree_split.md` | 22nd pass (copies each cross-core tpop's split/pipe-id onto its matching tfree op; runs right after SplitVectorKernel finalizes split, before SkewCrossCorePipeline clones tpop/tfree pairs) |
| 23 | `23-normalize_return_order.md` | 23rd pass |
| 24 | `24-skew_cross_core_pipeline.md` | 24th pass (cross-core cube/vector software-pipeline skew; runs immediately before LowerPipelineLoops) |
| 25 | `25-lower_pipeline_loops.md` | 25th pass |
| 26 | `26-canonicalize_io_order.md` | 26th pass |
| 27 | `27-materialize_tensor_strides.md` | 27th pass (RFC #1300 P3 — wired into Default starting from P6) |
| 28 | `28-init_memref.md` | 28th pass |
| 29 | `29-memory_reuse.md` | 29th pass (also enforces the Ascend910B load + tpop_from_aic in-place hazard guard) |
| 30 | `30-allocate_memory_addr.md` | 30th pass |
| 31 | `31-fold_no_op_reshape.md` | 31st pass |
| 32 | `32-fuse_create_assemble_to_slice.md` | 32nd pass |
| 33 | `33-derive_call_directions.md` | 33rd pass (two-phase: arg directions + manual-scope lowering) |
| 34 | `34-auto_derive_task_dependencies.md` | 34th pass (manual-scope compiler deps; opt-in AUTO-scope analysis/emission via compile-time switch; default behavior unchanged) |
| 35 | `35-expand_manual_phase_fence.md` | 35th pass (manual-scope phase-fence TaskId dep compression; runs after AutoDeriveTaskDependencies) |
| 36 | `36-materialize_comm_domain_scopes.md` | 36th pass (distributed: WindowBuffer + CommDomainScopeStmt wrappers in each host_orch body; runs immediately before LowerHostTensorCollectives) |
| 37 | `37-lower_host_tensor_collectives.md` | 37th pass (host-level tensor collectives -> internal builtin chip dispatches; runs after comm-domain scopes) |
| 38 | `38-materialize_runtime_scopes.md` | Last pass (after the final Simplify; inserts AUTO RuntimeScopeStmt so orchestration codegen emits PTO2_SCOPE 1:1) |
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
4. Update both `docs/en/dev/passes/` and `docs/zh-cn/dev/passes/`
5. Update any cross-references in other docs

## When Reordering Passes

If the pass manager execution order changes, renumber the doc files to match.
