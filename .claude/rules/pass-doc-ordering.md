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
| 07 | `07-split_chunked_loops.md` | 7th pass |
| 08 | `08-interchange_chunk_loops.md` | 8th pass |
| 09 | `09-outline_hierarchy_scopes.md` | 9th pass |
| 10 | `10-outline_incore_scopes.md` | 10th pass |
| 11 | `11-outline_cluster_scopes.md` | 11th pass |
| 12 | `12-convert_tensor_to_tile_ops.md` | 12th pass |
| 13 | `13-optimize_orch_tensors.md` | 13th pass |
| 14 | `14-lower_composite_ops.md` | 14th pass (first tile_pto pass) |
| 15 | `15-flatten_tile_nd_to_2d.md` | 15th pass |
| 16 | `16-auto_tile_matmul_l0.md` | 16th pass |
| 17 | `17-canonicalize_mat_slice.md` | Runs immediately after `AutoTileMatmulL0` (lowers Mat `tile.slice` → `tile.extract`) |
| 18 | `18-infer_tile_memory_space.md` | 18th pass |
| 19 | `19-lower_transpose_load_param_layout.md` | 19th pass (RFC #1300 P6 — replaces ResolveTransposeLayout) |
| 20 | `20-resolve_backend_op_layouts.md` | 20th pass |
| 21 | `21-expand_mixed_kernel.md` | 21st pass |
| 22 | `22-inject_gm_pipe_buffer.md` | Runs immediately after `ExpandMixedKernel` (backend-gated, Ascend910B) |
| 23 | `23-split_vector_kernel.md` | 23rd pass |
| 24 | `24-normalize_return_order.md` | 24th pass |
| 25 | `25-lower_pipeline_loops.md` | 25th pass |
| 26 | `26-canonicalize_io_order.md` | 26th pass |
| 27 | `27-materialize_tensor_strides.md` | 27th pass (RFC #1300 P3 — wired into Default starting from P6) |
| 28 | `28-init_memref.md` | 28th pass |
| 29 | `29-memory_reuse.md` | 29th pass |
| 30 | `30-legalize_pto_buffer_reuse.md` | 30th pass |
| 31 | `31-allocate_memory_addr.md` | 31st pass |
| 32 | `32-fold_no_op_reshape.md` | 32nd pass |
| 33 | `33-fuse_create_assemble_to_slice.md` | 33rd pass |
| 34 | `34-derive_call_directions.md` | 34th pass (two-phase: arg directions + manual-scope lowering) |
| 35 | `35-auto_derive_task_dependencies.md` | 35th pass (manual-scope compiler deps; no-op for auto scope) |
| 36 | `36-collect_comm_groups.md` | 36th pass (distributed: WindowBuffer + Program.comm_groups_; runs immediately before the final Simplify) |
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
