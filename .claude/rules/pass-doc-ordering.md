# Pass Documentation Ordering

## Rule

Pass documentation files in `docs/en/dev/passes/` (and `docs/zh-cn/dev/passes/`) must be numbered to match the pass execution order in the pass manager (`python/pypto/ir/pass_manager.py`).

## Why

Developers read pass docs sequentially to understand the compilation pipeline. If numbering doesn't match execution order, the reading experience is confusing.

## Current Order

| Number | File | Pass Manager Position |
| ------ | ---- | --------------------- |
| 00 | `00-pass_manager.md` | Overview (not a pass) |
| 01 | `01-unroll_loops.md` | 1st pass |
| 02 | `02-ctrl_flow_transform.md` | 2nd pass |
| 03 | `03-convert_to_ssa.md` | 3rd pass |
| 04 | `04-simplify.md` | 4th pass (also runs as the last pass of the tile pipeline) |
| 05 | `05-flatten_call_expr.md` | 5th pass |
| 06 | `06-split_chunked_loops.md` | 6th pass |
| 07 | `07-interchange_chunk_loops.md` | 7th pass |
| 08 | `08-outline_hierarchy_scopes.md` | 8th pass |
| 09 | `09-outline_incore_scopes.md` | 9th pass |
| 10 | `10-outline_cluster_scopes.md` | 10th pass |
| 11 | `11-convert_tensor_to_tile_ops.md` | 11th pass |
| 12 | `12-optimize_orch_tensors.md` | 12th pass |
| 13 | `13-flatten_tile_nd_to_2d.md` | 13th pass |
| 14 | `14-infer_tile_memory_space.md` | 14th pass |
| 15 | `15-resolve_transpose_layout.md` | 15th pass |
| 16 | `16-resolve_backend_op_layouts.md` | 16th pass |
| 17 | `17-expand_mixed_kernel.md` | 17th pass |
| 18 | `18-inject_gm_pipe_buffer.md` | Runs immediately after `ExpandMixedKernel` (backend-gated, Ascend910B) |
| 19 | `19-split_vector_kernel.md` | 19th pass |
| 20 | `20-normalize_return_order.md` | 20th pass |
| 21 | `21-lower_pipeline_loops.md` | 21st pass |
| 22 | `22-canonicalize_io_order.md` | 22nd pass |
| 23 | `23-init_memref.md` | 23rd pass |
| 24 | `24-memory_reuse.md` | 24th pass |
| 25 | `25-legalize_pto_buffer_reuse.md` | 25th pass |
| 26 | `26-allocate_memory_addr.md` | 26th pass |
| 27 | `27-fuse_create_assemble_to_slice.md` | 27th pass |
| 28 | `28-derive_call_directions.md` | 28th pass |
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
