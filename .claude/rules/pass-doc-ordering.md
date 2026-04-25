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
| 04 | `04-flatten_call_expr.md` | 4th pass |
| 05 | `05-split_chunked_loops.md` | 5th pass |
| 06 | `06-interchange_chunk_loops.md` | 6th pass |
| 07 | `07-outline_incore_scopes.md` | 7th pass |
| 08 | `08-outline_cluster_scopes.md` | 8th pass |
| 09 | `09-convert_tensor_to_tile_ops.md` | 9th pass |
| 10 | `10-optimize_orch_tensors.md` | 10th pass |
| 11 | `11-flatten_tile_nd_to_2d.md` | 11th pass |
| 12 | *(no doc yet)* | 12th pass (`InferTileMemorySpace`) |
| 13 | *(no doc yet)* | 13th pass (`ResolveTransposeLayout`) |
| 14 | `14-expand_mixed_kernel.md` | 14th pass |
| 15 | `15-inject_gm_pipe_buffer.md` | Runs immediately after `ExpandMixedKernel` (backend-gated, Ascend910B) |
| 16 | `16-init_memref.md` | 16th pass |
| 17 | `17-memory_reuse.md` | 17th pass |
| 18 | `18-allocate_memory_addr.md` | 18th pass |
| 91 | `91-utility_passes.md` | Not in Default strategy |
| 99 | `99-verifier.md` | Infrastructure (not a pipeline pass) |

**Gaps**: When a pass has no documentation yet, reserve its number and note it in the table. This keeps subsequent numbering aligned with execution order.

## When Adding a New Pass

1. Check where the pass appears in `pass_manager.py` default strategy
2. Assign the doc file number matching that execution position
3. Renumber subsequent files if needed (use `git mv` with temp names to avoid collisions)
4. Update both `docs/en/dev/passes/` and `docs/zh-cn/dev/passes/`
5. Update any cross-references in other docs

## When Reordering Passes

If the pass manager execution order changes, renumber the doc files to match.
