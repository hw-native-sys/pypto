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
| 09 | *(no doc yet)* | 9th pass (`ConvertTensorToTileOps`) |
| 10 | `10-flatten_tile_nd_to_2d.md` | 10th pass |
| 11 | `11-expand_mixed_kernel.md` | Reserved for `ExpandMixedKernel` (planned, not in any strategy yet) |
| 12 | `12-init_memref.md` | 12th pass |
| 13 | `13-basic_memory_reuse.md` | 13th pass |
| 14 | `14-insert_sync.md` | 14th pass (CCE strategy only) |
| 15 | `15-allocate_memory_addr.md` | 15th pass |
| 16 | `16-utility_passes.md` | Not in default strategy |
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
