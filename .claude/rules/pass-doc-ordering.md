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
| 02 | `02-convert_to_ssa.md` | 2nd pass |
| 03 | `03-flatten_call_expr.md` | 3rd pass |
| 04 | `04-split_chunked_loops.md` | 4th pass |
| 05 | `05-interchange_chunk_loops.md` | 5th pass |
| 06 | `06-outline_incore_scopes.md` | 6th pass |
| 07 | *(no doc yet)* | 7th pass (`ConvertTensorToTileOps`) |
| 08 | `08-init_memref.md` | 8th pass |
| 09 | `09-basic_memory_reuse.md` | 9th pass |
| 10 | `10-insert_sync.md` | 10th pass (Default only) |
| 11 | `11-allocate_memory_addr.md` | 11th pass |
| 12 | `12-utility_passes.md` | Not in default strategy |
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
