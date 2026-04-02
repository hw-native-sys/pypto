# Pass Documentation Ordering

## Rule

Pass documentation files in `docs/en/dev/passes/` (and `docs/zh-cn/dev/passes/`) must be numbered to match the pass execution order in the pass manager (`python/pypto/ir/pass_manager.py`).

## Why

Developers read pass docs sequentially to understand the compilation pipeline. If numbering doesn't match execution order, the reading experience is confusing.

## Current Order (Default strategy)

Order follows `OptimizationStrategy.Default`: `tensor_prefix_passes` + `tensor_only_passes` + `tile_pto_passes`.

| Pos | Pass | Doc file (when present) |
| --- | ---- | ----------------------- |
| 00 | *(overview)* | `00-pass_manager.md` |
| 1 | `UnrollLoops` | `01-unroll_loops.md` |
| 2 | `CtrlFlowTransform` | `02-ctrl_flow_transform.md` |
| 3 | `ConvertToSSA` | `03-convert_to_ssa.md` |
| 4 | `NormalizeStmtStructure` | *(no standalone doc)* |
| 5 | `FlattenCallExpr` | `04-flatten_call_expr.md` |
| 6 | `SplitChunkedLoops` | `05-split_chunked_loops.md` |
| 7 | `InterchangeChunkLoops` | `06-interchange_chunk_loops.md` |
| 8 | `OutlineHierarchyScopes` | *(no doc yet)* |
| 9 | `OutlineIncoreScopes` | `07-outline_incore_scopes.md` |
| 10 | `OutlineClusterScopes` | `08-outline_cluster_scopes.md` |
| 11 | `ConvertTensorToTileOps` | *(no doc yet)* |
| 12 | `FlattenTileNdTo2D` | `10-flatten_tile_nd_to_2d.md` |
| 13 | `InferTileMemorySpace` | *(no doc yet)* |
| 14 | `ResolveTransposeLayout` | *(no doc yet)* |
| 15 | `ResolveBackendOpLayouts` | *(no doc yet)* |
| 16 | `NormalizeStmtStructure` | *(second run; no standalone doc)* |
| 17 | `ExpandMixedKernel` | `11-expand_mixed_kernel.md` *(legacy prefix 11)* |
| 18 | `SplitVectorKernel` | *(no doc yet)* |
| 19 | `NormalizeTupleReturnOrder` | `19-normalize_tuple_return_order.md` |
| 20 | `InitMemRef` | `12-init_memref.md` *(legacy prefix 12)* |
| 21 | `MemoryReuse` | `15-memory_reuse.md` *(legacy prefix 15)* |
| 22 | `LegalizePTOBufferReuse` | *(no doc yet)* |
| 23 | `AllocateMemoryAddr` | `15-allocate_memory_addr.md` *(legacy prefix 15)* |

### Not in Default strategy

- `14-insert_sync.md` — `InsertSync`
- `16-utility_passes.md` — Utility passes

### Infrastructure (not a pipeline pass)

- `99-verifier.md` — Verifier / property checks

**Legacy numbering**: Some doc filenames still use older prefixes (e.g. `11-` for pass 17). When adding or renaming docs, prefer matching `pass_manager.py` position; renumber with `git mv` when touching those files.

**Gaps**: When a pass has no documentation yet, reserve its number in this table or mark *(no doc yet)*.

## When Adding a New Pass

1. Check where the pass appears in `pass_manager.py` default strategy
2. Assign the doc file number matching that execution position
3. Renumber subsequent files if needed (use `git mv` with temp names to avoid collisions)
4. Update both `docs/en/dev/passes/` and `docs/zh-cn/dev/passes/`
5. Update any cross-references in other docs

## When Reordering Passes

If the pass manager execution order changes, renumber the doc files to match.
