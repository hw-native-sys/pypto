# Passes

Reference for PyPTO IR transformations, organized for reading. Consult the pass
manager for execution order.

## Documentation numbering

The tables below are the current pass documentation index. Use the pass manager
implementation (`python/pypto/ir/pass_manager.py`) to verify execution order and
the conditions under which optional or backend-dependent passes run.

- `00` is the pass manager overview.
- `01`–`89` are reserved for pipeline pass pages, ordered by the default pipeline.
  Optional passes are identified in their entries and individual pages.
- `90` and above are reserved for utility passes and infrastructure. Utility
  passes that run at multiple positions are documented together rather than
  receiving a page number for every invocation.
- Documentation numbers are reading-order labels, not absolute execution slots:
  repeated utility passes and conditional passes can make those numbers differ.
- If a pipeline pass has no page yet, reserve its number and note the gap in the
  index so subsequent page numbers remain aligned.

When adding, removing, or reordering passes, update the affected page numbers,
this index, the pass manager overview, site navigation, and cross-references.
Keep the English and Chinese documentation synchronized.

## Framework

| Page | What it covers |
| ---- | -------------- |
| [Pass, PassContext, PassPipeline, and PassManager](00-pass_manager.md) | Organizing and executing passes with property tracking, instrumentation, and strategy-based pipelines |

## Default pipeline

| Order | Pass | What it does |
| ----- | ---- | ------------ |
| 01 | [OutlineSimtScopes](01-outline_simt_scopes.md) | Outlines per-thread scopes into canonical helpers and launches before SSA |
| 02 | [InlineFunctions](02-inline_functions.md) | Splices `FunctionType.Inline` bodies into every call site |
| 03 | [UnrollLoops](03-unroll_loops.md) | Expands `ForKind::Unroll` loops at compile time |
| 04 | [CtrlFlowTransform](04-ctrl_flow_transform.md) | Rewrites `break` / `continue` into structured control flow |
| 05 | [ConvertToSSA](05-convert_to_ssa.md) | Converts to SSA form with variable renaming, phi nodes, and iter_args |
| 06 | [Simplify](06-simplify.md) | Folds arithmetic, shape expressions, and scalar constant bindings |
| 07 | [FlattenCallExpr](07-flatten_call_expr.md) | Flattens nested call expressions into three-address form |
| 08 | [OutlineHierarchyScopes](08-outline_hierarchy_scopes.md) | Outlines Hierarchy scopes into functions carrying `level` / `role` metadata |
| 09 | [OutlineGraphScopes](09-outline_graph_scopes.md) | Outlines `pl.graph` regions into `FunctionType.Graph` functions, converging the scope form with `@pl.jit.graph` |
| 10 | [OutlineIncoreScopes](10-outline_incore_scopes.md) | Outlines InCore scopes into separate functions |
| 11 | [OutlineClusterScopes](11-outline_cluster_scopes.md) | Outlines Cluster scopes into Group functions and standalone Spmd scopes into Spmd functions |
| 12 | [DeduplicateOutlinedKernels](12-deduplicate_outlined_kernels.md) | Reuses closed outlined kernels and exclusive Group bundles while preserving every dispatch |
| 13 | [ConvertTensorToTileOps](13-convert_tensor_to_tile_ops.md) | Converts tensor ops to tile ops in InCore functions, updating orchestration call sites |
| 14 | [OptimizeOrchTensors](14-optimize_orch_tensors.md) | Eliminates redundant orchestration allocations and improves data flow |
| 15 | [LowerCompositeOps](15-lower_composite_ops.md) | Decomposes composite tile / distributed ops into primitives |
| 16 | [FlattenTileNdTo2D](16-flatten_tile_nd_to_2d.md) | Flattens 3D+ tile operations to 2D by merging all but the last dimension |
| 17 | [BlockNzTensorViews](17-block_nz_tensor_views.md) | Rewrites logical `pl.NZ` tensors into pto-isa's blocked rank-5 form and retargets their `tile.load` coordinates |
| 18 | [BlockMxScaleTensorViews](18-block_mx_scale_tensor_views.md) | Migrates logical MX scale views into canonical packed rank-5 physical form |
| 19 | [LegalizeTileCast](19-legalize_tile_cast.md) | Expands `tile.cast` pairs the ISA cannot emit as one instruction into the shortest native chain |
| 20 | [AutoTileMatmulL0](20-auto_tile_matmul_l0.md) | Picks an L0 tile shape `(m, n, k)` from the backend's L0 capacities and tiles matmuls to it |
| 21 | [CanonicalizeTileSlice](21-canonicalize_tile_slice.md) | Lowers `tile.slice` into the canonical `tile.extract` form |
| 22 | [InferTileMemorySpace](22-infer_tile_memory_space.md) | Infers the on-chip `MemorySpace` of every tile and inserts `tile.move` to legalize mismatches |
| 23 | [InsertMxScaleAddr](23-insert_mx_scale_addr.md) | Inserts `tile.tget_scale_addr` before MX matmul consumers after memory spaces are resolved |
| 24 | [ResolveBackendOpLayouts](24-resolve_backend_op_layouts.md) | Repairs backend-required tile layouts for elementwise ops |
| 25 | [LowerAutoVectorSplit](25-lower_auto_vector_split.md) | Converts AUTO `pl.split` mixed InCore functions into the explicit `split_aiv` form |
| 26 | [ExpandMixedKernel](26-expand_mixed_kernel.md) | Splits mixed InCore functions into separate AIC (Cube) and AIV (Vector) kernels |
| 27 | [InjectGMPipeBuffer](27-inject_gm_pipe_buffer.md) | Injects the `__gm_pipe_buffer` workspace for GM-routed cross-core pipes (Ascend910) |
| 28 | [SplitVectorKernel](28-split_vector_kernel.md) | Stamps split attributes and handles the no-split dual-AIV path |
| 29 | [StampTfreeSplit](29-stamp_tfree_split.md) | Copies each cross-core tpop's split and pipe id onto its matching tfree op |
| 30 | [NormalizeReturnOrder](30-normalize_return_order.md) | Reorders every InCore function's return tuple into the canonical order |
| 31 | [SkewCrossCorePipeline](31-skew_cross_core_pipeline.md) | Software-pipelines mixed cube/vector loops so the two cores overlap |
| 32 | [LowerPipelineToSlots](32-lower_pipeline_to_slots.md) | Opt-in single-level Vec prefetch and rotating slots under PyPTO, or same-iteration slot rotation under PTOAS; declined loops retain unroll |
| 33 | [LowerPipelineLoops](33-lower_pipeline_loops.md) | Replicates `pl.pipeline(N, stage=F)` bodies `F` times to enable ping-pong buffering |
| 34 | [CanonicalizeIOOrder](34-canonicalize_io_order.md) | Reorders pipeline-body statements along the scalar → load → compute → store ladder |
| 35 | [MaterializeTensorStrides](35-materialize_tensor_strides.md) | Fills in the packed canonical stride for every tensor view that carries none |
| 36 | [MaterializeSimtOutputs](36-materialize_simt_outputs.md) | Copies writable Tile inputs into dedicated initialized storage; preserves GM references |
| 37 | [SpecializeSimtHelpers](37-specialize_simt_helpers.md) | Specializes helpers by block dimensions, storage metadata, and target; derives resources |
| 38 | [InitMemRef](38-init_memref.md) | Initializes MemRefs and creates alloc operations with unallocated addresses |
| 39 | [SelectBufferIRFunctions / MaterializeSemanticAliases](39-materialize_semantic_aliases.md) | Forces buffers that program semantics require to be one allocation (loop-carry, in-place) |
| 40 | [MemoryReuse](40-memory_reuse.md) | Reuses buffers by lifetime analysis and removes redundant allocs; skipped with `memory_planner=DSA_RP` or `PTOAS` |
| 41 | [AllocateMemoryAddr](41-allocate_memory_addr.md) | Assigns real addresses to existing alloc operations; skipped with `memory_planner=PTOAS` |
| 42 | [FoldNoOpReshape](42-fold_no_op_reshape.md) | Folds `tile.reshape` calls that change neither physical shape nor allocation |
| 43 | [FuseCreateAssembleToSlice](43-fuse_create_assemble_to_slice.md) | Fuses `tensor.create` + `tensor.assemble` into one `tensor.slice` view |
| 44 | [LowerL2TensorCollectives](44-lower_l2_tensor_collectives.md) | Rewrites a managed collective written in a CHIP orchestration body into one local builtin AIV task, with no per-device fan-out and no nested L2 dispatch |
| 45 | [LegalizeSpmdLaunches](45-legalize_spmd_launches.md) | Guard potentially empty SPMD launches |
| 46 | [DeriveCallDirections](46-derive_call_directions.md) | Materializes wrapper `ParamDirection`s, then derives a per-argument `ArgDirection` at every call |
| 47 | [AutoDeriveTaskDependencies](47-auto_derive_task_dependencies.md) | Derives conservative task-to-task dependency edges |
| 48 | [ExpandManualPhaseFence](48-expand_manual_phase_fence.md) | Compresses profitable full-array `TaskId` dependencies in manual scopes |
| 49 | [SynthesizeAllReduceSignals](49-synthesize_allreduce_signals.md) | Turns a host allreduce's optional signal into explicit internal signal IR |
| 50 | [MaterializeCommDomainScopes](50-materialize_comm_domain_scopes.md) | Assembles `WindowBuffer` and `CommDomainScopeStmt` wrappers in each host orchestration body |
| 51 | [LowerHostTensorCollectives](51-lower_host_tensor_collectives.md) | Rewrites host-level tensor collectives into internal builtin chip dispatches |
| 52 | [MaterializeDistTensorCtx](52-materialize_dist_tensor_ctx.md) | Materializes an explicit `CommCtx` parameter and argument per `DistributedTensor` |
| 53 | [LegalizeGraphBoundary](53-legalize_graph_boundary.md) | Hoists the boundary scalars a `Graph` body derives out to its call sites, and rejects boundaries the `host_build_graph` runtime could not record |
| 54 | [MaterializeRuntimeScopes](54-materialize_runtime_scopes.md) | Inserts AUTO `RuntimeScopeStmt` nodes so orchestration codegen emits `SIMPLER_SCOPE` 1:1 |
| 55 | [ClassifyIterArgCarry](55-classify_iter_arg_carry.md) | Classifies each orchestration `ForStmt` iter_arg as a trivial alias or a materialised rebind carry |
| 56 | [InsertCommFence](56-insert_comm_fence.md) | Marks each publishing write (region `system.cacheinvalid` + `system.fence` locally, fence only for a remote write, whole-GM for an opaque one) and each wait (whole-GM `system.cacheinvalid`); the notify itself gets no marker |
| 57 | [MaterializeValidShapeSymbols](57-materialize_valid_shape_symbols.md) | Turns each device-kernel `valid_shape` symbol the kernel cannot bind into a leading `Scalar[INDEX]` parameter, fed the caller's actual valid extent |
| 58 | [LowerTileToBuffer](58-lower_tile_to_buffer.md) | Final conversion of selected kernels from planned Tile storage to explicit Buffer operations |

## Outside the default pipeline

| Page | What it covers |
| ---- | -------------- |
| [Utility Passes](91-utility_passes.md) | Normalization and cleanup passes that run at several pipeline positions |
| [Diagnostics](92-diagnostics.md) | The advisory channel for compile-time warnings and performance hints |
| [IR Verifier](99-verifier.md) | Pluggable property verifiers that validate IR correctness between passes |

## Shared material

| Page | What it covers |
| ---- | -------------- |
| [Shared Pass Utilities](utils.md) | Reusable helpers in `include/pypto/ir/transforms/utils/` |
| [Loop-Carried Compiler Dependency Compression](loop-carried-dep-compression.md) | How loop-carried dependency edges are compressed |

## See Also

- [IR](../ir/index.md) — the representation these passes transform.
- [Backend](../backend/index.md) — how passes get per-architecture answers without branching on the backend.
- [Code Generation](../codegen/index.md) — what runs once the pipeline is done.
