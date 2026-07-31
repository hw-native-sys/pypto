# Passes

Every transformation PyPTO runs over the IR, numbered to match its position in the
default pipeline.

Pass documentation is numbered so that reading it front to back walks the
compilation pipeline in execution order. `01`–`43` are pipeline passes; `91`+ is
reserved for passes that run at several positions and for infrastructure that is not
a pipeline pass at all.

## Framework

| Page | What it covers |
| ---- | -------------- |
| [Pass, PassContext, PassPipeline, and PassManager](00-pass_manager.md) | Organizing and executing passes with property tracking, instrumentation, and strategy-based pipelines |

## Default pipeline

| Order | Pass | What it does |
| ----- | ---- | ------------ |
| 01 | [InlineFunctions](01-inline_functions.md) | Splices `FunctionType.Inline` bodies into every call site |
| 02 | [UnrollLoops](02-unroll_loops.md) | Expands `ForKind::Unroll` loops at compile time |
| 03 | [CtrlFlowTransform](03-ctrl_flow_transform.md) | Rewrites `break` / `continue` into structured control flow |
| 04 | [ConvertToSSA](04-convert_to_ssa.md) | Converts to SSA form with variable renaming, phi nodes, and iter_args |
| 05 | [Simplify](05-simplify.md) | Folds arithmetic, shape expressions, and scalar constant bindings |
| 06 | [FlattenCallExpr](06-flatten_call_expr.md) | Flattens nested call expressions into three-address form |
| 07 | [OutlineHierarchyScopes](07-outline_hierarchy_scopes.md) | Outlines Hierarchy scopes into functions carrying `level` / `role` metadata |
| 08 | [OutlineIncoreScopes](08-outline_incore_scopes.md) | Outlines InCore scopes into separate functions |
| 09 | [OutlineClusterScopes](09-outline_cluster_scopes.md) | Outlines Cluster scopes into Group functions and standalone Spmd scopes into Spmd functions |
| 10 | [ConvertTensorToTileOps](10-convert_tensor_to_tile_ops.md) | Converts tensor ops to tile ops in InCore functions, updating orchestration call sites |
| 11 | [OptimizeOrchTensors](11-optimize_orch_tensors.md) | Eliminates redundant orchestration allocations and improves data flow |
| 12 | [LowerCompositeOps](12-lower_composite_ops.md) | Decomposes composite tile / distributed ops into primitives |
| 13 | [FlattenTileNdTo2D](13-flatten_tile_nd_to_2d.md) | Flattens 3D+ tile operations to 2D by merging all but the last dimension |
| 14 | [LegalizeTileCast](14-legalize_tile_cast.md) | Expands `tile.cast` pairs the ISA cannot emit as one instruction into the shortest native chain |
| 15 | [AutoTileMatmulL0](15-auto_tile_matmul_l0.md) | Picks an L0 tile shape `(m, n, k)` from the backend's L0 capacities and tiles matmuls to it |
| 16 | [CanonicalizeTileSlice](16-canonicalize_tile_slice.md) | Lowers `tile.slice` into the canonical `tile.extract` form |
| 17 | [InferTileMemorySpace](17-infer_tile_memory_space.md) | Infers the on-chip `MemorySpace` of every tile and inserts `tile.move` to legalize mismatches |
| 18 | [InsertMxScaleAddr](18-insert_mx_scale_addr.md) | Inserts `tile.tget_scale_addr` before MX matmul consumers after memory spaces are resolved |
| 19 | [ResolveBackendOpLayouts](19-resolve_backend_op_layouts.md) | Repairs backend-required tile layouts for elementwise ops |
| 20 | [LowerAutoVectorSplit](20-lower_auto_vector_split.md) | Converts AUTO `pl.split` mixed InCore functions into the explicit `split_aiv` form |
| 21 | [ExpandMixedKernel](21-expand_mixed_kernel.md) | Splits mixed InCore functions into separate AIC (Cube) and AIV (Vector) kernels |
| 22 | [InjectGMPipeBuffer](22-inject_gm_pipe_buffer.md) | Injects the `__gm_pipe_buffer` workspace for GM-routed cross-core pipes (Ascend910B) |
| 23 | [SplitVectorKernel](23-split_vector_kernel.md) | Stamps split attributes and handles the no-split dual-AIV path |
| 24 | [StampTfreeSplit](24-stamp_tfree_split.md) | Copies each cross-core tpop's split and pipe id onto its matching tfree op |
| 25 | [NormalizeReturnOrder](25-normalize_return_order.md) | Reorders every InCore function's return tuple into the canonical order |
| 26 | [SkewCrossCorePipeline](26-skew_cross_core_pipeline.md) | Software-pipelines mixed cube/vector loops so the two cores overlap |
| 27 | [LowerPipelineLoops](27-lower_pipeline_loops.md) | Replicates `pl.pipeline(N, stage=F)` bodies `F` times to enable ping-pong buffering |
| 28 | [CanonicalizeIOOrder](28-canonicalize_io_order.md) | Reorders pipeline-body statements along the scalar → load → compute → store ladder |
| 29 | [MaterializeTensorStrides](29-materialize_tensor_strides.md) | Fills in the packed canonical stride for every tensor view that carries none |
| 30 | [InitMemRef](30-init_memref.md) | Initializes MemRefs and creates alloc operations with unallocated addresses |
| 31 | [MaterializeSemanticAliases](31-materialize_semantic_aliases.md) | Forces buffers that program semantics require to be one allocation (loop-carry, in-place) |
| 32 | [MemoryReuse](32-memory_reuse.md) | Reuses buffers by lifetime analysis and removes redundant allocs |
| 33 | [AllocateMemoryAddr](33-allocate_memory_addr.md) | Assigns real addresses to existing alloc operations |
| 34 | [FoldNoOpReshape](34-fold_no_op_reshape.md) | Folds `tile.reshape` calls that change neither physical shape nor allocation |
| 35 | [FuseCreateAssembleToSlice](35-fuse_create_assemble_to_slice.md) | Fuses `tensor.create` + `tensor.assemble` into one `tensor.slice` view |
| 36 | [DeriveCallDirections](36-derive_call_directions.md) | Materializes wrapper `ParamDirection`s, then derives a per-argument `ArgDirection` at every call |
| 37 | [AutoDeriveTaskDependencies](37-auto_derive_task_dependencies.md) | Derives conservative task-to-task dependency edges |
| 38 | [ExpandManualPhaseFence](38-expand_manual_phase_fence.md) | Compresses profitable full-array `TaskId` dependencies in manual scopes |
| 39 | [SynthesizeAllReduceSignals](39-synthesize_allreduce_signals.md) | Turns a host allreduce's optional signal into explicit internal signal IR |
| 40 | [MaterializeCommDomainScopes](40-materialize_comm_domain_scopes.md) | Assembles `WindowBuffer` and `CommDomainScopeStmt` wrappers in each host orchestration body |
| 41 | [LowerHostTensorCollectives](41-lower_host_tensor_collectives.md) | Rewrites host-level tensor collectives into internal builtin chip dispatches |
| 42 | [MaterializeDistTensorCtx](42-materialize_dist_tensor_ctx.md) | Materializes an explicit `CommCtx` parameter and argument per `DistributedTensor` |
| 43 | [MaterializeRuntimeScopes](43-materialize_runtime_scopes.md) | Inserts AUTO `RuntimeScopeStmt` nodes so orchestration codegen emits `PTO2_SCOPE` 1:1 |
| 44 | [ClassifyIterArgCarry](44-classify_iter_arg_carry.md) | Classifies each orchestration `ForStmt` iter_arg as a trivial alias or a materialised rebind carry |
| 45 | [InsertCommFence](45-insert_comm_fence.md) | Inserts a whole-tensor `system.cacheinvalid` + GM `system.fence` between each publishing write and the `pld.system.notify` that releases it |

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
