# Passes

Every transformation PyPTO runs over the IR, numbered to match its position in the
default pipeline.

Pass documentation is numbered so that reading it front to back walks the
compilation pipeline in execution order. `01`–`49` are pipeline passes; `91`+ is
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
| 08 | [OutlineGraphScopes](08-outline_graph_scopes.md) | Outlines `pl.graph` regions into `FunctionType.Graph` functions, converging the scope form with `@pl.jit.graph` |
| 09 | [OutlineIncoreScopes](09-outline_incore_scopes.md) | Outlines InCore scopes into separate functions |
| 10 | [OutlineClusterScopes](10-outline_cluster_scopes.md) | Outlines Cluster scopes into Group functions and standalone Spmd scopes into Spmd functions |
| 11 | [ConvertTensorToTileOps](11-convert_tensor_to_tile_ops.md) | Converts tensor ops to tile ops in InCore functions, updating orchestration call sites |
| 12 | [OptimizeOrchTensors](12-optimize_orch_tensors.md) | Eliminates redundant orchestration allocations and improves data flow |
| 13 | [LowerCompositeOps](13-lower_composite_ops.md) | Decomposes composite tile / distributed ops into primitives |
| 13 | [FlattenTileNdTo2D](14-flatten_tile_nd_to_2d.md) | Flattens 3D+ tile operations to 2D by merging all but the last dimension |
| 15 | [BlockNzTensorViews](15-block_nz_tensor_views.md) | Rewrites logical `pl.NZ` tensors into pto-isa's blocked rank-(r+2) form and retargets their `tile.load` coordinates |
| 16 | [LegalizeTileCast](16-legalize_tile_cast.md) | Expands `tile.cast` pairs the ISA cannot emit as one instruction into the shortest native chain |
| 16 | [AutoTileMatmulL0](17-auto_tile_matmul_l0.md) | Picks an L0 tile shape `(m, n, k)` from the backend's L0 capacities and tiles matmuls to it |
| 18 | [CanonicalizeTileSlice](18-canonicalize_tile_slice.md) | Lowers `tile.slice` into the canonical `tile.extract` form |
| 19 | [InferTileMemorySpace](19-infer_tile_memory_space.md) | Infers the on-chip `MemorySpace` of every tile and inserts `tile.move` to legalize mismatches |
| 20 | [InsertMxScaleAddr](20-insert_mx_scale_addr.md) | Inserts `tile.tget_scale_addr` before MX matmul consumers after memory spaces are resolved |
| 21 | [ResolveBackendOpLayouts](21-resolve_backend_op_layouts.md) | Repairs backend-required tile layouts for elementwise ops |
| 22 | [LowerAutoVectorSplit](22-lower_auto_vector_split.md) | Converts AUTO `pl.split` mixed InCore functions into the explicit `split_aiv` form |
| 23 | [ExpandMixedKernel](23-expand_mixed_kernel.md) | Splits mixed InCore functions into separate AIC (Cube) and AIV (Vector) kernels |
| 24 | [InjectGMPipeBuffer](24-inject_gm_pipe_buffer.md) | Injects the `__gm_pipe_buffer` workspace for GM-routed cross-core pipes (Ascend910B) |
| 25 | [SplitVectorKernel](25-split_vector_kernel.md) | Stamps split attributes and handles the no-split dual-AIV path |
| 26 | [StampTfreeSplit](26-stamp_tfree_split.md) | Copies each cross-core tpop's split and pipe id onto its matching tfree op |
| 27 | [NormalizeReturnOrder](27-normalize_return_order.md) | Reorders every InCore function's return tuple into the canonical order |
| 28 | [SkewCrossCorePipeline](28-skew_cross_core_pipeline.md) | Software-pipelines mixed cube/vector loops so the two cores overlap |
| 29 | [LowerPipelineToSlots](29-lower_pipeline_to_slots.md) | Rotates a `pl.pipeline` body through the slots of one allocation instead of replicating it (`memory_planner=PTOAS`) |
| 30 | [LowerPipelineLoops](30-lower_pipeline_loops.md) | Replicates `pl.pipeline(N, stage=F)` bodies `F` times to enable ping-pong buffering |
| 31 | [CanonicalizeIOOrder](31-canonicalize_io_order.md) | Reorders pipeline-body statements along the scalar → load → compute → store ladder |
| 32 | [MaterializeTensorStrides](32-materialize_tensor_strides.md) | Fills in the packed canonical stride for every tensor view that carries none |
| 33 | [InitMemRef](33-init_memref.md) | Initializes MemRefs and creates alloc operations with unallocated addresses |
| 34 | [MaterializeSemanticAliases](34-materialize_semantic_aliases.md) | Forces buffers that program semantics require to be one allocation (loop-carry, in-place) |
| 35 | [MemoryReuse](35-memory_reuse.md) | Reuses buffers by lifetime analysis and removes redundant allocs |
| 36 | [AllocateMemoryAddr](36-allocate_memory_addr.md) | Assigns real addresses to existing alloc operations |
| 37 | [FoldNoOpReshape](37-fold_no_op_reshape.md) | Folds `tile.reshape` calls that change neither physical shape nor allocation |
| 38 | [FuseCreateAssembleToSlice](38-fuse_create_assemble_to_slice.md) | Fuses `tensor.create` + `tensor.assemble` into one `tensor.slice` view |
| 39 | [DeriveCallDirections](39-derive_call_directions.md) | Materializes wrapper `ParamDirection`s, then derives a per-argument `ArgDirection` at every call |
| 40 | [AutoDeriveTaskDependencies](40-auto_derive_task_dependencies.md) | Derives conservative task-to-task dependency edges |
| 41 | [ExpandManualPhaseFence](41-expand_manual_phase_fence.md) | Compresses profitable full-array `TaskId` dependencies in manual scopes |
| 42 | [SynthesizeAllReduceSignals](42-synthesize_allreduce_signals.md) | Turns a host allreduce's optional signal into explicit internal signal IR |
| 43 | [MaterializeCommDomainScopes](43-materialize_comm_domain_scopes.md) | Assembles `WindowBuffer` and `CommDomainScopeStmt` wrappers in each host orchestration body |
| 44 | [LowerHostTensorCollectives](44-lower_host_tensor_collectives.md) | Rewrites host-level tensor collectives into internal builtin chip dispatches |
| 45 | [MaterializeDistTensorCtx](45-materialize_dist_tensor_ctx.md) | Materializes an explicit `CommCtx` parameter and argument per `DistributedTensor` |
| 46 | [LegalizeGraphBoundary](46-legalize_graph_boundary.md) | Hoists the boundary scalars a `Graph` body derives out to its call sites, and rejects boundaries the `host_build_graph` runtime could not record |
| 47 | [MaterializeRuntimeScopes](47-materialize_runtime_scopes.md) | Inserts AUTO `RuntimeScopeStmt` nodes so orchestration codegen emits `SIMPLER_SCOPE` 1:1 |
| 48 | [ClassifyIterArgCarry](48-classify_iter_arg_carry.md) | Classifies each orchestration `ForStmt` iter_arg as a trivial alias or a materialised rebind carry |
| 49 | [InsertCommFence](49-insert_comm_fence.md) | Marks each publishing write (region `system.cacheinvalid` + `system.fence` locally, fence only for a remote write, whole-GM for an opaque one) and each wait (whole-GM `system.cacheinvalid`); the notify itself gets no marker |
| 50 | [MaterializeValidShapeSymbols](50-materialize_valid_shape_symbols.md) | Turns each device-kernel `valid_shape` symbol the kernel cannot bind into a leading `Scalar[INDEX]` parameter, fed the caller's actual valid extent |

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
