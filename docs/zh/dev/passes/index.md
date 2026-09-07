# Passes

PyPTO 在 IR 之上运行的全部变换，编号与其在默认流水线中的位置一致。

pass 文档按编号组织，因此从头读到尾就是按执行顺序走完整条编译流水。`01`–`49` 是流水线
pass；`91` 及以后保留给"在多个位置运行的 pass"以及"根本不是流水线 pass 的基础设施"。

## 框架

| 页面 | 内容 |
| ---- | ---- |
| [Pass、PassContext、PassPipeline 和 PassManager](00-pass_manager.md) | 带属性跟踪、插桩与策略化流水线的 pass 组织与执行框架 |

## 默认流水线

| 序号 | Pass | 作用 |
| ---- | ---- | ---- |
| 01 | [InlineFunctions](01-inline_functions.md) | 把 `FunctionType.Inline` 函数体展开到每个调用点 |
| 02 | [UnrollLoops](02-unroll_loops.md) | 在编译期展开 `ForKind::Unroll` 循环 |
| 03 | [CtrlFlowTransform](03-ctrl_flow_transform.md) | 把 `break` / `continue` 改写为结构化控制流 |
| 04 | [ConvertToSSA](04-convert_to_ssa.md) | 转换为 SSA 形式，含变量重命名、phi 节点与 iter_args |
| 05 | [Simplify](05-simplify.md) | 折叠算术表达式、shape 表达式与标量常量绑定 |
| 06 | [FlattenCallExpr](06-flatten_call_expr.md) | 把嵌套调用表达式拍平为三地址形式 |
| 07 | [OutlineHierarchyScopes](07-outline_hierarchy_scopes.md) | 把 Hierarchy 作用域外提为带 `level` / `role` 元数据的函数 |
| 08 | [OutlineGraphScopes](08-outline_graph_scopes.md) | 把 `pl.graph` 区域外提为 `FunctionType.Graph` 函数，使作用域形式与 `@pl.jit.graph` 汇聚 |
| 09 | [OutlineIncoreScopes](09-outline_incore_scopes.md) | 把 InCore 作用域外提为独立函数 |
| 10 | [OutlineClusterScopes](10-outline_cluster_scopes.md) | 把 Cluster 作用域外提为 Group 函数，独立 Spmd 作用域外提为 Spmd 函数 |
| 11 | [ConvertTensorToTileOps](11-convert_tensor_to_tile_ops.md) | 在 InCore 函数中把 tensor 算子转为 tile 算子，并更新编排层调用点 |
| 12 | [OptimizeOrchTensors](12-optimize_orch_tensors.md) | 消除编排层冗余分配并改善数据流 |
| 13 | [LowerCompositeOps](13-lower_composite_ops.md) | 把复合 tile / 分布式算子分解为基础原语 |
| 13 | [FlattenTileNdTo2D](14-flatten_tile_nd_to_2d.md) | 合并除最后一维外的所有维度，把 3D+ tile 操作拍平为 2D |
| 15 | [BlockNzTensorViews](15-block_nz_tensor_views.md) | 把逻辑 `pl.NZ` 张量改写为 pto-isa 的分块 rank-(r+2) 形式，并同步改写其 `tile.load` 坐标 |
| 16 | [BlockMxScaleTensorViews](16-block_mx_scale_tensor_views.md) | 将逻辑 MX scale 视图迁移为规范的 rank-5 物理分块形式 |
| 17 | [LegalizeTileCast](17-legalize_tile_cast.md) | 把 ISA 无法单条指令完成的 `tile.cast` 展开为最短的原生 cast 链 |
| 18 | [AutoTileMatmulL0](18-auto_tile_matmul_l0.md) | 依据后端 L0 容量选择 L0 tile 形状 `(m, n, k)` 并据此分块 matmul |
| 19 | [CanonicalizeTileSlice](19-canonicalize_tile_slice.md) | 把 `tile.slice` 下降为规范的 `tile.extract` 形式 |
| 20 | [InferTileMemorySpace](20-infer_tile_memory_space.md) | 推断每个 tile 的片上 `MemorySpace`，并插入 `tile.move` 消解残留不匹配 |
| 21 | [InsertMxScaleAddr](21-insert_mx_scale_addr.md) | 在 memory space 解析完成后，于 MX matmul 消费者前插入 `tile.tget_scale_addr` |
| 22 | [ResolveBackendOpLayouts](22-resolve_backend_op_layouts.md) | 修正逐元素算子所需的后端 tile layout |
| 23 | [LowerAutoVectorSplit](23-lower_auto_vector_split.md) | 把 AUTO `pl.split` 的混合 InCore 函数转换为显式 `split_aiv` 形式 |
| 24 | [ExpandMixedKernel](24-expand_mixed_kernel.md) | 把混合 InCore 函数拆分为独立的 AIC（Cube）与 AIV（Vector）kernel |
| 25 | [InjectGMPipeBuffer](25-inject_gm_pipe_buffer.md) | 为经 GM 路由的跨核 pipe 注入 `__gm_pipe_buffer` workspace（Ascend910B） |
| 26 | [SplitVectorKernel](26-split_vector_kernel.md) | 标记 split 属性并处理不拆分的双 AIV 路径 |
| 27 | [StampTfreeSplit](27-stamp_tfree_split.md) | 把每个跨核 tpop 的 split 与 pipe id 复制到与之配对的 tfree 上 |
| 28 | [NormalizeReturnOrder](28-normalize_return_order.md) | 把每个 InCore 函数的返回元组重排为规范顺序 |
| 29 | [SkewCrossCorePipeline](29-skew_cross_core_pipeline.md) | 对混合 cube/vector 循环做软流水，使两个核重叠执行 |
| 30 | [LowerPipelineToSlots](30-lower_pipeline_to_slots.md) | 把 `pl.pipeline` 循环体改为轮转一个分配的多个 slot，而不是复制（`memory_planner=PTOAS`） |
| 31 | [LowerPipelineLoops](31-lower_pipeline_loops.md) | 把 `pl.pipeline(N, stage=F)` 的循环体复制 `F` 份以启用乒乓缓冲 |
| 32 | [CanonicalizeIOOrder](32-canonicalize_io_order.md) | 按 scalar → load → compute → store 阶梯重排流水循环体内的语句 |
| 33 | [MaterializeTensorStrides](33-materialize_tensor_strides.md) | 为每个尚无 stride 的 tensor view 填入紧致规范 stride |
| 34 | [InitMemRef](34-init_memref.md) | 初始化 MemRef 并创建地址未分配的 alloc 操作 |
| 35 | [MaterializeSemanticAliases](35-materialize_semantic_aliases.md) | 强制语义要求同一分配的缓冲区真正共用一块（循环携带、原地更新） |
| 36 | [MemoryReuse](36-memory_reuse.md) | 基于生命周期分析复用缓冲区并删除冗余 alloc |
| 37 | [AllocateMemoryAddr](37-allocate_memory_addr.md) | 为已有 alloc 操作分配真实地址 |
| 38 | [FoldNoOpReshape](38-fold_no_op_reshape.md) | 折叠既不改变物理形状也不改变分配的 `tile.reshape` |
| 39 | [FuseCreateAssembleToSlice](39-fuse_create_assemble_to_slice.md) | 把 `tensor.create` + `tensor.assemble` 融合为单个 `tensor.slice` 视图 |
| 40 | [DeriveCallDirections](40-derive_call_directions.md) | 先物化包装函数的 `ParamDirection`，再为每个调用逐实参推导 `ArgDirection` |
| 41 | [AutoDeriveTaskDependencies](41-auto_derive_task_dependencies.md) | 推导保守的任务间依赖边 |
| 42 | [ExpandManualPhaseFence](42-expand_manual_phase_fence.md) | 压缩 manual scope 中收益明确的全数组 `TaskId` 依赖 |
| 43 | [SynthesizeAllReduceSignals](43-synthesize_allreduce_signals.md) | 把 host allreduce 的可选 signal 转为显式的内部 signal IR |
| 44 | [MaterializeCommDomainScopes](44-materialize_comm_domain_scopes.md) | 在每个 host 编排函数体内装配 `WindowBuffer` 与 `CommDomainScopeStmt` 包装 |
| 45 | [LowerHostTensorCollectives](45-lower_host_tensor_collectives.md) | 把 host 级 tensor 集合通信改写为内部 builtin chip 派发 |
| 46 | [MaterializeDistTensorCtx](46-materialize_dist_tensor_ctx.md) | 为每个 `DistributedTensor` 物化显式的 `CommCtx` 参数与实参 |
| 47 | [LegalizeGraphBoundary](47-legalize_graph_boundary.md) | 把 `Graph` 函数体内派生的边界标量外提到调用点，并拒绝 `host_build_graph` runtime 无法录制的边界 |
| 48 | [MaterializeRuntimeScopes](48-materialize_runtime_scopes.md) | 插入 AUTO `RuntimeScopeStmt` 使编排 codegen 能 1:1 发射 `SIMPLER_SCOPE` |
| 49 | [ClassifyIterArgCarry](49-classify_iter_arg_carry.md) | 把编排层 `ForStmt` 的每个 iter_arg 分类为平凡别名或需物化的重绑定携带 |
| 50 | [InsertCommFence](50-insert_comm_fence.md) | 为每个发布性写入打标记（本地：region `system.cacheinvalid` + `system.fence`；远端写：仅 fence；opaque 写：whole-GM），并为每个 wait 插入 whole-GM `system.cacheinvalid`；notify 本身不加任何标记 |
| 51 | [MaterializeValidShapeSymbols](51-materialize_valid_shape_symbols.md) | 将设备 kernel 中无法绑定的 `valid_shape` 符号转换为前置的 `Scalar[INDEX]` 参数，并传入调用方的实际有效范围 |

## 默认流水线之外

| 页面 | 内容 |
| ---- | ---- |
| [工具 Pass](91-utility_passes.md) | 在流水线多个位置运行的归一化与清理 pass |
| [诊断系统](92-diagnostics.md) | 编译期警告与性能提示的统一咨询通道 |
| [IR 验证器](99-verifier.md) | 在 pass 之间校验 IR 正确性的可插拔属性验证器 |

## 共享材料

| 页面 | 内容 |
| ---- | ---- |
| [共享 Pass 工具函数](utils.md) | `include/pypto/ir/transforms/utils/` 中的可复用工具 |
| [Loop-Carried Compiler Dependency 压缩](loop-carried-dep-compression.md) | 循环携带依赖边的压缩方式 |

## 另请参阅

- [IR](../ir/index.md) —— 这些 pass 所变换的表示。
- [后端](../backend/index.md) —— pass 如何在不对后端分支的前提下获得逐架构答案。
- [代码生成](../codegen/index.md) —— 流水线跑完之后运行的部分。
