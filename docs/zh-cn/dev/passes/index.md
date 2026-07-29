# Passes

PyPTO 在 IR 之上运行的全部变换，编号与其在默认流水线中的位置一致。

pass 文档按编号组织，因此从头读到尾就是按执行顺序走完整条编译流水。`01`–`43` 是流水线
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
| 08 | [OutlineIncoreScopes](08-outline_incore_scopes.md) | 把 InCore 作用域外提为独立函数 |
| 09 | [OutlineClusterScopes](09-outline_cluster_scopes.md) | 把 Cluster 作用域外提为 Group 函数，独立 Spmd 作用域外提为 Spmd 函数 |
| 10 | [ConvertTensorToTileOps](10-convert_tensor_to_tile_ops.md) | 在 InCore 函数中把 tensor 算子转为 tile 算子，并更新编排层调用点 |
| 11 | [OptimizeOrchTensors](11-optimize_orch_tensors.md) | 消除编排层冗余分配并改善数据流 |
| 12 | [LowerCompositeOps](12-lower_composite_ops.md) | 把复合 tile / 分布式算子分解为基础原语 |
| 13 | [FlattenTileNdTo2D](13-flatten_tile_nd_to_2d.md) | 合并除最后一维外的所有维度，把 3D+ tile 操作拍平为 2D |
| 14 | [LegalizeTileCast](14-legalize_tile_cast.md) | 把 ISA 无法单条指令完成的 `tile.cast` 展开为最短的原生 cast 链 |
| 15 | [AutoTileMatmulL0](15-auto_tile_matmul_l0.md) | 依据后端 L0 容量选择 L0 tile 形状 `(m, n, k)` 并据此分块 matmul |
| 16 | [CanonicalizeTileSlice](16-canonicalize_tile_slice.md) | 把 `tile.slice` 下降为规范的 `tile.extract` 形式 |
| 17 | [InferTileMemorySpace](17-infer_tile_memory_space.md) | 推断每个 tile 的片上 `MemorySpace`，并插入 `tile.move` 消解残留不匹配 |
| 18 | [ResolveBackendOpLayouts](18-resolve_backend_op_layouts.md) | 修正逐元素算子所需的后端 tile layout |
| 19 | [LowerAutoVectorSplit](19-lower_auto_vector_split.md) | 把 AUTO `pl.split` 的混合 InCore 函数转换为显式 `split_aiv` 形式 |
| 20 | [ExpandMixedKernel](20-expand_mixed_kernel.md) | 把混合 InCore 函数拆分为独立的 AIC（Cube）与 AIV（Vector）kernel |
| 21 | [InjectGMPipeBuffer](21-inject_gm_pipe_buffer.md) | 为经 GM 路由的跨核 pipe 注入 `__gm_pipe_buffer` workspace（Ascend910B） |
| 22 | [SplitVectorKernel](22-split_vector_kernel.md) | 标记 split 属性并处理不拆分的双 AIV 路径 |
| 23 | [StampTfreeSplit](23-stamp_tfree_split.md) | 把每个跨核 tpop 的 split 与 pipe id 复制到与之配对的 tfree 上 |
| 24 | [NormalizeReturnOrder](24-normalize_return_order.md) | 把每个 InCore 函数的返回元组重排为规范顺序 |
| 25 | [SkewCrossCorePipeline](25-skew_cross_core_pipeline.md) | 对混合 cube/vector 循环做软流水，使两个核重叠执行 |
| 26 | [LowerPipelineLoops](26-lower_pipeline_loops.md) | 把 `pl.pipeline(N, stage=F)` 的循环体复制 `F` 份以启用乒乓缓冲 |
| 27 | [CanonicalizeIOOrder](27-canonicalize_io_order.md) | 按 scalar → load → compute → store 阶梯重排流水循环体内的语句 |
| 28 | [MaterializeTensorStrides](28-materialize_tensor_strides.md) | 为每个尚无 stride 的 tensor view 填入紧致规范 stride |
| 29 | [InitMemRef](29-init_memref.md) | 初始化 MemRef 并创建地址未分配的 alloc 操作 |
| 30 | [MaterializeSemanticAliases](30-materialize_semantic_aliases.md) | 强制语义要求同一分配的缓冲区真正共用一块（循环携带、原地更新） |
| 31 | [MemoryReuse](31-memory_reuse.md) | 基于生命周期分析复用缓冲区并删除冗余 alloc |
| 32 | [AllocateMemoryAddr](32-allocate_memory_addr.md) | 为已有 alloc 操作分配真实地址 |
| 33 | [FoldNoOpReshape](33-fold_no_op_reshape.md) | 折叠既不改变物理形状也不改变分配的 `tile.reshape` |
| 34 | [FuseCreateAssembleToSlice](34-fuse_create_assemble_to_slice.md) | 把 `tensor.create` + `tensor.assemble` 融合为单个 `tensor.slice` 视图 |
| 35 | [DeriveCallDirections](35-derive_call_directions.md) | 先物化包装函数的 `ParamDirection`，再为每个调用逐实参推导 `ArgDirection` |
| 36 | [AutoDeriveTaskDependencies](36-auto_derive_task_dependencies.md) | 推导保守的任务间依赖边 |
| 37 | [ExpandManualPhaseFence](37-expand_manual_phase_fence.md) | 压缩 manual scope 中收益明确的全数组 `TaskId` 依赖 |
| 38 | [SynthesizeAllReduceSignals](38-synthesize_allreduce_signals.md) | 把 host allreduce 的可选 signal 转为显式的内部 signal IR |
| 39 | [MaterializeCommDomainScopes](39-materialize_comm_domain_scopes.md) | 在每个 host 编排函数体内装配 `WindowBuffer` 与 `CommDomainScopeStmt` 包装 |
| 40 | [LowerHostTensorCollectives](40-lower_host_tensor_collectives.md) | 把 host 级 tensor 集合通信改写为内部 builtin chip 派发 |
| 41 | [MaterializeDistTensorCtx](41-materialize_dist_tensor_ctx.md) | 为每个 `DistributedTensor` 物化显式的 `CommCtx` 参数与实参 |
| 42 | [MaterializeRuntimeScopes](42-materialize_runtime_scopes.md) | 插入 AUTO `RuntimeScopeStmt` 使编排 codegen 能 1:1 发射 `PTO2_SCOPE` |
| 43 | [ClassifyIterArgCarry](43-classify_iter_arg_carry.md) | 把编排层 `ForStmt` 的每个 iter_arg 分类为平凡别名或需物化的重绑定携带 |

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
