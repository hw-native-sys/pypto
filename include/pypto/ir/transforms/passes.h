/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef PYPTO_IR_TRANSFORMS_PASSES_H_
#define PYPTO_IR_TRANSFORMS_PASSES_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/ir_property.h"
#include "pypto/ir/transforms/pass_context.h"

namespace pypto {
namespace ir {

/**
 * @brief Internal base class for pass implementations
 *
 * Most passes should use CreateFunctionPass() or CreateProgramPass() helpers.
 * Only inherit from PassImpl for complex passes with custom state.
 */
class PassImpl {
 public:
  virtual ~PassImpl() = default;

  /**
   * @brief Execute the pass on a program
   */
  virtual ProgramPtr operator()(const ProgramPtr& program) = 0;

  /**
   * @brief Get the name of the pass (for debugging)
   */
  [[nodiscard]] virtual std::string GetName() const { return "UnnamedPass"; }

  /**
   * @brief Get properties required before this pass can run
   */
  [[nodiscard]] virtual IRPropertySet GetRequiredProperties() const { return {}; }

  /**
   * @brief Get properties produced (guaranteed) after this pass runs
   */
  [[nodiscard]] virtual IRPropertySet GetProducedProperties() const { return {}; }

  /**
   * @brief Get properties invalidated (broken) by this pass
   */
  [[nodiscard]] virtual IRPropertySet GetInvalidatedProperties() const { return {}; }
};

/**
 * @brief Base class for IR transformation passes
 *
 * Pass uses a pimpl pattern to hide implementation details.
 * Users should create passes using factory functions.
 */
class Pass {
 public:
  Pass();
  explicit Pass(std::shared_ptr<PassImpl> impl);
  ~Pass();

  // Copy and move
  Pass(const Pass& other);
  Pass& operator=(const Pass& other);
  Pass(Pass&& other) noexcept;
  Pass& operator=(Pass&& other) noexcept;

  /**
   * @brief Execute the pass on a program (primary API)
   */
  ProgramPtr operator()(const ProgramPtr& program) const;

  /**
   * @brief Execute the pass on a program (backward compatible API)
   */
  [[nodiscard]] ProgramPtr run(const ProgramPtr& program) const;

  /**
   * @brief Get the name of the pass
   */
  [[nodiscard]] std::string GetName() const;

  /**
   * @brief Get properties required before this pass can run
   */
  [[nodiscard]] IRPropertySet GetRequiredProperties() const;

  /**
   * @brief Get properties produced (guaranteed) after this pass runs
   */
  [[nodiscard]] IRPropertySet GetProducedProperties() const;

  /**
   * @brief Get properties invalidated (broken) by this pass
   */
  [[nodiscard]] IRPropertySet GetInvalidatedProperties() const;

 private:
  std::shared_ptr<PassImpl> impl_;
};

// Factory functions for built-in passes
namespace pass {

/**
 * @brief Create a pass from a function-level transform function (RECOMMENDED)
 *
 * @param transform Function that transforms a Function
 * @param name Optional name for the pass (for debugging)
 * @param properties Optional property declarations
 * @return Pass that applies the transform to each function
 */
Pass CreateFunctionPass(std::function<FunctionPtr(const FunctionPtr&)> transform,
                        const std::string& name = "", const PassProperties& properties = {});

/**
 * @brief Create a pass from a program-level transform function
 *
 * @param transform Function that transforms a Program
 * @param name Optional name for the pass (for debugging)
 * @param properties Optional property declarations
 * @return Pass that applies the transform
 */
Pass CreateProgramPass(std::function<ProgramPtr(const ProgramPtr&)> transform, const std::string& name = "",
                       const PassProperties& properties = {});

/**
 * @brief Create an init memref pass
 *
 * Initializes MemRef for all variables in functions.
 * Sets memory space to UB by default, or DDR for tile.load/tile.store operands.
 */
Pass InitMemRef();

/**
 * @brief Create the semantic must-alias materialization pass
 *
 * Propagates each loop-carried iter_arg/initValue MemRef down the yield/producer
 * chain so accumulator producers (and other loop-carry chains) write directly
 * into the carried buffer. This is a semantics-required aliasing (the loop
 * accumulator must live in one buffer), split out of MemoryReuse so it can run
 * without the opportunistic lifetime-reuse phase (when DSA-RP or ptoas owns
 * reuse). Runs after InitMemRef, before MemoryReuse/DSA placement.
 */
Pass MaterializeSemanticAliases();

/**
 * @brief Create a memory reuse pass
 *
 * Uses dependency analysis to identify memory reuse opportunities.
 * Variables with non-overlapping lifetimes in the same memory space can share MemRef objects.
 */
Pass MemoryReuse();

/**
 * @brief Create an allocate memory address pass
 *
 * Allocates real memory addresses for existing alloc operations.
 * Updates MemRef addresses and alloc statement arguments in place.
 */
Pass AllocateMemoryAddr();

/**
 * @brief Eliminate FunctionType::Inline functions by splicing their bodies
 *        into every call site.
 *
 * Runs as the first pipeline pass. After this pass, no Function with
 * func_type == Inline remains, and no Call expression resolves to one.
 * Subsequent passes do not need to handle Inline functions.
 *
 * Algorithm:
 *  - Detects cycles in the Inline → Inline call graph (raises pypto::ValueError).
 *  - Iteratively splices each top-level `LHS = inline_call(args)` (and
 *    `EvalStmt(inline_call(args))`) until fixpoint, supporting nested
 *    Inline-calls-Inline expansion.
 *  - Alpha-renames inlined locals to avoid collisions across multiple call
 *    sites and substitutes formal params with actual args.
 *  - Multi-return inline functions emit `LHS = MakeTuple([rets...])` at the
 *    call site.
 */
Pass InlineFunctions();

/**
 * @brief Synthesize private signal windows for host-level allreduce calls that omit signal.
 *
 * Rewrites host orchestration ``pld.tensor.allreduce(target, op=...)`` calls to
 * the internal explicit-signal form by inserting ordinary
 * ``pld.tensor.alloc_window_buffer`` and ``pld.tensor.window`` assignments
 * immediately before the call. Existing explicit-signal calls are preserved.
 */
Pass SynthesizeAllReduceSignals();

/**
 * @brief Materialise comm-domain scope statements for distributed window-buffer allocations.
 *
 * Runs at the end of the pipeline, just before the final Simplify. None of
 * the intervening passes touches the host_orch alloc/window/dispatch chain
 * (host_orch is never tile-lowered and L2 orch is never inlined into L3),
 * so the ``host_orch → chip_orch → InCore`` dispatch chain is still
 * discoverable here.
 *
 * For each ``@pl.program`` host_orch function:
 *
 *  1. Find every ``pld.tensor.alloc_window_buffer(size, *, name)`` Call op; its
 *     ``AssignStmt`` LHS is a plain ``Var(PtrType)`` (``ptr_var``).
 *  2. Follow def-use to the ``pld.tensor.window(ptr_var, shape, *, dtype)`` views
 *     materialised over each ``ptr_var`` and the dispatch calls that consume
 *     those views (recursing through chip_orch formal-param bindings).
 *  3. From each dispatch call's ``attrs["device"]`` expression, derive a
 *     ``DeviceDescriptor`` (``kAll`` / explicit subset). Merge across all
 *     consuming dispatches for the same alloc.
 *  4. Construct a :class:`WindowBuffer` per alloc (``base = ptr_var``,
 *     ``size = size_expr``). Rewrite every ``pld.tensor.window`` result Var's type
 *     so ``DistributedTensorType.window_buffer_`` points to the new
 *     ``WindowBuffer`` (host_orch only — chip_orch / InCore param types
 *     remain ``nullopt``).
 *  5. Cluster ``WindowBuffer`` s by ``DeviceDescriptor`` (same descriptor →
 *     one comm domain, slots in alloc-source order) and wrap the host_orch
 *     body in nested ``CommDomainScopeStmt`` nodes (outer = first declared
 *     domain, inner = last).
 *
 * Sanity-checks (``pypto::ValueError`` on failure):
 *  - Every alloc must have at least one ``pld.tensor.window`` materialisation and
 *    at least one dispatch consumer.
 *  - Allocation names are unique within a group (parser-enforced globally;
 *    re-asserted here).
 */
Pass MaterializeCommDomainScopes();

/**
 * @brief Lower host-level ``pld.tensor.allreduce`` calls to internal builtin chip dispatches.
 */
Pass LowerHostTensorCollectives();

/**
 * @brief Materialize one CommCtx parameter/argument per DistributedTensor parameter.
 */
Pass MaterializeDistTensorCtx();

/**
 * @brief Materialize a scalar parameter per unbindable device-kernel valid_shape symbol.
 *
 * A ``pl.dynamic()`` symbol named only in a parameter's
 * ``pl.TensorView(valid_shape=...)`` is neither a physical tensor dimension (which
 * the kernel wrapper recovers from the runtime tensor's ``shapes[]``) nor a scalar
 * parameter, so a precompiled kernel has no value for it. This pass adds the
 * symbol itself as a *leading* ``Scalar[INDEX]`` parameter and passes the caller's
 * actual extent at every call/submit site, lowering the annotation form into the
 * scalar-parameter form the backend already supports. The parameter leads because
 * the text form declares parameters left to right, and the annotation that names
 * the symbol has to resolve to it.
 */
Pass MaterializeValidShapeSymbols();

/**
 * @brief Make every FunctionType::Graph function legal to record and replay
 *
 * Hoists each boundary scalar a Graph body *derives* out to its call sites: the
 * host_build_graph runtime tracks a boundary scalar by the address of its
 * argument slot, so a value computed inside the region has no slot and would be
 * frozen at its first-call value on every later replay, with no warning.
 *
 * Also rejects, at compile time, the boundary shapes the runtime would decline
 * to cache — an oversized or empty tensor boundary, runtime-allocated outputs,
 * return values, nested graphs, and call sites carrying explicit dependencies or
 * a dispatch predicate. Almost all of those degrade to a silent non-graph
 * fallback at runtime, which no numerical test can detect.
 *
 * @return Program-level pass
 */
Pass LegalizeGraphBoundary();

/**
 * @brief Create a loop unrolling pass
 *
 * Expands ForStmt nodes with ForKind::Unroll into inlined copies of the loop
 * body, substituting the loop variable with each iteration's constant value.
 * Must run before ConvertToSSA.
 */
Pass UnrollLoops();

/**
 * @brief Skew cross-core (cube/vector) ``pl.pipeline`` loops; runs immediately
 *        before ``LowerPipelineLoops``.
 *
 * For a mixed-core pipeline loop whose body has both a cross-core ``tile.tpush_*``
 * and ``tile.tpop_*`` (``F > 1``), rewrites it to overlap the two cores:
 *   - Single round-trip, producer role (one tpush + one tpop, the tpush's
 *     backward slice does not feed the body via SSA): run the producer one
 *     iteration ahead — produce(start) prologue, a ``ForKind::Sequential`` steady
 *     loop pairing produce(k) with the trailing consume(k-step), and a
 *     consume(last) epilogue.
 *   - Consumer role or multi-round-trip: demote to a plain ``ForKind::Sequential``
 *     loop (order-preserving; cross-core overlap comes from the peer's producer
 *     skew). Demotion avoids reordering the in-order cross-core FIFO.
 *
 * The output carries no ``pipeline_stages`` marker and ``ForKind::Sequential``, so
 * the downstream ``LowerPipelineLoops`` skips it and ``CanonicalizeIOOrder`` does
 * not re-sort the hand-ordered skew. Every NON-cross-core pipeline loop (same-core
 * GM->L1 / L1->L0 / nested matmul stage loops) is left intact for
 * ``LowerPipelineLoops`` to replicate.
 */
Pass SkewCrossCorePipeline();

/**
 * @brief Rotate a ``pl.pipeline`` loop through the slots of one declared
 *        allocation; runs immediately before ``LowerPipelineLoops``.
 *
 * Where ``LowerPipelineLoops`` buys ping-pong by replicating the body ``F`` times
 * so each copy owns a distinct buffer, this pass keeps ONE body and gives the
 * buffer ``F`` slots: every top-level ``tile.load`` / ``tile.read`` in the body
 * whose arguments read the induction variable is rebound onto
 * ``pl.MemRef(name, slots=F)[iv % F]``, and the loop is demoted to
 * ``ForKind::Sequential`` with ``pipeline_stages`` stripped. Bounds, step and
 * ``iter_args`` are untouched, so no remainder dispatch is needed and a dynamic
 * trip count needs no special case.
 *
 * The synthesized MemRef is shaped exactly like an author's declaration, so
 * ``InitMemRef`` resolves it and PTO codegen lowers it to one
 * ``pto.alloc_multi_tile`` plus a ``pto.multi_tile_get`` per use — no new IR op
 * and no new user-facing switch.
 *
 * **Self-gated on ``memory_planner=PTOAS``.** Only that planner emits a ptoas
 * region today — PTO codegen's ``PlanMultiBufferRegions`` bails under the PyPTO
 * planner — so this pass returns the function untouched there and the default
 * pipeline is byte-identical. The gate tracks that codegen limitation, not a
 * limitation of ptoas: an addressed region synchronizes identically at
 * ``--pto-level=level3``. Widening it is follow-up work in the address
 * allocator.
 *
 * Loops it declines — a slot count outside ptoas' ``[2, 16]``, a step other than 1,
 * a start not a multiple of ``F``, a body with no eligible load, a tile in a space
 * other than Vec / Mat / Acc, a runtime valid shape, a tile carried out as a phi or
 * consumed by a view / in-place op, or any loop nested under a pipeline loop this
 * pass already declined — are left intact for ``LowerPipelineLoops`` to replicate.
 * Every rejection is a fallback to the existing behaviour, never an error: codegen
 * refuses a region it cannot describe, so synthesizing a doubtful one would turn a
 * kernel that compiles today into a compile failure.
 */
Pass LowerPipelineToSlots();

/**
 * @brief Lower ``pl.pipeline(N, stage=F)`` loops at the tile level
 *
 * Triggers on ``ForStmt`` nodes with ``kind_ == ForKind::Pipeline`` and
 * ``attrs_["pipeline_stages"] == F`` where ``F > 1``. Produces an outer loop
 * of ``N/F`` iterations whose body is a ``SeqStmts`` of ``F`` deep-cloned
 * copies of the original body, each with the loop variable substituted as
 * ``new_var + k * step``. A trailing remainder covers ``N % F`` if non-zero —
 * a bare ``SeqStmts`` flattened into the outer scope for static bounds, or a
 * cascaded ``IfStmt`` dispatch on ``rem`` for dynamic bounds.
 *
 * The produced outer loop **keeps ``ForKind::Pipeline`` and downgrades
 * ``pipeline_stages`` to ``1``** as the post-lowering marker for the
 * downstream ``CanonicalizeIOOrder`` pass (which scopes its IO reorder to
 * pipeline bodies and demotes the kind / strips the attr on exit). Keeping
 * the (kind, attr) pair together at every observable state preserves the
 * bidirectional structural invariant ``kind == Pipeline ⇔ pipeline_stages
 * attr present`` (verified by ``PipelineLoopValid``), so the IR survives
 * print/parse round-trip throughout. Re-running this pass on its own output
 * sees ``factor == 1`` and skips (idempotent).
 *
 * Runs at the tile level (after NormalizeReturnOrder, before InitMemRef) so
 * each clone's tile variables become candidates for distinct MemRef allocations
 * — enabling ping-pong buffering for the cloned bodies.
 */
Pass LowerPipelineLoops();

/**
 * @brief Canonicalize IO order inside every ``SeqStmts`` in the program
 *
 * For every ``SeqStmts`` with two or more statements, performs a priority-aware
 * stable topological sort over its members, using four priority tiers:
 *   - scalar-producing assigns (e.g. address arithmetic) — lifted as far up as
 *     the dependency graph permits so downstream loads become ready together
 *   - ``tile.load`` / ``tile.read`` assignments — clustered next, near the top
 *   - remaining tile/tensor compute — settles in the middle
 *   - ``tile.store`` / ``tile.write`` calls — sunk as far down as the
 *     dependency graph permits
 *
 * The result is `[scalar…, loads…, tile compute…, stores…]` whenever the
 * dataflow allows. Within replicated regions produced by
 * ``LowerPipelineLoops``, sibling clones' input tiles become co-live near
 * the top and output tiles co-live near the bottom — preventing ``MemoryReuse``
 * from coalescing them and enabling symmetric ping-pong execution.
 *
 * Soundness is enforced by checking the InOut-use discipline via
 * ``stmt_dep::CollectInOutUseDisciplineDiagnostics`` once per function before
 * any reordering. If any diagnostics are present, the pass leaves the function
 * untouched rather than attempting to reorder. Dependency constraints inside
 * each region are derived from ``stmt_dep::BuildStmtDependencyGraph``.
 */
Pass CanonicalizeIOOrder();

/**
 * @brief Transform break/continue into structured control flow
 *
 * Converts BreakStmt/ContinueStmt into equivalent if-else and while constructs.
 * For loops with break: ForStmt is converted to WhileStmt with a break flag.
 * For loops with continue: remaining body is wrapped in else branches.
 * Must run before ConvertToSSA and after UnrollLoops.
 */
Pass CtrlFlowTransform();

/**
 * @brief Create an SSA conversion pass
 */
Pass ConvertToSSA();

/**
 * @brief Outline InCore scopes into separate functions
 *
 * Requirements:
 * - Input IR must be in SSA form (run ConvertToSSA first)
 * - Only processes Opaque functions
 */
Pass OutlineIncoreScopes();

/**
 * @brief Outline Hierarchy scopes into separate functions with level/role
 *
 * Requirements:
 * - Input IR must be in SSA form (run ConvertToSSA first)
 * - Only processes Opaque functions containing Hierarchy scopes
 * - Should run before OutlineIncoreScopes and OutlineClusterScopes
 */
Pass OutlineHierarchyScopes();

/**
 * @brief Outline Cluster scopes into separate Group functions
 *
 * Requirements:
 * - Input IR must be in SSA form (run ConvertToSSA first)
 * - Only processes Opaque/Orchestration functions containing Cluster scopes
 */
Pass OutlineClusterScopes();

/**
 * @brief Convert tensor ops to tile ops in InCore functions
 *
 * Inserts tile.load at InCore function entry, converts tensor ops to tile ops
 * using the OpConversionRegistry, inserts tile.store at exit, and updates
 * orchestration call sites with tensor.create for output parameters.
 *
 * Requirements:
 * - Input IR must have InCore scopes outlined (run OutlineIncoreScopes first)
 */
Pass ConvertTensorToTileOps();

/**
 * @brief Optimize tensor buffer usage in orchestration and InCore functions
 *
 * Three optimization patterns, applied in order:
 * - Pattern 1 (iter-arg reuse): Merges Out params into In params (promoted
 *   to InOut) when the InCore result feeds back as a ForStmt/WhileStmt
 *   iter-arg, eliminating redundant tensor.create per iteration.
 * - Pattern 2 (assemble parent strides): Attaches parent-tensor strides
 *   (via TensorView) to InCore Out params when orchestration uses
 *   tensor.assemble to scatter InCore results into a larger tensor.
 * - Pattern 3 (assemble-loop rewrite): Rewrites InCore ForStmt loops that
 *   accumulate via tile.assemble to use tile.store directly, initializing
 *   the iter-arg from the Out param.
 *
 * Requirements:
 * - Input IR must have tile ops in InCore functions (run ConvertTensorToTileOps first)
 */
Pass OptimizeOrchTensors();

/**
 * @brief Rewrite logical ``pl.NZ`` tensors into pto-isa's blocked NZ form
 *
 * A ``pl.Tensor[[..., R, C], dtype, pl.NZ]`` annotation asserts that the GM
 * bytes are already in PTO-native NZ fractal order while the DSL keeps the
 * logical shape and slicing. pto-isa describes such a buffer with a blocked
 * rank-(r+2) GlobalTensor, so this pass rewrites:
 *
 *   - every NZ ``TensorType`` shape to ``[..., C/c0, R/16, 16, c0]``, where
 *     ``c0`` is the number of elements in a 32-byte C0 line (``256 / bits``);
 *     strides stay empty for ``MaterializeTensorStrides``, whose plain
 *     row-major rule already yields pto-isa's NZ strides once blocked;
 *   - every consuming ``tile.load``'s offsets / shapes / valid_shape into
 *     blocked coordinates, preserving the logical 2-D destination ``TileType``.
 *
 * Milestone 1 scope: read-only, ``target_memory=Mat`` only, whole-byte dtypes,
 * static shapes, ``R % 16 == 0`` and ``C % c0 == 0`` with equally aligned slice
 * offsets. Anything else is rejected rather than silently mis-addressed.
 *
 * Requirements:
 * - Input IR must have tile ops (run ConvertTensorToTileOps first)
 * - Must run **after** FlattenTileNdTo2D (requires ``TileOps2D``): the
 *   destination tile must already be the logical 2-D operand, or the rewritten
 *   ``tile.load`` has a type annotation and argument ranks that cannot both be
 *   printed. FlattenTileNdTo2D skips its ND2NZ window collapse for NZ sources,
 *   so the logical window is still intact here.
 */
Pass BlockNzTensorViews();

/**
 * @brief Flatten ND tile ops to 2D in InCore functions
 *
 * Merges all dimensions except the last into a single dimension.
 * E.g., a tile [A, B, C] becomes [A*B, C]. Inserts tile.reshape
 * after tile.load and before tile.store. Only converts tiles with
 * 3+ dimensions; 1D and 2D tiles are unchanged.
 *
 * Preconditions:
 * - All tile reduce ops must reduce along the last axis
 * - All tile shapes must be static (ConstInt dimensions)
 * - All tile memory must be contiguous
 *
 * Requirements:
 * - Input IR must have tile ops (run ConvertTensorToTileOps first)
 */
Pass FlattenTileNdTo2D();

/**
 * @brief Expand hardware-unsupported ``tile.cast`` pairs into native cast chains.
 *
 * ``pto.tcvt`` only supports a profile-dependent subset of (src, dst) dtype
 * pairs. This pass rewrites each non-native ``tile.cast`` into the shortest
 * sequence of native casts (BFS over the ISA adjacency table for A5 / A2A3).
 * Among equal-length paths it prefers "same byte-width -> float, then adjust
 * width" edges (e.g. A5 ``INT32->FP16`` becomes ``INT32->FP32->FP16``).
 *
 * Already-native casts (including FIXPIPE-foldable ``FP32->BF16/FP16`` with
 * ``mode=rint``) are left untouched. Runs after ``FlattenTileNdTo2D`` (so
 * newly inserted casts are also legalized) and before ``AutoTileMatmulL0``.
 *
 * Requirements:
 * - Input IR must have 2D tile ops (run FlattenTileNdTo2D first)
 * - A BackendHandler must be configured (PassContext) so the arch table
 *   (``a5`` / ``a2a3``) can be selected via ``GetPtoTargetArch()``
 */
Pass LegalizeTileCast();

/**
 * @brief Auto-tile static 2D matmul-family calls for the backend's L0
 *
 * Queries ``utils::ChooseL0Tile`` for an ``(m, n, k, stationarity, dbC)``
 * design point and rewrites Mat-resident operands into aligned Left/Right
 * ``tile.extract`` operations.  K-split reductions use a 2-stage pipelined
 * loop; a non-divisor aligned K tail is peeled into a final
 * ``tile.matmul_acc``.
 *
 * L0C legality uses the backend's physical accumulator-row alignment, which
 * may be stricter than the logical cube shape (for example, an INT32 M=16
 * result occupies 32 rows on Ascend910B).  When that physical ``[M, N]``
 * footprint exceeds L0c, fresh ``tile.matmul`` and ``tile.matmul_bias`` calls
 * are M/N-tiled.  A result consumed by one output store uses direct-to-GM
 * placement; a result consumed entirely as a later matmul operand is assembled
 * into an on-chip Mat scratch.  The Mat-scratch route also folds a compatible
 * f32-to-bf16/f16
 * ``tile.cast(mode="rint")`` into the FIXPIPE writeback.
 *
 * The canonical frontend split-K form -- a full-output accumulator placeholder,
 * a pipeline carrying it through ``tile.matmul`` / ``tile.matmul_acc``, then
 * one store -- is M/N-tiled at the enclosing-loop level.  Each output sub-tile
 * completes the whole source K reduction before the next sub-tile starts, so
 * an oversized full Acc is never materialized.  Arbitrary standalone
 * ``tile.matmul_acc`` calls with caller-owned accumulators remain deferred with
 * ``PH-AT-006``.
 *
 * Full-K M/N grids may use output-, A-, or B-stationary loop orders.  L0C
 * double-buffering is enabled automatically under DSA_RP and PTOAS and is an
 * opt-in under the legacy PYPTO planner.  Under PYPTO, chained Mat-scratch
 * producers remain output-stationary to avoid the allocator offset-packing
 * limitation tracked by issue #1908; some dbC-enabled layouts can still exceed
 * operand capacity there.
 * DSA_RP and PTOAS retain operand-stationary choices because their lifetime-aware
 * placement can subdivide the released operand range.
 *
 * The pass also recognizes a user-authored, static pipeline (stage >= 2, trip
 * count divisible by the stage count) containing exactly one already-L0
 * ``tile.matmul``.  Its selected moving operand must be produced by a direct
 * per-iteration Mat-to-L0 transfer, while the other operand is loop-invariant.
 * Its Acc result must have one canonical loop-carried drain: direct-to-GM
 * ``tile.store`` needs at least four iterations; Acc-to-Mat ``tile.assemble``
 * needs at least eight iterations and an aligned Acc tile occupying at least
 * one quarter of L0C.  When the
 * conservative whole-function Acc footprint (including the physical stage
 * multiplicity of other pipelined Acc producers) plus one slot per profitable
 * loop fits in L0C, it enables the existing two-accumulator drain-overlap
 * schedule automatically under the PyPTO memory planner.  Deeper operand
 * pipelines are scheduled in depth-two compute/drain chunks and still rotate
 * only two L0C slots.  Explicit loop policies, nested control flow, multiple
 * cube matmuls or Acc values, non-canonical drain/yield chains, additional
 * stores, and insufficient L0C capacity stay unchanged.
 *
 * Eligible calls require static 2D operands with B in Mat and A in Mat or Vec.
 * ``tile.matmul_bias`` is supported when both matrix operands are Mat and its
 * static ``[1, N]`` bias is Mat- or Bias-resident and has the accumulator dtype
 * (FP32 for floating-point matrix operands, INT32 for integer operands). The
 * bias is applied once on the first K block. M/N tiling requires a full
 * rectangular ``[1, N]`` defining load (physical shape equals valid shape),
 * reconstructs each N window from that single-use ``tile.load`` when it is
 * separated from the call only by other sibling loads, then moves that
 * independent Mat tile to Bias;
 * candidate N is bounded by the
 * backend's bias-table capacity and its emitted pipeline replication depth.
 * An already-Bias-resident source stays outside the emitted grid and consumes
 * one slot rather than inheriting those replication factors.
 * The backend must support the exact Mat-to-Bias dtype pair, and matrix tile
 * dimensions must satisfy their layout-derived boxed alignment.
 * A manually materialized Left/Right operand is otherwise left untouched, but
 * if its static physical footprint alone exceeds L0A/L0B the pass raises an
 * operation-specific error with the operand name, required/available bytes,
 * source location, and both automatic- and manual-tiling fixes.
 * When the chooser returns the full ``(M, N, K)`` shape, no tiling rewrite is
 * needed, although a chained result may still be remapped to Mat by the
 * compatible cast-fold placement above.  Other unsupported regimes are left
 * untouched; useful deferred cases emit ``PerfHint`` diagnostics.
 * An already-Bias-resident source that itself needs N tiling is deferred
 * because Bias-to-Bias sub-window extraction is unsupported. A Mat source that
 * needs N tiling is also deferred unless it is a single-use 2D load in the same
 * statement scope with only sibling loads between it and the matmul, since
 * reloading across an intervening effect changes snapshot semantics and boxed
 * one-row Mat subviews are not PTOAS-legal. Direct-store placement additionally
 * requires that its consumer store be the first non-load statement after the
 * matmul, so deferred emission cannot move computation across an effect.
 *
 * Requirements:
 * - Input IR must have static 2D tile ops (run FlattenTileNdTo2D first)
 */
Pass AutoTileMatmulL0();

/**
 * @brief Canonicalize Mat-resident ``tile.slice`` into ``tile.extract``
 *
 * A ``tile.slice`` whose result tile is ``Mem.Mat`` is a legal high-level
 * "sub-window of a Mat tile" construct (e.g. emitted by ``FlattenTileNdTo2D``
 * when it unrolls a ``tile.batch_matmul`` batch dimension).  It has no direct
 * hardware lowering — codegen would materialize it as an unsupported
 * ``loc=mat -> loc=mat`` ``pto.tmov``.
 *
 * This pass lowers every Mat-resident ``tile.slice`` into the canonical
 * ``tile.extract`` form by folding the slice offset into its consumer:
 * - consumed by ``tile.extract`` — the slice offset is added to the extract
 *   index and the extract reads the slice's source directly;
 * - consumed by a ``tile.matmul`` family operand — the operand is replaced by
 *   a ``tile.extract(src, off, shape, target=Left|Right)``.
 *
 * The now-dead ``tile.slice`` is dropped.  Result: Mat->Left/Right movement
 * is unified on ``tile.extract`` / ``pto.textract``.
 *
 * Requirements:
 * - Input IR must have tile ops in 2D form; runs after ``AutoTileMatmulL0``
 */
Pass CanonicalizeTileSlice();

/**
 * @brief Infer target memory space for TileType variables in InCore functions
 *
 * Sets TileType::memory_space_ based on the producing tile operation:
 * - tile.load/tile.move/tile.create: from target_memory kwarg
 * - tile.matmul and variants: Acc
 * - tile.reshape: inherit from first tile-typed input
 * - Other tile ops: Vec (default)
 *
 * The pass also runs a focused internal transform that hoists
 * compiler-generated, loop-invariant GM->Mat matmul operand paths across
 * statically non-empty sequential loops.  The rewrite
 * requires a direct root-orchestration Call whose candidate storage was
 * created by tensor.create, plus a capacity-safe whole-function
 * Mat/Left/Right footprint.  K-tiled fanout may retain the whole GM->Mat panel
 * while leaving K-dependent Left/Right staging inside its original pipeline.
 * External inputs, Submit sites, and direct/external InCore entries decline.
 * Private bridge provenance is consumed before the pass returns.
 *
 * Requirements:
 * - Input IR must have tile ops (run ConvertTensorToTileOps first)
 */
Pass InferTileMemorySpace();

/**
 * @brief Insert tile.tget_scale_addr bindings before MX matmul consumers
 *
 * After InferTileMemorySpace has resolved Left/LeftScale and Right/RightScale
 * operand spaces, inserts compiler-generated ``tile.tget_scale_addr(scale, data)``
 * immediately before each ``tile.matmul_mx`` / ``_acc`` / ``_bias`` and rewrites
 * the matmul to consume the bound scale SSA values.
 *
 * Bindings are not reused across consumers because tget mutates a shared
 * physical scale buffer whose aliases cannot be represented by SSA identity.
 * The pass therefore inserts a fresh binding at every consumer even when its
 * scale operand is already the result of an earlier binding.
 *
 * Requirements:
 * - Tile memory spaces must already be inferred (``TileMemoryInferred``)
 * - Statement structure must be normalized (``NormalizedStmtStructure``)
 */
Pass InsertMxScaleAddr();

/**
 * @brief Materialize implicit ND/DN strides on every TensorType (RFC #1300 §2.4)
 *
 * Walks every TensorType reachable from the program and rewrites any
 * ``view.has_value() && view.stride.empty()`` slot to its packed canonical
 * form (per ``BuildLogicalStridesFromLayout``). Bare TensorTypes
 * (``!view.has_value()``) are left untouched — they are implicitly
 * ND-packed and the strict ``TensorViewCanonical`` verifier accepts them.
 *
 * After this pass runs, the codegen-entry contract holds: every TensorType
 * that carries a TensorView has explicit stride matching its layout / shape.
 * The pass is idempotent — re-running it on already-canonical IR is a no-op.
 *
 * Produces ``IRProperty::TensorViewCanonical`` so PassPipeline auto-verifies
 * (via the registry's weak-mode verifier; the strict form is a P3 follow-up
 * once consumers depend on materialized stride).
 */
Pass MaterializeTensorStrides();

/**
 * @brief Repair backend-required layouts for constrained elementwise tile ops
 *
 * For current layout-constrained elementwise ops, rewrites `[N, 1]`
 * col-major vector inputs into `[1, N]` row-major reshapes at the use-site,
 * executes the consumer in row-major form, and reshapes the result back when
 * the original output is a col-major column vector.
 */
Pass ResolveBackendOpLayouts();

/**
 * @brief Expand mixed InCore functions into AIC + AIV + Group
 *
 * Splits InCore functions containing both Cube ops (tile.matmul) and Vector ops
 * (tile.load, tile.add, etc.) into separate AIC and AIV kernels communicating
 * via TPUSH/TPOP, wrapped in a Group function.
 *
 * Requirements:
 * - Input IR must have tile ops (run ConvertTensorToTileOps first)
 * - Input IR must have InCore scopes outlined (run OutlineIncoreScopes first)
 */
Pass ExpandMixedKernel();

/**
 * @brief Lower AUTO pl.split mixed InCore functions into the explicit split_aiv
 *        form before ExpandMixedKernel (RFC #1300 staged convergence).
 *
 * For each mixed InCore function carrying a function-level split mode (UP_DOWN /
 * LEFT_RIGHT), inserts tile.aiv_shard at C->V boundaries and tile.aic_gather at
 * V->C boundaries, halves only the VECTOR sub-region (affinity-gated reuse of the
 * split_axis machinery), injects get_subblock_idx, and stamps split + split_aiv.
 * ExpandMixedKernel then folds aiv_shard/aic_gather into split-stamped tpush/tpop
 * via its op-driven boundary arm, and SplitVectorKernel takes its "already
 * explicit" arm (attribute stamping only).
 *
 * This is the LIVE auto-split lowering path: it always runs, immediately before
 * ExpandMixedKernel. SplitVectorKernel's former per-op halving driver was deleted
 * once this pass became unconditional — the halving machinery now lives only in
 * split_axis_utils, shared by this pass.
 */
Pass LowerAutoVectorSplit();

/**
 * @brief Inject __gm_pipe_buffer workspace parameter for cross-core pipes
 *
 * Backend-gated (BackendHandler::RequiresGMPipeBuffer()). On Ascend910B the
 * cross-core tpush/tpop path rides through a shared GM buffer; this pass adds
 * the workspace parameter and propagates it upward through callers, stopping
 * at Orchestration functions which materialize the buffer locally.
 *
 * Must run after ExpandMixedKernel.
 */
Pass InjectGMPipeBuffer();

/**
 * @brief Split vector kernel pass
 *
 * For AIV/AIC functions with a non-None split mode:
 * 1. Sets the split kwarg on tpush/tpop operations
 * 2. Halves the tpop result tile shape in the split dimension
 * 3. Adjusts tile.store offsets for tiles originating from tpop
 *
 * Must run after ExpandMixedKernel.
 */
Pass SplitVectorKernel();

/**
 * @brief Create a verifier pass with opt-in property verification
 *
 * @param properties Properties to verify. Pass GetDefaultVerifyProperties() for the default set.
 * @return Pass that runs IR verification for the given properties
 */
Pass RunVerifier(const IRPropertySet& properties);

/**
 * @brief Simplify scalar expressions and statements in the program
 *
 * Uses algebraic rewrite rules and bound analysis to reduce complexity.
 * Automatically binds ForStmt loop variables to their iteration ranges for
 * range-aware simplification (e.g., i // 8 == 0 when i is in [0, 8)).
 * Propagates if-branch constraints for tighter bounds in then/else bodies.
 */
Pass Simplify();

/**
 * @brief Decompose composite tile/distributed ops into primitive ops.
 *
 * Lowering rules live in a file-local dispatch table inside
 * ``src/ir/transforms/lower_composite_ops_pass.cpp``. Today the pass handles
 * ``tile.sin`` / ``tile.cos`` and explicit-signal InCore
 * ``pld.tensor.allreduce``; host-level allreduce is skipped and lowered later
 * by ``LowerHostTensorCollectives``. Future composite ops (softmax, gelu,
 * layernorm, ...) are added by appending a rule function + one dispatch-table
 * row, without touching the mutator.
 *
 * FP32-only for the trig rules — non-FP32 inputs are rejected at
 * op-construction time by the op deducer, never reaching this pass.
 *
 * Idempotent: every lowering rule must emit only primitive ops that are not
 * themselves in the dispatch table, so running the pass twice yields the same
 * IR after the first run.
 */
Pass LowerCompositeOps();

/**
 * @brief Create a pass that flattens nested call expressions
 */
Pass FlattenCallExpr();

/**
 * @brief Create a pass that normalizes statement structure
 */
Pass NormalizeStmtStructure();

/**
 * @brief Normalize return tuple order in InCore functions
 *
 * Reorders ReturnStmt::value_ so that return[i] corresponds to the i-th
 * Out/InOut parameter in declaration order, and updates TupleGetItemExpr
 * indices at call sites accordingly.  After this pass, orchestration codegen
 * can map tuple element indices to output parameters sequentially without
 * tracing through tile.store / ForStmt yield chains.
 *
 * Requirements:
 * - Input IR must have InCore scopes outlined and tile ops
 */
Pass NormalizeReturnOrder();

/**
 * @brief Fuse tensor.create + tensor.assemble into tensor.slice in Orchestration functions
 *
 * When a tensor.create result is assembled into a target tensor exactly once,
 * replaces create with tensor.slice(target, shape, offsets) and removes the assemble.
 * This enables the orchestration codegen to emit .view() directly.
 *
 * Requirements:
 * - Must run after AllocateMemoryAddr (pipeline final position)
 * - Only processes Orchestration functions
 */
Pass FuseCreateAssembleToSlice();

/**
 * @brief Derive Call::GetArgDirections() (stored in attrs_["arg_directions"])
 *        from callee param directions and buffer lineage.
 *
 * For every non-builtin call in Orchestration / Group / Spmd functions,
 * compute the runtime call-site direction
 * (Input/Output/InOut/OutputExisting/Scalar) for each argument and write it
 * into Call::attrs_ under the reserved key ``"arg_directions"``.
 *
 * Mapping:
 *   - scalar argument                        -> ArgDirection::Scalar
 *   - tensor + callee dir == In              -> ArgDirection::Input
 *   - tensor + callee dir == InOut           -> ArgDirection::InOut
 *   - tensor + callee dir == Out, locally allocated buffer
 *                                            -> ArgDirection::InOut (WAW promotion)
 *   - tensor + callee dir == Out, external (param-rooted) buffer
 *                                            -> ArgDirection::OutputExisting
 *
 * Builtin ops (tensor.*, tile.*, system.*) are left untouched (arg_directions empty).
 *
 * Manual-scope dependency edges (typed ``Submit::deps_``) are written directly
 * by the parser from a ``pl.submit(...)`` ``deps=[...]`` kwarg — this pass
 * does not synthesise or lower them (ManualDepsOnSubmitOnly invariant).
 *
 * Requirements:
 *   - InCore scopes outlined (run OutlineIncoreScopes first)
 */
Pass DeriveCallDirections();

/**
 * @brief Expand profitable manual_scope Array[TASK_ID] fanout deps into
 *        explicit dependency-only dummy barrier calls.
 *
 * Rewrites selected consumer Submits' ``deps_=[source_array]`` to
 * ``deps_=[barrier_tid]`` after inserting a marked ``system.task_dummy``
 * assignment at the chosen phase-fence placement point.
 */
Pass ExpandManualPhaseFence();

/**
 * @brief Derive explicit task-to-task dependency edges inside runtime scopes.
 *
 * User-written manual runtime scopes are skipped: the user's explicit
 * ``deps=[...]`` edges are treated as the complete scheduling contract, and the
 * pass does not rewrite their call-site directions to ``NoDep`` or
 * ``OutputExisting``. AUTO scopes are skipped by default; pass
 * ``analyze_auto_scopes=true`` to analyze them while keeping ``manual=false`` in
 * the output IR. For each analyzed AUTO scope, the pass computes a conservative
 * storage access summary from
 * ``arg_directions`` and attaches RAW/WAR/WAW hazards against prior calls in the
 * same scope under ``Call.attrs["compiler_manual_dep_edges"]``. On unanalyzable
 * hazards, partial compiler deps are stripped and AUTO tracking remains active.
 *
 * User-provided ``manual_dep_edges`` remain authoritative and separate; codegen
 * merges both attrs before emitting ``Arg::set_dependencies``.
 * Requirements:
 *   - Call directions resolved (run DeriveCallDirections first)
 */
Pass AutoDeriveTaskDependencies(bool analyze_auto_scopes = false);

/**
 * @brief Fold no-op tile.reshape assignments into Var-to-Var assignments
 *
 * After InitMemRef and MaterializeSemanticAliases have finalized allocation
 * identities, two TileType variables can share the same MemRef and the same TileBufSignature — in that
 * case the `tile.reshape` connecting them is a no-op at the PTO level. This pass rewrites such
 * `lhs = tile.reshape(rhs, shape)` AssignStmts into plain `lhs = rhs`,
 * removing the reshape Call. PTO codegen previously dropped the emission
 * via a peephole; folding into the IR makes codegen 1:1.
 *
 * Requirements:
 * - InCore-type functions only (Opaque/Orchestration are unaffected)
 * - Must run after semantic alias materialization; PyPTO-owned planners also
 *   run AllocateMemoryAddr first, while PTOAS keeps the finalized root identity
 *   and assigns physical addresses later
 */
Pass FoldNoOpReshape();

/**
 * @brief Materialize implicit orchestration scopes as explicit RuntimeScopeStmt nodes
 *
 * The simpler runtime wraps regions of an Orchestration function in
 * ``SIMPLER_SCOPE()`` blocks. Historically the orchestration codegen decided where
 * to emit those wrappers from the for/if structure: the whole function body, and
 * each ForStmt / IfStmt branch body, were wrapped implicitly (suppressed inside a
 * manual ``RuntimeScopeStmt``). That embedded codegen policy in the printer.
 *
 * This pass moves the policy into the IR. For every ``FunctionType::Orchestration``
 * function it inserts explicit AUTO ``RuntimeScopeStmt`` (``manual_ = false``) nodes:
 *  - wrapping the entire function body, and
 *  - wrapping each ForStmt body and each IfStmt then/else body,
 *
 * while skipping insertion anywhere inside a manual ``RuntimeScopeStmt`` (the
 * runtime forbids AUTO nested in MANUAL). Codegen then emits ``SIMPLER_SCOPE`` only
 * from ``RuntimeScopeStmt`` nodes, staying 1:1 with the IR.
 *
 * Runs last in the pipeline (after the final Simplify) so no other transform has
 * to reason about the inserted scopes. Only Orchestration functions are touched.
 */
Pass MaterializeRuntimeScopes();

/**
 * @brief Classify ForStmt iter_arg carries and size TaskId array carries
 *
 * An Orchestration ``ForStmt`` iter_arg lowers one of two ways:
 *  - **trivial**: the yield value aliases the iter_arg (same backing buffer), so
 *    iter_arg and return_var share the init value's emit name. Materialising a
 *    fresh ``ChipTensor`` would break the runtime dependency tracker, which keys off
 *    ``ChipTensor*`` identity.
 *  - **rebind**: the yield value is a different buffer, so a mutable carry
 *    variable is declared and the yield assigns back to it (issue #1286).
 *
 * Inside a ``pl.manual_scope`` a ``Scalar[TASK_ID]`` carry additionally lowers to
 * a ``TaskId[N]`` array whose extent N comes from the loop's (or a threaded
 * inner loop's) constant trip count.
 *
 * The orchestration codegen used to derive both from an alias-equivalence
 * fixpoint over the loop body. This pass moves that analysis into the IR: it
 * stamps ``iter_arg_rebind_<i>`` (bool, every slot) and ``iter_arg_array_size_<i>``
 * (int, positive extents only) onto ``ForStmt::attrs_``, and codegen degenerates
 * to an attr read. Only ``FunctionType::Orchestration`` functions are touched.
 *
 * Runs last, after ``MaterializeRuntimeScopes``, so the classified IR is exactly
 * the IR codegen lowers.
 */
Pass ClassifyIterArgCarry();

/**
 * @brief Copy each cross-core tpop's split/pipe-id onto its matching tfree op
 *
 * A `system.tfree_to_ai{c,v}` carries no split/id of its own — those live on the
 * matching `tile.tpop_from_ai{c,v}` call. This pass stamps them onto the tfree op
 * so codegen reads them directly from the op (no codegen-side tpop lookup table).
 * Covers both finalizer-created (mixed-kernel) and user-written (explicit AIC/AIV)
 * tfrees. Runs late (after split is finalized on tpops), before codegen.
 */
Pass StampTfreeSplit();

/**
 * @brief Insert the ptoas data-before-signal markers around cross-rank publish
 *        and consume points (all via `system.cacheinvalid`).
 *
 * The `pld.system.notify` itself needs no marker:
 *   - after each **local** publishing write (`tile.store` or `tensor.write` into
 *     a window-bound destination, or `get` into a window-bound local
 *     destination): a region `system.cacheinvalid` of the written region
 *     immediately followed by a GM `system.fence`;
 *   - after each **remote** publishing write (`remote_store` / `put`): only a GM
 *     `system.fence`. Its data lands at a peer-offset address whose offset is not
 *     yet expressible in the IR, so the peer-region `pto.cmo.cacheinvalid` is
 *     emitted by the op's codegen as a workaround; the release fence is always an
 *     explicit `system.fence` op inserted here (codegen must not embed it);
 *   - after each **opaque** publishing write (a `Submit`, or a call to an
 *     unregistered user function whose body is not analysed here, so it has no
 *     single addressable region): a conservative no-arg (whole-GM)
 *     `system.cacheinvalid` followed by a GM `system.fence`;
 *   - after each **wait** (`pld.system.wait`): a no-arg (whole-GM)
 *     `system.cacheinvalid`.
 *
 * The pass carries no control-flow state and is idempotent. Runs last, after all
 * statement-reordering passes, so the markers stay adjacent through codegen.
 */
Pass InsertCommFence();

/**
 * @brief Verify properties on a program and throw on errors
 *
 * Uses PropertyVerifierRegistry to verify the given properties and throws
 * a VerificationError if any errors are found. Used by PassPipeline::Run()
 * and the Python dump_ir path for automatic verification.
 *
 * @param properties Properties to verify
 * @param program Program to verify
 * @param pass_name Name of the pass that produced these properties (for error context)
 */
void VerifyProperties(const IRPropertySet& properties, const ProgramPtr& program,
                      const std::string& pass_name);

}  // namespace pass

/**
 * @brief A pipeline of passes executed in sequence
 *
 * PassPipeline maintains an ordered sequence of passes and executes them in order.
 * Instrumentation (verification, logging, etc.) is handled by PassContext and its
 * PassInstruments — the pipeline itself is a simple pass list.
 *
 * Usage:
 * @code
 *   PassPipeline pipeline;
 *   pipeline.AddPass(pass::ConvertToSSA());
 *   pipeline.AddPass(pass::FlattenCallExpr());
 *   pipeline.AddPass(pass::RunVerifier(GetDefaultVerifyProperties()));
 *   auto result = pipeline.Run(program);
 * @endcode
 */
class PassPipeline {
 public:
  PassPipeline();

  /**
   * @brief Add a pass to the pipeline
   */
  void AddPass(Pass pass);

  /**
   * @brief Execute all passes in sequence
   * @param program Input program
   * @return Transformed program
   */
  [[nodiscard]] ProgramPtr Run(const ProgramPtr& program) const;

  /**
   * @brief Get the names of all passes in the pipeline
   */
  [[nodiscard]] std::vector<std::string> GetPassNames() const;

  /**
   * @brief Get copies of the passes in execution order
   *
   * Pass is a lightweight shared handle, so returning a vector by value keeps
   * the pipeline's ordered storage private while allowing callers to inspect or
   * compose a new pipeline without maintaining a second pass list.
   */
  [[nodiscard]] std::vector<Pass> GetPasses() const;

 private:
  std::vector<Pass> passes_;
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_PASSES_H_
