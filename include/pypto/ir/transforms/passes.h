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
 * @brief Create a memory reuse pass
 *
 * Uses dependency analysis to identify memory reuse opportunities.
 * Variables with non-overlapping lifetimes in the same memory space can share MemRef objects.
 */
Pass MemoryReuse();

/**
 * @brief Create a PTO buffer reuse legalisation pass
 *
 * After generic MemoryReuse, multiple tile variables with different
 * TileBufSignatures may share the same MemRef.  PTO codegen requires that
 * every non-view writer sharing a MemRef produces the same typed alloc_tile
 * signature.  This pass detects illegal cross-type sharing and splits the
 * offending MemRef into distinct allocations.
 */
Pass LegalizePTOBufferReuse();

/**
 * @brief Create an allocate memory address pass
 *
 * Allocates real memory addresses for existing alloc operations.
 * Updates MemRef addresses and alloc statement arguments in place.
 */
Pass AllocateMemoryAddr();

/**
 * @brief Create a loop chunking pass
 *
 * Splits ForStmt nodes with chunk_size into nested loops: an outer loop
 * iterating over chunk indices and an inner loop iterating within each chunk.
 * Requires SSA form input and produces SSA form output.
 */
Pass SplitChunkedLoops();

/**
 * @brief Interchange chunk loops and insert InCore scopes
 *
 * Reorders nested ChunkOuter/ChunkInner loop pairs so that all outer loops
 * are on top, then wraps the inner loops + body in a ScopeStmt(InCore).
 * Only interchanges when all ChunkInner loops are Parallel.
 * Requires SSA form input and produces SSA form output.
 */
Pass InterchangeChunkLoops();

/**
 * @brief Create a loop unrolling pass
 *
 * Expands ForStmt nodes with ForKind::Unroll into inlined copies of the loop
 * body, substituting the loop variable with each iteration's constant value.
 * Must run before ConvertToSSA.
 */
Pass UnrollLoops();

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
 * @brief Auto-tile Mat-resident matmul / matmul_acc into a C-stationary K-loop
 *
 * For each ``tile.matmul`` or ``tile.matmul_acc`` whose Mat operands have
 * static 2D shape, queries ``utils::ChooseL0Tile`` against the active
 * ``BackendHandler``'s L0 capacities and rewrites the call into a single
 * ``range(0, K, k)`` loop:
 *
 *   - For ``tile.matmul``: the loop body branches on ``ko == 0`` between
 *     ``tile.matmul`` (fresh accumulator) and ``tile.matmul_acc``
 *     (accumulating into the iter-arg).  The iter-arg init is an Acc-resident
 *     ``tile.create`` placeholder so the iter-arg / yield / return_var chain
 *     is Acc-typed end-to-end.
 *   - For ``tile.matmul_acc``: every iteration is ``tile.matmul_acc``; the
 *     iter-arg init is the caller-provided accumulator directly, so the
 *     accumulator chain is uniform from the first iteration and no if-else
 *     is needed.
 *
 * Operand extraction uses ``tile.extract(src, idx_row, idx_col, shape,
 * target_memory=Left|Right)`` directly — the SSA-form fusion of the older
 * ``tile.slice`` (Mat-resident result) + ``tile.mov`` (Mat→Left/Right) pair.
 * No intermediate Mat-resident slice tile is materialised, and the call
 * lowers to ``pto.textract`` rather than ``pto.subview``.
 *
 * The K-loop is marked ``ForKind::Pipeline`` + ``pipeline_stages=2`` so the
 * downstream ``LowerPipelineLoops`` pass clones the body for a 2-deep
 * ping-pong on the per-iter Mat→Left/Right extracts.
 *
 * Supported today (extensions to follow):
 *   - ``tile.matmul`` and ``tile.matmul_acc``.  ``tile.matmul_bias`` is
 *     deferred (bias add only after the final iteration needs extra
 *     rewriting).
 *   - K tiling only (``m == M`` and ``n == N``).  Cases where the chooser
 *     selects ``m < M`` or ``n < N`` emit a ``PerfHint`` and skip.
 *   - ``K % k == 0``.  K-boundary handling is deferred.
 *
 * Already-L0-sized matmuls (chooser returns ``(M, N, K)``) are left
 * untouched.  Runs after ``FlattenTileNdTo2D``: by the time this pass runs,
 * all tile ops have static 2D shapes.
 *
 * Requirements:
 * - Input IR must have tile ops in 2D form (run FlattenTileNdTo2D first)
 */
Pass AutoTileMatmulL0();

/**
 * @brief Infer target memory space for TileType variables in InCore functions
 *
 * Sets TileType::memory_space_ based on the producing tile operation:
 * - tile.load/tile.move/tile.create: from target_memory kwarg
 * - tile.matmul and variants: Acc
 * - tile.reshape: inherit from first tile-typed input
 * - Other tile ops: Vec (default)
 *
 * Requirements:
 * - Input IR must have tile ops (run ConvertTensorToTileOps first)
 */
Pass InferTileMemorySpace();

/**
 * @brief Resolve transpose layout for tile.load with transpose=True
 *
 * For each InCore function, detects tile.load(..., transpose=True) whose source
 * is a function parameter and annotates that parameter's TensorType with the DN
 * (column-major) layout. The shape is preserved -- DN is a codegen hint only.
 * Orchestration and Opaque functions are returned unchanged.
 *
 * Requirements:
 * - Input IR must have tile ops (run ConvertTensorToTileOps first)
 * - Input IR must have InCore scopes outlined (run OutlineIncoreScopes first)
 */
Pass ResolveTransposeLayout();

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
 * @brief Derive Call::GetArgDirections() (stored in attrs_["arg_directions"]) from
 *        callee param directions and buffer lineage.
 *
 * For every non-builtin call in Orchestration / Group / Spmd functions,
 * compute the runtime call-site direction (Input/Output/InOut/OutputExisting/Scalar)
 * for each argument and write it into Call::attrs_ under the reserved key
 * ``"arg_directions"``.
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
 * Requirements:
 *   - InCore scopes outlined (run OutlineIncoreScopes first)
 */
Pass DeriveCallDirections();

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

 private:
  std::vector<Pass> passes_;
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_PASSES_H_
