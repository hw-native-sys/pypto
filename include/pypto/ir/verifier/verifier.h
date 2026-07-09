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

#ifndef PYPTO_IR_VERIFIER_VERIFIER_H_
#define PYPTO_IR_VERIFIER_VERIFIER_H_

#include <memory>
#include <string>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/program.h"

namespace pypto {
namespace ir {

/**
 * @brief Base class for IR property verifiers
 *
 * Each verifier implements a specific check on IR programs.
 * Verifiers can detect errors or warnings and add them to a diagnostics vector.
 * Each verifier receives a ProgramPtr and internally decides whether to iterate
 * over functions or check program-level properties.
 *
 * To create a new property verifier:
 * 1. Inherit from PropertyVerifier
 * 2. Implement GetName() to return a unique name
 * 3. Implement Verify() to perform the verification logic
 *
 * Example:
 * @code
 *   class MyVerifier : public PropertyVerifier {
 *    public:
 *     std::string GetName() const override { return "MyVerifier"; }
 *     void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
 *       for (const auto& [gv, func] : program->functions_) {
 *         // Verification logic per function
 *       }
 *     }
 *   };
 * @endcode
 */
class PropertyVerifier {
 public:
  virtual ~PropertyVerifier() = default;

  /**
   * @brief Get the name of this verifier
   * @return Unique name (e.g., "SSAVerify", "TypeCheck")
   */
  [[nodiscard]] virtual std::string GetName() const = 0;

  /**
   * @brief Verify a program and collect diagnostics
   * @param program Program to verify
   * @param diagnostics Vector to append diagnostics to
   *
   * This method should examine the program and add any detected issues
   * to the diagnostics vector. It should not throw exceptions - all issues
   * should be reported through diagnostics.
   */
  virtual void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) = 0;
};

/// Shared pointer to a property verifier
using PropertyVerifierPtr = std::shared_ptr<PropertyVerifier>;

/**
 * @brief Factory function for creating SSA property verifier
 * @return Shared pointer to SSA PropertyVerifier
 */
PropertyVerifierPtr CreateSSAPropertyVerifier();

/**
 * @brief Factory function for creating type check property verifier
 * @return Shared pointer to TypeCheck PropertyVerifier
 */
PropertyVerifierPtr CreateTypeCheckPropertyVerifier();

/**
 * @brief Factory function for creating no nested call property verifier
 * @return Shared pointer to NoNestedCall PropertyVerifier
 */
PropertyVerifierPtr CreateNoNestedCallPropertyVerifier();

/**
 * @brief Factory function for creating NormalizedStmtStructure property verifier
 * @return Shared pointer to NormalizedStmtStructure PropertyVerifier
 */
PropertyVerifierPtr CreateNormalizedStmtPropertyVerifier();

/**
 * @brief Factory function for creating NoRedundantBlocks property verifier
 *
 * Verifies that no SeqStmts has exactly one child (should be unwrapped),
 * and no SeqStmts contains a nested SeqStmts (should be flattened).
 * @return Shared pointer to NoRedundantBlocks PropertyVerifier
 */
PropertyVerifierPtr CreateNoRedundantBlocksPropertyVerifier();

/**
 * @brief Factory function for creating SplitIncoreOrch property verifier
 * @return Shared pointer to SplitIncoreOrch PropertyVerifier
 */
PropertyVerifierPtr CreateSplitIncoreOrchPropertyVerifier();

/**
 * @brief Factory function for creating ClusterOutlined property verifier
 * @return Shared pointer to ClusterOutlined PropertyVerifier
 */
PropertyVerifierPtr CreateClusterOutlinedPropertyVerifier();

/**
 * @brief Factory function for creating HierarchyOutlined property verifier
 * @return Shared pointer to HierarchyOutlined PropertyVerifier
 */
PropertyVerifierPtr CreateHierarchyOutlinedPropertyVerifier();

/**
 * @brief Factory function for creating HasMemRefs property verifier
 * @return Shared pointer to HasMemRefs PropertyVerifier
 */
PropertyVerifierPtr CreateHasMemRefsPropertyVerifier();

/**
 * @brief Factory function for creating IncoreTileOps property verifier
 * @return Shared pointer to IncoreTileOps PropertyVerifier
 */
PropertyVerifierPtr CreateIncoreTileOpsPropertyVerifier();

/**
 * @brief Factory function for creating MixedKernelExpanded property verifier
 *
 * Verifies that no InCore function contains both Cube and Vector tile ops, and
 * that split AIC/AIV functions keep cross-core tpop results in their required
 * bridge memory spaces.
 * @return Shared pointer to MixedKernelExpanded PropertyVerifier
 */
PropertyVerifierPtr CreateMixedKernelExpandedPropertyVerifier();

/**
 * @brief Factory function for creating AivSplitValid property verifier
 *
 * Structural verifier for the first-class ``SplitAivScopeStmt`` region (live
 * between OutlineIncoreScopes and LowerAutoVectorSplit). Keyed on the node, it
 * checks, per region: (a) no cube compute inside a region (each AIV lane holds
 * only half the tile, so cube ops cannot be vector-split); (b) no AIV reduce
 * over the split axis inside a region (partial per-lane reduction); (c) the
 * ``tile.aiv_shard`` / ``tile.aic_gather`` boundary ops appear only inside a
 * region. Full-width vector compute outside a region is legal (multi-mode), so
 * "bare vector compute outside a region" is intentionally not checked.
 * @return Shared pointer to AivSplitValid PropertyVerifier
 */
PropertyVerifierPtr CreateAivSplitValidPropertyVerifier();

/**
 * @brief Factory function for creating AllocatedMemoryAddr property verifier
 *
 * Verifies that all non-DDR MemRefs have valid allocated addresses and
 * that total memory usage per space does not exceed platform buffer limits.
 * @return Shared pointer to AllocatedMemoryAddr PropertyVerifier
 */
PropertyVerifierPtr CreateAllocatedMemoryAddrPropertyVerifier();

/**
 * @brief Factory function for creating TileOps2D property verifier
 *
 * Verifies that all tile op calls (excluding tile.load, tile.store,
 * tile.reshape) in InCore functions operate on ≤2D tiles.
 * @return Shared pointer to TileOps2D PropertyVerifier
 */
PropertyVerifierPtr CreateTileOps2DPropertyVerifier();

/**
 * @brief Factory function for creating BreakContinueCheck property verifier
 *
 * Verifies that break/continue statements only appear inside sequential
 * (ForKind::Sequential) or while loops. Reports errors for break/continue
 * in parallel or unrolled loops, or outside any loop.
 * @return Shared pointer to BreakContinueCheck PropertyVerifier
 */
PropertyVerifierPtr CreateBreakContinuePropertyVerifier();

/**
 * @brief Factory function for creating InlineFunctionsEliminated property verifier
 *
 * Verifies that no FunctionType::Inline function and no Call expression
 * resolving to one survives in the program. Should hold after the
 * InlineFunctions pass runs (i.e. for the entire post-pass-01 pipeline).
 * @return Shared pointer to InlineFunctionsEliminated PropertyVerifier
 */
PropertyVerifierPtr CreateInlineFunctionsEliminatedPropertyVerifier();

/**
 * @brief Factory function for creating TileMemoryInferred property verifier
 *
 * Verifies that all TileType variables in InCore functions have
 * memory_space_ set (not nullopt).
 * @return Shared pointer to TileMemoryInferred PropertyVerifier
 */
PropertyVerifierPtr CreateTileMemoryInferredPropertyVerifier();

/**
 * @brief Factory function for creating UseAfterDef property verifier
 *
 * Verifies that every Var reference in an expression is dominated by a
 * definition (function parameter, ForStmt/WhileStmt loop variable,
 * iter_arg, return_var, or AssignStmt).
 * @return Shared pointer to UseAfterDef PropertyVerifier
 */
PropertyVerifierPtr CreateUseAfterDefPropertyVerifier();

/**
 * @brief Factory function for creating StructuredCtrlFlow property verifier
 *
 * Verifies that no BreakStmt or ContinueStmt remains in InCore-type function
 * bodies (InCore, AIC, AIV). Host and Orchestration functions are skipped
 * because they support break/continue natively.
 * @return Shared pointer to StructuredCtrlFlow PropertyVerifier
 */
PropertyVerifierPtr CreateStructuredCtrlFlowPropertyVerifier();

/**
 * @brief Factory function for creating OutParamNotShadowed property verifier
 *
 * Verifies that no Out/InOut function parameter is reassigned via a
 * tensor-creating call (tensor.create, tensor.full), which would shadow
 * the external output tensor.
 * @return Shared pointer to OutParamNotShadowed PropertyVerifier
 */
PropertyVerifierPtr CreateOutParamNotShadowedPropertyVerifier();

/**
 * @brief Factory function for creating ArrayNotEscaped property verifier
 *
 * Verifies that ``ArrayType`` never appears as a function parameter or
 * return type. ArrayType values live on the on-core scalar register file /
 * C stack and cannot be passed across function boundaries (the storage
 * backing them disappears when the function returns). Local create/use
 * within a function/region is allowed; this verifier only blocks escape.
 *
 * @return Shared pointer to ArrayNotEscaped PropertyVerifier
 */
PropertyVerifierPtr CreateArrayNotEscapedPropertyVerifier();

/**
 * @brief Factory function for creating NoNestedInCore property verifier
 *
 * Verifies that no ScopeStmt(InCore) is nested inside another ScopeStmt(InCore).
 * @return Shared pointer to NoNestedInCore PropertyVerifier
 */
PropertyVerifierPtr CreateNoNestedIncorePropertyVerifier();

/**
 * @brief Factory function for creating InOutUseValid property verifier
 *
 * Verifies the InOut-use discipline (RFC #1026): no statement reachable in CFG
 * order from a user-function call that passes variable `v` as InOut or Out may
 * read `v`. Post-mutation values must flow through the call's return slots.
 * Built-in ops (tile.*, tensor.*, system.*) are out of scope — their memory
 * effects are handled separately.
 * @return Shared pointer to InOutUseValid PropertyVerifier
 */
PropertyVerifierPtr CreateInOutUseValidPropertyVerifier();

/**
 * @brief Factory function for creating PipelineLoopValid property verifier
 *
 * Verifies the bidirectional structural invariant on every ``ForStmt``:
 *   - ``kind_ == ForKind::Pipeline``  ⇔  ``HasAttr("pipeline_stages")``
 *
 * Either direction failing indicates a malformed pipeline loop. Listed in
 * ``GetStructuralProperties()``, so ``VerificationInstrument`` checks it
 * before/after every pass — not just post-``CanonicalizeIOOrder``.
 *
 * @return Shared pointer to PipelineLoopValid PropertyVerifier
 */
PropertyVerifierPtr CreatePipelineLoopValidPropertyVerifier();

/**
 * @brief Factory function for creating ManualDepsOnSubmitOnly property verifier
 *
 * Verifies that no plain cross-function Call (GlobalVar callee) carries
 * ``attrs["manual_dep_edges"]`` — manual dependency edges live in the typed
 * ``Submit::deps_`` field. Op calls (``system.task_dummy``) keep the attr as
 * their codegen fanin contract and are exempt. Listed in
 * ``GetStructuralProperties()``, so ``VerificationInstrument`` checks it
 * before/after every pass.
 *
 * @return Shared pointer to ManualDepsOnSubmitOnly PropertyVerifier
 */
PropertyVerifierPtr CreateManualDepsOnSubmitOnlyPropertyVerifier();

/**
 * @brief Factory function for creating PipelineResolved property verifier
 *
 * Verifies the post-canonicalize invariant: no ``ForStmt`` may carry
 * ``kind_ == ForKind::Pipeline``. ``ForKind::Pipeline`` is a transient marker
 * lowered by ``LowerPipelineLoops`` and demoted by ``CanonicalizeIOOrder``;
 * any survivor downstream of CanonicalizeIOOrder indicates a missing demotion.
 *
 * The bidirectional kind ⇔ attr invariant is checked separately by
 * ``PipelineLoopValid`` (a structural property).
 *
 * @return Shared pointer to PipelineResolved PropertyVerifier
 */
PropertyVerifierPtr CreatePipelineResolvedPropertyVerifier();

/**
 * @brief Factory function for creating UnrollResolved property verifier
 *
 * Verifies the post-unroll invariant: no ``ForStmt`` may carry
 * ``kind_ == ForKind::Unroll``. ``ForKind::Unroll`` is a compile-time marker
 * expanded by ``UnrollLoops`` into ``SeqStmts``; any survivor downstream of
 * UnrollLoops indicates the pass failed to expand it.
 *
 * @return Shared pointer to UnrollResolved PropertyVerifier
 */
PropertyVerifierPtr CreateUnrollResolvedPropertyVerifier();

/**
 * @brief Factory function for creating IterArgCarryClassified property verifier
 *
 * Verifies that every ``ForStmt`` with iter_args inside a
 * ``FunctionType::Orchestration`` function carries the per-slot
 * ``iter_arg_rebind_<i>`` attr stamped by ``ClassifyIterArgCarry``, and that a
 * positive ``iter_arg_array_size_<i>`` only accompanies a rebind carry.
 *
 * @return Shared pointer to IterArgCarryClassified PropertyVerifier
 */
PropertyVerifierPtr CreateIterArgCarryClassifiedPropertyVerifier();

/**
 * @brief Factory function for creating CallDirectionsResolved property verifier
 *
 * Verifies that every non-builtin ``Call`` in the program carries a fully
 * populated ``attrs_["arg_directions"]`` vector (accessed via
 * ``Call::GetArgDirections``) that is internally consistent with the callee's
 * ``param_directions_``. Specifically, for each call:
 *   - the ``arg_directions`` attr is present, has size ``args_.size()`` and
 *     is non-empty;
 *   - tensor arguments carry a non-Scalar direction; scalar arguments carry
 *     ``ArgDirection::Scalar``;
 *   - the per-argument ``ArgDirection`` is consistent with the callee's
 *     ``ParamDirection`` (``In`` ↔ ``Input``; ``InOut`` ↔ ``InOut`` /
 *     ``OutputExisting`` after auto-deps rewrite; ``Out`` ↔ ``Output`` /
 *     ``OutputExisting`` / ``InOut`` for WAW promotion).
 *
 * The runtime requirement that ``add_input/add_output`` come before
 * ``add_scalar`` is satisfied by orchestration codegen (``stable_partition``
 * over ``ParamEntry``); the IR Call itself is allowed to interleave tensors
 * and scalars to preserve the user's parameter order.
 *
 * This is the post-condition of the ``DeriveCallDirections`` pass and is
 * automatically run by ``PassPipeline`` whenever that pass is executed.
 *
 * @return Shared pointer to CallDirectionsResolved PropertyVerifier
 */
PropertyVerifierPtr CreateCallDirectionsResolvedPropertyVerifier();

/**
 * @brief Factory function for creating TileTypeCoherence property verifier
 *
 * Asserts the canonical encoding of TileType: a TileView matching the implicit
 * semantics for the tile's (shape, memory_space) is stored as nullopt. The
 * TileType constructor enforces this at construction; the verifier catches
 * passes that mutate the public ``tile_view_`` field directly without going
 * through the constructor.
 *
 * @return Shared pointer to TileTypeCoherence PropertyVerifier
 */
PropertyVerifierPtr CreateTileTypeCoherencePropertyVerifier();

/**
 * @brief Factory function for creating OrchestrationReferencesResolved property verifier
 *
 * Verifies that every non-builtin Call inside an Orchestration function targets
 * a Function that exists in the surrounding Program. Replaces the codegen-side
 * ValidateOrchestrationReferences check that used to throw at codegen time.
 * @return Shared pointer to OrchestrationReferencesResolved PropertyVerifier
 */
PropertyVerifierPtr CreateOrchestrationReferencesResolvedPropertyVerifier();

/**
 * @brief Factory function for creating TensorViewCanonical property verifier
 *
 * Verifies that every TensorType.tensor_view_ in the program satisfies the
 * canonical form per RFC #1300 §2.2:
 *   - layout is one of {ND, DN} (NZ is tile-only and rejected)
 *   - when stride is non-empty: rank matches shape, innermost-stride is 1 at
 *     the layout-specific axis, and outer dims are valid packed/strided
 *   - when stride is empty: this is allowed in the "weak" mode (default —
 *     pre-MaterializeTensorStrides) and rejected in the "strict" mode
 *     (post-MaterializeTensorStrides, P3+)
 *
 * Symbolic strides are accepted under relaxed_symbolic semantics (RFC Open Q2).
 *
 * @param require_materialized When true, reject tensor_view_ with empty stride
 *   (the strict, codegen-entry contract). When false, accept empty stride as
 *   "implicitly packed canonical" — used during early pipeline stages.
 * @return Shared pointer to TensorViewCanonical PropertyVerifier
 */
PropertyVerifierPtr CreateTensorViewCanonicalPropertyVerifier(bool require_materialized = false);

/**
 * @brief Factory function for creating CommDomainScopesMaterialized property verifier
 *
 * Walks each function body for ``CommDomainScopeStmt`` nodes and verifies:
 * (a) every scope's ``slots_`` is non-empty (an empty domain would emit
 * ``window_size=0`` and the runtime would reject it); (b) every slot is
 * non-null; (c) each ``WindowBuffer`` appears as a slot in at most one
 * scope program-wide. ``DistributedCodegen`` relies on this shared_ptr-identity
 * uniqueness to route each Submit's per-arg ``device_ctx`` handle to the
 * correct domain; a duplicate slot would misroute communication traffic at
 * runtime. The ``MaterializeCommDomainScopes`` pass enforces this invariant by
 * construction; the verifier independently checks it.
 *
 * @return Shared pointer to CommDomainScopesMaterialized PropertyVerifier
 */
PropertyVerifierPtr CreateCommDomainScopesMaterializedPropertyVerifier();

/**
 * @brief Factory function for creating AssignTypeSymmetry property verifier
 *
 * Verifies that every ``AssignStmt(var, value)`` satisfies
 * ``structural_equal(var->GetType(), value->GetType())``. ``structural_equal``
 * is the IR's own type-equality contract (the one the roundtrip verifier uses).
 * It compares, per type kind: TileType → dtype, shape, tile_view, memory_space;
 * TensorType → dtype, shape, tensor_view (DistributedTensorType also compares
 * window_buffer); TupleType → every element recursively. It intentionally
 * excludes ``memref_`` — a MemRef is an allocation detail bound to the Var, not
 * part of the value's structural type — so MemRef asymmetry (legitimate after
 * ``InitMemRef``) is out of scope here and is governed instead by
 * ``HasMemRefs`` / ``AllocatedMemoryAddr``. (``memory_space`` exists only on
 * TileType, not TensorType.)
 *
 * Catches passes that mutate one side of an AssignStmt without keeping the
 * other in sync — e.g. #1262, where ``InferTileMemorySpace`` wrote ``Mem.Acc``
 * onto a Var whose producing ``tile.full`` Call still declared ``Mem.Vec``.
 * Turning silent type corruption into a hard error localises the bug to the
 * pass that caused it instead of a downstream consumer.
 *
 * @return Shared pointer to AssignTypeSymmetry PropertyVerifier
 */
PropertyVerifierPtr CreateAssignTypeSymmetryPropertyVerifier();

/**
 * @brief Create a verifier for IRProperty::ReturnParamsExplicit
 *
 * Checks every InCore/Group/Spmd function: each tensor return value that is a
 * param writeback must reference the param by pointer identity (not an SSA
 * alias of it), and tensor-returning functions must carry a ReturnStmt.
 * Kernel-allocated tensors (untraceable to any param) and scalars are exempt.
 * Keeps orchestration return->arg aliasing a lookup (#1702).
 *
 * @return Shared pointer to ReturnParamsExplicit PropertyVerifier
 */
PropertyVerifierPtr CreateReturnParamsExplicitPropertyVerifier();

/**
 * @brief Factory function for creating HardSyncallOccupancyValid property verifier
 *
 * Verifies that every hard (FFTS) ``system.syncall`` is launched at full core
 * occupancy: the enclosing ``pl.spmd(N)`` fills all physical cores of the
 * barrier's ``core_type``. Runs after ExpandMixedKernel (kernel FunctionType
 * resolved), mapping spmd blocks to physical cores per launch shape:
 *   - Spmd -> standalone AIV kernel : required N == #VECTOR cores (aiv_only)
 *   - Spmd -> standalone AIC kernel : required N == #CUBE cores (aic_only)
 *   - Spmd -> Group (mixed kernel)  : required N == #CUBE core-groups (any barrier)
 * A partial or over-occupancy launch deadlocks on device (507018); the error
 * message points users at ``mode="soft"`` for partial occupancy. Bare
 * hard-syncall kernels with no ``pl.spmd`` launch are not checked.
 *
 * @return Shared pointer to HardSyncallOccupancy PropertyVerifier
 */
PropertyVerifierPtr CreateHardSyncallOccupancyPropertyVerifier();

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_VERIFIER_VERIFIER_H_
