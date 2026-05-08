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

#ifndef PYPTO_IR_TRANSFORMS_IR_PROPERTY_H_
#define PYPTO_IR_TRANSFORMS_IR_PROPERTY_H_

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace pypto {
namespace ir {

/**
 * @brief Enumeration of verifiable IR properties
 *
 * Each value represents a property that the IR may or may not satisfy.
 * Passes can declare which properties they require, produce, and invalidate.
 * Not all passes produce properties — performance optimization passes
 * like MemoryReuse only have requirements but don't produce new
 * verifiable properties. This is by design.
 */
enum class IRProperty : uint64_t {
  SSAForm = 0,              ///< IR is in SSA form
  TypeChecked,              ///< IR has passed type checking
  NoNestedCalls,            ///< No nested call expressions
  NormalizedStmtStructure,  ///< Statement structure normalized
  NoRedundantBlocks,        ///< No single-child or nested SeqStmts
  SplitIncoreOrch,          ///< InCore scopes outlined into separate functions
  HasMemRefs,               ///< MemRef objects initialized on variables
  IncoreTileOps,            ///< InCore functions use tile ops (tile types, load/store)
  AllocatedMemoryAddr,      ///< All MemRefs have valid addresses within buffer limits
  MixedKernelExpanded,      ///< Mixed InCore functions split into AIC+AIV
  ClusterOutlined,          ///< Cluster scopes outlined into Group functions
  TileOps2D,                ///< All tile ops in InCore functions use ≤2D tiles
  TileMemoryInferred,       ///< TileType memory_space_ populated in InCore functions
  BreakContinueValid,       ///< Break/continue only in sequential/while loops
  UseAfterDef,              ///< All variable uses are dominated by a definition
  HierarchyOutlined,        ///< Hierarchy scopes outlined into level/role functions
  StructuredCtrlFlow,       ///< No BreakStmt/ContinueStmt — only structured control flow
  VectorKernelSplit,        ///< AIV functions with split mode have tpop shapes and store offsets adjusted
  OutParamNotShadowed,      ///< Out/InOut params are not reassigned with tensor-creating ops
  NoNestedInCore,           ///< No nested InCore scopes (ScopeStmt inside ScopeStmt)
  InOutUseValid,            ///< No reads of InOut/Out-passed variables after the call (RFC #1026)
  PipelineLoopValid,        ///< Bidirectional invariant: ForStmt.kind_ == Pipeline ⇔ has pipeline_stages attr
  PipelineResolved,         ///< No ForKind::Pipeline survives; produced by CanonicalizeIOOrder
  CallDirectionsResolved,   ///< Every non-builtin Call has explicit attrs['arg_directions']
  TileTypeCoherence,        ///< Every TileType has canonical tile_view (implicit views stored as nullopt)
  InlineFunctionsEliminated,        ///< No FunctionType::Inline functions or Calls to them remain
  OrchestrationReferencesResolved,  ///< Every non-builtin Call in an Orchestration function targets a
                                    ///< Function that exists in the Program
  kCount                            ///< Sentinel (must be last)
};

static_assert(
    static_cast<uint64_t>(IRProperty::kCount) <= 64,
    "IRProperty count exceeds 64, which is the maximum supported by IRPropertySet's uint64_t bitset");

/**
 * @brief Convert an IRProperty to its string name
 */
std::string IRPropertyToString(IRProperty prop);

/**
 * @brief A set of IR properties backed by a uint64_t bitset
 *
 * Efficient O(1) insert/remove/contains operations. Supports up to 64 properties.
 */
class IRPropertySet {
 public:
  IRPropertySet() : bits_(0) {}

  /**
   * @brief Construct from a list of properties
   */
  IRPropertySet(std::initializer_list<IRProperty> props) : bits_(0) {
    for (auto p : props) {
      Insert(p);
    }
  }

  /**
   * @brief Insert a property into the set
   */
  void Insert(IRProperty prop) { bits_ |= Bit(prop); }

  /**
   * @brief Remove a property from the set
   */
  void Remove(IRProperty prop) { bits_ &= ~Bit(prop); }

  /**
   * @brief Check if the set contains a property
   */
  [[nodiscard]] bool Contains(IRProperty prop) const { return (bits_ & Bit(prop)) != 0; }

  /**
   * @brief Check if this set contains all properties in another set
   */
  [[nodiscard]] bool ContainsAll(const IRPropertySet& other) const {
    return (bits_ & other.bits_) == other.bits_;
  }

  /**
   * @brief Return the union of this set and another
   */
  [[nodiscard]] IRPropertySet Union(const IRPropertySet& other) const {
    IRPropertySet result;
    result.bits_ = bits_ | other.bits_;
    return result;
  }

  /**
   * @brief Return the intersection of this set and another
   */
  [[nodiscard]] IRPropertySet Intersection(const IRPropertySet& other) const {
    IRPropertySet result;
    result.bits_ = bits_ & other.bits_;
    return result;
  }

  /**
   * @brief Return this set minus another (set difference)
   */
  [[nodiscard]] IRPropertySet Difference(const IRPropertySet& other) const {
    IRPropertySet result;
    result.bits_ = bits_ & ~other.bits_;
    return result;
  }

  /**
   * @brief Check if the set is empty
   */
  [[nodiscard]] bool Empty() const { return bits_ == 0; }

  /**
   * @brief Convert to a vector of the contained properties
   */
  [[nodiscard]] std::vector<IRProperty> ToVector() const;

  /**
   * @brief Convert to a human-readable string (e.g., "{SSAForm, TypeChecked}")
   */
  [[nodiscard]] std::string ToString() const;

  bool operator==(const IRPropertySet& other) const { return bits_ == other.bits_; }
  bool operator!=(const IRPropertySet& other) const { return bits_ != other.bits_; }

 private:
  uint64_t bits_;

  static uint64_t Bit(IRProperty prop) { return uint64_t{1} << static_cast<uint64_t>(prop); }
};

/**
 * @brief Property declarations for a pass
 *
 * Used with CreateFunctionPass/CreateProgramPass to declare pass requirements.
 */
struct PassProperties {
  IRPropertySet required;     ///< Preconditions: must hold before the pass runs
  IRPropertySet produced;     ///< New properties guaranteed after running
  IRPropertySet invalidated;  ///< Properties this pass breaks
};

/**
 * @brief Controls automatic verification in PassPipeline
 *
 * When VerificationLevel is Basic, PassPipeline::Run() automatically verifies
 * a small set of lightweight properties exactly once each, throwing on errors.
 */
enum class VerificationLevel {
  None,       ///< No automatic verification (fastest)
  Basic,      ///< Verify lightweight properties once per pipeline (default)
  Roundtrip,  ///< Basic + print→parse structural-equality check after every pass
};

/**
 * @brief Get the set of properties automatically verified during compilation
 *
 * Returns {SSAForm, TypeChecked, MixedKernelExpanded, AllocatedMemoryAddr,
 * BreakContinueValid, NoRedundantBlocks} — lightweight checks that catch the
 * most common IR errors.
 */
const IRPropertySet& GetVerifiedProperties();

/**
 * @brief Structural invariants that must hold at all pipeline stages
 *
 * These are verified automatically at pipeline start and never declared
 * in per-pass PassProperties. Returns {TypeChecked, BreakContinueValid,
 * NoRedundantBlocks, UseAfterDef, OutParamNotShadowed, NoNestedInCore,
 * InOutUseValid, PipelineLoopValid}.
 */
const IRPropertySet& GetStructuralProperties();

/**
 * @brief Default property set for explicit verification
 *
 * Returns {SSAForm, TypeChecked, NoNestedCalls, BreakContinueValid,
 * NoRedundantBlocks, UseAfterDef, OutParamNotShadowed, NoNestedInCore} — the properties checked by
 * run_verifier() when no explicit set is given.
 */
const IRPropertySet& GetDefaultVerifyProperties();

/**
 * @brief Get the default verification level from environment
 *
 * Checks the PYPTO_VERIFY_LEVEL environment variable on first call
 * (values: "none", "basic"). Defaults to Basic.
 * Used as the default for PassContext when no explicit level is provided.
 */
VerificationLevel GetDefaultVerificationLevel();

/**
 * @brief Controls when DiagnosticInstrument runs registered checks
 *
 * Checks (warnings + performance hints) declare a phase at registration time;
 * the instrument fires each registered check at its declared phase. The
 * `disabled` parameter on PassContext suppresses individual checks regardless
 * of phase.
 *
 * Phases are independent of severity: a Warning may run at PrePipeline, and a
 * PerfHint may run at PostPipeline — declared per check, not per severity.
 */
enum class DiagnosticPhase {
  None,          ///< Disable the diagnostic channel entirely
  PrePipeline,   ///< Run once before the first pass (default for warnings)
  PostPass,      ///< Run after every pass (debugging)
  PostPipeline,  ///< Run once after the last pass (default for performance hints)
};

/**
 * @brief Get the default diagnostic phase from environment
 *
 * Checks the PYPTO_WARNING_LEVEL environment variable on first call
 * (values: "none", "pre_pipeline", "post_pass", "post_pipeline").
 * Defaults to PrePipeline. The variable name is retained for backward
 * compatibility with shells that have it set.
 */
DiagnosticPhase GetDefaultDiagnosticPhase();

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_IR_PROPERTY_H_
