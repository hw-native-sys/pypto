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

#ifndef PYPTO_IR_TRANSFORMS_AUTO_LAYOUT_EXTRACTOR_H_
#define PYPTO_IR_TRANSFORMS_AUTO_LAYOUT_EXTRACTOR_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/auto_layout/model.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace auto_layout {

namespace detail {

/// Strip compiler SSA suffixes from a variable name hint.
std::string CleanName(const std::string& name_hint);

/// Extract static integer dimensions from a shape vector, or nullopt if any dim is dynamic.
std::optional<std::vector<int64_t>> StaticShape(const std::vector<ExprPtr>& exprs);

/// Return the physical GM TensorLayout of a TensorType (defaults to ND when unannotated).
TensorLayout TensorTypeLayout(const TensorTypePtr& tensor_type);

/// Return immediate child statements of compound statements (SeqStmts, IfStmt, ForStmt, WhileStmt,
/// ScopeStmt).
std::vector<StmtPtr> IterChildStmts(const StmtPtr& stmt);

/// Op sets shared between instance extraction and fallback config inference in writeback.
const std::unordered_set<std::string>& RowReductionOps();
const std::unordered_set<std::string>& ColReductionOps();
const std::unordered_set<std::string>& OtherRowExpandOps();
const std::unordered_set<std::string>& ColExpandOps();

/// Tile operands in call order, retaining their original registry argument positions.
std::vector<std::pair<size_t, VarPtr>> TileOperands(const CallPtr& call);

/// Whether the registry identifies a tile operand as scratch, independent of its data shape.
bool IsTileWorkspace(const CallPtr& call, size_t index);

/// Whether this call may run as flat elementwise work over an NZ physical buffer.
bool CanUseVecNzPointwise(const CallPtr& call);

/// Whether the operation depends on its original logical/carrier descriptors.
bool PreservesTileDescriptors(const CallPtr& call);

/// Whether a row sum reduces one valid element per row of a dense compact column.
bool CanCopySingletonRowSum(const CallPtr& call);

/// Geometry shared by flat Vec transpose extraction and materialization.
ValueNode VecTransposeGeometry(const TileTypePtr& tile);
bool AliasesFirstTileOperand(const CallPtr& call);

/// Return true if the configured backend requires `input_index` of `op_name` to be `row_major`.
bool HasRowMajorInputRequirement(const std::string& op_name, size_t input_index);

}  // namespace detail

/// Call-site and static invocation multiplicity metadata gathered from Orchestration functions.
struct OrchCallAnalysis {
  using GmNames = std::unordered_map<const Var*, std::string>;

  struct CallSite {
    std::vector<std::optional<std::string>> arg_names;
    int64_t multiplicity = 1;
    GmNames gm_names;
    std::vector<const Expr*> calls;
  };

  std::unordered_map<std::string, int64_t> call_counts;
  std::unordered_map<std::string, std::vector<std::optional<std::string>>> call_arg_names;
  std::unordered_map<std::string, std::vector<CallSite>> call_sites;
  std::unordered_map<std::string, size_t> call_order;
  std::map<std::pair<std::string, size_t>, int64_t> param_steps;
  std::unordered_set<std::string> reshaped_names;
  std::unordered_set<std::string> nz_ineligible_names;
  std::unordered_set<std::string> dn_ineligible_names;
  std::unordered_set<std::string> internal_buffers;
  std::unordered_set<std::string> external_outputs;
  GmNames gm_names;
};

/// Analyze Orchestration functions in `program` for static call multiplicities and GM aliases.
OrchCallAnalysis AnalyzeOrchestration(const ProgramPtr& program);

/// Return the effective on-chip `LayoutSpec` of a post-`InferTileMemorySpace` `TileType`.
LayoutSpec EffectiveTileLayout(const TileTypePtr& tile_type);

/// Extract a `LayoutProblemInstance` from a post-`InferTileMemorySpace` `Program`.
LayoutProblemInstance ExtractLayoutInstance(const ProgramPtr& program, const LayoutCostModel& cost_model = {},
                                            const std::unordered_set<std::string>& reusable_tensors = {});

}  // namespace auto_layout
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_AUTO_LAYOUT_EXTRACTOR_H_
