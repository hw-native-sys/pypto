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

#ifndef PYPTO_SRC_IR_TRANSFORMS_FLATTEN_TILE_ND_TO_2D_REWRITE_INTERNAL_H_
#define PYPTO_SRC_IR_TRANSFORMS_FLATTEN_TILE_ND_TO_2D_REWRITE_INTERNAL_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace flatten_tile_nd_to_2d {
namespace rewrite_internal {

struct FlattenContext {
  std::unordered_map<const Var*, VarPtr> var_map;

  void Insert(const VarPtr& old_var, const VarPtr& new_var) { var_map[old_var.get()] = new_var; }
  void Erase(const VarPtr& var) { var_map.erase(var.get()); }
};

bool IsNdTile(const TileTypePtr& tile_type);
int64_t GetStaticDim(const ExprPtr& expr, const std::string& context);
std::pair<int64_t, int64_t> ComputeMergedShape(const std::vector<ExprPtr>& shape, const std::string& context);
ExprPtr MakeShapeTupleFromInts(const std::vector<int64_t>& dims, const Span& span);
std::vector<ExprPtr> Make2DShapeExprs(int64_t merged, int64_t last, const Span& span);
std::vector<ExprPtr> ComputeMergedValidShape(const std::vector<ExprPtr>& valid, const Span& span);
ExprPtr MakeCanonicalIndexAdd(const ExprPtr& lhs, const ExprPtr& rhs, const Span& span);
bool BatchOperandsWholeFit(const TileTypePtr& lhs_type, const TileTypePtr& rhs_type);
std::vector<int64_t> ToStaticDims(const std::vector<ExprPtr>& shape, const std::string& context);
int64_t MultiplyStaticDims(const std::vector<int64_t>& dims, const std::string& context);
std::vector<int64_t> BuildBatchIndices(int64_t flat_index, const std::vector<int64_t>& batch_shape);
int64_t BuildOperandFlatBatchIndex(const std::vector<int64_t>& operand_batch_shape,
                                   const std::vector<int64_t>& output_batch_shape,
                                   const std::vector<int64_t>& output_batch_indices);
int64_t NormalizeAxisIndex(int64_t axis, size_t ndim, const std::string& context);
bool IsTrailingMatrixAxisSwap(int64_t axis1, int64_t axis2, size_t ndim);
std::vector<TypePtr> FindYieldTypes(const std::vector<StmtPtr>& stmts);

using AssignDefMap = std::unordered_map<const Var*, AssignStmtPtr>;

AssignDefMap BuildAssignDefMap(const std::vector<StmtPtr>& stmts);
bool IsSafePeelableBatchMatmulReshape(const CallPtr& reshape_call);
bool KeepOperandWhole(bool capacity_fits, const CallPtr& base_load);
CallPtr TraceOperandBaseLoad(const ExprPtr& operand_expr, const AssignDefMap& def_map);

struct BatchMatmulResult {
  std::vector<StmtPtr> stmts;
  VarPtr output_var;
  bool fused_store = false;
  VarPtr store_result_var;
  VarPtr store_orig_var;
};

BatchMatmulResult LowerBatchMatmul(const AssignStmtPtr& assign, const CallPtr& call,
                                   const std::vector<StmtPtr>& stmts, size_t stmt_index,
                                   const FlattenContext& ctx, const OpRegistry& op_registry,
                                   const Span& span);

struct BatchMatmulAccResult {
  std::vector<StmtPtr> stmts;
  VarPtr output_var;
};

BatchMatmulAccResult LowerBatchMatmulAcc(const AssignStmtPtr& assign, const CallPtr& call,
                                         const std::vector<StmtPtr>& stmts, const FlattenContext& ctx,
                                         const OpRegistry& op_registry, const Span& span);

struct NdTransposeResult {
  std::vector<StmtPtr> stmts;
  VarPtr output_var;
};

NdTransposeResult LowerNdTranspose(const AssignStmtPtr& assign, const CallPtr& call,
                                   const FlattenContext& ctx, const OpRegistry& op_registry,
                                   const Span& span);

}  // namespace rewrite_internal
}  // namespace flatten_tile_nd_to_2d
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_SRC_IR_TRANSFORMS_FLATTEN_TILE_ND_TO_2D_REWRITE_INTERNAL_H_
