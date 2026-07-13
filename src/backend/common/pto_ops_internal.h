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

#ifndef SRC_BACKEND_COMMON_PTO_OPS_INTERNAL_H_
#define SRC_BACKEND_COMMON_PTO_OPS_INTERNAL_H_

/**
 * @file pto_ops_internal.h
 * @brief Internal (not installed) declarations shared by the per-category PTO op
 *        registration translation units split out of pto_ops_common.cpp.
 *
 * Declares the per-category ``RegisterXxxOps`` entry points (called by
 * ``RegisterPTOOps``) and the ``pto_ops_detail`` shared codegen helper toolkit
 * (defined in pto_ops_shared.cpp).
 */

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/codegen/codegen_base.h"
#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/codegen/pto/pto_type_utils.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace backend {

// Per-category registration entry points. Each mirrors the RegisterPTOOps
// signature and registers one op category to the backend (skipping exclude_ops).
void RegisterElementwiseOps(Backend& backend, const std::unordered_set<std::string>& exclude_ops);
void RegisterMemoryOps(Backend& backend, const std::unordered_set<std::string>& exclude_ops);
void RegisterDataMoveOps(Backend& backend, const std::unordered_set<std::string>& exclude_ops);
void RegisterCrossCoreOps(Backend& backend, const std::unordered_set<std::string>& exclude_ops);
void RegisterDistributedOps(Backend& backend, const std::unordered_set<std::string>& exclude_ops);

// Shared PTO codegen helper toolkit. Defined in pto_ops_shared.cpp; each
// category .cpp pulls in the names it needs via `using pto_ops_detail::X;`
// declarations.
namespace pto_ops_detail {

extern const std::vector<std::string> cmp_modes;
extern const std::vector<std::string> round_modes;
extern const std::vector<std::string> mask_patterns;

codegen::PTOCodegen& AsPto(codegen::CodegenBase& codegen_base);
void CheckArity(const ir::CallPtr& op, std::string_view pto_op_name, size_t arity);
void CheckSafeIdentifier(const std::string& value, const std::string& attr_name);
std::string MakePartitionTensorViewType(const std::vector<std::string>& dims, const std::string& dtype_str);
std::vector<std::string> GetIndexOffsetCodes(const std::vector<ir::ExprPtr>& exprs,
                                             codegen::PTOCodegen& codegen);
std::vector<std::string> GetDimStrings(const std::vector<ir::ExprPtr>& exprs);
std::vector<std::string> GetSizeCodes(const std::vector<ir::ExprPtr>& exprs, codegen::PTOCodegen& codegen);
codegen::TileTypeComponents InferSubviewTileTypeComponents(const ir::TileType& result_tile_type,
                                                           const ir::MakeTuple& shape_tuple,
                                                           const std::string& dtype_str);
std::string EmitPartitionViewPTO(const std::string& name_hint, const std::string& tensor_view,
                                 const std::string& tensor_view_type, const std::string& partition_type,
                                 const std::vector<std::string>& offset_codes,
                                 const std::vector<std::string>& size_codes, codegen::PTOCodegen& codegen);
std::string EmitFlatOffsetSSAFromValues(const std::vector<std::string>& indices,
                                        const std::vector<ir::ExprPtr>& shape, codegen::PTOCodegen& codegen,
                                        const std::string& name_hint);
std::string GenerateInsOutsClause(const ir::CallPtr& op, codegen::PTOCodegen& codegen,
                                  const std::string& config_attr = "");
void EmitInsOuts(codegen::PTOCodegen& codegen, std::string_view pto_op_name,
                 const std::vector<std::pair<std::string, std::string>>& ins);
std::string MaterializeSubviewOperandIfNeeded(const ir::ExprPtr& expr, codegen::PTOCodegen& codegen,
                                              const std::string& name_hint);
void CheckSubviewTileCompat(const ir::TileType& source, const ir::TileType& result,
                            const std::string& op_name);
std::string EmitIndexOperand(codegen::PTOCodegen& codegen, const ir::ExprPtr& expr, std::string_view context);

}  // namespace pto_ops_detail

}  // namespace backend
}  // namespace pypto

#endif  // SRC_BACKEND_COMMON_PTO_OPS_INTERNAL_H_
