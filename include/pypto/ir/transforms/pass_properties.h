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

#ifndef PYPTO_IR_TRANSFORMS_PASS_PROPERTIES_H_
#define PYPTO_IR_TRANSFORMS_PASS_PROPERTIES_H_

#include "pypto/ir/transforms/ir_property.h"

namespace pypto {
namespace ir {
namespace pass {

/// @brief Central registry of PassProperties for all built-in passes.
///
/// Each constant declares the required, produced, and invalidated IRProperties
/// for one pass.  Using `inline const` (not `constexpr`) because
/// IRPropertySet's initializer_list constructor is not constexpr in C++17.

// -- SSA conversion pass ------------------------------------------------------

inline const PassProperties kConvertToSSAProperties{
    .required = {IRProperty::TypeChecked},
    .produced = {IRProperty::TypeChecked, IRProperty::SSAForm},
    .invalidated = {IRProperty::NormalizedStmtStructure, IRProperty::FlattenedSingleStmt}};

// -- Expression / statement normalisation passes ------------------------------

inline const PassProperties kFlattenCallExprProperties{
    .required = {IRProperty::TypeChecked},
    .produced = {IRProperty::TypeChecked, IRProperty::NoNestedCalls},
    .invalidated = {IRProperty::NormalizedStmtStructure, IRProperty::FlattenedSingleStmt}};

inline const PassProperties kNormalizeStmtStructureProperties{
    .required = {IRProperty::TypeChecked},
    .produced = {IRProperty::TypeChecked, IRProperty::NormalizedStmtStructure},
    .invalidated = {IRProperty::FlattenedSingleStmt}};

inline const PassProperties kFlattenSingleStmtProperties{
    .required = {IRProperty::TypeChecked},
    .produced = {IRProperty::TypeChecked, IRProperty::FlattenedSingleStmt},
    .invalidated = {IRProperty::NormalizedStmtStructure}};

// -- Outlining pass -----------------------------------------------------------

inline const PassProperties kOutlineIncoreScopesProperties{
    .required = {IRProperty::TypeChecked, IRProperty::SSAForm}, .produced = {IRProperty::SplitIncoreOrch}};

// -- Tensor-to-block conversion pass ------------------------------------------

inline const PassProperties kConvertTensorToBlockOpsProperties{.required = {IRProperty::SplitIncoreOrch},
                                                               .produced = {IRProperty::IncoreBlockOps}};

// -- Memory / codegen passes --------------------------------------------------

inline const PassProperties kInitMemRefProperties{
    .required = {IRProperty::TypeChecked, IRProperty::SSAForm, IRProperty::SplitIncoreOrch,
                 IRProperty::IncoreBlockOps},
    .produced = {IRProperty::HasMemRefs, IRProperty::NormalizedStmtStructure},
    .invalidated = {IRProperty::SSAForm}};

inline const PassProperties kBasicMemoryReuseProperties{
    .required = {IRProperty::TypeChecked, IRProperty::SplitIncoreOrch, IRProperty::IncoreBlockOps,
                 IRProperty::HasMemRefs}};

inline const PassProperties kInsertSyncProperties{
    .required = {IRProperty::TypeChecked, IRProperty::SplitIncoreOrch, IRProperty::IncoreBlockOps,
                 IRProperty::HasMemRefs}};

inline const PassProperties kAllocateMemoryAddrProperties{
    .required = {IRProperty::TypeChecked, IRProperty::SplitIncoreOrch, IRProperty::IncoreBlockOps,
                 IRProperty::HasMemRefs}};

}  // namespace pass
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_PASS_PROPERTIES_H_
