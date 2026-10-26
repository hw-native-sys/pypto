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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_STORAGE_REFERENCE_ANALYSIS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_STORAGE_REFERENCE_ANALYSIS_H_

#include <set>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/stmt.h"

namespace pypto::ir {

// A reference selects existing storage rather than owning an allocation. The
// SSA region edges are its persistent representation; this summary is rebuilt
// after rewrites and never substitutes an allocation identity for a selection.
struct StorageReferenceAnalysis {
  std::set<const Var*> references;
  std::set<const Var*> source_bases;
};

// Only region results/carries and their required aliases with proven storage
// origins are accepted. An arbitrary missing MemRef is not a reference.
StorageReferenceAnalysis AnalyzeStorageReferences(const StmtPtr& body,
                                                  const std::vector<VarPtr>& parameters = {});

// Retain the ordinary branch-local producer destination path, but represent
// incoming external values (and nonmovable local windows) by reference SSA.
// Concrete explicit result bindings remain materialization obligations.
StmtPtr NormalizeStorageReferences(const StmtPtr& body, const std::vector<VarPtr>& parameters = {},
                                   const std::set<const Var*>& required_references = {});

// Validate native mutations against independently observable old value
// versions, recurrent carried values, and operation no-alias contracts. The
// may-alias components used here protect observations; they never merge or
// choose allocation groups. The default runs after reference normalization,
// before planning. The optional postplanning check adds concrete Acc If-result
// components after their canonical destinations settle, before boundary fixup.
// Standalone concrete loops retain their byte/slot and overwrite-mask proofs.
void ValidateStorageReferenceWrites(const StmtPtr& body, const std::vector<VarPtr>& parameters = {},
                                    const std::vector<ParamDirection>& directions = {},
                                    bool include_concrete_acc_regions = false);

}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_STORAGE_REFERENCE_ANALYSIS_H_
