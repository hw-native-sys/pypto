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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_DEEP_CLONE_UTILS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_DEEP_CLONE_UTILS_H_

#include <unordered_map>

#include "pypto/ir/expr.h"
#include "pypto/ir/stmt.h"

namespace pypto {
namespace ir {

/// Result of a deep clone operation.
struct DeepCloneResult {
  StmtPtr cloned_body;
  /// Mapping from original Var pointers to their fresh definition-site clones.
  /// Does not include pre-seeded substitutions that map to non-Var expressions (e.g. ConstInt).
  std::unordered_map<const Var*, VarPtr> var_map;
};

/// Deep-clone a statement subtree, creating fresh Var/IterArg/MemRef objects
/// at every definition site (DefField). All references to those variables within
/// the subtree are consistently remapped to the fresh copies.
///
/// @param body    Statement subtree to clone
/// @param var_map Pre-seeded substitutions. Entries map original Var pointers to
///               replacement expressions. Supports both Var→Var (function param
///               cloning) and Var→ConstInt (loop unrolling).
///               Variables NOT in this map get fresh copies at definition sites.
///               Variables in this map are substituted directly at use sites.
/// @param clone_def_vars If true (default), create fresh Var copies at every
///               definition site. If false, keep original Var objects at def
///               sites — useful when the IR uses shared Var pointers for the
///               same source-level variable across multiple assignments.
/// @return DeepCloneResult with cloned body and definition-site var mapping.
DeepCloneResult DeepClone(const StmtPtr& body, const std::unordered_map<const Var*, ExprPtr>& var_map = {},
                          bool clone_def_vars = true);

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_DEEP_CLONE_UTILS_H_
