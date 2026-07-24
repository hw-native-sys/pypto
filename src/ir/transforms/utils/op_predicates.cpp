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

#include "pypto/ir/transforms/utils/op_predicates.h"

#include <memory>
#include <string>

#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/op_registry.h"

namespace pypto {
namespace ir {
namespace op_predicates {

using core_affinity::CrossCoreRole;

namespace {

bool HasRole(const CallPtr& call, CrossCoreRole expected) {
  if (!call || !call->op_) return false;
  auto op = std::dynamic_pointer_cast<const Op>(call->op_);
  if (!op) return false;
  auto& registry = OpRegistry::GetInstance();
  if (!registry.IsRegistered(op->name_)) return false;
  auto role = registry.GetEntry(op->name_).GetCrossCoreRole();
  return role.has_value() && *role == expected;
}

}  // namespace

bool IsTPop(const CallPtr& call) { return HasRole(call, CrossCoreRole::TPop); }
bool IsTPush(const CallPtr& call) { return HasRole(call, CrossCoreRole::TPush); }
bool IsTFree(const CallPtr& call) { return HasRole(call, CrossCoreRole::TFree); }
bool IsInitializePipe(const CallPtr& call) { return HasRole(call, CrossCoreRole::InitializePipe); }

bool IsBufferAliasingViewOp(const std::string& op_name) {
  auto& registry = OpRegistry::GetInstance();
  if (!registry.IsRegistered(op_name)) return false;
  const auto& entry = registry.GetEntry(op_name);
  // An inherit-input op aliases its input buffer only if it is also in-place
  // safe. tile.transpose inherits input memory but PERMUTES data into a fresh
  // buffer (pto.ttrans, registered not_inplace_safe()), so its output does NOT
  // alias the input — IsInplaceSafe() excludes it without hardcoding op names.
  return entry.OutputMemoryInheritsInput() && entry.IsInplaceSafe();
}

bool IsBuiltinOp(const std::string& op_name) {
  return op_name.rfind("tile.", 0) == 0 || op_name.rfind("tensor.", 0) == 0 ||
         op_name.rfind("system.", 0) == 0 || op_name.rfind("array.", 0) == 0;
}

}  // namespace op_predicates
}  // namespace ir
}  // namespace pypto
