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

#include <any>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

TypePtr DeduceUnknownType(const std::vector<ExprPtr>& args,
                          const std::vector<std::pair<std::string, std::any>>& kwargs) {
  return GetUnknownType();
}

// Shared validation for the (signal: 1-element INT32 tensor, value: INT32 scalar)
// operand contract of tile.comm_notify / tile.comm_wait / tile.comm_test.
void CheckCommSignalArgs(const std::vector<ExprPtr>& args, const char* op_name, const char* value_arg_name) {
  CHECK(args.size() == 2) << op_name << " requires 2 arguments (signal, " << value_arg_name << "), got "
                          << args.size();
  auto sig_ty = As<TensorType>(args[0]->GetType());
  CHECK(sig_ty) << op_name << " signal must be a TensorType, got " << args[0]->GetType()->TypeName();
  CHECK(sig_ty->dtype_ == DataType::INT32)
      << op_name << " signal must be INT32, got " << DataTypeToString(sig_ty->dtype_);
  CHECK(!sig_ty->shape_.empty()) << op_name << " signal must be rank >= 1, got rank-0 tensor";

  // Enforce the single-slot contract: when every shape dim is a ConstInt,
  // their product must equal 1. Dynamic dims are allowed (could be 1 at
  // runtime) but a statically-known non-singleton extent is rejected here so
  // the error surfaces at IR construction instead of late during PTO lowering.
  bool all_static = true;
  int64_t prod = 1;
  for (const auto& d : sig_ty->shape_) {
    auto c = As<ConstInt>(d);
    if (!c) {
      all_static = false;
      continue;
    }
    CHECK(c->value_ >= 1) << op_name << " signal shape dim must be positive, got " << c->value_;
    CHECK(c->value_ == 1) << op_name << " signal must hold exactly one INT32 slot, got static dim "
                          << c->value_;
    prod *= c->value_;
  }
  if (all_static) {
    CHECK(prod == 1) << op_name << " signal must hold exactly one INT32 slot, got element count " << prod;
  }

  auto val_ty = As<ScalarType>(args[1]->GetType());
  CHECK(val_ty) << op_name << " " << value_arg_name << " must be a ScalarType, got "
                << args[1]->GetType()->TypeName();
  CHECK(val_ty->dtype_ == DataType::INT32)
      << op_name << " " << value_arg_name << " must be INT32 scalar, got "
      << DataTypeToString(val_ty->dtype_);
}

TypePtr DeduceTileCommNotifyType(const std::vector<ExprPtr>& args,
                                 const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CheckCommSignalArgs(args, "tile.comm_notify", "value");
  return GetUnknownType();
}

TypePtr DeduceTileCommWaitType(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CheckCommSignalArgs(args, "tile.comm_wait", "cmp_value");
  return GetUnknownType();
}

TypePtr DeduceTileCommTestType(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CheckCommSignalArgs(args, "tile.comm_test", "cmp_value");
  return std::static_pointer_cast<const Type>(std::make_shared<ScalarType>(DataType::BOOL));
}

}  // namespace

// ============================================================================
// Cross-Core Tile Transfer Operations (tpush / tpop)
// ============================================================================

// Push tile data to AIV (from AIC)
REGISTER_OP("tile.tpush_to_aiv")
    .set_description("Push tile data from AIC to AIV via cross-core pipe")
    .set_op_category("CrossCoreOp")
    .set_core_affinity(core_affinity::CoreAffinity::CUBE)
    .set_cross_core_role(core_affinity::CrossCoreRole::TPush)
    .add_argument("tile", "Tile data to transfer")
    .set_attr<int>("split")
    .set_attr<int>("id")
    .no_memory_spec()
    .f_deduce_type(DeduceUnknownType);

// Push tile data to AIC (from AIV)
REGISTER_OP("tile.tpush_to_aic")
    .set_description("Push tile data from AIV to AIC via cross-core pipe")
    .set_op_category("CrossCoreOp")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .set_cross_core_role(core_affinity::CrossCoreRole::TPush)
    .add_argument("tile", "Tile data to transfer")
    .set_attr<int>("split")
    .set_attr<int>("id")
    .no_memory_spec()
    .f_deduce_type(DeduceUnknownType);

// Pop tile data from AIC (into AIV)
REGISTER_OP("tile.tpop_from_aic")
    .set_description("Pop tile data from AIC cross-core pipe into AIV")
    .set_op_category("CrossCoreOp")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .set_cross_core_role(core_affinity::CrossCoreRole::TPop)
    .no_argument()
    .set_attr<int>("split")
    .set_attr<int>("id")
    .no_memory_spec()
    .f_deduce_type(DeduceUnknownType);

// Pop tile data from AIV (into AIC)
REGISTER_OP("tile.tpop_from_aiv")
    .set_description("Pop tile data from AIV cross-core pipe into AIC")
    .set_op_category("CrossCoreOp")
    .set_core_affinity(core_affinity::CoreAffinity::CUBE)
    .set_cross_core_role(core_affinity::CrossCoreRole::TPop)
    .no_argument()
    .set_attr<int>("split")
    .set_attr<int>("id")
    .no_memory_spec()
    .f_deduce_type(DeduceUnknownType);

// ============================================================================
// Cross-Rank Signal Operations (notify / wait)
// ============================================================================

// Notify a remote rank by writing or atomic-adding a value into its signal slot.
// The signal operand is a 1-element INT32 Tensor that views the destination
// rank's GM signal location (typically obtained via import_peer_buffer);
// codegen lowers this to `pto::comm::TNOTIFY` via PTOAS `pto.comm.tnotify`.
REGISTER_OP("tile.comm_notify")
    .set_description(
        "Send a flag notification to a remote rank: write or atomic-add an INT32 value "
        "into the destination signal slot")
    .set_op_category("CrossCoreOp")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .add_argument("signal", "Destination signal tensor (1-element INT32, GM, remote-rank window)")
    .add_argument("value", "INT32 scalar value to write or atomic-add")
    .set_attr<std::string>("op")  // "atomic_add" | "set"
    .no_memory_spec()
    .f_deduce_type(DeduceTileCommNotifyType);

// Block until the local rank's signal slot satisfies `signal <cmp> cmp_value`.
// The signal operand is a 1-element INT32 Tensor in local-rank GM (the slot
// peers atomic-add or set into via tile.comm_notify); codegen lowers this to
// `pto::comm::TWAIT` via PTOAS `pto.comm.twait`.
REGISTER_OP("tile.comm_wait")
    .set_description("Block until a local INT32 signal slot satisfies the given comparison against cmp_value")
    .set_op_category("CrossCoreOp")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .add_argument("signal", "Local signal tensor (1-element INT32, GM) to poll")
    .add_argument("cmp_value", "INT32 scalar comparison value")
    .set_attr<std::string>("cmp")  // "eq" | "ne" | "gt" | "ge" | "lt" | "le"
    .no_memory_spec()
    .f_deduce_type(DeduceTileCommWaitType);

// Non-blocking poll of the local signal slot: returns a BOOL result equal to
// `signal <cmp> cmp_value`. Same operand shape as tile.comm_wait, but does not
// block; codegen lowers this to `pto::comm::TTEST` via PTOAS `pto.comm.ttest`,
// which produces an MLIR `i1`.
REGISTER_OP("tile.comm_test")
    .set_description(
        "Non-blocking poll: return BOOL = (local INT32 signal slot <cmp> cmp_value); does not block")
    .set_op_category("CrossCoreOp")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .add_argument("signal", "Local signal tensor (1-element INT32, GM) to poll")
    .add_argument("cmp_value", "INT32 scalar comparison value")
    .set_attr<std::string>("cmp")  // "eq" | "ne" | "gt" | "ge" | "lt" | "le"
    .no_memory_spec()
    .f_deduce_type(DeduceTileCommTestType);

}  // namespace ir
}  // namespace pypto
