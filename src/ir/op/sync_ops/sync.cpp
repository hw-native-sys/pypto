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

#include <string>
#include <vector>

#include "pypto/codegen/cce/cce_codegen.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

// Helper to deduce UnknownType (for ops with no return value)
TypePtr DeduceUnknownType(const std::vector<ExprPtr>& args,
                          const std::vector<std::pair<std::string, std::any>>& kwargs) {
  return GetUnknownType();
}

}  // namespace

// ============================================================================
// Registration Function for Sync Operations
// ============================================================================

// Convert PipeType enum to CCE macro string
std::string PipeTypeToString(PipeType pipe) {
  switch (pipe) {
    case PipeType::MTE1:
      return "PIPE_MTE1";
    case PipeType::MTE2:
      return "PIPE_MTE2";
    case PipeType::MTE3:
      return "PIPE_MTE3";
    case PipeType::M:
      return "PIPE_M";
    case PipeType::V:
      return "PIPE_V";
    case PipeType::S:
      return "PIPE_S";
    case PipeType::FIX:
      return "PIPE_FIX";
    case PipeType::ALL:
      return "PIPE_ALL";
    default:
      INTERNAL_CHECK(false) << "Internal error: Invalid PipeType value " << static_cast<int>(pipe);
  }
}

// Helper for set_flag/wait_flag codegen (sync_src and sync_dst)
CCECodegenFunc MakeSyncCodegenCCE(const std::string& isa_name) {
  return [isa_name](const CallPtr& op, codegen::CCECodegen& codegen) -> std::string {
    // Extract kwargs: set_pipe, wait_pipe, event_id
    auto set_pipe = static_cast<PipeType>(op->GetKwarg<int>("set_pipe"));
    auto wait_pipe = static_cast<PipeType>(op->GetKwarg<int>("wait_pipe"));
    int event_id = op->GetKwarg<int>("event_id");

    std::string set_pipe_str = PipeTypeToString(set_pipe);
    std::string wait_pipe_str = PipeTypeToString(wait_pipe);
    std::string event_id_str = "EVENT_ID" + std::to_string(event_id);

    codegen.Emit(isa_name + "(" + set_pipe_str + ", " + wait_pipe_str + ", " + event_id_str + ");");
    return "";  // No return value
  };
}

// Helper for pipe_barrier codegen (bar_v, bar_m, bar_all)
CCECodegenFunc MakeBarrierCodegenCCE(const std::string& pipe_str) {
  return [pipe_str](const CallPtr& op, codegen::CCECodegen& codegen) -> std::string {
    codegen.Emit("pipe_barrier(" + pipe_str + ");");
    return "";  // No return value
  };
}

// Register system.sync_src (Set Flag)
// Attributes: set_pipe, wait_pipe, event_id
REGISTER_OP("system.sync_src")
    .set_description("Send a synchronization signal (Set Flag)")
    .set_op_category("SyncOp")
    .set_pipe(PipeType::S)
    .no_argument()
    .set_attr<int>("set_pipe")
    .set_attr<int>("wait_pipe")
    .set_attr<int>("event_id")
    .f_deduce_type(DeduceUnknownType)
    .f_codegen_cce(MakeSyncCodegenCCE("set_flag"));

// Register system.sync_dst (Wait Flag)
// Attributes: set_pipe, wait_pipe, event_id
REGISTER_OP("system.sync_dst")
    .set_description("Wait for a synchronization signal (Wait Flag)")
    .set_op_category("SyncOp")
    .set_pipe(PipeType::S)
    .no_argument()
    .set_attr<int>("set_pipe")
    .set_attr<int>("wait_pipe")
    .set_attr<int>("event_id")
    .f_deduce_type(DeduceUnknownType)
    .f_codegen_cce(MakeSyncCodegenCCE("wait_flag"));

// Register system.bar_v (Vector Barrier)
// Attributes: None
REGISTER_OP("system.bar_v")
    .set_description("Vector unit barrier")
    .set_op_category("SyncOp")
    .set_pipe(PipeType::S)
    .no_argument()
    .f_deduce_type(DeduceUnknownType)
    .f_codegen_cce(MakeBarrierCodegenCCE("PIPE_V"));

// Register system.bar_m (Matrix Barrier)
// Attributes: None
REGISTER_OP("system.bar_m")
    .set_description("Matrix unit barrier")
    .set_op_category("SyncOp")
    .set_pipe(PipeType::S)
    .no_argument()
    .f_deduce_type(DeduceUnknownType)
    .f_codegen_cce(MakeBarrierCodegenCCE("PIPE_M"));

// Register system.bar_all (Global Barrier)
// Attributes: None
REGISTER_OP("system.bar_all")
    .set_description("Global barrier synchronization")
    .set_op_category("SyncOp")
    .set_pipe(PipeType::S)
    .no_argument()
    .f_deduce_type(DeduceUnknownType)
    .f_codegen_cce(MakeBarrierCodegenCCE("PIPE_ALL"));

}  // namespace ir
}  // namespace pypto
