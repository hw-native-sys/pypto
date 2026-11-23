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
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/acc_push_epilogue.h"
#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

TypePtr DeduceUnknownType(const std::vector<ExprPtr>& args,
                          const std::vector<std::pair<std::string, std::any>>& kwargs) {
  return GetUnknownType();
}

// PTO emits `pto.reserve_buffer` / `pto.import_reserved_buffer` with `-> i32` result.
TypePtr DeduceI32ScalarType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs) {
  (void)args;
  (void)kwargs;
  return std::make_shared<ScalarType>(DataType::INT32);
}

TypePtr DeduceInitializePipeType(const std::vector<ExprPtr>& args,
                                 const std::vector<std::pair<std::string, std::any>>& kwargs) {
  bool structured = false, shorthand = false;
  std::optional<bool> nosplit;
  for (const auto& [key, value] : kwargs) {
    structured |= key == "acc_push_epilogue";
    shorthand |= key == "c2v_dtype" || (key == "pre_relu" && AnyCast<bool>(value, "kwarg key: pre_relu"));
    if (key == "nosplit") nosplit = AnyCast<bool>(value, "kwarg key: nosplit");
  }
  CHECK(!structured || !shorthand) << "acc_push_epilogue cannot be combined with c2v_dtype/pre_relu";
  CHECK(!shorthand || !nosplit.has_value() || *nosplit) << "c2v_dtype requires nosplit=true";
  for (const auto& [key, value] : kwargs) {
    if (key != "acc_push_epilogue") continue;
    const auto e = AnyCast<AccPushEpilogue>(value);
    const auto error = ValidateAccPushEpilogue(e);
    CHECK(error.empty()) << error;
    CHECK(GetIntKwarg(kwargs, "dir_mask", -1) == 1) << "acc_push_epilogue requires dir_mask=1 (C2V only)";
    CHECK(nosplit.value_or(false)) << "acc_push_epilogue requires nosplit=true";
  }
  return GetUnknownType();
}

TypePtr DeduceQuantConfigType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs, bool vector) {
  CHECK(args.size() == 1) << "set_quant requires exactly one operand";
  CHECK(GetIntKwarg(kwargs, "id", 0) >= 0) << "set_quant pipe id must be non-negative";
  if (vector) {
    auto tile = As<TileType>(args[0]->GetType());
    CHECK(tile && tile->memory_space_ == MemorySpace::Scaling &&
          (tile->dtype_ == DataType::UINT64 || tile->dtype_ == DataType::INT64))
        << "set_quant_vector requires packed 64-bit controls in loc=scaling "
           "(MemorySpace.Scaling), not a floating-point scale tile";
    CHECK(tile->shape_.size() == 2) << "set_quant_vector requires shape [1, N]";
    auto rows = As<ConstInt>(tile->shape_[0]);
    auto cols = As<ConstInt>(tile->shape_[1]);
    const auto view = tile_view_semantics::GetEffectiveTileView(*tile);
    CHECK(rows && rows->value_ == 1 && cols && cols->value_ > 0 && cols->value_ % 16 == 0)
        << "set_quant_vector requires shape [1, N] with positive N % 16 == 0 (128-byte alignment)";
    const auto valid_rows = view.valid_shape.size() == 2 ? As<ConstInt>(view.valid_shape[0]) : nullptr;
    const auto valid_cols = view.valid_shape.size() == 2 ? As<ConstInt>(view.valid_shape[1]) : nullptr;
    CHECK(valid_rows && valid_rows->value_ == 1 && valid_cols && valid_cols->value_ == cols->value_)
        << "set_quant_vector requires valid_shape to cover the full [1, N] shape";
    CHECK(view.blayout == TileLayout::row_major && view.slayout == TileLayout::none_box)
        << "set_quant_vector requires unboxed row-major packed controls";
  } else {
    auto scalar = As<ScalarType>(args[0]->GetType());
    CHECK(scalar && scalar->dtype_ == DataType::FP32) << "set_quant_scalar requires an FP32 scalar";
  }
  return GetUnknownType();
}

constexpr int kMaxUserCrossCoreEventId = 13;
constexpr int64_t kMinFFTSWorkspaceElements = 256;

TypePtr DeduceSetFFTSType(const std::vector<ExprPtr>& args,
                          const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(kwargs.empty()) << "system.set_ffts does not accept attributes";
  CHECK(args.size() == 1) << "system.set_ffts requires one workspace tensor, got " << args.size();
  auto tensor_type = AsTensorTypeLike(args[0]->GetType());
  CHECK(tensor_type) << "system.set_ffts workspace must be a Tensor";
  CHECK(tensor_type->dtype_ == DataType::INT64)
      << "system.set_ffts workspace must have INT64 dtype, got " << tensor_type->dtype_.ToString();
  CHECK(tensor_type->shape_.size() == 1)
      << "system.set_ffts workspace must be 1-D, got rank " << tensor_type->shape_.size();
  auto extent = As<ConstInt>(tensor_type->shape_[0]);
  CHECK(extent && extent->value_ >= kMinFFTSWorkspaceElements)
      << "system.set_ffts workspace must have a static length of at least " << kMinFFTSWorkspaceElements
      << " INT64 elements";
  return GetUnknownType();
}

TypePtr DeduceCrossCoreSyncType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs,
                                const std::string& op_name, bool allow_ffts_mode) {
  CHECK(args.size() <= 1) << op_name << " accepts at most one dynamic event-id operand, got " << args.size();

  bool has_static_event_id = false;
  bool has_pipe = false;
  for (const auto& [key, value] : kwargs) {
    if (key == "event_id") {
      const int event_id = AnyCast<int>(value, "kwarg key: event_id");
      CHECK(event_id >= 0 && event_id <= kMaxUserCrossCoreEventId)
          << op_name << " event_id must be in the user-available range [0, " << kMaxUserCrossCoreEventId
          << "], got " << event_id;
      has_static_event_id = true;
    } else if (key == "pipe") {
      const int pipe = AnyCast<int>(value, "kwarg key: pipe");
      CHECK(pipe >= static_cast<int>(PipeType::MTE1) && pipe <= static_cast<int>(PipeType::ALL))
          << op_name << " pipe is invalid: " << pipe;
      has_pipe = true;
    } else if (key == "ffts_mode") {
      CHECK(allow_ffts_mode) << op_name << " does not support ffts_mode";
      const int ffts_mode = AnyCast<int>(value, "kwarg key: ffts_mode");
      CHECK(ffts_mode >= 0 && ffts_mode <= 2) << op_name << " ffts_mode must be in [0, 2], got " << ffts_mode;
    } else if (key == "core_type") {
      const auto core_type = AnyCast<std::string>(value, "kwarg key: core_type");
      CHECK(core_type == "aic" || core_type == "aiv")
          << op_name << " core_type must be 'aic' or 'aiv', got '" << core_type << "'";
    }
  }

  CHECK(has_pipe) << op_name << " requires a pipe attribute";
  const bool has_dynamic_event_id = args.size() == 1;
  CHECK(has_static_event_id != has_dynamic_event_id)
      << op_name << " requires exactly one static event_id attribute or dynamic event-id operand";
  if (has_dynamic_event_id) {
    auto event_type = std::dynamic_pointer_cast<const ScalarType>(args[0]->GetType());
    CHECK(event_type && event_type->dtype_ == DataType::INDEX)
        << op_name << " dynamic event id must have ScalarType(INDEX), got " << args[0]->GetType()->TypeName();
  }
  return GetUnknownType();
}

// Named intra-block events have their own A5 contract; FFTS keeps 0..13/INDEX.
TypePtr DeduceIntraBlockSyncType(const std::vector<ExprPtr>& args,
                                 const std::vector<std::pair<std::string, std::any>>& kwargs,
                                 const std::string& op_name) {
  CHECK(args.size() <= 1) << op_name << " accepts at most one dynamic event-id operand, got " << args.size();
  bool has_static_event_id = false;
  bool has_pipe = false;
  for (const auto& [key, value] : kwargs) {
    if (key == "event_id") {
      const int event_id = AnyCast<int>(value, "kwarg key: event_id");
      CHECK(event_id >= 0 && event_id <= 31)
          << op_name << " static event_id must be in [0, 31] on A5, got " << event_id;
      has_static_event_id = true;
    } else if (key == "pipe") {
      const int pipe = AnyCast<int>(value, "kwarg key: pipe");
      CHECK(pipe == PipeType::FIX || pipe == PipeType::MTE1 || pipe == PipeType::MTE2 ||
            pipe == PipeType::MTE3 || pipe == PipeType::V || pipe == PipeType::S)
          << op_name << " pipe must be FIX, MTE1, MTE2, MTE3, V, or S on A5, got " << pipe;
      has_pipe = true;
    } else if (key == "core_type") {
      const auto core_type = AnyCast<std::string>(value, "kwarg key: core_type");
      CHECK(core_type == "aic" || core_type == "aiv")
          << op_name << " core_type must be 'aic' or 'aiv', got '" << core_type << "'";
    } else {
      CHECK(false) << op_name << " does not support attribute '" << key << "'";
    }
  }
  CHECK(has_pipe) << op_name << " requires a pipe attribute";
  CHECK(has_static_event_id != (args.size() == 1))
      << op_name << " requires exactly one static event_id attribute or dynamic event-id operand";
  if (!args.empty()) {
    auto type = As<ScalarType>(args[0]->GetType());
    CHECK(type && (type->dtype_ == DataType::INT32 || type->dtype_ == DataType::INT64))
        << op_name << " dynamic event id must have ScalarType(INT32) or ScalarType(INT64); got "
        << (type ? type->dtype_.ToString() : args[0]->GetType()->TypeName())
        << "; cast INDEX explicitly before calling";
  }
  return GetUnknownType();
}

}  // namespace

// ============================================================================
// Registration Function for Cross-Core System Operations
// (tile.tpush/tpop are registered in tile_ops/cross_core.cpp)
// ============================================================================

REGISTER_OP("system.set_ffts")
    .set_description("Declare the A3 FFTS setup operand for explicit cross-core synchronization")
    .set_op_category("CrossCoreOp")
    .add_argument("workspace", "One-dimensional INT64 FFTS workspace")
    // Hands the workspace *pointer* to the FFTS unit (`pto.set_ffts %ws :
    // !pto.ptr<i64>`) — it declares where the hardware's scratch lives rather
    // than moving any data itself. The FFTS unit writes that region on its own
    // schedule, which no PyPTO dependency edge models or could usefully order.
    .no_arg_writes()
    .f_deduce_type(DeduceSetFFTSType);

REGISTER_OP("system.sync_set")
    .set_description("Set an explicit cross-core synchronization event")
    .set_op_category("CrossCoreOp")
    .add_argument("event_id_dyn", "Optional dynamic event id (ScalarType(INDEX))")
    .set_attr<int>("pipe")
    .set_attr<int>("event_id")
    .set_attr<int>("ffts_mode")
    .set_attr<std::string>("core_type")
    .f_deduce_type([](const auto& args, const auto& kwargs) {
      return DeduceCrossCoreSyncType(args, kwargs, "system.sync_set", true);
    });

REGISTER_OP("system.sync_wait")
    .set_description("Wait for an explicit cross-core synchronization event")
    .set_op_category("CrossCoreOp")
    .add_argument("event_id_dyn", "Optional dynamic event id (ScalarType(INDEX))")
    .set_attr<int>("pipe")
    .set_attr<int>("event_id")
    .set_attr<std::string>("core_type")
    .f_deduce_type([](const auto& args, const auto& kwargs) {
      return DeduceCrossCoreSyncType(args, kwargs, "system.sync_wait", false);
    });

REGISTER_OP("system.set_intra_block")
    .set_description("Set a named intra-block synchronization event")
    .set_op_category("CrossCoreOp")
    .add_argument("event_id_dyn", "Optional dynamic event id (INT32 or INT64)")
    .set_attr<int>("pipe")
    .set_attr<int>("event_id")
    .set_attr<std::string>("core_type")
    .f_deduce_type([](const auto& args, const auto& kwargs) {
      return DeduceIntraBlockSyncType(args, kwargs, "system.set_intra_block");
    });

REGISTER_OP("system.wait_intra_block")
    .set_description("Wait for a named intra-block synchronization event")
    .set_op_category("CrossCoreOp")
    .add_argument("event_id_dyn", "Optional dynamic event id (INT32 or INT64)")
    .set_attr<int>("pipe")
    .set_attr<int>("event_id")
    .set_attr<std::string>("core_type")
    .f_deduce_type([](const auto& args, const auto& kwargs) {
      return DeduceIntraBlockSyncType(args, kwargs, "system.wait_intra_block");
    });

// Release slot back to AIC producer (called by AIV consumer after tpop_from_aic)
REGISTER_OP("system.tfree_to_aic")
    .set_description("Release ring buffer slot back to AIC producer")
    .set_op_category("CrossCoreOp")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .set_cross_core_role(core_affinity::CrossCoreRole::TFree)
    .add_argument("tile", "Tile buffer obtained from tpop to release")
    .set_attr<int>("split")
    .set_attr<int>("id")
    .f_deduce_type(DeduceUnknownType);

// Release slot back to AIV producer (called by AIC consumer after tpop_from_aiv)
REGISTER_OP("system.tfree_to_aiv")
    .set_description("Release ring buffer slot back to AIV producer")
    .set_op_category("CrossCoreOp")
    .set_core_affinity(core_affinity::CoreAffinity::CUBE)
    .set_cross_core_role(core_affinity::CrossCoreRole::TFree)
    .add_argument("tile", "Tile buffer obtained from tpop to release")
    .set_attr<int>("split")
    .set_attr<int>("id")
    .f_deduce_type(DeduceUnknownType);

// Initialize pipe on AIC side
REGISTER_OP("system.aic_initialize_pipe")
    .set_description("Initialize cross-core pipe on AIC side")
    .set_op_category("CrossCoreOp")
    .set_core_affinity(core_affinity::CoreAffinity::CUBE)
    .set_cross_core_role(core_affinity::CrossCoreRole::InitializePipe)
    .add_argument("c2v_consumer_buf", "C2V consumer buffer base (i32 SSA)")
    .add_argument("v2c_consumer_buf", "V2C consumer buffer base (i32 SSA)")
    .set_attr<int>("dir_mask")
    .set_attr<int>("slot_size")
    .set_attr<int>("slot_num")
    .set_attr<int>("local_slot_num")
    .set_attr<int>("id")
    .set_attr<DataType>("c2v_dtype")
    .set_attr<bool>("pre_relu")
    .set_attr<bool>("ptoas_plan_memory")
    .set_attr<bool>("ptoas_buffer_id_sync")
    .set_attr<bool>("nosplit")
    .set_attr<AccPushEpilogue>("acc_push_epilogue")
    .f_deduce_type(DeduceInitializePipeType);

// Initialize pipe on AIV side
REGISTER_OP("system.aiv_initialize_pipe")
    .set_description("Initialize cross-core pipe on AIV side")
    .set_op_category("CrossCoreOp")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .set_cross_core_role(core_affinity::CrossCoreRole::InitializePipe)
    .add_argument("c2v_consumer_buf", "C2V consumer buffer base (i32 SSA)")
    .add_argument("v2c_consumer_buf", "V2C consumer buffer base (i32 SSA)")
    .set_attr<int>("dir_mask")
    .set_attr<int>("slot_size")
    .set_attr<int>("slot_num")
    .set_attr<int>("local_slot_num")
    .set_attr<int>("id")
    .set_attr<DataType>("c2v_dtype")
    .set_attr<bool>("pre_relu")
    .set_attr<bool>("ptoas_plan_memory")
    .set_attr<bool>("ptoas_buffer_id_sync")
    .set_attr<bool>("nosplit")
    .set_attr<AccPushEpilogue>("acc_push_epilogue")
    .f_deduce_type(DeduceInitializePipeType);

REGISTER_OP("system.set_quant_scalar")
    .set_description("Set the FP32 scalar quantization state of a Fixpipe pipe")
    .set_op_category("CrossCoreOp")
    .set_core_affinity(core_affinity::CoreAffinity::CUBE)
    .add_argument("scale", "FP32 scalar scale")
    .no_arg_writes()
    .set_attr<int>("id")
    .f_deduce_type([](const auto& args, const auto& kwargs) {
      return DeduceQuantConfigType(args, kwargs, false);
    });

REGISTER_OP("system.set_quant_vector")
    .set_description("Set packed 64-bit per-channel Fixpipe quantization controls")
    .set_op_category("CrossCoreOp")
    .set_core_affinity(core_affinity::CoreAffinity::CUBE)
    .add_argument("controls", "Packed 64-bit loc=scaling tile")
    .no_arg_writes()
    .set_attr<int>("id")
    .f_deduce_type([](const auto& args, const auto& kwargs) {
      return DeduceQuantConfigType(args, kwargs, true);
    });

// Reserve a named buffer in a kernel
REGISTER_OP("system.reserve_buffer")
    .set_description("Reserve a named buffer for cross-core communication")
    .set_op_category("CrossCoreOp")
    .no_argument()
    .set_attr<std::string>("name")
    .set_attr<int>("size")
    .set_attr<int>("base")
    .f_deduce_type(DeduceI32ScalarType);

// Import a peer function's buffer
REGISTER_OP("system.import_peer_buffer")
    .set_description("Import a buffer from a peer function in the same group")
    .set_op_category("CrossCoreOp")
    .no_argument()
    .set_attr<std::string>("name")
    .set_attr<std::string>("peer_func")
    .f_deduce_type(DeduceI32ScalarType);

}  // namespace ir
}  // namespace pypto
