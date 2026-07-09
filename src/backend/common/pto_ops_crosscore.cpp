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

/**
 * @file pto_ops_crosscore.cpp
 * @brief PTO codegen registration for cross-core (TPUSH/TPOP/TFREE/pipe) ops.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/backend/common/pto_ops_common.h"
#include "pypto/codegen/codegen_base.h"
#include "pypto/codegen/distributed/comm_layout.h"
#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/codegen/pto/pto_type_utils.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/tile_conversion_utils.h"
#include "pypto/ir/type.h"
#include "src/backend/common/pto_ops_internal.h"

namespace pypto {
namespace backend {

using ir::As;
using ir::AsTensorTypeLike;
using ir::AsVarLike;
using ir::CallPtr;
using ir::ExprPtr;
using ir::ScalarType;
using ir::TensorType;
using ir::Var;

using pto_ops_detail::AsPto;
using pto_ops_detail::CheckSafeIdentifier;
using pto_ops_detail::EmitIndexOperand;
using pto_ops_detail::EmitPartitionViewPTO;
using pto_ops_detail::GetDimStrings;
using pto_ops_detail::GetSizeCodes;
using pto_ops_detail::MakePartitionTensorViewType;

static bool IsSameDimExpr(const ExprPtr& lhs, const ExprPtr& rhs) {
  if (lhs == rhs) {
    return true;
  }
  auto lhs_const = As<ir::ConstInt>(lhs);
  auto rhs_const = As<ir::ConstInt>(rhs);
  return lhs_const && rhs_const && lhs_const->value_ == rhs_const->value_;
}

static std::shared_ptr<const ir::TileType> GetTpushTileType(const ExprPtr& tile_expr) {
  if (auto tile_type = ir::GetTileTypeWithMemRef(tile_expr->GetType())) {
    return tile_type;
  }
  return As<ir::TileType>(tile_expr->GetType());
}

static bool EmitSplitTpushTransportValidShape(const CallPtr& op, codegen::PTOCodegen& codegen,
                                              const std::string& tile_buf, const std::string& tile_type,
                                              int split) {
  // split == 0 normally means no cross-core split: the single consumer reads
  // exactly the producer's (possibly narrowed) valid_shape, so no full-box
  // transport is needed. BUT the 910B no-split dual-AIV dispatch path
  // (function attr `dual_aiv_dispatch`) runs the producer on TWO AIV subblocks
  // that share one FIFO slot while the single cube consumer pops the FULL
  // slot. If the producer narrowed its valid_shape (e.g. set_validshape on a
  // partial attention block), the un-narrowed rows/cols of the slot stay stale
  // and feed garbage into the consumer's matmul. So for that mode we must
  // still transport the full box, exactly as for split==1/2 — this extends
  // PR #1454's fix to the split==0 dual-dispatch case.
  const bool dual_aiv_no_split = (split == 0) && codegen.IsDualAivDispatchFunction();
  if ((split == 0 && !dual_aiv_no_split) || tile_buf.empty() || tile_type.empty()) {
    return false;
  }

  auto source_tile_type = GetTpushTileType(op->args_[0]);
  if (!source_tile_type || source_tile_type->shape_.size() < 2) {
    return false;
  }
  const auto tile_view = ir::tile_view_semantics::GetEffectiveTileView(*source_tile_type);
  if (tile_view.valid_shape.size() < 2) {
    return false;
  }

  // The transport must carry the full box for both subblocks to receive
  // complete data: each subblock reads its half of the slot regardless of the
  // user-declared valid_shape. Narrowing valid_shape on the producer side
  // before tpush would leave the non-split axis under-written and (on LR
  // splits in particular) make even subblock 0 see zeros for the cells the
  // producer skipped. Localization back to the user's logical valid happens
  // on the consumer side via LocalizeValidDimForSplit.
  const auto& shape = source_tile_type->shape_;
  const auto& valid_shape = tile_view.valid_shape;
  ExprPtr transport_row = shape[0];
  ExprPtr transport_col = shape[1];

  // For the 910B no-split dual-AIV path there is NO genuine cross-core row
  // split: subblock 0 runs the full computation while subblock 1 is a
  // pipe-balancing replay whose tile carries valid_shape (0, 0). So here we
  // widen the COLUMNS only -- carrying the producer's fillpad'd cols >=
  // valid_col, which fixes the stale-col feed into the consumer matmul --
  // while PRESERVING the producer's row valid_shape. Widening the rows to the
  // full box would push subblock-1's garbage rows into the shared FIFO slot
  // and race/overwrite subblock-0's real data. Genuine split==1/2 paths keep
  // widening both axes because the row split is real there.
  if (dual_aiv_no_split) {
    transport_row = valid_shape[0];
    // A statically-zero-row producer IS the subblock-1 pipe-balancing replay:
    // it moves no data regardless of the column box, so a col-widening
    // transport is pure overhead AND (on 910B) perturbs the shared-slot
    // dual-AIV merge -- emitting it regressed the cross_core_v2c_nosplit
    // golden. Only the real subblock-0 push (non-zero rows, possibly narrowed
    // by set_validshape) needs the full-column transport.
    if (auto row_const = As<ir::ConstInt>(transport_row); row_const && row_const->value_ == 0) {
      return false;
    }
  }

  if (IsSameDimExpr(transport_row, valid_shape[0]) && IsSameDimExpr(transport_col, valid_shape[1])) {
    return false;
  }

  std::string row = EmitIndexOperand(codegen, transport_row, "tpush transport valid_row");
  std::string col = EmitIndexOperand(codegen, transport_col, "tpush transport valid_col");
  codegen.Emit("pto.set_validshape " + tile_buf + ", " + row + ", " + col + " : " + tile_type);
  return true;
}

static void EmitLogicalTpushValidShapeRestore(const CallPtr& op, codegen::PTOCodegen& codegen,
                                              const std::string& tile_buf, const std::string& tile_type) {
  auto source_tile_type = GetTpushTileType(op->args_[0]);
  INTERNAL_CHECK(source_tile_type) << "Internal error: tpush validShape restore requires a TileType source";
  const auto tile_view = ir::tile_view_semantics::GetEffectiveTileView(*source_tile_type);
  INTERNAL_CHECK(tile_view.valid_shape.size() >= 2)
      << "Internal error: tpush validShape restore requires rank-2 validShape";
  const auto& valid_shape = tile_view.valid_shape;
  std::string row = EmitIndexOperand(codegen, valid_shape[0], "tpush logical valid_row");
  std::string col = EmitIndexOperand(codegen, valid_shape[1], "tpush logical valid_col");
  codegen.Emit("pto.set_validshape " + tile_buf + ", " + row + ", " + col + " : " + tile_type);
}

static std::string FormatFrontendPipeAttrs(const CallPtr& op, int split) {
  std::ostringstream oss;
  oss << "{";
  if (op->HasKwarg("id")) {
    const int id = op->GetKwarg<int>("id", 0);
    CHECK(id >= 0) << "Frontend pipe 'id' attribute must be non-negative, got " << id;
    oss << "id = " << id << ", ";
  }
  oss << "split = " << split << "}";
  return oss.str();
}

static std::string FormatInitializePipeAttrs(const CallPtr& op, int dir_mask, int slot_size) {
  std::ostringstream oss;
  oss << "{";
  if (op->HasKwarg("id")) {
    const int id = op->GetKwarg<int>("id", 0);
    CHECK(id >= 0) << "Frontend initialize_pipe 'id' attribute must be non-negative, got " << id;
    oss << "id = " << id << ", ";
  }
  oss << "dir_mask = " << dir_mask << ", slot_size = " << slot_size;
  if (op->HasKwarg("slot_num")) {
    const int slot_num = op->GetKwarg<int>("slot_num", 0);
    CHECK(slot_num > 0) << "Frontend initialize_pipe 'slot_num' attribute must be positive, got " << slot_num;
    oss << ", slot_num = " << slot_num;
  }
  if (op->HasKwarg("local_slot_num")) {
    const int local_slot_num = op->GetKwarg<int>("local_slot_num", 0);
    CHECK(local_slot_num > 0) << "Frontend initialize_pipe 'local_slot_num' attribute must be positive, got "
                              << local_slot_num;
    oss << ", local_slot_num = " << local_slot_num;
  }
  oss << "}";
  return oss.str();
}

// tile.tpush_to_{aic,aiv}: Push a tile across cores (Cube<->Vector). `target`
// is "aic" or "aiv" and selects both the emitted pto op name and diagnostics.
static std::string MakeTpushCodegenPTO(const char* target, const CallPtr& op,
                                       codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  const std::string op_name = std::string("tpush_to_") + target;

  CHECK(op->args_.size() == 1) << op_name << " requires 1 argument (tile), got " << op->args_.size();
  auto tile = AsVarLike(op->args_[0]);
  INTERNAL_CHECK_SPAN(tile, op->span_) << op_name << " first argument must be a Var or IterArg";

  const int split = op->GetKwarg<int>("split", -1);
  CHECK(split >= 0 && split <= 2) << op_name
                                  << " requires 'split' attribute (0=none, 1=up-down, 2=left-right), got "
                                  << split;

  std::string tile_buf = codegen.GetExprAsCode(op->args_[0]);
  std::string tile_type = codegen.GetExprTypeAnnotation(op->args_[0]);
  const bool restore_valid_shape = EmitSplitTpushTransportValidShape(op, codegen, tile_buf, tile_type, split);

  std::ostringstream oss;
  oss << "pto.tpush_to_" << target << "(" << tile_buf;
  if (!tile_type.empty()) {
    oss << " : " << tile_type;
  }
  oss << ") " << FormatFrontendPipeAttrs(op, split);
  codegen.Emit(oss.str());
  if (restore_valid_shape) {
    EmitLogicalTpushValidShapeRestore(op, codegen, tile_buf, tile_type);
  }

  return "";
}

// tile.tpop_from_{aic,aiv}: Pop a tile another core pushed. `target` is "aic"
// (from Cube into Vector) or "aiv" (from Vector into Cube).
static std::string MakeTpopCodegenPTO(const char* target, const CallPtr& op,
                                      codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  const std::string op_name = std::string("tpop_from_") + target;

  CHECK(op->args_.size() == 0) << op_name << " takes no arguments, got " << op->args_.size();

  const int split = op->GetKwarg<int>("split", 0);
  CHECK(split >= 0 && split <= 2) << op_name
                                  << " requires 'split' attribute (0=none, 1=up-down, 2=left-right), got "
                                  << split;

  std::string result_buf = codegen.GetCurrentResultTarget();
  INTERNAL_CHECK_SPAN(!result_buf.empty(), op->span_) << op_name << " requires assignment target (tile_buf)";
  std::string result_type = codegen.GetCurrentResultTileBufTypeString();
  auto [valid_row, valid_col] = codegen.GetCurrentResultTpopValidShapeOperands();

  std::ostringstream oss;
  oss << result_buf << " = pto.tpop_from_" << target;
  if (!valid_row.empty() || !valid_col.empty()) {
    INTERNAL_CHECK_SPAN(!valid_row.empty() && !valid_col.empty(), op->span_)
        << "Internal error: " << op_name << " dynamic valid_shape requires both valid_row and valid_col";
    oss << "(" << valid_row << ", " << valid_col << ")";
  }
  oss << " " << FormatFrontendPipeAttrs(op, split);
  if (!result_type.empty()) {
    oss << " -> " << result_type;
  }
  codegen.Emit(oss.str());

  return "";
}

/// tfree codegen for system.tfree_to_{aic,aiv}: emits pto.tfree_from_{aic,aiv} {split = N}
static std::string MakeTfreeCodegenPTO(const char* target, const CallPtr& op,
                                       codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  const std::string op_name = std::string("tfree_to_") + target;

  CHECK(op->args_.size() == 1) << op_name << " requires 1 argument (tile from tpop), got "
                               << op->args_.size();
  INTERNAL_CHECK_SPAN(op->HasKwarg("split"), op->span_)
      << "Internal error: system." << op_name
      << " is missing its 'split' kwarg; StampTfreeSplit must "
         "run before codegen to copy it from the originating tpop";
  const int split = op->GetKwarg<int>("split", 0);

  std::ostringstream oss;
  oss << "pto.tfree_from_" << target << " " << FormatFrontendPipeAttrs(op, split);
  codegen.Emit(oss.str());

  return "";
}

static bool ExprIsI32Scalar(const ir::ExprPtr& expr) {
  if (auto st = As<ScalarType>(expr->GetType())) {
    return st->dtype_ == DataType::INT32;
  }
  return false;
}

// Pipe buffer operands are i32 SSA. GetExprAsCode(ConstInt) uses index constants; use i32 here.
static std::string GetPipeBufOperandI32SSA(codegen::PTOCodegen& codegen, const ir::ExprPtr& expr) {
  if (auto c = As<ir::ConstInt>(expr)) {
    return codegen.GetOrEmitConstant(static_cast<int64_t>(static_cast<int32_t>(c->value_)), DataType::INT32);
  }
  INTERNAL_CHECK_SPAN(ExprIsI32Scalar(expr), expr->span_)
      << "Initialize-pipe buffer operand must be INT32 scalar SSA or integral ConstInt placeholder";
  return codegen.GetExprAsCode(expr);
}

// Helper to format initialize_pipe operand list
static void EmitInitializePipeOperands(std::ostringstream& oss, const std::string& gm_ssa,
                                       const std::string& c2v_ssa, const std::string& v2c_ssa) {
  if (!gm_ssa.empty()) {
    oss << "\n      (gm_slot_buffer = " << gm_ssa << " : !pto.ptr<f32>"
        << ", c2v_consumer_buf = " << c2v_ssa << " : i32"
        << ", v2c_consumer_buf = " << v2c_ssa << " : i32)";
  } else {
    oss << " (c2v_consumer_buf = " << c2v_ssa << " : i32"
        << ", v2c_consumer_buf = " << v2c_ssa << " : i32)";
  }
}

// system.{aic,aiv}_initialize_pipe: Initialize a cross-core pipe. `target` is
// "aic" (Cube side) or "aiv" (Vector side); operands are explicit i32 SSAs
// (validated by the MixedKernelExpanded verifier).
static std::string MakeInitializePipeCodegenPTO(const char* target, const CallPtr& op,
                                                codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  const std::string op_name = std::string(target) + "_initialize_pipe";

  CHECK(op->args_.size() == 2) << op_name
                               << " requires 2 arguments (c2v_consumer_buf, v2c_consumer_buf), got "
                               << op->args_.size();
  const int dir_mask = op->GetKwarg<int>("dir_mask", -1);
  const int slot_size = op->GetKwarg<int>("slot_size", -1);
  CHECK(dir_mask >= 0) << op_name << " requires 'dir_mask' attribute";
  CHECK(slot_size > 0) << op_name << " requires 'slot_size' attribute";

  std::string c2v_ssa = GetPipeBufOperandI32SSA(codegen, op->args_[0]);
  std::string v2c_ssa = GetPipeBufOperandI32SSA(codegen, op->args_[1]);
  CHECK(!c2v_ssa.empty() && !v2c_ssa.empty()) << op_name << ": failed to lower buffer operands to SSA names";

  std::ostringstream oss;
  oss << "pto." << target << "_initialize_pipe " << FormatInitializePipeAttrs(op, dir_mask, slot_size);
  const int pipe_id = op->GetKwarg<int>("id", 0);
  EmitInitializePipeOperands(oss, codegen.GetGMSlotBufferSSAForPipe(pipe_id, dir_mask), c2v_ssa, v2c_ssa);
  codegen.Emit(oss.str());

  return "";
}

void RegisterCrossCoreOps(Backend& backend, const std::unordered_set<std::string>& exclude_ops) {
  // Register ops with custom codegen logic
  auto reg = [&](const char* op_name, BackendCodegenFunc fn) {
    if (exclude_ops.count(op_name) > 0) return;
    backend.RegisterOp(op_name).f_codegen(std::move(fn));
  };

  reg("tile.tpush_to_aiv", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTpushCodegenPTO("aiv", op, codegen);
  });
  reg("tile.tpop_from_aiv", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTpopCodegenPTO("aiv", op, codegen);
  });
  reg("tile.tpush_to_aic", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTpushCodegenPTO("aic", op, codegen);
  });
  reg("tile.tpop_from_aic", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTpopCodegenPTO("aic", op, codegen);
  });
  reg("system.tfree_to_aic", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTfreeCodegenPTO("aic", op, codegen);
  });
  reg("system.tfree_to_aiv", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTfreeCodegenPTO("aiv", op, codegen);
  });
  reg("system.aic_initialize_pipe", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeInitializePipeCodegenPTO("aic", op, codegen);
  });
  reg("system.aiv_initialize_pipe", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeInitializePipeCodegenPTO("aiv", op, codegen);
  });

  reg("system.reserve_buffer", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    auto& codegen = AsPto(codegen_base);
    CHECK(op->args_.size() == 0) << "reserve_buffer takes no arguments, got " << op->args_.size();

    const auto name = op->GetKwarg<std::string>("name");
    const int size = op->GetKwarg<int>("size", -1);
    const int base = op->GetKwarg<int>("base", -1);
    // Under memory_planner=PtoAS, AllocateMemoryAddr is skipped and ptoas PlanMemory places the
    // reserved region itself (`auto = true`, base absent — ptoas rejects both being present).
    const bool auto_alloc = !codegen.EmitTileAddr();
    CHECK(!name.empty()) << "reserve_buffer requires 'name' attribute";
    CHECK(size > 0) << "reserve_buffer requires positive 'size' attribute, got " << size;
    INTERNAL_CHECK_SPAN(auto_alloc || base >= 0, op->span_)
        << "reserve_buffer requires AllocateMemoryAddr to resolve 'base' before PTO emission, got " << base;
    CheckSafeIdentifier(name, "reserve_buffer 'name'");

    std::string ssa_name = codegen.GetCurrentResultTarget();
    if (ssa_name.empty()) {
      // EvalStmt context — derive SSA name from buffer name hint
      ssa_name = codegen.NewNamedTemp(name);
    }

    std::string location;
    if (codegen.IsAICFunction()) {
      location = "mat";
    } else if (codegen.IsAIVFunction()) {
      location = "vec";
    } else {
      location = "undefined";
    }

    std::ostringstream oss;
    oss << ssa_name << " = pto.reserve_buffer {name = \"" << name << "\", size = " << size
        << ", location = #pto.address_space<" << location << ">, auto = " << (auto_alloc ? "true" : "false");
    if (!auto_alloc) {
      oss << ", base = " << base;
    }
    oss << "} -> i32";
    codegen.Emit(oss.str());

    return std::string("");
  });

  reg("system.import_peer_buffer", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    auto& codegen = AsPto(codegen_base);
    CHECK(op->args_.size() == 0) << "import_peer_buffer takes no arguments, got " << op->args_.size();

    const auto name = op->GetKwarg<std::string>("name");
    const auto peer_func = op->GetKwarg<std::string>("peer_func");
    CHECK(!name.empty()) << "import_peer_buffer requires 'name' attribute";
    CHECK(!peer_func.empty()) << "import_peer_buffer requires 'peer_func' attribute";
    CheckSafeIdentifier(name, "import_peer_buffer 'name'");
    CheckSafeIdentifier(peer_func, "import_peer_buffer 'peer_func'");

    std::string ssa_name = codegen.GetCurrentResultTarget();
    if (ssa_name.empty()) {
      ssa_name = codegen.NewNamedTemp(name + "_import");
    }

    std::ostringstream oss;
    oss << ssa_name << " = pto.import_reserved_buffer {name = \"" << name << "\", peer_func = @" << peer_func
        << "} -> i32";
    codegen.Emit(oss.str());

    return std::string("");
  });

  reg("system.syncall", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    auto& codegen = AsPto(codegen_base);
    // Cross-core all-participant barrier (pto::SYNCALL). `mode` selects the
    // hard/FFTS form (no operands) or the soft/GM-polling form (operands).
    const auto core_type = op->GetKwarg<std::string>("core_type", "mix");
    const auto mode = op->GetKwarg<std::string>("mode", "hard");
    CHECK(core_type == "aiv_only" || core_type == "aic_only" || core_type == "mix")
        << "system.syncall: core_type must be aiv_only|aic_only|mix, got " << core_type;

    if (mode == "hard") {
      CHECK(op->args_.empty()) << "system.syncall (hard form) takes no arguments, got " << op->args_.size();
      codegen.Emit("pto.syncall() mode = #pto.sync_all_mode<hard>, core_type = #pto.sync_core_type<" +
                   core_type + ">");
      return std::string("");
    }

    CHECK(mode == "soft") << "system.syncall: mode must be hard|soft, got " << mode;
    // Soft form operands (all validated below):
    //   aiv_only: [gm_workspace, ub_scratch, used_cores]
    //   aic_only: [gm_workspace, l1_scratch, used_cores]
    //   mix:      [gm_workspace, ub_scratch, l1_scratch, used_cores]
    // gm_workspace is a shared 1-D GM int32 buffer (used_cores*8 slots, zero-init);
    // the scratch tiles are local int32 staging (UB=Vec on the vector lane, flat
    // L1=Mat on the cube lane); used_cores is an i32 participant count. A mix
    // barrier carries both scratch tiles and is emitted on both lanes (SHARED);
    // pto-isa's soft-mix lowering uses the L1 tile on the cube path and the UB
    // tile on the vector path (the other is dead on each lane).
    const bool is_mix = core_type == "mix";
    const size_t num_scratch = is_mix ? 2 : 1;
    const size_t expected_args = num_scratch + 2;  // gm_workspace + scratch(es) + used_cores
    CHECK(op->args_.size() == expected_args) << "system.syncall (soft " << core_type << ") requires "
                                             << expected_args << " operands, got " << op->args_.size();
    const size_t used_idx = op->args_.size() - 1;

    // gm_workspace: shared 1-D GM int32 tensor -> pto.partition_view over the
    // whole buffer.
    auto gm_var = AsVarLike(op->args_[0]);
    CHECK_SPAN(gm_var, op->span_) << "system.syncall soft: gm_workspace must be a tensor variable";
    auto gm_tt = As<ir::TensorType>(gm_var->GetType());
    CHECK_SPAN(gm_tt && gm_tt->shape_.size() == 1, op->span_)
        << "system.syncall soft: gm_workspace must be a 1-D tensor";
    const std::string dtype_str = codegen.GetTypeString(gm_tt->dtype_);
    // Workspace contract (user-facing): the soft barrier indexes int32 counter
    // slots, so the GM buffer must be INT32 with >= used_cores * 8 elements.
    CHECK_SPAN(dtype_str == "i32", op->span_)
        << "system.syncall soft: gm_workspace must be an INT32 tensor, got " << dtype_str;
    constexpr int64_t kSyncAllSoftSlotInt32 = 8;  // pto::SYNCALL soft: 8 int32 slots per core
    if (auto used_const = As<ir::ConstInt>(op->args_[used_idx])) {
      const int64_t required = used_const->value_ * kSyncAllSoftSlotInt32;
      if (auto gm_dim = As<ir::ConstInt>(gm_tt->shape_[0])) {
        CHECK_SPAN(gm_dim->value_ >= required, op->span_)
            << "system.syncall soft: gm_workspace needs >= used_cores*8 (" << required
            << ") int32 slots, got " << gm_dim->value_;
      }
    }
    // Each scratch tile must be an INT32 staging tile (it mirrors the int32 GM slots).
    for (size_t i = 1; i <= num_scratch; ++i) {
      if (auto scratch_tt = As<ir::TileType>(op->args_[i]->GetType())) {
        CHECK_SPAN(codegen.GetTypeString(scratch_tt->dtype_) == "i32", op->span_)
            << "system.syncall soft: scratch tile must be INT32";
      }
    }
    const std::string gm_view = codegen.GetOrCreateTensorView(gm_var);
    const std::string gm_view_type = codegen.GetTensorViewTypeString(gm_tt.get());
    const std::string partition_type = MakePartitionTensorViewType(GetDimStrings(gm_tt->shape_), dtype_str);
    const std::vector<std::string> offset_codes = {codegen.GetOrEmitConstant(int64_t{0}, DataType::INDEX)};
    const std::vector<std::string> size_codes = GetSizeCodes(gm_tt->shape_, codegen);
    const std::string gm_pview = EmitPartitionViewPTO(gm_var->name_hint_ + "_syncgm", gm_view, gm_view_type,
                                                      partition_type, offset_codes, size_codes, codegen);

    // Assemble the operand + type-annotation lists: gm_pview, scratch(es), used_cores.
    std::vector<std::string> operands = {gm_pview};
    std::vector<std::string> types = {partition_type};
    for (size_t i = 1; i <= num_scratch; ++i) {
      const std::string scratch = codegen.GetExprAsCode(op->args_[i]);
      const std::string scratch_type = codegen.GetExprTypeAnnotation(op->args_[i]);
      CHECK_SPAN(!scratch_type.empty(), op->span_)
          << "system.syncall soft: scratch tile has no tile_buf type annotation";
      operands.push_back(scratch);
      types.push_back(scratch_type);
    }
    operands.push_back(codegen.GetExprAsCode(op->args_[used_idx]));  // used_cores (i32)
    types.push_back("i32");

    std::ostringstream oss;
    oss << "pto.syncall(";
    for (size_t i = 0; i < operands.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << operands[i];
    }
    oss << " : ";
    for (size_t i = 0; i < types.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << types[i];
    }
    oss << ") mode = #pto.sync_all_mode<soft>, core_type = #pto.sync_core_type<" << core_type << ">";
    codegen.Emit(oss.str());
    return std::string("");
  });
}
}  // namespace backend
}  // namespace pypto
