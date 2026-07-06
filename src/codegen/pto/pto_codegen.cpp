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

#include "pypto/codegen/pto/pto_codegen.h"

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <ios>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/codegen/distributed/comm_layout.h"
#include "pypto/codegen/pto/pto_type_utils.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

using ir::As;
using ir::AssignStmtPtr;
using ir::BinaryExprPtr;
using ir::CallPtr;
using ir::CommCtxType;
using ir::EvalStmtPtr;
using ir::ExprPtr;
using ir::ForStmtPtr;
using ir::FunctionPtr;
using ir::IfStmtPtr;
using ir::MemRefPtr;
using ir::ProgramPtr;
using ir::ScalarType;
using ir::StmtPtr;
using ir::TensorType;
using ir::TileType;
using ir::VarPtr;
using ir::WhileStmtPtr;
using ir::YieldStmtPtr;

namespace transform_utils = ir::transform_utils;

namespace {

// Full-MemRef-identity key used by PTOAS memory-planner codegen to decide when
// two tile variables denote the *same* buffer (and must share one tile_buf
// handle so the op writes in place). Same base + byte_offset + size = same
// buffer (loop-carried accumulator, in-place op result). A view shares the
// base but differs in offset and/or size, so it gets a distinct key.
std::string MemRefIdentityKey(const ir::MemRefPtr& memref) {
  std::ostringstream key;
  key << static_cast<const void*>(memref->base_.get()) << '|';
  if (auto off = As<ir::ConstInt>(memref->byte_offset_)) {
    key << "off" << off->value_;
  } else {
    key << "off@" << static_cast<const void*>(memref->byte_offset_.get());
  }
  key << "|sz" << memref->size_;
  return key.str();
}

bool IsSameDimExpr(const ExprPtr& lhs, const ExprPtr& rhs) {
  if (lhs == rhs) {
    return true;
  }
  auto lhs_const = As<ir::ConstInt>(lhs);
  auto rhs_const = As<ir::ConstInt>(rhs);
  return lhs_const && rhs_const && lhs_const->value_ == rhs_const->value_;
}

// Extract the (row, col) valid_shape expressions from a TileType's tile_view.
// Returns nullptr for a dimension when it is missing or is a ConstInt (static).
// Non-ConstInt expressions (Var, Call, BinaryOp, ...) flow through as dynamic
// and must be lowered to MLIR via GetExprAsCode at the call site.
std::pair<ExprPtr, ExprPtr> GetTileValidShapeExprs(const std::shared_ptr<const ir::TileType>& tile_type) {
  ExprPtr valid_row_expr;
  ExprPtr valid_col_expr;
  if (!tile_type) {
    return {valid_row_expr, valid_col_expr};
  }

  const auto& optional_tile_view = tile_type->tile_view_;
  if (!optional_tile_view) {
    return {valid_row_expr, valid_col_expr};
  }

  const auto& tile_view = *optional_tile_view;
  if (tile_view.valid_shape.size() >= 1 && tile_view.valid_shape[0] &&
      !As<ir::ConstInt>(tile_view.valid_shape[0])) {
    valid_row_expr = tile_view.valid_shape[0];
  }
  if (tile_view.valid_shape.size() >= 2 && tile_view.valid_shape[1] &&
      !As<ir::ConstInt>(tile_view.valid_shape[1])) {
    valid_col_expr = tile_view.valid_shape[1];
  }
  return {valid_row_expr, valid_col_expr};
}

bool HasDynamicTileValidShape(const std::shared_ptr<const ir::TileType>& tile_type) {
  auto [valid_row_expr, valid_col_expr] = GetTileValidShapeExprs(tile_type);
  return valid_row_expr || valid_col_expr;
}

// Collect Vars referenced by a shape expression in first-seen order (for trailing
// %argN: index in MLIR). Single source of truth: both the in-translation-unit
// caller ``CollectTensorShapeDynVars`` (driving the trailing index params on the
// emitted ``func.func`` signature) and the Python kernel-wrapper codegen
// (recovering a Var from runtime ``tensor->shapes[]`` inside
// ``_generate_arg_unpacking`` in python/pypto/backend/pto_backend.py) go through
// this walker. The Python side reaches it via the public ``CollectVarsFromShapeExpr``
// wrapper exposed through the codegen nanobind binding
// ``collect_vars_from_shape_expr``. There is no Python-side mirror to keep in sync.
//
// Dedup key: raw ``Var*`` is sound here because the IR holds the canonical
// shared_ptr graph (each Var has exactly one address).
//
// Unknown node kinds fail loudly: silently skipping them would recreate the
// very bug this function exists to fix (lost dynamic-dim params in the kernel
// signature) the next time a new Expr subclass is introduced in shapes.
void CollectVarsFromShapeExprImpl(const ExprPtr& expr, std::set<const ir::Var*>& seen,
                                  std::vector<VarPtr>& out) {
  if (!expr) {
    return;
  }
  if (auto var = As<ir::Var>(expr)) {
    if (seen.insert(var.get()).second) {
      out.push_back(var);
    }
    return;
  }
  if (auto binary = As<ir::BinaryExpr>(expr)) {
    CollectVarsFromShapeExprImpl(binary->left_, seen, out);
    CollectVarsFromShapeExprImpl(binary->right_, seen, out);
    return;
  }
  if (auto unary = As<ir::UnaryExpr>(expr)) {
    CollectVarsFromShapeExprImpl(unary->operand_, seen, out);
    return;
  }
  if (auto call = As<ir::Call>(expr)) {
    for (const auto& arg : call->args_) {
      CollectVarsFromShapeExprImpl(arg, seen, out);
    }
    return;
  }
  if (auto tget = As<ir::TupleGetItemExpr>(expr)) {
    CollectVarsFromShapeExprImpl(tget->tuple_, seen, out);
    return;
  }
  if (As<ir::ConstInt>(expr) || As<ir::ConstFloat>(expr) || As<ir::ConstBool>(expr)) {
    return;
  }
  INTERNAL_UNREACHABLE_SPAN(expr->span_) << "CollectVarsFromShapeExpr: unsupported shape expression node";
}

// Collect tensor-shape dyn Vars across a function's tensor params.
// Used both to reserve %argN names upfront (so NewNamedTemp does not collide)
// and to emit the trailing index params on the MLIR func.func signature.
std::vector<VarPtr> CollectTensorShapeDynVars(const FunctionPtr& func) {
  std::vector<VarPtr> dyn_vars;
  std::set<const ir::Var*> seen;
  for (const auto& param : func->params_) {
    if (auto tensor_type = ir::AsTensorTypeLike(param->GetType())) {
      for (const auto& dim : tensor_type->shape_) {
        CollectVarsFromShapeExprImpl(dim, seen, dyn_vars);
      }
    }
  }
  return dyn_vars;
}

int GetGMPipeSlotCount(int dir_mask) {
  const int bidirectional = ir::core_affinity::kDirMaskC2V | ir::core_affinity::kDirMaskV2C;
  if (dir_mask == bidirectional) {
    return 4;
  }
  if (dir_mask == ir::core_affinity::kDirMaskC2V || dir_mask == ir::core_affinity::kDirMaskV2C) {
    return 8;
  }
  return 0;
}

// In-place DPS ops that write into input 0 rather than a freshly-allocated
// result tile:
//   * scatter family (`set_output_reuses_input(0)`): a tscatter into a fresh
//     uninitialized tile would lose the rows it does not write;
//   * `tile.assemble` (`set_output_memory_inherit_input()`): the result is the
//     target with one window overwritten — written in place so the out-of-window
//     data is preserved (and the Acc->Mat pto.tmov stays a clean converting move,
//     not an unsupported Mat->Mat preservation copy).
// The aliasing is gated below on the result and input actually sharing a base
// memref, so it only triggers when memory reuse merged them in place.
bool IsInPlaceInput0DpsOp(const ir::OpPtr& op) {
  return ir::IsOp(op, "tile.scatter") || ir::IsOp(op, "tile.scatter_mask") || ir::IsOp(op, "tile.assemble");
}

bool ShouldAliasScatterResultToInput(const AssignStmtPtr& stmt) {
  auto call = As<ir::Call>(stmt->value_);
  if (!call || !IsInPlaceInput0DpsOp(call->op_) || call->args_.empty()) {
    return false;
  }

  auto result_tile_type = ir::GetTileTypeWithMemRef(stmt->var_->GetType());
  auto input_tile_type = ir::GetTileTypeWithMemRef(call->args_[0]->GetType());
  if (!result_tile_type || !input_tile_type) {
    return false;
  }

  auto result_memref = ir::GetDefinedMemRef(result_tile_type);
  auto input_memref = ir::GetDefinedMemRef(input_tile_type);
  return result_memref && input_memref && result_memref->base_.get() == input_memref->base_.get();
}

// `array.update_element` is SSA-functional in the IR (returns a fresh
// ArrayType value), but on-core arrays lower to a single `pto.declare_local_array`
// that is mutated in place. Aliasing the result Var to the input array's SSA
// name lets the emitted `pto.local_array_set` write the same storage — no copy.
bool ShouldAliasArrayUpdateResultToInput(const AssignStmtPtr& stmt) {
  auto call = As<ir::Call>(stmt->value_);
  return call && ir::IsOp(call, "array.update_element") && !call->args_.empty() &&
         As<ir::ArrayType>(stmt->var_->GetType());
}

const auto& FlattenBody = transform_utils::FlattenToStmts;

// Collects `<var> = TupleGetItemExpr(tuple_var, i)` AssignStmts. IRVisitor
// auto-recurses through all statement kinds (Seq/For/If/While/Scope/Inline/...),
// so this stays correct regardless of where the tuple-returning call is nested.
class TupleConsumerCollector : public ir::IRVisitor {
 public:
  explicit TupleConsumerCollector(const ir::Var* tuple_var, size_t arity)
      : tuple_var_(tuple_var), elements_(arity, nullptr) {}

  [[nodiscard]] const std::vector<ir::VarPtr>& elements() const { return elements_; }

 protected:
  void VisitStmt_(const ir::AssignStmtPtr& op) override {
    if (auto tge = As<ir::TupleGetItemExpr>(op->value_)) {
      if (auto base = As<ir::Var>(tge->tuple_)) {
        if (base.get() == tuple_var_ && tge->index_ >= 0 &&
            static_cast<size_t>(tge->index_) < elements_.size()) {
          elements_[tge->index_] = op->var_;
        }
      }
    }
    ir::IRVisitor::VisitStmt_(op);
  }

 private:
  const ir::Var* tuple_var_;
  std::vector<ir::VarPtr> elements_;
};

}  // namespace

std::vector<VarPtr> CollectVarsFromShapeExpr(const ExprPtr& expr) {
  std::vector<VarPtr> out;
  std::set<const ir::Var*> seen;
  CollectVarsFromShapeExprImpl(expr, seen, out);
  return out;
}

// Visitor to collect all MemRef objects from TileType variables. Also
// piggy-backs SPMD identity detection (tile.get_block_idx / tile.get_block_num
// / tile.get_subblock_idx) on the same body walk so callers do not need a
// separate IR traversal.
class MemRefCollectorVisitor : public ir::IRVisitor {
 public:
  MemRefCollectorVisitor() = default;

  [[nodiscard]] const std::vector<MemRefPtr>& GetMemRefs() const { return memrefs_; }
  [[nodiscard]] const std::map<const ir::Var*, std::shared_ptr<const TileType>>& GetMemRefTileTypes() const {
    return memref_tile_types_;
  }

  /// Returns true when the visited body invokes tile.get_block_idx or
  /// tile.get_block_num. Drives PTOCodegen's decision to append two synthetic
  /// i32 params to the emitted func.func signature; the kernel wrapper
  /// resolves those values from intrinsic.h::get_block_idx(args) /
  /// get_block_num(args) at dispatch time.
  [[nodiscard]] bool UsesSpmdBlockOps() const { return uses_spmd_block_ops_; }

  /// Returns true when the visited body invokes tile.get_subblock_idx. Drives
  /// PTOCodegen's decision to append a synthetic i32 param to the func.func
  /// signature; the kernel wrapper resolves it from
  /// intrinsic.h::get_sub_block_id(args) at dispatch time, rather than reading
  /// the ccec get_subblockid() register.
  [[nodiscard]] bool UsesSubblockOp() const { return uses_subblock_op_; }

  void VisitExpr_(const VarPtr& op) override {
    if (iter_arg_ids_.count(op->UniqueId())) return;
    if (auto tile_type = ir::GetTileTypeWithMemRef(op->GetType())) {
      AddMemRefIfUnique(ir::GetDefinedMemRef(tile_type), tile_type);
    }
  }

  void VisitExpr_(const ir::IterArgPtr& op) override {
    iter_arg_ids_.insert(op->UniqueId());
    ir::IRVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const ir::CallPtr& op) override {
    if (op->op_) {
      if (!uses_spmd_block_ops_ &&
          (ir::IsOp(op, "tile.get_block_idx") || ir::IsOp(op, "tile.get_block_num"))) {
        uses_spmd_block_ops_ = true;
      }
      if (!uses_subblock_op_ && ir::IsOp(op, "tile.get_subblock_idx")) {
        uses_subblock_op_ = true;
      }
    }
    ir::IRVisitor::VisitExpr_(op);
  }

 private:
  std::vector<MemRefPtr> memrefs_;
  std::set<const ir::Var*> seen_bases_;
  std::map<const ir::Var*, std::shared_ptr<const TileType>> memref_tile_types_;
  std::set<uint64_t> iter_arg_ids_;
  bool uses_spmd_block_ops_ = false;
  bool uses_subblock_op_ = false;

  void AddMemRefIfUnique(const MemRefPtr& memref, const std::shared_ptr<const TileType>& tile_type) {
    const ir::Var* base_ptr = memref->base_.get();
    if (seen_bases_.insert(base_ptr).second) {
      memrefs_.push_back(memref);
      memref_tile_types_[base_ptr] = tile_type;
    } else {
      // Merge TileView properties when multiple tiles share the same allocation:
      // - Keep valid_shape from the original tile (e.g., from load)
      // - Take pad from the new tile if it has a non-null pad (e.g., from fillpad)
      // This ensures fillpad's pad_value is used while preserving the original valid_shape
      auto existing = memref_tile_types_[base_ptr];
      if (const auto& tile_view = tile_type->tile_view_;
          tile_view.has_value() && tile_view->pad != ir::PadValue::null) {
        // Merge: keep valid_shape from existing, take pad from new tile
        ir::TileView merged_view;
        if (const auto& existing_view = existing->tile_view_) {
          merged_view = *existing_view;
        }
        merged_view.pad = tile_view->pad;
        auto merged_tile_type = std::make_shared<TileType>(
            existing->shape_, existing->dtype_, existing->memref_, merged_view, existing->memory_space_);
        memref_tile_types_[base_ptr] = merged_tile_type;
      }
    }
  }
};

// ========================================================================
// Constructors
// ========================================================================

PTOCodegen::PTOCodegen() : backend_(backend::GetBackend()) {
  CHECK(backend_ != nullptr && backend_->GetHandler() != nullptr)
      << "PTOCodegen requires a configured backend that exposes a BackendHandler";
}

PTOCodegen::PTOCodegen(const backend::Backend* backend) : backend_(backend) {
  CHECK(backend != nullptr) << "Backend cannot be null";
  CHECK(backend->GetHandler() != nullptr) << "PTOCodegen requires a backend that exposes a BackendHandler";
}

const backend::BackendHandler* PTOCodegen::GetBackendHandler() const { return backend_->GetHandler(); }

// ========================================================================
// Generate entry and GenerateFunction
// ========================================================================

std::string PTOCodegen::Generate(const ProgramPtr& program, bool emit_tile_addr) {
  emit_tile_addr_ = emit_tile_addr;
  stream_.str("");
  stream_.clear();
  fs_.constants_section.str("");
  fs_.constants_section.clear();
  fs_.body_section.str("");
  fs_.body_section.clear();
  gm_slot_buffer_offsets_.clear();
  remote_offset_dtypes_.clear();
  PrepareGMSlotBufferLayout(program);

  const std::string target_arch = backend_->GetHandler()->GetPtoTargetArch();
  stream_ << "module attributes {pto.target_arch = \"" << target_arch << "\"} {\n";

  for (const auto& [gvar, func] : program->functions_) {
    INTERNAL_CHECK_SPAN(ir::IsInCoreType(func->func_type_), func->span_)
        << "PTO backend only supports InCore-variant functions (InCore, AIC, AIV), but function '"
        << func->name_ << "' has type " << ir::FunctionTypeToString(func->func_type_);
    GenerateFunction(func);
  }

  // Emit `@CommRemoteOffset_<dtype>` helpers at module end. Dtypes were
  // registered lazily during op lowering via RegisterCommRemoteOffsetHelper
  // (see EmitCommRemoteView in src/backend/common/pto_ops_common.cpp). MLIR
  // resolves func.call symbols whole-module, so call sites in user functions
  // above can forward-reference these helpers without issue.
  EmitCommRemoteOffsetHelpers();

  stream_ << "}\n";
  return stream_.str();
}

void PTOCodegen::PrepareGMSlotBufferLayout(const ProgramPtr& program) {
  std::map<std::pair<int, int>, int> slot_size_by_pipe;

  std::function<void(const std::vector<StmtPtr>&)> scan_stmts;
  scan_stmts = [&](const std::vector<StmtPtr>& stmts) {
    for (const auto& stmt : stmts) {
      auto call = transform_utils::GetCallFromStmt(stmt);
      if (ir::op_predicates::IsInitializePipe(call)) {
        const int pipe_id = call->GetKwarg<int>("id", 0);
        const int dir_mask = call->GetKwarg<int>("dir_mask", 0);
        const int slot_size = call->GetKwarg<int>("slot_size", 0);
        if (dir_mask > 0 && slot_size > 0) {
          const auto key = std::make_pair(pipe_id, dir_mask);
          auto [it, inserted] = slot_size_by_pipe.emplace(key, slot_size);
          CHECK(inserted || it->second == slot_size)
              << "initialize_pipe for frontend pipe id " << pipe_id << " and dir_mask " << dir_mask
              << " uses inconsistent slot_size values: " << it->second << " and " << slot_size;
        }
      }
      if (auto for_stmt = As<ir::ForStmt>(stmt)) {
        scan_stmts(FlattenBody(for_stmt->body_));
      } else if (auto if_stmt = As<ir::IfStmt>(stmt)) {
        scan_stmts(FlattenBody(if_stmt->then_body_));
        if (if_stmt->else_body_.has_value()) {
          scan_stmts(FlattenBody(if_stmt->else_body_.value()));
        }
      } else if (auto while_stmt = As<ir::WhileStmt>(stmt)) {
        scan_stmts(FlattenBody(while_stmt->body_));
      }
    }
  };

  for (const auto& [gvar, func] : program->functions_) {
    (void)gvar;
    if (func->body_) {
      scan_stmts(FlattenBody(func->body_));
    }
  }

  int64_t byte_offset = 0;
  for (const auto& [key, slot_size] : slot_size_by_pipe) {
    gm_slot_buffer_offsets_[key] = byte_offset;
    const int dir_mask = key.second;
    const int slot_count = GetGMPipeSlotCount(dir_mask);
    CHECK(slot_count > 0) << "initialize_pipe has invalid dir_mask for GM slot buffer: " << dir_mask;
    CHECK(byte_offset <= std::numeric_limits<int64_t>::max() - static_cast<int64_t>(slot_count) * slot_size)
        << "GM slot buffer offset overflow while assigning frontend pipe id " << key.first;
    byte_offset += static_cast<int64_t>(slot_count) * slot_size;
  }
}

// ========================================================================
// Distributed N6: CommRemoteOffset helper emission
// ========================================================================

std::string PTOCodegen::GetCommRemoteOffsetFuncName(const DataType& dtype) {
  return "CommRemoteOffset_" + DataTypeToMLIR(dtype);
}

std::string PTOCodegen::RegisterCommRemoteOffsetHelper(const DataType& dtype) {
  // Sub-byte dtypes (bool / 4-bit) have no whole-byte element stride, so the
  // byte→element division at the bottom of the helper body is ill-defined.
  // Fail at the op call site, where the CHECK message still has caller context.
  const size_t elem_bits = dtype.GetBit();
  CHECK(elem_bits >= 8 && elem_bits % 8 == 0)
      << "Distributed remote ops only support byte-sized element types, got " << dtype.ToString() << " ("
      << elem_bits << " bits)";
  remote_offset_dtypes_.insert(dtype);
  return GetCommRemoteOffsetFuncName(dtype);
}

void PTOCodegen::EmitCommRemoteOffsetHelpers() {
  if (remote_offset_dtypes_.empty()) return;

  namespace cl = codegen::distributed::comm_layout;
  // CommContext field indices, expressed in u64 slots (one ``pto.load_scalar``
  // step = one slot). Pinned via static_assert in
  // include/pypto/codegen/distributed/comm_layout.h so a runtime ABI shift
  // fails PyPTO compilation rather than silently emitting wrong addresses.
  const int64_t k_rank_idx = static_cast<int64_t>(cl::kRankIdOffset / cl::kWindowSlotStride);
  const int64_t k_win_idx = static_cast<int64_t>(cl::kWindowsInOffset / cl::kWindowSlotStride);

  const std::string body_indent = std::string(4, ' ');
  for (const DataType& dtype : remote_offset_dtypes_) {
    // Bit-width validated at registration time (RegisterCommRemoteOffsetHelper),
    // so the division below is always well-defined.
    const int64_t elem_size_bytes = static_cast<int64_t>(dtype.GetBit() / 8);
    const std::string func_name = GetCommRemoteOffsetFuncName(dtype);

    // ``private`` visibility tells PTOAS to emit the helper with C++
    // ``static`` linkage — without it AIV mis-lowers the call and the
    // remote pointer arithmetic silently returns garbage.
    stream_ << "  func.func private @" << func_name << "(%ctx: !pto.ptr<i64>, %peer: index) -> index {\n";
    stream_ << body_indent << "%c_r = arith.constant " << k_rank_idx << " : index\n";
    stream_ << body_indent << "%c_w = arith.constant " << k_win_idx << " : index\n";
    // Read rankId (the low 32 bits of the (rankId, rankNum) 8-byte slot at
    // u64 index k_rank_idx).
    stream_ << body_indent << "%rk_pair = pto.load_scalar %ctx[%c_r] : !pto.ptr<i64> -> i64\n";
    stream_ << body_indent << "%rk_i32 = arith.trunci %rk_pair : i64 to i32\n";
    stream_ << body_indent << "%rk_idx = arith.index_cast %rk_i32 : i32 to index\n";
    // local_base = windowsIn[rankId]
    stream_ << body_indent << "%lb_off = arith.addi %c_w, %rk_idx : index\n";
    stream_ << body_indent << "%lbase = pto.load_scalar %ctx[%lb_off] : !pto.ptr<i64> -> i64\n";
    // peer_base = windowsIn[peer]
    stream_ << body_indent << "%pb_off = arith.addi %c_w, %peer : index\n";
    stream_ << body_indent << "%pbase = pto.load_scalar %ctx[%pb_off] : !pto.ptr<i64> -> i64\n";
    // delta_bytes = peer_base - local_base; converted to an element offset
    // because pto.addptr takes element counts, not bytes.
    stream_ << body_indent << "%dbytes = arith.subi %pbase, %lbase : i64\n";
    stream_ << body_indent << "%esize = arith.constant " << elem_size_bytes << " : i64\n";
    stream_ << body_indent << "%delems_i = arith.divsi %dbytes, %esize : i64\n";
    stream_ << body_indent << "%delems = arith.index_cast %delems_i : i64 to index\n";
    stream_ << body_indent << "return %delems : index\n";
    stream_ << "  }\n";
  }
}

void PTOCodegen::GenerateFunction(const FunctionPtr& func) {
  fs_.Reset();
  fs_.current_function = func;

  // Collect dyn-dim Vars from tensor-parameter shapes once. The same list
  // drives both name reservation (Site A below) and the trailing %argN: index
  // params on the MLIR signature (Site B further down) -- a single source of
  // truth keeps the two in lockstep.
  const std::vector<VarPtr> dyn_vars = CollectTensorShapeDynVars(func);

  // Reserve %argN names upfront so NewNamedTemp never collides with them
  for (size_t i = 0; i < func->params_.size(); i++) {
    fs_.used_ssa_names.insert("arg" + std::to_string(i));
  }
  // Reserve extra %argN slots for the trailing signature args: one
  // CommContext ptr per DistributedTensor param, then the dyn-dim Vars
  // (``dyn_vars`` computed at the top of GenerateFunction). Order here only
  // affects the reserved name count; the actual layout is emitted below.
  size_t dist_tensor_count = 0;
  for (const auto& param : func->params_) {
    if (As<ir::DistributedTensorType>(param->GetType())) {
      ++dist_tensor_count;
    }
  }
  for (size_t i = 0; i < dist_tensor_count + dyn_vars.size(); i++) {
    fs_.used_ssa_names.insert("arg" + std::to_string(func->params_.size() + i));
  }

  BuildVarToMemRefMapping(func);

  // One body walk: collects MemRefs and detects SPMD identity usage. SPMD
  // identity params are injected at codegen time (not at IR level) when the
  // function body invokes tile.get_block_idx / tile.get_block_num /
  // tile.get_subblock_idx; they are appended at the end of the func.func
  // signature with named SSAs, and the ops lower to arith.index_cast of those
  // params (the kernel wrapper supplies the runtime values via
  // intrinsic.h::get_block_idx(args) / get_block_num(args) /
  // get_sub_block_id(args)).
  MemRefCollectorVisitor collector;
  if (func->body_) {
    collector.VisitStmt(func->body_);
  }
  const bool uses_spmd_params = collector.UsesSpmdBlockOps();
  const bool uses_subblock_param = collector.UsesSubblockOp();
  if (uses_spmd_params) {
    fs_.used_ssa_names.insert("__pypto_spmd_block_idx");
    fs_.used_ssa_names.insert("__pypto_spmd_block_num");
  }
  if (uses_subblock_param) {
    fs_.used_ssa_names.insert("__pypto_spmd_subblock_idx");
  }

  // Still collect fs_.memref_to_tile_type for GetTileBufTypeString fallback paths
  fs_.memref_to_tile_type = collector.GetMemRefTileTypes();

  // Per-var SSA binding: each tile variable gets its own SSA name — except in
  // PTOAS memory-planner mode (no addr baked), where variables denoting the
  // *same* buffer (same MemRef base+offset+size, e.g. a loop-carried
  // accumulator coalesced by MemoryReuse) must share one tile_buf handle. In
  // level3 that aliasing was carried by an identical `addr`; without addr, ptoas
  // PlanMemory would otherwise allocate them separately, so we instead emit a
  // single alloc_tile and let the op write in place (`outs(%acc)`).
  for (const auto& [tile_var, tile_type] : fs_.tile_var_allocs) {
    auto memref = ir::GetDefinedMemRef(tile_type);

    std::string ssa_name;
    if (!emit_tile_addr_) {
      const std::string ident = MemRefIdentityKey(memref);
      auto it = fs_.memref_identity_to_mlir.find(ident);
      if (it != fs_.memref_identity_to_mlir.end()) {
        ssa_name = it->second;  // reuse the shared handle (in-place aliasing)
      } else {
        ssa_name = NewNamedTemp(tile_var->name_hint_);
        fs_.memref_identity_to_mlir[ident] = ssa_name;
      }
    } else {
      ssa_name = NewNamedTemp(tile_var->name_hint_);
    }
    BindVarToMlir(tile_var, ssa_name);

    // Pre-populate type so body visitors (e.g., tile.reshape no-op check)
    // can query it before per-variable alloc_tile emission runs.
    std::string type_str = GetTileBufTypeStringFromTileType(tile_type);
    fs_.ssa_to_tile_buf_type[ssa_name] = type_str;

    // Also maintain fs_.memref_to_mlir for compatibility (first var per allocation)
    const ir::Var* base_ptr = memref->base_.get();
    if (fs_.memref_to_mlir.find(base_ptr) == fs_.memref_to_mlir.end()) {
      fs_.memref_to_mlir[base_ptr] = ssa_name;
    }
  }

  // ``dyn_vars`` was computed at the top of GenerateFunction; it carries the
  // trailing %argN: index parameters in first-seen order.

  // Collect ordered DistributedTensor params (in IR-param order). One
  // CommContext pointer arg is appended per DistributedTensor at the end
  // of the func.func signature.
  std::vector<VarPtr> dist_tensor_params;
  for (const auto& param : func->params_) {
    if (As<ir::DistributedTensorType>(param->GetType())) {
      dist_tensor_params.push_back(param);
    }
  }

  stream_ << "  func.func @" << func->name_ << "(";

  // Separate params into tensors and scalars for tensors-first dispatch order.
  // PTOParam dispatches args as [tensors..., scalars...] regardless of function
  // signature order, so the MLIR function signature must match that layout.
  // DistributedTensorType inherits TensorType and uses the same `!pto.ptr<T>`
  // signature slot — fold it into the tensor partition via ir::AsTensorTypeLike.
  std::vector<size_t> tensor_param_indices;
  std::vector<size_t> scalar_param_indices;
  for (size_t i = 0; i < func->params_.size(); i++) {
    if (ir::AsTensorTypeLike(func->params_[i]->GetType())) {
      tensor_param_indices.push_back(i);
    } else {
      scalar_param_indices.push_back(i);
    }
  }

  // Assign %argN names: tensors get indices 0..N_tensors-1, scalars get N_tensors..
  size_t scalar_start_idx = tensor_param_indices.size();
  std::set<const ir::Var*> param_keys;
  for (size_t j = 0; j < tensor_param_indices.size(); j++) {
    const auto& param = func->params_[tensor_param_indices[j]];
    BindVarToMlir(param, "%arg" + std::to_string(j));
    param_keys.insert(GetVarKey(param));
  }
  for (size_t j = 0; j < scalar_param_indices.size(); j++) {
    const auto& param = func->params_[scalar_param_indices[j]];
    BindVarToMlir(param, "%arg" + std::to_string(scalar_start_idx + j));
    param_keys.insert(GetVarKey(param));
  }

  // Emit signature: tensors first, then scalars
  bool first_param = true;
  for (size_t j = 0; j < tensor_param_indices.size(); j++) {
    if (!first_param) stream_ << ", ";
    first_param = false;
    const auto& param = func->params_[tensor_param_indices[j]];
    auto tensor_type = ir::AsTensorTypeLike(param->GetType());
    stream_ << "%arg" << j << ": !pto.ptr<" << GetTypeString(tensor_type->dtype_) << ">";
  }
  for (size_t j = 0; j < scalar_param_indices.size(); j++) {
    if (!first_param) stream_ << ", ";
    first_param = false;
    const auto& param = func->params_[scalar_param_indices[j]];
    stream_ << "%arg" << (scalar_start_idx + j) << ": ";
    if (auto scalar_type = As<ScalarType>(param->GetType())) {
      stream_ << GetTypeString(scalar_type->dtype_);
    } else {
      stream_ << "!pto.ptr<f32>";
    }
  }

  // Append one CommContext pointer arg per DistributedTensor param (in IR-param
  // order). The runtime CommContext is passed as a GM ``uint64_t*`` (see
  // ``runtime/src/common/platform_comm/comm_context.h``); codegen indexes its
  // fields via ``pto.load_scalar`` and the ``comm_layout::k*`` constants. The
  // host-side L2 orch flattens these as ``add_scalar(ctx)`` calls — i.e. they
  // are passed as trailing scalar slots, mirroring dynamic-shape flattening.
  size_t next_arg_idx = func->params_.size();
  for (const auto& dist_param : dist_tensor_params) {
    std::string arg_name = "%arg" + std::to_string(next_arg_idx++);
    stream_ << ", " << arg_name << ": !pto.ptr<i64>";
    fs_.dist_tensor_to_ctx[GetVarKey(dist_param)] = arg_name;
  }

  // Append trailing index parameters for each unique dynamic dimension variable
  for (const auto& dyn_var : dyn_vars) {
    std::string arg_name = "%arg" + std::to_string(next_arg_idx++);
    stream_ << ", " << arg_name << ": index";
    BindVarToMlir(dyn_var, arg_name);
  }

  // Append SPMD identity params after dynamic-dim args, in canonical order
  // (block_idx, block_num, subblock_idx). Each is appended independently based
  // on the ops the function actually uses; the Python kernel wrapper
  // (pto_backend.py) mirrors this exact order when forwarding the call args.
  // Named SSAs make the synthetic origin obvious in the emitted MLIR and let
  // lowerings refer to them via PTOCodegen::GetSpmd{Block,Subblock}*ArgSSA().
  if (uses_spmd_params) {
    fs_.spmd_block_idx_arg = "%__pypto_spmd_block_idx";
    fs_.spmd_block_num_arg = "%__pypto_spmd_block_num";
    stream_ << ", " << fs_.spmd_block_idx_arg << ": i32, " << fs_.spmd_block_num_arg << ": i32";
  }
  if (uses_subblock_param) {
    fs_.spmd_subblock_idx_arg = "%__pypto_spmd_subblock_idx";
    stream_ << ", " << fs_.spmd_subblock_idx_arg << ": i32";
  }

  stream_ << ")";
  switch (func->func_type_) {
    case ir::FunctionType::AIC:
      stream_ << " attributes {pto.kernel_kind = #pto.kernel_kind<cube>}";
      break;
    case ir::FunctionType::AIV:
      stream_ << " attributes {pto.kernel_kind = #pto.kernel_kind<vector>}";
      break;
    default:
      // Other function types like InCore are not expected here and have no kernel_kind.
      break;
  }
  stream_ << " {\n";
  indent_level_++;
  fs_.constants_indent = GetIndent();

  // Pre-emit alloc_tile address constants now that indent_level_ is set.
  // For addr constants specifically, codegen preserves the IR ConstInt
  // dtype 1:1 (other operands like valid_row/valid_col adapt to the
  // consumer's type via cast_to_index — see ComputeAllocTileFields).
  if (emit_tile_addr_) {
    for (const auto& [tile_var, tile_type] : fs_.tile_var_allocs) {
      auto memref = ir::GetDefinedMemRef(tile_type);
      if (auto const_offset = memref ? As<ir::ConstInt>(memref->byte_offset_) : nullptr) {
        GetOrEmitConstant(const_offset->value_, const_offset->dtype());
      }
    }
  }

  // Parameters are already bound; non-param tile vars are bound above in per-var SSA binding

  for (const auto& var : func->params_) {
    if (auto tensor_type = ir::AsTensorTypeLike(var->GetType())) {
      // Skip tensor view for GM slot buffer workspace parameter (raw pointer, no view needed)
      if (var->name_hint_ == "__gm_pipe_buffer") {
        RecordGMSlotBufferSSA(GetVarName(var), tensor_type->dtype_);
        continue;
      }
      std::string tensor_view = NewNamedTemp(var->name_hint_ + "_view");
      BindTensorView(var, tensor_view);
      // Remember the base pointer so mid-body pl.read/pl.write resolve to !pto.ptr
      // even after a slice-assign rebinds the var to its tensor_view.
      RegisterBasePtr(var, GetVarName(var));

      for (const auto& j : tensor_type->shape_) {
        if (As<ir::ConstInt>(j)) {
          GetOrEmitConstant(GetConstIntValue(j), DataType::INDEX);
        }
      }
      // Pre-emit stride constants: use explicit tensor_view_.stride if available,
      // otherwise fall back to shape-based stride computation.
      bool has_explicit_stride =
          tensor_type->tensor_view_.has_value() && !tensor_type->tensor_view_->stride.empty();
      if (has_explicit_stride) {
        for (const auto& s : tensor_type->tensor_view_->stride) {
          if (As<ir::ConstInt>(s)) {
            GetOrEmitConstant(GetConstIntValue(s), DataType::INDEX);
          }
        }
      } else if (tensor_type->shape_.size() == 2) {
        if (As<ir::ConstInt>(tensor_type->shape_[1])) {
          GetOrEmitConstant(GetConstIntValue(tensor_type->shape_[1]), DataType::INDEX);
        }
        GetOrEmitConstant(static_cast<int64_t>(1), DataType::INDEX);
      } else {
        // 1-D and N-D (N>2): pre-emit constant 1 (innermost stride). For N>2,
        // other strides are computed dynamically via arith.muli in
        // EmitMakeTensorViews to support dynamic dims.
        GetOrEmitConstant(static_cast<int64_t>(1), DataType::INDEX);
      }
    }
  }

  auto saved_stream = std::move(stream_);
  stream_ = std::move(fs_.body_section);

  if (func->body_) {
    VisitStmt(func->body_);
  }

  std::string body_content = stream_.str();

  // Render the prologue before flushing constants so constants unique to a
  // shape/stride expression (e.g. the 2 in M * 2) are declared before use.
  stream_.str("");
  stream_.clear();
  EmitMakeTensorViews(func);
  EmitExtraAllocTiles();
  std::string prologue_content = stream_.str();

  stream_ = std::move(saved_stream);

  stream_ << fs_.constants_section.str();
  stream_ << prologue_content;
  stream_ << body_content;
  stream_ << GetIndent() << "return\n";

  indent_level_--;
  stream_ << "  }\n";
}

void PTOCodegen::BuildVarToMemRefMapping(const FunctionPtr& func) {
  class VarMemRefMapper : public ir::IRVisitor {
   public:
    std::map<const ir::Var*, const ir::Var*>& var_to_memref;    ///< tile var → base_ Ptr
    std::map<const ir::Var*, std::string>& memref_to_var_name;  ///< base_ Ptr → var name
    std::vector<std::pair<VarPtr, std::shared_ptr<const TileType>>>& tile_var_allocs;

    VarMemRefMapper(std::map<const ir::Var*, const ir::Var*>& mapping,
                    std::map<const ir::Var*, std::string>& reverse_mapping,
                    std::vector<std::pair<VarPtr, std::shared_ptr<const TileType>>>& allocs)
        : var_to_memref(mapping), memref_to_var_name(reverse_mapping), tile_var_allocs(allocs) {}

    void VisitStmt_(const AssignStmtPtr& op) override {
      if (auto tile_type = ir::GetTileTypeWithMemRef(op->var_->GetType())) {
        const auto memref = ir::GetDefinedMemRef(tile_type);
        const ir::Var* base_ptr = memref->base_.get();
        var_to_memref[op->var_.get()] = base_ptr;
        if (memref_to_var_name.find(base_ptr) == memref_to_var_name.end()) {
          memref_to_var_name[base_ptr] = op->var_->name_hint_;
        }
        tile_var_allocs.emplace_back(op->var_, tile_type);
      }
      ir::IRVisitor::VisitStmt_(op);
    }
  };

  VarMemRefMapper mapper(fs_.var_to_memref, fs_.memref_to_var_name, fs_.tile_var_allocs);
  if (func->body_) {
    mapper.VisitStmt(func->body_);
  }
}

void PTOCodegen::EmitMakeTensorViews(const FunctionPtr& func) {
  // RFC #1300 P7 (canonical codegen).
  //
  // Emit ``pto.make_tensor_view`` directly from the IR's canonical
  // ``(shape, stride, layout)`` triple. There are no implicit swaps or
  // post-emit dn_swap path here — every layout-aware transform (RFC §3.3
  // canonical promotion, ``MaterializeTensorStrides``) has already run by the
  // time codegen executes, so the IR's TensorView fields can be transcribed
  // verbatim.
  //
  // The one exception is the ``[M, 1]`` column-vector special case: PTOAS
  // *infers* DN for shape ``[M, 1]`` with degenerate strides regardless of
  // the IR-declared layout, so the codegen forces DN + ``[1, M]`` strides
  // here to match what PTOAS expects.
  for (const auto& param : func->params_) {
    auto tensor_type = ir::AsTensorTypeLike(param->GetType());
    if (!tensor_type) continue;
    if (param->name_hint_ == "__gm_pipe_buffer") continue;  // GM slot buffer is a raw pointer

    std::string tensor_view = fs_.tensor_to_view.at(GetVarKey(param));
    const size_t rank = tensor_type->shape_.size();

    // ``[..., M, 1]`` column-vector legacy path: PTOAS infers DN for any
    // shape whose innermost dim is constant 1, so the codegen forces DN to
    // match what ``tile.load`` produces (memory.cpp DeduceTileLoadType emits
    // a ColMajor BLayout tile whenever the load shape ends with a constant 1
    // — see test_tensor_expand_clone[broadcast_dim=2] where input
    // ``[B, N, 1]`` is loaded into a ColMajor tile and PTOAS TLoad enforces
    // ``tile.BLayout == tensor.Layout``).
    bool is_column_vector = false;
    if (rank >= 2) {
      auto last_dim = As<ir::ConstInt>(tensor_type->shape_.back());
      if (last_dim && last_dim->value_ == 1) {
        is_column_vector = true;
      }
    }

    ir::TensorLayout layout = ir::TensorLayout::ND;
    if (tensor_type->tensor_view_.has_value()) {
      layout = tensor_type->tensor_view_->layout;
    }
    if (is_column_vector) layout = ir::TensorLayout::DN;

    // Materialize one shape dimension as an MLIR SSA value.
    auto get_shape_dim_mlir = [&](size_t dim_idx) -> std::string {
      const auto& dim_expr = tensor_type->shape_[dim_idx];
      if (auto const_int = As<ir::ConstInt>(dim_expr)) {
        return GetOrEmitConstant(const_int->value_, DataType::INDEX);
      }
      return EmitCastToIndex(dim_expr, GetExprAsCode(dim_expr));
    };
    // Materialize a stride ExprPtr as an MLIR SSA value.
    auto get_stride_mlir = [&](const ir::ExprPtr& stride_expr) -> std::string {
      if (auto const_int = As<ir::ConstInt>(stride_expr)) {
        return GetOrEmitConstant(const_int->value_, DataType::INDEX);
      }
      return EmitCastToIndex(stride_expr, GetExprAsCode(stride_expr));
    };
    // Precompute shape dim SSA names. Dynamic shape exprs may need cast SSA
    // ops (``EmitCastToIndex``) emitted before the ``pto.make_tensor_view``
    // line — materialize them all up-front so the main statement is a single
    // contiguous line.
    std::vector<std::string> shape_dim_names(rank);
    for (size_t j = 0; j < rank; ++j) {
      shape_dim_names[j] = get_shape_dim_mlir(j);
    }

    // Emit one stride multiply ``lhs * shape_dim_names[dim_idx]`` and return
    // the resulting SSA, used for fallback stride derivation when
    // ``tensor_view_->stride`` is empty.
    auto emit_stride_mul = [&](const std::string& lhs, size_t dim_idx, size_t stride_slot) -> std::string {
      std::string mul_name = NewNamedTemp(param->name_hint_ + "_s" + std::to_string(stride_slot));
      stream_ << GetIndent() << mul_name << " = arith.muli " << lhs << ", " << shape_dim_names[dim_idx]
              << " : index\n";
      return mul_name;
    };

    // Build the stride SSA names. Prefer explicit ``tensor_view_->stride``;
    // fall back to canonical derivation per ``layout`` when absent
    // (``MaterializeTensorStrides`` should normally have populated it by now,
    // but the codegen tolerates absent strides for any path that constructs
    // IR ad-hoc and skips the pipeline).
    std::vector<std::string> stride_names(rank);
    bool has_explicit_stride =
        tensor_type->tensor_view_.has_value() && !tensor_type->tensor_view_->stride.empty();
    if (has_explicit_stride) {
      const auto& strides = tensor_type->tensor_view_->stride;
      CHECK(strides.size() == rank) << "EmitMakeTensorViews: explicit stride rank " << strides.size()
                                    << " does not match tensor shape rank " << rank;
      for (size_t j = 0; j < rank; ++j) {
        stride_names[j] = get_stride_mlir(strides[j]);
      }
    } else if (is_column_vector) {
      // Forced-DN ``[..., M, 1]`` legacy stride pattern (PTOAS column-vector
      // convention): trailing pair degenerates to ``stride[rank-2]=1`` and
      // ``stride[rank-1]=shape[rank-1]=1``; outer dims walk row-major over the
      // ``M`` extent (``stride[rank-3]=shape[rank-2]``, ``stride[k-1]=stride[k]*shape[k]``).
      // For rank 2 this collapses to the legacy ``[1, shape[0]]``.
      stride_names[rank - 2] = GetOrEmitConstant(static_cast<int64_t>(1), DataType::INDEX);
      if (rank == 2) {
        stride_names[rank - 1] = shape_dim_names[0];
      } else {
        // rank >= 3: stride[rank-1] = shape[rank-1] (= 1), stride[rank-3] = shape[rank-2].
        stride_names[rank - 1] = shape_dim_names[rank - 1];
        stride_names[rank - 3] = shape_dim_names[rank - 2];
        for (int j = static_cast<int>(rank) - 4; j >= 0; --j) {
          size_t dim = static_cast<size_t>(j);
          stride_names[dim] = emit_stride_mul(stride_names[dim + 1], dim + 1, dim);
        }
      }
    } else if (layout == ir::TensorLayout::DN) {
      CHECK(rank >= 2) << "EmitMakeTensorViews: DN layout requires rank >= 2, got " << rank;
      // RFC §2.3 canonical DN: stride[-2]=1, stride[-1]=shape[-2], outer
      // strides walk row-major over the DN-block volume. Use direct shape
      // references for the trailing pair so 2D DN avoids a spurious
      // ``arith.muli %c1, shape`` step.
      stride_names[rank - 2] = GetOrEmitConstant(static_cast<int64_t>(1), DataType::INDEX);
      stride_names[rank - 1] = shape_dim_names[rank - 2];
      if (rank >= 3) {
        // stride[n-3] = shape[n-2] * shape[n-1] (one full DN-block volume).
        stride_names[rank - 3] = emit_stride_mul(shape_dim_names[rank - 2], rank - 1, rank - 3);
        for (int j = static_cast<int>(rank) - 4; j >= 0; --j) {
          size_t dim = static_cast<size_t>(j);
          stride_names[dim] = emit_stride_mul(stride_names[dim + 1], dim + 1, dim);
        }
      }
    } else {
      // Canonical ND (row-major): stride[-1]=1, stride[k]=stride[k+1]*shape[k+1].
      // For rank 2 specifically, stride[0] = shape[1] directly (avoids a
      // spurious ``arith.muli %c1, shape[1]`` step).
      stride_names[rank - 1] = GetOrEmitConstant(static_cast<int64_t>(1), DataType::INDEX);
      if (rank >= 2) {
        stride_names[rank - 2] = shape_dim_names[rank - 1];
        for (int j = static_cast<int>(rank) - 3; j >= 0; --j) {
          size_t dim = static_cast<size_t>(j);
          stride_names[dim] = emit_stride_mul(stride_names[dim + 1], dim + 1, dim);
        }
      }
    }

    stream_ << GetIndent() << tensor_view << " = pto.make_tensor_view ";
    stream_ << GetVarName(param);

    // Emit shape (verbatim from IR — canonical).
    stream_ << ", shape = [";
    for (size_t j = 0; j < rank; ++j) {
      if (j > 0) stream_ << ", ";
      stream_ << shape_dim_names[j];
    }
    stream_ << "],";

    // Emit strides.
    stream_ << " strides = [";
    for (size_t j = 0; j < rank; ++j) {
      if (j > 0) stream_ << ", ";
      stream_ << stride_names[j];
    }
    stream_ << "]";

    std::string layout_str = "nd";
    switch (layout) {
      case ir::TensorLayout::DN:
        layout_str = "dn";
        break;
      case ir::TensorLayout::NZ:
        layout_str = "nz";
        break;
      case ir::TensorLayout::ND:
        break;
    }
    stream_ << " {layout = #pto.layout<" << layout_str << ">}";

    stream_ << ": !pto.tensor_view<";
    for (size_t j = 0; j < rank; ++j) {
      if (j > 0) stream_ << "x";
      stream_ << "?";
    }
    stream_ << "x" << GetTypeString(tensor_type->dtype_) << ">\n";
  }
}

PTOCodegen::AllocTileFields PTOCodegen::ComputeAllocTileFields(
    const std::shared_ptr<const ir::TileType>& tile_type) {
  AllocTileFields fields;

  // Type string always uses dynamic valid dims (v_row=?, v_col=?); the actual
  // extent is conveyed via valid_row / valid_col operands below.
  fields.type_str = GetTileBufTypeStringFromTileType(tile_type);

  // Cast a non-index integer SSA to `index` (PTOAS expects index typed
  // valid_row / valid_col operands). Floating-point operands are rejected.
  auto cast_to_index = [&](const std::string& ssa, const ir::ExprPtr& expr) -> std::string {
    auto scalar_type = As<ScalarType>(expr->GetType());
    if (!scalar_type || scalar_type->dtype_ == DataType::INDEX) return ssa;
    CHECK(scalar_type->dtype_.IsInt())
        << "alloc_tile valid_row/valid_col operand must be integer or index typed, got "
        << GetTypeString(scalar_type->dtype_);
    std::string idx = NewTemp();
    Emit(idx + " = arith.index_cast " + ssa + " : " + GetTypeString(scalar_type->dtype_) + " to index");
    return idx;
  };

  // Lower a single valid_shape dim expression to an `index` SSA value.
  auto lower_dim = [&](const ir::ExprPtr& expr) -> std::string {
    if (!expr) return "";
    if (auto ci = As<ir::ConstInt>(expr)) {
      return GetOrEmitConstant(ci->value_, DataType::INDEX);
    }
    return cast_to_index(GetExprAsCode(expr), expr);
  };

  // Source of truth for valid_row / valid_col operand values:
  //   - tile_view.valid_shape when populated (preferred — captures user intent
  //     such as a smaller load region or a runtime ctx_len);
  //   - tile_type->shape_ otherwise (physical dims).
  const std::vector<ir::ExprPtr>* dims = nullptr;
  if (const auto& tile_view = tile_type->tile_view_;
      tile_view.has_value() && !tile_view->valid_shape.empty()) {
    dims = &tile_view->valid_shape;
  } else if (!tile_type->shape_.empty()) {
    dims = &tile_type->shape_;
  }

  if (dims != nullptr) {
    if (dims->size() == 1) {
      // Match ExtractTileTypeInfo: 1-D tile maps to rows=1, cols=shape[0].
      fields.valid_row_ssa = GetOrEmitConstant(static_cast<int64_t>(1), DataType::INDEX);
      fields.valid_col_ssa = lower_dim((*dims)[0]);
    } else {
      if (dims->size() >= 1) fields.valid_row_ssa = lower_dim((*dims)[0]);
      if (dims->size() >= 2) fields.valid_col_ssa = lower_dim((*dims)[1]);
    }
  }

  auto memref = ir::GetDefinedMemRef(tile_type);
  if (memref && emit_tile_addr_) {
    if (auto const_offset = As<ir::ConstInt>(memref->byte_offset_)) {
      fields.addr_ssa = GetOrEmitConstant(const_offset->value_, const_offset->dtype());
    }
  }
  return fields;
}

void PTOCodegen::EmitAllocTileForVar(const ir::VarPtr& tile_var,
                                     const std::shared_ptr<const ir::TileType>& tile_type) {
  auto var_key = GetVarKey(tile_var);
  if (!fs_.emitted_tile_alloc_vars.insert(var_key).second) {
    return;
  }

  auto mlir_it = fs_.var_to_mlir.find(var_key);
  INTERNAL_CHECK_SPAN(mlir_it != fs_.var_to_mlir.end(), tile_var->span_)
      << "Tile var " << tile_var->name_hint_ << " not found in fs_.var_to_mlir";
  std::string tile_buf = mlir_it->second;

  // In PTOAS mode several vars may share one handle (in-place aliasing); emit
  // the alloc_tile only once per handle so the shared buffer has a single def.
  if (!fs_.emitted_tile_alloc_names.insert(tile_buf).second) {
    return;
  }

  AllocTileFields fields = ComputeAllocTileFields(tile_type);

  std::ostringstream line;
  line << tile_buf << " = pto.alloc_tile";
  if (!fields.addr_ssa.empty()) line << " addr = " << fields.addr_ssa;
  if (!fields.valid_row_ssa.empty()) line << " valid_row = " << fields.valid_row_ssa;
  if (!fields.valid_col_ssa.empty()) line << " valid_col = " << fields.valid_col_ssa;
  line << " : " << fields.type_str;
  Emit(line.str());

  fs_.ssa_to_tile_buf_type[tile_buf] = fields.type_str;
}

// ========================================================================
// Private helpers
// ========================================================================

std::string PTOCodegen::GetIndent() const { return std::string(static_cast<size_t>(indent_level_) * 2, ' '); }

std::string PTOCodegen::GetOrEmitConstant(int64_t value, DataType dt) {
  auto key = std::make_pair(value, dt.Code());
  auto it = fs_.emitted_numeric_constants.find(key);
  if (it != fs_.emitted_numeric_constants.end()) return it->second;

  std::string mlir_type = GetTypeString(dt);
  // MLIR's arith.constant requires signless integer return types (upstream
  // ArithOps.cpp ConstantOp::verify). For unsigned dtypes, emit the constant
  // at the signless type and bridge to the unsigned type via
  // builtin.unrealized_conversion_cast; some consumer ops (e.g. pto.tci) in
  // turn require the operand type to match the destination dtype exactly.
  bool is_unsigned = dt.IsUnsignedInt() && !mlir_type.empty() && mlir_type[0] == 'u';
  std::string signless_type = is_unsigned ? mlir_type.substr(1) : mlir_type;
  std::string ssa_suffix = "_" + mlir_type;

  std::string ssa_id;
  if (value == 0) {
    ssa_id = "c0" + ssa_suffix;
  } else if (value < 0) {
    uint64_t mag = static_cast<uint64_t>(-(value + 1)) + 1;
    ssa_id = "cn" + std::to_string(mag) + ssa_suffix;
  } else {
    ssa_id = "c" + std::to_string(value) + ssa_suffix;
  }

  std::string name;
  if (!fs_.used_ssa_names.count(ssa_id)) {
    fs_.used_ssa_names.insert(ssa_id);
    name = "%" + ssa_id;
  } else {
    name = NewTemp();
  }

  if (is_unsigned) {
    std::string signless_name = NewTemp();
    fs_.constants_section << fs_.constants_indent << signless_name << " = arith.constant " << value << " : "
                          << signless_type << "\n";
    fs_.constants_section << fs_.constants_indent << name << " = builtin.unrealized_conversion_cast "
                          << signless_name << " : " << signless_type << " to " << mlir_type << "\n";
  } else {
    fs_.constants_section << fs_.constants_indent << name << " = arith.constant " << value << " : "
                          << mlir_type << "\n";
  }
  fs_.emitted_numeric_constants[key] = name;
  return name;
}

std::string PTOCodegen::GetOrEmitConstant(double value, DataType dt) {
  int64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  auto key = std::make_pair(bits, dt.Code());
  auto it = fs_.emitted_numeric_constants.find(key);
  if (it != fs_.emitted_numeric_constants.end()) return it->second;

  std::string mlir_type = GetTypeString(dt);
  std::string ssa_id = "cst";
  if (!fs_.emitted_numeric_constants.empty()) {
    ssa_id += "_" + std::to_string(fs_.emitted_numeric_constants.size());
  }
  std::string name;
  if (!fs_.used_ssa_names.count(ssa_id)) {
    fs_.used_ssa_names.insert(ssa_id);
    name = "%" + ssa_id;
  } else {
    name = NewTemp();
  }
  std::ostringstream val_str;
  val_str << std::scientific << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
  fs_.constants_section << fs_.constants_indent << name << " = arith.constant " << val_str.str() << " : "
                        << mlir_type << "\n";
  fs_.emitted_numeric_constants[key] = name;
  return name;
}

std::string PTOCodegen::GetTileBufForMemRef(const MemRefPtr& memref) const {
  INTERNAL_CHECK(memref != nullptr) << "Internal error: null MemRef passed to GetTileBufForMemRef";
  auto it = fs_.memref_to_mlir.find(memref->base_.get());
  INTERNAL_CHECK_SPAN(it != fs_.memref_to_mlir.end(), memref->span_)
      << "Internal error: no MLIR mapping for MemRef base '" << memref->base_->name_hint_ << "'";
  return it->second;
}

std::string PTOCodegen::AllocNewTileBuf(const std::string& tile_buf_type_string, const std::string& name_hint,
                                        const std::string& addr_ssa, const std::string& valid_row_ssa,
                                        const std::string& valid_col_ssa) {
  std::string name = NewNamedTemp(name_hint);
  fs_.extra_alloc_tiles.push_back(
      FunctionState::ExtraAllocTile{name, tile_buf_type_string, addr_ssa, valid_row_ssa, valid_col_ssa});
  fs_.ssa_to_tile_buf_type[name] = tile_buf_type_string;
  return name;
}

void PTOCodegen::SetCurrentResultBuf(const std::string& buf) { fs_.current_result_buf = buf; }

void PTOCodegen::RegisterTileBufType(const std::string& ssa_name, const std::string& type_string) {
  fs_.ssa_to_tile_buf_type[ssa_name] = type_string;
}

std::string PTOCodegen::GetSSATileBufType(const std::string& ssa_name) const {
  auto it = fs_.ssa_to_tile_buf_type.find(ssa_name);
  return it != fs_.ssa_to_tile_buf_type.end() ? it->second : std::string{};
}

void PTOCodegen::RegisterSubviewMaterialization(const std::string& subview_ssa,
                                                const SubviewMaterializationInfo& info) {
  fs_.subview_materializations[subview_ssa] = info;
}

PTOCodegen::SubviewMaterializationInfo* PTOCodegen::GetSubviewMaterialization(
    const std::string& subview_ssa) {
  auto it = fs_.subview_materializations.find(subview_ssa);
  return it != fs_.subview_materializations.end() ? &it->second : nullptr;
}

const PTOCodegen::SubviewMaterializationInfo* PTOCodegen::GetSubviewMaterialization(
    const std::string& subview_ssa) const {
  auto it = fs_.subview_materializations.find(subview_ssa);
  return it != fs_.subview_materializations.end() ? &it->second : nullptr;
}

void PTOCodegen::RecordGMSlotBufferSSA(const std::string& ssa, const DataType& dtype) {
  CHECK(dtype == DataType::FP32) << "__gm_pipe_buffer must use FP32 elements, got " << dtype.ToString();
  fs_.gm_slot_buffer_ssa = ssa;
  fs_.gm_slot_buffer_dtype = dtype;
}

std::string PTOCodegen::GetGMSlotBufferSSA() const { return fs_.gm_slot_buffer_ssa; }

std::string PTOCodegen::GetCommCtxSSAFor(const ir::Var* dist_var) const {
  if (dist_var == nullptr) return "";
  auto it = fs_.dist_tensor_to_ctx.find(dist_var);
  if (it == fs_.dist_tensor_to_ctx.end()) return "";
  return it->second;
}

void PTOCodegen::RegisterCommCtxFor(const ir::VarPtr& dist_var, const std::string& ctx_ssa) {
  if (!dist_var || ctx_ssa.empty()) return;
  fs_.dist_tensor_to_ctx[GetVarKey(dist_var)] = ctx_ssa;
}

std::string PTOCodegen::GetGMSlotBufferSSAForPipe(int pipe_id, int dir_mask) {
  if (fs_.gm_slot_buffer_ssa.empty()) {
    return "";
  }

  const auto key = std::make_pair(pipe_id, dir_mask);
  auto it = fs_.gm_slot_buffer_region_by_pipe.find(key);
  if (it != fs_.gm_slot_buffer_region_by_pipe.end()) {
    return it->second;
  }

  auto offset_it = gm_slot_buffer_offsets_.find(key);
  CHECK(offset_it != gm_slot_buffer_offsets_.end())
      << "Internal error: missing GM slot buffer offset for frontend pipe id " << pipe_id << " and dir_mask "
      << dir_mask;
  const int64_t byte_offset = offset_it->second;

  std::string region_ssa = fs_.gm_slot_buffer_ssa;
  if (byte_offset != 0) {
    const auto element_bytes = static_cast<int64_t>((fs_.gm_slot_buffer_dtype.GetBit() + 7) / 8);
    CHECK(element_bytes > 0) << "Unsupported __gm_pipe_buffer dtype: " << fs_.gm_slot_buffer_dtype.ToString();
    CHECK(byte_offset % element_bytes == 0)
        << "GM slot buffer byte offset must be aligned to " << fs_.gm_slot_buffer_dtype.ToString()
        << " elements, got " << byte_offset;
    const int64_t elem_offset = byte_offset / element_bytes;
    std::string offset_ssa = GetOrEmitConstant(elem_offset, DataType::INDEX);
    region_ssa = NewTemp();
    const std::string elem_type = GetTypeString(fs_.gm_slot_buffer_dtype);
    Emit(region_ssa + " = pto.addptr " + fs_.gm_slot_buffer_ssa + ", " + offset_ssa + " : <" + elem_type +
         "> -> <" + elem_type + ">");
  }

  fs_.gm_slot_buffer_region_by_pipe[key] = region_ssa;
  return region_ssa;
}

bool PTOCodegen::IsAICFunction() const {
  return fs_.current_function && fs_.current_function->func_type_ == ir::FunctionType::AIC;
}

bool PTOCodegen::IsAIVFunction() const {
  return fs_.current_function && fs_.current_function->func_type_ == ir::FunctionType::AIV;
}

bool PTOCodegen::IsDualAivDispatchFunction() const {
  return fs_.current_function && fs_.current_function->HasAttr("dual_aiv_dispatch") &&
         fs_.current_function->GetAttr<bool>("dual_aiv_dispatch", false);
}

void PTOCodegen::EmitExtraAllocTiles() {
  for (const auto& alloc : fs_.extra_alloc_tiles) {
    stream_ << GetIndent() << alloc.name << " = pto.alloc_tile";
    if (emit_tile_addr_ && !alloc.addr_ssa.empty()) {
      stream_ << " addr = " << alloc.addr_ssa;
    }
    if (!alloc.valid_row_ssa.empty()) {
      stream_ << " valid_row = " << alloc.valid_row_ssa;
    }
    if (!alloc.valid_col_ssa.empty()) {
      stream_ << " valid_col = " << alloc.valid_col_ssa;
    }
    stream_ << " : " << alloc.type_string << "\n";
  }
}

// ========================================================================
// Statement visitors
// ========================================================================

void PTOCodegen::VisitStmt(const ir::StmtPtr& stmt) {
  // Defensive: the first-class SplitAivScopeStmt region is consumed and erased
  // by LowerAutoVectorSplit (pass 21), ~19 passes before codegen. There is no
  // ScopeStmt handler here, so a survivor would be silently unwrapped by the
  // base visitor — losing the region semantics. Fail loudly instead.
  INTERNAL_CHECK_SPAN(!ir::As<ir::SplitAivScopeStmt>(stmt), stmt->span_)
      << "Internal error: SplitAivScopeStmt reached PTO codegen; it must be lowered and erased by "
         "LowerAutoVectorSplit (pass 21).";
  ir::IRVisitor::VisitStmt(stmt);
}

void PTOCodegen::VisitStmt_(const AssignStmtPtr& op) {
  auto call = As<ir::Call>(op->value_);
  const bool is_set_validshape = ir::IsOp(call, "tile.set_validshape");
  const bool alias_scatter_result_to_input = ShouldAliasScatterResultToInput(op);
  const bool alias_array_update_to_input = ShouldAliasArrayUpdateResultToInput(op);

  if (auto tile_type = ir::GetTileTypeWithMemRef(op->var_->GetType())) {
    if (!is_set_validshape && !alias_scatter_result_to_input) {
      EmitAllocTileForVar(op->var_, tile_type);
    }
  }

  if (call) {
    if (backend_ != nullptr && backend_->GetOpInfo(call->op_->name_) != nullptr) {
      std::string result_buf =
          op->var_->name_hint_;  // Seed for readable MLIR names when no tile buffer exists.
      std::shared_ptr<const TileType> result_tile_type;
      if (auto tile_type = ir::GetTileTypeWithMemRef(op->var_->GetType())) {
        if (alias_scatter_result_to_input) {
          result_buf = GetExprAsCode(call->args_[0]);
          INTERNAL_CHECK(!result_buf.empty())
              << "Internal error: " << call->op_->name_ << " result must alias the input tile SSA";
          BindVarToMlir(op->var_, result_buf);
        } else {
          // Prefer per-var SSA name from fs_.var_to_mlir (set during per-var alloc binding)
          auto var_it = fs_.var_to_mlir.find(GetVarKey(op->var_));
          if (var_it != fs_.var_to_mlir.end()) {
            result_buf = var_it->second;
          } else {
            result_buf = GetTileBufForMemRef(ir::GetDefinedMemRef(tile_type));
          }
        }
        result_tile_type = tile_type;
      } else if (auto tile_type = As<TileType>(op->var_->GetType())) {
        // A MemRef-less tile result (e.g. a cross-core tpop result, whose data
        // lives in the reserved C2V/V2C slot) still needs a %-SSA name bound so
        // consumers resolve it; its tile_buf type comes from the TileType since
        // there is no MemRef to read. Register it before the op codegen runs so
        // GetCurrentResultTileBufTypeString() can emit the `-> type` annotation.
        result_tile_type = tile_type;
        result_buf = NewNamedTemp(op->var_->name_hint_);
        BindVarToMlir(op->var_, result_buf);
        std::string type_str = GetTileBufTypeStringFromTileType(tile_type);
        if (!type_str.empty()) {
          fs_.ssa_to_tile_buf_type[result_buf] = type_str;
        }
      } else if (alias_array_update_to_input) {
        // array.update_element: alias the result Var to the input array's SSA so
        // the emitted pto.local_array_set mutates the same declare_local_array
        // storage in place (mirrors the SSA-functional -> in-place lowering).
        result_buf = GetExprAsCode(call->args_[0]);
        INTERNAL_CHECK_SPAN(!result_buf.empty(), op->span_)
            << "Internal error: array.update_element result must alias the input array SSA";
        BindVarToMlir(op->var_, result_buf);
      } else {
        // Pre-allocate a %-prefixed SSA name for non-tile backend ops (e.g., scalar
        // results like tile.getval, or i32 results like reserve_buffer / import_peer_buffer).
        // Register it in fs_.var_to_mlir so subsequent expressions can resolve the variable.
        result_buf = NewNamedTemp(op->var_->name_hint_);
        BindVarToMlir(op->var_, result_buf);
      }
      fs_.current_result_var = op->var_;
      fs_.current_result_buf = result_buf;
      fs_.current_result_tile_type = result_tile_type;
      VisitExpr(op->value_);
      // If codegen changed the result buffer (e.g., reshape allocated a new tile),
      // update variable mapping so subsequent references use the new buffer
      if (!fs_.current_result_buf.empty() && (is_set_validshape || fs_.current_result_buf != result_buf)) {
        BindVarToMlir(op->var_, fs_.current_result_buf);
      }
      // Register per-variable tile_buf type from the variable's own TileType.
      // This ensures that even when multiple variables share a MemRef, each
      // variable's SSA value carries its correct typed annotation.
      if (result_tile_type && !fs_.current_result_buf.empty() && fs_.current_result_buf == result_buf) {
        std::string var_type_str = GetTileBufTypeStringFromTileType(result_tile_type);
        if (!var_type_str.empty()) {
          fs_.ssa_to_tile_buf_type[fs_.current_result_buf] = var_type_str;
        }
      }
      fs_.current_result_var.reset();
      fs_.current_result_buf.clear();
      fs_.current_result_tile_type = nullptr;
      return;
    }
  }

  // Plain tensor alias: `lhs_tensor = rhs_var` with no Call on the RHS. This
  // arises when Simplify folds an empty loop-result ForStmt into a plain
  // AssignStmt — e.g. a constant-trip `pl.pipeline`'s statically-empty main
  // loop becomes `t__rv_vN_main = t__iter_vM`. The deleted ForStmt would have
  // registered the loop-result tensor view / SSA name for its return var (see
  // VisitStmt_(ForStmtPtr) in pto_control_flow_codegen.cpp), so a later
  // tile.store into the alias can resolve its view instead of
  // GetOrCreateTensorView tripping its INTERNAL_CHECK on the synthetic var.
  // We additionally propagate the base-ptr mapping so element-wise alias
  // consumers (pl.read / pl.write / store_scalar) resolve to the backing
  // pointer rather than the view SSA — as the IfStmt in-place-return path
  // (VisitStmt_(IfStmtPtr)) does for merged tensors.
  // Non-fatal: if the RHS has no registered view, fall through to the generic
  // handling rather than throwing eagerly on a view that may never be consumed.
  if (auto rhs_var = AsVarLike(op->value_)) {
    if (ir::AsTensorTypeLike(op->var_->GetType())) {
      const std::string view = TryGetTensorView(rhs_var);
      if (!view.empty()) {
        BindTensorView(op->var_, view);
        BindVarToMlir(op->var_, view);  // view name == SSA name, as in ForStmt
        RegisterBasePtr(op->var_, GetTensorBasePtr(rhs_var));
        return;
      }
    }
  }

  fs_.current_expr_value = "";
  VisitExpr(op->value_);
  // Register scalar/index/CommCtx result so subsequent expressions can look up
  // this variable. N7: CommCtxType is a singleton marker; the bound SSA is the
  // matching ``!pto.ptr<i64>`` ctx ptr from the func.func trailing-ctx segment
  // (no MLIR is emitted for ``pld.system.get_comm_ctx`` — its lambda just sets
  // ``current_expr_value`` to the ctx SSA). Treating it like a scalar here lets
  // downstream ``pld.system.rank(ctx)`` / ``pld.system.nranks(ctx)`` codegen
  // resolve ``ctx`` via the standard ``GetExprAsCode(call->args_[0])`` path.
  const auto& var_type = op->var_->GetType();
  if ((As<ScalarType>(var_type) || As<CommCtxType>(var_type)) && !fs_.current_expr_value.empty()) {
    BindVarToMlir(op->var_, fs_.current_expr_value);
  }
}

// ========================================================================
// Expression visitors
// ========================================================================

void PTOCodegen::VisitExpr_(const CallPtr& op) {
  const std::string& op_name = op->op_->name_;

  CHECK(backend_ != nullptr) << "Backend must not be null; use PTOCodegen(backend) or default backend";
  const auto* op_info = backend_->GetOpInfo(op_name);
  if (op_info == nullptr) {
    ThrowNoCodegenForCall(op_name);
  }
  std::string mlir_line = op_info->codegen_func(op, *this);
  if (!mlir_line.empty()) {
    Emit(mlir_line);
  }
}

// ========================================================================
// CodegenBase interface and PTO-specific helper methods
// ========================================================================

std::string PTOCodegen::GetCurrentResultTarget() const { return fs_.current_result_buf; }

ir::VarPtr PTOCodegen::GetCurrentResultVar() const { return fs_.current_result_var; }

std::vector<ir::VarPtr> PTOCodegen::ResolveTupleResultElements(const ir::VarPtr& tuple_var,
                                                               size_t arity) const {
  INTERNAL_CHECK(tuple_var) << "Internal error: ResolveTupleResultElements requires non-null tuple_var";
  INTERNAL_CHECK(fs_.current_function)
      << "Internal error: ResolveTupleResultElements requires current_function";
  TupleConsumerCollector collector(tuple_var.get(), arity);
  collector.VisitStmt(fs_.current_function->body_);
  return collector.elements();
}

void PTOCodegen::Emit(const std::string& line) { stream_ << GetIndent() << line << "\n"; }

std::string PTOCodegen::GetExprAsCode(const ExprPtr& expr) {
  if (auto var = As<ir::Var>(expr)) {
    return GetVarName(var);
  }
  if (auto const_int = As<ir::ConstInt>(expr)) {
    return GetOrEmitConstant(const_int->value_, const_int->dtype());
  }
  if (auto const_float = As<ir::ConstFloat>(expr)) {
    return GetOrEmitConstant(const_float->value_, const_float->dtype());
  }

  // Fall back to visitor pattern for complex expressions (arithmetic, comparisons)
  fs_.current_expr_value = "";
  VisitExpr(expr);
  std::string result = fs_.current_expr_value;
  fs_.current_expr_value = "";
  if (!result.empty()) {
    return result;
  }

  LOG_ERROR << "GetExprAsCode for unsupported expression type";
  return "";
}

std::string PTOCodegen::GetTypeString(const DataType& dtype) const { return DataTypeToMLIR(dtype); }

const ir::Var* PTOCodegen::GetVarKey(const VarPtr& var) const {
  INTERNAL_CHECK(var != nullptr) << "Internal error: variable key requested for null Var";
  return var.get();
}

void PTOCodegen::BindVarToMlir(const VarPtr& var, const std::string& mlir_name) {
  fs_.var_to_mlir[GetVarKey(var)] = mlir_name;
}

void PTOCodegen::BindTensorView(const VarPtr& var, const std::string& tensor_view_name) {
  fs_.tensor_to_view[GetVarKey(var)] = tensor_view_name;
}

void PTOCodegen::BindVarToMemRef(const VarPtr& var, const ir::Var* base_ptr) {
  fs_.var_to_memref[GetVarKey(var)] = base_ptr;
}

std::string PTOCodegen::GetVarName(const VarPtr& var) const {
  auto key = GetVarKey(var);
  auto it = fs_.var_to_mlir.find(key);
  if (it != fs_.var_to_mlir.end()) {
    return it->second;
  }
  auto memref_it = fs_.var_to_memref.find(key);
  if (memref_it != fs_.var_to_memref.end()) {
    auto mlir_it = fs_.memref_to_mlir.find(memref_it->second);
    if (mlir_it != fs_.memref_to_mlir.end()) {
      return mlir_it->second;
    }
  }
  if (auto tile_type = ir::GetTileTypeWithMemRef(var->GetType())) {
    return GetTileBufForMemRef(ir::GetDefinedMemRef(tile_type));
  }
  for (const auto& [mapped_var, mlir_name] : fs_.var_to_mlir) {
    if (mapped_var && mapped_var->name_hint_ == var->name_hint_) {
      return mlir_name;
    }
  }
  LOG_ERROR << "Variable " << var->name_hint_ << " not found in MLIR mapping";
  return "";
}

std::string PTOCodegen::NewTemp() {
  std::string name = std::to_string(fs_.temp_counter++);
  while (fs_.used_ssa_names.count(name)) {
    name = std::to_string(fs_.temp_counter++);
  }
  fs_.used_ssa_names.insert(name);
  return "%" + name;
}

std::string PTOCodegen::NewNamedTemp(const std::string& name) {
  // Sanitize name to be a valid MLIR SSA identifier: [a-zA-Z_][a-zA-Z0-9_$.]*
  std::string sanitized = name;
  if (!sanitized.empty()) {
    for (auto& c : sanitized) {
      if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '.' && c != '$') {
        c = '_';
      }
    }
    if (std::isdigit(static_cast<unsigned char>(sanitized[0]))) {
      sanitized.insert(0, 1, '_');
    }
  }

  if (!sanitized.empty() && fs_.used_ssa_names.find(sanitized) == fs_.used_ssa_names.end()) {
    fs_.used_ssa_names.insert(sanitized);
    return "%" + sanitized;
  }
  return NewTemp();
}

void PTOCodegen::RegisterVarToMlir(const VarPtr& var, const std::string& mlir_name) {
  BindVarToMlir(var, mlir_name);
}

void PTOCodegen::RegisterTensorView(const VarPtr& var, const std::string& tensor_view_name) {
  BindTensorView(var, tensor_view_name);
}

void PTOCodegen::RegisterBasePtr(const VarPtr& var, const std::string& ptr_name) {
  if (var && !ptr_name.empty()) fs_.tensor_to_base_ptr[GetVarKey(var)] = ptr_name;
}

std::string PTOCodegen::GetTensorBasePtr(const VarPtr& tensor) const {
  auto it = fs_.tensor_to_base_ptr.find(GetVarKey(tensor));
  if (it != fs_.tensor_to_base_ptr.end()) return it->second;
  // For IterArg, follow initValue_ to the original tensor parameter (mirrors GetOrCreateTensorView).
  if (auto iter_arg = As<ir::IterArg>(tensor)) {
    if (auto init_var = AsVarLike(iter_arg->initValue_)) return GetTensorBasePtr(init_var);
  }
  return GetVarName(tensor);
}

int64_t PTOCodegen::GetConstIntValue(const ExprPtr& expr) const {
  if (auto const_int = As<ir::ConstInt>(expr)) {
    return const_int->value_;
  }
  LOG_ERROR << "Expected ConstInt expression";
  return 0;
}

std::string PTOCodegen::TryGetTensorView(const VarPtr& tensor_var) const {
  auto it = fs_.tensor_to_view.find(GetVarKey(tensor_var));
  if (it != fs_.tensor_to_view.end()) return it->second;
  // For IterArg, follow initValue_ chain to the original tensor parameter.
  if (auto iter_arg = As<ir::IterArg>(tensor_var)) {
    if (auto init_var = As<ir::Var>(iter_arg->initValue_)) return TryGetTensorView(init_var);
    if (auto init_iter = As<ir::IterArg>(iter_arg->initValue_)) return TryGetTensorView(init_iter);
  }
  return "";
}

std::string PTOCodegen::GetOrCreateTensorView(const VarPtr& tensor_var) {
  std::string view = TryGetTensorView(tensor_var);
  INTERNAL_CHECK_SPAN(!view.empty(), tensor_var->span_)
      << "Tensor view not found for parameter: " << tensor_var->name_hint_;
  return view;
}

std::string PTOCodegen::GetTensorViewTypeString(const ir::TensorType* tensor_type) const {
  std::ostringstream oss;
  oss << "!pto.tensor_view<";
  for (size_t i = 0; i < tensor_type->shape_.size(); i++) {
    if (i > 0) oss << "x";
    oss << "?";
  }
  oss << "x" << GetTypeString(tensor_type->dtype_) << ">";
  return oss.str();
}

std::string PTOCodegen::GetTileBufTypeString(const ir::Var* base_ptr) const {
  INTERNAL_CHECK(base_ptr != nullptr) << "Internal error: null base_ptr passed to GetTileBufTypeString";
  auto tile_it = fs_.memref_to_tile_type.find(base_ptr);
  INTERNAL_CHECK_SPAN(tile_it != fs_.memref_to_tile_type.end(), base_ptr->span_)
      << "Internal error: missing tile type for base Ptr '" << base_ptr->name_hint_ << "'";
  auto memory_space = tile_it->second->GetMemorySpace();
  INTERNAL_CHECK_SPAN(memory_space.has_value(), base_ptr->span_)
      << "Internal error: tile type must have memory_space";

  std::string loc = MemorySpaceToMLIR(*memory_space);
  auto c = ExtractTileTypeInfo(*tile_it->second, GetTypeString(tile_it->second->dtype_));
  return FormatTileBufTypeString(loc, c.dtype_str, c.rows, c.cols, c.blayout, c.slayout, c.fractal, c.pad,
                                 c.v_row, c.v_col, c.v_row_dynamic, c.v_col_dynamic);
}

std::string PTOCodegen::GetTileBufTypeStringFromTileType(
    const std::shared_ptr<const ir::TileType>& tile_type) const {
  INTERNAL_CHECK(tile_type) << "Internal error: tile_type must not be null";
  auto memory_space = tile_type->GetMemorySpace();
  INTERNAL_CHECK(memory_space.has_value()) << "Internal error: tile_type must have memory_space";

  std::string loc = MemorySpaceToMLIR(*memory_space);
  auto c = ExtractTileTypeInfo(*tile_type, GetTypeString(tile_type->dtype_));
  return FormatTileBufTypeString(loc, c.dtype_str, c.rows, c.cols, c.blayout, c.slayout, c.fractal, c.pad,
                                 c.v_row, c.v_col, c.v_row_dynamic, c.v_col_dynamic);
}

std::string PTOCodegen::GetExprTypeAnnotation(const ir::ExprPtr& expr) {
  if (auto var = As<ir::Var>(expr)) {
    auto key = GetVarKey(var);
    // Primary lookup: SSA name → tile_buf type (covers root allocs AND view results)
    auto mlir_it = fs_.var_to_mlir.find(key);
    if (mlir_it != fs_.var_to_mlir.end()) {
      auto ssa_it = fs_.ssa_to_tile_buf_type.find(mlir_it->second);
      if (ssa_it != fs_.ssa_to_tile_buf_type.end()) {
        return ssa_it->second;
      }
    }
    // Per-variable TileType: derives the type from the variable's own
    // TileType, which is correct for view op results (slice, reshape,
    // fillpad) whose type differs from the root alloc's type.
    if (auto tile_type = As<TileType>(var->GetType())) {
      if (tile_type->memref_.has_value()) {
        return GetTileBufTypeStringFromTileType(tile_type);
      }
    }
    // Fallback: var → memref → root alloc type
    auto memref_it = fs_.var_to_memref.find(key);
    if (memref_it != fs_.var_to_memref.end()) {
      return GetTileBufTypeString(memref_it->second);
    }
    if (auto scalar_type = As<ScalarType>(var->GetType())) {
      return GetTypeString(scalar_type->dtype_);
    }
  }
  if (auto iter_arg = As<ir::IterArg>(expr)) {
    auto key = GetVarKey(std::dynamic_pointer_cast<const ir::Var>(iter_arg));
    auto mlir_it = fs_.var_to_mlir.find(key);
    if (mlir_it != fs_.var_to_mlir.end()) {
      auto ssa_it = fs_.ssa_to_tile_buf_type.find(mlir_it->second);
      if (ssa_it != fs_.ssa_to_tile_buf_type.end()) {
        return ssa_it->second;
      }
    }
    if (auto tile_type = ir::GetTileTypeWithMemRef(iter_arg->GetType())) {
      return GetTileBufTypeStringFromTileType(tile_type);
    }
    auto memref_it = fs_.var_to_memref.find(key);
    if (memref_it != fs_.var_to_memref.end()) {
      return GetTileBufTypeString(memref_it->second);
    }
    if (auto scalar_type = As<ScalarType>(iter_arg->GetType())) {
      return GetTypeString(scalar_type->dtype_);
    }
  }
  if (auto const_float = As<ir::ConstFloat>(expr)) {
    return GetTypeString(const_float->dtype());
  }
  if (auto const_int = As<ir::ConstInt>(expr)) {
    // The SSA value produced by GetOrEmitConstant is cast back to the dtype's
    // MLIR type (via unrealized_conversion_cast for unsigned), so the use-site
    // annotation matches the declared dtype directly.
    return GetTypeString(const_int->dtype());
  }
  // Fallback: derive annotation from any ScalarType expression (e.g. Cast results,
  // arith expression results). Their SSA value carries the declared dtype.
  if (auto scalar_type = As<ScalarType>(expr->GetType())) {
    return GetTypeString(scalar_type->dtype_);
  }
  return "";
}

std::string PTOCodegen::GetCurrentResultTileBufTypeString() const {
  // Prefer the type registered by alloc_tile (always dynamic
  // `v_row=?, v_col=?` per ComputeAllocTileFields).
  if (!fs_.current_result_buf.empty()) {
    auto ssa_it = fs_.ssa_to_tile_buf_type.find(fs_.current_result_buf);
    if (ssa_it != fs_.ssa_to_tile_buf_type.end()) {
      return ssa_it->second;
    }
  }
  if (fs_.current_result_tile_type) {
    if (const auto& memref = fs_.current_result_tile_type->memref_) {
      return GetTileBufTypeString((*memref)->base_.get());
    }
  }
  return "";
}

std::string PTOCodegen::GetCurrentResultTileBufTypeStringFromTileType() const {
  if (fs_.current_result_tile_type && fs_.current_result_tile_type->memref_.has_value()) {
    return GetTileBufTypeStringFromTileType(fs_.current_result_tile_type);
  }
  return "";
}

std::pair<std::string, std::string> PTOCodegen::GetCurrentResultTpopValidShapeOperands() {
  if (!fs_.current_result_tile_type) {
    return {"", ""};
  }

  const auto& tile_view = fs_.current_result_tile_type->tile_view_;
  if (!tile_view) {
    return {"", ""};
  }

  const auto& valid_shape = tile_view->valid_shape;
  ExprPtr valid_row_expr;
  ExprPtr valid_col_expr;
  bool has_dynamic_valid_shape = false;
  if (valid_shape.size() >= 1 && valid_shape[0]) {
    valid_row_expr = valid_shape[0];
    has_dynamic_valid_shape = !As<ir::ConstInt>(valid_row_expr);
  }
  if (valid_shape.size() >= 2 && valid_shape[1]) {
    valid_col_expr = valid_shape[1];
    has_dynamic_valid_shape = has_dynamic_valid_shape || !As<ir::ConstInt>(valid_col_expr);
  }
  bool valid_shape_matches_shape = !valid_row_expr && !valid_col_expr;
  if (valid_row_expr && valid_col_expr) {
    const auto& shape = fs_.current_result_tile_type->shape_;
    ExprPtr shape_row_expr;
    ExprPtr shape_col_expr;
    if (shape.size() >= 2) {
      shape_row_expr = shape[0];
      shape_col_expr = shape[1];
    } else if (shape.size() == 1) {
      shape_row_expr = std::make_shared<ir::ConstInt>(1, DataType::INDEX, ir::Span::unknown());
      shape_col_expr = shape[0];
    }
    valid_shape_matches_shape =
        IsSameDimExpr(valid_row_expr, shape_row_expr) && IsSameDimExpr(valid_col_expr, shape_col_expr);
  }
  if (!has_dynamic_valid_shape && valid_shape_matches_shape) {
    return {"", ""};
  }

  auto cast_scalar_to_index = [&](const std::string& ssa, const ScalarType* scalar_type) -> std::string {
    bool is_integer_or_index = scalar_type->dtype_.IsInt() || scalar_type->dtype_ == DataType::INDEX;
    CHECK(is_integer_or_index && scalar_type->dtype_.GetBit() != 1)
        << "tpop valid_shape operand must be integer or index type, got "
        << GetTypeString(scalar_type->dtype_);
    if (scalar_type->dtype_ == DataType::INDEX) {
      return ssa;
    }
    std::string idx = NewTemp();
    std::string src_type = GetTypeString(scalar_type->dtype_);
    Emit(idx + " = arith.index_cast " + ssa + " : " + src_type + " to index");
    return idx;
  };

  auto get_index_operand = [&](const ExprPtr& expr, size_t dim_idx) -> std::string {
    if (expr) {
      if (auto const_int = As<ir::ConstInt>(expr)) {
        return GetOrEmitConstant(const_int->value_, DataType::INDEX);
      }
      std::string ssa = GetExprAsCode(expr);
      if (auto scalar_type = As<ScalarType>(expr->GetType())) {
        return cast_scalar_to_index(ssa, scalar_type.get());
      }
      return ssa;
    }

    const auto& shape = fs_.current_result_tile_type->shape_;
    ExprPtr shape_dim;
    if (shape.size() >= 2 && dim_idx < shape.size()) {
      shape_dim = shape[dim_idx];
    } else if (shape.size() == 1) {
      if (dim_idx == 0) {
        return GetOrEmitConstant(static_cast<int64_t>(1), DataType::INDEX);
      }
      shape_dim = shape[0];
    }
    INTERNAL_CHECK(shape_dim) << "Internal error: tpop result tile type is missing shape dim " << dim_idx;
    if (auto const_int = As<ir::ConstInt>(shape_dim)) {
      return GetOrEmitConstant(const_int->value_, DataType::INDEX);
    }
    std::string ssa = GetExprAsCode(shape_dim);
    if (auto scalar_type = As<ScalarType>(shape_dim->GetType())) {
      return cast_scalar_to_index(ssa, scalar_type.get());
    }
    return ssa;
  };

  return {get_index_operand(valid_row_expr, 0), get_index_operand(valid_col_expr, 1)};
}

}  // namespace codegen
}  // namespace pypto
