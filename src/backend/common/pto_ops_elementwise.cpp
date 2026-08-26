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
 * @file pto_ops_elementwise.cpp
 * @brief PTO codegen registration for elementwise / compute tile ops.
 */

#include <algorithm>
#include <cstddef>
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
#include "pypto/codegen/codegen_base.h"
#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"
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
using pto_ops_detail::CheckArity;
using pto_ops_detail::cmp_modes;
using pto_ops_detail::EmitInsOuts;
using pto_ops_detail::GenerateInsOutsClause;
using pto_ops_detail::MaterializeSubviewOperandIfNeeded;
using pto_ops_detail::round_modes;

static bool RequiresRowMajorLayout(std::string_view op_name) {
  static const std::unordered_set<std::string_view> kRowMajorOps = {
      // Tile x Tile binary ops
      "tile.add",
      "tile.and",
      "tile.fmod",
      "tile.maximum",
      "tile.minimum",
      "tile.mul",
      "tile.or",
      "tile.part_add",
      "tile.part_max",
      "tile.part_min",
      "tile.part_mul",
      "tile.rem",
      "tile.shl",
      "tile.shr",
      "tile.sub",
      "tile.xor",
      // Unary ops
      "tile.abs",
      "tile.exp",
      "tile.sqrt",
      "tile.not",
      "tile.prelu",
      "tile.relu",
      // Tile x Scalar ops
      "tile.adds",
      "tile.subs",
      "tile.muls",
      "tile.divs",
      "tile.fmods",
      "tile.maximums",
      "tile.lrelu",
      "tile.sels",
      // Gather operands and result are linearly addressed.
      "tile.gatherb",
      // Ternary scalar ops (Tile x Scalar x Tile)
      "tile.addsc",
      "tile.subsc",
  };
  return kRowMajorOps.count(op_name) > 0;
}

// Helper function for N-ary operations (unary, binary, ternary, etc.)
static std::string MakeNaryCodegenPTO(const std::string& pto_op_name, size_t arity, const CallPtr& op,
                                      codegen::CodegenBase& codegen_base,
                                      std::optional<size_t> i32_operand_idx = std::nullopt) {
  auto& codegen = AsPto(codegen_base);
  CheckArity(op, pto_op_name, arity);
  if (i32_operand_idx.has_value()) {
    INTERNAL_CHECK_SPAN(*i32_operand_idx < op->args_.size(), op->span_)
        << "Internal error: " << pto_op_name << " i32 operand index " << *i32_operand_idx
        << " is outside the " << op->args_.size() << " input operands";
    std::vector<std::pair<std::string, std::string>> ins;
    ins.reserve(op->args_.size());
    for (size_t i = 0; i < op->args_.size(); ++i) {
      const ExprPtr& arg = op->args_[i];
      std::string operand = codegen.GetExprAsCode(arg);
      std::string type = codegen.GetExprTypeAnnotation(arg);
      if (i == *i32_operand_idx) {
        operand = codegen.EmitCastToI32(arg, operand);
        type = codegen.GetTypeString(DataType::INT32);
      }
      ins.emplace_back(std::move(operand), std::move(type));
    }
    EmitInsOuts(codegen, pto_op_name, ins);
    return "";
  }
  // The pto.tcolexpand* family requires materialized tile data — their hardware
  // lowering reads physical tile rows/cols from the operand type, which is
  // incorrect for a pto.subview alias.  Other tile ops (tmov, tfillpad, tadd,
  // ...) accept subview SSAs natively, so only the tcolexpand family needs
  // eager materialization.
  if (pto_op_name == "pto.tcolexpandmul" || pto_op_name == "pto.tcolexpandadd" ||
      pto_op_name == "pto.tcolexpanddiv" || pto_op_name == "pto.tcolexpandsub" ||
      pto_op_name == "pto.tcolexpandmax" || pto_op_name == "pto.tcolexpandmin" ||
      pto_op_name == "pto.tcolexpandexpdif") {
    // Derive a debug hint from the op name (e.g. "pto.tcolexpandmul" -> "colexpandmul").
    const std::string mat_tag = pto_op_name.substr(std::string("pto.t").size());
    auto lhs_operand = MaterializeSubviewOperandIfNeeded(op->args_[0], codegen, mat_tag + "_mat");
    auto rhs_operand = MaterializeSubviewOperandIfNeeded(op->args_[1], codegen, mat_tag + "_vec");
    std::string lhs_orig = codegen.GetExprAsCode(op->args_[0]);
    std::string rhs_orig = codegen.GetExprAsCode(op->args_[1]);
    if (lhs_operand != lhs_orig || rhs_operand != rhs_orig) {
      // Resolve type annotations: use the materialized target type when
      // the operand was a subview, otherwise use the original annotation.
      std::string lhs_type = codegen.GetExprTypeAnnotation(op->args_[0]);
      auto* lhs_mat = codegen.GetSubviewMaterialization(lhs_orig);
      if (lhs_mat) lhs_type = lhs_mat->materialize_target_type;

      std::string rhs_type = codegen.GetExprTypeAnnotation(op->args_[1]);
      auto* rhs_mat = codegen.GetSubviewMaterialization(rhs_orig);
      if (rhs_mat) rhs_type = rhs_mat->materialize_target_type;

      std::ostringstream oss;
      oss << pto_op_name << " ins(" << lhs_operand << ", " << rhs_operand;
      if (!lhs_type.empty() && !rhs_type.empty()) {
        oss << " : " << lhs_type << ", " << rhs_type;
      }
      std::string result_target = codegen.GetCurrentResultTarget();
      std::string result_type = codegen.GetCurrentResultTileBufTypeString();
      oss << ") outs(" << result_target;
      if (!result_type.empty()) oss << " : " << result_type;
      oss << ")";
      codegen.Emit(oss.str());
      return "";
    }
  }
  codegen.Emit(pto_op_name + " " + GenerateInsOutsClause(op, codegen));
  return "";
}

static std::string GemvAccPhaseAttr(const CallPtr& op) {
  const auto acc_phase = op->GetKwarg<std::string>("acc_phase", "unspecified");
  CHECK(acc_phase == "unspecified" || acc_phase == "partial" || acc_phase == "final")
      << "GEMV acc_phase must be one of {unspecified, partial, final}, but got " << acc_phase;
  if (acc_phase == "unspecified") return "";
  return " {accPhase = #pto<acc_phase " + acc_phase + ">}";
}

static std::string MakeGemvCodegenPTO(const std::string& pto_op_name, size_t arity, const CallPtr& op,
                                      codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CheckArity(op, pto_op_name, arity);
  codegen.Emit(pto_op_name + " " + GenerateInsOutsClause(op, codegen) + GemvAccPhaseAttr(op));
  return "";
}

static std::string MakeTileSelCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CheckArity(op, "pto.tsel", 4);
  codegen.Emit("pto.tsel " + GenerateInsOutsClause(op, codegen));
  return "";
}

// pto.ttrans ins(%src, %tmp : tile_type, tile_type). IR form: tile.transpose(src, axis0, axis1, tmp).
// tmp is pre-allocated by an IR-level tile.create so the memory allocator gives it a real UB
// address before codegen (required at --pto-level=level3).
static std::string MakeTileTransposeCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  INTERNAL_CHECK_SPAN(op->args_.size() == 4, op->span_)
      << "tile.transpose requires 4 arguments (src, axis0, axis1, tmp), got " << op->args_.size();

  std::string src_ssa = codegen.GetExprAsCode(op->args_[0]);
  std::string src_type = codegen.GetExprTypeAnnotation(op->args_[0]);
  std::string tmp_ssa = codegen.GetExprAsCode(op->args_[3]);
  // Fall back to tmp's annotation when src lacks a MemRef (ForStmt result var, tile.reshape view).
  if (src_type.empty()) {
    src_type = codegen.GetExprTypeAnnotation(op->args_[3]);
  }

  // Both operands carry src's annotation (pto.ttrans requires a matched type
  // pair); when src_type is empty, EmitInsOuts omits the whole `: types` clause.
  EmitInsOuts(codegen, "pto.ttrans", {{src_ssa, src_type}, {tmp_ssa, src_type}});
  return std::string("");
}

// Single-operand tile ops whose output shape/type come from the AssignStmt
// context, so exactly one args_ entry is emitted as the ins() operand:
//   tile.col_expand -> pto.tcolexpand: emits the column vector (args_[1]); args_[0]
//                      (target) is kept only for shape/type inference.
//   tile.row_expand -> pto.trowexpand: emits the row vector (args_[1]); ditto.
//   tile.fillpad_expand -> pto.tfillpad_expand: emits the source tile (args_[0]);
//                      args_[1] (shape tuple) is type-deduction only. The pad value
//                      and dst extents ride on the result tile-buf type.
struct SingleOperandOp {
  const char* ir_name;   // IR op name, for the arity CHECK message
  const char* pto_op;    // emitted pto op name
  size_t operand_idx;    // which args_ entry becomes the ins() operand
  const char* arg_desc;  // extra description in the arity message (e.g. " (src, shape)")
};

static std::string MakeSingleOperandCodegenPTO(const SingleOperandOp& spec, const CallPtr& op,
                                               codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  INTERNAL_CHECK_SPAN(op->args_.size() == 2, op->span_)
      << spec.ir_name << " requires 2 arguments" << spec.arg_desc << ", got " << op->args_.size();
  const ir::ExprPtr& operand = op->args_[spec.operand_idx];
  EmitInsOuts(codegen, spec.pto_op,
              {{codegen.GetExprAsCode(operand), codegen.GetExprTypeAnnotation(operand)}});
  return "";
}

// Shared driver for tile ops that carry an integer `mode` kwarg selecting an
// enum name, emitted as a `{<attr_key> = #pto<<attr_kind> NAME>}` config:
//   tile.cmp / tile.cmps -> {cmpMode = #pto<cmp NAME>}       (cmp_modes,   arity 2)
//   tile.cvt             -> {rmode   = #pto<round_mode NAME>} (round_modes, arity 1)
static std::string MakeModalCodegenPTO(const std::string& pto_op_name, size_t arity, const char* kwarg,
                                       const std::vector<std::string>& modes, const char* range_label,
                                       const char* attr_key, const char* attr_kind, const CallPtr& op,
                                       codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CheckArity(op, pto_op_name, arity);
  int mode = op->GetKwarg<int>(kwarg);
  CHECK(mode >= 0 && mode < static_cast<int>(modes.size())) << range_label << " mode out of range: " << mode;
  std::string config_attr =
      std::string("{") + attr_key + " = #pto<" + attr_kind + " " + modes.at(mode) + ">}";
  codegen.Emit(pto_op_name + " " + GenerateInsOutsClause(op, codegen, config_attr));
  return "";
}

// Emit the default PTO form without an explicit precision attribute, or append
// the exact PTOAS enum attribute after outs(...) for high-precision mode.
// Unlike cmp/cvt attributes, precision-op assembly formats place their
// attr-dict after the complete ins()/outs() clause.
static std::string MakePrecisionCodegenPTO(const std::string& pto_op_name, size_t arity,
                                           const char* attr_kind, const CallPtr& op,
                                           codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CheckArity(op, pto_op_name, arity);
  const bool high_precision = op->GetKwarg<bool>("high_precision", false);
  std::string code = pto_op_name + " " + GenerateInsOutsClause(op, codegen);
  if (high_precision) {
    code += " {precisionType = #pto<";
    code += attr_kind;
    code += " high_precision>}";
  }
  codegen.Emit(code);
  return "";
}

// Helper function for full op
static std::string MakeFullCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                      codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CheckArity(op, pto_op_name, 2);
  const ir::ExprPtr& scalar = op->args_[1];
  EmitInsOuts(codegen, pto_op_name, {{codegen.GetExprAsCode(scalar), codegen.GetExprTypeAnnotation(scalar)}});
  return "";
}

// Helper function for Assign
static std::string MakeAssignCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                        codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CheckArity(op, pto_op_name, 2);
  std::string tile = codegen.GetExprAsCode(op->args_[0]);
  std::string addr = codegen.GetExprAsCode(op->args_[1]);
  codegen.Emit(pto_op_name + " ins(" + tile + ", " + addr + ")");
  return "";
}

// Helper function for Ci
static std::string MakeCiCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                    codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  INTERNAL_CHECK_SPAN(op->args_.size() == 2, op->span_)
      << "Operation:[" << pto_op_name << "] requires 2 arguments (start, shape), but got "
      << op->args_.size();
  bool descending = op->GetKwarg<bool>("descending");
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string src_type = codegen.GetExprTypeAnnotation(op->args_[0]);
  std::string config_attr = descending ? "{descending = true}" : "{descending = false}";
  std::string dst = codegen.GetCurrentResultTarget();
  std::string dst_type = codegen.GetCurrentResultTileBufTypeString();
  std::ostringstream oss;
  oss << pto_op_name << " ins(" << src;
  if (!src_type.empty()) {
    oss << " : " << src_type;
  }
  oss << ") outs(" << dst;
  if (!dst_type.empty()) {
    oss << " : " << dst_type;
  }
  oss << ") " << config_attr;
  codegen.Emit(oss.str());
  return "";
}

// TTRI's upper/lower selector is only accepted through the generic MLIR form.
// Shape and optional valid_shape operands are type-only and are not emitted.
static std::string MakeTriCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  INTERNAL_CHECK_SPAN(op->args_.size() == 2 || op->args_.size() == 3, op->span_)
      << "Operation:[pto.ttri] requires 2 or 3 arguments (diagonal, shape, [valid_shape]), but got "
      << op->args_.size();
  auto result_type = As<ir::TileType>(op->GetType());
  INTERNAL_CHECK(result_type) << "tile.tri result must be a TileType";
  const auto* handler = codegen.GetBackendHandler();
  INTERNAL_CHECK(handler) << "tile.tri requires a backend handler";
  if (handler->GetPtoTargetArch() == "a2a3") {
    CHECK_SPAN(result_type->dtype_ != DataType::INT8 && result_type->dtype_ != DataType::UINT8 &&
                   result_type->dtype_ != DataType::BF16,
               op->span_)
        << "tile.tri dtype " << result_type->dtype_.ToString()
        << " is not supported on the 'a2a3' backend; use the A5 backend";
  }
  const bool upper = op->GetKwarg<bool>("upper", false);
  const std::string diagonal = codegen.GetExprAsCode(op->args_[0]);
  const std::string diagonal_type = codegen.GetExprTypeAnnotation(op->args_[0]);
  const std::string dst = codegen.GetCurrentResultTarget();
  const std::string dst_type = codegen.GetCurrentResultTileBufTypeString();

  std::ostringstream oss;
  oss << "\"pto.ttri\"(" << diagonal << ", " << dst << ") {upperOrLower = " << (upper ? 1 : 0)
      << " : i32} : (" << diagonal_type << ", " << dst_type << ") -> ()";
  codegen.Emit(oss.str());
  return "";
}

static std::string MakeGatherbCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CheckArity(op, "pto.tgatherb", 2);
  auto src = AsVarLike(op->args_[0]);
  INTERNAL_CHECK(src) << "tile.gatherb src must be a Var or IterArg";
  auto src_type = As<ir::TileType>(src->GetType());
  INTERNAL_CHECK(src_type) << "tile.gatherb src must be a TileType";
  const std::string src_ssa = codegen.GetExprAsCode(op->args_[0]);
  if (const auto* subview = codegen.GetSubviewMaterialization(src_ssa)) {
    CHECK_SPAN(subview->byte_offset_mod_32 == 0, op->span_)
        << "tile.gatherb source subview byte offset must be provably 32-byte aligned";
  }
  if (src_type->memref_.has_value()) {
    auto byte_offset = As<ir::ConstInt>((*src_type->memref_)->byte_offset_);
    CHECK_SPAN(byte_offset, op->span_)
        << "tile.gatherb source base byte offset must be statically known and 32-byte aligned";
    // PtoAS memory planning deliberately keeps root-allocation offsets at the -1
    // sentinel. Concrete offsets are assigned by the conventional planner.
    if (byte_offset->value_ >= 0) {
      CHECK_SPAN(byte_offset->value_ % 32 == 0, op->span_)
          << "tile.gatherb source base byte offset must be 32-byte aligned";
    }
  }
  codegen.Emit("pto.tgatherb " + GenerateInsOutsClause(op, codegen));
  return "";
}

// Helper function for Random: emits pto.trandom.
// IR tile.random(key0, key1, counter0..3, shape) carries the shape tuple as the
// last arg for type deduction only; the hardware reads the destination extent
// from the result type, so only the 6 seed scalars are emitted as operands.
//
// The `rounds` attribute is special: ptoas' custom assembly format for
// pto.trandom has no trailing attr-dict slot (a `... {rounds = N}` suffix fails
// to parse). The PTOAS template defaults rounds to 10 when the attribute is
// absent, so the common rounds==10 case is emitted as the clean DPS custom form
//   pto.trandom ins(k0..c3 : i32 x6) outs(dst : dst_type)
// and a non-default rounds is attached via the MLIR generic op form, the only
// spelling ptoas accepts the attribute in:
//   "pto.trandom"(k0..c3, dst) {rounds = N : i32} : (i32 x6, dst_type) -> ()
static std::string MakeRandomCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                        codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  INTERNAL_CHECK_SPAN(op->args_.size() == 7 || op->args_.size() == 8, op->span_)
      << "Operation:[" << pto_op_name
      << "] requires 7 or 8 arguments (key0, key1, counter0, counter1, counter2, "
         "counter3, shape, [valid_shape]), but got "
      << op->args_.size();
  int rounds = op->GetKwarg<int>("rounds", 10);
  std::vector<std::string> seeds;
  std::vector<std::string> seed_types;
  seeds.reserve(6);
  seed_types.reserve(6);
  for (size_t i = 0; i < 6; ++i) {
    seeds.push_back(codegen.GetExprAsCode(op->args_[i]));
    seed_types.push_back(codegen.GetExprTypeAnnotation(op->args_[i]));
  }
  const std::string dst = codegen.GetCurrentResultTarget();
  const std::string dst_type = codegen.GetCurrentResultTileBufTypeString();

  std::ostringstream oss;
  if (rounds == 10) {
    oss << pto_op_name << " ins(";
    for (size_t i = 0; i < 6; ++i) oss << (i ? ", " : "") << seeds[i];
    oss << " : ";
    for (size_t i = 0; i < 6; ++i) oss << (i ? ", " : "") << seed_types[i];
    oss << ") outs(" << dst;
    if (!dst_type.empty()) oss << " : " << dst_type;
    oss << ")";
  } else {
    oss << "\"" << pto_op_name << "\"(";
    for (size_t i = 0; i < 6; ++i) oss << seeds[i] << ", ";
    oss << dst << ") {rounds = " << rounds << " : i32} : (";
    for (size_t i = 0; i < 6; ++i) oss << seed_types[i] << ", ";
    oss << dst_type << ") -> ()";
  }
  codegen.Emit(oss.str());
  return "";
}

// Helper function for Print
static std::string MakePrintCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                       codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  INTERNAL_CHECK_SPAN(op->args_.size() == 1, op->span_)
      << "Operation:" << pto_op_name << "] requires 1 argument, but got " << op->args_.size();
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  codegen.Emit(pto_op_name + " ins(" + src + " | !pto.partition_tensor_view<MxNxdtype>)");
  return "";
}

static std::string MakeSelsCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CheckArity(op, "pto.tsels", 4);
  auto mask_type = As<ir::TileType>(op->args_[0]->GetType());
  auto src_type = As<ir::TileType>(op->args_[1]->GetType());
  auto tmp_type = As<ir::TileType>(op->args_[2]->GetType());
  INTERNAL_CHECK(mask_type && src_type && tmp_type);
  const auto* handler = codegen.GetBackendHandler();
  const bool is_a5 = handler->GetPtoTargetArch() == "a5";
  const auto dtype = src_type->dtype_;
  const bool supported_on_a2a3 = dtype == DataType::INT16 || dtype == DataType::UINT16 ||
                                 dtype == DataType::INT32 || dtype == DataType::UINT32 ||
                                 dtype == DataType::FP16 || dtype == DataType::FP32;
  CHECK_SPAN(supported_on_a2a3 || is_a5, op->span_)
      << "tile.sels with integer src dtype " << src_type->dtype_.ToString()
      << " is only supported on the 'a5' backend; A2/A3 supports 16/32-bit integers, FP16, and FP32";

  auto dst_var = codegen.GetCurrentResultVar();
  INTERNAL_CHECK_SPAN(dst_var, op->span_) << "Internal error: tile.sels requires an assignment target";
  auto dst_type = As<ir::TileType>(dst_var->GetType());
  INTERNAL_CHECK_SPAN(dst_type, op->span_) << "Internal error: tile.sels result must be a TileType";

  std::vector<std::pair<std::string_view, std::shared_ptr<const ir::TileType>>> operands = {
      {"mask", mask_type}, {"src", src_type}, {"tmp", tmp_type}, {"dst", dst_type}};
  std::vector<std::pair<std::string_view, ir::MemRefPtr>> regions;
  regions.reserve(operands.size());
  for (const auto& [name, type] : operands) {
    INTERNAL_CHECK_SPAN(type->memref_.has_value(), op->span_)
        << "Internal error: tile.sels " << name << " must carry a MemRef before PTO codegen";
    regions.emplace_back(name, *type->memref_);
  }
  CHECK_SPAN(!ir::MemRef::MayAlias(regions[0].second, regions[3].second), op->span_)
      << "tile.sels requires mask and dst to use non-overlapping memory regions";
  if (!is_a5) {
    for (const size_t other : {size_t{0}, size_t{1}}) {
      CHECK_SPAN(!ir::MemRef::MayAlias(regions[2].second, regions[other].second), op->span_)
          << "tile.sels on A2/A3 requires tmp not to overlap mask or src, but tmp overlaps "
          << regions[other].first;
    }
  }
  return MakeNaryCodegenPTO("pto.tsels", 4, op, codegen_base);
}

static std::string MakePreluCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CheckArity(op, "pto.tprelu", 3);
  auto src_type = As<ir::TileType>(op->args_[0]->GetType());
  auto slope_type = As<ir::TileType>(op->args_[1]->GetType());
  auto tmp_type = As<ir::TileType>(op->args_[2]->GetType());
  INTERNAL_CHECK(src_type && slope_type && tmp_type);

  auto dst_var = codegen.GetCurrentResultVar();
  INTERNAL_CHECK_SPAN(dst_var, op->span_) << "Internal error: tile.prelu requires an assignment target";
  auto dst_type = As<ir::TileType>(dst_var->GetType());
  INTERNAL_CHECK_SPAN(dst_type, op->span_) << "Internal error: tile.prelu result must be a TileType";

  if (codegen.GetBackendHandler()->GetPtoTargetArch() == "a5") {
    INTERNAL_CHECK_SPAN(src_type->memref_.has_value(), op->span_)
        << "Internal error: tile.prelu src must carry a MemRef before PTO codegen";
    INTERNAL_CHECK_SPAN(slope_type->memref_.has_value(), op->span_)
        << "Internal error: tile.prelu slope must carry a MemRef before PTO codegen";
    INTERNAL_CHECK_SPAN(tmp_type->memref_.has_value(), op->span_)
        << "Internal error: tile.prelu tmp must carry a MemRef before PTO codegen";
    INTERNAL_CHECK_SPAN(dst_type->memref_.has_value(), op->span_)
        << "Internal error: tile.prelu dst must carry a MemRef before PTO codegen";
    CHECK_SPAN(!ir::MemRef::MayAlias(*src_type->memref_, *dst_type->memref_), op->span_)
        << "tile.prelu on A5 requires dst not to overlap src";
    CHECK_SPAN(!ir::MemRef::MayAlias(*slope_type->memref_, *dst_type->memref_), op->span_)
        << "tile.prelu on A5 requires dst not to overlap slope";
    EmitInsOuts(codegen, "pto.tprelu",
                {{codegen.GetExprAsCode(op->args_[0]), codegen.GetExprTypeAnnotation(op->args_[0])},
                 {codegen.GetExprAsCode(op->args_[1]), codegen.GetExprTypeAnnotation(op->args_[1])},
                 {codegen.GetExprAsCode(op->args_[2]), codegen.GetExprTypeAnnotation(op->args_[2])}});
    return "";
  }

  CHECK_SPAN(tmp_type->dtype_ == DataType::UINT8, op->args_[2]->span_)
      << "tile.prelu on A2/A3 requires UINT8 tmp scratch, but got " << tmp_type->dtype_.ToString();
  const auto src_valid_shape = ir::GetValidShape(src_type);
  const auto tmp_valid_shape = ir::GetValidShape(tmp_type);
  const auto required_rows = ir::MakeAdd(
      src_valid_shape[0], std::make_shared<ir::ConstInt>(1, DataType::INDEX, op->span_), op->span_);
  ir::ExprPtr required_cols;
  if (auto const_cols = As<ir::ConstInt>(src_valid_shape[1])) {
    required_cols =
        std::make_shared<ir::ConstInt>((const_cols->value_ + 7) / 8, DataType::INDEX, const_cols->span_);
  } else {
    required_cols = ir::MakeFloorDiv(
        ir::MakeAdd(src_valid_shape[1],
                    std::make_shared<ir::ConstInt>(7, DataType::INDEX, src_valid_shape[1]->span_),
                    src_valid_shape[1]->span_),
        std::make_shared<ir::ConstInt>(8, DataType::INDEX, src_valid_shape[1]->span_),
        src_valid_shape[1]->span_);
  }
  CHECK_SPAN(ir::ProveValidExtentLessEqual(required_rows, tmp_type->shape_[0]) == ir::ProofResult::kTrue,
             op->args_[2]->span_)
      << "tile.prelu on A2/A3 requires UINT8 tmp physical rows >= src valid rows + 1";
  CHECK_SPAN(ir::ProveValidExtentLessEqual(required_cols, tmp_valid_shape[1]) == ir::ProofResult::kTrue,
             op->args_[2]->span_)
      << "tile.prelu on A2/A3 requires UINT8 tmp valid columns >= ceil(src valid columns / 8)";

  std::vector<std::pair<std::string_view, std::shared_ptr<const ir::TileType>>> operands = {
      {"src", src_type}, {"slope", slope_type}, {"tmp", tmp_type}, {"dst", dst_type}};
  std::vector<std::pair<std::string_view, ir::MemRefPtr>> regions;
  regions.reserve(operands.size());
  for (const auto& [name, type] : operands) {
    INTERNAL_CHECK_SPAN(type->memref_.has_value(), op->span_)
        << "Internal error: tile.prelu " << name << " must carry a MemRef before PTO codegen";
    regions.emplace_back(name, *type->memref_);
  }
  for (size_t i = 0; i < regions.size(); ++i) {
    for (size_t j = i + 1; j < regions.size(); ++j) {
      CHECK_SPAN(!ir::MemRef::MayAlias(regions[i].second, regions[j].second), op->span_)
          << "tile.prelu on A2/A3 requires src, slope, tmp, and dst to use pairwise non-overlapping "
             "memory regions, but "
          << regions[i].first << " overlaps " << regions[j].first;
    }
  }

  EmitInsOuts(codegen, "pto.tprelu",
              {{codegen.GetExprAsCode(op->args_[0]), codegen.GetExprTypeAnnotation(op->args_[0])},
               {codegen.GetExprAsCode(op->args_[1]), codegen.GetExprTypeAnnotation(op->args_[1])},
               {codegen.GetExprAsCode(op->args_[2]), codegen.GetExprTypeAnnotation(op->args_[2])}});
  return "";
}

struct SimpleOpEntry {
  const char* op_name;
  const char* pto_op_name;
  size_t arity;
  std::optional<size_t> i32_operand_idx = std::nullopt;
};

// clang-format off
static const SimpleOpEntry kSimpleOps[] = {
    // Tile x Tile arithmetic operations
    {"tile.add",             "pto.tadd",             2},
    {"tile.sub",             "pto.tsub",             2},
    {"tile.mul",             "pto.tmul",             2},
    {"tile.rem",             "pto.trem",             3},  // src0, src1, tmp
    // Tile x Tile partial-combine operations
    {"tile.part_add",        "pto.tpartadd",         2},
    {"tile.part_mul",        "pto.tpartmul",         2},
    {"tile.part_max",        "pto.tpartmax",         2},
    {"tile.part_min",        "pto.tpartmin",         2},
    {"tile.fmod",            "pto.tfmod",            2},
    // Tile x Tile bitwise operations
    {"tile.and",             "pto.tand",             2},
    {"tile.or",              "pto.tor",              2},
    {"tile.xor",             "pto.txor",             3},  // src0, src1, tmp
    {"tile.shl",             "pto.tshl",             2},
    {"tile.shr",             "pto.tshr",             2},
    // Tile x Tile comparison/selection operations
    {"tile.maximum",         "pto.tmax",             2},
    {"tile.minimum",         "pto.tmin",             2},
    // Unary operations
    {"tile.abs",             "pto.tabs",             1},
    {"tile.exp",             "pto.texp",             1},
    {"tile.sqrt",            "pto.tsqrt",            1},
    // tile.rsqrt is registered with a custom codegen handler below (supports 1 or 2 args).
    {"tile.neg",             "pto.tneg",             1},
    {"tile.not",             "pto.tnot",             1},
    {"tile.relu",            "pto.trelu",            1},
    // Ternary operations (tile x tile + carry/select)
    {"tile.addc",            "pto.taddc",            3},
    {"tile.subc",            "pto.tsubc",            3},
    // Tile x Scalar operations
    {"tile.adds",            "pto.tadds",            2},
    {"tile.subs",            "pto.tsubs",            2},
    {"tile.muls",            "pto.tmuls",            2},
    {"tile.divs",            "pto.tdivs",            2},
    {"tile.rems",            "pto.trems",            3},  // src0, scalar, tmp
    {"tile.fmods",           "pto.tfmods",           2},
    {"tile.ands",            "pto.tands",            2, 1},
    {"tile.ors",             "pto.tors",             2, 1},
    {"tile.xors",            "pto.txors",            3, 1},  // src0, scalar, tmp
    {"tile.shls",            "pto.tshls",            2, 1},
    {"tile.shrs",            "pto.tshrs",            2, 1},
    {"tile.maximums",        "pto.tmaxs",            2},
    {"tile.minimums",        "pto.tmins",            2},
    {"tile.lrelu",           "pto.tlrelu",           2},
    // Ternary scalar operations (tile x scalar + carry/select)
    {"tile.addsc",           "pto.taddsc",           3},
    {"tile.subsc",           "pto.tsubsc",           3},
    {"tile.selc",            "pto.tselc",            3},
    // Axis reduction/expansion operations
    {"tile.row_sum",         "pto.trowsum",          2},
    {"tile.row_max",         "pto.trowmax",          2},
    {"tile.row_min",         "pto.trowmin",          2},
    {"tile.row_prod",        "pto.trowprod",         2},
    {"tile.col_max",         "pto.tcolmax",          1},
    {"tile.col_min",         "pto.tcolmin",          1},
    {"tile.col_prod",        "pto.tcolprod",         1},
    // Argmax/argmin reductions — int32 index output, require a tmp scratch tile.
    {"tile.row_argmax",      "pto.trowargmax",       2},
    {"tile.row_argmin",      "pto.trowargmin",       2},
    {"tile.col_argmax",      "pto.tcolargmax",       2},
    {"tile.col_argmin",      "pto.tcolargmin",       2},
    {"tile.col_expand_mul",  "pto.tcolexpandmul",    2},
    {"tile.col_expand_add",  "pto.tcolexpandadd",    2},
    {"tile.col_expand_div",  "pto.tcolexpanddiv",    2},
    {"tile.col_expand_sub",  "pto.tcolexpandsub",    2},
    {"tile.col_expand_max",  "pto.tcolexpandmax",    2},
    {"tile.col_expand_min",  "pto.tcolexpandmin",    2},
    {"tile.col_expand_expdif", "pto.tcolexpandexpdif", 2},
    {"tile.row_expand_div",  "pto.trowexpanddiv",    2},
    {"tile.row_expand_mul",  "pto.trowexpandmul",    2},
    {"tile.row_expand_sub",  "pto.trowexpandsub",    2},
    {"tile.row_expand_max",  "pto.trowexpandmax",    2},
    {"tile.row_expand_min",  "pto.trowexpandmin",    2},
    {"tile.row_expand_expdif", "pto.trowexpandexpdif", 2},
    // Padding operations
    {"tile.fillpad",         "pto.tfillpad",         1},
    // Inplace variant: set_output_reuses_input(0) makes src/dst share UB addr.
    {"tile.fillpad_inplace", "pto.tfillpad",         1},
    // Matrix multiplication operations (PipeType::M → CUBE/AIC core)
    {"tile.matmul",          "pto.tmatmul",          2},
    {"tile.matmul_mx",       "pto.tmatmul.mx",       4},
    {"tile.matmul_mx_bias",  "pto.tmatmul.mx.bias",  5},
    // tile.matmul_acc / tile.gemv_acc / tile.matmul_mx_acc have custom codegen
    // (in-place accumulation: ptoas requires ins(acc) == outs).
    {"tile.matmul_bias",     "pto.tmatmul.bias",     3},
    // tile.gemv_acc has custom codegen (in-place accumulation)
    // Data movement/layout operations
    {"tile.concat",          "pto.tconcat",          2},
    // tile.move has custom codegen (PTOAS same-handle elision and baked-address validation)
    {"tile.move_fp",         "pto.tmov.fp",          2},
    // tile.transpose has custom codegen (MakeTileTransposeCodegenPTO): pto.ttrans needs
    // ins(%src, %tmp : tile_type, tile_type) where %tmp is a scratch workspace tile, NOT
    // the axis-index integers that tile.transpose(src, axis0, axis1) carries in the IR.
    // tile.extract has custom codegen (see reg("tile.extract") below): the IR carries the
    // shape tuple as args_[3] purely for type deduction, so the generic N-ary lowering
    // would emit the tuple as a PTO operand — not what pto.textract expects.
    // Gather/scatter operations
    {"tile.gather",          "pto.tgather",          3},
    // tile.scatter and tile.scatter_mask are registered with custom codegen
    // handlers below (DPS — dst is `args_[0]`, aliased to the result via
    // set_output_reuses_input(0)).
    // Partial reduction operations
    {"tile.partadd",         "pto.tpartadd",         2},
    {"tile.partmax",         "pto.tpartmax",         2},
    {"tile.partmin",         "pto.tpartmin",         2},
};
// clang-format on

void RegisterElementwiseOps(Backend& backend, const std::unordered_set<std::string>& exclude_ops) {
  // Register simple N-ary ops
  for (const auto& entry : kSimpleOps) {
    if (exclude_ops.count(entry.op_name) > 0) continue;
    std::string pto_op = entry.pto_op_name;
    size_t arity = entry.arity;
    std::optional<size_t> i32_operand_idx = entry.i32_operand_idx;
    auto reg_entry = backend.RegisterOp(entry.op_name);
    reg_entry.f_codegen([pto_op, arity, i32_operand_idx](const CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeNaryCodegenPTO(pto_op, arity, op, codegen, i32_operand_idx);
    });
    if (RequiresRowMajorLayout(entry.op_name)) {
      for (size_t i = 0; i < arity; ++i) {
        reg_entry.set_input_layout(i, ir::TileLayout::row_major);
      }
      reg_entry.set_output_layout(ir::TileLayout::row_major);
    }
  }

  if (exclude_ops.count("tile.sels") == 0) {
    auto entry = backend.RegisterOp("tile.sels");
    entry.f_codegen(MakeSelsCodegenPTO);
    for (size_t i = 0; i < 4; ++i) {
      entry.set_input_layout(i, ir::TileLayout::row_major);
    }
    entry.set_output_layout(ir::TileLayout::row_major);
  }

  if (exclude_ops.count("tile.prelu") == 0) {
    auto entry = backend.RegisterOp("tile.prelu");
    entry.f_codegen(MakePreluCodegenPTO);
    for (size_t i = 0; i < 3; ++i) {
      entry.set_input_layout(i, ir::TileLayout::row_major);
    }
    entry.set_output_layout(ir::TileLayout::row_major);
  }

  // Register ops with custom codegen logic
  auto reg = [&](const char* op_name, BackendCodegenFunc fn) {
    if (exclude_ops.count(op_name) > 0) return;
    backend.RegisterOp(op_name).f_codegen(std::move(fn));
  };

  auto register_precision_op = [&](const char* op_name, const char* pto_op_name, size_t arity,
                                   const char* attr_kind) {
    if (exclude_ops.count(op_name) > 0) return;
    auto reg_entry = backend.RegisterOp(op_name);
    reg_entry.f_codegen([pto_op = std::string(pto_op_name), arity, attr_kind = std::string(attr_kind)](
                            const CallPtr& op, codegen::CodegenBase& codegen) {
      return MakePrecisionCodegenPTO(pto_op, arity, attr_kind.c_str(), op, codegen);
    });
    for (size_t i = 0; i < arity; ++i) {
      reg_entry.set_input_layout(i, ir::TileLayout::row_major);
    }
    reg_entry.set_output_layout(ir::TileLayout::row_major);
  };
  register_precision_op("tile.div", "pto.tdiv", 2, "div_precision");
  register_precision_op("tile.log", "pto.tlog", 1, "log_precision");
  register_precision_op("tile.recip", "pto.trecip", 1, "recip_precision");

  // tile.row_expand_add follows the PTOAS overloads with and without tmp.
  // Its row-sensitive layout contract is validated by the IR op: the generic
  // backend layout repair may reshape [M, 1] to [1, M], which changes semantics.
  if (exclude_ops.count("tile.row_expand_add") == 0) {
    auto reg_entry = backend.RegisterOp("tile.row_expand_add");
    reg_entry.f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      const size_t arity = op->args_.size();
      INTERNAL_CHECK_SPAN(arity == 2 || arity == 3, op->span_)
          << "tile.row_expand_add requires 2 or 3 arguments, but got " << arity;
      return MakeNaryCodegenPTO("pto.trowexpandadd", arity, op, codegen);
    });
  }

  // tile.move → pto.tmov.
  //
  // tile.move is registered not_inplace_safe(), so the PyPTO and DSA-RP
  // planners must assign distinct source and destination addresses. Validate
  // that invariant here as well: explicit MemRef bindings and hand-built IR can
  // bypass planner-created no-alias constraints, and TMOV does not support an
  // in-place same-address instruction.
  reg("tile.move", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    auto& codegen = AsPto(codegen_base);
    INTERNAL_CHECK_SPAN(op->args_.size() == 1, op->span_)
        << "tile.move requires 1 argument, got " << op->args_.size();

    // Under memory_planner=PtoAS there is no baked address (AllocateMemoryAddr
    // and the reuse-packer's not_inplace_safe gate are both skipped). A
    // redundant loop-carry write-back that YieldFixupMutator inserts collapses
    // onto a single tile_buf handle, and PTO codegen re-points the producer at
    // the phi handle (#1956/#1985). Elide only that exact case — src and dst
    // denote one handle — so we never emit an illegal same-handle pto.tmov.
    if (!codegen.EmitTileAddr()) {
      std::string src_ssa = codegen.GetExprAsCode(op->args_[0]);
      if (!src_ssa.empty() && src_ssa == codegen.GetCurrentResultTarget()) {
        return std::string("");  // no-op: one handle, the op already wrote in place
      }
      codegen.Emit("pto.tmov " + GenerateInsOutsClause(op, codegen));
      return std::string("");
    }

    const auto src_var = AsVarLike(op->args_[0]);
    const auto dst_var = codegen.GetCurrentResultVar();
    if (src_var && dst_var) {
      const auto src_tile = As<ir::TileType>(src_var->GetType());
      const auto dst_tile = As<ir::TileType>(dst_var->GetType());
      if (src_tile && dst_tile && src_tile->memref_.has_value() && dst_tile->memref_.has_value()) {
        const auto src_space = src_tile->GetMemorySpace();
        const auto dst_space = dst_tile->GetMemorySpace();
        if (src_space.has_value() && dst_space.has_value() && *src_space == *dst_space) {
          const ir::MemRefPtr& src_memref = *src_tile->memref_;
          const ir::MemRefPtr& dst_memref = *dst_tile->memref_;
          if (src_memref && dst_memref && src_memref->byte_offset_ && dst_memref->byte_offset_ &&
              ir::AreExprsEqual(src_memref->byte_offset_, dst_memref->byte_offset_)) {
            const auto const_offset = As<ir::ConstInt>(src_memref->byte_offset_);
            const std::string address = const_offset ? "byte offset " + std::to_string(const_offset->value_)
                                                     : "the same symbolic byte offset";
            CHECK_SPAN(false, op->span_)
                << "tile.move requires distinct source and destination addresses in "
                << ir::MemorySpaceToString(*src_space) << ", but both resolve to " << address;
          }
        }
      }
    }

    codegen.Emit("pto.tmov " + GenerateInsOutsClause(op, codegen));
    return std::string("");
  });

  reg("tile.transpose", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTileTransposeCodegenPTO(op, codegen);
  });

  if (exclude_ops.count("tile.sel") == 0) {
    backend.RegisterOp("tile.sel")
        .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
          return MakeTileSelCodegenPTO(op, codegen);
        })
        .set_input_layout(0, ir::TileLayout::row_major)
        .set_input_layout(1, ir::TileLayout::row_major)
        .set_input_layout(2, ir::TileLayout::row_major)
        .set_input_layout(3, ir::TileLayout::row_major)
        .set_output_layout(ir::TileLayout::row_major);
  }

  reg("tile.col_expand", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeSingleOperandCodegenPTO({"tile.col_expand", "pto.tcolexpand", 1, ""}, op, codegen);
  });
  reg("tile.row_expand", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeSingleOperandCodegenPTO({"tile.row_expand", "pto.trowexpand", 1, ""}, op, codegen);
  });
  reg("tile.fillpad_expand", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeSingleOperandCodegenPTO({"tile.fillpad_expand", "pto.tfillpad_expand", 0, " (src, shape)"}, op,
                                       codegen);
  });

  reg("tile.cmp", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeModalCodegenPTO("pto.tcmp", 2, "cmp_type", cmp_modes, "Tile cmp", "cmpMode", "cmp", op,
                               codegen);
  });

  // tile.cast (TCVT): pto.tcvt mis-orders elements on a col_major source, so per
  // ISA the input and output must be row_major (see #1549).
  if (exclude_ops.count("tile.cast") == 0) {
    backend.RegisterOp("tile.cast")
        .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
          return MakeModalCodegenPTO("pto.tcvt", 1, "mode", round_modes, "Round", "rmode", "round_mode", op,
                                     codegen);
        })
        .set_input_layout(0, ir::TileLayout::row_major)
        .set_output_layout(ir::TileLayout::row_major);
  }

  // tile.rsqrt accepts 1 arg (basic) or 2 args (high-precision with tmp workspace).
  // Both forms emit pto.trsqrt with the appropriate ins() arity. Per ISA, both
  // inputs (when present) and the output must be row_major.
  if (exclude_ops.count("tile.rsqrt") == 0) {
    backend.RegisterOp("tile.rsqrt")
        .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
          size_t arity = op->args_.size();
          INTERNAL_CHECK_SPAN(arity == 1 || arity == 2, op->span_)
              << "tile.rsqrt requires 1 or 2 arguments, but got " << arity;
          return MakeNaryCodegenPTO("pto.trsqrt", arity, op, codegen);
        })
        .set_input_layout(0, ir::TileLayout::row_major)
        .set_input_layout(1, ir::TileLayout::row_major)
        .set_output_layout(ir::TileLayout::row_major);
  }

  // tile.col_sum (TCOLSUM): accepts 1 arg (sequential) or 2 args (tile + tmp for binary-tree).
  // PTOAS pairs tmp operand with isBinary attribute; both present or both absent.
  if (exclude_ops.count("tile.col_sum") == 0) {
    backend.RegisterOp("tile.col_sum")
        .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
          auto& codegen = AsPto(codegen_base);
          INTERNAL_CHECK_SPAN(op->args_.size() == 1 || op->args_.size() == 2, op->span_)
              << "tile.col_sum requires 1 or 2 arguments, but got " << op->args_.size();
          std::string config_attr = op->args_.size() == 2 ? " {isBinary = true}" : "";
          codegen.Emit("pto.tcolsum " + GenerateInsOutsClause(op, codegen, config_attr));
          return std::string("");
        });
  }

  // tile.full (TEXPANDS): output is row_major per ISA
  if (exclude_ops.count("tile.full") == 0) {
    backend.RegisterOp("tile.full")
        .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
          return MakeFullCodegenPTO("pto.texpands", op, codegen);
        })
        .set_output_layout(ir::TileLayout::row_major);
  }

  // tile.cmps (TCMPS): tile input and output must be row_major per ISA
  if (exclude_ops.count("tile.cmps") == 0) {
    backend.RegisterOp("tile.cmps")
        .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
          return MakeModalCodegenPTO("pto.tcmps", 2, "cmp_type", cmp_modes, "Tile cmp", "cmpMode", "cmp", op,
                                     codegen);
        })
        .set_input_layout(0, ir::TileLayout::row_major)
        .set_output_layout(ir::TileLayout::row_major);
  }

  reg("tile.assign", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeAssignCodegenPTO("pto.tassign", op, codegen);
  });
  if (exclude_ops.count("tile.gatherb") == 0) {
    backend.RegisterOp("tile.gatherb")
        .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
          return MakeGatherbCodegenPTO(op, codegen);
        })
        .set_input_layout(0, ir::TileLayout::row_major)
        .set_input_layout(1, ir::TileLayout::row_major)
        .set_output_layout(ir::TileLayout::row_major);
  }

  reg("tile.ci", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeCiCodegenPTO("pto.tci", op, codegen);
  });

  if (exclude_ops.count("tile.tri") == 0) {
    backend.RegisterOp("tile.tri")
        .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
          return MakeTriCodegenPTO(op, codegen);
        })
        .set_output_layout(ir::TileLayout::row_major);
  }

  // tile.random (TRANDOM): output must be row_major per ISA
  if (exclude_ops.count("tile.random") == 0) {
    backend.RegisterOp("tile.random")
        .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
          return MakeRandomCodegenPTO("pto.trandom", op, codegen);
        })
        .set_output_layout(ir::TileLayout::row_major);
  }

  reg("tile.print", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakePrintCodegenPTO("pto.tprint", op, codegen);
  });

  // In-place accumulation ops (matmul_acc, gemv_acc): ptoas expects the
  // accumulator in ins() to be the same SSA value as outs().  InitMemRef
  // guarantees that the output shares the MemRef of the accumulator input
  // (via set_output_reuses_input), so we use the result buffer (dst) as the
  // accumulator operand instead of the IR-level input arg.
  //
  // The optional `init_cond` operand (args_[3]) makes the accumulator's initial
  // value conditional: where it holds, `dst` is overwritten with `lhs @ rhs`
  // rather than accumulated into.  The ISA carries this as one bit of the MAD's
  // Xt register, but the `pto.*` tile ops expose it only as the choice between
  // the accumulating and the non-accumulating op, so a runtime predicate lowers
  // to a branch over the two.  No phi is needed: both arms write `dst` in place.
  // `supports_init_cond` must track the op's own type deduction: `tile.gemv_acc`
  // still accepts exactly 3 arguments, so accepting a 4th here would only create
  // an unreachable branch behind a `CHECK` that fires earlier in deduction.
  auto make_acc_codegen = [](const std::string& pto_op, const std::string& init_pto_op,
                             bool supports_init_cond) {
    return [pto_op, init_pto_op, supports_init_cond](const ir::CallPtr& op,
                                                     codegen::CodegenBase& codegen_base) -> std::string {
      auto& codegen = AsPto(codegen_base);
      INTERNAL_CHECK_SPAN(op->args_.size() == 3 || (supports_init_cond && op->args_.size() == 4), op->span_)
          << pto_op << " requires 3 arguments (acc, lhs, rhs)"
          << (supports_init_cond ? " or 4 with init_cond" : "") << ", but got " << op->args_.size();

      std::string dst = codegen.GetCurrentResultTarget();
      std::string lhs = codegen.GetExprAsCode(op->args_[1]);
      std::string rhs = codegen.GetExprAsCode(op->args_[2]);
      std::string dst_type = codegen.GetCurrentResultTileBufTypeString();
      std::string lhs_type = codegen.GetExprTypeAnnotation(op->args_[1]);
      std::string rhs_type = codegen.GetExprTypeAnnotation(op->args_[2]);
      const std::string acc_phase = GemvAccPhaseAttr(op);

      // ins() carries the accumulator only on the accumulating form; the
      // initializing form reads lhs/rhs alone and writes dst from scratch.
      auto build = [&](bool initializing) {
        std::vector<std::string> operands = {lhs, rhs};
        std::vector<std::string> types = {lhs_type, rhs_type};
        if (!initializing) {
          operands.insert(operands.begin(), dst);
          types.insert(types.begin(), dst_type);
        }
        std::ostringstream inst;
        inst << (initializing ? init_pto_op : pto_op) << " ins(";
        for (size_t i = 0; i < operands.size(); ++i) {
          if (i > 0) inst << ", ";
          inst << operands[i];
        }
        // Type annotations must be all present or all absent: the `: t0, t1, ...`
        // clause is positional, so emitting a filtered subset would bind the
        // remaining types to the wrong operands. Mirrors make_mx_acc_codegen.
        const bool any_type_present =
            std::any_of(types.begin(), types.end(), [](const std::string& t) { return !t.empty(); });
        const bool all_types_present =
            std::all_of(types.begin(), types.end(), [](const std::string& t) { return !t.empty(); });
        INTERNAL_CHECK(!any_type_present || all_types_present)
            << "Internal error: " << (initializing ? init_pto_op : pto_op)
            << " operand type annotations must all be present or all absent, got a partial set";
        if (all_types_present) {
          inst << " : ";
          for (size_t i = 0; i < types.size(); ++i) {
            if (i > 0) inst << ", ";
            inst << types[i];
          }
        }
        inst << ") outs(" << dst;
        if (!dst_type.empty()) inst << " : " << dst_type;
        inst << ")" << acc_phase;
        return inst.str();
      };

      if (op->args_.size() == 3) {
        codegen.Emit(build(/*initializing=*/false));
        return "";
      }

      // A literal predicate picks one arm outright; only a runtime one branches.
      // Both spellings reach here: a DSL `init_cond=True/False` arrives as a
      // BOOL-typed ConstInt, while a predicate the arithmetic simplifier folded
      // (e.g. `ko == 0` after LowerPipelineLoops replicates the K-loop) arrives
      // as a ConstBool.  Missing either one leaves an `scf.if` on a compile-time
      // constant, doubling the emitted MADs.
      if (auto init_const = As<ir::ConstInt>(op->args_[3])) {
        codegen.Emit(build(/*initializing=*/init_const->value_ != 0));
        return "";
      }
      if (auto init_bool = As<ir::ConstBool>(op->args_[3])) {
        codegen.Emit(build(/*initializing=*/init_bool->value_));
        return "";
      }

      // Resolve the condition before opening the region so any instruction its
      // evaluation emits lands outside (and so dominates) both arms.
      std::string cond = codegen.GetExprAsCode(op->args_[3]);
      codegen.EmitStructural("scf.if " + cond + " {");
      codegen.IncreaseIndent();
      codegen.Emit(build(/*initializing=*/true));
      codegen.DecreaseIndent();
      codegen.EmitStructural("} else {");
      codegen.IncreaseIndent();
      codegen.Emit(build(/*initializing=*/false));
      codegen.DecreaseIndent();
      codegen.EmitStructural("}");
      return "";
    };
  };

  // MX in-place acc (5 operands): same c_in==dst contract as make_acc_codegen,
  // kept separate so the non-MX 3-arg helper stays untouched.
  auto make_mx_acc_codegen = [](const std::string& pto_op) {
    return [pto_op](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) -> std::string {
      auto& codegen = AsPto(codegen_base);
      INTERNAL_CHECK_SPAN(op->args_.size() == 5, op->span_)
          << pto_op << " requires 5 arguments: acc, lhs, lhs_scale, rhs, rhs_scale, but got "
          << op->args_.size();

      std::string dst = codegen.GetCurrentResultTarget();
      INTERNAL_CHECK(!dst.empty()) << "Internal error: " << pto_op
                                   << " Acc SSA must resolve (in-place c_in==dst)";
      std::string dst_type = codegen.GetCurrentResultTileBufTypeString();

      std::ostringstream acc_inst;
      acc_inst << pto_op << " ins(" << dst;
      std::vector<std::string> operand_types = {dst_type};
      for (size_t i = 1; i < op->args_.size(); ++i) {
        acc_inst << ", " << codegen.GetExprAsCode(op->args_[i]);
        operand_types.push_back(codegen.GetExprTypeAnnotation(op->args_[i]));
      }
      // Type annotations must be all present or all absent; a partial set
      // would desync the `: t0, t1, ...` clause from operand positions.
      const bool any_type_present = std::any_of(operand_types.begin(), operand_types.end(),
                                                [](const std::string& t) { return !t.empty(); });
      const bool all_types_present = std::all_of(operand_types.begin(), operand_types.end(),
                                                 [](const std::string& t) { return !t.empty(); });
      INTERNAL_CHECK(!any_type_present || all_types_present)
          << "Internal error: " << pto_op
          << " operand type annotations must all be present or all absent, got a partial set";
      if (all_types_present) {
        acc_inst << " : ";
        for (size_t i = 0; i < operand_types.size(); ++i) {
          if (i > 0) acc_inst << ", ";
          acc_inst << operand_types[i];
        }
      }
      acc_inst << ") outs(" << dst;
      if (!dst_type.empty()) acc_inst << " : " << dst_type;
      acc_inst << ")";
      codegen.Emit(acc_inst.str());
      return "";
    };
  };

  reg("tile.matmul_acc", make_acc_codegen("pto.tmatmul.acc", "pto.tmatmul", /*supports_init_cond=*/true));
  reg("tile.gemv", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeGemvCodegenPTO("pto.tgemv", 2, op, codegen);
  });
  reg("tile.gemv_acc", make_acc_codegen("pto.tgemv.acc", "pto.tgemv", /*supports_init_cond=*/false));
  reg("tile.matmul_mx_acc", make_mx_acc_codegen("pto.tmatmul.mx.acc"));
  reg("tile.gemv_bias", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeGemvCodegenPTO("pto.tgemv.bias", 3, op, codegen);
  });
}
}  // namespace backend
}  // namespace pypto
