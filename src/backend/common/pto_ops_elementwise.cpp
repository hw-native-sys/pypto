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
      "tile.div",
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
      "tile.log",
      "tile.sqrt",
      "tile.recip",
      "tile.not",
      "tile.relu",
      // Tile x Scalar ops
      "tile.adds",
      "tile.muls",
      "tile.divs",
      "tile.fmods",
      "tile.maximums",
      "tile.lrelu",
      // Ternary scalar ops (Tile x Scalar x Tile)
      "tile.addsc",
      "tile.subsc",
  };
  return kRowMajorOps.count(op_name) > 0;
}

// Helper function for N-ary operations (unary, binary, ternary, etc.)
static std::string MakeNaryCodegenPTO(const std::string& pto_op_name, size_t arity, const CallPtr& op,
                                      codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CheckArity(op, pto_op_name, arity);
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
  CHECK(op->args_.size() == 4) << "tile.transpose requires 4 arguments (src, axis0, axis1, tmp), got "
                               << op->args_.size();

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
  CHECK(op->args_.size() == 2) << spec.ir_name << " requires 2 arguments" << spec.arg_desc << ", got "
                               << op->args_.size();
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
  CHECK(op->args_.size() == 2) << "Operation:[" << pto_op_name
                               << "] requires 2 arguments (start, shape), but got " << op->args_.size();
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
  CHECK(op->args_.size() == 7 || op->args_.size() == 8)
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
  CHECK(op->args_.size() == 1) << "Operation:" << pto_op_name << "] requires 1 argument, but got "
                               << op->args_.size();
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  codegen.Emit(pto_op_name + " ins(" + src + " | !pto.partition_tensor_view<MxNxdtype>)");
  return "";
}

struct SimpleOpEntry {
  const char* op_name;
  const char* pto_op_name;
  size_t arity;
};

// clang-format off
static const SimpleOpEntry kSimpleOps[] = {
    // Memory operations
    {"tile.mgather",         "pto.tmgather",         2},
    // Tile x Tile arithmetic operations
    {"tile.add",             "pto.tadd",             2},
    {"tile.sub",             "pto.tsub",             2},
    {"tile.mul",             "pto.tmul",             2},
    {"tile.div",             "pto.tdiv",             2},
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
    {"tile.prelu",           "pto.tprelu",           2},
    // Unary operations
    {"tile.abs",             "pto.tabs",             1},
    {"tile.exp",             "pto.texp",             1},
    {"tile.log",             "pto.tlog",             1},
    {"tile.sqrt",            "pto.tsqrt",            1},
    // tile.rsqrt is registered with a custom codegen handler below (supports 1 or 2 args).
    {"tile.recip",           "pto.trecip",           1},
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
    {"tile.ands",            "pto.tands",            2},
    {"tile.ors",             "pto.tors",             2},
    {"tile.xors",            "pto.txors",            3},  // src0, scalar, tmp
    {"tile.shls",            "pto.tshls",            2},
    {"tile.shrs",            "pto.tshrs",            2},
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
    {"tile.row_expand_add",  "pto.trowexpandadd",    2},
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
    {"tile.matmul_mx_acc",   "pto.tmatmul.mx.acc",   5},
    {"tile.matmul_mx_bias",  "pto.tmatmul.mx.bias",  5},
    // tile.matmul_acc and tile.gemv_acc have custom codegen (in-place accumulation)
    {"tile.matmul_bias",     "pto.tmatmul.bias",     3},
    {"tile.gemv",            "pto.tgemv",            2},
    // tile.gemv_acc has custom codegen (in-place accumulation)
    {"tile.gemv_bias",       "pto.tgemv.bias",       3},
    // Data movement/layout operations
    {"tile.concat",          "pto.tconcat",          2},
    // tile.move has custom codegen (no-op elision for same-space same-address moves)
    {"tile.move_fp",         "pto.tmov.fp",          2},
    // tile.transpose has custom codegen (MakeTileTransposeCodegenPTO): pto.ttrans needs
    // ins(%src, %tmp : tile_type, tile_type) where %tmp is a scratch workspace tile, NOT
    // the axis-index integers that tile.transpose(src, axis0, axis1) carries in the IR.
    // tile.extract has custom codegen (see reg("tile.extract") below): the IR carries the
    // shape tuple as args_[3] purely for type deduction, so the generic N-ary lowering
    // would emit the tuple as a PTO operand — not what pto.textract expects.
    // Gather/scatter operations
    {"tile.gather",          "pto.tgather",          3},
    {"tile.gatherb",         "pto.tgatherb",         2},
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
    auto reg_entry = backend.RegisterOp(entry.op_name);
    reg_entry.f_codegen([pto_op, arity](const CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeNaryCodegenPTO(pto_op, arity, op, codegen);
    });
    if (RequiresRowMajorLayout(entry.op_name)) {
      for (size_t i = 0; i < arity; ++i) {
        reg_entry.set_input_layout(i, ir::TileLayout::row_major);
      }
      reg_entry.set_output_layout(ir::TileLayout::row_major);
    }
  }

  // Register ops with custom codegen logic
  auto reg = [&](const char* op_name, BackendCodegenFunc fn) {
    if (exclude_ops.count(op_name) > 0) return;
    backend.RegisterOp(op_name).f_codegen(std::move(fn));
  };

  // tile.move → pto.tmov with no-op elision.
  // When MemoryReuse inserts a tile.move between two MemRefs that end up at the
  // same physical address after AllocateMemoryAddr (e.g. acc→acc at the same Acc
  // offset), the move is a no-op. Elide it to avoid emitting pto.tmov with
  // unsupported same-space address pairs (fixes #1310).
  reg("tile.move", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    auto& codegen = AsPto(codegen_base);
    CHECK(op->args_.size() == 1) << "tile.move requires 1 argument, got " << op->args_.size();

    auto src_var = AsVarLike(op->args_[0]);
    auto dst_var = codegen.GetCurrentResultVar();
    if (src_var && dst_var) {
      auto src_tile = As<ir::TileType>(src_var->GetType());
      auto dst_tile = As<ir::TileType>(dst_var->GetType());
      if (src_tile && dst_tile && src_tile->memref_.has_value() && dst_tile->memref_.has_value()) {
        auto src_space = src_tile->GetMemorySpace();
        auto dst_space = dst_tile->GetMemorySpace();
        if (src_space.has_value() && dst_space.has_value() && *src_space == *dst_space) {
          auto src_offset = As<ir::ConstInt>((*src_tile->memref_)->byte_offset_);
          auto dst_offset = As<ir::ConstInt>((*dst_tile->memref_)->byte_offset_);
          if (src_offset && dst_offset && src_offset->value_ == dst_offset->value_) {
            // Alias the destination to the source SSA value so downstream
            // references use the source's defined buffer, not the destination's
            // alloc_tile (which would be unwritten after eliding the tmov).
            codegen.SetCurrentResultBuf(codegen.GetExprAsCode(op->args_[0]));
            return std::string("");  // no-op: same space, same address
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
          CHECK(arity == 1 || arity == 2) << "tile.rsqrt requires 1 or 2 arguments, but got " << arity;
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
          CHECK(op->args_.size() == 1 || op->args_.size() == 2)
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

  reg("tile.ci", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeCiCodegenPTO("pto.tci", op, codegen);
  });

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
  auto make_acc_codegen = [](const std::string& pto_op) {
    return [pto_op](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) -> std::string {
      auto& codegen = AsPto(codegen_base);
      CHECK(op->args_.size() == 3) << pto_op << " requires 3 arguments: acc, lhs, rhs";

      std::string dst = codegen.GetCurrentResultTarget();
      std::string lhs = codegen.GetExprAsCode(op->args_[1]);
      std::string rhs = codegen.GetExprAsCode(op->args_[2]);
      std::string dst_type = codegen.GetCurrentResultTileBufTypeString();
      std::string lhs_type = codegen.GetExprTypeAnnotation(op->args_[1]);
      std::string rhs_type = codegen.GetExprTypeAnnotation(op->args_[2]);

      std::ostringstream acc_inst;
      acc_inst << pto_op << " ins(" << dst << ", " << lhs << ", " << rhs;
      std::vector<std::string> ins_type_parts;
      for (const auto& t : {dst_type, lhs_type, rhs_type}) {
        if (!t.empty()) ins_type_parts.push_back(t);
      }
      if (!ins_type_parts.empty()) {
        acc_inst << " : ";
        for (size_t i = 0; i < ins_type_parts.size(); ++i) {
          if (i > 0) acc_inst << ", ";
          acc_inst << ins_type_parts[i];
        }
      }
      acc_inst << ") outs(" << dst;
      if (!dst_type.empty()) acc_inst << " : " << dst_type;
      acc_inst << ")";
      codegen.Emit(acc_inst.str());
      return "";
    };
  };

  reg("tile.matmul_acc", make_acc_codegen("pto.tmatmul.acc"));
  reg("tile.gemv_acc", make_acc_codegen("pto.tgemv.acc"));
}
}  // namespace backend
}  // namespace pypto
