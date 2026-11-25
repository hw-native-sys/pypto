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
 * @file matmul_mx.cpp
 * @brief MX block-scale matmul tile ops and the inferred scale-address binding op
 */

#include <any>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/phase.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

static bool IsSupportedMxDataType(const DataType& dtype) {
  return IsMxFp8DataType(dtype) || dtype.IsPackedFp4();
}

// Static-dim checks only: when either side is symbolic (non-ConstInt), the match
// is skipped. Full PTOAS alignment (M%16 / K%64 / N%32 / ceil(K/32) groups) is
// enforced solely for constant extents; dynamic shapes rely on the declared
// scale tile geometry and later PTOAS verification.
static void CheckDimMatch(const ExprPtr& expected, const ExprPtr& actual, const std::string& op_name,
                          const char* scale_name, const char* which, const char* axis) {
  auto e = As<ConstInt>(expected);
  auto a = As<ConstInt>(actual);
  if (e && a) {
    CHECK(e->value_ == a->value_) << "The operator " << op_name << " requires " << scale_name << " " << which
                                  << " " << axis << "=" << e->value_ << ", but got " << a->value_;
  }
}

/// Validate MX scale tile physical shape and valid_shape separately (PTOAS contract).
static void CheckMxScaleTile(const TileTypePtr& scale_type, const ExprPtr& phys_rows,
                             const ExprPtr& phys_cols, const ExprPtr& valid_rows, const ExprPtr& valid_cols,
                             const std::string& op_name, const char* scale_name) {
  CHECK(scale_type) << "The operator " << op_name << " requires " << scale_name << " to be a TileType";
  CHECK(scale_type->dtype_ == DataType::FP8E8M0)
      << "The operator " << op_name << " requires " << scale_name << " dtype FP8E8M0, but got "
      << scale_type->dtype_.ToString();
  CHECK(scale_type->shape_.size() == 2)
      << "The operator " << op_name << " requires " << scale_name << " to be 2D, but got "
      << scale_type->shape_.size() << " dimensions";

  CheckDimMatch(phys_rows, scale_type->shape_[0], op_name, scale_name, "physical", "rows");
  CheckDimMatch(phys_cols, scale_type->shape_[1], op_name, scale_name, "physical", "cols");

  const auto scale_valid = GetValidShape(scale_type);
  CheckDimMatch(valid_rows, scale_valid[0], op_name, scale_name, "valid", "rows");
  CheckDimMatch(valid_cols, scale_valid[1], op_name, scale_name, "valid", "cols");
}

/// MX scale groups along K: ceil(K / 32). Valid K need not be a multiple of 32.
static ExprPtr MxScaleKCeil(const ExprPtr& k_dim) {
  auto k_const = As<ConstInt>(k_dim);
  if (!k_const) return nullptr;
  CHECK(k_const->value_ > 0) << "MX scale K must be positive, got K=" << k_const->value_;
  const int64_t groups = (k_const->value_ + 31) / 32;
  return std::make_shared<ConstInt>(groups, DataType::INDEX, Span::unknown());
}

TypePtr DeduceTileMatMulMxType(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs,
                               const std::string& op_name, bool is_gemv = false) {
  const int acc_phase = GetKwargOr<int>(kwargs, "acc_phase", static_cast<int>(AccPhase::kUnspecified));
  CHECK(IsValidAccPhase(acc_phase))
      << "The operator " << op_name
      << " requires acc_phase to be AccPhase.Unspecified, AccPhase.Partial, or AccPhase.Final, but got "
      << acc_phase;
  CHECK(args.size() == 4) << "The operator " << op_name
                          << " requires exactly 4 arguments (lhs, lhs_scale, rhs, rhs_scale), but got "
                          << args.size();
  CHECK(args[1].get() != args[3].get())
      << "The operator " << op_name << " requires distinct lhs_scale and rhs_scale tiles";

  auto lhs_type = As<TileType>(args[0]->GetType());
  auto lhs_scale_type = As<TileType>(args[1]->GetType());
  auto rhs_type = As<TileType>(args[2]->GetType());
  auto rhs_scale_type = As<TileType>(args[3]->GetType());

  CHECK(lhs_type) << "The operator " << op_name << " requires lhs to be a TileType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(rhs_type) << "The operator " << op_name << " requires rhs to be a TileType, but got "
                  << args[2]->GetType()->TypeName();
  CHECK(lhs_type->shape_.size() == 2 && rhs_type->shape_.size() == 2)
      << "The operator " << op_name << " requires 2D lhs/rhs tiles";
  CHECK(IsSupportedMxDataType(lhs_type->dtype_))
      << "The operator " << op_name << " requires lhs dtype FP8E4M3FN, FP8E5M2, or packed FP4, but got "
      << lhs_type->dtype_.ToString();
  CHECK(IsSupportedMxDataType(rhs_type->dtype_))
      << "The operator " << op_name << " requires rhs dtype FP8E4M3FN, FP8E5M2, or packed FP4, but got "
      << rhs_type->dtype_.ToString();
  CHECK(lhs_type->dtype_.IsPackedFp4() == rhs_type->dtype_.IsPackedFp4())
      << "The operator " << op_name << " does not support mixed FP4 and FP8 operands";

  const auto lhs_shape = GetMatmulLogicalShape(lhs_type);
  const auto rhs_shape = GetMatmulLogicalShape(rhs_type);
  const ExprPtr& m_phys = lhs_shape[0];
  const ExprPtr& k_phys_lhs = lhs_shape[1];
  const ExprPtr& k_phys_rhs = rhs_shape[0];
  ExprPtr n_phys = rhs_shape[1];

  const auto lhs_valid = GetMatmulLogicalShape(lhs_type, true);
  const auto rhs_valid = GetMatmulLogicalShape(rhs_type, true);
  ExprPtr m_valid = lhs_valid[0];
  const ExprPtr& k_valid_lhs = lhs_valid[1];
  const ExprPtr& k_valid_rhs = rhs_valid[0];
  ExprPtr n_valid = rhs_valid[1];

  auto k_phys_lhs_c = As<ConstInt>(k_phys_lhs);
  auto k_phys_rhs_c = As<ConstInt>(k_phys_rhs);
  if (is_gemv) {
    const auto one = std::make_shared<ConstInt>(1, DataType::INDEX, m_phys->span_);
    auto is_one_row = [&one](const ExprPtr& extent) {
      if (auto c = As<ConstInt>(extent)) return c->value_ == 1;
      return ProveValidExtentEqual(extent, one) == ProofResult::kTrue;
    };
    CHECK(is_one_row(m_phys) && is_one_row(m_valid))
        << "The operator " << op_name << " requires lhs physical and valid M to be 1";
    // PTOAS cannot give a packed FP4 descriptor an instruction-local static
    // valid view with TRESHAPE. The assembler compares the physical extents
    // of these descriptors, so packed inputs need matching full valid extents.
    CHECK(!lhs_type->dtype_.IsPackedFp4() ||
          (ProveValidExtentEqual(k_valid_lhs, k_phys_lhs) == ProofResult::kTrue &&
           ProveValidExtentEqual(k_valid_rhs, k_phys_rhs) == ProofResult::kTrue &&
           ProveValidExtentEqual(n_valid, n_phys) == ProofResult::kTrue))
        << "The operator " << op_name << " requires full valid K and N for packed FP4";
    // A5 PTO-ISA's TGEMV_MX_IMPL takes K and N from the operand valid columns,
    // checks MMAD_MAX_SUPPORT_LENGTH (4095), and passes them to mad_mx.
    // These are active logical element counts, including the two values per
    // packed FP4 carrier expanded by GetMatmulLogicalShape. The padded physical
    // extents describe storage pitch and can be larger than the active region.
    constexpr int64_t kMaxValidExtent = 4095;
    for (const auto& dim : {k_valid_lhs, k_valid_rhs, n_valid}) {
      const auto extent = As<ConstInt>(dim);
      CHECK(!extent || (extent->value_ > 0 && extent->value_ <= kMaxValidExtent))
          << "The operator " << op_name << " requires valid K and N between 1 and " << kMaxValidExtent;
    }
    // The standard one-row Mat-to-Left path loads a complete 512-byte block.
    const int64_t k_align = lhs_type->dtype_.IsPackedFp4() ? 1024 : 512;
    CHECK(!k_phys_lhs_c || (k_phys_lhs_c->value_ > 0 && k_phys_lhs_c->value_ % k_align == 0))
        << "The operator " << op_name << " requires lhs physical K divisible by " << k_align
        << " for a 512-byte L0A vector block";
  } else {
    auto m_phys_c = As<ConstInt>(m_phys);
    auto n_phys_c = As<ConstInt>(n_phys);
    if (m_phys_c) {
      CHECK(m_phys_c->value_ > 0 && m_phys_c->value_ % 16 == 0)
          << "The operator " << op_name
          << " requires physical M divisible by 16 (ISA/PTOAS tmatmul_mx), but got M=" << m_phys_c->value_;
    }
    if (n_phys_c) {
      const int64_t n_align = rhs_type->dtype_.IsPackedFp4() ? 64 : 32;
      CHECK(n_phys_c->value_ > 0 && n_phys_c->value_ % n_align == 0)
          << "The operator " << op_name << " requires physical N divisible by " << n_align << " for "
          << rhs_type->dtype_.ToString()
          << " (ISA/PTOAS tmatmul_mx 32-byte row), but got N=" << n_phys_c->value_;
    }
    if (k_phys_lhs_c && k_phys_rhs_c) {
      CHECK(k_phys_lhs_c->value_ == k_phys_rhs_c->value_)
          << "The operator " << op_name
          << " requires matching physical K, but got lhs K=" << k_phys_lhs_c->value_
          << " and rhs K=" << k_phys_rhs_c->value_;
    }
    if (k_phys_lhs_c) {
      CHECK(k_phys_lhs_c->value_ > 0 && k_phys_lhs_c->value_ % 64 == 0)
          << "The operator " << op_name
          << " requires physical K divisible by 64 (ISA/PTOAS tmatmul_mx), but got lhs K="
          << k_phys_lhs_c->value_;
    }
    if (k_phys_rhs_c) {
      CHECK(k_phys_rhs_c->value_ > 0 && k_phys_rhs_c->value_ % 64 == 0)
          << "The operator " << op_name
          << " requires physical K divisible by 64 (ISA/PTOAS tmatmul_mx), but got rhs K="
          << k_phys_rhs_c->value_;
    }
  }

  auto k_valid_lhs_c = As<ConstInt>(k_valid_lhs);
  auto k_valid_rhs_c = As<ConstInt>(k_valid_rhs);
  if (k_valid_lhs_c && k_valid_rhs_c) {
    CHECK(k_valid_lhs_c->value_ == k_valid_rhs_c->value_)
        << "The operator " << op_name
        << " requires matching valid K, but got lhs valid K=" << k_valid_lhs_c->value_
        << " and rhs valid K=" << k_valid_rhs_c->value_;
  }
  if (k_valid_lhs_c) {
    CHECK(k_valid_lhs_c->value_ > 0) << "The operator " << op_name
                                     << " requires positive valid K, but got lhs valid K="
                                     << k_valid_lhs_c->value_;
  }
  if (k_valid_rhs_c) {
    CHECK(k_valid_rhs_c->value_ > 0) << "The operator " << op_name
                                     << " requires positive valid K, but got rhs valid K="
                                     << k_valid_rhs_c->value_;
  }
  // PTOAS v0.48: matmul_mx derives the scale physical group count from physical K
  // (ceil(K/32)), while tget_scale_addr derives it from valid K (ceil(validK/32)).
  // They must agree, otherwise no scale tile satisfies both verifiers — valid K
  // must round up to the same scale-group count as physical K.
  auto check_scale_group_agreement = [&](const ConstIntPtr& phys_c, const ConstIntPtr& valid_c,
                                         const char* side) {
    if (!phys_c || !valid_c) return;
    const int64_t phys_groups = (phys_c->value_ + 31) / 32;
    const int64_t valid_groups = (valid_c->value_ + 31) / 32;
    CHECK(valid_groups == phys_groups)
        << "The operator " << op_name << " requires " << side
        << " valid K to round up to the same scale-group count as physical K "
           "(ceil(validK/32) == ceil(physicalK/32)); otherwise the PTOAS matmul_mx and tget_scale_addr "
           "verifiers conflict. Got physical K="
        << phys_c->value_ << ", valid K=" << valid_c->value_;
  };
  if (!is_gemv) {
    check_scale_group_agreement(k_phys_lhs_c, k_valid_lhs_c, "lhs");
    check_scale_group_agreement(k_phys_rhs_c, k_valid_rhs_c, "rhs");
  }

  // GEMV scales cover the active K groups even if a staged operand has a
  // wider physical pitch (for example a one-row L0A load with padding).
  ExprPtr lhs_scale_k_phys = MxScaleKCeil(is_gemv ? k_valid_lhs : k_phys_lhs);
  ExprPtr lhs_scale_k_valid = MxScaleKCeil(k_valid_lhs);
  if (!lhs_scale_k_phys) {
    // Symbolic lhs K: ceil(K/32) is not computable, so the scale-group count is read off
    // the declared lhs_scale tile instead. Split so the rank message can name the actual
    // rank without dereferencing a null scale type.
    CHECK(lhs_scale_type) << "The operator " << op_name << " requires lhs_scale to be a TileType, but got "
                          << args[1]->GetType()->TypeName();
    CHECK(lhs_scale_type->shape_.size() == 2)
        << "The operator " << op_name << " requires lhs_scale to be 2D, but got "
        << lhs_scale_type->shape_.size() << " dimensions";
    lhs_scale_k_phys = lhs_scale_type->shape_[1];
  }
  if (!lhs_scale_k_valid) {
    lhs_scale_k_valid = lhs_scale_k_phys;
  }
  ExprPtr rhs_scale_k_phys = MxScaleKCeil(is_gemv ? k_valid_rhs : k_phys_rhs);
  ExprPtr rhs_scale_k_valid = MxScaleKCeil(k_valid_rhs);
  if (!rhs_scale_k_phys) {
    // Symbolic rhs K -- see the lhs_scale note above.
    CHECK(rhs_scale_type) << "The operator " << op_name << " requires rhs_scale to be a TileType, but got "
                          << args[3]->GetType()->TypeName();
    CHECK(rhs_scale_type->shape_.size() == 2)
        << "The operator " << op_name << " requires rhs_scale to be 2D, but got "
        << rhs_scale_type->shape_.size() << " dimensions";
    rhs_scale_k_phys = rhs_scale_type->shape_[0];
  }
  if (!rhs_scale_k_valid) {
    rhs_scale_k_valid = rhs_scale_k_phys;
  }
  CheckMxScaleTile(lhs_scale_type, m_phys, lhs_scale_k_phys, m_valid, lhs_scale_k_valid, op_name,
                   "lhs_scale");
  CheckMxScaleTile(rhs_scale_type, rhs_scale_k_phys, n_phys, rhs_scale_k_valid, n_valid, op_name,
                   "rhs_scale");

  // MAD writes a full 16-row L0C pitch even in its one-row GEMV mode.
  auto acc_rows =
      is_gemv ? std::make_shared<ConstInt>(kAccFractalRows, DataType::INDEX, m_phys->span_) : m_phys;
  std::vector<ExprPtr> output_shape = {acc_rows, n_phys};
  TileView tile_view;
  tile_view.blayout = TileLayout::col_major;
  tile_view.slayout = TileLayout::row_major;
  tile_view.fractal = 1024;
  tile_view.valid_shape = {m_valid, n_valid};
  if (!is_gemv) StampCompactForNarrowedAccRows(tile_view, output_shape);
  return std::make_shared<TileType>(output_shape, DataType::FP32, std::nullopt, tile_view);
}

TypePtr DeduceTileMatMulMxAccType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs,
                                  const std::string& op_name, bool is_gemv = false) {
  CHECK(args.size() == 5) << "The operator " << op_name
                          << " requires exactly 5 arguments (acc, lhs, lhs_scale, rhs, rhs_scale), but got "
                          << args.size();
  auto acc_type = As<TileType>(args[0]->GetType());
  CHECK(acc_type) << "The operator " << op_name << " requires acc to be a TileType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(acc_type->shape_.size() == 2) << "The operator " << op_name << " requires acc to be 2D, but got "
                                      << acc_type->shape_.size() << " dimensions";
  CHECK(acc_type->dtype_ == DataType::FP32)
      << "The operator " << op_name << " requires acc dtype FP32, but got " << acc_type->dtype_.ToString();
  const auto acc_valid = GetValidShape(acc_type);

  std::vector<ExprPtr> mx_args = {args[1], args[2], args[3], args[4]};
  auto out_type = DeduceTileMatMulMxType(mx_args, kwargs, op_name, is_gemv);
  auto out_tile = As<TileType>(out_type);
  auto m_acc = As<ConstInt>(acc_type->shape_[0]);
  auto n_acc = As<ConstInt>(acc_type->shape_[1]);
  auto m_out = As<ConstInt>(out_tile->shape_[0]);
  auto n_out = As<ConstInt>(out_tile->shape_[1]);
  if (m_acc && m_out) {
    CHECK(m_acc->value_ == m_out->value_)
        << "The operator " << op_name << " requires acc rows to match output M";
  }
  if (n_acc && n_out) {
    CHECK(n_acc->value_ == n_out->value_)
        << "The operator " << op_name << " requires acc cols to match output N";
  }

  const auto out_valid = GetValidShape(out_tile);
  if (GetKwargOr<int>(kwargs, "acc_phase", static_cast<int>(AccPhase::kUnspecified)) ==
      static_cast<int>(AccPhase::kFinal)) {
    // Every block the matching final store checks must have been set by MAD.
    CHECK(ProveValidExtentEqual(out_valid[0], acc_valid[0]) == ProofResult::kTrue &&
          ProveValidExtentEqual(out_valid[1], acc_valid[1]) == ProofResult::kTrue)
        << "The operator " << op_name
        << " with AccPhase.Final requires the product valid shape to match the accumulator valid shape";
  }
  CheckDimMatch(out_valid[0], acc_valid[0], op_name, "acc", "valid", "rows");
  CheckDimMatch(out_valid[1], acc_valid[1], op_name, "acc", "valid", "cols");
  return out_type;
}

TypePtr DeduceTileMatMulMxBiasType(const std::vector<ExprPtr>& args,
                                   const std::vector<std::pair<std::string, std::any>>& kwargs,
                                   const std::string& op_name, bool is_gemv = false) {
  CHECK(args.size() == 5) << "The operator " << op_name
                          << " requires exactly 5 arguments (lhs, lhs_scale, rhs, rhs_scale, bias), but got "
                          << args.size();
  auto bias_type = As<TileType>(args[4]->GetType());
  CHECK(bias_type) << "The operator " << op_name << " requires bias to be a TileType, but got "
                   << args[4]->GetType()->TypeName();
  CHECK(bias_type->shape_.size() == 2) << "The operator " << op_name << " requires bias to be 2D, but got "
                                       << bias_type->shape_.size() << " dimensions";
  CHECK(bias_type->dtype_ == DataType::FP32)
      << "The operator " << op_name << " requires bias dtype FP32, but got " << bias_type->dtype_.ToString();
  const auto bias_valid = GetValidShape(bias_type);

  std::vector<ExprPtr> mx_args = {args[0], args[1], args[2], args[3]};
  auto out_type = DeduceTileMatMulMxType(mx_args, kwargs, op_name, is_gemv);
  auto out_tile = As<TileType>(out_type);
  auto n_out = As<ConstInt>(out_tile->shape_[1]);
  auto bias_rows = As<ConstInt>(bias_type->shape_[0]);
  auto bias_cols = As<ConstInt>(bias_type->shape_[1]);
  if (bias_rows) {
    CHECK(bias_rows->value_ == 1) << "The operator " << op_name << " requires bias shape [1, N]";
  }
  if (n_out && bias_cols) {
    CHECK(n_out->value_ == bias_cols->value_)
        << "The operator " << op_name << " requires bias cols to match N";
  }

  const auto out_valid = GetValidShape(out_tile);
  auto bias_valid_rows = As<ConstInt>(bias_valid[0]);
  if (bias_valid_rows) {
    CHECK(bias_valid_rows->value_ == 1)
        << "The operator " << op_name << " requires bias valid rows to be 1, but got "
        << bias_valid_rows->value_;
  }
  CheckDimMatch(out_valid[1], bias_valid[1], op_name, "bias", "valid", "cols");
  return out_type;
}

REGISTER_OP("tile.matmul_mx")
    .set_op_category("TileOp")
    .set_description("MX block-scale matrix multiplication: C = matmul_mx(A, A_scale, B, B_scale)")
    .add_argument(
        "lhs",
        "Left-hand side tile (TileType, 2D, FP8 E4M3/E5M2 or packed FP4); physical M % 16 == 0, K % 64 == 0")
    .add_argument("lhs_scale", "Left scale tile (TileType, 2D, FP8E8M0); physical/valid [M, ceil(K/32)]")
    .add_argument(
        "rhs",
        "Right-hand side tile (TileType, 2D, FP8 E4M3/E5M2 or packed FP4); physical K % 64 == 0, N % 32 "
        "== 0 (64 for FP4)")
    .add_argument("rhs_scale", "Right scale tile (TileType, 2D, FP8E8M0); physical/valid [ceil(K/32), N]")
    .set_attr<int>("acc_phase")
    .set_input_memory(0, MemorySpace::Left)
    .set_input_memory(1, MemorySpace::LeftScale)
    .set_input_memory(2, MemorySpace::Right)
    .set_input_memory(3, MemorySpace::RightScale)
    .set_output_memory(MemorySpace::Acc)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileMatMulMxType(args, kwargs, "tile.matmul_mx");
    });

REGISTER_OP("tile.matmul_mx_acc")
    .set_op_category("TileOp")
    .set_description("MX block-scale matmul with accumulation: acc += matmul_mx(...)")
    .add_argument("acc", "Accumulator tile (TileType, 2D, FP32)")
    .add_argument("lhs", "Left-hand side tile (TileType, 2D, FP8 E4M3/E5M2 or packed FP4)")
    .add_argument("lhs_scale", "Left scale tile (TileType, 2D, FP8E8M0)")
    .add_argument("rhs", "Right-hand side tile (TileType, 2D, FP8 E4M3/E5M2 or packed FP4)")
    .add_argument("rhs_scale", "Right scale tile (TileType, 2D, FP8E8M0)")
    .set_attr<int>("acc_phase")
    .set_input_memory(0, MemorySpace::Acc)
    .set_input_memory(1, MemorySpace::Left)
    .set_input_memory(2, MemorySpace::LeftScale)
    .set_input_memory(3, MemorySpace::Right)
    .set_input_memory(4, MemorySpace::RightScale)
    .set_output_memory(MemorySpace::Acc)
    .set_output_reuses_input(0)
    // Accumulates into `acc`, same as tile.matmul_acc.
    .set_arg_effect(0, ArgEffect::ReadWrite)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileMatMulMxAccType(args, kwargs, "tile.matmul_mx_acc");
    });

REGISTER_OP("tile.matmul_mx_bias")
    .set_op_category("TileOp")
    .set_description("MX block-scale matmul with bias: C = matmul_mx(...) + bias")
    .add_argument("lhs", "Left-hand side tile (TileType, 2D, FP8 E4M3/E5M2 or packed FP4)")
    .add_argument("lhs_scale", "Left scale tile (TileType, 2D, FP8E8M0)")
    .add_argument("rhs", "Right-hand side tile (TileType, 2D, FP8 E4M3/E5M2 or packed FP4)")
    .add_argument("rhs_scale", "Right scale tile (TileType, 2D, FP8E8M0)")
    .add_argument("bias", "Bias tile (TileType, [1, N], FP32)")
    .set_attr<int>("acc_phase")
    .set_input_memory(0, MemorySpace::Left)
    .set_input_memory(1, MemorySpace::LeftScale)
    .set_input_memory(2, MemorySpace::Right)
    .set_input_memory(3, MemorySpace::RightScale)
    .set_input_memory(4, MemorySpace::Bias)
    .set_output_memory(MemorySpace::Acc)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileMatMulMxBiasType(args, kwargs, "tile.matmul_mx_bias");
    });

REGISTER_OP("tile.gemv_mx")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("MX block-scale matrix-vector multiplication: C = gemv_mx(A, A_scale, B, B_scale)")
    .add_argument("lhs", "Left row vector (TileType, [1, K], FP8 E4M3/E5M2 or packed FP4)")
    .add_argument("lhs_scale", "Left scale tile (TileType, 2D, FP8E8M0); physical/valid [M, ceil(K/32)]")
    .add_argument("rhs", "Right-hand side tile (TileType, [K, N], FP8 E4M3/E5M2 or packed FP4)")
    .add_argument("rhs_scale", "Right scale tile (TileType, 2D, FP8E8M0); physical/valid [ceil(K/32), N]")
    .set_attr<int>("acc_phase")
    .set_input_memory(0, MemorySpace::Left)
    .set_input_memory(1, MemorySpace::LeftScale)
    .set_input_memory(2, MemorySpace::Right)
    .set_input_memory(3, MemorySpace::RightScale)
    .set_output_memory(MemorySpace::Acc)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileMatMulMxType(args, kwargs, "tile.gemv_mx", /*is_gemv=*/true);
    });

REGISTER_OP("tile.gemv_mx_acc")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("MX block-scale GEMV with accumulation: acc += gemv_mx(...)")
    .add_argument("acc", "Accumulator tile (TileType, 2D, FP32)")
    .add_argument("lhs", "Left-hand side tile (TileType, 2D, FP8 E4M3/E5M2 or packed FP4)")
    .add_argument("lhs_scale", "Left scale tile (TileType, 2D, FP8E8M0)")
    .add_argument("rhs", "Right-hand side tile (TileType, 2D, FP8 E4M3/E5M2 or packed FP4)")
    .add_argument("rhs_scale", "Right scale tile (TileType, 2D, FP8E8M0)")
    .set_attr<int>("acc_phase")
    .set_input_memory(0, MemorySpace::Acc)
    .set_input_memory(1, MemorySpace::Left)
    .set_input_memory(2, MemorySpace::LeftScale)
    .set_input_memory(3, MemorySpace::Right)
    .set_input_memory(4, MemorySpace::RightScale)
    .set_output_memory(MemorySpace::Acc)
    .set_output_reuses_input(0)
    // Accumulates into `acc`, same as tile.matmul_acc.
    .set_arg_effect(0, ArgEffect::ReadWrite)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileMatMulMxAccType(args, kwargs, "tile.gemv_mx_acc", /*is_gemv=*/true);
    });

REGISTER_OP("tile.gemv_mx_bias")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("MX block-scale GEMV with bias: C = gemv_mx(...) + bias")
    .add_argument("lhs", "Left-hand side tile (TileType, 2D, FP8 E4M3/E5M2 or packed FP4)")
    .add_argument("lhs_scale", "Left scale tile (TileType, 2D, FP8E8M0)")
    .add_argument("rhs", "Right-hand side tile (TileType, 2D, FP8 E4M3/E5M2 or packed FP4)")
    .add_argument("rhs_scale", "Right scale tile (TileType, 2D, FP8E8M0)")
    .add_argument("bias", "Bias tile (TileType, [1, N], FP32)")
    .set_attr<int>("acc_phase")
    .set_input_memory(0, MemorySpace::Left)
    .set_input_memory(1, MemorySpace::LeftScale)
    .set_input_memory(2, MemorySpace::Right)
    .set_input_memory(3, MemorySpace::RightScale)
    .set_input_memory(4, MemorySpace::Bias)
    .set_output_memory(MemorySpace::Acc)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileMatMulMxBiasType(args, kwargs, "tile.gemv_mx_bias", /*is_gemv=*/true);
    });

static void CheckScaleAddressTypes(const TileTypePtr& dst_type, const TileTypePtr& src_type,
                                   const std::string& op_name) {
  CHECK(dst_type) << "The operator " << op_name << " requires dst_scale to be a TileType";
  CHECK(src_type) << "The operator " << op_name << " requires src to be a TileType";
  // PTOAS treats ui8+loc=scaling as Fixpipe Scaling; LeftScale/RightScale must already
  // be FP8E8M0 (UINT8 is only allowed on the Mat side before Mat→Scale tmov).
  CHECK(dst_type->dtype_ == DataType::FP8E8M0)
      << "The operator " << op_name << " requires dst_scale dtype FP8E8M0, but got "
      << dst_type->dtype_.ToString();
  CHECK(IsSupportedMxDataType(src_type->dtype_))
      << "The operator " << op_name << " requires src dtype FP8E4M3FN, FP8E5M2, or packed FP4, but got "
      << src_type->dtype_.ToString();
  CHECK(dst_type->shape_.size() == 2 && src_type->shape_.size() == 2)
      << "The operator " << op_name << " requires 2D dst_scale and src tiles";
  const auto src_shape = GetMatmulLogicalShape(src_type);
  const auto src_valid = GetMatmulLogicalShape(src_type, true);

  const auto dst_space = dst_type->GetMemorySpace();
  const auto src_space = src_type->GetMemorySpace();
  CHECK(dst_space.has_value() && src_space.has_value())
      << "The operator " << op_name << " requires resolved dst_scale and src memory spaces";
  const bool is_left = *dst_space == MemorySpace::LeftScale && *src_space == MemorySpace::Left;
  const bool is_right = *dst_space == MemorySpace::RightScale && *src_space == MemorySpace::Right;
  CHECK(is_left || is_right) << "The operator " << op_name
                             << " requires LeftScale↔Left or RightScale↔Right pairing, but got dst="
                             << MemorySpaceToString(*dst_space) << " src=" << MemorySpaceToString(*src_space);

  // Both the physical and valid scale K axes cover the source's valid K.
  // A symbolic valid K remains unchecked here and is verified by PTOAS.
  if (is_left) {
    const ExprPtr scale_k = MxScaleKCeil(src_valid[1]);
    CheckMxScaleTile(dst_type, src_shape[0], scale_k, src_valid[0], scale_k, op_name, "dst_scale");
  } else {
    const ExprPtr scale_k = MxScaleKCeil(src_valid[0]);
    CheckMxScaleTile(dst_type, scale_k, src_shape[1], scale_k, src_valid[1], op_name, "dst_scale");
  }
}

REGISTER_OP("tile.tget_scale_addr")
    .set_op_category("TileOp")
    .set_description(
        "Bind an inferred MX scale-tile address from a Left/Right data tile (A5): "
        "dst_addr = src_addr >> SHIFT_MX_ADDR. Maps to pto.tget_scale_addr.")
    .add_argument("dst_scale", "Resolved LeftScale/RightScale destination tile (FP8E8M0 only)")
    .add_argument(
        "src", "Resolved Left/Right MX data tile (FP8E4M3FN, FP8E5M2, or packed FP4) whose address is scaled")
    .set_output_memory_inherit_input()
    .set_output_reuses_input(0)
    // Binds a derived address into the shared physical scale buffer `dst_scale`
    // names. Only an unchanged scale/data pair with no intervening invalidation
    // can reuse a binding; declaring the write keeps aliasing hazards visible
    // to analyses that order accesses.
    .set_arg_effect(0, ArgEffect::Write)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      CHECK(args.size() == 2) << "tile.tget_scale_addr requires dst_scale and src";
      auto dst_type = As<TileType>(args[0]->GetType());
      CheckScaleAddressTypes(dst_type, As<TileType>(args[1]->GetType()), "tile.tget_scale_addr");
      return std::make_shared<TileType>(dst_type->shape_, dst_type->dtype_, std::nullopt,
                                        dst_type->tile_view_, dst_type->memory_space_);
    });

REGISTER_OP("tile.bind_scale_addr")
    .set_op_category("TileOp")
    .set_description("Create an MX scale destination at the address derived from its payload tile")
    .add_argument("payload", "Resolved Left or Right MX payload tile")
    .add_argument("shape", "Physical shape of the scale destination")
    .add_argument("valid_shape", "Valid shape of the scale destination")
    .set_attr<MemorySpace>("target_memory")
    .set_output_memory_from_kwarg("target_memory")
    // The binding changes a tile handle without accessing payload or scale bytes.
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 3) << "tile.bind_scale_addr requires payload, shape and valid_shape";
      auto shape = As<MakeTuple>(args[1]);
      auto valid = As<MakeTuple>(args[2]);
      CHECK(shape && valid && shape->elements_.size() == 2 && valid->elements_.size() == 2)
          << "tile.bind_scale_addr requires 2D shape and valid_shape tuples";
      for (const auto& dim : shape->elements_) {
        auto extent = As<ConstInt>(dim);
        CHECK(extent && extent->value_ > 0) << "tile.bind_scale_addr requires positive static dimensions";
      }
      const auto space = GetKwargOr(kwargs, "target_memory", MemorySpace::Mat);
      CHECK(space == MemorySpace::LeftScale || space == MemorySpace::RightScale)
          << "tile.bind_scale_addr requires a LeftScale or RightScale destination";
      TileView view;
      view.valid_shape = valid->elements_;
      tile_view_semantics::SetTileLayout(view,
                                         tile_view_semantics::GetImplicitTileLayout(shape->elements_, space));
      auto result =
          std::make_shared<TileType>(shape->elements_, DataType::FP8E8M0, std::nullopt, view, space);
      CheckScaleAddressTypes(result, As<TileType>(args[0]->GetType()), "tile.bind_scale_addr");
      return result;
    });

}  // namespace ir
}  // namespace pypto
