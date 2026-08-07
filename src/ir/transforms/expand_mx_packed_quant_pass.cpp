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
 * @file expand_mx_packed_quant_pass.cpp
 * @brief Expand ``tile.tquant_mx(..., layout=MX_A_ZZ|MX_B_NN)`` into per-box flat
 *        quant + continuous ZZ/NN scale packing (B also INT8-transposes to [K,N]).
 *
 * Preferred lowering matches the onboard-proven rearrange recipe: resolve the
 * source ``tile.load``, then per-box ``tile.load`` + flat ``tquant_mx`` +
 * ``tile.store`` into the consumer GM buffers. Vec ``slice``/``assemble`` packing
 * is only a fallback when stores are not visible and is less reliable on A5.
 */

#include <any>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace pass {
namespace {

constexpr int64_t kMxPackTileM = 16;
constexpr int64_t kMxPackTileK = 64;
constexpr int64_t kMxPackGroup = 32;
constexpr int64_t kMxPackBoxRows = kMxPackTileM * (kMxPackTileK / kMxPackGroup);  // 32
constexpr int64_t kMxPackReuseChunkBoxes = 16;

std::optional<TensorLayout> GetMxPackLayout(const CallPtr& call) {
  if (!call || !IsOp(call, "tile.tquant_mx")) {
    return std::nullopt;
  }
  for (const auto& [key, value] : call->kwargs_) {
    if (key != "layout") {
      continue;
    }
    auto layout = AnyCast<TensorLayout>(value, "kwarg key: layout");
    CHECK(layout == TensorLayout::MX_A_ZZ || layout == TensorLayout::MX_B_NN)
        << "tile.tquant_mx layout must be MX_A_ZZ or MX_B_NN (ND/None are not allowed), got "
        << TensorLayoutToString(layout);
    return layout;
  }
  return std::nullopt;
}

ExprPtr MakeIndex(int64_t value, const Span& span) {
  return std::make_shared<ConstInt>(value, DataType::INDEX, span);
}

ExprPtr MakeShape2(int64_t d0, int64_t d1, const Span& span) {
  return std::make_shared<MakeTuple>(std::vector<ExprPtr>{MakeIndex(d0, span), MakeIndex(d1, span)}, span);
}

const Var* GetVarIdentity(const ExprPtr& expr) {
  if (auto var = As<Var>(expr)) return var.get();
  if (auto iter_arg = As<IterArg>(expr)) return iter_arg.get();
  return nullptr;
}

bool ConstOffset2(const ExprPtr& offsets, int64_t* r, int64_t* c) {
  auto tup = As<MakeTuple>(offsets);
  if (!tup || tup->elements_.size() != 2) return false;
  auto a = As<ConstInt>(tup->elements_[0]);
  auto b = As<ConstInt>(tup->elements_[1]);
  if (!a || !b) return false;
  *r = a->value_;
  *c = b->value_;
  return true;
}

struct ResolvedTileLoad {
  ExprPtr tensor;
  int64_t row0 = 0;
  int64_t col0 = 0;
  std::vector<std::pair<std::string, std::any>> kwargs;
};

std::optional<ResolvedTileLoad> ResolveTileLoad(ExprPtr expr,
                                                const std::unordered_map<const Var*, ExprPtr>& defs) {
  for (int depth = 0; depth < 8 && expr; ++depth) {
    if (auto call = As<Call>(expr); call && IsOp(call, "tile.load")) {
      CHECK(call->args_.size() >= 4) << "tile.load expects tensor, offsets, shapes, valid_shape";
      int64_t r = 0;
      int64_t c = 0;
      // Dynamic base offsets cannot be folded into the per-box reloads.  They
      // are still valid inputs: leave the aggregate load in place and let the
      // caller use the slice/assemble fallback.
      if (!ConstOffset2(call->args_[1], &r, &c)) return std::nullopt;
      ResolvedTileLoad out;
      out.tensor = call->args_[0];
      out.row0 = r;
      out.col0 = c;
      out.kwargs = call->kwargs_;
      return out;
    }
    if (const Var* v = GetVarIdentity(expr)) {
      auto it = defs.find(v);
      if (it == defs.end()) return std::nullopt;
      expr = it->second;
      continue;
    }
    return std::nullopt;
  }
  return std::nullopt;
}

class ExpandBuilder {
 public:
  ExpandBuilder(std::string base_name, std::size_t& temp_counter)
      : base_name_(std::move(base_name)), temp_counter_(temp_counter) {}

  ExprPtr Bind(const std::string& qualifier, const ExprPtr& expr, const Span& span) {
    auto var = std::make_shared<Var>(MakeTempName(qualifier), expr->GetType(), span);
    stmts_.push_back(std::make_shared<AssignStmt>(var, expr, span));
    return var;
  }

  void DrainChunk(const std::vector<ExprPtr>& tiles, const Span& span,
                  const std::string& qualifier = "chunk_keep") {
    for (const auto& tile : tiles) {
      Bind(qualifier, tile, span);
    }
    auto barrier = OpRegistry::GetInstance().Create("system.bar_all", {}, span);
    stmts_.push_back(std::make_shared<EvalStmt>(barrier, span));
  }

  std::vector<StmtPtr> TakeStmts() { return std::move(stmts_); }

 private:
  std::string MakeTempName(const std::string& qualifier) {
    return auto_name::BuildName(auto_name::GetBaseName(base_name_), qualifier, "tmp",
                                static_cast<int>(temp_counter_++));
  }

  std::string base_name_;
  std::size_t& temp_counter_;
  std::vector<StmtPtr> stmts_;
};

ExprPtr LoadBox(const ResolvedTileLoad& ld, int64_t row, int64_t col, ExpandBuilder& b, const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  auto shape = MakeShape2(kMxPackTileM, kMxPackTileK, span);
  return b.Bind(
      "box",
      reg.Create("tile.load", {ld.tensor, MakeShape2(ld.row0 + row, ld.col0 + col, span), shape, shape},
                 ld.kwargs, span),
      span);
}

ExprPtr SliceBox(const ExprPtr& src, int64_t row, int64_t col, ExpandBuilder& b, const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  return b.Bind(
      "box",
      reg.Create("tile.slice",
                 {src, MakeShape2(kMxPackTileM, kMxPackTileK, span), MakeShape2(row, col, span)}, {}, span),
      span);
}

struct BoxQuant {
  ExprPtr q_mk;  // [16, 64] FP8
  ExprPtr s;     // [1, 32] E8M0
};

BoxQuant QuantizeBox(const ExprPtr& box, ExpandBuilder& b, const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  auto box32 = b.Bind(
      "box32", reg.Create("tile.reshape", {box, MakeShape2(kMxPackBoxRows, kMxPackGroup, span)}, {}, span),
      span);
  std::string mode = "mxfp8_e4m3";
  auto pair = b.Bind("tq", reg.Create("tile.tquant_mx", {box32}, {{"mode", mode}}, span), span);
  auto q_box = b.Bind("qb", std::make_shared<TupleGetItemExpr>(pair, 0, span), span);
  auto s_box = b.Bind("sb", std::make_shared<TupleGetItemExpr>(pair, 1, span), span);
  auto q_mk = b.Bind(
      "qmk", reg.Create("tile.reshape", {q_box, MakeShape2(kMxPackTileM, kMxPackTileK, span)}, {}, span),
      span);
  return {q_mk, s_box};
}

ExprPtr CreateMxScaleU8Buffer(ExpandBuilder& b, int64_t groups, const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  auto groups_dim = MakeIndex(groups, span);
  auto one = MakeIndex(1, span);
  // Plain ND UINT8 (no MX fractal): assemble packs bytes like consecutive GM stores.
  TileView scale_u8_view;
  scale_u8_view.valid_shape = {one, groups_dim};
  scale_u8_view.blayout = TileLayout::row_major;
  scale_u8_view.slayout = TileLayout::none_box;
  auto scale_u8_type = std::make_shared<TileType>(std::vector<ExprPtr>{one, groups_dim}, DataType::UINT8,
                                                  std::nullopt, scale_u8_view, MemorySpace::Vec);
  auto scale_shape = MakeShape2(1, groups, span);
  auto raw_create = As<Call>(reg.Create(
      "tile.create", {scale_shape}, {{"dtype", DataType::UINT8}, {"target_memory", MemorySpace::Vec}}, span));
  INTERNAL_CHECK_SPAN(raw_create, span) << "Internal error: MX scale tile.create did not produce a Call";
  auto typed = std::make_shared<Call>(raw_create->op_, raw_create->args_, raw_create->kwargs_,
                                      raw_create->attrs_, scale_u8_type, span);
  return b.Bind("s_u8", typed, span);
}

ExprPtr ReinterpretMxScaleBuffer(const ExprPtr& scale_u8, int64_t groups, ExpandBuilder& b,
                                 const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  auto one = MakeIndex(1, span);
  auto groups_dim = MakeIndex(groups, span);
  auto raw = As<Call>(reg.Create("tile.reinterpret_view", {scale_u8}, {{"dtype", DataType::FP8E8M0}}, span));
  INTERNAL_CHECK_SPAN(raw, span) << "Internal error: MX scale reinterpret_view did not produce a Call";

  TileView scale_view;
  scale_view.valid_shape = {one, groups_dim};
  scale_view.blayout = TileLayout::row_major;
  scale_view.slayout = TileLayout::none_box;
  scale_view.fractal = tile_view_semantics::kMXScaleFractal;
  auto scale_type = std::make_shared<TileType>(std::vector<ExprPtr>{one, groups_dim}, DataType::FP8E8M0,
                                               std::nullopt, scale_view, MemorySpace::Vec);
  auto typed = std::make_shared<Call>(raw->op_, raw->args_, raw->kwargs_, raw->attrs_, scale_type, span);
  return b.Bind("s", typed, span);
}

ExprPtr CreatePlainU8Buffer(ExpandBuilder& b, int64_t rows, int64_t cols, const std::string& name,
                            const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  auto r = MakeIndex(rows, span);
  auto c = MakeIndex(cols, span);
  TileView view;
  view.valid_shape = {r, c};
  view.blayout = TileLayout::row_major;
  view.slayout = TileLayout::none_box;
  auto type = std::make_shared<TileType>(std::vector<ExprPtr>{r, c}, DataType::UINT8, std::nullopt, view,
                                         MemorySpace::Vec);
  auto shape = MakeShape2(rows, cols, span);
  auto raw = As<Call>(reg.Create("tile.create", {shape},
                                 {{"dtype", DataType::UINT8}, {"target_memory", MemorySpace::Vec}}, span));
  INTERNAL_CHECK_SPAN(raw, span) << "Internal error: quant tile.create did not produce a Call";
  auto typed = std::make_shared<Call>(raw->op_, raw->args_, raw->kwargs_, raw->attrs_, type, span);
  return b.Bind(name, typed, span);
}

ExprPtr StoreTile(const ExprPtr& tile, int64_t row, int64_t col, const ExprPtr& tensor, ExpandBuilder& b,
                  const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  return b.Bind("st", reg.Create("tile.store", {tile, MakeShape2(row, col, span), tensor}, {}, span), span);
}

struct StoreSite {
  ExprPtr tensor;
  int64_t row0 = 0;
  int64_t col0 = 0;
  const Var* result_var = nullptr;  // LHS of the store assign
  const Var* stored_tile_var = nullptr;
  const SeqStmts* parent_seq = nullptr;
  size_t stmt_index = 0;
};

struct MxPackSite {
  const Var* quant_var = nullptr;
  TensorLayout layout = TensorLayout::ND;
  ExprPtr src;
  const SeqStmts* parent_seq = nullptr;
  size_t stmt_index = 0;
  std::optional<StoreSite> q_store;
  std::optional<StoreSite> s_store;
  bool outputs_are_store_only = false;
};

struct ExpansionResult {
  ExprPtr quant;  // tile result (loaded back or assembled)
  ExprPtr scale;
  ExprPtr q_store_result;  // final stored tensor (if store-fused)
  ExprPtr s_store_result;
};

struct FusedStoreReplacement {
  ExprPtr tensor_result;
};

FusedStoreReplacement MakeFusedStoreReplacement(const ExprPtr& tensor_result) { return {tensor_result}; }

ExpansionResult ExpandMxAZzStoreFused(const ResolvedTileLoad& ld, int64_t m, int64_t k,
                                      const StoreSite& q_store, const StoreSite& s_store, ExpandBuilder& b,
                                      const Span& span) {
  const int64_t mb = m / kMxPackTileM;
  const int64_t kb = k / kMxPackTileK;
  ExprPtr q_t = q_store.tensor;
  ExprPtr s_t = s_store.tensor;
  std::vector<ExprPtr> chunk_tiles;
  const int64_t box_count = mb * kb;
  int64_t boxes_done = 0;
  for (int64_t mi = 0; mi < mb; ++mi) {
    for (int64_t ki = 0; ki < kb; ++ki) {
      auto box = LoadBox(ld, mi * kMxPackTileM, ki * kMxPackTileK, b, span);
      auto qq = QuantizeBox(box, b, span);
      chunk_tiles.insert(chunk_tiles.end(), {box, qq.q_mk, qq.s});
      q_t = StoreTile(qq.q_mk, q_store.row0 + mi * kMxPackTileM, q_store.col0 + ki * kMxPackTileK, q_t, b,
                      span);
      s_t = StoreTile(qq.s, s_store.row0, s_store.col0 + (mi * kb + ki) * kMxPackBoxRows, s_t, b, span);
      ++boxes_done;
      if (boxes_done % kMxPackReuseChunkBoxes == 0 || boxes_done == box_count) {
        b.DrainChunk(chunk_tiles, span);
        chunk_tiles.clear();
      }
    }
  }
  return {nullptr, nullptr, q_t, s_t};
}

ExpansionResult ExpandMxBNnStoreFused(const ResolvedTileLoad& ld, int64_t n, int64_t k,
                                      const StoreSite& q_store, const StoreSite& s_store, ExpandBuilder& b,
                                      const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  const int64_t nb = n / kMxPackTileM;
  const int64_t kb = k / kMxPackTileK;

  // The [N,K] transpose input is compiler-owned temporary storage.  Reusing a
  // shape-compatible Out/InOut parameter would silently overwrite user data.
  auto nk_rows = MakeIndex(n, span);
  auto nk_cols = MakeIndex(k, span);
  TileView nk_view;
  nk_view.valid_shape = {nk_rows, nk_cols};
  nk_view.blayout = TileLayout::row_major;
  nk_view.slayout = TileLayout::none_box;
  auto nk_type = std::make_shared<TileType>(std::vector<ExprPtr>{nk_rows, nk_cols}, DataType::FP8E4M3FN,
                                            std::nullopt, nk_view, MemorySpace::Vec);
  auto nk_shape = MakeShape2(n, k, span);
  auto raw_nk =
      As<Call>(reg.Create("tile.create", {nk_shape},
                          {{"dtype", DataType::FP8E4M3FN}, {"target_memory", MemorySpace::Vec}}, span));
  INTERNAL_CHECK_SPAN(raw_nk, span) << "Internal error: B quant tile.create did not produce a Call";
  auto typed_nk =
      std::make_shared<Call>(raw_nk->op_, raw_nk->args_, raw_nk->kwargs_, raw_nk->attrs_, nk_type, span);
  ExprPtr nk_t = b.Bind("nk_q", typed_nk, span);

  ExprPtr s_t = s_store.tensor;
  std::vector<ExprPtr> chunk_tiles;
  const int64_t box_count = nb * kb;
  int64_t boxes_done = 0;
  for (int64_t ni = 0; ni < nb; ++ni) {
    for (int64_t ki = 0; ki < kb; ++ki) {
      auto box = LoadBox(ld, ni * kMxPackTileM, ki * kMxPackTileK, b, span);
      auto qq = QuantizeBox(box, b, span);
      chunk_tiles.insert(chunk_tiles.end(), {box, qq.q_mk, qq.s});
      nk_t = b.Bind(
          "nk_q",
          reg.Create("tile.assemble", {nk_t, qq.q_mk, MakeShape2(ni * kMxPackTileM, ki * kMxPackTileK, span)},
                     {}, span),
          span);
      s_t = StoreTile(qq.s, s_store.row0, s_store.col0 + (ni * kb + ki) * kMxPackBoxRows, s_t, b, span);
      ++boxes_done;
      if (boxes_done % kMxPackReuseChunkBoxes == 0 || boxes_done == box_count) {
        b.DrainChunk(chunk_tiles, span);
        chunk_tiles.clear();
      }
    }
  }

  ExprPtr nk_q = nk_t;
  auto nk_i8 =
      b.Bind("nk_i8", reg.Create("tile.reinterpret_view", {nk_q}, {{"dtype", DataType::INT8}}, span), span);
  auto kn_i8 = b.Bind(
      "kn_i8", reg.Create("tile.transpose", {nk_i8, MakeIndex(0, span), MakeIndex(1, span)}, {}, span), span);
  auto kn_q = b.Bind(
      "kn_q", reg.Create("tile.reinterpret_view", {kn_i8}, {{"dtype", DataType::FP8E4M3FN}}, span), span);
  ExprPtr q_t = StoreTile(kn_q, q_store.row0, q_store.col0, q_store.tensor, b, span);
  b.DrainChunk({nk_q, kn_q}, span, "transpose_keep");

  return {nullptr, nullptr, q_t, s_t};
}

std::pair<ExprPtr, ExprPtr> ExpandMxAZzAssemble(const ExprPtr& src, const std::optional<ResolvedTileLoad>& ld,
                                                int64_t m, int64_t k, ExpandBuilder& b, const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  const int64_t mb = m / kMxPackTileM;
  const int64_t kb = k / kMxPackTileK;
  const int64_t groups = m * k / kMxPackGroup;

  auto q_u8 = CreatePlainU8Buffer(b, m, k, "q_u8", span);
  auto s_u8 = CreateMxScaleU8Buffer(b, groups, span);

  for (int64_t mi = 0; mi < mb; ++mi) {
    for (int64_t ki = 0; ki < kb; ++ki) {
      ExprPtr box = ld ? LoadBox(*ld, mi * kMxPackTileM, ki * kMxPackTileK, b, span)
                       : SliceBox(src, mi * kMxPackTileM, ki * kMxPackTileK, b, span);
      auto qq = QuantizeBox(box, b, span);
      auto q_mk_u8 = b.Bind(
          "qmk_u8", reg.Create("tile.reinterpret_view", {qq.q_mk}, {{"dtype", DataType::UINT8}}, span), span);
      q_u8 = b.Bind(
          "q_u8",
          reg.Create("tile.assemble", {q_u8, q_mk_u8, MakeShape2(mi * kMxPackTileM, ki * kMxPackTileK, span)},
                     {}, span),
          span);
      auto s_box_u8 = b.Bind(
          "sb_u8", reg.Create("tile.reinterpret_view", {qq.s}, {{"dtype", DataType::UINT8}}, span), span);
      s_u8 =
          b.Bind("s_u8",
                 reg.Create("tile.assemble",
                            {s_u8, s_box_u8, MakeShape2(0, (mi * kb + ki) * kMxPackBoxRows, span)}, {}, span),
                 span);
    }
  }
  auto q_acc =
      b.Bind("q", reg.Create("tile.reinterpret_view", {q_u8}, {{"dtype", DataType::FP8E4M3FN}}, span), span);
  auto s_acc = ReinterpretMxScaleBuffer(s_u8, groups, b, span);
  return {q_acc, s_acc};
}

std::pair<ExprPtr, ExprPtr> ExpandMxBNnAssemble(const ExprPtr& src, const std::optional<ResolvedTileLoad>& ld,
                                                int64_t n, int64_t k, ExpandBuilder& b, const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  const int64_t nb = n / kMxPackTileM;
  const int64_t kb = k / kMxPackTileK;
  const int64_t groups = (k / kMxPackGroup) * n;

  auto nk_u8 = CreatePlainU8Buffer(b, n, k, "nk_u8", span);
  auto s_u8 = CreateMxScaleU8Buffer(b, groups, span);

  for (int64_t ni = 0; ni < nb; ++ni) {
    for (int64_t ki = 0; ki < kb; ++ki) {
      ExprPtr box = ld ? LoadBox(*ld, ni * kMxPackTileM, ki * kMxPackTileK, b, span)
                       : SliceBox(src, ni * kMxPackTileM, ki * kMxPackTileK, b, span);
      auto qq = QuantizeBox(box, b, span);
      auto q_nk_u8 = b.Bind(
          "qnk_u8", reg.Create("tile.reinterpret_view", {qq.q_mk}, {{"dtype", DataType::UINT8}}, span), span);
      nk_u8 = b.Bind(
          "nk_u8",
          reg.Create("tile.assemble",
                     {nk_u8, q_nk_u8, MakeShape2(ni * kMxPackTileM, ki * kMxPackTileK, span)}, {}, span),
          span);
      auto s_box_u8 = b.Bind(
          "sb_u8", reg.Create("tile.reinterpret_view", {qq.s}, {{"dtype", DataType::UINT8}}, span), span);
      s_u8 =
          b.Bind("s_u8",
                 reg.Create("tile.assemble",
                            {s_u8, s_box_u8, MakeShape2(0, (ni * kb + ki) * kMxPackBoxRows, span)}, {}, span),
                 span);
    }
  }

  auto nk_q = b.Bind(
      "nk_q", reg.Create("tile.reinterpret_view", {nk_u8}, {{"dtype", DataType::FP8E4M3FN}}, span), span);
  auto nk_i8 =
      b.Bind("nk_i8", reg.Create("tile.reinterpret_view", {nk_q}, {{"dtype", DataType::INT8}}, span), span);
  auto kn_i8 = b.Bind(
      "kn_i8", reg.Create("tile.transpose", {nk_i8, MakeIndex(0, span), MakeIndex(1, span)}, {}, span), span);
  auto kn_q = b.Bind(
      "kn_q", reg.Create("tile.reinterpret_view", {kn_i8}, {{"dtype", DataType::FP8E4M3FN}}, span), span);
  auto s_acc = ReinterpretMxScaleBuffer(s_u8, groups, b, span);
  return {kn_q, s_acc};
}

void CheckPackShape(TensorLayout layout, int64_t d0, int64_t d1, const Span& span) {
  const char* name = layout == TensorLayout::MX_A_ZZ ? "MX_A_ZZ" : "MX_B_NN";
  CHECK_SPAN(d0 % kMxPackTileM == 0 && d1 % kMxPackTileK == 0, span)
      << "tile.tquant_mx(layout=" << name << ") requires dim0%" << kMxPackTileM << "==0 and dim1%"
      << kMxPackTileK << "==0, got [" << d0 << ", " << d1 << "]";
}

class ExpandMxPackedQuantMutator : public IRMutator {
 public:
  ExprPtr VisitExpr_(const TupleGetItemExprPtr& op) override {
    if (const Var* tuple_var = GetVarIdentity(op->tuple_)) {
      if (auto it = tuple_outputs_.find(tuple_var); it != tuple_outputs_.end()) {
        INTERNAL_CHECK_SPAN(op->index_ >= 0 && op->index_ < 2, op->span_)
            << "Internal error: expanded MX packed quant tuple index out of range: " << op->index_;
        return it->second[static_cast<size_t>(op->index_)];
      }
    }
    return IRMutator::VisitExpr_(op);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    // Store-fused expansion reloads every box directly from the source tensor.
    // Drop the aggregate tile.load when the collector proved that packed quant
    // was its only consumer, while retaining the definition for LoadBox
    // resolution below.
    if (dead_source_loads_.count(op->var_.get()) != 0) {
      def_map_[op->var_.get()] = op->value_;
      return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, op->span_);
    }

    // Both tuple projections are dead after their only consumers (the two
    // stores) were fused into the expansion.
    if (dead_result_aliases_.count(op->var_.get()) != 0) {
      return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, op->span_);
    }

    // Remap stores that were already emitted by a fused expand. Per-box async
    // lifetimes are bounded by the PIPE_ALL chunk drains emitted inside the
    // expansion.
    if (auto it = store_results_.find(op->var_.get()); it != store_results_.end()) {
      def_map_[op->var_.get()] = it->second.tensor_result;
      return std::make_shared<AssignStmt>(op->var_, it->second.tensor_result, op->span_);
    }

    auto call = As<Call>(op->value_);
    auto layout = GetMxPackLayout(call);
    if (!layout.has_value()) {
      if (!call) {
        if (const Var* source = GetVarIdentity(op->value_)) {
          if (auto it = tuple_outputs_.find(source); it != tuple_outputs_.end()) {
            tuple_outputs_[op->var_.get()] = it->second;
            auto tuple_value =
                std::make_shared<MakeTuple>(std::vector<ExprPtr>{it->second[0], it->second[1]}, op->span_);
            auto stmt = std::make_shared<AssignStmt>(op->var_, tuple_value, op->span_);
            def_map_[op->var_.get()] = tuple_value;
            return stmt;
          }
        }
      }
      auto new_stmt = IRMutator::VisitStmt_(op);
      if (auto as = As<AssignStmt>(new_stmt)) {
        def_map_[as->var_.get()] = as->value_;
      }
      return new_stmt;
    }

    const Span& span = op->span_;
    auto src = call->args_[0];
    auto src_type = As<TileType>(src->GetType());
    INTERNAL_CHECK_SPAN(src_type && src_type->shape_.size() == 2, span)
        << "Internal error: MX packed quant expand requires 2D src";
    auto d0 = As<ConstInt>(src_type->shape_[0]);
    auto d1 = As<ConstInt>(src_type->shape_[1]);
    INTERNAL_CHECK_SPAN(d0 && d1, span) << "Internal error: MX packed quant expand requires static shape";
    CheckPackShape(*layout, d0->value_, d1->value_, span);

    auto resolved = ResolveTileLoad(src, def_map_);
    ExpandBuilder builder(op->var_->name_hint_, temp_counter_);

    // Prefer store-fused expand when both consumer stores were recorded.
    auto site_it = pack_sites_.find(op->var_.get());
    const bool can_fuse = site_it != pack_sites_.end() && site_it->second.q_store &&
                          site_it->second.s_store && site_it->second.outputs_are_store_only &&
                          resolved.has_value();

    ExpansionResult outs;
    if (can_fuse) {
      const auto& site = site_it->second;
      if (*layout == TensorLayout::MX_A_ZZ) {
        outs = ExpandMxAZzStoreFused(*resolved, d0->value_, d1->value_, *site.q_store, *site.s_store, builder,
                                     span);
      } else {
        outs = ExpandMxBNnStoreFused(*resolved, d0->value_, d1->value_, *site.q_store, *site.s_store, builder,
                                     span);
      }
      if (site.q_store->result_var) {
        store_results_[site.q_store->result_var] = MakeFusedStoreReplacement(outs.q_store_result);
      }
      if (site.s_store->result_var) {
        store_results_[site.s_store->result_var] = MakeFusedStoreReplacement(outs.s_store_result);
      }
    } else {
      auto pair = (*layout == TensorLayout::MX_A_ZZ)
                      ? ExpandMxAZzAssemble(src, resolved, d0->value_, d1->value_, builder, span)
                      : ExpandMxBNnAssemble(src, resolved, d0->value_, d1->value_, builder, span);
      outs.quant = pair.first;
      outs.scale = pair.second;
    }

    auto stmts = builder.TakeStmts();
    if (can_fuse && site_it->second.outputs_are_store_only) {
      return std::make_shared<SeqStmts>(std::move(stmts), span);
    }

    tuple_outputs_[op->var_.get()] = {outs.quant, outs.scale};
    auto tuple_val = std::make_shared<MakeTuple>(std::vector<ExprPtr>{outs.quant, outs.scale}, span);
    stmts.push_back(std::make_shared<AssignStmt>(op->var_, tuple_val, span));
    def_map_[op->var_.get()] = tuple_val;
    return std::make_shared<SeqStmts>(std::move(stmts), span);
  }

  void SetPackSites(std::unordered_map<const Var*, MxPackSite> sites) { pack_sites_ = std::move(sites); }

  void SetDeadSourceLoads(std::unordered_set<const Var*> loads) { dead_source_loads_ = std::move(loads); }

  void SetDeadResultAliases(std::unordered_set<const Var*> aliases) {
    dead_result_aliases_ = std::move(aliases);
  }

 private:
  std::size_t temp_counter_ = 0;
  std::unordered_map<const Var*, std::array<ExprPtr, 2>> tuple_outputs_;
  std::unordered_map<const Var*, ExprPtr> def_map_;
  std::unordered_map<const Var*, MxPackSite> pack_sites_;
  std::unordered_map<const Var*, FusedStoreReplacement> store_results_;
  std::unordered_set<const Var*> dead_source_loads_;
  std::unordered_set<const Var*> dead_result_aliases_;
};

class CollectMxPackSites : public IRVisitor {
 public:
  explicit CollectMxPackSites(const std::vector<VarPtr>& params) {
    for (const auto& param : params) {
      function_params_.insert(param.get());
    }
  }

  void VisitStmt_(const SeqStmtsPtr& op) override {
    const SeqStmts* saved_seq = current_seq_;
    const size_t saved_index = current_stmt_index_;
    current_seq_ = op.get();
    for (size_t i = 0; i < op->stmts_.size(); ++i) {
      current_stmt_index_ = i;
      VisitStmt(op->stmts_[i]);
    }
    current_seq_ = saved_seq;
    current_stmt_index_ = saved_index;
  }

  void VisitExpr_(const VarPtr& op) override {
    ++use_counts_[op.get()];
    IRVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    def_map_[op->var_.get()] = op->value_;

    if (auto layout = GetMxPackLayout(As<Call>(op->value_))) {
      MxPackSite site;
      site.quant_var = op->var_.get();
      site.layout = *layout;
      site.src = As<Call>(op->value_)->args_[0];
      site.parent_seq = current_seq_;
      site.stmt_index = current_stmt_index_;
      sites_.emplace(op->var_.get(), std::move(site));
      VisitExpr(op->value_);
      return;
    }

    auto call = As<Call>(op->value_);
    if (call && IsOp(call, "tile.store") && call->args_.size() >= 3) {
      TryAttachStore(op, call);
    }
    // The assignment target is a definition, not a use.  Visiting the default
    // AssignStmt traversal would count it and prevent single-consumer source
    // loads from being recognized as dead.
    VisitExpr(op->value_);
  }

  std::unordered_map<const Var*, MxPackSite> TakeSites() { return std::move(sites_); }

  void DiscardUnsafeStoreSites() {
    for (auto& [_, site] : sites_) {
      if (!IsSafeStoreFusion(site)) {
        site.q_store.reset();
        site.s_store.reset();
      }
    }
  }

  std::unordered_set<const Var*> FindDeadSourceLoads() const {
    std::unordered_set<const Var*> result;
    for (const auto& [_, site] : sites_) {
      if (!site.outputs_are_store_only) continue;
      if (!ResolveTileLoad(site.src, def_map_)) continue;
      const Var* source = GetVarIdentity(site.src);
      if (!source) continue;
      auto use_it = use_counts_.find(source);
      if (use_it == use_counts_.end() || use_it->second != 1) continue;
      auto def_it = def_map_.find(source);
      auto load = def_it == def_map_.end() ? nullptr : As<Call>(def_it->second);
      if (load && IsOp(load, "tile.load")) result.insert(source);
    }
    return result;
  }

  std::unordered_set<const Var*> MarkStoreOnlyOutputs() {
    std::unordered_set<const Var*> result;
    for (auto& [_, site] : sites_) {
      if (!site.q_store || !site.s_store) continue;
      // Projection aliases can be removed only when the mutator will take the
      // store-fused path.  A transformed source such as abs(tile.load(...))
      // falls back to tile.slice/assemble and still needs both aliases for the
      // original stores.
      if (!ResolveTileLoad(site.src, def_map_)) continue;
      const Var* q_var = site.q_store->stored_tile_var;
      const Var* s_var = site.s_store->stored_tile_var;
      if (!IsExclusiveProjection(q_var, site.quant_var, 0) ||
          !IsExclusiveProjection(s_var, site.quant_var, 1)) {
        continue;
      }
      auto tuple_uses = use_counts_.find(site.quant_var);
      if (tuple_uses == use_counts_.end() || tuple_uses->second != 2) continue;
      site.outputs_are_store_only = true;
      result.insert(q_var);
      result.insert(s_var);
    }
    return result;
  }

 private:
  class VarUseFinder : public IRVisitor {
   public:
    explicit VarUseFinder(const Var* target) : target_(target) {}

    void VisitExpr_(const VarPtr& op) override {
      if (op.get() == target_) used_ = true;
      IRVisitor::VisitExpr_(op);
    }

    [[nodiscard]] bool used() const { return used_; }

   private:
    const Var* target_;
    bool used_ = false;
  };

  static bool StmtUsesVar(const StmtPtr& stmt, const Var* target) {
    VarUseFinder finder(target);
    finder.VisitStmt(stmt);
    return finder.used();
  }

  static bool HasInterveningUse(const MxPackSite& site, const StoreSite& store) {
    const Var* tensor = GetVarIdentity(store.tensor);
    if (!tensor || !site.parent_seq) return true;
    for (size_t i = site.stmt_index + 1; i < store.stmt_index; ++i) {
      if (StmtUsesVar(site.parent_seq->stmts_[i], tensor)) return true;
    }
    return false;
  }

  bool IsSafeStoreFusion(const MxPackSite& site) const {
    if (!site.q_store || !site.s_store || !site.parent_seq) return false;
    const auto& q_store = *site.q_store;
    const auto& s_store = *site.s_store;
    if (q_store.parent_seq != site.parent_seq || s_store.parent_seq != site.parent_seq) return false;
    if (q_store.stmt_index <= site.stmt_index || s_store.stmt_index <= site.stmt_index) return false;

    // Fusion emits the destination writes at the quantization site.  Do not
    // move either write across an intervening read/update of that same tensor;
    // the replacement AssignStmt remains at the old store site, but the GM
    // side effect itself would otherwise become observable too early.
    if (HasInterveningUse(site, q_store) || HasInterveningUse(site, s_store)) return false;

    // The fused implementation reloads the source one box at a time.  If a
    // destination is also the source tensor, those writes could clobber boxes
    // that the original aggregate load had already captured.
    auto source_load = ResolveTileLoad(site.src, def_map_);
    if (!source_load) return false;
    const Var* source_tensor = GetVarIdentity(source_load->tensor);
    const Var* q_tensor = GetVarIdentity(q_store.tensor);
    const Var* s_tensor = GetVarIdentity(s_store.tensor);
    return !source_tensor || (source_tensor != q_tensor && source_tensor != s_tensor);
  }

  bool IsExclusiveProjection(const Var* projection, const Var* tuple, int index) const {
    if (!projection) return false;
    auto uses = use_counts_.find(projection);
    if (uses == use_counts_.end() || uses->second != 1) return false;
    auto def = def_map_.find(projection);
    if (def == def_map_.end()) return false;
    auto get = As<TupleGetItemExpr>(def->second);
    return get && get->index_ == index && GetVarIdentity(get->tuple_) == tuple;
  }

  void TryAttachStore(const AssignStmtPtr& op, const CallPtr& store_call) {
    ExprPtr tile = store_call->args_[0];
    int index = -1;
    const Var* quant_var = nullptr;

    if (auto get = As<TupleGetItemExpr>(tile)) {
      quant_var = GetVarIdentity(get->tuple_);
      index = get->index_;
    } else if (const Var* tv = GetVarIdentity(tile)) {
      // Follow aliases: q = tuple[0]
      auto it = def_map_.find(tv);
      if (it != def_map_.end()) {
        if (auto get = As<TupleGetItemExpr>(it->second)) {
          quant_var = GetVarIdentity(get->tuple_);
          index = get->index_;
        }
      }
    }
    if (!quant_var || (index != 0 && index != 1)) return;
    auto sit = sites_.find(quant_var);
    if (sit == sites_.end()) return;
    if (!current_seq_ || sit->second.parent_seq != current_seq_) return;

    // Moving the store to the quantization site is safe only for destination
    // buffers that already exist at function entry.  A later SSA tensor value
    // would become a free variable, and an intervening update would be
    // reordered across the fused store.
    const Var* store_tensor = GetVarIdentity(store_call->args_[2]);
    if (!store_tensor || function_params_.count(store_tensor) == 0) return;

    int64_t r = 0;
    int64_t c = 0;
    if (!ConstOffset2(store_call->args_[1], &r, &c)) return;
    StoreSite ss;
    ss.tensor = store_call->args_[2];
    ss.row0 = r;
    ss.col0 = c;
    ss.result_var = op->var_.get();
    ss.stored_tile_var = GetVarIdentity(tile);
    ss.parent_seq = current_seq_;
    ss.stmt_index = current_stmt_index_;
    if (index == 0) {
      sit->second.q_store = ss;
    } else {
      sit->second.s_store = ss;
    }
  }

  std::unordered_map<const Var*, ExprPtr> def_map_;
  std::unordered_map<const Var*, MxPackSite> sites_;
  std::unordered_map<const Var*, size_t> use_counts_;
  std::unordered_set<const Var*> function_params_;
  const SeqStmts* current_seq_ = nullptr;
  size_t current_stmt_index_ = 0;
};

}  // namespace

Pass ExpandMxPackedQuant() {
  auto pass_func = [](const FunctionPtr& func) -> FunctionPtr {
    if (!func || !func->body_) {
      return func;
    }
    if (!IsInCoreType(func->func_type_)) {
      return func;
    }
    CollectMxPackSites collector(func->params_);
    collector.VisitStmt(func->body_);
    collector.DiscardUnsafeStoreSites();
    auto dead_result_aliases = collector.MarkStoreOnlyOutputs();
    auto dead_source_loads = collector.FindDeadSourceLoads();
    auto sites = collector.TakeSites();
    if (sites.empty()) {
      return func;
    }
    ExpandMxPackedQuantMutator mutator;
    mutator.SetPackSites(std::move(sites));
    mutator.SetDeadSourceLoads(std::move(dead_source_loads));
    mutator.SetDeadResultAliases(std::move(dead_result_aliases));
    auto new_body = mutator.VisitStmt(func->body_);
    if (new_body.get() == func->body_.get()) {
      return func;
    }
    return std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                      func->return_types_, new_body, func->span_, func->func_type_,
                                      func->level_, func->role_, func->attrs_);
  };
  return CreateFunctionPass(pass_func, "ExpandMxPackedQuant", kExpandMxPackedQuantProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
