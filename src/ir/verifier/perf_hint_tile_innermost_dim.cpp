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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/diagnostic_check_registry.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

constexpr int kTileInnermostDimGranularityCode = 1;  // PH001 — issue #1180

/// Compute the innermost-dim size in bytes for a TileType result. Returns
/// nullopt if the shape is symbolic, the dtype has unknown bit width, or the
/// type is not a tile.
///
/// We use the result type for tile.load (which produces a tile) and the first
/// argument's type for tile.store (which consumes a tile) — see CheckCall.
std::optional<uint64_t> InnermostBytesOfTile(const TypePtr& type) {
  auto tile = std::dynamic_pointer_cast<const TileType>(type);
  if (!tile) return std::nullopt;
  if (tile->shape_.empty()) return std::nullopt;

  // Innermost dimension must be a constant integer to compute byte size.
  auto last = std::dynamic_pointer_cast<const ConstInt>(tile->shape_.back());
  if (!last) return std::nullopt;
  if (last->value_ <= 0) return std::nullopt;

  size_t bits = tile->dtype_.GetBit();
  if (bits == 0) return std::nullopt;

  // Multiply by element count first, then round up to bytes — for sub-byte
  // dtypes (int4, bool, ...) per-element rounding overestimates the row size
  // and would mask hints that should fire. Innermost-dim granularity is a
  // bus / cache concern measured in bytes.
  return (static_cast<uint64_t>(last->value_) * static_cast<uint64_t>(bits) + 7u) / 8u;
}

class TileInnermostDimVisitor : public IRVisitor {
 public:
  TileInnermostDimVisitor(std::vector<Diagnostic>& diagnostics, uint32_t recommended_bytes,
                          uint32_t l2_cache_line_bytes, std::string arch)
      : diagnostics_(diagnostics),
        recommended_bytes_(recommended_bytes),
        l2_cache_line_bytes_(l2_cache_line_bytes),
        arch_(std::move(arch)) {}

 protected:
  void VisitExpr_(const CallPtr& op) override {
    IRVisitor::VisitExpr_(op);  // recurse into children first
    if (!op || !op->op_) return;

    const std::string& name = op->op_->name_;
    if (name == "tile.load") {
      // tile.load returns a TileType — innermost dim is on the result.
      EmitIfBelowThreshold(name, op->GetType(), op->span_);
    } else if (name == "tile.store") {
      // tile.store's first arg is the source tile; innermost dim lives there.
      if (op->args_.empty() || !op->args_[0]) return;
      EmitIfBelowThreshold(name, op->args_[0]->GetType(), op->span_);
    }
  }

 private:
  void EmitIfBelowThreshold(const std::string& op_name, const TypePtr& tile_type, const Span& span) {
    auto bytes_opt = InnermostBytesOfTile(tile_type);
    if (!bytes_opt.has_value()) return;
    uint64_t bytes = *bytes_opt;
    if (bytes >= recommended_bytes_) return;

    std::ostringstream msg;
    msg << op_name << " has innermost dim = " << bytes << "B; recommended >= " << recommended_bytes_
        << "B for backend " << arch_ << " (L2 cache line = " << l2_cache_line_bytes_
        << "B). Consider increasing tile shape on the innermost axis.";

    // The registry stamps severity / hint_code; we still set them for
    // self-documenting verifier behaviour and for direct-invocation tests.
    diagnostics_.emplace_back(DiagnosticSeverity::PerfHint, "TileInnermostDimGranularity",
                              kTileInnermostDimGranularityCode, /*hint_code=*/"PH001", msg.str(), span);
  }

  std::vector<Diagnostic>& diagnostics_;
  uint32_t recommended_bytes_;
  uint32_t l2_cache_line_bytes_;
  std::string arch_;
};

class TileInnermostDimVerifier : public PropertyVerifier {
 public:
  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;

    // Backend thresholds come from the active PassContext per the
    // `pass-context-config` rule. With no context (e.g. a verifier run in
    // isolation), there is no backend to consult — silently skip rather
    // than emit advice that may be wrong for the real target.
    const auto* ctx = PassContext::Current();
    if (ctx == nullptr) return;
    const auto* handler = ctx->GetBackendHandler();
    if (handler == nullptr) return;

    TileInnermostDimVisitor visitor(diagnostics, handler->GetRecommendedInnermostDimBytes(),
                                    handler->GetL2CacheLineBytes(), handler->GetPtoTargetArch());

    for (const auto& [global_var, func] : program->functions_) {
      (void)global_var;
      if (!func || !func->body_) continue;
      visitor.VisitStmt(func->body_);
    }
  }

  [[nodiscard]] std::string GetName() const override { return "TileInnermostDimGranularity"; }
};

}  // namespace

PropertyVerifierPtr CreateTileInnermostDimGranularityVerifier() {
  return std::make_shared<TileInnermostDimVerifier>();
}

}  // namespace ir
}  // namespace pypto
