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
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/pipe.h"
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

/// Evaluated facts about a flagged tile transfer, carried from inspection to
/// message construction so the hint can echo the exact (shape, dtype,
/// target_memory) tuple it reasoned about — see issue #1305 ask (5).
struct TileTransferInfo {
  uint64_t innermost_bytes;           ///< Innermost-dim size in bytes.
  int64_t innermost_elems;            ///< Innermost-dim element count.
  std::string dtype_name;             ///< Element dtype, e.g. "int8".
  std::optional<MemorySpace> memory;  ///< target_memory of the tile, if known.
};

/// Inspect a TileType and compute the innermost-dim transfer facts. Returns
/// nullopt if the shape is symbolic, the dtype has unknown bit width, or the
/// type is not a tile.
///
/// We use the result type for tile.load (which produces a tile) and the first
/// argument's type for tile.store (which consumes a tile) — see VisitExpr_.
std::optional<TileTransferInfo> InspectTile(const TypePtr& type) {
  auto tile = std::dynamic_pointer_cast<const TileType>(type);
  if (!tile) return std::nullopt;
  if (tile->shape_.empty()) return std::nullopt;

  // Innermost dimension must be a constant integer to compute byte size.
  auto last = std::dynamic_pointer_cast<const ConstInt>(tile->shape_.back());
  if (!last) return std::nullopt;
  if (last->value_ <= 0) return std::nullopt;

  size_t bits = tile->dtype_.GetBit();
  if (bits == 0) return std::nullopt;

  TileTransferInfo info;
  // Multiply by element count first, then round up to bytes — for sub-byte
  // dtypes (int4, bool, ...) per-element rounding overestimates the row size
  // and would mask hints that should fire. Innermost-dim granularity is a
  // bus / cache concern measured in bytes.
  info.innermost_bytes = (static_cast<uint64_t>(last->value_) * static_cast<uint64_t>(bits) + 7u) / 8u;
  info.innermost_elems = last->value_;
  info.dtype_name = tile->dtype_.ToString();
  info.memory = tile->GetMemorySpace();
  return info;
}

class TileInnermostDimVisitor : public IRVisitor {
 public:
  TileInnermostDimVisitor(std::vector<Diagnostic>& diagnostics, uint32_t recommended_bytes,
                          uint32_t l2_cache_line_bytes, std::string arch)
      : diagnostics_(diagnostics),
        recommended_bytes_(recommended_bytes),
        l2_cache_line_bytes_(l2_cache_line_bytes),
        arch_(std::move(arch)) {}

  /// Flush deduplicated sites into `diagnostics_`. Call once after the walk.
  void Flush() {
    for (auto& [key, site] : sites_) {
      (void)key;
      diagnostics_.emplace_back(DiagnosticSeverity::PerfHint, "TileInnermostDimGranularity",
                                kTileInnermostDimGranularityCode, /*hint_code=*/"PH001", BuildMessage(site),
                                site.span);
    }
  }

 protected:
  void VisitExpr_(const CallPtr& op) override {
    IRVisitor::VisitExpr_(op);  // recurse into children first
    if (!op || !op->op_) return;

    const std::string& name = op->op_->name_;
    if (name == "tile.load") {
      // tile.load returns a TileType — innermost dim is on the result.
      RecordIfBelowThreshold(name, op->GetType(), op->span_);
    } else if (name == "tile.store") {
      // tile.store's first arg is the source tile; innermost dim lives there.
      if (op->args_.empty() || !op->args_[0]) return;
      RecordIfBelowThreshold(name, op->args_[0]->GetType(), op->span_);
    }
  }

 private:
  /// One deduplicated diagnostic site, keyed by (file, line, col, op_name).
  /// `Span` has const members (no default ctor / copy-assign), so Site is
  /// constructed once via try_emplace and never reassigned; `count` is the
  /// only field that mutates after insertion.
  struct Site {
    Site(TileTransferInfo i, Span s, std::string op)
        : info(std::move(i)), span(std::move(s)), op_name(std::move(op)) {}
    TileTransferInfo info;
    Span span;
    std::string op_name;
    uint32_t count = 0;
  };

  // Dedup key per issue #1305 ask (4): (file, line, col, op_name). Loop-unroll
  // and per-fragment expansion produce many tile ops at the same source span;
  // collapsing them to one hint with a count keeps the signal readable.
  using SiteKey = std::tuple<std::string, int, int, std::string>;

  void RecordIfBelowThreshold(const std::string& op_name, const TypePtr& tile_type, const Span& span) {
    auto info_opt = InspectTile(tile_type);
    if (!info_opt.has_value()) return;
    const TileTransferInfo& info = *info_opt;

    // Memory-space awareness (ask 1): the recommended-bytes threshold is an L2
    // cache-line concern, which only applies to transfers that traverse L2
    // (GM <-> Vec). Cube-private L0/L1 buffers (Left/Right/Acc/Mat) never touch
    // L2, so the L2 threshold is meaningless for them — skip to avoid
    // false-positive noise on fully-tuned cube kernels.
    if (info.memory.has_value() && IsCubeMemorySpace(*info.memory)) return;

    if (info.innermost_bytes >= recommended_bytes_) return;

    SiteKey key{span.filename_, span.begin_line_, span.begin_column_, op_name};
    auto [it, inserted] = sites_.try_emplace(key, info, span, op_name);
    (void)inserted;
    ++it->second.count;
  }

  std::string BuildMessage(const Site& site) const {
    const TileTransferInfo& info = site.info;
    const std::string mem_str =
        info.memory.has_value() ? MemorySpaceToString(*info.memory) : std::string("unknown");

    std::ostringstream msg;
    msg << site.op_name << " has innermost dim = " << info.innermost_bytes << "B"
        << " (tile " << info.dtype_name << "[" << info.innermost_elems << "], target_memory=" << mem_str
        << ")"
        << "; recommended >= " << recommended_bytes_ << "B for backend " << arch_
        << " (L2 cache line = " << l2_cache_line_bytes_
        << "B). Consider increasing tile shape on the innermost axis.";
    if (site.count > 1) {
      msg << " (" << site.count << " occurrences at this source location)";
    }
    // Ask (2)/(3): the span below is the post-pipeline IR-text location
    // ("<string>:line:col"), not the originating DSL pl.at / slicing
    // expression, and the inner-dim chunk constant is not recoverable here.
    // Mapping back to the user's source requires DSL source spans to be
    // threaded through the parser/IR onto the tile op (none of TileType, the
    // Call op, or Span currently carry the originating DSL file:line for the
    // slicing expression), so we cannot name the controlling constant from
    // inside this verifier today. The (dtype, innermost, target_memory) tuple
    // above lets users reconcile the hint against the IR they are inspecting.
    return msg.str();
  }

  std::vector<Diagnostic>& diagnostics_;
  uint32_t recommended_bytes_;
  uint32_t l2_cache_line_bytes_;
  std::string arch_;
  std::map<SiteKey, Site> sites_;
};

class TileInnermostDimVerifier : public PropertyVerifier {
 public:
  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;

    // Backend thresholds come from the active PassContext per the
    // `pass-context-config` rule. With no context (e.g. a verifier run in
    // isolation), there is no backend to consult — silently skip rather
    // than emit advice that may be wrong for the real target.
    // GetBackendHandler() is documented as non-null (throws ValueError if
    // the backend type has not been configured), so no null-check is needed.
    const auto* ctx = PassContext::Current();
    if (ctx == nullptr) return;
    const auto* handler = ctx->GetBackendHandler();

    TileInnermostDimVisitor visitor(diagnostics, handler->GetRecommendedInnermostDimBytes(),
                                    handler->GetL2CacheLineBytes(), handler->GetPtoTargetArch());

    for (const auto& [global_var, func] : program->functions_) {
      (void)global_var;
      if (!func || !func->body_) continue;
      visitor.VisitStmt(func->body_);
    }
    // Deduplicated sites are accumulated during the walk; emit one diagnostic
    // per (file, line, col, op) site now (issue #1305 ask 4).
    visitor.Flush();
  }

  [[nodiscard]] std::string GetName() const override { return "TileInnermostDimGranularity"; }
};

}  // namespace

PropertyVerifierPtr CreateTileInnermostDimGranularityVerifier() {
  return std::make_shared<TileInnermostDimVerifier>();
}

}  // namespace ir
}  // namespace pypto
