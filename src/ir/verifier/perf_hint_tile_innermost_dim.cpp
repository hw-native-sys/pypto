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
#include "pypto/ir/op_registry.h"
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
  /// Number of separately-addressed innermost rows in one transfer (the product
  /// of all but the last dim), and the whole-tile byte volume. Both are nullopt
  /// when an outer dim is symbolic. Volume is what separates a hot streaming
  /// operand from an incidental one at the same innermost size — see issue
  /// #2309, where a [1024, 64] weight panel and a [16, 64] activation panel
  /// share every other fact.
  std::optional<uint64_t> rows;
  std::optional<uint64_t> transfer_bytes;
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

  // Outer dims give the row (descriptor) count. A single symbolic outer dim
  // makes the volume unknowable; the innermost facts above still stand, so the
  // hint is emitted without the volume clause rather than dropped.
  uint64_t rows = 1;
  bool rows_static = true;
  for (size_t i = 0; i + 1 < tile->shape_.size(); ++i) {
    auto dim = std::dynamic_pointer_cast<const ConstInt>(tile->shape_[i]);
    if (!dim || dim->value_ <= 0) {
      rows_static = false;
      break;
    }
    rows *= static_cast<uint64_t>(dim->value_);
  }
  if (rows_static) {
    info.rows = rows;
    // Same element-count-first rounding as the row size above.
    info.transfer_bytes =
        (rows * static_cast<uint64_t>(last->value_) * static_cast<uint64_t>(bits) + 7u) / 8u;
  }
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
      EmitSite(site);
    }
  }

 protected:
  void VisitExpr_(const CallPtr& op) override {
    IRVisitor::VisitExpr_(op);  // recurse into children first
    if (!op || !op->op_) return;

    const std::string& name = op->op_->name_;
    if (IsOp(op, "tile.load")) {
      // tile.load returns a TileType — innermost dim is on the result.
      RecordIfBelowThreshold(name, op->GetType(), op->span_);
    } else if (IsOp(op, "tile.store")) {
      // tile.store's first arg is the source tile; innermost dim lives there.
      if (op->args_.empty() || !op->args_[0]) return;
      RecordIfBelowThreshold(name, op->args_[0]->GetType(), op->span_);
    }
  }

 private:
  /// One deduplicated diagnostic site, keyed by (file, line, col, op_name) plus
  /// the transfer facts (see SiteKey). `Span` has const members (no default
  /// ctor / copy-assign), so Site is constructed once via try_emplace and never
  /// reassigned; `count` is the only field that mutates after insertion.
  struct Site {
    Site(TileTransferInfo i, Span s, std::string op)
        : info(std::move(i)), span(std::move(s)), op_name(std::move(op)) {}
    TileTransferInfo info;
    Span span;
    std::string op_name;
    uint32_t count = 0;
  };

  // Dedup key per issue #1305 ask (4): (file, line, col, op_name) plus the
  // transfer facts the hint actually renders (dtype, innermost_bytes, memory).
  // Loop-unroll and per-fragment expansion produce many tile ops at the same
  // source span; collapsing *identical* transfers to one hint with a count
  // keeps the signal readable. The transfer facts are part of the key so that
  // distinct tiles sharing a span (e.g. multiple loads on one line, or macro
  // expansion) are not conflated — each (size, dtype, memory) gets its own
  // hint with an accurate count, rather than all collapsing onto the
  // first-seen tuple.
  // `transfer_bytes` is part of the key (issue #2309): two loads can agree on
  // span, op, dtype, innermost size and memory space and still differ by orders
  // of magnitude in volume — a [1024, 64] weight panel and a [16, 64]
  // activation panel under one `pl.spmd` line. Collapsing them onto one hint
  // reported an occurrence count that named neither operand.
  using SiteKey = std::tuple<std::string, int, int, std::string, std::string, uint64_t,
                             std::optional<MemorySpace>, std::optional<uint64_t>>;

  void RecordIfBelowThreshold(const std::string& op_name, const TypePtr& tile_type, const Span& span) {
    auto info_opt = InspectTile(tile_type);
    if (!info_opt.has_value()) return;
    const TileTransferInfo& info = *info_opt;

    // Deliberately NOT skipped by memory space (issue #2309). An earlier
    // revision returned early for cube-private spaces (Mat/Left/Right/Acc) on
    // the grounds that they never traverse L2. That reasoning is sound for
    // chip-internal movement, but this check inspects only `tile.load` and
    // `tile.store`, whose non-tile side is a `TensorType` — always off-chip.
    // Both are therefore GM<->chip transfers whatever space the tile lands in,
    // and the GM side always crosses L2 at the innermost-dim granularity. The
    // genuinely cube-private moves (Mat->L0 `tile.move` / `tile.extract`) are
    // not visited here at all, so the skip could only ever suppress true
    // positives: it silenced the GM->Mat weight load of a `b_trans` matmul,
    // whose loaded shape is the caller's [N, K] window transposed, at 128B per
    // row against a 512B recommendation, for a measured 16-25% penalty.
    if (info.innermost_bytes >= recommended_bytes_) return;

    // Invalid/unknown spans (Span::unknown() -> "", -1, -1) carry no real
    // source location, so they cannot be meaningfully deduplicated by site:
    // every such op would collapse onto the same ("", -1, -1, op, ...) key,
    // conflating unrelated transfers across the whole program into one bogus
    // "N occurrences at this source location" hint. Emit them individually.
    if (!span.is_valid()) {
      Site site(info, span, op_name);
      site.count = 1;
      EmitSite(site);
      return;
    }

    SiteKey key{span.filename_,  span.begin_line_,     span.begin_column_, op_name,
                info.dtype_name, info.innermost_bytes, info.memory,        info.transfer_bytes};
    auto [it, inserted] = sites_.try_emplace(key, info, span, op_name);
    (void)inserted;
    ++it->second.count;
  }

  /// Append one diagnostic for a fully-populated site.
  void EmitSite(const Site& site) {
    diagnostics_.emplace_back(DiagnosticSeverity::PerfHint, "TileInnermostDimGranularity",
                              kTileInnermostDimGranularityCode, /*hint_code=*/"PH001", BuildMessage(site),
                              site.span);
  }

  [[nodiscard]] std::string BuildMessage(const Site& site) const {
    const TileTransferInfo& info = site.info;
    const std::string mem_str =
        info.memory.has_value() ? MemorySpaceToString(*info.memory) : std::string("unknown");

    std::ostringstream msg;
    msg << site.op_name << " has innermost dim = " << info.innermost_bytes << "B"
        << " (tile " << info.dtype_name << "[" << info.innermost_elems << "], target_memory=" << mem_str
        << ")";
    // Volume clause (issue #2309): the row count is the number of separate
    // short bus transactions this one transfer issues, and the total is how
    // much traffic rides on them — together they rank a hint against its
    // siblings at the same innermost size. Omitted when an outer dim is
    // symbolic.
    if (info.rows.has_value() && info.transfer_bytes.has_value()) {
      msg << "; moves " << *info.transfer_bytes << "B as " << *info.rows << " x " << info.innermost_bytes
          << "B rows";
    }
    msg << "; recommended >= " << recommended_bytes_ << "B for backend " << arch_
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
    // per (file, line, col, op, dtype, bytes, memory) site now (issue #1305
    // ask 4). Hits at invalid spans were already emitted individually.
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
