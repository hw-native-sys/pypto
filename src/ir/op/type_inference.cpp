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

#include "pypto/ir/type_inference.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/printer.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

// Constant-folding arithmetic on INDEX-typed valid extents. When both operands are
// ``ConstInt`` the result is folded to a single ``ConstInt`` so a partially-static
// valid_shape never carries an ``Add``/``Max``/``Min`` of two literals; otherwise the
// scalar-expr node is emitted. Shared by the assemble-union and matmul-bias valid rules.
ExprPtr FoldAdd(const ExprPtr& a, const ExprPtr& b, const Span& span) {
  auto ca = As<ConstInt>(a);
  auto cb = As<ConstInt>(b);
  if (ca && ca->value_ == 0) return b;
  if (cb && cb->value_ == 0) return a;
  if (ca && cb) return std::make_shared<ConstInt>(ca->value_ + cb->value_, DataType::INDEX, span);
  return MakeAdd(a, b, span);
}

ExprPtr FoldMax(const ExprPtr& a, const ExprPtr& b, const Span& span) {
  auto ca = As<ConstInt>(a);
  auto cb = As<ConstInt>(b);
  if (ca && cb) {
    return std::make_shared<ConstInt>(std::max(ca->value_, cb->value_), DataType::INDEX, span);
  }
  if (ProveValidExtentLessEqual(a, b) == ProofResult::kTrue) return b;
  if (ProveValidExtentLessEqual(b, a) == ProofResult::kTrue) return a;
  return MakeMax(a, b, span);
}

ExprPtr FoldMin(const ExprPtr& a, const ExprPtr& b, const Span& span) {
  auto ca = As<ConstInt>(a);
  auto cb = As<ConstInt>(b);
  if (ca && cb) {
    return std::make_shared<ConstInt>(std::min(ca->value_, cb->value_), DataType::INDEX, span);
  }
  if (ProveValidExtentLessEqual(a, b) == ProofResult::kTrue) return a;
  if (ProveValidExtentLessEqual(b, a) == ProofResult::kTrue) return b;
  return MakeMin(a, b, span);
}

ExprPtr FoldMul(const ExprPtr& a, const ExprPtr& b, const Span& span) {
  auto ca = As<ConstInt>(a);
  auto cb = As<ConstInt>(b);
  if ((ca && ca->value_ == 0) || (cb && cb->value_ == 0)) {
    return std::make_shared<ConstInt>(0, DataType::INDEX, span);
  }
  if (ca && ca->value_ == 1) return b;
  if (cb && cb->value_ == 1) return a;
  if (ca && cb) {
    return std::make_shared<ConstInt>(ca->value_ * cb->value_, DataType::INDEX, span);
  }
  return MakeMul(a, b, span);
}

// True iff ``e`` is a compile-time ``ConstInt`` of value 1. Used to detect a
// physical broadcast dim (shape 1) for the elementwise valid_shape agreement rule.
bool IsConstOne(const ExprPtr& e) {
  auto c = As<ConstInt>(e);
  return c && c->value_ == 1;
}

arith::Analyzer& GetTypeInferenceAnalyzer() {
  thread_local arith::Analyzer analyzer;
  return analyzer;
}

}  // namespace

ProofResult ProveValidExtentEqual(const ExprPtr& lhs, const ExprPtr& rhs) {
  if (lhs == rhs) return ProofResult::kTrue;

  auto lhs_const = As<ConstInt>(lhs);
  auto rhs_const = As<ConstInt>(rhs);
  if (lhs_const && rhs_const) {
    return lhs_const->value_ == rhs_const->value_ ? ProofResult::kTrue : ProofResult::kFalse;
  }

  auto& analyzer = GetTypeInferenceAnalyzer();
  if (analyzer.CanProveEqual(lhs, rhs)) return ProofResult::kTrue;
  if (analyzer.CanProve(MakeNe(lhs, rhs))) return ProofResult::kFalse;
  return ProofResult::kUnknown;
}

ProofResult ProveValidExtentLessEqual(const ExprPtr& lhs, const ExprPtr& rhs) {
  if (ProveValidExtentEqual(lhs, rhs) == ProofResult::kTrue) return ProofResult::kTrue;

  auto lhs_const = As<ConstInt>(lhs);
  auto rhs_const = As<ConstInt>(rhs);
  if (lhs_const && rhs_const) {
    return lhs_const->value_ <= rhs_const->value_ ? ProofResult::kTrue : ProofResult::kFalse;
  }

  auto& analyzer = GetTypeInferenceAnalyzer();
  if (analyzer.CanProve(MakeLe(lhs, rhs))) return ProofResult::kTrue;
  if (analyzer.CanProve(MakeGt(lhs, rhs))) return ProofResult::kFalse;
  return ProofResult::kUnknown;
}

void ValidateValidShapeBounds(const std::vector<ExprPtr>& valid_shape,
                              const std::vector<ExprPtr>& physical_shape, const Span& span,
                              const std::string& op_name) {
  CHECK_SPAN(valid_shape.size() == physical_shape.size(), span)
      << op_name << ": valid_shape rank (" << valid_shape.size() << ") must match physical shape rank ("
      << physical_shape.size() << ")";

  auto zero = std::make_shared<ConstInt>(0, DataType::INDEX, span);
  for (size_t i = 0; i < valid_shape.size(); ++i) {
    const ProofResult non_negative = ProveValidExtentLessEqual(zero, valid_shape[i]);
    CHECK_SPAN(non_negative != ProofResult::kFalse, span)
        << op_name << ": valid_shape[" << i << "] must be >= 0, got " << PythonPrint(valid_shape[i]);

    const ProofResult within_bound = ProveValidExtentLessEqual(valid_shape[i], physical_shape[i]);
    CHECK_SPAN(within_bound != ProofResult::kFalse, span)
        << op_name << ": valid_shape[" << i << "]=" << PythonPrint(valid_shape[i])
        << " exceeds physical shape[" << i << "]=" << PythonPrint(physical_shape[i]);
  }
}

void ValidatePhysicalWindowBounds(const std::vector<ExprPtr>& window_shape,
                                  const std::vector<ExprPtr>& offsets,
                                  const std::vector<ExprPtr>& target_shape, const Span& span,
                                  const std::string& op_name) {
  CHECK_SPAN(window_shape.size() == target_shape.size() && offsets.size() == target_shape.size(), span)
      << op_name << ": physical window rank (" << window_shape.size() << "), offset rank (" << offsets.size()
      << "), and target rank (" << target_shape.size() << ") must match";

  auto zero = std::make_shared<ConstInt>(0, DataType::INDEX, span);
  for (size_t i = 0; i < target_shape.size(); ++i) {
    const ProofResult non_negative = ProveValidExtentLessEqual(zero, offsets[i]);
    CHECK_SPAN(non_negative != ProofResult::kFalse, span)
        << op_name << ": offset[" << i << "] must be >= 0, got " << PythonPrint(offsets[i]);

    const ExprPtr end = FoldAdd(offsets[i], window_shape[i], span);
    const ProofResult in_bounds = ProveValidExtentLessEqual(end, target_shape[i]);
    CHECK_SPAN(in_bounds != ProofResult::kFalse, span)
        << op_name << ": physical window on dim " << i << " ends at " << PythonPrint(end)
        << ", beyond target extent " << PythonPrint(target_shape[i]) << " (offset=" << PythonPrint(offsets[i])
        << ", window=" << PythonPrint(window_shape[i]) << ")";
  }
}

std::vector<ExprPtr> LiftValidShapeForBroadcast(const std::vector<ExprPtr>& operand_shape,
                                                const std::vector<ExprPtr>& operand_valid,
                                                const std::vector<ExprPtr>& out_shape, const Span& span,
                                                const std::string& op_name) {
  ValidateValidShapeBounds(operand_valid, operand_shape, span, op_name);
  CHECK_SPAN(operand_shape.size() <= out_shape.size(), span)
      << op_name << ": operand rank " << operand_shape.size() << " exceeds broadcast output rank "
      << out_shape.size();

  const size_t offset = out_shape.size() - operand_shape.size();
  std::vector<ExprPtr> lifted;
  lifted.reserve(out_shape.size());
  for (size_t i = 0; i < out_shape.size(); ++i) {
    if (i < offset) {
      // A missing leading dimension is an implicit, fully-valid singleton.
      lifted.push_back(out_shape[i]);
      continue;
    }

    const size_t d = i - offset;
    if (DimensionsEqual(operand_shape[d], out_shape[i])) {
      lifted.push_back(operand_valid[d]);
      continue;
    }

    CHECK_SPAN(IsConstOne(operand_shape[d]), span)
        << op_name << ": operand physical shape " << FormatShape(operand_shape)
        << " cannot broadcast to output shape " << FormatShape(out_shape) << " at dim " << i;
    // The explicit singleton covers the output dim iff its sole cell is valid.
    // The standing bound invariant makes a dynamic singleton validity a runtime
    // 0/1 value, so valid * out_extent is the exact lifted extent.
    lifted.push_back(FoldMul(operand_valid[d], out_shape[i], span));
  }
  return lifted;
}

std::vector<ExprPtr> ComputeReshapeValidShape(const std::vector<ExprPtr>& src_valid,
                                              const std::vector<ExprPtr>& in_shape,
                                              const std::vector<ExprPtr>& new_shape, const Span& span,
                                              const std::string& op_name) {
  ValidateValidShapeBounds(src_valid, in_shape, span, op_name);
  CHECK_SPAN(!src_valid.empty() && !new_shape.empty(), span)
      << op_name << ": reshape validity mapping requires non-empty input and output ranks";

  bool fully_valid = true;
  for (size_t i = 0; i < src_valid.size(); ++i) {
    if (ProveValidExtentEqual(src_valid[i], in_shape[i]) != ProofResult::kTrue) {
      fully_valid = false;
      break;
    }
  }
  if (fully_valid) return new_shape;

  // The empty set stays empty under every reshape. Handle it before the
  // rectangular-prefix proof below: a box such as [1, 0, N] is not a flat
  // prefix according to that syntactic form, but it still denotes no cells and
  // therefore has an exact representation in every target shape.
  if (std::any_of(src_valid.begin(), src_valid.end(), [](const ExprPtr& extent) {
        auto value = As<ConstInt>(extent);
        return value && value->value_ == 0;
      })) {
    std::vector<ExprPtr> empty;
    empty.reserve(new_shape.size());
    for (size_t i = 0; i < new_shape.size(); ++i) {
      empty.push_back(std::make_shared<ConstInt>(0, DataType::INDEX, span));
    }
    return empty;
  }

  // Inserting or removing provably-full physical unit axes is a coordinate-only
  // rank change, not a flatten/repartition of data. It preserves an arbitrary
  // rectangular valid region exactly (for example [16,16] valid [8,8] ->
  // [1,16,16] valid [1,8,8]). The generic flat-prefix rule below would reject
  // that safe rank lift because [8,8] is not a contiguous prefix of a 16x16
  // buffer. Use a small sequence-alignment search so ambiguous runs of unit
  // axes retain an empty/partial unit axis by matching it when possible; an
  // input unit axis may be erased only when its sole coordinate is provably
  // valid.
  using UnitMap = std::optional<std::vector<ExprPtr>>;
  std::function<UnitMap(size_t, size_t)> map_unit_axes = [&](size_t input_dim, size_t output_dim) -> UnitMap {
    if (input_dim == in_shape.size() && output_dim == new_shape.size()) return std::vector<ExprPtr>{};
    if (input_dim == in_shape.size()) {
      std::vector<ExprPtr> inserted;
      inserted.reserve(new_shape.size() - output_dim);
      for (size_t i = output_dim; i < new_shape.size(); ++i) {
        if (!IsConstOne(new_shape[i])) return std::nullopt;
        inserted.push_back(new_shape[i]);
      }
      return inserted;
    }
    if (output_dim == new_shape.size()) {
      for (size_t i = input_dim; i < in_shape.size(); ++i) {
        if (!IsConstOne(in_shape[i]) ||
            ProveValidExtentEqual(src_valid[i], in_shape[i]) != ProofResult::kTrue) {
          return std::nullopt;
        }
      }
      return std::vector<ExprPtr>{};
    }

    // Prefer matching equal axes so a partial/empty unit axis is preserved
    // rather than accidentally erased and recreated as fully valid.
    if (ProveValidExtentEqual(in_shape[input_dim], new_shape[output_dim]) == ProofResult::kTrue) {
      if (auto tail = map_unit_axes(input_dim + 1, output_dim + 1)) {
        tail->insert(tail->begin(), src_valid[input_dim]);
        return tail;
      }
    }
    if (IsConstOne(in_shape[input_dim]) &&
        ProveValidExtentEqual(src_valid[input_dim], in_shape[input_dim]) == ProofResult::kTrue) {
      if (auto tail = map_unit_axes(input_dim + 1, output_dim)) return tail;
    }
    if (IsConstOne(new_shape[output_dim])) {
      if (auto tail = map_unit_axes(input_dim, output_dim + 1)) {
        tail->insert(tail->begin(), new_shape[output_dim]);
        return tail;
      }
    }
    return std::nullopt;
  };
  if (auto unit_mapped = map_unit_axes(0, 0)) return *unit_mapped;

  const size_t input_rank = src_valid.size();
  // Skip leading dimensions pinned to a single valid coordinate. The first
  // remaining axis is the one free extent in the contiguous prefix.
  size_t free_dim = 0;
  while (free_dim + 1 < input_rank) {
    auto extent = As<ConstInt>(src_valid[free_dim]);
    if (!extent || extent->value_ != 1) break;
    ++free_dim;
  }

  for (size_t i = free_dim + 1; i < input_rank; ++i) {
    const ProofResult full = ProveValidExtentEqual(src_valid[i], in_shape[i]);
    CHECK_SPAN(full == ProofResult::kTrue, span)
        << op_name << ": input dim " << i << " is not provably fully valid below free dim " << free_dim
        << " (valid=" << PythonPrint(src_valid[i]) << ", shape=" << PythonPrint(in_shape[i])
        << (full == ProofResult::kUnknown ? "; symbolic equality cannot be proven"
                                          : "; the extents are provably different")
        << "). The valid region is not a contiguous flat prefix and cannot be reshaped "
           "without widening.";
  }

  int64_t trailing_volume = 1;
  for (size_t i = free_dim + 1; i < input_rank; ++i) {
    auto extent = As<ConstInt>(in_shape[i]);
    CHECK_SPAN(extent, span) << op_name << ": cannot map a partial valid region when physical dim " << i
                             << " is dynamic";
    trailing_volume *= extent->value_;
  }
  auto free_physical = As<ConstInt>(in_shape[free_dim]);
  CHECK_SPAN(free_physical, span)
      << op_name << ": cannot map a partial valid region when the free physical dimension is dynamic";

  std::vector<int64_t> target(new_shape.size());
  for (size_t i = 0; i < new_shape.size(); ++i) {
    auto extent = As<ConstInt>(new_shape[i]);
    CHECK_SPAN(extent, span) << op_name << ": cannot map a partial valid region onto dynamic target dim "
                             << i;
    target[i] = extent->value_;
  }

  std::vector<int64_t> suffix(target.size(), 1);
  for (size_t i = target.size(); i-- > 0;) {
    suffix[i] = i + 1 < target.size() ? suffix[i + 1] * target[i + 1] : 1;
  }

  auto build_box = [&](size_t output_free_dim, const ExprPtr& free_extent) {
    std::vector<ExprPtr> output(new_shape.size());
    for (size_t i = 0; i < new_shape.size(); ++i) {
      if (i < output_free_dim) {
        output[i] = std::make_shared<ConstInt>(1, DataType::INDEX, span);
      } else if (i == output_free_dim) {
        output[i] = free_extent;
      } else {
        output[i] = new_shape[i];
      }
    }
    return output;
  };

  const ExprPtr& free_valid = src_valid[free_dim];
  if (auto extent = As<ConstInt>(free_valid)) {
    const int64_t prefix_elements = extent->value_ * trailing_volume;
    if (prefix_elements == 0) {
      return std::vector<ExprPtr>(new_shape.size(), std::make_shared<ConstInt>(0, DataType::INDEX, span));
    }
    for (size_t i = 0; i < new_shape.size(); ++i) {
      if (prefix_elements % suffix[i] != 0) continue;
      const int64_t output_extent = prefix_elements / suffix[i];
      if (output_extent >= 1 && output_extent <= target[i]) {
        return build_box(i, std::make_shared<ConstInt>(output_extent, DataType::INDEX, span));
      }
    }
    CHECK_SPAN(false, span) << op_name << ": the input valid prefix of " << prefix_elements
                            << " elements does not align to a rectangular sub-region of target shape "
                            << FormatShape(new_shape);
  }

  for (size_t i = 0; i < new_shape.size(); ++i) {
    if (suffix[i] == trailing_volume && free_physical->value_ <= target[i]) {
      return build_box(i, free_valid);
    }
  }
  CHECK_SPAN(false, span)
      << op_name << ": cannot prove the dynamic valid prefix reshapes to a representable rectangle; no "
      << "target axis preserves trailing volume " << trailing_volume << " with room for the free extent";
  return {};
}

std::vector<ExprPtr> ComputeConcatValidShape(const std::vector<ExprPtr>& lhs_shape,
                                             const std::vector<ExprPtr>& lhs_valid,
                                             const std::vector<ExprPtr>& rhs_shape,
                                             const std::vector<ExprPtr>& rhs_valid, const Span& span,
                                             const std::string& op_name) {
  CHECK_SPAN(lhs_shape.size() == 2 && rhs_shape.size() == 2, span)
      << op_name << ": concat validity inference requires 2-D operands";
  ValidateValidShapeBounds(lhs_valid, lhs_shape, span, op_name);
  ValidateValidShapeBounds(rhs_valid, rhs_shape, span, op_name);

  const ProofResult physical_rows = ProveValidExtentEqual(lhs_shape[0], rhs_shape[0]);
  CHECK_SPAN(physical_rows == ProofResult::kTrue, span)
      << op_name << ": cannot prove matching physical row counts (" << PythonPrint(lhs_shape[0]) << " vs "
      << PythonPrint(rhs_shape[0]) << ")";

  const ProofResult valid_rows = ProveValidExtentEqual(lhs_valid[0], rhs_valid[0]);
  CHECK_SPAN(valid_rows == ProofResult::kTrue, span)
      << op_name << ": operands disagree on the valid row extent (src0 valid " << FormatShape(lhs_valid)
      << ", src1 valid " << FormatShape(rhs_valid) << ")"
      << (valid_rows == ProofResult::kUnknown ? "; symbolic equality cannot be proven" : "");

  const ProofResult lhs_columns_full = ProveValidExtentEqual(lhs_valid[1], lhs_shape[1]);
  CHECK_SPAN(lhs_columns_full == ProofResult::kTrue, span)
      << op_name << ": the first operand is only partially valid along the concatenation (column) dimension "
      << "(src0 valid " << FormatShape(lhs_valid) << ", physical shape " << FormatShape(lhs_shape) << ")"
      << (lhs_columns_full == ProofResult::kUnknown ? "; symbolic equality cannot be proven" : "");

  return {lhs_valid[0], FoldAdd(lhs_shape[1], rhs_valid[1], span)};
}

ExprPtr IntersectWindowValidDim(const ExprPtr& src_valid, const ExprPtr& offset, const ExprPtr& window,
                                const ExprPtr& valid_arg, const Span& span) {
  auto vt = As<ConstInt>(src_valid);
  auto off = As<ConstInt>(offset);
  auto sh = As<ConstInt>(window);
  auto va = As<ConstInt>(valid_arg);

  // A provably-negative constant offset makes the window non-origin-anchored: its
  // valid cells begin at local index ``-offset`` (local rows [0, -offset) map to
  // source rows [offset, 0), which are out of bounds — padding), so the valid
  // region no longer starts at the result origin and a per-dim ``valid_shape``
  // cannot represent it. Reject rather than widen it to a full/anchored extent
  // (``valid_shape`` North Star: never default to the full shape). A symbolic
  // offset cannot be proven negative and defers to runtime. (``tile.extract``
  // additionally guards this earlier; the shared clip is the sole guard for the
  // ``tensor.slice`` / ``tile.slice`` clamp paths.)
  CHECK_SPAN(!off || off->value_ >= 0, span)
      << "slice/load window offset is negative (" << off->value_
      << "): a window that starts before the source origin is not origin-anchored, so "
         "its valid region cannot be expressed as a per-dim valid_shape (local cells "
         "[0, "
      << (-off->value_) << ") would read out-of-bounds source padding). Offsets must be >= 0.";

  // Fully static: fold to a single ConstInt (no Min/Sub/Max nodes).
  if (vt && off && sh && va) {
    int64_t avail = vt->value_ - off->value_;
    if (avail < 0) avail = 0;
    if (avail > sh->value_) avail = sh->value_;
    int64_t result = std::min(avail, va->value_);
    return std::make_shared<ConstInt>(result, DataType::INDEX, span);
  }

  // avail = src_valid - offset, floored at 0. A zero offset means the window starts
  // at the source origin, so avail == src_valid (already >= 0) with no subtract/floor.
  ExprPtr avail;
  if (off && off->value_ == 0) {
    avail = src_valid;
  } else {
    ExprPtr sub = MakeSub(src_valid, offset, span);
    avail = MakeMax(sub, std::make_shared<ConstInt>(0, DataType::INDEX, span), span);
  }

  // inherited = min(avail, window); when avail is structurally the window the clamp
  // is a no-op.
  ExprPtr inherited = AreExprsEqual(avail, window) ? window : MakeMin(avail, window, span);

  // result = min(valid_arg, inherited). When valid_arg is the full window or already
  // equals inherited it adds no narrowing beyond inherited (already <= window) — drop
  // the min.
  if (AreExprsEqual(valid_arg, window) || AreExprsEqual(valid_arg, inherited)) {
    return inherited;
  }
  return MakeMin(valid_arg, inherited, span);
}

std::vector<ExprPtr> ComputeAssembleUnionValidShape(const std::vector<ExprPtr>& target_valid,
                                                    const std::vector<ExprPtr>& source_valid,
                                                    const std::vector<ExprPtr>& offset,
                                                    const std::vector<ExprPtr>& shape, const Span& span,
                                                    const std::string& op_name) {
  const size_t ndim = shape.size();
  // The written region is [offset, offset + valid(source)); its per-dim bounding
  // box with the target's existing valid region needs a source valid extent and an
  // offset for every target dim, so all three share the target rank. A rank
  // mismatch yields a region that a per-dim valid_shape cannot represent — reject
  // (never widen to the full shape). CHECK, not INTERNAL_CHECK: an assemble whose
  // operands disagree in rank is a user-authored (DSL) error.
  CHECK_SPAN(target_valid.size() == ndim, span)
      << op_name << ": target valid_shape rank (" << target_valid.size()
      << ") must match the target physical rank (" << ndim << ")";
  CHECK_SPAN(source_valid.size() == ndim && offset.size() == ndim, span)
      << op_name << ": source valid_shape rank (" << source_valid.size() << ") and offset rank ("
      << offset.size() << ") must both match the target physical rank (" << ndim
      << ") to infer the assembled valid region";

  // Keep the shared region helper memory-safe even when an internal operator
  // calls it directly instead of going through tensor.assemble/tile.assemble.
  // Unknown symbolic offsets defer, but a provably-negative offset can never
  // describe an origin-anchored write.
  auto zero = std::make_shared<ConstInt>(0, DataType::INDEX, span);
  for (size_t i = 0; i < ndim; ++i) {
    const ProofResult non_negative = ProveValidExtentLessEqual(zero, offset[i]);
    CHECK_SPAN(non_negative != ProofResult::kFalse, span)
        << op_name << ": offset[" << i << "] must be >= 0, got " << PythonPrint(offset[i]);
  }

  // Compile-time value of each operand dim (nullopt when symbolic).
  std::vector<std::optional<int64_t>> off_c(ndim), src_c(ndim), tgt_c(ndim), shp_c(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    if (auto c = As<ConstInt>(offset[i])) off_c[i] = c->value_;
    if (auto c = As<ConstInt>(source_valid[i])) src_c[i] = c->value_;
    if (auto c = As<ConstInt>(target_valid[i])) tgt_c[i] = c->value_;
    if (auto c = As<ConstInt>(shape[i])) shp_c[i] = c->value_;
  }

  // A provably-empty source (some valid extent == 0) writes nothing: the written
  // rectangle [offset, offset + valid(source)) is degenerate, so the result is the
  // target's valid region unchanged. Short-circuit — the bounding box below assumes
  // a non-empty write and would otherwise widen a no-op assemble (and the no-gap
  // check would false-reject a shifted empty write). target_valid already satisfies
  // the verifier invariant valid <= shape, so it needs no re-clamp.
  for (size_t i = 0; i < ndim; ++i) {
    // ``optional == value`` is false when empty, i.e. exactly "provably zero".
    if (src_c[i] == 0) {
      return target_valid;
    }
  }

  // User error: the written region [offset, offset + valid(source)) must lie within
  // the target's physical extent. Only provable when the offset, the source extent
  // and the physical shape are all static; a symbolic write is clamped by the min
  // below.
  for (size_t i = 0; i < ndim; ++i) {
    const std::optional<int64_t> off = off_c[i];
    const std::optional<int64_t> src = src_c[i];
    const std::optional<int64_t> shp = shp_c[i];
    if (off && src && shp) {
      CHECK_SPAN(*off + *src <= *shp, span)
          << op_name << ": dim " << i << " write region [" << *off << ", " << (*off + *src)
          << ") exceeds the target physical extent " << *shp << " (out-of-bounds assemble)";
    }
  }

  // Representability. assemble's true valid region is the UNION of the target's
  // valid rectangle [0, target_valid) and the written rectangle
  // [offset, offset + source_valid). A per-dim valid_shape describes ONE
  // origin-anchored rectangle, so it can represent that union only when the union
  // IS such a rectangle — exactly the per-dim bounding box computed below.
  // Otherwise the box would mark cells written by neither operand as valid
  // (silently-wrong data); per the valid_shape North Star that is REJECTED, never
  // approximated by its box. We therefore accept the box ONLY when the union is
  // *provably* a rectangle; a union whose representability cannot be proven at
  // compile time (e.g. a symbolic extent that may or may not line up) is rejected
  // rather than widened. CHECK_SPAN, not INTERNAL: the offending assemble is
  // user-authored via ``pl.tile.assemble`` / ``pl.tensor.assemble``.
  //
  // The proof is PER-DIM and must NOT be gated on a global "all dims static" flag:
  // a single symbolic passenger dim (e.g. a shared, fully-covering batch extent)
  // must not disable the rejection of a provably non-rectangular union in the
  // remaining static dims. A global gate silently widened a provable static column
  // grow / L-shape whenever any unrelated dim was dynamic.
  //
  // Per-dim provable predicates (constants + structural equality). ``off_c`` /
  // ``src_c`` / ``tgt_c`` are the ConstInt values (nullopt when symbolic).
  auto off_zero = [&](size_t i) { return off_c[i] == 0; };
  // "passenger" (equal): the written interval [offset, offset+source) provably
  // coincides with the target's valid interval [0, target) AND the box — offset 0
  // and source == target — so the dim neither grows nor shrinks the region.
  auto is_equal = [&](size_t i) { return off_zero(i) && AreExprsEqual(source_valid[i], target_valid[i]); };
  // offset + source provably <= target: the write does not grow the region here.
  auto within = [&](size_t i) {
    if (is_equal(i)) return true;
    const std::optional<int64_t> off = off_c[i];
    const std::optional<int64_t> src = src_c[i];
    const std::optional<int64_t> tgt = tgt_c[i];
    return off && src && tgt && (*off + *src <= *tgt);
  };
  // offset 0 AND target provably <= source: the source covers the target here. A
  // provably-empty target dim (target == 0) is covered by any source (0 <= source).
  auto covers = [&](size_t i) {
    if (!off_zero(i)) return false;
    if (is_equal(i) || tgt_c[i] == 0) return true;
    const std::optional<int64_t> src = src_c[i];
    const std::optional<int64_t> tgt = tgt_c[i];
    return src && tgt && (*tgt <= *src);
  };
  // no gap in a free dim: the write starts within the target's valid extent so
  // [0, target) and [offset, offset+source) touch/overlap (offset <= target).
  //
  // The structural-equality arm proves ``offset == target_valid`` for SYMBOLIC
  // extents, which is the contiguous-append idiom an accumulator loop emits:
  // ``assemble(acc /*valid [v, ..]*/, src, offset=[v, ..])`` appends exactly at the
  // current boundary, so the union stays an origin-anchored rectangle even though
  // neither extent is a ConstInt. Without it a provable append is rejected merely
  // for being dynamic. Two *different* symbolic extents remain unprovable and are
  // still rejected — this arm only discharges the case where the two expressions
  // are the same value.
  auto no_gap = [&](size_t i) {
    if (off_zero(i) || AreExprsEqual(offset[i], target_valid[i])) return true;
    const std::optional<int64_t> off = off_c[i];
    const std::optional<int64_t> tgt = tgt_c[i];
    return off && tgt && (*off <= *tgt);
  };

  // The union is PROVABLY an origin-anchored rectangle iff one of these holds:
  bool representable = false;
  // (T) target provably fully valid (valid == physical shape): the source lies
  // within the shape, so the union is the full target and the box clamps to shape.
  // Accepts a dynamic (symbolic) source accumulated into a fully-valid accumulator.
  if (AreExprVectorsEqual(target_valid, shape)) representable = true;
  // (S⊆T) source provably within the target in every dim: union == target.
  if (!representable) {
    representable = true;
    for (size_t i = 0; i < ndim; ++i) representable = representable && within(i);
  }
  // (T⊆S) target provably covered by the source in every dim (origin-anchored):
  // union == source. Accepts an empty accumulator (target valid 0) with any source.
  if (!representable) {
    representable = true;
    for (size_t i = 0; i < ndim; ++i) representable = representable && covers(i);
  }
  // (single free dim) exactly one dim is not a passenger, and that free dim has no
  // gap: the union stacks contiguously along it while every other dim coincides.
  if (!representable) {
    size_t free_count = 0, free_dim = 0;
    for (size_t i = 0; i < ndim; ++i) {
      if (!is_equal(i)) {
        ++free_count;
        free_dim = i;
      }
    }
    if (free_count == 1 && no_gap(free_dim)) representable = true;
  }

  CHECK_SPAN(representable, span)
      << op_name << ": the assembled valid region is not representable as a single origin-anchored "
      << "valid_shape. assemble unions the target's valid rectangle [0, valid(target)) with the "
      << "written rectangle [offset, offset + valid(source)); a valid_shape can express only ONE "
      << "origin-anchored rectangle, so a non-rectangular union — a gap, an L-shape, or one whose "
      << "representability cannot be proven at compile time — would mark cells written by neither "
      << "operand as valid, which is never widened away (valid_shape North Star). Restructure so "
      << "each assemble fills a fully-valid target, overwrites it from the origin with a source that "
      << "covers it, or grows a single dimension contiguously (offset <= the target's valid extent "
      << "there) while fully covering the target's valid extent in every other dimension. target "
      << "valid=" << FormatShape(target_valid) << ", source valid=" << FormatShape(source_valid)
      << ", offset=" << FormatShape(offset) << ", shape=" << FormatShape(shape);

  // Produce the clamped per-dim bounding box min(shape, max(target_valid,
  // offset + source_valid)), folding when both operands are constant so a
  // partially-static dim never carries a Max/Add of two ConstInts.
  std::vector<ExprPtr> out_valid(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    // Fold a zero offset (offset + x == x) so a fresh accumulator's [0, ...] offset
    // stays clean.
    ExprPtr written = (off_c[i] == 0) ? source_valid[i] : FoldAdd(offset[i], source_valid[i], span);
    ExprPtr box = FoldMax(target_valid[i], written, span);
    out_valid[i] = FoldMin(shape[i], box, span);
  }
  return out_valid;
}

void CheckMatMulValidKCompat(const ExprPtr& k_lhs_valid, const ExprPtr& k_rhs_valid, const Span& span,
                             const std::string& op_name) {
  const ProofResult proof = ProveValidExtentLessEqual(k_lhs_valid, k_rhs_valid);
  if (proof == ProofResult::kFalse) {
    CHECK_SPAN(false, span)
        << op_name << ": lhs valid contraction length K=" << PythonPrint(k_lhs_valid)
        << " exceeds rhs valid row extent=" << PythonPrint(k_rhs_valid)
        << ". PTO matmul contracts over lhs.valid_K, so rhs must provide at least that many valid rows.";
  }
  if (proof == ProofResult::kUnknown) {
    CHECK_SPAN(false, span) << op_name
                            << ": cannot prove lhs valid contraction length K=" << PythonPrint(k_lhs_valid)
                            << " is <= rhs valid row extent=" << PythonPrint(k_rhs_valid)
                            << ". Use the same extent expression, make the relation statically provable, or "
                               "guard it before matmul.";
  }
}

std::vector<ExprPtr> ComputeMatMulBiasValidShape(const std::vector<ExprPtr>& lhs_valid,
                                                 const std::vector<ExprPtr>& rhs_valid,
                                                 const std::vector<ExprPtr>& bias_valid, const Span& span) {
  // C[i,j] = (A@B)[i,j] + bias[0,j]. The matmul term is valid over rows lhs_valid[M]
  // and cols rhs_valid[N]; the bias term is real only where bias[0,j] is real — column
  // j < valid(bias)[N] AND the single bias row itself valid. A column whose bias is
  // padding yields C[:,j] = A@B + garbage, which is NOT the intended A@B + bias, so it is
  // invalid. The output N extent therefore INTERSECTS the matmul N with the bias N —
  // never widening past the real bias columns (the North Star forbids claiming padding
  // as valid). M is unaffected: the [1, N] bias is broadcast down the rows.
  //
  // bias_valid[1] <= bias physical N == output physical N and rhs_valid[1] <= output
  // physical N, so the intersected N is already within the output shape — no re-clamp
  // to the physical shape is needed for the verifier invariant.
  auto one = std::make_shared<ConstInt>(1, DataType::INDEX, span);
  const ProofResult row_valid = ProveValidExtentEqual(bias_valid[0], one);
  CHECK_SPAN(row_valid == ProofResult::kTrue, span)
      << "matmul_bias: the broadcast bias singleton row must be provably valid (valid extent == 1), got "
      << PythonPrint(bias_valid[0])
      << (row_valid == ProofResult::kUnknown ? "; symbolic equality cannot be proven" : "");
  return {lhs_valid[0], FoldMin(rhs_valid[1], bias_valid[1], span)};
}

std::shared_ptr<const TileType> PickElementwiseLayoutSource(
    const std::vector<ExprPtr>& out_shape, const std::vector<std::shared_ptr<const TileType>>& operands) {
  INTERNAL_CHECK(!operands.empty()) << "PickElementwiseLayoutSource requires at least one tile operand";
  for (const auto& op : operands) {
    // The shaping operand is not broadcast in any dim: its physical shape equals
    // the broadcast output. Its layout is the one the full-shaped result should
    // carry — a shape-1 broadcast operand's layout would leak otherwise.
    if (AreExprVectorsEqual(op->shape_, out_shape)) return op;
  }
  return nullptr;
}

std::vector<ExprPtr> ComputeBroadcastElementwiseValidShape(
    const std::vector<ExprPtr>& out_shape, const std::vector<std::vector<ExprPtr>>& operand_shapes,
    const std::vector<std::vector<ExprPtr>>& operand_valids, const Span& span, const std::string& op_name) {
  INTERNAL_CHECK_SPAN(!operand_shapes.empty() && operand_shapes.size() == operand_valids.size(), span)
      << op_name << ": ComputeBroadcastElementwiseValidShape requires matching non-empty operand lists";
  const size_t out_ndim = out_shape.size();

  // Per-operand effective valid_shape, physical shape, and right-alignment offset
  // (broadcasting aligns shapes from the trailing dim, so a lower-rank operand's
  // leading dims are implicit size-1 broadcast dims).
  struct Info {
    const std::vector<ExprPtr>* valid;
    const std::vector<ExprPtr>* shape;
    size_t offset;
  };
  std::vector<Info> infos;
  infos.reserve(operand_shapes.size());
  for (size_t k = 0; k < operand_shapes.size(); ++k) {
    INTERNAL_CHECK_SPAN(operand_shapes[k].size() <= out_ndim, span)
        << op_name << ": operand rank " << operand_shapes[k].size() << " exceeds the broadcast output rank "
        << out_ndim;
    ValidateValidShapeBounds(operand_valids[k], operand_shapes[k], span, op_name);
    infos.push_back({&operand_valids[k], &operand_shapes[k], out_ndim - operand_shapes[k].size()});
  }

  // Every non-broadcast contributor must be provably equal. An explicit singleton
  // broadcast is exempt only when its sole element is provably valid; otherwise
  // every output cell would read padding from that operand.
  std::vector<ExprPtr> out_valid = out_shape;
  for (size_t i = 0; i < out_ndim; ++i) {
    const bool out_is_one = IsConstOne(out_shape[i]);
    ExprPtr rep = nullptr;
    for (const auto& info : infos) {
      if (i < info.offset) continue;  // operand lacks this dim => implicit size-1 broadcast
      const size_t d = i - info.offset;
      const ExprPtr& v = (*info.valid)[d];

      if (IsConstOne((*info.shape)[d]) && !out_is_one) {
        const ProofResult singleton_valid = ProveValidExtentEqual(v, (*info.shape)[d]);
        CHECK_SPAN(singleton_valid == ProofResult::kTrue, span)
            << op_name << ": broadcast operand at dim " << i << " has physical extent 1 but valid extent "
            << PythonPrint(v)
            << ". Its singleton value must be provably valid (valid extent == 1); an empty or unprovable "
               "singleton would broadcast padding into the output.";
        continue;
      }

      INTERNAL_CHECK_SPAN(DimensionsEqual((*info.shape)[d], out_shape[i]), span)
          << op_name << ": non-singleton operand shape does not match broadcast output at dim " << i;
      if (!rep) {
        rep = v;
        continue;
      }

      const ProofResult agreement = ProveValidExtentEqual(rep, v);
      CHECK_SPAN(agreement == ProofResult::kTrue, span)
          << op_name << ": operands disagree on the valid extent along dim " << i << " (" << PythonPrint(rep)
          << " vs " << PythonPrint(v) << "). "
          << (agreement == ProofResult::kUnknown
                  ? "Their symbolic equality cannot be proven and this op emits no runtime guard. "
                  : "The extents are provably different. ")
          << "Make the operands use the same valid extent expression or fillpad them before the operation.";

      // A proven constant representative is deterministic under operand reversal.
      if (!As<ConstInt>(rep) && As<ConstInt>(v)) rep = v;
    }
    if (rep) out_valid[i] = rep;
  }
  return out_valid;
}

namespace {

void CheckInputFullyValid(const std::vector<ExprPtr>& shape, const std::vector<ExprPtr>& valid,
                          const std::string& op_name, const std::string& arg_desc, const Span& span) {
  ValidateValidShapeBounds(valid, shape, span, op_name);
  for (size_t i = 0; i < shape.size(); ++i) {
    const ProofResult full = ProveValidExtentEqual(valid[i], shape[i]);
    CHECK_SPAN(full == ProofResult::kTrue, span)
        << op_name << ": " << arg_desc << " is not provably fully valid at dim " << i
        << " (valid=" << PythonPrint(valid[i]) << ", physical=" << PythonPrint(shape[i]) << ")"
        << (full == ProofResult::kUnknown ? "; symbolic equality cannot be proven" : "")
        << ". This op cannot safely consume a partial valid_shape region; fillpad the input first.";
  }
}

}  // namespace

void CheckTensorInputFullyValid(const std::shared_ptr<const TensorType>& t, const std::string& op_name,
                                const std::string& arg_desc, const Span& span) {
  INTERNAL_CHECK(t) << op_name << ": CheckTensorInputFullyValid requires a non-null tensor type";
  CheckInputFullyValid(t->shape_, GetValidShape(t), op_name, arg_desc, span);
}

void CheckTileInputFullyValid(const std::shared_ptr<const TileType>& t, const std::string& op_name,
                              const std::string& arg_desc, const Span& span) {
  INTERNAL_CHECK(t) << op_name << ": CheckTileInputFullyValid requires a non-null tile type";
  CheckInputFullyValid(t->shape_, GetValidShape(t), op_name, arg_desc, span);
}

void CheckBatchAxesFullyValid(const std::vector<ExprPtr>& batch_shape,
                              const std::vector<ExprPtr>& batch_valid, const Span& span,
                              const std::string& op_name, const std::string& operand_name) {
  ValidateValidShapeBounds(batch_valid, batch_shape, span, op_name);
  for (size_t i = 0; i < batch_shape.size(); ++i) {
    const ProofResult full = ProveValidExtentEqual(batch_valid[i], batch_shape[i]);
    CHECK_SPAN(full == ProofResult::kTrue, span)
        << op_name << ": " << operand_name << " batch axis " << i
        << " must be provably full before ND-to-2D flattening (valid=" << PythonPrint(batch_valid[i])
        << ", physical=" << PythonPrint(batch_shape[i]) << ")"
        << (full == ProofResult::kUnknown ? "; symbolic equality cannot be proven" : "");
  }
}

TileView DeduceBroadcastElementwiseTileView(const std::vector<ExprPtr>& out_shape,
                                            const std::vector<std::shared_ptr<const TileType>>& operands,
                                            const Span& span, const std::string& op_name) {
  INTERNAL_CHECK(!operands.empty())
      << op_name << ": DeduceBroadcastElementwiseTileView requires at least one tile operand";

  // Delegate the valid-region agreement to the shared type-agnostic helper so the
  // tile and tensor elementwise deducers stay bit-for-bit identical.
  std::vector<std::vector<ExprPtr>> shapes;
  std::vector<std::vector<ExprPtr>> valids;
  shapes.reserve(operands.size());
  valids.reserve(operands.size());
  for (const auto& op : operands) {
    shapes.push_back(op->shape_);
    valids.push_back(GetValidShape(op));
  }

  TileView view;
  view.valid_shape = ComputeBroadcastElementwiseValidShape(out_shape, shapes, valids, span, op_name);
  if (auto layout_source = PickElementwiseLayoutSource(out_shape, operands)) {
    InheritFreshTileComputeLayout(view, layout_source);
  } else {
    view.blayout = tile_view_semantics::InferImplicitTileLayoutFromShape(out_shape);
  }
  return view;
}

BroadcastResult BroadcastShapes(const std::vector<ExprPtr>& shape1, const std::vector<ExprPtr>& shape2) {
  // Handle empty shapes
  if (shape1.empty() && shape2.empty()) {
    return BroadcastResult::Success({});
  }
  if (shape1.empty()) {
    return BroadcastResult::Success(shape2);
  }
  if (shape2.empty()) {
    return BroadcastResult::Success(shape1);
  }

  // Broadcast from right to left
  size_t max_ndim = std::max(shape1.size(), shape2.size());
  std::vector<ExprPtr> result_shape;
  result_shape.reserve(max_ndim);

  for (size_t i = 0; i < max_ndim; ++i) {
    // Get dimensions from right to left
    int64_t idx1 = static_cast<int64_t>(shape1.size()) - 1 - i;  // NOLINT
    int64_t idx2 = static_cast<int64_t>(shape2.size()) - 1 - i;  // NOLINT

    ExprPtr dim1 = (idx1 >= 0) ? shape1[idx1] : nullptr;
    ExprPtr dim2 = (idx2 >= 0) ? shape2[idx2] : nullptr;

    // If one dimension is missing, use the other
    if (!dim1) {
      result_shape.push_back(dim2);
      continue;
    }
    if (!dim2) {
      result_shape.push_back(dim1);
      continue;
    }

    // Check if dimensions are equal
    if (DimensionsEqual(dim1, dim2)) {
      result_shape.push_back(dim1);
      continue;
    }

    // Check if either dimension is 1 (broadcastable)
    auto const_dim1 = GetConstantDimension(dim1);
    auto const_dim2 = GetConstantDimension(dim2);

    if (const_dim1 && *const_dim1 == 1) {
      result_shape.push_back(dim2);
      continue;
    }
    if (const_dim2 && *const_dim2 == 1) {
      result_shape.push_back(dim1);
      continue;
    }

    // Dimensions are incompatible for broadcasting
    std::ostringstream oss;
    oss << "Cannot broadcast shapes: dimension " << i << " mismatch";
    return BroadcastResult::Failure(oss.str());
  }

  // Reverse result since we built it from right to left
  std::reverse(result_shape.begin(), result_shape.end());
  return BroadcastResult::Success(std::move(result_shape));
}

std::optional<DataType> PromoteDataTypes(DataType dtype1, DataType dtype2) {
  // If types are the same, return that type
  if (dtype1 == dtype2) {
    return dtype1;
  }

  // Float types take precedence
  bool is_float1 = dtype1.IsFloat();
  bool is_float2 = dtype2.IsFloat();

  if (is_float1 && !is_float2) {
    return dtype1;
  }
  if (is_float2 && !is_float1) {
    return dtype2;
  }

  // Both are floats or both are integers
  // Return the larger type
  size_t bits1 = dtype1.GetBit();
  size_t bits2 = dtype2.GetBit();

  if (bits1 > bits2) {
    return dtype1;
  }
  if (bits2 > bits1) {
    return dtype2;
  }

  // Same size - prefer signed over unsigned for integers
  if (!is_float1 && dtype1.IsSignedInt()) {
    return dtype1;
  }
  if (!is_float2 && dtype2.IsSignedInt()) {
    return dtype2;
  }

  // Default to first type
  return dtype1;
}

bool CheckTypeCompatibility(const TypePtr& type1, const TypePtr& type2) {
  // Check if both are scalar types
  auto scalar1 = As<ScalarType>(type1);
  auto scalar2 = As<ScalarType>(type2);
  if (scalar1 && scalar2) {
    return true;
  }

  // Check if both are tensor types
  auto tensor1 = As<TensorType>(type1);
  auto tensor2 = As<TensorType>(type2);
  if (tensor1 && tensor2) {
    return true;
  }

  // Check if both are tile types
  auto tile1 = As<TileType>(type1);
  auto tile2 = As<TileType>(type2);
  if (tile1 && tile2) {
    return true;
  }

  // Types are not compatible
  return false;
}

std::optional<DataType> ExtractDataType(const TypePtr& type) {
  // Try ScalarType
  if (auto scalar = As<ScalarType>(type)) {
    return scalar->dtype_;
  }

  // Try TensorType
  if (auto tensor = As<TensorType>(type)) {
    return tensor->dtype_;
  }

  // Try TileType
  if (auto tile = As<TileType>(type)) {
    return tile->dtype_;
  }

  return std::nullopt;
}

std::vector<ExprPtr> ExtractShape(const TypePtr& type) {
  // Try TensorType
  if (auto tensor = As<TensorType>(type)) {
    return tensor->shape_;
  }

  // Try TileType
  if (auto tile = As<TileType>(type)) {
    return tile->shape_;
  }

  // Not a shaped type
  return {};
}

std::optional<int64_t> GetConstantDimension(const ExprPtr& dim) {
  // Try to cast to ConstInt
  if (auto const_int = As<ConstInt>(dim)) {
    return const_int->value_;
  }

  // Not a constant
  return std::nullopt;
}

bool DimensionsEqual(const ExprPtr& dim1, const ExprPtr& dim2) {
  return ProveValidExtentEqual(dim1, dim2) == ProofResult::kTrue;
}

bool IsBroadcastable(const ExprPtr& source_dim, const ExprPtr& target_dim) {
  // If dimensions are equal, they're broadcastable
  if (DimensionsEqual(source_dim, target_dim)) {
    return true;
  }

  // Check if source is constant 1
  auto const_source = GetConstantDimension(source_dim);
  if (const_source && *const_source == 1) {
    return true;
  }

  // Check if target is constant 1
  auto const_target = GetConstantDimension(target_dim);
  if (const_target && *const_target == 1) {
    return true;
  }

  return false;
}

std::string FormatShape(const std::vector<ExprPtr>& shape) {
  if (shape.empty()) {
    return "[]";
  }

  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << PythonPrint(shape[i]);
  }
  oss << "]";
  return oss.str();
}

// ============================================================================
// Slice rank-reduction (drop_dims) helpers
// ============================================================================

std::vector<int64_t> ParseSliceDropDims(const ExprPtr& drop_dims_arg, const std::vector<ExprPtr>& full_shape,
                                        const std::string& op_name) {
  if (!drop_dims_arg) {
    return {};
  }
  auto tuple = As<MakeTuple>(drop_dims_arg);
  CHECK(tuple) << op_name << " drop_dims must be a MakeTuple of compile-time int constants";

  std::vector<int64_t> axes;
  axes.reserve(tuple->elements_.size());
  std::vector<bool> seen(full_shape.size(), false);
  for (size_t i = 0; i < tuple->elements_.size(); ++i) {
    auto const_int = As<ConstInt>(tuple->elements_[i]);
    CHECK(const_int) << op_name << " drop_dims element " << i << " must be a compile-time int constant";
    int64_t axis = const_int->value_;
    CHECK(axis >= 0 && axis < static_cast<int64_t>(full_shape.size()))
        << op_name << " drop_dims index " << axis << " out of range for rank " << full_shape.size();
    CHECK(!seen[static_cast<size_t>(axis)]) << op_name << " drop_dims index " << axis << " is repeated";
    seen[static_cast<size_t>(axis)] = true;
    auto dim = GetConstantDimension(full_shape[static_cast<size_t>(axis)]);
    CHECK(dim.has_value() && *dim == 1)
        << op_name << " drop_dims index " << axis
        << " must select a static unit dimension (rank reduction only erases size-1 dims), but dim " << axis
        << " is " << (dim.has_value() ? std::to_string(*dim) : std::string("dynamic"));
    axes.push_back(axis);
  }
  std::sort(axes.begin(), axes.end());
  return axes;
}

std::vector<ExprPtr> ApplyDropDims(const std::vector<ExprPtr>& shape, const std::vector<int64_t>& drop_dims) {
  if (drop_dims.empty()) {
    return shape;
  }
  std::vector<bool> drop(shape.size(), false);
  for (int64_t d : drop_dims) {
    if (d >= 0 && d < static_cast<int64_t>(shape.size())) {
      drop[static_cast<size_t>(d)] = true;
    }
  }
  std::vector<ExprPtr> result;
  result.reserve(shape.size() - drop_dims.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    if (!drop[i]) {
      result.push_back(shape[i]);
    }
  }
  return result;
}

// ============================================================================
// Cross-function call return type deduction
// ============================================================================

std::vector<TypePtr> DeduceCallReturnType(const std::vector<VarPtr>& callee_params,
                                          const std::vector<ExprPtr>& args,
                                          const std::vector<TypePtr>& return_types) {
  if (return_types.empty()) return return_types;
  CHECK(callee_params.size() == args.size())
      << "DeduceCallReturnType: callee_params size (" << callee_params.size() << ") must match args size ("
      << args.size() << ")";

  // 1. Build Var* -> ExprPtr mapping from param shapes vs arg shapes
  std::unordered_map<const Var*, ExprPtr> var_map;
  size_t n = callee_params.size();
  for (size_t i = 0; i < n; ++i) {
    auto param_type = callee_params[i]->GetType();
    auto arg_type = args[i]->GetType();
    if (!param_type || !arg_type) continue;
    auto p_shaped = As<ShapedType>(param_type);
    auto a_shaped = As<ShapedType>(arg_type);
    if (!p_shaped || !a_shaped) continue;
    size_t ndim = std::min(p_shaped->shape_.size(), a_shaped->shape_.size());
    for (size_t d = 0; d < ndim; ++d) {
      if (auto var = As<Var>(p_shaped->shape_[d])) {
        auto [it, inserted] = var_map.emplace(var.get(), a_shaped->shape_[d]);
        // Validate consistency only when both are statically known constants.
        // Symbolic dims (Vars, exprs) may be equal at runtime — defer to runtime.
        if (!inserted) {
          auto existing_const = GetConstantDimension(it->second);
          auto new_const = GetConstantDimension(a_shaped->shape_[d]);
          if (existing_const && new_const) {
            CHECK(*existing_const == *new_const)
                << "Dynamic shape variable '" << var->name_hint_
                << "' has conflicting bindings: " << FormatShape({it->second}) << " vs "
                << FormatShape({a_shaped->shape_[d]}) << " (from argument " << i << ", dimension " << d
                << ")";
          }
        }
      }
    }
  }
  if (var_map.empty()) return return_types;

  // 2. Substitution helpers
  // Substitute recursively so composite metadata expressions such as M + 1
  // and cast(M) are rebuilt with the call-site dimension. The shared helper
  // preserves every reconstructed expression's ObjectKind, dtype, and span.
  auto subst_dim = [&](const ExprPtr& dim) -> ExprPtr {
    return dim ? transform_utils::Substitute(dim, var_map) : dim;
  };

  auto subst_dims = [&](const std::vector<ExprPtr>& dims) {
    std::vector<ExprPtr> result;
    result.reserve(dims.size());
    bool changed = false;
    for (const auto& d : dims) {
      auto nd = subst_dim(d);
      if (nd.get() != d.get()) changed = true;
      result.push_back(nd);
    }
    return std::pair{std::move(result), changed};
  };

  std::function<TypePtr(const TypePtr&)> subst_type;
  subst_type = [&](const TypePtr& type) -> TypePtr {
    if (!type) return type;
    if (auto t = AsTensorTypeLike(type)) {
      auto [new_shape, changed] = subst_dims(t->shape_);
      std::optional<TensorView> new_tv = t->tensor_view_;
      if (new_tv.has_value()) {
        auto [new_stride, s_changed] = subst_dims(new_tv->stride);
        auto [new_vs, vs_changed] = subst_dims(new_tv->valid_shape);
        if (s_changed || vs_changed) {
          new_tv->stride = std::move(new_stride);
          new_tv->valid_shape = std::move(new_vs);
          changed = true;
        }
      }
      if (!changed) return type;
      if (auto distributed = As<DistributedTensorType>(type)) {
        return std::make_shared<DistributedTensorType>(std::move(new_shape), distributed->dtype_,
                                                       distributed->memref_, std::move(new_tv),
                                                       distributed->window_buffer_);
      }
      return std::make_shared<TensorType>(std::move(new_shape), t->dtype_, t->memref_, std::move(new_tv));
    }
    if (auto t = As<TileType>(type)) {
      auto [new_shape, changed] = subst_dims(t->shape_);
      std::optional<TileView> new_tv = t->tile_view_;
      if (new_tv.has_value()) {
        auto [new_vs, vs_changed] = subst_dims(new_tv->valid_shape);
        auto [new_stride, s_changed] = subst_dims(new_tv->stride);
        auto new_start = subst_dim(new_tv->start_offset);
        bool so_changed = (new_start.get() != new_tv->start_offset.get());
        if (vs_changed || s_changed || so_changed) {
          new_tv->valid_shape = std::move(new_vs);
          new_tv->stride = std::move(new_stride);
          new_tv->start_offset = std::move(new_start);
          changed = true;
        }
      }
      if (!changed) return type;
      return std::make_shared<TileType>(std::move(new_shape), t->dtype_, t->memref_, std::move(new_tv),
                                        t->memory_space_);
    }
    if (auto t = As<TupleType>(type)) {
      std::vector<TypePtr> new_types;
      bool changed = false;
      for (const auto& inner : t->types_) {
        auto nt = subst_type(inner);
        if (nt.get() != inner.get()) changed = true;
        new_types.push_back(nt);
      }
      if (!changed) return type;
      return std::make_shared<TupleType>(std::move(new_types));
    }
    return type;  // ScalarType, etc. — no shape dims
  };

  // 3. Apply to all return types
  std::vector<TypePtr> result;
  result.reserve(return_types.size());
  for (const auto& rt : return_types) {
    result.push_back(subst_type(rt));
  }
  return result;
}

}  // namespace ir
}  // namespace pypto
