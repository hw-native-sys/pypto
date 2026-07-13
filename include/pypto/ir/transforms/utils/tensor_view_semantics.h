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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_TENSOR_VIEW_SEMANTICS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_TENSOR_VIEW_SEMANTICS_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"  // CHECK
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"

namespace pypto::ir::tensor_view_semantics {

/// Compute the product of static shape dimensions; returns -1 if any dim is dynamic.
inline int64_t ComputeShapeProduct(const std::vector<ExprPtr>& shape) {
  int64_t product = 1;
  for (const auto& dim : shape) {
    auto const_dim = As<ConstInt>(dim);
    if (!const_dim) {
      return -1;
    }
    product *= const_dim->value_;
  }
  return product;
}

/// Build an INDEX-typed multiply, folding ConstInt * ConstInt and the
/// multiplicative identity (×1) so that downstream codegen sees the same
/// strides whether the source shape is static or dynamic.
///
/// Uses ``__builtin_mul_overflow`` to detect signed overflow in the constant
/// fold path; on overflow, falls back to a symbolic ``Mul`` rather than
/// silently wrapping (which would yield an incorrect stride that the
/// canonical-view verifier cannot detect).
inline ExprPtr MakeIndexMul(const ExprPtr& lhs, const ExprPtr& rhs) {
  auto const_lhs = As<ConstInt>(lhs);
  auto const_rhs = As<ConstInt>(rhs);
  if (const_lhs && const_rhs) {
    int64_t folded = 0;
    if (!__builtin_mul_overflow(const_lhs->value_, const_rhs->value_, &folded)) {
      return std::make_shared<ConstInt>(folded, DataType::INDEX, Span::unknown());
    }
    // Overflow — drop to symbolic so callers / verifiers see a non-folded form.
  }
  if (const_rhs && const_rhs->value_ == 1) return lhs;
  if (const_lhs && const_lhs->value_ == 1) return rhs;
  return std::make_shared<Mul>(lhs, rhs, DataType::INDEX, Span::unknown());
}

/// Build row-major (ND-packed) strides for the given shape:
///   strides[ndim-1] = 1; strides[i] = strides[i+1] * shape[i+1].
/// Works for both static and dynamic dims; ConstInt chains collapse via MakeIndexMul.
inline std::vector<ExprPtr> BuildRowMajorStrides(const std::vector<ExprPtr>& shape) {
  size_t ndim = shape.size();
  if (ndim == 0) return {};
  std::vector<ExprPtr> strides(ndim);
  strides[ndim - 1] = std::make_shared<ConstInt>(1, DataType::INDEX, Span::unknown());
  for (int i = static_cast<int>(ndim) - 2; i >= 0; --i) {
    strides[i] = MakeIndexMul(strides[i + 1], shape[i + 1]);
  }
  return strides;
}

/// Build packed canonical strides for the given (shape, layout).
///
/// Definitions (per RFC #1300 §2.3):
///   ND : strides[n-1] = 1; strides[k] = strides[k+1] * shape[k+1]
///   DN : strides[n-2] = 1; strides[n-1] = shape[n-2];
///        strides[n-3] = shape[n-2] * shape[n-1];
///        strides[k]   = strides[k+1] * shape[k+1]   (k = n-4 .. 0)
///   NZ : not representable as logical strides — CHECK-fails.
///
/// Throws ``pypto::ValueError`` for NZ layout or for DN layout with rank < 2.
inline std::vector<ExprPtr> BuildLogicalStridesFromLayout(const std::vector<ExprPtr>& shape,
                                                          TensorLayout layout) {
  size_t ndim = shape.size();
  if (ndim == 0) return {};

  if (layout == TensorLayout::ND) {
    return BuildRowMajorStrides(shape);
  }

  if (layout == TensorLayout::DN) {
    CHECK(ndim >= 2) << "BuildLogicalStridesFromLayout: DN layout requires rank >= 2, got " << ndim;
    std::vector<ExprPtr> strides(ndim);
    auto one = std::make_shared<ConstInt>(1, DataType::INDEX, Span::unknown());
    // Innermost two dims: stride[n-2]=1, stride[n-1]=shape[n-2].
    strides[ndim - 2] = one;
    strides[ndim - 1] = shape[ndim - 2];
    if (ndim >= 3) {
      // The dim immediately preceding the trailing pair gets the product of
      // the trailing two shape dims (one full DN-block worth of elements).
      strides[ndim - 3] = MakeIndexMul(shape[ndim - 2], shape[ndim - 1]);
      // Outer dims: row-major over the DN-block volume.
      for (int i = static_cast<int>(ndim) - 4; i >= 0; --i) {
        strides[i] = MakeIndexMul(strides[i + 1], shape[i + 1]);
      }
    }
    return strides;
  }

  // NZ is fractal; cannot be represented as flat logical strides.
  CHECK(false) << "BuildLogicalStridesFromLayout: layout '" << TensorLayoutToString(layout)
               << "' has no logical-stride representation (NZ is tile-only and fractal)";
  return {};
}

/// Static structural pattern detection from (shape, stride).
///
/// Returns:
///   - ``TensorLayout::ND`` if ``stride[-1]`` is the static constant 1
///     (covers ND-packed and ND-strided families)
///   - ``TensorLayout::DN`` if ``stride[-2]`` is the static constant 1 and
///     the trailing-stride structural condition holds
///     (covers DN-packed and DN-strided families)
///   - ``std::nullopt`` for symbolic / ambiguous / non-canonical cases
///
/// This is purely structural — it does not enforce the strided-family
/// inequality (``stride[-2] >= shape[-1]`` for ND, ``stride[-1] >= shape[-2]``
/// for DN); the verifier handles that with optional symbolic relaxation.
inline std::optional<TensorLayout> DeriveLayoutFromStrides(const std::vector<ExprPtr>& shape,
                                                           const std::vector<ExprPtr>& stride) {
  if (shape.size() != stride.size() || shape.empty()) {
    return std::nullopt;
  }
  size_t n = stride.size();

  auto trailing = As<ConstInt>(stride[n - 1]);
  if (trailing && trailing->value_ == 1) {
    return TensorLayout::ND;
  }

  if (n >= 2) {
    auto second_last = As<ConstInt>(stride[n - 2]);
    if (second_last && second_last->value_ == 1) {
      return TensorLayout::DN;
    }
  }

  return std::nullopt;
}

/// Result of a canonical-view check: ``ok`` plus a human-readable reason on
/// failure (empty when ``ok``).
struct CanonicalCheckResult {
  bool ok;
  std::string reason;
};

namespace detail {

/// Return true iff two index expressions are structurally equal as static
/// constants. Symbolic exprs are not compared (``relaxed_symbolic`` controls
/// whether the caller treats that as a pass or fail).
inline bool StaticEqual(const ExprPtr& lhs, const ExprPtr& rhs) {
  if (lhs == rhs) return true;
  auto lc = As<ConstInt>(lhs);
  auto rc = As<ConstInt>(rhs);
  return lc && rc && lc->value_ == rc->value_;
}

inline bool IsConstOne(const ExprPtr& e) {
  auto c = As<ConstInt>(e);
  return c != nullptr && c->value_ == 1;
}

/// Check ``lhs >= rhs`` when both are static ConstInt. Returns std::nullopt
/// when either operand is symbolic.
inline std::optional<bool> StaticGreaterEqual(const ExprPtr& lhs, const ExprPtr& rhs) {
  auto lc = As<ConstInt>(lhs);
  auto rc = As<ConstInt>(rhs);
  if (!lc || !rc) return std::nullopt;
  return lc->value_ >= rc->value_;
}

}  // namespace detail

/// Verify (shape, stride, layout) is canonical per RFC #1300 §2.2:
///   - rank consistency
///   - innermost-stride constant 1 at the layout-specific axis
///   - strided-family inequality (when statically decidable)
///
/// ``relaxed_symbolic`` (default true): when an inequality cannot be statically
/// decided due to symbolic dims, accept the relaxed form (only the innermost
/// stride structural equality is enforced). When false, symbolic cases that
/// cannot prove the inequality are flagged.
inline CanonicalCheckResult CheckCanonicalView(const std::vector<ExprPtr>& shape,
                                               const std::vector<ExprPtr>& stride, TensorLayout layout,
                                               bool relaxed_symbolic = true) {
  if (layout == TensorLayout::NZ) {
    return {false, "NZ layout is tile-only and not allowed on TensorType"};
  }
  // 0-rank tensors (scalar tensors) are canonical iff stride is also empty.
  // Check this before the generic stride.empty() rejection so a scalar tensor
  // doesn't trip the "must be materialized" error.
  if (shape.empty() && stride.empty()) {
    return {true, ""};
  }
  if (stride.empty()) {
    return {false, "stride is empty (must be materialized via MaterializeTensorStrides)"};
  }
  if (shape.size() != stride.size()) {
    std::ostringstream oss;
    oss << "stride rank " << stride.size() << " does not match shape rank " << shape.size();
    return {false, oss.str()};
  }

  size_t n = shape.size();

  if (layout == TensorLayout::ND) {
    if (!detail::IsConstOne(stride[n - 1])) {
      return {false, "ND layout requires innermost stride to be ConstInt(1)"};
    }
    // Outer-dim strided family: stride[k] >= stride[k+1] * shape[k+1].
    // Statically decidable cases enforce; symbolic cases pass under relaxed_symbolic.
    for (int k = static_cast<int>(n) - 2; k >= 0; --k) {
      auto packed = MakeIndexMul(stride[k + 1], shape[k + 1]);
      auto cmp = detail::StaticGreaterEqual(stride[k], packed);
      if (cmp.has_value() && !*cmp) {
        std::ostringstream oss;
        oss << "ND stride[" << k << "] is smaller than packed stride[" << (k + 1) << "] * shape[" << (k + 1)
            << "]";
        return {false, oss.str()};
      }
      if (!cmp.has_value() && !relaxed_symbolic) {
        return {false, "ND outer-dim stride relation is symbolic and cannot be statically verified"};
      }
    }
    return {true, ""};
  }

  // layout == DN
  if (n < 2) {
    return {false, "DN layout requires rank >= 2"};
  }
  if (!detail::IsConstOne(stride[n - 2])) {
    return {false, "DN layout requires stride[-2] to be ConstInt(1)"};
  }
  // Trailing stride: stride[-1] >= shape[-2].
  auto trailing_cmp = detail::StaticGreaterEqual(stride[n - 1], shape[n - 2]);
  if (trailing_cmp.has_value() && !*trailing_cmp) {
    return {false, "DN stride[-1] is smaller than shape[-2]"};
  }
  if (!trailing_cmp.has_value() && !relaxed_symbolic) {
    return {false, "DN trailing-stride relation is symbolic and cannot be statically verified"};
  }
  // Outer-dim relation: stride[k] >= stride[k+1] * shape[k+1] for k <= n-3.
  for (int k = static_cast<int>(n) - 3; k >= 0; --k) {
    auto packed = MakeIndexMul(stride[k + 1], shape[k + 1]);
    auto cmp = detail::StaticGreaterEqual(stride[k], packed);
    if (cmp.has_value() && !*cmp) {
      std::ostringstream oss;
      oss << "DN stride[" << k << "] is smaller than packed stride[" << (k + 1) << "] * shape[" << (k + 1)
          << "]";
      return {false, oss.str()};
    }
    if (!cmp.has_value() && !relaxed_symbolic) {
      return {false, "DN outer-dim stride relation is symbolic and cannot be statically verified"};
    }
  }
  return {true, ""};
}

/// Convenience wrapper around CheckCanonicalView returning only the ok flag.
inline bool IsCanonicalView(const std::vector<ExprPtr>& shape, const std::vector<ExprPtr>& stride,
                            TensorLayout layout, bool relaxed_symbolic = true) {
  return CheckCanonicalView(shape, stride, layout, relaxed_symbolic).ok;
}

/// Build a packed canonical TensorView for (shape, layout). Used by the
/// MaterializeTensorStrides pass to fill stride.empty() slots.
inline TensorView CanonicalizeView(const std::vector<ExprPtr>& shape, TensorLayout layout) {
  return TensorView(BuildLogicalStridesFromLayout(shape, layout), layout, /*valid_shape=*/{});
}

}  // namespace pypto::ir::tensor_view_semantics

#endif  // PYPTO_IR_TRANSFORMS_UTILS_TENSOR_VIEW_SEMANTICS_H_
