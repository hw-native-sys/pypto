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

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/type.h"

namespace pypto::ir {
namespace {
using Args = std::vector<ExprPtr>;
using Kwargs = std::vector<std::pair<std::string, std::any>>;

bool Supported(DataType t) {
  return t == DataType::FP16 || t == DataType::BF16 || t == DataType::FP32 || t == DataType::INT8 ||
         t == DataType::UINT8 || t == DataType::INT16 || t == DataType::UINT16 || t == DataType::INT32 ||
         t == DataType::UINT32;
}

bool IsLinearContiguousVecTile(const TileType& tile) {
  if (tile.shape_.empty() || tile.shape_.size() > 2) return false;
  const TileView view = tile_view_semantics::GetEffectiveTileView(tile);
  if (view.pad != PadValue::null || view.compact != CompactMode::null ||
      view.slayout != TileLayout::none_box ||
      (!view.stride.empty() && view.stride.size() != tile.shape_.size())) {
    return false;
  }
  // Append follows flattened row-major element order. Explicit dense strides
  // describe the same storage as implicit ones; singleton axes create no gaps.
  int64_t expected_stride = 1;
  bool vector_shaped = tile.shape_.size() == 1;
  for (size_t i = tile.shape_.size(); i > 0; --i) {
    const auto dimension = As<ConstInt>(tile.shape_[i - 1]);
    if (!dimension || dimension->value_ <= 0 ||
        dimension->value_ > std::numeric_limits<int64_t>::max() / expected_stride) {
      return false;
    }
    vector_shaped |= dimension->value_ == 1;
    if (!view.stride.empty() && dimension->value_ != 1) {
      const auto stride = As<ConstInt>(view.stride[i - 1]);
      if (!stride || stride->value_ != expected_stride) return false;
    }
    expected_stride *= dimension->value_;
  }
  return view.blayout == TileLayout::row_major || (view.blayout == TileLayout::col_major && vector_shaped);
}

RegTypePtr Register(const ExprPtr& e, RegKind kind, bool allow_fp8 = false) {
  auto t = As<RegType>(e->GetType());
  CHECK_SPAN(t && t->GetRegKind() == kind, e->span_)
      << "Expected " << (kind == RegKind::VecReg ? "VecReg" : "MaskReg");
  CHECK_SPAN(kind == RegKind::MaskReg || Supported(t->GetDType()) ||
                 (allow_fp8 && t->GetDType() == DataType::FP8E4M3FN),
             e->span_)
      << "Unsupported register dtype " << t->GetDType().ToString();
  return t;
}

void Match(const RegTypePtr& a, const RegTypePtr& b) {
  CHECK(a->GetShape() == b->GetShape()) << "Register lane shape mismatch";
  CHECK(a->GetDType() == b->GetDType()) << "Register dtype mismatch (including signedness)";
}

int Lanes(const Kwargs& kw) {
  int n = GetRequiredKwarg<int>(kw, "lanes", "reg");
  CHECK(n > 0) << "Register lanes must be positive";
  return n;
}

int64_t Constant(const ExprPtr& e, const char* label) {
  // TODO(reg-vmi-generalize): the P0 allocation/layout ABI still requires
  // static physical geometry and strides. Relax this when symbolic physical
  // layouts can be lowered; dynamic offsets and valid_shape pass through.
  auto c = As<ConstInt>(e);
  CHECK_SPAN(c, e->span_) << label << " must be a static integer in P0";
  return c->value_;
}

int64_t Product(const std::vector<ExprPtr>& shape) {
  int64_t result = 1;
  CHECK(!shape.empty()) << "Register memory access requires nonempty shape";
  for (const auto& e : shape) {
    auto n = Constant(e, "Tile dimension");
    CHECK(n > 0 && result <= std::numeric_limits<int64_t>::max() / n)
        << "Tile capacity must be a positive representable static extent";
    result *= n;
  }
  return result;
}

// Physical-register loads distinguish the complete 256-byte register from
// the source memory footprint. For example UNPK4 reads 64 bytes and places
// them in every fourth byte; BRC_B32 reads only one scalar.
int64_t NativeLoadExtent(const Kwargs& kwargs) {
  const auto dtype = GetRequiredKwarg<DataType>(kwargs, "dtype", "reg.load_native");
  const auto lanes = Lanes(kwargs);
  CHECK(lanes * dtype.GetByte() == 256) << "Native load requires a full 256-byte register";
  const auto mode = GetRequiredKwarg<std::string>(kwargs, "dist", "reg.load_native");
  if (mode == "UNPK4") {
    CHECK(dtype == DataType::FP4E2M1X2 || dtype == DataType::FP8E4M3FN)
        << "UNPK4 native load requires FP4E2M1X2 or FP8E4M3FN";
    return 64;
  }
  if (mode == "BRC_B32") {
    CHECK(dtype.GetByte() == 4 && Supported(dtype)) << "BRC_B32 requires a supported 32-bit dtype";
    return 1;
  }
  if (mode == "E2B_B16") {
    CHECK(dtype == DataType::BF16 || dtype == DataType::UINT16 || dtype == DataType::INT16)
        << "E2B_B16 requires BF16/UINT16/INT16";
    return 8;
  }
  CHECK(mode == "NORM" && Supported(dtype)) << "Unsupported native load distribution";
  return lanes;
}

int64_t NativeStoreExtent(const RegTypePtr& value, const Kwargs& kwargs) {
  CHECK(value->GetShape() * value->GetDType().GetByte() == 256)
      << "Native store requires a full 256-byte register";
  const auto mode = GetRequiredKwarg<std::string>(kwargs, "dist", "reg.store_native");
  if (mode == "PK_B32") {
    CHECK(value->GetDType() == DataType::BF16) << "PK_B32 native store requires BF16";
    return 64;
  }
  if (mode == "ONEPT_B32") {
    CHECK(value->GetDType().GetByte() == 4) << "ONEPT_B32 requires a 32-bit dtype";
    return 1;
  }
  CHECK(mode == "NORM") << "Unsupported native store distribution";
  return value->GetShape();
}

// The bounding extent includes gaps for bounds/dependency checks, but gaps
// must never be credited as initialized. Arithmetic is checked before multiplying.
int64_t StoreExtent(const Args& args, const Kwargs& kwargs) {
  auto value = Register(args[2], RegKind::VecReg);
  if (GetKwargOr<bool>(kwargs, "native", false)) return NativeStoreExtent(value, kwargs);
  int stride = GetKwargOr<int>(kwargs, "block_stride", 1);
  CHECK_SPAN(stride > 0 && stride <= std::numeric_limits<int16_t>::max(), args[0]->span_)
      << "block_stride must be a positive i16 integer (1..32767)";
  const auto bytes = static_cast<int64_t>(value->GetDType().GetByte());
  CHECK_SPAN(bytes > 0 && bytes <= 32, args[2]->span_)
      << "Register store element size must be in [1, 32] bytes";
  const int64_t block = 32 / bytes;
  const int64_t last = value->GetShape() - 1;
  const int64_t tail = last % block + 1;
  CHECK(last / block <= (std::numeric_limits<int64_t>::max() - tail) / (stride * block))
      << "Register strided extent overflows";
  return (last / block) * stride * block + tail;
}

TileTypePtr Memory(const Args& args, int64_t lanes, bool read, bool linear_destination = false) {
  auto t = As<TileType>(args[0]->GetType());
  CHECK_SPAN(t, args[0]->span_) << "reg.load/store requires a Vec Tile";
  CHECK(t->GetMemorySpace() == MemorySpace::Vec) << "reg.load/store requires Vec UB memory";
  CHECK(Supported(t->dtype_)) << "Unsupported load/store dtype";
  auto offset_type = As<ScalarType>(args[1]->GetType());
  CHECK(offset_type && offset_type->dtype_ == DataType::INDEX) << "offset must be an INDEX scalar";
  int64_t capacity = Product(t->shape_);
  auto offset_const = As<ConstInt>(args[1]);
  int64_t offset = offset_const ? offset_const->value_ : 0;
  CHECK(lanes <= capacity && (!offset_const || (offset >= 0 && offset <= capacity - lanes)))
      << "Register access exceeds capacity bounds";
  if (linear_destination) {
    CHECK_SPAN(IsLinearContiguousVecTile(*t), args[0]->span_)
        << "squeeze_store destination requires a contiguous one- or two-dimensional Vec Tile view";
  }
  if (t->tile_view_) {
    const auto& v = *t->tile_view_;
    // Offsets and block strides address flat storage. A boxed/padded View
    // needs a separate physical-layout contract; block_stride alone does not
    // make its logical indexing contiguous.
    if (!linear_destination) {
      CHECK(v.blayout == TileLayout::row_major && v.slayout == TileLayout::none_box &&
            v.compact == CompactMode::null)
          << "Register memory requires a contiguous row-major View";
    }
    // Verify the full physical stride, not just the last dimension: flattening
    // a padded row would otherwise read padding as the next logical row.
    if (!linear_destination && !v.stride.empty()) {
      CHECK(v.stride.size() == t->shape_.size()) << "View stride rank mismatch";
      int64_t stride = 1;
      for (size_t i = t->shape_.size(); i > 0; --i) {
        CHECK(Constant(v.stride[i - 1], "stride") == stride) << "View must be contiguous";
        stride *= Constant(t->shape_[i - 1], "shape");
      }
    }
    if (auto start = As<ConstInt>(v.start_offset)) CHECK(start->value_ >= 0) << "Negative View offset";
    if (read && !v.valid_shape.empty()) {
      CHECK(v.valid_shape.size() == t->shape_.size()) << "valid_shape rank mismatch";
      // A contiguous read can cross rows; both endpoints and every crossed
      // row must lie in the initialized rectangular valid_shape.
      int64_t stride = capacity;
      int64_t last = offset + lanes - 1;
      for (size_t i = 0; i < t->shape_.size(); ++i) {
        auto dim = Constant(t->shape_[i], "shape");
        stride /= dim;
        auto valid_const = As<ConstInt>(v.valid_shape[i]);
        // Unknown validity/offsets pass through to VMI. Check only facts that
        // are known statically; callers own runtime bounds and initialization.
        if (!valid_const) continue;
        auto valid = valid_const->value_;
        CHECK(valid >= 0 && valid <= dim) << "Invalid valid_shape";
        if (!offset_const) continue;
        bool wraps = offset / (stride * dim) != last / (stride * dim);
        int64_t max_coordinate = wraps ? dim - 1 : (last / stride) % dim;
        CHECK(max_coordinate < valid) << "Load interval is not proven initialized by valid_shape";
      }
    }
  } else {
    // Stream destinations have already proved their effective (possibly
    // implicit) layout above; ordinary register accesses retain their contract.
    CHECK(linear_destination || t->shape_.size() == 1)
        << "2D register memory access requires an explicit contiguous View";
  }
  if (t->memref_ && !(*t->memref_)->is_pinned_ && offset_const) {
    // A View may be smaller than its allocation, but its base offset still
    // contributes to the accessed range. Check the actual MemRef region too.
    ExprPtr start_expr = t->tile_view_ ? t->tile_view_->start_offset : nullptr;
    auto start_const = As<ConstInt>(start_expr);
    if (!start_expr || start_const) {
      uint64_t start = start_const ? static_cast<uint64_t>(start_const->value_) : 0;
      uint64_t elements = (*t->memref_)->size_ / t->dtype_.GetByte();
      CHECK(start <= elements && static_cast<uint64_t>(offset) <= elements - start &&
            static_cast<uint64_t>(lanes) <= elements - start - static_cast<uint64_t>(offset))
          << "Register View access exceeds root capacity bounds";
    }
  }
  return t;
}

void Scalar(const ExprPtr& e, DataType dtype) {
  auto scalar = As<ScalarType>(e->GetType());
  CHECK(scalar) << "Expected scalar base/value";
  // DSL literals are normalized to the requested dtype before construction.
  // Raw IR construction must obey the same contract so printing and reparsing
  // cannot change the operand type or reject an otherwise accepted literal.
  CHECK(scalar->dtype_ == dtype) << "Scalar dtype must match register dtype";
}

TypePtr Binary(const Args& a, const Kwargs& k, const std::string& name) {
  CHECK(a.size() == 2 || a.size() == 3) << "Binary register operation expects 2 operands and optional mask";
  auto x = Register(a[0], RegKind::VecReg);
  Match(x, Register(a[1], RegKind::VecReg));
  if (a.size() == 3) {
    CHECK(Register(a[2], RegKind::MaskReg)->GetShape() == x->GetShape()) << "Mask lane mismatch";
  }
  CHECK(GetStringKwarg(k, "pmode", "zero") == "zero") << "P0 pmode must be zero (merge unsupported)";
  CHECK(name != "mul" || (x->GetDType() != DataType::INT8 && x->GetDType() != DataType::UINT8))
      << "reg.mul rejects 8-bit dtype";
  CHECK(name != "div" || x->GetDType() == DataType::FP16 || x->GetDType() == DataType::FP32)
      << "reg.div requires FP16/FP32 dtype";
  return x;
}

TypePtr Convert(const Args& a, const Kwargs& k) {
  if (GetKwargOr<bool>(k, "native", false)) {
    CHECK(a.size() == 2) << "reg.convert_native requires source and mask";
    auto x = As<RegType>(a[0]->GetType());
    CHECK(x && x->GetRegKind() == RegKind::VecReg && x->GetShape() * x->GetDType().GetByte() == 256)
        << "Native convert requires a full 256-byte input register";
    auto mask = Register(a[1], RegKind::MaskReg);
    CHECK(mask->GetShape() == x->GetShape()) << "Native convert mask must match the source width";
    auto dst = GetRequiredKwarg<DataType>(k, "dtype", "reg.convert_native");
    auto part = GetRequiredKwarg<std::string>(k, "part", "reg.convert_native");
    auto src = x->GetDType();
    bool widening = (src == DataType::FP4E2M1X2 && dst == DataType::BF16) ||
                    (src == DataType::FP8E4M3FN && dst == DataType::FP32);
    bool narrowing = src == DataType::FP32 && dst == DataType::BF16;
    CHECK(widening || narrowing) << "Unsupported native convert dtype pair";
    CHECK((widening && part == "PART_P0") || (narrowing && (part == "PART_EVEN" || part == "PART_ODD")))
        << "Unsupported native convert part for dtype pair";
    if (narrowing) {
      CHECK(GetKwargOr<std::string>(k, "rounding", "") == "R" && !GetKwargOr<bool>(k, "saturate", true))
          << "Native FP32 -> BF16 requires ROUND_R and saturation disabled";
    } else {
      CHECK(!std::any_of(k.begin(), k.end(), [](const auto& attr) {
        return attr.first == "rounding" || attr.first == "saturate";
      })) << "Native widening conversion does not take rounding or saturation";
    }
    auto lanes = Lanes(k);
    CHECK(lanes * dst.GetByte() == 256) << "Native convert requires a full 256-byte result register";
    return std::make_shared<RegType>(lanes, dst, RegKind::VecReg);
  }
  CHECK(a.size() == 1) << "reg.convert expects one VecReg";
  auto x = Register(a[0], RegKind::VecReg);
  auto dst = GetRequiredKwarg<DataType>(k, "dtype", "reg.convert");
  const auto src = x->GetDType();
  // Pooling and normalization round through BF16 without changing logical lanes.
  bool narrow_float =
      src == DataType::FP32 && (dst == DataType::FP16 || dst == DataType::BF16 || dst == DataType::FP8E4M3FN);
  bool float_int = src == DataType::FP32 && dst == DataType::INT32;
  bool narrow_int = src == DataType::INT32 && dst == DataType::INT8;
  bool plain = ((src == DataType::FP16 || src == DataType::BF16) && dst == DataType::FP32) ||
               (src == DataType::INT32 && dst == DataType::FP32) ||
               (src == DataType::INT8 && dst == DataType::FP16) ||
               (src == DataType::INT8 && dst == DataType::INT16) ||
               (src == DataType::INT16 && dst == DataType::INT32) ||
               (src == DataType::UINT8 && dst == DataType::UINT16) ||
               (src == DataType::UINT16 && dst == DataType::UINT32);
  CHECK(plain || narrow_float || float_int || narrow_int || src == dst) << "Unsupported convert dtype pair";
  bool rounding = false, saturate = false;
  for (const auto& [key, value] : k) {
    if (key == "rounding") {
      rounding = true;
      auto mode = AnyCast<std::string>(value, key);
      CHECK((narrow_float && (mode == "R" || mode == "A" || mode == "Z")) ||
            (float_int && (mode == "R" || mode == "A" || mode == "F" || mode == "C" || mode == "Z")))
          << "Unsupported rounding for convert pair";
    }
    if (key == "saturate") {
      saturate = true;
      CHECK(narrow_float || float_int || (narrow_int && !AnyCast<bool>(value, key)))
          << "Unsupported saturate for convert pair";
    }
  }
  CHECK(!(narrow_float || float_int || narrow_int) || saturate) << "convert requires explicit saturate";
  CHECK(!(narrow_float || float_int) || rounding) << "convert requires normalized rounding";
  return std::make_shared<RegType>(x->GetShape(), dst, RegKind::VecReg);
}
bool Integer(DataType dtype) { return dtype.IsInt() && dtype.GetBit() <= 32 && dtype.GetBit() >= 8; }
bool Float(DataType dtype) { return dtype == DataType::FP16 || dtype == DataType::FP32; }

RegTypePtr Vec(const ExprPtr& expr) {
  auto type = Register(expr, RegKind::VecReg);
  CHECK_SPAN(Integer(type->GetDType()) || Float(type->GetDType()), expr->span_)
      << "Unsupported VF dtype " << type->GetDType().ToString();
  return type;
}

RegTypePtr Mask(const ExprPtr& expr, const RegTypePtr& vector) {
  auto mask = Register(expr, RegKind::MaskReg);
  CHECK_SPAN(mask->GetShape() == vector->GetShape(), expr->span_) << "Mask lane mismatch";
  return mask;
}

void Pmode(const Kwargs& kwargs) {
  CHECK(GetStringKwarg(kwargs, "pmode", "zero") == "zero") << "VF pmode must be zero";
}

TypePtr Unary(const Args& args, const Kwargs& kwargs, const std::string& name) {
  CHECK(args.size() == 1 || args.size() == 2) << "Unary VF expects value and optional mask";
  auto x = Vec(args[0]);
  auto dtype = x->GetDType();
  if (name == "abs") {
    CHECK(Float(dtype) || dtype.IsSignedInt()) << "reg.abs requires float or signed integer dtype";
  } else if (name == "relu") {
    CHECK(Float(dtype) || dtype == DataType::INT32) << "reg.relu requires FP16/FP32/INT32 dtype";
  } else if (name == "bitwise_not") {
    CHECK(Integer(dtype)) << "Bitwise operations require integer dtype";
  } else if (name != "neg") {
    CHECK(Float(dtype)) << "Transcendental VF requires FP16/FP32 dtype";
  }
  if (args.size() == 2) Mask(args[1], x);
  Pmode(kwargs);
  return x;
}

TypePtr IntegerBinary(const Args& args, const Kwargs& kwargs) {
  CHECK(args.size() == 2 || args.size() == 3) << "Binary VF expects two vectors and optional mask";
  auto x = Vec(args[0]);
  auto y = Vec(args[1]);
  CHECK(Integer(x->GetDType())) << "Bitwise and shift operations require integer dtype";
  Match(x, y);
  if (args.size() == 3) Mask(args[2], x);
  Pmode(kwargs);
  return x;
}

TypePtr ScalarOperation(const Args& args, const Kwargs& kwargs, const std::string& name) {
  CHECK(args.size() == 3) << "Scalar VF expects vector, scalar and mask";
  auto x = Vec(args[0]);
  auto scalar = As<ScalarType>(args[1]->GetType());
  bool shift = name == "shift_lefts" || name == "shift_rights";
  auto dtype = shift ? DataType::INT16 : x->GetDType();
  CHECK(scalar && scalar->dtype_ == dtype) << "Scalar dtype mismatch (shifts require i16)";
  if (shift) CHECK(Integer(x->GetDType())) << "Scalar shift requires integer dtype";
  if (name == "muls") CHECK(x->GetDType().GetBit() >= 16) << "reg.muls rejects 8-bit dtype";
  auto mask = Mask(args[2], x);
  Pmode(kwargs);
  if (name != "compares") return x;
  CHECK(Float(x->GetDType()) || x->GetDType() == DataType::INT32 || x->GetDType() == DataType::UINT32)
      << "Unsupported compares dtype";
  auto predicate = GetRequiredKwarg<std::string>(kwargs, "predicate", "reg.compares");
  CHECK(predicate == "eq" || predicate == "ne" || predicate == "lt" || predicate == "le" ||
        predicate == "gt" || predicate == "ge")
      << "Invalid compare predicate";
  return mask;
}

// A5 executes a grouped reduction natively only for these group counts, and
// only when the count divides the source lane count. Outside that set PTOAS has
// no layout to assign, so reject it here where the lane count is still visible
// in DSL terms rather than letting it fail during layout assignment.
bool GroupCount(int64_t groups) { return groups == 1 || groups == 2 || groups == 4 || groups == 8; }

TypePtr Reduction(const Args& args, const Kwargs& kwargs, const std::string& name) {
  CHECK(args.size() == 2) << "Reduction requires source and mask";
  auto x = Vec(args[0]);
  CHECK(x->GetDType().GetBit() >= 16) << "Reduction rejects 8-bit dtype";
  Mask(args[1], x);
  if (name == "reduce_sum" && Float(x->GetDType())) {
    CHECK(GetKwargOr<bool>(kwargs, "reassoc", false)) << "Floating reduce_sum requires reassoc=True";
  }
  if (GetKwargOr<bool>(kwargs, "native", false)) {
    CHECK(GetKwargOr<int>(kwargs, "group_size", 0) == 0) << "Native reduction does not take group_size";
    CHECK(x->GetShape() * x->GetDType().GetByte() == 256 && x->GetDType() == DataType::FP32)
        << "Native reduction requires a full FP32 register";
    return x;
  }
  // An absent group_size means a full reduction to one lane, the original
  // contract. Detect absence by key, not by a zero default: raw IR may pass
  // group_size explicitly, and a bogus 0 must be rejected rather than silently
  // degrading to a full reduction.
  const auto entry = std::find_if(kwargs.begin(), kwargs.end(),
                                  [](const auto& kwarg) { return kwarg.first == "group_size"; });
  if (entry == kwargs.end()) return std::make_shared<RegType>(1, x->GetDType(), RegKind::VecReg);
  const auto group_size = AnyCast<int>(entry->second, "group_size");
  CHECK(group_size > 0 && x->GetShape() % group_size == 0)
      << "reg." << name << " group_size must be positive and divide the " << x->GetShape() << " source lanes";
  const int64_t groups = x->GetShape() / group_size;
  CHECK(GroupCount(groups)) << "reg." << name << " supports 1, 2, 4 or 8 groups; group_size " << group_size
                            << " over " << x->GetShape() << " lanes requests " << groups;
  return std::make_shared<RegType>(groups, x->GetDType(), RegKind::VecReg);
}

TypePtr Fused(const Args& a, const Kwargs& k, const std::string& name) {
  const size_t inputs = (name == "axpy" || name == "mula") ? 3 : 2;
  CHECK(a.size() == inputs + 1 || (name == "mula" && a.size() == inputs))
      << "reg." << name << " has invalid operand count";
  auto x = Register(a[0], RegKind::VecReg);
  const auto dtype = x->GetDType();
  if (name == "mula") {
    CHECK(dtype != DataType::INT8 && dtype != DataType::UINT8)
        << "reg.mula requires floating-point or 16/32-bit integer registers";
  } else {
    CHECK(dtype == DataType::FP16 || dtype == DataType::FP32)
        << "reg." << name << " requires FP16/FP32 registers";
  }
  for (size_t i = 1; i < inputs; ++i) {
    if ((name == "axpy" && i == 2) || name == "lrelu") {
      Scalar(a[i], dtype);
    } else {
      Match(x, Register(a[i], RegKind::VecReg));
    }
  }
  if (a.size() > inputs) Mask(a.back(), x);
  Pmode(k);
  // EXPDIF widens its FP16 inputs; it is not a dtype-preserving binary op.
  if (name == "expdif") return std::make_shared<RegType>(x->GetShape(), DataType::FP32, RegKind::VecReg);
  return x;
}

TypePtr Paired(const Args& a, const Kwargs& k, const std::string& name) {
  const bool carry = name == "addc" || name == "subc" || name == "addcs" || name == "subcs";
  const bool carry_in = name == "addcs" || name == "subcs";
  const bool interleave = name == "intlv" || name == "dintlv";
  CHECK(a.size() == (carry_in ? 4 : 3) || (interleave && a.size() == 2))
      << "reg." << name << " has invalid operand count";
  auto x = Register(a[0], RegKind::VecReg);
  Match(x, Register(a[1], RegKind::VecReg));
  if (a.size() >= 3) Mask(a.back(), x);
  if (carry_in) Mask(a[2], x);
  if (carry || name == "mull") {
    CHECK(x->GetDType() == DataType::INT32 || x->GetDType() == DataType::UINT32)
        << "reg." << name << " requires INT32/UINT32 registers";
  }
  if (name == "mull") {
    CHECK(x->GetShape() == 64 || x->GetShape() == 128 || x->GetShape() == 256)
        << "reg.mull requires 64, 128, or 256 lanes";
  } else if (!carry) {
    CHECK(x->GetShape() % 2 == 0) << "Interleave/deinterleave require even lane counts";
  }
  if (!carry) Pmode(k);
  TypePtr second = carry ? std::make_shared<RegType>(x->GetShape(), DataType::BOOL, RegKind::MaskReg) : x;
  return std::make_shared<TupleType>(std::vector<TypePtr>{x, second});
}

TileTypePtr IndexedMemory(const Args& a, bool read) {
  // The index vector can repeat addresses and need not fit in the Tile's lane
  // count. Validate the storage geometry only; bounds and initialization of
  // active indexed addresses are runtime obligations, as with dynamic offsets.
  auto zero = std::make_shared<ConstInt>(0, DataType::INDEX, a[0]->span_);
  const size_t base_index = read ? 3 : 4;
  // Shift the scalar pointer, not the unsigned lane indices. This preserves
  // narrow index ranges and lets dynamic row origins stay scalar arithmetic.
  auto base = a.size() > base_index ? a[base_index] : zero;
  auto tile = Memory({a[0], base}, 1, false);
  auto offsets = Register(a[1], RegKind::VecReg);
  CHECK(offsets->GetDType() == DataType::UINT16 || offsets->GetDType() == DataType::UINT32)
      << "Indexed register memory requires UINT16/UINT32 element offsets";
  // The physical indexed paths use 16-bit offsets for byte/halfword data,
  // and 32-bit offsets for word data. Byte gathers promote their result.
  const auto offset_dtype = tile->dtype_.GetBit() == 32 ? DataType::UINT32 : DataType::UINT16;
  CHECK(offsets->GetDType() == offset_dtype)
      << "Indexed memory requires UINT16 offsets for 8/16-bit data and UINT32 offsets for 32-bit data";
  Mask(a[read ? 2 : 3], offsets);
  if (!read) {
    auto value = Register(a[2], RegKind::VecReg);
    CHECK(value->GetShape() == offsets->GetShape()) << "Indexed value/offset lane mismatch";
    CHECK(value->GetDType() == tile->dtype_) << "Scatter dtype mismatch";
  }
  return tile;
}
}  // namespace

// Zero-operand fences are observable ordering effects, handled explicitly by DCE.
REGISTER_OP("reg.mem_bar")
    .set_op_category("RegOp")
    .set_reg_op("pto.mem_bar")
    .set_description("Order UB memory operations within a register scope")
    .no_argument()
    .set_attr<std::string>("kind")
    .f_deduce_type([](const Args& a, const Kwargs& k) -> TypePtr {
      CHECK(a.empty()) << "reg.mem_bar expects no operands";
      const auto kind = GetRequiredKwarg<std::string>(k, "kind", "reg.mem_bar");
      static const std::vector<std::string> kinds{
          "VV_ALL", "VST_VLD", "VLD_VST", "VST_VST", "VS_ALL", "VST_LD", "VLD_ST", "VST_ST",
          "SV_ALL", "ST_VLD",  "LD_VST",  "ST_VST",  "SS_ALL", "ST_LD",  "LD_ST",  "ST_ST"};
      CHECK(std::find(kinds.begin(), kinds.end(), kind) != kinds.end())
          << "Invalid reg.mem_bar kind: " << kind;
      return GetUnknownType();
    });

REGISTER_OP("reg.load")
    .set_op_category("RegOp")
    .set_description("Read a continuous initialized Vec UB interval")
    .set_reg_op("pto.vmi.vload")
    .add_argument("tile", "Source Vec Tile")
    .add_argument("offset", "Element offset")
    .set_attr<int>("lanes")
    .set_attr<bool>("native")
    .set_attr<DataType>("dtype")
    .set_attr<std::string>("dist")
    .set_input_memory(0, MemorySpace::Vec)
    .set_arg_effect(0, ArgEffect::Read)
    .f_deduce_type([](const Args& a, const Kwargs& k) -> TypePtr {
      CHECK(a.size() == 2) << "reg.load expects tile, offset";
      auto lanes = Lanes(k);
      if (GetKwargOr<bool>(k, "native", false)) {
        auto dtype = GetRequiredKwarg<DataType>(k, "dtype", "reg.load_native");
        auto tile = Memory(a, NativeLoadExtent(k), true);
        CHECK(tile->dtype_.GetByte() == dtype.GetByte())
            << "Native load storage/result byte widths must agree";
        return std::make_shared<RegType>(lanes, dtype, RegKind::VecReg);
      }
      auto t = Memory(a, lanes, true);
      return std::make_shared<RegType>(lanes, t->dtype_, RegKind::VecReg);
    });
REGISTER_OP("reg.store")
    .set_op_category("RegOp")
    .set_description("Write active lanes in contiguous or block-strided Vec UB storage")
    .set_reg_op("pto.vmi.vstore")
    .add_argument("tile", "Destination Vec Tile")
    .add_argument("offset", "Element offset")
    .add_argument("value", "VecReg value")
    .add_argument("mask", "Optional MaskReg")
    .set_attr<int>("block_stride")
    .set_attr<bool>("native")
    .set_attr<std::string>("dist")
    .set_arg_effect(0, ArgEffect::Write)
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .set_output_reuses_input(0)
    .f_deduce_type([](const Args& a, const Kwargs& k) -> TypePtr {
      CHECK(a.size() == 3 || a.size() == 4) << "reg.store expects tile, offset, value, optional mask";
      auto v = Register(a[2], RegKind::VecReg);
      if (GetKwargOr<bool>(k, "native", false)) {
        CHECK(a.size() == 4 && GetKwargOr<int>(k, "block_stride", 0) == 0)
            << "Native store requires mask and does not take block_stride";
      }
      auto t = Memory(a, StoreExtent(a, k), false);
      CHECK(t->dtype_ == v->GetDType()) << "Store dtype mismatch";
      if (a.size() == 4) {
        auto mask_lanes = v->GetShape();
        if (GetKwargOr<bool>(k, "native", false) && GetStringKwarg(k, "dist", "") == "PK_B32") {
          mask_lanes = 64;
        }
        CHECK(Register(a[3], RegKind::MaskReg)->GetShape() == mask_lanes) << "Mask lane mismatch";
      }
      return t;
    });

// These operations share their exact operand and attribute contract. Keeping
// the declaration and inference in one loop prevents accidental promotion drift.
static bool RegisterArithmetic() {
  for (const auto& name : {"add", "sub", "mul", "div", "minimum", "maximum"}) {
    auto& e = OpRegistry::GetInstance().Register(std::string("reg.") + name);
    e.set_op_category("RegOp")
        .set_description("Same-type register arithmetic with inactive-zero lanes")
        .set_reg_op(std::string("pto.vmi.") + (std::string(name) == "minimum"   ? "vmin"
                                               : std::string(name) == "maximum" ? "vmax"
                                                                                : std::string("v") + name))
        .add_argument("lhs", "VecReg")
        .add_argument("rhs", "VecReg")
        .add_argument("mask", "Optional MaskReg")
        .set_attr<std::string>("pmode")
        .no_arg_writes()
        .f_deduce_type([name](const Args& a, const Kwargs& k) { return Binary(a, k, name); });
  }
  return true;
}
static const bool arithmetic_registered = RegisterArithmetic();

REGISTER_OP("reg.convert")
    .set_op_category("RegOp")
    .set_description("Explicit whitelist register conversion")
    .set_reg_op("pto.vmi.vcvt")
    .add_argument("value", "VecReg")
    .add_argument("mask", "Optional native source-width MaskReg")
    .set_attr<DataType>("dtype")
    .set_attr<std::string>("rounding")
    .set_attr<bool>("saturate")
    .set_attr<bool>("native")
    .set_attr<std::string>("part")
    .set_attr<int>("lanes")
    .no_arg_writes()
    .f_deduce_type(Convert);

REGISTER_OP("reg.interpret_cast")
    .set_op_category("RegOp")
    .set_description("Reinterpret register bits without numeric conversion")
    .set_reg_op("pto.vmi.vinterpret_cast")
    .add_argument("value", "VecReg")
    .set_attr<DataType>("dtype")
    .no_arg_writes()
    .f_deduce_type([](const Args& a, const Kwargs& k) -> TypePtr {
      CHECK(a.size() == 1) << "reg.interpret_cast expects one VecReg";
      auto x = Register(a[0], RegKind::VecReg, true);
      auto dtype = GetRequiredKwarg<DataType>(k, "dtype", "reg.interpret_cast");
      CHECK(Supported(dtype)) << "Unsupported reinterpret destination dtype";
      const auto source_bits = static_cast<int64_t>(x->GetDType().GetBit());
      const auto target_bits = static_cast<int64_t>(dtype.GetBit());
      INTERNAL_CHECK(source_bits > 0 && target_bits > 0) << "Supported register widths must be positive";
      CHECK(x->GetShape() <= std::numeric_limits<int64_t>::max() / source_bits)
          << "Reinterpret total bit count is too large";
      const int64_t bits = x->GetShape() * source_bits;
      CHECK(bits % target_bits == 0) << "Reinterpret total bits must form whole destination lanes";
      CHECK(bits / target_bits <= std::numeric_limits<int>::max())
          << "Reinterpret result lane count is too large";
      return std::make_shared<RegType>(bits / target_bits, dtype, RegKind::VecReg);
    });

REGISTER_OP("reg.group_broadcast")
    .set_op_category("RegOp")
    .set_description("Broadcast one group-slot lane back across each logical group")
    .set_reg_op("pto.vmi.group_broadcast")
    .add_argument("value", "VecReg holding one lane per group")
    .set_attr<int>("lanes")
    .no_arg_writes()
    .f_deduce_type([](const Args& a, const Kwargs& k) -> TypePtr {
      CHECK(a.size() == 1) << "reg.group_broadcast expects one VecReg";
      auto x = Vec(a[0]);
      // The group count is the source lane count: this op consumes exactly the
      // grouped-reduction result, so passing num_groups separately could only
      // disagree with it. `lanes` is the width to expand back to.
      const auto lanes = Lanes(k);
      CHECK(GroupCount(x->GetShape()))
          << "reg.group_broadcast source must hold 1, 2, 4 or 8 group lanes, got " << x->GetShape();
      CHECK(lanes % x->GetShape() == 0) << "reg.group_broadcast requires the " << x->GetShape()
                                        << " groups to divide the " << lanes << " result lanes";
      return std::make_shared<RegType>(lanes, x->GetDType(), RegKind::VecReg);
    });

REGISTER_OP("reg.create_group_mask")
    .set_op_category("RegOp")
    .set_description("Activate a prefix within each logical lane group")
    .set_reg_op("pto.vmi.create_group_mask")
    .add_argument("active_elems_per_group", "INDEX scalar")
    .set_attr<int>("num_groups")
    .set_attr<int>("group_size")
    .no_arg_writes()
    .f_deduce_type([](const Args& a, const Kwargs& k) -> TypePtr {
      CHECK(a.size() == 1) << "reg.create_group_mask expects one INDEX scalar";
      Scalar(a[0], DataType::INDEX);
      const auto groups = GetRequiredKwarg<int>(k, "num_groups", "reg.create_group_mask");
      const auto size = GetRequiredKwarg<int>(k, "group_size", "reg.create_group_mask");
      CHECK(groups > 0 && size > 0 && groups <= std::numeric_limits<int>::max() / size)
          << "Grouped mask requires positive num_groups and group_size with representable lane count";
      if (auto active = As<ConstInt>(a[0])) {
        CHECK_SPAN(active->value_ >= 0 && active->value_ <= size, a[0]->span_)
            << "active_elems_per_group must be in [0, group_size]";
      } else {
        CHECK((size & (size - 1)) == 0) << "Dynamic grouped mask requires power-of-two group_size";
      }
      return std::make_shared<RegType>(groups * size, DataType::BOOL, RegKind::MaskReg);
    });

static bool RegisterHistograms() {
  for (const auto& name : {"chist", "dhist"}) {
    OpRegistry::GetInstance()
        .Register(std::string("reg.") + name)
        .set_op_category("RegOp")
        .set_description("Accumulate masked UINT8 histogram counts into UINT16 bins")
        .set_reg_op(std::string("pto.vmi.v") + name)
        .add_argument("acc", "UINT16 VecReg with 128 or 256 bins")
        .add_argument("source", "UINT8 VecReg")
        .add_argument("mask", "MaskReg selecting source lanes")
        .no_arg_writes()
        .f_deduce_type([](const Args& a, const Kwargs&) -> TypePtr {
          CHECK(a.size() == 3) << "Histogram expects accumulator, source and mask";
          auto acc = Register(a[0], RegKind::VecReg);
          auto source = Register(a[1], RegKind::VecReg);
          CHECK(acc->GetDType() == DataType::UINT16 && (acc->GetShape() == 128 || acc->GetShape() == 256))
              << "Histogram accumulator must be UINT16 with 128 or 256 bins";
          CHECK(source->GetDType() == DataType::UINT8) << "Histogram source must be UINT8";
          Mask(a[2], source);
          return acc;
        });
  }
  return true;
}
static const bool histograms_registered = RegisterHistograms();

static bool RegisterGenerators() {
  for (const auto& name : {"broadcast", "index_gen", "full_mask", "active_mask"}) {
    auto& e = OpRegistry::GetInstance().Register(std::string("reg.") + name);
    e.set_op_category("RegOp")
        .set_description("Register or mask generator")
        .set_reg_op(std::string(name) == "broadcast"   ? "pto.vmi.vbrc"
                    : std::string(name) == "index_gen" ? "pto.vmi.vci"
                    : std::string(name) == "full_mask" ? "pto.vmi.pset"
                                                       : "pto.vmi.create_mask")
        .set_attr<int>("lanes")
        .no_arg_writes();
    std::string op(name);
    if (op == "full_mask") {
      e.no_argument();
    } else {
      e.add_argument("value", "Scalar input");
    }
    if (op == "broadcast" || op == "index_gen") e.set_attr<DataType>("dtype");
    if (op == "index_gen") e.set_attr<std::string>("order");
    e.f_deduce_type([op](const Args& a, const Kwargs& k) -> TypePtr {
      CHECK(a.size() == (op == "full_mask" ? 0 : 1)) << "Invalid generator operand count";
      int n = Lanes(k);
      bool mask = op == "full_mask" || op == "active_mask";
      auto dtype = mask ? DataType::BOOL : GetRequiredKwarg<DataType>(k, "dtype", "reg");
      if (!mask) {
        CHECK(Supported(dtype)) << "Unsupported generator dtype";
        if (op == "broadcast" && As<RegType>(a[0]->GetType())) {
          // Broadcast reductions directly without a UB store/load or scalar transfer.
          auto value = Register(a[0], RegKind::VecReg);
          CHECK(value->GetShape() == 1) << "Register broadcast requires a one-lane VecReg";
          CHECK(value->GetDType() == dtype) << "Register broadcast dtype mismatch";
        } else {
          Scalar(a[0], dtype);
        }
      }
      if (op == "active_mask") {
        auto t = As<ScalarType>(a[0]->GetType());
        CHECK(t && t->dtype_ == DataType::INDEX) << "active_lanes must be INDEX";
        // Runtime counts are caller-bounded; retain early checks for constants.
        if (auto active = As<ConstInt>(a[0])) {
          CHECK(active->value_ >= 0 && active->value_ <= n) << "active_lanes must be in [0, lanes]";
        }
      }
      if (op == "index_gen") {
        // A5 VCI has no BF16 form. Reject at the shared schema boundary so
        // DSL, Builder and raw-IR verification fail before reaching codegen.
        CHECK_SPAN(dtype != DataType::BF16, a[0]->span_)
            << "reg.index_gen does not support BF16 VCI; use an integer index dtype";
        CHECK(GetStringKwarg(k, "order", "ascending") == "ascending") << "P0 order must be ascending";
      }
      return std::make_shared<RegType>(n, dtype, mask ? RegKind::MaskReg : RegKind::VecReg);
    });
  }
  return true;
}
static const bool generators_registered = RegisterGenerators();

REGISTER_OP("reg.compare")
    .set_op_category("RegOp")
    .set_description("Seeded register comparison")
    .set_reg_op("pto.vmi.vcmp")
    .add_argument("lhs", "VecReg")
    .add_argument("rhs", "VecReg")
    .add_argument("seed", "MaskReg")
    .set_attr<std::string>("predicate")
    .no_arg_writes()
    .f_deduce_type([](const Args& a, const Kwargs& k) -> TypePtr {
      CHECK(a.size() == 3) << "reg.compare expects lhs, rhs, seed";
      auto x = Register(a[0], RegKind::VecReg);
      Match(x, Register(a[1], RegKind::VecReg));
      auto m = Register(a[2], RegKind::MaskReg);
      CHECK(m->GetShape() == x->GetShape()) << "Mask lane mismatch";
      CHECK(x->GetDType() == DataType::FP16 || x->GetDType() == DataType::FP32 ||
            x->GetDType() == DataType::INT32 || x->GetDType() == DataType::UINT32)
          << "Unsupported compare dtype";
      auto p = GetRequiredKwarg<std::string>(k, "predicate", "reg.compare");
      CHECK(p == "eq" || p == "ne" || p == "lt" || p == "le" || p == "gt" || p == "ge")
          << "Invalid compare predicate";
      return m;
    });
REGISTER_OP("reg.select")
    .set_op_category("RegOp")
    .set_description("Select register lanes")
    .set_reg_op("pto.vmi.vsel")
    .add_argument("mask", "MaskReg")
    .add_argument("true_value", "VecReg")
    .add_argument("false_value", "VecReg")
    .no_arg_writes()
    .f_deduce_type([](const Args& a, const Kwargs&) -> TypePtr {
      CHECK(a.size() == 3) << "reg.select expects mask, true_value, false_value";
      auto m = Register(a[0], RegKind::MaskReg);
      auto v = Register(a[1], RegKind::VecReg);
      Match(v, Register(a[2], RegKind::VecReg));
      CHECK(v->GetShape() == m->GetShape()) << "Mask lane mismatch";
      CHECK(v->GetDType() != DataType::BF16) << "select rejects BF16 dtype";
      return v;
    });
static bool RegisterMaskLogic() {
  for (const auto& name : {"and", "or", "xor", "not"}) {
    std::string op(name);
    auto& e = OpRegistry::GetInstance().Register("reg.mask_" + op);
    e.set_op_category("RegOp")
        .set_description("Boolean mask logic")
        .set_reg_op("pto.vmi.v" + op)
        .add_argument("lhs", "MaskReg")
        .no_arg_writes();
    if (op != "not") e.add_argument("rhs", "MaskReg");
    e.f_deduce_type([op](const Args& a, const Kwargs&) -> TypePtr {
      CHECK(a.size() == (op == "not" ? 1 : 2)) << "Invalid mask operand count";
      auto m = Register(a[0], RegKind::MaskReg);
      if (a.size() == 2) Match(m, Register(a[1], RegKind::MaskReg));
      return m;
    });
  }
  return true;
}
static const bool masks_registered = RegisterMaskLogic();

static bool RegisterComputeFamilies() {
  for (const auto& [name, opcode] :
       std::vector<std::pair<std::string, std::string>>{{"abs", "vabs"},
                                                        {"neg", "vneg"},
                                                        {"relu", "vrelu"},
                                                        {"exp", "vexp"},
                                                        {"log", "vln"},
                                                        {"sqrt", "vsqrt"},
                                                        {"bitwise_not", "vnot"}}) {
    OpRegistry::GetInstance()
        .Register("reg." + name)
        .set_op_category("RegOp")
        .set_description("Predicated lane-wise unary VF operation")
        .set_reg_op("pto.vmi." + opcode)
        .add_argument("value", "VecReg")
        .add_argument("mask", "Optional MaskReg")
        .set_attr<std::string>("pmode")
        .no_arg_writes()
        .f_deduce_type([op_name = name](const Args& args, const Kwargs& kwargs) {
          return Unary(args, kwargs, op_name);
        });
  }
  for (const auto& [name, opcode] :
       std::vector<std::pair<std::string, std::string>>{{"bitwise_and", "vand"},
                                                        {"bitwise_or", "vor"},
                                                        {"bitwise_xor", "vxor"},
                                                        {"shift_left", "vshl"},
                                                        {"shift_right", "vshr"}}) {
    OpRegistry::GetInstance()
        .Register("reg." + name)
        .set_op_category("RegOp")
        .set_description("Predicated integer binary VF operation")
        .set_reg_op("pto.vmi." + opcode)
        .add_argument("lhs", "VecReg")
        .add_argument("rhs", "VecReg")
        .add_argument("mask", "Optional MaskReg")
        .set_attr<std::string>("pmode")
        .no_arg_writes()
        .f_deduce_type(IntegerBinary);
  }
  for (const auto& [name, opcode] :
       std::vector<std::pair<std::string, std::string>>{{"adds", "vadds"},
                                                        {"muls", "vmuls"},
                                                        {"maximums", "vmaxs"},
                                                        {"minimums", "vmins"},
                                                        {"shift_lefts", "vshls"},
                                                        {"shift_rights", "vshrs"},
                                                        {"compares", "vcmps"}}) {
    auto& entry = OpRegistry::GetInstance().Register("reg." + name);
    entry.set_op_category("RegOp")
        .set_description("Seeded vector-scalar VF operation")
        .set_reg_op("pto.vmi." + opcode)
        .add_argument("value", "VecReg")
        .add_argument("scalar", "Scalar")
        .add_argument("mask", "MaskReg")
        .no_arg_writes()
        .f_deduce_type([op_name = name](const Args& args, const Kwargs& kwargs) {
          return ScalarOperation(args, kwargs, op_name);
        });
    if (name == "compares") {
      entry.set_attr<std::string>("predicate");
    } else {
      entry.set_attr<std::string>("pmode");
    }
  }
  for (const auto& [name, opcode] : std::vector<std::pair<std::string, std::string>>{
           {"reduce_sum", "vcadd"}, {"reduce_max", "vcmax"}, {"reduce_min", "vcmin"}}) {
    auto& entry = OpRegistry::GetInstance().Register("reg." + name);
    entry.set_op_category("RegOp")
        .set_description("Masked reduction to one lane, or to one lane per logical group")
        .set_reg_op("pto.vmi." + opcode)
        .add_argument("value", "VecReg")
        .add_argument("mask", "MaskReg")
        .set_attr<int>("group_size")
        .set_attr<bool>("native")
        .no_arg_writes()
        .f_deduce_type([op_name = name](const Args& args, const Kwargs& kwargs) {
          return Reduction(args, kwargs, op_name);
        });
    if (name == "reduce_sum") {
      entry.set_attr<bool>("reassoc");
    }
  }
  return true;
}
static const bool compute_families_registered = RegisterComputeFamilies();

REGISTER_OP("reg.gather_lanes")
    .set_op_category("RegOp")
    .set_description("Dynamic register lane permutation")
    .set_reg_op("pto.vmi.vselr")
    .add_argument("value", "VecReg")
    .add_argument("indices", "VecReg")
    .no_arg_writes()
    .f_deduce_type([](const Args& args, const Kwargs&) -> TypePtr {
      CHECK(args.size() == 2) << "gather_lanes requires source and indices";
      auto x = Vec(args[0]);
      auto indices = Vec(args[1]);
      auto width = x->GetDType().GetBit();
      CHECK(Integer(indices->GetDType()) && indices->GetDType().GetBit() == width) << "Index width mismatch";
      CHECK(indices->GetShape() == x->GetShape()) << "Index lane mismatch";
      CHECK(x->GetShape() == 64 || (width <= 16 && x->GetShape() == 128) ||
            (width == 8 && x->GetShape() == 256))
          << "Unsupported vselr lane count";
      return x;
    });

static bool RegisterFusedOps() {
  for (const std::string name : {"expdif", "axpy", "mula", "lrelu", "prelu"}) {
    auto& e = OpRegistry::GetInstance().Register("reg." + name);
    e.set_op_category("RegOp")
        .set_reg_op("pto.vmi.v" + name)
        .set_description("Fused register arithmetic/activation")
        .add_argument(name == "mula" ? "acc" : "x", "VecReg");
    if (name == "mula") {
      e.add_argument("lhs", "VecReg").add_argument("rhs", "VecReg");
    } else if (name == "axpy") {
      e.add_argument("acc", "VecReg").add_argument("alpha", "Matching float scalar");
    } else {
      e.add_argument(name == "lrelu" ? "slope" : (name == "prelu" ? "alpha" : "max_value"),
                     name == "lrelu" ? "Matching float scalar" : "VecReg");
    }
    e.add_argument("mask", "MaskReg (optional for mula)")
        .set_attr<std::string>("pmode")
        .no_arg_writes()
        .f_deduce_type([name](const Args& a, const Kwargs& k) { return Fused(a, k, name); });
  }
  return true;
}
static const bool fused_registered = RegisterFusedOps();

static bool RegisterPairedOps() {
  for (const std::string name : {"mull", "addc", "subc", "addcs", "subcs", "intlv", "dintlv"}) {
    auto& e = OpRegistry::GetInstance().Register("reg." + name);
    e.set_op_category("RegOp")
        .set_reg_op("pto.vmi.v" + name)
        .set_description("Two-result register operation")
        .add_argument("lhs", "VecReg")
        .add_argument("rhs", "VecReg");
    if (name == "addcs" || name == "subcs") e.add_argument("carry_in", "Carry/not-borrow MaskReg");
    e.add_argument("mask", name == "intlv" || name == "dintlv" ? "Optional MaskReg" : "MaskReg")
        .set_output_arity(2)
        .no_arg_writes();
    if (name == "mull" || name == "intlv" || name == "dintlv") e.set_attr<std::string>("pmode");
    e.f_deduce_type([name](const Args& a, const Kwargs& k) { return Paired(a, k, name); });
  }
  return true;
}
static const bool paired_registered = RegisterPairedOps();

REGISTER_OP("reg.gather")
    .set_op_category("RegOp")
    .set_reg_op("pto.vmi.vgather")
    .set_description("Masked indexed UB read with element offsets")
    .add_argument("tile", "Source Vec Tile")
    .add_argument("offsets", "Unsigned index VecReg")
    .add_argument("mask", "MaskReg")
    .add_argument("base_offset", "Optional scalar INDEX element offset")
    .set_attr<std::string>("pmode")
    .set_input_memory(0, MemorySpace::Vec)
    .set_arg_effect(0, ArgEffect::Read)
    .f_deduce_type([](const Args& a, const Kwargs& k) -> TypePtr {
      CHECK(a.size() == 3 || a.size() == 4) << "reg.gather expects tile, offsets, mask, optional base_offset";
      auto tile = IndexedMemory(a, true);
      auto indices = Register(a[1], RegKind::VecReg);
      auto dtype = tile->dtype_;
      // The byte gather form zero-extends, including signed byte sources.
      if (dtype == DataType::INT8) dtype = DataType::INT16;
      if (dtype == DataType::UINT8) dtype = DataType::UINT16;
      CHECK(indices->GetDType() != DataType::UINT16 || dtype.GetBit() == 16)
          << "UINT16 gather offsets require 16-bit results";
      Pmode(k);
      return std::make_shared<RegType>(indices->GetShape(), dtype, RegKind::VecReg);
    });
REGISTER_OP("reg.scatter")
    .set_op_category("RegOp")
    .set_reg_op("pto.vmi.vscatter")
    .set_description("Masked indexed UB write; return destination alias")
    .add_argument("tile", "Destination Vec Tile")
    .add_argument("offsets", "Unsigned index VecReg")
    .add_argument("value", "VecReg")
    .add_argument("mask", "MaskReg")
    .add_argument("base_offset", "Optional scalar INDEX element offset")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .set_arg_effect(0, ArgEffect::Write)
    .set_output_reuses_input(0)
    .f_deduce_type([](const Args& a, const Kwargs&) -> TypePtr {
      CHECK(a.size() == 4 || a.size() == 5)
          << "reg.scatter expects tile, offsets, value, mask, optional base_offset";
      return IndexedMemory(a, false);
    });

// A stream is represented by explicit state edges, including through loop
// carries. It is not an ordinary Tile version or a per-call compress_store.
REGISTER_OP("reg.squeeze")
    .set_op_category("RegOp")
    .set_description("Pack selected UINT16/UINT32 lanes in stable order")
    .set_reg_op("pto.vmi.compress")
    .add_argument("value", "One physical UINT16/UINT32 VecReg")
    .add_argument("mask", "Matching MaskReg")
    .no_arg_writes()
    .f_deduce_type([](const Args& a, const Kwargs&) -> TypePtr {
      CHECK(a.size() == 2) << "reg.squeeze expects value and mask";
      auto value = Register(a[0], RegKind::VecReg);
      CHECK(value->GetDType() == DataType::UINT16 || value->GetDType() == DataType::UINT32)
          << "reg.squeeze requires UINT16/UINT32";
      CHECK(value->GetShape() == 256 / value->GetDType().GetByte())
          << "reg.squeeze requires one full 256-byte register";
      CHECK(Register(a[1], RegKind::MaskReg)->GetShape() == value->GetShape())
          << "Squeeze mask lane mismatch";
      return value;
    });

REGISTER_OP("reg.squeeze_store_init")
    .set_op_category("RegOp")
    .set_description("Initialize a region-local UReg squeeze-store state")
    .set_reg_op("pto.init_align")
    .add_argument("dst", "Destination Vec Tile")
    .add_argument("offset", "Initial element offset")
    .set_input_memory(0, MemorySpace::Vec)
    .no_arg_writes()
    .f_deduce_type([](const Args& a, const Kwargs&) -> TypePtr {
      CHECK(a.size() == 2) << "squeeze_store_init expects destination and offset";
      auto tile = Memory(a, 0, false, /*linear_destination=*/true);
      CHECK(tile->dtype_ == DataType::UINT16 || tile->dtype_ == DataType::UINT32)
          << "squeeze_store requires UINT16/UINT32 destination";
      return RegType::GetUReg();
    });

REGISTER_OP("reg.squeeze_store_step")
    .set_op_category("RegOp")
    .set_description("Append selected lanes and return the next UReg squeeze-store state")
    .set_reg_op("pto.vstur")
    .add_argument("ureg", "Current UReg squeeze-store state")
    .add_argument("dst", "The stream destination")
    .add_argument("value", "One physical UINT16/UINT32 VecReg")
    .add_argument("mask", "Matching MaskReg")
    .set_input_memory(1, MemorySpace::Vec)
    .set_arg_effect(1, ArgEffect::Write)
    .f_deduce_type([](const Args& a, const Kwargs& k) -> TypePtr {
      CHECK(a.size() == 4) << "squeeze_store_step expects UReg value, destination, VecReg and MaskReg";
      CHECK(IsURegType(a[0]->GetType())) << "Expected current UReg squeeze-store state";
      auto packed = OpRegistry::GetInstance().Create("reg.squeeze", {a[2], a[3]}, k, a[2]->span_);
      auto tile = As<TileType>(a[1]->GetType());
      CHECK(tile && tile->GetMemorySpace() == MemorySpace::Vec) << "Expected Vec Tile destination";
      CHECK(tile->dtype_ == As<RegType>(packed->GetType())->GetDType())
          << "Squeeze destination dtype mismatch";
      CHECK(IsLinearContiguousVecTile(*tile))
          << "squeeze_store destination must have a contiguous linear Vec Tile view; flatten the Tile or "
             "use a contiguous row-major view";
      return RegType::GetUReg();
    });

REGISTER_OP("reg.squeeze_store_flush")
    .set_op_category("RegOp")
    .set_description("Finalize the UReg squeeze-store state and return the destination alias")
    .set_reg_op("pto.vstar")
    .add_argument("ureg", "Final UReg squeeze-store state")
    .add_argument("dst", "The stream destination")
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .set_arg_effect(1, ArgEffect::Write)
    .set_output_reuses_input(1)
    .f_deduce_type([](const Args& a, const Kwargs&) -> TypePtr {
      CHECK(a.size() == 2) << "squeeze_store_flush expects final UReg value and destination";
      CHECK(IsURegType(a[0]->GetType())) << "Expected final UReg squeeze-store state";
      auto tile = As<TileType>(a[1]->GetType());
      CHECK(tile && tile->GetMemorySpace() == MemorySpace::Vec &&
            (tile->dtype_ == DataType::UINT16 || tile->dtype_ == DataType::UINT32))
          << "Expected UINT16/UINT32 Vec Tile destination";
      CHECK(IsLinearContiguousVecTile(*tile))
          << "squeeze_store destination must have a contiguous linear Vec Tile view; flatten the Tile or "
             "use a contiguous row-major view";
      return tile;
    });

RegMemoryEffect GetRegMemoryEffect(const CallPtr& call) {
  if (IsOp(call, "reg.squeeze_store_step") || IsOp(call, "reg.squeeze_store_flush")) {
    (void)OpRegistry::GetInstance().Create(call->op_->name_, call->args_, call->kwargs_, call->span_);
    auto tile = As<TileType>(call->args_[1]->GetType());
    // Selected counts and the post-updated cursor are dynamic. Neither a full
    // Tile overwrite nor initialization of its untouched tail can be inferred.
    return {call->args_[1],   tile->memref_ ? (*tile->memref_)->base_ : nullptr,
            ArgEffect::Write, std::nullopt,
            std::nullopt,     IsOp(call, "reg.squeeze_store_step") ? call->args_[3] : nullptr};
  }

  CHECK(IsOp(call, "reg.load") || IsOp(call, "reg.store") || IsOp(call, "reg.gather") ||
        IsOp(call, "reg.scatter"))
      << "Memory effect requires reg.load/store/gather/scatter";
  // This query is public and can receive a manually created Call. Validate
  // operand arity and types before indexing, even if its result type is forged.
  auto validated =
      OpRegistry::GetInstance().Create(call->op_->name_, call->args_, call->kwargs_, call->span_);
  bool load = IsOp(call, "reg.load") || IsOp(call, "reg.gather");
  bool indexed = IsOp(call, "reg.gather") || IsOp(call, "reg.scatter");
  auto tile = As<TileType>(call->args_[0]->GetType());
  auto validated_reg = As<RegType>(load ? validated->GetType() : call->args_[2]->GetType());
  INTERNAL_CHECK_SPAN(validated_reg, call->span_)
      << "Register memory effect validation produced a non-register result";
  int64_t extent = IsOp(call, "reg.store")
                       ? StoreExtent(call->args_, call->kwargs_)
                       : (IsOp(call, "reg.load") && GetKwargOr<bool>(call->kwargs_, "native", false)
                              ? NativeLoadExtent(call->kwargs_)
                              : validated_reg->GetShape());
  // Endpoints are half-open ELEMENT coordinates relative to MemRef.base,
  // including both byte_offset / element_bytes and the TileView origin.
  // An absent root or endpoint means unknown, never an empty/zero access.
  // Optional static initialization analysis converts these coordinates to
  // bytes; codegen emits the pointer-relative offset independently.
  RegMemoryEffect result{
      call->args_[0],
      tile->memref_ ? (*tile->memref_)->base_ : nullptr,
      load ? ArgEffect::Read : ArgEffect::Write,
      std::nullopt,
      std::nullopt,
      indexed ? call->args_[load ? 2 : 3] : (call->args_.size() == 4 ? call->args_[3] : nullptr)};
  // An index vector describes an unknown set, never a contiguous lane interval.
  if (indexed) return result;
  auto offset = As<ConstInt>(call->args_[1]);
  ExprPtr base = tile->tile_view_ ? tile->tile_view_->start_offset : nullptr;
  auto base_const = As<ConstInt>(base);
  if (offset && (!base || base_const)) {
    int64_t start = base_const ? base_const->value_ : 0;
    if (tile->memref_) {
      auto bytes = As<ConstInt>((*tile->memref_)->byte_offset_);
      int64_t element_bytes = static_cast<int64_t>(tile->dtype_.GetByte());
      if (!bytes || bytes->value_ < 0 || bytes->value_ % element_bytes != 0 ||
          start > std::numeric_limits<int64_t>::max() - bytes->value_ / element_bytes) {
        return result;
      }
      start += bytes->value_ / element_bytes;
    }
    if (start >= 0 && offset->value_ >= 0 && start <= std::numeric_limits<int64_t>::max() - offset->value_ &&
        start + offset->value_ <= std::numeric_limits<int64_t>::max() - extent) {
      result.begin = start + offset->value_;
      result.end = *result.begin + extent;
    }
  }
  // TODO(reg-vmi-generalize): carry symbolic ranges for optional analysis of
  // dynamic accesses. Unknown endpoints already pass through to PTOAS VMI.
  return result;
}
}  // namespace pypto::ir
