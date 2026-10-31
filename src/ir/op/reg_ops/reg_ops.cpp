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

RegTypePtr Register(const ExprPtr& e, RegKind kind) {
  auto t = As<RegType>(e->GetType());
  CHECK_SPAN(t && t->kind_ == kind, e->span_)
      << "Expected " << (kind == RegKind::VecReg ? "VecReg" : "MaskReg");
  CHECK_SPAN(kind == RegKind::MaskReg || Supported(t->dtype_), e->span_)
      << "Unsupported register dtype " << t->dtype_.ToString();
  return t;
}

void Match(const RegTypePtr& a, const RegTypePtr& b) {
  CHECK(a->shape_ == b->shape_) << "Register lane shape mismatch";
  CHECK(a->dtype_ == b->dtype_) << "Register dtype mismatch (including signedness)";
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

TileTypePtr Memory(const Args& args, int64_t lanes, bool read) {
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
  if (t->tile_view_) {
    const auto& v = *t->tile_view_;
    // TODO(reg-vmi-generalize): PTOAS VMI supports block/group/strided loads,
    // but the P0 reg.load/store ABI emits only the contiguous vload/vstore
    // form. Remove this restriction when those access modes are represented
    // in the RegOp attributes and lowered by reg_codegen.cpp.
    CHECK(v.blayout == TileLayout::row_major && v.slayout == TileLayout::none_box &&
          v.compact == CompactMode::null)
        << "Register memory requires a contiguous row-major View";
    // Verify the full physical stride, not just the last dimension: flattening
    // a padded row would otherwise read padding as the next logical row.
    if (!v.stride.empty()) {
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
    CHECK(t->shape_.size() == 1) << "2D register memory access requires an explicit contiguous View";
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
  if (a.size() == 3) CHECK(Register(a[2], RegKind::MaskReg)->shape_ == x->shape_) << "Mask lane mismatch";
  CHECK(GetStringKwarg(k, "pmode", "zero") == "zero") << "P0 pmode must be zero (merge unsupported)";
  CHECK(name != "mul" || (x->dtype_ != DataType::INT8 && x->dtype_ != DataType::UINT8))
      << "reg.mul rejects 8-bit dtype";
  CHECK(name != "div" || x->dtype_ == DataType::FP16 || x->dtype_ == DataType::FP32)
      << "reg.div requires FP16/FP32 dtype";
  return x;
}

TypePtr Convert(const Args& a, const Kwargs& k) {
  CHECK(a.size() == 1) << "reg.convert expects one VecReg";
  auto x = Register(a[0], RegKind::VecReg);
  auto dst = GetRequiredKwarg<DataType>(k, "dtype", "reg.convert");
  const auto src = x->dtype_;
  bool narrow_float = src == DataType::FP32 && dst == DataType::FP16;
  bool float_int = src == DataType::FP32 && dst == DataType::INT32;
  bool narrow_int = src == DataType::INT32 && dst == DataType::INT8;
  bool plain =
      (src == DataType::FP16 && dst == DataType::FP32) || (src == DataType::INT32 && dst == DataType::FP32) ||
      (src == DataType::INT8 && dst == DataType::FP16) || (src == DataType::INT8 && dst == DataType::INT16) ||
      (src == DataType::INT16 && dst == DataType::INT32) ||
      (src == DataType::UINT8 && dst == DataType::UINT16) ||
      (src == DataType::UINT16 && dst == DataType::UINT32);
  CHECK(plain || narrow_float || float_int || narrow_int || src == dst) << "Unsupported convert dtype pair";
  bool rounding = false, saturate = false;
  for (const auto& [key, value] : k) {
    if (key == "rounding") {
      rounding = true;
      auto mode = AnyCast<std::string>(value, key);
      CHECK((narrow_float && (mode == "R" || mode == "A" || mode == "H" || mode == "Z")) ||
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
  return std::make_shared<RegType>(x->shape_, dst, RegKind::VecReg);
}
bool Integer(DataType dtype) { return dtype.IsInt() && dtype.GetBit() <= 32 && dtype.GetBit() >= 8; }
bool Float(DataType dtype) { return dtype == DataType::FP16 || dtype == DataType::FP32; }

RegTypePtr Vec(const ExprPtr& expr) {
  auto type = Register(expr, RegKind::VecReg);
  CHECK_SPAN(Integer(type->dtype_) || Float(type->dtype_), expr->span_)
      << "Unsupported VF dtype " << type->dtype_.ToString();
  return type;
}

RegTypePtr Mask(const ExprPtr& expr, const RegTypePtr& vector) {
  auto mask = Register(expr, RegKind::MaskReg);
  CHECK_SPAN(mask->shape_ == vector->shape_, expr->span_) << "Mask lane mismatch";
  return mask;
}

void Pmode(const Kwargs& kwargs) {
  CHECK(GetStringKwarg(kwargs, "pmode", "zero") == "zero") << "VF pmode must be zero";
}

TypePtr Unary(const Args& args, const Kwargs& kwargs, const std::string& name) {
  CHECK(args.size() == 1 || args.size() == 2) << "Unary VF expects value and optional mask";
  auto x = Vec(args[0]);
  auto dtype = x->dtype_;
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
  CHECK(Integer(x->dtype_)) << "Bitwise and shift operations require integer dtype";
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
  auto dtype = shift ? DataType::INT16 : x->dtype_;
  CHECK(scalar && scalar->dtype_ == dtype) << "Scalar dtype mismatch (shifts require i16)";
  if (shift) CHECK(Integer(x->dtype_)) << "Scalar shift requires integer dtype";
  if (name == "muls") CHECK(x->dtype_.GetBit() >= 16) << "reg.muls rejects 8-bit dtype";
  auto mask = Mask(args[2], x);
  Pmode(kwargs);
  if (name != "compares") return x;
  CHECK(Float(x->dtype_) || x->dtype_ == DataType::INT32 || x->dtype_ == DataType::UINT32)
      << "Unsupported compares dtype";
  auto predicate = GetRequiredKwarg<std::string>(kwargs, "predicate", "reg.compares");
  CHECK(predicate == "eq" || predicate == "ne" || predicate == "lt" || predicate == "le" ||
        predicate == "gt" || predicate == "ge")
      << "Invalid compare predicate";
  return mask;
}

TypePtr Reduction(const Args& args, const Kwargs& kwargs, const std::string& name) {
  CHECK(args.size() == 2) << "Reduction requires source and mask";
  auto x = Vec(args[0]);
  CHECK(x->dtype_.GetBit() >= 16) << "Reduction rejects 8-bit dtype";
  Mask(args[1], x);
  if (name == "reduce_sum" && Float(x->dtype_)) {
    CHECK(GetKwargOr<bool>(kwargs, "reassoc", false)) << "Floating reduce_sum requires reassoc=True";
  }
  return std::make_shared<RegType>(1, x->dtype_, RegKind::VecReg);
}

}  // namespace

REGISTER_OP("reg.load")
    .set_op_category("RegOp")
    .set_description("Read a continuous initialized Vec UB interval")
    .set_reg_op("pto.vmi.vload")
    .add_argument("tile", "Source Vec Tile")
    .add_argument("offset", "Element offset")
    .set_attr<int>("lanes")
    .set_input_memory(0, MemorySpace::Vec)
    .set_arg_effect(0, ArgEffect::Read)
    .f_deduce_type([](const Args& a, const Kwargs& k) -> TypePtr {
      CHECK(a.size() == 2) << "reg.load expects tile, offset";
      auto lanes = Lanes(k);
      auto t = Memory(a, lanes, true);
      return std::make_shared<RegType>(lanes, t->dtype_, RegKind::VecReg);
    });
REGISTER_OP("reg.store")
    .set_op_category("RegOp")
    .set_description("Write a continuous Vec UB interval; return the destination alias")
    .set_reg_op("pto.vmi.vstore")
    .add_argument("tile", "Destination Vec Tile")
    .add_argument("offset", "Element offset")
    .add_argument("value", "VecReg value")
    .add_argument("mask", "Optional MaskReg")
    .set_arg_effect(0, ArgEffect::Write)
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .set_output_reuses_input(0)
    .f_deduce_type([](const Args& a, const Kwargs&) -> TypePtr {
      CHECK(a.size() == 3 || a.size() == 4) << "reg.store expects tile, offset, value, optional mask";
      auto v = Register(a[2], RegKind::VecReg);
      auto t = Memory(a, v->shape_, false);
      CHECK(t->dtype_ == v->dtype_) << "Store dtype mismatch";
      if (a.size() == 4) CHECK(Register(a[3], RegKind::MaskReg)->shape_ == v->shape_) << "Mask lane mismatch";
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
    .set_attr<DataType>("dtype")
    .set_attr<std::string>("rounding")
    .set_attr<bool>("saturate")
    .no_arg_writes()
    .f_deduce_type(Convert);

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
        Scalar(a[0], dtype);
      }
      if (op == "active_mask") {
        auto t = As<ScalarType>(a[0]->GetType());
        CHECK(t && t->dtype_ == DataType::INDEX) << "active_lanes must be INDEX";
        auto active = Constant(a[0], "active_lanes");
        CHECK(active >= 0 && active <= n) << "active_lanes must be in [0, lanes]";
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
      CHECK(m->shape_ == x->shape_) << "Mask lane mismatch";
      CHECK(x->dtype_ == DataType::FP16 || x->dtype_ == DataType::FP32 || x->dtype_ == DataType::INT32 ||
            x->dtype_ == DataType::UINT32)
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
      CHECK(v->shape_ == m->shape_) << "Mask lane mismatch";
      CHECK(v->dtype_ != DataType::BF16) << "select rejects BF16 dtype";
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
        .f_deduce_type([name](const Args& args, const Kwargs& kwargs) { return Unary(args, kwargs, name); });
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
        .f_deduce_type(
            [name](const Args& args, const Kwargs& kwargs) { return ScalarOperation(args, kwargs, name); });
    if (name == "compares")
      entry.set_attr<std::string>("predicate");
    else
      entry.set_attr<std::string>("pmode");
  }
  for (const auto& [name, opcode] : std::vector<std::pair<std::string, std::string>>{
           {"reduce_sum", "vcadd"}, {"reduce_max", "vcmax"}, {"reduce_min", "vcmin"}}) {
    auto& entry = OpRegistry::GetInstance().Register("reg." + name);
    entry.set_op_category("RegOp")
        .set_description("Masked reduction to a one-lane vector")
        .set_reg_op("pto.vmi." + opcode)
        .add_argument("value", "VecReg")
        .add_argument("mask", "MaskReg")
        .no_arg_writes()
        .f_deduce_type(
            [name](const Args& args, const Kwargs& kwargs) { return Reduction(args, kwargs, name); });
    if (name == "reduce_sum") entry.set_attr<bool>("reassoc");
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
      auto width = x->dtype_.GetBit();
      CHECK(Integer(indices->dtype_) && indices->dtype_.GetBit() == width) << "Index width mismatch";
      CHECK(indices->shape_ == x->shape_) << "Index lane mismatch";
      CHECK(x->shape_ == 64 || (width <= 16 && x->shape_ == 128) || (width == 8 && x->shape_ == 256))
          << "Unsupported vselr lane count";
      return x;
    });

RegMemoryEffect GetRegMemoryEffect(const CallPtr& call) {
  CHECK(call && call->op_ && (call->op_->name_ == "reg.load" || call->op_->name_ == "reg.store"))
      << "Memory effect requires reg.load/store";
  // This query is public and can receive a manually created Call. Validate
  // operand arity and types before indexing, even if its result type is forged.
  auto validated =
      OpRegistry::GetInstance().Create(call->op_->name_, call->args_, call->kwargs_, call->span_);
  bool load = call->op_->name_ == "reg.load";
  auto tile = As<TileType>(call->args_[0]->GetType());
  auto validated_reg = As<RegType>(load ? validated->GetType() : call->args_[2]->GetType());
  INTERNAL_CHECK_SPAN(validated_reg, call->span_)
      << "Register memory effect validation produced a non-register result";
  int64_t lanes = validated_reg->shape_;
  // Endpoints are half-open ELEMENT coordinates relative to MemRef.base,
  // including both byte_offset / element_bytes and the TileView origin.
  // An absent root or endpoint means unknown, never an empty/zero access.
  // Optional static initialization analysis converts these coordinates to
  // bytes; codegen emits the pointer-relative offset independently.
  RegMemoryEffect result{call->args_[0],
                         tile->memref_ ? (*tile->memref_)->base_ : nullptr,
                         load ? ArgEffect::Read : ArgEffect::Write,
                         std::nullopt,
                         std::nullopt,
                         call->args_.size() == 4 ? call->args_[3] : nullptr};
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
        start + offset->value_ <= std::numeric_limits<int64_t>::max() - lanes) {
      result.begin = start + offset->value_;
      result.end = *result.begin + lanes;
    }
  }
  // TODO(reg-vmi-generalize): carry symbolic ranges for optional analysis of
  // dynamic accesses. Unknown endpoints already pass through to PTOAS VMI.
  return result;
}
}  // namespace pypto::ir
