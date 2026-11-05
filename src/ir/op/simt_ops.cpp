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
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/simt.h"
#include "pypto/ir/type.h"

namespace pypto::ir {
namespace {
using Kwargs = std::vector<std::pair<std::string, std::any>>;

TypePtr QueryType(const std::vector<ExprPtr>&, const Kwargs& kwargs) {
  bool found = false;
  for (const auto& [key, value] : kwargs) {
    if (key == "axis") {
      // Checking the enum value also rejects forged out-of-domain native values.
      SimtAxisToString(AnyCast<SimtAxis>(value, "axis"));
      found = true;
    }
  }
  CHECK(found) << "SIMT coordinate query requires a typed SimtAxis axis";
  return std::make_shared<ScalarType>(DataType::INT32);
}

TypePtr MemoryType(const std::vector<ExprPtr>& args, const Kwargs& kwargs, bool store) {
  // Functional op dispatch reaches type deduction before enforcing positional arity.
  const size_t expected_args = store ? 3 : 2;
  CHECK(args.size() == expected_args) << "simt." << (store ? "store" : "load") << " requires exactly "
                                      << expected_args << " arguments, got " << args.size();
  const auto tile = As<TileType>(args[0]->GetType());
  const auto tensor = As<TensorType>(args[0]->GetType());
  CHECK(tile || tensor) << "SIMT memory access requires Tile (UB) or Tensor (GM) storage";
  if (tile) {
    CHECK(!tile->memory_space_ || tile->memory_space_ == MemorySpace::Vec)
        << "SIMT Tile access requires Vec (UB) memory";
  }
  if (tensor && tensor->tensor_view_) {
    CHECK(tensor->tensor_view_->layout == TensorLayout::ND ||
          tensor->tensor_view_->layout == TensorLayout::DN)
        << "SIMT Tensor access requires an affine ND or DN layout";
  }
  const auto& shape = tile ? tile->shape_ : tensor->shape_;
  const auto dtype = tile ? tile->dtype_ : tensor->dtype_;
  // Packed FP4 carriers occupy a byte but represent two elements, not one scalar.
  CHECK(dtype.GetBit() >= 8 && !dtype.IsPackedFp4() && dtype != DataType::INDEX && dtype != DataType::TASK_ID)
      << "SIMT scalar storage requires an unpacked byte-addressable numeric element dtype";
  auto indices = As<MakeTuple>(args[1]);
  CHECK(indices && indices->elements_.size() == shape.size())
      << "SIMT indices must be a MakeTuple with one index per storage dimension";
  for (const auto& index : indices->elements_) {
    auto scalar = As<ScalarType>(index->GetType());
    CHECK(scalar && (scalar->dtype_.IsInt() || scalar->dtype_ == DataType::INDEX))
        << "SIMT indices must have integer scalar types";
  }
  for (const auto& [key, value] : kwargs) {
    CHECK(tensor) << "SIMT cache policies apply only to GM Tensor storage";
    auto policy = AnyCast<std::string>(value, key);
    if (key == "l1cache") {
      CHECK(policy == "cache" || policy == "uncache") << "Invalid SIMT l1cache policy: " << policy;
    } else if (key == "l2cache") {
      static const std::unordered_set<std::string> load_policies{
          "nmfv",  "nmlv",   "nmprs",   "nmpref", "nakeep", "naclean", "nadrop", "idsfv",
          "idslv", "idsprs", "idspref", "exfv",   "exlv",   "exprs",   "expref"};
      static const std::unordered_set<std::string> store_policies{
          "nmfv",  "nmlv",  "nmprs",  "nmred",  "naci",  "napw",  "napi",   "nared",
          "wbhfv", "wbhlv", "wbhprs", "wbhred", "wtsfv", "wtslv", "wtsprs", "wtsred"};
      CHECK((store ? store_policies : load_policies).count(policy))
          << "Invalid SIMT " << (store ? "store" : "load") << " l2cache policy: " << policy;
    }
  }
  if (store) {
    auto value = As<ScalarType>(args[2]->GetType());
    CHECK(value && value->dtype_ == dtype) << "SIMT store value dtype must match storage element dtype";
    return GetVoidType();
  }
  return std::make_shared<ScalarType>(dtype);
}
}  // namespace

REGISTER_OP("simt.thread_idx")
    .set_description("Query thread coordinate along a typed axis")
    .set_op_category("SimtOp")
    .no_argument()
    .set_attr<SimtAxis>("axis")
    .no_arg_writes()
    .f_deduce_type(QueryType);
REGISTER_OP("simt.block_idx")
    .set_description("Query block coordinate along a typed axis")
    .set_op_category("SimtOp")
    .no_argument()
    .set_attr<SimtAxis>("axis")
    .no_arg_writes()
    .f_deduce_type(QueryType);
REGISTER_OP("simt.block_dim")
    .set_description("Query block extent along a typed axis")
    .set_op_category("SimtOp")
    .no_argument()
    .set_attr<SimtAxis>("axis")
    .no_arg_writes()
    .f_deduce_type(QueryType);
REGISTER_OP("simt.grid_dim")
    .set_description("Query grid extent along a typed axis")
    .set_op_category("SimtOp")
    .no_argument()
    .set_attr<SimtAxis>("axis")
    .no_arg_writes()
    .f_deduce_type(QueryType);
REGISTER_OP("simt.lane_id")
    .set_description("Query lane identifier in the current warp")
    .set_op_category("SimtOp")
    .no_argument()
    .no_arg_writes()
    .f_deduce_type([](const std::vector<ExprPtr>&, const Kwargs& kwargs) {
      CHECK(kwargs.empty()) << "simt.lane_id does not accept axis or other attributes";
      return std::make_shared<ScalarType>(DataType::INT32);
    });
REGISTER_OP("simt.load")
    .set_description("Load one scalar element directly from UB or GM")
    .set_op_category("SimtOp")
    .add_argument("storage", "Borrowed UB Tile or GM Tensor")
    .add_argument("indices", "Logical element coordinates")
    .set_attr<std::string>("l1cache")
    .set_attr<std::string>("l2cache")
    .no_arg_writes()
    // Scalar access is a subrange effect, not a functional whole-storage access.
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return MemoryType(args, kwargs, false);
    });
REGISTER_OP("simt.store")
    .set_description("Store one scalar element directly to UB or GM")
    .set_op_category("SimtOp")
    .set_output_arity(0)
    .add_argument("storage", "Borrowed UB Tile or GM Tensor")
    .add_argument("indices", "Logical element coordinates")
    .add_argument("value", "Scalar element to store")
    .set_attr<std::string>("l1cache")
    .set_attr<std::string>("l2cache")
    // Physical store is Write. The launch preserves untouched elements and is
    // logically InOut; consumers must use the helper access summary for that.
    .set_arg_effect(0, ArgEffect::Write)
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return MemoryType(args, kwargs, true);
    });
}  // namespace pypto::ir
