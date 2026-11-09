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
#include "pypto/ir/scalar_expr.h"
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
  // A distributed window is this rank's local GM slice, with the same
  // addressing and permission contract as an ordinary Tensor.
  const auto tensor = AsTensorTypeLike(args[0]->GetType());
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

// Atomics reuse logical scalar addressing, but are always RMW, including Exch:
// the returned old value reads storage even when the caller discards it.
TypePtr AtomicType(const std::vector<ExprPtr>& args, const Kwargs& kwargs, bool cas) {
  const size_t arity = cas ? 4 : 3;
  CHECK(args.size() == arity) << "SIMT atomic requires exactly " << arity << " arguments";
  MemoryType({args[0], args[1], args.back()}, kwargs, true);
  const auto value = As<ScalarType>(args.back()->GetType());
  CHECK(value->dtype_ == DataType::INT32 || value->dtype_ == DataType::FP32)
      << "SIMT atomic requires INT32 or FP32 storage and values";
  if (cas) {
    const auto compare = As<ScalarType>(args[2]->GetType());
    CHECK(compare && compare->dtype_ == value->dtype_) << "SIMT CAS compare dtype must match storage";
  }
  return args.back()->GetType();
}

TypePtr SynchronizationType(const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
  CHECK(args.empty() && kwargs.empty()) << "SIMT synchronization takes no arguments or attributes";
  return GetVoidType();
}

// Lane collectives depend on the active execution mask as well as their
// explicit operands. no_arg_writes is only a storage-effect declaration: it
// must not be interpreted as permission to hoist them across control flow.
enum class CollectiveKind { Vote, Ballot, Shuffle, Redux };

TypePtr CollectiveType(const std::vector<ExprPtr>& args, const Kwargs& kwargs, const std::string& name,
                       CollectiveKind kind) {
  const bool shuffle = kind == CollectiveKind::Shuffle;
  const size_t arity = shuffle ? 2 : 1;
  CHECK(args.size() == arity) << name << " requires exactly " << arity << " arguments, got " << args.size();
  const auto value = As<ScalarType>(args[0]->GetType());
  if (kind == CollectiveKind::Vote || kind == CollectiveKind::Ballot) {
    CHECK(value && value->dtype_ == DataType::BOOL) << name << " requires a BOOL predicate";
    CHECK(kwargs.empty()) << name << " does not accept attributes";
    // The ballot is an INT32 bit carrier, including bit 31, not a lane count.
    return std::make_shared<ScalarType>(kind == CollectiveKind::Ballot ? DataType::INT32 : DataType::BOOL);
  }
  CHECK(value && (value->dtype_ == DataType::INT32 || value->dtype_ == DataType::FP32))
      << name << " requires an INT32 or FP32 scalar value";
  if (shuffle) {
    const auto control = As<ScalarType>(args[1]->GetType());
    CHECK(control && control->dtype_ == DataType::INT32) << name << " requires an INT32 lane control";
    if (const auto constant = As<ConstInt>(args[1])) {
      CHECK(constant->value_ >= std::numeric_limits<int32_t>::min() &&
            constant->value_ <= std::numeric_limits<int32_t>::max())
          << name << " lane control must fit INT32";
    }
    for (const auto& [key, attr] : kwargs) {
      CHECK(key == "width") << name << " only accepts the width attribute";
      const auto width = AnyCast<int>(attr, key);
      CHECK(width == 16 || width == 32) << name << " width must be 16 or 32, got " << width;
    }
  } else {
    CHECK(kwargs.empty()) << name << " does not accept attributes; signedness follows the value dtype";
  }
  return args[0]->GetType();
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
    .set_write_channel(WriteChannel::Simt)
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return MemoryType(args, kwargs, true);
    });

REGISTER_OP("simt.vote_all")
    .set_description("Active-lane vote_all collective")
    .set_op_category("SimtOp")
    .add_argument("predicate", "Current lane scalar")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return CollectiveType(args, kwargs, "simt.vote_all", CollectiveKind::Vote);
    });
REGISTER_OP("simt.vote_any")
    .set_description("Active-lane vote_any collective")
    .set_op_category("SimtOp")
    .add_argument("predicate", "Current lane scalar")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return CollectiveType(args, kwargs, "simt.vote_any", CollectiveKind::Vote);
    });
REGISTER_OP("simt.vote_uni")
    .set_description("Active-lane vote_uni collective")
    .set_op_category("SimtOp")
    .add_argument("predicate", "Current lane scalar")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return CollectiveType(args, kwargs, "simt.vote_uni", CollectiveKind::Vote);
    });
REGISTER_OP("simt.vote_ballot")
    .set_description("Active-lane vote_ballot collective")
    .set_op_category("SimtOp")
    .add_argument("predicate", "Current lane scalar")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return CollectiveType(args, kwargs, "simt.vote_ballot", CollectiveKind::Ballot);
    });
REGISTER_OP("simt.shuffle_idx")
    .set_description("Active-lane shuffle_idx collective")
    .set_op_category("SimtOp")
    .add_argument("value", "Current lane scalar")
    .add_argument("control", "INT32 source lane, offset or XOR mask")
    .set_attr<int>("width")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return CollectiveType(args, kwargs, "simt.shuffle_idx", CollectiveKind::Shuffle);
    });
REGISTER_OP("simt.shuffle_up")
    .set_description("Active-lane shuffle_up collective")
    .set_op_category("SimtOp")
    .add_argument("value", "Current lane scalar")
    .add_argument("control", "INT32 source lane, offset or XOR mask")
    .set_attr<int>("width")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return CollectiveType(args, kwargs, "simt.shuffle_up", CollectiveKind::Shuffle);
    });
REGISTER_OP("simt.shuffle_down")
    .set_description("Active-lane shuffle_down collective")
    .set_op_category("SimtOp")
    .add_argument("value", "Current lane scalar")
    .add_argument("control", "INT32 source lane, offset or XOR mask")
    .set_attr<int>("width")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return CollectiveType(args, kwargs, "simt.shuffle_down", CollectiveKind::Shuffle);
    });
REGISTER_OP("simt.shuffle_bfly")
    .set_description("Active-lane shuffle_bfly collective")
    .set_op_category("SimtOp")
    .add_argument("value", "Current lane scalar")
    .add_argument("control", "INT32 source lane, offset or XOR mask")
    .set_attr<int>("width")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return CollectiveType(args, kwargs, "simt.shuffle_bfly", CollectiveKind::Shuffle);
    });
REGISTER_OP("simt.redux_add")
    .set_description("Active-lane redux_add collective")
    .set_op_category("SimtOp")
    .add_argument("value", "Current lane scalar")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return CollectiveType(args, kwargs, "simt.redux_add", CollectiveKind::Redux);
    });
REGISTER_OP("simt.redux_min")
    .set_description("Active-lane redux_min collective")
    .set_op_category("SimtOp")
    .add_argument("value", "Current lane scalar")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return CollectiveType(args, kwargs, "simt.redux_min", CollectiveKind::Redux);
    });
REGISTER_OP("simt.redux_max")
    .set_description("Active-lane redux_max collective")
    .set_op_category("SimtOp")
    .add_argument("value", "Current lane scalar")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return CollectiveType(args, kwargs, "simt.redux_max", CollectiveKind::Redux);
    });
REGISTER_OP("simt.atomic_add")
    .set_description("Atomic add on one initialized UB or GM element; return its old value")
    .set_op_category("SimtOp")
    .add_argument("storage", "Initialized borrowed UB Tile or GM Tensor")
    .add_argument("indices", "Logical element coordinates")
    .add_argument("value", "Scalar update value")
    .set_attr<std::string>("l2cache")
    .set_arg_effect(0, ArgEffect::ReadWrite)
    .set_write_channel(WriteChannel::Simt)
    // Partial destination access intentionally retains Unknown execution-memory metadata.
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return AtomicType(args, kwargs, false);
    });
REGISTER_OP("simt.atomic_min")
    .set_description("Atomic min on one initialized UB or GM element; return its old value")
    .set_op_category("SimtOp")
    .add_argument("storage", "Initialized borrowed UB Tile or GM Tensor")
    .add_argument("indices", "Logical element coordinates")
    .add_argument("value", "Scalar update value")
    .set_attr<std::string>("l2cache")
    .set_arg_effect(0, ArgEffect::ReadWrite)
    .set_write_channel(WriteChannel::Simt)
    // Partial destination access intentionally retains Unknown execution-memory metadata.
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return AtomicType(args, kwargs, false);
    });
REGISTER_OP("simt.atomic_max")
    .set_description("Atomic max on one initialized UB or GM element; return its old value")
    .set_op_category("SimtOp")
    .add_argument("storage", "Initialized borrowed UB Tile or GM Tensor")
    .add_argument("indices", "Logical element coordinates")
    .add_argument("value", "Scalar update value")
    .set_attr<std::string>("l2cache")
    .set_arg_effect(0, ArgEffect::ReadWrite)
    .set_write_channel(WriteChannel::Simt)
    // Partial destination access intentionally retains Unknown execution-memory metadata.
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return AtomicType(args, kwargs, false);
    });
REGISTER_OP("simt.atomic_exch")
    .set_description("Atomic exch on one initialized UB or GM element; return its old value")
    .set_op_category("SimtOp")
    .add_argument("storage", "Initialized borrowed UB Tile or GM Tensor")
    .add_argument("indices", "Logical element coordinates")
    .add_argument("value", "Scalar update value")
    .set_attr<std::string>("l2cache")
    .set_arg_effect(0, ArgEffect::ReadWrite)
    .set_write_channel(WriteChannel::Simt)
    // Partial destination access intentionally retains Unknown execution-memory metadata.
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return AtomicType(args, kwargs, false);
    });
REGISTER_OP("simt.atomic_cas")
    .set_description("Atomic cas on one initialized UB or GM element; return its old value")
    .set_op_category("SimtOp")
    .add_argument("storage", "Initialized borrowed UB Tile or GM Tensor")
    .add_argument("indices", "Logical element coordinates")
    .add_argument("compare", "Expected old value")
    .add_argument("value", "Scalar update value")
    .set_attr<std::string>("l2cache")
    .set_arg_effect(0, ArgEffect::ReadWrite)
    .set_write_channel(WriteChannel::Simt)
    // Partial destination access intentionally retains Unknown execution-memory metadata.
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return AtomicType(args, kwargs, true);
    });
REGISTER_OP("simt.syncthreads")
    .set_description("Explicit SIMT syncthreads synchronization effect")
    .set_op_category("SimtOp")
    .set_output_arity(0)
    .no_argument()
    // No explicit storage arguments; this does not imply purity. Calls remain
    // ordered effects, and Unknown execution-memory metadata is conservative.
    .no_arg_writes()
    .f_deduce_type(SynchronizationType);
REGISTER_OP("simt.threadfence")
    .set_description("Explicit SIMT threadfence synchronization effect")
    .set_op_category("SimtOp")
    .set_output_arity(0)
    .no_argument()
    // No explicit storage arguments; this does not imply purity. Calls remain
    // ordered effects, and Unknown execution-memory metadata is conservative.
    .no_arg_writes()
    .f_deduce_type(SynchronizationType);
REGISTER_OP("simt.threadfence_block")
    .set_description("Explicit SIMT threadfence_block synchronization effect")
    .set_op_category("SimtOp")
    .set_output_arity(0)
    .no_argument()
    // No explicit storage arguments; this does not imply purity. Calls remain
    // ordered effects, and Unknown execution-memory metadata is conservative.
    .no_arg_writes()
    .f_deduce_type(SynchronizationType);
}  // namespace pypto::ir
