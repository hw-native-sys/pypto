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

#include "pypto/ir/transforms/op_conversion_registry.h"

#include <any>
#include <atomic>
#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

OpConversionRegistry& OpConversionRegistry::GetInstance() {
  static OpConversionRegistry instance;
  return instance;
}

OpConversionRegistry::OpConversionRegistry() {
  // Register default simple conversions (tensor op -> block op)
  auto register_row_reduce = [this](const std::string& from_op, const std::string& to_op) {
    RegisterCustom(from_op, [from_op, to_op](const std::vector<ExprPtr>& args,
                                             const std::vector<std::pair<std::string, std::any>>& kwargs,
                                             const Span& span) -> ConversionResult {
      CHECK(args.size() == 1) << from_op << " conversion expects exactly 1 argument, but got " << args.size();
      auto tile_type = ::pypto::ir::As<TileType>(args[0]->GetType());
      CHECK(tile_type) << from_op << " conversion requires TileType input, but got "
                       << args[0]->GetType()->TypeName();

      static std::atomic<int64_t> counter{0};
      auto tmp_id = counter.fetch_add(1);
      auto tmp_name = "tmp_row_reduce_" + std::to_string(tmp_id);
      auto shape_tuple = std::make_shared<MakeTuple>(tile_type->shape_, span);
      std::vector<std::pair<std::string, std::any>> create_kwargs;
      create_kwargs.emplace_back("dtype", tile_type->dtype_);
      create_kwargs.emplace_back("target_memory", ::pypto::ir::MemorySpace::Vec);
      std::vector<ExprPtr> create_args{shape_tuple};
      auto create_call = OpRegistry::GetInstance().Create("block.create_tile", create_args, create_kwargs, span);
      auto tmp_var = std::make_shared<Var>(tmp_name, create_call->GetType(), span);
      std::vector<ExprPtr> result_args{args[0], tmp_var};
      auto result_call = OpRegistry::GetInstance().Create(to_op, result_args, span);
      return ConversionResult({std::make_shared<AssignStmt>(tmp_var, create_call, span)}, result_call);
    });
  };

  // Elementwise binary ops
  RegisterSimple("tensor.add", "block.add");
  RegisterSimple("tensor.sub", "block.sub");
  RegisterSimple("tensor.mul", "block.mul");
  RegisterSimple("tensor.div", "block.div");
  RegisterSimple("tensor.maximum", "block.maximum");

  // Scalar ops
  RegisterSimple("tensor.add_scalar", "block.adds");
  RegisterSimple("tensor.sub_scalar", "block.subs");
  RegisterSimple("tensor.mul_scalar", "block.muls");
  RegisterSimple("tensor.div_scalar", "block.divs");
  RegisterSimple("tensor.adds", "block.adds");
  RegisterSimple("tensor.subs", "block.subs");
  RegisterSimple("tensor.muls", "block.muls");
  RegisterSimple("tensor.divs", "block.divs");
  RegisterSimple("tensor.rems", "block.rems");
  RegisterSimple("tensor.maxs", "block.maxs");
  RegisterSimple("tensor.mins", "block.mins");
  RegisterSimple("tensor.ands", "block.ands");
  RegisterSimple("tensor.ors", "block.ors");
  RegisterSimple("tensor.shls", "block.shls");
  RegisterSimple("tensor.shrs", "block.shrs");
  RegisterSimple("tensor.cmps", "block.cmps");
  RegisterSimple("tensor.expands", "block.expands");
  RegisterSimple("tensor.lrelu", "block.lrelu");
  RegisterSimple("tensor.full", "block.full");

  // Unary ops
  RegisterSimple("tensor.neg", "block.neg");
  RegisterSimple("tensor.exp", "block.exp");
  RegisterSimple("tensor.recip", "block.recip");
  RegisterSimple("tensor.sqrt", "block.sqrt");
  RegisterSimple("tensor.rsqrt", "block.rsqrt");
  RegisterSimple("tensor.log", "block.log");
  RegisterSimple("tensor.abs", "block.abs");
  RegisterSimple("tensor.relu", "block.relu");
  RegisterSimple("tensor.not", "block.not");
  RegisterSimple("tensor.cast", "block.cast");
  RegisterSimple("tensor.fillpad", "block.fillpad");

  // Transform ops
  RegisterSimple("tensor.reshape", "block.reshape");
  RegisterSimple("tensor.transpose", "block.transpose");

  // Broadcast / expand ops
  RegisterSimple("tensor.row_expand", "block.row_expand");
  RegisterSimple("tensor.row_expand_sub", "block.row_expand_sub");
  RegisterSimple("tensor.row_expand_div", "block.row_expand_div");
  RegisterSimple("tensor.row_expand_mul", "block.row_expand_mul");
  RegisterSimple("tensor.row_expand_add", "block.row_expand_add");
  RegisterSimple("tensor.col_expand", "block.col_expand");
  RegisterSimple("tensor.col_expand_mul", "block.col_expand_mul");
  RegisterSimple("tensor.col_expand_div", "block.col_expand_div");
  RegisterSimple("tensor.col_expand_sub", "block.col_expand_sub");

  // Additional elementwise ops
  RegisterSimple("tensor.minimum", "block.minimum");
  RegisterSimple("tensor.rem", "block.rem");
  RegisterSimple("tensor.and", "block.and");
  RegisterSimple("tensor.or", "block.or");
  RegisterSimple("tensor.shl", "block.shl");
  RegisterSimple("tensor.shr", "block.shr");
  RegisterSimple("tensor.xor", "block.xor");
  RegisterSimple("tensor.xors", "block.xors");
  RegisterSimple("tensor.prelu", "block.prelu");
  RegisterSimple("tensor.addc", "block.addc");
  RegisterSimple("tensor.subc", "block.subc");
  RegisterSimple("tensor.addsc", "block.addsc");
  RegisterSimple("tensor.subsc", "block.subsc");
  RegisterSimple("tensor.sel", "block.sel");
  RegisterSimple("tensor.sels", "block.sels");
  RegisterSimple("tensor.cmp", "block.cmp");

  // Reduction ops with matching signatures
  RegisterSimple("tensor.sum", "block.sum");
  RegisterSimple("tensor.max", "block.max");
  RegisterSimple("tensor.min", "block.min");

  // Matmul family with matching signatures
  RegisterSimple("tensor.matmul_acc", "block.matmul_acc");
  RegisterSimple("tensor.matmul_bias", "block.matmul_bias");
  RegisterSimple("tensor.gemv", "block.gemv");
  RegisterSimple("tensor.gemv_acc", "block.gemv_acc");
  RegisterSimple("tensor.gemv_bias", "block.gemv_bias");

  // Row reductions need a synthesized tmp tile on the block side.
  register_row_reduce("tensor.row_max", "block.row_max");
  register_row_reduce("tensor.row_sum", "block.row_sum");
  register_row_reduce("tensor.row_min", "block.row_min");
}

void OpConversionRegistry::RegisterSimple(const std::string& from_op, const std::string& to_op) {
  // Capture to_op by value for the lambda
  conversions_[from_op] = [to_op](const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs,
                                  const Span& span) -> ConversionResult {
    auto& reg = OpRegistry::GetInstance();
    CallPtr call;
    if (kwargs.empty()) {
      call = reg.Create(to_op, args, span);
    } else {
      call = reg.Create(to_op, args, kwargs, span);
    }
    return ConversionResult{call};
  };
}

void OpConversionRegistry::RegisterCustom(const std::string& from_op, ConversionFunc func) {
  conversions_[from_op] = std::move(func);
}

const ConversionFunc* OpConversionRegistry::Lookup(const std::string& op_name) const {
  auto it = conversions_.find(op_name);
  if (it == conversions_.end()) {
    return nullptr;
  }
  return &it->second;
}

bool OpConversionRegistry::HasConversion(const std::string& op_name) const {
  return conversions_.count(op_name) > 0;
}

}  // namespace ir
}  // namespace pypto
