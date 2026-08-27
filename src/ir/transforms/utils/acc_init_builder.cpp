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

#include "pypto/ir/transforms/utils/acc_init_builder.h"

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {
namespace acc_init {

AssignStmtPtr BuildAccStorage(const std::vector<ExprPtr>& shape, const DataType& dtype,
                              const std::string& name_hint, const Span& span, bool compact) {
  INTERNAL_CHECK_SPAN(!shape.empty(), span) << "Internal error: an Acc box needs a non-empty shape";
  auto& reg = OpRegistry::GetInstance();
  std::vector<std::pair<std::string, std::any>> kwargs = {{"dtype", dtype},
                                                          {"target_memory", MemorySpace::Acc}};
  if (compact) {
    kwargs.emplace_back("compact", true);
  }
  auto call = reg.Create("tile.create", {std::make_shared<MakeTuple>(shape, span)}, kwargs, span);
  auto var = std::make_shared<Var>(name_hint, call->GetType(), span);
  return std::make_shared<AssignStmt>(var, call, span);
}

AccInit BuildNarrowedAccInit(const std::vector<ExprPtr>& shape, const std::vector<ExprPtr>& valid,
                             const DataType& dtype, const std::string& name_hint, const Span& span) {
  INTERNAL_CHECK_SPAN(shape.size() == 2 && valid.size() == 2, span)
      << "Internal error: a narrowed Acc accumulator is 2D; got shape rank " << shape.size()
      << " and valid rank " << valid.size();
  INTERNAL_CHECK_SPAN(valid[0] && valid[1], span)
      << "Internal error: accumulator initializer requires two valid extents";

  // A statically full rectangle keeps the historical single-`tile.create` form.
  auto valid_rows_const = As<ConstInt>(valid[0]);
  auto valid_cols_const = As<ConstInt>(valid[1]);
  auto rows_const = As<ConstInt>(shape[0]);
  auto cols_const = As<ConstInt>(shape[1]);
  if (valid_rows_const && valid_cols_const && rows_const && cols_const &&
      valid_rows_const->value_ == rows_const->value_ && valid_cols_const->value_ == cols_const->value_) {
    auto init = BuildAccStorage(shape, dtype, name_hint, span);
    return AccInit{{init}, init->var_};
  }
  const bool rows_fill_box = ProveValidExtentEqual(valid[0], shape[0]) == ProofResult::kTrue;

  // Declare the buffer compact when its valid rows are not provably its physical rows -- the same
  // predicate `StampCompactForNarrowedAccRows` applies to a matmul result, because the `mad` that
  // writes this buffer takes M from the L0A operand's valid rows. `tile.set_validshape` below
  // inherits the mode, so the iter_arg, every `tile.matmul_acc` that accumulates into it (the op
  // inherits its accumulator's compact mode) and the reader after the loop all agree with the pitch
  // the hardware used. Leaving the seed non-compact silently skews every N-fractal above the first
  // (#2470, #2510).
  auto storage = BuildAccStorage(shape, dtype, name_hint + "_storage", span, !rows_fill_box);
  auto& reg = OpRegistry::GetInstance();
  auto narrowed_call = reg.Create("tile.set_validshape", {storage->var_, valid[0], valid[1]}, span);
  auto narrowed_var = std::make_shared<Var>(name_hint, narrowed_call->GetType(), span);
  auto narrowed = std::make_shared<AssignStmt>(narrowed_var, narrowed_call, span);
  return AccInit{{storage, narrowed}, narrowed_var};
}

}  // namespace acc_init
}  // namespace ir
}  // namespace pypto
