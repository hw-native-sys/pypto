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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_ACC_INIT_BUILDER_H_
#define PYPTO_IR_TRANSFORMS_UTILS_ACC_INIT_BUILDER_H_

#include <string>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"

namespace pypto {
namespace ir {
namespace acc_init {

/// A freshly declared L0C accumulator: the statements that declare it, in
/// order, and the value later statements should read it through.
struct AccInit {
  std::vector<StmtPtr> stmts;
  VarPtr value;
};

/**
 * @brief Declare an Acc (L0C) box, optionally packed at the valid-region pitch
 *
 * ``tile.create``'s deducer honors ``target_memory=Acc`` and emits the Nz TileView
 * ``(col_major, row_major, fractal)`` a matmul result carries, so an accumulator declared here
 * lines up structurally with the ``tile.matmul[_acc]`` values that flow into the same carry.
 *
 * @param shape Physical box shape (compile-time constants, as ``tile.create`` requires)
 * @param dtype Accumulator element type
 * @param name_hint Name for the bound Var
 * @param span Source location to attribute the declaration to
 * @param compact Declare the box written at ``ceil(validRow/16)*16`` rather than at its physical rows
 * @return The ``AssignStmt`` binding the fresh box
 */
AssignStmtPtr BuildAccStorage(const std::vector<ExprPtr>& shape, const DataType& dtype,
                              const std::string& name_hint, const Span& span, bool compact = false);

/**
 * @brief Declare an Acc accumulator whose valid rectangle may be narrower than its box
 *
 * The narrowed form is a ``tile.create`` followed by ``tile.set_validshape``. Declaring the box
 * compact (rather than stamping the mode onto a type afterwards) is what makes the mode survive:
 * a pass-applied refinement is discarded the moment any pass re-deduces the call, whereas the
 * ``compact`` kwarg is re-read by the deducer. ``tile.set_validshape`` then inherits the mode onto
 * the narrowed view without re-interpreting bytes it did not write.
 *
 * A box whose valid rectangle provably fills it keeps the historical single-``tile.create`` form.
 *
 * @param shape Physical box shape (compile-time constants)
 * @param valid Valid extents, same rank as @p shape; row extent may be dynamic
 * @param dtype Accumulator element type
 * @param name_hint Name for the value the caller reads the accumulator through
 * @param span Source location to attribute the declaration to
 * @return The declaring statements and the value to use
 */
AccInit BuildNarrowedAccInit(const std::vector<ExprPtr>& shape, const std::vector<ExprPtr>& valid,
                             const DataType& dtype, const std::string& name_hint, const Span& span);

}  // namespace acc_init
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_ACC_INIT_BUILDER_H_
