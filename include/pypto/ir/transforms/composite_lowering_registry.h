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

#ifndef PYPTO_IR_TRANSFORMS_COMPOSITE_LOWERING_REGISTRY_H_
#define PYPTO_IR_TRANSFORMS_COMPOSITE_LOWERING_REGISTRY_H_

#include <cstddef>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"

namespace pypto {
namespace ir {

/**
 * @brief Per-call scratchpad handed to a composite-lowering rule.
 *
 * A rule appends one ``AssignStmt`` per intermediate temp via ``Bind`` and
 * returns the final result ``ExprPtr``. The mutator wraps that result in the
 * original target ``Var`` (or a fresh result ``Var`` for ``ReturnStmt`` calls)
 * before splicing the accumulated statements into the surrounding sequence.
 *
 * The temp counter is borrowed from the mutator so distinct trig calls in the
 * same function get distinct temp names — see ``LowerCompositeOpsMutator``.
 */
class LoweringBuilder {
 public:
  /**
   * @param base_name   Name hint to derive temp names from (typically the
   *                    AssignStmt's LHS ``Var`` name).
   * @param temp_counter Reference to a mutator-owned counter; bumped per Bind.
   */
  LoweringBuilder(std::string base_name, std::size_t& temp_counter);

  /// Append an ``AssignStmt`` binding a fresh ``Var`` to ``expr`` and return
  /// the new ``Var`` so it can be used as input to subsequent ops. The
  /// ``qualifier`` is woven into the temp name for debuggability.
  ExprPtr Bind(const std::string& qualifier, const ExprPtr& expr, const Span& span);

  // Primitive op builders -- type deduction is delegated to OpRegistry so the
  // result preserves the input TileType's shape/layout/dtype.
  ExprPtr Muls(const ExprPtr& x, float c, const Span& span);
  ExprPtr Adds(const ExprPtr& x, float c, const Span& span);
  ExprPtr Add(const ExprPtr& a, const ExprPtr& b, const Span& span);
  ExprPtr Sub(const ExprPtr& a, const ExprPtr& b, const Span& span);
  ExprPtr Mul(const ExprPtr& a, const ExprPtr& b, const Span& span);
  ExprPtr Cast(const ExprPtr& x, DataType to, int mode, const Span& span);

  /// Drain accumulated statements (called by the mutator after the rule
  /// returns).
  std::vector<StmtPtr> TakeStmts() { return std::move(stmts_); }

  const std::string& base_name() const { return base_name_; }

 private:
  std::string MakeTempName(const std::string& qualifier);

  std::string base_name_;
  std::size_t& temp_counter_;
  std::vector<StmtPtr> stmts_;
};

/**
 * @brief Signature for a composite-lowering rule.
 *
 * @param args     Visited operand expressions (var-remap already applied).
 * @param span     Source location of the original call.
 * @param builder  Scratchpad: rule appends intermediate temps via builder.Bind
 *                 and returns the final result expression.
 * @return Final result expression. The mutator binds this to the target ``Var``
 *         and splices the builder's accumulated statements before it.
 */
using CompositeLoweringFn =
    std::function<ExprPtr(const std::vector<ExprPtr>& args, const Span& span, LoweringBuilder& builder)>;

/**
 * @brief Registry of op-name -> lowering rule for ``LowerCompositeOps``.
 *
 * Per-op rules live in their own translation units (e.g.
 * ``src/ir/transforms/composite_ops/sin_cos_lowering.cpp``) and expose a free
 * registrar function (e.g. ``RegisterSinCosLoweringRules``). The registry
 * constructor calls every such registrar so registration is independent of
 * static-initializer ordering and link order.
 *
 * Adding a new composite op:
 *   1. Implement the rule in ``src/ir/transforms/composite_ops/<op>_lowering.cpp``.
 *   2. Expose a ``void Register<Op>LoweringRules(CompositeLoweringRegistry&)``
 *      free function and call it from the registry constructor.
 *   3. Add the new source to ``CMakeLists.txt``.
 *
 * No edits to the dispatch pass are needed.
 */
class CompositeLoweringRegistry {
 public:
  static CompositeLoweringRegistry& GetInstance();

  void Register(const std::string& op_name, CompositeLoweringFn fn);

  /// Returns the rule for ``op_name`` if registered, else ``nullptr``.
  const CompositeLoweringFn* Lookup(const std::string& op_name) const;

 private:
  CompositeLoweringRegistry();
  std::unordered_map<std::string, CompositeLoweringFn> rules_;
};

// Free-function registrars implemented in per-op TUs. The registry constructor
// invokes each one so rules are pulled in regardless of static-initializer
// order. Add a forward declaration here when introducing a new composite op.
void RegisterSinCosLoweringRules(CompositeLoweringRegistry& reg);

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_COMPOSITE_LOWERING_REGISTRY_H_
