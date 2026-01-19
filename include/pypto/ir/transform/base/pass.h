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

#ifndef PYPTO_IR_TRANSFORM_BASE_PASS_H_
#define PYPTO_IR_TRANSFORM_BASE_PASS_H_

#include <memory>
#include <vector>

#include "pypto/ir/function.h"
#include "pypto/ir/transform/base/mutator.h"

namespace pypto {
namespace ir {

/**
 * @brief Base class for IR transformation passes
 *
 * Pass is an abstract base class that extends IRMutator to provide function-level transformations.
 * Each pass operates on a Function and returns a transformed Function.
 * Passes maintain immutability - they return new FunctionPtr instances rather than modifying in place.
 */
class Pass : public IRMutator {
 public:
  ~Pass() override = default;

  /**
   * @brief Execute the pass on a function
   *
   * This is the main entry point for pass execution. Subclasses must implement this method
   * to define their transformation logic.
   *
   * @param func Input function to transform
   * @return Transformed function (may be the same pointer if no changes were made)
   */
  virtual FunctionPtr Run(const FunctionPtr& func) = 0;

 protected:
  /**
   * @brief Helper method to transform a function
   *
   * This utility method helps create a new Function with transformed components.
   * It applies copy-on-write: only creates a new Function if any component changed.
   *
   * @param func Original function
   * @param new_params Transformed parameters (pass func->params_ if unchanged)
   * @param new_return_types Transformed return types (pass func->return_types_ if unchanged)
   * @param new_body Transformed body (pass func->body_ if unchanged)
   * @return New FunctionPtr if any component changed, otherwise the original func
   */
  FunctionPtr TransformFunction(const FunctionPtr& func, const std::vector<VarPtr>& new_params,
                                const std::vector<TypePtr>& new_return_types, const StmtPtr& new_body) {
    // Copy-on-write: only create new Function if something changed
    bool params_changed = false;
    if (new_params.size() != func->params_.size()) {
      params_changed = true;
    } else {
      for (size_t i = 0; i < new_params.size(); ++i) {
        if (new_params[i].get() != func->params_[i].get()) {
          params_changed = true;
          break;
        }
      }
    }

    bool return_types_changed = false;
    if (new_return_types.size() != func->return_types_.size()) {
      return_types_changed = true;
    } else {
      for (size_t i = 0; i < new_return_types.size(); ++i) {
        if (new_return_types[i].get() != func->return_types_[i].get()) {
          return_types_changed = true;
          break;
        }
      }
    }

    bool body_changed = new_body.get() != func->body_.get();

    if (params_changed || return_types_changed || body_changed) {
      return std::make_shared<const Function>(func->name_, new_params, new_return_types, new_body, func->span_);
    } else {
      return func;
    }
  }
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORM_BASE_PASS_H_
