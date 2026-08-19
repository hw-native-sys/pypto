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

/**
 * @file testing.cpp
 * @brief Testing operations for operator registration
 *
 * This file provides test operators used exclusively for testing the operator
 * registration system. These operators should not be used in production code.
 */

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

// ============================================================================
// Test Operator Registration
// ============================================================================

REGISTER_OP("test.op")
    .set_op_category("TestOp")
    .set_description("Test operation for operator registration system")
    .add_argument("arg1", "First test argument")
    .add_argument("arg2", "Second test argument")
    .set_attr<int>("int_attr")
    .set_attr<std::string>("string_attr")
    .set_attr<bool>("bool_attr")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return args[0]->GetType();
    });

// Type deduction that fails an internal invariant. Used exclusively to verify that
// OpRegistry::CreateImpl surfaces the concrete exception type (InternalError) and the
// stack trace of the real throw site, rather than flattening both into a ValueError
// raised from the registry itself.
REGISTER_OP("test.deduce_raises_internal")
    .set_op_category("TestOp")
    .set_description("Test-only op whose type deduction fails an internal invariant")
    .add_argument("x", "Input")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) -> TypePtr {
      INTERNAL_CHECK(false) << "Internal error: test.deduce_raises_internal always fails";
      return nullptr;
    });

// Sibling of the above for the user-error half of the same contract: a TypeError raised
// during deduction must reach the caller as a TypeError, not a ValueError.
REGISTER_OP("test.deduce_raises_type")
    .set_op_category("TestOp")
    .set_description("Test-only op whose type deduction raises TypeError")
    .add_argument("x", "Input")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) -> TypePtr {
      throw TypeError("test.deduce_raises_type always fails");
    });

// Used exclusively to test the "missing conversion" error path in ConvertTensorToTileOps.
REGISTER_OP("test.tensor_op_no_conv")
    .set_op_category("TensorOp")
    .set_description("Test-only op: TensorOp with no registered tile conversion")
    .add_argument("x", "Input tensor")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return args[0]->GetType();
    });

}  // namespace ir
}  // namespace pypto
