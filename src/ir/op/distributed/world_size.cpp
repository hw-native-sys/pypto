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
 * @file world_size.cpp
 * @brief ``pld.world_size`` — host-only IR op returning the L3 world size.
 *
 * Returns a scalar ``INT64`` value equal to the number of devices participating
 * in the current distributed execution (``len(contexts)`` at codegen time).
 *
 * The op takes no positional args, no kwargs, and no attrs; the parser is
 * responsible for ensuring the call appears inside a host-level orchestrator
 * function body. The op itself only deduces its return type.
 */

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

TypePtr DeduceWorldSizeType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.empty()) << "pld.world_size takes no positional arguments, but got " << args.size();
  CHECK(kwargs.empty()) << "pld.world_size takes no kwargs, but got " << kwargs.size();
  return std::make_shared<ScalarType>(DataType::INT64);
}

}  // namespace

// ============================================================================
// pld.world_size — host-only scalar producing the distributed world size
// ============================================================================

REGISTER_OP("pld.world_size")
    .set_description(
        "Return the number of devices participating in the current distributed execution "
        "as a scalar INT64. Host-only: must appear inside a host-level orchestrator function "
        "body. Codegen lowers each call site to ``len(contexts)``.")
    .set_op_category("DistributedOp")
    .no_argument()
    .no_memory_spec()
    .f_deduce_type(DeduceWorldSizeType);

}  // namespace ir
}  // namespace pypto
