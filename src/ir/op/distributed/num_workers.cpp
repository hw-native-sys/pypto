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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace {

TypePtr DeduceNumWorkersType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.empty()) << "pld.system.num_workers takes no positional arguments, but got " << args.size();
  CHECK(kwargs.empty()) << "pld.system.num_workers takes no kwargs, but got " << kwargs.size();
  return std::make_shared<ScalarType>(DataType::INT64);
}

}  // namespace

REGISTER_OP("pld.system.num_workers")
    .set_description(
        "Return the immediate-child count as scalar INT64: local devices for a HOST orchestrator, "
        "nodes for a CLUSTER_0 orchestrator. Orchestration-only, outside InCore/SPMD scopes. "
        "This is not the size of a global communication domain.")
    .set_op_category("DistributedOp")
    .no_argument()
    .no_memory_spec()
    .f_deduce_type(DeduceNumWorkersType);

}  // namespace ir
}  // namespace pypto
