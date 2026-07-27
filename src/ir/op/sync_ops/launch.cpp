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
#include "pypto/ir/expr.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

TypePtr DeduceInt32ScalarType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs) {
  (void)args;
  (void)kwargs;
  return std::make_shared<ScalarType>(DataType::INT32);
}

}  // namespace

// system.available_cluster_count — this run's MIX cluster (= AIC) count.
//
// The count is a property of the device the run lands on, not of the program:
// the runtime latches it during bring-up and orchestration reads it back. A
// literal block count baked at compile time is wrong the moment a run gets a
// device with fewer usable clusters, which silently under- or over-fills an
// SPMD launch. Used as the ``core_num`` of a mixed (AIC+AIV) or cube-only
// launch — the only value that guarantees full occupancy for a hard
// ``pl.system.syncall``.
//
// Orchestration codegen lowers it to ``rt_available_cluster_count()``.
REGISTER_OP("system.available_cluster_count")
    .set_description("This run's MIX cluster (= AIC) count, resolved by the runtime from the device")
    .set_op_category("TaskOp")
    .no_argument()
    .f_deduce_type(DeduceInt32ScalarType);

// system.available_aiv_count — this run's standalone AIV core count.
//
// The AIV counterpart of ``system.available_cluster_count``: the ``core_num``
// of a vector-only SPMD launch. Note this is the *standalone* AIV count, not
// the AIVs reachable through clusters — a mixed launch sizes itself on
// ``system.available_cluster_count`` instead.
//
// Orchestration codegen lowers it to ``rt_available_aiv_count()``.
REGISTER_OP("system.available_aiv_count")
    .set_description("This run's standalone AIV core count, resolved by the runtime from the device")
    .set_op_category("TaskOp")
    .no_argument()
    .f_deduce_type(DeduceInt32ScalarType);

}  // namespace ir
}  // namespace pypto
