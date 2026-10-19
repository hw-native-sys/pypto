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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_NESTED_PIPELINE_H_
#define PYPTO_IR_TRANSFORMS_UTILS_NESTED_PIPELINE_H_

#include <cstdint>
#include <map>
#include <set>
#include <string>

#include "pypto/ir/expr.h"
#include "pypto/ir/stmt.h"

namespace pypto::ir {

/// Storage identity and logical coordinate of a specialized child iteration.
/// Scheduling uses stride * enclosing_logical_iteration + phase. Storage uses
/// a separate mixed-radix coordinate: storage_stride is the product of child
/// stage counts, and storage_phase contains their iteration residues. The root
/// stage multiplies storage_stride to give the complete region's slot count.
struct NestedPipelineSlot {
  const Var* family = nullptr;
  int64_t slots = 0;
  int64_t stride = 1;
  int64_t phase = 0;
  const Var* anchor = nullptr;
  int64_t storage_stride = 1;
  int64_t storage_phase = 0;
};

struct NestedPipelinePlan {
  ForStmtPtr loop;
  std::map<const Var*, NestedPipelineSlot> storage;
  std::set<const Var*> anchors;
  bool nested = false;
  int64_t max_stride = 1;
};

/// Specialize bounded local child scopes without allocating or moving effects.
/// Scheduling and alias proofs run on the result before it is committed.
/// Unsupported nests return false and leave the original IR untouched.
bool PrepareNestedPipeline(const ForStmtPtr& loop, NestedPipelinePlan& plan, std::string& reason,
                           bool allow_root_transfer = false);

}  // namespace pypto::ir
#endif  // PYPTO_IR_TRANSFORMS_UTILS_NESTED_PIPELINE_H_
