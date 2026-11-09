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

#ifndef SRC_IR_TRANSFORMS_UTILS_SIMT_UTILS_H_
#define SRC_IR_TRANSFORMS_UTILS_SIMT_UTILS_H_

#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"

namespace pypto::ir::simt_detail {

/// Gate SIMT-specific storage analysis without inspecting unrelated operations.
inline bool ContainsLaunch(const StmtPtr& body) {
  class Finder : public IRVisitor {
   public:
    bool found = false;
    void VisitStmt_(const SimtLaunchStmtPtr&) override { found = true; }
  } finder;
  finder.VisitStmt(body);
  return finder.found;
}

}  // namespace pypto::ir::simt_detail
#endif  // SRC_IR_TRANSFORMS_UTILS_SIMT_UTILS_H_
