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

#include "pypto/ir/stmt.h"

#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/function.h"

namespace pypto {
namespace ir {

HierarchyScopeStmt::HierarchyScopeStmt(Level level, std::optional<Role> role, std::optional<SplitMode> split,
                                       std::string name_hint, StmtPtr body, Span span,
                                       std::vector<std::string> leading_comments)
    : ScopeStmt(std::move(name_hint), std::move(body), std::move(span), std::move(leading_comments)),
      level_(level),
      role_(role),
      split_(split) {
  CHECK(!split_.has_value() || level_ == Level::CORE_GROUP)
      << "HierarchyScopeStmt split is only valid at Level::CORE_GROUP";
}

}  // namespace ir
}  // namespace pypto
