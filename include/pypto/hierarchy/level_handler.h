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

#ifndef PYPTO_HIERARCHY_LEVEL_HANDLER_H_
#define PYPTO_HIERARCHY_LEVEL_HANDLER_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "pypto/ir/function.h"

namespace pypto {
namespace hierarchy {

/// Inputs to a complete ordinary host-tensor append statement. The generator
/// owns argument grouping and supplies the tensor-pool ordinal, not the IR
/// parameter index (scalar parameters do not occupy tensor-pool entries).
/// worker_expr is the local Worker expression for HOST and the logical child
/// node index for CLUSTER_0; the latter selects the node's routing binding.
struct HostArgSpec {
  std::string task_args;
  std::string tensor_expr;
  std::string tag;
  std::string worker_expr;
  std::string callee_name;
  size_t tensor_index;
};

/// Internal, stateless formatting policy. Allocation preparation, ownership,
/// dependency tracking and IR traversal remain with the code generator.
class LevelHandler {
 public:
  virtual ~LevelHandler() = default;
  [[nodiscard]] virtual std::string GetDomainAllocator() const = 0;
  /// An empty child list denotes the full runtime world, not an empty domain.
  [[nodiscard]] virtual std::string FormatDomainMembership(const std::vector<int64_t>& children) const = 0;
  /// Format a locally owned domain context; explicit ctx Vars bypass this path.
  [[nodiscard]] virtual std::string FormatCommCtxValue(const std::string& table,
                                                       const std::string& child) const = 0;
  [[nodiscard]] virtual std::string FormatHostTensorAppend(const HostArgSpec& spec) const = 0;
  [[nodiscard]] virtual bool MaterializesWindowTensor() const = 0;
  /// The generator retains the existing provider-acceptance classification.
  [[nodiscard]] virtual std::vector<std::string> GetEntryKeywordParams(
      bool accepts_domain_provider) const = 0;
  [[nodiscard]] virtual std::vector<std::string> GetImports() const = 0;
};

/// Return the process-lifetime HOST/CLUSTER_0 policy, or nullptr for an unsupported level.
/// Accepts ir::Level enum values, not Linqu level numbers. Having a handler does
/// not establish collective legality or enable dispatch at another level.
[[nodiscard]] const LevelHandler* GetLevelHandler(ir::Level level);

}  // namespace hierarchy
}  // namespace pypto

#endif  // PYPTO_HIERARCHY_LEVEL_HANDLER_H_
