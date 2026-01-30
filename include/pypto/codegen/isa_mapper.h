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

#ifndef PYPTO_CODEGEN_ISA_MAPPER_H_
#define PYPTO_CODEGEN_ISA_MAPPER_H_

#include <map>
#include <optional>
#include <string>
#include <unordered_map>

#include "pypto/ir/expr.h"

namespace pypto {
namespace codegen {

/**
 * @brief Mapping information for an IR operation to pto-isa instruction
 */
struct ISAMapping {
  std::string isa_name;  ///< The pto-isa instruction name (e.g., "TADD", "TLOAD")
};

/**
 * @brief Mapper from IR operations to pto-isa instructions
 *
 * ISAMapper provides the translation from PyPTO IR operation names
 * (e.g., "block.add", "block.load") to corresponding pto-isa instruction
 * names (e.g., "TADD", "TLOAD").
 */
class ISAMapper {
 public:
  ISAMapper();

  /**
   * @brief Get ISA mapping for an operation
   *
   * Returns the ISA instruction mapping for a given IR operation name.
   * For operations that depend on attributes (like block.sum with axis),
   * the attributes map can be provided to determine the correct instruction.
   *
   * @param op_name The IR operation name (e.g., "block.add")
   * @param attrs Optional attributes map for operations that need them
   * @return ISA mapping if found, nullopt otherwise
   */
  [[nodiscard]] std::optional<ISAMapping> GetMapping(
      const std::string& op_name, const std::map<std::string, ir::ExprPtr>& attrs = {}) const;

 private:
  /**
   * @brief Initialize the operation mapping table
   */
  void InitializeMappings();

  std::unordered_map<std::string, ISAMapping> mappings_;  ///< Operation name → ISA mapping
};

}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_ISA_MAPPER_H_
