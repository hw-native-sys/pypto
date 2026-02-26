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

#ifndef PYPTO_IR_VERIFIER_PROPERTY_VERIFIER_REGISTRY_H_
#define PYPTO_IR_VERIFIER_PROPERTY_VERIFIER_REGISTRY_H_

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/ir_property.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

/**
 * @brief Registry mapping IRProperty values to their PropertyVerifier factories
 *
 * The registry is a singleton that holds factory functions for creating verifiers
 * for each IR property. This allows the PassPipeline to automatically verify
 * properties before/after passes.
 */
class PropertyVerifierRegistry {
 public:
  /**
   * @brief Get the singleton registry instance
   */
  static PropertyVerifierRegistry& GetInstance();

  /**
   * @brief Register a verifier factory for a property
   * @param prop The property this verifier checks
   * @param factory Function that creates a new PropertyVerifier instance
   */
  void Register(IRProperty prop, std::function<PropertyVerifierPtr()> factory);

  /**
   * @brief Get the verifier for a property
   * @param prop The property to get a verifier for
   * @return New PropertyVerifier instance, or nullptr if none registered
   */
  [[nodiscard]] PropertyVerifierPtr GetVerifier(IRProperty prop) const;

  /**
   * @brief Check if a verifier is registered for a property
   */
  [[nodiscard]] bool HasVerifier(IRProperty prop) const;

  /**
   * @brief Verify a set of properties on a program
   * @param properties Properties to verify
   * @param program Program to verify against
   * @return Diagnostics from all verifiers (empty if all pass)
   */
  [[nodiscard]] std::vector<Diagnostic> VerifyProperties(const IRPropertySet& properties,
                                                         const ProgramPtr& program) const;

 private:
  PropertyVerifierRegistry();

  std::unordered_map<uint32_t, std::function<PropertyVerifierPtr()>> factories_;
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_VERIFIER_PROPERTY_VERIFIER_REGISTRY_H_
