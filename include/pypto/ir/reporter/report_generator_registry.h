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

#ifndef PYPTO_IR_REPORTER_REPORT_GENERATOR_REGISTRY_H_
#define PYPTO_IR_REPORTER_REPORT_GENERATOR_REGISTRY_H_

#include <cstdint>
#include <functional>
#include <set>
#include <unordered_map>
#include <vector>

#include "pypto/ir/program.h"
#include "pypto/ir/reporter/report.h"

namespace pypto {
namespace ir {

/**
 * @brief Registry mapping ReportType values to their ReportGenerator factories
 * (analogous to PropertyVerifierRegistry)
 *
 * The registry is a singleton that holds factory functions for creating generators
 * for each report type. Built-in generators are registered in the constructor.
 */
class ReportGeneratorRegistry {
 public:
  /**
   * @brief Get the singleton registry instance
   */
  static ReportGeneratorRegistry& GetInstance();

  /**
   * @brief Register a generator factory for a report type
   * @param type The report type this generator produces
   * @param factory Function that creates a new ReportGenerator instance
   */
  void Register(ReportType type, std::function<ReportGeneratorPtr()> factory);

  /**
   * @brief Get a generator for a report type
   * @param type The report type to get a generator for
   * @return New ReportGenerator instance, or nullptr if none registered
   */
  [[nodiscard]] ReportGeneratorPtr GetGenerator(ReportType type) const;

  /**
   * @brief Check if a generator is registered for a report type
   */
  [[nodiscard]] bool HasGenerator(ReportType type) const;

  /**
   * @brief Generate reports for a set of report types
   * (analogous to PropertyVerifierRegistry::VerifyProperties)
   *
   * @param types Set of report types to generate
   * @param pass The pass that triggered report generation
   * @param program The program after the pass
   * @return All generated reports
   */
  std::vector<ReportPtr> GenerateReports(const std::set<ReportType>& types, const Pass& pass,
                                         const ProgramPtr& program) const;

 private:
  ReportGeneratorRegistry();

  std::unordered_map<uint32_t, std::function<ReportGeneratorPtr()>> factories_;
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_REPORTER_REPORT_GENERATOR_REGISTRY_H_
