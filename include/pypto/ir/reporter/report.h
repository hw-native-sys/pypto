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

#ifndef PYPTO_IR_REPORTER_REPORT_H_
#define PYPTO_IR_REPORTER_REPORT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "pypto/ir/memref.h"
#include "pypto/ir/program.h"

namespace pypto {
namespace ir {

// Forward declare Pass to avoid circular include
class Pass;

/**
 * @brief Report type identifier (analogous to IRProperty in the Verification system)
 */
enum class ReportType {
  Memory,  ///< Memory usage per MemorySpace
  // Future: Timing, DependencyGraph, ...
};

/**
 * @brief Abstract base class for report data (analogous to Diagnostic)
 *
 * Defines a unified output interface for all report types.
 * Subclass this to implement new report formats.
 */
class Report {
 public:
  virtual ~Report() = default;

  /**
   * @brief Get the report title (used for filenames and headers)
   */
  [[nodiscard]] virtual std::string GetTitle() const = 0;

  /**
   * @brief Format the report as human-readable text
   */
  [[nodiscard]] virtual std::string Format() const = 0;
};

using ReportPtr = std::unique_ptr<Report>;

/**
 * @brief Abstract base class for report generators (analogous to PropertyVerifier)
 *
 * Each generator implements logic to collect data from IR and produce Report objects.
 * To create a new report generator:
 * 1. Inherit from ReportGenerator
 * 2. Implement GetName() to return a unique name
 * 3. Implement Generate() to produce reports from the IR
 */
class ReportGenerator {
 public:
  virtual ~ReportGenerator() = default;

  /**
   * @brief Get the name of this generator
   */
  [[nodiscard]] virtual std::string GetName() const = 0;

  /**
   * @brief Generate reports from the given pass and program
   * @param pass The pass that just executed
   * @param program The program after the pass
   * @return Vector of reports (may contain one per function)
   */
  virtual std::vector<ReportPtr> Generate(const Pass& pass, const ProgramPtr& program) = 0;
};

using ReportGeneratorPtr = std::shared_ptr<ReportGenerator>;

/**
 * @brief Memory usage report for all InCore functions in a program.
 *
 * Collects per-MemorySpace usage statistics for each InCore function
 * and compares with platform limits.
 */
class MemoryReport : public Report {
 public:
  struct MemorySpaceUsage {
    MemorySpace space;
    uint64_t used;   ///< High-water mark in bytes
    uint64_t limit;  ///< Platform limit in bytes (0 = unknown)
    uint32_t count;  ///< Number of MemRef allocations
  };

  struct FunctionMemoryUsage {
    std::string function_name;
    std::vector<MemorySpaceUsage> entries;
  };

  MemoryReport(std::string pass_name, std::string backend_name, std::vector<FunctionMemoryUsage> functions);

  [[nodiscard]] std::string GetTitle() const override;
  [[nodiscard]] std::string Format() const override;

 private:
  std::string pass_name_;
  std::string backend_name_;
  std::vector<FunctionMemoryUsage> functions_;
};

/**
 * @brief Factory function for creating MemoryReportGenerator
 * (analogous to CreateAllocatedMemoryAddrPropertyVerifier)
 */
ReportGeneratorPtr CreateMemoryReportGenerator();

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_REPORTER_REPORT_H_
