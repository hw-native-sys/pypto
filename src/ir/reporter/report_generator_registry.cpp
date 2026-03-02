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

#include "pypto/ir/reporter/report_generator_registry.h"

#include <cstdint>
#include <functional>
#include <iterator>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pypto/ir/program.h"
#include "pypto/ir/reporter/report.h"

namespace pypto {
namespace ir {

ReportGeneratorRegistry& ReportGeneratorRegistry::GetInstance() {
  static ReportGeneratorRegistry instance;
  return instance;
}

ReportGeneratorRegistry::ReportGeneratorRegistry() {
  Register(ReportType::Memory, CreateMemoryReportGenerator);
}

void ReportGeneratorRegistry::Register(ReportType type, std::function<ReportGeneratorPtr()> factory) {
  if (!factory) {
    throw std::invalid_argument("ReportGeneratorRegistry::Register received empty factory");
  }
  factories_[static_cast<uint32_t>(type)] = std::move(factory);
}

ReportGeneratorPtr ReportGeneratorRegistry::GetGenerator(ReportType type) const {
  auto it = factories_.find(static_cast<uint32_t>(type));
  if (it == factories_.end()) {
    return nullptr;
  }
  return it->second();
}

bool ReportGeneratorRegistry::HasGenerator(ReportType type) const {
  return factories_.count(static_cast<uint32_t>(type)) > 0;
}

std::vector<ReportPtr> ReportGeneratorRegistry::GenerateReports(const std::set<ReportType>& types,
                                                                const Pass& pass,
                                                                const ProgramPtr& program) const {
  std::vector<ReportPtr> all_reports;
  if (!program) return all_reports;

  for (auto type : types) {
    auto generator = GetGenerator(type);
    if (generator) {
      auto reports = generator->Generate(pass, program);
      all_reports.insert(all_reports.end(), std::make_move_iterator(reports.begin()),
                         std::make_move_iterator(reports.end()));
    }
  }
  return all_reports;
}

}  // namespace ir
}  // namespace pypto
