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

#include "pypto/ir/reporter/report.h"

#include <cstdint>
#include <iomanip>
#include <ios>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/memref.h"

namespace pypto {
namespace ir {

namespace {

std::string FormatBytes(uint64_t bytes) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(1);
  if (bytes >= static_cast<uint64_t>(1024) * 1024) {
    os << static_cast<double>(bytes) / (1024.0 * 1024.0) << " MB";
  } else if (bytes >= 1024) {
    os << static_cast<double>(bytes) / 1024.0 << " KB";
  } else {
    os << bytes << " B";
  }
  return os.str();
}

std::string FormatPercent(uint64_t used, uint64_t limit) {
  if (limit == 0) return "N/A";
  std::ostringstream os;
  os << std::fixed << std::setprecision(1) << static_cast<double>(used) / static_cast<double>(limit) * 100.0
     << "%";
  return os.str();
}

}  // namespace

MemoryReport::MemoryReport(std::string pass_name, std::string backend_name,
                           std::vector<FunctionMemoryUsage> functions)
    : pass_name_(std::move(pass_name)),
      backend_name_(std::move(backend_name)),
      functions_(std::move(functions)) {}

std::string MemoryReport::GetTitle() const { return "memory"; }

std::string MemoryReport::Format() const {
  std::ostringstream os;
  os << "=== Memory Usage Report ===\n";
  os << "Pass: " << pass_name_ << "\n";
  os << "Backend: " << backend_name_ << "\n";
  os << "Functions: " << functions_.size() << " InCore\n";

  for (const auto& func : functions_) {
    os << "\n--- " << func.function_name << " ---\n\n";

    os << "  " << std::left << std::setw(7) << "Space"
       << "|  " << std::setw(11) << "Used"
       << "|  " << std::setw(11) << "Limit"
       << "|  " << std::setw(8) << "Usage"
       << "|  " << "MemRefs" << "\n";
    os << "  -------+-------------+-------------+----------+---------\n";

    for (const auto& entry : func.entries) {
      std::string space_name = MemorySpaceToString(entry.space);
      std::string used_str = FormatBytes(entry.used);
      std::string limit_str = entry.limit > 0 ? FormatBytes(entry.limit) : "N/A";
      std::string usage_str = FormatPercent(entry.used, entry.limit);

      os << "  " << std::left << std::setw(7) << space_name << "|  " << std::right << std::setw(9) << used_str
         << "  "
         << "|  " << std::setw(9) << limit_str << "  "
         << "|  " << std::setw(6) << usage_str << "  "
         << "|  " << entry.count << "\n";
    }
  }

  return os.str();
}

}  // namespace ir
}  // namespace pypto
