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

#include "pypto/ir/transforms/pass_context.h"

#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/program.h"
#include "pypto/ir/reporter/report.h"
#include "pypto/ir/reporter/report_generator_registry.h"
#include "pypto/ir/transforms/ir_property.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/verifier/property_verifier_registry.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

// Thread-local current context (top of stack)
thread_local PassContext* PassContext::current_ = nullptr;

// VerificationInstrument

VerificationInstrument::VerificationInstrument(VerificationMode mode) : mode_(mode) {}

void VerificationInstrument::RunBeforePass(const Pass& pass, const ProgramPtr& program) {
  if (mode_ != VerificationMode::Before && mode_ != VerificationMode::BeforeAndAfter) {
    return;
  }

  auto required = pass.GetRequiredProperties();
  if (required.Empty()) {
    return;
  }

  auto& registry = PropertyVerifierRegistry::GetInstance();
  auto diagnostics = registry.VerifyProperties(required, program);
  if (diagnostics.empty()) {
    return;
  }

  bool has_errors = false;
  for (const auto& d : diagnostics) {
    if (d.severity == DiagnosticSeverity::Error) {
      has_errors = true;
      break;
    }
  }
  if (has_errors) {
    std::string report = IRVerifier::GenerateReport(diagnostics);
    throw pypto::ValueError("Pre-verification failed before pass '" + pass.GetName() + "':\n" + report);
  }
}

void VerificationInstrument::RunAfterPass(const Pass& pass, const ProgramPtr& program) {
  if (mode_ != VerificationMode::After && mode_ != VerificationMode::BeforeAndAfter) {
    return;
  }

  auto produced = pass.GetProducedProperties();
  if (produced.Empty()) {
    return;
  }

  auto& registry = PropertyVerifierRegistry::GetInstance();
  auto diagnostics = registry.VerifyProperties(produced, program);
  if (diagnostics.empty()) {
    return;
  }

  bool has_errors = false;
  for (const auto& d : diagnostics) {
    if (d.severity == DiagnosticSeverity::Error) {
      has_errors = true;
      break;
    }
  }
  if (has_errors) {
    std::string report = IRVerifier::GenerateReport(diagnostics);
    throw pypto::ValueError("Post-verification failed after pass '" + pass.GetName() + "':\n" + report);
  }
}

std::string VerificationInstrument::GetName() const { return "VerificationInstrument"; }

// CallbackInstrument

CallbackInstrument::CallbackInstrument(Callback before_pass, Callback after_pass, std::string name)
    : before_pass_(std::move(before_pass)), after_pass_(std::move(after_pass)), name_(std::move(name)) {}

void CallbackInstrument::RunBeforePass(const Pass& pass, const ProgramPtr& program) {
  if (before_pass_) before_pass_(pass, program);
}

void CallbackInstrument::RunAfterPass(const Pass& pass, const ProgramPtr& program) {
  if (after_pass_) after_pass_(pass, program);
}

std::string CallbackInstrument::GetName() const { return name_; }

// ReportInstrument

ReportInstrument::ReportInstrument(std::string output_dir) : output_dir_(std::move(output_dir)) {}

void ReportInstrument::EnableReport(ReportType type, std::string trigger_pass) {
  triggers_[std::move(trigger_pass)].insert(type);
}

void ReportInstrument::RunBeforePass(const Pass& /*pass*/, const ProgramPtr& /*program*/) {}

void ReportInstrument::RunAfterPass(const Pass& pass, const ProgramPtr& program) {
  auto it = triggers_.find(pass.GetName());
  if (it == triggers_.end()) return;

  auto& registry = ReportGeneratorRegistry::GetInstance();
  auto reports = registry.GenerateReports(it->second, pass, program);

  for (const auto& report : reports) {
    std::string filename = report->GetTitle() + "_after_" + pass.GetName() + ".txt";
    WriteReport(*report, filename);
  }
}

std::string ReportInstrument::GetName() const { return "ReportInstrument"; }

void ReportInstrument::WriteReport(const Report& report, const std::string& filename) {
  std::string filepath = output_dir_ + "/" + filename;
  std::ofstream file(filepath);
  if (!file.is_open()) {
    LOG_ERROR << "Failed to open report file: " << filepath;
    return;
  }
  file << report.Format();
  if (file.fail()) {
    LOG_ERROR << "Failed to write report file: " << filepath;
  }
}

// PassContext

PassContext::PassContext(std::vector<PassInstrumentPtr> instruments, VerificationLevel verification_level)
    : instruments_(std::move(instruments)), verification_level_(verification_level), previous_(nullptr) {}

VerificationLevel PassContext::GetVerificationLevel() const { return verification_level_; }

const std::vector<PassInstrumentPtr>& PassContext::GetInstruments() const { return instruments_; }

void PassContext::EnterContext() {
  previous_ = current_;
  current_ = this;
}

void PassContext::ExitContext() {
  INTERNAL_CHECK(current_ == this)
      << "PassContext::ExitContext called out of order or without a matching EnterContext";
  current_ = previous_;
  previous_ = nullptr;
}

void PassContext::RunBeforePass(const Pass& pass, const ProgramPtr& program) {
  for (const auto& instrument : instruments_) {
    INTERNAL_CHECK(instrument != nullptr) << "PassContext contains a null PassInstrument";
    instrument->RunBeforePass(pass, program);
  }
}

void PassContext::RunAfterPass(const Pass& pass, const ProgramPtr& program) {
  for (const auto& instrument : instruments_) {
    INTERNAL_CHECK(instrument != nullptr) << "PassContext contains a null PassInstrument";
    instrument->RunAfterPass(pass, program);
  }
}

PassContext* PassContext::Current() { return current_; }

}  // namespace ir
}  // namespace pypto
