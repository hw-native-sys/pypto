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

#include <algorithm>
#include <fstream>
#include <ios>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/program.h"
#include "pypto/ir/reporter/report.h"
#include "pypto/ir/reporter/report_generator_registry.h"
#include "pypto/ir/transforms/ir_property.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/verifier/diagnostic_check_registry.h"
#include "pypto/ir/verifier/property_verifier_registry.h"

namespace pypto {
namespace ir {

// Thread-local current context (top of stack)
thread_local PassContext* PassContext::current_ = nullptr;

// VerificationInstrument

VerificationInstrument::VerificationInstrument(VerificationMode mode) : mode_(mode) {}

namespace {

/**
 * @brief Verify properties and throw ValueError on errors (used by VerificationInstrument)
 */
void VerifyOrThrowWithContext(const IRPropertySet& properties, const ProgramPtr& program,
                              const std::string& context_msg) {
  if (properties.Empty()) {
    return;
  }

  auto& registry = PropertyVerifierRegistry::GetInstance();
  auto diagnostics = registry.VerifyProperties(properties, program);

  bool has_errors = std::any_of(diagnostics.begin(), diagnostics.end(),
                                [](const Diagnostic& d) { return d.severity == DiagnosticSeverity::Error; });
  if (has_errors) {
    std::string report = PropertyVerifierRegistry::GenerateReport(diagnostics);
    throw pypto::ValueError(context_msg + ":\n" + report);
  }
}

}  // namespace

void VerificationInstrument::RunBeforePass(const Pass& pass, const ProgramPtr& program) {
  if (mode_ != VerificationMode::Before && mode_ != VerificationMode::BeforeAndAfter) {
    return;
  }
  VerifyOrThrowWithContext(pass.GetRequiredProperties().Union(GetStructuralProperties()), program,
                           "Pre-verification failed before pass '" + pass.GetName() + "'");
}

void VerificationInstrument::RunAfterPass(const Pass& pass, const ProgramPtr& program) {
  if (mode_ != VerificationMode::After && mode_ != VerificationMode::BeforeAndAfter) {
    return;
  }
  VerifyOrThrowWithContext(pass.GetProducedProperties().Union(GetStructuralProperties()), program,
                           "Post-verification failed after pass '" + pass.GetName() + "'");
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

// Diagnostic emission helpers ------------------------------------------------

namespace {

/// Format one diagnostic as a single line of text. Used both for stderr and
/// the perf_hints.log file so the two views stay consistent.
std::string FormatDiagnosticLine(const Diagnostic& d, const std::string& phase_label) {
  std::ostringstream out;
  switch (d.severity) {
    case DiagnosticSeverity::Warning:
      out << "[warning] [" << d.rule_name << "]";
      if (!phase_label.empty()) out << " (" << phase_label << ")";
      out << " " << d.message;
      if (d.span.is_valid()) out << " at " << d.span.to_string();
      break;
    case DiagnosticSeverity::PerfHint:
      out << "[perf_hint";
      if (!d.hint_code.empty()) out << " " << d.hint_code;
      out << "] " << d.rule_name << ": " << d.message;
      if (d.span.is_valid()) out << " at " << d.span.to_string();
      break;
    case DiagnosticSeverity::Error:
      // EmitDiagnostics never receives Error severity; guarded below.
      out << "[error] [" << d.rule_name << "] " << d.message;
      break;
  }
  return out.str();
}

/// Find the output_dir of the first ReportInstrument in the active context,
/// or the empty string if there is no context or no ReportInstrument.
std::string FindReportOutputDir() {
  const auto* ctx = PassContext::Current();
  if (ctx == nullptr) return {};
  for (const auto& inst : ctx->GetInstruments()) {
    if (auto* r = dynamic_cast<ReportInstrument*>(inst.get())) {
      return r->GetOutputDir();
    }
  }
  return {};
}

}  // namespace

void EmitDiagnostics(const std::vector<Diagnostic>& diags, const std::string& phase_label) {
  if (diags.empty()) return;

  // 1. stderr — every diagnostic, gated by LogLevel.
  for (const auto& d : diags) {
    INTERNAL_CHECK(d.severity != DiagnosticSeverity::Error)
        << "Error severity must not flow through DiagnosticInstrument: " << d.rule_name;
    const std::string line = FormatDiagnosticLine(d, phase_label);
    if (d.severity == DiagnosticSeverity::Warning) {
      LOG_WARN << line;
    } else {
      LOG_INFO << line;
    }
  }

  // 2. File — only PerfHint, only when a ReportInstrument is registered.
  const std::string dir = FindReportOutputDir();
  if (dir.empty()) return;

  bool has_perf_hint = false;
  for (const auto& d : diags) {
    if (d.severity == DiagnosticSeverity::PerfHint) {
      has_perf_hint = true;
      break;
    }
  }
  if (!has_perf_hint) return;

  // PassContext is thread-local, but multiple threads can run concurrent
  // pipelines whose distinct ReportInstruments happen to share an output
  // directory. Serialise file appends so per-line writes don't interleave.
  static std::mutex perf_hints_log_mu;
  std::scoped_lock lock(perf_hints_log_mu);

  const std::string path = dir + "/perf_hints.log";
  std::ofstream f(path, std::ios::app);
  if (!f.is_open()) {
    LOG_WARN << "Failed to open " << path << " for perf-hint append";
    return;
  }
  for (const auto& d : diags) {
    if (d.severity == DiagnosticSeverity::PerfHint) {
      f << FormatDiagnosticLine(d, phase_label) << "\n";
    }
  }
}

// DiagnosticInstrument

DiagnosticInstrument::DiagnosticInstrument(DiagnosticCheckSet checks)
    : checks_(checks), pre_pipeline_done_(false) {}

namespace {

/// Whether the active context disables the diagnostic channel. Honoring this
/// from the instrument (in addition to PassPipeline) means
/// `diagnostic_phase=NONE` reliably silences output regardless of which
/// driver runs the passes.
bool DiagnosticsDisabledByContext() {
  const auto* ctx = PassContext::Current();
  return ctx != nullptr && ctx->GetDiagnosticPhase() == DiagnosticPhase::None;
}

}  // namespace

void DiagnosticInstrument::RunBeforePass(const Pass& /*pass*/, const ProgramPtr& program) {
  if (pre_pipeline_done_) return;
  pre_pipeline_done_ = true;
  if (DiagnosticsDisabledByContext()) return;
  auto diags =
      DiagnosticCheckRegistry::GetInstance().RunChecks(checks_, DiagnosticPhase::PrePipeline, program);
  EmitDiagnostics(diags, "pipeline_input");
}

void DiagnosticInstrument::RunAfterPass(const Pass& pass, const ProgramPtr& program) {
  if (DiagnosticsDisabledByContext()) return;
  auto diags = DiagnosticCheckRegistry::GetInstance().RunChecks(checks_, DiagnosticPhase::PostPass, program);
  EmitDiagnostics(diags, pass.GetName());
}

void DiagnosticInstrument::RunAfterPipeline(const ProgramPtr& program) {
  if (DiagnosticsDisabledByContext()) return;
  auto diags =
      DiagnosticCheckRegistry::GetInstance().RunChecks(checks_, DiagnosticPhase::PostPipeline, program);
  EmitDiagnostics(diags, "pipeline_output");
}

std::string DiagnosticInstrument::GetName() const { return "DiagnosticInstrument"; }

// PassContext

PassContext::PassContext(std::vector<PassInstrumentPtr> instruments, VerificationLevel verification_level,
                         DiagnosticPhase diagnostic_phase, DiagnosticCheckSet disabled_diagnostics)
    : instruments_(std::move(instruments)),
      verification_level_(verification_level),
      diagnostic_phase_(diagnostic_phase),
      disabled_diagnostics_(disabled_diagnostics),
      previous_(nullptr) {}

VerificationLevel PassContext::GetVerificationLevel() const { return verification_level_; }

DiagnosticPhase PassContext::GetDiagnosticPhase() const { return diagnostic_phase_; }

const DiagnosticCheckSet& PassContext::GetDisabledDiagnostics() const { return disabled_diagnostics_; }

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

void PassContext::RunAfterPipeline(const ProgramPtr& program) {
  for (const auto& instrument : instruments_) {
    INTERNAL_CHECK(instrument != nullptr) << "PassContext contains a null PassInstrument";
    instrument->RunAfterPipeline(program);
  }
}

PassContext* PassContext::Current() { return current_; }

const backend::BackendHandler* PassContext::GetBackendHandler() const {
  // Handler ownership lives with the Backend itself; PassContext is just a
  // convenient access path that satisfies the "pass-context-config" rule.
  return backend::BackendConfig::GetBackend()->GetHandler();
}

}  // namespace ir
}  // namespace pypto
