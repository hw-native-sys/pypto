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

#ifndef PYPTO_IR_TRANSFORMS_PASS_CONTEXT_H_
#define PYPTO_IR_TRANSFORMS_PASS_CONTEXT_H_

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/program.h"
#include "pypto/ir/reporter/report.h"
#include "pypto/ir/transforms/ir_property.h"
#include "pypto/ir/verifier/diagnostic_check_registry.h"

namespace pypto {

namespace backend {
class BackendHandler;
}  // namespace backend

namespace ir {

// Forward declare Pass to avoid circular include (pass_context.h <-> passes.h)
class Pass;

/**
 * @brief Emit a batch of diagnostics from the unified diagnostic channel.
 *
 * Warning severity always prints in full via LOG_WARN. PerfHint severity is
 * appended to `${ReportInstrument.output_dir}/perf_hints.log` when a
 * `ReportInstrument` is in the active context; in that case the console gets a
 * single LOG_INFO summary line (`[perf_hint] N hints across M sites; see
 * <path>`) instead of one line per hint, so build logs stay readable. With no
 * `ReportInstrument` there is no file, so each PerfHint is printed in full via
 * LOG_INFO. Error severity is rejected by INTERNAL_CHECK — errors should be
 * thrown via `VerificationError`, not emitted here.
 *
 * @param diags Diagnostics to emit. May be empty (no-op).
 * @param phase_label Human-readable label for the source phase
 *        (e.g. "pipeline_input", "pipeline_output", a pass name).
 */
void EmitDiagnostics(const std::vector<Diagnostic>& diags, const std::string& phase_label);

/**
 * @brief Controls when property verification runs
 */
enum class VerificationMode {
  None,           ///< No automatic verification
  Before,         ///< Verify required properties before each pass
  After,          ///< Verify produced properties after each pass
  BeforeAndAfter  ///< Verify both before and after each pass
};

/**
 * @brief Abstract base class for pass instrumentation
 *
 * PassInstruments are callbacks that run before/after each pass execution.
 * Subclass this to implement custom instrumentation (verification, logging, profiling, etc.).
 */
class PassInstrument {
 public:
  virtual ~PassInstrument() = default;

  /**
   * @brief Called before a pass is executed
   * @param pass The pass about to run
   * @param program The program before transformation
   */
  virtual void RunBeforePass(const Pass& pass, const ProgramPtr& program) = 0;

  /**
   * @brief Called after a pass is executed
   * @param pass The pass that just ran
   * @param program The program after transformation
   */
  virtual void RunAfterPass(const Pass& pass, const ProgramPtr& program) = 0;

  /**
   * @brief Called once after the final pass in the pipeline.
   *
   * Default no-op. Override to run end-of-pipeline analyses such as
   * performance-hint checks (issue #1180) that need a fully lowered IR.
   */
  virtual void RunAfterPipeline(const ProgramPtr& /*program*/) {}

  /**
   * @brief Get the name of this instrument
   */
  [[nodiscard]] virtual std::string GetName() const = 0;
};

using PassInstrumentPtr = std::shared_ptr<PassInstrument>;

/**
 * @brief Instrument that verifies IR properties before/after passes
 *
 * Uses PropertyVerifierRegistry to check that passes' required properties hold
 * before execution and produced properties hold after execution.
 */
class VerificationInstrument : public PassInstrument {
 public:
  explicit VerificationInstrument(VerificationMode mode);

  void RunBeforePass(const Pass& pass, const ProgramPtr& program) override;
  void RunAfterPass(const Pass& pass, const ProgramPtr& program) override;
  [[nodiscard]] std::string GetName() const override;

 private:
  VerificationMode mode_;
};

/**
 * @brief Instrument that invokes user-provided callbacks before/after each pass
 *
 * Enables lightweight, ad-hoc instrumentation (e.g., IR dumping, logging)
 * without subclassing PassInstrument. Null callbacks are silently skipped.
 */
class CallbackInstrument : public PassInstrument {
 public:
  using Callback = std::function<void(const Pass&, const ProgramPtr&)>;

  explicit CallbackInstrument(Callback before_pass = nullptr, Callback after_pass = nullptr,
                              std::string name = "CallbackInstrument");

  void RunBeforePass(const Pass& pass, const ProgramPtr& program) override;
  void RunAfterPass(const Pass& pass, const ProgramPtr& program) override;
  [[nodiscard]] std::string GetName() const override;

 private:
  Callback before_pass_;
  Callback after_pass_;
  std::string name_;
};

/**
 * @brief Instrument that generates reports to files after specified passes
 * (analogous to VerificationInstrument using PropertyVerifierRegistry)
 *
 * Uses ReportGeneratorRegistry to dispatch report generation. Enable specific
 * report types for specific passes via EnableReport().
 *
 * Usage (Python):
 * @code
 *   instrument = passes.ReportInstrument("/path/to/output")
 *   instrument.enable_report(passes.ReportType.Memory, "AllocateMemoryAddr")
 *   with passes.PassContext([instrument]):
 *       pipeline.run(program)
 * @endcode
 */
class ReportInstrument : public PassInstrument {
 public:
  explicit ReportInstrument(std::string output_dir);

  /**
   * @brief Enable a report type to be generated after a specific pass
   * @param type Report type to enable
   * @param trigger_pass Name of the pass that triggers this report
   */
  void EnableReport(ReportType type, std::string trigger_pass);

  void RunBeforePass(const Pass& pass, const ProgramPtr& program) override;
  void RunAfterPass(const Pass& pass, const ProgramPtr& program) override;
  [[nodiscard]] std::string GetName() const override;

  /**
   * @brief Path of the directory that holds report files.
   *
   * Exposed so that `DiagnosticInstrument` can append its perf-hint log
   * (`perf_hints.log`) into the same folder when this instrument is present
   * in the active context.
   */
  [[nodiscard]] const std::string& GetOutputDir() const { return output_dir_; }

 private:
  std::string output_dir_;
  std::unordered_map<std::string, std::set<ReportType>> triggers_;

  void WriteReport(const Report& report, const std::string& filename);
};

/**
 * @brief Instrument that runs registered diagnostic checks (warnings + perf hints).
 *
 * Checks declare their phase at registration; this instrument fires each
 * check at every pass boundary and the registry filters by phase. Output is
 * dispatched by `EmitDiagnostics`: Warnings print in full via `LOG_WARN`;
 * PerfHints are appended to `${ReportInstrument.output_dir}/perf_hints.log`
 * when a `ReportInstrument` is in the active context, with the console getting
 * a single `LOG_INFO` summary line in that case (and the full per-hint lines
 * otherwise).
 *
 * For advanced use outside PassPipeline or fine-grained per-instrument
 * control. PassPipeline::Run runs the registered checks directly without
 * needing this instrument; constructing one explicitly is only required when
 * driving passes outside the pipeline.
 */
class DiagnosticInstrument : public PassInstrument {
 public:
  explicit DiagnosticInstrument(DiagnosticCheckSet checks = DiagnosticCheckRegistry::GetAllChecks());

  void RunBeforePass(const Pass& pass, const ProgramPtr& program) override;
  void RunAfterPass(const Pass& pass, const ProgramPtr& program) override;
  void RunAfterPipeline(const ProgramPtr& program) override;
  [[nodiscard]] std::string GetName() const override;

 private:
  DiagnosticCheckSet checks_;
  bool pre_pipeline_done_;
};

/**
 * @brief Context that holds instruments and manages a thread-local stack
 *
 * PassContext provides a `with`-style nesting mechanism. When active, Pass::operator()
 * will run the context's instruments before/after each pass execution.
 *
 * Usage (Python):
 * @code
 *   with PassContext([VerificationInstrument(VerificationMode.AFTER)]):
 *       result = some_pass(program)  # instruments fire automatically
 * @endcode
 */
class PassContext {
 public:
  /**
   * @brief Create a context with instruments and optional verification/diagnostic settings
   * @param instruments List of pass instruments
   * @param verification_level Verification level (default: Basic)
   * @param diagnostic_phase Default phase gate for the diagnostic channel
   *        (default: PrePipeline). Setting this to None disables warnings AND
   *        performance hints; finer-grained control uses `disabled_diagnostics`.
   * @param disabled_diagnostics Diagnostic checks to skip — keyed by
   *        DiagnosticCheck enum (default: UnusedControlFlowResult).
   *        Performance hints are on by default; disable individual hints by
   *        adding their DiagnosticCheck values here.
   */
  explicit PassContext(std::vector<PassInstrumentPtr> instruments,
                       VerificationLevel verification_level = VerificationLevel::Basic,
                       DiagnosticPhase diagnostic_phase = DiagnosticPhase::PrePipeline,
                       DiagnosticCheckSet disabled_diagnostics = {DiagnosticCheck::UnusedControlFlowResult});

  /**
   * @brief Push this context onto the thread-local stack
   */
  void EnterContext();

  /**
   * @brief Pop this context from the thread-local stack
   */
  void ExitContext();

  /**
   * @brief Run all instruments' RunBeforePass
   */
  void RunBeforePass(const Pass& pass, const ProgramPtr& program);

  /**
   * @brief Run all instruments' RunAfterPass
   */
  void RunAfterPass(const Pass& pass, const ProgramPtr& program);

  /**
   * @brief Run all instruments' RunAfterPipeline (called by PassPipeline once
   *        after the final pass has finished).
   */
  void RunAfterPipeline(const ProgramPtr& program);

  /**
   * @brief Get the verification level for this context
   */
  [[nodiscard]] VerificationLevel GetVerificationLevel() const;

  /**
   * @brief Get the diagnostic phase gate for this context.
   *
   * `None` disables warnings and performance hints entirely. Other values
   * are passed through to the instrument; per-check phase still comes from
   * the registry.
   */
  [[nodiscard]] DiagnosticPhase GetDiagnosticPhase() const;

  /**
   * @brief Get the diagnostic checks suppressed by this context (keyed by
   *        `DiagnosticCheck`). Applies to both warnings and performance hints.
   */
  [[nodiscard]] const DiagnosticCheckSet& GetDisabledDiagnostics() const;

  /**
   * @brief Get the instruments registered on this context
   */
  [[nodiscard]] const std::vector<PassInstrumentPtr>& GetInstruments() const;

  /**
   * @brief Get the currently active context (top of thread-local stack)
   * @return Pointer to current context, or nullptr if none
   */
  static PassContext* Current();

  /**
   * @brief Convenience accessor for the currently configured BackendHandler.
   *
   * Equivalent to `BackendConfig::GetBackend()->GetHandler()`. Provided so
   * that passes can dispatch backend-specific behaviour through the
   * PassContext (per the `pass-context-config` rule) without taking a
   * direct dependency on the global BackendConfig from every call site.
   *
   * @return Pointer to the active backend handler (never null).
   * @throws pypto::ValueError if backend type has not been configured.
   */
  [[nodiscard]] const backend::BackendHandler* GetBackendHandler() const;

 private:
  std::vector<PassInstrumentPtr> instruments_;
  VerificationLevel verification_level_;
  DiagnosticPhase diagnostic_phase_;
  DiagnosticCheckSet disabled_diagnostics_;
  PassContext* previous_;

  static thread_local PassContext* current_;
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_PASS_CONTEXT_H_
