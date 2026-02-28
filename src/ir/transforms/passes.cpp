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

#include "pypto/ir/transforms/passes.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/ir_property.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/verifier/property_verifier_registry.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

// Pass class implementation using pimpl pattern

Pass::Pass() : impl_(nullptr) {}

Pass::Pass(std::shared_ptr<PassImpl> impl) : impl_(std::move(impl)) {}

Pass::~Pass() = default;

Pass::Pass(const Pass& other) = default;
Pass& Pass::operator=(const Pass& other) = default;
Pass::Pass(Pass&& other) noexcept = default;
Pass& Pass::operator=(Pass&& other) noexcept = default;

ProgramPtr Pass::operator()(const ProgramPtr& program) const {
  INTERNAL_CHECK(impl_) << "Pass has null implementation";
  INTERNAL_CHECK(program) << "Pass cannot run on null program";

  auto* ctx = PassContext::Current();
  if (ctx) {
    ctx->RunBeforePass(*this, program);
  }

  ProgramPtr result = (*impl_)(program);
  INTERNAL_CHECK(result) << "Pass '" << GetName() << "' returned null program";

  if (ctx) {
    ctx->RunAfterPass(*this, result);
  }

  return result;
}

ProgramPtr Pass::run(const ProgramPtr& program) const { return (*this)(program); }

std::string Pass::GetName() const {
  if (!impl_) {
    return "NullPass";
  }
  return impl_->GetName();
}

IRPropertySet Pass::GetRequiredProperties() const {
  if (!impl_) {
    return {};
  }
  return impl_->GetRequiredProperties();
}

IRPropertySet Pass::GetProducedProperties() const {
  if (!impl_) {
    return {};
  }
  return impl_->GetProducedProperties();
}

IRPropertySet Pass::GetInvalidatedProperties() const {
  if (!impl_) {
    return {};
  }
  return impl_->GetInvalidatedProperties();
}

// Utility pass implementations

namespace {

/**
 * @brief Pass implementation that wraps a program transform function
 */
class ProgramPassImpl : public PassImpl {
 public:
  ProgramPassImpl(std::function<ProgramPtr(const ProgramPtr&)> transform, std::string name,
                  PassProperties properties)
      : transform_(std::move(transform)), name_(std::move(name)), properties_(properties) {}

  ProgramPtr operator()(const ProgramPtr& program) override {
    INTERNAL_CHECK(program) << "ProgramPass cannot run on null program";
    return transform_(program);
  }

  [[nodiscard]] std::string GetName() const override { return name_.empty() ? "ProgramPass" : name_; }
  [[nodiscard]] IRPropertySet GetRequiredProperties() const override { return properties_.required; }
  [[nodiscard]] IRPropertySet GetProducedProperties() const override { return properties_.produced; }
  [[nodiscard]] IRPropertySet GetInvalidatedProperties() const override { return properties_.invalidated; }

 private:
  std::function<ProgramPtr(const ProgramPtr&)> transform_;
  std::string name_;
  PassProperties properties_;
};

/**
 * @brief Pass implementation that applies a function transform to each function in program
 */
class FunctionPassImpl : public PassImpl {
 public:
  FunctionPassImpl(std::function<FunctionPtr(const FunctionPtr&)> transform, std::string name,
                   PassProperties properties)
      : transform_(std::move(transform)), name_(std::move(name)), properties_(properties) {}

  ProgramPtr operator()(const ProgramPtr& program) override {
    INTERNAL_CHECK(program) << "FunctionPass cannot run on null program";

    // Apply the function transform to each function in the program
    std::vector<FunctionPtr> transformed_functions;
    transformed_functions.reserve(program->functions_.size());

    for (const auto& [global_var, func] : program->functions_) {
      FunctionPtr transformed_func = transform_(func);
      transformed_functions.push_back(transformed_func);
    }

    // Create a new program with the transformed functions
    return std::make_shared<const Program>(transformed_functions, program->name_, program->span_);
  }

  [[nodiscard]] std::string GetName() const override { return name_.empty() ? "FunctionPass" : name_; }
  [[nodiscard]] IRPropertySet GetRequiredProperties() const override { return properties_.required; }
  [[nodiscard]] IRPropertySet GetProducedProperties() const override { return properties_.produced; }
  [[nodiscard]] IRPropertySet GetInvalidatedProperties() const override { return properties_.invalidated; }

 private:
  std::function<FunctionPtr(const FunctionPtr&)> transform_;
  std::string name_;
  PassProperties properties_;
};

}  // namespace

// Factory functions for utility passes
namespace pass {

Pass CreateProgramPass(std::function<ProgramPtr(const ProgramPtr&)> transform, const std::string& name,
                       const PassProperties& properties) {
  return Pass(std::make_shared<ProgramPassImpl>(std::move(transform), name, properties));
}

Pass CreateFunctionPass(std::function<FunctionPtr(const FunctionPtr&)> transform, const std::string& name,
                        const PassProperties& properties) {
  return Pass(std::make_shared<FunctionPassImpl>(std::move(transform), name, properties));
}

Pass RunVerifier(const std::vector<std::string>& disabled_rules) {
  auto disabled_rules_snapshot = std::make_shared<const std::vector<std::string>>(disabled_rules);
  return CreateProgramPass(
      [disabled_rules_snapshot](const ProgramPtr& program) -> ProgramPtr {
        // Create default verifier with all rules
        IRVerifier verifier = IRVerifier::CreateDefault();

        // Disable requested rules
        for (const auto& rule_name : *disabled_rules_snapshot) {
          verifier.DisableRule(rule_name);
        }

        // Run verification and collect diagnostics
        auto diagnostics = verifier.Verify(program);

        // Log diagnostics
        if (!diagnostics.empty()) {
          std::string report = IRVerifier::GenerateReport(diagnostics);
          LOG_INFO << "IR Verification Report:\n" << report;
        }

        // Return the same program (verification doesn't modify IR)
        return program;
      },
      "IRVerifier");
}

void VerifyProperties(const IRPropertySet& properties, const ProgramPtr& program,
                      const std::string& pass_name) {
  auto& registry = PropertyVerifierRegistry::GetInstance();
  auto diagnostics = registry.VerifyProperties(properties, program);

  bool has_errors = std::any_of(diagnostics.begin(), diagnostics.end(),
                                [](const Diagnostic& d) { return d.severity == DiagnosticSeverity::Error; });

  if (has_errors) {
    std::string report = "Verification failed after '" + pass_name + "' for properties " +
                         properties.ToString() + ":\n" + IRVerifier::GenerateReport(diagnostics);
    throw VerificationError(report, std::move(diagnostics));
  }
}

}  // namespace pass

// PassPipeline implementation

PassPipeline::PassPipeline() = default;

void PassPipeline::AddPass(Pass pass) { passes_.push_back(std::move(pass)); }

ProgramPtr PassPipeline::Run(const ProgramPtr& program) const {
  CHECK(program) << "PassPipeline cannot run on null program";

  ProgramPtr current = program;

  // Automatic verification: verify lightweight properties exactly once each.
  // Level comes from active PassContext, or env-var default if no context.
  auto* ctx = PassContext::Current();
  VerificationLevel level = ctx ? ctx->GetVerificationLevel() : GetDefaultVerificationLevel();
  const bool should_verify = level != VerificationLevel::None;
  IRPropertySet verified;

  for (const auto& p : passes_) {
    current = p(current);

    if (should_verify) {
      // Remove invalidated properties so they get re-verified if re-produced
      auto invalidated = p.GetInvalidatedProperties().Intersection(GetVerifiedProperties());
      if (!invalidated.Empty()) {
        verified = verified.Difference(invalidated);
      }
      auto to_verify = p.GetProducedProperties().Intersection(GetVerifiedProperties()).Difference(verified);
      if (!to_verify.Empty()) {
        pass::VerifyProperties(to_verify, current, p.GetName());
        verified = verified.Union(to_verify);
      }
    }
  }
  return current;
}

std::vector<std::string> PassPipeline::GetPassNames() const {
  std::vector<std::string> names;
  names.reserve(passes_.size());
  for (const auto& p : passes_) {
    names.push_back(p.GetName());
  }
  return names;
}

}  // namespace ir
}  // namespace pypto
