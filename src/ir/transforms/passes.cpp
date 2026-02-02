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

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/verifier.h"

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
  return (*impl_)(program);
}

ProgramPtr Pass::run(const ProgramPtr& program) const { return (*this)(program); }

// Utility pass implementations

namespace {

/**
 * @brief Pass implementation that wraps a program transform function
 */
class ProgramPassImpl : public PassImpl {
 public:
  ProgramPassImpl(std::function<ProgramPtr(const ProgramPtr&)> transform, std::string name)
      : transform_(std::move(transform)), name_(std::move(name)) {}

  ProgramPtr operator()(const ProgramPtr& program) override {
    INTERNAL_CHECK(program) << "ProgramPass cannot run on null program";
    return transform_(program);
  }

  [[nodiscard]] std::string GetName() const override { return name_.empty() ? "ProgramPass" : name_; }

 private:
  std::function<ProgramPtr(const ProgramPtr&)> transform_;
  std::string name_;
};

/**
 * @brief Pass implementation that applies a function transform to each function in program
 */
class FunctionPassImpl : public PassImpl {
 public:
  FunctionPassImpl(std::function<FunctionPtr(const FunctionPtr&)> transform, std::string name)
      : transform_(std::move(transform)), name_(std::move(name)) {}

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

 private:
  std::function<FunctionPtr(const FunctionPtr&)> transform_;
  std::string name_;
};

}  // namespace

// Factory functions for utility passes
namespace pass {

Pass CreateProgramPass(std::function<ProgramPtr(const ProgramPtr&)> transform, const std::string& name) {
  return Pass(std::make_shared<ProgramPassImpl>(std::move(transform), name));
}

Pass CreateFunctionPass(std::function<FunctionPtr(const FunctionPtr&)> transform, const std::string& name) {
  return Pass(std::make_shared<FunctionPassImpl>(std::move(transform), name));
}

Pass RunVerifier(const std::vector<std::string>& disabled_rules) {
  return CreateProgramPass(
      [disabled_rules](const ProgramPtr& program) -> ProgramPtr {
        // Create default verifier with all rules
        IRVerifier verifier = IRVerifier::CreateDefault();

        // Disable requested rules
        for (const auto& rule_name : disabled_rules) {
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

}  // namespace pass
}  // namespace ir
}  // namespace pypto
