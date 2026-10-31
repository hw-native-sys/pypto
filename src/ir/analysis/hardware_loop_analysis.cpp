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
#include "pypto/ir/analysis/hardware_loop_analysis.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/type.h"

namespace pypto::ir {
namespace {
std::optional<int64_t> Constant(const ExprPtr& expr) {
  if (auto value = As<ConstInt>(expr)) return value->value_;
  return std::nullopt;
}

struct BodySummary {
  int depth = 0;
  int64_t nodes = 0;
  bool forbidden_control = false;
  bool unknown = false;
  bool unsupported_carry = false;
};

HardwareLoopQualification Qualify(const ForStmtPtr& loop, const HardwareLoopProfile& profile,
                                  const BodySummary& body, const std::string& id) {
  HardwareLoopQualification result;
  result.profile_id = profile.profile_id;
  result.toolchain_id = profile.toolchain_id;
  result.loop_id = id;
  result.lower = Constant(loop->start_);
  result.upper = Constant(loop->stop_);
  result.step = Constant(loop->step_);
  auto finish = [&](HardwareLoopState state, const std::string& reason) {
    result.state = state;
    result.reason = reason;
    return result;
  };
  if (!result.step) return finish(HardwareLoopState::Unknown, "HWL_DYNAMIC_STEP");
  if (*result.step == 0) return finish(HardwareLoopState::Ineligible, "HWL_ZERO_STEP");
  if (!result.lower || !result.upper) return finish(HardwareLoopState::Unknown, "HWL_DYNAMIC_BOUND");

  // Subtraction, ceil division and the terminating induction update can exceed
  // int64 even when every input fits. Never divide before validating the step.
  using Wide = __int128;
  const Wide lower = *result.lower;
  const Wide upper = *result.upper;
  const Wide step = *result.step;
  const Wide distance = step > 0 ? upper - lower : lower - upper;
  const Wide stride = step > 0 ? step : -step;
  const Wide count = distance <= 0 ? 0 : (distance + stride - 1) / stride;
  const Wide post = lower + count * step;
  if (count > std::numeric_limits<int64_t>::max() || post > std::numeric_limits<int64_t>::max() ||
      post < std::numeric_limits<int64_t>::min()) {
    return finish(HardwareLoopState::Ineligible, "HWL_ARITHMETIC_OVERFLOW");
  }
  result.trip_count = static_cast<int64_t>(count);
  result.post_iv = static_cast<int64_t>(post);
  if (loop->kind_ != ForKind::Sequential) {
    return finish(HardwareLoopState::Ineligible, "HWL_FOR_KIND");
  }
  // Empty loops have no hardware realization requirement, even for profiles
  // without hardware-loop support. Simplify owns replacement of loop results.
  if (count == 0) return finish(HardwareLoopState::Eligible, "HWL_STATIC_EMPTY");
  if (step < 0 &&
      (loop->hardware_loop_policy_ == HardwareLoopPolicy::Require || !profile.allow_negative_step)) {
    return finish(HardwareLoopState::Ineligible, "HWL_STEP_SIGN");
  }
  if (profile.counter_bits < 2 || profile.counter_bits > 64 || profile.max_trip_count < 0 ||
      profile.max_depth < 1 || profile.max_body_nodes < 0 || profile.profile_id.empty() ||
      profile.toolchain_id.empty()) {
    return finish(HardwareLoopState::Unknown, "HWL_INVALID_PROFILE");
  }
  const Wide limit = Wide{1} << (profile.counter_bits - 1);
  if (lower < -limit || lower >= limit || post < -limit || post >= limit) {
    return finish(HardwareLoopState::Ineligible, "HWL_COUNTER_RANGE");
  }
  if (count > profile.max_trip_count) return finish(HardwareLoopState::Ineligible, "HWL_TRIP_COUNT");
  if (body.depth + 1 > profile.max_depth) return finish(HardwareLoopState::Ineligible, "HWL_DEPTH");
  if (body.nodes > profile.max_body_nodes) return finish(HardwareLoopState::Ineligible, "HWL_RESOURCES");
  if (body.forbidden_control) return finish(HardwareLoopState::Ineligible, "HWL_CONTROL_FLOW");
  if (body.unsupported_carry) return finish(HardwareLoopState::Ineligible, "HWL_CARRY_TYPE");
  if (!profile.supported) return finish(HardwareLoopState::Unknown, "HWL_UNSUPPORTED_PROFILE");
  if (body.unknown) return finish(HardwareLoopState::Unknown, "HWL_UNKNOWN_BODY");
  return finish(HardwareLoopState::Eligible, "HWL_QUALIFIED");
}

/// A single postorder walk summarizes each body once. Children contribute only
/// their summaries; the analyzer never walks a loop body per iteration or per
/// ancestor. IDs use deterministic program/function/preorder traversal positions.
class Analyzer : public IRVisitor {
 public:
  explicit Analyzer(const HardwareLoopProfile& profile)
      : profile_(profile), allowed_ops_(profile.allowed_ops.begin(), profile.allowed_ops.end()) {}
  std::vector<HardwareLoopQualification> results;
  std::vector<ForStmtPtr> loops;
  std::vector<bool> unreachable;

  void VisitExpr(const ExprPtr& expr) override {
    ++summary_.nodes;
    IRVisitor::VisitExpr(expr);
  }
  void VisitStmt(const StmtPtr& stmt) override {
    ++summary_.nodes;
    IRVisitor::VisitStmt(stmt);
  }

  void VisitFunction(const FunctionPtr& func) override {
    function_ = func->name_;
    ordinal_ = 0;
    IRVisitor::VisitFunction(func);
  }

 protected:
  void VisitStmt_(const ForStmtPtr& op) override {
    const size_t index = results.size();
    const std::string id = function_ + "/loop" + std::to_string(ordinal_++);
    results.emplace_back();
    loops.push_back(op);
    const bool parent_empty = in_empty_loop_;
    // Descendant Require loops are unreachable under a static-empty ancestor.
    // Keep reports, but never request hardware evidence for dead code.
    const auto own_bounds = Qualify(op, profile_, {}, id);
    unreachable.push_back(parent_empty);
    in_empty_loop_ |= own_bounds.trip_count == 0;
    // Headers and carry initializers execute in the enclosing body. Account
    // for them there before starting this loop's per-iteration summary.
    VisitExpr(op->start_);
    VisitExpr(op->stop_);
    VisitExpr(op->step_);
    for (const auto& arg : op->iter_args_) VisitExpr(arg->initValue_);
    const BodySummary parent = summary_;
    summary_ = {};
    for (const auto& arg : op->iter_args_) {
      const auto& type = arg->GetType();
      const bool allowed = (As<ScalarType>(type) && profile_.allow_scalar_carry) ||
                           (As<RegType>(type) && profile_.allow_reg_carry) ||
                           (As<TileType>(type) && profile_.allow_tile_carry);
      summary_.unsupported_carry |= !allowed;
    }
    VisitStmt(op->body_);
    const BodySummary body = summary_;
    in_empty_loop_ = parent_empty;
    results[index] = Qualify(op, profile_, body, id);
    summary_ = parent;
    summary_.nodes += body.nodes;
    summary_.depth = std::max(summary_.depth, body.depth + 1);
    summary_.forbidden_control |= body.forbidden_control;
    summary_.unsupported_carry |= body.unsupported_carry;
    summary_.unknown |= body.unknown || results[index].state != HardwareLoopState::Eligible;
  }
  // A carry use is a value edge, not another execution of its initializer.
  // Explicit loop-header traversal above visits each initializer exactly once.
  void VisitExpr_(const IterArgPtr&) override {}

  void VisitExpr_(const CallPtr& op) override {
    if (std::dynamic_pointer_cast<const GlobalVar>(op->op_) || allowed_ops_.count(op->op_->name_) == 0) {
      summary_.unknown = true;
    }
    IRVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const SubmitPtr& op) override {
    summary_.unknown = true;
    IRVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const WhileStmtPtr& op) override {
    summary_.unknown = true;
    IRVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const IfStmtPtr& op) override {
    // Conditional control flow requires target-specific resource/path proofs.
    summary_.unknown = true;
    IRVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const ReturnStmtPtr&) override { summary_.forbidden_control = true; }
  void VisitStmt_(const BreakStmtPtr&) override { summary_.forbidden_control = true; }
  void VisitStmt_(const ContinueStmtPtr&) override { summary_.forbidden_control = true; }

 private:
  const HardwareLoopProfile& profile_;
  const std::set<std::string> allowed_ops_;
  BodySummary summary_;
  std::string function_ = "root";
  int64_t ordinal_ = 0;
  bool in_empty_loop_ = false;
};

void Enforce(const ForStmtPtr& loop, const HardwareLoopQualification& result) {
  if (loop->hardware_loop_policy_ == HardwareLoopPolicy::Auto || result.trip_count == 0) return;
  const char* state = result.state == HardwareLoopState::Eligible     ? "Eligible"
                      : result.state == HardwareLoopState::Ineligible ? "Ineligible"
                                                                      : "Unknown";
  CHECK_SPAN(result.state == HardwareLoopState::Eligible, loop->span_)
      << "HardwareLoopPolicy.Require " << state << " [" << result.loop_id << ", " << result.profile_id << ", "
      << result.toolchain_id << "]: " << result.reason;
  CHECK_SPAN(false, loop->span_) << "HardwareLoopPolicy.Require backend evidence missing [" << result.loop_id
                                 << ", " << result.profile_id << ", " << result.toolchain_id
                                 << "]: qualification is not hardware evidence";
}
}  // namespace

HardwareLoopQualification AnalyzeHardwareLoop(const ForStmtPtr& loop, const HardwareLoopProfile& profile) {
  CHECK(loop != nullptr) << "AnalyzeHardwareLoop requires a loop";
  Analyzer analyzer(profile);
  analyzer.VisitStmt(loop);
  return analyzer.results.front();
}
std::vector<HardwareLoopQualification> AnalyzeHardwareLoops(const ProgramPtr& program,
                                                            const HardwareLoopProfile& profile) {
  CHECK(program != nullptr) << "AnalyzeHardwareLoops requires a program";
  Analyzer analyzer(profile);
  analyzer.VisitProgram(program);
  return analyzer.results;
}
void VerifyHardwareLoopRequirement(const ForStmtPtr& loop, const HardwareLoopProfile& profile) {
  Analyzer analyzer(profile);
  CHECK(loop != nullptr) << "VerifyHardwareLoopRequirement requires a loop";
  analyzer.VisitStmt(loop);
  for (size_t i = 0; i < analyzer.results.size(); ++i) {
    if (!analyzer.unreachable[i]) Enforce(analyzer.loops[i], analyzer.results[i]);
  }
}
void VerifyHardwareLoopRequirements(const ProgramPtr& program, const HardwareLoopProfile& profile) {
  CHECK(program != nullptr) << "VerifyHardwareLoopRequirements requires a program";
  Analyzer analyzer(profile);
  analyzer.VisitProgram(program);
  for (size_t i = 0; i < analyzer.results.size(); ++i) {
    if (!analyzer.unreachable[i]) Enforce(analyzer.loops[i], analyzer.results[i]);
  }
}
}  // namespace pypto::ir
