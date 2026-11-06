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

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/simt.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/result_alias_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"
#include "src/ir/transforms/memory_reuse/storage.h"
#include "src/ir/transforms/utils/simt_utils.h"

namespace pypto::ir {
namespace {

bool IsOutputSeed(const ExprPtr& value) {
  auto call = As<Call>(value);
  if (!call || !IsOp(call, "tile.move")) return false;
  for (const auto& [key, attribute] : call->attrs_) {
    if (key == kSimtOutputSeedAttr) {
      const auto* flag = std::any_cast<bool>(&attribute);
      return flag && *flag;
    }
  }
  return false;
}

/// Each writable Tile gets an initialized destination. InOut alone does not
/// permit overwriting the old Tile value: later statements and loop backedges
/// can still observe it. Tensor handles instead denote mutable GM references.
class OutputMaterializer : public IRMutator {
 public:
  explicit OutputMaterializer(const FunctionPtr& function) {
    names_.Collect(function->body_, function->params_);
  }

 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto result = As<AssignStmt>(IRMutator::VisitStmt_(op));
    if (IsOutputSeed(result->value_)) seeds_.insert(result->var_.get());
    if (auto source = AsVarLike(result->value_)) {
      roots_[result->var_.get()] = RootOf(source);
    } else if (auto call = As<Call>(result->value_)) {
      auto aliased = ResultAliasedArgIndex(call);
      if (!aliased && !call->args_.empty() &&
          (op_predicates::IsBufferAliasingViewOp(call->op_->name_) || IsOp(call, "tensor.slice") ||
           IsOp(call, "tensor.view") || IsOp(call, "tensor.reshape"))) {
        aliased = 0;
      }
      if (aliased) {
        if (auto source = AsVarLike(call->args_[*aliased])) roots_[result->var_.get()] = RootOf(source);
      }
    }
    return result;
  }

  StmtPtr VisitStmt_(const SimtLaunchStmtPtr& op) override {
    auto launch = MutableCopy(As<SimtLaunchStmt>(IRMutator::VisitStmt_(op)));
    CheckAliases(launch);
    std::vector<StmtPtr> statements;
    for (size_t i = 0; i < launch->results_.size(); ++i) {
      const auto index = launch->result_param_indices_[i];
      auto source = AsVarLike(launch->args_[index]);
      INTERNAL_CHECK_SPAN(source, op->span_) << "Internal error: SIMT output source must be a variable";
      auto tile = As<TileType>(source->GetType());
      if (!tile) {
        roots_[launch->results_[i].get()] = RootOf(source);
        continue;
      }
      CHECK_SPAN(tile->memory_space_ == MemorySpace::Vec, op->span_)
          << "SIMT output materialization requires Vec/UB Tile storage";

      // An existing compiler-created seed may serve one launch. Consuming the
      // marker makes the pass idempotent without letting a later launch mutate
      // the first launch's logical result through the same seed.
      if (seeds_.erase(source.get()) != 0) {
        roots_[launch->results_[i].get()] = RootOf(source);
        continue;
      }
      auto move = MutableCopy(OpRegistry::GetInstance().Create(
          "tile.move", {source}, {{"target_memory", MemorySpace::Vec}}, op->span_));
      move->attrs_.emplace_back(kSimtOutputSeedAttr, true);
      auto seed = std::make_shared<Var>(names_.Fresh("simt_seed_"), move->GetType(), op->span_);
      statements.push_back(std::make_shared<AssignStmt>(seed, move, op->span_));
      launch->args_[index] = seed;

      auto result =
          std::make_shared<Var>(launch->results_[i]->name_hint_, seed->GetType(), launch->results_[i]->span_);
      var_remap_[op->results_[i].get()] = result;
      roots_[result.get()] = RootOf(seed);
      launch->results_[i] = std::move(result);
    }
    statements.push_back(launch);
    return SeqStmts::Flatten(std::move(statements), op->span_);
  }

 private:
  const Var* RootOf(const VarPtr& value) {
    auto found = roots_.find(value.get());
    if (found != roots_.end()) return found->second;
    const Var* root = value.get();
    if (auto memory = GetTypeMemRef(value->GetType()); memory && (*memory)->base_) {
      root = (*memory)->base_.get();
    } else if (auto argument = As<IterArg>(value)) {
      if (auto initial = AsVarLike(argument->initValue_)) root = RootOf(initial);
    }
    roots_[value.get()] = root;
    return root;
  }

  void CheckAliases(const SimtLaunchStmtPtr& launch) {
    struct Window {
      int64_t begin = std::numeric_limits<int64_t>::min();
      int64_t end = std::numeric_limits<int64_t>::max();
      bool writes = false;
    };
    std::vector<bool> writes(launch->args_.size(), false);
    for (int index : launch->result_param_indices_) writes[index] = true;
    std::map<const Var*, std::vector<Window>> windows;
    for (size_t i = 0; i < launch->args_.size(); ++i) {
      auto value = AsVarLike(launch->args_[i]);
      if (!value || !As<ShapedType>(value->GetType())) continue;
      Window window;
      window.writes = writes[i];
      if (auto memory = GetTypeMemRef(value->GetType())) {
        auto offset = As<ConstInt>((*memory)->byte_offset_);
        if (offset && offset->value_ >= 0 && (*memory)->size_ > 0 &&
            (*memory)->size_ <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - offset->value_)) {
          window.begin = offset->value_;
          window.end = window.begin + static_cast<int64_t>((*memory)->size_);
        }
      }
      windows[RootOf(value)].push_back(window);
    }
    // Reject known overlapping aliases before creating copies: output
    // materialization is not a substitute for the helper's nonoverlap
    // contract. Unknown runtime GM aliases remain a caller precondition.
    // Sorting intervals avoids a quadratic all-pairs argument comparison.
    for (auto& [_, group] : windows) {
      std::sort(group.begin(), group.end(),
                [](const Window& lhs, const Window& rhs) { return lhs.begin < rhs.begin; });
      int64_t last_end = std::numeric_limits<int64_t>::min();
      int64_t last_write_end = last_end;
      for (const auto& window : group) {
        CHECK_SPAN((window.writes ? last_end : last_write_end) <= window.begin, launch->span_)
            << "SIMT writable arguments have known overlapping storage";
        last_end = std::max(last_end, window.end);
        if (window.writes) last_write_end = std::max(last_write_end, window.end);
      }
    }
  }

  memory_reuse::StorageNameSupply names_;
  std::unordered_set<const Var*> seeds_;
  std::unordered_map<const Var*, const Var*> roots_;
};

class OutputVerifier : public IRVisitor {
 public:
  explicit OutputVerifier(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (IsOutputSeed(op->value_)) seeds_.insert(op->var_.get());
    IRVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const SimtLaunchStmtPtr& op) override {
    for (int index : op->result_param_indices_) {
      if (!As<TileType>(op->args_[index]->GetType())) continue;
      auto source = AsVarLike(op->args_[index]);
      if (!source || seeds_.erase(source.get()) == 0) {
        diagnostics_.emplace_back(DiagnosticSeverity::Error, "SimtOutputsMaterialized", 0,
                                  "SIMT writable Tile requires a dedicated initialized output seed",
                                  op->span_);
      }
    }
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
  std::unordered_set<const Var*> seeds_;
};

class OutputsPropertyVerifier : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "SimtOutputsMaterialized"; }
  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    for (const auto& [_, function] : program->functions_) {
      OutputVerifier verifier(diagnostics);
      verifier.VisitStmt(function->body_);
    }
  }
};

FunctionPtr MaterializeOutputs(const FunctionPtr& function) {
  if (function->func_type_ == FunctionType::SIMT) return function;
  // No-op for ordinary programs, including the Buffer IR pipeline. Buffer
  // lowering of SIMT needs its own destination protocol and is not inferred
  // from this Functional IR copy convention.
  if (!simt_detail::ContainsLaunch(function->body_)) return function;
  CHECK_SPAN(function->ir_stage_ == FunctionIRStage::Functional, function->span_)
      << "SIMT currently supports Functional IR only";
  OutputMaterializer materializer(function);
  return materializer.VisitFunction(function);
}
}  // namespace

namespace pass {
Pass MaterializeSimtOutputs() {
  return CreateFunctionPass(MaterializeOutputs, "MaterializeSimtOutputs", kMaterializeSimtOutputsProperties);
}
}  // namespace pass

PropertyVerifierPtr CreateSimtOutputsMaterializedPropertyVerifier() {
  return std::make_shared<OutputsPropertyVerifier>();
}
}  // namespace pypto::ir
