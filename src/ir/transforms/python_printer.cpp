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

#include "src/ir/transforms/python_printer.h"

#include <algorithm>
#include <any>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/printer.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

// IRPythonPrinter implementation
std::string IRPythonPrinter::Print(const IRNodePtr& node) {
  stream_.str("");
  stream_.clear();
  indent_level_ = 0;

  // Try each type in order
  if (auto program = As<Program>(node)) {
    VisitProgram(program);
  } else if (auto func = As<Function>(node)) {
    VisitFunction(func);
  } else if (auto stmt = As<Stmt>(node)) {
    // Bare-stmt roots (e.g. for_stmt.as_python(), seq.as_python()) need the
    // same disambiguation a function body gets — without this, two distinct
    // Var*/IterArg* sharing a name_hint collapse to one printed identifier.
    BuildVarRenameMap({}, stmt);
    VisitStmt(stmt);
  } else if (auto expr = As<Expr>(node)) {
    VisitExpr(expr);
  } else {
    // Unsupported node type
    stream_ << "<unsupported IRNode type>";
  }

  return stream_.str();
}

std::string IRPythonPrinter::GetIndent() const {
  return std::string(static_cast<size_t>(indent_level_ * 4), ' ');
}

void IRPythonPrinter::IncreaseIndent() { indent_level_++; }

void IRPythonPrinter::DecreaseIndent() {
  if (indent_level_ > 0) {
    indent_level_--;
  }
}

std::string IRPythonPrinter::GetVarName(const Var* var) const {
  // Free variables print with a visible suffix so a dump of malformed IR is
  // unmistakably invalid instead of silently reusing an ordinary-looking name
  // (the marked use-site pinpoints the SSAVerify "used outside its defining
  // scope" diagnostic). dyn_var_rename_map_ entries are never flagged free.
  std::string suffix = free_body_vars_.count(var) ? "__FREE_VAR" : "";
  auto it = var_rename_map_.find(var);
  if (it != var_rename_map_.end()) return it->second + suffix;
  auto dyn_it = dyn_var_rename_map_.find(var);
  if (dyn_it != dyn_var_rename_map_.end()) return dyn_it->second;
  return var->name_hint_ + suffix;
}

std::string IRPythonPrinter::PrintSubExpr(const ExprPtr& expr, bool typed_index) const {
  IRPythonPrinter sub_printer(prefix_);
  sub_printer.var_rename_map_ = var_rename_map_;
  sub_printer.dyn_var_rename_map_ = dyn_var_rename_map_;
  sub_printer.free_body_vars_ = free_body_vars_;
  sub_printer.typed_index_consts_ = typed_index;
  return sub_printer.Print(expr);
}

void IRPythonPrinter::BuildVarRenameMap(const std::vector<VarPtr>& params, const StmtPtr& body,
                                        bool is_function) {
  // Collect Var/IterArg pointers in DFS pre-order: params, then body defs,
  // then body uses. Defs precede uses so a pointer appearing as both keeps
  // its def-site canonical name; pointers appearing only as uses (e.g. a
  // dangling reference from a buggy transform, or a sibling-scope iter_arg
  // sharing a name_hint) still get their own suffix. (#1244)
  std::vector<const Var*> ordered_refs;
  ordered_refs.reserve(params.size());
  std::unordered_set<const Var*> param_ptrs;
  for (const auto& p : params) {
    ordered_refs.push_back(p.get());
    param_ptrs.insert(p.get());
  }
  body_defined_vars_.clear();
  free_body_vars_.clear();
  if (body) {
    var_collectors::VarDefUseCollector body_collector;
    body_collector.VisitStmt(body);
    ordered_refs.insert(ordered_refs.end(), body_collector.var_defs_ordered.begin(),
                        body_collector.var_defs_ordered.end());
    ordered_refs.insert(ordered_refs.end(), body_collector.var_uses_ordered.begin(),
                        body_collector.var_uses_ordered.end());
    // A function is a closed scope: every body use must resolve to a parameter
    // or a body-local definition. A use that resolves to neither is a free
    // variable — malformed IR — so flag it (see free_body_vars_). Skipped for
    // bare-stmt roots, whose absent context legitimately supplies such vars.
    if (is_function) {
      for (const Var* use : body_collector.var_uses_ordered) {
        if (!param_ptrs.count(use) && !body_collector.var_defs.count(use)) {
          free_body_vars_.insert(use);
        }
      }
    }
    body_defined_vars_ = std::move(body_collector.var_defs);
  }
  // Drop program-level dynamic dim vars: they are already disambiguated
  // globally in dyn_var_rename_map_, and re-registering them here would
  // shadow that map in GetVarName's two-tier lookup. They are program-scoped,
  // so they are also legitimately "free" in a function body — drop them from
  // free_body_vars_ too to avoid false positives.
  if (!dyn_var_rename_map_.empty()) {
    ordered_refs.erase(std::remove_if(ordered_refs.begin(), ordered_refs.end(),
                                      [this](const Var* v) { return dyn_var_rename_map_.count(v) > 0; }),
                       ordered_refs.end());
    for (auto it = free_body_vars_.begin(); it != free_body_vars_.end();) {
      if (dyn_var_rename_map_.count(*it)) {
        it = free_body_vars_.erase(it);
      } else {
        ++it;
      }
    }
  }
  auto_name::BuildRenameMapForDefs(ordered_refs, var_rename_map_);
}

void IRPythonPrinter::VisitFunction(const FunctionPtr& func) {
  const bool saved_orchestration =
      std::exchange(orchestration_function_, IsOrchestrationLike(func->func_type_));
  // Build rename map for this function to handle SSA name shadowing.
  BuildVarRenameMap(func);

  // A ``split`` of ``SplitMode::None`` is filtered because it is a non-canonical
  // spelling of "no split" (``Function::GetSplitMode`` maps a stored 0 to
  // ``nullopt``, exactly as an absent key does), and the parser drops it — so
  // emitting it would make print -> parse lossy. No pass produces it; this only
  // normalizes IR that bypassed them (a ``.pto`` blob written before
  // OutlineIncoreScopes stopped stamping it, a programmatically built Function).
  // ``auto_scope`` rides in attrs_ but prints as a dedicated kwarg.
  // The pointer form of ``any_cast`` keeps a mistyped attr failing at the
  // existing print site rather than in this predicate, which also runs for the
  // ``has_attrs`` tests.
  // Every function attr prints as a ``pl.func_attr({...})`` body prologue except
  // the two the parser must read BEFORE it can walk the body — a body-position
  // declaration would arrive too late to take effect. Those two print as their
  // own ``@pl.function(...)`` keyword (``auto_scope=`` / ``external_source=``),
  // which is also the user-facing spelling, so reparsing never goes through the
  // deprecated ``attrs=`` keyword and the compiler never warns on its own
  // output. Kept in sync with the parser's ``_DECORATOR_ONLY_FUNC_ATTRS``.
  auto is_filtered_attr = [](const std::string& k, const std::any& v) {
    if (k == kAttrAutoScope || k == kAttrExternalSource) return true;
    if (k != "split") return false;
    const int* split_value = std::any_cast<int>(&v);
    return split_value != nullptr && *split_value == static_cast<int>(SplitMode::None);
  };
  auto print_func_attr_value = [&](const std::string& key, const std::any& value) {
    if (key == "split") {
      int split_value = AnyCast<int>(value, "func attr key: " + key);
      auto split_mode = static_cast<SplitMode>(split_value);
      stream_ << prefix_ << ".SplitMode." << SplitModeToPythonString(split_mode);
    } else {
      PrintAttrValue(value, func->span_);
    }
  };

  // Buffer dumps are diagnostic: the DSL parser does not construct this stage.
  if (func->ir_stage_ != FunctionIRStage::Functional) {
    stream_ << GetIndent() << "# ir_stage: " << FunctionIRStageToString(func->ir_stage_) << "\n";
  }

  // Print decorator
  stream_ << GetIndent() << "@" << prefix_ << ".function";
  {
    bool has_type = func->func_type_ != FunctionType::Opaque;
    bool has_level = func->level_.has_value();
    bool has_role = func->role_.has_value();
    // Absent ⇒ default True ⇒ not printed.
    bool auto_scope_off = !func->GetAttr<bool>(kAttrAutoScope, true);
    std::string external_source = func->GetAttr<std::string>(kAttrExternalSource, "");
    if (has_type || has_level || has_role || auto_scope_off || !external_source.empty()) {
      stream_ << "(";
      bool first = true;
      if (has_type) {
        stream_ << "type=" << prefix_ << ".FunctionType." << FunctionTypeToString(func->func_type_);
        first = false;
      }
      if (has_level) {
        if (!first) stream_ << ", ";
        stream_ << "level=" << prefix_ << ".Level." << LevelToString(*func->level_);
        first = false;
      }
      if (has_role) {
        if (!first) stream_ << ", ";
        stream_ << "role=" << prefix_ << ".Role." << RoleToString(*func->role_);
        first = false;
      }
      if (auto_scope_off) {
        if (!first) stream_ << ", ";
        stream_ << "auto_scope=False";
        first = false;
      }
      // Last keyword: nothing reads `first` after this, so it is not updated.
      if (!external_source.empty()) {
        if (!first) stream_ << ", ";
        stream_ << "external_source=" << std::quoted(external_source);
      }
      stream_ << ")";
    }
  }
  stream_ << "\n";

  // Print function signature
  stream_ << GetIndent() << "def " << func->name_ << "(";

  // Add 'self' as first parameter when inside @pl.program — except for
  // SubWorker functions, which are declared self-contained.
  bool is_sub_worker = func->role_.has_value() && *func->role_ == Role::SubWorker;
  bool emit_self = current_program_ && !is_sub_worker;
  if (emit_self) {
    stream_ << "self";
  }

  // Everything from here to the closing ":" is the signature, which Python
  // evaluates before the body binds any name. Flag it so a MemRef annotation
  // referring to a body-allocated base Ptr prints its name as a string instead
  // of an unbound forward reference (see PrintMemRef).
  printing_signature_ = true;

  // Print parameters with type annotations and direction wrappers
  for (size_t i = 0; i < func->params_.size(); ++i) {
    if (i > 0 || emit_self) stream_ << ", ";
    const auto& var = func->params_[i];
    const auto& dir = func->param_directions_[i];
    stream_ << GetVarName(var.get()) << ": ";
    if (dir == ParamDirection::InOut) {
      stream_ << prefix_ << ".InOut[" << Print(var->GetType()) << "]";
    } else if (dir == ParamDirection::Out) {
      stream_ << prefix_ << ".Out[" << Print(var->GetType()) << "]";
    } else {
      stream_ << Print(var->GetType());
    }
  }

  stream_ << ")";

  // Print return type annotation
  if (!func->return_types_.empty()) {
    stream_ << " -> ";
    if (func->return_types_.size() == 1) {
      stream_ << Print(func->return_types_[0]);
    } else {
      stream_ << "tuple[";
      for (size_t i = 0; i < func->return_types_.size(); ++i) {
        if (i > 0) stream_ << ", ";
        stream_ << Print(func->return_types_[i]);
      }
      stream_ << "]";
    }
  }

  printing_signature_ = false;

  stream_ << ":\n";

  // Print body - convert yield to return in function context
  IncreaseIndent();

  // ``pl.func_attr({...})`` prologue: every attr that is not decorator-only.
  // Emitted before the body statements so it reparses as a prologue. Skipped
  // for bodyless functions (abstract SubWorker / external kernel), which must
  // round-trip as a bare ``...`` — those carry only decorator-only attrs.
  std::vector<const std::pair<std::string, std::any>*> prologue_attrs;
  for (const auto& kv : func->attrs_) {
    if (is_filtered_attr(kv.first, kv.second)) continue;
    prologue_attrs.push_back(&kv);
  }
  // An abstract SubWorker prints its prologue too, before the bare ``...``.
  // ``_is_abstract_subworker_body`` skips the directive exactly as it skips a
  // docstring, so the body still reads as abstract on reparse and the attrs
  // survive — they would otherwise have no printable position at all.
  if (!prologue_attrs.empty()) {
    stream_ << GetIndent() << prefix_ << ".func_attr({";
    for (size_t i = 0; i < prologue_attrs.size(); ++i) {
      if (i > 0) stream_ << ", ";
      stream_ << std::quoted(prologue_attrs[i]->first) << ": ";
      print_func_attr_value(prologue_attrs[i]->first, prologue_attrs[i]->second);
    }
    stream_ << "})\n";
  }

  if (func->requires_runtime_binding_) {
    // Abstract SubWorker: runtime-bound callback. Round-trips as `...`, which
    // the parser re-detects (see `_is_abstract_subworker_body`).
    stream_ << GetIndent() << "...";
  } else if (func->body_) {
    if (auto seq_stmts = As<SeqStmts>(func->body_)) {
      if (seq_stmts->stmts_.empty()) {
        stream_ << GetIndent() << "pass";
      } else {
        for (size_t i = 0; i < seq_stmts->stmts_.size(); ++i) {
          if (ShouldSuppressPlaceholder(seq_stmts->stmts_, i)) continue;
          // Convert yield to return in function context
          if (auto yield_stmt = As<YieldStmt>(seq_stmts->stmts_[i])) {
            stream_ << GetIndent();
            PrintLeadingComments(yield_stmt);
            stream_ << "return";
            if (!yield_stmt->value_.empty()) {
              stream_ << " ";
              for (size_t j = 0; j < yield_stmt->value_.size(); ++j) {
                if (j > 0) stream_ << ", ";
                VisitExpr(yield_stmt->value_[j]);
              }
            }
          } else {
            PrintStmtBlock(seq_stmts->stmts_[i]);
          }
          if (i < seq_stmts->stmts_.size() - 1) {
            stream_ << "\n";
          }
        }
      }
    } else if (auto yield_stmt = As<YieldStmt>(func->body_)) {
      stream_ << GetIndent();
      PrintLeadingComments(yield_stmt);
      stream_ << "return";
      if (!yield_stmt->value_.empty()) {
        stream_ << " ";
        for (size_t i = 0; i < yield_stmt->value_.size(); ++i) {
          if (i > 0) stream_ << ", ";
          VisitExpr(yield_stmt->value_[i]);
        }
      }
    } else {
      PrintStmtBlock(func->body_);
    }
  }
  DecreaseIndent();
  orchestration_function_ = saved_orchestration;
}

// Helper class to collect GlobalVar references from a function's body
class GlobalVarCollector : public IRVisitor {
 public:
  std::set<GlobalVarPtr, GlobalVarPtrLess> collected_gvars;

  void VisitExpr_(const CallPtr& op) override {
    // Visit the op field (which may be a GlobalVar for cross-function calls)
    INTERNAL_CHECK_SPAN(op->op_, op->span_) << "Call has null op";
    if (auto gvar = As<GlobalVar>(op->op_)) {
      collected_gvars.insert(gvar);
    }
    // Visit arguments
    IRVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const SubmitPtr& op) override {
    // Submit's callee is a GlobalVar (the kernel being launched). Treat the
    // same as Call so cross-function dependencies through pl.submit are
    // captured by the topological sort.
    INTERNAL_CHECK_SPAN(op->op_, op->span_) << "Submit has null op";
    if (auto gvar = As<GlobalVar>(op->op_)) {
      collected_gvars.insert(gvar);
    }
    IRVisitor::VisitExpr_(op);
  }
};

// Topologically sort functions so called functions come before callers
// This ensures that when reparsing, function return types are known when needed
static std::vector<std::pair<GlobalVarPtr, FunctionPtr>> TopologicalSortFunctions(
    const std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess>& functions) {
  // Build dependency graph: function -> set of functions it calls
  std::map<GlobalVarPtr, std::set<GlobalVarPtr, GlobalVarPtrLess>, GlobalVarPtrLess> dependencies;
  std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> gvar_to_func;

  for (const auto& [gvar, func] : functions) {
    gvar_to_func[gvar] = func;
    // Collect all GlobalVars referenced in the function body
    GlobalVarCollector collector;
    if (func->body_) {
      collector.VisitStmt(func->body_);
    }
    // Only keep GlobalVars that are actually functions in this program
    for (const auto& called_gvar : collector.collected_gvars) {
      if (functions.count(called_gvar) > 0) {
        dependencies[gvar].insert(called_gvar);
      }
    }
  }

  // Topological sort using DFS
  std::vector<std::pair<GlobalVarPtr, FunctionPtr>> sorted;
  std::set<GlobalVarPtr, GlobalVarPtrLess> visited;
  std::set<GlobalVarPtr, GlobalVarPtrLess> in_progress;  // For cycle detection

  std::function<bool(const GlobalVarPtr&)> dfs = [&](const GlobalVarPtr& gvar) -> bool {
    if (visited.count(gvar)) return true;
    if (in_progress.count(gvar)) return false;  // Cycle detected

    in_progress.insert(gvar);

    // Visit dependencies first (dependencies = functions this function calls)
    if (dependencies.count(gvar)) {
      for (const auto& dep : dependencies[gvar]) {
        if (!dfs(dep)) return false;  // Cycle detected
      }
    }

    in_progress.erase(gvar);
    visited.insert(gvar);
    // Add to sorted AFTER visiting dependencies, so dependencies come first
    sorted.emplace_back(gvar, gvar_to_func[gvar]);
    return true;
  };

  // Visit all functions
  for (const auto& [gvar, func] : functions) {
    if (!dfs(gvar)) {
      // Cycle detected, fall back to original order
      sorted.clear();
      for (const auto& pair : functions) {
        sorted.emplace_back(pair);
      }
      return sorted;
    }
  }

  return sorted;
}

static std::unordered_map<const Var*, std::string> CollectDynVarMapping(const ProgramPtr& program) {
  // Phase 1: Collect all distinct Var* pointers used as dynamic dims,
  // preserving insertion order for deterministic output.
  std::vector<const Var*> dyn_var_ptrs;
  std::unordered_set<const Var*> seen_ptrs;

  auto try_insert = [&](const Var* var) {
    if (seen_ptrs.insert(var).second) {
      dyn_var_ptrs.push_back(var);
    }
  };

  // Walk an expression tree to find all Var nodes (handles both bare Var dims
  // and complex expressions like M + 1 where Var is a sub-expression).
  std::function<void(const ExprPtr&)> collect_vars_from_expr = [&](const ExprPtr& expr) {
    if (!expr) return;
    if (auto var = As<Var>(expr)) {
      try_insert(var.get());
    } else if (auto bin = As<BinaryExpr>(expr)) {
      collect_vars_from_expr(bin->left_);
      collect_vars_from_expr(bin->right_);
    } else if (auto unary = As<UnaryExpr>(expr)) {
      collect_vars_from_expr(unary->operand_);
    }
  };

  std::function<void(const TypePtr&)> collect_from_type = [&](const TypePtr& type) {
    // AsTensorTypeLike, not As<TensorType>: DistributedTensorType has its own
    // ObjectKind, so the exact-match As<TensorType> misses it and the symbols a
    // pld.DistributedTensor annotation declares never reach dyn_var_rename_map_.
    if (auto tensor_type = AsTensorTypeLike(type)) {
      for (const auto& dim : tensor_type->shape_) {
        collect_vars_from_expr(dim);
      }
      if (tensor_type->tensor_view_.has_value()) {
        for (const auto& dim : tensor_type->tensor_view_->valid_shape) {
          collect_vars_from_expr(dim);
        }
        for (const auto& dim : tensor_type->tensor_view_->stride) {
          collect_vars_from_expr(dim);
        }
      }
    } else if (auto tile_type = As<TileType>(type)) {
      for (const auto& dim : tile_type->shape_) {
        collect_vars_from_expr(dim);
      }
      if (tile_type->tile_view_.has_value()) {
        for (const auto& dim : tile_type->tile_view_->valid_shape) {
          collect_vars_from_expr(dim);
        }
        for (const auto& dim : tile_type->tile_view_->stride) {
          collect_vars_from_expr(dim);
        }
        collect_vars_from_expr(tile_type->tile_view_->start_offset);
      }
    } else if (auto tuple_type = As<TupleType>(type)) {
      for (const auto& elem_type : tuple_type->types_) {
        collect_from_type(elem_type);
      }
    }
  };
  // Use a full IRVisitor so that dynamic-dimension Var names are found in every
  // expression context: loop bounds (ForStmt start/stop/step),
  // if/while conditions, EvalStmt expressions, AssignStmt values, etc.
  // The ad-hoc collect_from_stmt only inspected AssignStmt variable types and
  // missed all those other locations.
  class DynVarCollector : public IRVisitor {
   public:
    explicit DynVarCollector(const std::function<void(const TypePtr&)>& collect_from_type)
        : collect_from_type_(collect_from_type) {}

    void VisitExpr_(const VarPtr& op) override {
      // Collect dynamic dimension names from the variable's type annotation.
      // Handles TensorType/TileType shapes and their tensor_view_/tile_view_ fields.
      collect_from_type_(op->GetType());
    }

   private:
    const std::function<void(const TypePtr&)>& collect_from_type_;
  };

  // Collect dynamic dim vars from type annotations, and simultaneously track
  // all defined vars (params + body defs) so we can filter them out below.
  DynVarCollector collector(collect_from_type);
  std::unordered_set<const Var*> defined_vars;
  for (const auto& [gvar, func] : program->functions_) {
    for (const auto& param : func->params_) {
      collector.VisitExpr(param);
      defined_vars.insert(param.get());
    }
    for (const auto& ret_type : func->return_types_) {
      collect_from_type(ret_type);
    }
    if (func->body_) {
      collector.VisitStmt(func->body_);
      var_collectors::VarDefUseCollector body_def_collector;
      body_def_collector.VisitStmt(func->body_);
      defined_vars.insert(body_def_collector.var_defs.begin(), body_def_collector.var_defs.end());
    }
  }

  // A parameter whose name is ALSO read inside a parameter *type annotation*
  // still needs its pl.dynamic() declaration: Python evaluates annotations in the
  // enclosing scope, before any parameter exists, so without the declaration the
  // printed signature raises NameError on re-parse. MaterializeValidShapeSymbols
  // produces exactly this shape — the valid_shape symbol becomes a scalar
  // parameter while the tensor parameter's annotation keeps naming it — and the
  // parser re-unifies the two (see ast_parser's DynVar re-point on f.param).
  // Collected separately from the pass above: that one dedups through seen_ptrs,
  // which would swallow every repeat occurrence.
  // Deliberately narrow (issue #854 keeps the other two cases undeclared):
  //   * a body-local var read in a valid_shape stays undeclared — it is defined
  //     in the body, so the annotation resolves against the local;
  //   * a param read as a *physical* dim stays undeclared — orchestration codegen
  //     defines those symbols from the runtime tensor's shapes[].
  // Only a param read in a valid_shape has neither escape, which is exactly what
  // MaterializeValidShapeSymbols produces.
  std::unordered_set<const Var*> read_in_param_types;
  {
    std::unordered_set<const Var*> param_vars;
    for (const auto& [gvar, func] : program->functions_) {
      for (const auto& param : func->params_) param_vars.insert(param.get());
    }
    std::function<void(const ExprPtr&)> collect_reads = [&](const ExprPtr& expr) {
      if (!expr) return;
      if (auto var = As<Var>(expr)) {
        if (param_vars.count(var.get()) > 0) read_in_param_types.insert(var.get());
      } else if (auto bin = As<BinaryExpr>(expr)) {
        collect_reads(bin->left_);
        collect_reads(bin->right_);
      } else if (auto unary = As<UnaryExpr>(expr)) {
        collect_reads(unary->operand_);
      }
    };
    for (const auto& [gvar, func] : program->functions_) {
      for (const auto& param : func->params_) {
        // AsTensorTypeLike, not As<TensorType>: DistributedTensorType has its own
        // ObjectKind and carries valid_shape symbols the same way.
        auto tensor_type = AsTensorTypeLike(param->GetType());
        if (!tensor_type || !tensor_type->tensor_view_.has_value()) continue;
        for (const auto& dim : tensor_type->tensor_view_->valid_shape) collect_reads(dim);
      }
    }
  }

  // Filter out locally-defined vars and function params: they should not get
  // pl.dynamic() declarations — only truly free dimension variables should, plus
  // the params that a parameter annotation reads (see read_in_param_types above).
  dyn_var_ptrs.erase(std::remove_if(dyn_var_ptrs.begin(), dyn_var_ptrs.end(),
                                    [&defined_vars, &read_in_param_types](const Var* v) {
                                      return defined_vars.count(v) > 0 && read_in_param_types.count(v) == 0;
                                    }),
                     dyn_var_ptrs.end());

  // Phase 2: Assign unique printed names, disambiguating collisions.
  // Reuses the same rename logic as BuildVarRenameMap.
  // include_unique_names=true so VisitProgram can iterate the full map for
  // pl.dynamic() declarations.
  std::unordered_map<const Var*, std::string> result;
  auto_name::BuildRenameMapForDefs(dyn_var_ptrs, result, /*include_unique_names=*/true);
  return result;
}

void IRPythonPrinter::VisitProgram(const ProgramPtr& program) {
  CHECK(prefix_ != "pld")
      << "Python printer prefix 'pld' is reserved for pypto.language.distributed; choose another prefix";

  // Render everything below the imports into a scratch buffer first. Whether the
  // program needs ``import pypto.language.distributed as pld`` is only knowable
  // once the body exists: every distributed spelling is hardcoded with the
  // ``pld.`` prefix (types, ops, and the comm-domain comment alike), so a plain
  // substring search over the rendered body is an exact test. Without the import
  // the output references an unbound name and is not valid standalone Python --
  // it only re-parses because the text parser injects ``pld`` into the exec
  // namespace itself.
  std::ostringstream body;
  stream_.swap(body);

  // Emit pl.dynamic() declarations for dynamic shape variables used in function signatures.
  // Uses pointer-identity-aware collection so distinct Var* with the same name_hint_
  // get disambiguated printed names (issue #618).
  dyn_var_rename_map_ = CollectDynVarMapping(program);
  if (!dyn_var_rename_map_.empty()) {
    // Sort by disambiguated name for deterministic output
    std::vector<std::pair<const Var*, std::string>> sorted_dyn_vars(dyn_var_rename_map_.begin(),
                                                                    dyn_var_rename_map_.end());
    std::sort(sorted_dyn_vars.begin(), sorted_dyn_vars.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    for (const auto& [var, name] : sorted_dyn_vars) {
      stream_ << name << " = " << prefix_ << ".dynamic(\"" << name << "\")\n";
    }
    stream_ << "\n";
  }

  // Print as @pl.program class with @pl.function methods
  stream_ << "@" << prefix_ << ".program\n";
  stream_ << "class " << (program->name_.empty() ? "Program" : program->name_) << ":\n";

  IncreaseIndent();

  // Sort functions in dependency order (called functions before callers)
  auto sorted_functions = TopologicalSortFunctions(program->functions_);

  // Print each function as a method, delegating to VisitFunction
  // Setting current_program_ enables self parameter and self.method() call printing
  auto prev_program = current_program_;
  current_program_ = program;

  bool first = true;
  for (const auto& [gvar, func] : sorted_functions) {
    if (!first) {
      stream_ << "\n";  // Blank line between functions
    }
    first = false;

    VisitFunction(func);
  }

  current_program_ = prev_program;
  DecreaseIndent();

  // Restore the caller's stream and prepend the header + imports the body needs.
  const std::string body_str = stream_.str();
  stream_.swap(body);

  stream_ << "# pypto.program: " << (program->name_.empty() ? "Program" : program->name_) << "\n";
  if (prefix_ == "pl") {
    stream_ << "import pypto.language as pl\n";
  } else {
    stream_ << "from pypto import language as " << prefix_ << "\n";
  }
  if (body_str.find("pld.") != std::string::npos) {
    stream_ << "import pypto.language.distributed as pld\n";
  }
  if (body_str.find("pypto.ir.") != std::string::npos) {
    stream_ << "import pypto\n";
  }
  stream_ << "\n" << body_str;
}

// ================================
// Public API
// ================================
std::string PythonPrint(const IRNodePtr& node, const std::string& prefix, bool concise,
                        bool explicit_layout) {
  IRPythonPrinter printer(prefix, concise, explicit_layout);
  return printer.Print(node);
}

std::string PythonPrint(const TypePtr& type, const std::string& prefix, bool explicit_layout) {
  IRPythonPrinter printer(prefix, /*concise=*/false, explicit_layout);
  return printer.Print(type);
}

// ================================
// Format Callback
// ================================
namespace {
FormatCallback g_format_callback;  // set once at import time, read-only after
}  // namespace

void RegisterFormatCallback(FormatCallback callback) { g_format_callback = std::move(callback); }

std::string ApplyFormatCallback(const std::string& code) {
  if (!g_format_callback) {
    return code;
  }
  try {
    return g_format_callback(code);
  } catch (...) {
    // Best-effort: return raw output on any failure (e.g., Python exception in ruff)
    return code;
  }
}

}  // namespace ir
}  // namespace pypto
