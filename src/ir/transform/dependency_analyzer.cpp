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

#include "pypto/ir/transform/dependency_analyzer.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transform/base/visitor.h"

namespace pypto {
namespace ir {

DependencyGraph DependencyAnalyzer::Analyze(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "DependencyAnalyzer cannot analyze null function";

  // Step 1: Identify basic blocks from control flow
  std::vector<BasicBlock> blocks = IdentifyBasicBlocks(func->body_);
  LOG_INFO << "Identified " << blocks.size() << " basic blocks";

  // Step 2: Build dependency graph for each basic block
  std::vector<DependencyEdge> all_dependencies;
  for (const auto& block : blocks) {
    auto block_deps = AnalyzeBlockDependencies(block);
    all_dependencies.insert(all_dependencies.end(), block_deps.begin(), block_deps.end());
  }
  LOG_INFO << "Found " << all_dependencies.size() << " dependency edges";

  // Step 3: Log dependency statistics
  int raw_count = 0, war_count = 0, waw_count = 0;
  for (const auto& edge : all_dependencies) {
    switch (edge.type) {
      case DependencyEdge::RAW:
        raw_count++;
        break;
      case DependencyEdge::WAR:
        war_count++;
        break;
      case DependencyEdge::WAW:
        waw_count++;
        break;
    }
  }
  LOG_INFO << "Dependency types: RAW=" << raw_count << ", WAR=" << war_count << ", WAW=" << waw_count;

  return DependencyGraph(std::move(blocks), std::move(all_dependencies));
}

std::vector<BasicBlock> DependencyAnalyzer::AnalyzeBasicBlocks(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "DependencyAnalyzer cannot analyze null function";
  return IdentifyBasicBlocks(func->body_);
}

std::vector<DependencyEdge> DependencyAnalyzer::AnalyzeDependencies(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "DependencyAnalyzer cannot analyze null function";

  // Create a single basic block containing all statements
  std::vector<BasicBlock> blocks = IdentifyBasicBlocks(func->body_);

  std::vector<DependencyEdge> all_dependencies;
  for (const auto& block : blocks) {
    auto block_deps = AnalyzeBlockDependencies(block);
    all_dependencies.insert(all_dependencies.end(), block_deps.begin(), block_deps.end());
  }

  return all_dependencies;
}

std::vector<BasicBlock> DependencyAnalyzer::IdentifyBasicBlocks(const StmtPtr& stmt) {
  std::vector<BasicBlock> blocks;
  int next_id = 0;

  // Helper class to traverse statements and build basic blocks
  class BasicBlockBuilder {
   public:
    explicit BasicBlockBuilder(std::vector<BasicBlock>& blocks, int& next_id)
        : blocks_(blocks), next_id_(next_id) {}

    // Process a statement and create basic blocks
    int ProcessStmt(const StmtPtr& stmt, const std::vector<int>& predecessors) {
      if (!stmt) {
        return -1;
      }

      // Handle different statement types
      if (auto seq = As<SeqStmts>(stmt)) {
        return ProcessSeqStmts(seq, predecessors);
      } else if (auto if_stmt = As<IfStmt>(stmt)) {
        return ProcessIfStmt(if_stmt, predecessors);
      } else if (auto for_stmt = As<ForStmt>(stmt)) {
        return ProcessForStmt(for_stmt, predecessors);
      } else {
        // Single statement forms a basic block
        return CreateSingleStmtBlock(stmt, predecessors, false);
      }
    }

   private:
    int ProcessSeqStmts(const std::shared_ptr<const SeqStmts>& seq, const std::vector<int>& predecessors) {
      if (seq->stmts_.empty()) {
        return -1;
      }

      // Process statements sequentially
      std::vector<int> current_preds = predecessors;
      int last_block_id = -1;

      for (const auto& sub_stmt : seq->stmts_) {
        last_block_id = ProcessStmt(sub_stmt, current_preds);
        if (last_block_id != -1) {
          current_preds = {last_block_id};
        }
      }

      return last_block_id;
    }

    int ProcessIfStmt(const std::shared_ptr<const IfStmt>& if_stmt, const std::vector<int>& predecessors) {
      // Create a block for the condition (if needed)
      // For simplicity, we'll process then/else bodies as separate blocks

      // Process then body
      int then_exit = ProcessStmt(if_stmt->then_body_, predecessors);

      // Process else body (if exists)
      int else_exit = -1;
      if (if_stmt->else_body_.has_value()) {
        else_exit = ProcessStmt(*if_stmt->else_body_, predecessors);
      }

      // Create a virtual merge block (both branches merge here)
      // In a more complete implementation, we would track this merge point
      // For now, we'll return the then_exit (simplified)
      return then_exit;
    }

    int ProcessForStmt(const std::shared_ptr<const ForStmt>& for_stmt, const std::vector<int>& predecessors) {
      // Create a basic block for the loop body
      BasicBlock loop_block;
      loop_block.id = next_id_++;
      loop_block.is_loop_body = true;
      loop_block.predecessors = predecessors;

      // Collect statements from loop body
      CollectStmtsInBlock(for_stmt->body_, loop_block.statements);

      // Loop body can jump back to itself (loop-carried dependency)
      loop_block.successors.push_back(loop_block.id);

      // Add a successor for loop exit (next block after loop)
      // We'll model this as continuing to the next block
      int exit_id = next_id_;  // The next block would have this ID
      loop_block.successors.push_back(exit_id);

      blocks_.push_back(loop_block);

      return loop_block.id;
    }

    int CreateSingleStmtBlock(const StmtPtr& stmt, const std::vector<int>& predecessors, bool is_loop) {
      BasicBlock block;
      block.id = next_id_++;
      block.predecessors = predecessors;
      block.is_loop_body = is_loop;

      // Collect statements
      CollectStmtsInBlock(stmt, block.statements);

      blocks_.push_back(block);

      return block.id;
    }

    void CollectStmtsInBlock(const StmtPtr& stmt, std::vector<StmtPtr>& statements) {
      if (!stmt) {
        return;
      }

      // Recursively collect all non-control-flow statements
      if (auto seq = As<SeqStmts>(stmt)) {
        for (const auto& sub_stmt : seq->stmts_) {
          CollectStmtsInBlock(sub_stmt, statements);
        }
      } else if (IsA<IfStmt>(stmt) || IsA<ForStmt>(stmt)) {
        // Control flow statements are not added to the current block
        // They create their own blocks
      } else {
        // Regular statement (AssignStmt, EvalStmt, etc.)
        statements.push_back(stmt);
      }
    }

    std::vector<BasicBlock>& blocks_;
    int& next_id_;
  };

  // Build basic blocks starting from the root statement
  BasicBlockBuilder builder(blocks, next_id);
  builder.ProcessStmt(stmt, {});

  LOG_DEBUG << "Created " << blocks.size() << " basic blocks";

  return blocks;
}

std::vector<DependencyEdge> DependencyAnalyzer::AnalyzeBlockDependencies(const BasicBlock& block) {
  std::vector<DependencyEdge> dependencies;

  // Track last write and last read for each variable
  std::map<std::string, StmtPtr> last_write;               // var_name -> stmt that last wrote it
  std::map<std::string, std::vector<StmtPtr>> last_reads;  // var_name -> stmts that read it

  // Helper class to collect variable reads and writes
  class VarCollector : public IRVisitor {
   public:
    std::set<VarPtr> read_vars;
    std::set<VarPtr> write_vars;

    void VisitExpr_(const VarPtr& var) override {
      read_vars.insert(var);
      IRVisitor::VisitExpr_(var);
    }
  };

  // Process each statement in the block
  for (const auto& stmt : block.statements) {
    VarCollector collector;

    // Identify the defined (written) variable and used (read) variables
    if (auto assign = As<AssignStmt>(stmt)) {
      // The assigned variable is written
      VarPtr def_var = assign->var_;

      // Collect variables read from the value expression
      collector.VisitExpr(assign->value_);

      // Check for RAW dependencies (Read-After-Write)
      for (const auto& read_var : collector.read_vars) {
        std::string var_name = read_var->name_;
        if (last_write.count(var_name)) {
          DependencyEdge edge;
          edge.producer = last_write[var_name];
          edge.consumer = stmt;
          edge.variable = read_var;
          edge.type = DependencyEdge::RAW;
          edge.producer_pipe = GetPipeTypeFromStmt(last_write[var_name]);
          edge.consumer_pipe = GetPipeTypeFromStmt(stmt);
          dependencies.push_back(edge);
        }
      }

      // Check for WAR dependencies (Write-After-Read)
      std::string def_name = def_var->name_;
      if (last_reads.count(def_name)) {
        for (const auto& read_stmt : last_reads[def_name]) {
          DependencyEdge edge;
          edge.producer = read_stmt;
          edge.consumer = stmt;
          edge.variable = def_var;
          edge.type = DependencyEdge::WAR;
          edge.producer_pipe = GetPipeTypeFromStmt(read_stmt);
          edge.consumer_pipe = GetPipeTypeFromStmt(stmt);
          dependencies.push_back(edge);
        }
      }

      // Check for WAW dependencies (Write-After-Write)
      if (last_write.count(def_name)) {
        DependencyEdge edge;
        edge.producer = last_write[def_name];
        edge.consumer = stmt;
        edge.variable = def_var;
        edge.type = DependencyEdge::WAW;
        edge.producer_pipe = GetPipeTypeFromStmt(last_write[def_name]);
        edge.consumer_pipe = GetPipeTypeFromStmt(stmt);
        dependencies.push_back(edge);
      }

      // Update tracking: record this write
      last_write[def_name] = stmt;

      // Record reads for this statement
      for (const auto& read_var : collector.read_vars) {
        last_reads[read_var->name_].push_back(stmt);
      }

    } else if (auto eval_stmt = As<EvalStmt>(stmt)) {
      // EvalStmt doesn't define variables, only reads
      collector.VisitExpr(eval_stmt->expr_);

      // Check for RAW dependencies
      for (const auto& read_var : collector.read_vars) {
        std::string var_name = read_var->name_;
        if (last_write.count(var_name)) {
          DependencyEdge edge;
          edge.producer = last_write[var_name];
          edge.consumer = stmt;
          edge.variable = read_var;
          edge.type = DependencyEdge::RAW;
          edge.producer_pipe = GetPipeTypeFromStmt(last_write[var_name]);
          edge.consumer_pipe = GetPipeTypeFromStmt(stmt);
          dependencies.push_back(edge);
        }
      }

      // Record reads
      for (const auto& read_var : collector.read_vars) {
        last_reads[read_var->name_].push_back(stmt);
      }
    }
  }

  LOG_DEBUG << "Found " << dependencies.size() << " dependencies in block " << block.id;

  return dependencies;
}

// Helper method to extract pipe type from a statement
std::string DependencyAnalyzer::GetPipeTypeFromStmt(const StmtPtr& stmt) {
  if (!stmt) {
    return "UNKNOWN";
  }

  // Try to extract Call expression from the statement
  if (auto assign = As<AssignStmt>(stmt)) {
    if (auto call = As<Call>(assign->value_)) {
      return GetPipeType(call);
    }
  } else if (auto eval_stmt = As<EvalStmt>(stmt)) {
    if (auto call = As<Call>(eval_stmt->expr_)) {
      return GetPipeType(call);
    }
  }

  return "UNKNOWN";
}

std::vector<DependencyEdge> DependencyAnalyzer::MergeDependencies(
    const std::vector<std::vector<DependencyEdge>>& path_dependencies) {
  std::vector<DependencyEdge> merged;

  // Take union of all dependencies from all paths
  // Use a set to track unique edges (by producer, consumer, variable, type)
  std::set<std::tuple<StmtPtr, StmtPtr, std::string, DependencyEdge::Type>> seen;

  for (const auto& path : path_dependencies) {
    for (const auto& edge : path) {
      auto key = std::make_tuple(edge.producer, edge.consumer, edge.variable->name_, edge.type);
      if (seen.find(key) == seen.end()) {
        seen.insert(key);
        merged.push_back(edge);
      }
    }
  }

  LOG_DEBUG << "Merged " << merged.size() << " unique dependencies from " << path_dependencies.size()
            << " paths";

  return merged;
}

std::string DependencyAnalyzer::GetPipeType(const CallPtr& call_expr) {
  if (!call_expr || !call_expr->op_) {
    return "UNKNOWN";
  }

  // Try to get pipe type from the Op
  auto pipe_opt = call_expr->op_->GetPipe();
  if (pipe_opt.has_value()) {
    // Convert PipeType enum to string
    PipeType pipe = *pipe_opt;
    switch (pipe) {
      case PipeType::M:
        return "CUBE";
      case PipeType::V:
        return "VECTOR";
      case PipeType::MTE1:
        return "MTE1";
      case PipeType::MTE2:
        return "MTE2";
      case PipeType::MTE3:
        return "MTE3";
      case PipeType::S:
        return "SCALAR";
      case PipeType::FIX:
        return "FIX";
      case PipeType::ALL:
        return "ALL";
      default:
        return "UNKNOWN";
    }
  }

  // Try to get pipe_type from kwargs
  if (call_expr->HasKwarg("pipe_type")) {
    try {
      std::string pipe_str = call_expr->GetKwarg<std::string>("pipe_type", "UNKNOWN");
      return pipe_str;
    } catch (...) {
      // If type conversion fails, return UNKNOWN
      return "UNKNOWN";
    }
  }

  return "UNKNOWN";
}

}  // namespace ir
}  // namespace pypto
