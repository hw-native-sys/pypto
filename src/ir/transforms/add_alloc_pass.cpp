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

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

// Visitor to collect all MemRef objects from TileType variables
class MemRefCollectorVisitor : public IRVisitor {
 public:
  MemRefCollectorVisitor() = default;

  [[nodiscard]] const std::vector<MemRefPtr>& GetMemRefs() const { return memrefs_; }

  void VisitExpr_(const VarPtr& op) override {
    // Check if this variable has a TileType with MemRef
    auto tile_type = std::dynamic_pointer_cast<const TileType>(op->GetType());
    if (tile_type && tile_type->memref_.has_value()) {
      AddMemRefIfUnique(tile_type->memref_.value());
    }
  }

  void VisitExpr_(const IterArgPtr& op) override {
    // Check if this iteration argument has a TileType with MemRef
    auto tile_type = std::dynamic_pointer_cast<const TileType>(op->GetType());
    if (tile_type && tile_type->memref_.has_value()) {
      AddMemRefIfUnique(tile_type->memref_.value());
    }
  }

 private:
  std::vector<MemRefPtr> memrefs_;
  std::set<const MemRef*> seen_ptrs_;  // Track raw MemRef pointers to avoid duplicates

  void AddMemRefIfUnique(const MemRefPtr& memref) {
    // Use raw pointer address to check uniqueness (same shared_ptr)
    const MemRef* raw_ptr = memref.get();
    if (seen_ptrs_.find(raw_ptr) == seen_ptrs_.end()) {
      memrefs_.push_back(memref);
      seen_ptrs_.insert(raw_ptr);
    }
  }
};

/**
 * @brief Helper function to collect MemRefs from a statement
 */
void CollectMemRefsFromStatement(const StmtPtr& stmt, std::vector<MemRefPtr>& memrefs) {
  // Create a visitor to traverse the statement
  MemRefCollectorVisitor visitor;
  visitor.VisitStmt(stmt);

  // Add collected MemRefs to the vector (avoiding duplicates by comparing raw pointers)
  std::set<const ir::MemRef*> existing_ptrs;
  for (const auto& mr : memrefs) {
    existing_ptrs.insert(mr.get());
  }

  for (const auto& mr : visitor.GetMemRefs()) {
    if (existing_ptrs.find(mr.get()) == existing_ptrs.end()) {
      memrefs.push_back(mr);
      existing_ptrs.insert(mr.get());
    }
  }
}

/**
 * @brief Transform a function by adding alloc operations for TileType MemRefs
 */
FunctionPtr TransformAddAlloc(const FunctionPtr& func) {
  // Step 1: Collect all unique MemRef objects from TileType variables in the function
  std::vector<MemRefPtr> memrefs;
  CollectMemRefsFromStatement(func->body_, memrefs);

  // Step 2: Create alloc operations for each MemRef
  std::vector<StmtPtr> alloc_stmts;

  for (const auto& memref : memrefs) {
    // Create block.alloc operation with all MemRef fields as arguments
    auto alloc_op = std::make_shared<Op>("block.alloc");

    // Create expressions for each MemRef field:
    // 1. memory_space - Convert enum to ConstInt
    auto memspace_expr = std::make_shared<ConstInt>(static_cast<int64_t>(memref->memory_space_),
                                                    DataType::INT64, Span::unknown());

    // 2. addr - Already an ExprPtr
    ExprPtr addr_expr = memref->addr_;

    // 3. size - Convert uint64_t to ConstInt
    auto size_expr =
        std::make_shared<ConstInt>(static_cast<int64_t>(memref->size_), DataType::INT64, Span::unknown());

    // 4. id - Convert uint64_t to ConstInt
    auto id_expr =
        std::make_shared<ConstInt>(static_cast<int64_t>(memref->id_), DataType::INT64, Span::unknown());

    // Build argument vector: [memspace, addr, size, id]
    std::vector<ExprPtr> alloc_args;
    alloc_args.push_back(memspace_expr);
    alloc_args.push_back(addr_expr);
    alloc_args.push_back(size_expr);
    alloc_args.push_back(id_expr);

    // Create a Call expression for the alloc operation
    // The alloc operation now returns MemRefType
    auto alloc_call = std::make_shared<Call>(alloc_op, alloc_args, GetMemRefType(), Span::unknown());

    // Create an assignment statement: mem_123: MemRefType = block.alloc(memspace, addr, size, id)
    // where mem_123 is the MemRef variable itself (which is already a Var)
    auto assign_stmt = std::make_shared<AssignStmt>(memref, alloc_call, Span::unknown());
    alloc_stmts.push_back(assign_stmt);
  }

  // Step 3: Prepend alloc statements to function body
  StmtPtr new_body = func->body_;

  if (!alloc_stmts.empty()) {
    // If there are alloc statements, create a sequence
    if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(func->body_)) {
      // Append alloc statements before existing statements
      std::vector<StmtPtr> all_stmts = alloc_stmts;
      all_stmts.insert(all_stmts.end(), seq->stmts_.begin(), seq->stmts_.end());
      new_body = std::make_shared<SeqStmts>(all_stmts, func->body_->span_);
    } else {
      // Wrap existing body in sequence with alloc statements
      alloc_stmts.push_back(func->body_);
      new_body = std::make_shared<SeqStmts>(alloc_stmts, func->body_->span_);
    }
  }

  // Step 4: Return transformed function
  return std::make_shared<Function>(func->name_, func->params_, func->return_types_, new_body, func->span_);
}

}  // namespace

namespace pass {
// Factory function
Pass AddAlloc() { return CreateFunctionPass(TransformAddAlloc, "AddAlloc"); }
}  // namespace pass

}  // namespace ir
}  // namespace pypto
