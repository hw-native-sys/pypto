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
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

// Helper function to align address to 32-byte boundary
inline uint64_t Align32(uint64_t addr) { return (addr + 31) & ~31ULL; }

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

// Mutator to update MemRef addresses in IR
class MemRefUpdateMutator : public IRMutator {
 public:
  explicit MemRefUpdateMutator(const std::vector<std::pair<const MemRef*, MemRefPtr>>& memref_pairs) {
    // Build lookup map from vector
    for (const auto& [old_ptr, new_memref] : memref_pairs) {
      memref_map_[old_ptr] = new_memref;
    }
  }

  ExprPtr VisitExpr_(const VarPtr& op) override {
    TypePtr new_type = UpdateTypeMemRef(op->GetType());
    if (new_type != op->GetType()) {
      return std::make_shared<Var>(op->name_, new_type, op->span_);
    }
    return op;
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    // Visit initValue first
    auto new_init = VisitExpr(op->initValue_);
    TypePtr new_type = UpdateTypeMemRef(op->GetType());

    if (new_init != op->initValue_ || new_type != op->GetType()) {
      return std::make_shared<IterArg>(op->name_, new_type, new_init, op->span_);
    }
    return op;
  }

 private:
  std::unordered_map<const MemRef*, MemRefPtr> memref_map_;

  // Helper to update MemRef in a type
  TypePtr UpdateTypeMemRef(const TypePtr& type) {
    if (auto tensor_type = std::dynamic_pointer_cast<const TensorType>(type)) {
      if (tensor_type->memref_.has_value()) {
        auto it = memref_map_.find(tensor_type->memref_.value().get());
        if (it != memref_map_.end()) {
          return std::make_shared<TensorType>(tensor_type->shape_, tensor_type->dtype_, it->second);
        }
      }
    } else if (auto tile_type = std::dynamic_pointer_cast<const TileType>(type)) {
      if (tile_type->memref_.has_value()) {
        auto it = memref_map_.find(tile_type->memref_.value().get());
        if (it != memref_map_.end()) {
          return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, it->second,
                                            tile_type->tile_view_);
        }
      }
    }
    return type;
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
 * @brief Allocate memory addresses for non-DDR memory spaces
 */
std::vector<std::pair<const MemRef*, MemRefPtr>> AllocateMemoryAddresses(
    const std::vector<MemRefPtr>& memrefs) {
  // Group MemRefs by memory space
  std::unordered_map<MemorySpace, std::vector<MemRefPtr>> space_to_memrefs;
  for (const auto& memref : memrefs) {
    space_to_memrefs[memref->memory_space_].push_back(memref);
  }

  // Create new MemRefs with allocated addresses for each memory space
  std::vector<std::pair<const MemRef*, MemRefPtr>> memref_pairs;

  for (auto& [space, refs] : space_to_memrefs) {
    // Skip DDR space - keep original MemRefs
    if (space == MemorySpace::DDR) {
      continue;
    }

    // Sort by ID for deterministic allocation
    std::sort(refs.begin(), refs.end(),
              [](const MemRefPtr& a, const MemRefPtr& b) { return a->id_ < b->id_; });

    // Allocate sequential aligned addresses
    uint64_t current_addr = 0;
    for (const auto& old_memref : refs) {
      // Create new MemRef with allocated address
      auto addr_expr =
          std::make_shared<ConstInt>(static_cast<int64_t>(current_addr), DataType::INDEX, Span::unknown());
      auto new_memref = std::make_shared<MemRef>(old_memref->memory_space_, addr_expr, old_memref->size_,
                                                 old_memref->id_, old_memref->span_);
      memref_pairs.emplace_back(old_memref.get(), new_memref);

      // Next address = align(current + size)
      current_addr = Align32(current_addr + old_memref->size_);
    }
  }

  // Sort by address (ascending order) so alloc statements are in address order
  std::sort(memref_pairs.begin(), memref_pairs.end(),
            [](const std::pair<const MemRef*, MemRefPtr>& a, const std::pair<const MemRef*, MemRefPtr>& b) {
              // Extract address values for comparison
              auto addr_a = std::dynamic_pointer_cast<const ConstInt>(a.second->addr_);
              auto addr_b = std::dynamic_pointer_cast<const ConstInt>(b.second->addr_);
              if (addr_a && addr_b) {
                return addr_a->value_ < addr_b->value_;
              }
              // Fallback: sort by ID if addresses are not ConstInt
              return a.second->id_ < b.second->id_;
            });

  return memref_pairs;
}

/**
 * @brief Create alloc statements for allocated MemRefs
 */
std::vector<StmtPtr> CreateAllocStatements(
    const std::vector<std::pair<const MemRef*, MemRefPtr>>& memref_pairs) {
  std::vector<StmtPtr> alloc_stmts;
  alloc_stmts.reserve(memref_pairs.size());

  // Create alloc statements in order (already sorted by address)
  for (const auto& [old_memref_ptr, new_memref] : memref_pairs) {
    // Create block.alloc operation with all MemRef fields as arguments
    auto alloc_op = std::make_shared<Op>("block.alloc");

    // Create expressions for each MemRef field:
    // 1. memory_space - Convert enum to ConstInt
    auto memspace_expr = std::make_shared<ConstInt>(static_cast<int64_t>(new_memref->memory_space_),
                                                    DataType::INDEX, Span::unknown());

    // 2. addr - Use the new allocated address from new_memref
    ExprPtr addr_expr = new_memref->addr_;

    // 3. size - Convert uint64_t to ConstInt
    auto size_expr =
        std::make_shared<ConstInt>(static_cast<int64_t>(new_memref->size_), DataType::INDEX, Span::unknown());

    // 4. id - Convert uint64_t to ConstInt
    auto id_expr =
        std::make_shared<ConstInt>(static_cast<int64_t>(new_memref->id_), DataType::INDEX, Span::unknown());

    // Build argument vector: [memspace, addr, size, id]
    std::vector<ExprPtr> alloc_args;
    alloc_args.push_back(memspace_expr);
    alloc_args.push_back(addr_expr);
    alloc_args.push_back(size_expr);
    alloc_args.push_back(id_expr);

    // Create a Call expression for the alloc operation
    auto alloc_call = std::make_shared<Call>(alloc_op, alloc_args, GetMemRefType(), Span::unknown());

    // Create an assignment statement: mem_xxx: MemRefType = block.alloc(memspace, addr, size, id)
    auto assign_stmt = std::make_shared<AssignStmt>(new_memref, alloc_call, Span::unknown());
    alloc_stmts.push_back(assign_stmt);
  }

  return alloc_stmts;
}

/**
 * @brief Prepend alloc statements to function body
 */
StmtPtr PrependAllocStatements(const StmtPtr& body, const std::vector<StmtPtr>& alloc_stmts) {
  if (alloc_stmts.empty()) {
    return body;
  }

  // If there are alloc statements, create a sequence
  if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(body)) {
    // Append alloc statements before existing statements
    std::vector<StmtPtr> all_stmts = alloc_stmts;
    all_stmts.insert(all_stmts.end(), seq->stmts_.begin(), seq->stmts_.end());
    return std::make_shared<SeqStmts>(all_stmts, body->span_);
  } else {
    // Wrap existing body in sequence with alloc statements
    std::vector<StmtPtr> all_stmts = alloc_stmts;
    all_stmts.push_back(body);
    return std::make_shared<SeqStmts>(all_stmts, body->span_);
  }
}

/**
 * @brief Transform a function by adding alloc operations for TileType MemRefs
 */
FunctionPtr TransformAddAlloc(const FunctionPtr& func) {
  // Step 1: Collect all unique MemRef objects from TileType variables in the function
  std::vector<MemRefPtr> memrefs;
  CollectMemRefsFromStatement(func->body_, memrefs);

  // Step 2: Allocate memory addresses for non-DDR spaces
  // Returns vector of (old MemRef, new MemRef) pairs sorted by allocated address
  auto memref_pairs = AllocateMemoryAddresses(memrefs);

  // If no MemRefs need allocation (e.g., all are DDR), return early
  if (memref_pairs.empty()) {
    return func;
  }

  // Step 3: Update all MemRef references in the IR with new MemRefs
  MemRefUpdateMutator mutator(memref_pairs);

  // Update function parameters
  std::vector<VarPtr> new_params;
  for (const auto& param : func->params_) {
    auto new_param_expr = mutator.VisitExpr(param);
    auto new_param = std::dynamic_pointer_cast<const Var>(new_param_expr);
    INTERNAL_CHECK(new_param) << "Failed to cast mutated param to Var";
    new_params.push_back(new_param);
  }

  // Update function body
  auto new_body = mutator.VisitStmt(func->body_);

  // Step 4: Create alloc statements for each new MemRef (in address order)
  auto alloc_stmts = CreateAllocStatements(memref_pairs);

  // Step 5: Prepend alloc statements to function body
  new_body = PrependAllocStatements(new_body, alloc_stmts);

  // Step 6: Return transformed function
  return std::make_shared<Function>(func->name_, new_params, func->return_types_, new_body, func->span_,
                                    func->func_type_);
}

}  // namespace

// Factory function
namespace pass {
Pass AddAlloc() { return CreateFunctionPass(TransformAddAlloc, "AddAlloc", kAddAllocProperties); }
}  // namespace pass

}  // namespace ir
}  // namespace pypto
