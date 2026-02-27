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

// Mutator to update MemRef addresses in IR (both variable types and alloc statements)
class MemRefUpdateMutator : public IRMutator {
 public:
  explicit MemRefUpdateMutator(const std::vector<std::pair<const MemRef*, MemRefPtr>>& memref_pairs) {
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
    auto new_init = VisitExpr(op->initValue_);
    TypePtr new_type = UpdateTypeMemRef(op->GetType());

    if (new_init != op->initValue_ || new_type != op->GetType()) {
      return std::make_shared<IterArg>(op->name_, new_type, new_init, op->span_);
    }
    return op;
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    // Handle block.alloc statements: update LHS MemRef and Call addr argument
    auto memref_var = std::dynamic_pointer_cast<const MemRef>(op->var_);
    if (memref_var) {
      auto call = std::dynamic_pointer_cast<const Call>(op->value_);
      if (call && call->op_->name_ == "block.alloc") {
        auto it = memref_map_.find(memref_var.get());
        if (it != memref_map_.end()) {
          const auto& new_memref = it->second;
          // Rebuild Call with updated addr argument (index 1)
          std::vector<ExprPtr> new_args = call->args_;
          new_args[1] = new_memref->addr_;
          auto new_call =
              std::make_shared<Call>(call->op_, new_args, call->kwargs_, call->GetType(), call->span_);
          return std::make_shared<AssignStmt>(new_memref, new_call, op->span_);
        }
        return op;
      }
    }
    return IRMutator::VisitStmt_(op);
  }

 private:
  std::unordered_map<const MemRef*, MemRefPtr> memref_map_;

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
 * @brief Allocate real memory addresses for existing alloc operations
 *
 * Alloc statements already exist (created by InitMemRef with addr=-1).
 * This pass assigns real addresses and updates both variable MemRef references
 * and the alloc statement arguments in place.
 */
FunctionPtr TransformAllocateMemoryAddr(const FunctionPtr& func) {
  // Step 1: Collect all unique MemRef objects from TileType variables
  std::vector<MemRefPtr> memrefs;
  CollectMemRefsFromStatement(func->body_, memrefs);

  // Step 2: Allocate memory addresses for non-DDR spaces
  auto memref_pairs = AllocateMemoryAddresses(memrefs);

  if (memref_pairs.empty()) {
    return func;
  }

  // Step 3: Update all MemRef references AND alloc statements in the IR
  MemRefUpdateMutator mutator(memref_pairs);

  std::vector<VarPtr> new_params;
  for (const auto& param : func->params_) {
    auto new_param_expr = mutator.VisitExpr(param);
    auto new_param = std::dynamic_pointer_cast<const Var>(new_param_expr);
    INTERNAL_CHECK(new_param) << "Failed to cast mutated param to Var";
    new_params.push_back(new_param);
  }

  auto new_body = mutator.VisitStmt(func->body_);

  return std::make_shared<Function>(func->name_, new_params, func->param_directions_, func->return_types_,
                                    new_body, func->span_, func->func_type_);
}

}  // namespace

// Factory function
namespace pass {
Pass AllocateMemoryAddr() {
  return CreateFunctionPass(TransformAllocateMemoryAddr, "AllocateMemoryAddr", kAllocateMemoryAddrProperties);
}
}  // namespace pass

}  // namespace ir
}  // namespace pypto
