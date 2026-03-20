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
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

// Helper function to align address to 32-byte boundary
inline uint64_t Align32(uint64_t addr) { return (addr + 31) & ~31ULL; }

using MemRefWithSpace = std::pair<MemRefPtr, MemorySpace>;

// Visitor to collect all MemRef objects from TileType variables
class MemRefCollectorVisitor : public IRVisitor {
 public:
  MemRefCollectorVisitor() = default;

  [[nodiscard]] const std::vector<MemRefWithSpace>& GetMemRefs() const { return memrefs_; }

  void VisitVarLike_(const VarPtr& op) override {
    if (auto tile_type = GetTileTypeWithMemRef(op->GetType())) {
      AddMemRefIfUnique(tile_type);
    }
  }

 private:
  std::vector<MemRefWithSpace> memrefs_;
  std::map<const MemRef*, MemorySpace> seen_ptrs_;  // Track canonical space per shared MemRef

  void AddMemRefIfUnique(const std::shared_ptr<const TileType>& tile_type) {
    auto memory_space = tile_type->GetMemorySpace();
    CHECK(memory_space.has_value())
        << "TileType with MemRef must have memory_space before address allocation";
    CHECK(tile_type->memref_.has_value()) << "TileType must carry MemRef before address allocation";
    const MemorySpace canonical_space = memory_space.value();

    const auto& memref = tile_type->memref_.value();
    if (TryRegisterUniqueMemRef(memref, canonical_space, seen_ptrs_)) {
      memrefs_.emplace_back(memref, canonical_space);
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
    // Check if already remapped (same old pointer seen again).
    auto it = var_remap_.find(op.get());
    if (it != var_remap_.end()) {
      return it->second;
    }
    TypePtr new_type = UpdateTypeMemRef(op->GetType());
    if (new_type != op->GetType()) {
      auto new_var = std::make_shared<Var>(op->name_hint_, new_type, op->span_);
      var_remap_[op.get()] = new_var;
      return new_var;
    }
    return op;
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    // Check if already remapped.
    auto it = var_remap_.find(op.get());
    if (it != var_remap_.end()) {
      return it->second;
    }
    auto new_init = VisitExpr(op->initValue_);
    TypePtr new_type = UpdateTypeMemRef(op->GetType());

    if (new_init != op->initValue_ || new_type != op->GetType()) {
      auto new_iter_arg = std::make_shared<IterArg>(op->name_hint_, new_type, new_init, op->span_);
      var_remap_[op.get()] = new_iter_arg;
      return new_iter_arg;
    }
    return op;
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    // Handle tile.alloc statements: update LHS MemRef and Call addr argument
    auto memref_var = std::dynamic_pointer_cast<const MemRef>(op->var_);
    if (memref_var) {
      auto call = std::dynamic_pointer_cast<const Call>(op->value_);
      if (call && call->op_->name_ == "tile.alloc") {
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
  std::unordered_map<const Expr*, ExprPtr> var_remap_;

  TypePtr UpdateTypeMemRef(const TypePtr& type) {
    auto memref = GetTypeMemRef(type);
    auto new_memref = memref;
    if (memref.has_value()) {
      auto it = memref_map_.find(memref.value().get());
      if (it != memref_map_.end()) {
        new_memref = it->second;
      }
    }
    return CloneTypeWithMemRefAndRemapExprs(type, new_memref,
                                            [this](const ExprPtr& expr) { return VisitExpr(expr); });
  }
};

/**
 * @brief Helper function to collect MemRefs from a statement
 */
void CollectMemRefsFromStatement(const StmtPtr& stmt, std::vector<MemRefWithSpace>& memrefs) {
  // Create a visitor to traverse the statement
  MemRefCollectorVisitor visitor;
  visitor.VisitStmt(stmt);

  // Add collected MemRefs to the vector (avoiding duplicates by comparing raw pointers)
  std::set<const ir::MemRef*> existing_ptrs;
  for (const auto& [memref, memory_space] : memrefs) {
    (void)memory_space;
    existing_ptrs.insert(memref.get());
  }

  for (const auto& [memref, memory_space] : visitor.GetMemRefs()) {
    if (existing_ptrs.find(memref.get()) == existing_ptrs.end()) {
      memrefs.emplace_back(memref, memory_space);
      existing_ptrs.insert(memref.get());
    }
  }
}

/**
 * @brief Allocate memory addresses for non-DDR memory spaces
 */
std::vector<std::pair<const MemRef*, MemRefPtr>> AllocateMemoryAddresses(
    const std::vector<MemRefWithSpace>& memrefs) {
  // Group MemRefs by memory space
  std::unordered_map<MemorySpace, std::vector<MemRefPtr>> space_to_memrefs;
  for (const auto& [memref, memory_space] : memrefs) {
    space_to_memrefs[memory_space].push_back(memref);
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
      CHECK(old_memref->size_ > 0)
          << "AllocateMemoryAddr encountered zero-sized MemRef '" << old_memref->name_hint_
          << "'. InitMemRef should reject dynamic or invalid allocation shapes before address assignment.";
      // Create new MemRef with allocated address
      auto addr_expr =
          std::make_shared<ConstInt>(static_cast<int64_t>(current_addr), DataType::INDEX, Span::unknown());
      auto new_memref =
          std::make_shared<MemRef>(space, addr_expr, old_memref->size_, old_memref->id_, old_memref->span_);
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
  std::vector<MemRefWithSpace> memrefs;
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
                                    new_body, func->span_, func->func_type_, func->level_, func->role_);
}

}  // namespace

// Factory function
namespace pass {
Pass AllocateMemoryAddr() {
  return CreateFunctionPass(TransformAllocateMemoryAddr, "AllocateMemoryAddr", kAllocateMemoryAddrProperties);
}
}  // namespace pass

// ============================================================================
// AllocatedMemoryAddr property verifier
// ============================================================================

namespace {

/**
 * @brief Collects non-DDR MemRefs and checks address validity.
 *
 * Records diagnostics for MemRefs whose address is still -1 (unallocated).
 * Also tracks the high-water mark (addr + size) per memory space so the
 * caller can compare against platform buffer limits.
 */
class AllocatedMemoryAddrVerifier : public IRVisitor {
 public:
  explicit AllocatedMemoryAddrVerifier(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

  void VisitVarLike_(const VarPtr& op) override {
    if (!op || !op->GetType()) return;
    auto tile_type = As<TileType>(op->GetType());
    if (tile_type && tile_type->memref_.has_value()) {
      auto memory_space = tile_type->GetMemorySpace();
      CHECK(memory_space.has_value())
          << "TileType with MemRef must have memory_space for address verification";
      CheckMemRefAddr(tile_type->memref_.value(), *memory_space, op->name_hint_, op->span_);
    }
  }

  [[nodiscard]] const std::unordered_map<MemorySpace, uint64_t>& GetHighWaterMarks() const {
    return high_water_;
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
  std::set<const MemRef*> seen_;
  std::unordered_map<MemorySpace, uint64_t> high_water_;

  void CheckMemRefAddr(const MemRefPtr& memref, MemorySpace memory_space, const std::string& var_name,
                       const Span& span) {
    if (memory_space == MemorySpace::DDR) return;
    if (!seen_.insert(memref.get()).second) return;

    auto const_addr = std::dynamic_pointer_cast<const ConstInt>(memref->addr_);
    if (!const_addr || const_addr->value_ < 0) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "AllocatedMemoryAddr", 0,
                                "MemRef for variable '" + var_name + "' in " +
                                    MemorySpaceToString(memory_space) +
                                    " has no valid address allocated (addr=-1)",
                                span);
      return;
    }

    uint64_t end = static_cast<uint64_t>(const_addr->value_) + memref->size_;
    auto& hw = high_water_[memory_space];
    if (end > hw) hw = end;
  }
};

}  // namespace

class AllocatedMemoryAddrPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "AllocatedMemoryAddr"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;

    const backend::Backend* be = backend::BackendConfig::IsConfigured() ? backend::GetBackend() : nullptr;

    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;

      AllocatedMemoryAddrVerifier verifier(diagnostics);
      verifier.VisitStmt(func->body_);

      if (!be) continue;

      for (const auto& [space, used] : verifier.GetHighWaterMarks()) {
        uint64_t limit = be->GetMemSize(space);
        if (limit > 0 && used > limit) {
          diagnostics.emplace_back(DiagnosticSeverity::Error, "AllocatedMemoryAddr", 1,
                                   "Function '" + func->name_ + "': " + MemorySpaceToString(space) +
                                       " buffer usage (" + std::to_string(used) +
                                       " bytes) exceeds platform limit (" + std::to_string(limit) + " bytes)",
                                   func->span_);
        }
      }
    }
  }
};

PropertyVerifierPtr CreateAllocatedMemoryAddrPropertyVerifier() {
  return std::make_shared<AllocatedMemoryAddrPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
