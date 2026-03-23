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

/**
 * @file legalize_pto_buffer_reuse_pass.cpp
 * @brief PTO backend-specific buffer reuse legalisation
 *
 * After generic MemoryReuse, multiple tile variables with different
 * TileBufSignatures may share the same MemRef.  PTO codegen requires that
 * every non-view writer sharing a MemRef produces the same typed alloc_tile
 * signature.  This pass detects illegal cross-type sharing and splits the
 * offending MemRef into distinct allocations.
 *
 * "Legal" cross-type sharing is defined as differences that existing PTO view
 * ops (treshape, textract, tfillpad) can materialise.  All other differences
 * are illegal and trigger a MemRef split.
 */

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/codegen/pto/tile_buf_signature.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/ir_property.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace {

using codegen::TileBufSignature;

/// Set of IR op names whose output shares the input MemRef and can be
/// expressed as a PTO view instruction (the output type may differ from the
/// MemRef's root alloc type).
static bool IsLegalViewOp(const std::string& op_name) {
  return op_name == "tile.reshape" || op_name == "tile.extract" || op_name == "tile.slice" ||
         op_name == "tile.fillpad";
}

// -------------------------------------------------------------------------
// Phase 1 — Collect per-MemRef tile usage information
// -------------------------------------------------------------------------

struct MemRefUsageInfo {
  const MemRef* ptr = nullptr;
  std::vector<std::pair<const Var*, TileBufSignature>> writers;
  std::vector<std::pair<const Var*, TileBufSignature>> view_users;
  std::vector<std::pair<const Var*, const Var*>> view_edges;
};

class MemRefUsageCollector : public IRVisitor {
 public:
  void VisitStmt_(const AssignStmtPtr& op) override {
    auto tile_type = GetTileTypeWithMemRef(op->var_->GetType());
    if (!tile_type) {
      IRVisitor::VisitStmt_(op);
      return;
    }

    const MemRef* memref_ptr = GetDefinedMemRef(tile_type).get();
    auto sig = TileBufSignature::FromTileType(*tile_type);

    CallPtr call;
    bool is_view = false;
    if (auto maybe_call = As<Call>(op->value_)) {
      call = maybe_call;
      if (IsLegalViewOp(call->op_->name_)) {
        is_view = true;
      }
    }

    auto& info = GetOrCreate(memref_ptr);
    if (is_view) {
      info.view_users.emplace_back(op->var_.get(), sig);
      for (const auto& arg : call->args_) {
        if (auto source_var = As<Var>(arg)) {
          if (auto source_tile_type = GetTileTypeWithMemRef(source_var->GetType())) {
            if (GetDefinedMemRef(source_tile_type).get() == memref_ptr) {
              info.view_edges.emplace_back(source_var.get(), op->var_.get());
              break;
            }
          }
        }
      }
    } else {
      info.writers.emplace_back(op->var_.get(), sig);
    }

    IRVisitor::VisitStmt_(op);
  }

  [[nodiscard]] const std::map<const MemRef*, MemRefUsageInfo>& GetUsages() const { return usages_; }

 private:
  std::map<const MemRef*, MemRefUsageInfo> usages_;

  MemRefUsageInfo& GetOrCreate(const MemRef* ptr) {
    auto it = usages_.find(ptr);
    if (it == usages_.end()) {
      MemRefUsageInfo info;
      info.ptr = ptr;
      it = usages_.emplace(ptr, std::move(info)).first;
    }
    return it->second;
  }
};

// -------------------------------------------------------------------------
// Phase 2 — Decide which MemRefs must be split
// -------------------------------------------------------------------------

/// For each MemRef that has multiple writers with incompatible root
/// signatures, collect the set of Var* that need a fresh MemRef.
///
/// Strategy: the first writer keeps the original MemRef.  Every subsequent
/// writer whose root signature differs gets a new MemRef.
void PropagateSplitToViewUsers(const MemRefUsageInfo& info, const std::vector<const Var*>& split_roots,
                               const MemRefPtr& new_memref, std::map<const Var*, MemRefPtr>& splits) {
  std::vector<const Var*> worklist = split_roots;
  std::map<const Var*, bool> visited;
  for (const Var* root : split_roots) {
    visited[root] = true;
  }

  for (size_t i = 0; i < worklist.size(); ++i) {
    const Var* source = worklist[i];
    for (const auto& [view_source, view_user] : info.view_edges) {
      if (view_source != source || visited[view_user]) {
        continue;
      }
      visited[view_user] = true;
      splits[view_user] = new_memref;
      worklist.push_back(view_user);
    }
  }
}

std::map<const Var*, MemRefPtr> PlanMemRefSplits(const std::map<const MemRef*, MemRefUsageInfo>& usages,
                                                 uint64_t& next_id) {
  std::map<const Var*, MemRefPtr> splits;

  for (const auto& [memref_ptr, info] : usages) {
    if (info.writers.size() <= 1) continue;

    const auto& root_sig = info.writers[0].second.RootSignature();
    bool needs_split = false;
    for (size_t i = 1; i < info.writers.size(); ++i) {
      if (info.writers[i].second.RootSignature() != root_sig) {
        needs_split = true;
        break;
      }
    }
    if (!needs_split) continue;

    // Group writers by root signature; first group keeps original MemRef
    std::map<int, std::vector<size_t>> sig_groups;
    std::vector<TileBufSignature> distinct_roots;

    for (size_t i = 0; i < info.writers.size(); ++i) {
      auto root = info.writers[i].second.RootSignature();
      int group_id = -1;
      for (size_t g = 0; g < distinct_roots.size(); ++g) {
        if (distinct_roots[g] == root) {
          group_id = static_cast<int>(g);
          break;
        }
      }
      if (group_id < 0) {
        group_id = static_cast<int>(distinct_roots.size());
        distinct_roots.push_back(root);
      }
      sig_groups[group_id].push_back(i);
    }

    // Group 0 keeps original MemRef; groups 1..N get fresh MemRefs
    for (auto& [gid, indices] : sig_groups) {
      if (gid == 0) continue;

      auto memory_space = info.writers[indices[0]].second.memory_space;
      auto addr = std::make_shared<ConstInt>(-1, DataType::INDEX, Span::unknown());
      auto new_memref = std::make_shared<MemRef>(memory_space, addr, memref_ptr->size_, next_id++);
      std::vector<const Var*> split_roots;

      for (size_t idx : indices) {
        splits[info.writers[idx].first] = new_memref;
        split_roots.push_back(info.writers[idx].first);
      }
      PropagateSplitToViewUsers(info, split_roots, new_memref, splits);
    }
  }
  return splits;
}

// -------------------------------------------------------------------------
// Phase 3 — Mutate: replace MemRef in split variables
// -------------------------------------------------------------------------

class MemRefSplitMutator : public IRMutator {
 public:
  explicit MemRefSplitMutator(const std::map<const Var*, MemRefPtr>& splits) : splits_(splits) {}

  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = var_remap_.find(op.get());
    if (it != var_remap_.end()) return it->second;

    auto split_it = splits_.find(op.get());
    if (split_it == splits_.end()) return op;

    const auto& new_memref = split_it->second;
    auto tile_type = As<TileType>(op->GetType());
    if (!tile_type) return op;

    auto new_type = std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, new_memref,
                                               tile_type->tile_view_, tile_type->memory_space_);
    auto new_var = std::make_shared<Var>(op->name_hint_, new_type, op->span_);
    var_remap_[op.get()] = new_var;
    return new_var;
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    auto it = var_remap_.find(op.get());
    if (it != var_remap_.end()) return it->second;

    auto new_init = VisitExpr(op->initValue_);

    auto split_it = splits_.find(op.get());
    if (split_it == splits_.end() && new_init == op->initValue_) return op;

    TypePtr new_type = op->GetType();
    if (split_it != splits_.end()) {
      if (auto tile_type = As<TileType>(op->GetType())) {
        new_type = std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, split_it->second,
                                              tile_type->tile_view_, tile_type->memory_space_);
      }
    }

    auto new_iter = std::make_shared<IterArg>(op->name_hint_, new_type, new_init, op->span_);
    var_remap_[op.get()] = new_iter;
    return new_iter;
  }

 private:
  const std::map<const Var*, MemRefPtr>& splits_;
  std::map<const Expr*, ExprPtr> var_remap_;
};

// -------------------------------------------------------------------------
// Phase 4 — Create alloc statements for newly-split MemRefs
// -------------------------------------------------------------------------

StmtPtr InsertNewAllocStatements(const StmtPtr& body, const std::map<const Var*, MemRefPtr>& splits) {
  // Collect unique new MemRefs
  std::map<const MemRef*, std::pair<MemRefPtr, MemorySpace>> new_memrefs;
  for (const auto& [var, memref] : splits) {
    if (new_memrefs.count(memref.get()) > 0) continue;
    auto tile_type = As<TileType>(var->GetType());
    MemorySpace space =
        tile_type && tile_type->memory_space_.has_value() ? *tile_type->memory_space_ : MemorySpace::Vec;
    new_memrefs[memref.get()] = {memref, space};
  }
  if (new_memrefs.empty()) return body;

  // Build alloc statements
  std::vector<StmtPtr> alloc_stmts;
  for (const auto& [_, pair] : new_memrefs) {
    const auto& [memref, space] = pair;
    auto alloc_op = std::make_shared<Op>("tile.alloc");
    auto memspace_expr =
        std::make_shared<ConstInt>(static_cast<int64_t>(space), DataType::INDEX, Span::unknown());
    auto addr_expr = memref->addr_;
    auto size_expr =
        std::make_shared<ConstInt>(static_cast<int64_t>(memref->size_), DataType::INDEX, Span::unknown());
    auto id_expr =
        std::make_shared<ConstInt>(static_cast<int64_t>(memref->id_), DataType::INDEX, Span::unknown());

    std::vector<ExprPtr> alloc_args = {memspace_expr, addr_expr, size_expr, id_expr};
    auto alloc_call = std::make_shared<Call>(alloc_op, alloc_args, GetMemRefType(), Span::unknown());
    alloc_stmts.push_back(std::make_shared<AssignStmt>(memref, alloc_call, Span::unknown()));
  }

  auto seq = As<SeqStmts>(body);
  if (!seq || seq->stmts_.empty()) return body;

  std::vector<StmtPtr> new_stmts;
  new_stmts.reserve(alloc_stmts.size() + seq->stmts_.size());
  new_stmts.insert(new_stmts.end(), alloc_stmts.begin(), alloc_stmts.end());
  new_stmts.insert(new_stmts.end(), seq->stmts_.begin(), seq->stmts_.end());
  return SeqStmts::Flatten(std::move(new_stmts), body->span_);
}

// -------------------------------------------------------------------------
// Top-level transform
// -------------------------------------------------------------------------

/// Find the highest MemRef id in the function (for generating fresh ids)
class MaxMemRefIdCollector : public IRVisitor {
 public:
  void VisitVarLike_(const VarPtr& op) override {
    if (auto tile_type = GetTileTypeWithMemRef(op->GetType())) {
      auto memref = GetDefinedMemRef(tile_type);
      if (memref->id_ >= max_id_) max_id_ = memref->id_ + 1;
    }
  }
  [[nodiscard]] uint64_t GetNextId() const { return max_id_; }

 private:
  uint64_t max_id_ = 0;
};

FunctionPtr TransformLegalizePTOBufferReuse(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "LegalizePTOBufferReuse cannot run on null function";

  // Phase 1: Collect MemRef usage
  MemRefUsageCollector collector;
  if (func->body_) collector.VisitStmt(func->body_);

  const auto& usages = collector.GetUsages();
  if (usages.empty()) return func;

  // Phase 2: Plan splits
  MaxMemRefIdCollector id_collector;
  if (func->body_) id_collector.VisitStmt(func->body_);
  uint64_t next_id = id_collector.GetNextId();

  auto splits = PlanMemRefSplits(usages, next_id);
  if (splits.empty()) return func;

  LOG_DEBUG << "LegalizePTOBufferReuse: splitting " << splits.size() << " variable(s) into new MemRefs";

  // Phase 3: Mutate
  MemRefSplitMutator mutator(splits);
  StmtPtr new_body = mutator.VisitStmt(func->body_);

  // Phase 4: Insert alloc statements for new MemRefs
  new_body = InsertNewAllocStatements(new_body, splits);

  return std::make_shared<const Function>(func->name_, func->params_, func->param_directions_,
                                          func->return_types_, new_body, func->span_, func->func_type_);
}

}  // namespace

namespace pass {

Pass LegalizePTOBufferReuse() {
  static const PassProperties kProps{.required = {IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps,
                                                  IRProperty::HasMemRefs, IRProperty::TileOps2D}};
  return CreateFunctionPass(TransformLegalizePTOBufferReuse, "LegalizePTOBufferReuse", kProps);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
