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
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/codegen/pto/tile_buf_signature.h"
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
         op_name == "tile.fillpad" || op_name == "tile.fillpad_inplace" || op_name == "tensor.slice";
}

// -------------------------------------------------------------------------
// Phase 1 — Collect per-MemRef tile usage information
// -------------------------------------------------------------------------

struct MemRefUsageInfo {
  const Var* base_ptr = nullptr;  ///< base_ Ptr identity key
  uint64_t alloc_size = 0;        ///< Size of the root allocation in bytes
  struct WriterInfo {
    const Var* var = nullptr;
    TileBufSignature signature;
    std::string op_name;
    std::vector<const Var*> input_vars;
  };

  std::vector<WriterInfo> writers;
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

    auto memref = GetDefinedMemRef(tile_type);
    const Var* base_ptr = memref->base_.get();
    auto sig = TileBufSignature::FromTileType(*tile_type);

    CallPtr call;
    bool is_view = false;
    if (auto maybe_call = As<Call>(op->value_)) {
      call = maybe_call;
      if (IsLegalViewOp(call->op_->name_)) {
        is_view = true;
      }
    }

    auto& info = GetOrCreate(base_ptr, memref->size_);
    if (is_view) {
      info.view_users.emplace_back(op->var_.get(), sig);
      for (const auto& arg : call->args_) {
        if (auto source_var = As<Var>(arg)) {
          if (auto source_tile_type = GetTileTypeWithMemRef(source_var->GetType())) {
            if (GetDefinedMemRef(source_tile_type)->base_.get() == base_ptr) {
              info.view_edges.emplace_back(source_var.get(), op->var_.get());
              break;
            }
          }
        }
      }
    } else {
      std::vector<const Var*> input_vars;
      std::string op_name;
      if (call && call->op_) {
        op_name = call->op_->name_;
        for (const auto& arg : call->args_) {
          if (auto input_var = As<Var>(arg)) {
            input_vars.push_back(input_var.get());
          }
        }
        if (op_name == "tile.tpop_from_aic") {
          tpop_from_aic_vars_.insert(op->var_.get());
        }
      }
      info.writers.push_back(
          MemRefUsageInfo::WriterInfo{op->var_.get(), sig, std::move(op_name), std::move(input_vars)});
    }

    IRVisitor::VisitStmt_(op);
  }

  [[nodiscard]] const std::map<const Var*, MemRefUsageInfo>& GetUsages() const { return usages_; }
  [[nodiscard]] const std::unordered_set<const Var*>& GetTpopFromAicVars() const {
    return tpop_from_aic_vars_;
  }

 private:
  std::map<const Var*, MemRefUsageInfo> usages_;
  std::unordered_set<const Var*> tpop_from_aic_vars_;

  MemRefUsageInfo& GetOrCreate(const Var* base_ptr, uint64_t size) {
    auto it = usages_.find(base_ptr);
    if (it == usages_.end()) {
      MemRefUsageInfo info;
      info.base_ptr = base_ptr;
      info.alloc_size = size;
      it = usages_.emplace(base_ptr, std::move(info)).first;
    } else {
      // Keep the max size across all uses
      if (size > it->second.alloc_size) it->second.alloc_size = size;
    }
    return it->second;
  }
};

// -------------------------------------------------------------------------
// Phase 2 — Decide which MemRefs must be split
// -------------------------------------------------------------------------

bool NeedsAscend910BSplitLoadTpopHazardWorkaround(const FunctionPtr& func) {
  if (backend::GetBackendType() != backend::BackendType::Ascend910B ||
      func->func_type_ != FunctionType::AIV) {
    return false;
  }

  const auto split_mode = func->GetSplitMode();
  return split_mode.has_value() && *split_mode != SplitMode::None;
}

std::unordered_set<const Var*> CollectLoadFamilyVars(const MemRefUsageInfo& info) {
  std::unordered_set<const Var*> load_family;
  std::vector<const Var*> worklist;
  for (const auto& writer : info.writers) {
    if (writer.op_name != "tile.load") continue;
    load_family.insert(writer.var);
    worklist.push_back(writer.var);
  }
  for (size_t i = 0; i < worklist.size(); ++i) {
    const Var* source = worklist[i];
    for (const auto& [view_source, view_user] : info.view_edges) {
      if (view_source != source || load_family.count(view_user) != 0) continue;
      load_family.insert(view_user);
      worklist.push_back(view_user);
    }
  }
  return load_family;
}

std::unordered_set<size_t> CollectForcedSplitWriterIndices(
    const MemRefUsageInfo& info, const std::unordered_set<const Var*>& tpop_from_aic_vars,
    bool enable_ascend910b_split_workaround) {
  std::unordered_set<size_t> forced_indices;
  if (!enable_ascend910b_split_workaround) return forced_indices;

  const auto load_family = CollectLoadFamilyVars(info);
  if (load_family.empty()) return forced_indices;

  for (size_t i = 0; i < info.writers.size(); ++i) {
    const auto& writer = info.writers[i];
    if (writer.op_name.empty() || writer.op_name == "tile.load") continue;

    bool uses_shared_load = false;
    bool uses_tpop = false;
    for (const Var* input_var : writer.input_vars) {
      uses_shared_load = uses_shared_load || load_family.count(input_var) != 0;
      uses_tpop = uses_tpop || tpop_from_aic_vars.count(input_var) != 0;
    }
    if (uses_shared_load && uses_tpop) {
      forced_indices.insert(i);
    }
  }
  return forced_indices;
}

/// For each MemRef that has multiple writers with incompatible signatures,
/// collect the set of Var* that need a fresh MemRef.
///
/// Strategy: the first writer keeps the original MemRef. Every subsequent
/// writer that is not PTO-materializable from the first writer's signature,
/// or that matches an Ascend910B split-AIV load+tpop hazard pattern, gets a
/// new MemRef.
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

std::map<const Var*, MemRefPtr> PlanMemRefSplits(const std::map<const Var*, MemRefUsageInfo>& usages,
                                                 const std::unordered_set<const Var*>& tpop_from_aic_vars,
                                                 bool enable_ascend910b_split_workaround, uint64_t& next_id) {
  std::map<const Var*, MemRefPtr> splits;

  for (const auto& [base_ptr, info] : usages) {
    if (info.writers.size() <= 1) continue;

    const auto& ref_sig = info.writers[0].signature;
    const auto forced_split_indices =
        CollectForcedSplitWriterIndices(info, tpop_from_aic_vars, enable_ascend910b_split_workaround);

    bool needs_split = false;
    for (size_t i = 1; i < info.writers.size(); ++i) {
      if (forced_split_indices.count(i) != 0 || !ref_sig.IsPTOMaterializable(info.writers[i].signature)) {
        needs_split = true;
        break;
      }
    }
    if (!needs_split) continue;

    // Group writers by materializable-compatibility; first group keeps original MemRef
    std::vector<int> group_ids(info.writers.size(), -1);
    std::vector<TileBufSignature> group_reps{info.writers[0].signature};
    group_ids[0] = 0;

    for (size_t i = 1; i < info.writers.size(); ++i) {
      if (forced_split_indices.count(i) != 0) continue;

      const auto& sig = info.writers[i].signature;
      int group_id = -1;
      for (size_t g = 0; g < group_reps.size(); ++g) {
        if (group_reps[g].IsPTOMaterializable(sig)) {
          group_id = static_cast<int>(g);
          break;
        }
      }
      if (group_id < 0) {
        group_id = static_cast<int>(group_reps.size());
        group_reps.push_back(sig);
      }
      group_ids[i] = group_id;
    }

    for (size_t i = 1; i < info.writers.size(); ++i) {
      if (forced_split_indices.count(i) == 0) continue;
      group_ids[i] = static_cast<int>(group_reps.size());
      group_reps.push_back(info.writers[i].signature);
    }

    std::map<int, std::vector<size_t>> sig_groups;
    for (size_t i = 0; i < group_ids.size(); ++i) {
      sig_groups[group_ids[i]].push_back(i);
    }

    // Group 0 keeps original MemRef; groups 1..N get fresh MemRefs
    for (auto& [gid, indices] : sig_groups) {
      if (gid == 0) continue;

      auto memory_space = info.writers[indices[0]].signature.memory_space;
      auto new_base =
          std::make_shared<Var>(BuildBasePtrName(memory_space, next_id++), GetPtrType(), Span::unknown());
      auto new_memref = std::make_shared<MemRef>(new_base, static_cast<int64_t>(0), info.alloc_size);
      std::vector<const Var*> split_roots;

      for (size_t idx : indices) {
        splits[info.writers[idx].var] = new_memref;
        split_roots.push_back(info.writers[idx].var);
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
  // Collect unique new MemRefs (keyed by base_ Ptr identity)
  std::map<const Var*, std::pair<MemRefPtr, MemorySpace>> new_memrefs;
  for (const auto& [var, memref] : splits) {
    if (new_memrefs.count(memref->base_.get()) > 0) continue;
    auto tile_type = As<TileType>(var->GetType());
    MemorySpace space = MemorySpace::Vec;
    if (tile_type) {
      const auto& memory_space = tile_type->memory_space_;
      if (memory_space.has_value()) {
        space = *memory_space;
      }
    }
    new_memrefs[memref->base_.get()] = {memref, space};
  }
  if (new_memrefs.empty()) return body;

  // Build alloc statements
  std::vector<StmtPtr> alloc_stmts;
  for (const auto& [_, pair] : new_memrefs) {
    const auto& [memref, space] = pair;
    alloc_stmts.push_back(CreateAllocStatement(memref, space));
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

/// Find the highest MemRef base name counter in the function (for generating fresh ids).
class MaxMemRefIdCollector : public IRVisitor {
 public:
  void VisitVarLike_(const VarPtr& op) override {
    if (auto tile_type = GetTileTypeWithMemRef(op->GetType())) {
      auto memref = GetDefinedMemRef(tile_type);
      auto counter = ExtractNameCounter(memref->base_->name_hint_);
      if (counter.has_value() && *counter >= max_id_) max_id_ = *counter + 1;
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
  const bool enable_ascend910b_split_workaround = NeedsAscend910BSplitLoadTpopHazardWorkaround(func);

  auto splits =
      PlanMemRefSplits(usages, collector.GetTpopFromAicVars(), enable_ascend910b_split_workaround, next_id);
  if (splits.empty()) return func;

  LOG_DEBUG << "LegalizePTOBufferReuse: splitting " << splits.size() << " variable(s) into new MemRefs";

  // Phase 3: Mutate
  MemRefSplitMutator mutator(splits);
  StmtPtr new_body = mutator.VisitStmt(func->body_);

  // Phase 4: Insert alloc statements for new MemRefs
  new_body = InsertNewAllocStatements(new_body, splits);

  return std::make_shared<const Function>(func->name_, func->params_, func->param_directions_,
                                          func->return_types_, new_body, func->span_, func->func_type_,
                                          func->level_, func->role_, func->attrs_);
}

}  // namespace

namespace pass {

Pass LegalizePTOBufferReuse() {
  static const PassProperties kProps{.required = {IRProperty::HierarchyOutlined, IRProperty::IncoreTileOps,
                                                  IRProperty::HasMemRefs, IRProperty::TileOps2D}};
  return CreateFunctionPass(TransformLegalizePTOBufferReuse, "LegalizePTOBufferReuse", kProps);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
