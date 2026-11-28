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

#ifndef PYPTO_CODEGEN_PTO_TILE_STORAGE_PLAN_H_
#define PYPTO_CODEGEN_PTO_TILE_STORAGE_PLAN_H_

#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

namespace pypto::codegen {

struct TileExtents {
  ir::ExprPtr row;
  ir::ExprPtr col;
};

struct TileAllocationOperands {
  TileExtents valid;
  TileExtents physical;
  ir::ExprPtr address;
  // A head declaration cannot reference valid extents computed inside a loop.
  // The original IR column (in carriers for packed Tiles) records the update.
  std::optional<ir::ExprPtr> restore_valid_column;
};

struct BranchTileTransfer {
  ir::VarPtr source;
  bool redirect_producer = false;
  bool copy_if_distinct = false;
  bool borrowed = false;
};

struct TileBranchArm {
  ir::YieldStmtPtr yield;
  std::vector<BranchTileTransfer> transfers;
};

// A mixed-type window can share a handle only if the two prepared operands
// materialize as the same dominating, correctly typed SSA declaration.
struct TileJoinBinding {
  const ir::Var* allocation_owner = nullptr;
  bool use_common_source = false;
};

struct TileBranchPlan {
  size_t position = 0;
  TileBranchArm then_arm;
  TileBranchArm else_arm;
  std::vector<TileJoinBinding> results;
};

struct TileAssignmentBinding {
  ir::ExprPtr writeback_input;
  bool allocate = false;
  bool prepare_destination = false;
  bool same_window_move = false;
};

struct TileStoragePlan {
  // A staged head allocation remains visible throughout the function. An inline
  // declaration expires when its lexical region ends, including sibling arms.
  static constexpr size_t kFunctionEnd = std::numeric_limits<size_t>::max();
  std::map<const ir::Var*, size_t> scope_end;

  std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>> allocations;
  std::map<const ir::Var*, const ir::Var*> allocation_owner;
  std::map<const ir::Var*, const ir::Var*> var_to_memref;
  std::map<const ir::Var*, ir::TileTypePtr> memref_to_tile_type;
  std::map<const ir::TileType*, TileAllocationOperands> operands;
  std::map<const ir::IfStmt*, TileBranchPlan> branches;
  std::map<const ir::AssignStmt*, TileAssignmentBinding> assignments;
  std::set<const ir::Var*> parameter_bases;
  bool has_software_pipeline_slots = false;
};

TileStoragePlan PrepareTileStorage(const ir::StmtPtr& body, const std::vector<ir::VarPtr>& params,
                                   bool emit_tile_addr);

}  // namespace pypto::codegen

#endif  // PYPTO_CODEGEN_PTO_TILE_STORAGE_PLAN_H_
