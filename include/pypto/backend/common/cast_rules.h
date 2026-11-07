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

#ifndef PYPTO_BACKEND_COMMON_CAST_RULES_H_
#define PYPTO_BACKEND_COMMON_CAST_RULES_H_

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/type.h"

namespace pypto::backend {

/// Descriptor preparation selected by the same rule that admits a direct cast.
enum class CastDescriptor { Default, StaticValidShape, StaticPlannedAddress };

/// An alternative implementation of a dtype edge. Missing layout permits the
/// ordinary backend layout-repair pipeline; constrained variants must already
/// satisfy their layouts and can request that those layouts be preserved.
struct TcvtConditions {
  std::optional<ir::TileLayout> layout;
  bool unboxed = false;
  bool null_compact = false;
  bool static_shape = false;
  bool static_valid = false;
  bool matching_valid_shape = false;
  bool full_valid = false;
  bool dense_stride = false;
  bool zero_start_offset = false;
  bool vec_only = false;
  bool preserve_layout = false;
  bool allow_empty_replay = false;
  int64_t source_col_alignment = 1;
  int64_t max_rows = std::numeric_limits<int64_t>::max();
  int64_t max_columns = std::numeric_limits<int64_t>::max();
  int64_t max_byte_span = std::numeric_limits<int64_t>::max();
  int64_t source_register_bytes = 1;
  int64_t max_source_registers = std::numeric_limits<int64_t>::max();
  CastDescriptor descriptor = CastDescriptor::Default;
};

struct TcvtRule {
  DataType source;
  DataType destination;
  // Empty means an unconstrained native implementation; IR validity is checked upstream.
  // Alternatives describe e.g. ND and dense DN implementations of one edge.
  std::vector<TcvtConditions> alternatives;
};

/// Optional diagnostics and emission requirements from IsDirectCast. Queries
/// never mutate IR; unknown extents are accepted only by unconstrained rules.
struct DirectCastInfo {
  std::string reason;
  CastDescriptor descriptor = CastDescriptor::Default;
  bool preserve_layout = false;
  bool empty_replay = false;
};

/// Single source of truth for native dtype edges and implementation conditions.
/// BFS uses these edges, then queries each concrete candidate hop; codegen and
/// layout passes use the identical query on their current IR.
struct TcvtAdjacency {
  std::vector<TcvtRule> edges;
};

/// Query native implementation conditions for a well-typed tile.cast.
/// IR construction and TypeCheck own attribute/type validity. The caller
/// supplies its target's table; this query never reads the global backend,
/// mutates IR, or validates post-allocation address overlap.
[[nodiscard]] bool IsDirectCast(const ir::CallPtr& call, const TcvtAdjacency& rules,
                                DirectCastInfo* info = nullptr);

}  // namespace pypto::backend

#endif  // PYPTO_BACKEND_COMMON_CAST_RULES_H_
