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

/// Conditions one endpoint (the source or the result tile) must satisfy.
///
/// Held separately per endpoint rather than shared, because the asymmetric
/// implementations need it: the A5 packed FP4 decode reads an ND source and may
/// write either an ND or an NZ destination, so a single layout field could not
/// describe the NZ form at all. An endpoint whose fields are all at their
/// defaults imposes nothing.
struct TcvtEndpointConditions {
  /// Required block layout; nullopt permits the ordinary layout-repair pipeline.
  std::optional<ir::TileLayout> blayout;
  /// Required scatter layout. Combined with `blayout`, this is how a boxed
  /// (NZ / ZN) endpoint is spelled: NZ is col_major/row_major (see
  /// tile_view_semantics::GetImplicitTileLayout for the space->layout table).
  std::optional<ir::TileLayout> slayout;
  /// Required boxing granularity in bytes; only meaningful for a boxed endpoint.
  std::optional<uint64_t> fractal;
  bool null_compact = false;
  bool static_shape = false;
  bool static_valid = false;
  bool full_valid = false;
  bool dense_stride = false;
  bool zero_start_offset = false;
  bool vec_only = false;
  /// Physical row-count granularity. A boxed endpoint whose rows are not a whole
  /// number of boxes has no addressable fractal grid.
  int64_t row_alignment = 1;
  int64_t max_rows = std::numeric_limits<int64_t>::max();
  int64_t max_columns = std::numeric_limits<int64_t>::max();
  int64_t max_byte_span = std::numeric_limits<int64_t>::max();
};

/// An alternative implementation of a dtype edge. An endpoint with no layout
/// constraint permits the ordinary backend layout-repair pipeline; constrained
/// variants must already satisfy their layouts and can request that those
/// layouts be preserved.
struct TcvtConditions {
  TcvtEndpointConditions source;
  TcvtEndpointConditions result;
  bool matching_valid_shape = false;
  /// Suppress the backend's row-major *operand* repair for this implementation.
  bool preserve_input_layout = false;
  /// Suppress the backend's row-major *result* repair. Set by an implementation
  /// that writes a non-row-major destination natively, so that layout
  /// resolution must not insert a repair tile.move around it.
  bool preserve_output_layout = false;
  bool allow_empty_replay = false;
  int64_t source_col_alignment = 1;
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
  bool preserve_input_layout = false;
  bool preserve_output_layout = false;
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
