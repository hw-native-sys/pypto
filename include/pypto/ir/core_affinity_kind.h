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

#ifndef PYPTO_IR_CORE_AFFINITY_KIND_H_
#define PYPTO_IR_CORE_AFFINITY_KIND_H_

namespace pypto {
namespace ir {
namespace core_affinity {

/// Execution-core classification for a Call or a compound statement.
///
/// CUBE / VECTOR: runs on AIC / AIV side.
/// SHARED:        runs on both cores (pure declaration, no work — e.g. tile.create).
/// MIXED:         either a compound stmt whose children span both cores, OR
///                a leaf tile.move that crosses the C/V boundary in either
///                direction (Vec↔Acc/Mat/Left/Right).
///
/// Cross-core data motion — whether a single tile.move is actually a boundary
/// that needs tpush/tpop splitting — is tracked *separately* via the
/// boundary_moves map rather than as a distinct enum value, because the
/// boundary status is derivable from ClassifyMoveDirection at call sites
/// that need it (CollectCVBoundaryMoves, BuildCoreBody).
enum class CoreAffinity { CUBE, VECTOR, SHARED, MIXED };

/// Cross-core communication role. Set via OpRegistryEntry::set_cross_core_role().
/// Used by op_predicates (IsTPop, IsInitializePipe, ...) so passes do not
/// have to string-compare on specific op names.
enum class CrossCoreRole { TPush, TPop, TFree, InitializePipe };

inline CoreAffinity CombineAffinity(CoreAffinity a, CoreAffinity b) {
  if (a == b) return a;
  if (a == CoreAffinity::SHARED) return b;
  if (b == CoreAffinity::SHARED) return a;
  return CoreAffinity::MIXED;
}

}  // namespace core_affinity
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_CORE_AFFINITY_KIND_H_
