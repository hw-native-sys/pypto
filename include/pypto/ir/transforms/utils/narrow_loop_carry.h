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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_NARROW_LOOP_CARRY_H_
#define PYPTO_IR_TRANSFORMS_UTILS_NARROW_LOOP_CARRY_H_

#include "pypto/ir/function.h"

namespace pypto {
namespace ir {
namespace narrow_loop_carry {

/**
 * @brief Re-declare each Acc (L0C) loop carry at the valid extent its yields prove
 *
 * A loop carry is typed from its init value alone: ``ConvertToSSA`` mints the ``IterArg``
 * from the reaching definition before the loop, ``ConvertTensorToTileOps`` re-mints it
 * from the converted seed, and both force the loop's ``return_var`` to that same type.
 * The yields are never consulted.
 *
 * That is a lie whenever the body yields a narrower valid extent than the seed declares:
 * from the second iteration on the carried value *is* the yield. For an accumulator it is
 * also a miscompile -- ``mad`` lays its product out in L0C at an N-fractal stride of
 * ``ceil(validRow/16)*16`` taken from the L0A operand's valid rows, while a reader that
 * believes the seed's full height walks the buffer at the physical row pitch, scrambling
 * every N-fractal above the first (issue #2470).
 *
 * Whoever narrows a matmul result inside a carry therefore has to repair the carry before
 * it returns, or it publishes IR its own ``TypeCheck`` and ``AccCompactValid`` verifiers
 * reject. Two passes do: ``ConvertTensorToTileOps`` (a 2D seed, narrowed the moment
 * ``tensor.matmul`` becomes ``tile.matmul``) and ``FlattenTileNdTo2D`` (an ND seed, narrowed
 * when ``tile.batch_matmul`` is unrolled into 2D matmuls). Both call this.
 *
 * The repair re-declares the *seed* -- the only place the rest of the pipeline reads a
 * carry's type from -- through the same ``tile.create(compact=...)`` plus
 * ``tile.set_validshape`` form ``AutoTileMatmulL0`` builds when it splits K, and re-types
 * the body's def-use closure through the operators' own deducers.
 *
 * Deliberately narrow in scope:
 *   * only **Acc** carries, where a stale extent changes the stride a reader uses. A Vec
 *     seed may hold bytes the first iteration is entitled to read at full height.
 *   * only a seed defined by ``tile.create``; anything else may carry bytes whose layout
 *     this repair must not re-interpret.
 *   * only a dimension whose yield extent is adoptable: either provably ``<=`` the extent
 *     the init declares, or declared against an init that still fills its physical box --
 *     every ``valid_shape`` is bounded by that box, so a dynamic yield extent is already
 *     trusted to fit inside it. An init that is *itself* already narrowed is never widened
 *     on an undecidable relation.
 *   * only where the two readings of the buffer would actually disagree
 *     (``AccPitchesCoincide``, shared with the ``AccCompactValid`` verifier). A
 *     single-fractal-block box packs to its physical rows whatever its valid rows, so a
 *     ``[16, N]`` accumulator keeps the exact form it has today.
 *   * only where the narrowed extents are visible *before* the loop. The re-declared seed
 *     sits there, and the common spelling puts the row count next to the slice it bounds,
 *     inside the body -- hoisting that would leave codegen with a symbol it cannot bind.
 *     Such a carry is declined; where its pitches genuinely differ, ``AccCompactValid``
 *     then reports it as a compile error rather than letting it corrupt data.
 *
 * Cost is O(N log N) in the size of the function: three linear sweeps -- a scope index, a
 * decision pass over types, and one top-down rewrite -- over ordered-map lookups. The
 * rewrite settles each loop's carries *before* visiting its body, so one visit types the
 * body against the narrowed carry; deciding afterwards would re-type it a second time,
 * and a nested carry would compound that per level.
 *
 * One round: a carry whose yields only narrow *because* an inner carry was repaired is
 * decided against the types as they were, and is left alone. Nothing miscompiles -- an
 * unrepaired pitch disagreement is what ``AccCompactValid`` rejects.
 *
 * @param func Function to repair (any type; only loop carries inside it are touched)
 * @return The repaired function, or @p func itself when nothing narrows
 */
FunctionPtr NarrowAccCarries(const FunctionPtr& func);

}  // namespace narrow_loop_carry
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_NARROW_LOOP_CARRY_H_
