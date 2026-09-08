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

#ifndef PYPTO_IR_CAST_SATURATION_H_
#define PYPTO_IR_CAST_SATURATION_H_

#include <any>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"

namespace pypto {
namespace ir {

// Destination saturation for `tensor.cast` / `tile.cast`, carried as the
// optional integer `saturation_mode` kwarg and lowered to PTOAS's
// `satmode = #pto<saturation_mode ON|OFF>` on `pto.tcvt`.
//
// kOn clamps a rounded result that falls outside the destination range to that
// range. kOff selects the target's non-saturating conversion, whose overflow
// and non-finite behavior is architecture-defined. The two agree only for inputs
// the destination can already represent.
//
// The integer values are part of the IR ABI and match the DSL's "off"/"on".
enum class SaturationMode : int {
  kOff = 0,
  kOn = 1,
};

/// What a cast means when it carries no `saturation_mode` kwarg, given where it
/// is converting *to*. `nullopt` means "whatever the target does" -- no `satmode`
/// is emitted and the lowering is exactly what it was before this kwarg existed.
///
/// The default is only ON for an **integer** destination. That is where the two
/// modes are a genuine choice: no standard fixes what a float-to-int or a
/// narrowing int-to-int overflow produces, clamping is the safer of the two to
/// get by accident, and on A2/A3 it is also the conversion the assembler
/// performs natively rather than emulating with a chunked vector sequence.
///
/// A **float** destination is a different question with an existing answer: IEEE
/// says an out-of-range narrowing yields an infinity, `torch` agrees, and
/// `docs/en/user/precision/00-workflow.md` asserts PyPTO matches them
/// bit-for-bit on `INT32 -> FP16`. Defaulting those to ON broke that block on
/// the a2a3 simulator (65520 clamped to 65504 instead of overflowing to inf), so
/// float destinations keep the target's own behavior unless the author asks
/// otherwise.
inline std::optional<SaturationMode> DefaultSaturationModeFor(DataType dst) {
  if (!dst.IsInt()) return std::nullopt;
  return SaturationMode::kOn;
}

/// The effective mode of a cast, reading the destination's default through for an
/// absent kwarg. `nullopt` means no `satmode` should be emitted.
inline std::optional<int> GetSaturationMode(const CallPtr& call) {
  constexpr int kAbsent = -1;
  const int explicit_mode = call->GetKwarg<int>("saturation_mode", kAbsent);
  if (explicit_mode != kAbsent) return explicit_mode;
  const auto fallback = DefaultSaturationModeFor(call->GetKwarg<DataType>("target_type"));
  if (!fallback.has_value()) return std::nullopt;
  return static_cast<int>(*fallback);
}

inline bool IsValidSaturationMode(int value) {
  return value == static_cast<int>(SaturationMode::kOff) || value == static_cast<int>(SaturationMode::kOn);
}

/// The DSL spelling ("off" / "on"), for round-trippable IR printing.
inline std::string SaturationModeToName(int value) {
  if (!IsValidSaturationMode(value)) {
    throw pypto::TypeError("Unknown SaturationMode: " + std::to_string(value));
  }
  return value == static_cast<int>(SaturationMode::kOn) ? "on" : "off";
}

/// The PTOAS enum member ("OFF" / "ON") used in the emitted `satmode` attribute.
inline std::string SaturationModeToPTOString(int value) {
  if (!IsValidSaturationMode(value)) {
    throw pypto::TypeError("Unknown SaturationMode: " + std::to_string(value));
  }
  return value == static_cast<int>(SaturationMode::kOn) ? "ON" : "OFF";
}

/// Reject an out-of-contract `saturation_mode` at construction time, so a bad
/// value is reported against the cast the caller wrote rather than surfacing as
/// an unrenderable PTOAS attribute in codegen. A missing kwarg is valid — it
/// means "leave the backend default alone" — so this only checks what is there.
inline void ValidateCastSaturationModeKwarg(const std::vector<std::pair<std::string, std::any>>& kwargs,
                                            const std::string& op_name) {
  for (const auto& [key, value] : kwargs) {
    if (key != "saturation_mode") continue;
    const int saturation_mode = AnyCast<int>(value, "kwarg key: saturation_mode");
    CHECK(IsValidSaturationMode(saturation_mode))
        << op_name << ": saturation_mode must be off(" << static_cast<int>(SaturationMode::kOff) << ") or on("
        << static_cast<int>(SaturationMode::kOn) << "), got " << saturation_mode;
    return;
  }
}

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_CAST_SATURATION_H_
