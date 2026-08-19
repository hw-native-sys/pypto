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

#ifndef PYPTO_IR_PHASE_H_
#define PYPTO_IR_PHASE_H_

#include <string>

#include "pypto/core/error.h"

namespace pypto {
namespace ir {

// AccPhase selects the producer-side unit-flag behavior of phased accumulator
// operations such as tile.gemv. STPhase selects the matching consumer-side
// behavior of tile.store. Keep these as distinct types: the producer and store
// protocols are not interchangeable even where PTO-ISA assigns the same wire
// values.
//
// The integer values are part of the IR ABI and intentionally match PTO-ISA.
// AccPhase exposes the full producer protocol: unspecified = off, partial =
// check-only, and final = check-and-set. PyPTO deliberately limits STPhase to
// unspecified = off and final = check-and-clear. PTO-ISA's check-only store
// phase requires an ordered multi-consumer lifecycle that PyPTO does not expose.
enum class AccPhase : int {
  kUnspecified = 0x0,
  kPartial = 0x2,
  kFinal = 0x3,
};

enum class STPhase : int {
  kUnspecified = 0x0,
  kFinal = 0x3,
};

namespace detail {

inline bool IsValidPhaseValue(int value) { return value == 0x0 || value == 0x2 || value == 0x3; }

inline std::string PhaseValueToPythonMember(int value, const std::string& enum_name) {
  switch (value) {
    case 0x0:
      return "Unspecified";
    case 0x2:
      return "Partial";
    case 0x3:
      return "Final";
    default:
      throw pypto::TypeError("Unknown " + enum_name + ": " + std::to_string(value));
  }
}

inline std::string PhaseValueToPTOString(int value, const std::string& enum_name) {
  switch (value) {
    case 0x0:
      return "unspecified";
    case 0x2:
      return "partial";
    case 0x3:
      return "final";
    default:
      throw pypto::TypeError("Unknown " + enum_name + ": " + std::to_string(value));
  }
}

}  // namespace detail

inline bool IsValidAccPhase(int value) { return detail::IsValidPhaseValue(value); }

inline bool IsValidSTPhase(int value) { return value == 0x0 || value == 0x3; }

inline std::string AccPhaseToString(AccPhase phase) {
  return detail::PhaseValueToPythonMember(static_cast<int>(phase), "AccPhase");
}

inline std::string STPhaseToString(STPhase phase) {
  const int value = static_cast<int>(phase);
  if (!IsValidSTPhase(value)) throw pypto::TypeError("Unknown STPhase: " + std::to_string(value));
  return detail::PhaseValueToPythonMember(value, "STPhase");
}

inline std::string AccPhaseToPTOString(AccPhase phase) {
  return detail::PhaseValueToPTOString(static_cast<int>(phase), "AccPhase");
}

inline std::string STPhaseToPTOString(STPhase phase) {
  const int value = static_cast<int>(phase);
  if (!IsValidSTPhase(value)) throw pypto::TypeError("Unknown STPhase: " + std::to_string(value));
  return detail::PhaseValueToPTOString(value, "STPhase");
}

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_PHASE_H_
