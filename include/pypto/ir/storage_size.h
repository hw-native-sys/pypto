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

#ifndef PYPTO_IR_STORAGE_SIZE_H_
#define PYPTO_IR_STORAGE_SIZE_H_

#include <cstdint>
#include <limits>
#include <optional>

#include "pypto/core/dtype.h"

namespace pypto::ir::storage_size {

/// Physical storage width of one logical element.
///
/// Every semantic 4-bit dtype uses packed nibble storage. Other sub-byte
/// dtypes, notably BOOL, remain byte-addressable to preserve their existing
/// buffer ABI.
inline uint64_t GetStorageBitWidth(const DataType& dtype) {
  return dtype.GetBit() == 4 ? 4 : static_cast<uint64_t>(dtype.GetByte()) * 8;
}

/// Bytes occupied by ``logical_elements`` contiguous logical elements.
/// Invalid dtypes and multiplication overflow return nullopt.
inline std::optional<uint64_t> StaticStorageBytes(uint64_t logical_elements, const DataType& dtype) {
  const uint64_t storage_bits = GetStorageBitWidth(dtype);
  if (storage_bits == 0 || logical_elements > std::numeric_limits<uint64_t>::max() / storage_bits) {
    return std::nullopt;
  }
  const uint64_t total_bits = logical_elements * storage_bits;
  return total_bits / 8 + static_cast<uint64_t>(total_bits % 8 != 0);
}

/// Byte offset of a statically known logical element offset.
///
/// A packed-nibble origin cannot be represented by MemRef's byte_offset field,
/// so odd nibble offsets return nullopt rather than being rounded.
inline std::optional<uint64_t> StaticLogicalOffsetToByte(uint64_t logical_offset, const DataType& dtype) {
  const uint64_t storage_bits = GetStorageBitWidth(dtype);
  if (storage_bits == 0 || logical_offset > std::numeric_limits<uint64_t>::max() / storage_bits) {
    return std::nullopt;
  }
  const uint64_t bit_offset = logical_offset * storage_bits;
  if (bit_offset % 8 != 0) return std::nullopt;
  return bit_offset / 8;
}

}  // namespace pypto::ir::storage_size

#endif  // PYPTO_IR_STORAGE_SIZE_H_
