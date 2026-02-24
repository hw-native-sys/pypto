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
 * @file dtype.h
 * @brief Data type definitions for PyPTO tensors and operations
 *
 * This file defines the DataType class which represents all supported numeric types
 * in the PyPTO framework, including integers, unsigned integers, floating point,
 * bfloat16, and Hisilicon float formats.
 */

#ifndef PYPTO_CORE_DTYPE_H_
#define PYPTO_CORE_DTYPE_H_

#include <cstddef>
#include <cstdint>
#include <string>

namespace pypto {

/**
 * @brief Data type representation for PyPTO
 *
 * This class encapsulates all numeric data types supported by PyPTO tensors and operations.
 * It includes:
 * - Signed integers: INT4, INT8, INT16, INT32, INT64
 * - Unsigned integers: UINT4, UINT8, UINT16, UINT32, UINT64
 * - Floating point: FP4, FP8, FP16, FP32
 * - Brain floating point: BF16
 * - Hisilicon float formats: HF4, HF8
 * - Boolean: BOOL
 */
class DataType {
 public:
  // Type code constants
  // Organized by category with gaps for future extension

  // Boolean types: 0x00-0x0F
  static constexpr uint8_t kBoolCode = 0x00;

  // Signed integer types: 0x10-0x1F (16 slots reserved)
  static constexpr uint8_t kSignedIntRangeStart = 0x10;
  static constexpr uint8_t kInt4Code = 0x10;
  static constexpr uint8_t kInt8Code = 0x11;
  static constexpr uint8_t kInt16Code = 0x12;
  static constexpr uint8_t kInt32Code = 0x13;
  static constexpr uint8_t kInt64Code = 0x14;
  static constexpr uint8_t kSignedIntRangeEnd = 0x1F;
  // 0x15-0x1F reserved for future signed integer types

  // Unsigned integer types: 0x20-0x2F (16 slots reserved)
  static constexpr uint8_t kUnsignedIntRangeStart = 0x20;
  static constexpr uint8_t kUInt4Code = 0x20;
  static constexpr uint8_t kUInt8Code = 0x21;
  static constexpr uint8_t kUInt16Code = 0x22;
  static constexpr uint8_t kUInt32Code = 0x23;
  static constexpr uint8_t kUInt64Code = 0x24;
  static constexpr uint8_t kUnsignedIntRangeEnd = 0x2F;
  // 0x25-0x2F reserved for future unsigned integer types

  // IEEE floating point types: 0x30-0x3F (16 slots reserved)
  static constexpr uint8_t kIeeeFloatRangeStart = 0x30;
  static constexpr uint8_t kFp4Code = 0x30;
  static constexpr uint8_t kFp8e4m3fnCode = 0x31;
  static constexpr uint8_t kFp8e5m2Code = 0x32;
  static constexpr uint8_t kFp16Code = 0x33;
  static constexpr uint8_t kFp32Code = 0x34;
  static constexpr uint8_t kFp64Code = 0x35;  // Reserved for future FP64 support
  static constexpr uint8_t kIeeeFloatRangeEnd = 0x3F;
  // 0x36-0x3F reserved for future IEEE float types

  // Brain/Hisilicon float types: 0x40-0x4F (16 slots reserved)
  static constexpr uint8_t kBrainFloatRangeStart = 0x40;
  static constexpr uint8_t kBf16Code = 0x40;
  static constexpr uint8_t kHf4Code = 0x41;
  static constexpr uint8_t kHf8Code = 0x42;
  static constexpr uint8_t kBrainFloatRangeEnd = 0x4F;
  // 0x43-0x4F reserved for future brain/Hisilicon float types

  // Static constants for all data types
  static const DataType BOOL;       // Boolean (true/false)
  static const DataType INT4;       // 4-bit signed integer
  static const DataType INT8;       // 8-bit signed integer
  static const DataType INT16;      // 16-bit signed integer
  static const DataType INT32;      // 32-bit signed integer
  static const DataType INT64;      // 64-bit signed integer
  static const DataType UINT4;      // 4-bit unsigned integer
  static const DataType UINT8;      // 8-bit unsigned integer
  static const DataType UINT16;     // 16-bit unsigned integer
  static const DataType UINT32;     // 32-bit unsigned integer
  static const DataType UINT64;     // 64-bit unsigned integer
  static const DataType FP4;        // 4-bit floating point
  static const DataType FP8E4M3FN;  // 8-bit floating point (IEEE 754 e4m3fn format)
  static const DataType FP8E5M2;    // 8-bit floating point (IEEE 754 e5m2 format)
  static const DataType FP16;       // 16-bit floating point (IEEE 754 half precision)
  static const DataType FP32;       // 32-bit floating point (IEEE 754 single precision)
  static const DataType BF16;       // 16-bit brain floating point
  static const DataType HF4;        // 4-bit Hisilicon float
  static const DataType HF8;        // 8-bit Hisilicon float

  // Semantic alias for index computations (loop variables, dimensions, offsets, strides)
  static const DataType INDEX;  // Alias for INT64

  // Default dtypes for bare constant literals (used by printer/parser for round-trip)
  static const DataType DEFAULT_CONST_INT;    // Default dtype for ConstInt (= INDEX)
  static const DataType DEFAULT_CONST_FLOAT;  // Default dtype for ConstFloat (= FP32)

  /**
   * @brief Default constructor, initializes to BOOL type
   */
  constexpr DataType() : code_(kBoolCode) {}

  /**
   * @brief Construct from type code
   * @param code The type code
   */
  constexpr explicit DataType(uint8_t code) : code_(code) {}

  /**
   * @brief Get the size in bits of this data type
   *
   * Returns the storage size in bits for each data type. This accurately
   * represents sub-byte types like INT4, UINT4, FP4, and HF4.
   *
   * @return Size in bits
   */
  [[nodiscard]] size_t GetBit() const {
    switch (code_) {
      case kBoolCode:
        return 1;
      case kHf4Code:
      case kFp4Code:
      case kUInt4Code:
      case kInt4Code:
        return 4;
      case kHf8Code:
      case kFp8e4m3fnCode:
      case kFp8e5m2Code:
      case kUInt8Code:
      case kInt8Code:
        return 8;
      case kBf16Code:
      case kFp16Code:
      case kUInt16Code:
      case kInt16Code:
        return 16;
      case kFp32Code:
      case kUInt32Code:
      case kInt32Code:
        return 32;
      case kUInt64Code:
      case kInt64Code:
        return 64;
      default:
        return 0;
    }
  }

  /**
   * @brief Get a human-readable string name for this data type
   *
   * @return String representation of the data type
   */
  [[nodiscard]] std::string ToString() const {
    switch (code_) {
      case kInt4Code:
        return "int4";
      case kInt8Code:
        return "int8";
      case kInt16Code:
        return "int16";
      case kInt32Code:
        return "int32";
      case kInt64Code:
        return "int64";
      case kUInt4Code:
        return "uint4";
      case kUInt8Code:
        return "uint8";
      case kUInt16Code:
        return "uint16";
      case kUInt32Code:
        return "uint32";
      case kUInt64Code:
        return "uint64";
      case kFp4Code:
        return "fp4";
      case kFp8e4m3fnCode:
        return "fp8e4m3fn";
      case kFp8e5m2Code:
        return "fp8e5m2";
      case kFp16Code:
        return "fp16";
      case kFp32Code:
        return "fp32";
      case kBf16Code:
        return "bfloat16";
      case kHf4Code:
        return "hf4";
      case kHf8Code:
        return "hf8";
      case kBoolCode:
        return "bool";
      default:
        return "unknown";
    }
  }

  /**
   * @brief Get C style type string for code generation
   *
   * Returns the C/C++ type string representation used in code generation.
   * Covers all DataType variants: signed/unsigned integers (incl. INT4/UINT4),
   * IEEE float (FP16, FP32, FP64), FP4/FP8, BF16, HF4/HF8, and BOOL.
   *
   * @return C style type string (e.g. "float", "int32_t", "half", "bfloat16")
   */
  [[nodiscard]] std::string ToCTypeString() const {
    switch (code_) {
      case kBoolCode:
        return "bool";
      case kInt8Code:
        return "int8_t";
      case kInt16Code:
        return "int16_t";
      case kInt32Code:
        return "int32_t";
      case kInt64Code:
        return "int64_t";
      case kUInt8Code:
        return "uint8_t";
      case kUInt16Code:
        return "uint16_t";
      case kUInt32Code:
        return "uint32_t";
      case kUInt64Code:
        return "uint64_t";
      case kFp16Code:
        return "half";
      case kFp32Code:
        return "float";
      case kFp64Code:
        return "double";
      case kBf16Code:
        return "bfloat16";
      default:
        return "unknown";
    }
  }

  /**
   * @brief Check if this data type is a floating point type
   *
   * @return true if this is FP4, FP8, FP16, FP32, BF16, HF4, or HF8
   */
  [[nodiscard]] bool IsFloat() const {
    // IEEE float types or Brain/Hisilicon float types
    return (code_ >= kIeeeFloatRangeStart && code_ <= kIeeeFloatRangeEnd) ||
           (code_ >= kBrainFloatRangeStart && code_ <= kBrainFloatRangeEnd);
  }

  /**
   * @brief Check if this data type is a signed integer type
   *
   * @return true if this is INT4, INT8, INT16, INT32, or INT64
   */
  [[nodiscard]] bool IsSignedInt() const {
    return code_ >= kSignedIntRangeStart && code_ <= kSignedIntRangeEnd;
  }

  /**
   * @brief Check if this data type is an unsigned integer type
   *
   * @return true if this is UINT4, UINT8, UINT16, UINT32, or UINT64
   */
  [[nodiscard]] bool IsUnsignedInt() const {
    return code_ >= kUnsignedIntRangeStart && code_ <= kUnsignedIntRangeEnd;
  }

  /**
   * @brief Check if this data type is any integer type (signed or unsigned)
   *
   * @return true if this is any integer type
   */
  [[nodiscard]] bool IsInt() const { return IsSignedInt() || IsUnsignedInt(); }

  /**
   * @brief Equality comparison operator
   *
   * @param other The other DataType to compare with
   * @return true if both types have the same code
   */
  constexpr bool operator==(const DataType& other) const { return code_ == other.code_; }

  /**
   * @brief Inequality comparison operator
   *
   * @param other The other DataType to compare with
   * @return true if types have different codes
   */
  constexpr bool operator!=(const DataType& other) const { return code_ != other.code_; }

  /**
   * @brief Get the underlying type code
   *
   * @return The uint8_t code representing this type
   */
  [[nodiscard]] constexpr uint8_t Code() const { return code_; }

 private:
  uint8_t code_;  // Internal type code
};

// Static constant definitions
inline constexpr DataType DataType::BOOL = DataType(kBoolCode);
inline constexpr DataType DataType::INT4 = DataType(kInt4Code);
inline constexpr DataType DataType::INT8 = DataType(kInt8Code);
inline constexpr DataType DataType::INT16 = DataType(kInt16Code);
inline constexpr DataType DataType::INT32 = DataType(kInt32Code);
inline constexpr DataType DataType::INT64 = DataType(kInt64Code);
inline constexpr DataType DataType::UINT4 = DataType(kUInt4Code);
inline constexpr DataType DataType::UINT8 = DataType(kUInt8Code);
inline constexpr DataType DataType::UINT16 = DataType(kUInt16Code);
inline constexpr DataType DataType::UINT32 = DataType(kUInt32Code);
inline constexpr DataType DataType::UINT64 = DataType(kUInt64Code);
inline constexpr DataType DataType::FP4 = DataType(kFp4Code);
inline constexpr DataType DataType::FP8E4M3FN = DataType(kFp8e4m3fnCode);
inline constexpr DataType DataType::FP8E5M2 = DataType(kFp8e5m2Code);
inline constexpr DataType DataType::FP16 = DataType(kFp16Code);
inline constexpr DataType DataType::FP32 = DataType(kFp32Code);
inline constexpr DataType DataType::BF16 = DataType(kBf16Code);
inline constexpr DataType DataType::HF4 = DataType(kHf4Code);
inline constexpr DataType DataType::HF8 = DataType(kHf8Code);
inline constexpr DataType DataType::INDEX = DataType(kInt64Code);
inline constexpr DataType DataType::DEFAULT_CONST_INT = DataType(kInt64Code);   // = INDEX
inline constexpr DataType DataType::DEFAULT_CONST_FLOAT = DataType(kFp32Code);  // = FP32

}  // namespace pypto

#endif  // PYPTO_CORE_DTYPE_H_
