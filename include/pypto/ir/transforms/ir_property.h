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

#ifndef PYPTO_IR_TRANSFORMS_IR_PROPERTY_H_
#define PYPTO_IR_TRANSFORMS_IR_PROPERTY_H_

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace pypto {
namespace ir {

/**
 * @brief Enumeration of verifiable IR properties
 *
 * Each value represents a property that the IR may or may not satisfy.
 * Passes can declare which properties they require, produce, and invalidate.
 * Not all passes produce properties — performance optimization passes
 * (BasicMemoryReuse, InsertSync, AddAlloc) only have requirements but
 * don't produce new verifiable properties. This is by design.
 */
enum class IRProperty : uint64_t {
  SSAForm = 0,              ///< IR is in SSA form
  TypeChecked,              ///< IR has passed type checking
  NoNestedCalls,            ///< No nested call expressions
  NormalizedStmtStructure,  ///< Statement structure normalized
  FlattenedSingleStmt,      ///< Single-statement blocks flattened
  SplitIncoreOrch,          ///< InCore scopes outlined into separate functions
  HasMemRefs,               ///< MemRef objects initialized on variables
  kCount                    ///< Sentinel (must be last)
};

static_assert(
    static_cast<uint64_t>(IRProperty::kCount) <= 64,
    "IRProperty count exceeds 64, which is the maximum supported by IRPropertySet's uint64_t bitset");

/**
 * @brief Convert an IRProperty to its string name
 */
std::string IRPropertyToString(IRProperty prop);

/**
 * @brief A set of IR properties backed by a uint64_t bitset
 *
 * Efficient O(1) insert/remove/contains operations. Supports up to 64 properties.
 */
class IRPropertySet {
 public:
  IRPropertySet() : bits_(0) {}

  /**
   * @brief Construct from a list of properties
   */
  IRPropertySet(std::initializer_list<IRProperty> props) : bits_(0) {
    for (auto p : props) {
      Insert(p);
    }
  }

  /**
   * @brief Insert a property into the set
   */
  void Insert(IRProperty prop) { bits_ |= Bit(prop); }

  /**
   * @brief Remove a property from the set
   */
  void Remove(IRProperty prop) { bits_ &= ~Bit(prop); }

  /**
   * @brief Check if the set contains a property
   */
  [[nodiscard]] bool Contains(IRProperty prop) const { return (bits_ & Bit(prop)) != 0; }

  /**
   * @brief Check if this set contains all properties in another set
   */
  [[nodiscard]] bool ContainsAll(const IRPropertySet& other) const {
    return (bits_ & other.bits_) == other.bits_;
  }

  /**
   * @brief Return the union of this set and another
   */
  [[nodiscard]] IRPropertySet Union(const IRPropertySet& other) const {
    IRPropertySet result;
    result.bits_ = bits_ | other.bits_;
    return result;
  }

  /**
   * @brief Return the intersection of this set and another
   */
  [[nodiscard]] IRPropertySet Intersection(const IRPropertySet& other) const {
    IRPropertySet result;
    result.bits_ = bits_ & other.bits_;
    return result;
  }

  /**
   * @brief Return this set minus another (set difference)
   */
  [[nodiscard]] IRPropertySet Difference(const IRPropertySet& other) const {
    IRPropertySet result;
    result.bits_ = bits_ & ~other.bits_;
    return result;
  }

  /**
   * @brief Check if the set is empty
   */
  [[nodiscard]] bool Empty() const { return bits_ == 0; }

  /**
   * @brief Convert to a vector of the contained properties
   */
  [[nodiscard]] std::vector<IRProperty> ToVector() const;

  /**
   * @brief Convert to a human-readable string (e.g., "{SSAForm, TypeChecked}")
   */
  [[nodiscard]] std::string ToString() const;

  bool operator==(const IRPropertySet& other) const { return bits_ == other.bits_; }
  bool operator!=(const IRPropertySet& other) const { return bits_ != other.bits_; }

 private:
  uint64_t bits_;

  static uint64_t Bit(IRProperty prop) { return uint64_t{1} << static_cast<uint64_t>(prop); }
};

/**
 * @brief Property declarations for a pass
 *
 * Used with CreateFunctionPass/CreateProgramPass to declare pass requirements.
 */
struct PassProperties {
  IRPropertySet required;     ///< Preconditions: must hold before the pass runs
  IRPropertySet produced;     ///< New properties guaranteed after running
  IRPropertySet invalidated;  ///< Properties this pass breaks
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_IR_PROPERTY_H_
