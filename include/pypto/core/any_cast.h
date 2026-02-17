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
 * @file any_cast.h
 * @brief Enhanced any_cast utilities with better error reporting
 *
 * This header provides wrapper functions around std::any_cast that offer
 * improved error messages when type casting fails. Instead of generic
 * bad_any_cast exceptions, these utilities provide detailed information
 * about expected vs. actual types using demangled type names.
 */

#ifndef PYPTO_CORE_ANY_CAST_H_
#define PYPTO_CORE_ANY_CAST_H_

#include <cxxabi.h>

#include <any>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <typeinfo>

#include "pypto/core/error.h"

namespace pypto {

/**
 * @brief Demangle C++ type names for better error messages
 *
 * Converts mangled C++ type names (from typeid().name()) into human-readable
 * format using abi::__cxa_demangle. Also simplifies common pypto types by
 * removing the "pypto::" prefix for brevity.
 *
 * @param mangled_name The mangled type name from typeid().name()
 * @return Human-readable type name string
 *
 * @example
 *   DemangleTypeName(typeid(int).name()) -> "int"
 *   DemangleTypeName(typeid(pypto::DataType).name()) -> "DataType"
 */
inline std::string DemangleTypeName(const char* mangled_name) {
  int status = 0;
  char* demangled = abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status);
  if (status == 0 && demangled) {
    std::string result(demangled);
    free(demangled);

    // Simplify common pypto types for readability
    size_t pos = result.find("pypto::");
    if (pos != std::string::npos) {
      result = result.substr(pos + 7);  // Remove "pypto::" prefix
    }

    return result;
  }
  // If demangling fails, return the original mangled name
  return mangled_name;
}

/**
 * @brief Cast std::any to type T with enhanced error reporting
 *
 * This function wraps std::any_cast and provides detailed error messages
 * when the cast fails, including the expected type, actual type (both
 * demangled), and optional context information.
 *
 * @tparam T The target type to cast to
 * @param value The std::any value to cast
 * @param context Optional context string for error messages (e.g., "kwarg key: out_dtype")
 * @return The value cast to type T
 * @throws TypeError if the cast fails, with detailed type information
 *
 * @example
 *   std::any val = 42;
 *   int x = AnyCast<int>(val, "kwarg key: value");
 *   // If val contains wrong type, throws:
 *   // "Invalid type for kwarg key: value, expected int, but got std::string"
 */
template <typename T>
T AnyCast(const std::any& value, const std::string& context = "") {
  try {
    return std::any_cast<T>(value);
  } catch (const std::bad_any_cast& e) {
    std::string expected_type = DemangleTypeName(typeid(T).name());
    std::string actual_type = DemangleTypeName(value.type().name());
    std::string error_msg = "Invalid type";
    if (!context.empty()) {
      error_msg += " for ";
      error_msg += context;
    }
    error_msg += ", expected ";
    error_msg += expected_type;
    error_msg += ", but got ";
    error_msg += actual_type;
    throw TypeError(error_msg);
  }
}

/**
 * @brief Cast std::any to const reference of type T with enhanced error reporting
 *
 * This function wraps std::any_cast<const T&> and provides detailed error
 * messages when the cast fails. Use this variant when you need a const
 * reference to avoid copying large objects.
 *
 * @tparam T The target type to cast to (will be cast as const T&)
 * @param value The std::any value to cast
 * @param context Optional context string for error messages
 * @return Const reference to the value cast to type T
 * @throws TypeError if the cast fails, with detailed type information
 *
 * @example
 *   std::any val = std::string("hello");
 *   const std::string& str = AnyCastRef<std::string>(val);
 */
template <typename T>
const T& AnyCastRef(const std::any& value, const std::string& context = "") {
  try {
    return std::any_cast<const T&>(value);
  } catch (const std::bad_any_cast& e) {
    std::string expected_type = DemangleTypeName(typeid(T).name());
    std::string actual_type = DemangleTypeName(value.type().name());
    std::string error_msg = "Invalid type";
    if (!context.empty()) {
      error_msg += " for ";
      error_msg += context;
    }
    error_msg += ", expected ";
    error_msg += expected_type;
    error_msg += ", but got ";
    error_msg += actual_type;
    throw TypeError(error_msg);
  }
}

}  // namespace pypto

#endif  // PYPTO_CORE_ANY_CAST_H_
