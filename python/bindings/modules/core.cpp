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

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "../module.h"
#include "pypto/core/dtype.h"

namespace nb = nanobind;

namespace pypto {
namespace python {

void BindCore(nb::module_& m) {
  // Bind DataType class
  nb::class_<DataType>(m, "DataType", "Data type representation for PyPTO tensors and operations")
      // Static type constants
      .def_ro_static("BOOL", &DataType::BOOL, "Boolean (true/false)")
      .def_ro_static("INT4", &DataType::INT4, "4-bit signed integer")
      .def_ro_static("INT8", &DataType::INT8, "8-bit signed integer")
      .def_ro_static("INT16", &DataType::INT16, "16-bit signed integer")
      .def_ro_static("INT32", &DataType::INT32, "32-bit signed integer")
      .def_ro_static("INT64", &DataType::INT64, "64-bit signed integer")
      .def_ro_static("UINT4", &DataType::UINT4, "4-bit unsigned integer")
      .def_ro_static("UINT8", &DataType::UINT8, "8-bit unsigned integer")
      .def_ro_static("UINT16", &DataType::UINT16, "16-bit unsigned integer")
      .def_ro_static("UINT32", &DataType::UINT32, "32-bit unsigned integer")
      .def_ro_static("UINT64", &DataType::UINT64, "64-bit unsigned integer")
      .def_ro_static("FP4", &DataType::FP4, "4-bit floating point")
      .def_ro_static("FP8E4M3FN", &DataType::FP8E4M3FN, "8-bit floating point (E4M3FN format)")
      .def_ro_static("FP8E5M2", &DataType::FP8E5M2, "8-bit floating point (E5M2 format)")
      .def_ro_static("FP16", &DataType::FP16, "16-bit floating point (IEEE 754 half precision)")
      .def_ro_static("FP32", &DataType::FP32, "32-bit floating point (IEEE 754 single precision)")
      .def_ro_static("BF16", &DataType::BF16, "16-bit brain floating point")
      .def_ro_static("HF4", &DataType::HF4, "4-bit Hisilicon float")
      .def_ro_static("HF8", &DataType::HF8, "8-bit Hisilicon float")
      .def_ro_static("INDEX", &DataType::INDEX,
                     "Index type for loop variables, dimensions, offsets (alias for INT64)")
      // Member methods
      .def("get_bit", &DataType::GetBit,
           "Get the size in bits of this data type. Returns the actual bit size for sub-byte types (e.g., 4 "
           "bits "
           "for INT4).")
      .def("to_string", &DataType::ToString, "Get a human-readable string name for this data type.")
      .def("to_c_type_string", &DataType::ToCTypeString,
           "Get C style type string for code generation (e.g., 'float', 'half', 'int32_t').")
      .def("is_float", &DataType::IsFloat,
           "Check if this data type is a floating point type (FP4, FP8, FP16, FP32, BF16, HF4, HF8).")
      .def("is_signed_int", &DataType::IsSignedInt,
           "Check if this data type is a signed integer type (INT4, INT8, INT16, INT32, INT64).")
      .def("is_unsigned_int", &DataType::IsUnsignedInt,
           "Check if this data type is an unsigned integer type (UINT4, UINT8, UINT16, UINT32, UINT64).")
      .def("is_int", &DataType::IsInt, "Check if this data type is any integer type (signed or unsigned).")
      .def("code", &DataType::Code, "Get the underlying type code as uint8_t.")
      // Operators
      .def("__eq__", &DataType::operator==, nb::arg("other"), "Equality comparison operator")
      .def("__ne__", &DataType::operator!=, nb::arg("other"), "Inequality comparison operator")
      .def("__repr__", &DataType::ToString, "String representation for debugging")
      .def("__str__", &DataType::ToString, "String representation for printing");
}

}  // namespace python
}  // namespace pypto
