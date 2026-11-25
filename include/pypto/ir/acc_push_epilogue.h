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

#ifndef PYPTO_IR_ACC_PUSH_EPILOGUE_H_
#define PYPTO_IR_ACC_PUSH_EPILOGUE_H_

#include <map>
#include <set>
#include <string>

#include "pypto/core/dtype.h"

namespace pypto::ir {

/// Value-semantic pipe record with required layout, quant and relu fields.
using AccPushEpilogue = std::map<std::string, std::string>;

inline std::string ValidateAccPushEpilogue(const AccPushEpilogue& e) {
  if (e.size() != 3 || !e.count("layout") || !e.count("quant") || !e.count("relu")) {
    return "acc_push_epilogue requires exactly layout, quant and relu fields";
  }
  if (e.at("layout") != "nz2nd" && e.at("layout") != "nz2dn" && e.at("layout") != "nz2nz") {
    return "acc_push_epilogue.layout must be nz2nd, nz2dn or nz2nz";
  }
  // Enable only modes with a complete PyPTO operand and result contract.
  static const std::set<std::string> modes = {"no_convert",    "f32_f16",    "f32_bf16",
                                              "deqf16_scalar", "deqf16_vec", "qf322f16_pre_scalar"};
  if (!modes.count(e.at("quant"))) return "unsupported acc_push_epilogue.quant: " + e.at("quant");
  if (e.at("relu") != "no_relu" && e.at("relu") != "normal_relu") {
    return "acc_push_epilogue.relu must be no_relu or normal_relu";
  }
  return "";
}

inline bool IsScalarAccPushQuant(const std::string& quant) {
  return quant == "deqf16_scalar" || quant == "qf322f16_pre_scalar";
}
inline bool IsVectorAccPushQuant(const std::string& quant) { return quant == "deqf16_vec"; }
inline DataType AccPushSourceDtype(const std::string& quant) {
  return quant == "deqf16_scalar" || quant == "deqf16_vec" ? DataType::INT32 : DataType::FP32;
}
inline DataType AccPushResultDtype(const std::string& quant, DataType source) {
  if (quant == "no_convert") return source;
  return quant == "f32_bf16" ? DataType::BF16 : DataType::FP16;
}

}  // namespace pypto::ir
#endif  // PYPTO_IR_ACC_PUSH_EPILOGUE_H_
