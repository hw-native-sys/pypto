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

#include "pypto/backend/common/tcvt_path.h"

#include <stdexcept>
#include <vector>

namespace {

using pypto::DataType;
using pypto::backend::FindTcvtPath;
using pypto::backend::TcvtAdjacency;

void RequireEqual(const std::vector<DataType>& actual, const std::vector<DataType>& expected) {
  if (actual != expected) throw std::runtime_error("cast path depends on adjacency edge order");
}

void TestOrderAndDuplicatesDoNotChangePath() {
  const TcvtAdjacency ordered{{
      {DataType::FP32, DataType::FP16},
      {DataType::FP32, DataType::BF16},
      {DataType::FP16, DataType::INT8},
      {DataType::BF16, DataType::INT8},
  }};
  const TcvtAdjacency reordered{{
      {DataType::BF16, DataType::INT8},
      {DataType::FP32, DataType::BF16},
      {DataType::FP16, DataType::INT8},
      {DataType::FP32, DataType::FP16},
      {DataType::FP32, DataType::BF16},
      {DataType::FP32, DataType::FP32},
  }};
  const std::vector<DataType> expected{DataType::FP16, DataType::INT8};

  RequireEqual(FindTcvtPath(ordered, DataType::FP32, DataType::INT8), expected);
  RequireEqual(FindTcvtPath(reordered, DataType::FP32, DataType::INT8), expected);
}

}  // namespace

int main() {
  TestOrderAndDuplicatesDoNotChangePath();
  return 0;
}
