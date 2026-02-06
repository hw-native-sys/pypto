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

#include "pypto/backend/910B_PTO/backend_910b_pto.h"

#include <string>

#include "pypto/backend/common/soc.h"
#include "pypto/codegen/pto/pto_codegen.h"

namespace pypto {
namespace backend {

Backend910B_PTO::Backend910B_PTO() : Backend(Create910BSoC()) {
  // Operators are registered via REGISTER_BACKEND_OP macro
  // in backend_910b_pto_ops.cpp during static initialization
}

Backend910B_PTO& Backend910B_PTO::Instance() {
  static Backend910B_PTO instance;
  return instance;
}

std::string Backend910B_PTO::GenerateCode(const ir::ProgramPtr& program) {
  codegen::PTOCodegen codegen(this);
  return codegen.Generate(program);
}

}  // namespace backend
}  // namespace pypto
