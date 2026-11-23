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

#ifndef PYPTO_BACKEND_LINX_BACKEND_LINX_H_
#define PYPTO_BACKEND_LINX_BACKEND_LINX_H_

#include <string>

#include "pypto/backend/common/backend.h"

namespace pypto::backend {

class BackendLinx final : public Backend {
 public:
  static const BackendLinx& Instance();
  [[nodiscard]] std::string GetTypeName() const override { return "Linx"; }
  [[nodiscard]] bool HasPtoCodegen() const override { return false; }
  [[nodiscard]] const BackendHandler* GetHandler() const override;

 private:
  BackendLinx();
};

}  // namespace pypto::backend

#endif  // PYPTO_BACKEND_LINX_BACKEND_LINX_H_
