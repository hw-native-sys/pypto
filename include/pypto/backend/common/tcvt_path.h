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

#ifndef PYPTO_BACKEND_COMMON_TCVT_PATH_H_
#define PYPTO_BACKEND_COMMON_TCVT_PATH_H_

#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/dtype.h"

namespace pypto {
namespace backend {

/** Return whether one `pto.tcvt` instruction supports the conversion. */
[[nodiscard]] bool IsNativeTcvt(const TcvtAdjacency& table, DataType from, DataType to);

/**
 * Return the preferred shortest native conversion path, excluding `from` and
 * including `to`. An empty result means either `from == to` or no legal path;
 * callers that require a conversion must distinguish those cases.
 */
[[nodiscard]] std::vector<DataType> FindTcvtPath(const TcvtAdjacency& table, DataType from, DataType to);

}  // namespace backend
}  // namespace pypto

#endif  // PYPTO_BACKEND_COMMON_TCVT_PATH_H_
