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

#include "pypto/backend/common/backend_config.h"

#include "pypto/core/logging.h"

namespace pypto {
namespace backend {

std::optional<BackendType> BackendConfig::backend_type_;
std::mutex BackendConfig::mutex_;

void BackendConfig::SetBackendType(BackendType type) {
  std::scoped_lock<std::mutex> lock(mutex_);

  if (backend_type_.has_value()) {
    // Idempotent: allow setting the same type multiple times
    CHECK(*backend_type_ == type) << "Backend type already set to "
                                  << (*backend_type_ == BackendType::CCE ? "CCE" : "PTO")
                                  << ", cannot change to " << (type == BackendType::CCE ? "CCE" : "PTO");
    return;
  }

  backend_type_ = type;
}

const Backend* BackendConfig::GetBackend() {
  std::scoped_lock<std::mutex> lock(mutex_);

  CHECK(backend_type_.has_value())
      << "Backend type not configured. "
      << "Please call SetBackendType() or use compile() with backend_type parameter.";

  return GetBackendInstance(*backend_type_);
}

BackendType BackendConfig::GetBackendType() {
  std::scoped_lock<std::mutex> lock(mutex_);

  CHECK(backend_type_.has_value())
      << "Backend type not configured. "
      << "Please call SetBackendType() or use compile() with backend_type parameter.";

  return *backend_type_;
}

bool BackendConfig::IsConfigured() {
  std::scoped_lock<std::mutex> lock(mutex_);
  return backend_type_.has_value();
}

void BackendConfig::ResetForTesting() {
  std::scoped_lock<std::mutex> lock(mutex_);
  backend_type_.reset();
}

}  // namespace backend
}  // namespace pypto
