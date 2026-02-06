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

#ifndef PYPTO_BACKEND_COMMON_BACKEND_CONFIG_H_
#define PYPTO_BACKEND_COMMON_BACKEND_CONFIG_H_

#include <mutex>
#include <optional>

#include "pypto/backend/common/backend.h"

namespace pypto {
namespace backend {

/*!
 * \brief Global backend configuration manager
 *
 * Manages the global backend type selection. The backend type must be
 * explicitly configured before use - there is no default value.
 *
 * Thread-safe singleton pattern with lazy initialization.
 */
class BackendConfig {
 public:
  /*!
   * \brief Set the global backend type
   *
   * Must be called before any backend operations. Can be called multiple
   * times with the same type (idempotent), but will throw an error if
   * attempting to change to a different type.
   *
   * \param type The backend type to use (CCE or PTO)
   * \throws pypto::ValueError if attempting to change an already-set type
   */
  static void SetBackendType(BackendType type);

  /*!
   * \brief Get the configured backend instance
   *
   * Returns the backend instance corresponding to the configured type.
   *
   * \return Pointer to the backend instance (never null)
   * \throws pypto::ValueError if backend type has not been configured
   */
  static const Backend* GetBackend();

  /*!
   * \brief Get the configured backend type
   *
   * \return The configured backend type
   * \throws pypto::ValueError if backend type has not been configured
   */
  static BackendType GetBackendType();

  /*!
   * \brief Check if backend type has been configured
   *
   * \return true if SetBackendType() has been called, false otherwise
   */
  static bool IsConfigured();

  /*!
   * \brief Reset backend configuration (for testing only)
   *
   * WARNING: This function should ONLY be used in tests to reset the
   * backend configuration between test cases. Do NOT use in production code.
   */
  static void ResetForTesting();

 private:
  static std::optional<BackendType> backend_type_;
  static std::mutex mutex_;
};

/*!
 * \brief Convenience function to get the configured backend
 *
 * Equivalent to BackendConfig::GetBackend()
 *
 * \return Pointer to the backend instance
 * \throws pypto::ValueError if backend type has not been configured
 */
inline const Backend* GetBackend() { return BackendConfig::GetBackend(); }

/*!
 * \brief Convenience function to get the configured backend type
 *
 * Equivalent to BackendConfig::GetBackendType()
 *
 * \return The configured backend type
 * \throws pypto::ValueError if backend type has not been configured
 */
inline BackendType GetBackendType() { return BackendConfig::GetBackendType(); }

}  // namespace backend
}  // namespace pypto

#endif  // PYPTO_BACKEND_COMMON_BACKEND_CONFIG_H_
