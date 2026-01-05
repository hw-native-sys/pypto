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
 * @file logging.cpp
 * @brief Implementation of Python bindings for logging framework
 */

#include "pypto/core/logging.h"

#include <pybind11/pybind11.h>

#include <string>

#include "../bindings.h"

namespace py = pybind11;

namespace pypto {
namespace python {

/**
 * @brief Log a message at the DEBUG level
 * @param message Message to log
 */
void log_debug(const std::string& message) { LOG_DEBUG << message; }

/**
 * @brief Log a message at the INFO level
 * @param message Message to log
 */
void log_info(const std::string& message) { LOG_INFO << message; }

/**
 * @brief Log a message at the WARN level
 * @param message Message to log
 */
void log_warn(const std::string& message) { LOG_WARN << message; }

/**
 * @brief Log a message at the ERROR level
 * @param message Message to log
 */
void log_error(const std::string& message) { LOG_ERROR << message; }

/**
 * @brief Log a message at the FATAL level
 * @param message Message to log
 */
void log_fatal(const std::string& message) { LOG_FATAL << message; }

/**
 * @brief Log a message at the EVENT level
 * @param message Message to log
 */
void log_event(const std::string& message) { LOG_EVENT << message; }

/**
 * @brief Check a condition and throw ValueError if it fails
 * @param condition Condition to check
 * @param message Error message to include if check fails
 */
void check(bool condition, const std::string& message) { CHECK(condition) << message; }

/**
 * @brief Check an internal invariant and throw InternalError if it fails
 * @param condition Condition to check
 * @param message Error message to include if check fails
 */
void internal_check(bool condition, const std::string& message) { INTERNAL_CHECK(condition) << message; }

void BindLogging(py::module_& m) {
  // Bind LogLevel enum
  py::enum_<LogLevel>(m, "LogLevel", "Enumeration of available log levels")
      .value("DEBUG", LogLevel::DEBUG, "Detailed information for debugging")
      .value("INFO", LogLevel::INFO, "General informational messages")
      .value("WARN", LogLevel::WARN, "Warning messages for potentially harmful situations")
      .value("ERROR", LogLevel::ERROR, "Error messages for failures")
      .value("FATAL", LogLevel::FATAL, "Critical errors that may cause termination")
      .value("EVENT", LogLevel::EVENT, "Special events and milestones")
      .value("NONE", LogLevel::NONE, "Disable all logging")
      .export_values();  // Export values to module scope for convenience

  // Bind LoggerManager functions
  m.def("set_log_level", &LoggerManager::ResetLevel, py::arg("level"),
        "Set the global log level threshold. Only messages at or above this level will be logged.");
  m.def("log_debug", &log_debug, py::arg("message"), "Log a message at the DEBUG level");
  m.def("log_info", &log_info, py::arg("message"), "Log a message at the INFO level");
  m.def("log_warn", &log_warn, py::arg("message"), "Log a message at the WARN level");
  m.def("log_error", &log_error, py::arg("message"), "Log a message at the ERROR level");
  m.def("log_fatal", &log_fatal, py::arg("message"), "Log a message at the FATAL level");
  m.def("log_event", &log_event, py::arg("message"), "Log a message at the EVENT level");
  m.def("check", &check, py::arg("condition"), py::arg("message"),
        "Check a condition and throw ValueError if it fails. "
        "Usage: check(x > 0, 'x must be positive')");
  m.def("internal_check", &internal_check, py::arg("condition"), py::arg("message"),
        "Check an internal invariant and throw InternalError if it fails. "
        "Usage: internal_check(ptr is not None, 'pointer should never be None')");
}

}  // namespace python
}  // namespace pypto
