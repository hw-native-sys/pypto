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
 * @file logging.h
 * @brief Logging framework with support for console, file, and in-memory logging
 *
 * This header provides a flexible logging system with:
 * - Multiple log levels (DEBUG, INFO, WARN, ERROR, FATAL, EVENT)
 * - Colored terminal output using ANSI escape codes
 * - File-based logging with append/overwrite modes
 * - In-memory line-based logging for programmatic access
 * - Thread-safe logging operations
 * - Stream-style and printf-style logging interfaces
 */

#ifndef PYPTO_CORE_LOGGING_H_
#define PYPTO_CORE_LOGGING_H_

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/error.h"

namespace pypto {

/**
 * @brief TTY command for colored terminal output
 *
 * Wraps ANSI escape codes for terminal text formatting and coloring.
 */
class TTYCmd {
 public:
  unsigned char code;

  explicit TTYCmd(int code_in) : code(code_in) {}

  /**
   * @brief Convert the TTY command to an ANSI escape sequence string
   * @return ANSI escape sequence as a string
   */
  [[nodiscard]] std::string Str() const { return "\033[" + std::to_string(code) + "m"; }
};

// TTY color macros for convenient colored output
#define TTY_COLOR(n, ...) TTYCmd(n), ##__VA_ARGS__, TTYCmd(0)
#define TTY_RED(...) TTY_COLOR(31, __VA_ARGS__)
#define TTY_GREEN(...) TTY_COLOR(32, __VA_ARGS__)
#define TTY_YELLOW(...) TTY_COLOR(33, __VA_ARGS__)
#define TTY_BLUE(...) TTY_COLOR(34, __VA_ARGS__)
#define TTY_MAGENTA(...) TTY_COLOR(35, __VA_ARGS__)
#define TTY_CYAN(...) TTY_COLOR(36, __VA_ARGS__)
#define TTY_WHITE(...) TTY_COLOR(37, __VA_ARGS__)

/**
 * @brief Enumeration of available log levels
 *
 * Log levels in ascending order of severity:
 * - DEBUG: Detailed information for debugging
 * - INFO: General informational messages
 * - WARN: Warning messages for potentially harmful situations
 * - ERROR: Error messages for failures
 * - FATAL: Critical errors that may cause termination
 * - EVENT: Special events and milestones
 * - NONE: Disable all logging
 */
enum class LogLevel : uint8_t {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3,
  FATAL = 4,
  EVENT = 5,
  NONE = 6,
};

/**
 * @brief Standard error logger that writes to stderr
 *
 * Supports colored output using TTY commands.
 */
class StdLogger {
 public:
  /**
   * @brief Log a TTY command (for colored output)
   * @param cmd TTY command to output
   * @return Reference to this logger for chaining
   */
  StdLogger& Log(TTYCmd&& cmd) {
    std::cerr << cmd.Str();
    return *this;
  }

  /**
   * @brief Log a value of any type
   * @tparam T Type of value to log
   * @param t Value to log
   * @return Reference to this logger for chaining
   */
  template <typename T>
  StdLogger& Log(T&& t) {
    std::cerr << (std::forward<T>(t));
    return *this;
  }

  StdLogger() = default;
  StdLogger(const StdLogger&) = delete;
  StdLogger& operator=(const StdLogger&) = delete;
};

/**
 * @brief File-based logger that writes to a file stream
 *
 * Supports both append and overwrite modes. TTY commands are ignored
 * since file output typically doesn't support colored text.
 */
class FileLogger {
 public:
  std::ofstream ofs;

  /**
   * @brief Construct a file logger
   * @param filepath Path to the log file
   * @param append If true, append to existing file; otherwise overwrite
   */
  FileLogger(const std::string& filepath, bool append) {
    if (append) {
      ofs.open(filepath, std::ios_base::app);
    } else {
      ofs.open(filepath);
    }
  }

  /**
   * @brief Log a TTY command (no-op for file logging)
   * @param cmd TTY command (ignored)
   * @return Reference to this logger for chaining
   */
  FileLogger& Log([[maybe_unused]] TTYCmd&& cmd) { return *this; }

  /**
   * @brief Log a value to the file
   * @tparam T Type of value to log
   * @param t Value to log
   * @return Reference to this logger for chaining
   */
  template <typename T>
  FileLogger& Log(T&& t) {
    ofs << (std::forward<T>(t));
    return *this;
  }
  FileLogger(const FileLogger&) = delete;
  FileLogger& operator=(const FileLogger&) = delete;
};

/**
 * @brief In-memory logger that stores log lines as strings
 *
 * Useful for programmatic access to log messages or testing.
 * Inherits from std::vector<std::string> for direct access to stored lines.
 */
class LineLogger : public std::vector<std::string> {
 public:
  /**
   * @brief Log a TTY command (no-op for line logging)
   * @param cmd TTY command (ignored)
   * @return Reference to this logger for chaining
   */
  LineLogger& Log([[maybe_unused]] TTYCmd&& cmd) { return *this; }

  /**
   * @brief Log a string value
   * @param t String to store
   * @return Reference to this logger for chaining
   */
  LineLogger& Log(std::string&& t) {
    this->emplace_back(t);
    return *this;
  }
};

/**
 * @brief Central manager for all loggers
 *
 * This singleton class manages:
 * - Global log level threshold
 * - Standard output logger enable/disable
 * - Multiple file loggers
 * - Multiple line (in-memory) loggers
 *
 * All logging operations are thread-safe.
 */
class LoggerManager {
 public:
  std::mutex log_mtx;
  LogLevel level{LogLevel::ERROR};
  bool std_enabled{true};
  StdLogger std_logger;
  std::unordered_map<std::string, std::unique_ptr<FileLogger>> file_logger_dict;
  std::unordered_map<std::string, std::shared_ptr<LineLogger>> line_logger_dict;

  LoggerManager() = default;

  /**
   * @brief Log a message to all active loggers
   * @tparam T Type of the log message
   * @param l Log level
   * @param t Plain message (for file/line loggers)
   * @param t_rich Rich message with formatting (for std logger)
   */
  template <typename T>
  void Log(LogLevel l, T&& t, T&& t_rich) {
    std::lock_guard lock(log_mtx);
    if (l >= level) {
      if (std_enabled) {
        std_logger.Log(std::forward<T>(t_rich));
      }
    }
    for (auto& [filepath, logger] : file_logger_dict) {
      (void)filepath;
      logger->Log(std::forward<T>(t));
    }
    for (auto& [name, logger] : line_logger_dict) {
      (void)name;
      logger->Log(std::forward<T>(t));
    }
  }

  /**
   * @brief Set the global log level threshold
   * @param l New log level
   */
  static void ResetLevel(LogLevel l) { GetManager().level = l; }

  /**
   * @brief Enable or disable standard output logging
   * @param enabled True to enable, false to disable
   */
  static void StdLoggerEnable(bool enabled) { GetManager().std_enabled = enabled; }

  /**
   * @brief Register a file logger
   * @param filepath Path to the log file
   * @param append If true, append to existing file; otherwise overwrite
   */
  static void FileLoggerRegister(const std::string& filepath, bool append) {
    GetManager().file_logger_dict.try_emplace(filepath, std::make_unique<FileLogger>(filepath, append));
  }

  /**
   * @brief Unregister and close a file logger
   * @param filepath Path to the log file to unregister
   */
  static void FileLoggerUnregister(const std::string& filepath) {
    GetManager().file_logger_dict.erase(filepath);
  }

  /**
   * @brief Replace one file logger with another
   * @param old_filepath Path to the old log file
   * @param new_filepath Path to the new log file
   * @param append If true, append to new file; otherwise overwrite
   */
  static void FileLoggerReplace(const std::string& old_filepath, const std::string& new_filepath,
                                bool append) {
    FileLoggerUnregister(old_filepath);
    FileLoggerRegister(new_filepath, append);
  }

  /**
   * @brief Register an in-memory line logger
   * @param name Name identifier for the logger
   * @return Shared pointer to the line logger for access to stored lines
   */
  static std::shared_ptr<LineLogger> LineLoggerRegister(const std::string& name) {
    auto logger = std::make_shared<LineLogger>();
    GetManager().line_logger_dict[name] = logger;
    return logger;
  }

  /**
   * @brief Unregister an in-memory line logger
   * @param name Name identifier of the logger to unregister
   */
  static void LineLoggerUnregister(const std::string& name) { GetManager().line_logger_dict.erase(name); }

  friend class Logger;

  /**
   * @brief Get the singleton LoggerManager instance
   * @return Reference to the singleton LoggerManager
   */
  static LoggerManager& GetManager() {
    static LoggerManager manager;
    return manager;
  }
};

constexpr uint32_t MAX_LOG_BUF_SIZE = 1024;

/**
 * @brief Main logger class for creating log messages
 *
 * This class handles:
 * - Automatic timestamp generation
 * - Log level prefixing
 * - Message buffering
 * - Stream-style output via operator<<
 * - Variadic output via operator()
 *
 * Logger objects should be created per log message (they flush on destruction).
 */
class Logger {
 private:
  std::stringstream ss;
  std::stringstream ss_rich;
  LogLevel level{LogLevel::ERROR};
  bool enable_log = false;

 public:
  /**
   * @brief Construct a logger for a single log message
   * @param level_in Log level for this message
   * @param func Function name (currently unused but available for future use)
   * @param line Line number (currently unused but available for future use)
   */
  Logger(LogLevel level_in, [[maybe_unused]] const std::string& func, [[maybe_unused]] int line)
      : level(level_in) {
    enable_log = LoggerManager::GetManager().level <= level;
    if (enable_log) {
      static const char* MSG = "DIWEFVN";
      auto now = std::chrono::system_clock::now();
      auto time = std::chrono::system_clock::to_time_t(now);
      auto tm = *std::localtime(&time);

      // Format timestamp
      char time_buf[128];
      std::strftime(time_buf, sizeof(time_buf), "%F %T.", &tm);
      Log(time_buf);

      // Add milliseconds and log level
      char buf[MAX_LOG_BUF_SIZE];
      auto epoch = now.time_since_epoch();
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(epoch).count() % 1000;
      std::snprintf(buf, MAX_LOG_BUF_SIZE, "%03d %c | ", static_cast<int>(ms), MSG[static_cast<int>(level)]);

      Log(buf);
      // Uncomment the following line to include function and line number in logs:
      // Log(func + ":" + std::to_string(line) + " | ");
    }
  }

  /**
   * @brief Destructor flushes the log message to all active loggers
   */
  ~Logger() {
    if (enable_log) {
      Log("\n");
      LoggerManager::GetManager().Log(level, ss.str(), ss_rich.str());
    }
  }

  /**
   * @brief Log a TTY command for colored output
   * @param val TTY command
   * @return Reference to this logger for chaining
   */
  Logger& Log(TTYCmd&& val) {
    ss_rich << val.Str();
    return *this;
  }

  /**
   * @brief Log a value
   * @tparam T Type of value to log
   * @param val Value to log
   * @return Reference to this logger for chaining
   */
  template <typename T>
  Logger& Log(T&& val) {
    ss << (std::forward<T>(val));
    ss_rich << (std::forward<T>(val));
    return *this;
  }

  /**
   * @brief Stream operator for convenient logging
   * @tparam T Type of value to log
   * @param val Value to log
   * @return Reference to this logger for chaining
   */
  template <typename T>
  Logger& operator<<(T&& val) {
    if (enable_log) {
      return Log(std::forward<T>(val));
    } else {
      return *this;
    }
  }

  /**
   * @brief Variadic operator for logging multiple values at once
   * @tparam Tys Types of values to log
   * @param vals Values to log
   * @return Reference to this logger for chaining
   */
  template <typename... Tys>
  Logger& operator()(Tys&&... vals) {
    if (enable_log) {
      if constexpr (sizeof...(Tys) > 0) {
        (Log(std::forward<Tys>(vals)), ...);
      }
    }
    return *this;
  }
};

// Convenience macros for logging at different levels
#define LOG_LEVEL(lvl) pypto::Logger(lvl, __func__, __LINE__)
#define LOG_DEBUG LOG_LEVEL(pypto::LogLevel::DEBUG)
#define LOG_INFO LOG_LEVEL(pypto::LogLevel::INFO)
#define LOG_WARN LOG_LEVEL(pypto::LogLevel::WARN)
#define LOG_ERROR LOG_LEVEL(pypto::LogLevel::ERROR)
#define LOG_FATAL LOG_LEVEL(pypto::LogLevel::FATAL)
#define LOG_EVENT LOG_LEVEL(pypto::LogLevel::EVENT)

// Printf-style logging macros
#define LOG_F(lvl, args...)                                                 \
  do {                                                                      \
    if (pypto::LoggerManager::GetManager().level <= pypto::LogLevel::lvl) { \
      constexpr int default_buf_size = 1024;                                \
      std::string buf(default_buf_size, '\0');                              \
      int msg_length = std::snprintf(buf.data(), buf.size(), ##args) + 1;   \
      if (msg_length > default_buf_size) {                                  \
        buf.resize(msg_length, '\0');                                       \
        std::snprintf(buf.data(), buf.size(), ##args);                      \
      }                                                                     \
      LOG_##lvl(buf.data());                                                \
    }                                                                       \
  } while (false)

#define LOG_DEBUG_F(fmt, args...) LOG_F(DEBUG, fmt, ##args)
#define LOG_INFO_F(fmt, args...) LOG_F(INFO, fmt, ##args)
#define LOG_WARN_F(fmt, args...) LOG_F(WARN, fmt, ##args)
#define LOG_ERROR_F(fmt, args...) LOG_F(ERROR, fmt, ##args)
#define LOG_EVENT_F(fmt, args...) LOG_F(EVENT, fmt, ##args)

/**
 * @brief Helper class for CHECK, INTERNAL_CHECK, UNREACHABLE, and INTERNAL_UNREACHABLE macros
 *
 * This class collects error messages via operator<< and throws
 * an exception on destruction if the check condition failed.
 *
 * @tparam ExceptionType The type of exception to throw (ValueError or InternalError)
 * @tparam AlwaysFail If true, the destructor is [[noreturn]] and always throws (for UNREACHABLE)
 */
template <typename ExceptionType, bool AlwaysFail = false>
class CheckLogger;

// Specialization for conditional checks (CHECK, INTERNAL_CHECK)
template <typename ExceptionType>
class CheckLogger<ExceptionType, false> {
 private:
  std::stringstream ss;
  bool failed;
  const char* file;
  int line;
  const char* expr_str;

 public:
  CheckLogger(bool condition, const char* expr_str, const char* file, int line)
      : failed(!condition), file(file), line(line), expr_str(expr_str) {}

  ~CheckLogger() noexcept(false) {
    if (failed) {
      ss << "\n" << "Check failed: " << expr_str << " at " << file << ":" << line;
      throw ExceptionType(ss.str());
    }
  }

  template <typename T>
  CheckLogger& operator<<(T&& val) {
    if (failed) {
      ss << std::forward<T>(val);
    }
    return *this;
  }

  CheckLogger(const CheckLogger&) = delete;
  CheckLogger& operator=(const CheckLogger&) = delete;
  CheckLogger(CheckLogger&&) = delete;
  CheckLogger& operator=(CheckLogger&&) = delete;
};

// Specialization for unconditional failure (UNREACHABLE, INTERNAL_UNREACHABLE)
template <typename ExceptionType>
class CheckLogger<ExceptionType, true> {
 private:
  std::stringstream ss;
  const char* file;
  int line;
  const char* expr_str;

 public:
  CheckLogger(bool /*condition*/, const char* expr_str, const char* file, int line)
      : file(file), line(line), expr_str(expr_str) {}

  [[noreturn]] ~CheckLogger() noexcept(false) {
    ss << "\n" << "Check failed: " << expr_str << " at " << file << ":" << line;
    throw ExceptionType(ss.str());
  }

  template <typename T>
  CheckLogger& operator<<(T&& val) {
    ss << std::forward<T>(val);
    return *this;
  }

  CheckLogger(const CheckLogger&) = delete;
  CheckLogger& operator=(const CheckLogger&) = delete;
  CheckLogger(CheckLogger&&) = delete;
  CheckLogger& operator=(CheckLogger&&) = delete;
};

/**
 * @brief Check a condition and throw ValueError if it fails
 *
 * Usage: CHECK(condition) << "error message";
 */
#define CHECK(expr) pypto::CheckLogger<pypto::ValueError>(static_cast<bool>(expr), #expr, __FILE__, __LINE__)

/**
 * @brief Check an internal invariant and throw InternalError if it fails
 *
 * Usage: INTERNAL_CHECK(condition) << "error message";
 */
#define INTERNAL_CHECK(expr) \
  pypto::CheckLogger<pypto::InternalError>(static_cast<bool>(expr), #expr, __FILE__, __LINE__)

/**
 * @brief Mark a code path as unreachable and throw ValueError if reached
 *
 * Usage: UNREACHABLE << "optional message";
 */
#define UNREACHABLE pypto::CheckLogger<pypto::ValueError, true>(false, "unreachable", __FILE__, __LINE__)

/**
 * @brief Mark a code path as internally unreachable and throw InternalError if reached
 *
 * Usage: INTERNAL_UNREACHABLE << "optional message";
 */
#define INTERNAL_UNREACHABLE \
  pypto::CheckLogger<pypto::InternalError, true>(false, "unreachable", __FILE__, __LINE__)

}  // namespace pypto

#endif  // PYPTO_CORE_LOGGING_H_
