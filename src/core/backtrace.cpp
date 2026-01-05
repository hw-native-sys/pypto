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

#include <cxxabi.h>
#include <dlfcn.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "pypto/core/error.h"

namespace pypto {

/// Patterns to filter out from backtraces (internal/infrastructure frames)
const std::vector<std::string> kFileNameFilter = {
    "libbacktrace",  // backtrace infrastructure
    "pybind11",      // Python binding layer
    "__libc_",       // C library internals
    "include/c++/",  // C++ standard library
    "object.h",      // Python object.h
    "error.h"        // exception throwing infrastructure
};

std::string StackFrame::to_string() const {
  std::ostringstream oss;

  if (!function.empty()) {
    oss << function;
  } else {
    oss << "0x" << std::hex << pc;
  }

  if (!filename.empty()) {
    oss << " at " << filename;
    if (lineno > 0) {
      oss << ":" << std::dec << lineno;
    }
  }

  return oss.str();
}

Backtrace& Backtrace::GetInstance() {
  static Backtrace instance;
  return instance;
}

Backtrace::Backtrace() {
  // Get the path of the current shared library using dladdr
  Dl_info info;
  const char* filename = nullptr;

  // Use the address of this function to find the shared library path
  if (dladdr(reinterpret_cast<void*>(&Backtrace::GetInstance), &info)) {
    filename = info.dli_fname;
  }

  // Use the filename from dladdr - this is the shared library path
  state_ = backtrace_create_state(filename, 1, ErrorCallback, nullptr);
}

void Backtrace::ErrorCallback(void* data, const char* msg, int errnum) {
  // Log errors in backtrace generation to stderr for debugging
  if (msg) {
    fprintf(stderr, "libbacktrace error: %s (errno: %d)\n", msg, errnum);
  }
}

int Backtrace::FullCallback(void* data, uintptr_t pc, const char* filename, int lineno,
                            const char* function) {
  auto* frames = static_cast<std::vector<StackFrame>*>(data);

  std::string func_str = function ? function : "";
  std::string file_str = filename ? filename : "";

  frames->emplace_back(func_str, file_str, lineno, pc);
  return 0;  // Continue collecting frames
}

std::vector<StackFrame> Backtrace::CaptureStackTrace(int skip) {
  std::vector<StackFrame> frames;

  if (state_ != nullptr) {
    // Skip one additional frame for this function itself
    backtrace_full(state_, skip + 1, FullCallback, ErrorCallback, &frames);
  }

  return frames;
}

// Helper function to read a specific line from a file
std::string ReadSourceLine(const std::string& filename, int lineno) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    return "";
  }

  std::string line;
  int current_line = 0;
  while (std::getline(file, line)) {
    current_line++;
    if (current_line == lineno) {
      // Trim leading whitespace for display
      size_t start = line.find_first_not_of(" \t");
      if (start != std::string::npos) {
        return line.substr(start);
      }
      return line;
    }
  }
  return "";
}

std::string Backtrace::FormatStackTrace(const std::vector<StackFrame>& frames) {
  if (frames.empty()) {
    return "";
  }

  std::ostringstream oss;

  // Reverse the frames to show most recent last (like Python)
  std::vector<StackFrame> reversed_frames(frames.rbegin(), frames.rend());

  auto is_file_name_filtered = [](const std::string& filename) {
    return std::any_of(
        kFileNameFilter.begin(), kFileNameFilter.end(),
        [&filename](const std::string& filter) { return filename.find(filter) != std::string::npos; });
  };

  // Deduplicate frames by PC address to handle Clang's debug info issues.
  // When Clang generates DWARF info for inlined functions/templates, it may
  // report multiple "virtual" frames for the same PC with incorrect source
  // locations. We keep only the last frame for each unique PC, as that's
  // typically the actual call site.
  std::vector<StackFrame> deduplicated_frames;
  for (const auto& frame : reversed_frames) {
    if (frame.pc != 0 && !deduplicated_frames.empty() && deduplicated_frames.back().pc == frame.pc) {
      // Same PC as the previous frame - this is likely a spurious inline frame.
      // Replace with the current frame (which is typically more accurate).
      deduplicated_frames.back() = frame;
    } else {
      deduplicated_frames.push_back(frame);
    }
  }

  for (const auto& frame : deduplicated_frames) {
    // Format: File "filename", line X in function_name
    if (!frame.filename.empty()) {
      if (is_file_name_filtered(frame.filename)) {
        // Skip libbacktrace and pybind11 frames
        continue;
      }
      oss << " File \"" << frame.filename << "\", line " << frame.lineno << "\n";

      // Try to read and display the source line
      std::string source_line = ReadSourceLine(frame.filename, frame.lineno);
      if (!source_line.empty()) {
        oss << "   " << source_line << "\n";
      }
    } else if (frame.pc != 0) {
      // If we don't have filename info, skip this frame
    }
  }

  return oss.str();
}

}  // namespace pypto
