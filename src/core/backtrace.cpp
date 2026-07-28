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

#include <backtrace.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <ios>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/error.h"

namespace pypto {

/// Patterns to filter out from backtraces (internal/infrastructure frames)
const std::vector<std::string> kFileNameFilter = {
    "libbacktrace",   // backtrace infrastructure
    "nanobind",       // Python binding layer
    "__libc_",        // C library internals
    "include/c++/",   // C++ standard library
    "object.h",       // Python object.h
    "error.h",        // exception throwing infrastructure
    "core/logging.h"  // CHECK / INTERNAL_CHECK macro throw site
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
  // The filename argument names the *main executable*, not this module. Passing nullptr lets
  // libbacktrace find it itself (/proc/self/exe on Linux); every loaded shared object, including
  // this one, is then registered separately via dl_iterate_phdr at its true load address.
  //
  // Do not pass this module's own path (e.g. from dladdr): libbacktrace treats it as the
  // executable, elf_add() rejects the ET_DYN file, and phdr_callback pairs the descriptor with the
  // real executable instead. That registers *this* module's DWARF at the *executable's* load base,
  // so every PC in the CPython interpreter resolves to whichever PyPTO line sits at the same
  // offset — real-looking frames that were never on the call path.
  state_ = backtrace_create_state(nullptr, 1, ErrorCallback, nullptr);
}

void Backtrace::ErrorCallback(void* data, const char* msg, int errnum) {
  // Report libbacktrace errors to stderr so failures in backtrace generation are not silent, but
  // report each distinct message only once. When a platform cannot supply symbol information at
  // all, the same failure repeats without bound. macOS is the case in practice: upstream
  // libbacktrace accepts only MH_EXECUTE / MH_DYLIB / MH_DSYM Mach-O files, and a CPython
  // extension module is an MH_BUNDLE. Its dyld initialization path still succeeds overall, so it
  // installs macho_nodebug as the fileline handler — and that fires once per *frame* of every
  // captured trace, on top of one rejection message per loaded bundle.
  if (msg == nullptr) {
    return;
  }

  // The state is created with threaded=1, so the dedupe set needs its own lock. Holding it across
  // the write also keeps concurrent reports from interleaving.
  static std::mutex reported_mutex;
  static std::set<std::pair<std::string, int>> reported;
  std::scoped_lock lock(reported_mutex);
  if (!reported.emplace(msg, errnum).second) {
    return;
  }

  fprintf(stderr, "libbacktrace error: %s (errno: %d)\n", msg, errnum);
}

/// Clean up file paths from debug info that may contain temp build directory prefixes.
/// When building via pip in a temp directory, paths may look like:
///   /private/var/folders/.../build/./python/nanobind/modules/logging.cpp
/// This function extracts just the relative path portion.
std::string CleanupFilePath(const std::string& path) {
  if (path.empty()) {
    return path;
  }

  // Look for "/./", which indicates where the relative path begins
  // (this is created by -fdebug-prefix-map=${CMAKE_SOURCE_DIR}=.)
  // Replace the prefix up to "/./" with "./"
  size_t marker_pos = path.find("/./");
  if (marker_pos != std::string::npos) {
    return "./" + path.substr(marker_pos + 3);
  }

  // If path already starts with "./", keep it as-is
  if (path.size() >= 2 && path[0] == '.' && path[1] == '/') {
    return path;
  }

  return path;
}

int Backtrace::FullCallback(void* data, uintptr_t pc, const char* filename, int lineno,
                            const char* function) {
  auto* frames = static_cast<std::vector<StackFrame>*>(data);

  std::string func_str = function ? function : "";
  std::string file_str = filename ? CleanupFilePath(filename) : "";

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

  // Filter and deduplicate frames by PC address to handle Clang's debug info issues.
  // When Clang generates DWARF info for inlined functions/templates, it may
  // report multiple "virtual" frames for the same PC with incorrect source
  // locations. We keep only the first frame for each unique PC.
  std::vector<StackFrame> deduplicated_frames;
  for (const auto& frame : reversed_frames) {
    // Filter out libbacktrace and nanobind frames before deduplication.
    // This prevents filtered frames from being used in duplicate PC checks.
    // Also skip duplicate PC addresses (likely spurious inline frames from Clang).
    if ((!frame.filename.empty() && is_file_name_filtered(frame.filename)) ||
        (frame.pc != 0 && !deduplicated_frames.empty() && deduplicated_frames.back().pc == frame.pc)) {
      continue;
    }
    deduplicated_frames.push_back(frame);
  }

  for (const auto& frame : deduplicated_frames) {
    // Format: File "filename", line X in function_name
    if (!frame.filename.empty()) {
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
