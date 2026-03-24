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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_AUTO_NAME_UTILS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_AUTO_NAME_UTILS_H_

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/core/error.h"

namespace pypto {
namespace ir {
namespace auto_name {

struct ParsedName {
  std::string base_name;
  std::string qualifier;
  std::optional<std::string> role;
  std::optional<int> version;
  bool has_auto_suffix = false;
};

inline bool IsKnownRole(std::string_view token) {
  return token == "ssa" || token == "iter" || token == "rv" || token == "phi" || token == "tile" ||
         token == "tmp" || token == "out" || token == "store" || token == "idx";
}

inline bool IsVersionToken(std::string_view token) {
  if (token.size() < 2 || token[0] != 'v') {
    return false;
  }
  return std::all_of(token.begin() + 1, token.end(), [](unsigned char ch) { return std::isdigit(ch); });
}

inline std::optional<int> ParseVersionToken(std::string_view token) {
  if (!IsVersionToken(token)) {
    return std::nullopt;
  }
  return std::stoi(std::string(token.substr(1)));
}

inline std::string JoinQualifierParts(const std::vector<std::string>& qualifier_parts) {
  std::string qualifier;
  for (const auto& part : qualifier_parts) {
    if (part.empty()) {
      continue;
    }
    if (!qualifier.empty()) {
      qualifier += "_";
    }
    qualifier += part;
  }
  return qualifier;
}

inline std::string ChunkOuterQualifier() { return "co"; }

inline std::string ChunkInnerQualifier() { return "ci"; }

inline std::string ChunkRemainderQualifier() { return "cr"; }

inline std::string LoopLevelQualifier(int level) { return "l" + std::to_string(level); }

inline std::string RowMajorQualifier() { return "rm"; }

inline std::string ArgQualifier(size_t index) { return "a" + std::to_string(index); }

inline void ValidateBaseName(const std::string& base_name) {
  if (base_name.find("__") != std::string::npos) {
    throw pypto::ValueError("IR auto-name base cannot contain reserved delimiter '__': " + base_name);
  }
}

inline std::string BuildName(const std::string& base_name, const std::string& qualifier = "",
                             const std::optional<std::string>& role = std::nullopt,
                             const std::optional<int>& version = std::nullopt) {
  ValidateBaseName(base_name);
  if (qualifier.empty() && !role.has_value() && !version.has_value()) {
    return base_name;
  }

  std::string suffix;
  if (!qualifier.empty()) {
    suffix += qualifier;
  }
  if (role.has_value()) {
    if (!suffix.empty()) {
      suffix += "_";
    }
    suffix += *role;
  }
  if (version.has_value()) {
    if (!suffix.empty()) {
      suffix += "_";
    }
    suffix += "v" + std::to_string(*version);
  }
  return base_name + "__" + suffix;
}

inline std::string BuildName(const std::string& base_name, const std::vector<std::string>& qualifier_parts,
                             const std::optional<std::string>& role = std::nullopt,
                             const std::optional<int>& version = std::nullopt) {
  return BuildName(base_name, JoinQualifierParts(qualifier_parts), role, version);
}

inline std::string BuildName(const ParsedName& parsed) {
  return BuildName(parsed.base_name, parsed.qualifier, parsed.role, parsed.version);
}

inline ParsedName Parse(const std::string& name) {
  ParsedName parsed;
  parsed.base_name = name;

  size_t boundary = name.find("__");
  if (boundary == std::string::npos) {
    return parsed;
  }

  parsed.base_name = name.substr(0, boundary);
  parsed.has_auto_suffix = true;

  std::string suffix = name.substr(boundary + 2);
  size_t last_sep = suffix.rfind('_');
  if (last_sep != std::string::npos) {
    auto version = ParseVersionToken(std::string_view(suffix).substr(last_sep + 1));
    if (version.has_value()) {
      parsed.version = version;
      suffix.resize(last_sep);
    }
  } else {
    auto version = ParseVersionToken(suffix);
    if (version.has_value()) {
      parsed.version = version;
      suffix.clear();
    }
  }

  last_sep = suffix.rfind('_');
  if (last_sep != std::string::npos) {
    std::string role = suffix.substr(last_sep + 1);
    if (IsKnownRole(role)) {
      parsed.role = role;
      suffix.resize(last_sep);
    }
  } else if (!suffix.empty() && IsKnownRole(suffix)) {
    parsed.role = suffix;
    suffix.clear();
  }

  parsed.qualifier = suffix;
  return parsed;
}

inline std::string GetBaseName(const std::string& name) { return Parse(name).base_name; }

inline std::string StripLegacyBaseName(const std::string& name) {
  auto strip_suffix = [](std::string& str, const char* suffix, size_t len) -> bool {
    if (str.size() > len && str.compare(str.size() - len, len, suffix) == 0) {
      str.resize(str.size() - len);
      return true;
    }
    return false;
  };

  auto strip_numeric_suffix = [](std::string& str, char prefix) -> bool {
    size_t pos = str.rfind('_');
    if (pos == std::string::npos || pos == 0 || pos + 1 >= str.size() || str[pos + 1] != prefix ||
        pos + 2 >= str.size()) {
      return false;
    }
    if (!std::all_of(str.begin() + static_cast<ptrdiff_t>(pos + 2), str.end(),
                     [](unsigned char ch) { return std::isdigit(ch); })) {
      return false;
    }
    str.resize(pos);
    return true;
  };

  auto strip_plain_numeric_suffix = [](std::string& str) -> bool {
    size_t pos = str.rfind('_');
    if (pos == std::string::npos || pos == 0 || pos + 1 >= str.size()) {
      return false;
    }
    if (!std::all_of(str.begin() + static_cast<ptrdiff_t>(pos + 1), str.end(),
                     [](unsigned char ch) { return std::isdigit(ch); })) {
      return false;
    }
    str.resize(pos);
    return true;
  };

  std::string current = name;
  bool changed = true;
  while (changed) {
    changed = false;
    if (strip_suffix(current, "_rv", 3) || strip_suffix(current, "_phi", 4) ||
        strip_suffix(current, "_iter", 5) || strip_suffix(current, "_tile", 5) ||
        strip_suffix(current, "_tmp", 4) || strip_suffix(current, "_store", 6) ||
        strip_suffix(current, "_out", 4) || strip_suffix(current, "_idx", 4) ||
        strip_suffix(current, "_outer", 6) || strip_suffix(current, "_inner", 6) ||
        strip_suffix(current, "_rem", 4) || strip_suffix(current, "_ssa", 4) ||
        strip_suffix(current, "_store_ret", 10)) {
      changed = true;
      continue;
    }
    if (strip_numeric_suffix(current, 'l') || strip_numeric_suffix(current, 'v') ||
        strip_plain_numeric_suffix(current)) {
      changed = true;
      continue;
    }
  }
  return current;
}

inline std::string GetCompatibleBaseName(const std::string& name) {
  ParsedName parsed = Parse(name);
  if (parsed.has_auto_suffix) {
    return parsed.base_name;
  }
  return parsed.base_name;
}

inline std::string GetLegacyCompatibleBaseName(const std::string& name) {
  ParsedName parsed = Parse(name);
  if (parsed.has_auto_suffix) {
    return parsed.base_name;
  }
  return StripLegacyBaseName(name);
}

inline std::string ReserveUniqueName(const std::string& base_name, std::set<std::string>& used_names) {
  std::string candidate = base_name;
  int suffix = 0;
  while (!used_names.insert(candidate).second) {
    ++suffix;
    candidate = base_name + "_" + std::to_string(suffix);
  }
  return candidate;
}

template <typename DefNode>
void BuildRenameMapForDefs(const std::vector<const DefNode*>& defs,
                           std::unordered_map<const DefNode*, std::string>& rename_map,
                           bool include_unique_names = false) {
  rename_map.clear();

  std::vector<const DefNode*> unique_defs;
  unique_defs.reserve(defs.size());
  std::unordered_set<const DefNode*> seen_defs;
  for (const DefNode* def : defs) {
    if (seen_defs.insert(def).second) unique_defs.push_back(def);
  }

  std::unordered_map<std::string, int> name_counts;
  for (const DefNode* def : unique_defs) {
    name_counts[def->name_hint_]++;
  }

  // Reserve unique names first so colliding defs never steal them.
  std::set<std::string> used_names;
  for (const DefNode* def : unique_defs) {
    if (name_counts[def->name_hint_] == 1) {
      used_names.insert(def->name_hint_);
      if (include_unique_names) rename_map[def] = def->name_hint_;
    }
  }

  for (const DefNode* def : unique_defs) {
    const std::string& base_name = def->name_hint_;
    if (name_counts[base_name] == 1) continue;
    rename_map[def] = ReserveUniqueName(base_name, used_names);
  }
}

inline std::string AddQualifier(const std::string& name, const std::string& qualifier) {
  ParsedName parsed = Parse(name);
  if (!parsed.has_auto_suffix) {
    return BuildName(parsed.base_name, qualifier);
  }
  std::string combined = qualifier;
  if (!parsed.qualifier.empty()) {
    combined += "_" + parsed.qualifier;
  }
  return BuildName(parsed.base_name, combined, parsed.role, parsed.version);
}

inline std::string ReplaceRole(const std::string& name, const std::string& role) {
  ParsedName parsed = Parse(name);
  return BuildName(parsed.base_name, parsed.qualifier, role, parsed.version);
}

inline std::string WithVersion(const std::string& name, int version) {
  ParsedName parsed = Parse(name);
  return BuildName(parsed.base_name, parsed.qualifier, parsed.role, version);
}

inline std::string BuildFreshVersion(const std::string& base_name, int version,
                                     const std::string& role = "ssa") {
  return BuildName(base_name, "", role, version);
}

inline std::string GenerateFreshNameLike(const std::string& name,
                                         const std::unordered_set<std::string>& used_names) {
  ParsedName parsed = Parse(name);
  std::optional<std::string> role = parsed.role;
  if (!role.has_value()) {
    role = "ssa";
  }
  int next_version = parsed.version.value_or(-1) + 1;
  std::string candidate;
  do {
    candidate = BuildName(parsed.base_name, parsed.qualifier, role, next_version);
    next_version++;
  } while (used_names.count(candidate) > 0);
  return candidate;
}

}  // namespace auto_name
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_AUTO_NAME_UTILS_H_
