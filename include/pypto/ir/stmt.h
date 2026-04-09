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

#ifndef PYPTO_IR_STMT_H_
#define PYPTO_IR_STMT_H_

#include <algorithm>
#include <any>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// Forward-declare Level and Role from function.h to avoid circular include.
// ScopeStmt uses these as optional fields; the full definitions are in function.h.
namespace pypto {
namespace ir {
enum class Level : uint8_t;
enum class Role : uint8_t;
}  // namespace ir
}  // namespace pypto

#include "pypto/core/any_cast.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/reflection/field_traits.h"
#include "pypto/ir/span.h"

namespace pypto {
namespace ir {

// Forward declarations for friend classes
class IRVisitor;
class IRMutator;

/**
 * @brief Distinguishes sequential, parallel, and unroll for loops
 */
enum class ForKind : uint8_t {
  Sequential = 0,  ///< Standard sequential for loop (default)
  Parallel = 1,    ///< Parallel for loop
  Unroll = 2       ///< Compile-time unrolled for loop
};

/**
 * @brief Chunk policy for loop chunking
 *
 * Controls how iterations are distributed across chunks.
 */
enum class ChunkPolicy : uint8_t {
  LeadingFull = 0  ///< Full chunks first, smaller remainder at end
};

/**
 * @brief Convert ChunkPolicy to string
 */
inline std::string ChunkPolicyToString(ChunkPolicy policy) {
  switch (policy) {
    case ChunkPolicy::LeadingFull:
      return "LeadingFull";
    default:
      INTERNAL_CHECK(false) << "Unknown ChunkPolicy: " << static_cast<int>(policy);
      return "";  // Unreachable
  }
}

/**
 * @brief Convert string to ChunkPolicy
 */
inline ChunkPolicy StringToChunkPolicy(const std::string& str) {
  if (str == "LeadingFull" || str == "leading_full") {
    return ChunkPolicy::LeadingFull;
  }
  throw pypto::TypeError("Unknown ChunkPolicy: " + str);
}

/**
 * @brief Chunk configuration for parallel loop splitting.
 *
 * Groups chunk_size and chunk_policy which always appear together.
 * Only meaningful on parallel (chunked) loops.
 */
struct ChunkConfig {
  ExprPtr size;                                   ///< Chunk size expression
  ChunkPolicy policy = ChunkPolicy::LeadingFull;  ///< Distribution policy
};

/**
 * @brief Loop origin classification for tracking how a loop was generated
 *
 * Used by SplitChunkedLoops to tag each generated loop with its origin.
 */
enum class LoopOrigin : uint8_t {
  Original = 0,       ///< Regular loop (default)
  ChunkOuter = 1,     ///< Outer loop from chunk splitting
  ChunkInner = 2,     ///< Inner loop from chunk splitting
  ChunkRemainder = 3  ///< Remainder loop from chunk splitting
};

/**
 * @brief Convert LoopOrigin to string
 */
inline std::string LoopOriginToString(LoopOrigin origin) {
  switch (origin) {
    case LoopOrigin::Original:
      return "Original";
    case LoopOrigin::ChunkOuter:
      return "ChunkOuter";
    case LoopOrigin::ChunkInner:
      return "ChunkInner";
    case LoopOrigin::ChunkRemainder:
      return "ChunkRemainder";
    default:
      INTERNAL_CHECK(false) << "Unknown LoopOrigin: " << static_cast<int>(origin);
      return "";  // Unreachable
  }
}

/**
 * @brief Convert string to LoopOrigin
 */
inline LoopOrigin StringToLoopOrigin(const std::string& str) {
  if (str == "Original") {
    return LoopOrigin::Original;
  } else if (str == "ChunkOuter") {
    return LoopOrigin::ChunkOuter;
  } else if (str == "ChunkInner") {
    return LoopOrigin::ChunkInner;
  } else if (str == "ChunkRemainder") {
    return LoopOrigin::ChunkRemainder;
  } else {
    throw pypto::TypeError("Unknown LoopOrigin: " + str);
  }
}

/**
 * @brief Distinguishes different scope kinds
 */
enum class ScopeKind : uint8_t {
  InCore = 0,      ///< InCore scope for AICore sub-graphs
  AutoInCore = 1,  ///< AutoInCore scope for automatic chunking
  Cluster = 2,     ///< Cluster scope for co-scheduled AIC + AIV groups
  Hierarchy = 3    ///< Distributed hierarchy scope (uses level_/role_ on ScopeStmt)
};

/**
 * @brief Split mode for cross-core data transfer
 *
 * Controls how tile data is split when transferred between AIC and AIV cores:
 * - None: No splitting, full tile transferred
 * - UpDown: Split vertically (height halved, width unchanged)
 * - LeftRight: Split horizontally (height unchanged, width halved)
 */
enum class SplitMode : uint8_t {
  None = 0,       ///< No split
  UpDown = 1,     ///< Split vertically (height halved)
  LeftRight = 2,  ///< Split horizontally (width halved)
};

/**
 * @brief Convert SplitMode to string
 * @param mode The split mode
 * @return String representation ("None", "UpDown", or "LeftRight")
 */
inline std::string SplitModeToString(SplitMode mode) {
  switch (mode) {
    case SplitMode::None:
      return "None";
    case SplitMode::UpDown:
      return "UpDown";
    case SplitMode::LeftRight:
      return "LeftRight";
  }
  throw pypto::TypeError("Unknown SplitMode");
}

/**
 * @brief Convert string to SplitMode
 * @param str String representation
 * @return SplitMode enum value
 * @throws pypto::TypeError if string is not recognized
 */
inline SplitMode StringToSplitMode(const std::string& str) {
  if (str == "None") {
    return SplitMode::None;
  } else if (str == "UpDown") {
    return SplitMode::UpDown;
  } else if (str == "LeftRight") {
    return SplitMode::LeftRight;
  } else {
    throw pypto::TypeError("Unknown SplitMode: " + str);
  }
}

/**
 * @brief Convert ForKind to string
 * @param kind The for loop kind
 * @return String representation ("Sequential", "Parallel", or "Unroll")
 */
inline std::string ForKindToString(ForKind kind) {
  switch (kind) {
    case ForKind::Sequential:
      return "Sequential";
    case ForKind::Parallel:
      return "Parallel";
    case ForKind::Unroll:
      return "Unroll";
  }
  throw pypto::TypeError("Unknown ForKind");
}

/**
 * @brief Convert string to ForKind
 * @param str String representation
 * @return ForKind enum value
 * @throws pypto::TypeError if string is not recognized
 */
inline ForKind StringToForKind(const std::string& str) {
  if (str == "Sequential") {
    return ForKind::Sequential;
  } else if (str == "Parallel") {
    return ForKind::Parallel;
  } else if (str == "Unroll") {
    return ForKind::Unroll;
  } else {
    throw pypto::TypeError("Unknown ForKind: " + str);
  }
}

/**
 * @brief Convert ScopeKind to string
 * @param kind The scope kind
 * @return String representation ("InCore", "AutoInCore", or "Cluster")
 */
inline std::string ScopeKindToString(ScopeKind kind) {
  switch (kind) {
    case ScopeKind::InCore:
      return "InCore";
    case ScopeKind::AutoInCore:
      return "AutoInCore";
    case ScopeKind::Cluster:
      return "Cluster";
    case ScopeKind::Hierarchy:
      return "Hierarchy";
  }
  throw pypto::TypeError("Unknown ScopeKind");
}

/**
 * @brief Convert string to ScopeKind
 * @param str String representation
 * @return ScopeKind enum value
 * @throws pypto::TypeError if string is not recognized
 */
inline ScopeKind StringToScopeKind(const std::string& str) {
  if (str == "InCore") {
    return ScopeKind::InCore;
  } else if (str == "AutoInCore") {
    return ScopeKind::AutoInCore;
  } else if (str == "Cluster") {
    return ScopeKind::Cluster;
  } else if (str == "Hierarchy") {
    return ScopeKind::Hierarchy;
  } else {
    throw pypto::TypeError("Unknown ScopeKind: " + str);
  }
}

/**
 * @brief Base class for all statements in the IR
 *
 * Statements represent operations that perform side effects or control flow.
 * All statements are immutable.
 */
class Stmt : public IRNode {
 public:
  /**
   * @brief Create a statement
   *
   * @param span Source location
   */
  explicit Stmt(Span s) : IRNode(std::move(s)) {}
  ~Stmt() override = default;

  /**
   * @brief Get the type name of this statement
   *
   * @return Human-readable type name (e.g., "Stmt", "Assign", "Return")
   */
  [[nodiscard]] std::string TypeName() const override { return "Stmt"; }

  static constexpr auto GetFieldDescriptors() { return IRNode::GetFieldDescriptors(); }
};

using StmtPtr = std::shared_ptr<const Stmt>;

/**
 * @brief Assignment statement
 *
 * Represents an assignment operation: var = value
 * where var is a variable and value is an expression.
 */
class AssignStmt : public Stmt {
 public:
  VarPtr var_;     // Variable
  ExprPtr value_;  // Expression

  /**
   * @brief Create an assignment statement
   *
   * @param var Variable
   * @param value Expression
   * @param span Source location
   */
  AssignStmt(VarPtr var, ExprPtr value, Span span)
      : Stmt(std::move(span)), var_(std::move(var)), value_(std::move(value)) {}

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::AssignStmt; }
  [[nodiscard]] std::string TypeName() const override { return "AssignStmt"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (var and value as DEF and USUAL fields)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Stmt::GetFieldDescriptors(),
                          std::make_tuple(reflection::DefField(&AssignStmt::var_, "var"),
                                          reflection::UsualField(&AssignStmt::value_, "value")));
  }
};

using AssignStmtPtr = std::shared_ptr<const AssignStmt>;

/**
 * @brief Conditional statement
 *
 * Represents an if-else statement: if condition then then_body else else_body
 * where condition is an expression and then_body/else_body is statement.
 */
class IfStmt : public Stmt {
 public:
  /**
   * @brief Create a conditional statement with then and else branches
   *
   * @param condition Condition expression
   * @param then_body Then branch statement
   * @param else_body Else branch statement (can be optional)
   * @param return_vars Return variables (can be empty)
   * @param span Source location
   */
  IfStmt(ExprPtr condition, StmtPtr then_body, std::optional<StmtPtr> else_body,
         std::vector<VarPtr> return_vars, Span span)
      : Stmt(std::move(span)),
        condition_(std::move(condition)),
        then_body_(std::move(then_body)),
        else_body_(std::move(else_body)),
        return_vars_(std::move(return_vars)) {}

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::IfStmt; }
  [[nodiscard]] std::string TypeName() const override { return "IfStmt"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (condition, then_body, else_body as USUAL fields)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Stmt::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&IfStmt::condition_, "condition"),
                                          reflection::UsualField(&IfStmt::then_body_, "then_body"),
                                          reflection::UsualField(&IfStmt::else_body_, "else_body"),
                                          reflection::DefField(&IfStmt::return_vars_, "return_vars")));
  }

 public:
  ExprPtr condition_;                 // Condition expression
  StmtPtr then_body_;                 // Then branch statement
  std::optional<StmtPtr> else_body_;  // Else branch statement (optional)
  std::vector<VarPtr> return_vars_;   // Return variables (can be empty)
};

using IfStmtPtr = std::shared_ptr<const IfStmt>;

/**
 * @brief Yield statement
 *
 * Represents a yield operation: yield value
 * where value is a list of variables to yield.
 */
class YieldStmt : public Stmt {
 public:
  /**
   * @brief Create a yield statement
   *
   * @param value List of variables to yield (can be empty)
   * @param span Source location
   */
  YieldStmt(std::vector<ExprPtr> value, Span span) : Stmt(std::move(span)), value_(std::move(value)) {}

  /**
   * @brief Create a yield statement without values
   *
   * @param span Source location
   */
  explicit YieldStmt(Span span) : Stmt(std::move(span)), value_() {}

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::YieldStmt; }
  [[nodiscard]] std::string TypeName() const override { return "YieldStmt"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (value as USUAL field)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Stmt::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&YieldStmt::value_, "value")));
  }

 public:
  std::vector<ExprPtr> value_;  // List of expressions to yield
};

using YieldStmtPtr = std::shared_ptr<const YieldStmt>;

/**
 * @brief Return statement
 *
 * Represents a return operation: return value
 * where value is a list of expressions to return.
 */
class ReturnStmt : public Stmt {
 public:
  /**
   * @brief Create a return statement
   *
   * @param value List of expressions to return (can be empty)
   * @param span Source location
   */
  ReturnStmt(std::vector<ExprPtr> value, Span span) : Stmt(std::move(span)), value_(std::move(value)) {}

  /**
   * @brief Create a return statement without values
   *
   * @param span Source location
   */
  explicit ReturnStmt(Span span) : Stmt(std::move(span)), value_() {}

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ReturnStmt; }
  [[nodiscard]] std::string TypeName() const override { return "ReturnStmt"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (value as USUAL field)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Stmt::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&ReturnStmt::value_, "value")));
  }

 public:
  std::vector<ExprPtr> value_;  // List of expressions to return
};

using ReturnStmtPtr = std::shared_ptr<const ReturnStmt>;

/**
 * @brief For loop statement
 *
 * Represents a for loop with optional loop-carried values (SSA-style iteration).
 *
 * **Basic loop:** for loop_var in range(start, stop, step): body
 *
 * **Loop with iteration arguments:**
 * for loop_var, (iter_arg1, iter_arg2) in pl.range(start, stop, step, init_values=(...)):
 *     iter_arg1, iter_arg2 = pl.yield_(new_val1, new_val2)
 * return_var1 = iter_arg1
 * return_var2 = iter_arg2
 *
 * **Key Relationships:**
 * - iter_args: IterArg variables scoped to loop body, carry values between iterations
 * - return_vars: Var variables that capture final iteration values, accessible after loop
 * - Number of iter_args must equal number of return_vars
 * - Number of yielded values must equal number of iter_args
 * - IterArgs cannot be directly accessed outside the loop; use return_vars instead
 */
class ForStmt : public Stmt {
 public:
  /**
   * @brief Create a for loop statement
   *
   * @param loop_var Loop variable
   * @param start Start value expression
   * @param stop Stop value expression
   * @param step Step value expression
   * @param iter_args Iteration arguments (loop-carried values, scoped to loop body)
   * @param body Loop body statement (must yield values matching iter_args if non-empty)
   * @param return_vars Return variables (capture final values, accessible after loop)
   * @param span Source location
   * @param kind Loop kind (Sequential, Parallel, or Unroll; default: Sequential)
   * @param chunk_config Optional chunk configuration (nullopt = no chunking)
   * @param attrs Loop-level attributes (key-value metadata, default: empty)
   */
  ForStmt(VarPtr loop_var, ExprPtr start, ExprPtr stop, ExprPtr step, std::vector<IterArgPtr> iter_args,
          StmtPtr body, std::vector<VarPtr> return_vars, Span span, ForKind kind = ForKind::Sequential,
          std::optional<ChunkConfig> chunk_config = std::nullopt,
          std::vector<std::pair<std::string, std::any>> attrs = {})
      : Stmt(std::move(span)),
        loop_var_(std::move(loop_var)),
        start_(std::move(start)),
        stop_(std::move(stop)),
        step_(std::move(step)),
        iter_args_(std::move(iter_args)),
        body_(std::move(body)),
        return_vars_(std::move(return_vars)),
        kind_(kind),
        chunk_config_(std::move(chunk_config)),
        attrs_(std::move(attrs)) {}

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ForStmt; }
  [[nodiscard]] std::string TypeName() const override { return "ForStmt"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (loop_var as DEF field, others as USUAL fields)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Stmt::GetFieldDescriptors(),
                          std::make_tuple(reflection::DefField(&ForStmt::loop_var_, "loop_var"),
                                          reflection::UsualField(&ForStmt::start_, "start"),
                                          reflection::UsualField(&ForStmt::stop_, "stop"),
                                          reflection::UsualField(&ForStmt::step_, "step"),
                                          reflection::DefField(&ForStmt::iter_args_, "iter_args"),
                                          reflection::UsualField(&ForStmt::body_, "body"),
                                          reflection::DefField(&ForStmt::return_vars_, "return_vars"),
                                          reflection::UsualField(&ForStmt::kind_, "kind"),
                                          reflection::UsualField(&ForStmt::chunk_config_, "chunk_config"),
                                          reflection::UsualField(&ForStmt::attrs_, "attrs")));
  }

 public:
  VarPtr loop_var_;                    // Loop variable (e.g., i in "for i in range(...)")
  ExprPtr start_;                      // Start value expression
  ExprPtr stop_;                       // Stop value expression
  ExprPtr step_;                       // Step value expression
  std::vector<IterArgPtr> iter_args_;  // Loop-carried values (scoped to loop body)
  StmtPtr body_;                       // Loop body statement (must yield if iter_args non-empty)
  std::vector<VarPtr> return_vars_;    // Variables capturing final iteration values (accessible after loop)
  ForKind kind_;                       // Loop kind (Sequential, Parallel, or Unroll)
  std::optional<ChunkConfig> chunk_config_;              // Chunk configuration (nullopt = no chunking)
  std::vector<std::pair<std::string, std::any>> attrs_;  // Loop-level attributes (key-value metadata)

  /// Get a typed attribute value (returns default_value if key not found)
  template <typename T>
  [[nodiscard]] T GetAttr(const std::string& key, const T& default_value = T{}) const {
    for (const auto& [k, v] : attrs_) {
      if (k == key) return AnyCast<T>(v, "for_stmt attr key: " + key);
    }
    return default_value;
  }

  /// Check if an attribute exists
  [[nodiscard]] bool HasAttr(const std::string& key) const {
    return std::any_of(attrs_.begin(), attrs_.end(), [&key](const auto& pair) { return pair.first == key; });
  }

  /// Get all attributes
  [[nodiscard]] const std::vector<std::pair<std::string, std::any>>& GetAttrs() const { return attrs_; }
};

using ForStmtPtr = std::shared_ptr<const ForStmt>;

/**
 * @brief While loop statement
 *
 * Represents a while loop with optional loop-carried state (iter_args) and return variables.
 *
 * **Syntax:**
 * Natural form (non-SSA, parsed from user code):
 *   while condition:
 *       body
 *
 * SSA form (after ConvertToSSA or explicit Python DSL):
 *   for iter_args in pl.while_(init_values=(...)):
 *       with pl.cond(condition):
 *           body
 *           iter_args = pl.yield_(new_values)
 *   return_vars = iter_args
 *
 * Note: The Python surface syntax for pl.while_ does not accept a positional
 * condition argument; conditions must be expressed via pl.cond(...) inside
 * the loop body as shown above.
 *
 * **Semantics:**
 * - Each iteration: evaluate condition using current iter_arg values (via pl.cond)
 * - If condition is true, execute body
 * - Body ends with YieldStmt feeding next iteration
 * - When condition is false, return_vars get final iter_arg values
 *
 * **Key Relationships:**
 * - condition: Boolean expression evaluated each iteration using current iter_args
 * - iter_args: IterArg variables scoped to loop body, carry values between iterations
 * - return_vars: Var variables that capture final iteration values, accessible after loop
 * - Number of iter_args must equal number of return_vars
 * - Number of yielded values must equal number of iter_args
 */
class WhileStmt : public Stmt {
 public:
  /**
   * @brief Create a while loop statement
   *
   * @param condition Boolean condition expression
   * @param iter_args Iteration arguments (loop-carried values, scoped to loop body)
   * @param body Loop body statement (must yield values matching iter_args if non-empty)
   * @param return_vars Return variables (capture final values, accessible after loop)
   * @param span Source location
   */
  WhileStmt(ExprPtr condition, std::vector<IterArgPtr> iter_args, StmtPtr body,
            std::vector<VarPtr> return_vars, Span span)
      : Stmt(std::move(span)),
        condition_(std::move(condition)),
        iter_args_(std::move(iter_args)),
        body_(std::move(body)),
        return_vars_(std::move(return_vars)) {}

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::WhileStmt; }
  [[nodiscard]] std::string TypeName() const override { return "WhileStmt"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (condition as USUAL, iter_args as DEF, body as USUAL, return_vars as
   * DEF)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Stmt::GetFieldDescriptors(),
                          std::make_tuple(reflection::DefField(&WhileStmt::iter_args_, "iter_args"),
                                          reflection::UsualField(&WhileStmt::condition_, "condition"),
                                          reflection::UsualField(&WhileStmt::body_, "body"),
                                          reflection::DefField(&WhileStmt::return_vars_, "return_vars")));
  }

 public:
  ExprPtr condition_;                  // Condition expression (evaluated each iteration)
  std::vector<IterArgPtr> iter_args_;  // Loop-carried values (scoped to loop body)
  StmtPtr body_;                       // Loop body statement (must yield if iter_args non-empty)
  std::vector<VarPtr> return_vars_;    // Variables capturing final iteration values (accessible after loop)
};

using WhileStmtPtr = std::shared_ptr<const WhileStmt>;

/**
 * @brief Scope statement
 *
 * Represents a scoped region of code with a specific execution context.
 * This is NOT a control flow node — it executes its body exactly once, linearly.
 *
 * **Syntax:**
 * with pl.incore():    # InCore scope
 *     body
 * with pl.cluster():   # Cluster scope
 *     body
 * with pl.at(level=pl.Level.HOST, role=pl.Role.Worker):  # Hierarchy scope
 *     body
 *
 * **Semantics:**
 * - Marks a region of code as belonging to a specific scope (e.g., InCore, Cluster)
 * - Executes body exactly once (no iteration, no branching)
 * - Variables flow through transparently (no iter_args/return_vars needed)
 * - SSA conversion treats it as transparent (just visits body)
 * - OutlineIncoreScopes extracts InCore scopes into InCore functions
 * - OutlineClusterScopes extracts Cluster scopes into Group functions
 * - Hierarchy scopes (planned) will be outlined into level-/role-annotated functions
 *
 * **Key Properties:**
 * - scope_kind: The kind of scope (InCore, AutoInCore, Cluster, or Hierarchy)
 * - body: The nested statements to execute within this scope
 * - level: (Hierarchy only) Target hierarchy level
 * - role: (Hierarchy only) Function role (Orchestrator or Worker)
 */
class ScopeStmt : public Stmt {
 public:
  /**
   * @brief Create a scope statement (legacy 3-arg constructor)
   *
   * @param scope_kind The kind of scope
   * @param body The nested statements
   * @param span Source location
   */
  ScopeStmt(ScopeKind scope_kind, StmtPtr body, Span span)
      : Stmt(std::move(span)), scope_kind_(scope_kind), body_(std::move(body)) {}

  /**
   * @brief Create a scope statement with hierarchy level, role, and split mode
   *
   * @param scope_kind The kind of scope
   * @param body The nested statements
   * @param span Source location
   * @param level Hierarchy level (for Hierarchy scopes)
   * @param role Function role (for Hierarchy scopes)
   * @param split Split mode for cross-core transfer (for AutoInCore scopes)
   */
  ScopeStmt(ScopeKind scope_kind, StmtPtr body, Span span, std::optional<Level> level,
            std::optional<Role> role = std::nullopt, std::optional<SplitMode> split = std::nullopt)
      : Stmt(std::move(span)),
        scope_kind_(scope_kind),
        body_(std::move(body)),
        level_(level),
        role_(role),
        split_(split) {
    CHECK(scope_kind != ScopeKind::Hierarchy || level_.has_value()) << "Hierarchy scope requires a level";
  }

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ScopeStmt; }
  [[nodiscard]] std::string TypeName() const override { return "ScopeStmt"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (scope_kind, level, role, split, and body as USUAL fields)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Stmt::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&ScopeStmt::scope_kind_, "scope_kind"),
                                          reflection::UsualField(&ScopeStmt::level_, "level"),
                                          reflection::UsualField(&ScopeStmt::role_, "role"),
                                          reflection::UsualField(&ScopeStmt::split_, "split"),
                                          reflection::UsualField(&ScopeStmt::body_, "body")));
  }

 public:
  ScopeKind scope_kind_;            // The kind of scope (e.g., InCore, Hierarchy)
  std::optional<Level> level_;      // Hierarchy level (nullopt for non-Hierarchy scopes)
  std::optional<Role> role_;        // Function role (nullopt for non-Hierarchy scopes)
  std::optional<SplitMode> split_;  // Split mode (nullopt or None for no split)
  StmtPtr body_;                    // The nested statements
};

using ScopeStmtPtr = std::shared_ptr<const ScopeStmt>;

/**
 * @brief Sequence of statements
 *
 * Represents a sequence of statements: stmt1; stmt2; ... stmtN
 * where stmts is a list of statements.
 */
class SeqStmts : public Stmt {
 public:
  /**
   * @brief Create a sequence of statements
   *
   * @param stmts List of statements
   * @param span Source location
   */
  SeqStmts(std::vector<StmtPtr> stmts, Span span) : Stmt(std::move(span)), stmts_(std::move(stmts)) {}

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::SeqStmts; }
  [[nodiscard]] std::string TypeName() const override { return "SeqStmts"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (stmts as USUAL field)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Stmt::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&SeqStmts::stmts_, "stmts")));
  }

  /**
   * @brief Create a normalized statement from a list of statements
   *
   * Flattens nested SeqStmts and unwraps single-child sequences:
   * - Flatten({a, SeqStmts({b, c}), d}, span) → SeqStmts({a, b, c, d})
   * - Flatten({a}, span) → a
   * - Flatten({}, span) → SeqStmts({})
   */
  static StmtPtr Flatten(std::vector<StmtPtr> stmts, Span span) {
    std::vector<StmtPtr> flat;
    for (auto& s : stmts) {
      if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(s)) {
        // Recursively flatten nested SeqStmts
        for (const auto& inner : seq->stmts_) {
          if (auto inner_seq = std::dynamic_pointer_cast<const SeqStmts>(inner)) {
            flat.insert(flat.end(), inner_seq->stmts_.begin(), inner_seq->stmts_.end());
          } else {
            flat.push_back(inner);
          }
        }
      } else {
        flat.push_back(std::move(s));
      }
    }
    if (flat.size() == 1) {
      return flat[0];
    }
    return std::make_shared<SeqStmts>(std::move(flat), std::move(span));
  }

 public:
  std::vector<StmtPtr> stmts_;  // List of statements
};

using SeqStmtsPtr = std::shared_ptr<const SeqStmts>;

/**
 * @brief Evaluation statement
 *
 * Represents an expression executed as a statement: expr
 * where expr is an expression (typically a Call).
 * This is used for expressions that have side effects but no return value
 * (or return value is ignored).
 */
class EvalStmt : public Stmt {
 public:
  /**
   * @brief Create an evaluation statement
   *
   * @param expr Expression to execute
   * @param span Source location
   */
  EvalStmt(ExprPtr expr, Span span) : Stmt(std::move(span)), expr_(std::move(expr)) {}

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::EvalStmt; }
  [[nodiscard]] std::string TypeName() const override { return "EvalStmt"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (expr as USUAL field)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Stmt::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&EvalStmt::expr_, "expr")));
  }

 public:
  ExprPtr expr_;  // Expression
};

using EvalStmtPtr = std::shared_ptr<const EvalStmt>;

/**
 * @brief Break statement
 *
 * Represents a break statement used to exit a loop: break
 */
class BreakStmt : public Stmt {
 public:
  explicit BreakStmt(Span span) : Stmt(std::move(span)) {}

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::BreakStmt; }
  [[nodiscard]] std::string TypeName() const override { return "BreakStmt"; }

  static constexpr auto GetFieldDescriptors() { return Stmt::GetFieldDescriptors(); }
};

using BreakStmtPtr = std::shared_ptr<const BreakStmt>;

/**
 * @brief Continue statement
 *
 * Represents a continue statement used to skip to the next loop iteration: continue
 */
class ContinueStmt : public Stmt {
 public:
  explicit ContinueStmt(Span span) : Stmt(std::move(span)) {}

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ContinueStmt; }
  [[nodiscard]] std::string TypeName() const override { return "ContinueStmt"; }

  static constexpr auto GetFieldDescriptors() { return Stmt::GetFieldDescriptors(); }
};

using ContinueStmtPtr = std::shared_ptr<const ContinueStmt>;

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_STMT_H_
