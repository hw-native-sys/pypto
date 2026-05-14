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
 * @brief Distinguishes sequential, parallel, unroll, and pipeline for loops
 */
enum class ForKind : uint8_t {
  Sequential = 0,  ///< Standard sequential for loop (default)
  Parallel = 1,    ///< Parallel for loop
  Unroll = 2,      ///< Compile-time unrolled for loop
  Pipeline = 3     ///< Software-pipelined loop (lowered by LowerPipelineLoops; transient marker kept through
                   ///< CanonicalizeIOOrder)
};

/**
 * @brief Chunk policy for loop chunking
 *
 * Controls how iterations are distributed across chunks.
 */
enum class ChunkPolicy : uint8_t {
  LeadingFull = 0,  ///< Full chunks first, smaller remainder at end (splits into two kernels)
  Guarded = 1       ///< Single loop over ceil(N/C) chunks with per-iteration if-guard (default)
};

/**
 * @brief Convert ChunkPolicy to string
 */
inline std::string ChunkPolicyToString(ChunkPolicy policy) {
  switch (policy) {
    case ChunkPolicy::LeadingFull:
      return "LeadingFull";
    case ChunkPolicy::Guarded:
      return "Guarded";
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
  if (str == "Guarded" || str == "guarded") {
    return ChunkPolicy::Guarded;
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
  ExprPtr size;                               ///< Chunk size expression
  ChunkPolicy policy = ChunkPolicy::Guarded;  ///< Distribution policy (default: Guarded)
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
  Hierarchy = 3,   ///< Distributed hierarchy scope (uses level_/role_ on ScopeStmt)
  Spmd = 4,        ///< SPMD dispatch scope (core_num/sync_start on ScopeStmt)
  Runtime = 5      ///< Runtime orchestration scope (PTO2_SCOPE wrapper, manual on/off)
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
 * @return String representation ("Sequential", "Parallel", "Unroll", or "Pipeline")
 */
inline std::string ForKindToString(ForKind kind) {
  switch (kind) {
    case ForKind::Sequential:
      return "Sequential";
    case ForKind::Parallel:
      return "Parallel";
    case ForKind::Unroll:
      return "Unroll";
    case ForKind::Pipeline:
      return "Pipeline";
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
  } else if (str == "Pipeline") {
    return ForKind::Pipeline;
  } else {
    throw pypto::TypeError("Unknown ForKind: " + str);
  }
}

/**
 * @brief Convert ScopeKind to string
 * @param kind The scope kind
 * @return String representation ("InCore", "AutoInCore", "Cluster", "Hierarchy", "Spmd", or "Runtime")
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
    case ScopeKind::Spmd:
      return "Spmd";
    case ScopeKind::Runtime:
      return "Runtime";
  }
  throw pypto::TypeError("Unknown ScopeKind");
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
   * @param leading_comments Source-level comments to print above this statement (defaults to empty)
   */
  explicit Stmt(Span s, std::vector<std::string> leading_comments = {})
      : IRNode(std::move(s)), leading_comments_(std::move(leading_comments)) {}
  ~Stmt() override = default;

  /**
   * @brief Get the type name of this statement
   *
   * @return Human-readable type name (e.g., "Stmt", "Assign", "Return")
   */
  [[nodiscard]] std::string TypeName() const override { return "Stmt"; }

  // Source-level comments printed above this statement.
  //
  // Registered as IgnoreField so it does NOT participate in structural equality
  // or hashing (purely DSL-level annotation metadata). It IS serialized/deserialized
  // so comments survive `serialize_to_file` round-trips. Passed through the Stmt
  // constructor (symmetric with span_). Exposed to Python as read-only; the
  // late-binding mutation channel is AttachLeadingComments (used by the parser
  // builder and comment-merging passes).
  std::vector<std::string> leading_comments_;

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(IRNode::GetFieldDescriptors(), std::make_tuple(reflection::IgnoreField(
                                                             &Stmt::leading_comments_, "leading_comments")));
  }
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
  AssignStmt(VarPtr var, ExprPtr value, Span span, std::vector<std::string> leading_comments = {})
      : Stmt(std::move(span), std::move(leading_comments)), var_(std::move(var)), value_(std::move(value)) {}

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
         std::vector<VarPtr> return_vars, Span span, std::vector<std::string> leading_comments = {})
      : Stmt(std::move(span), std::move(leading_comments)),
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
  YieldStmt(std::vector<ExprPtr> value, Span span, std::vector<std::string> leading_comments = {})
      : Stmt(std::move(span), std::move(leading_comments)), value_(std::move(value)) {}

  /**
   * @brief Create a yield statement without values
   *
   * @param span Source location
   */
  explicit YieldStmt(Span span, std::vector<std::string> leading_comments = {})
      : Stmt(std::move(span), std::move(leading_comments)), value_() {}

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
  ReturnStmt(std::vector<ExprPtr> value, Span span, std::vector<std::string> leading_comments = {})
      : Stmt(std::move(span), std::move(leading_comments)), value_(std::move(value)) {}

  /**
   * @brief Create a return statement without values
   *
   * @param span Source location
   */
  explicit ReturnStmt(Span span, std::vector<std::string> leading_comments = {})
      : Stmt(std::move(span), std::move(leading_comments)), value_() {}

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
          std::vector<std::pair<std::string, std::any>> attrs = {},
          std::vector<std::string> leading_comments = {})
      : Stmt(std::move(span), std::move(leading_comments)),
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
            std::vector<VarPtr> return_vars, Span span, std::vector<std::string> leading_comments = {})
      : Stmt(std::move(span), std::move(leading_comments)),
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
 * @brief Scope statement (abstract base — see derived classes below)
 *
 * Represents a scoped region of code with a specific execution context.
 * This is NOT a control flow node — it executes its body exactly once, linearly.
 *
 * **Class hierarchy** (issue #1047):
 * - `ScopeStmt` (abstract): common fields `name_hint_`, `body_`
 *   - `InCoreScopeStmt`: optional `split_`
 *   - `AutoInCoreScopeStmt`: optional `split_`
 *   - `ClusterScopeStmt`: no extra fields
 *   - `HierarchyScopeStmt`: required `level_`, optional `role_`
 *   - `SpmdScopeStmt`: required `core_num_`, `sync_start_` (default false)
 *
 * **Syntax:**
 * with pl.incore():    # InCore scope -> InCoreScopeStmt
 *     body
 * with pl.cluster():   # Cluster scope -> ClusterScopeStmt
 *     body
 * with pl.at(level=pl.Level.HOST, role=pl.Role.SubWorker):  # -> HierarchyScopeStmt
 *     body
 * with pl.spmd(8):           # -> SpmdScopeStmt
 *     body
 * for i in pl.spmd(8):       # -> SpmdScopeStmt(body=InCoreScopeStmt(...))
 *     body
 *
 * **Semantics:**
 * - Marks a region of code as belonging to a specific scope (e.g., InCore, Cluster)
 * - Executes body exactly once (no iteration, no branching)
 * - Variables flow through transparently (no iter_args/return_vars needed)
 * - SSA conversion treats it as transparent (just visits body)
 * - OutlineIncoreScopes extracts InCore scopes into InCore functions
 * - OutlineClusterScopes extracts Cluster scopes into Group functions
 * - Hierarchy scopes are outlined into level-/role-annotated functions
 */
class ScopeStmt : public Stmt {
 public:
  ScopeStmt(std::string name_hint, StmtPtr body, Span span, std::vector<std::string> leading_comments = {})
      : Stmt(std::move(span), std::move(leading_comments)),
        name_hint_(std::move(name_hint)),
        body_(std::move(body)) {}

  /// Each derived class returns its ScopeKind. Used for switch-style dispatch.
  [[nodiscard]] virtual ScopeKind GetScopeKind() const = 0;

  [[nodiscard]] std::string TypeName() const override { return "ScopeStmt"; }

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Stmt::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&ScopeStmt::name_hint_, "name_hint"),
                                          reflection::UsualField(&ScopeStmt::body_, "body")));
  }

 public:
  std::string name_hint_;  // User-provided scope name hint (empty = auto-generate)
  StmtPtr body_;           // The nested statements
};

using ScopeStmtPtr = std::shared_ptr<const ScopeStmt>;

/**
 * @brief InCore scope: AICore sub-graph region.
 *
 * Carries an optional `split` for cross-core transfer mode.
 */
class InCoreScopeStmt : public ScopeStmt {
 public:
  InCoreScopeStmt(std::optional<SplitMode> split, std::string name_hint, StmtPtr body, Span span,
                  std::vector<std::string> leading_comments = {})
      : ScopeStmt(std::move(name_hint), std::move(body), std::move(span), std::move(leading_comments)),
        split_(split) {}

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::InCoreScopeStmt; }
  [[nodiscard]] ScopeKind GetScopeKind() const override { return ScopeKind::InCore; }
  [[nodiscard]] std::string TypeName() const override { return "InCoreScopeStmt"; }

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(ScopeStmt::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&InCoreScopeStmt::split_, "split")));
  }

 public:
  std::optional<SplitMode> split_;  // Split mode (nullopt or None for no split)
};

using InCoreScopeStmtPtr = std::shared_ptr<const InCoreScopeStmt>;

/**
 * @brief AutoInCore scope: InCore region with automatic chunking.
 *
 * Carries an optional `split` for cross-core transfer mode.
 */
class AutoInCoreScopeStmt : public ScopeStmt {
 public:
  AutoInCoreScopeStmt(std::optional<SplitMode> split, std::string name_hint, StmtPtr body, Span span,
                      std::vector<std::string> leading_comments = {})
      : ScopeStmt(std::move(name_hint), std::move(body), std::move(span), std::move(leading_comments)),
        split_(split) {}

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::AutoInCoreScopeStmt; }
  [[nodiscard]] ScopeKind GetScopeKind() const override { return ScopeKind::AutoInCore; }
  [[nodiscard]] std::string TypeName() const override { return "AutoInCoreScopeStmt"; }

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(ScopeStmt::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&AutoInCoreScopeStmt::split_, "split")));
  }

 public:
  std::optional<SplitMode> split_;  // Split mode (nullopt or None for no split)
};

using AutoInCoreScopeStmtPtr = std::shared_ptr<const AutoInCoreScopeStmt>;

/**
 * @brief Cluster scope: co-scheduled AIC + AIV group.
 *
 * No kind-specific fields; only inherits `name_hint_` and `body_` from base.
 */
class ClusterScopeStmt : public ScopeStmt {
 public:
  ClusterScopeStmt(std::string name_hint, StmtPtr body, Span span,
                   std::vector<std::string> leading_comments = {})
      : ScopeStmt(std::move(name_hint), std::move(body), std::move(span), std::move(leading_comments)) {}

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ClusterScopeStmt; }
  [[nodiscard]] ScopeKind GetScopeKind() const override { return ScopeKind::Cluster; }
  [[nodiscard]] std::string TypeName() const override { return "ClusterScopeStmt"; }

  static constexpr auto GetFieldDescriptors() { return ScopeStmt::GetFieldDescriptors(); }
};

using ClusterScopeStmtPtr = std::shared_ptr<const ClusterScopeStmt>;

/**
 * @brief Hierarchy scope: distributed-hierarchy region.
 *
 * Required `level`, optional `role`. Outlined into level-/role-annotated functions.
 */
class HierarchyScopeStmt : public ScopeStmt {
 public:
  HierarchyScopeStmt(Level level, std::optional<Role> role, std::string name_hint, StmtPtr body, Span span,
                     std::vector<std::string> leading_comments = {})
      : ScopeStmt(std::move(name_hint), std::move(body), std::move(span), std::move(leading_comments)),
        level_(level),
        role_(role) {}

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::HierarchyScopeStmt; }
  [[nodiscard]] ScopeKind GetScopeKind() const override { return ScopeKind::Hierarchy; }
  [[nodiscard]] std::string TypeName() const override { return "HierarchyScopeStmt"; }

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(ScopeStmt::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&HierarchyScopeStmt::level_, "level"),
                                          reflection::UsualField(&HierarchyScopeStmt::role_, "role")));
  }

 public:
  Level level_;               ///< Hierarchy level (required)
  std::optional<Role> role_;  ///< Function role (Orchestrator or Worker)
};

using HierarchyScopeStmtPtr = std::shared_ptr<const HierarchyScopeStmt>;

/**
 * @brief SPMD dispatch scope.
 *
 * Required `core_num` expression; `sync_start` defaults to false.
 *
 * `core_num_` is a generic `ExprPtr` so compile-time-known integer
 * expressions (closure variables, closure arithmetic, etc.) flow through
 * the parser unchanged. Codegen emits it as a scalar C++ expression —
 * constants produce a literal, scalar Vars resolve to the enclosing scope's
 * parameter. `Simplify` folds closure-derived arithmetic to `ConstInt`
 * whenever possible.
 */
class SpmdScopeStmt : public ScopeStmt {
 public:
  SpmdScopeStmt(ExprPtr core_num, bool sync_start, std::string name_hint, StmtPtr body, Span span,
                std::vector<std::string> leading_comments = {})
      : ScopeStmt(std::move(name_hint), std::move(body), std::move(span), std::move(leading_comments)),
        core_num_(std::move(core_num)),
        sync_start_(sync_start) {
    INTERNAL_CHECK(core_num_ != nullptr) << "SpmdScopeStmt core_num must not be null";
  }

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::SpmdScopeStmt; }
  [[nodiscard]] ScopeKind GetScopeKind() const override { return ScopeKind::Spmd; }
  [[nodiscard]] std::string TypeName() const override { return "SpmdScopeStmt"; }

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(ScopeStmt::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&SpmdScopeStmt::core_num_, "core_num"),
                                          reflection::UsualField(&SpmdScopeStmt::sync_start_, "sync_start")));
  }

 public:
  ExprPtr core_num_;  ///< SPMD block count expression
  bool sync_start_;   ///< Require sync-start for SPMD dispatch
};

using SpmdScopeStmtPtr = std::shared_ptr<const SpmdScopeStmt>;

/**
 * @brief Runtime orchestration scope: a PTO2_SCOPE wrapper at codegen.
 *
 * Marks a region wrapped by the simpler runtime's PTO2_SCOPE block. The
 * ``manual_`` flag picks between two modes:
 *   - manual_ = false → PTO2_SCOPE()                       (auto-dep via TensorMap)
 *   - manual_ = true  → PTO2_SCOPE(PTO2ScopeMode::MANUAL)  (no auto-dep, compiler emits add_dep)
 *
 * Inside a manual=true region, the manual-scope lowering phase of
 * ``DeriveCallDirections`` walks SSA data flow plus user-supplied
 * ``deps=[...]`` on each kernel call to compute explicit dependency edges
 * (``Call.attrs[manual_dep_edges]``). Codegen reads those edges and emits
 * ``params.add_dep(...)`` calls.
 *
 * The runtime forbids:
 *   - Manual scope nested inside another manual scope
 *   - Auto scope nested inside a manual scope (codegen suppresses the
 *     implicit ``PTO2_SCOPE()`` wrap on ForStmt/IfStmt bodies inside manual)
 */
class RuntimeScopeStmt : public ScopeStmt {
 public:
  RuntimeScopeStmt(bool manual, std::string name_hint, StmtPtr body, Span span,
                   std::vector<std::string> leading_comments = {})
      : ScopeStmt(std::move(name_hint), std::move(body), std::move(span), std::move(leading_comments)),
        manual_(manual) {}

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::RuntimeScopeStmt; }
  [[nodiscard]] ScopeKind GetScopeKind() const override { return ScopeKind::Runtime; }
  [[nodiscard]] std::string TypeName() const override { return "RuntimeScopeStmt"; }

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(ScopeStmt::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&RuntimeScopeStmt::manual_, "manual")));
  }

 public:
  bool manual_;  ///< true = MANUAL scope; false = AUTO scope (default PTO2_SCOPE())
};

using RuntimeScopeStmtPtr = std::shared_ptr<const RuntimeScopeStmt>;

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
  SeqStmts(std::vector<StmtPtr> stmts, Span span, std::vector<std::string> leading_comments = {})
      : Stmt(std::move(span), std::move(leading_comments)), stmts_(std::move(stmts)) {
    INTERNAL_CHECK(leading_comments_.empty())
        << "SeqStmts is a transparent container and must not carry leading comments; "
           "attach to an inner (non-Seq) stmt instead";
  }

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
  EvalStmt(ExprPtr expr, Span span, std::vector<std::string> leading_comments = {})
      : Stmt(std::move(span), std::move(leading_comments)), expr_(std::move(expr)) {}

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
  explicit BreakStmt(Span span, std::vector<std::string> leading_comments = {})
      : Stmt(std::move(span), std::move(leading_comments)) {}

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
  explicit ContinueStmt(Span span, std::vector<std::string> leading_comments = {})
      : Stmt(std::move(span), std::move(leading_comments)) {}

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ContinueStmt; }
  [[nodiscard]] std::string TypeName() const override { return "ContinueStmt"; }

  static constexpr auto GetFieldDescriptors() { return Stmt::GetFieldDescriptors(); }
};

using ContinueStmtPtr = std::shared_ptr<const ContinueStmt>;

/**
 * @brief Language carried by an InlineStmt body.
 */
enum class InlineLanguage : uint8_t {
  Python = 0,
};

/**
 * @brief Inline statement carrying a verbatim source body in a target language.
 *
 * Used to embed a block of source text (e.g. a SubWorker's Python body) directly
 * into the IR. Passes treat it as an opaque leaf — no children to traverse.
 */
class InlineStmt : public Stmt {
 public:
  InlineStmt(std::string body, InlineLanguage language, Span span,
             std::vector<std::string> leading_comments = {})
      : Stmt(std::move(span), std::move(leading_comments)), body_(std::move(body)), language_(language) {}

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::InlineStmt; }
  [[nodiscard]] std::string TypeName() const override { return "InlineStmt"; }

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Stmt::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&InlineStmt::body_, "body"),
                                          reflection::UsualField(&InlineStmt::language_, "language")));
  }

 public:
  std::string body_;
  InlineLanguage language_;
};

using InlineStmtPtr = std::shared_ptr<const InlineStmt>;

/**
 * @brief Attach leading comments to an existing statement
 *
 * Stmts are handed around as `shared_ptr<const Stmt>` to discourage mutation of
 * semantic fields, but `leading_comments_` is IgnoreField metadata that has no
 * effect on structural equality or hashing. This helper is the single sanctioned
 * mutation channel (e.g., for the Python parser attaching extracted comments
 * after building a stmt). The `const_cast` is safe because the helper requires
 * the caller already holds a StmtPtr referencing a concrete, owned stmt.
 *
 * Rejects `SeqStmts` targets: it is a transparent container and the printer
 * enforces that its leading_comments_ is always empty. Attach comments to an
 * inner stmt instead.
 *
 * Not thread-safe: concurrent callers must coordinate since the mutation is
 * performed through `const_cast` on the shared target.
 *
 * @param stmt Statement to annotate (must be non-null, not a SeqStmts)
 * @param comments Comment lines (without leading '#')
 * @return The same StmtPtr, with `leading_comments_` replaced by `comments`
 */
inline StmtPtr AttachLeadingComments(StmtPtr stmt, std::vector<std::string> comments) {
  CHECK(stmt) << "AttachLeadingComments: stmt must not be null";
  CHECK(stmt->GetKind() != ObjectKind::SeqStmts)
      << "AttachLeadingComments: cannot attach to SeqStmts; attach to an inner stmt instead";
  const_cast<Stmt&>(*stmt).leading_comments_ = std::move(comments);
  return stmt;
}

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_STMT_H_
