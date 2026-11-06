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

#include "pypto/ir/op_registry.h"

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/span.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

/// Format an IR span as a " at <file>:<line>:<col>" suffix, empty when unknown.
///
/// Kept out of line so `Span::to_string()` is only paid on the error path.
std::string LocationSuffix(const Span& span) { return span.is_valid() ? " at " + span.to_string() : ""; }

bool IsBufferHandle(const TypePtr& type) { return As<BufferType>(type) || As<MultiBufferType>(type); }

bool IsNonMemoryBufferOperand(const TypePtr& type) {
  if (As<ScalarType>(type)) return true;
  if (auto tuple = As<TupleType>(type)) {
    return std::all_of(tuple->types_.begin(), tuple->types_.end(), IsNonMemoryBufferOperand);
  }
  return false;
}

/// Render an allowed-space list as "Acc" / "Vec or Acc" / "Vec, Mat or Acc".
std::string FormatAllowedSpaces(const std::vector<MemorySpace>& allowed) {
  std::string out;
  for (size_t i = 0; i < allowed.size(); ++i) {
    if (i > 0) out += (i + 1 == allowed.size()) ? " or " : ", ";
    out += MemorySpaceToString(allowed[i]);
  }
  return out;
}

/// Reject an operand whose explicit memory space no pass can legalize.
///
/// Three states, three outcomes:
///
///  * **unset** — always legal. `nullopt` is the IR's "not decided yet", and
///    `InferTileMemorySpace` (pass 20) places the tile from consumer demand.
///    The `TileMemoryInferred` verifier checks the outcome afterwards.
///  * **set and allowed** — legal, nothing to do.
///  * **set, not allowed, but reachable by a `tile.move`** — also legal here.
///    Pass 20's MoveCollector inserts the move. This is the ordinary case for
///    `tile.matmul`'s Left/Right operands, reached from Mat by an MTE1 tmov.
///  * **set, not allowed, and unreachable** — a user error, reported here.
///
/// Only the last case is rejected, and the axis is *reachability in the ISA's
/// move graph*, not aliasing. In practice it fires on `Acc`: nothing writes L0C
/// except the MAD unit, so no target's SoC memory graph has an inbound
/// edge to `Acc` on any target. A tile that must be an accumulator therefore has
/// to be *created* in `Acc` — no copy can put it there afterwards. Without this
/// check the pass emits a `pto.tmov` into L0C that PTOAS rejects much later,
/// naming neither the tile nor the line that created it.
///
/// Deliberately narrow. `OpRegistry::Create` is not only the user-authoring
/// path: passes call it constantly on half-rewritten IR, where an operand may
/// still be the pre-legalization value (a GM tensor awaiting its `tile.load`, an
/// operand a later phase will bridge). Rejecting every constraint violation here
/// would reject those transients. So this checks only what *no* later phase can
/// repair, whatever order they run in:
///
///  * A `DDR`-resident operand is skipped outright — GM values reach on-chip by
///    `tile.load`, a different mechanism from `tile.move`, and always available.
///  * Otherwise the operand is rejected only when the constraint set contains no
///    space that any target can move into from where the operand actually is.
///    Today that means exactly one thing: a constraint of `{Acc}`. Nothing writes
///    L0C but the MAD unit, so an accumulator must be *created* in `Acc`.
///
/// Everything else — a Vec operand needing Left, a Mat operand needing Vec — is
/// left alone here even when it is also unimplementable, because at construction
/// time we cannot tell a settled operand from a transient one. Those belong to
/// `InferTileMemorySpace`'s MoveCollector, which runs on settled IR and knows the
/// configured target's exact adjacency via `SoC::GetMemoryGraph()`.
void CheckOperandMemorySpaceReachable(const OpMemorySpaceSpec& spec, const std::string& op_name,
                                      const std::vector<ExprPtr>& args, const Span& span) {
  if (spec.input_constraints.empty()) return;

  const size_t n = std::min(spec.input_constraints.size(), args.size());
  for (size_t idx = 0; idx < n; ++idx) {
    const auto& allowed = spec.input_constraints[idx];
    if (allowed.empty() || !args[idx]) continue;

    auto tile_type = As<TileType>(args[idx]->GetType());
    if (!tile_type) continue;
    const auto space = tile_type->memory_space_;
    if (!space.has_value()) continue;          // unset: the compiler will place it
    if (*space == MemorySpace::DDR) continue;  // reached by tile.load, not tile.move
    if (std::find(allowed.begin(), allowed.end(), *space) != allowed.end()) continue;

    // Reachable from ANY on-chip space, not just this operand's: that is what
    // distinguishes "this particular hop is missing" (leave it to pass 20) from
    // "this destination has no inbound edge at all" (unfixable, reject now).
    const bool destination_ever_reachable =
        std::any_of(allowed.begin(), allowed.end(), [](MemorySpace target) {
          for (MemorySpace from : {MemorySpace::Vec, MemorySpace::Mat, MemorySpace::Acc, MemorySpace::Left,
                                   MemorySpace::Right, MemorySpace::Bias}) {
            if (IsTileMoveEverSupported(from, target)) return true;
          }
          return false;
        });
    if (destination_ever_reachable) continue;

    const std::string wanted = FormatAllowedSpaces(allowed);
    std::string msg = "The operator " + op_name + " requires argument " + std::to_string(idx) +
                      " to live in " + wanted + " memory, but it is in " + MemorySpaceToString(*space) +
                      " memory. No target has any data path into " + wanted +
                      " memory -- only the matrix unit writes it -- so the compiler " +
                      "cannot insert a copy to bridge them. The value has to be produced there " +
                      "in the first place: either by a matmul, or by an allocation that names " +
                      "the space (target_memory=pl.MemorySpace." + MemorySpaceToString(allowed[0]) +
                      "), or by an allocation left unset for the compiler to place.";
    if (allowed.size() == 1 && allowed[0] == MemorySpace::Acc) {
      // The common way to land here is a zero-initialized accumulator written as
      // `tile.full`, whose output space is fixed to UB and so can never be it.
      // Name the replacement, because there is no in-place rewrite of `tile.full`
      // that would work: `init_cond` removes the need to pre-zero at all.
      msg +=
          " Note that `tile.full` fills UB and cannot produce an accumulator. To start an"
          " accumulation from zero, drop the pre-zeroed tile and pass"
          " `init_cond=<true on the first step>` to the accumulating op instead -- it overwrites"
          " on that step rather than accumulating into it.";
    }
    throw ValueError(msg + LocationSuffix(span));
  }
}

}  // namespace

OpRegistryEntry& OpRegistryEntry::set_buffer_arg_effect(size_t arg_index, BufferAccess data,
                                                        BufferAccess metadata) {
  CHECK(ir_stage_ == OpIRStage::Buffer)
      << "Operator '" << name_ << "' must select Buffer stage before declaring buffer effects";
  CHECK(buffer_arg_effects_.emplace(arg_index, BufferArgEffect{data, metadata, false}).second)
      << "Operator '" << name_ << "' already classified buffer argument " << arg_index;
  return *this;
}

OpRegistryEntry& OpRegistryEntry::set_buffer_non_memory_arg(size_t arg_index) {
  set_buffer_arg_effect(arg_index, BufferAccess::None, BufferAccess::None);
  buffer_arg_effects_.at(arg_index).non_memory = true;
  return *this;
}

OpRegistryEntry& OpRegistryEntry::set_buffer_result_behavior(size_t result_index,
                                                             BufferResultBehavior behavior,
                                                             std::optional<size_t> alias_arg) {
  CHECK(ir_stage_ == OpIRStage::Buffer)
      << "Operator '" << name_ << "' must select Buffer stage before declaring buffer results";
  CHECK(buffer_results_.emplace(result_index, BufferResultSpec{behavior, alias_arg}).second)
      << "Operator '" << name_ << "' already classified result " << result_index;
  return *this;
}

const BufferArgEffect& OpRegistryEntry::GetBufferArgEffect(size_t arg_index) const {
  CHECK(ir_stage_ == OpIRStage::Buffer) << "Operator '" << name_ << "' is not a buffer operator";
  auto it = buffer_arg_effects_.find(arg_index);
  CHECK(it != buffer_arg_effects_.end())
      << "Operator '" << name_ << "' has no buffer effect for argument " << arg_index;
  return it->second;
}

const BufferResultSpec& OpRegistryEntry::GetBufferResultSpec(size_t result_index) const {
  CHECK(ir_stage_ == OpIRStage::Buffer) << "Operator '" << name_ << "' is not a buffer operator";
  auto it = buffer_results_.find(result_index);
  CHECK(it != buffer_results_.end()) << "Operator '" << name_ << "' has no buffer result behavior for result "
                                     << result_index;
  return it->second;
}

OpRegistryEntry& OpRegistryEntry::set_tuple_result_alias(size_t result_index, size_t arg_index) {
  CHECK(ir_stage_ == OpIRStage::Functional && output_arity_ > 1 && result_index < output_arity_)
      << "Operator '" << name_ << "' requires a declared tuple element for result alias " << result_index;
  CHECK(arg_index < GetArgumentCount()) << "Operator '" << name_ << "' has no argument " << arg_index;
  CHECK(tuple_result_aliases_.emplace(result_index, arg_index).second)
      << "Operator '" << name_ << "' declares result alias " << result_index << " twice";
  return *this;
}

void OpRegistryEntry::ValidateIRStage() const {
  CHECK(!validate_explicit_type_.has_value() || (ir_stage_ == OpIRStage::Buffer && internal_only_))
      << "Operator '" << name_ << "' explicit type validation requires an internal-only Buffer operator";
  if (ir_stage_ == OpIRStage::Functional) {
    CHECK(output_arity_ > 0 || op_category_ == "SimtOp")
        << "Functional operator '" << name_ << "' must produce at least one value";
    CHECK(buffer_arg_effects_.empty() && buffer_results_.empty())
        << "Functional operator '" << name_ << "' cannot carry buffer contracts";
    return;
  }
  CHECK(internal_only_) << "Buffer operator '" << name_ << "' must be internal-only";
  CHECK(output_arity_declared_) << "Buffer operator '" << name_ << "' must explicitly declare output arity";
  CHECK(!arg_effects_.has_value()) << "Buffer operator '" << name_
                                   << "' must use buffer data and metadata effects, not ArgEffect";
  CHECK(arguments_.has_value()) << "Buffer operator '" << name_ << "' must declare its arguments";
  for (size_t i = 0; i < GetArgumentCount(); ++i) {
    CHECK(buffer_arg_effects_.count(i) > 0)
        << "Buffer operator '" << name_ << "' has no buffer effect for argument " << i;
  }
  CHECK(buffer_arg_effects_.size() == GetArgumentCount())
      << "Buffer operator '" << name_ << "' declares an effect for an argument outside its schema";
  const size_t result_count = std::max(size_t{1}, output_arity_);
  for (size_t i = 0; i < result_count; ++i) {
    const auto& result = GetBufferResultSpec(i);
    CHECK((result.behavior == BufferResultBehavior::None) == (output_arity_ == 0))
        << "Buffer operator '" << name_ << "' must declare None behavior exactly for zero results";
    const bool aliases =
        result.behavior == BufferResultBehavior::Alias || result.behavior == BufferResultBehavior::Borrow;
    CHECK(aliases == result.alias_arg.has_value())
        << "Buffer operator '" << name_ << "' must name a source argument exactly for Alias/Borrow results";
    if (result.alias_arg.has_value()) {
      CHECK(*result.alias_arg < GetArgumentCount() && !GetBufferArgEffect(*result.alias_arg).non_memory)
          << "Buffer operator '" << name_ << "' result " << i << " has no memory source argument";
    }
  }
  CHECK(buffer_results_.size() == result_count)
      << "Buffer operator '" << name_ << "' declares behavior for a result outside its output arity";
}

void OpRegistryEntry::ValidateCall(const std::vector<ExprPtr>& args, const TypePtr& result_type,
                                   const Span& span) const {
  INTERNAL_CHECK_SPAN(result_type, span) << "Type deduction failed for '" << name_ << "'";
  if (mutable_session_arg_) {
    const size_t i = *mutable_session_arg_;
    INTERNAL_CHECK_SPAN(i < args.size() && args[i] && IsA<AsyncSessionType>(args[i]->GetType()), span)
        << "Operator '" << name_ << "' argument " << i << " must be an AsyncSession";
  }
  auto tuple = As<TupleType>(result_type);
  if (output_arity_ == 0) {
    INTERNAL_CHECK_SPAN(
        (ir_stage_ == OpIRStage::Buffer || op_category_ == "SimtOp") && As<VoidType>(result_type), span)
        << "Operator '" << name_ << "' declares zero results but did not deduce VoidType";
  } else {
    INTERNAL_CHECK_SPAN(!As<VoidType>(result_type), span)
        << "Operator '" << name_ << "' declares SSA results but deduced VoidType";
    if (output_arity_ > 1) {
      INTERNAL_CHECK_SPAN(tuple, span)
          << "Operator '" << name_ << "' declares set_output_arity(" << output_arity_
          << ") but deduced a non-tuple " << result_type->TypeName();
      INTERNAL_CHECK_SPAN(tuple->types_.size() == output_arity_, span)
          << "Operator '" << name_ << "' declares set_output_arity(" << output_arity_
          << ") but deduced a TupleType with " << tuple->types_.size() << " elements";
      for (const auto& [result, arg] : tuple_result_aliases_) {
        INTERNAL_CHECK_SPAN(result < tuple->types_.size() && arg < args.size() && args[arg] &&
                                AsTensorTypeLike(args[arg]->GetType()) &&
                                (tuple->types_[result] == args[arg]->GetType() ||
                                 structural_equal(tuple->types_[result], args[arg]->GetType())),
                            span)
            << "Operator '" << name_ << "' tuple result " << result << " must return tensor argument " << arg;
      }
    } else {
      INTERNAL_CHECK_SPAN(!tuple, span)
          << "Operator '" << name_ << "' deduced a TupleType result without declaring set_output_arity("
          << tuple->types_.size() << ")";
    }
  }
  if (ir_stage_ != OpIRStage::Buffer) return;

  CHECK(args.size() <= GetArgumentCount()) << "Buffer operator '" << name_ << "' received " << args.size()
                                           << " arguments, but its schema has " << GetArgumentCount();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << "Buffer operator '" << name_ << "' argument " << i << " is null";
    const auto& type = args[i]->GetType();
    const auto& effect = GetBufferArgEffect(i);
    const bool memory = IsBufferHandle(type) || AsTensorTypeLike(type);
    CHECK(effect.non_memory ? IsNonMemoryBufferOperand(type) : memory)
        << "Buffer operator '" << name_ << "' argument " << i << " has type " << type->TypeName()
        << " inconsistent with its " << (effect.non_memory ? "non-memory" : "memory") << " effect";
  }
  if (output_arity_ == 0) return;
  for (size_t i = 0; i < output_arity_; ++i) {
    const TypePtr& type = tuple ? tuple->types_[i] : result_type;
    const auto& result = GetBufferResultSpec(i);
    if (result.behavior == BufferResultBehavior::Value) {
      INTERNAL_CHECK_SPAN(IsNonMemoryBufferOperand(type), span)
          << "Buffer operator '" << name_ << "' Value result " << i << " must contain only scalar values";
      continue;
    }
    INTERNAL_CHECK_SPAN(IsBufferHandle(type), span)
        << "Buffer operator '" << name_ << "' storage result " << i << " must be a buffer handle";
    if (result.alias_arg.has_value()) {
      const size_t source = *result.alias_arg;
      CHECK(source < args.size() && IsBufferHandle(args[source]->GetType()))
          << "Buffer operator '" << name_ << "' result " << i << " must alias an actual buffer operand";
    }
  }
}

void ValidateKwargs(const std::vector<std::pair<std::string, std::any>>& kwargs,
                    const std::unordered_map<std::string, std::type_index>& allowed_kwargs,
                    const std::string& op_name) {
  for (const auto& [key, value] : kwargs) {
    auto it = allowed_kwargs.find(key);
    if (it == allowed_kwargs.end()) {
      throw ValueError("Unknown kwarg '" + key + "' for operator '" + op_name + "'");
    }

    // For DataType, accept both DataType and int (since Python may pass as int for backward compatibility)
    if (it->second == std::type_index(typeid(DataType))) {
      std::type_index value_type(value.type());
      if (value_type != std::type_index(typeid(DataType)) && value_type != std::type_index(typeid(int))) {
        throw TypeError("Kwarg '" + key + "' for operator '" + op_name +
                        "' expects DataType or int, but got incompatible type");
      }
    } else if (it->second == std::type_index(typeid(MemorySpace))) {
      if (std::type_index(value.type()) != std::type_index(typeid(MemorySpace))) {
        throw TypeError("Kwarg '" + key + "' for operator '" + op_name +
                        "' expects MemorySpace, but got incompatible type");
      }
    } else if (it->second == std::type_index(typeid(TileLayout))) {
      if (std::type_index(value.type()) != std::type_index(typeid(TileLayout))) {
        throw TypeError("Kwarg '" + key + "' for operator '" + op_name +
                        "' expects TileLayout, but got incompatible type");
      }
    } else if (std::type_index(value.type()) != it->second) {
      throw TypeError("Kwarg '" + key + "' for operator '" + op_name + "' has incompatible type");
    }
  }
}

std::vector<std::pair<std::string, std::any>> NormalizeRegKwargs(
    const std::string& op_name, const std::vector<std::pair<std::string, std::any>>& kwargs) {
  auto normalized = kwargs;
  auto has_kwarg = [&normalized](const std::string& key) {
    return std::any_of(normalized.begin(), normalized.end(),
                       [&key](const auto& item) { return item.first == key; });
  };
  if (op_name.rfind("reg.", 0) != 0) return normalized;
  const auto op = OpRegistry::GetInstance().GetOp(op_name);
  // Every public register pmode currently defaults to zero. Drive this from
  // the schema so DSL, Builder and raw IR construction cannot drift as new
  // operations are added; carry ops have no pmode and remain attribute-free.
  if (op->GetAttrs().count("pmode") && !has_kwarg("pmode")) {
    normalized.emplace_back("pmode", std::string("zero"));
  }
  if (IsOp(op, "reg.index_gen") && !has_kwarg("order")) {
    normalized.emplace_back("order", std::string("ascending"));
  }
  return normalized;
}

void RejectRawNoOpConvert(const std::string& op_name, const std::vector<ExprPtr>& args,
                          const std::vector<std::pair<std::string, std::any>>& kwargs, const Span& span) {
  if (op_name != "reg.convert" || args.size() != 1) return;
  bool has_rounding = false;
  bool has_saturate = false;
  const DataType* target = nullptr;
  for (const auto& [key, value] : kwargs) {
    if (key == "rounding") has_rounding = true;
    if (key == "saturate") has_saturate = true;
    if (key == "dtype") target = std::any_cast<DataType>(&value);
  }
  if (has_rounding || has_saturate || target == nullptr || !args[0]) return;
  auto source = std::dynamic_pointer_cast<const RegType>(args[0]->GetType());
  if (source && source->GetRegKind() == RegKind::VecReg && source->GetDType() == *target) {
    CHECK_SPAN(false, span) << "Raw reg.convert same-dtype no-op is non-canonical; use the operand directly";
  }
}

OpRegistry& OpRegistry::GetInstance() {
  static OpRegistry instance;
  return instance;
}

OpRegistryEntry& OpRegistry::Register(const std::string& op_name) {
  // Check if operator is already registered
  CHECK(registry_.find(op_name) == registry_.end()) << "Operator '" + op_name + "' is already registered";

  // Create and insert the entry into the registry
  auto result = registry_.emplace(op_name, OpRegistryEntry());
  auto& entry = result.first->second;
  entry.set_name(op_name);

  // Create the operator instance with the operator name
  entry.op_ = std::make_shared<Op>(op_name);

  return entry;
}

// ============================================================================
// OpRegistry Implementation
// ============================================================================

CallPtr OpRegistry::Create(const std::string& op_name, const std::vector<ExprPtr>& args, Span span) const {
  // Call new version with empty kwargs for backward compatibility
  return Create(op_name, args, {}, std::move(span));
}

CallPtr OpRegistry::Create(const std::string& op_name, const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs, Span span) const {
  return CreateImpl(op_name, args, NormalizeRegKwargs(op_name, kwargs), std::move(span),
                    /*allow_internal=*/true);
}

CallPtr OpRegistry::CreateUserFacing(const std::string& op_name, const std::vector<ExprPtr>& args,
                                     Span span) const {
  return CreateUserFacing(op_name, args, {}, std::move(span));
}

CallPtr OpRegistry::CreateUserFacing(const std::string& op_name, const std::vector<ExprPtr>& args,
                                     const std::vector<std::pair<std::string, std::any>>& kwargs,
                                     Span span) const {
  auto normalized = NormalizeRegKwargs(op_name, kwargs);
  RejectRawNoOpConvert(op_name, args, normalized, span);
  return CreateImpl(op_name, args, normalized, std::move(span), /*allow_internal=*/false);
}

CallPtr OpRegistry::CreateInternal(const std::string& op_name, const std::vector<ExprPtr>& args,
                                   Span span) const {
  return CreateInternal(op_name, args, {}, std::move(span));
}

CallPtr OpRegistry::CreateInternal(const std::string& op_name, const std::vector<ExprPtr>& args,
                                   const std::vector<std::pair<std::string, std::any>>& kwargs,
                                   Span span) const {
  return CreateImpl(op_name, args, NormalizeRegKwargs(op_name, kwargs), std::move(span),
                    /*allow_internal=*/true);
}

CallPtr OpRegistry::CreateInternal(const std::string& op_name, const std::vector<ExprPtr>& args,
                                   const std::vector<std::pair<std::string, std::any>>& kwargs,
                                   const TypePtr& result_type, Span span) const {
  CHECK_SPAN(result_type, span) << "Operator '" << op_name << "' explicit result type must not be null";
  return CreateImpl(op_name, args, NormalizeRegKwargs(op_name, kwargs), std::move(span),
                    /*allow_internal=*/true, result_type);
}

TypePtr OpRegistry::ResolveAndValidateCallType(const OpRegistryEntry& entry, const std::vector<ExprPtr>& args,
                                               const std::vector<std::pair<std::string, std::any>>& kwargs,
                                               const TypePtr& explicit_type, const Span& span) const {
  const auto& op = entry.GetOp();  // Also validates the registration's stage and complete contracts.
  const auto& op_name = entry.GetName();
  TypePtr result_type;
  try {
    if (!kwargs.empty()) {
      const auto& allowed_kwargs = op->GetAttrs();
      if (!allowed_kwargs.empty() || entry.GetIRStage() == OpIRStage::Buffer || entry.GetRegSpec()) {
        ValidateKwargs(kwargs, allowed_kwargs, op_name);
      }
    }

    if (entry.validate_explicit_type_.has_value()) {
      CHECK(explicit_type) << "Operator '" << op_name << "' requires an explicit result type";
      (*entry.validate_explicit_type_)(args, kwargs, explicit_type);
      result_type = explicit_type;
    } else {
      CHECK(!explicit_type) << "Operator '" << op_name
                            << "' uses type deduction and does not accept an explicit result type";
      result_type = entry.GetDeduceType()(args, kwargs);
    }
  } catch (const Error& e) {
    // Preserve the original exception class and attach the source location.
    e.RethrowWithMessage(std::string(e.what()) + LocationSuffix(span));
  } catch (const std::exception& e) {
    throw ValueError(std::string(e.what()) + LocationSuffix(span));
  }
  entry.ValidateCall(args, result_type, span);
  return result_type;
}

void OpRegistry::ValidateBufferCall(const CallPtr& call) const {
  CHECK(call) << "Cannot validate a null Buffer call";
  CHECK_SPAN(call->op_, call->span_) << "Buffer call must name a registered operator";
  CHECK_SPAN(!As<GlobalVar>(call->op_), call->span_)
      << "A GlobalVar call cannot use a registered buffer operator contract";
  const auto& entry = GetEntry(call->op_->name_);
  CHECK_SPAN(entry.GetIRStage() == OpIRStage::Buffer, call->span_)
      << "Operator '" << entry.GetName() << "' is not a buffer operator";
  const auto& original_type = call->GetType();
  CHECK_SPAN(original_type, call->span_) << "Buffer call must have a result type";
  const TypePtr expected_type = ResolveAndValidateCallType(
      entry, call->args_, call->kwargs_, entry.RequiresExplicitType() ? original_type : nullptr, call->span_);
  CHECK_SPAN(structural_equal(original_type, expected_type), call->span_)
      << "Buffer operator '" << entry.GetName() << "' stored result type does not match its deduced type";
}

CallPtr OpRegistry::BindResultStorage(const CallPtr& call,
                                      const std::vector<std::optional<MemRefPtr>>& destinations) const {
  CHECK(call && call->op_) << "Storage binding requires a detached registered producer call";
  // `GlobalVar` derives from `Op` and carries a function name; looking that
  // up could validate the call against a same-named registered operator's
  // storage contract (kind first, name second).
  CHECK_SPAN(!As<GlobalVar>(call->op_), call->span_) << "A GlobalVar call has no registered storage contract";
  const auto& entry = GetEntry(call->op_->name_);
  CHECK(entry.GetIRStage() == OpIRStage::Functional && !entry.IsInternalOnly())
      << "Storage binding requires a public Functional-stage producer";
  CHECK(entry.GetOpCategory() == "TileOp")
      << "Storage binding requires a Tile producer; communication results use their transport storage";
  const auto& spec = entry.GetMemorySpec();
  // `no_memory_spec()` engages an empty spec on purpose; that must read as
  // "no contract", not as one satisfied by the engaged optional.
  CHECK(spec.has_value() && !spec->opted_out)
      << "Operator '" << entry.GetName()
      << "' declares no output memory contract; its result storage cannot be bound";
  CHECK(!spec->output_reuses_input_arg.has_value())
      << "Operator '" << entry.GetName() << "' writes its result into input argument "
      << *spec->output_reuses_input_arg << "; bind that input's allocation instead";
  CHECK(!(entry.OutputMemoryInheritsInput() && entry.IsInplaceSafe()))
      << "Operator '" << entry.GetName() << "' inherits storage; bind its allocating input instead";
  // Input MemRefs for alias checking; null for operands without storage.
  // As<ShapedType> is the supported widening cast (kind_traits.h): it matches
  // ShapedType, TensorType, DistributedTensorType, TileType and ArrayType,
  // the same subtree the dynamic cast in GetTypeMemRef
  // (transforms/utils/memref_utils.h) accepts.
  std::vector<MemRefPtr> arg_memrefs(call->args_.size());
  for (size_t a = 0; a < call->args_.size(); ++a) {
    if (auto shaped = As<ShapedType>(call->args_[a]->GetType())) {
      arg_memrefs[a] = shaped->memref_.value_or(nullptr);
    }
  }
  std::vector<TypePtr> results;
  const auto tuple = As<TupleType>(call->GetType());
  if (tuple) {
    results = tuple->types_;
  } else {
    results.push_back(call->GetType());
  }
  CHECK(destinations.size() == results.size())
      << "Storage binding requires one destination per result, got " << destinations.size() << " for "
      << results.size() << " results";
  // Destinations that passed validation, paired with their result index: the
  // pairwise check below runs after the loop, once every destination has been
  // validated on its own terms.
  std::vector<std::pair<size_t, MemRefPtr>> bound_destinations;
  bound_destinations.reserve(results.size());
  auto aliases_dest = [](const MemRefPtr& a, const MemRefPtr& b) {
    // Same conservativeness as the input check below: sharing the allocation
    // base is an alias unless both select provably-different constant slots,
    // and an unresolved byte extent (0) means "unknown", not "empty".
    if (a->base_.get() != b->base_.get()) return false;
    auto a_slot = a->slot_index_ ? As<ConstInt>(*a->slot_index_) : nullptr;
    auto b_slot = b->slot_index_ ? As<ConstInt>(*b->slot_index_) : nullptr;
    if (a_slot && b_slot && a_slot->value_ != b_slot->value_) return false;
    return a->size_ == 0 || b->size_ == 0 || MemRef::MayAlias(a, b);
  };
  // A producer with any placed result is itself already placed: a second
  // request leaves the placed positions out of both the per-result and the
  // pairwise checks below (they only see destinations supplied here), so a
  // partial rebind could hand two results the same allocation.
  for (const auto& result : results) {
    if (auto tile = As<TileType>(result)) {
      CHECK(!tile->memref_.has_value()) << "Producer already has an explicit storage binding";
    }
  }
  for (size_t i = 0; i < results.size(); ++i) {
    if (!destinations[i].has_value()) continue;
    auto tile = As<TileType>(results[i]);
    CHECK(tile) << "Storage binding result " << i << " is not a Tile";
    CHECK(tile->memory_space_.has_value())
        << "Storage binding requires a resolved output memory space; specify target_memory";
    const auto& memory = *destinations[i];
    CHECK(memory && memory->is_pinned_) << "Storage binding requires an allocation declaration";
    auto aliases_arg = [&](size_t arg_index) {
      const auto& arg = arg_memrefs[arg_index];
      if (!arg) return false;
      // Binding runs before InitMemRef sizes a pinned destination, so a zero
      // byte extent means "unresolved", not "empty": MayAlias's range
      // overlap is no disjointness proof for an unsized region. The
      // conservative floor is sharing the allocation base, lifted only by
      // constant slot selections that provably name different slots.
      if (memory->base_.get() != arg->base_.get()) return false;
      auto dest_slot = memory->slot_index_ ? As<ConstInt>(*memory->slot_index_) : nullptr;
      auto arg_slot = arg->slot_index_ ? As<ConstInt>(*arg->slot_index_) : nullptr;
      if (dest_slot && arg_slot && dest_slot->value_ != arg_slot->value_) return false;
      return memory->size_ == 0 || arg->size_ == 0 || MemRef::MayAlias(memory, arg);
    };
    if (!entry.IsInplaceSafe()) {
      // Deliberately target-independent, which over-rejects one known case:
      // A5's TPRELU keeps the ABI-required tmp operand of tile.prelu but never
      // reads it, and its codegen checks dst only against src and slope. Binding
      // is a type-layer statement made before (or without) a target, while
      // A2/A3's TPRELU reads tmp while writing dst — so a destination sharing
      // tmp's allocation is refused for every target. The target-aware memory
      // planners make the opposite call when they do the packing
      // (allocation_constraints / semantic_aliases exempt A5 tmp).
      for (size_t a = 0; a < call->args_.size(); ++a) {
        CHECK(!aliases_arg(a)) << "Operator '" << entry.GetName() << "' is not in-place safe: result " << i
                               << " storage must not alias input argument " << a;
      }
    } else {
      for (size_t a : entry.ForbidOutputAliasArgs()) {
        if (a >= call->args_.size()) continue;
        CHECK(!aliases_arg(a)) << "Operator '" << entry.GetName() << "' reads input argument " << a
                               << " while writing its output: result " << i << " storage must not alias it";
      }
    }
    if (memory->slot_index_.has_value()) {
      const auto& index = *memory->slot_index_;
      auto scalar = index ? As<ScalarType>(index->GetType()) : nullptr;
      CHECK(scalar && scalar->dtype_.IsInt()) << "Storage slot index must be an integer scalar";
      if (auto constant = As<ConstInt>(index)) {
        CHECK(constant->value_ >= 0 && static_cast<uint64_t>(constant->value_) < memory->slot_count_)
            << "Storage slot index is outside the declared slot count";
      }
    }
    CHECK(memory->slot_count_ == 1 || memory->slot_index_.has_value())
        << "Select a slot before binding a multi-slot allocation";
    bound_destinations.emplace_back(i, memory);
    results[i] =
        std::make_shared<TileType>(tile->shape_, tile->dtype_, memory, tile->tile_view_, tile->memory_space_);
  }
  // Two results of one producer never see each other through the input-alias
  // checks above, so compare the destinations to each other: the second write
  // would otherwise overwrite the first.
  for (size_t a = 0; a + 1 < bound_destinations.size(); ++a) {
    for (size_t b = a + 1; b < bound_destinations.size(); ++b) {
      CHECK(!aliases_dest(bound_destinations[a].second, bound_destinations[b].second))
          << "Storage binding results " << bound_destinations[a].first << " and "
          << bound_destinations[b].first
          << " must not alias each other; give each result its own allocation or slot";
    }
  }
  TypePtr result_type = tuple ? std::make_shared<TupleType>(results) : results[0];
  return std::make_shared<Call>(call->op_, call->args_, call->kwargs_, call->attrs_, result_type,
                                call->span_);
}

CallPtr OpRegistry::CreateImpl(const std::string& op_name, const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs, Span span,
                               bool allow_internal, const TypePtr& explicit_type) const {
  // Look up operator in registry
  auto it = registry_.find(op_name);
  if (it == registry_.end()) {
    std::string msg = "Operator '" + op_name + "' not found in registry";
    if (op_name.find('.') == std::string::npos) {
      msg +=
          ". This looks like a function name (GlobalVar), not a registered operator. "
          "Callers should check for GlobalVar before using OpRegistry::Create.";
    }
    throw ValueError(msg);
  }

  const auto& entry = it->second;
  if (entry.IsInternalOnly() && !allow_internal) {
    throw ValueError("Operator '" + op_name +
                     "' is internal-only and cannot be created from user-facing op creation paths");
  }

  TypePtr result_type = ResolveAndValidateCallType(entry, args, kwargs, explicit_type, span);
  OpPtr op = entry.op_;  // ResolveAndValidateCallType already validated the registration.

  // Apply OpMemorySpaceSpec to TileType results that lack memory_space.
  // This ensures the deduced type carries memory_space even when individual
  // type deduction functions omit it (fixes issue #553).
  //
  // Single-output ops: patch the result TileType directly.
  // Tuple-output ops (e.g. tile.gather_compare): patch each TileType element
  // that lacks a memory_space. Heterogeneous-output ops should set
  // memory_space_ inside f_deduce_type rather than relying on this fallback.
  const auto& mem_spec = entry.GetMemorySpec();
  if (mem_spec.has_value()) {
    CheckOperandMemorySpaceReachable(*mem_spec, op_name, args, span);
  }
  if (mem_spec.has_value() && mem_spec->deduce_output_memory) {
    auto resolve_memory_space = [&]() -> std::optional<MemorySpace> {
      auto resolved = mem_spec->deduce_output_memory(kwargs);
      if (resolved.has_value()) {
        return resolved;
      }
      // Inherit from first tile-typed input
      for (const auto& arg : args) {
        if (auto input_tile = As<TileType>(arg->GetType())) {
          if (input_tile->memory_space_.has_value()) {
            return input_tile->memory_space_;
          }
        }
      }
      return std::nullopt;
    };
    auto apply_memory_space = [](const TileTypePtr& tile_type, MemorySpace space) {
      return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, tile_type->memref_,
                                        tile_type->tile_view_, space);
    };

    if (auto tile_type = As<TileType>(result_type)) {
      // Single-output case: result is a TileType.
      if (!tile_type->memory_space_.has_value()) {
        if (auto space = resolve_memory_space(); space.has_value()) {
          result_type = apply_memory_space(tile_type, *space);
        }
      }
    } else if (auto tuple_type = As<TupleType>(result_type)) {
      // Multi-output case: result is a TupleType. Patch every TileType
      // element that is missing a memory_space.
      bool any_missing = false;
      for (const auto& elem_ty : tuple_type->types_) {
        if (auto elem_tile = As<TileType>(elem_ty); elem_tile && !elem_tile->memory_space_.has_value()) {
          any_missing = true;
          break;
        }
      }
      if (any_missing) {
        if (auto space = resolve_memory_space(); space.has_value()) {
          std::vector<TypePtr> new_elems;
          new_elems.reserve(tuple_type->types_.size());
          for (const auto& elem_ty : tuple_type->types_) {
            if (auto elem_tile = As<TileType>(elem_ty); elem_tile && !elem_tile->memory_space_.has_value()) {
              new_elems.push_back(apply_memory_space(elem_tile, *space));
            } else {
              new_elems.push_back(elem_ty);
            }
          }
          result_type = std::make_shared<TupleType>(std::move(new_elems));
        }
      }
    }
  }

  // Create Call with deduced type
  return std::make_shared<Call>(op, args, kwargs, result_type, std::move(span));
}

const OpRegistryEntry& OpRegistry::GetEntry(const std::string& op_name) const {
  auto it = registry_.find(op_name);
  CHECK(it != registry_.end()) << "Operator '" + op_name + "' not found in registry";
  return it->second;
}

OpPtr OpRegistry::GetOp(const std::string& op_name) const {
  auto it = registry_.find(op_name);
  CHECK(it != registry_.end()) << "Operator '" + op_name + "' not found in registry";
  return it->second.GetOp();
}

void OpRegistry::ValidateTileOps() const {
  std::vector<std::string> missing;
  for (const auto& [name, entry] : registry_) {
    if (name.rfind("tile.", 0) != 0) continue;
    if (entry.GetMemorySpec().has_value()) continue;
    missing.push_back(name);
  }
  if (!missing.empty()) {
    std::sort(missing.begin(), missing.end());
    std::string msg =
        "The following tile ops are missing a memory spec "
        "(add set_output_memory/set_input_memory or no_memory_spec()):";
    for (const auto& name : missing) {
      msg += "\n  - " + name;
    }
    throw ValueError(msg);
  }
}

void OpRegistry::ValidateArgEffects() const {
  std::vector<std::string> unclassified;
  std::vector<std::string> channel_without_write;
  for (const auto& [name, entry] : registry_) {
    if (entry.GetIRStage() != OpIRStage::Functional) continue;
    // A write channel describes *how* an operator writes, so declaring one
    // while writing nothing is incoherent — and it is the shape that hides a
    // missing classification, since `set_write_channel()` creates the effect
    // spec as a side effect and would otherwise make the operator look
    // classified.
    if (entry.GetWriteChannel().has_value() && !entry.WritesAnyArg()) {
      channel_without_write.push_back(name);
    }
    const auto& spec = entry.GetMemorySpec();
    if (!spec.has_value() || !spec->output_reuses_input_arg.has_value()) continue;
    const size_t reused = *spec->output_reuses_input_arg;
    // Ask about the reused argument specifically. A registration that named a
    // different argument still leaves this one defaulting to `Read`, and the
    // whole point of the gate is that such a default is a decision nobody made.
    if (entry.HasDeclaredArgEffect(reused)) continue;
    unclassified.push_back(name + " (in-place on argument " + std::to_string(reused) + ")");
  }
  if (!channel_without_write.empty()) {
    std::sort(channel_without_write.begin(), channel_without_write.end());
    std::string msg =
        "The following ops declare a write channel but write through no argument. A channel says "
        "how an op writes, so one without a write is either a stray declaration or a missing "
        "one — add the .set_arg_effect(<index>, ...) that was meant to accompany it, or drop the "
        ".set_write_channel(...):";
    for (const auto& name : channel_without_write) {
      msg += "\n  - " + name;
    }
    throw ValueError(msg);
  }
  if (!unclassified.empty()) {
    std::sort(unclassified.begin(), unclassified.end());
    std::string msg =
        "The following ops update an argument in place but never declared what they do to it. "
        "Direction inference reads an undeclared operator as a pure consumer, so the write is "
        "silently dropped. Add .set_arg_effect(<index>, ArgEffect::Write) — ArgEffect::ReadWrite "
        "when the op accumulates into the slot — or .no_arg_writes() when the slot is metadata "
        "rather than data:";
    for (const auto& name : unclassified) {
      msg += "\n  - " + name;
    }
    throw ValueError(msg);
  }
}

void OpRegistry::ValidateMultiOutputOps() const {
  std::vector<std::string> unclassified;
  std::vector<std::string> leaked_destinations;
  std::vector<std::string> workspace_out_of_range;
  std::vector<std::string> workspace_never_written;
  std::vector<std::string> reuses_input;
  for (const auto& [name, entry] : registry_) {
    if (entry.GetIRStage() != OpIRStage::Functional) continue;
    if (entry.GetOutputArity() <= 1) continue;
    const size_t arg_count = entry.GetArgumentCount();
    for (size_t i = 0; i < arg_count; ++i) {
      // An argument nobody classified defaults to Read, and a destination tile
      // reads exactly like an input under that default. Demanding a verdict is
      // what turns the leak from invisible into a registration a reviewer sees.
      if (!entry.HasDeclaredArgEffect(i)) {
        unclassified.push_back(name + " (argument " + std::to_string(i) + ")");
        continue;
      }
      // Tensor writes require a returned alias or a workspace declaration.
      // Submissions can also update an opaque session whose identity stays the same.
      const auto& aliases = entry.GetTupleResultAliases();
      const bool returned =
          std::any_of(aliases.begin(), aliases.end(), [i](const auto& alias) { return alias.second == i; });
      if (entry.MayWriteArg(i) && !entry.IsWorkspaceArg(i) && entry.GetMutableSessionArg() != i &&
          !returned) {
        leaked_destinations.push_back(name + " (argument " + std::to_string(i) + ")");
      }
    }
    // A workspace declaration is a claim about an argument that exists and that
    // the hardware writes. Neither half is implied by the loop above: an index
    // past the end names nothing, and `.no_arg_writes().set_workspace_arg(0)`
    // classifies argument 0 as Read while calling it hardware-written scratch.
    // Either way the operator reads as a pure consumer of a slot it writes.
    for (size_t i : entry.GetWorkspaceArgs()) {
      if (i >= arg_count) {
        workspace_out_of_range.push_back(name + " (argument " + std::to_string(i) + " of " +
                                         std::to_string(arg_count) + ")");
      } else if (!entry.MayWriteArg(i)) {
        workspace_never_written.push_back(name + " (argument " + std::to_string(i) + ")");
      }
    }
    const auto& spec = entry.GetMemorySpec();
    if (spec.has_value() && spec->output_reuses_input_arg.has_value()) {
      reuses_input.push_back(name);
    }
  }
  auto fail = [](std::string msg, std::vector<std::string> names) {
    std::sort(names.begin(), names.end());
    for (const auto& name : names) msg += "\n  - " + name;
    throw ValueError(msg);
  };
  if (!unclassified.empty()) {
    fail(
        "The following multi-output ops left an argument unclassified. An operator that "
        "returns several values must reach a verdict about every argument it takes: an "
        "undeclared slot defaults to Read, which is indistinguishable from a destination "
        "tile smuggled into the argument list. Add .set_arg_effect(<index>, ...) for each "
        "argument — or .no_arg_writes() when the operator writes through none of them:",
        std::move(unclassified));
  }
  if (!leaked_destinations.empty()) {
    fail(
        "The following multi-output ops write an argument without declaring its role. Use "
        "set_workspace_arg for scratch, set_mutable_session_arg for queue state, or set_tuple_result_alias "
        "for a tensor "
        "returned as a tuple element. Fresh tile outputs belong in the TupleType:",
        std::move(leaked_destinations));
  }
  if (!workspace_out_of_range.empty()) {
    fail(
        "The following multi-output ops declare a workspace argument that does not exist. "
        "set_workspace_arg() names a positional argument, so an index past the end is a "
        "typo that silently protects nothing:",
        std::move(workspace_out_of_range));
  }
  if (!workspace_never_written.empty()) {
    fail(
        "The following multi-output ops declare a workspace argument the operator never "
        "writes. A workspace is hardware-written scratch by definition; declaring one Read "
        "-- or reaching that classification through no_arg_writes() -- leaves direction "
        "inference treating a real write as a read, which is the dropped dependency edge "
        "this check exists to prevent. Declare the effect that matches what the hardware "
        "does, or drop the workspace marker:",
        std::move(workspace_never_written));
  }
  if (!reuses_input.empty()) {
    fail(
        "The following multi-output ops declare set_output_reuses_input(N). With several "
        "results, \"the output reuses input N\" cannot say which one, and InitMemRef would "
        "bind the tuple temporary rather than an element. Drop the declaration:",
        std::move(reuses_input));
  }
}

void OpRegistry::ValidateBufferOps() const {
  for (const auto& [name, entry] : registry_) {
    entry.ValidateIRStage();
  }
}

}  // namespace ir
}  // namespace pypto
