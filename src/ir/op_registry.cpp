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
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

/// Format an IR span as a " at <file>:<line>:<col>" suffix, empty when unknown.
///
/// Kept out of line so `Span::to_string()` is only paid on the error path.
std::string LocationSuffix(const Span& span) { return span.is_valid() ? " at " + span.to_string() : ""; }

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
  return CreateImpl(op_name, args, kwargs, std::move(span), /*allow_internal=*/true);
}

CallPtr OpRegistry::CreateUserFacing(const std::string& op_name, const std::vector<ExprPtr>& args,
                                     Span span) const {
  return CreateUserFacing(op_name, args, {}, std::move(span));
}

CallPtr OpRegistry::CreateUserFacing(const std::string& op_name, const std::vector<ExprPtr>& args,
                                     const std::vector<std::pair<std::string, std::any>>& kwargs,
                                     Span span) const {
  return CreateImpl(op_name, args, kwargs, std::move(span), /*allow_internal=*/false);
}

CallPtr OpRegistry::CreateInternal(const std::string& op_name, const std::vector<ExprPtr>& args,
                                   Span span) const {
  return CreateInternal(op_name, args, {}, std::move(span));
}

CallPtr OpRegistry::CreateInternal(const std::string& op_name, const std::vector<ExprPtr>& args,
                                   const std::vector<std::pair<std::string, std::any>>& kwargs,
                                   Span span) const {
  return CreateImpl(op_name, args, kwargs, std::move(span), /*allow_internal=*/true);
}

CallPtr OpRegistry::CreateImpl(const std::string& op_name, const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs, Span span,
                               bool allow_internal) const {
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

  // Get operator instance (shared definition)
  OpPtr op = entry.GetOp();

  // Validate kwargs against allowed attributes (stored in Op)
  if (!kwargs.empty()) {
    const auto& allowed_kwargs = op->GetAttrs();
    if (!allowed_kwargs.empty()) {
      ValidateKwargs(kwargs, allowed_kwargs, op_name);
    }
  }

  const auto& deduce_type_fn = entry.GetDeduceType();

  // Deduce result type (pass args and kwargs separately)
  TypePtr result_type;
  try {
    result_type = deduce_type_fn(args, kwargs);
  } catch (const Error& e) {
    // Append the IR location but keep the concrete exception type and the stack trace
    // captured at the original throw. Flattening every PyPTO exception to ValueError
    // here erased the CHECK / INTERNAL_CHECK distinction for all op type deduction.
    e.RethrowWithMessage(std::string(e.what()) + LocationSuffix(span));
  } catch (const std::exception& e) {
    // Non-PyPTO exceptions (e.g. std::bad_any_cast from a wrong-typed kwarg) stay
    // ValueError: they are reachable from user input and carry no PyPTO trace to keep.
    throw ValueError(std::string(e.what()) + LocationSuffix(span));
  }
  INTERNAL_CHECK_SPAN(result_type, span) << "Type deduction failed for '" + op_name + "'";

  // The declared output arity and the deduced shape must agree. A mismatch means
  // the registration and its f_deduce_type disagree about what the operator
  // produces, which would surface much later as a null element var inside
  // multi-output codegen. The reverse direction matters just as much: a tuple
  // result nobody declared has no arity for codegen to read, so its elements
  // would never be resolved.
  const size_t declared_arity = entry.GetOutputArity();
  auto deduced_tuple = As<TupleType>(result_type);
  if (declared_arity > 1) {
    INTERNAL_CHECK_SPAN(deduced_tuple, span)
        << "Internal error: '" << op_name << "' declares set_output_arity(" << declared_arity
        << ") but deduced a non-tuple " << result_type->TypeName();
    INTERNAL_CHECK_SPAN(deduced_tuple->types_.size() == declared_arity, span)
        << "Internal error: '" << op_name << "' declares set_output_arity(" << declared_arity
        << ") but deduced a TupleType with " << deduced_tuple->types_.size() << " elements";
  } else {
    INTERNAL_CHECK_SPAN(!deduced_tuple, span)
        << "Internal error: '" << op_name << "' deduced a TupleType result without declaring "
        << "set_output_arity(" << deduced_tuple->types_.size()
        << "); multi-output codegen reads the arity from the registry and would not "
           "resolve this call's elements";
  }

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
      // A written argument is either scratch the hardware needs or a result the
      // caller reads. The first is legitimate and must say so; the second is a
      // destination that belongs in the TupleType.
      if (entry.MayWriteArg(i) && !entry.IsWorkspaceArg(i)) {
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
        "The following multi-output ops write through an argument that was never declared "
        "scratch. A written argument is either a workspace the hardware needs — say so with "
        ".set_workspace_arg(<index>) — or a destination the caller reads, which must be an "
        "element of the deduced TupleType instead. A destination in the argument list makes "
        "the caller pre-allocate a buffer InitMemRef owns:",
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

}  // namespace ir
}  // namespace pypto
