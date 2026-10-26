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

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/python_printer.h"

namespace pypto {
namespace ir {

std::string IRPythonPrinter::Print(const TypePtr& type) {
  // Buffer IR is internal: print native constructors without adding DSL types.
  if (As<VoidType>(type)) {
    return "pypto.ir.VoidType()";
  }
  if (auto buffer = As<BufferType>(type)) {
    std::ostringstream oss;
    auto print_dims = [&oss](const std::vector<int64_t>& dims) {
      oss << "[";
      for (size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << dims[i];
      }
      oss << "]";
    };
    oss << "pypto.ir.BufferType(";
    print_dims(buffer->shape_);
    oss << ", pypto.ir.DataType." << DataTypeToString(buffer->dtype_) << ", pypto.ir.MemorySpace."
        << MemorySpaceToString(buffer->memory_space_) << ", valid_shape=";
    print_dims(buffer->valid_shape_);
    oss << ", blayout=pypto.ir.TileLayout." << TileLayoutToString(buffer->blayout_)
        << ", slayout=pypto.ir.TileLayout." << TileLayoutToString(buffer->slayout_)
        << ", fractal=" << buffer->fractal_ << ", pad=pypto.ir.PadValue.";
    switch (buffer->pad_) {
      case PadValue::null:
        oss << "null";
        break;
      case PadValue::zero:
        oss << "zero";
        break;
      case PadValue::max:
        oss << "max";
        break;
      case PadValue::min:
        oss << "min";
        break;
    }
    oss << ", compact=pypto.ir.CompactMode." << CompactModeToString(buffer->compact_) << ")";
    return oss.str();
  }
  if (auto multi_buffer = As<MultiBufferType>(type)) {
    return "pypto.ir.MultiBufferType(" + Print(multi_buffer->element_type_) + ", " +
           std::to_string(multi_buffer->slot_count_) + ")";
  }
  if (auto reg = As<RegType>(type)) {
    return prefix_ + ".Reg[" + std::to_string(reg->shape_) + ", " + prefix_ + "." +
           DataTypeToString(reg->dtype_) + ", " + prefix_ + ".RegKind." +
           (reg->kind_ == RegKind::VecReg ? "VecReg" : "MaskReg") + "]";
  }
  if (auto scalar_type = As<ScalarType>(type)) {
    // Print as pl.Scalar[pl.INT64] for proper round-trip support
    return prefix_ + ".Scalar[" + prefix_ + "." + DataTypeToString(scalar_type->dtype_) + "]";
  }

  // Tensor / DistributedTensor share the same rendering surface — only the
  // subscript head differs (``pl.Tensor`` vs ``pld.DistributedTensor``). Note
  // ``As<TensorType>`` is precise-match and would not fire for the subclass,
  // so dispatch on DistributedTensorType first and pass it through the
  // TensorType base for shared field access.
  auto dt_tensor = As<DistributedTensorType>(type);
  TensorTypePtr tensor_type;
  std::string tensor_head;
  if (dt_tensor) {
    tensor_type = dt_tensor;
    tensor_head = "pld.DistributedTensor";
  } else if (auto plain_tensor = As<TensorType>(type)) {
    tensor_type = plain_tensor;
    tensor_head = prefix_ + ".Tensor";
  }
  if (tensor_type) {
    std::ostringstream oss;
    oss << tensor_head << "[[";
    PrintShapeDims(oss, tensor_type->shape_);
    oss << "], " << prefix_ << "." << DataTypeToString(tensor_type->dtype_);

    // Add optional tensor_view parameter if present.
    // Always emit something when tensor_view is set so that print->parse roundtrip
    // preserves presence: structural equality distinguishes present vs absent.
    if (tensor_type->tensor_view_.has_value()) {
      auto tv_str = PrintTensorView(tensor_type->tensor_view_.value(), tensor_type->shape_);
      if (!tv_str.empty()) {
        oss << ", " << tv_str;
      } else {
        oss << ", " << prefix_ << ".TensorView()";
      }
    }

    // Add optional memref as positional arg
    if (tensor_type->memref_.has_value()) {
      oss << ", " << PrintMemRef(*tensor_type->memref_.value());
    }

    if (dt_tensor && dt_tensor->window_buffer_.has_value()) {
      oss << ", " << prefix_ << "._dump.window_ref("
          << std::quoted(GetWindowBufferName(*dt_tensor->window_buffer_)) << ")";
    }

    oss << "]";
    return oss.str();
  }

  if (auto array_type = As<ArrayType>(type)) {
    std::ostringstream oss;
    // Subscript-style: pl.Array[N, dtype]. Use PrintShapeDims (which routes through
    // PrintExprForType into a temp stream) so the recursive printing doesn't reset
    // the main stream_ buffer.
    oss << prefix_ << ".Array[";
    PrintShapeDims(oss, array_type->shape_);
    oss << ", " << prefix_ << "." << DataTypeToString(array_type->dtype_);
    oss << "]";
    return oss.str();
  }

  if (auto tile_type = As<TileType>(type)) {
    std::ostringstream oss;
    // Subscript-style: pl.Tile[[shape], dtype]
    oss << prefix_ << ".Tile[[";
    PrintShapeDims(oss, tile_type->shape_);
    oss << "], " << prefix_ << "." << DataTypeToString(tile_type->dtype_);

    // Add optional memref as positional arg
    if (tile_type->memref_.has_value()) {
      oss << ", " << PrintMemRef(*tile_type->memref_.value());
    }

    if (tile_type->memory_space_.has_value()) {
      auto mem_str = MemorySpaceToString(tile_type->memory_space_.value());
      oss << ", " << prefix_ << ".Mem." << mem_str;
    }

    // Pick the view to render, then print once. explicit_layout_ resolves an
    // absent canonical view (tile_view_ == nullopt) to its implicit form via
    // GetEffectiveTileView so the annotation states its real blayout/slayout/
    // fractal (issue #2088) — PrintTileView forces those fields in that mode, so
    // it never returns "" for the explicit view. The concise branch only renders
    // a present view; when a present view happens to match the implicit one
    // (possible only on incoherent IR — canonical IR stores that as nullopt), the
    // empty-string fallback keeps the output parseable so TileTypeCoherence can
    // flag the real bug.
    std::optional<TileView> view;
    if (explicit_layout_) {
      view = tile_view_semantics::GetEffectiveTileView(*tile_type);
    } else if (tile_type->tile_view_.has_value()) {
      view = tile_type->tile_view_.value();
    }
    if (view.has_value()) {
      auto view_str = PrintTileView(*view, tile_type->shape_, tile_type->memory_space_);
      oss << ", " << (view_str.empty() ? prefix_ + ".TileView()" : view_str);
    }

    oss << "]";
    return oss.str();
  }

  if (auto tuple_type = As<TupleType>(type)) {
    std::ostringstream oss;
    if (tuple_type->types_.empty()) {
      oss << prefix_ << ".Tuple[()]";
    } else {
      oss << prefix_ << ".Tuple[";
      for (size_t i = 0; i < tuple_type->types_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << Print(tuple_type->types_[i]);
      }
      oss << "]";
    }
    return oss.str();
  }

  if (auto memref_type = As<MemRefType>(type)) {
    return prefix_ + ".MemRefType";
  }

  if (auto ptr_type = As<PtrType>(type)) {
    return prefix_ + ".Ptr";
  }

  if (As<WindowBufferType>(type)) {
    return prefix_ + "._dump.WindowBuffer";
  }

  if (As<CommCtxType>(type)) {
    // Singleton marker — printed with the public DSL wrapper name, matching the
    // ``pl.AsyncEvent`` family below. ``pld`` is hardcoded rather than derived
    // from ``prefix_`` because that prefix aliases only ``pypto.language``; the
    // text parser injects ``pld`` into the exec namespace itself.
    return "pld.CommCtx";
  }

  // Async-prefetch handle markers — fieldless singletons, rendered as bare
  // public wrapper attributes on the main DSL namespace so the output can be
  // evaluated by normal Python as well as round-tripped through the parser.
  if (As<PrefetchAsyncContextType>(type)) {
    return prefix_ + ".PrefetchAsyncContext";
  }

  if (As<AsyncEventType>(type)) {
    return prefix_ + ".AsyncEvent";
  }

  if (As<AsyncSessionType>(type)) {
    return prefix_ + ".AsyncSession";
  }

  return prefix_ + "._dump.Unknown";
}

std::string IRPythonPrinter::PrintExprForType(const ExprPtr& expr) {
  // Bare dimensions reparse as INDEX. Preserve every other stored dtype.
  if (auto const_int = As<ConstInt>(expr); const_int && const_int->dtype() == DataType::INDEX) {
    return std::to_string(const_int->value_);
  }
  if (auto var = As<Var>(expr)) {
    return GetVarName(var.get());
  }
  // Composite dims retain their enclosing rename context and typed constant
  // leaves so print/parse preserves bindings and does not fold the IR tree.
  // Simplification stays the Simplify pass's job.
  return PrintSubExpr(expr, /*typed_index=*/true);
}

void IRPythonPrinter::PrintShapeDims(std::ostringstream& oss, const std::vector<ExprPtr>& shape) {
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << PrintExprForType(shape[i]);
  }
}

// Helper methods for MemRef and TileView printing
std::string IRPythonPrinter::PrintMemRef(const MemRef& memref) {
  std::ostringstream oss;

  // An author-declared allocation prints in the one-argument form: it carries no
  // size or address to print, since InitMemRef derives both. Printing it with the
  // invented pair would make it indistinguishable from a compiler allocation on
  // reparse.
  if (memref.is_pinned_) {
    oss << prefix_ << ".MemRef(";
    if (!printing_signature_ && body_defined_vars_.count(memref.base_.get())) {
      oss << GetVarName(memref.base_.get());
    } else {
      oss << std::quoted(GetVarName(memref.base_.get()));
    }
    // `slots=` is what makes the subscript below legal on reparse, so it has to
    // ride along; omitted at 1 so an unsubscripted declaration prints as before.
    if (memref.slot_count_ > 1) oss << ", slots=" << memref.slot_count_;
    oss << ")";
    if (memref.slot_index_.has_value() && *memref.slot_index_) {
      // The index may be a runtime expression, so print it through the expression
      // printer rather than assuming a constant.
      oss << "[" << PrintSubExpr(*memref.slot_index_) << "]";
    }
    return oss.str();
  }

  oss << prefix_ << ".MemRef(";

  // Base Ptrs defined in the function body (by alloc statements) are printed as
  // bare variable references; everything else uses a string literal (forward
  // references in parameter annotations, standalone type printing, etc.).
  //
  // A signature annotation is exactly such a forward reference: it is emitted
  // before the body, so even a base Ptr that *is* body-defined is not yet bound
  // there and must take the string form, or the reparse fails with NameError.
  if (!printing_signature_ && body_defined_vars_.count(memref.base_.get())) {
    oss << GetVarName(memref.base_.get());
  } else {
    oss << "\"" << GetVarName(memref.base_.get()) << "\"";
  }

  // Print byte offset off the main stream, but with this printer's naming state:
  // a dynamic offset (a tile.slice view) names loop vars, which may have been
  // disambiguated.
  oss << ", " << PrintSubExpr(memref.byte_offset_, true);

  // Print size
  oss << ", " << memref.size_;

  // Slot geometry survives InitMemRef (it says which slot of what this MemRef is,
  // not merely how to compute its offset), so a resolved slot has to print it too
  // or the reparsed MemRef silently becomes an ordinary one at the same offset —
  // and PTO codegen would emit N unrelated allocs instead of one multi-buffer.
  if (memref.slot_count_ > 1) oss << ", slots=" << memref.slot_count_;
  oss << ")";
  if (memref.slot_index_.has_value() && *memref.slot_index_) {
    // The index may be a runtime expression, so print it through the expression
    // printer rather than assuming a constant.
    oss << "[" << PrintSubExpr(*memref.slot_index_) << "]";
  }
  return oss.str();
}

std::string IRPythonPrinter::PrintTileView(const TileView& tile_view, const std::vector<ExprPtr>& tile_shape,
                                           const std::optional<MemorySpace>& memory_space) {
  // Caller already gated on has_value() (or passed the resolved effective view in
  // explicit_layout_ mode); a present view is non-implicit by the TileType
  // canonical-encoding invariant, so always render the explicit form.
  std::ostringstream oss;
  oss << prefix_ << ".TileView(";

  bool first = true;
  auto maybe_comma = [&]() {
    if (!first) oss << ", ";
    first = false;
  };

  // Compute the implicit view so we can elide layout fields that match it — but
  // only in concise mode. Under explicit_layout_ the blayout/slayout/fractal
  // guards short-circuit on `explicit_layout_ ||` and never read it, so skipping
  // it avoids a wasted GetImplicitTileView (already folded into the effective
  // view the caller resolved for absent-view tiles).
  TileView implicit_view;
  if (!explicit_layout_) {
    implicit_view = tile_view_semantics::GetImplicitTileView(tile_shape, memory_space);
  }

  // valid_shape — omit if it matches the parent tile's shape
  bool valid_shape_matches = tile_view.valid_shape.empty() ||
                             tile_view_semantics::ShapeExprListsEquivalent(tile_view.valid_shape, tile_shape);
  if (!valid_shape_matches) {
    maybe_comma();
    oss << "valid_shape=[";
    for (size_t i = 0; i < tile_view.valid_shape.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << PrintExprForType(tile_view.valid_shape[i]);
    }
    oss << "]";
  }

  // stride — omit if empty
  if (!tile_view.stride.empty()) {
    maybe_comma();
    oss << "stride=[";
    for (size_t i = 0; i < tile_view.stride.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << PrintExprForType(tile_view.stride[i]);
    }
    oss << "]";
  }

  // start_offset — omit if null
  if (tile_view.start_offset) {
    maybe_comma();
    oss << "start_offset=";
    oss << PrintExprForType(tile_view.start_offset);
  }

  // blayout — omit if matches the implicit view for this shape+memory_space
  // (kept in explicit_layout_ mode so every tile self-describes its layout).
  if (explicit_layout_ || tile_view.blayout != implicit_view.blayout) {
    maybe_comma();
    oss << "blayout=" << prefix_ << ".TileLayout.";
    switch (tile_view.blayout) {
      case TileLayout::none_box:
        oss << "none_box";
        break;
      case TileLayout::row_major:
        oss << "row_major";
        break;
      case TileLayout::col_major:
        oss << "col_major";
        break;
    }
  }

  // slayout — omit if matches the implicit view for this shape+memory_space
  // (kept in explicit_layout_ mode so every tile self-describes its layout).
  if (explicit_layout_ || tile_view.slayout != implicit_view.slayout) {
    maybe_comma();
    oss << "slayout=" << prefix_ << ".TileLayout.";
    switch (tile_view.slayout) {
      case TileLayout::none_box:
        oss << "none_box";
        break;
      case TileLayout::row_major:
        oss << "row_major";
        break;
      case TileLayout::col_major:
        oss << "col_major";
        break;
    }
  }

  // fractal — omit if matches the implicit view for this shape+memory_space
  // (kept in explicit_layout_ mode so every tile self-describes its layout).
  if (explicit_layout_ || tile_view.fractal != implicit_view.fractal) {
    maybe_comma();
    oss << "fractal=" << tile_view.fractal;
  }

  // pad — omit if null (default)
  if (tile_view.pad != PadValue::null) {
    maybe_comma();
    oss << "pad=" << prefix_ << ".PadValue.";
    switch (tile_view.pad) {
      case PadValue::null:
        oss << "null";
        break;
      case PadValue::zero:
        oss << "zero";
        break;
      case PadValue::max:
        oss << "max";
        break;
      case PadValue::min:
        oss << "min";
        break;
    }
  }

  // compact — omit if null (default)
  if (tile_view.compact != CompactMode::null) {
    maybe_comma();
    oss << "compact=" << prefix_ << ".CompactMode.";
    switch (tile_view.compact) {
      case CompactMode::null:
        oss << "null";
        break;
      case CompactMode::normal:
        oss << "normal";
        break;
    }
  }

  // If all fields were at defaults, return empty string to skip tile_view entirely
  if (first) return "";

  oss << ")";
  return oss.str();
}

std::string IRPythonPrinter::PrintTensorView(const TensorView& tensor_view,
                                             const std::vector<ExprPtr>& tensor_shape) {
  std::ostringstream oss;
  oss << prefix_ << ".TensorView(";

  bool first = true;
  auto maybe_comma = [&]() {
    if (!first) oss << ", ";
    first = false;
  };

  // valid_shape — omit if it matches the parent tensor's shape
  bool valid_shape_matches = (tensor_view.valid_shape.size() == tensor_shape.size());
  if (valid_shape_matches) {
    for (size_t i = 0; i < tensor_shape.size(); ++i) {
      const auto& vs_expr = tensor_view.valid_shape[i];
      const auto& ts_expr = tensor_shape[i];
      if (vs_expr == ts_expr) continue;
      auto vs = As<ConstInt>(vs_expr);
      auto ts = As<ConstInt>(ts_expr);
      if (!vs || !ts || vs->value_ != ts->value_) {
        valid_shape_matches = false;
        break;
      }
    }
  }
  if (!valid_shape_matches && !tensor_view.valid_shape.empty()) {
    maybe_comma();
    oss << "valid_shape=[";
    for (size_t i = 0; i < tensor_view.valid_shape.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << PrintExprForType(tensor_view.valid_shape[i]);
    }
    oss << "]";
  }

  bool has_stride = !tensor_view.stride.empty();
  bool has_non_default_layout = (tensor_view.layout != TensorLayout::ND);
  bool has_non_default_pad = (tensor_view.pad != PadValue::null);

  // If all fields are at defaults, skip TensorView entirely
  if (first && !has_stride && !has_non_default_layout && !has_non_default_pad) return "";

  // When TensorView is non-trivial, always emit both stride and layout to satisfy
  // the C++ constructor signature TensorView(stride, layout, valid_shape=[], pad=null).
  // Omitting either required arg causes TypeError when Python eagerly evaluates
  // function parameter annotations during exec() in the text parser.
  maybe_comma();
  oss << "stride=[";
  for (size_t i = 0; i < tensor_view.stride.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << PrintExprForType(tensor_view.stride[i]);
  }
  oss << "]";

  maybe_comma();
  oss << "layout=" << prefix_ << ".TensorLayout." << TensorLayoutToString(tensor_view.layout);

  // pad — omit if null (default)
  if (has_non_default_pad) {
    maybe_comma();
    oss << "pad=" << prefix_ << ".PadValue.";
    switch (tensor_view.pad) {
      case PadValue::null:
        oss << "null";
        break;
      case PadValue::zero:
        oss << "zero";
        break;
      case PadValue::max:
        oss << "max";
        break;
      case PadValue::min:
        oss << "min";
        break;
    }
  }

  oss << ")";
  return oss.str();
}

}  // namespace ir
}  // namespace pypto
