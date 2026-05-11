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

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <any>
#include <cctype>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../module.h"
#include "pypto/core/any_cast.h"
#include "pypto/core/common.h"
#include "pypto/core/error.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/program.h"
#include "pypto/ir/reflection/field_visitor.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/serialization/deserializer.h"
#include "pypto/ir/serialization/serializer.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/op_conversion_registry.h"
#include "pypto/ir/transforms/printer.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/parent_stmt_analysis.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace nb = nanobind;

namespace pypto {
namespace python {

using namespace pypto::ir;  // NOLINT(build/namespaces)
using pypto::DataType;

template <typename T>
bool TryConvertAnyToPy(const std::any& value, nb::object& out) {
  if (value.type() != typeid(T)) {
    return false;
  }
  out = nb::cast(AnyCastRef<T>(value, "converting to Python"));
  return true;
}

template <typename... Ts>
nb::object AnyToPyObject(const std::any& value, const std::string& key) {
  nb::object out;
  if ((TryConvertAnyToPy<Ts>(value, out) || ...)) {
    return out;
  }
  throw pypto::TypeError("Attribute '" + key + "' has unsupported type");
}

// Helper to bind a single field using reflection
template <typename ClassType, typename PyClassType, typename FieldDesc>
void BindField(PyClassType& nb_class, const FieldDesc& desc) {
  nb_class.def_ro(desc.name, desc.field_ptr);
}

// Helper to bind all fields from a tuple of field descriptors
template <typename ClassType, typename PyClassType, typename DescTuple, std::size_t... Is>
void BindFieldsImpl(PyClassType& nb_class, const DescTuple& descriptors, std::index_sequence<Is...>) {
  (BindField<ClassType>(nb_class, std::get<Is>(descriptors)), ...);
}

// Main function to bind all fields using reflection
template <typename ClassType, typename PyClassType>
void BindFields(PyClassType& nb_class) {
  constexpr auto descriptors = ClassType::GetFieldDescriptors();
  constexpr auto num_fields = std::tuple_size_v<decltype(descriptors)>;
  BindFieldsImpl<ClassType>(nb_class, descriptors, std::make_index_sequence<num_fields>{});
}

// Helper function to convert nb::dict to vector<pair<string, any>>
std::vector<std::pair<std::string, std::any>> ConvertKwargsDict(const nb::dict& kwargs_dict) {
  std::vector<std::pair<std::string, std::any>> kwargs;
  for (auto item : kwargs_dict) {
    std::string key = nb::cast<std::string>(item.first);

    // Try to cast to common types
    // NOTE: Check DataType/MemorySpace/PipeType/CoreType/PadValue BEFORE int, and bool BEFORE int
    if (nb::isinstance<DataType>(item.second)) {
      kwargs.emplace_back(key, nb::cast<DataType>(item.second));
    } else if (nb::isinstance<MemorySpace>(item.second)) {
      kwargs.emplace_back(key, nb::cast<MemorySpace>(item.second));
    } else if (nb::isinstance<TensorLayout>(item.second)) {
      kwargs.emplace_back(key, nb::cast<TensorLayout>(item.second));
    } else if (nb::isinstance<TileLayout>(item.second)) {
      kwargs.emplace_back(key, nb::cast<TileLayout>(item.second));
    } else if (nb::isinstance<PipeType>(item.second)) {
      // Cast enum to int for storage
      kwargs.emplace_back(key, static_cast<int>(nb::cast<PipeType>(item.second)));
    } else if (nb::isinstance<CoreType>(item.second)) {
      // Cast enum to int for storage
      kwargs.emplace_back(key, static_cast<int>(nb::cast<CoreType>(item.second)));
    } else if (nb::isinstance<SplitMode>(item.second)) {
      // Cast enum to int for storage
      kwargs.emplace_back(key, static_cast<int>(nb::cast<SplitMode>(item.second)));
    } else if (nb::isinstance<PadValue>(item.second)) {
      kwargs.emplace_back(key, nb::cast<PadValue>(item.second));
    } else if (nb::isinstance<LoopOrigin>(item.second)) {
      kwargs.emplace_back(key, nb::cast<LoopOrigin>(item.second));
    } else if (nb::isinstance<ArgDirection>(item.second)) {
      kwargs.emplace_back(key, nb::cast<ArgDirection>(item.second));
    } else if (nb::isinstance<Expr>(item.second)) {
      // IR expression (e.g. SpmdScopeStmt core_num stored as Function attr).
      kwargs.emplace_back(key, nb::cast<ExprPtr>(item.second));
    } else if (nb::isinstance<nb::bool_>(item.second)) {
      kwargs.emplace_back(key, nb::cast<bool>(item.second));
    } else if (nb::isinstance<nb::int_>(item.second)) {
      kwargs.emplace_back(key, nb::cast<int>(item.second));
    } else if (nb::isinstance<nb::str>(item.second)) {
      kwargs.emplace_back(key, nb::cast<std::string>(item.second));
    } else if (nb::isinstance<nb::float_>(item.second)) {
      kwargs.emplace_back(key, nb::cast<double>(item.second));
    } else if (nb::isinstance<nb::list>(item.second) || nb::isinstance<nb::tuple>(item.second)) {
      // Lists/tuples carry exactly one element type, dispatched by attr key:
      //   - kAttrArgDirections          -> vector<ArgDirection>
      //   - kAttrArgDirectionOverrides  -> vector<int32_t>
      //   - kAttrManualDepEdges /
      //     kAttrUserManualDepEdges     -> vector<VarPtr>
      // Inferring from the first element would silently accept mismatched
      // payloads (e.g. ``manual_dep_edges=[1]``) and fail later in codegen
      // instead of raising at parse time.
      auto seq = nb::cast<nb::sequence>(item.second);
      if (key == kAttrArgDirectionOverrides) {
        std::vector<int32_t> idxs;
        for (auto elem : seq) {
          if (nb::isinstance<nb::bool_>(elem) || !nb::isinstance<nb::int_>(elem)) {
            throw pypto::TypeError("Unsupported list element type for key: " + key + " (expected int)");
          }
          int64_t v = nb::cast<int64_t>(elem);
          if (v < std::numeric_limits<int32_t>::min() || v > std::numeric_limits<int32_t>::max()) {
            throw pypto::ValueError("List value " + std::to_string(v) + " for key: " + key +
                                    " is out of int32 range");
          }
          idxs.push_back(static_cast<int32_t>(v));
        }
        kwargs.emplace_back(key, std::move(idxs));
      } else if (key == kAttrManualDepEdges || key == kAttrUserManualDepEdges) {
        std::vector<VarPtr> vars;
        for (auto elem : seq) {
          if (!nb::isinstance<Var>(elem)) {
            throw pypto::TypeError("Unsupported list element type for key: " + key + " (expected Var)");
          }
          vars.push_back(nb::cast<VarPtr>(elem));
        }
        kwargs.emplace_back(key, std::move(vars));
      } else {
        // Default: kAttrArgDirections and any future ArgDirection list key.
        std::vector<ArgDirection> dirs;
        for (auto elem : seq) {
          if (!nb::isinstance<ArgDirection>(elem)) {
            throw pypto::TypeError("Unsupported list element type for key: " + key +
                                   " (expected ArgDirection)");
          }
          dirs.push_back(nb::cast<ArgDirection>(elem));
        }
        kwargs.emplace_back(key, std::move(dirs));
      }
    } else {
      throw pypto::TypeError("Unsupported kwarg type for key: " + key);
    }
  }
  return kwargs;
}

std::vector<std::pair<std::string, std::any>> ConvertAttrsFromPython(const nb::object& attrs_or_none) {
  std::vector<std::pair<std::string, std::any>> attrs;
  if (attrs_or_none.is_none()) {
    // no-op
  } else if (nb::isinstance<nb::dict>(attrs_or_none)) {
    attrs = ConvertKwargsDict(nb::cast<nb::dict>(attrs_or_none));
  } else if (nb::isinstance<nb::list>(attrs_or_none)) {
    for (auto item : nb::cast<nb::list>(attrs_or_none)) {
      auto tup = nb::cast<nb::tuple>(item);
      nb::dict d;
      d[tup[0]] = tup[1];
      auto converted = ConvertKwargsDict(d);
      attrs.push_back(converted[0]);
    }
  } else {
    throw pypto::TypeError("attrs must be a dict, list of (key, value) tuples, or None");
  }
  // Ergonomic auto-wrap: Function attrs["core_num"] is typed as ExprPtr, but
  // users and text-parser reparse sites commonly supply a plain int — wrap it
  // as ConstInt(DataType::INDEX) so the codegen-side ExprPtr read is uniform.
  for (auto& [key, value] : attrs) {
    if (key == "core_num" && value.type() == typeid(int)) {
      auto n = std::any_cast<int>(value);
      value = std::static_pointer_cast<const Expr>(
          std::make_shared<const ConstInt>(n, DataType::INDEX, Span::unknown()));
    }
  }
  return attrs;
}

/// Conditionally apply the registered format callback.
/// When `format` is true, post-processes `code` through the callback (e.g., ruff).
/// When false, returns `code` unchanged — useful for tests that match exact substrings.
std::string MaybeFormat(const std::string& code, bool format) {
  return format ? ApplyFormatCallback(code) : code;
}

void BindIR(nb::module_& m) {
  nb::module_ ir = m.def_submodule("ir", "PyPTO IR (Intermediate Representation) module");

  // Span - value type, copy semantics
  nb::class_<Span>(ir, "Span", "Source location information tracking file, line, and column positions")
      .def(nb::init<std::string, int, int, int, int>(), nb::arg("filename"), nb::arg("begin_line"),
           nb::arg("begin_column"), nb::arg("end_line") = -1, nb::arg("end_column") = -1,
           "Create a source span")
      .def("to_string", &Span::to_string, "Convert span to string representation")
      .def("is_valid", &Span::is_valid, "Check if the span has valid coordinates")
      .def_static("unknown", &Span::unknown,
                  "Create an unknown/invalid span for cases where source location is unavailable")
      .def("__repr__", &Span::to_string)
      .def("__str__", &Span::to_string)
      .def_ro("filename", &Span::filename_, "Source filename")
      .def_ro("begin_line", &Span::begin_line_, "Beginning line (1-indexed)")
      .def_ro("begin_column", &Span::begin_column_, "Beginning column (1-indexed)")
      .def_ro("end_line", &Span::end_line_, "Ending line (1-indexed)")
      .def_ro("end_column", &Span::end_column_, "Ending column (1-indexed)");

  // Op - operation/function
  nb::class_<Op>(ir, "Op",
                 "Represents callable operations in the IR. Stores the schema of allowed kwargs (key -> type "
                 "mapping). Actual kwarg values are stored per-Call instance in Call.kwargs")
      .def(nb::init<std::string>(), nb::arg("name"), "Create an operation with the given name")
      .def_ro("name", &Op::name_, "Operation name")
      .def("has_attr", &Op::HasAttr, nb::arg("key"), "Check if a kwarg is registered in the schema")
      .def("get_attr_keys", &Op::GetAttrKeys, "Get all registered kwarg keys from the schema");

  // GlobalVar - global function reference
  nb::class_<GlobalVar, Op>(ir, "GlobalVar",
                            "Global variable reference for functions in a program. "
                            "Can be used in Call expressions to invoke functions within the same program.")
      .def(nb::init<std::string>(), nb::arg("name"),
           "Create a global variable reference with the given name");

  // Type - abstract base, const shared_ptr
  auto type_class = nb::class_<Type>(ir, "Type", "Base class for type representations");
  BindFields<Type>(type_class);
  type_class.def(
      "__str__", [](const TypePtr& self) { return ApplyFormatCallback(PythonPrint(self, "pl")); },
      "Python-style string representation");
  type_class.def(
      "__eq__", [](const TypePtr& self, const TypePtr& other) { return structural_equal(self, other); },
      "Equality comparison");
  type_class.def(
      "__hash__", [](const TypePtr& self) { return structural_hash(self); },
      "Hash by structural identity (consistent with __eq__)");

  // UnknownType - const shared_ptr
  auto unknown_type_class =
      nb::class_<UnknownType, Type>(ir, "UnknownType", "Unknown or unspecified type representation");
  unknown_type_class.def(nb::init<>(), "Create an unknown type");
  unknown_type_class.def_static(
      "get", []() { return GetUnknownType(); }, "Get the singleton UnknownType instance");
  BindFields<UnknownType>(unknown_type_class);

  // ScalarType - const shared_ptr
  auto scalar_type_class = nb::class_<ScalarType, Type>(ir, "ScalarType", "Scalar type representation");
  scalar_type_class.def(nb::init<DataType>(), nb::arg("dtype"), "Create a scalar type");
  BindFields<ScalarType>(scalar_type_class);

  // IRNode - abstract base, const shared_ptr
  auto irnode_class = nb::class_<IRNode>(ir, "IRNode", "Base class for all IR nodes");
  BindFields<IRNode>(irnode_class);
  irnode_class
      .def(
          "same_as", [](const IRNodePtr& self, const IRNodePtr& other) { return self == other; },
          nb::arg("other"), "Check if this IR node is the same as another IR node.")
      .def(
          "__str__", [](const IRNodePtr& self) { return ApplyFormatCallback(PythonPrint(self, "pl")); },
          "Python-style string representation")
      .def(
          "as_python",
          [](const IRNodePtr& self, const std::string& prefix, bool concise, bool format) {
            return MaybeFormat(PythonPrint(self, prefix, concise), format);
          },
          nb::arg("prefix") = "pl", nb::arg("concise") = false, nb::arg("format") = true,
          "Convert to Python-style string representation.\n\n"
          "Args:\n"
          "    prefix: Module prefix (default 'pl' for 'import pypto.language as pl')\n"
          "    concise: If true, omit intermediate type annotations (default false)\n"
          "    format: If true, apply registered format callback (default true)");

  // Expr - abstract base, const shared_ptr
  auto expr_class = nb::class_<Expr, IRNode>(ir, "Expr", "Base class for all expressions");
  BindFields<Expr>(expr_class);

  // ShapedType - abstract base for types with shape and optional memref
  auto shaped_type_class =
      nb::class_<ShapedType, Type>(ir, "ShapedType", "Base class for shaped types (tensors and tiles)");
  BindFields<ShapedType>(shaped_type_class);
  shaped_type_class.def_prop_ro("memory_space", &ShapedType::GetMemorySpace,
                                "Canonical memory space for this shaped type");
  shaped_type_class.def(
      "shares_memref_with",
      [](const ShapedTypePtr& self, const ShapedTypePtr& other) {
        if (!self->memref_.has_value() || !other->memref_.has_value()) {
          return false;
        }
        return self->memref_.value().get() == other->memref_.value().get();
      },
      nb::arg("other"), "Check if this ShapedType shares the same MemRef object with another ShapedType");

  // TensorLayout enum - must be before TensorView and TensorType
  nb::enum_<TensorLayout>(ir, "TensorLayout", "Tensor layout enumeration")
      .value("ND", TensorLayout::ND, "ND layout")
      .value("DN", TensorLayout::DN, "DN layout")
      .value("NZ", TensorLayout::NZ, "NZ layout")
      .export_values();

  // PadValue enum - must be before both TensorView and TileView since both carry it
  nb::enum_<PadValue>(ir, "PadValue", "Pad mode enumeration for tile/tensor views")
      .value("null", PadValue::null, "No padding")
      .value("zero", PadValue::zero, "Zero padding")
      .value("max", PadValue::max, "Max value padding")
      .value("min", PadValue::min, "Min value padding")
      .export_values();

  // TensorView - struct for tensor view information - must be before TensorType
  nb::class_<TensorView>(ir, "TensorView",
                         "Tensor view representation with stride, layout, valid shape, and pad mode")
      .def(nb::init<>(), "Create an empty tensor view")
      .def(nb::init<const std::vector<ExprPtr>&, TensorLayout, const std::vector<ExprPtr>&, PadValue>(),
           nb::arg("stride"), nb::arg("layout"), nb::arg("valid_shape") = std::vector<ExprPtr>{},
           nb::arg("pad") = PadValue::null,
           "Create a tensor view with stride, layout, optional valid shape, and optional pad")
      .def(nb::init<const std::vector<int64_t>&, TensorLayout, const std::vector<int64_t>&, PadValue>(),
           nb::arg("stride"), nb::arg("layout"), nb::arg("valid_shape") = std::vector<int64_t>{},
           nb::arg("pad") = PadValue::null,
           "Create a tensor view with integer stride, layout, optional integer valid shape, and optional pad")
      .def_rw("stride", &TensorView::stride, "Stride for each dimension")
      .def_rw("layout", &TensorView::layout, "Tensor layout type")
      .def_rw("valid_shape", &TensorView::valid_shape, "Valid shape for each dimension")
      .def_rw("pad", &TensorView::pad, "Pad mode for out-of-valid-shape accesses");

  // ---------------------------------------------------------------------------
  // tensor_view_semantics free functions (RFC #1300 §2.2/§2.3)
  // Exposed under ir.tensor_view_semantics so passes/verifiers/tests share one
  // implementation of the canonical (shape, stride, layout) invariants.
  auto tvs = ir.def_submodule("tensor_view_semantics",
                              "Canonical-form helpers for TensorType.tensor_view_ (RFC #1300).");

  tvs.def("build_logical_strides_from_layout", &tensor_view_semantics::BuildLogicalStridesFromLayout,
          nb::arg("shape"), nb::arg("layout"),
          "Build packed canonical strides for (shape, layout). "
          "Raises ValueError on NZ layout or DN with rank < 2.");

  tvs.def(
      "derive_layout_from_strides",
      [](const std::vector<ExprPtr>& shape,
         const std::vector<ExprPtr>& stride) -> std::optional<TensorLayout> {
        return tensor_view_semantics::DeriveLayoutFromStrides(shape, stride);
      },
      nb::arg("shape"), nb::arg("stride"),
      "Statically derive layout from (shape, stride). "
      "Returns None for symbolic / non-canonical cases.");

  tvs.def(
      "check_canonical_view",
      [](const std::vector<ExprPtr>& shape, const std::vector<ExprPtr>& stride, TensorLayout layout,
         bool relaxed_symbolic) -> std::pair<bool, std::string> {
        auto r = tensor_view_semantics::CheckCanonicalView(shape, stride, layout, relaxed_symbolic);
        return {r.ok, r.reason};
      },
      nb::arg("shape"), nb::arg("stride"), nb::arg("layout"), nb::arg("relaxed_symbolic") = true,
      "Verify (shape, stride, layout) is canonical. Returns (ok, reason).");

  tvs.def("is_canonical_view", &tensor_view_semantics::IsCanonicalView, nb::arg("shape"), nb::arg("stride"),
          nb::arg("layout"), nb::arg("relaxed_symbolic") = true,
          "Convenience wrapper around check_canonical_view returning only the ok flag.");

  tvs.def("canonicalize_view", &tensor_view_semantics::CanonicalizeView, nb::arg("shape"), nb::arg("layout"),
          "Build a packed canonical TensorView for (shape, layout).");

  tvs.def("compute_shape_product", &tensor_view_semantics::ComputeShapeProduct, nb::arg("shape"),
          "Static product of shape dimensions; -1 if any dim is dynamic.");

  // TensorType - const shared_ptr
  auto tensor_type_class = nb::class_<TensorType, ShapedType>(ir, "TensorType", "Tensor type representation");
  tensor_type_class.def(nb::init<const std::vector<ExprPtr>&, DataType, std::optional<MemRefPtr>>(),
                        nb::arg("shape"), nb::arg("dtype"), nb::arg("memref") = nb::none(),
                        "Create a tensor type");
  tensor_type_class.def(nb::init<const std::vector<int64_t>&, DataType, std::optional<MemRefPtr>>(),
                        nb::arg("shape"), nb::arg("dtype"), nb::arg("memref") = nb::none(),
                        "Create a tensor type");
  tensor_type_class.def(
      nb::init<const std::vector<ExprPtr>&, DataType, std::optional<MemRefPtr>, std::optional<TensorView>>(),
      nb::arg("shape"), nb::arg("dtype"), nb::arg("memref") = nb::none(), nb::arg("tensor_view") = nb::none(),
      "Create a tensor type with optional memory reference and tensor view");
  tensor_type_class.def(
      nb::init<const std::vector<int64_t>&, DataType, std::optional<MemRefPtr>, std::optional<TensorView>>(),
      nb::arg("shape"), nb::arg("dtype"), nb::arg("memref") = nb::none(), nb::arg("tensor_view") = nb::none(),
      "Create a tensor type with constant shape, optional memory reference and tensor view");
  BindFields<TensorType>(tensor_type_class);

  // DistributedTensorType - subclass of TensorType used as the param type for
  // cross-rank ops (pld.tile.remote_load / pld.system.notify / pld.system.wait).
  // Distinguished from TensorType only by ObjectKind so verifiers can reject
  // plain Tensors. Otherwise mirrors TensorType's ctor overloads — memref /
  // tensor_view variants are equally valid on the distributed flavour.
  auto dist_tensor_type_class = nb::class_<DistributedTensorType, TensorType>(
      ir, "DistributedTensorType", "Tensor backed by a per-rank slice of a CommGroup HCCL window buffer");
  dist_tensor_type_class.def(nb::init<const std::vector<ExprPtr>&, DataType>(), nb::arg("shape"),
                             nb::arg("dtype"), "Create a distributed tensor type");
  dist_tensor_type_class.def(nb::init<const std::vector<int64_t>&, DataType>(), nb::arg("shape"),
                             nb::arg("dtype"), "Create a distributed tensor type with constant shape");
  dist_tensor_type_class.def(nb::init<const std::vector<ExprPtr>&, DataType, std::optional<MemRefPtr>>(),
                             nb::arg("shape"), nb::arg("dtype"), nb::arg("memref") = nb::none(),
                             "Create a distributed tensor type with optional memref");
  dist_tensor_type_class.def(nb::init<const std::vector<int64_t>&, DataType, std::optional<MemRefPtr>>(),
                             nb::arg("shape"), nb::arg("dtype"), nb::arg("memref") = nb::none(),
                             "Create a distributed tensor type with constant shape and optional memref");
  dist_tensor_type_class.def(
      nb::init<const std::vector<ExprPtr>&, DataType, std::optional<MemRefPtr>, std::optional<TensorView>>(),
      nb::arg("shape"), nb::arg("dtype"), nb::arg("memref") = nb::none(), nb::arg("tensor_view") = nb::none(),
      "Create a distributed tensor type with optional memref and tensor_view");
  dist_tensor_type_class.def(
      nb::init<const std::vector<int64_t>&, DataType, std::optional<MemRefPtr>, std::optional<TensorView>>(),
      nb::arg("shape"), nb::arg("dtype"), nb::arg("memref") = nb::none(), nb::arg("tensor_view") = nb::none(),
      "Create a distributed tensor type with constant shape, optional memref and tensor_view");
  BindFields<DistributedTensorType>(dist_tensor_type_class);

  // TileType - const shared_ptr
  auto tile_type_class =
      nb::class_<TileType, ShapedType>(ir, "TileType", "Tile type representation (multi-dimensional tensor)");
  tile_type_class.def(
      nb::init<const std::vector<ExprPtr>&, DataType, std::optional<MemRefPtr>, std::optional<TileView>,
               std::optional<MemorySpace>>(),
      nb::arg("shape"), nb::arg("dtype"), nb::arg("memref") = nb::none(), nb::arg("tile_view") = nb::none(),
      nb::arg("memory_space") = nb::none(),
      "Create a tile type (supports multi-dimensional tensors; code generation has constraints)");
  tile_type_class.def(
      nb::init<const std::vector<int64_t>&, DataType, std::optional<MemRefPtr>, std::optional<TileView>,
               std::optional<MemorySpace>>(),
      nb::arg("shape"), nb::arg("dtype"), nb::arg("memref") = nb::none(), nb::arg("tile_view") = nb::none(),
      nb::arg("memory_space") = nb::none(),
      "Create a tile type (supports multi-dimensional tensors; code generation has constraints)");
  tile_type_class.def(
      "get_effective_tile_view",
      [](const TileType& self) { return tile_view_semantics::GetEffectiveTileView(self); },
      "Return the effective TileView: the stored tile_view if present, else the implicit "
      "view derived from (shape, memory_space). An implicit view is stored as None under "
      "canonicalization, so callers that need to inspect layout fields should use this.");
  BindFields<TileType>(tile_type_class);

  // ArrayType - on-core fixed-size 1-D homogeneous array (C-stack local)
  auto array_type_class = nb::class_<ArrayType, ShapedType>(
      ir, "ArrayType",
      "On-core array type (lives on scalar register file / C stack). "
      "Homogeneous, 1-D, fixed extent, integer dtype. Cannot cross function boundaries.");
  array_type_class.def(nb::init<DataType, ExprPtr>(), nb::arg("dtype"), nb::arg("extent"),
                       "Create an ArrayType from a dtype and a ConstInt extent");
  array_type_class.def(nb::init<DataType, int64_t>(), nb::arg("dtype"), nb::arg("extent"),
                       "Create an ArrayType from a dtype and an int extent");
  array_type_class.def_prop_ro(
      "extent", [](const ArrayType& self) { return self.extent(); },
      "Number of elements (always a ConstInt)");
  BindFields<ArrayType>(array_type_class);

  // TupleType - const shared_ptr
  auto tuple_type_class =
      nb::class_<TupleType, Type>(ir, "TupleType", "Tuple type representation (contains multiple types)");
  tuple_type_class.def(nb::init<const std::vector<TypePtr>&>(), nb::arg("types"),
                       "Create a tuple type from a list of types");
  BindFields<TupleType>(tuple_type_class);

  // PtrType - allocation identity token type
  auto ptr_type_class =
      nb::class_<PtrType, Type>(ir, "PtrType", "Pointer type for allocation identity tokens");
  ptr_type_class.def(nb::init<>(), "Create a Ptr type");
  ptr_type_class.def_static("get", &GetPtrType, "Get the singleton PtrType instance");
  BindFields<PtrType>(ptr_type_class);

  // MemorySpace enum
  nb::enum_<MemorySpace>(ir, "MemorySpace", "Memory space enumeration")
      .value("DDR", MemorySpace::DDR, "DDR memory (off-chip)")
      .value("Vec", MemorySpace::Vec, "Vector/unified buffer (on-chip)")
      .value("Mat", MemorySpace::Mat, "Matrix/L1 buffer")
      .value("Left", MemorySpace::Left, "Left matrix operand buffer")
      .value("Right", MemorySpace::Right, "Right matrix operand buffer")
      .value("Acc", MemorySpace::Acc, "Accumulator buffer")
      .value("Bias", MemorySpace::Bias, "Bias buffer")
      .value("ScalarLocal", MemorySpace::ScalarLocal, "On-core scalar register file / C stack (ArrayType)")
      .export_values();

  // Short alias: ir.Mem = ir.MemorySpace
  ir.attr("Mem") = ir.attr("MemorySpace");

  // PipeType enum
  nb::enum_<PipeType>(ir, "PipeType", nb::is_arithmetic(), "Pipeline type enumeration")
      .value("MTE1", PipeType::MTE1, "Memory Transfer Engine 1")
      .value("MTE2", PipeType::MTE2, "Memory Transfer Engine 2")
      .value("MTE3", PipeType::MTE3, "Memory Transfer Engine 3")
      .value("M", PipeType::M, "Matrix Unit")
      .value("V", PipeType::V, "Vector Unit")
      .value("S", PipeType::S, "Scalar Unit")
      .value("FIX", PipeType::FIX, "Fix Pipe")
      .value("ALL", PipeType::ALL, "All Pipes")
      .export_values();

  // CoreType enum
  nb::enum_<CoreType>(ir, "CoreType", nb::is_arithmetic(), "Core type enumeration")
      .value("VECTOR", CoreType::VECTOR, "Vector Core")
      .value("CUBE", CoreType::CUBE, "Cube Core")
      .export_values();

  // TileLayout enum - must be before TileView
  nb::enum_<TileLayout>(ir, "TileLayout", "Tile layout enumeration")
      .value("none_box", TileLayout::none_box, "No layout constraint")
      .value("row_major", TileLayout::row_major, "Row-major layout")
      .value("col_major", TileLayout::col_major, "Column-major layout")
      .export_values();

  // TileView - immutable struct for tile view information.
  //
  // Fields are read-only from Python so that hash and equality remain stable
  // across an instance's lifetime (Python's hash contract). Construct via the
  // constructor, not by mutating fields.
  nb::class_<TileView>(
      ir, "TileView",
      "Tile view representation with valid shape, stride, start offset, layouts, fractal, and pad. "
      "Immutable from Python — set all fields at construction time.")
      .def(nb::init<const std::vector<ExprPtr>&, const std::vector<ExprPtr>&, ExprPtr, TileLayout, TileLayout,
                    uint64_t, PadValue>(),
           nb::arg("valid_shape") = std::vector<ExprPtr>{}, nb::arg("stride") = std::vector<ExprPtr>{},
           nb::arg("start_offset") = ExprPtr{}, nb::arg("blayout") = TileLayout::row_major,
           nb::arg("slayout") = TileLayout::none_box, nb::arg("fractal") = static_cast<uint64_t>(512),
           nb::arg("pad") = PadValue::null,
           "Create a tile view; all fields default to empty/null/row_major/none_box/512/null")
      .def(nb::init<const std::vector<int64_t>&, const std::vector<int64_t>&, ExprPtr, TileLayout, TileLayout,
                    uint64_t, PadValue>(),
           nb::arg("valid_shape"), nb::arg("stride"), nb::arg("start_offset"),
           nb::arg("blayout") = TileLayout::row_major, nb::arg("slayout") = TileLayout::none_box,
           nb::arg("fractal") = static_cast<uint64_t>(512), nb::arg("pad") = PadValue::null,
           "Create a tile view with integer valid_shape and stride, auto-converted to ConstInt")
      .def_ro("valid_shape", &TileView::valid_shape, "Valid shape dimensions")
      .def_ro("stride", &TileView::stride, "Stride for each dimension")
      .def_ro("start_offset", &TileView::start_offset, "Starting offset")
      .def_ro("blayout", &TileView::blayout, "Block layout")
      .def_ro("slayout", &TileView::slayout, "Scatter layout")
      .def_ro("fractal", &TileView::fractal, "Fractal size")
      .def_ro("pad", &TileView::pad, "Pad mode")
      .def(
          "__eq__", [](const TileView& self, const TileView& other) { return self == other; },
          nb::arg("other"), "Structural equality comparison")
      .def(
          "__ne__", [](const TileView& self, const TileView& other) { return self != other; },
          nb::arg("other"), "Structural inequality comparison")
      .def(
          "__hash__", [](const TileView& self) { return Hash(self); },
          "Hash consistent with __eq__ (ConstInt by value, other ExprPtrs by pointer identity)");

  // Dynamic dimension constant
  ir.attr("DYNAMIC_DIM") = kDynamicDim;

  // OpRegistry
  ir.def(
      "create_op_call",
      [](const std::string& op_name, const std::vector<ExprPtr>& args, const Span& span) {
        return OpRegistry::GetInstance().Create(op_name, args, span);
      },
      nb::arg("op_name"), nb::arg("args"), nb::arg("span"),
      "Create a Call expression (backward compatibility)");

  ir.def(
      "create_op_call",
      [](const std::string& op_name, const std::vector<ExprPtr>& args, const nb::dict& kwargs_dict,
         const Span& span) {
        // Convert Python dict to C++ vector<pair<string, any>> to preserve order
        auto kwargs = ConvertKwargsDict(kwargs_dict);
        return OpRegistry::GetInstance().Create(op_name, args, kwargs, span);
      },
      nb::arg("op_name"), nb::arg("args"), nb::arg("kwargs"), nb::arg("span"),
      "Create a Call expression with args and kwargs");

  ir.def(
      "is_op_registered",
      [](const std::string& op_name) { return OpRegistry::GetInstance().IsRegistered(op_name); },
      nb::arg("op_name"), "Check if an operator is registered");

  ir.def(
      "get_op", [](const std::string& op_name) { return OpRegistry::GetInstance().GetOp(op_name); },
      nb::arg("op_name"), "Get an operator instance by name");

  ir.def(
      "get_op_memory_spec",
      [](const std::string& op_name) -> nb::object {
        auto& registry = OpRegistry::GetInstance();
        if (!registry.IsRegistered(op_name)) return nb::none();
        const auto& entry = registry.GetEntry(op_name);
        const auto& spec = entry.GetMemorySpec();
        if (!spec.has_value()) return nb::none();
        // Empty spec (from no_memory_spec()) — no constraints and no resolver
        if (spec->input_constraints.empty() && !spec->deduce_output_memory) return nb::none();

        nb::dict result;
        // Input constraints
        nb::list inputs;
        for (const auto& c : spec->input_constraints) {
          nb::list allowed;
          for (auto ms : c) allowed.append(ms);
          inputs.append(nb::cast(allowed));
        }
        result["input_constraints"] = inputs;
        // Output (resolve with empty kwargs for display). Distinguishes:
        //   - Fixed/default-seeded: resolver returns a concrete MemorySpace.
        //   - Inherit-input (slice/reshape/...): marker string "inherit_from_input".
        //   - Retargetable with no kwarg default (tile.load/tile.create without
        //     target_memory): deferred — InferTileMemorySpace resolves from
        //     consumer demand. Reported as the string "deferred".
        if (entry.OutputMemoryInheritsInput()) {
          result["output_memory"] = "inherit_from_input";
        } else if (spec->deduce_output_memory) {
          auto out = spec->deduce_output_memory({});
          if (out.has_value()) {
            result["output_memory"] = *out;
          } else {
            result["output_memory"] = "deferred";
          }
        } else {
          result["output_memory"] = nb::none();
        }
        return result;
      },
      nb::arg("op_name"), "Get memory space specification for a registered operator");

  // Var - const shared_ptr
  auto var_class = nb::class_<Var, Expr>(ir, "Var", "Variable reference expression");

  var_class.def(
      nb::init<const std::string&, const TypePtr&, const Span&>(), nb::arg("name_hint"), nb::arg("type"),
      nb::arg("span"),
      "Create a variable reference (memory reference is stored in ShapedType for Tensor/Tile types)");
  BindFields<Var>(var_class);

  // IterArg - const shared_ptr
  auto iterarg_class = nb::class_<IterArg, Var>(ir, "IterArg", "Iteration argument variable");
  iterarg_class.def(nb::init<const std::string&, const TypePtr&, const ExprPtr&, const Span&>(),
                    nb::arg("name_hint"), nb::arg("type"), nb::arg("initValue"), nb::arg("span"),
                    "Create an iteration argument with initial value");
  BindFields<IterArg>(iterarg_class);

  // MemRef - now inherits from Var (first-class expression)
  auto memref_class =
      nb::class_<MemRef, Var>(ir, "MemRef", "Memory reference variable for shaped types (inherits from Var)");
  memref_class
      .def(
          "__init__",
          [](MemRef* self, const VarPtr& base, int64_t byte_offset, uint64_t size, const Span& span) {
            new (self) MemRef(base, byte_offset, size, span);
          },
          nb::arg("base"), nb::arg("byte_offset"), nb::arg("size"), nb::arg("span") = Span::unknown(),
          "Create a memory reference with base Ptr, integer byte_offset, and size")
      .def(nb::init<VarPtr, ExprPtr, uint64_t, Span>(), nb::arg("base"), nb::arg("byte_offset"),
           nb::arg("size"), nb::arg("span") = Span::unknown(),
           "Create a memory reference with base Ptr, byte_offset expression, and size")
      // String base constructor: MemRef("base_name", byte_offset, size) — for forward references
      // in printed IR where the base Ptr variable appears in annotations before its alloc statement
      .def(
          "__init__",
          [](MemRef* self, const std::string& base_name, int64_t byte_offset, uint64_t size,
             const Span& span) {
            auto base = std::make_shared<Var>(base_name, GetPtrType(), Span::unknown());
            new (self) MemRef(base, byte_offset, size, span);
          },
          nb::arg("base"), nb::arg("byte_offset"), nb::arg("size"), nb::arg("span") = Span::unknown(),
          "Create a memory reference from base name string, integer byte_offset, and size")
      // Legacy constructor: MemRef(MemorySpace, addr_expr, size, id) → auto-creates base Ptr
      .def(
          "__init__",
          [](MemRef* self, MemorySpace memory_space, const ExprPtr& addr_expr, uint64_t size, uint64_t id,
             const Span& span) {
            // Build a name like "mem_vec_0" from memory_space and id
            std::string space_str = MemorySpaceToString(memory_space);
            std::transform(space_str.begin(), space_str.end(), space_str.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            std::string base_name = "mem_" + space_str + "_" + std::to_string(id);
            auto base = std::make_shared<Var>(base_name, GetPtrType(), Span::unknown());
            // Use addr_expr as byte_offset for backward compatibility
            new (self) MemRef(base, addr_expr, size, span);
          },
          nb::arg("memory_space"), nb::arg("addr"), nb::arg("size"), nb::arg("id"),
          nb::arg("span") = Span::unknown(),
          "Legacy constructor: create MemRef from memory_space, addr, size, id")
      // Legacy constructor: MemRef(addr_int, size, id) → auto-creates base Ptr
      .def(
          "__init__",
          [](MemRef* self, int64_t addr, uint64_t size, uint64_t id, const Span& span) {
            std::string base_name = "mem_" + std::to_string(id);
            auto base = std::make_shared<Var>(base_name, GetPtrType(), Span::unknown());
            auto addr_expr = std::make_shared<ConstInt>(addr, DataType::INDEX, Span::unknown());
            new (self) MemRef(base, addr_expr, size, span);
          },
          nb::arg("addr"), nb::arg("size"), nb::arg("id"), nb::arg("span") = Span::unknown(),
          "Legacy constructor: create MemRef from integer addr, size, id")
      .def_rw("base_", &MemRef::base_, "Base Ptr variable (allocation identity)")
      .def_rw("byte_offset_", &MemRef::byte_offset_, "Byte offset from base")
      .def_rw("size_", &MemRef::size_, "Size in bytes (64-bit unsigned)")
      .def_static("same_allocation", &MemRef::SameAllocation, nb::arg("a"), nb::arg("b"),
                  "Check if two MemRefs share the same allocation (same base_ Ptr)")
      .def_static("may_alias", &MemRef::MayAlias, nb::arg("a"), nb::arg("b"),
                  "Check if two MemRefs may alias (same base + overlapping byte ranges)");

  // ConstInt - const shared_ptr
  auto constint_class = nb::class_<ConstInt, Expr>(ir, "ConstInt", "Constant integer expression");
  constint_class.def(nb::init<int64_t, DataType, const Span&>(), nb::arg("value"), nb::arg("dtype"),
                     nb::arg("span"), "Create a constant integer expression");
  BindFields<ConstInt>(constint_class);
  constint_class.def_prop_ro("dtype", &ConstInt::dtype, "Data type of the expression");

  // ConstFloat - const shared_ptr
  auto constfloat_class = nb::class_<ConstFloat, Expr>(ir, "ConstFloat", "Constant float expression");
  constfloat_class.def(nb::init<double, DataType, const Span&>(), nb::arg("value"), nb::arg("dtype"),
                       nb::arg("span"), "Create a constant float expression");
  BindFields<ConstFloat>(constfloat_class);
  constfloat_class.def_prop_ro("dtype", &ConstFloat::dtype, "Data type of the expression");

  // ConstBool - const shared_ptr
  auto constbool_class = nb::class_<ConstBool, Expr>(ir, "ConstBool", "Constant boolean expression");
  constbool_class.def(nb::init<bool, const Span&>(), nb::arg("value"), nb::arg("span"),
                      "Create a constant boolean expression");
  BindFields<ConstBool>(constbool_class);
  constbool_class.def_prop_ro("dtype", &ConstBool::dtype, "Data type of the expression (always BOOL)");

  // Call - const shared_ptr
  auto call_class = nb::class_<Call, Expr>(ir, "Call", "Function call expression");

  // Constructors without kwargs (backward compatibility)
  call_class.def(nb::init<const OpPtr&, const std::vector<ExprPtr>&, const Span&>(), nb::arg("op"),
                 nb::arg("args"), nb::arg("span"), "Create a function call expression");
  call_class.def(nb::init<const OpPtr&, const std::vector<ExprPtr>&, const TypePtr&, const Span&>(),
                 nb::arg("op"), nb::arg("args"), nb::arg("type"), nb::arg("span"),
                 "Create a function call expression with explicit type");

  // Constructors with kwargs (using nb::dict) - use factory functions
  call_class.def(
      "__init__",
      [](Call* self, const OpPtr& op, const std::vector<ExprPtr>& args, const nb::dict& kwargs_dict,
         const Span& span) {
        auto kwargs = ConvertKwargsDict(kwargs_dict);
        new (self) Call(op, args, kwargs, span);
      },
      nb::arg("op"), nb::arg("args"), nb::arg("kwargs"), nb::arg("span"),
      "Create a function call expression with kwargs");

  call_class.def(
      "__init__",
      [](Call* self, const OpPtr& op, const std::vector<ExprPtr>& args, const nb::dict& kwargs_dict,
         const TypePtr& type, const Span& span) {
        auto kwargs = ConvertKwargsDict(kwargs_dict);
        new (self) Call(op, args, kwargs, type, span);
      },
      nb::arg("op"), nb::arg("args"), nb::arg("kwargs"), nb::arg("type"), nb::arg("span"),
      "Create a function call expression with kwargs and explicit type");

  // Constructor with kwargs and explicit attrs (e.g. {"arg_directions": [...]}) and type
  call_class.def(
      "__init__",
      [](Call* self, const OpPtr& op, const std::vector<ExprPtr>& args, const nb::dict& kwargs_dict,
         const nb::object& attrs_or_none, const TypePtr& type, const Span& span) {
        auto kwargs = ConvertKwargsDict(kwargs_dict);
        auto attrs = ConvertAttrsFromPython(attrs_or_none);
        new (self) Call(op, args, std::move(kwargs), std::move(attrs), type, span);
      },
      nb::arg("op"), nb::arg("args"), nb::arg("kwargs"), nb::arg("attrs").none(), nb::arg("type"),
      nb::arg("span"),
      "Create a function call expression with kwargs and explicit attrs map and type. "
      "Reserved attrs keys: 'arg_directions' -> list[ArgDirection].");

  BindFields<Call>(call_class);

  // Convert a vector<pair<string, any>> kwargs/attrs container to a Python dict.
  auto kwargs_to_pydict = [](const std::vector<std::pair<std::string, std::any>>& items) {
    nb::dict result;
    for (const auto& [key, value] : items) {
      if (value.type() == typeid(int)) {
        result[key.c_str()] = AnyCast<int>(value, "converting to Python: " + key);
      } else if (value.type() == typeid(bool)) {
        result[key.c_str()] = AnyCast<bool>(value, "converting to Python: " + key);
      } else if (value.type() == typeid(std::string)) {
        result[key.c_str()] = AnyCast<std::string>(value, "converting to Python: " + key);
      } else if (value.type() == typeid(double)) {
        result[key.c_str()] = AnyCast<double>(value, "converting to Python: " + key);
      } else if (value.type() == typeid(float)) {
        result[key.c_str()] = AnyCast<float>(value, "converting to Python: " + key);
      } else if (value.type() == typeid(DataType)) {
        result[key.c_str()] = AnyCast<DataType>(value, "converting to Python: " + key);
      } else if (value.type() == typeid(MemorySpace)) {
        result[key.c_str()] = AnyCast<MemorySpace>(value, "converting to Python: " + key);
      } else if (value.type() == typeid(TensorLayout)) {
        result[key.c_str()] = AnyCast<TensorLayout>(value, "converting to Python: " + key);
      } else if (value.type() == typeid(TileLayout)) {
        result[key.c_str()] = AnyCast<TileLayout>(value, "converting to Python: " + key);
      } else if (value.type() == typeid(PadValue)) {
        result[key.c_str()] = AnyCast<PadValue>(value, "converting to Python: " + key);
      } else if (value.type() == typeid(LoopOrigin)) {
        result[key.c_str()] = AnyCast<LoopOrigin>(value, "converting to Python: " + key);
      } else if (value.type() == typeid(std::vector<ArgDirection>)) {
        const auto& dirs = AnyCast<std::vector<ArgDirection>>(value, "converting to Python: " + key);
        nb::list lst;
        for (auto d : dirs) {
          lst.append(nb::cast(d));
        }
        result[key.c_str()] = lst;
      } else if (value.type() == typeid(std::vector<int32_t>)) {
        // Used by attrs["arg_direction_overrides"] (per-arg NoDep override
        // indices). Surface as a Python list[int] so attribute readback
        // round-trips through the binding.
        const auto& idxs = AnyCast<std::vector<int32_t>>(value, "converting to Python: " + key);
        nb::list lst;
        for (auto i : idxs) {
          lst.append(nb::cast(i));
        }
        result[key.c_str()] = lst;
      } else if (value.type() == typeid(std::vector<VarPtr>)) {
        // Used by attrs["manual_dep_edges"] / attrs["user_manual_dep_edges"].
        const auto& vars = AnyCast<std::vector<VarPtr>>(value, "converting to Python: " + key);
        nb::list lst;
        for (const auto& v : vars) {
          lst.append(nb::cast(v));
        }
        result[key.c_str()] = lst;
      }
    }
    return result;
  };

  call_class.def_prop_ro(
      "kwargs", [kwargs_to_pydict](const CallPtr& self) { return kwargs_to_pydict(self->kwargs_); },
      "Keyword arguments (metadata) for this call");

  call_class.def_prop_ro(
      "attrs", [kwargs_to_pydict](const CallPtr& self) { return kwargs_to_pydict(self->attrs_); },
      "Compiler-internal node metadata. Reserved keys: 'arg_directions' -> list[ArgDirection].");

  call_class.def_prop_ro(
      "arg_directions",
      [](const CallPtr& self) {
        nb::list result;
        for (auto d : self->GetArgDirections()) {
          result.append(nb::cast(d));
        }
        return result;
      },
      "Resolved per-argument call-site directions (empty list when not yet derived). "
      "Stored under attrs['arg_directions'].");

  // MakeTuple - const shared_ptr
  auto make_tuple_class = nb::class_<MakeTuple, Expr>(ir, "MakeTuple", "Tuple construction expression");
  make_tuple_class.def(nb::init<const std::vector<ExprPtr>&, const Span&>(), nb::arg("elements"),
                       nb::arg("span"), "Create a tuple construction expression");
  BindFields<MakeTuple>(make_tuple_class);

  // TupleGetItemExpr - const shared_ptr
  auto tuple_get_item_class =
      nb::class_<TupleGetItemExpr, Expr>(ir, "TupleGetItemExpr", "Tuple element access expression");
  tuple_get_item_class.def(nb::init<const ExprPtr&, int, const Span&>(), nb::arg("tuple"), nb::arg("index"),
                           nb::arg("span"), "Create a tuple element access expression");
  BindFields<TupleGetItemExpr>(tuple_get_item_class);

  // BinaryExpr - abstract, const shared_ptr
  auto binaryexpr_class = nb::class_<BinaryExpr, Expr>(ir, "BinaryExpr", "Base class for binary operations");
  BindFields<BinaryExpr>(binaryexpr_class);

  // UnaryExpr - abstract, const shared_ptr
  auto unaryexpr_class = nb::class_<UnaryExpr, Expr>(ir, "UnaryExpr", "Base class for unary operations");
  BindFields<UnaryExpr>(unaryexpr_class);

// Macro to bind binary expression nodes
#define BIND_BINARY_EXPR(OpName, Description)                                                  \
  nb::class_<OpName, BinaryExpr>(ir, #OpName, Description)                                     \
      .def(nb::init<const ExprPtr&, const ExprPtr&, DataType, const Span&>(), nb::arg("left"), \
           nb::arg("right"), nb::arg("dtype"), nb::arg("span"), "Create " Description);

  // Bind all binary expression nodes
  BIND_BINARY_EXPR(Add, "Addition expression (left + right)")
  BIND_BINARY_EXPR(Sub, "Subtraction expression (left - right)")
  BIND_BINARY_EXPR(Mul, "Multiplication expression (left * right)")
  BIND_BINARY_EXPR(FloorDiv, "Floor division expression (left // right)")
  BIND_BINARY_EXPR(FloorMod, "Floor modulo expression (left % right)")
  BIND_BINARY_EXPR(FloatDiv, "Float division expression (left / right)")
  BIND_BINARY_EXPR(Min, "Minimum expression (min(left, right))")
  BIND_BINARY_EXPR(Max, "Maximum expression (max(left, right))")
  BIND_BINARY_EXPR(Pow, "Power expression (left ** right)")
  BIND_BINARY_EXPR(Eq, "Equality expression (left == right)")
  BIND_BINARY_EXPR(Ne, "Inequality expression (left != right)")
  BIND_BINARY_EXPR(Lt, "Less than expression (left < right)")
  BIND_BINARY_EXPR(Le, "Less than or equal to expression (left <= right)")
  BIND_BINARY_EXPR(Gt, "Greater than expression (left > right)")
  BIND_BINARY_EXPR(Ge, "Greater than or equal to expression (left >= right)")
  BIND_BINARY_EXPR(And, "Logical and expression (left and right)")
  BIND_BINARY_EXPR(Or, "Logical or expression (left or right)")
  BIND_BINARY_EXPR(Xor, "Logical xor expression (left xor right)")
  BIND_BINARY_EXPR(BitAnd, "Bitwise and expression (left & right)")
  BIND_BINARY_EXPR(BitOr, "Bitwise or expression (left | right)")
  BIND_BINARY_EXPR(BitXor, "Bitwise xor expression (left ^ right)")
  BIND_BINARY_EXPR(BitShiftLeft, "Bitwise left shift expression (left << right)")
  BIND_BINARY_EXPR(BitShiftRight, "Bitwise right shift expression (left >> right)")

#undef BIND_BINARY_EXPR

// Macro to bind unary expression nodes
#define BIND_UNARY_EXPR(OpName, Description)                                                        \
  nb::class_<OpName, UnaryExpr>(ir, #OpName, Description)                                           \
      .def(nb::init<const ExprPtr&, DataType, const Span&>(), nb::arg("operand"), nb::arg("dtype"), \
           nb::arg("span"), "Create " Description);

  // Bind all unary expression nodes
  BIND_UNARY_EXPR(Abs, "Absolute value expression (abs(operand))")
  BIND_UNARY_EXPR(Neg, "Negation expression (-operand)")
  BIND_UNARY_EXPR(Not, "Logical not expression (not operand)")
  BIND_UNARY_EXPR(BitNot, "Bitwise not expression (~operand)")
  BIND_UNARY_EXPR(Cast, "Cast expression (cast operand to dtype)")

#undef BIND_UNARY_EXPR

  // Bind structural hash and equality functions
  // structural_hash overloads share the same auto-mapping semantics:
  //   enable_auto_mapping=True  -> variable names are ignored (x+1 and y+1 hash the same)
  //   enable_auto_mapping=False -> variable identity is preserved deterministically
  ir.def("structural_hash", static_cast<uint64_t (*)(const IRNodePtr&, bool)>(&structural_hash),
         nb::arg("node"), nb::arg("enable_auto_mapping") = false,
         "Compute deterministic structural hash of an IR node (ignores Span). "
         "If enable_auto_mapping=True, variable names are ignored (e.g., x+1 and y+1 hash the same). "
         "If enable_auto_mapping=False (default), different variable objects produce different hashes.");
  ir.def("structural_hash", static_cast<uint64_t (*)(const TypePtr&, bool)>(&structural_hash),
         nb::arg("type"), nb::arg("enable_auto_mapping") = false,
         "Compute deterministic structural hash of a type. "
         "enable_auto_mapping only affects variables embedded in the type (e.g., shape expressions).");

  ir.def("structural_equal",
         static_cast<bool (*)(const IRNodePtr&, const IRNodePtr&, bool)>(&structural_equal), nb::arg("lhs"),
         nb::arg("rhs"), nb::arg("enable_auto_mapping") = false,
         "Check if two IR nodes are structurally equal. "
         "Ignores source location (Span). Returns True if IR nodes have identical structure. "
         "If enable_auto_mapping=True, automatically map variables (e.g., x+1 equals y+1). "
         "If enable_auto_mapping=False (default), variable objects must be exactly the same (not just same "
         "name).");
  ir.def("structural_equal", static_cast<bool (*)(const TypePtr&, const TypePtr&, bool)>(&structural_equal),
         nb::arg("lhs"), nb::arg("rhs"), nb::arg("enable_auto_mapping") = false,
         "Check if two types are structurally equal. "
         "Ignores source location (Span). Returns True if types have identical structure. "
         "If enable_auto_mapping=True, automatically map variables (e.g., x+1 equals y+1). "
         "If enable_auto_mapping=False (default), variable objects must be exactly the same (not just same "
         "name).");

  ir.def("assert_structural_equal",
         static_cast<void (*)(const IRNodePtr&, const IRNodePtr&, bool)>(&assert_structural_equal),
         nb::arg("lhs"), nb::arg("rhs"), nb::arg("enable_auto_mapping") = false,
         "Assert two IR nodes are structurally equal. "
         "Raises ValueError with detailed error message showing the first mismatch location if they differ. "
         "Ignores source location (Span). "
         "If enable_auto_mapping=True, automatically map variables (e.g., x+1 equals y+1). "
         "If enable_auto_mapping=False (default), variable objects must be exactly the same (not just same "
         "name).");
  ir.def("assert_structural_equal",
         static_cast<void (*)(const TypePtr&, const TypePtr&, bool)>(&assert_structural_equal),
         nb::arg("lhs"), nb::arg("rhs"), nb::arg("enable_auto_mapping") = false,
         "Assert two types are structurally equal. "
         "Raises ValueError with detailed error message showing the first mismatch location if they differ. "
         "Ignores source location (Span). "
         "If enable_auto_mapping=True, automatically map variables (e.g., x+1 equals y+1). "
         "If enable_auto_mapping=False (default), variable objects must be exactly the same (not just same "
         "name).");

  // Serialization functions
  ir.def(
      "serialize",
      [](const IRNodePtr& node) {
        auto data = serialization::Serialize(node);
        return nb::bytes(reinterpret_cast<const char*>(data.data()), data.size());
      },
      nb::arg("node"), "Serialize an IR node to MessagePack bytes");

  ir.def(
      "deserialize",
      [](const nb::bytes& data) {
        std::vector<uint8_t> vec(static_cast<const uint8_t*>(data.data()),
                                 static_cast<const uint8_t*>(data.data()) + data.size());
        return serialization::Deserialize(vec);
      },
      nb::arg("data"), "Deserialize an IR node from MessagePack bytes");

  ir.def("serialize_to_file", &serialization::SerializeToFile, nb::arg("node"), nb::arg("path"),
         "Serialize an IR node to a file");

  ir.def("deserialize_from_file", &serialization::DeserializeFromFile, nb::arg("path"),
         "Deserialize an IR node from a file");

  // ========== Statements ==========

  // Stmt - abstract base, const shared_ptr
  auto stmt_class = nb::class_<Stmt, IRNode>(ir, "Stmt", "Base class for all statements");
  BindFields<Stmt>(stmt_class);
  // Manually expose leading_comments as read-only (intentionally outside
  // GetFieldDescriptors — see stmt.h).
  stmt_class.def_ro("leading_comments", &Stmt::leading_comments_,
                    "Source-level comments printed above this statement (read-only; "
                    "use ir.attach_leading_comments to modify).");

  // Free function: attach leading comments to an existing statement.
  // Python-side `Stmt.leading_comments` is read-only; this helper is the one
  // sanctioned mutation channel (e.g., for the Python parser after building a
  // stmt). See stmt.h:AttachLeadingComments for the const-safety rationale.
  ir.def("attach_leading_comments", &AttachLeadingComments, nb::arg("stmt"), nb::arg("comments"),
         "Attach leading comments to an existing statement (mutates IgnoreField metadata only). "
         "Returns the same statement for convenient chaining.");

  // AssignStmt - const shared_ptr
  auto assign_stmt_class =
      nb::class_<AssignStmt, Stmt>(ir, "AssignStmt", "Assignment statement: var = value");
  assign_stmt_class.def(nb::init<const VarPtr&, const ExprPtr&, const Span&>(), nb::arg("var"),
                        nb::arg("value"), nb::arg("span"), "Create an assignment statement");
  BindFields<AssignStmt>(assign_stmt_class);

  // IfStmt - const shared_ptr
  auto if_stmt_class = nb::class_<IfStmt, Stmt>(
      ir, "IfStmt", "Conditional statement: if condition then then_body else else_body");
  if_stmt_class.def(nb::init<const ExprPtr&, const StmtPtr&, const std::optional<StmtPtr>&,
                             const std::vector<VarPtr>&, const Span&>(),
                    nb::arg("condition"), nb::arg("then_body"), nb::arg("else_body") = nb::none(),
                    nb::arg("return_vars"), nb::arg("span"),
                    "Create a conditional statement with then and else branches (else_body can be None)");
  BindFields<IfStmt>(if_stmt_class);

  // YieldStmt - const shared_ptr
  auto yield_stmt_class = nb::class_<YieldStmt, Stmt>(ir, "YieldStmt", "Yield statement: yield value");
  yield_stmt_class.def(nb::init<const std::vector<ExprPtr>&, const Span&>(), nb::arg("value"),
                       nb::arg("span"), "Create a yield statement with a list of expressions");
  yield_stmt_class.def(nb::init<const Span&>(), nb::arg("span"), "Create a yield statement without values");
  BindFields<YieldStmt>(yield_stmt_class);

  // ReturnStmt - const shared_ptr
  auto return_stmt_class = nb::class_<ReturnStmt, Stmt>(ir, "ReturnStmt", "Return statement: return value");
  return_stmt_class.def(nb::init<const std::vector<ExprPtr>&, const Span&>(), nb::arg("value"),
                        nb::arg("span"), "Create a return statement with a list of expressions");
  return_stmt_class.def(nb::init<const Span&>(), nb::arg("span"), "Create a return statement without values");
  BindFields<ReturnStmt>(return_stmt_class);

  // ForKind enum (must be before ForStmt which uses it)
  nb::enum_<ForKind>(ir, "ForKind", "For loop kind classification")
      .value("Sequential", ForKind::Sequential, "Standard sequential for loop (default)")
      .value("Parallel", ForKind::Parallel, "Parallel for loop")
      .value("Unroll", ForKind::Unroll, "Compile-time unrolled for loop")
      .value("Pipeline", ForKind::Pipeline,
             "Software-pipelined loop (pre-lowering user marker + transient post-lowering marker)")
      .export_values();

  // ChunkPolicy enum (must be before ForStmt which uses it)
  nb::enum_<ChunkPolicy>(ir, "ChunkPolicy", "Chunk policy for loop chunking")
      .value("LeadingFull", ChunkPolicy::LeadingFull, "Full chunks first, smaller remainder at end")
      .value("Guarded", ChunkPolicy::Guarded,
             "Single loop over ceil(N/C) chunks with per-iteration if-guard (default)")
      .export_values();

  // ChunkConfig struct (must be before ForStmt which uses it)
  nb::class_<ChunkConfig>(ir, "ChunkConfig", "Chunk configuration for parallel loop splitting")
      .def(nb::init<ExprPtr, ChunkPolicy>(), nb::arg("size"), nb::arg("policy") = ChunkPolicy::Guarded,
           "Create a chunk configuration")
      .def_ro("size", &ChunkConfig::size, "Chunk size expression")
      .def_ro("policy", &ChunkConfig::policy, "Chunk distribution policy");

  // LoopOrigin enum (must be before ForStmt which uses it)
  nb::enum_<LoopOrigin>(ir, "LoopOrigin", "Loop origin classification")
      .value("Original", LoopOrigin::Original, "Regular loop (default)")
      .value("ChunkOuter", LoopOrigin::ChunkOuter, "Outer loop from chunk splitting")
      .value("ChunkInner", LoopOrigin::ChunkInner, "Inner loop from chunk splitting")
      .value("ChunkRemainder", LoopOrigin::ChunkRemainder, "Remainder loop from chunk splitting")
      .export_values();

  // ForStmt - const shared_ptr
  auto for_stmt_class = nb::class_<ForStmt, Stmt>(
      ir, "ForStmt", "For loop statement: for loop_var in range(start, stop, step): body");
  for_stmt_class.def(
      "__init__",
      [](ForStmt* self, const VarPtr& loop_var, const ExprPtr& start, const ExprPtr& stop,
         const ExprPtr& step, const std::vector<IterArgPtr>& iter_args, const StmtPtr& body,
         const std::vector<VarPtr>& return_vars, const Span& span, ForKind kind,
         const std::optional<ExprPtr>& chunk_size, ChunkPolicy chunk_policy,
         const nb::object& attrs_or_none) {
        auto attrs = ConvertAttrsFromPython(attrs_or_none);
        std::optional<ChunkConfig> chunk_config =
            chunk_size.has_value() ? std::optional{ChunkConfig{*chunk_size, chunk_policy}} : std::nullopt;
        new (self) ForStmt(loop_var, start, stop, step, iter_args, body, return_vars, span, kind,
                           std::move(chunk_config), std::move(attrs));
      },
      nb::arg("loop_var"), nb::arg("start"), nb::arg("stop"), nb::arg("step"), nb::arg("iter_args"),
      nb::arg("body"), nb::arg("return_vars"), nb::arg("span"), nb::arg("kind") = ForKind::Sequential,
      nb::arg("chunk_size") = nb::none(), nb::arg("chunk_policy") = ChunkPolicy::Guarded,
      nb::arg("attrs") = nb::none(), "Create a for loop statement");
  BindFields<ForStmt>(for_stmt_class);
  // Custom attrs property: convert vector<pair<string, any>> to Python dict
  for_stmt_class.def_prop_ro(
      "attrs",
      [](const ForStmtPtr& self) {
        nb::dict result;
        for (const auto& [key, value] : self->attrs_) {
          result[key.c_str()] = AnyToPyObject<int, bool, std::string, double, float, DataType, MemorySpace,
                                              TensorLayout, TileLayout, PadValue, LoopOrigin>(value, key);
        }
        return result;
      },
      "Loop-level attributes as a dictionary");
  // Convenience properties for chunk_size and chunk_policy (derived from chunk_config)
  for_stmt_class.def_prop_ro(
      "chunk_size",
      [](const ForStmtPtr& self) -> std::optional<ExprPtr> {
        if (self->chunk_config_.has_value()) return self->chunk_config_->size;
        return std::nullopt;
      },
      "Chunk size expression (None if no chunking)");
  for_stmt_class.def_prop_ro(
      "chunk_policy",
      [](const ForStmtPtr& self) -> ChunkPolicy {
        if (self->chunk_config_.has_value()) return self->chunk_config_->policy;
        return ChunkPolicy::Guarded;
      },
      "Chunk distribution policy");

  // WhileStmt - const shared_ptr
  auto while_stmt_class =
      nb::class_<WhileStmt, Stmt>(ir, "WhileStmt", "While loop statement: while condition: body");
  while_stmt_class.def(nb::init<const ExprPtr&, const std::vector<IterArgPtr>&, const StmtPtr&,
                                const std::vector<VarPtr>&, const Span&>(),
                       nb::arg("condition"), nb::arg("iter_args"), nb::arg("body"), nb::arg("return_vars"),
                       nb::arg("span"), "Create a while loop statement");
  BindFields<WhileStmt>(while_stmt_class);

  // ScopeKind enum
  nb::enum_<ScopeKind>(ir, "ScopeKind", "Scope kind classification")
      .value("InCore", ScopeKind::InCore, "InCore scope for AICore sub-graphs")
      .value("AutoInCore", ScopeKind::AutoInCore, "AutoInCore scope for automatic chunking")
      .value("Cluster", ScopeKind::Cluster, "Cluster scope for co-scheduled AIC + AIV groups")
      .value("Hierarchy", ScopeKind::Hierarchy, "Distributed hierarchy scope (uses level/role)")
      .value("Spmd", ScopeKind::Spmd, "SPMD dispatch scope (core_num/sync_start)")
      .value("Runtime", ScopeKind::Runtime, "Runtime orchestration scope (PTO2_SCOPE wrapper)")
      .export_values();

  // SplitMode enum
  nb::enum_<SplitMode>(ir, "SplitMode", "Split mode for cross-core data transfer")
      .value("NONE", SplitMode::None, "No split")
      .value("UP_DOWN", SplitMode::UpDown, "Split vertically (height halved)")
      .value("LEFT_RIGHT", SplitMode::LeftRight, "Split horizontally (width halved)")
      .export_values();

  // ScopeStmt - abstract base class for all scope statements (issue #1047).
  auto scope_stmt_class = nb::class_<ScopeStmt, Stmt>(
      ir, "ScopeStmt", "Scope statement: marks a region with specific execution context (abstract base)");
  scope_stmt_class.def_prop_ro("scope_kind", &ScopeStmt::GetScopeKind, "Discriminator for the scope kind");
  BindFields<ScopeStmt>(scope_stmt_class);  // exposes name_hint, body

  // InCoreScopeStmt
  auto in_core_scope_stmt_class =
      nb::class_<InCoreScopeStmt, ScopeStmt>(ir, "InCoreScopeStmt", "InCore scope: AICore sub-graph region");
  in_core_scope_stmt_class.def(nb::init<std::optional<SplitMode>, std::string, const StmtPtr&, const Span&>(),
                               nb::arg("split") = nb::none(), nb::arg("name_hint") = "", nb::arg("body"),
                               nb::arg("span"), "Create an InCore scope statement");
  BindFields<InCoreScopeStmt>(in_core_scope_stmt_class);

  // AutoInCoreScopeStmt
  auto auto_in_core_scope_stmt_class = nb::class_<AutoInCoreScopeStmt, ScopeStmt>(
      ir, "AutoInCoreScopeStmt", "AutoInCore scope: InCore region with automatic chunking");
  auto_in_core_scope_stmt_class.def(
      nb::init<std::optional<SplitMode>, std::string, const StmtPtr&, const Span&>(),
      nb::arg("split") = nb::none(), nb::arg("name_hint") = "", nb::arg("body"), nb::arg("span"),
      "Create an AutoInCore scope statement");
  BindFields<AutoInCoreScopeStmt>(auto_in_core_scope_stmt_class);

  // ClusterScopeStmt
  auto cluster_scope_stmt_class = nb::class_<ClusterScopeStmt, ScopeStmt>(
      ir, "ClusterScopeStmt", "Cluster scope: co-scheduled AIC + AIV group");
  cluster_scope_stmt_class.def(nb::init<std::string, const StmtPtr&, const Span&>(),
                               nb::arg("name_hint") = "", nb::arg("body"), nb::arg("span"),
                               "Create a Cluster scope statement");
  BindFields<ClusterScopeStmt>(cluster_scope_stmt_class);

  // HierarchyScopeStmt
  auto hierarchy_scope_stmt_class = nb::class_<HierarchyScopeStmt, ScopeStmt>(
      ir, "HierarchyScopeStmt", "Hierarchy scope: distributed-hierarchy region");
  hierarchy_scope_stmt_class.def(
      nb::init<Level, std::optional<Role>, std::string, const StmtPtr&, const Span&>(), nb::arg("level"),
      nb::arg("role") = nb::none(), nb::arg("name_hint") = "", nb::arg("body"), nb::arg("span"),
      "Create a Hierarchy scope statement");
  BindFields<HierarchyScopeStmt>(hierarchy_scope_stmt_class);

  // SpmdScopeStmt
  auto spmd_scope_stmt_class =
      nb::class_<SpmdScopeStmt, ScopeStmt>(ir, "SpmdScopeStmt", "SPMD dispatch scope");
  spmd_scope_stmt_class.def(nb::init<ExprPtr, bool, std::string, const StmtPtr&, const Span&>(),
                            nb::arg("core_num"), nb::arg("sync_start") = false, nb::arg("name_hint") = "",
                            nb::arg("body"), nb::arg("span"), "Create an SPMD scope statement");
  // Convenience overload: accept a plain Python int for core_num and wrap it as
  // ConstInt(DataType::INDEX) automatically, mirroring the surface of pl.spmd()
  // and IRBuilder.scope() so callers can write ir.SpmdScopeStmt(core_num=4, ...).
  spmd_scope_stmt_class.def(
      "__init__",
      [](SpmdScopeStmt* self, int64_t core_num, bool sync_start, std::string name_hint, const StmtPtr& body,
         const Span& span) {
        auto core_num_expr = std::make_shared<const ConstInt>(core_num, DataType::INDEX, span);
        new (self) SpmdScopeStmt(core_num_expr, sync_start, std::move(name_hint), body, span);
      },
      nb::arg("core_num"), nb::arg("sync_start") = false, nb::arg("name_hint") = "", nb::arg("body"),
      nb::arg("span"), "Create an SPMD scope statement (int core_num is wrapped as ConstInt)");
  BindFields<SpmdScopeStmt>(spmd_scope_stmt_class);

  // RuntimeScopeStmt
  auto runtime_scope_stmt_class = nb::class_<RuntimeScopeStmt, ScopeStmt>(
      ir, "RuntimeScopeStmt",
      "Runtime orchestration scope: emits PTO2_SCOPE() (manual=False) or "
      "PTO2_SCOPE(PTO2ScopeMode::MANUAL) (manual=True) wrappers in codegen");
  runtime_scope_stmt_class.def(nb::init<bool, std::string, const StmtPtr&, const Span&>(),
                               nb::arg("manual") = false, nb::arg("name_hint") = "", nb::arg("body"),
                               nb::arg("span"), "Create a Runtime scope statement");
  BindFields<RuntimeScopeStmt>(runtime_scope_stmt_class);

  // SeqStmts - const shared_ptr
  auto seq_stmts_class =
      nb::class_<SeqStmts, Stmt>(ir, "SeqStmts", "Sequence of statements: a sequence of statements");
  seq_stmts_class.def(nb::init<const std::vector<StmtPtr>&, const Span&>(), nb::arg("stmts"), nb::arg("span"),
                      "Create a sequence of statements");
  seq_stmts_class.def(
      "__getitem__",
      [](const std::shared_ptr<const SeqStmts>& self, int index) -> StmtPtr {
        int size = static_cast<int>(self->stmts_.size());
        if (index < -size || index >= size) {
          throw pypto::IndexError("SeqStmts index " + std::to_string(index) + " out of range [" +
                                  std::to_string(-size) + ", " + std::to_string(size - 1) + "]");
        }
        if (index < 0) index += size;
        return self->stmts_[index];
      },
      nb::arg("index"), "Get statement by index, supports negative indexing");
  BindFields<SeqStmts>(seq_stmts_class);

  // EvalStmt - const shared_ptr
  auto eval_stmt_class = nb::class_<EvalStmt, Stmt>(ir, "EvalStmt", "Evaluation statement: expr");
  eval_stmt_class.def(nb::init<const ExprPtr&, const Span&>(), nb::arg("expr"), nb::arg("span"),
                      "Create an evaluation statement");
  BindFields<EvalStmt>(eval_stmt_class);

  // BreakStmt - const shared_ptr
  auto break_stmt_class = nb::class_<BreakStmt, Stmt>(ir, "BreakStmt", "Break statement: break");
  break_stmt_class.def(nb::init<const Span&>(), nb::arg("span"), "Create a break statement");
  BindFields<BreakStmt>(break_stmt_class);

  // ContinueStmt - const shared_ptr
  auto continue_stmt_class =
      nb::class_<ContinueStmt, Stmt>(ir, "ContinueStmt", "Continue statement: continue");
  continue_stmt_class.def(nb::init<const Span&>(), nb::arg("span"), "Create a continue statement");
  BindFields<ContinueStmt>(continue_stmt_class);

  // InlineLanguage enum — language tag for InlineStmt bodies.
  nb::enum_<InlineLanguage>(ir, "InlineLanguage", "Source language carried by an InlineStmt body")
      .value("Python", InlineLanguage::Python, "Python source")
      .export_values();

  // InlineStmt - const shared_ptr
  auto inline_stmt_class = nb::class_<InlineStmt, Stmt>(
      ir, "InlineStmt", "Inline statement: opaque source body in a target language");
  inline_stmt_class.def(nb::init<std::string, InlineLanguage, const Span&>(), nb::arg("body"),
                        nb::arg("language"), nb::arg("span"), "Create an inline statement");
  BindFields<InlineStmt>(inline_stmt_class);

  // FunctionType enum
  nb::enum_<FunctionType>(ir, "FunctionType", "Function type classification")
      .value("Opaque", FunctionType::Opaque, "Unspecified function type (default)")
      .value("Orchestration", FunctionType::Orchestration, "Host/AICPU control and coordination")
      .value("InCore", FunctionType::InCore, "AICore sub-graph execution (unspecialized)")
      .value("AIC", FunctionType::AIC, "Cube core kernel (specialized InCore)")
      .value("AIV", FunctionType::AIV, "Vector core kernel (specialized InCore)")
      .value("Group", FunctionType::Group, "Co-scheduled group of AIC + AIV kernels")
      .value("Spmd", FunctionType::Spmd, "SPMD data-parallel dispatch")
      .value("Inline", FunctionType::Inline, "Whole-body substitution at every call site")
      .export_values();

  // Level enum — hierarchy level in the Linqu machine model
  nb::enum_<Level>(ir, "Level", "Hierarchy level in the Linqu machine model")
      .value("AIV", Level::AIV, "Single AIV (Vector) core")
      .value("AIC", Level::AIC, "Single AIC (Cube) core")
      .value("CORE_GROUP", Level::CORE_GROUP, "Core-group (e.g. 1 AIC + 2 AIV)")
      .value("CHIP_DIE", Level::CHIP_DIE, "Chip die")
      .value("CHIP", Level::CHIP, "Chip (UMA)")
      .value("HOST", Level::HOST, "Host (single OS instance)")
      .value("CLUSTER_0", Level::CLUSTER_0, "Cluster-level-0 (pod)")
      .value("CLUSTER_1", Level::CLUSTER_1, "Cluster-level-1 (supernode)")
      .value("CLUSTER_2", Level::CLUSTER_2, "Cluster-level-2 (cross-rack)")
      .value("GLOBAL", Level::GLOBAL, "Global coordinator")
      // Readability aliases
      .value("L2CACHE", Level::L2CACHE, "Alias for CHIP_DIE")
      .value("PROCESSOR", Level::PROCESSOR, "Alias for CHIP")
      .value("UMA", Level::UMA, "Alias for CHIP")
      .value("NODE", Level::NODE, "Alias for HOST")
      .value("POD", Level::POD, "Alias for CLUSTER_0")
      .value("CLOS1", Level::CLOS1, "Alias for CLUSTER_1")
      .value("CLOS2", Level::CLOS2, "Alias for CLUSTER_2")
      .export_values();

  // Role enum — function role at L3-L7 hierarchy levels
  nb::enum_<Role>(ir, "Role", "Function role at L3-L7 hierarchy levels")
      .value("Orchestrator", Role::Orchestrator, "Builds DAG, submits tasks")
      .value("SubWorker", Role::SubWorker,
             "Executes compute/data tasks dispatched by the orchestrator at the same level")
      .export_values();

  // ParamDirection enum
  nb::enum_<ParamDirection>(ir, "ParamDirection", "Parameter direction classification")
      .value("In", ParamDirection::In, "Read-only input (default)")
      .value("Out", ParamDirection::Out, "Write-only output")
      .value("InOut", ParamDirection::InOut, "Read-write input/output")
      .export_values();

  // ArgDirection enum (call-site task-submission semantics)
  nb::enum_<ArgDirection>(ir, "ArgDirection", "Call-site argument direction (mirrors runtime TensorArgType)")
      .value("Input", ArgDirection::Input, "Read-only input (add_input / TensorArgType::INPUT)")
      .value("Output", ArgDirection::Output,
             "Runtime-allocated output buffer (add_output(create_info) / TensorArgType::OUTPUT)")
      .value("InOut", ArgDirection::InOut, "Read-then-write (add_inout / TensorArgType::INOUT)")
      .value("OutputExisting", ArgDirection::OutputExisting,
             "Write-only into an existing tensor "
             "(add_output(tensor) / TensorArgType::OUTPUT_EXISTING)")
      .value("NoDep", ArgDirection::NoDep,
             "No-dependency existing tensor (add_no_dep / TensorArgType::NO_DEP)")
      .value("Scalar", ArgDirection::Scalar, "Scalar (non-tensor) argument (add_scalar)");
  // Intentionally do NOT call .export_values(): ArgDirection::InOut would shadow
  // ParamDirection::InOut at module scope. Users access values via the enum class
  // (ir.ArgDirection.InOut) to keep the two direction models cleanly separated.

  // IsInCoreType helper
  ir.def("is_incore_type", &IsInCoreType, nb::arg("func_type"),
         "Check if a FunctionType is an InCore variant (InCore, AIC, or AIV)");

  // LevelToLinquLevel helper
  ir.def("level_to_linqu_level", &LevelToLinquLevel, nb::arg("level"),
         "Map Level enum value to Linqu hierarchy level number (0-7)");

  // Function - const shared_ptr
  auto function_class = nb::class_<Function, IRNode>(
      ir, "Function", "Function definition with name, parameters, return types, and body");
  function_class.def(
      "__init__",
      [](Function* self, const std::string& name, const nb::list& params,
         const std::vector<TypePtr>& return_types, const StmtPtr& body, const Span& span, FunctionType type,
         std::optional<Level> level, std::optional<Role> role, const nb::object& attrs_or_none) {
        std::vector<VarPtr> param_vars;
        std::vector<ParamDirection> param_dirs;
        param_vars.reserve(nb::len(params));
        param_dirs.reserve(nb::len(params));
        for (auto item : params) {
          // Accept either a Var (default In) or a tuple (Var, ParamDirection)
          if (nb::isinstance<nb::tuple>(item)) {
            auto tup = nb::cast<nb::tuple>(item);
            if (nb::len(tup) != 2) {
              throw pypto::TypeError("Each tuple in 'params' must be (Var, ParamDirection)");
            }
            param_vars.push_back(nb::cast<VarPtr>(tup[0]));
            param_dirs.push_back(nb::cast<ParamDirection>(tup[1]));
          } else {
            param_vars.push_back(nb::cast<VarPtr>(item));
            param_dirs.push_back(ParamDirection::In);
          }
        }
        auto attrs = ConvertAttrsFromPython(attrs_or_none);
        new (self) Function(name, std::move(param_vars), std::move(param_dirs), return_types, body, span,
                            type, level, role, std::move(attrs));
      },
      nb::arg("name"), nb::arg("params"), nb::arg("return_types"), nb::arg("body"), nb::arg("span"),
      nb::arg("type") = FunctionType::Opaque, nb::arg("level") = nb::none(), nb::arg("role") = nb::none(),
      nb::arg("attrs") = nb::none(), "Create a function definition");
  BindFields<Function>(function_class);
  // Custom attrs property: convert vector<pair<string, any>> to Python dict
  function_class.def_prop_ro(
      "attrs",
      [](const FunctionPtr& self) {
        nb::dict result;
        for (const auto& [key, value] : self->attrs_) {
          if (value.type() == typeid(int)) {
            result[key.c_str()] = AnyCast<int>(value, "converting to Python: " + key);
          } else if (value.type() == typeid(bool)) {
            result[key.c_str()] = AnyCast<bool>(value, "converting to Python: " + key);
          } else if (value.type() == typeid(std::string)) {
            result[key.c_str()] = AnyCast<std::string>(value, "converting to Python: " + key);
          } else if (value.type() == typeid(double)) {
            result[key.c_str()] = AnyCast<double>(value, "converting to Python: " + key);
          } else if (value.type() == typeid(float)) {
            result[key.c_str()] = AnyCast<float>(value, "converting to Python: " + key);
          } else if (value.type() == typeid(DataType)) {
            result[key.c_str()] = AnyCast<DataType>(value, "converting to Python: " + key);
          } else if (value.type() == typeid(MemorySpace)) {
            result[key.c_str()] = AnyCast<MemorySpace>(value, "converting to Python: " + key);
          } else if (value.type() == typeid(TensorLayout)) {
            result[key.c_str()] = AnyCast<TensorLayout>(value, "converting to Python: " + key);
          } else if (value.type() == typeid(TileLayout)) {
            result[key.c_str()] = AnyCast<TileLayout>(value, "converting to Python: " + key);
          } else if (value.type() == typeid(PadValue)) {
            result[key.c_str()] = AnyCast<PadValue>(value, "converting to Python: " + key);
          } else if (value.type() == typeid(ExprPtr)) {
            result[key.c_str()] = nb::cast(AnyCast<ExprPtr>(value, "converting to Python: " + key));
          }
        }
        return result;
      },
      "Function-level attributes as a dictionary");
  // Backward-compat split property: extract SplitMode from attrs
  function_class.def_prop_ro(
      "split",
      [](const FunctionPtr& self) -> nb::object {
        auto mode = self->GetSplitMode();
        if (!mode.has_value()) return nb::none();
        return nb::cast(*mode);
      },
      "Split mode for cross-core transfer (convenience accessor into attrs)");

  // Program - const shared_ptr
  auto program_class =
      nb::class_<Program, IRNode>(ir, "Program",
                                  "Program definition with functions mapped by GlobalVar references. "
                                  "Functions are automatically sorted by name for deterministic ordering.");
  program_class.def(nb::init<const std::vector<FunctionPtr>&, const std::string&, const Span&>(),
                    nb::arg("functions"), nb::arg("name"), nb::arg("span"),
                    "Create a program from a list of functions. "
                    "GlobalVar references are created automatically from function names.");
  program_class.def(
      nb::init<const std::vector<FunctionPtr>&, std::vector<CommGroupPtr>, const std::string&, const Span&>(),
      nb::arg("functions"), nb::arg("comm_groups"), nb::arg("name"), nb::arg("span"),
      "Create a program from a list of functions and CommGroup metadata. "
      "GlobalVar references are created automatically from function names.");
  program_class.def("get_function", &Program::GetFunction, nb::arg("name"),
                    "Get a function by name, returns None if not found");
  program_class.def("get_global_var", &Program::GetGlobalVar, nb::arg("name"),
                    "Get a GlobalVar by name, returns None if not found");
  program_class.def(
      "__getitem__",
      [](const std::shared_ptr<const Program>& self, const std::string& name) {
        return self->GetFunction(name);
      },
      nb::arg("name"), "Get function by name, returns None if not found");
  // Custom property for functions_ map that converts to Python dict
  program_class.def_prop_ro(
      "functions",
      [](const std::shared_ptr<const Program>& self) {
        nb::dict result;
        for (const auto& [gvar, func] : self->functions_) {
          result[nb::cast(gvar)] = nb::cast(func);
        }
        return result;
      },
      "Map of GlobalVar references to their corresponding functions, sorted by GlobalVar name");
  program_class.def_ro("name", &Program::name_, "Program name");
  program_class.def_ro("span", &Program::span_, "Source location");
  program_class.def_ro("comm_groups", &Program::comm_groups_,
                       "List of CommGroups declared on the program. CommGroups participate "
                       "in structural equality / hashing through reflection.");

  // CommGroup / WindowBuffer — IRNode-typed host-side metadata attached to a Program.
  auto window_buffer_class = nb::class_<WindowBuffer, IRNode>(
      ir, "WindowBuffer",
      "Per-rank allocation spec for a named CommGroup HCCL window buffer. "
      "Maps 1:1 to simpler.task_interface.ChipBufferSpec at submit-time. "
      "size_ is the per-rank element count; load_from_host_ / store_to_host_ "
      "are bool flags marking host-staging participation (the actual host "
      "tensor binding is recorded on the alloc op, not here).");
  window_buffer_class.def(nb::init<std::string, ExprPtr, DataType, bool, bool, Span>(), nb::arg("name"),
                          nb::arg("size"), nb::arg("dtype"), nb::arg("load_from_host") = false,
                          nb::arg("store_to_host") = false, nb::arg("span") = Span::unknown(),
                          "Create a WindowBuffer.");
  BindFields<WindowBuffer>(window_buffer_class);

  auto comm_group_class =
      nb::class_<CommGroup, IRNode>(ir, "CommGroup",
                                    "A communication group inferred from pld.alloc_window_buffer ops: a "
                                    "device-id list (empty = all devices) and a list of WindowBuffer slots "
                                    "shared by every rank in the group.");
  comm_group_class.def(nb::init<std::vector<int64_t>, std::vector<WindowBufferPtr>, Span>(),
                       nb::arg("devices"), nb::arg("slots"), nb::arg("span") = Span::unknown(),
                       "Create a CommGroup. ``devices`` empty = all devices.");
  BindFields<CommGroup>(comm_group_class);

  // Python-style printer function - unified API for IRNode
  ir.def(
      "python_print",
      [](const IRNodePtr& node, const std::string& prefix, bool concise, bool format) {
        return MaybeFormat(PythonPrint(node, prefix, concise), format);
      },
      nb::arg("node"), nb::arg("prefix") = "pl", nb::arg("concise") = false, nb::arg("format") = true,
      "Print IR node (Expr, Stmt, Function, or Program) in Python IR syntax.\n\n"
      "Args:\n"
      "    node: IR node to print\n"
      "    prefix: Module prefix (default 'pl' for 'import pypto.language as pl')\n"
      "    concise: If true, omit intermediate type annotations (default false)\n"
      "    format: If true, apply registered format callback (default true)");

  // Python-style printer function for Type objects - use separate name to avoid overload ambiguity
  ir.def(
      "python_print_type",
      [](const TypePtr& type, const std::string& prefix, bool format) {
        return MaybeFormat(PythonPrint(type, prefix), format);
      },
      nb::arg("type"), nb::arg("prefix") = "pl", nb::arg("format") = true,
      "Print Type object in Python IR syntax.\n\n"
      "Args:\n"
      "    type: Type to print\n"
      "    prefix: Module prefix (default 'pl' for 'import pypto.language as pl')\n"
      "    format: If true, apply registered format callback (default true)");

  // Register a Python callable to format printed IR output (e.g., ruff).
  // Pass None to unregister. The callback receives a code string and returns formatted code.
  ir.def(
      "register_format_callback",
      [](nb::object cb) {
        if (cb.is_none()) {
          RegisterFormatCallback(nullptr);
        } else {
          // Store as nb::object to prevent garbage collection of the Python callback
          nb::object stored_cb = nb::borrow(cb);
          RegisterFormatCallback([stored_cb](const std::string& code) -> std::string {
            try {
              nb::gil_scoped_acquire guard;
              return nb::cast<std::string>(stored_cb(code));
            } catch (...) {
              // Best-effort: return raw output on any failure
              return code;
            }
          });
        }
      },
      nb::arg("callback"),
      "Register a Python callable to post-process printed IR output.\n\n"
      "The callback receives a code string and returns the formatted code.\n"
      "Pass None to unregister and revert to raw output.\n\n"
      "Note: Must be called during module initialization (single-threaded).\n"
      "The GIL is acquired automatically when the callback is invoked from C++.");

  // operator functions for Var (wrapped in Python for span capture and normalization)
  // Using standalone C++ API functions from scalar_expr.h
  // Note: first parameter (self) is implicit when binding as method
  ir.def("add", &MakeAdd, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Addition operator");
  ir.def("sub", &MakeSub, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Subtraction operator");
  ir.def("mul", &MakeMul, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Multiplication operator");
  ir.def("truediv", &MakeFloatDiv, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "True division operator");
  ir.def("floordiv", &MakeFloorDiv, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Floor division operator");
  ir.def("mod", &MakeFloorMod, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Modulo operator");
  ir.def("pow", &MakePow, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Power operator");
  ir.def("eq", &MakeEq, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Equality operator");
  ir.def("ne", &MakeNe, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Inequality operator");
  ir.def("lt", &MakeLt, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Less than operator");
  ir.def("le", &MakeLe, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Less than or equal operator");
  ir.def("gt", &MakeGt, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Greater than operator");
  ir.def("ge", &MakeGe, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Greater than or equal operator");
  ir.def("neg", &MakeNeg, nb::arg("operand"), nb::arg("span") = Span::unknown(), "Negation operator");
  ir.def("cast", &MakeCast, nb::arg("operand"), nb::arg("dtype"), nb::arg("span") = Span::unknown(),
         "Cast operator");
  ir.def("bit_and", &MakeBitAnd, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Bitwise and operator");
  ir.def("bit_or", &MakeBitOr, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Bitwise or operator");
  ir.def("bit_xor", &MakeBitXor, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Bitwise xor operator");
  ir.def("bit_shift_left", &MakeBitShiftLeft, nb::arg("lhs"), nb::arg("rhs"),
         nb::arg("span") = Span::unknown(), "Bitwise left shift operator");
  ir.def("bit_shift_right", &MakeBitShiftRight, nb::arg("lhs"), nb::arg("rhs"),
         nb::arg("span") = Span::unknown(), "Bitwise right shift operator");
  ir.def("bit_not", &MakeBitNot, nb::arg("operand"), nb::arg("span") = Span::unknown(),
         "Bitwise not operator");
  ir.def("not_", &MakeNot, nb::arg("operand"), nb::arg("span") = Span::unknown(), "Logical not operator");
  ir.def("and_", &MakeAnd, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Logical and operator (lhs and rhs)");
  ir.def("or_", &MakeOr, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Logical or operator (lhs or rhs)");
  ir.def("min_", &MakeMin, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Minimum operator");
  ir.def("max_", &MakeMax, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Maximum operator");

  // ParentStmtAnalysis - utility class for analyzing statement parent relationships
  auto parent_stmt_analysis_class = nb::class_<ParentStmtAnalysis>(
      ir, "ParentStmtAnalysis",
      "Utility class for analyzing parent-child relationships in statement trees.\n\n"
      "This class builds a mapping from each statement to its parent statement within\n"
      "a function's body. It is useful for passes that need to traverse upward in the\n"
      "IR tree or understand the context of a statement.\n\n"
      "Example usage:\n"
      "    analysis = ir.ParentStmtAnalysis()\n"
      "    analysis.build_map(function)\n"
      "    parent = analysis.get_parent(some_stmt)\n"
      "    if parent:\n"
      "        # Use parent statement\n\n"
      "Note: The analysis becomes invalid after IR transformations. Call build_map again\n"
      "if the IR tree is modified.");

  parent_stmt_analysis_class.def(nb::init<>(), "Create a ParentStmtAnalysis instance");

  parent_stmt_analysis_class.def(
      "build_map", &ParentStmtAnalysis::BuildMap, nb::arg("func"),
      "Build the parent mapping from a function's body.\n\n"
      "Traverses the function's statement tree and records parent-child relationships.\n"
      "This method clears any existing mapping before building the new one.\n\n"
      "Args:\n"
      "    func: The function to analyze (can be None, resulting in empty map)\n\n"
      "Parent relationships established:\n"
      "- For SeqStmts: Each child statement's parent is the SeqStmts\n"
      "- For IfStmt: then_body and else_body (if present) have IfStmt as parent\n"
      "- For ForStmt: body has ForStmt as parent\n"
      "- Root statement (function.body) has no parent");

  parent_stmt_analysis_class.def("get_parent", &ParentStmtAnalysis::GetParent, nb::arg("stmt"),
                                 "Get the parent statement of a given statement.\n\n"
                                 "Args:\n"
                                 "    stmt: The statement to query\n\n"
                                 "Returns:\n"
                                 "    Parent statement, or None if:\n"
                                 "    - stmt is the root statement (function body)\n"
                                 "    - stmt is not found in the analyzed tree\n"
                                 "    - stmt is None");

  parent_stmt_analysis_class.def("has_parent", &ParentStmtAnalysis::HasParent, nb::arg("stmt"),
                                 "Check if a statement has a recorded parent.\n\n"
                                 "Args:\n"
                                 "    stmt: The statement to check\n\n"
                                 "Returns:\n"
                                 "    True if stmt has a parent in the map, False otherwise");

  parent_stmt_analysis_class.def("clear", &ParentStmtAnalysis::Clear,
                                 "Clear the parent mapping.\n\n"
                                 "Removes all recorded parent-child relationships. Useful for reusing\n"
                                 "the same ParentStmtAnalysis instance with different functions.");

  // Op conversion registry bindings
  ir.def(
      "register_op_conversion",
      [](const std::string& from_op, const std::string& to_op) {
        OpConversionRegistry::GetInstance().RegisterSimple(from_op, to_op);
      },
      nb::arg("from_op"), nb::arg("to_op"),
      "Register a simple tensor-to-tile op name mapping.\n\n"
      "Args:\n"
      "    from_op: Source op name (e.g., 'tensor.add')\n"
      "    to_op: Target op name (e.g., 'tile.add')");

  ir.def(
      "register_op_conversion_custom",
      [](const std::string& from_op, const nb::object& func) {
        // Capture Python callable in a C++ ConversionFunc
        nb::object py_func = nb::borrow(func);
        OpConversionRegistry::GetInstance().RegisterCustom(
            from_op,
            [py_func](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs,
                      const Span& span) -> ConversionResult {
              nb::gil_scoped_acquire guard;
              // Convert kwargs to Python list of (key, value) tuples
              nb::list py_kwargs_list;
              for (const auto& [key, val] : kwargs) {
                nb::object py_val = AnyToPyObject<DataType, MemorySpace, TensorLayout, PadValue, bool, int,
                                                  std::string, double>(val, key);
                nb::tuple pair = nb::make_tuple(nb::cast(key), py_val);
                py_kwargs_list.append(pair);
              }
              nb::object result = py_func(nb::cast(args), py_kwargs_list, nb::cast(span));
              // Result can be:
              // 1. An ExprPtr (simple conversion)
              // 2. A tuple of (list[StmtPtr], ExprPtr) (complex conversion)
              if (nb::isinstance<nb::tuple>(result)) {
                nb::tuple result_tuple = nb::cast<nb::tuple>(result);
                auto prologue = nb::cast<std::vector<StmtPtr>>(result_tuple[0]);
                auto expr = nb::cast<ExprPtr>(result_tuple[1]);
                return ConversionResult{std::move(prologue), std::move(expr)};
              }
              return ConversionResult{nb::cast<ExprPtr>(result)};
            });
      },
      nb::arg("from_op"), nb::arg("func"),
      "Register a custom conversion function for a tensor op.\n\n"
      "The function receives (args, kwargs, span) and should return either:\n"
      "- An Expr (simple conversion)\n"
      "- A tuple (list[Stmt], Expr) for complex conversions with prologue statements");

  ir.def(
      "has_op_conversion",
      [](const std::string& op_name) { return OpConversionRegistry::GetInstance().HasConversion(op_name); },
      nb::arg("op_name"), "Check if a conversion rule exists for an operator.");

  // ---------------------------------------------------------------------------
  // transform_utils bindings
  // ---------------------------------------------------------------------------

  ir.def(
      "flatten_to_stmts", [](const StmtPtr& stmt) { return transform_utils::FlattenToStmts(stmt); },
      nb::arg("stmt"),
      "Unwrap a statement into a flat list. Returns children of SeqStmts, or a single-element list.");

  ir.def(
      "collect_def_vars", [](const StmtPtr& stmt) { return transform_utils::CollectDefVars(stmt); },
      nb::arg("stmt"), "Collect all AssignStmt LHS variables (definition sites) from a statement tree.");

  ir.def(
      "find_yield_stmt",
      [](const StmtPtr& body) -> nb::object {
        auto result = transform_utils::FindYieldStmt(body);
        if (!result) return nb::none();
        return nb::cast(result);
      },
      nb::arg("body"), "Find the first YieldStmt inside a statement body (searches through SeqStmts).");

  ir.def(
      "get_last_yield_stmt",
      [](const StmtPtr& body) -> nb::object {
        auto result = transform_utils::GetLastYieldStmt(body);
        if (!result) return nb::none();
        return nb::cast(result);
      },
      nb::arg("body"), "Find the trailing YieldStmt in a statement body (checks only the last element).");

  ir.def(
      "substitute_expr",
      [](const ExprPtr& expr, const std::vector<std::pair<VarPtr, VarPtr>>& var_map_pairs) {
        std::unordered_map<const Var*, VarPtr> var_map;
        for (const auto& [orig, replacement] : var_map_pairs) {
          var_map[orig.get()] = replacement;
        }
        return transform_utils::Substitute(expr, var_map);
      },
      nb::arg("expr"), nb::arg("var_map"),
      "Substitute variables in an expression using a list of (original_var, replacement_var) pairs.");

  ir.def(
      "substitute_stmt",
      [](const StmtPtr& body, const std::vector<std::pair<VarPtr, VarPtr>>& var_map_pairs) {
        std::unordered_map<const Var*, VarPtr> var_map;
        for (const auto& [orig, replacement] : var_map_pairs) {
          var_map[orig.get()] = replacement;
        }
        return transform_utils::Substitute(body, var_map);
      },
      nb::arg("body"), nb::arg("var_map"),
      "Substitute variable references in a statement subtree using a list of (original_var, replacement_var) "
      "pairs.");

  ir.def(
      "deep_clone",
      [](const StmtPtr& body, const std::vector<std::pair<VarPtr, ExprPtr>>& var_map_pairs) -> nb::tuple {
        std::unordered_map<const Var*, ExprPtr> seed_map;
        for (const auto& [orig, replacement] : var_map_pairs) {
          seed_map[orig.get()] = replacement;
        }
        auto result = DeepClone(body, seed_map);
        // Convert raw-pointer-keyed map to shared_ptr-keyed map for Python
        std::vector<std::pair<VarPtr, VarPtr>> out_pairs;
        out_pairs.reserve(result.var_map.size());
        for (const auto& [raw_ptr, new_var] : result.var_map) {
          // Find the original VarPtr from the raw pointer — wrap as non-owning shared_ptr
          // Since Python holds the original IR tree alive, the raw pointer is valid
          out_pairs.emplace_back(std::shared_ptr<const Var>(std::shared_ptr<const Var>{}, raw_ptr), new_var);
        }
        return nb::make_tuple(result.cloned_body, out_pairs);
      },
      nb::arg("body"), nb::arg("var_map") = std::vector<std::pair<VarPtr, ExprPtr>>{},
      "Deep-clone a statement subtree, creating fresh Var objects at definition sites.\n\n"
      "var_map seeds the substitution: each (original_var, replacement_expr) pair\n"
      "replaces references to original_var with replacement_expr inside the clone.\n\n"
      "Returns a tuple of (cloned_body, def_var_map) where def_var_map is a list of\n"
      "(original_var, cloned_var) pairs for the definition sites freshly cloned by\n"
      "the traversal. Seeded entries from var_map are NOT included — use the\n"
      "caller's own substitution map for those.");

  // Cross-function call return type deduction
  ir.def("deduce_call_return_type", &DeduceCallReturnType, nb::arg("callee_params"), nb::arg("args"),
         nb::arg("return_types"),
         "Deduce return types for a cross-function call by substituting "
         "dynamic shape variables from callee params with actual arg shapes.");
}

}  // namespace python
}  // namespace pypto
