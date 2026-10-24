# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""DSL operator dispatch, argument conversion and buffer metadata."""

import ast
from typing import Any, TypeGuard, cast

from pypto.ir import op as ir_op
from pypto.language.distributed import op as _dsl_pld
from pypto.language.op import array_ops as _dsl_array
from pypto.language.op import prefetch_ops as _dsl_prefetch
from pypto.language.op import system_ops as _dsl_system
from pypto.language.op import tensor_ops as _dsl_tensor
from pypto.language.op import tile_ops as _dsl_tile
from pypto.pypto_core import DataType, ir

from ._dsl_invoker import invoke_dsl
from .attr_parser import AttrParser
from .diagnostics import (
    BUG_CLASS_EXCEPTIONS,
    InvalidOperationError,
    ParserError,
    ParserSyntaxError,
    ParserTypeError,
    UnsupportedFeatureError,
    concise_error_message,
)
from .dispatch_semantics import DispatchSemantics
from .parser_state import (
    FunctionContext,
    ParseCallbacks,
    ParserServices,
    ProgramBindings,
)
from .region_context import RegionContext
from .syntax_helpers import (
    ENUM_VALUE_TYPES,
    PLD_CATEGORIES,
)
from .type_resolver import (
    _is_dump_call,  # noqa: PLC2701
)


class OpParser:
    """DSL operator dispatch, argument conversion and buffer metadata."""

    def __init__(
        self,
        *,
        attrs: AttrParser,
        bindings: ProgramBindings,
        callbacks: ParseCallbacks,
        dispatch: DispatchSemantics,
        function: FunctionContext,
        regions: RegionContext,
        services: ParserServices,
    ) -> None:
        self.attrs = attrs
        self.bindings = bindings
        self.callbacks = callbacks
        self.dispatch = dispatch
        self.function = function
        self.regions = regions
        self.services = services
        # Buffer metadata registry for reserve_buffer().base attribute access
        self.buffer_meta: dict[str, dict[str, Any]] = {}

    def parse_op_call(self, call: ast.Call) -> ir.Expr:
        """Parse operation call like pl.tensor.create_tensor() or pl.add().

        Args:
            call: Call AST node

        Returns:
            IR expression from operation
        """
        func = call.func

        # Navigate through attribute chain to find operation
        # e.g., pl.tensor.create_tensor -> ["pl", "tensor", "create_tensor"]
        # e.g., pl.add -> ["pl", "add"]
        attrs: list[str] = []
        node = func
        while isinstance(node, ast.Attribute):
            attrs.insert(0, node.attr)
            node = node.value

        if isinstance(node, ast.Name):
            attrs.insert(0, node.id)

        # pl.aiv_shard / pl.aic_gather (also pl.tile.aiv_shard / pl.tile.aic_gather
        # and the printed pl.tensor.aiv_shard / pl.tensor.aic_gather high-level
        # form): the split mode is inherited from the enclosing ``pl.split_aiv``
        # scope, so intercept before the generic dispatch (the DSL wrapper raises
        # since it cannot resolve the scope mode). The emitted op namespace is
        # type-dispatched inside ``_parse_split_transfer_op`` (tensor.* for a
        # high-level Tensor operand, tile.* otherwise), so a printed
        # ``pl.tensor.aiv_shard(...)`` must also route here for print -> parse.
        if (
            attrs
            and attrs[0] == "pl"
            and attrs[-1] in ("aiv_shard", "aic_gather")
            and (len(attrs) == 2 or (len(attrs) == 3 and attrs[1] in ("tile", "tensor")))
        ):
            return self.parse_split_transfer_op(attrs[-1], call)

        # pld.<op> (2-segment unified short form)
        if len(attrs) == 2 and attrs[0] == "pld":
            return self.parse_pld_op(attrs[1], call)

        # pld.<category>.<op> (3-segment canonical form; category = system/tensor/tile)
        if len(attrs) == 3 and attrs[0] == "pld":
            return self.parse_pld_category_op(attrs[1], attrs[2], call)

        # pl.builtin.<category>.<op> (4-segment) — printer-emitted internal
        # builtin dispatch, e.g. ``pl.builtin.tensor.allreduce(...)``. Matched
        # on ``attrs[1]`` alone so a malformed spelling gets the namespace's own
        # diagnostic instead of falling through to the 2-segment unified path
        # and reporting the useless "Unknown operation 'pl.builtin'".
        if len(attrs) >= 2 and attrs[0] == "pl" and attrs[1] == "builtin":
            return self.parse_builtin_op(attrs[2:], call)

        # pl.tensor.{operation} (3-segment)
        if len(attrs) >= 3 and attrs[0] == "pl" and attrs[1] == "tensor":
            op_name = attrs[2]
            return self.parse_tensor_op(op_name, call)

        # pl.tile.{operation} (3-segment)
        if len(attrs) >= 3 and attrs[0] == "pl" and attrs[1] == "tile":
            op_name = attrs[2]
            return self.parse_tile_op(op_name, call)

        # pl.system.{operation} (3-segment)
        if len(attrs) >= 3 and attrs[0] == "pl" and attrs[1] == "system":
            op_name = attrs[2]
            return self.parse_system_op(op_name, call)

        # pl.array.{operation} (3-segment)
        if len(attrs) >= 3 and attrs[0] == "pl" and attrs[1] == "array":
            op_name = attrs[2]
            return self.parse_array_op(op_name, call)

        # pl.prefetch.{operation} (3-segment)
        if len(attrs) >= 3 and attrs[0] == "pl" and attrs[1] == "prefetch":
            op_name = attrs[2]
            return self.parse_prefetch_op(op_name, call)

        # pl.const(value, dtype) — typed constant literal
        if len(attrs) >= 2 and attrs[0] == "pl" and attrs[1] == "const":
            return self.parse_typed_constant(call)

        # pl.<DataType>.get_byte() — fold into compile-time int constant.
        # e.g. ``pl.FP32.get_byte()`` → 4, ``pl.INT32.get_byte()`` → 4.
        # The call has no side effects (GetByte is a pure accessor) so folding
        # at parse time is safe and lets it work directly inside traced bodies
        # without a separate import of the _window_byte_constants module.
        if len(attrs) == 3 and attrs[0] == "pl" and attrs[-1] == "get_byte":
            return self.parse_dtype_get_byte(attrs[1], call)

        # pl.{operation} (2-segment, unified dispatch or promoted ops)
        if (
            len(attrs) >= 2
            and attrs[0] == "pl"
            and attrs[1] not in ("tensor", "tile", "system", "array", "prefetch")
        ):
            op_name = attrs[1]
            return self.parse_unified_op(op_name, call)

        raise UnsupportedFeatureError(
            f"Unsupported operation call: {ast.unparse(call)}",
            span=self.services.span_tracker.get_span(call),
            hint="Use pl.*, pl.tensor.*, pl.tile.*, or pl.system.* operations",
        )

    def parse_split_transfer_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse ``pl.aiv_shard(x)`` / ``pl.aic_gather(x)``.

        The emitted op namespace is **type-dispatched** on the operand:

        - A high-level **Tensor** operand (``@pl.jit`` / ``pl.spmd`` author-facing
          form, e.g. a ``pl.matmul`` result or a GM tensor param) lowers to
          ``tensor.aiv_shard`` / ``tensor.aic_gather``. This form is
          **region-only** — it is reachable solely through the high-level scoped
          path below (which requires an enclosing ``pl.split_aiv`` region). A
          ``tensor.*`` op is lowered 1:1 to the matching ``tile.*`` op at
          ``ConvertTensorToTileOps``.
        - A **Tile** (or not-yet-typed) operand keeps the ``tile.aiv_shard`` /
          ``tile.aic_gather`` form, preserving the legacy ``@pl.program`` path.
        - A **distributed** tensor operand is rejected (AIV/AIC split only).

        Two surface forms reach this method:

        - **High-level scoped form** ``pl.aiv_shard(x)`` inside a
          ``for aiv_id in pl.split_aiv(mode=...)`` loop. The op inherits the
          split mode from the enclosing scope — the user does not (and must not)
          pass a ``split=`` / ``mode=`` kwarg. The mode is read off
          :attr:`_split_aiv_mode_stack` and stamped as the ``split`` attr. Both
          the tile and tensor forms are valid here.
        - **Outlined low-level form** ``pl.tile.aiv_shard(tile, split=N)`` with an
          explicit ``split=`` kwarg and NO enclosing ``pl.split_aiv`` loop. This
          is what the python printer emits for a function already lowered into
          the explicit ``split_aiv`` form (e.g. after ``LowerAutoVectorSplit`` /
          ``OutlineIncoreScopes``, or a hand-written ``split_aiv`` kernel). The
          split is carried on the op itself, so the form must round-trip without
          re-synthesising the loop wrapper. The explicit ``split`` is taken
          verbatim and stamped as the ``split`` attr. This form is **tile-only**:
          a high-level Tensor operand is rejected (wrap it in a
          ``pl.split_aiv`` region instead).
        """
        span = self.services.span_tracker.get_span(call)
        hint = (
            f"Write 'x = pl.{op_name}(tile)' inside a "
            "'for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):' loop; "
            "the split mode is taken from that scope."
        )

        # Detect the outlined low-level form: an explicit ``split=`` kwarg. A
        # ``mode=`` kwarg is never accepted (the mode is an integer ``split``
        # attr on the lowered op, never a SplitMode literal); any other kwarg is
        # rejected as well.
        explicit_split: ast.expr | None = None
        explicit_lane_stride: ast.expr | None = None
        for kw in call.keywords:
            if kw.arg == "split":
                explicit_split = kw.value
                continue
            if kw.arg == "lane_stride":
                # Compiler bookkeeping stamped by LowerAutoVectorSplit when it
                # balances a ragged boundary across the two AIV lanes; it only
                # ever appears alongside an explicit ``split=`` in the printed
                # outlined form, and is accepted here so that round-trips.
                explicit_lane_stride = kw.value
                continue
            if kw.arg == "mode":
                raise ParserSyntaxError(
                    f"pl.{op_name}() does not take a mode= argument — pass the lowered "
                    "integer 'split=' (outlined form) or rely on the enclosing "
                    "pl.split_aiv(mode=...) scope (high-level form)",
                    span=span,
                    hint=hint,
                )
            raise ParserSyntaxError(
                f"pl.{op_name}() does not accept keyword argument '{kw.arg}'",
                span=span,
                hint=hint,
            )

        if len(call.args) != 1:
            raise ParserSyntaxError(
                f"pl.{op_name}() takes exactly one positional argument (the tile to "
                f"{'shard' if op_name == 'aiv_shard' else 'gather'}), got {len(call.args)}",
                span=span,
                hint=hint,
            )
        operand_expr = self.callbacks.expression(call.args[0])

        # Reject distributed tensors outright — AIV/AIC split only. This must
        # come BEFORE the TensorType dispatch: ``DistributedTensorType`` is a
        # subclass of ``TensorType`` and would otherwise route to the tensor op.
        if isinstance(operand_expr.type, ir.DistributedTensorType):
            raise ParserSyntaxError(
                f"pl.{op_name} does not support distributed tensors (AIV/AIC split only)",
                span=span,
                hint=hint,
            )
        # Type-dispatch the emitted op namespace: a high-level Tensor operand
        # lowers to ``tensor.{op}`` (region-only, converted to the tile op at
        # ConvertTensorToTileOps); a Tile / not-yet-typed operand keeps the
        # legacy ``tile.{op}`` form.
        is_tensor_operand = isinstance(operand_expr.type, ir.TensorType)
        op_ns = "tensor" if is_tensor_operand else "tile"

        if explicit_split is not None:
            # Outlined form — bypass the scope-stack requirement; the split is
            # carried on the op. The printer emits a plain integer literal.
            # An explicit ``split=`` is ONLY valid in the outlined form (no
            # enclosing ``pl.split_aiv`` loop); inside such a loop the mode is
            # inherited from the scope and passing ``split=`` would silently
            # override it, so reject it there.
            if self.regions.split_aiv_mode_stack:
                raise ParserSyntaxError(
                    f"pl.{op_name}() does not take a split= argument inside a "
                    "'for ... in pl.split_aiv(...)' loop — the split mode is inherited "
                    "from that scope",
                    span=span,
                    hint=hint,
                )
            # The outlined form is tile-only: a high-level Tensor operand is not
            # valid here. The tensor form is reachable solely through the
            # region-scoped path (which carries the split mode on the region).
            if is_tensor_operand:
                raise ParserSyntaxError(
                    f"pl.{op_name}(..., split=N) does not accept a high-level Tensor "
                    "operand — the outlined split= form is tile-only. Wrap the Tensor "
                    "in a 'for aiv_id in pl.split_aiv(...)' region instead, which supplies "
                    "the split mode and lowers to the tile op.",
                    span=span,
                    hint=hint,
                )
            if not (isinstance(explicit_split, ast.Constant) and isinstance(explicit_split.value, int)):
                raise ParserSyntaxError(
                    f"pl.{op_name}(..., split=N) requires an integer split (the lowered "
                    "SplitMode value), got "
                    f"'{ast.unparse(explicit_split)}'",
                    span=span,
                    hint=hint,
                )
            kwargs: dict[str, Any] = {"split": int(explicit_split.value)}
            if explicit_lane_stride is not None:
                if op_name == "aic_gather":
                    raise ParserSyntaxError(
                        "pl.aic_gather() does not take a lane_stride= argument: only the Cube -> "
                        "Vector shard is ever rebalanced onto a ragged boundary's valid region, so "
                        "the gather always re-joins the lanes on the box partition",
                        span=span,
                        hint=hint,
                    )
                if not (
                    isinstance(explicit_lane_stride, ast.Constant)
                    and isinstance(explicit_lane_stride.value, int)
                ):
                    raise ParserSyntaxError(
                        f"pl.{op_name}(..., lane_stride=N) requires an integer partition stride, got "
                        f"'{ast.unparse(explicit_lane_stride)}'",
                        span=span,
                        hint=hint,
                    )
                kwargs["lane_stride"] = int(explicit_lane_stride.value)
            return ir.create_op_call(f"{op_ns}.{op_name}", [operand_expr], kwargs, span)

        # High-level scoped form — inherit the mode from the enclosing scope.
        # This is the only path that emits the tensor form (region-only).
        if not self.regions.split_aiv_mode_stack:
            raise ParserSyntaxError(
                f"pl.{op_name}() must be used inside a 'for ... in pl.split_aiv(...)' loop "
                "(or pass an explicit integer 'split=' in the outlined form); it otherwise "
                "inherits the split mode from that scope",
                span=span,
                hint=hint,
            )
        if explicit_lane_stride is not None:
            raise ParserSyntaxError(
                f"pl.{op_name}() does not take a lane_stride= argument inside a "
                "'for ... in pl.split_aiv(...)' loop — the partition follows the region's own "
                "per-lane offsets",
                span=span,
                hint=hint,
            )
        mode = self.regions.split_aiv_mode_stack[-1]
        return ir.create_op_call(f"{op_ns}.{op_name}", [operand_expr], {"split": int(mode.value)}, span)

    def parse_op_positional_arg(self, arg: ast.expr) -> Any:
        """Parse a positional op argument.

        Positional and keyword op arguments must agree: ``pl.fillpad(t,
        pl.PadValue.zero)`` means exactly what ``pl.fillpad(t,
        pad_value=pl.PadValue.zero)`` means. Wrappers have non-Expr parameter
        slots (dtypes, memory spaces, pad modes) that ``parse_expression``
        cannot represent, so those spellings are resolved here the same way
        ``_parse_op_kwargs`` resolves them.

        For ``ast.Attribute`` nodes (e.g. ``pl.INDEX``, ``pl.FP32``), try
        dtype resolution first so wrappers receive a ``DataType`` for slots
        like ``pl.cast(value, dtype)``, then closure evaluation for enum and
        numeric constants (``pl.PadValue.zero``, ``math.inf``) — neither of
        which ``parse_expression`` accepts. ``ast.Name`` resolves enums from
        the closure only when the name is not an IR variable; numeric closure
        names stay on the existing path, which materializes them as
        ``ConstInt`` / ``ConstFloat`` carrying the parser's chosen dtype.
        Everything else falls through to ``parse_expression``, which keeps
        Tensor/Tile/Scalar var lookups, list literals, etc. unchanged.
        """
        if isinstance(arg, ast.Attribute):
            try:
                return self.services.type_resolver.resolve_dtype(arg)
            except ParserError:
                pass
            success, value = self.services.expr_evaluator.try_eval_expr(arg)
            # bool subclasses int — a positional ``True`` is not a numeric
            # constant slot, so leave it on the expression path.
            if success and (
                isinstance(value, ENUM_VALUE_TYPES)
                or (isinstance(value, (int, float)) and not isinstance(value, bool))
            ):
                return value
        elif isinstance(arg, ast.Name) and self.services.scope_manager.lookup_var(arg.id) is None:
            success, value = self.services.expr_evaluator.try_eval_expr(arg)
            if success and isinstance(value, ENUM_VALUE_TYPES):
                return value
        elif isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
            # ``-math.inf`` — parse_expression would descend into the operand
            # and hit the standalone-attribute rejection. Only attributes are
            # intercepted; ``-<name>`` and ``-<literal>`` keep their existing
            # expression handling.
            if isinstance(arg.operand, ast.Attribute):
                success, value = self.services.expr_evaluator.try_eval_expr(arg.operand)
                if success and isinstance(value, (int, float)) and not isinstance(value, bool):
                    return -value
        return self.callbacks.expression(arg)

    def parse_op_kwargs(self, call: ast.Call) -> dict[str, Any]:
        """Parse keyword arguments for an operation call.

        Shared helper for tensor, tile, system, and unified op parsing.

        Args:
            call: Call AST node

        Returns:
            Dictionary of keyword argument names to values
        """
        kwargs: dict[str, Any] = {}
        for keyword in call.keywords:
            # Op-call kwargs are always named; ``**`` spreads are not part of
            # the op-call grammar (keyword.arg is None only for those).
            key = cast("str", keyword.arg)
            value = keyword.value

            # ``attrs={...}`` is a generic compiler-internal attr dict (e.g.
            # ``pipeline_membership``), NOT an op kwarg. The dispatch helpers
            # extract it via ``_parse_op_attrs`` and re-attach it to the built
            # Call, so skip it here.
            if key == "attrs":
                continue

            # Handle dtype specially
            if key == "dtype":
                kwargs[key] = self.services.type_resolver.resolve_dtype(value)
            elif isinstance(value, ast.Constant):
                kwargs[key] = value.value
            elif isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
                kwargs[key] = self.resolve_unary_kwarg(value)
            elif isinstance(value, ast.Name):
                kwargs[key] = self.resolve_name_kwarg(value)
            elif isinstance(value, ast.Attribute):
                kwargs[key] = self.resolve_attribute_kwarg(value)
            elif isinstance(value, ast.List):
                kwargs[key] = self.resolve_list_kwarg(value)
            elif _is_dump_call(value, "float64"):
                # Same value semantics as a float literal in an op kwarg.
                float_expr = self.callbacks.expression(value)
                assert isinstance(float_expr, ir.ConstFloat)
                kwargs[key] = float_expr.value
            else:
                kwargs[key] = self.callbacks.expression(value)
        return kwargs

    def resolve_unary_kwarg(self, value: ast.UnaryOp) -> Any:
        """Resolve a unary op kwarg value (e.g., -1)."""
        if isinstance(value.operand, ast.Constant) and isinstance(value.operand.value, (int, float)):
            return -value.operand.value
        return self.callbacks.expression(value)

    def resolve_name_kwarg(self, value: ast.Name) -> Any:
        """Resolve a Name kwarg value via scope lookup or closure eval."""
        if value.id in ["True", "False"]:
            return value.id == "True"
        if self.services.scope_manager.lookup_var(value.id) is not None:
            return self.callbacks.expression(value)  # IR var from scope
        # Not in IR scope — evaluate from closure (raises ParserTypeError if undefined)
        return self.services.expr_evaluator.eval_expr(value)

    def resolve_attribute_kwarg(self, value: ast.Attribute) -> Any:
        """Resolve an Attribute kwarg value (e.g., pl.FP32, config.field, pipe_buf.base)."""
        try:
            return self.services.type_resolver.resolve_dtype(value)
        except ParserTypeError:
            pass

        # Check buffer metadata registry (e.g., pipe_buf.base from reserve_buffer)
        if isinstance(value.value, ast.Name):
            meta = self.buffer_meta.get(value.value.id)
            if meta is not None and value.attr in meta:
                return meta[value.attr]
            tensor = self.services.scope_manager.lookup_var(value.value.id)
            if (
                value.attr == "shape"
                and isinstance(tensor, ir.Expr)
                and isinstance(tensor.type, ir.TensorType)
            ):
                return self.callbacks.expression(value)

        # Not a dtype or buffer attr — evaluate as a general expression from closure.
        return self.services.expr_evaluator.eval_expr(value)

    def track_buffer_meta(self, var_name: str, call: ast.Call) -> None:
        """Track kwargs from reserve_buffer and import_peer_buffer calls for later attribute access.

        Enables patterns like:
            pipe_buf = pl.reserve_buffer(..., base=0x1000)
            pl.aic_initialize_pipe(..., v2c_consumer_buf=pipe_buf.base)

            peer_buf = pl.import_peer_buffer(name="c2v_slot_buffer", peer_func="vector_bidir")
            pl.aic_initialize_pipe(..., c2v_consumer_buf=peer_buf.base)
        """
        func = call.func
        if not isinstance(func, ast.Attribute):
            return
        if func.attr not in ("reserve_buffer", "import_peer_buffer"):
            return
        meta: dict[str, Any] = {}
        for kw in call.keywords:
            if kw.arg is not None and isinstance(kw.value, ast.Constant):
                meta[kw.arg] = kw.value.value
        if func.attr == "reserve_buffer":
            # Index by (func_name, buffer_name) for cross-function import_peer_buffer resolution
            buf_name = meta.get("name")
            if buf_name is not None:
                self.bindings.buffer_name_meta[(self.function.func_name, buf_name)] = meta
        elif func.attr == "import_peer_buffer":
            # Resolve base from peer's reserve_buffer if available
            buf_name = meta.get("name")
            peer_func_name = meta.get("peer_func")
            if buf_name is not None and peer_func_name is not None:
                peer_key = (peer_func_name, buf_name)
                if peer_key in self.bindings.buffer_name_meta:
                    peer_meta = self.bindings.buffer_name_meta[peer_key]
                    if "base" in peer_meta:
                        meta["base"] = peer_meta["base"]
            if "base" not in meta:
                meta["base"] = -1  # AUTO sentinel
        if meta:
            self.buffer_meta[var_name] = meta

    def resolve_list_kwarg(self, value: ast.List) -> Any:
        """Resolve a List kwarg value, trying closure eval first."""
        # If any element refers to a name in IR scope, parse as IR expressions
        # (mirrors _resolve_name_kwarg: IR scope takes priority over closure)
        if any(
            isinstance(elt, ast.Name) and self.services.scope_manager.lookup_var(elt.id) is not None
            for elt in value.elts
        ):
            return self.callbacks.expression(value)
        success, result = self.services.expr_evaluator.try_eval_expr(value)
        if success and isinstance(result, list):
            return cast("list[Any]", result)
        return self.callbacks.expression(value)

    def dispatch_op(self, module: Any, module_name: str, op_name: str, call: ast.Call) -> ir.Expr:
        """Dispatch an op call to a DSL wrapper module.

        The wrapper owns type-checking and dispatch (e.g. ``tile.add`` already
        routes scalar rhs to ``tile.adds``); the parser only parses args, wraps
        them as DSL types via :func:`invoke_dsl`, pins the call-site span
        through a contextvar, and unwraps the result.

        Args:
            module: A DSL op submodule (``_dsl_tensor`` / ``_dsl_tile`` /
                ``_dsl_system``) or the unified ``pypto.language.op`` namespace.
            module_name: Human-readable module name for error messages.
            op_name: Name of the operation to look up on the module.
            call: Call AST node.

        Returns:
            IR expression from the operation.
        """
        if not hasattr(module, op_name):
            raise InvalidOperationError(
                f"Unknown {module_name} operation: {op_name}",
                span=self.services.span_tracker.get_span(call),
                hint=f"Check if '{op_name}' is a valid {module_name} operation",
            )
        args = [self.parse_op_positional_arg(arg) for arg in call.args]
        kwargs = self.parse_op_kwargs(call)
        attrs = self.attrs.parse_op_attrs(call)
        op_func = getattr(module, op_name)
        span = self.services.span_tracker.get_span(call)
        try:
            return self.attrs.attach_op_attrs(invoke_dsl(op_func, args, kwargs, span), attrs)
        except ParserError:
            raise
        except BUG_CLASS_EXCEPTIONS:
            # Compiler bug, not a bad kernel - surface it with its type and trace intact.
            raise
        except (TypeError, ValueError) as e:
            # Wrapper may have prefixed its message (``pl.<op>:`` from
            # ``_raise_type_dispatch_error`` or ``pl.<module>.<op>:``) or raised
            # bare text from a deeper helper (e.g. ``ValueError("Invalid
            # rounding mode ...")``). Make sure the surfaced error always names
            # the op so users can locate the bad call. When any operand was a
            # Scalar, append a hint pointing at Python operators.
            #
            # Sanitize first: most ops type-check in C++, and a ``CHECK`` throws
            # ``pypto::ValueError`` -- so backend op-validation failures land *here*, not
            # in the ``except Exception`` branch below. Their message still carries
            # FatalLogger's "Check failed: <C++ expr> at <absolute path>.cpp:<line>" tail,
            # which the renderer would splice into the bold ``Error:`` header ahead of the
            # ``-->`` source arrow. It stays reachable through ``__cause__`` under
            # PTO_BACKTRACE=1. A pure-Python ValueError has no tail, so this is a no-op.
            # ``strip_trailing_span`` additionally drops the ``[<file>:<line>:<col>]`` a
            # ``CHECK_SPAN`` leaves in the payload: we raise with ``span=`` below, so the
            # arrow and snippet locate the call anyway, and the inline copy is an absolute
            # path in the middle of the header (often naming an operand's *definition*,
            # not the call, which reads as a contradiction against the arrow).
            msg = concise_error_message(e, strip_trailing_span=True)
            if not msg.startswith(f"pl.{op_name}") and not msg.startswith(f"{module_name}.{op_name}"):
                msg = f"{module_name} operation '{op_name}': {msg}"
            hint = self.scalar_operand_hint(args, kwargs)
            raise InvalidOperationError(msg, span=span, hint=hint) from e
        except Exception as e:
            # ``span=`` below gives the renderer its ``-->`` arrow, so strip the inline
            # ``[<file>:<line>:<col>]`` a ``CHECK_SPAN`` leaves in the payload rather than
            # printing an absolute path inside the bold ``Error:`` header.
            raise InvalidOperationError(
                f"Error in {module_name} operation '{op_name}': "
                f"{concise_error_message(e, strip_trailing_span=True)}",
                span=span,
            ) from e

    @staticmethod
    def scalar_operand_hint(args: list[Any], kwargs: dict[str, Any]) -> str | None:
        """Return the Python-operators hint when any operand has ScalarType.

        Replaces a fragile ``"Scalar" in msg`` substring check; the parsed
        operands carry their IR type explicitly, so we can decide based on
        that rather than the wrapper's error wording.
        """
        for v in (*args, *kwargs.values()):
            if isinstance(v, ir.Expr) and isinstance(v.type, ir.ScalarType):
                return (
                    "For scalar arithmetic / comparison, use Python operators directly "
                    "(e.g. `s1 + s2`, `s1 % s2`, `s1 == s2`)."
                )
        return None

    def parse_tensor_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse tensor operation."""
        if op_name == "alloc":
            return self.parse_printed_alloc_call(call)
        if op_name == "dim":
            folded = self.fold_tensor_dim(call)
            if folded is not None:
                return folded
        # Prefer the DSL wrapper (owns type-checking + dispatch); fall back to
        # the IR-builder layer for pass-internal ops like ``gather_mask``
        # that are emitted by the printer but have no DSL wrapper.
        if hasattr(_dsl_tensor, op_name):
            return self.dispatch_op(_dsl_tensor, "pl.tensor", op_name, call)
        return self.dispatch_ir_builder_op(ir_op.tensor, "pl.tensor", op_name, call)

    @staticmethod
    def static_int(node: ast.expr) -> int | None:
        """Return the value of an integer literal AST node (``0``, ``-1``), else None."""
        if isinstance(node, ast.Constant):
            value = node.value
            return value if isinstance(value, int) and not isinstance(value, bool) else None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = OpParser.static_int(node.operand)
            return None if inner is None else -inner
        return None

    def is_param_dim_symbol(self, expr: Any) -> TypeGuard[ir.Var]:
        """Whether ``expr`` is a dyn-dim symbol this signature declares (see __init__).

        Takes ``Any``: a scope lookup can yield ``None`` or a ``str`` placeholder
        (loop yields), and both must simply answer "no".
        """
        return isinstance(expr, ir.Var) and expr.unique_id in self.function.param_dim_symbols

    def fold_tensor_dim(self, call: ast.Call) -> ir.Expr | None:
        """Fold ``pl.tensor.dim(x, i)`` onto the extent ``x``'s type already names.

        A tensor's declared extent *is* its runtime extent, so reading it back
        mints a *second* scalar for the same quantity, which nothing downstream can
        prove equal to the symbol the tensor types carry. Shapes built from the copy
        then disagree structurally with shapes built from the symbol — breaking, for
        one, calls into a callee that declares that same symbol. Fold instead: one
        runtime extent, one IR name.

        Orchestration bodies only. That is where the symbol is a value: codegen
        defines it there from the declaring param's task-arg descriptor. An
        Inline/InCore callee's placeholder is not the caller's (``InlineFunctions``
        does not substitute callee symbols, and the callee may be reached with a
        statically-shaped actual), so there ``tensor.dim`` stays a runtime read.

        Returns the folded extent, or None to emit ``tensor.dim`` as usual.
        """
        # A Graph body is orchestration too. Skipping it here would mint a second
        # runtime scalar for an extent the signature already names, so shapes built
        # from it can disagree structurally with a callee that uses the symbol.
        if not ir.is_orchestration_like(self.function.func_type):
            return None
        # Every spelling the DSL accepts must fold. The printer normalizes them all
        # to ``dim(x, 0)``, so a spelling that did not fold here would fold on
        # reparse — breaking the round-trip, and leaving the original source with
        # the second-name-for-one-extent bug this fold exists to remove.
        by_keyword = {kw.arg: kw.value for kw in call.keywords}
        if set(by_keyword) - {"tensor", "axis"}:
            return None
        positional = list(call.args)
        tensor_arg = by_keyword.get("tensor") or (positional.pop(0) if positional else None)
        axis_arg = by_keyword.get("axis") or (positional.pop(0) if positional else None)
        if tensor_arg is None or axis_arg is None or positional:
            return None
        # The *tensor* argument must be a bare name: the generic dispatch re-parses
        # the AST, so parsing a non-trivial argument here would emit its statements
        # twice. The axis is under no such constraint — resolving it parses nothing.
        if not isinstance(tensor_arg, ast.Name):
            return None
        axis = self.static_int(axis_arg)
        if axis is None:
            # A named constant (``pl.tensor.dim(x, ROWS)``) is still a static axis.
            evaluated, value = self.services.expr_evaluator.try_eval_expr(axis_arg)
            if not evaluated or not isinstance(value, int) or isinstance(value, bool):
                return None
            axis = value
        tensor_var = self.services.scope_manager.lookup_var(tensor_arg.id)
        tensor_type = getattr(tensor_var, "type", None)
        if not isinstance(tensor_type, ir.TensorType):
            return None
        shape = tensor_type.shape
        if axis < 0:
            axis += len(shape)
        # An out-of-range axis is the op's error to raise, not ours to fold.
        if not 0 <= axis < len(shape):
            return None
        # Only a symbol this signature declares: those are the extents codegen
        # materializes, so the folded value is sure to have a definition in the
        # emitted host code. A local scalar in the shape may since have been
        # reassigned, so it is left alone.
        extent = shape[axis]
        return extent if self.is_param_dim_symbol(extent) else None

    def parse_tile_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse tile operation."""
        if op_name == "alloc":
            return self.parse_printed_alloc_call(call)
        if hasattr(_dsl_tile, op_name):
            return self.dispatch_op(_dsl_tile, "pl.tile", op_name, call)
        return self.dispatch_ir_builder_op(ir_op.tile, "pl.tile", op_name, call)

    def dispatch_ir_builder_op(self, module: Any, module_name: str, op_name: str, call: ast.Call) -> ir.Expr:
        """Dispatch to a raw IR-builder op (no DSL wrapping).

        Used as a fallback when the DSL layer doesn't expose an op that the
        printer emitted (e.g. pass-internal lowerings like ``tile.gather_mask``,
        ``tile.mrgsort_format1``). IR builders take ``span=`` explicitly and
        accept raw ``ir.Expr`` arguments — no DSL wrap/unwrap round-trip.
        """
        if not hasattr(module, op_name):
            raise InvalidOperationError(
                f"Unknown {module_name} operation: {op_name}",
                span=self.services.span_tracker.get_span(call),
                hint=f"Check if '{op_name}' is a valid {module_name} operation",
            )
        args = [self.parse_op_positional_arg(arg) for arg in call.args]
        kwargs = self.parse_op_kwargs(call)
        attrs = self.attrs.parse_op_attrs(call)
        op_func = getattr(module, op_name)
        span = self.services.span_tracker.get_span(call)
        try:
            return self.attrs.attach_op_attrs(op_func(*args, **kwargs, span=span), attrs)
        except ParserError:
            raise
        except BUG_CLASS_EXCEPTIONS:
            # Compiler bug, not a bad kernel - surface it with its type and trace intact.
            raise
        except Exception as e:
            # ``span=`` below gives the renderer its ``-->`` arrow, so strip the inline
            # ``[<file>:<line>:<col>]`` a ``CHECK_SPAN`` leaves in the payload rather than
            # printing an absolute path inside the bold ``Error:`` header.
            raise InvalidOperationError(
                f"Error in {module_name} operation '{op_name}': "
                f"{concise_error_message(e, strip_trailing_span=True)}",
                span=span,
            ) from e

    @staticmethod
    def is_printed_alloc_call(call: ast.Call) -> bool:
        """Return whether the AST call matches printer-emitted ``pl.tile.alloc(...)``
        or ``pl.tensor.alloc(...)``."""
        func = call.func
        return (
            isinstance(func, ast.Attribute)
            and func.attr == "alloc"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr in ("tile", "tensor")
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "pl"
        )

    def parse_printed_alloc_call(self, call: ast.Call) -> ir.Call:
        """Parse printer-emitted ``pl.{tile,tensor}.alloc(...)`` into a raw IR call."""
        func = call.func
        assert isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute)
        namespace = func.value.attr  # "tile" or "tensor"
        op_name = f"{namespace}.alloc"

        # `pinned=True` is the only keyword an alloc carries — it marks an
        # allocation the author declared via a one-argument `pl.MemRef(...)`,
        # which MemoryReuse must leave alone. Everything else stays positional.
        kwargs: dict[str, Any] = {}
        for keyword in call.keywords:
            if keyword.arg != "pinned":
                raise InvalidOperationError(
                    f"{op_name} in printed IR accepts no keyword argument '{keyword.arg}'",
                    span=self.services.span_tracker.get_span(call),
                )
            if not (isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, bool)):
                raise InvalidOperationError(
                    f"{op_name} 'pinned' must be a bool literal",
                    span=self.services.span_tracker.get_span(call),
                )
            kwargs["pinned"] = keyword.value.value
        args = [self.callbacks.expression(arg) for arg in call.args]
        if len(args) != 2:
            raise InvalidOperationError(
                f"{op_name} in printed IR expects 2 positional args (memory_space, size), got {len(args)}",
                span=self.services.span_tracker.get_span(call),
            )
        return ir.create_op_call(op_name, args, kwargs, self.services.span_tracker.get_span(call))

    def parse_system_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse system operation."""
        if op_name == "task_dummy":
            return self.parse_task_dummy(call)
        return self.dispatch_op(_dsl_system, "pl.system", op_name, call)

    def parse_task_dummy(self, call: ast.Call) -> ir.Expr:
        """Parse user-written ``pl.system.task_dummy(deps=[...])``."""
        span = self.services.span_tracker.get_span(call)
        if call.args:
            raise InvalidOperationError(
                "pl.system.task_dummy must not use positional arguments",
                span=span,
            )
        # ``attrs=`` is the machine-only round-trip surface, not a user API: the
        # printer's denylist emits every non-bespoke attr into it, so rejecting
        # it here would turn a printed attr into an unparseable program.
        allowed_kwargs = {"deps", "attrs"}
        for kw in call.keywords:
            if kw.arg not in allowed_kwargs:
                raise ParserTypeError(
                    f"pl.system.task_dummy does not accept keyword argument '{kw.arg}'",
                    span=span,
                    hint="Use pl.system.task_dummy(deps=[task_id]) where each dep is a TaskId value",
                )
        if not any(kw.arg == "deps" for kw in call.keywords):
            raise ParserTypeError(
                "pl.system.task_dummy requires keyword argument 'deps'",
                span=span,
                hint="Use pl.system.task_dummy(deps=[]) for an empty dummy barrier",
            )
        deps = self.dispatch.parse_submit_deps_kwarg("pl.system.task_dummy", call.keywords, span)
        base = ir.create_op_call("system.task_dummy", [], {}, span)
        attrs: list[tuple[str, Any]] = [("dummy_task", True)]
        if deps:
            attrs.append(("manual_dep_edges", deps))
        # ``deps=`` and the op identity are the bespoke carriers for
        # ``manual_dep_edges`` / ``dummy_task``, reconstructed above; both are
        # rejected inside a generic attrs dict, so everything left here is an
        # ordinary attr to recover.
        attrs.extend((self.attrs.parse_op_attrs(call) or {}).items())
        return ir.Call(base.op, base.args, base.kwargs, attrs, base.type, base.span)

    def parse_array_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse array operation (create / get_element / update_element)."""
        return self.dispatch_op(_dsl_array, "pl.array", op_name, call)

    def parse_prefetch_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse async-prefetch operation (make_context / async_prefetch / session / wait)."""
        return self.dispatch_op(_dsl_prefetch, "pl.prefetch", op_name, call)

    def validate_pld_op_call(self, op_name: str, call: ast.Call) -> None:
        """Parser-context checks shared by 2-segment and 3-segment pld paths.

        - ``alloc_window_buffer``: must be intercepted at assignment-LHS.
          Reaching here means it was used outside that context.
        - ``world_size``: host-only.
        """
        span = self.services.span_tracker.get_span(call)

        if op_name == "alloc_window_buffer":
            raise ParserSyntaxError(
                "pld.tensor.alloc_window_buffer must appear as the RHS of a simple assignment "
                "(its result must be bound to a named variable)",
                span=span,
                hint="Write 'buf = pld.tensor.alloc_window_buffer(N)' "
                "(or the short form 'buf = pld.alloc_window_buffer(N)')",
            )

        if op_name == "world_size":
            in_device_scope = any(
                self.regions.is_inside_scope(kind) for kind in (ir.ScopeKind.InCore, ir.ScopeKind.Spmd)
            )
            if self.function.func_level != ir.Level.HOST or in_device_scope:
                raise ParserSyntaxError(
                    "pld.system.world_size() can only be called in HOST orchestration context "
                    "(not inside InCore / SPMD scopes); "
                    f"current function level: {self.function.func_level}",
                    span=span,
                    hint="Use '@pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)' "
                    "on the enclosing function and call outside any nested device-side scope",
                )

    def parse_pld_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse ``pld.<op>(...)`` — 2-segment unified short form.

        Forwards to the 3-segment surface via the unified-dispatch shim in
        :mod:`pypto.language.distributed.op.unified_ops`.
        """
        self.validate_pld_op_call(op_name, call)
        span = self.services.span_tracker.get_span(call)

        if not hasattr(_dsl_pld, op_name):
            raise InvalidOperationError(
                f"Unknown distributed operation 'pld.{op_name}'",
                span=span,
                hint="Available short forms: pld.world_size, pld.get_comm_ctx, pld.rank, "
                "pld.nranks, pld.alloc_window_buffer, pld.window, pld.remote_load, "
                "pld.remote_store",
            )

        return self.dispatch_op(_dsl_pld, "pld", op_name, call)

    def parse_pld_category_op(self, category: str, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse ``pld.<category>.<op>(...)`` — 3-segment canonical form.

        Categories: ``system`` / ``tensor`` / ``tile`` (mirrors ``pl.system`` /
        ``pl.tensor`` / ``pl.tile``).
        """
        span = self.services.span_tracker.get_span(call)
        self.validate_pld_op_call(op_name, call)

        if category not in PLD_CATEGORIES or not hasattr(_dsl_pld, category):
            raise InvalidOperationError(
                f"Unknown distributed category 'pld.{category}'",
                span=span,
                hint="Available categories: pld.system, pld.tensor, pld.tile",
            )

        submodule = getattr(_dsl_pld, category)
        if not hasattr(submodule, op_name):
            raise InvalidOperationError(
                f"Unknown distributed operation 'pld.{category}.{op_name}'",
                span=span,
                hint=f"Check spelling, or list available ops in pypto.language.distributed.op.{category}_ops",
            )

        return self.dispatch_op(submodule, f"pld.{category}", op_name, call)

    def parse_builtin_op(self, segments: list[str], call: ast.Call) -> ir.Expr:
        """Parse printer-emitted ``pl.builtin.<category>.<op>(...)``.

        ``builtin.*`` operators are compiler-internal chip dispatches that
        passes synthesize — today ``builtin.tensor.*``, emitted by
        ``LowerHostTensorCollectives`` for the host ``pld.tensor.*``
        collectives. They are ``internal_only`` in the registry, so no DSL
        wrapper spells them and users write the composite ``pld.tensor.*``
        form instead.

        This path exists so the printer's output past those passes re-parses:
        the print -> parse round-trip must hold for every IR the pipeline can
        produce, and the printer renders a registered ``builtin.<ns>.<op>`` as
        ``pl.builtin.<ns>.<op>`` like any other non-``pld`` operator. It is the
        machine-only reader for that writer — the counterpart of
        ``_parse_printed_alloc_call`` — so it builds through
        ``ir._create_internal_op_call`` and is scoped to the ``builtin.``
        namespace: no other internal operator becomes reachable, and the
        user-facing ``create_op_call`` guard is untouched.

        Because this is a *reader for the printer*, it accepts only what the
        printer can write. Every ``builtin.*`` dispatch is built by one
        function (``MakeBuiltinCallWithAttrs`` in
        ``lower_host_tensor_collectives_pass.cpp``), which unconditionally
        stamps a ``device`` attr and an ``arg_directions`` attr covering every
        positional arg; orchestration codegen then reads both back behind
        internal checks. Hand-written source that omits them would be accepted
        here and blow up much later as an "internal error" — a compiler-bug
        diagnostic for what is really bad user input — so the invariants are
        required up front and a violation is reported as a user error.
        """
        span = self.services.span_tracker.get_span(call)
        if len(segments) != 2:
            raise InvalidOperationError(
                f"Unknown operation '{ast.unparse(call.func)}'",
                span=span,
                hint="pl.builtin is the compiler-internal operator namespace and is spelled "
                "pl.builtin.<category>.<op> (e.g. pl.builtin.tensor.allreduce); it is emitted by "
                "the printer, not written by hand — use the pld.tensor.* collective instead",
            )
        category, op_name = segments
        full_name = f"builtin.{category}.{op_name}"
        if not ir.is_op_registered(full_name):
            raise InvalidOperationError(
                f"Unknown builtin operation 'pl.builtin.{category}.{op_name}'",
                span=span,
                hint="pl.builtin.* names compiler-internal dispatches emitted by lowering passes "
                "(e.g. pl.builtin.tensor.allreduce); check spelling, or use the public "
                "pld.tensor.* collective",
            )

        args = [self.callbacks.expression(arg) for arg in call.args]
        kwargs = self.parse_op_kwargs(call)
        attrs = self.attrs.parse_op_attrs(call) or {}
        self.check_builtin_op_printer_invariants(full_name, args, attrs, span)
        try:
            built = ir._create_internal_op_call(full_name, args, kwargs, span)
        except BUG_CLASS_EXCEPTIONS:
            # Compiler bug, not a bad kernel - surface it with its type and trace intact.
            raise
        except Exception as e:
            raise InvalidOperationError(
                f"Error in builtin operation '{full_name}': "
                f"{concise_error_message(e, strip_trailing_span=True)}",
                span=span,
            ) from e
        return self.attrs.attach_op_attrs(built, attrs)

    def check_builtin_op_printer_invariants(
        self, full_name: str, args: list[Any], attrs: dict[str, object], span: ir.Span
    ) -> None:
        """Reject a ``pl.builtin.*`` call the printer could not have written.

        Keeps the accepted grammar equal to the printer's output, so this
        machine-only surface cannot be used to hand-build a dispatch that
        orchestration codegen would reject with an ``INTERNAL_CHECK`` (see
        ``EmitBuiltinWindowCollectiveDispatch``: it requires a ``device`` attr
        to resolve the rank expression, and one ``arg_directions`` entry per
        positional arg).
        """
        hint = (
            f"pl.builtin.* is emitted by the python printer, not written by hand — "
            f"{full_name} is reached by writing the public pld.{full_name.split('.', 1)[1]} "
            f"collective and letting LowerHostTensorCollectives lower it"
        )
        if "device" not in attrs:
            raise InvalidOperationError(
                f"'pl.{full_name}' requires a device attr naming the dispatching rank "
                f'(attrs={{"device": <rank>}})',
                span=span,
                hint=hint,
            )
        directions = attrs.get("arg_directions")
        if directions is None:
            raise InvalidOperationError(
                f"'pl.{full_name}' requires an arg_directions attr "
                f'(attrs={{"arg_directions": [pl.adir.<dir>, ...]}})',
                span=span,
                hint=hint,
            )
        if len(cast("list[object]", directions)) != len(args):
            raise InvalidOperationError(
                f"'pl.{full_name}' has {len(args)} positional args but "
                f"{len(cast('list[object]', directions))} arg_directions entries; "
                "orchestration codegen requires one direction per arg",
                span=span,
                hint=hint,
            )

    def parse_unified_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse a ``pl.<op>(...)`` call by delegating to the matching DSL wrapper.

        Lookup is restricted to ``pypto.language.op`` rather than the broader
        ``pypto.language`` namespace. The op package re-exports only callable
        ops (unified + promoted tensor/tile/system), so non-op DSL symbols
        like ``pl.range``, ``pl.dynamic``, or ``pl.const`` cannot be invoked
        here as ops — they're handled by their own parser entry points
        upstream of this method. The wrapper itself owns type-checking and
        dispatch (e.g. ``unified_ops.add`` routes Tile+Scalar to
        ``tile.adds``); the parser only forwards arguments and the call-site
        span.
        """
        import pypto.language.op as _pl_op  # noqa: PLC0415 (circular import — `pypto.language` re-exports parser)

        op_func = getattr(_pl_op, op_name, None)
        if op_func is None or not callable(op_func):
            raise InvalidOperationError(
                f"Unknown operation 'pl.{op_name}'",
                span=self.services.span_tracker.get_span(call),
                hint="Check spelling, or use pl.tensor.*/pl.tile.*/pl.system.* for explicit namespacing",
            )
        return self.dispatch_op(_pl_op, "pl", op_name, call)

    def parse_typed_constant(self, call: ast.Call) -> ir.Expr:
        """Parse pl.const(value, dtype) → ConstInt or ConstFloat.

        Args:
            call: Call AST node for pl.const(value, dtype)

        Returns:
            ConstInt or ConstFloat with the specified dtype
        """
        span = self.services.span_tracker.get_span(call)

        if len(call.args) != 2 or call.keywords:
            raise ParserSyntaxError(
                "pl.const() requires exactly 2 arguments: value and dtype",
                span=span,
                hint="Use pl.const(42, pl.INT32) or pl.const(1.0, pl.FP16)",
            )

        # Extract numeric value from first argument (handles Constant and -Constant)
        value_node = call.args[0]
        if _is_dump_call(value_node, "float64"):
            value_expr = self.callbacks.expression(value_node)
            assert isinstance(value_expr, ir.ConstFloat)
            dtype = self.services.type_resolver.resolve_dtype(call.args[1])
            if not dtype.is_float():
                raise ParserTypeError("float64 constants require a floating-point dtype", span=span)
            return ir.ConstFloat(value_expr.value, dtype, span)
        negate = False
        if isinstance(value_node, ast.UnaryOp) and isinstance(value_node.op, ast.USub):
            negate = True
            value_node = value_node.operand

        if not isinstance(value_node, ast.Constant) or not isinstance(value_node.value, (int, float)):
            raise ParserSyntaxError(
                "pl.const() first argument must be a numeric literal",
                span=span,
                hint="Use an int or float literal: pl.const(42, pl.INT32)",
            )

        value = value_node.value
        if negate:
            value = -value

        # Resolve dtype from second argument
        dtype = self.services.type_resolver.resolve_dtype(call.args[1])

        if isinstance(value, float):
            return ir.ConstFloat(value, dtype, span)
        else:
            return ir.ConstInt(value, dtype, span)

    def parse_dtype_get_byte(self, dtype_name: str, call: ast.Call) -> ir.Expr:
        """Parse ``pl.<DataType>.get_byte()`` as a compile-time int constant.

        ``DataType.get_byte()`` is a pure accessor returning ``ceil(bit_width / 8)``
        with no side effects, so folding at parse time is safe. This lets traced
        ``@pl.function`` / ``@pl.jit.host`` bodies use the canonical API directly:
        ``pld.alloc_window_buffer(256 * pl.FP32.get_byte())`` works without an
        external import.

        Args:
            dtype_name: The DataType attribute name (e.g. "FP32", "INT32", "BF16").
            call: The Call AST node (for span tracking).

        Returns:
            ``ir.ConstInt(nbytes, INDEX)`` with the byte-size of the data type.
        """
        span = self.services.span_tracker.get_span(call)

        # Lazily build the mapping from dtype attribute name → get_byte() value.
        # Module-level caching via a class attribute avoids re-scanning on every
        # call site within a single parser instance (a function body may contain
        # many alloc_window_buffer calls).
        try:
            cache = OpParser._dtype_byte_cache
        except AttributeError:
            import pypto.language as _pl  # noqa: PLC0415 (lazy import of DataType constants)

            cache: dict[str, int] = {}
            for attr_name in dir(_pl):
                val = getattr(_pl, attr_name)
                if isinstance(val, DataType):
                    cache[attr_name] = val.get_byte()
            OpParser._dtype_byte_cache = cache

        if dtype_name not in cache:
            known = ", ".join(sorted(cache.keys()))
            raise ParserTypeError(
                f"Unknown DataType '{dtype_name}' in pl.{dtype_name}.get_byte()",
                span=span,
                hint=f"Valid DataType names: {known}",
            )

        return ir.ConstInt(cache[dtype_name], DataType.INDEX, span)
