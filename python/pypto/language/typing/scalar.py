# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Scalar wrapper type for PyPTO Language DSL."""

from typing import Any, TypeAlias, cast

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import ConstInt, Expr, Span


def _validate_scalar_meta_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    """Validate ScalarMeta.__call__ argument structure."""
    allowed_kwargs = {"dtype", "expr", "_annotation_only"}
    unexpected = set(kwargs) - allowed_kwargs
    if unexpected:
        name = sorted(unexpected)[0]
        raise TypeError(f"Scalar() got an unexpected keyword argument '{name}'")

    if len(args) > 3:
        raise TypeError(f"Scalar() takes at most 3 positional arguments but {len(args)} were given")

    param_names = ("dtype", "expr", "_annotation_only")
    for index, name in enumerate(param_names[: len(args)]):
        if name in kwargs:
            raise TypeError(f"Scalar() got multiple values for argument '{name}'")


class ScalarMeta(type):
    """Metaclass for Scalar to enable subscript notation."""

    def __getitem__(cls, dtype: DataType) -> "Scalar":
        """Enable Scalar[dtype] syntax (recommended).

        Args:
            dtype: Data type

        Returns:
            Scalar instance with dtype (annotation-only mode)
        """
        return cls(dtype, _annotation_only=True)

    def __call__(cls, *args: Any, **kwargs: Any) -> "Scalar":
        """Enable both Scalar(dtype) syntax and runtime wrapping.

        Args:
            dtype: Data type (for annotation mode)
            expr: IR expression to wrap (for runtime mode)
            _annotation_only: Internal flag for annotation-only mode

        Returns:
            Scalar instance
        """
        # Subclasses (e.g. DynVar) bypass Scalar's arg reinterpretation logic
        # and delegate directly to their own __init__.
        if cls is not Scalar:
            return cast("Scalar", type.__call__(cls, *args, **kwargs))

        _validate_scalar_meta_call(args, kwargs)

        # When called with just dtype (legacy notation), treat it as annotation mode
        dtype = kwargs.get("dtype", args[0] if len(args) > 0 else None)
        expr = kwargs.get("expr", args[1] if len(args) > 1 else None)
        annotation_only = kwargs.get("_annotation_only", args[2] if len(args) > 2 else False)

        if dtype is not None and expr is None and not annotation_only:
            annotation_only = True
        return cast("Scalar", type.__call__(cls, dtype, expr, annotation_only))


class Scalar(metaclass=ScalarMeta):
    """Scalar type for PyPTO Language DSL.

    This class serves dual purposes:
    1. Type annotation helper for function signatures
    2. Runtime wrapper around IR Expr/Call objects

    Annotation mode (used in type hints):
        x: pl.Scalar[pl.FP32]
        count: pl.Scalar[pl.INT32]

    Runtime mode (wraps IR expressions):
        scalar_value = pl.scalar.create(3.14, dtype=pl.FP32)
        # Returns Scalar wrapping the Call expression

    Examples:
        >>> import pypto.language as pl
        >>>
        >>> @pl.function
        ... def add_scalar(
        ...     x: pl.Tensor[[64], pl.FP32],
        ...     scalar: pl.Scalar[pl.FP32]
        ... ) -> pl.Tensor[[64], pl.FP32]:
        ...     result: pl.Tensor[[64], pl.FP32] = pl.add(x, scalar)
        ...     return result
    """

    def __init__(
        self,
        dtype: DataType | None = None,
        expr: Expr | None = None,
        _annotation_only: bool = False,
    ):
        """Initialize Scalar.

        Args:
            dtype: Data type (for annotation mode)
            expr: IR expression to wrap (for runtime mode)
            _annotation_only: Internal flag for annotation-only mode

        Raises:
            ValueError: If neither dtype nor expr is provided
        """
        if _annotation_only:
            # Annotation mode: store dtype for type checking
            if dtype is None:
                raise ValueError("dtype is required for annotation mode")
            self.dtype = dtype
            self.expr = None
            self._annotation_only = True
        elif expr is not None:
            # Runtime mode: wrap IR expression
            self.expr = expr
            self.dtype = None
            self._annotation_only = False
        else:
            raise ValueError("Either dtype (for annotation) or expr (for runtime) must be provided")

    def unwrap(self) -> Expr:
        """Unwrap to get the underlying IR expression.

        Returns:
            The wrapped IR expression

        Raises:
            RuntimeError: If this is an annotation-only instance
        """
        if self._annotation_only:
            raise RuntimeError("Cannot unwrap annotation-only Scalar")
        if self.expr is None:
            raise RuntimeError("No expression to unwrap")
        return self.expr

    def __repr__(self) -> str:
        """Return string representation."""
        if self._annotation_only:
            return f"Scalar[{self.dtype}]"
        return f"Scalar(expr={self.expr})"

    def __bool__(self) -> bool:
        """Prevent implicit boolean conversion of symbolic Scalar values.

        Defined so that type checkers (pyright, mypy) do not infer that
        ``if scalar:`` is always truthy.  At runtime, a symbolic IR
        wrapper has no concrete truth value.

        Raises:
            TypeError: Always — Scalar cannot be converted to bool.
        """
        raise TypeError(
            "Cannot convert Scalar to bool. "
            "Scalar wraps a symbolic IR expression and has no concrete truth value."
        )

    # ------------------------------------------------------------------
    # Arithmetic operators — enable type-checked DSL expressions like
    # ``n * 2`` or ``n // 4`` where ``n`` is a Scalar parameter.
    # ------------------------------------------------------------------

    def __add__(self, other: "int | float | Scalar") -> "Scalar":
        return Scalar(expr=self.unwrap() + (other.unwrap() if isinstance(other, Scalar) else other))

    def __radd__(self, other: "int | float") -> "Scalar":
        return Scalar(expr=other + self.unwrap())

    def __sub__(self, other: "int | float | Scalar") -> "Scalar":
        return Scalar(expr=self.unwrap() - (other.unwrap() if isinstance(other, Scalar) else other))

    def __rsub__(self, other: "int | float") -> "Scalar":
        return Scalar(expr=other - self.unwrap())

    def __mul__(self, other: "int | float | Scalar") -> "Scalar":
        return Scalar(expr=self.unwrap() * (other.unwrap() if isinstance(other, Scalar) else other))

    def __rmul__(self, other: "int | float") -> "Scalar":
        return Scalar(expr=other * self.unwrap())

    def __truediv__(self, other: "int | float | Scalar") -> "Scalar":
        return Scalar(expr=self.unwrap() / (other.unwrap() if isinstance(other, Scalar) else other))

    def __rtruediv__(self, other: "int | float | Scalar") -> "Scalar":
        return Scalar(expr=(other.unwrap() if isinstance(other, Scalar) else other) / self.unwrap())

    def __floordiv__(self, other: "int | float | Scalar") -> "Scalar":
        return Scalar(expr=self.unwrap() // (other.unwrap() if isinstance(other, Scalar) else other))

    def __rfloordiv__(self, other: "int | float | Scalar") -> "Scalar":
        return Scalar(expr=(other.unwrap() if isinstance(other, Scalar) else other) // self.unwrap())

    def __mod__(self, other: "int | float | Scalar") -> "Scalar":
        return Scalar(expr=self.unwrap() % (other.unwrap() if isinstance(other, Scalar) else other))

    def __lshift__(self, other: "int | Scalar") -> "Scalar":
        return Scalar(expr=self.unwrap() << (other.unwrap() if isinstance(other, Scalar) else other))

    def __rlshift__(self, other: int) -> "Scalar":
        return Scalar(expr=other << self.unwrap())

    def __rshift__(self, other: "int | Scalar") -> "Scalar":
        return Scalar(expr=self.unwrap() >> (other.unwrap() if isinstance(other, Scalar) else other))

    def __rrshift__(self, other: int) -> "Scalar":
        return Scalar(expr=other >> self.unwrap())

    # ------------------------------------------------------------------
    # Comparison operators — return Scalar wrapping the IR comparison node.
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> "Scalar":  # type: ignore[override]
        if not isinstance(other, (Scalar, int, float)):
            return NotImplemented  # type: ignore[return-value]
        return Scalar(expr=self.unwrap() == (other.unwrap() if isinstance(other, Scalar) else other))

    def __ne__(self, other: object) -> "Scalar":  # type: ignore[override]
        if not isinstance(other, (Scalar, int, float)):
            return NotImplemented  # type: ignore[return-value]
        return Scalar(expr=self.unwrap() != (other.unwrap() if isinstance(other, Scalar) else other))

    def __hash__(self) -> int:
        # Required when __eq__ is overridden. Scalar wraps a symbolic expression,
        # so we fall back to identity-based hashing.
        return id(self)

    def __lt__(self, other: "int | float | Scalar") -> "Scalar":
        return Scalar(expr=self.unwrap() < (other.unwrap() if isinstance(other, Scalar) else other))

    def __le__(self, other: "int | float | Scalar") -> "Scalar":
        return Scalar(expr=self.unwrap() <= (other.unwrap() if isinstance(other, Scalar) else other))

    def __gt__(self, other: "int | float | Scalar") -> "Scalar":
        return Scalar(expr=self.unwrap() > (other.unwrap() if isinstance(other, Scalar) else other))

    def __ge__(self, other: "int | float | Scalar") -> "Scalar":
        return Scalar(expr=self.unwrap() >= (other.unwrap() if isinstance(other, Scalar) else other))

    # ------------------------------------------------------------------
    # In-place operators for RangeIterator compatibility.
    # ------------------------------------------------------------------

    def __iadd__(self, other: "int | float | Scalar") -> "Scalar":
        return self.__add__(other)

    @classmethod
    def __class_getitem__(cls, item: DataType) -> "Scalar":
        """Support static type checkers for Scalar[dtype] syntax."""
        return cls.__getitem__(item)


class RuntimeScalarMarker(Scalar):
    """Marker for a scalar parameter whose value is supplied at dispatch.

    A ``pl.Scalar[dtype]`` annotation carries a type but no value, so
    annotation-driven signature mode (``compile()`` / ``lower()`` with no
    tensor arguments) needs one value per scalar parameter. Passing a literal
    **specializes** that value into the compiled artifact; passing
    :data:`RUNTIME` leaves the parameter **unspecialized** — it stays a real
    ``pl.Scalar`` parameter in the generated program and its value is supplied
    at dispatch, exactly like a ``pl.dynamic`` dimension extent. Unspecialized
    scalars also drop out of the specialization cache key, so one artifact
    serves every runtime value.

    Subclasses :class:`Scalar` so that a type checker accepts it as the default
    of a scalar parameter — ``n: pl.Scalar[dtype] = pl.RUNTIME`` — for the same
    reason :class:`~pypto.language.typing.dynamic.DynVar` does: the marker
    stands in wherever a ``Scalar`` is expected. It carries no dtype of its own;
    the parameter's annotation supplies that.

    Use the :data:`RUNTIME` singleton rather than instantiating this class.

    Examples:
        >>> import pypto.language as pl
        >>>
        >>> # num_tokens varies per step: keep it out of the artifact.
        >>> compiled = prefill_fwd.compile(num_tokens=pl.RUNTIME)  # doctest: +SKIP
    """

    def __init__(self) -> None:
        """Initialize the marker with no dtype and no wrapped expression.

        Bypasses :meth:`Scalar.__init__`, which requires one of the two — this
        marker deliberately has neither, and its dtype comes from the annotated
        parameter it defaults.
        """
        self.dtype = None
        self.expr = None
        self._annotation_only = False

    def unwrap(self) -> Expr:
        """Reject use in an expression.

        Raises:
            RuntimeError: Always — the marker has no value to unwrap.
        """
        raise RuntimeError(
            "pl.RUNTIME is a compile-time marker with no value. Pass it to compile() or "
            "lower() to leave a scalar parameter unspecialized; it cannot take part in an "
            "expression."
        )

    def __repr__(self) -> str:
        """Return the marker's canonical spelling."""
        return "pl.RUNTIME"


RUNTIME = RuntimeScalarMarker()
"""Singleton :class:`RuntimeScalarMarker` — see the class docstring."""


BoolLike: TypeAlias = bool | Scalar | Expr
"""Type alias for predicate parameters accepting a Python bool, a Scalar, or a raw Expr."""


def predicate_to_expr(value: BoolLike | None, span: Span | None = None) -> Expr | None:
    """Coerce an optional boolean predicate operand to an ``Expr``.

    A Python ``bool`` becomes ``ConstInt(.., BOOL)`` — a compile-time constant an
    operator's lowering can fold away. A :class:`Scalar` (typically a comparison
    such as ``k == 0``) is unwrapped to the symbolic expression it carries, which
    stays a runtime value.

    Args:
        value: Predicate to coerce, or ``None`` to pass through
        span: Optional span for a materialized constant

    Returns:
        The corresponding ``Expr``, or ``None`` when @p value is ``None``
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return ConstInt(int(value), DataType.BOOL, span if span is not None else Span.unknown())
    if isinstance(value, Scalar):
        return value.unwrap()
    return value


__all__ = ["RUNTIME", "BoolLike", "RuntimeScalarMarker", "Scalar", "predicate_to_expr"]
