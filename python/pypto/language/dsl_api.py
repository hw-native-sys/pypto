# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""DSL API helpers for writing IR functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Optional, TypeVar, Union, overload

if TYPE_CHECKING:
    from pypto.language.typing import Scalar, Tensor, Tile

# Range argument type: int literal or Scalar variable
RangeArg = Union[int, "Scalar"]

# Condition argument type: bool literal or Scalar variable
CondArg = Union[bool, "Scalar"]

ExprType = TypeVar("ExprType", "Scalar", "Tensor", "Tile")


T = TypeVar("T")
W = TypeVar("W")

# TypeVars for overloads
T1 = TypeVar("T1", "Scalar", "Tensor", "Tile")
T2 = TypeVar("T2", "Scalar", "Tensor", "Tile")
T3 = TypeVar("T3", "Scalar", "Tensor", "Tile")
T4 = TypeVar("T4", "Scalar", "Tensor", "Tile")
T5 = TypeVar("T5", "Scalar", "Tensor", "Tile")


class RangeIterator(Generic[T]):
    """Iterator for pl.range() that supports tuple unpacking."""

    def __init__(
        self,
        stop: RangeArg,
        start: RangeArg = 0,
        step: RangeArg = 1,
        init_values: Optional[tuple[Any, ...]] = None,
    ):
        """Initialize range iterator.

        Args:
            stop: Stop value (int or Scalar)
            start: Start value (default 0, int or Scalar)
            step: Step value (default 1, int or Scalar)
            init_values: Initial values for iter_args
        """
        self.start = start
        self.stop = stop
        self.step = step
        self.init_values = init_values or ()
        self.current = start

    def __iter__(self) -> "RangeIterator[T]":
        """Return iterator."""
        return self

    @overload
    def __next__(self: "RangeIterator[int]") -> int: ...

    @overload
    def __next__(
        self: "RangeIterator[tuple[int, tuple[T1]]]",
    ) -> tuple[int, tuple[T1]]: ...

    @overload
    def __next__(
        self: "RangeIterator[tuple[int, tuple[T1, T2]]]",
    ) -> tuple[int, tuple[T1, T2]]: ...

    @overload
    def __next__(
        self: "RangeIterator[tuple[int, tuple[T1, T2, T3]]]",
    ) -> tuple[int, tuple[T1, T2, T3]]: ...

    @overload
    def __next__(
        self: "RangeIterator[tuple[int, tuple[T1, T2, T3, T4]]]",
    ) -> tuple[int, tuple[T1, T2, T3, T4]]: ...

    @overload
    def __next__(
        self: "RangeIterator[tuple[int, tuple[T1, T2, T3, T4, T5]]]",
    ) -> tuple[int, tuple[T1, T2, T3, T4, T5]]: ...

    def __next__(self) -> Union[int, tuple[int, tuple[Any, ...]]]:
        """Get next iteration value.

        Returns:
            If no init_values: just the loop variable (int)
            If init_values provided: Tuple of (loop_var, (iter_arg_values...))
        """
        if self.current >= self.stop:  # type: ignore[operator]
            raise StopIteration

        value = self.current
        self.current += self.step  # type: ignore[operator]

        # Return just the value if no init_values, otherwise return (value, iter_args_tuple)
        if not self.init_values:
            return value  # type: ignore[return-value]
        return (value, self.init_values)  # type: ignore[return-value]


def _make_range_iterator(
    *args: RangeArg, init_values: Optional[tuple[Any, ...]] = None, func_name: str = "range"
) -> Union[RangeIterator[int], RangeIterator[tuple[int, tuple[Any, ...]]]]:
    """Shared implementation for range() and parallel()."""
    if len(args) == 1:
        return RangeIterator(args[0], init_values=init_values)
    elif len(args) == 2:
        return RangeIterator(args[1], args[0], init_values=init_values)
    elif len(args) == 3:
        return RangeIterator(args[1], args[0], args[2], init_values=init_values)
    else:
        raise ValueError(f"{func_name}() takes 1 to 3 positional arguments")


@overload
def range(*args: RangeArg, init_values: None = None) -> RangeIterator[int]: ...


@overload
def range(*args: RangeArg, init_values: tuple[T1]) -> RangeIterator[tuple[int, tuple[T1]]]: ...


@overload
def range(*args: RangeArg, init_values: tuple[T1, T2]) -> RangeIterator[tuple[int, tuple[T1, T2]]]: ...


@overload
def range(
    *args: RangeArg, init_values: tuple[T1, T2, T3]
) -> RangeIterator[tuple[int, tuple[T1, T2, T3]]]: ...


@overload
def range(
    *args: RangeArg, init_values: tuple[T1, T2, T3, T4]
) -> RangeIterator[tuple[int, tuple[T1, T2, T3, T4]]]: ...


@overload
def range(
    *args: RangeArg, init_values: tuple[T1, T2, T3, T4, T5]
) -> RangeIterator[tuple[int, tuple[T1, T2, T3, T4, T5]]]: ...


def range(
    *args: RangeArg, init_values: Optional[tuple[Any, ...]] = None
) -> Union[RangeIterator[int], RangeIterator[tuple[int, tuple[Any, ...]]]]:
    """Create a range iterator for for loops.

    Supports two patterns:
        Simple:    for i in pl.range(10):
        Iter args: for i, (var1, var2) in pl.range(16, init_values=(init1, init2)):

    Args can be int literals or Scalar variables:
        for i in pl.range(n):  # n is pl.Scalar[pl.INT64]
        for i in pl.range(0, n, 1):
        for i in pl.range(n * 2 + 1):

    Args:
        *args: Positional arguments (stop) or (start, stop) or (start, stop, step).
            Each argument can be an int literal or a pl.Scalar value.
        init_values: Initial values for iteration arguments

    Returns:
        If no init_values: RangeIterator yielding loop variable (int)
        If init_values: RangeIterator yielding (loop_var, (iter_args...))

    Examples:
        >>> for i in pl.range(10):
        ...     result = pl.add(x, 1.0)
        >>> for i in pl.range(n):  # n: pl.Scalar[pl.INT64]
        ...     result = pl.add(x, 1.0)
        >>> for i, (sum,) in pl.range(10, init_values=(0,)):
        ...     sum = sum + i
        ...     sum_out = pl.yield_(sum)
    """
    return _make_range_iterator(*args, init_values=init_values, func_name="range")


@overload
def parallel(*args: RangeArg, init_values: None = None) -> RangeIterator[int]: ...


@overload
def parallel(*args: RangeArg, init_values: tuple[T1]) -> RangeIterator[tuple[int, tuple[T1]]]: ...


@overload
def parallel(*args: RangeArg, init_values: tuple[T1, T2]) -> RangeIterator[tuple[int, tuple[T1, T2]]]: ...


@overload
def parallel(
    *args: RangeArg, init_values: tuple[T1, T2, T3]
) -> RangeIterator[tuple[int, tuple[T1, T2, T3]]]: ...


@overload
def parallel(
    *args: RangeArg, init_values: tuple[T1, T2, T3, T4]
) -> RangeIterator[tuple[int, tuple[T1, T2, T3, T4]]]: ...


@overload
def parallel(
    *args: RangeArg, init_values: tuple[T1, T2, T3, T4, T5]
) -> RangeIterator[tuple[int, tuple[T1, T2, T3, T4, T5]]]: ...


def parallel(
    *args: RangeArg, init_values: Optional[tuple[Any, ...]] = None
) -> Union[RangeIterator[int], RangeIterator[tuple[int, tuple[Any, ...]]]]:
    """Create a parallel range iterator for parallel for loops.

    Behaves identically to range() at runtime. The distinction is used by the
    parser to emit ForKind.Parallel instead of ForKind.Sequential.

    Args:
        *args: Positional arguments (stop) or (start, stop) or (start, stop, step).
            Each argument can be an int literal or a pl.Scalar value.
        init_values: Initial values for iteration arguments

    Returns:
        If no init_values: RangeIterator yielding loop variable (int)
        If init_values: RangeIterator yielding (loop_var, (iter_args...))
    """
    return _make_range_iterator(*args, init_values=init_values, func_name="parallel")


class WhileIterator(Generic[W]):
    """Iterator for pl.while_() that supports tuple unpacking for iter_args."""

    def __init__(self, *, init_values: Optional[tuple[Any, ...]] = None):
        """Initialize while iterator.

        Args:
            init_values: Initial values for iter_args (required for while loops)
        """
        if init_values is None:
            raise ValueError("while_() requires init_values to be specified")
        self.init_values = init_values
        self._exhausted = False

    def __iter__(self) -> "WhileIterator[W]":
        """Return iterator."""
        return self

    @overload
    def __next__(self: "WhileIterator[tuple[T1]]") -> tuple[T1]: ...

    @overload
    def __next__(self: "WhileIterator[tuple[T1, T2]]") -> tuple[T1, T2]: ...

    @overload
    def __next__(self: "WhileIterator[tuple[T1, T2, T3]]") -> tuple[T1, T2, T3]: ...

    @overload
    def __next__(
        self: "WhileIterator[tuple[T1, T2, T3, T4]]",
    ) -> tuple[T1, T2, T3, T4]: ...

    @overload
    def __next__(
        self: "WhileIterator[tuple[T1, T2, T3, T4, T5]]",
    ) -> tuple[T1, T2, T3, T4, T5]: ...

    @overload
    def __next__(self: "WhileIterator[tuple[Any, ...]]") -> tuple[Any, ...]: ...

    def __next__(self) -> tuple[Any, ...]:
        """Get next iteration value.

        Returns:
            Tuple of iter_arg values
        """
        if self._exhausted:
            raise StopIteration

        # Only iterate once - the parser will handle the while loop
        self._exhausted = True
        return self.init_values  # type: ignore[return-value]


@overload
def while_(*, init_values: tuple[T1]) -> WhileIterator[tuple[T1]]: ...


@overload
def while_(*, init_values: tuple[T1, T2]) -> WhileIterator[tuple[T1, T2]]: ...


@overload
def while_(*, init_values: tuple[T1, T2, T3]) -> WhileIterator[tuple[T1, T2, T3]]: ...


@overload
def while_(*, init_values: tuple[T1, T2, T3, T4]) -> WhileIterator[tuple[T1, T2, T3, T4]]: ...


@overload
def while_(*, init_values: tuple[T1, T2, T3, T4, T5]) -> WhileIterator[tuple[T1, T2, T3, T4, T5]]: ...


def while_(*, init_values: Optional[tuple[ExprType, ...]] = None) -> WhileIterator[tuple[ExprType, ...]]:
    """Create a while iterator for while loops.

    Always requires init_values to specify loop-carried state.
    The loop condition must be specified as the first statement in the loop body using pl.cond().

    Pattern:
        for (var1, var2) in pl.while_(init_values=(init1, init2)):
            pl.cond(condition)
            # loop body
            var1_out, var2_out = pl.yield_(var1_updated, var2_updated)

    Args:
        init_values: Initial values for iteration arguments (required)

    Returns:
        WhileIterator yielding tuple of iter_args

    Raises:
        ValueError: If init_values is not provided

    Examples:
        >>> for (x,) in pl.while_(init_values=(0,)):
        ...     pl.cond(x < 10)
        ...     x = x + 1
        ...     x_out = pl.yield_(x)
        >>>
        >>> for (x, y) in pl.while_(init_values=(0, 1)):
        ...     pl.cond(x < n)
        ...     x_new = x + 1
        ...     y_new = y * 2
        ...     x_out, y_out = pl.yield_(x_new, y_new)
    """
    return WhileIterator(init_values=init_values)


@overload
def yield_(value: T1, /) -> T1: ...


@overload
def yield_(v1: T1, v2: T2, /) -> tuple[T1, T2]: ...


@overload
def yield_(v1: T1, v2: T2, v3: T3, /) -> tuple[T1, T2, T3]: ...


@overload
def yield_(v1: T1, v2: T2, v3: T3, v4: T4, /) -> tuple[T1, T2, T3, T4]: ...


@overload
def yield_(v1: T1, v2: T2, v3: T3, v4: T4, v5: T5, /) -> tuple[T1, T2, T3, T4, T5]: ...


def yield_(*values: Any) -> Union[Any, tuple[Any, ...]]:
    """Yield values from a scope (for, if).

    This function is used to explicitly return values from nested scopes
    and create SSA phi nodes.

    Args:
        *values: Values to yield

    Returns:
        The yielded value(s). For single value, returns the value.
        For multiple values, returns tuple.

    Examples:
        >>> # Single value yield
        >>> result = pl.yield_(x + 1)
        >>>
        >>> # Multiple value yield
        >>> a, b = pl.yield_(x, y)
    """
    if len(values) == 1:
        return values[0]
    return tuple(values)


def cond(condition: CondArg) -> None:
    """Specify the condition for a pl.while_() loop.

    This function must be the first statement in a pl.while_() loop body.
    It is purely syntactic - the parser extracts the condition and sets it on the WhileStmt.

    Args:
        condition: While loop condition (bool literal or Scalar variable)

    Examples:
        >>> for (x,) in pl.while_(init_values=(0,)):
        ...     pl.cond(x < 10)
        ...     x = x + 1
        ...     x_out = pl.yield_(x)
    """
    # Runtime no-op - parser handles semantics
    pass


__all__ = [
    "range",
    "parallel",
    "while_",
    "yield_",
    "cond",
    "RangeIterator",
    "WhileIterator",
    "RangeArg",
    "CondArg",
]
