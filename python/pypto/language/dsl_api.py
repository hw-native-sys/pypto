# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""DSL API helpers for writing IR functions."""

from typing import Any, Generic, Optional, TypeVar, Union, overload

T = TypeVar("T", int, tuple[int, tuple[Any, ...]])


class RangeIterator(Generic[T]):
    """Iterator for pl.range() that supports tuple unpacking."""

    def __init__(
        self,
        stop: int,
        start: int = 0,
        step: int = 1,
        init_values: Optional[list[Any]] = None,
    ):
        """Initialize range iterator.

        Args:
            stop: Stop value
            start: Start value (default 0)
            step: Step value (default 1)
            init_values: Initial values for iter_args
        """
        self.start = start
        self.stop = stop
        self.step = step
        self.init_values = init_values or []
        self.current = start

    def __iter__(self) -> "RangeIterator[T]":
        """Return iterator."""
        return self

    @overload
    def __next__(self: "RangeIterator[int]") -> int: ...

    @overload
    def __next__(self: "RangeIterator[tuple[int, tuple[Any, ...]]]") -> tuple[int, tuple[Any, ...]]: ...

    def __next__(self) -> Union[int, tuple[int, tuple[Any, ...]]]:
        """Get next iteration value.

        Returns:
            If no init_values: just the loop variable (int)
            If init_values provided: Tuple of (loop_var, (iter_arg_values...))
        """
        if self.current >= self.stop:
            raise StopIteration

        value = self.current
        self.current += self.step

        # Return just the value if no init_values, otherwise return (value, iter_args_tuple)
        if not self.init_values:
            return value  # type: ignore[return-value]
        return (value, tuple(self.init_values))  # type: ignore[return-value]


@overload
def range(*args: int, init_values: None = None) -> RangeIterator[int]: ...


@overload
def range(*args: int, init_values: list[Any]) -> RangeIterator[tuple[int, tuple[Any, ...]]]: ...


def range(
    *args: int, init_values: Optional[list[Any]] = None
) -> Union[RangeIterator[int], RangeIterator[tuple[int, tuple[Any, ...]]]]:
    """Create a range iterator for for loops.

    Supports two patterns:
        Simple:    for i in pl.range(10):
        Iter args: for i, (var1, var2) in pl.range(16, init_values=[init1, init2]):

    Args:
        *args: Positional arguments (stop) or (start, stop) or (start, stop, step)
        init_values: Initial values for iteration arguments

    Returns:
        If no init_values: RangeIterator yielding loop variable (int)
        If init_values: RangeIterator yielding (loop_var, (iter_args...))

    Examples:
        >>> for i in pl.range(10):
        ...     result = pl.op.tensor.add(x, 1.0)
        >>> for i, (sum,) in pl.range(10, init_values=[0]):
        ...     sum = sum + i
        ...     sum_out = pl.yield_(sum)
    """
    if len(args) == 1:
        return RangeIterator(args[0], init_values=init_values)
    elif len(args) == 2:
        return RangeIterator(args[1], args[0], init_values=init_values)
    elif len(args) == 3:
        return RangeIterator(args[1], args[0], args[2], init_values=init_values)
    else:
        raise ValueError("range() takes 1 to 3 positional arguments")


def yield_(*values: Any) -> Any:
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


__all__ = ["range", "yield_", "RangeIterator"]
