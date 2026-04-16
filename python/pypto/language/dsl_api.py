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

from typing import TYPE_CHECKING, Any, Generic, TypeVar, Union, cast, overload

if TYPE_CHECKING:
    from pypto.language.typing import Scalar, Tensor, Tile
    from pypto.pypto_core import ir

from pypto.pypto_core.ir import SplitMode

from .optimizations import Optimization


class _ChunkedLoopOptimizerCall:
    """Result of calling chunked_loop_optimizer(split=...).

    Stores the split mode to pass to the AutoInCore scope.
    """

    def __init__(self, split: SplitMode = SplitMode.UP_DOWN) -> None:
        self.split = split

    def __repr__(self) -> str:
        return f"chunked_loop_optimizer(split={self.split!r})"


class _ChunkedLoopOptimizer:
    """Sentinel type for optimization=pl.chunked_loop_optimizer in pl.at().

    Can be used bare or called with a split mode:
    - ``optimization=pl.chunked_loop_optimizer``
    - ``optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.UP_DOWN)``
    """

    def __call__(self, *, split: SplitMode = SplitMode.UP_DOWN) -> _ChunkedLoopOptimizerCall:
        """Create an optimizer specification with an explicit split mode.

        Args:
            split: Split mode for cross-core data transfer (default: SplitMode.UP_DOWN)

        Returns:
            Optimizer call with the given split mode
        """
        return _ChunkedLoopOptimizerCall(split=split)

    def __repr__(self) -> str:
        return "chunked_loop_optimizer"


chunked_loop_optimizer: _ChunkedLoopOptimizer = _ChunkedLoopOptimizer()
"""Sentinel for optimization=pl.chunked_loop_optimizer in pl.at().

Use with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer)
to request compiler-driven chunked loop outlining (replaces pl.auto_incore()).
Can also be called with a split mode:
pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.UP_DOWN))
"""

# Range argument type: int literal or Scalar variable
RangeArg = Union[int, "Scalar"]

# Condition argument type: bool literal or Scalar variable
CondArg = Union[bool, "Scalar"]

ExprType = TypeVar("ExprType", int, float, "Scalar", "Tensor", "Tile")


T = TypeVar("T")
W = TypeVar("W")

# TypeVars for overloads (int/float included so yield_(1) is valid in DSL)
T1 = TypeVar("T1", int, float, "Scalar", "Tensor", "Tile")
T2 = TypeVar("T2", int, float, "Scalar", "Tensor", "Tile")
T3 = TypeVar("T3", int, float, "Scalar", "Tensor", "Tile")
T4 = TypeVar("T4", int, float, "Scalar", "Tensor", "Tile")
T5 = TypeVar("T5", int, float, "Scalar", "Tensor", "Tile")


class RangeIterator(Generic[T]):
    """Iterator for pl.range() that supports tuple unpacking."""

    def __init__(
        self,
        stop: RangeArg,
        start: RangeArg = 0,
        step: RangeArg = 1,
        init_values: tuple[Any, ...] | None = None,
        chunk: int | None = None,
        chunk_policy: str = "guarded",
        unroll_factor: int | None = None,
    ):
        """Initialize range iterator.

        Args:
            stop: Stop value (int or Scalar)
            start: Start value (default 0, int or Scalar)
            step: Step value (default 1, int or Scalar)
            init_values: Initial values for iter_args
            chunk: Chunk size for loop chunking (None = no chunking)
            chunk_policy: Chunk distribution policy (default: "guarded")
            unroll_factor: Partial unroll factor for tile-level ping-pong (None = no unroll)
        """
        self.start = start
        self.stop = stop
        self.step = step
        self.init_values = init_values or ()
        self.chunk = chunk
        self.chunk_policy = chunk_policy
        self.unroll_factor = unroll_factor
        self.current = start

    def __iter__(self) -> RangeIterator[T]:
        """Return iterator."""
        return self

    @overload
    def __next__(self: RangeIterator[Scalar]) -> Scalar: ...

    @overload
    def __next__(
        self: RangeIterator[tuple[Scalar, tuple[T1]]],
    ) -> tuple[Scalar, tuple[T1]]: ...

    @overload
    def __next__(
        self: RangeIterator[tuple[Scalar, tuple[T1, T2]]],
    ) -> tuple[Scalar, tuple[T1, T2]]: ...

    @overload
    def __next__(
        self: RangeIterator[tuple[Scalar, tuple[T1, T2, T3]]],
    ) -> tuple[Scalar, tuple[T1, T2, T3]]: ...

    @overload
    def __next__(
        self: RangeIterator[tuple[Scalar, tuple[T1, T2, T3, T4]]],
    ) -> tuple[Scalar, tuple[T1, T2, T3, T4]]: ...

    @overload
    def __next__(
        self: RangeIterator[tuple[Scalar, tuple[T1, T2, T3, T4, T5]]],
    ) -> tuple[Scalar, tuple[T1, T2, T3, T4, T5]]: ...

    def __next__(self) -> Scalar | tuple[Scalar, tuple[Any, ...]]:
        """Get next iteration value.

        Returns:
            If no init_values: just the loop variable (Scalar)
            If init_values provided: Tuple of (loop_var, (iter_arg_values...))
        """
        if self.current >= self.stop:
            raise StopIteration

        value = self.current
        self.current += self.step

        # Return just the value if no init_values, otherwise return (value, iter_args_tuple)
        if not self.init_values:
            return cast("Scalar", value)
        return cast(tuple["Scalar", tuple[Any, ...]], (value, self.init_values))


def _make_range_iterator(
    *args: RangeArg,
    init_values: tuple[Any, ...] | None = None,
    chunk: int | None = None,
    chunk_policy: str = "guarded",
    unroll_factor: int | None = None,
    func_name: str = "range",
) -> RangeIterator[Scalar] | RangeIterator[tuple[Scalar, tuple[Any, ...]]]:
    """Shared implementation for range(), parallel(), and unroll()."""
    if chunk is not None and (not isinstance(chunk, int) or isinstance(chunk, bool) or chunk <= 0):
        raise ValueError(f"{func_name}() chunk must be a positive integer, got {chunk!r}")
    if unroll_factor is not None:
        if not isinstance(unroll_factor, int) or isinstance(unroll_factor, bool) or unroll_factor < 1:
            raise ValueError(f"{func_name}() unroll factor must be a positive integer, got {unroll_factor!r}")
        if chunk is not None:
            raise ValueError(f"{func_name}() unroll= and chunk= are mutually exclusive")
    kwargs = {
        "init_values": init_values,
        "chunk": chunk,
        "chunk_policy": chunk_policy,
        "unroll_factor": unroll_factor,
    }
    if len(args) == 1:
        return RangeIterator(args[0], **kwargs)
    elif len(args) == 2:
        return RangeIterator(args[1], args[0], **kwargs)
    elif len(args) == 3:
        return RangeIterator(args[1], args[0], args[2], **kwargs)
    else:
        raise ValueError(f"{func_name}() takes 1 to 3 positional arguments")


@overload
def range(
    *args: RangeArg,
    init_values: None = None,
    chunk: int | None = None,
    chunk_policy: str = "guarded",
    unroll: int | None = None,
) -> RangeIterator[Scalar]: ...


@overload
def range(
    *args: RangeArg,
    init_values: tuple[T1],
    chunk: int | None = None,
    chunk_policy: str = "guarded",
    unroll: int | None = None,
) -> RangeIterator[tuple[Scalar, tuple[T1]]]: ...


@overload
def range(
    *args: RangeArg,
    init_values: tuple[T1, T2],
    chunk: int | None = None,
    chunk_policy: str = "guarded",
    unroll: int | None = None,
) -> RangeIterator[tuple[Scalar, tuple[T1, T2]]]: ...


@overload
def range(
    *args: RangeArg,
    init_values: tuple[T1, T2, T3],
    chunk: int | None = None,
    chunk_policy: str = "guarded",
    unroll: int | None = None,
) -> RangeIterator[tuple[Scalar, tuple[T1, T2, T3]]]: ...


@overload
def range(
    *args: RangeArg,
    init_values: tuple[T1, T2, T3, T4],
    chunk: int | None = None,
    chunk_policy: str = "guarded",
    unroll: int | None = None,
) -> RangeIterator[tuple[Scalar, tuple[T1, T2, T3, T4]]]: ...


@overload
def range(
    *args: RangeArg,
    init_values: tuple[T1, T2, T3, T4, T5],
    chunk: int | None = None,
    chunk_policy: str = "guarded",
    unroll: int | None = None,
) -> RangeIterator[tuple[Scalar, tuple[T1, T2, T3, T4, T5]]]: ...


def range(
    *args: RangeArg,
    init_values: tuple[Any, ...] | None = None,
    chunk: int | None = None,
    chunk_policy: str = "guarded",
    unroll: int | None = None,
) -> RangeIterator[Scalar] | RangeIterator[tuple[Scalar, tuple[Any, ...]]]:
    """Create a range iterator for for loops.

    Supports several patterns:
        Simple:        for i in pl.range(10):
        Iter args:     for i, (var1, var2) in pl.range(16, init_values=(init1, init2)):
        Chunked:       for i in pl.range(0, 10, chunk=5):
        Partial unroll: for i in pl.range(64, unroll=4):  # 16 outer iters, 4 body copies each

    Args can be int literals or Scalar variables:
        for i in pl.range(n):  # n is pl.Scalar[pl.INT64]
        for i in pl.range(0, n, 1):
        for i in pl.range(n * 2 + 1):

    Args:
        *args: Positional arguments (stop) or (start, stop) or (start, stop, step).
            Each argument can be an int literal or a pl.Scalar value.
        init_values: Initial values for iteration arguments
        chunk: Chunk size for loop chunking (splits loop into nested loops)
        chunk_policy: Chunk distribution policy (default: "guarded")
        unroll: Partial unroll factor for tile-level ping-pong buffering. Replicates
            the loop body ``unroll`` times per outer iteration; the outer loop steps
            by ``step * unroll``. Mutually exclusive with ``chunk``. Lowered by the
            ``PartialUnrollTileLoops`` pass at the tile level.

    Returns:
        If no init_values: RangeIterator yielding loop variable (Scalar)
        If init_values: RangeIterator yielding (loop_var, (iter_args...))
    """
    return _make_range_iterator(
        *args,
        init_values=init_values,
        chunk=chunk,
        chunk_policy=chunk_policy,
        unroll_factor=unroll,
        func_name="range",
    )


@overload
def parallel(
    *args: RangeArg, init_values: None = None, chunk: int | None = None, chunk_policy: str = "guarded"
) -> RangeIterator[Scalar]: ...


@overload
def parallel(
    *args: RangeArg, init_values: tuple[T1], chunk: int | None = None, chunk_policy: str = "guarded"
) -> RangeIterator[tuple[Scalar, tuple[T1]]]: ...


@overload
def parallel(
    *args: RangeArg, init_values: tuple[T1, T2], chunk: int | None = None, chunk_policy: str = "guarded"
) -> RangeIterator[tuple[Scalar, tuple[T1, T2]]]: ...


@overload
def parallel(
    *args: RangeArg,
    init_values: tuple[T1, T2, T3],
    chunk: int | None = None,
    chunk_policy: str = "guarded",
) -> RangeIterator[tuple[Scalar, tuple[T1, T2, T3]]]: ...


@overload
def parallel(
    *args: RangeArg,
    init_values: tuple[T1, T2, T3, T4],
    chunk: int | None = None,
    chunk_policy: str = "guarded",
) -> RangeIterator[tuple[Scalar, tuple[T1, T2, T3, T4]]]: ...


@overload
def parallel(
    *args: RangeArg,
    init_values: tuple[T1, T2, T3, T4, T5],
    chunk: int | None = None,
    chunk_policy: str = "guarded",
) -> RangeIterator[tuple[Scalar, tuple[T1, T2, T3, T4, T5]]]: ...


def parallel(
    *args: RangeArg,
    init_values: tuple[Any, ...] | None = None,
    chunk: int | None = None,
    chunk_policy: str = "guarded",
) -> RangeIterator[Scalar] | RangeIterator[tuple[Scalar, tuple[Any, ...]]]:
    """Create a parallel range iterator for parallel for loops.

    Behaves identically to range() at runtime. The distinction is used by the
    parser to emit ForKind.Parallel instead of ForKind.Sequential.

    Args:
        *args: Positional arguments (stop) or (start, stop) or (start, stop, step).
            Each argument can be an int literal or a pl.Scalar value.
        init_values: Initial values for iteration arguments
        chunk: Chunk size for loop chunking
        chunk_policy: Chunk distribution policy (default: "guarded")

    Returns:
        If no init_values: RangeIterator yielding loop variable (Scalar)
        If init_values: RangeIterator yielding (loop_var, (iter_args...))
    """
    return _make_range_iterator(
        *args, init_values=init_values, chunk=chunk, chunk_policy=chunk_policy, func_name="parallel"
    )


def unroll(
    *args: RangeArg,
    chunk: int | None = None,
    chunk_policy: str = "guarded",
) -> RangeIterator[Scalar]:
    """Create an unroll range iterator for compile-time loop unrolling.

    Behaves identically to range() at runtime. The distinction is used by the
    parser to emit ForKind.Unroll instead of ForKind.Sequential.

    Unrolled loops do not support init_values (loop-carried state).

    Args:
        *args: Positional arguments (stop) or (start, stop) or (start, stop, step).
            Each argument must be an int literal (compile-time constant).
        chunk: Chunk size for loop chunking
        chunk_policy: Chunk distribution policy (default: "guarded")

    Returns:
        RangeIterator yielding loop variable (Scalar)

    Examples:
        >>> for i in pl.unroll(4):
        ...     x = pl.add(x, 1.0)
        >>> for i in pl.unroll(0, 6, 2):
        ...     x = pl.add(x, i)
    """
    return cast(
        RangeIterator["Scalar"],
        _make_range_iterator(*args, chunk=chunk, chunk_policy=chunk_policy, func_name="unroll"),
    )


class WhileIterator(Generic[W]):
    """Iterator for pl.while_() that supports tuple unpacking for iter_args."""

    def __init__(self, *, init_values: tuple[Any, ...] | None = None):
        """Initialize while iterator.

        Args:
            init_values: Initial values for iter_args (required for while loops)
        """
        if init_values is None:
            raise ValueError("while_() requires init_values to be specified")
        self.init_values = init_values
        self._exhausted = False

    def __iter__(self) -> WhileIterator[W]:
        """Return iterator."""
        return self

    @overload
    def __next__(self: WhileIterator[tuple[T1]]) -> tuple[T1]: ...

    @overload
    def __next__(self: WhileIterator[tuple[T1, T2]]) -> tuple[T1, T2]: ...

    @overload
    def __next__(self: WhileIterator[tuple[T1, T2, T3]]) -> tuple[T1, T2, T3]: ...

    @overload
    def __next__(
        self: WhileIterator[tuple[T1, T2, T3, T4]],
    ) -> tuple[T1, T2, T3, T4]: ...

    @overload
    def __next__(
        self: WhileIterator[tuple[T1, T2, T3, T4, T5]],
    ) -> tuple[T1, T2, T3, T4, T5]: ...

    @overload
    def __next__(self: WhileIterator[tuple[Any, ...]]) -> tuple[Any, ...]: ...

    def __next__(self) -> tuple[Any, ...]:
        """Get next iteration value.

        Returns:
            Tuple of iter_arg values
        """
        if self._exhausted:
            raise StopIteration

        # Only iterate once - the parser will handle the while loop
        self._exhausted = True
        return self.init_values


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


def while_(*, init_values: tuple[ExprType, ...] | None = None) -> WhileIterator[tuple[ExprType, ...]]:
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


def yield_(*values: Any) -> Any | tuple[Any, ...]:
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


def const(value: int | float, dtype: Any) -> int | float:
    """Create a typed constant with an explicit dtype.

    Used by the printer to preserve non-default constant dtypes in round-trip.
    The parser intercepts pl.const() calls and creates ConstInt/ConstFloat
    with the specified dtype.

    Args:
        value: Numeric value (int or float)
        dtype: DataType for the constant

    Returns:
        Parser builds ``ConstInt``/``ConstFloat`` IR; the runtime stub returns
        the numeric value unchanged (type checking sees ``int``/``float``).
    """
    return value


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


def static_print(*args: Any) -> None:
    """Print compile-time information about IR objects.

    At parse time, prints type/value info to stdout. At runtime, no-op.

    Args:
        *args: Values to print (variables, expressions, string labels)
    """


def static_assert(condition: Any, msg: str = "") -> None:
    """Assert a condition at compile time (parse time).

    At parse time, evaluates ``condition``. If false, raises ``ParserError``.
    At runtime, this is a no-op (all semantics are handled by the parser).

    Notes:
        * This is a **statement-only** construct. It must be used as a
          standalone statement, not as part of an expression.
        * The ``msg`` argument must be a **string literal** at the call site.
          Passing a variable or expression for ``msg`` will raise
          ``ParserSyntaxError``.
        * The check is evaluated at parse time only; it does not run at
          execution time.

    Args:
        condition: Condition to check (must be compile-time evaluable)
        msg: Optional error message as a string literal
    """


class IncoreContext:
    """Context manager for InCore scope.

    This is returned by pl.incore() and used with the 'with' statement.
    The parser recognizes this pattern and creates a ScopeStmt(InCore).
    """

    def __init__(self, split: SplitMode = SplitMode.NONE, name_hint: str = "") -> None:
        self.split = split
        self.name_hint = name_hint

    def __enter__(self) -> None:
        """Enter the InCore scope context."""
        pass

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the InCore scope context."""
        pass


class AutoIncoreContext:
    """Context manager for AutoInCore scope.

    This is returned by pl.auto_incore() and used with the 'with' statement.
    The parser recognizes this pattern and creates a ScopeStmt(AutoInCore).
    """

    def __init__(self, split: SplitMode = SplitMode.NONE, name_hint: str = "") -> None:
        self.split = split
        self.name_hint = name_hint

    def __enter__(self) -> None:
        """Enter the AutoInCore scope context."""
        pass

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the AutoInCore scope context."""
        pass


def auto_incore(split: SplitMode = SplitMode.UP_DOWN, *, name_hint: str = "") -> AutoIncoreContext:
    """Mark a region of code for automatic incore chunking.

    This function returns a context manager that should be used with the 'with' statement.
    The parser recognizes this pattern and creates a ScopeStmt with ScopeKind.AutoInCore.

    Args:
        split: Split mode for cross-core data transfer (default: SplitMode.UP_DOWN)

    Returns:
        Context manager for AutoInCore scope

    Examples:
        >>> with pl.auto_incore():
        ...     for i in pl.parallel(0, 8, 1, chunk=4):
        ...         x = pl.add(x, x)
        >>> with pl.auto_incore(split=pl.SplitMode.UP_DOWN):
        ...     for i in pl.parallel(0, 8, 1, chunk=4):
        ...         x = pl.add(x, x)
    """
    if split == SplitMode.NONE:
        raise ValueError("SplitMode.NONE is not supported by pto-isa now")
    return AutoIncoreContext(split=split, name_hint=name_hint)


def incore(split: SplitMode = SplitMode.NONE, *, name_hint: str = "") -> IncoreContext:
    """Mark a region of code as belonging to the InCore execution context.

    This function returns a context manager that should be used with the 'with' statement.
    The parser recognizes this pattern and creates a ScopeStmt with ScopeKind.InCore.

    Args:
        split: Split mode for cross-core data transfer (default: SplitMode.NONE).
            When set, the outlined InCore function will use the specified split
            mode for data transfer between AIC and AIV cores.
        name_hint: Optional name hint for the outlined function (must be a valid identifier)

    Returns:
        Context manager for InCore scope

    Examples:
        >>> with pl.incore():
        ...     y = pl.ops.add(x, x)
        ...     z = pl.ops.mul(y, y)
        >>> with pl.incore(split=pl.SplitMode.UP_DOWN):
        ...     y = pl.ops.add(x, x)
    """
    return IncoreContext(split=split, name_hint=name_hint)


class ClusterContext:
    """Context manager for Cluster scope.

    This is returned by pl.cluster() and used with the 'with' statement.
    The parser recognizes this pattern and creates a ScopeStmt(Cluster).
    """

    def __init__(self, *, name_hint: str = "") -> None:
        self.name_hint = name_hint

    def __enter__(self) -> None:
        """Enter the Cluster scope context."""
        pass

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the Cluster scope context."""
        pass


def cluster(*, name_hint: str = "") -> ClusterContext:
    """Mark a region of code as belonging to a Cluster execution context.

    A cluster groups co-scheduled AIC (Cube) and AIV (Vector) kernels that
    share the same physical cluster resources. The OutlineClusterScopes pass
    extracts Cluster scopes into separate Group-typed functions.

    Args:
        name_hint: Optional name hint for the outlined function.

    Returns:
        Context manager for Cluster scope

    Examples:
        >>> with pl.cluster():
        ...     with pl.incore():
        ...         y = pl.add(x, x)
    """
    return ClusterContext(name_hint=name_hint)


class SpmdContext:
    """Context manager for SPMD dispatch scope.

    The parser recognizes this pattern and creates a ScopeStmt(Spmd).
    """

    def __init__(
        self,
        core_num: int,
        sync_start: bool = False,
        name_hint: str = "",
    ) -> None:
        self.core_num = core_num
        self.sync_start = sync_start
        self.name_hint = name_hint

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


def spmd(
    *,
    core_num: int,
    sync_start: bool = False,
    name_hint: str = "",
) -> SpmdContext:
    """Dispatch a kernel with SPMD (Single Program Multiple Data) multi-block execution.

    The body must contain exactly one function call. Can be used standalone
    (creates an implicit cluster) or nested inside pl.cluster().

    Args:
        core_num: Number of blocks for SPMD dispatch. Must be a positive integer.
        sync_start: If True, all blocks start execution simultaneously (default: False).
        name_hint: Optional name hint for the outlined function.

    Returns:
        Context manager for SPMD scope

    Examples:
        >>> # Standalone SPMD (pure AIV kernel)
        >>> with pl.spmd(core_num=4):
        ...     out = self.kernel(a, b, out)
        >>>
        >>> # SPMD inside cluster (mixed kernel)
        >>> with pl.cluster():
        ...     with pl.spmd(core_num=4, sync_start=True):
        ...         out = self.kernel(a, b, out)
    """
    if not isinstance(core_num, int) or isinstance(core_num, bool) or core_num <= 0:
        raise ValueError(f"core_num must be a positive integer, got {core_num!r}")
    return SpmdContext(core_num=core_num, sync_start=sync_start, name_hint=name_hint)


class AtContext:
    """Context manager for hierarchy-level scope.

    Returned by pl.at(level=..., role=..., optimizations=[...]) and used with the
    'with' statement. The parser recognizes this pattern and creates:
    - ScopeStmt(InCore) when level=CORE_GROUP (no optimizations)
    - ScopeStmt(InCore, split=...) when level=CORE_GROUP with optimizations=[pl.split(...)]
    - ScopeStmt(AutoInCore) when level=CORE_GROUP with optimizations=[pl.auto_chunk]
    - ScopeStmt(Hierarchy) for all other levels
    """

    def __init__(
        self,
        level: ir.Level,
        role: ir.Role | None = None,
        *,
        optimizations: list[Optimization] | None = None,
        # Deprecated kwargs (kept for back-compat; emit DeprecationWarning at parse time):
        optimization: _ChunkedLoopOptimizer | _ChunkedLoopOptimizerCall | None = None,
        split: SplitMode | None = None,
        name_hint: str = "",
    ) -> None:
        self.level = level
        self.role = role
        self.optimizations = optimizations
        self.optimization = optimization
        self.split = split
        self.name_hint = name_hint

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


def at(
    level: ir.Level,
    role: ir.Role | None = None,
    *,
    optimizations: list[Optimization] | None = None,
    # Deprecated kwargs (kept for back-compat; emit DeprecationWarning at parse time):
    optimization: _ChunkedLoopOptimizer | _ChunkedLoopOptimizerCall | None = None,
    split: SplitMode | None = None,
    name_hint: str = "",
) -> AtContext:
    """Mark a region of code for execution at a specific hierarchy level.

    With ``level=pl.Level.CORE_GROUP``, the ``optimizations=`` list controls
    the resulting scope kind:

    - no entries → ``ScopeStmt(InCore)``
    - ``pl.split(mode)`` → ``ScopeStmt(InCore, split=mode)``
    - ``pl.auto_chunk`` → ``ScopeStmt(AutoInCore)``
    - both entries → ``ScopeStmt(AutoInCore, split=mode)``

    For all other levels, this creates a Hierarchy scope.

    Args:
        level: Target hierarchy level (e.g. pl.Level.HOST, pl.Level.CORE_GROUP).
        role: Function role (Orchestrator or Worker). Default: None.
        optimizations: Optional list literal of optimization entries. Each
            entry must be one of ``pl.auto_chunk`` or ``pl.split(mode)`` —
            written inline at the call site, since the DSL parser inspects
            the AST and does not accept dynamically built variables here.
            Entries are independent and may be combined.
        optimization: **Deprecated.** Use ``optimizations=[pl.auto_chunk]`` (or
            ``optimizations=[pl.auto_chunk, pl.split(mode)]``) instead.
        split: **Deprecated.** Use ``optimizations=[pl.split(mode)]`` instead.
        name_hint: Optional name hint for the outlined function (must be a
            valid identifier).

    Returns:
        Context manager for the appropriate scope.

    Examples:
        >>> # InCore scope (replaces pl.incore()):
        >>> with pl.at(level=pl.Level.CORE_GROUP):
        ...     y = pl.ops.add(x, x)

        >>> # InCore scope with split hint:
        >>> with pl.at(level=pl.Level.CORE_GROUP,
        ...            optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
        ...     y = pl.ops.add(x, x)

        >>> # AutoInCore scope (replaces pl.auto_incore()):
        >>> with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
        ...     for i in pl.parallel(0, 8, 1, chunk=4):
        ...         x = pl.add(x, x)

        >>> # AutoInCore + split hint (combined, independent entries):
        >>> with pl.at(level=pl.Level.CORE_GROUP,
        ...            optimizations=[pl.auto_chunk, pl.split(pl.SplitMode.UP_DOWN)]):
        ...     for i in pl.parallel(0, 8, 1, chunk=4):
        ...         x = pl.add(x, x)

        >>> # Hierarchy scope (unchanged behavior):
        >>> with pl.at(level=pl.Level.HOST, role=pl.Role.Worker):
        ...     y = pl.add(x, x)
    """
    return AtContext(
        level,
        role,
        optimizations=optimizations,
        optimization=optimization,
        split=split,
        name_hint=name_hint,
    )


__all__ = [
    "const",
    "range",
    "parallel",
    "unroll",
    "while_",
    "yield_",
    "cond",
    "static_print",
    "static_assert",
    "incore",
    "auto_incore",
    "at",
    "cluster",
    "spmd",
    "chunked_loop_optimizer",
    "RangeIterator",
    "WhileIterator",
    "IncoreContext",
    "AutoIncoreContext",
    "ClusterContext",
    "SpmdContext",
    "AtContext",
    "RangeArg",
    "CondArg",
]
