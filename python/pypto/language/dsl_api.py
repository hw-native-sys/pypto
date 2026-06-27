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

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Union, cast, overload

if TYPE_CHECKING:
    from pypto.language.typing import Array, Scalar, Tensor, Tile
    from pypto.pypto_core import ir

from pypto.pypto_core import ir as _ir

from .optimizations import Optimization

# Range argument type: int literal or Scalar variable
RangeArg = Union[int, "Scalar"]

# Condition argument type: bool literal or Scalar variable
CondArg = Union[bool, "Scalar"]

ExprType = TypeVar("ExprType", int, float, "Scalar", "Tensor", "Tile", "Array")


T = TypeVar("T")
W = TypeVar("W")

# TypeVars for overloads (int/float included so yield_(1) is valid in DSL)
T1 = TypeVar("T1", int, float, "Scalar", "Tensor", "Tile", "Array")
T2 = TypeVar("T2", int, float, "Scalar", "Tensor", "Tile", "Array")
T3 = TypeVar("T3", int, float, "Scalar", "Tensor", "Tile", "Array")
T4 = TypeVar("T4", int, float, "Scalar", "Tensor", "Tile", "Array")
T5 = TypeVar("T5", int, float, "Scalar", "Tensor", "Tile", "Array")


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
        pipeline_stages: int | None = None,
    ):
        """Initialize range iterator.

        Args:
            stop: Stop value (int or Scalar)
            start: Start value (default 0, int or Scalar)
            step: Step value (default 1, int or Scalar)
            init_values: Initial values for iter_args
            chunk: Chunk size for loop chunking (None = no chunking)
            chunk_policy: Chunk distribution policy (default: "guarded")
            pipeline_stages: Software-pipelining depth — replicates the body this
                many times per outer iteration (None = no pipelining). Only set by
                pl.pipeline(); validated by the parser.
        """
        self.start = start
        self.stop = stop
        self.step = step
        self.init_values = init_values or ()
        self.chunk = chunk
        self.chunk_policy = chunk_policy
        self.pipeline_stages = pipeline_stages
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
    pipeline_stages: int | None = None,
    func_name: str = "range",
) -> RangeIterator[Scalar] | RangeIterator[tuple[Scalar, tuple[Any, ...]]]:
    """Shared implementation for range(), parallel(), unroll(), and pipeline()."""
    if chunk is not None and (not isinstance(chunk, int) or isinstance(chunk, bool) or chunk <= 0):
        raise ValueError(f"{func_name}() chunk must be a positive integer, got {chunk!r}")
    if pipeline_stages is not None:
        if not isinstance(pipeline_stages, int) or isinstance(pipeline_stages, bool) or pipeline_stages < 1:
            raise ValueError(f"{func_name}() stage must be a positive integer, got {pipeline_stages!r}")
        if chunk is not None:
            raise ValueError(f"{func_name}() stage= and chunk= are mutually exclusive")
    kwargs = {
        "init_values": init_values,
        "chunk": chunk,
        "chunk_policy": chunk_policy,
        "pipeline_stages": pipeline_stages,
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
) -> RangeIterator[Scalar]: ...


@overload
def range(
    *args: RangeArg,
    init_values: tuple[T1],
    chunk: int | None = None,
    chunk_policy: str = "guarded",
) -> RangeIterator[tuple[Scalar, tuple[T1]]]: ...


@overload
def range(
    *args: RangeArg,
    init_values: tuple[T1, T2],
    chunk: int | None = None,
    chunk_policy: str = "guarded",
) -> RangeIterator[tuple[Scalar, tuple[T1, T2]]]: ...


@overload
def range(
    *args: RangeArg,
    init_values: tuple[T1, T2, T3],
    chunk: int | None = None,
    chunk_policy: str = "guarded",
) -> RangeIterator[tuple[Scalar, tuple[T1, T2, T3]]]: ...


@overload
def range(
    *args: RangeArg,
    init_values: tuple[T1, T2, T3, T4],
    chunk: int | None = None,
    chunk_policy: str = "guarded",
) -> RangeIterator[tuple[Scalar, tuple[T1, T2, T3, T4]]]: ...


@overload
def range(
    *args: RangeArg,
    init_values: tuple[T1, T2, T3, T4, T5],
    chunk: int | None = None,
    chunk_policy: str = "guarded",
) -> RangeIterator[tuple[Scalar, tuple[T1, T2, T3, T4, T5]]]: ...


def range(
    *args: RangeArg,
    init_values: tuple[Any, ...] | None = None,
    chunk: int | None = None,
    chunk_policy: str = "guarded",
) -> RangeIterator[Scalar] | RangeIterator[tuple[Scalar, tuple[Any, ...]]]:
    """Create a range iterator for for loops.

    Supports several patterns:
        Simple:        for i in pl.range(10):
        Iter args:     for i, (var1, var2) in pl.range(16, init_values=(init1, init2)):
        Chunked:       for i in pl.range(0, 10, chunk=5):

    For software pipelining (body replication for ping-pong buffering), use
    ``pl.pipeline(N, stage=F)`` instead — it is a sibling loop iterator.

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

    Returns:
        If no init_values: RangeIterator yielding loop variable (Scalar)
        If init_values: RangeIterator yielding (loop_var, (iter_args...))
    """
    return _make_range_iterator(
        *args,
        init_values=init_values,
        chunk=chunk,
        chunk_policy=chunk_policy,
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


@overload
def pipeline(*args: RangeArg, stage: int, init_values: None = None) -> RangeIterator[Scalar]: ...


@overload
def pipeline(
    *args: RangeArg, stage: int, init_values: tuple[T1]
) -> RangeIterator[tuple[Scalar, tuple[T1]]]: ...


@overload
def pipeline(
    *args: RangeArg, stage: int, init_values: tuple[T1, T2]
) -> RangeIterator[tuple[Scalar, tuple[T1, T2]]]: ...


@overload
def pipeline(
    *args: RangeArg, stage: int, init_values: tuple[T1, T2, T3]
) -> RangeIterator[tuple[Scalar, tuple[T1, T2, T3]]]: ...


@overload
def pipeline(
    *args: RangeArg, stage: int, init_values: tuple[T1, T2, T3, T4]
) -> RangeIterator[tuple[Scalar, tuple[T1, T2, T3, T4]]]: ...


@overload
def pipeline(
    *args: RangeArg, stage: int, init_values: tuple[T1, T2, T3, T4, T5]
) -> RangeIterator[tuple[Scalar, tuple[T1, T2, T3, T4, T5]]]: ...


def pipeline(
    *args: RangeArg,
    stage: int,
    init_values: tuple[Any, ...] | None = None,
) -> RangeIterator[Scalar] | RangeIterator[tuple[Scalar, tuple[Any, ...]]]:
    """Create a software-pipelined loop iterator.

    Replicates the loop body ``stage`` times per outer iteration to enable
    ping-pong buffering. The outer loop advances in strides of ``stage * step``;
    a tail dispatch covers the remainder when the trip count is not divisible
    by ``stage``. Lowered by the ``LowerPipelineLoops`` pass at the tile level.

    Positional args match ``pl.range``: (stop) / (start, stop) / (start, stop, step).
    The ``stage`` kwarg is required and must be a positive integer.

    Args:
        *args: 1-3 positional args — same shape as ``pl.range()``.
        stage: Pipeline depth (positive integer, typically 2-4).
        init_values: Loop-carried state, same semantics as ``pl.range()``.

    Examples:
        >>> for i in pl.pipeline(64, stage=4):
        ...     tile = pl.tile.load(...)
        ...     pl.tile.store(...)
    """
    return _make_range_iterator(
        *args,
        init_values=init_values,
        pipeline_stages=stage,
        func_name="pipeline",
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


def const(value: int | float, dtype: Any) -> Scalar:
    """Create a typed constant with an explicit dtype.

    Used by the printer to preserve non-default constant dtypes in round-trip.
    The parser intercepts pl.const() calls and creates ConstInt/ConstFloat
    with the specified dtype.

    Statically typed ``-> Scalar`` to mirror its IR semantics: ``pl.const``
    builds a ``ConstInt``/``ConstFloat`` expression, so it can be returned
    from a ``-> pl.Scalar`` function and combined with other scalars. At
    runtime the stub returns the numeric value unchanged (``cast`` is a no-op).

    Args:
        value: Numeric value (int or float)
        dtype: DataType for the constant

    Returns:
        Parser builds ``ConstInt``/``ConstFloat`` IR; the runtime stub returns
        the numeric value unchanged (type checking sees a ``Scalar``).
    """
    return cast("Scalar", value)


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
        ...     with pl.at(level=pl.Level.CORE_GROUP):
        ...         y = pl.add(x, x)
    """
    return ClusterContext(name_hint=name_hint)


class SpmdContext:
    """Context manager / loop iterator for SPMD dispatch scope.

    The parser recognizes ``with pl.spmd(...):`` (builds a ``ScopeStmt(Spmd)``
    whose body must be a single function call), ``with pl.spmd(...) as tid:``
    (captures the grid dispatch's producer ``Scalar[TASK_ID]`` and accepts an
    inline multi-statement body), and ``for i in pl.spmd(...):`` (auto-outlines
    the loop body into an InCore function with ``i`` bound to
    ``pl.tile.get_block_idx()``).
    """

    def __init__(
        self,
        core_num: int | _ir.Expr,
        sync_start: bool = False,
        name_hint: str = "",
        optimizations: list[Optimization] | None = None,
        deps: list[Any] | None = None,
        allow_early_resolve: bool = False,
    ) -> None:
        self.core_num = core_num
        self.sync_start = sync_start
        self.name_hint = name_hint
        self.optimizations = optimizations
        self.deps = deps
        self.allow_early_resolve = allow_early_resolve

    def __enter__(self) -> Any:
        # The parser intercepts the ``with pl.spmd(...) [as tid]:`` pattern and
        # binds ``tid`` (when present) to the dispatch's producer TaskId. This
        # runtime return value is not consumed in practice — the ``@pl.program``
        # decorator parses the function source rather than executing it — but
        # returning ``self`` keeps ``as`` syntactically legal (and ``tid``
        # non-``None`` under static checking) when a script is executed directly
        # (e.g. for linting), matching :meth:`AtContext.__enter__`.
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def __iter__(self) -> Iterator[Any]:
        # Lets `for i in pl.spmd(...)` type-check and parse at the Python
        # level. The @pl.program / @pl.function decorators intercept the AST,
        # so this method should never actually run — if it does, the caller
        # is using pl.spmd() outside the DSL interception path. Raise
        # loudly rather than silently yielding zero iterations.
        raise RuntimeError(
            "pl.spmd(...) loop form is only valid inside a @pl.program / "
            "@pl.function body; the parser replaces the for-loop with an "
            "SpmdScopeStmt. If you're seeing this, the surrounding function "
            "was not decorated."
        )


def spmd(
    core_num: RangeArg,
    *,
    sync_start: bool = False,
    name_hint: str = "",
    optimizations: list[Optimization] | None = None,
    deps: list[Any] | None = None,
    allow_early_resolve: bool = False,
) -> SpmdContext:
    """Dispatch a kernel with SPMD (Single Program Multiple Data) multi-block execution.

    The first argument is the number of blocks and is positional — mirroring
    ``range(n)``. Loop start is fixed at 0 and step at 1; each block gets an
    index ``i`` in ``[0, core_num)``.

    Usage forms:

    1. ``with pl.spmd(n):`` — body must be a single call to a pre-defined
       InCore kernel. Can stand alone (implicit cluster) or nest inside
       ``pl.cluster()``.

    2. ``for i in pl.spmd(n):`` — loop-style. The iteration variable binds
       the per-block index (equivalent to ``pl.tile.get_block_idx()``); the
       body is auto-outlined into a synthetic InCore function, so inline
       tile/tensor ops work without a separate ``@pl.function(type=InCore)``
       declaration.

    3. ``with pl.spmd(n, deps=[...]) as tid:`` — captures the grid dispatch's
       producer ``Scalar[TASK_ID]`` in ``tid`` (mirroring ``with pl.at(...) as
       tid:``), usable as a ``deps=`` edge on later tasks, stored into a
       ``pl.array.create(N, pl.TASK_ID)``, or crossing into ``pl.manual_scope``.
       Unlike form 1, this form accepts an inline multi-statement body (like the
       loop form); read the per-block index inside via ``pl.tile.get_block_idx()``.

    Optional ``optimizations=[pl.split(mode)]`` applies to the inner InCore scope
    (auto-generated for the for-form and the ``as tid`` form, wrapped around the
    call for the plain with-form). ``pl.auto_chunk`` is not supported on
    ``pl.spmd`` — use ``pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk])``
    inside the loop body for chunked parallel loops.

    Args:
        core_num: Number of blocks for SPMD dispatch. Positional; accepts a
            Python ``int`` or any ``ir.Expr`` of integer type. Closure-captured
            integer constants and closure arithmetic are folded to ``ConstInt``
            by the parser and ``Simplify``; non-foldable expressions flow
            through to codegen unchanged.
        sync_start: If True, all blocks start execution simultaneously (default: False).
        name_hint: Optional name hint for the outlined function.
        optimizations: Optional list literal containing only ``pl.split(mode)``
            entries (the parser inspects the AST).
        deps: Optional explicit producer-edge list (TaskId Vars and/or ``None``
            sentinels), accepted only with the ``as tid`` form. Lowered to the
            outlined Submit's ``manual_dep_edges``; codegen packs it into a
            ``set_dependencies(...)`` invocation (union'd with auto-deps).
        allow_early_resolve: Opt the grid dispatch in as a speculative
            early-dispatch producer (simpler#1065). Same hint as
            ``pl.submit(..., allow_early_resolve=True)`` / ``pl.at(...,
            allow_early_resolve=True)``: the scheduler may pre-stage this task's
            consumers before it completes. Forces the dispatch to lower to an
            ``ir.Submit`` (even without ``as tid``) so the flag rides to codegen,
            where it emits ``Arg::set_allow_early_resolve(true)``. Pure scheduling
            hint. Rejected on a ``pl.cluster()``-nested ``pl.spmd`` (such a scope
            is unwrapped into the Group function and never produces a Submit, so
            the hint would be lost).

    Returns:
        Context manager / loop iterator for the SPMD scope.

    Examples:
        >>> # Single-kernel context-manager form
        >>> with pl.spmd(4):
        ...     out = self.kernel(a, b, out)
        >>>
        >>> # Loop form — body runs per-block with i = tile.get_block_idx()
        >>> for i in pl.spmd(4):
        ...     offset = i * 128
        ...     tile_a = pl.load(a, [offset, 0], [128, 128])
        ...     tile_b = pl.load(b, [offset, 0], [128, 128])
        ...     out = pl.store(pl.add(tile_a, tile_b), [offset, 0], out)
        >>>
        >>> # Capture-form: inline body + producer TaskId for explicit dep wiring
        >>> with pl.spmd(4, name_hint="stage1") as tid_a:
        ...     i = pl.tile.get_block_idx()
        ...     offset = i * 128
        ...     out = pl.store(pl.add(a[offset], b[offset]), [offset, 0], out)
        >>> with pl.spmd(4, name_hint="stage2", deps=[tid_a]) as tid_b:
        ...     i = pl.tile.get_block_idx()
        ...     out2 = pl.store(pl.relu(out[i * 128]), [i * 128, 0], out2)
        >>>
        >>> # With-form with split hint on the inner InCore wrapper
        >>> with pl.spmd(4, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
        ...     out = self.kernel(a, b, out)
        >>>
        >>> # SPMD inside cluster (mixed kernel)
        >>> with pl.cluster():
        ...     with pl.spmd(4, sync_start=True):
        ...         out = self.kernel(a, b, out)
    """
    if isinstance(core_num, bool) or not isinstance(core_num, (int, _ir.Expr)):
        raise ValueError(f"core_num must be a positive integer or ir.Expr, got {core_num!r}")
    if isinstance(core_num, int) and core_num <= 0:
        raise ValueError(f"core_num must be a positive integer, got {core_num!r}")
    return SpmdContext(
        core_num=core_num,
        sync_start=sync_start,
        name_hint=name_hint,
        optimizations=optimizations,
        deps=deps,
        allow_early_resolve=allow_early_resolve,
    )


class SplitAivContext:
    """Loop iterator for the explicit AIV-split scope.

    The parser recognizes ``for aiv_id in pl.split_aiv(2, mode=...):`` and opens
    a single bare ``InCoreScopeStmt`` marking an explicit AIV-split body (it
    carries the requested ``SplitMode`` and a ``("split_aiv", True)`` attr). The
    loop variable is bound to ``pl.tile.get_subblock_idx()`` — the AIV lane /
    sub-core index.
    """

    def __init__(self, n: int, mode: ir.SplitMode) -> None:
        self.n = n
        self.mode = mode

    def __iter__(self) -> Iterator[Any]:
        # Lets `for aiv_id in pl.split_aiv(...)` type-check and parse at the
        # Python level. The @pl.program / @pl.function decorators intercept the
        # AST, so this should never actually run — if it does, the caller is
        # using pl.split_aiv() outside the DSL interception path. Raise loudly
        # rather than silently yielding zero iterations (mirrors SpmdContext).
        raise RuntimeError(
            "pl.split_aiv(...) loop form is only valid inside a @pl.program / "
            "@pl.function body; the parser replaces the for-loop with an explicit "
            "AIV-split InCoreScopeStmt. If you're seeing this, the surrounding "
            "function was not decorated."
        )


def split_aiv(n: int, *, mode: ir.SplitMode) -> SplitAivContext:
    """Open an explicit AIV-split scope as an SPMD-style loop.

    Usage::

        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            ...  # body runs per AIV lane; aiv_id = pl.tile.get_subblock_idx()

    The loop opens exactly ONE bare ``InCore`` scope (unlike ``pl.spmd``, which
    wraps an InCore body in a Spmd scope). The scope carries the requested
    ``SplitMode`` (same mechanism as ``pl.split``) plus a ``("split_aiv", True)``
    attr so later passes can identify the explicit-split body. The loop variable
    binds the AIV lane index (equivalent to ``pl.tile.get_subblock_idx()``).

    Args:
        n: The AIV sub-core count. Positional; hardware-fixed at 2 (the two AIV
            lanes of one AICore). Any other value is rejected by the parser.
        mode: Required split mode (``pl.SplitMode.UP_DOWN`` or
            ``pl.SplitMode.LEFT_RIGHT``). No silent default.

    Returns:
        Loop iterator for the explicit AIV-split scope.

    Examples:
        >>> for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
        ...     offset = aiv_id * 128
        ...     t = pl.load(a, [offset, 0], [128, 128])
        ...     out = pl.store(t, [offset, 0], out)
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError(f"pl.split_aiv(n): n must be the integer 2, got {n!r}")
    if n != 2:
        raise ValueError(f"pl.split_aiv(n): n must be the integer 2 (the two AIV lanes), got {n}")
    if not isinstance(mode, _ir.SplitMode):
        raise ValueError(f"pl.split_aiv(mode=...): mode must be a pl.SplitMode, got {mode!r}")
    return SplitAivContext(n=n, mode=mode)


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
        deps: list[Any] | None = None,
        no_dep_args: list[Any] | None = None,
        dumps: list[Any] | None = None,
        allow_early_resolve: bool = False,
        name_hint: str = "",
        windowize: bool = False,
    ) -> None:
        self.level = level
        self.role = role
        self.optimizations = optimizations
        self.deps = deps
        self.no_dep_args = no_dep_args
        self.dumps = dumps
        self.allow_early_resolve = allow_early_resolve
        self.name_hint = name_hint
        self.windowize = windowize

    def __enter__(self) -> Any:
        # The parser intercepts the ``with pl.at(...) [as tid]:`` pattern and
        # binds ``tid`` (when present) to the outlined kernel Call's producer
        # TaskId. This runtime return value is not consumed in practice — the
        # ``@pl.program`` decorator parses the function source rather than
        # executing it — but returning ``self`` keeps ``as`` syntactically
        # legal in case a script is executed directly (e.g. for linting).
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


def at(
    level: ir.Level,
    role: ir.Role | None = None,
    *,
    optimizations: list[Optimization] | None = None,
    deps: list[Any] | None = None,
    no_dep_args: list[Any] | None = None,
    dumps: list[Any] | None = None,
    allow_early_resolve: bool = False,
    name_hint: str = "",
    windowize: bool = False,
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
        deps: Optional explicit producer-edge list (TaskId Vars and/or
            ``None`` sentinels). Lowered to the resulting Call's
            ``manual_dep_edges`` attr, which codegen packs into a
            ``set_dependencies(...)`` invocation. Operates at the *TaskId*
            level — orthogonal to ``no_dep_args=``, which operates at the
            *arg-slot* level on captured tensors.
        no_dep_args: Optional list literal of outer-scope tensor names
            captured by the scope body. Each entry must be a bare tensor
            name; the parser resolves it to a Var, the outliner translates
            the Var list into positional indices into the synthesised
            Call's args, and ``DeriveCallDirections`` overwrites those
            slots to ``ArgDirection.NoDep``. Equivalent to wrapping the
            same tensor with ``pl.no_dep(t)`` at an explicit kernel call
            site — use this form when the kernel call is synthesised by
            the ``pl.at`` outliner and there is no syntactic call-arg slot
            to wrap. Legal for both read-only captures and captures that
            the scope body mutates via ``pl.assemble`` / ``pl.store`` (the
            outliner classifies the latter as ``InOut`` on the synthesised
            kernel, and ``NoDep`` overrides ``InOut`` just as it overrides
            ``Input``); the user is asserting that sibling fan-outs touch
            disjoint regions of the tensor and therefore do not need
            OverlapMap dep tracking. Note: ``deps=`` takes TaskIds, while
            ``no_dep_args=`` takes tensors — they describe different things.
        dumps: Optional list literal of outer-scope tensor names to mark for
            selective tensor dump on the synthesised kernel dispatch. The
            scope-level selective-dump surface, symmetric with ``deps=``: the
            outliner translates the captured-tensor entries into the dispatch's
            ``dump_vars`` by Var identity. Equivalent to declaring the same
            tensors with ``pl.dump_tag(t)`` before the scope; use ``dumps=``
            when you want the dump targets listed explicitly at the scope.
        allow_early_resolve: Opt the outlined dispatch in as a speculative
            early-dispatch producer (simpler#1065). Same hint as
            ``pl.submit(..., allow_early_resolve=True)``: the scheduler may
            pre-stage this task's consumers before it completes. Forces the
            scope to lower to an ``ir.Submit`` (even without ``as tid``) so the
            flag can ride to codegen, where it emits
            ``Arg::set_allow_early_resolve(true)``. Pure scheduling hint.
        name_hint: Optional name hint for the outlined function (must be a
            valid identifier).
        windowize: Explicitly allow local windowization for the outlined InCore
            kernel. The default is False.

    Returns:
        Context manager for the appropriate scope.

    Examples:
        >>> # InCore scope:
        >>> with pl.at(level=pl.Level.CORE_GROUP):
        ...     y = pl.ops.add(x, x)

        >>> # InCore scope with split hint:
        >>> with pl.at(level=pl.Level.CORE_GROUP,
        ...            optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
        ...     y = pl.ops.add(x, x)

        >>> # AutoInCore scope:
        >>> with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
        ...     for i in pl.parallel(0, 8, 1, chunk=4):
        ...         x = pl.add(x, x)

        >>> # AutoInCore + split hint (combined, independent entries):
        >>> with pl.at(level=pl.Level.CORE_GROUP,
        ...            optimizations=[pl.auto_chunk, pl.split(pl.SplitMode.UP_DOWN)]):
        ...     for i in pl.parallel(0, 8, 1, chunk=4):
        ...         x = pl.add(x, x)

        >>> # Hierarchy scope (unchanged behavior):
        >>> with pl.at(level=pl.Level.HOST, role=pl.Role.SubWorker):
        ...     y = pl.add(x, x)
    """
    return AtContext(
        level,
        role,
        optimizations=optimizations,
        deps=deps,
        no_dep_args=no_dep_args,
        dumps=dumps,
        allow_early_resolve=allow_early_resolve,
        name_hint=name_hint,
        windowize=windowize,
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
    "at",
    "cluster",
    "spmd",
    "split_aiv",
    "RangeIterator",
    "WhileIterator",
    "ClusterContext",
    "SpmdContext",
    "SplitAivContext",
    "AtContext",
    "RangeArg",
    "CondArg",
]
