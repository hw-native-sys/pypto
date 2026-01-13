# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
PyPTO Runtime - Simulated NPU runtime with AICPU and AICORE

This module provides a runtime machine simulator for testing and developing
NPU programs before deploying to actual hardware.

Example:
    >>> import pypto.runtime as rt
    >>>
    >>> # Create a runtime machine
    >>> machine = rt.RuntimeMachine(num_aicore=4)
    >>>
    >>> # Register AICORE tasks
    >>> def my_task(a, b):
    ...     return a + b
    >>> machine.register_task("add", my_task)
    >>>
    >>> # Create a program
    >>> program = rt.RuntimeProgram()
    >>> r0, r1, r2 = (program.reg(i) for i in range(3))
    >>> program.const(r0, 10)
    >>> program.const(r1, 20)
    >>> program.dispatch(r2, "add", [r0, r1])
    >>> program.wait(r2)
    >>> program.halt()
    >>>
    >>> # Execute
    >>> machine.load_and_run_program(program)
"""

from typing import Any, Callable, List, Optional, Union

from ..pypto_core.runtime import RuntimeMachine as _RuntimeMachine
from ..pypto_core.runtime import RuntimeProgram as _RuntimeProgram
from ..pypto_core.runtime import SharedMemory as _SharedMemory
from ..pypto_core.runtime import Value as _Value

# Re-export core classes
RuntimeMachine = _RuntimeMachine
RuntimeProgram = _RuntimeProgram
SharedMemory = _SharedMemory
Value = _Value

__all__ = [
    "RuntimeMachine",
    "RuntimeProgram",
    "SharedMemory",
    "Value",
    "ProgramBuilder",
]


class ProgramBuilder:
    """
    Fluent API for building RuntimePrograms with cleaner syntax.

    Example:
        >>> builder = ProgramBuilder()
        >>> r0, r1, r2 = (builder.reg(i) for i in range(3))
        >>> builder.const(r0, 100) \\
        ...        .const(r1, 200) \\
        ...        .add(r2, r0, r1) \\
        ...        .halt()
        >>> program = builder.build()
    """

    def __init__(self, program: Optional[_RuntimeProgram] = None):
        """Create a program builder."""
        self._program = program if program is not None else _RuntimeProgram()

    def const(self, dst: str, value: Union[int, float, str]) -> "ProgramBuilder":
        """Add CONST instruction."""
        self._program.const(dst, _Value(value))
        return self

    def add(self, dst: str, src1: str, src2: str) -> "ProgramBuilder":
        """Add ADD instruction."""
        self._program.add(dst, src1, src2)
        return self

    def sub(self, dst: str, src1: str, src2: str) -> "ProgramBuilder":
        """Add SUB instruction."""
        self._program.sub(dst, src1, src2)
        return self

    def mul(self, dst: str, src1: str, src2: str) -> "ProgramBuilder":
        """Add MUL instruction."""
        self._program.mul(dst, src1, src2)
        return self

    def cmp_eq(self, dst: str, src1: str, src2: str) -> "ProgramBuilder":
        """Add CMP_EQ instruction."""
        self._program.cmp_eq(dst, src1, src2)
        return self

    def jump(self, label: str) -> "ProgramBuilder":
        """Add JUMP instruction."""
        self._program.jump(label)
        return self

    def jump_if_zero(self, cond: str, label: str) -> "ProgramBuilder":
        """Add JUMP_IF_ZERO instruction."""
        self._program.jump_if_zero(cond, label)
        return self

    def jump_if_not_zero(self, cond: str, label: str) -> "ProgramBuilder":
        """Add JUMP_IF_NOT_ZERO instruction."""
        self._program.jump_if_not_zero(cond, label)
        return self

    def label(self, name: str) -> "ProgramBuilder":
        """Add a label at current position."""
        self._program.add_label(name)
        return self

    def dispatch(self, handle: str, task_name: str, args: List[str]) -> "ProgramBuilder":
        """Add DISPATCH instruction."""
        self._program.dispatch(handle, task_name, args)
        return self

    def wait(self, handle: str) -> "ProgramBuilder":
        """Add WAIT instruction."""
        self._program.wait(handle)
        return self

    def wait_all(self, handles: List[str]) -> "ProgramBuilder":
        """Add WAIT_ALL instruction."""
        self._program.wait_all(handles)
        return self

    def store_mem(self, addr: int, reg: str) -> "ProgramBuilder":
        """Add STORE_MEM instruction."""
        self._program.store_mem(addr, reg)
        return self

    def load_mem(self, reg: str, addr: int) -> "ProgramBuilder":
        """Add LOAD_MEM instruction."""
        self._program.load_mem(reg, addr)
        return self

    def nop(self) -> "ProgramBuilder":
        """Add NOP instruction."""
        self._program.nop()
        return self

    def halt(self) -> "ProgramBuilder":
        """Add HALT instruction."""
        self._program.halt()
        return self

    @staticmethod
    def reg(n: int) -> str:
        """Generate register name from integer (e.g., reg(0) -> 'r0')."""
        return _RuntimeProgram.reg(n)

    def build(self) -> _RuntimeProgram:
        """Get the built program."""
        return self._program

    def __str__(self) -> str:
        """Get Python syntax representation."""
        return self._program.to_python_syntax()
