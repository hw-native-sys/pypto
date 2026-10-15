# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Identify HBG arguments whose storage is accessed by host orchestration."""

from pypto.pypto_core import ir


class _HostAccess(ir.IRVisitor):
    """Conservative alias components, in O(N log N) using union by size."""

    def __init__(self) -> None:
        super().__init__()
        self.parents: dict[int, int] = {}
        self.sizes: dict[int, int] = {}
        self.host: set[ir.Var] = set()
        self.device: set[ir.Var] = set()
        self.yields: list[list[ir.Var]] = []
        self.predicate: ir.Expr | None = None
        self.in_predicate = False

    def root(self, value: ir.Var) -> int:
        var = value.unique_id
        self.parents.setdefault(var, var)
        self.sizes.setdefault(var, 1)
        while self.parents[var] != var:
            var = self.parents[var]
        return var

    def alias(self, left: ir.Expr, right: ir.Expr) -> None:
        if not isinstance(left, ir.Var) or not isinstance(right, ir.Var):
            return
        if not isinstance(left.type, ir.TensorType) or not isinstance(right.type, ir.TensorType):
            return
        a, b = self.root(left), self.root(right)
        if a == b:
            return
        if self.sizes[a] < self.sizes[b]:
            a, b = b, a
        self.parents[b] = a
        self.sizes[a] += self.sizes[b]

    def visit_assign_stmt(self, op: ir.AssignStmt) -> None:
        self.alias(op.var, op.value)
        # These descriptor-only operations preserve their first argument's storage.
        if isinstance(op.value, ir.Call) and op.value.op.name in {
            ir.get_op("tensor.slice").name,
            ir.get_op("tensor.view").name,
            ir.get_op("tensor.reshape").name,
            ir.get_op("tensor.transpose").name,
            ir.get_op("tensor.reinterpret_view").name,
        }:
            self.alias(op.var, op.value.args[0])
        super().visit_assign_stmt(op)

    def record_call(self, op: ir.Call | ir.Submit) -> None:
        if op.op.name in {ir.get_op("tensor.read").name, ir.get_op("tensor.write").name}:
            arg = op.args[0]
            if not isinstance(arg, ir.Var):
                raise ValueError("HBG host tensor access requires an external tensor argument")
            (self.device if self.in_predicate else self.host).add(arg)
        elif isinstance(op.op, ir.GlobalVar):
            self.device.update(
                arg for arg in op.args if isinstance(arg, ir.Var) and isinstance(arg.type, ir.TensorType)
            )

    def visit_call(self, op: ir.Call) -> None:
        self.record_call(op)
        super().visit_call(op)

    def visit_submit(self, op: ir.Submit) -> None:
        self.record_call(op)
        previous = self.predicate
        self.predicate = op.predicate
        try:
            super().visit_submit(op)
        finally:
            self.predicate = previous

    def visit_expr(self, expr: ir.Expr) -> None:
        # Keep the base visitor's traversal of args, launch settings and attrs.
        # Only the predicate subtree is evaluated by the device scheduler.
        previous = self.in_predicate
        if self.predicate is not None and expr.same_as(self.predicate):
            self.in_predicate = True
        try:
            super().visit_expr(expr)
        finally:
            self.in_predicate = previous

    def visit_if_stmt(self, op: ir.IfStmt) -> None:
        self.yields.append(op.return_vars)
        super().visit_if_stmt(op)
        self.yields.pop()

    def loop(self, op: ir.ForStmt | ir.WhileStmt) -> None:
        for carry, result in zip(op.iter_args, op.return_vars, strict=True):
            self.alias(carry, carry.initValue)
            self.alias(carry, result)
        self.yields.append(op.return_vars)

    def visit_for_stmt(self, op: ir.ForStmt) -> None:
        self.loop(op)
        super().visit_for_stmt(op)
        self.yields.pop()

    def visit_while_stmt(self, op: ir.WhileStmt) -> None:
        self.loop(op)
        super().visit_while_stmt(op)
        self.yields.pop()

    def visit_yield_stmt(self, op: ir.YieldStmt) -> None:
        if self.yields:
            for result, value in zip(self.yields[-1], op.value, strict=True):
                self.alias(result, value)
        super().visit_yield_stmt(op)


def host_tensor_indices(func: ir.Function) -> list[int]:
    """Return tensor-pool slots to bind with explicit HOST/NONE transfers.

    Device tasks cannot consume these host views. Reject ambiguous mixed uses
    before dispatch, including aliases and branch/loop-carried descriptors.
    """
    access = _HostAccess()
    access.visit_stmt(func.body)
    host = {access.root(var) for var in access.host}
    device = {access.root(var) for var in access.device}
    if host & device:
        raise ValueError(
            "host_build_graph requires separate tensor arguments for host tensor.read/write "
            "and device kernels. Pass the control tensor and device tensor in separate parameters."
        )
    tensors = [param for param in func.params if isinstance(param.type, ir.TensorType)]
    external = {access.root(var) for var in tensors}
    if host - external:
        raise ValueError(
            "host_build_graph tensor.read/write requires a ready external host tensor; "
            "runtime allocations and device-produced tensors cannot be accessed during graph construction."
        )
    return [i for i, var in enumerate(tensors) if access.root(var) in host]
