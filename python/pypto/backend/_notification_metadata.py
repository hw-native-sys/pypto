# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Project notification selectors into a Host-only metadata provider.

Only scalar expressions that determine notification coordinates and values are
emitted. Kernel computation and user-visible task arguments are unchanged.
"""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from pypto._function_attrs import BUILTIN_TEMPLATE_DIR_ATTR, EXTERNAL_SOURCE_ATTR
from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir

_NOTIFY = _ir.get_op("pld.system.notify").name
_WAIT = _ir.get_op("pld.system.defer_wait").name
_RANK = _ir.get_op("pld.system.rank").name
_NRANKS = _ir.get_op("pld.system.nranks").name
_BLOCK = frozenset(_ir.get_op(name).name for name in ("tile.get_block_idx", "buffer.get_block_idx"))
_BLOCKS = frozenset(_ir.get_op(name).name for name in ("tile.get_block_num", "buffer.get_block_num"))


class _Unsupported(ValueError):
    pass


@dataclass
class _Projection:
    stmt: _ir.Stmt
    children: list["_Projection"]
    active: bool


class _CallFacts(_ir.IRVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.effect = False
        self.opaque = False
        self.flags = 0
        self.callees: set[str] = set()

    def visit_call(self, op: _ir.Call) -> None:
        if isinstance(op.op, _ir.GlobalVar):
            self.opaque = True
            self.callees.add(op.op.name)
        elif op.op.name in {_NOTIFY, _WAIT}:
            self.effect = True
            self.flags |= 1 if op.op.name == _NOTIFY else 2
        super().visit_call(op)

    def visit_submit(self, op: _ir.Submit) -> None:
        self.opaque = True
        self.callees.add(op.op.name)
        super().visit_submit(op)


class _Emitter:
    def __init__(self, func: _ir.Function) -> None:
        self.func = func
        self.definitions: dict[int, _ir.Expr] = {}
        self.bindings: dict[int, str] = {}
        self.binding_changes: list[tuple[int, str | None]] = []
        self.tensor_indices: dict[int, int] = {}
        self.scalar_indices: dict[int, int] = {}
        self.contexts: dict[int, int] = {}
        self.lines: list[str] = []
        self.depth = 1
        self.serial = 0
        tensors = [p for p in func.params if isinstance(p.type, _ir.TensorType)]
        scalars = [p for p in func.params if not isinstance(p.type, _ir.TensorType)]
        self.tensor_indices = {p.unique_id: i for i, p in enumerate(tensors)}
        self.scalar_indices = {p.unique_id: i for i, p in enumerate(scalars)}
        dist = [p for p in tensors if isinstance(p.type, _ir.DistributedTensorType)]
        # MaterializeDistTensorCtx appends one context per distributed parameter,
        # in parameter order. Explicit user context operands precede that suffix.
        suffix = list(func.params[-len(dist) :]) if dist else []
        if suffix and all(isinstance(p.type, _ir.CommCtxType) for p in suffix):
            self.contexts = {
                tensor.unique_id: self.scalar_indices[ctx.unique_id]
                for tensor, ctx in zip(dist, suffix, strict=True)
            }
        self.opaque = False
        self.tree = self._index(func.body)
        self.tensor_count = len(tensors)
        self.scalar_count = len(scalars)

    def _index(self, stmt: _ir.Stmt) -> _Projection:
        if isinstance(stmt, _ir.SeqStmts):
            children = [self._index(child) for child in stmt.stmts]
        elif isinstance(stmt, (_ir.ForStmt, _ir.ScopeStmt)):
            children = [self._index(stmt.body)]
        elif isinstance(stmt, _ir.IfStmt):
            children = [self._index(stmt.then_body)]
            if stmt.else_body is not None:
                children.append(self._index(stmt.else_body))
        else:
            children = []
            if isinstance(stmt, _ir.AssignStmt):
                self.definitions[stmt.var.unique_id] = stmt.value
            facts = _CallFacts()
            facts.visit_stmt(stmt)
            self.opaque |= facts.opaque
            return _Projection(stmt, [], facts.effect)
        return _Projection(stmt, children, any(child.active for child in children))

    def emit(self, line: str) -> None:
        self.lines.append("    " * self.depth + line)

    def fresh(self) -> str:
        self.serial += 1
        return f"notification_value_{self.serial}"

    def bind(self, uid: int, value: str) -> None:
        self.binding_changes.append((uid, self.bindings.get(uid)))
        self.bindings[uid] = value

    def restore(self, checkpoint: int) -> None:
        while len(self.binding_changes) > checkpoint:
            uid, old = self.binding_changes.pop()
            if old is None:
                del self.bindings[uid]
            else:
                self.bindings[uid] = old

    @staticmethod
    def ctype(expr: _ir.Expr) -> str:
        typ = expr.type
        if isinstance(typ, _ir.CommCtxType):
            return "uint64_t"
        if not isinstance(typ, _ir.ScalarType):
            raise _Unsupported("notification selector is not a scalar")
        if typ.dtype == DataType.INDEX:
            return "int64_t"
        if typ.dtype not in {
            DataType.BOOL,
            DataType.INT8,
            DataType.INT16,
            DataType.INT32,
            DataType.INT64,
            DataType.UINT8,
            DataType.UINT16,
            DataType.UINT32,
            DataType.UINT64,
        }:
            raise _Unsupported("notification selector is not an integral value")
        return typ.dtype.to_c_type_string()

    def expr(self, expr: _ir.Expr) -> str:
        if isinstance(expr, _ir.ConstBool):
            return "true" if expr.value else "false"
        if isinstance(expr, _ir.ConstInt):
            bits = int(expr.value) & ((1 << 64) - 1)
            return f"static_cast<{self.ctype(expr)}>(UINT64_C({bits}))"
        if isinstance(expr, _ir.Var):
            uid = expr.unique_id
            if uid in self.bindings:
                return self.bindings[uid]
            name = self.fresh()
            if uid in self.scalar_indices:
                self.emit(f"{self.ctype(expr)} {name}{{}};")
                self.emit(
                    f"std::memcpy(&{name}, &input->scalars[{self.scalar_indices[uid]}], sizeof({name}));"
                )
            elif uid in self.definitions:
                value = self.expr(self.definitions[uid])
                self.emit(f"const auto {name} = {value};")
            else:
                raise _Unsupported("selector uses an unavailable device value or loop-carried value")
            self.bind(uid, name)
            return name
        if isinstance(expr, _ir.Cast):
            return f"static_cast<{self.ctype(expr)}>({self.expr(expr.operand)})"
        if isinstance(expr, (_ir.Not, _ir.Neg)):
            operand = self.expr(expr.operand)
            if isinstance(expr, _ir.Not):
                return f"(!({operand}))"
            return f"static_cast<{self.ctype(expr)}>(UINT64_C(0) - static_cast<uint64_t>({operand}))"
        if isinstance(expr, _ir.BinaryExpr):
            return self.binary(expr)
        if isinstance(expr, _ir.Call) and not isinstance(expr.op, _ir.GlobalVar):
            return self.call(expr)
        raise _Unsupported("notification selector requires device computation")

    def binary(self, expr: _ir.BinaryExpr) -> str:
        left, right = self.expr(expr.left), self.expr(expr.right)
        arithmetic = {_ir.Add: "+", _ir.Sub: "-", _ir.Mul: "*"}
        for kind, symbol in arithmetic.items():
            if isinstance(expr, kind):
                return (
                    f"static_cast<{self.ctype(expr)}>(static_cast<uint64_t>({left}) "
                    f"{symbol} static_cast<uint64_t>({right}))"
                )
        operators = {
            _ir.Eq: "==",
            _ir.Ne: "!=",
            _ir.Lt: "<",
            _ir.Le: "<=",
            _ir.Gt: ">",
            _ir.Ge: ">=",
            _ir.And: "&&",
            _ir.Or: "||",
            _ir.BitAnd: "&",
            _ir.BitOr: "|",
            _ir.BitXor: "^",
        }
        for kind, symbol in operators.items():
            if isinstance(expr, kind):
                return f"(({left}) {symbol} ({right}))"
        if isinstance(expr, (_ir.Min, _ir.Max)):
            function = "min" if isinstance(expr, _ir.Min) else "max"
            return f"std::{function}<{self.ctype(expr)}>({left}, {right})"
        if isinstance(expr, (_ir.FloorDiv, _ir.FloorMod)):
            return self.division(expr, left, right)
        raise _Unsupported("unsupported scalar notification arithmetic")

    def division(self, expr: _ir.BinaryExpr, left: str, right: str) -> str:
        name, ctype = self.fresh(), self.ctype(expr)
        self.emit(f"const {ctype} {name}_left = {left}, {name}_right = {right};")
        self.emit(f"if ({name}_right == 0) return NotificationPlanStatus::INVALID;")
        signed = not ctype.startswith("uint")
        if signed:
            self.emit(
                f"if ({name}_left == std::numeric_limits<{ctype}>::min() && {name}_right == -1) "
                "return NotificationPlanStatus::INVALID;"
            )
        self.emit(f"[[maybe_unused]] auto {name}_q = {name}_left / {name}_right;")
        self.emit(f"[[maybe_unused]] auto {name}_r = {name}_left % {name}_right;")
        if signed:
            self.emit(
                f"if ({name}_r != 0 && (({name}_r < 0) != ({name}_right < 0))) "
                f"{{ --{name}_q; {name}_r += {name}_right; }}"
            )
        return f"{name}_q" if isinstance(expr, _ir.FloorDiv) else f"{name}_r"

    def call(self, expr: _ir.Call) -> str:
        if expr.op.name in _BLOCK:
            return "input->block_index"
        if expr.op.name in _BLOCKS:
            return "input->block_count"
        if expr.op.name in {_RANK, _NRANKS}:
            context = self.expr(expr.args[0])
            name = self.fresh()
            self.emit(f"int32_t {name}_rank = -1;")
            self.emit(f"uint32_t {name}_size = 0;")
            self.emit(
                "if (input->query_context == nullptr || "
                f"!input->query_context(input->context_user, {context}, "
                f"&{name}_rank, &{name}_size)) return NotificationPlanStatus::UNSUPPORTED;"
            )
            return f"{name}_rank" if expr.op.name == _RANK else f"{name}_size"
        raise _Unsupported("notification selector requires device computation")

    def effect(self, call: _ir.Call) -> None:
        signal = call.args[0]
        if not isinstance(signal, _ir.Var) or signal.unique_id not in self.tensor_indices:
            raise _Unsupported("notification signal is not a directly bound tensor parameter")
        notify = call.op.name == _NOTIFY
        index = 2 if notify else 1
        offsets = call.args[index]
        if not isinstance(offsets, _ir.MakeTuple) or not 1 <= len(offsets.elements) <= 5:
            raise _Unsupported("notification coordinates require one to five logical dimensions")
        peer = self.expr(call.args[1]) if notify else "-1"
        value = self.expr(call.args[index + 1])
        coordinates = [self.expr(offset) for offset in offsets.elements]
        if notify:
            operation = int(call.kwargs["op"])
            if operation not in {int(_ir.NotifyOp.Set), int(_ir.NotifyOp.AtomicAdd)}:
                raise _Unsupported("invalid notification operation")
            kind = "SET" if operation == int(_ir.NotifyOp.Set) else "ADD"
        else:
            if int(call.kwargs["cmp"]) != int(_ir.WaitCmp.Ge):
                raise _Unsupported("only GE deferred counter conditions are supported")
            kind = "WAIT_GE"
        self.emit("{")
        self.depth += 1
        self.emit("NotificationPlanEvent event{};")
        self.emit(f"event.kind = NotificationEventKind::{kind};")
        self.emit(f"event.tensor_index = {self.tensor_indices[signal.unique_id]};")
        self.emit(f"event.ndims = {len(coordinates)};")
        for i, coordinate in enumerate(coordinates):
            self.emit(f"event.offsets[{i}] = {coordinate};")
        self.emit(f"event.peer = {peer};")
        self.emit(f"event.value = {value};")
        context_slot = self.contexts.get(signal.unique_id)
        if context_slot is not None:
            self.emit(f"event.comm_context = input->scalars[{context_slot}];")
        self.emit("if (!sink(sink_context, &event)) return NotificationPlanStatus::INVALID;")
        self.depth -= 1
        self.emit("}")

    def body(self, node: _Projection) -> None:
        if not node.active:
            return
        stmt = node.stmt
        if isinstance(stmt, (_ir.SeqStmts, _ir.ScopeStmt)):
            for child in node.children:
                self.body(child)
        elif isinstance(stmt, _ir.ForStmt):
            if stmt.iter_args:
                raise _Unsupported("notification loop has loop-carried values")
            start, stop, step = self.expr(stmt.start), self.expr(stmt.stop), self.expr(stmt.step)
            loop = self.fresh()
            self.emit(f"const int64_t {loop}_stop = {stop};")
            self.emit(f"const int64_t {loop}_step = {step};")
            self.emit(f"if ({loop}_step == 0) return NotificationPlanStatus::INVALID;")
            self.emit(
                f"for (int64_t {loop} = {start}; "
                f"{loop}_step > 0 ? {loop} < {loop}_stop : {loop} > {loop}_stop; "
                f"{loop} = static_cast<int64_t>(static_cast<uint64_t>({loop}) "
                f"+ static_cast<uint64_t>({loop}_step))) {{"
            )
            self.depth += 1
            self.emit("if (++iterations > UINT64_C(1048576)) return NotificationPlanStatus::UNSUPPORTED;")
            checkpoint = len(self.binding_changes)
            self.bind(stmt.loop_var.unique_id, loop)
            self.body(node.children[0])
            self.restore(checkpoint)
            self.depth -= 1
            self.emit("}")
        elif isinstance(stmt, _ir.IfStmt):
            condition = self.expr(stmt.condition)
            self.emit(f"if ({condition}) {{")
            checkpoint = len(self.binding_changes)
            self.depth += 1
            self.body(node.children[0])
            self.restore(checkpoint)
            self.depth -= 1
            if len(node.children) == 2:
                self.emit("} else {")
                self.depth += 1
                self.body(node.children[1])
                self.depth -= 1
            self.restore(checkpoint)
            self.emit("}")
        else:
            expr = (
                stmt.expr
                if isinstance(stmt, _ir.EvalStmt)
                else stmt.value
                if isinstance(stmt, _ir.AssignStmt)
                else None
            )
            if not isinstance(expr, _ir.Call) or expr.op.name not in {_NOTIFY, _WAIT}:
                raise _Unsupported("notification occurs inside unsupported control flow")
            self.effect(expr)

    def function(self, func_id: int) -> str:
        self.emit(
            f"if (input->tensor_count != {self.tensor_count} || "
            f"input->scalar_count != {self.scalar_count}) return NotificationPlanStatus::INVALID;"
        )
        if (
            self.opaque
            or EXTERNAL_SOURCE_ATTR in self.func.attrs
            or BUILTIN_TEMPLATE_DIR_ATTR in self.func.attrs
        ):
            self.emit("return NotificationPlanStatus::OPAQUE;")
        else:
            try:
                self.emit("[[maybe_unused]] uint64_t iterations = 0;")
                self.body(self.tree)
                self.emit("return NotificationPlanStatus::COMPLETE;")
            except _Unsupported as error:
                # A partial event stream cannot certify a task's source set.
                self.lines = [f"    // {error}", "    return NotificationPlanStatus::UNSUPPORTED;"]
        return (
            f"static NotificationPlanStatus notification_kernel_{func_id}("
            "[[maybe_unused]] const NotificationPlanInput* input, "
            "[[maybe_unused]] NotificationEventSink sink, [[maybe_unused]] void* sink_context) {\n"
            + "\n".join(self.lines)
            + "\n}\n"
        )


def _function_flags(program: _ir.Program) -> dict[str, int]:
    """Propagate known effects through helpers without interpreting their bodies.

    A helper call still makes selector projection opaque, but its known waiter
    effect must reach the caller so it is not mistaken for ordinary computation.
    Recursion does not invent effects: propagate only observed bits through
    callers. Each of the three bits changes at most once per function.
    """
    facts: dict[str, _CallFacts] = {}
    for func in program.functions.values():
        info = _CallFacts()
        info.visit_stmt(func.body)
        if func.attrs.get("deferred_completion_waiter", False):
            info.flags |= 2
        if EXTERNAL_SOURCE_ATTR in func.attrs or BUILTIN_TEMPLATE_DIR_ATTR in func.attrs:
            info.opaque = True
        facts[func.name] = info
    result = {name: info.flags | (4 if info.opaque else 0) for name, info in facts.items()}
    callers: dict[str, set[str]] = {name: set() for name in facts}
    for name, info in facts.items():
        for callee in info.callees:
            if callee in callers:
                callers[callee].add(name)
            else:
                result[name] |= 4
    pending = deque(facts)
    while pending:
        name = pending.popleft()
        for caller in callers[name]:
            propagated = result[caller] | result[name]
            if propagated != result[caller]:
                result[caller] = propagated
                pending.append(caller)
    return result


def generate_notification_metadata(program: _ir.Program, registrations: Sequence[tuple[int, str]]) -> str:
    """Emit metadata for registered leaf kernels, including explicit unknowns."""
    bodies: list[str] = []
    cases: list[str] = []
    flags = _function_flags(program)
    flag_cases = [f"    case {fid}: return {flags.get(name, 4)};" for fid, name in registrations]
    for func_id, code_name in registrations:
        func = program.get_function(code_name)
        if func is None:
            cases.append(f"    case {func_id}: return NotificationPlanStatus::OPAQUE;")
            continue
        emitter = _Emitter(func)
        bodies.append(emitter.function(func_id))
        cases.append(f"    case {func_id}: return notification_kernel_{func_id}(input, sink, sink_context);")
    return (
        "\n#include <cstdint>\n#include <cstring>\n#include <limits>\n"
        '#include "host_build_graph/host/notification_metadata.h"\n'
        "namespace {\nusing namespace simpler::hbg;\n"
        + "\n".join(bodies)
        + '}\nextern "C" __attribute__((visibility("default"))) simpler::hbg::NotificationPlanStatus\n'
        "simpler_hbg_notification_plan_v1(uint32_t kernel_id, "
        "const simpler::hbg::NotificationPlanInput* input, "
        "simpler::hbg::NotificationEventSink sink, void* sink_context) {\n"
        "    using namespace simpler::hbg;\n"
        "    if (input == nullptr || input->version != 1 || sink == nullptr || "
        "(input->tensor_count != 0 && input->tensors == nullptr) || "
        "(input->scalar_count != 0 && input->scalars == nullptr)) "
        "return NotificationPlanStatus::INVALID;\n"
        "    switch (kernel_id) {\n"
        + "\n".join(cases)
        + "\n    default: return NotificationPlanStatus::INVALID;\n    }\n}\n"
        + 'extern "C" __attribute__((visibility("default"))) uint32_t\n'
        "simpler_hbg_notification_kernel_flags_v1(uint32_t kernel_id) {\n"
        "    switch (kernel_id) {\n" + "\n".join(flag_cases) + "\n    default: return UINT32_MAX;\n    }\n}\n"
    )
