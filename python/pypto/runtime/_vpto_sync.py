# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Fail-closed admission for the verified, static VF memory-analysis subset.

This checks canonical LLVM emitted by PTOAS, before Bisheng can optimize away
memory operations. It inserts no barriers and does not replace PTOAS analysis.
Unknown syntax is rejected, including after a compiler output-format change.
Only one straight-line AIV scope, contiguous FP32 accesses, one symbolic UB root
and constant byte displacements are admitted. Scalars and masks may be dynamic.
Required barrier coverage is checked at 32-byte granularity, conservatively
using all 64 lanes for a masked access. The walk is linear: each access touches
eight fixed-size blocks; pointer provenance is resolved in SSA definition order.
"""

import re
from typing import NoReturn

VERSION = "static-ub-f32-v3"
_SSA = r"%[-a-zA-Z$._0-9]+"
_PURE_OPS = frozenset(
    "add sub mul udiv sdiv urem srem shl lshr ashr and or xor fadd fsub fmul fdiv fneg "
    "icmp fcmp select trunc zext sext fptrunc fpext fptoui fptosi uitofp sitofp "
    "bitcast insertelement extractelement shufflevector freeze".split()
)
_PURE_INTRINSIC = re.compile(
    r"llvm\.hivm\.(?:pset\.b32|vci\.v64i32|vdups\.z\.v64(?:f32|i32)|"
    r"v(?:add|sub|mul|min|max)\.s\.x\.v64(?:f32|i32)|"
    r"vcmp\.(?:lt|le|gt|ge|eq|ne)\.s\.z\.v64(?:f32|i32))"
)


def _reject(reason: str) -> NoReturn:
    raise ValueError(f"unsupported VF synchronization contract: {reason}")


class _MemoryHistory:
    """Verify existing barriers in O(N) without inserting or moving any."""

    def __init__(self) -> None:
        self.reads: dict[int, int] = {}
        self.writes: dict[int, int] = {}
        # A frontier denotes every operation of that pipe up to its position.
        # Do not assume FIFO memory completion within either pipe. Only an
        # explicit barrier transfers the source pipe's issued operations and
        # their proven ancestors to subsequent operations of the target pipe.
        self.before = {pipe: {"vld": -1, "vst": -1} for pipe in ("vld", "vst")}
        self.issued = {pipe: {"vld": -1, "vst": -1} for pipe in ("vld", "vst")}

    def barrier(self, kind: str) -> None:
        source, target = kind.split(".")
        for pipe in ("vld", "vst"):
            self.before[target][pipe] = max(self.before[target][pipe], self.issued[source][pipe])

    def access(self, displacement: int, load: bool, position: int) -> None:
        blocks = range(displacement // 32, displacement // 32 + 8)
        target = "vld" if load else "vst"
        dependencies = [(self.writes, "vst")] if load else [(self.reads, "vld"), (self.writes, "vst")]
        for history, source in dependencies:
            if any(history.get(block, -1) > self.before[target][source] for block in blocks):
                _reject(
                    f"missing {source}.{target} barrier/order "
                    f"for overlapping UB access at byte {displacement}"
                )
        history = self.reads if load else self.writes
        for block in blocks:
            history[block] = position
        for pipe in ("vld", "vst"):
            self.issued[target][pipe] = max(self.issued[target][pipe], self.before[target][pipe])
        self.issued[target][target] = position


def _scope_instructions(lines: list[str]) -> tuple[list[str], list[str]]:
    """Recognize PTOAS's single-iteration AIV marker loop, not arbitrary CFGs."""
    blocks: list[tuple[str, list[str]]] = [("", [])]
    for line in lines:
        if match := re.fullmatch(r"([-a-zA-Z$._0-9]+):", line):
            blocks.append(("%" + match[1], []))
        else:
            blocks[-1][1].append(line)
    if len(blocks) == 1 and lines and lines[-1] == "ret void":
        # Useful for a scalar-only helper; UB instructions require an AIV scope.
        return lines[:-1], []
    if len(blocks) != 5:
        _reject("expected one straight-line AIV scope")
    (_, entry), (head_name, head), (body_name, body), (latch_name, latch), (end_name, end) = blocks
    if len(head) != 3 or len(latch) != 2 or not entry or not body or end != ["ret void"]:
        _reject("unexpected AIV control flow")
    phi = re.fullmatch(rf"({_SSA}) = phi i64 \[ ({_SSA}), ({_SSA}) \], \[ 0, ({_SSA}) \]", head[0])
    if phi is None or phi[3] != latch_name:
        _reject("AIV induction must start at zero")
    compare = re.fullmatch(rf"({_SSA}) = icmp slt i64 {re.escape(phi[1])}, 1", head[1])
    if compare is None:
        _reject("AIV scope must execute exactly once")
    if (
        entry[-1] != f"br label {head_name}"
        or head[2] != f"br i1 {compare[1]}, label {body_name}, label {end_name}"
        or body[-1] != f"br label {latch_name}"
        or latch[0] != f"{phi[2]} = add i64 {phi[1]}, 1"
        or re.fullmatch(rf"br label {re.escape(head_name)}, !llvm.loop !\d+", latch[1]) is None
    ):
        _reject("unexpected AIV control flow")
    return entry[:-1], body[:-1]


def validate_sync_contract(llvm: str, source: str) -> None:
    """Reject unproven relationships, access modes and control flow, with context."""
    try:
        _validate(llvm)
    except ValueError as error:
        raise ValueError(f"VPTO helper {source}: {error}") from error


def _validate(llvm: str) -> None:
    # LLVM has already parsed/verified the source inside PTOAS. This deliberately
    # recognizes a closed output subset instead of guessing at general LLVM IR.
    lines = [line.split(";", 1)[0].strip() for line in llvm.splitlines()]
    lines = [line for line in lines if line]
    signatures = [i for i, line in enumerate(lines) if line.startswith("define ")]
    if len(signatures) != 1:
        _reject("expected exactly one helper definition")
    start = signatures[0]
    signature = re.fullmatch(r"define void @[-a-zA-Z$._0-9]+\((.*)\)(?: #\d+)? \{", lines[start])
    if signature is None:
        _reject("unsupported helper signature")
    try:
        end = lines.index("}", start)
    except ValueError:
        _reject("missing helper end")
    for line in lines[:start] + lines[end + 1 :]:
        if not line.startswith(("source_filename =", "target ", "declare ", "attributes #", "!")):
            _reject("unsupported module content")
    pointers: dict[str, tuple[str, int]] = {}
    for argument in signature[1].split(",") if signature[1] else []:
        argument = argument.strip()
        if match := re.fullmatch(rf"ptr addrspace\(6\) ({_SSA})", argument):
            pointers[match[1]] = (match[1], 0)
        elif not re.fullmatch(rf"(?:float|i32|i64) {_SSA}", argument):
            _reject("only UB pointers and supported scalars are admitted")
    entry, body = _scope_instructions(lines[start + 1 : end])
    if body:
        loop = next(re.finditer(r"!llvm.loop (!\d+)", llvm), None)
        record = (
            re.search(rf"^{loop[1]} = distinct !\{{{loop[1]}, (!\d+)\}}$", llvm, re.MULTILINE)
            if loop
            else None
        )
        if record is None or f'{record[1]} = !{{!"llvm.loop.aivector_scope"}}' not in lines:
            _reject("missing AIV scope metadata")
    _validate_instructions(entry, body, pointers)


def _validate_instructions(entry: list[str], body: list[str], pointers: dict[str, tuple[str, int]]) -> None:
    roots: set[str] = set()
    history = _MemoryHistory()
    for position, (instruction, in_scope) in enumerate(
        [(x, False) for x in entry] + [(x, True) for x in body]
    ):
        assign = re.fullmatch(rf"({_SSA}) = (.*)", instruction)
        result, expression = (assign[1], assign[2]) if assign else ("", instruction)
        if expression.startswith("getelementptr "):
            gep = re.fullmatch(
                rf"getelementptr (?:inbounds )?float, ptr addrspace\(6\) ({_SSA}), i64 (-?\d+)",
                expression,
            )
            if not result or gep is None or gep[1] not in pointers:
                _reject("UB pointer derivation requires a known base and constant FP32 offset")
            root, offset = pointers[gep[1]]
            offset += int(gep[2]) * 4
            if not 0 <= offset < 2**31:
                _reject("UB pointer displacement is out of range")
            pointers[result] = (root, offset)
            continue
        call = re.fullmatch(r"call (.+?) @([-a-zA-Z$._0-9]+)\((.*)\)", expression)
        if call:
            name, args = call[2], call[3]
            if name in ("llvm.hivm.vldsx1.v64f32", "llvm.hivm.vstsx1.v64f32"):
                if not in_scope:
                    _reject("UB access outside the verified AIV scope")
                load = name == "llvm.hivm.vldsx1.v64f32"
                prefix = "" if load else rf"<64 x float> {_SSA}, "
                suffix = "" if load else rf", <256 x i1> {_SSA}"
                memory = re.fullmatch(
                    prefix
                    + rf"ptr addrspace\(6\) ({_SSA}), i32 (\d+), i32 "
                    + ("0" if load else "2")
                    + ", i32 0"
                    + suffix,
                    args,
                )
                if memory is None or memory[1] not in pointers:
                    _reject(
                        "contiguous UB access requires a proven pointer, static offset and supported mode"
                    )
                root, displacement = pointers[memory[1]]
                displacement += int(memory[2])
                if displacement % 32 or displacement + 256 >= 2**31:
                    _reject("UB access requires static 32-byte alignment and bounded displacement")
                roots.add(root)
                if len(roots) > 1:
                    _reject("UB accesses use unrelated pointer parameters; preserve a common base")
                history.access(displacement, load, position)
            elif (
                name
                in ("llvm.hivm.mem.bar.vst.vld", "llvm.hivm.mem.bar.vld.vst", "llvm.hivm.mem.bar.vst.vst")
                and not args
                and in_scope
            ):
                history.barrier(name.removeprefix("llvm.hivm.mem.bar."))
            elif _PURE_INTRINSIC.fullmatch(name) and "ptr" not in expression:
                pass
            else:
                _reject(f"unmodeled intrinsic or call {name}")
            continue
        opcode = expression.split(" ", 1)[0]
        if not result or opcode not in _PURE_OPS or "ptr" in expression:
            _reject(f"unmodeled instruction: {expression}")
