# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Recognize PTOAS's canonical single-loop FP32 helper and prove its pointers.

The input has already been verified as LLVM by PTOAS. This is deliberately a
closed subset recognizer, not a general LLVM parser. Unknown compiler output
fails admission. Pointer phis must advance exactly 256 bytes per iteration,
preserve one UB root, and return the same provenance on the backedge.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, replace

from ._vpto_memory import Access, reject, validate_loop_history

_SSA = r"%[-a-zA-Z$._0-9]+"


@dataclass(frozen=True)
class _Pointer:
    root: str
    offset: int
    # The phi name identifies an advancing stream; empty means loop-invariant.
    stream: str = ""


class _Instructions:
    def __init__(self, pointers: dict[str, tuple[str, int]], trips: int, pure: Callable[[str], None]) -> None:
        self.pointers = {name: _Pointer(*value) for name, value in pointers.items()}
        self.aggregates: dict[str, _Pointer] = {}
        self.trips = trips
        self.pure = pure
        self.roots: set[str] = set()
        self.ordinal = 0

    def memory(
        self, memory: re.Match[str], args: str, result: str, *, in_scope: bool, in_loop: bool
    ) -> Access:
        if not in_scope:
            reject("UB access outside the verified AIV scope")
        load, post = memory[1] == "vldsx1", bool(memory[2])
        prefix = "" if load else rf"<64 x float> {_SSA}, "
        suffix = "" if load else rf", <256 x i1> {_SSA}"
        operands = re.fullmatch(
            prefix
            + rf"ptr addrspace\(6\) ({_SSA}), i32 (\d+), i32 "
            + ("0" if load else "2")
            + ", i32 "
            + ("1" if post else "0")
            + suffix,
            args,
        )
        if operands is None or operands[1] not in self.pointers:
            reject("hardware-loop memory requires a proven pointer and supported mode")
        pointer = self.pointers[operands[1]]
        displacement = int(operands[2])
        if post:
            if not result or not in_loop or displacement != 256 or not pointer.stream:
                reject("hardware-loop post-increment must advance a loop pointer by 256 bytes")
            advanced = replace(pointer, offset=pointer.offset + displacement)
            if load:
                self.aggregates[result] = advanced
            else:
                self.pointers[result] = advanced
            displacement = 0  # The intrinsic accesses the incoming pointer, then advances it.
        begin = pointer.offset + displacement
        moving = bool(pointer.stream)
        end = begin + 256 * (self.trips if moving else 1)
        if begin < 0 or begin % 32 or end >= 2**31 or (moving and not in_loop):
            reject("hardware-loop access requires aligned bounded in-loop provenance")
        self.roots.add(pointer.root)
        if len(self.roots) != 1:
            reject("UB accesses use unrelated pointer parameters; preserve a common base")
        access = Access(begin, end, load, self.ordinal, moving)
        self.ordinal += 1
        return access

    def parse(self, lines: list[str], *, in_scope: bool, in_loop: bool = False) -> list[Access | str]:
        actions: list[Access | str] = []
        for line in lines:
            assignment = re.fullmatch(rf"({_SSA}) = (.*)", line)
            result, expression = (assignment[1], assignment[2]) if assignment else ("", line)
            if expression.startswith("getelementptr "):
                gep = re.fullmatch(
                    rf"getelementptr (?:inbounds )?float, ptr addrspace\(6\) ({_SSA}), i64 (-?\d+)",
                    expression,
                )
                if not result or gep is None or gep[1] not in self.pointers:
                    reject("hardware-loop pointer derivation requires a proven constant displacement")
                base = self.pointers[gep[1]]
                offset = base.offset + 4 * int(gep[2])
                if not 0 <= offset < 2**31:
                    reject("hardware-loop pointer displacement is out of range")
                self.pointers[result] = replace(base, offset=offset)
                continue
            if expression.startswith("extractvalue "):
                extract = re.fullmatch(
                    rf"extractvalue \{{ <64 x float>, ptr addrspace\(6\) \}} ({_SSA}), ([01])", expression
                )
                if not result or extract is None or extract[1] not in self.aggregates:
                    reject("hardware-loop pointer extraction requires a known post-increment load")
                if extract[2] == "1":
                    self.pointers[result] = self.aggregates[extract[1]]
                continue
            call = re.fullmatch(r"call (.+?) @([-a-zA-Z$._0-9]+)\((.*)\)", expression)
            if call:
                name, args = call[2], call[3]
                memory = re.fullmatch(r"llvm\.hivm\.(vldsx1|vstsx1)(\.post)?\.v64f32", name)
                if memory:
                    actions.append(self.memory(memory, args, result, in_scope=in_scope, in_loop=in_loop))
                    continue
                if (
                    name
                    in ("llvm.hivm.mem.bar.vst.vld", "llvm.hivm.mem.bar.vld.vst", "llvm.hivm.mem.bar.vst.vst")
                    and not args
                    and in_scope
                ):
                    actions.append(name.removeprefix("llvm.hivm.mem.bar."))
                    continue
            # Reuse the straight-line instruction whitelist for scalar/vector
            # arithmetic. It rejects unknown calls, pointer casts and control flow.
            self.pure(line)
        return actions


def validate_loop(
    lines: list[str], pointers: dict[str, tuple[str, int]], trips: int, pure: Callable[[str], None]
) -> None:
    """Validate one canonical i16 loop inside the single-iteration AIV marker."""
    if not 2 <= trips <= 32767:
        reject("hardware-loop trip count must be static and in [2, 32767]")
    blocks: list[tuple[str, list[str]]] = [("", [])]
    for line in lines:
        if match := re.fullmatch(r"([-a-zA-Z$._0-9]+):", line):
            blocks.append(("%" + match[1], []))
        else:
            blocks[-1][1].append(line)
    if len(blocks) != 8:
        reject("hardware-loop helper requires one canonical loop inside one AIV scope")
    (
        (_, entry),
        (scope_name, scope),
        (pre_name, pre),
        (head_name, head),
        (body_name, body),
        (post_name, post),
        (latch_name, latch),
        (end_name, end),
    ) = blocks
    if (
        not entry
        or len(scope) != 3
        or not pre
        or len(head) < 3
        or len(body) < 2
        or not post
        or len(latch) != 2
        or end != ["ret void"]
    ):
        reject("unexpected hardware-loop block structure")
    scope_phi = re.fullmatch(
        rf"({_SSA}) = phi i64 \[ ({_SSA}), {re.escape(latch_name)} \], \[ 0, {_SSA} \]", scope[0]
    )
    scope_cmp = re.fullmatch(rf"({_SSA}) = icmp slt i64 ({_SSA}), 1", scope[1])
    if scope_phi is None or scope_cmp is None or scope_cmp[2] != scope_phi[1]:
        reject("hardware-loop requires a single-iteration AIV marker")
    if (
        entry[-1] != f"br label {scope_name}"
        or scope[2] != f"br i1 {scope_cmp[1]}, label {pre_name}, label {end_name}"
        or pre[-1] != f"br label {head_name}"
        or post[-1] != f"br label {latch_name}"
        or latch[0] != f"{scope_phi[2]} = add i64 {scope_phi[1]}, 1"
        or re.fullmatch(rf"br label {re.escape(scope_name)}, !llvm.loop !\d+", latch[1]) is None
    ):
        reject("unexpected hardware-loop AIV control flow")
    phi = re.fullmatch(
        rf"({_SSA}) = phi i16 \[ ({_SSA}), {re.escape(body_name)} \], \[ 0, {re.escape(pre_name)} \]", head[0]
    )
    compare = re.fullmatch(rf"({_SSA}) = icmp slt i16 ({_SSA}), {trips}", head[-2])
    if phi is None or compare is None or compare[2] != phi[1]:
        reject("hardware-loop counter does not match the required trip count")
    if (
        head[-1] != f"br i1 {compare[1]}, label {body_name}, label {post_name}"
        or body[-2] != f"{phi[2]} = add i16 {phi[1]}, 1"
        or body[-1] != f"br label {head_name}"
    ):
        reject("hardware-loop counter must advance by one on its sole backedge")
    instructions = _Instructions(pointers, trips, pure)
    instructions.parse(entry[:-1], in_scope=False)
    prefix = instructions.parse(pre[:-1], in_scope=True)
    carries: list[tuple[str, str, _Pointer]] = []
    for line in head[1:-2]:
        pointer_phi = re.fullmatch(
            rf"({_SSA}) = phi ptr addrspace\(6\) \[ ({_SSA}), {re.escape(body_name)} \], "
            rf"\[ ({_SSA}), {re.escape(pre_name)} \]",
            line,
        )
        if pointer_phi is None or pointer_phi[3] not in instructions.pointers:
            reject("hardware-loop carries require a proven initial UB pointer")
        initial = instructions.pointers[pointer_phi[3]]
        if initial.stream:
            reject("nested hardware-loop pointer streams are unsupported")
        pointer = replace(initial, stream=pointer_phi[1])
        instructions.pointers[pointer_phi[1]] = pointer
        carries.append((pointer_phi[1], pointer_phi[2], pointer))
    actions = instructions.parse(body[:-2], in_scope=True, in_loop=True)
    for name, next_name, pointer in carries:
        if instructions.pointers.get(next_name) != replace(pointer, offset=pointer.offset + 256):
            reject(f"hardware-loop pointer {name} must advance exactly 256 bytes on the backedge")
    suffix = instructions.parse(post[:-1], in_scope=True)
    validate_loop_history(prefix, actions, suffix)
