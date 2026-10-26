# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Lexical statement traversal shared by Buffer migration tests."""

from collections.abc import Iterator

from pypto import ir


def statements(program: ir.Program) -> Iterator[ir.Stmt]:
    """Walk regions in source order without recursive native visitor callbacks."""
    for function in program.functions.values():
        pending = [function.body]
        while pending:
            statement = pending.pop()
            yield statement
            if isinstance(statement, ir.SeqStmts):
                pending.extend(reversed(statement.stmts))
            elif isinstance(statement, ir.IfStmt):
                if statement.else_body is not None:
                    pending.append(statement.else_body)
                pending.append(statement.then_body)
            elif isinstance(statement, (ir.ForStmt, ir.WhileStmt, ir.SplitAivScopeStmt, ir.RuntimeScopeStmt)):
                pending.append(statement.body)


class TileHandleOrigins:
    """Trace every incoming handle through the reference cases exercised here."""

    def __init__(self, program: ir.Program):
        self.values: dict[str, ir.Var] = {}
        self.edges: dict[int, list[ir.Expr]] = {}
        aliases = {ir.get_op(name).name for name in ("tile.reshape", "tile.gather_row")}
        for function in program.functions.values():
            for parameter in function.params:
                self.values[parameter.name_hint] = parameter
        for statement in statements(program):
            if isinstance(statement, ir.AssignStmt):
                self.values[statement.var.name_hint] = statement.var
                if isinstance(statement.value, (ir.Var, ir.IterArg)):
                    self.edges[statement.var.unique_id] = [statement.value]
                elif isinstance(statement.value, ir.Call) and statement.value.op.name in aliases:
                    self.edges[statement.var.unique_id] = [statement.value.args[0]]
            elif isinstance(statement, ir.IfStmt):
                assert statement.else_body is not None
                arms = [statement.then_body, statement.else_body]
                for slot, result in enumerate(statement.return_vars):
                    self.values[result.name_hint] = result
                    sources: list[ir.Expr] = []
                    for arm in arms:
                        yielded = ir.get_last_yield_stmt(arm)
                        assert yielded is not None and len(yielded.value) == len(statement.return_vars)
                        sources.append(yielded.value[slot])
                    self.edges[result.unique_id] = sources
            elif isinstance(statement, (ir.ForStmt, ir.WhileStmt)):
                yielded = ir.get_last_yield_stmt(statement.body)
                assert yielded is not None and len(yielded.value) == len(statement.iter_args)
                assert len(statement.return_vars) == len(statement.iter_args)
                for slot, (argument, result) in enumerate(
                    zip(statement.iter_args, statement.return_vars, strict=True)
                ):
                    self.values[argument.name_hint] = argument
                    self.values[result.name_hint] = result
                    self.edges[argument.unique_id] = [argument.initValue, yielded.value[slot]]
                    self.edges[result.unique_id] = [argument]

    def owners(self, value: ir.Expr) -> list[ir.Expr]:
        """Return all concrete incoming endpoints, rejecting an unrooted graph."""
        return self._owners([value])

    def _owners(self, pending: list[ir.Expr]) -> list[ir.Expr]:
        visited: set[int] = set()
        result: list[ir.Expr] = []
        while pending:
            current = pending.pop()
            assert isinstance(current, (ir.Var, ir.IterArg))
            if current.unique_id in visited:
                continue
            visited.add(current.unique_id)
            tile = current.type
            assert isinstance(tile, ir.TileType)
            if tile.memref is not None:
                result.append(current)
            else:
                sources = self.edges.get(current.unique_id)
                assert sources, f"Unrooted selected handle: {current.name_hint}"
                pending.extend(sources)
        assert result, "A selected handle must reach concrete storage"
        return result

    def references_allocation(self, expected: ir.MemRef) -> bool:
        """Prove that a reference graph protects this entire source allocation."""
        references: list[ir.Expr] = [
            value
            for value in self.values.values()
            if isinstance(value.type, ir.TileType) and value.type.memref is None
        ]
        if not references:
            return False
        for owner in self._owners(references):
            tile = owner.type
            assert isinstance(tile, ir.TileType) and tile.memref is not None
            if ir.MemRef.same_allocation(tile.memref, expected):
                return True
        return False

    def assert_window(self, value: ir.Expr, expected: ir.MemRef) -> None:
        """Check every possible owner, including its complete byte/slot window."""
        for owner in self.owners(value):
            tile = owner.type
            assert isinstance(tile, ir.TileType) and tile.memref is not None
            actual = tile.memref
            assert ir.MemRef.same_allocation(actual, expected)
            assert ir.structural_equal(actual.byte_offset_, expected.byte_offset_)
            assert actual.size_ == expected.size_
            assert actual.slot_count_ == expected.slot_count_
            if expected.slot_index_ is None:
                assert actual.slot_index_ is None
            else:
                assert actual.slot_index_ is not None
                assert ir.structural_equal(actual.slot_index_, expected.slot_index_)

    @staticmethod
    def assert_descriptor(value: ir.Expr, expected: ir.Expr) -> None:
        """Preserve the selected value's declared view independently of storage."""
        tile, original = value.type, expected.type
        assert isinstance(tile, ir.TileType) and isinstance(original, ir.TileType)
        assert tile.shape == original.shape and tile.dtype == original.dtype
        assert tile.memory_space == original.memory_space
        assert tile.get_effective_tile_view() == original.get_effective_tile_view()
