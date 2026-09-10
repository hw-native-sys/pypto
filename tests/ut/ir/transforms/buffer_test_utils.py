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
            elif isinstance(statement, (ir.ForStmt, ir.WhileStmt)):
                pending.append(statement.body)
