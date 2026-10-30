# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Ablate literal rendering and compare cache-free and cached algorithms.

Run like hbg_csa_cost_audit.py with CSA_COST_AUDIT=bench and --metadata produce.
All arms keep the normal resolver, namespace reads and folding behavior.
"""

import ast
import importlib
from typing import Any

import hbg_csa_cost_audit as audit


def plain_literal(value: Any) -> bool:
    kind = type(value)
    # Identity comparisons avoid invoking arbitrary metaclass __eq__ methods.
    return value is None or kind is bool or kind is int or kind is str


class DesignCandidates(audit.Candidates):
    stages = (
        "current_a",
        "current_b",
        "source_only",
        "callsite_only",
        "global_text",
        "request_text",
        "direct_ast",
        "direct_value",
    )

    def __init__(self) -> None:
        super().__init__()
        self.source = importlib.import_module("pypto.jit._source")
        self.kind = "global_text"

    def render(self, value: Any) -> str | None:
        if self.kind == "direct_value":
            return repr(value) if plain_literal(value) else self.original_constant(value)
        if self.kind == "direct_ast":
            node = self.specializer._render_free_value(value)
            if node is None:
                return None
            if type(node) is ast.Constant and node.kind is None and plain_literal(node.value):
                return repr(node.value)
            return ast.unparse(node)
        if self.kind == "request_text" and audit.eligible(value):
            snapshot = self.source._ACTIVE_NAMESPACES.get()
            if snapshot is not None:
                key = (self, (type(value), value))
                if key not in snapshot.values:
                    snapshot.values[key] = self.original_constant(value)
                return snapshot.values[key]
        return super().render(value) if self.kind != "request_text" else self.original_constant(value)

    def install(self, stage: str) -> None:
        self.restore()
        assert stage in self.stages, stage
        self.kind = stage
        if stage in ("current_a", "current_b"):
            return
        if stage != "callsite_only":
            for module in (self.specializer, self.binding, self.decorator):
                self.set(module, "free_name_source", self.free_name)
        if stage != "source_only":
            self.set(self.specializer, "constant_source", self.render)


def main() -> None:
    original = audit.Candidates
    audit.Candidates = DesignCandidates
    try:
        audit.main()
    finally:
        audit.Candidates = original


if __name__ == "__main__":
    main()
