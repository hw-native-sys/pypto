# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Generate a self-contained, whole-Program communication-domain roundtrip case."""

from pathlib import Path

import pypto.language as pl
from pypto import ir, passes

SOURCE = """import pypto.language as pl
import pypto.language.distributed as pld

@pl.program
class P:
    @pl.function(type=pl.FunctionType.Orchestration)
    def chip(self, x: pld.DistributedTensor[[4], pl.FP32]):
        return x

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host(self):
        buf = pld.alloc_window_buffer(16)
        x = pld.window(buf, [4], dtype=pl.FP32)
        for r in pl.range(2):
            self.chip(x, device=r)
        return 0
"""


def host_function(program: ir.Program) -> ir.Function:
    """Get the host function from the complete program."""
    gvar = program.get_global_var("host")
    assert gvar is not None
    return program.functions[gvar]


def main() -> None:
    """Save the actual printer output and compare complete Program objects."""
    output = Path(__file__).resolve().parents[4] / "build" / "comm_domain_program_roundtrip"
    output.mkdir(parents=True, exist_ok=True)
    (output / "input.py").write_text(SOURCE, encoding="utf-8")

    before = pl.parse_program(SOURCE, filename="<self-contained-input>")
    ir.assert_structural_equal(before, pl.parse_program(ir.python_print(before, format=False)))
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        original = passes.materialize_comm_domain_scopes()(before)

    printed = ir.python_print(original, format=False)
    (output / "materialized.py").write_text(printed + "\n", encoding="utf-8")
    restored = pl.parse_program(printed, filename="<whole-program-reparse>")
    restored_text = ir.python_print(restored, format=False)
    (output / "reparsed.py").write_text(restored_text + "\n", encoding="utf-8")

    scope = host_function(original).body
    assert isinstance(scope, ir.CommDomainScopeStmt)
    assert isinstance(scope.body, ir.SeqStmts)
    alloc, view, loop, _ = scope.body.stmts
    assert isinstance(alloc, ir.AssignStmt)
    assert isinstance(view, ir.AssignStmt)
    assert isinstance(view.var.type, ir.DistributedTensorType)
    assert isinstance(loop, ir.ForStmt)
    slot = scope.slots[0]
    assert slot.base is alloc.var
    assert view.var.type.window_buffer is slot
    chip_gvar = original.get_global_var("chip")
    assert chip_gvar is not None and chip_gvar in original.functions

    restored_scope = host_function(restored).body
    assert isinstance(restored_scope, ir.CommDomainScopeStmt)
    restored_body = restored_scope.body
    assert isinstance(restored_body, ir.SeqStmts)
    restored_view = restored_body.stmts[1]
    assert isinstance(restored_view, ir.AssignStmt)
    assert isinstance(restored_view.var.type, ir.DistributedTensorType)
    assert restored_view.var.type.window_buffer is restored_scope.slots[0]

    properties = passes.IRPropertySet()
    property_names = (
        "SSAForm",
        "UseAfterDef",
        "TypeChecked",
        "AssignTypeSymmetry",
        "CommDomainScopesMaterialized",
    )
    for name in property_names:
        properties.insert(getattr(passes.IRProperty, name))
    passes.verify_properties(properties, original, "SelfContainedProgramRepro")

    report = [
        "Comparison roots: original Program P vs restored Program P",
        "Pre-pass whole-program text roundtrip: PASS",
        f"Original program verification: PASS ({', '.join(property_names)})",
        f"Functions defined in program: {sorted(f.name for f in original.functions.values())}",
        f"original.host.body: {type(scope).__name__}",
        f"original.host.body.name_hint: {scope.name_hint}",
        f"original.host.body.devices: {list(scope.devices)}",
        f"original.host.body.slots: {[s.name_hint for s in scope.slots]}",
        f"slot.size: {ir.python_print(slot.size, format=False)}",
        f"slot.base is host allocation LHS: {slot.base is alloc.var}",
        f"x.type.window_buffer is scope.slots[0]: {view.var.type.window_buffer is slot}",
        f"restored.host.body: {type(restored_scope).__name__}",
        f"restored.x.type.window_buffer is restored scope slot: "
        f"{restored_view.var.type.window_buffer is restored_scope.slots[0]}",
        f"Printed text identical after reparse: {printed == restored_text}",
    ]
    ir.assert_structural_equal(original, restored)
    assert ir.structural_hash(original) == ir.structural_hash(restored)
    report.append("Whole-program print -> parse -> same structural IR: PASS (no lowering rerun)")
    (output / "comparison.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
