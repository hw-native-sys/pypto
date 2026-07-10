# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared helpers for the orchestration-codegen test files."""

import difflib
import re
import textwrap

from pypto import backend, codegen, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import ir

SELF_ALIAS_RE = re.compile(r"\bauto\s+(\w+)\s*=\s*\1\s*;")


def assert_code_equal(actual: str, expected: str) -> None:
    """Compare generated code against expected output, with unified diff on failure."""
    actual_stripped = actual.strip()
    expected_stripped = textwrap.dedent(expected).strip()
    if actual_stripped != expected_stripped:
        diff = "\n".join(
            difflib.unified_diff(
                expected_stripped.splitlines(),
                actual_stripped.splitlines(),
                fromfile="expected",
                tofile="actual",
                lineterm="",
            )
        )
        raise AssertionError(f"Code mismatch:\n{diff}")


def _ensure_return_params_explicit(program):
    """Phase-5 invariant: codegen requires IRProperty.ReturnParamsExplicit.

    Orchestration codegen reads each callee's return->param map straight off its
    ReturnStmt (pointer identity). NormalizeReturnOrder establishes that form.
    Tests that hand-build IR (without going through PassManager) need to invoke
    it before codegen; it is a no-op when the program already went through the
    pass pipeline.
    """
    return passes.normalize_return_order()(program)


def _ensure_arg_directions(program):
    """Phase-5 invariant: codegen requires Call.arg_directions to be populated.

    Tests that hand-build IR (without going through PassManager) need to invoke
    DeriveCallDirections before codegen so the Call sites carry explicit
    ArgDirection vectors. This helper makes that step a no-op when the program
    was already produced by the pass pipeline.
    """
    return passes.derive_call_directions()(program)


def _finalize_for_codegen(program):
    """Run the two codegen-entry passes on a hand-built program.

    MaterializeRuntimeScopes gives the orchestration function body and for/if
    bodies explicit AUTO RuntimeScopeStmt nodes (codegen no longer emits implicit
    PTO2_SCOPE() wrappers). ClassifyIterArgCarry then stamps each ForStmt's
    iter_arg carry plan, which codegen reads instead of deriving. Both are
    codegen preconditions and both are no-ops when the program already went
    through the pass pipeline. Must run after DeriveCallDirections (a declared
    requirement of both passes).
    """
    return passes.classify_iter_arg_carry()(passes.materialize_runtime_scopes()(program))


def _generate_orch_code(program) -> str:
    """Generate orchestration code using backend-agnostic codegen."""
    program = _finalize_for_codegen(_ensure_arg_directions(_ensure_return_params_explicit(program)))
    for func in program.functions.values():
        if func.func_type == ir.FunctionType.Orchestration:
            result = codegen.generate_orchestration(program, func)
            return result.code
    raise ValueError("No orchestration function found in program")


def _generate_orch_result(program) -> "codegen.OrchestrationResult":
    """Generate orchestration result using backend-agnostic codegen."""
    program = _finalize_for_codegen(_ensure_arg_directions(_ensure_return_params_explicit(program)))
    for func in program.functions.values():
        if func.func_type == ir.FunctionType.Orchestration:
            return codegen.generate_orchestration(program, func)
    raise ValueError("No orchestration function found in program")


def _out_of_scope_tensor_refs(code: str) -> list[str]:
    """Return tensor identifiers used outside the C++ brace scope that declares
    them — a lightweight stand-in for ``g++`` that catches the
    ``'<name>' was not declared in this scope`` class of orchestration codegen
    bugs (issues #1697, #1713) without invoking a real C++ compiler.

    Walks brace scopes recording the tensor identifiers declared in each, then
    flags any *use* that names a declared tensor not visible at the use site.
    Three use shapes are scanned (a name escaping a closed ``PTO2_SCOPE`` block
    can surface as any of them):

      * ``add_input/output/inout/no_dep(X)``         — call-arg reads (#1697)
      * ``X.reshape/.view/.transpose/.assemble/.slice/.get_ref(...)`` —
        method-receiver reads (the after-scope ``buf_rv.reshape(...)`` shape that
        an ``add_*``-only checker missed, #1713)
      * ``... = X;`` / ``... = X.method(...)``        — assignment-RHS reads

    A used name out of scope is flagged when it is *either* declared as a tensor
    somewhere (``declared_anywhere``) *or* an SSA-versioned tensor temp
    (``__ssa_v``/``__rv``/``__window``/...). The latter is checked independently
    so a dangling reference to a name whose declaration was wrongly *collapsed
    away* — which would never land in ``declared_anywhere`` — is still reported.
    Numeric literals (``= 0;``), casts, and scalar locals carry neither marker,
    so they never yield false positives.
    """
    decl_re = re.compile(r"\b(?:const\s+Tensor\s*&|Tensor|TaskOutputTensors|Arg)\s+(\w+)")
    declared_anywhere = set(decl_re.findall(code))
    # An SSA-versioned tensor temp is unambiguously a tensor regardless of whether
    # its declaration still exists, so an out-of-scope reference to one is always
    # a bug worth flagging (independent of declared_anywhere).
    ssa_tensor = re.compile(r"__(?:ssa_v\d|rv\b|rv_v\d|window|windowed|assembled)")
    use_add = re.compile(r"\badd_(?:input|output|inout|no_dep)\(\s*(\w+)\s*\)")
    use_method = re.compile(r"\b(\w+)\s*\.(?:reshape|view|transpose|assemble|slice|get_ref)\s*\(")
    use_rhs = re.compile(r"=\s*(\w+)\s*[;.]")
    scopes: list[set[str]] = [set()]
    bad: list[str] = []
    for raw in code.splitlines():
        line = raw.strip()
        # Declarations and uses are each emitted on their own line (never sharing
        # a line with a scope brace), so resolve them against the current scope
        # set first. Declarations are recorded before uses so a ``const Tensor& Y
        # = X`` line registers Y while still checking the RHS read of X.
        for m in decl_re.finditer(line):
            scopes[-1].add(m.group(1))
        names = [m.group(1) for m in use_add.finditer(line)]
        names += [m.group(1) for m in use_method.finditer(line)]
        names += [m.group(1) for m in use_rhs.finditer(line)]
        for name in names:
            if any(name in s for s in scopes):
                continue
            if name in declared_anywhere or ssa_tensor.search(name):
                bad.append(name)
        # Apply braces in source order so a ``} else {`` line closes the prior
        # block then opens a fresh one (counting separately would mis-nest it).
        for ch in line:
            if ch == "{":
                scopes.append(set())
            elif ch == "}" and len(scopes) > 1:
                scopes.pop()
    return bad


def _run_default_pipeline_with_auto_scope_deps(program):
    """Run the default pipeline with AUTO-scope auto-deps enabled in-place."""
    return PassManager.get_strategy(
        OptimizationStrategy.Default,
        analyze_auto_scopes_for_deps=True,
    ).run_passes(program)


def _generate_orch_full_pipeline(
    program_cls,
    *,
    analyze_auto_scopes_for_deps: bool = False,
    allow_relaxed_verification: bool = False,
) -> str:
    """Run the Default pass pipeline + orchestration codegen on the orch func.

    The issue #1577 witnesses use ``pl.submit`` against ``@pl.function(InCore)``
    kernels, which only reach orchestration form after the full Default pipeline
    (inline / outline / SSA / materialize scopes). Mirrors
    ``test_array_codegen._generate_pto`` but targets the Orchestration function.
    """
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    pm = PassManager.get_strategy(
        OptimizationStrategy.Default,
        analyze_auto_scopes_for_deps=analyze_auto_scopes_for_deps,
    )
    if allow_relaxed_verification:
        # Scope to BASIC (property) verification for tests that hit the known
        # ``pl.submit(..., deps=[...])`` print->parse roundtrip gap after
        # DeriveCallDirections. Property verification still runs so real IR
        # invariant violations are caught.
        with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
            optimized = pm.run_passes(program_cls)
    else:
        optimized = pm.run_passes(program_cls)
    for func in optimized.functions.values():
        if func.func_type == ir.FunctionType.Orchestration:
            return codegen.generate_orchestration(optimized, func).code
    raise AssertionError("no Orchestration function found in program")


def _generate_orch_from_transformed_program(program) -> str:
    program = _finalize_for_codegen(program)
    for func in program.functions.values():
        if func.func_type == ir.FunctionType.Orchestration:
            return codegen.generate_orchestration(program, func).code
    raise AssertionError("no Orchestration function found in program")
