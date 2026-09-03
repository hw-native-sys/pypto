# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reserved ``Function.attrs`` keys shared between the DSL, the backend and the JIT.

Every key here is also declared in C++ as an ``inline constexpr const char*`` in
``include/pypto/ir/function.h``, which carries the authoritative lifecycle
documentation for each one (which pass writes it, which reads it, whether it is
ever stripped). This module is the Python half of the same pair so a rename has
one place to change per layer instead of one place per call site.

``tests/lint/check_function_attr_key_parity.py`` fails when the two halves
disagree, so a key added or renamed on one side cannot silently drift from the
other.
"""

# AIV kernel running on BOTH vector sub-lanes of a mixed kernel (bool).
DUAL_AIV_DISPATCH_ATTR = "dual_aiv_dispatch"

# InCore function outlined from a scope holding ``pl.split_aiv`` region(s) (bool).
SPLIT_AIV_ATTR = "split_aiv"

# ``pl.split_aiv`` regions already transpose-hazard-checked per region (bool).
SPLIT_AIV_REGION_VALIDATED_ATTR = "split_aiv_region_validated"

# Absolute path to a hand-written external C++ kernel source (str). When present
# on an AIC/AIV function the DSL body is empty (``...``): the compiler assigns the
# function a kernel func_id and emits the orchestration submit as usual, but skips
# PyPTO codegen and instead compiles the referenced ``.cpp`` as the InCore kernel
# (see pto_backend).
EXTERNAL_SOURCE_ATTR = "external_source"

# Package-resource handle (":pypto.runtime.builtins.collectives.<op>") naming the
# builtin template package that supplies a compiler-synthesized kernel's C++
# source (str). Written by LowerL2TensorCollectives on the AIV function it
# synthesizes for a managed CHIP/L2 collective; the backend renders the package's
# ``templates/kernel.cpp.in`` instead of running PyPTO codegen for that function.
BUILTIN_TEMPLATE_DIR_ATTR = "builtin_template_dir"

# Comma-separated ``key=value`` substitutions for the template named by
# ``BUILTIN_TEMPLATE_DIR_ATTR`` (str), e.g. "dtype_cpp=float,ctx_arg_index=5".
BUILTIN_TEMPLATE_VARS_ATTR = "builtin_template_vars"

# Opt-out of automatic ``RuntimeScopeStmt`` materialization (bool). Absent means
# True, so only the opt-out (False) is ever stored.
AUTO_SCOPE_ATTR = "auto_scope"

__all__ = [
    "AUTO_SCOPE_ATTR",
    "BUILTIN_TEMPLATE_DIR_ATTR",
    "BUILTIN_TEMPLATE_VARS_ATTR",
    "DUAL_AIV_DISPATCH_ATTR",
    "EXTERNAL_SOURCE_ATTR",
    "SPLIT_AIV_ATTR",
    "SPLIT_AIV_REGION_VALIDATED_ATTR",
]
