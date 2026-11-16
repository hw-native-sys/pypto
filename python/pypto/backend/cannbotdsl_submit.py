# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Generate host-side CANNBotDSL batch ordering for an authorized TMR submit callback."""

import hashlib
import json

from pypto.jit.external.contracts import PreparedKernel, SourceArtifact
from pypto.jit.external.identity import prepared_digest
from pypto.jit.external.validation import ContractError

from .cannbotdsl_abi import build_binding_checker, build_conflict_checker
from .tmr_bridge import _file


def build_ordered_submitter(prepared: tuple[PreparedKernel, ...]) -> SourceArtifact:
    """Generate a synchronous submission helper for a fixed, ordered kernel batch.

    The generated function accepts an array of CoreTaskArgs pointers and a
    callback (index, args) -> TaskId that synchronously consumes each binding.
    It validates the whole batch before submission and adds DEP_WAIT for every
    conflicting predecessor, preserving existing dependency flags. It restores
    each argument's dependency pointers and early-dispatch setting after submit.

    The caller must keep producers referenceable in one active TMR scope and
    keep all bindings fixed and storage alive through execution. Calls outside
    the batch need independent ordering. The callback must submit exactly the
    indexed prepared kernel with the supplied args and return its fresh TaskId.
    It must not retain args/dependency pointers or mutate batch bindings.
    Submission failure stops the batch but cannot roll back submitted tasks.
    Resource, loading and completion authorization remains the caller's duty.
    """
    if type(prepared) is not tuple or not prepared:
        raise ContractError("CANNBotDSL submitter: expected a nonempty tuple of prepared kernels")
    count = len(prepared)
    digest = hashlib.sha256(json.dumps([prepared_digest(p) for p in prepared]).encode()).hexdigest()
    symbol = f"pypto_cannbotdsl_submit_{digest}"
    guard = f"{symbol.upper()}_HPP"
    files: dict[str, bytes] = {}
    checks: list[str] = []
    pairs: list[tuple[int, int, str]] = []
    # Include one self pair per kernel too: this validates the isolation policy
    # even for singleton batches. Digest paths and named guards deduplicate
    # repeated kernels and permit composition with standalone pair artifacts.
    for i, kernel in enumerate(prepared):
        checker = build_binding_checker(kernel)
        checks.append(checker.symbol)
        for j in range(i + 1):
            pair = build_conflict_checker(prepared[j], kernel)
            entry = f"{pair.symbol}.hpp"
            files.update({entry if f.path == pair.entry else f.path: f.content for f in pair.files})
            if j != i:
                pairs.append((j, i, pair.symbol))
    includes = [path for path in files if not path.startswith("checks/")]
    lines = ["#pragma once", f"#ifndef {guard}", f"#define {guard}"]
    lines += ["#include <array>", "#include <vector>", *(f'#include "{path}"' for path in includes)]
    lines += [
        f"template<class Submit> inline std::array<TaskId, {count}> {symbol}(",
        f"        const std::array<CoreTaskArgs*, {count}>& args, Submit&& submit) {{",
        f"    std::array<std::vector<TaskId>, {count}> deps;",
        f"    std::array<std::vector<DepFlags>, {count}> kinds;",
        f"    std::array<uint32_t, {count}> counts{{}};",
        f"    bool conflicts[{count}][{count}]{{}};",
        # Allocate/copy all dependency buffers before the first side effect.
        # Existing RETAIN bits must survive even for disjoint cache lines.
        f"    for (size_t i = 0; i < {count}; ++i) {{",
        '        always_assert(args[i] != nullptr && "CANNBotDSL batch: null arguments");',
        '        always_assert(!args[i]->has_error && "CANNBotDSL batch: invalid arguments");',
        "        const auto n = args[i]->explicit_dep_count();",
        f'        always_assert(n <= UINT32_MAX - {count} && "CANNBotDSL batch: dependency overflow");',
        "        deps[i].resize(size_t(n) + i);",
        "        kinds[i].resize(size_t(n) + i);",
        "        counts[i] = n;",
        "        for (uint32_t d = 0; d < n; ++d) {",
        "            deps[i][d] = args[i]->explicit_dep(d);",
        "            kinds[i][d] = args[i]->explicit_dep_kind(d);",
        '            always_assert(deps[i][d].is_valid() && "CANNBotDSL batch: invalid dependency");',
        "            always_assert((unsigned(kinds[i][d]) & ~3U) == 0 &&",
        '                          "CANNBotDSL batch: unknown dependency flags");',
        "        }",
        "    }",
        *(f"    {checker}(*args[{i}]);" for i, checker in enumerate(checks)),
        *(f"    conflicts[{i}][{j}] = {pair}(*args[{j}], *args[{i}]);" for j, i, pair in pairs),
        # CoreTaskArgs cannot be copied. Temporarily bind owned dependency
        # arrays, restoring borrowed original pointers even if submit throws.
        "    struct Restore {",
        "        CoreTaskArgs& arg;",
        "        const TaskId* deps;",
        "        const DepFlags* kinds;",
        "        uint32_t count;",
        "        bool early;",
        "        ~Restore() {",
        "            arg.set_dependencies(nullptr, 0);",
        "            if (kinds) arg.set_dependencies_with_kinds(deps, kinds, count);",
        "            else arg.set_dependencies(deps, count);",
        "            arg.set_allow_early_resolve(early);",
        "        }",
        "    };",
        f"    std::array<TaskId, {count}> ids{{}};",
        f"    for (size_t i = 0; i < {count}; ++i) {{",
        "        for (size_t j = 0; j < i; ++j) {",
        "            if (!conflicts[i][j]) continue;",
        "            uint32_t d = 0;",
        "            while (d < counts[i] && deps[i][d] != ids[j]) ++d;",
        "            if (d == counts[i]) {",
        "                deps[i][d] = ids[j];",
        "                kinds[i][d] = DEP_NONE;",
        "                ++counts[i];",
        "            }",
        "            kinds[i][d] |= DEP_WAIT;",
        "        }",
        "        auto& arg = *args[i];",
        "        Restore restore{arg, arg.explicit_deps_data(), arg.explicit_dep_kinds_data(),",
        "                        arg.explicit_dep_count(), arg.allow_early_resolve()};",
        "        arg.set_dependencies(nullptr, 0);",
        "        arg.set_dependencies_with_kinds(deps[i].data(), kinds[i].data(), counts[i]);",
        # Cache-line safety requires completion-before-start, not early staging
        # with a data-availability hint whose stronger prerequisites are unknown.
        "        arg.set_allow_early_resolve(false);",
        "        ids[i] = submit(i, arg);",
        '        always_assert(ids[i].is_valid() && "CANNBotDSL batch: submission failed");',
        "        for (size_t j = 0; j < i; ++j)",
        '            always_assert(ids[i] != ids[j] && "CANNBotDSL batch: reused task ID");',
        "    }",
        "    return ids;",
        "}",
        "#endif",
        "",
    ]
    files["ordered_submit.hpp"] = "\n".join(lines).encode()
    return SourceArtifact(
        tuple(_file(path, content) for path, content in files.items()), "ordered_submit.hpp", symbol
    )
