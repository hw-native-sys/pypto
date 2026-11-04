# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Environment policies shared by cache discovery and compilation consumers."""

import os
from pathlib import Path

# Supported deployments use injection for allocation/observation, without
# changing compiler selection, inputs, or generated semantics. This is a
# deployment contract, not an audit or an allowlist of shared-library names.
# Normalizing probes prevents injected dependencies, loader diagnostics and
# profiling output from becoming installation evidence or extra inventory work.
_PROBE_IGNORED_ENV = (
    "LD_PRELOAD",
    "LD_AUDIT",
    "LD_DEBUG",
    "LD_DEBUG_OUTPUT",
    "LD_PROFILE",
    "LD_PROFILE_OUTPUT",
)
_DISABLED_VALUES = frozenset(("0", "false", "no"))


def toolchain_probe_env(*, c_locale: bool = False) -> dict[str, str]:
    """Copy the environment for an existing metadata probe, never a build.

    Called only when a probe already needs to run, after identity/version memo
    lookup. The parent environment and actual compiler processes retain the
    caller's settings. Library search paths still select real dependencies.
    """
    environment = os.environ.copy()
    for name in _PROBE_IGNORED_ENV:
        environment.pop(name, None)
    if c_locale:
        # Discovery parses compiler diagnostics; translated markers are not
        # interchangeable with the C-locale grammar used by those parsers.
        environment.update(LC_ALL="C", LANG="C", LANGUAGE="")
    return environment


def env_flag_disabled(value: str | None) -> bool:
    """Recognize the existing opt-out spellings; unset/empty retain defaults."""
    return value is not None and value.strip().lower() in _DISABLED_VALUES


def explicit_env_request(value: str | None) -> bool:
    """Distinguish an explicit enabled request from unset or disabled flags."""
    return value is not None and not env_flag_disabled(value)


def default_output_root() -> Path:
    """Select placement for new private outputs without creating directories.

    Read only when creating a new build/runtime object. Existing objects retain
    their directories, and shared artifacts remain under the cache root.
    """
    return Path(os.environ.get("PYPTO_PROG_BUILD_DIR") or "build_output")
