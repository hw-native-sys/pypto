# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Bootstrap helpers for worktree-local PyPTO imports."""

import importlib.util
import sys
from pathlib import Path


def _bootstrap_pypto_core() -> None:
    pkg_dir = Path(__file__).resolve().parent
    bootstrap_init = pkg_dir / "pypto_core" / "__init__.py"
    if not bootstrap_init.exists():
        return

    package_name = __name__.rsplit(".", 1)[0]
    module_name = f"{package_name}.pypto_core"
    if module_name in sys.modules:
        return

    spec = importlib.util.spec_from_file_location(
        module_name,
        bootstrap_init,
        submodule_search_locations=[str(bootstrap_init.parent)],
    )
    if spec is None or spec.loader is None:
        return

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


_bootstrap_pypto_core()
