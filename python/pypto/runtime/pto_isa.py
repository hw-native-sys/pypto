# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Resolution of the pinned PTO-ISA checkout.

``runtime/pto_isa.pin`` is the single source of truth for the revision, and
``simpler_setup.pto_isa`` owns the only resolver. PyPTO deliberately keeps no
resolver of its own: a second implementation inevitably drifts, and
:class:`~pypto.runtime.kernel_compiler.KernelCompiler` skips its own revision
re-check *because* the resolver already guaranteed the pin -- handing it an
off-pin tree silently compiles kernels against the wrong ISA headers.

This module is deliberately separate from ``device_runner``. Resolving the pin
is a build-input question, not a device-execution one, so it must stay
importable in a codegen-only installation that has no ``simpler`` and therefore
no ``task_interface`` extension. Nothing here imports ``simpler_setup`` at
module scope.
"""

import os
from functools import lru_cache
from pathlib import Path


def ensure_pto_isa_root() -> str:
    """Resolve the pinned PTO-ISA checkout, cloning it when necessary.

    Delegates to :func:`simpler_setup.pto_isa.ensure_pto_isa_root`, for which
    ``pto_isa.pin`` is the single source of truth: it reuses a managed checkout
    only when that checkout is clean and already at the pin, re-clones it
    otherwise, serializes concurrent resolutions with a file lock, and verifies
    ``HEAD`` before returning.

    The resolved path is *exported* as ``PTO_ISA_ROOT`` for downstream consumers
    that build extern CCE kernels and need the ISA include directory. It is
    never *read* back: an ambient value is not the pin.

    Returns:
        Absolute path to the pinned PTO-ISA checkout.

    Raises:
        RuntimeError: If ``pto_isa.pin`` is missing or malformed.
        OSError: If the pinned checkout cannot be obtained.
    """
    resolved = _resolve_pinned_pto_isa_root()
    os.environ["PTO_ISA_ROOT"] = resolved
    return resolved


def pto_isa_include_dir() -> Path:
    """Return the include directory of the pinned PTO-ISA checkout.

    Use this instead of reading ``PTO_ISA_ROOT`` when a ``pl.jit.extern`` kernel
    or an out-of-tree build needs the PTO-ISA headers -- it resolves the pin on
    demand rather than trusting whatever the ambient environment happens to say.

    Example::

        from pypto.runtime import pto_isa_include_dir

        _INCLUDE = pto_isa_include_dir()

    Returns:
        Absolute path to ``<pto-isa>/include``.

    Raises:
        RuntimeError: If ``pto_isa.pin`` is missing or malformed.
        OSError: If the pinned checkout cannot be obtained.
    """
    return Path(ensure_pto_isa_root()) / "include"


@lru_cache(maxsize=1)
def _resolve_pinned_pto_isa_root() -> str:
    """Resolve the pin once per process (the underlying call takes a file lock)."""
    # noqa: PLC0415 -- importing at module scope would make this module, and
    # therefore ``pypto.runtime``, require simpler in a codegen-only install.
    from simpler_setup.pto_isa import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        ensure_pto_isa_root as _resolve,
    )

    return str(Path(_resolve(verbose=True)).resolve())
