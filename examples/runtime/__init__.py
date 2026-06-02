# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""User-facing examples for the explicit runtime dispatch APIs.

See :mod:`pypto.runtime.ChipWorker` for the dispatch surface used here:
``run(compiled, *args)`` for one-shot dispatch and ``register(compiled)``
returning a :class:`~pypto.runtime.RegistrationHandle` for hot loops.

Examples in this directory:

- ``explicit_dispatch.py`` — three end-to-end patterns:
   * ``mode_a_inference_service``: pre-register multiple compiled kernels and
     dispatch by name (recommended for serving runtimes).
   * ``mode_b_training_loop``: long-lived weight ``DeviceTensor`` + per-step
     inputs, hot loop on a single registration handle.
   * ``mode_c_register_dispatch_overhead``: verify that ``register`` warms the
     callable cache once (``aicpu_dlopen_count`` does not grow per dispatch).
"""
