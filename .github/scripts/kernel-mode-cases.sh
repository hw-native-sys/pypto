# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Sourced by CI and local validation to use the same explicit regression set.
case "${1:-}" in
  unit)
    cases=(
      tests/ut/test_optional_runtime_imports.py
      tests/ut/test_kernel_ci_download.py
      tests/ut/test_kernel_ci_results.py
      tests/ut/jit/test_kernel_eager.py
      tests/ut/jit/test_cache_config.py
      tests/ut/jit/test_jit_compile_extraction.py
      tests/ut/runtime/test_kernel_abi.py
      tests/ut/runtime/test_kernel_compiler.py
      tests/ut/runtime/test_kernel_context.py
      tests/ut/runtime/test_kernel_shutdown.py
      tests/ut/runtime/test_prebuilt_artifact.py
      tests/ut/torch/test_init.py
      tests/ut/torch/test_interop.py
      tests/ut/torch/test_launch.py
      tests/ut/torch/test_registration.py
      tests/ut/torch/test_capture.py
    )
    ;;
  eager)
    cases=(
      'tests/st/runtime/kernel/test_jit_eager.py::test_jit_eager[eager-0]'
      'tests/st/runtime/kernel/test_jit_eager.py::test_jit_eager[eager-1]'
      'tests/st/runtime/kernel/test_jit_eager.py::test_jit_eager[program-1]'
      'tests/st/runtime/kernel/test_torch_ops.py::test_torch_ops[1-compile]'
      'tests/st/runtime/kernel/test_torch_launch.py::test_torch_kernel_launch[failure-0]'
      'tests/st/runtime/kernel/test_torch_launch.py::test_torch_kernel_launch[failure-1]'
      'tests/st/runtime/kernel/test_torch_launch.py::test_torch_kernel_launch[delayed-1]'
      'tests/st/runtime/kernel/test_kernel_shutdown.py::test_kernel_shutdown[normal-1]'
      'tests/st/runtime/kernel/test_hot_path.py::test_kernel_hot_path[0]'
      'tests/st/runtime/kernel/test_hot_path.py::test_kernel_hot_path[1]'
      'tests/st/runtime/kernel/test_dfx.py'
    )
    ;;
  capture)
    cases=(
      'tests/st/runtime/kernel/test_capture.py::test_capture[1-cold-jit]'
      'tests/st/runtime/kernel/test_capture.py::test_capture[0-single-jit]'
      'tests/st/runtime/kernel/test_capture.py::test_capture[1-owners-torch_ops]'
      'tests/st/runtime/kernel/test_capture.py::test_capture[1-shutdown-torch_ops]'
      'tests/st/runtime/kernel/test_capture.py::test_capture_entry_interop[1-build-dir-mixed]'
      'tests/st/runtime/kernel/test_torch_ops.py::test_torch_ops[1-capture]'
    )
    ;;
  *) echo "Expected unit, eager or capture suite, got '${1:-}'" >&2; return 2 ;;
esac
