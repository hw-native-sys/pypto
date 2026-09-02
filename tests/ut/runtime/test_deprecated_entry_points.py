# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The directory-driven execute entry points warn, and still work.

``execute_compiled`` and ``execute_distributed_compiled`` are deprecated in
favour of reconstructing the artifact -- ``CompiledProgram.from_dir`` /
``DistributedCompiledProgram.from_dir`` -- and calling it. Both remain exported
for a deprecation cycle, so each must keep forwarding to the same
implementation the supported path uses; a shim that warns but silently changes
behaviour is worse than no shim.
"""

from ctypes import _SimpleCData
from unittest.mock import MagicMock, patch

import pytest
import torch
from pypto.runtime import DeviceTensor, RunConfig, execute_compiled, execute_distributed_compiled

_L3_FROM_DIR = "pypto.ir.distributed_compiled_program.DistributedCompiledProgram.from_dir"


def test_execute_compiled_warns_and_forwards(tmp_path):
    args: list[torch.Tensor | DeviceTensor | _SimpleCData] = [torch.zeros(4)]
    config = RunConfig(platform="a2a3sim", device_id=2)

    with patch("pypto.runtime.runner._execute_compiled") as impl:
        with pytest.warns(DeprecationWarning, match=r"CompiledProgram\.from_dir"):
            execute_compiled(
                tmp_path,
                args,
                platform="a2a3sim",
                device_id=2,
                dfx=config.dfx_options(),
                config=config,
            )

    impl.assert_called_once()
    assert impl.call_args.args[0] == tmp_path
    assert impl.call_args.args[1] is args
    assert impl.call_args.kwargs["platform"] == "a2a3sim"
    assert impl.call_args.kwargs["device_id"] == 2
    assert impl.call_args.kwargs["config"] is config


def test_execute_distributed_compiled_warns_and_forwards(tmp_path):
    args = [torch.zeros(4), torch.zeros(4)]
    config = RunConfig(platform="a2a3")
    reconstructed = MagicMock(name="DistributedCompiledProgram")

    with patch(_L3_FROM_DIR, return_value=reconstructed) as from_dir:
        with pytest.warns(DeprecationWarning, match=r"DistributedCompiledProgram\.from_dir"):
            result = execute_distributed_compiled(tmp_path, args, config=config, platform="a2a3")

    from_dir.assert_called_once_with(tmp_path, platform="a2a3", distributed_config=None)
    reconstructed.assert_called_once_with(*args, config=config)
    assert result is reconstructed.return_value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
