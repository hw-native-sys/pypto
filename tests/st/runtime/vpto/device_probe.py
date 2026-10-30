# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Bounded-child ACL probe; importing this module never initializes a device."""

import ctypes
import sys


def probe_device(device_id: int) -> str:
    """Select the reserved device and query its SoC without torch_npu."""
    if device_id < 0:
        raise ValueError(f"Device ID must be nonnegative, got {device_id}")
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclInit.argtypes = [ctypes.c_char_p]
    acl.aclInit.restype = ctypes.c_int
    acl.aclFinalize.argtypes = []
    acl.aclFinalize.restype = ctypes.c_int
    acl.aclrtSetDevice.argtypes = [ctypes.c_int32]
    acl.aclrtSetDevice.restype = ctypes.c_int
    acl.aclrtResetDevice.argtypes = [ctypes.c_int32]
    acl.aclrtResetDevice.restype = ctypes.c_int
    acl.aclrtGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int32)]
    acl.aclrtGetDevice.restype = ctypes.c_int
    acl.aclrtGetSocName.argtypes = []
    acl.aclrtGetSocName.restype = ctypes.c_char_p

    def check(status: int, operation: str) -> None:
        if status != 0:
            raise RuntimeError(f"{operation} failed for device {device_id}: ACL error {status}")

    check(acl.aclInit(None), "aclInit")
    try:
        check(acl.aclrtSetDevice(device_id), "aclrtSetDevice")
        try:
            selected = ctypes.c_int32(-1)
            check(acl.aclrtGetDevice(ctypes.byref(selected)), "aclrtGetDevice")
            if selected.value != device_id:
                raise RuntimeError(f"Selected device {selected.value}, expected {device_id}")
            name = acl.aclrtGetSocName()
            if not name:
                raise RuntimeError(f"aclrtGetSocName returned no name for device {device_id}")
            return name.decode("utf-8")
        finally:
            check(acl.aclrtResetDevice(device_id), "aclrtResetDevice")
    finally:
        check(acl.aclFinalize(), "aclFinalize")


if __name__ == "__main__":
    print(f"VPTO_DEVICE_NAME={probe_device(int(sys.argv[1]))}", flush=True)
