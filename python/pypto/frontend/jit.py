#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
import os
import subprocess
import sys
from pathlib import Path
import ctypes
import functools
import inspect
import torch
from pypto.pypto_core.codegen import PTOCodegen

_jit_functions = {}
_kernel_functions = {}
_compiled_cache = {}
_default_device = "cpu"


def _get_mlir_code(result):
    """Normalize generate() result to MLIR string (support both str and dict)."""
    return result if isinstance(result, str) else "".join(result.values())


def compile(prog, clean_up=False, timeout=20):
    Path("./build").mkdir(parents=True, exist_ok=True)
    ir_path = "./build/temp.pto"  # TODO: use Python `tempfile` module
    raw_cpp_path = "./build/temp_generated.cpp"
    edited_cpp_path = "./build/temp_edited.cpp"
    lib_path = "./build/temp_lib.so"

    # step 1, Program -> PtoAs-mlir
    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(prog))
    print("mlir code is:")
    print(mlir_code)
    return
    with open(ir_path, "w") as f:
        f.write(mlir_code)

    # step 2, IR -> CPP
    # TODO: use `ptoas --enable-insert-sync` so no need for explicit sync in frontend
    # need https://github.com/zhangstevenunity/PTOAS/issues/10
    subprocess.run(
        ["ptoas", ir_path, "-o", raw_cpp_path],
        timeout=timeout, stderr=subprocess.DEVNULL
    )

    # Step 3, preprocess cpp source
    # TODO: should extend `ptoas` emitc to largely replace this ad-doc editing
    content = Path(raw_cpp_path).read_text(encoding="utf-8")
    Path(edited_cpp_path).write_text(content, encoding="utf-8")

    # Step 4, cpp -> so
    PTO_LIB_PATH = os.environ["PTO_LIB_PATH"]
    ASCEND_HOME_PATH = os.environ.get("ASCEND_HOME_PATH")
    LD_LIB_PATH = ASCEND_HOME_PATH + "/lib64/"
    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        "--npu-arch=dav-2201",
        "-DMEMORY_BASE",  # here hardcoded for A2A3; TODO: expose this option to jit interface
        "-O2",
        "-std=c++17",
        f"-I{PTO_LIB_PATH}/include",
    ]

    subprocess.run(
        ["bisheng", *flags, edited_cpp_path, "-L", LD_LIB_PATH, "-lruntime", "-o", lib_path],
        timeout=timeout
    )

    if clean_up:
        os.remove(ir_path)
        os.remove(raw_cpp_path)
        os.remove(edited_cpp_path)

    return lib_path


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path, clean_up=False):
    lib = ctypes.CDLL(lib_path)

    default_block_dim = 1  # TODO: extend kernel to multi-core

    def func_wrapper(
        *tensors,
        block_dim=default_block_dim,
        stream=None
    ):
        for i, t in enumerate(tensors):
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"argument{i} must be torch.Tensor, real type is: {type(t)}")
            
        if stream is None:
            stream = torch.npu.current_stream()
        ptrs = [torch_to_ctypes(t) for t in tensors]
        # TODO (important): matching call signature to arg list information in Python `build_module`
        lib.call_kernel(
            block_dim,
            stream._as_parameter_,
            *ptrs
        )

    if clean_up:
        os.remove(lib_path)

    return func_wrapper


def launch(stream=None, block_dim=1, compiled_result="", *tensors):
    if compiled_result is "":
        raise RuntimeError("Compile error is empty")
    
    # compiled_func = load_lib(compiled_result)
    # if stream is None:
    #     stream = torch.npu.current_stream()
    # ptrs = [torch_to_ctypes(t) for t in tensors]
    # compiled_func(stream._as_parameter_, block_dim, ptrs)


def jit(target=None, optimize: bool = True, cache: bool = True, 
    preprocess: bool = True,
    *dargs, **kwargs):
    """
    @pto.jit 装饰器: 标记函数为JIT编译函数
    
    参数:
        func: 要装饰的函数
        target: 编译目标 ('cpu', 'npu')
        optimize: 是否启用优化
        cache: 是否缓存编译结果
    
    示例:
        @pto.jit
        def add(a, b):
            workspace_size = 100
            pto.launch(add_kernel)
            return workspace_size
    """
    
    def decorator(f):
        # 获取函数信息
        name = f.__name__
        signature = inspect.signature(f)
        
        # 获取源代码（去除装饰器行）
        try:
            source_lines = inspect.getsource(f).split('\n')
            # 移除装饰器行
            source_lines = [line for line in source_lines 
                          if '@pto.jit' not in line and '@pto.kernel' not in line]
            source_code = '\n'.join(source_lines).strip()
        except:
            source_code = "<source unavailable>"
        print("In jit decorator, registering JIT function:", name)
        # 存储JIT函数信息
        _jit_functions[name] = {
            'func': f,
            'name': name,
            'signature': signature,
            'source_code': source_code,
            'target': target or 'cpu',
            'optimize': optimize,
            'cache': cache,
            'kwargs': kwargs
        }
        
        @functools.wraps(f)
        def wrapper(*args, **kwargs_):
            # if name in _compiled_cache:
            #     return _compiled_cache[name](*args, **kwargs_)
            
            # jit函数没有命中缓存，回退到Python执行
            return f(*args, **kwargs_)
        
        return wrapper
    
    if len(dargs) == 1 and callable(dargs[0]):
        return decorator(dargs[0])
    else:
        return decorator