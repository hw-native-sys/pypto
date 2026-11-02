# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Opt-in A5 EmitC + VPTO validation through the real PyPTO Tensor wrapper.

Set PYPTO_RUN_VPTO_VF_VALIDATION=1 and ASCEND_HOME_PATH to run. Set
PYPTO_VPTO_VF_SIMULATOR=1 for CANN simulation, or PYPTO_VPTO_VF_ONBOARD=1
and PYPTO_VPTO_DEVICE_ID for the separate hardware executable.
This validates argument unpacking and device linking, not the persistent worker.
"""

import json
import os
import re
import runpy
import shlex
import shutil
import signal
import subprocess
from contextlib import suppress
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parents[3]
pytestmark = pytest.mark.skipif(
    os.environ.get("PYPTO_RUN_VPTO_VF_VALIDATION") != "1",
    reason="Explicitly opt in to the installed A5 compiler toolchain",
)


def run(
    output: Path,
    name: str,
    args: list[str | Path],
    env: dict[str, str] | None = None,
    *,
    timeout: int = 180,
) -> str:
    """Retain exact commands and diagnostics, including failures, in the build tree."""
    command = [str(arg) for arg in args]
    (output / f"{name}.command.txt").write_text(shlex.join(command) + "\n")
    with subprocess.Popen(
        command,
        cwd=output,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            log, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Compilers and runtime/profiler workers may spawn children. Bound the
            # entire process group so a timed-out acceptance leaves no orphan worker.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            log, _ = process.communicate()
            (output / f"{name}.log").write_text(f"TIMEOUT after {timeout}s\n{log}")
            raise
        (output / f"{name}.log").write_text(log)
        assert process.returncode == 0, f"{name} exited {process.returncode}:\n{log}"
    return log


def make_wrapper(caller_cpp: str) -> str:
    """Generate the real Tensor/scalar ABI shared by standalone and worker tests."""
    from pypto import DataType, backend, ir  # noqa: PLC0415
    from pypto.backend.pto_backend import _generate_kernel_wrapper  # noqa: PLC0415
    from pypto.ir import IRBuilder  # noqa: PLC0415

    builder = IRBuilder()
    with builder.function("mixed_ub_vadd", type=ir.FunctionType.InCore) as function:
        for name in ("lhs", "rhs", "out"):
            function.param(name, ir.TensorType([1, 1024], DataType.FP32))
        n = function.param("n", ir.ScalarType(DataType.INT32))
        function.return_type(ir.ScalarType(DataType.INT32))
        builder.return_stmt(n)
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend950)
    try:
        wrapper = _generate_kernel_wrapper(function.get_result(), caller_cpp)
    finally:
        backend.reset_for_testing()
    return wrapper


@pytest.fixture(scope="module")
def compiled() -> tuple[Path, Path, list[str]]:
    output = Path(
        os.environ.get("PYPTO_VPTO_VF_OUTPUT", ROOT / "build/vpto-investigation/validated")
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cann = Path(os.environ["ASCEND_HOME_PATH"]).resolve()
    ptoas = shutil.which("ptoas")
    assert ptoas, "ptoas must be on PATH"
    compiler = cann / "bin/bisheng"
    version = run(output, "ptoas-version", [ptoas, "--version"]).strip()
    (output / "environment.json").write_text(
        json.dumps({"ptoas": ptoas, "version": version, "cann": str(cann)}, indent=2)
    )
    run(
        output,
        "emit-helper",
        [
            ptoas,
            SOURCE / "helper.pto",
            "--pto-arch=a5",
            "--pto-backend=vpto",
            "--emit-vpto-llvm-ir",
            "-o",
            output / "helper.ll",
        ],
    )
    run(
        output,
        "emit-caller",
        [
            ptoas,
            SOURCE / "caller.pto",
            "--pto-arch=a5",
            "--pto-backend=emitc",
            "--pto-level=level3",
            "-o",
            output / "caller.cpp",
        ],
    )
    run(
        output,
        "compile-helper",
        [
            compiler,
            "--cce-aicore-arch=dav-c310-vec",
            "--cce-aicore-only",
            "-O2",
            "--cce-generic-addrspace=off",
            "-cce-bitcode-is-aicore",
            "-Wno-override-module",
            "-dc",
            "--cce-long-scbz=true",
            "-mllvm",
            "-cce-dyn-kernel-stack-size=true",
            "-c",
            "-x",
            "ir",
            output / "helper.ll",
            "-o",
            output / "helper.device.o",
        ],
    )
    from pypto.runtime.kernel_compiler import KernelCompiler  # noqa: PLC0415
    from pypto.runtime.pto_isa import ensure_pto_isa_root  # noqa: PLC0415

    wrapper = make_wrapper((output / "caller.cpp").read_text())
    (output / "wrapper.cpp").write_text(wrapper)
    runtime_compiler = KernelCompiler("a5")
    isa = ensure_pto_isa_root()
    binary = runtime_compiler.compile_incore(
        str(output / "wrapper.cpp"),
        pto_isa_root=isa,
        runtime_name="tensormap_and_ringbuffer",
        vpto_sources=[str(SOURCE / "helper.pto")],
    )
    (output / "runtime-kernel.elf").write_bytes(binary)
    includes = [
        str(Path(isa) / "include"),
        str(Path(isa) / "include/pto"),
        *runtime_compiler.get_incore_include_dirs(),
        *runtime_compiler.get_kernel_include_dirs("tensormap_and_ringbuffer"),
    ]
    (output / "includes.json").write_text(json.dumps(includes))
    ccec = runtime_compiler.sdk.ccec
    assert ccec is not None, "A5 validation requires the CCEC toolchain"
    run(
        output,
        "compile-wrapper",
        [
            ccec.cxx_path,
            *ccec.get_compile_flags(core_type="aiv"),
            *(f"-I{path}" for path in includes),
            "-c",
            output / "wrapper.cpp",
            "-o",
            output / "wrapper.device.o",
        ],
    )
    objects = [str(output / "wrapper.device.o"), str(output / "helper.device.o")]
    run(output, "link", [cann / "bin/ld.lld", "-e", "kernel_entry", "-o", output / "kernel.elf", *objects])
    return output, cann, objects


def test_ub_signature_and_internal_vf(compiled: tuple[Path, Path, list[str]]) -> None:
    output, _, _ = compiled
    llvm = (output / "helper.ll").read_text()
    signature = next(line for line in llvm.splitlines() if line.startswith("define void @run_vf("))
    assert signature.count("ptr addrspace(6)") == 1
    assert "i32 %1" in signature
    cpp = (output / "wrapper.cpp").read_text()
    assert 'extern "C" __aicore__ void run_vf(__ubuf__ float*, int32_t);' in cpp
    symbols = run(output, "helper-symbols", ["readelf", "-Ws", output / "helper.device.o"])
    assert re.search(r"FUNC\s+GLOBAL\s+DEFAULT\s+\d+\s+run_vf$", symbols, re.MULTILINE)
    assert "run_vf.vector.thread" in symbols


def test_linked_image_and_rebasing(compiled: tuple[Path, Path, list[str]]) -> None:
    output, cann, objects = compiled
    report = run(output, "kernel-readelf", ["readelf", "-h", "-SW", "-Ws", "-r", output / "kernel.elf"])
    assert "There are no relocations in this file." in report
    text_match = re.search(r"\]\s+\.text\s+PROGBITS\s+([0-9a-f]+)", report)
    entry_match = re.search(r"Entry point address:\s+(0x[0-9a-f]+)", report)
    symbol_match = re.search(
        r"\d+:\s+([0-9a-f]+)\s+\d+\s+FUNC\s+GLOBAL\s+DEFAULT\s+\d+\s+kernel_entry$",
        report,
        re.MULTILINE,
    )
    assert text_match is not None, "Linked image has no .text section"
    assert entry_match is not None, "Linked image has no ELF entry point"
    assert symbol_match is not None, "Linked image has no exported kernel_entry"
    text_address = int(text_match[1], 16)
    entry_address = int(entry_match[1], 16)
    symbol_address = int(symbol_match[1], 16)
    assert text_address == entry_address == symbol_address
    for line in report.splitlines():
        if " UND " in line:
            assert re.match(r"\s*0:", line), f"Unresolved symbol: {line}"
    # The loader retains only .text. Reject additional allocated code/data.
    for line in report.splitlines():
        section = re.match(r"\s*\[\s*\d+\]\s+(\S+)\s+\S+\s+\w+\s+\w+\s+([0-9a-f]+)\s+\w+\s+([A-Z]+)\s", line)
        if section and "A" in section[3] and int(section[2], 16):
            assert section[1] == ".text", line
    run(
        output,
        "link-rebased",
        [
            cann / "bin/ld.lld",
            "--image-base",
            "0x10000",
            "-e",
            "kernel_entry",
            "-o",
            output / "rebased.elf",
            *objects,
        ],
    )
    # Use the runtime extractor to compare the actual bytes retained by the loader.
    extract = runpy.run_path(str(ROOT / "python/pypto/runtime/elf_parser.py"))["extract_text_section"]
    code = extract(output / "kernel.elf")
    assert code and code == extract(output / "rebased.elf")
    assert code == extract(output / "runtime-kernel.elf")
    (output / "kernel.bin").write_bytes(code)


def test_simulator_numerics(compiled: tuple[Path, Path, list[str]]) -> None:
    if os.environ.get("PYPTO_VPTO_VF_SIMULATOR") != "1":
        pytest.skip("Set PYPTO_VPTO_VF_SIMULATOR=1 to run the standalone CANN simulator")
    _standalone_numerics(compiled)


def test_onboard_numerics(compiled: tuple[Path, Path, list[str]]) -> None:
    if os.environ.get("PYPTO_VPTO_VF_ONBOARD") != "1":
        pytest.skip("Set PYPTO_VPTO_VF_ONBOARD=1 and PYPTO_VPTO_DEVICE_ID for hardware")
    device_id = int(os.environ["PYPTO_VPTO_DEVICE_ID"])
    timeout = int(os.environ.get("PYPTO_VPTO_TIMEOUT", "180"))
    assert device_id >= 0 and timeout > 0
    assert os.environ.get("PYPTO_VPTO_VF_SIMULATOR") != "1", "Choose exactly one execution mode"
    # Refuse inherited injection; never silently turn a simulator environment into a hardware claim.
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
        value = os.environ.get(key, "").lower()
        assert not any(token in value for token in ("simulator", "camodel", "msopprof")), (key, value)
    _standalone_numerics(compiled, device_id=device_id, timeout=timeout)


def _standalone_numerics(
    compiled: tuple[Path, Path, list[str]],
    *,
    device_id: int | None = None,
    timeout: int = 180,
) -> None:
    base, cann, _ = compiled
    onboard = device_id is not None
    output = base / ("onboard" if onboard else "simulation")
    output.mkdir(exist_ok=True)
    shutil.copyfile(base / "wrapper.cpp", output / "wrapper.cpp")
    compiler = cann / "bin/bisheng"
    simulator = Path(os.environ.get("PYPTO_VPTO_SIM_LIB", cann / "tools/simulator/dav_3510/lib"))
    ptoas = shutil.which("ptoas")
    assert ptoas
    includes = json.loads((base / "includes.json").read_text())
    run(
        output,
        "compile-helper-fatobj",
        [
            ptoas,
            SOURCE / "helper.pto",
            "--pto-arch=a5",
            "--pto-backend=vpto",
            "-o",
            output / "helper.fatobj.o",
        ],
    )
    envelope = (
        '#if defined(__DAV_VEC__)\n#include "wrapper.cpp"\n#endif\n'
        'extern "C" __global__[aicore] void pypto_vf_launch(__gm__ int64_t* args) {\n'
        "#if defined(__DAV_VEC__)\n  kernel_entry(args);\n#endif\n}\n"
    )
    (output / "envelope.cpp").write_text("#include <stdint.h>\n" + envelope)
    run(
        output,
        "compile-envelope",
        [
            compiler,
            "-xcce",
            "-fPIC",
            "-dc",
            "-O3",
            "--cce-aicore-arch=dav-c310",
            "-std=c++17",
            "-DMEMORY_BASE",
            *(f"-I{path}" for path in includes),
            "-c",
            output / "envelope.cpp",
            "-o",
            output / "envelope.o",
        ],
    )
    run(
        output,
        "compile-launch",
        [
            compiler,
            "-xcce",
            "-fPIC",
            "--cce-aicore-arch=dav-c310",
            "-DREGISTER_BASE",
            "-std=c++17",
            "-I",
            cann / "include",
            "-c",
            SOURCE / "launch.cpp",
            "-o",
            output / "launch.o",
        ],
    )
    runtime_lib = "-lruntime" if onboard else "-lruntime_camodel"
    library_dirs = ["-L", str(cann / "lib64")]
    if not onboard:
        library_dirs += ["-L", str(simulator)]
    run(
        output,
        "link-onboard-library" if onboard else "link-simulator-library",
        [
            compiler,
            "-fPIC",
            "--cce-fatobj-link",
            "-shared",
            *library_dirs,
            "-o",
            output / "libvf_kernel.so",
            output / "envelope.o",
            output / "helper.fatobj.o",
            output / "launch.o",
            "-Wl,--no-as-needed",
            runtime_lib,
        ],
    )
    run(
        output,
        "compile-host",
        [
            compiler,
            "-xc++",
            "-std=c++17",
            SOURCE / "main.cpp",
            next(
                Path(path) / "assert_compat.cpp"
                for path in includes
                if (Path(path) / "assert_compat.cpp").is_file()
            ),
            *(f"-I{path}" for path in includes),
            "-I",
            cann / "include",
            "-L",
            output,
            *library_dirs,
            "-o",
            output / "check_vf",
            "-lvf_kernel",
            "-Wl,--allow-shlib-undefined",
            runtime_lib,
            "-lstdc++",
            "-lascendcl",
            "-lm",
            "-ltiling_api",
            "-lplatform",
            "-lc_sec",
            "-ldl",
            "-lnnopbase",
        ],
    )
    env = dict(os.environ)
    if onboard:
        env["LD_LIBRARY_PATH"] = f"{output}:{cann / 'lib64'}:{env.get('LD_LIBRARY_PATH', '')}"
        # Record actual dynamic-library resolution, not only linker arguments.
        dependencies = run(output, "dependencies", ["ldd", output / "check_vf"], env)
        assert "camodel" not in dependencies and "simulator" not in dependencies
        assert "not found" not in dependencies
        result = run(output, "onboard", [output / "check_vf", str(device_id), "3"], env, timeout=timeout)
        for n in (0, 1, 63, 64, 65, 127, 128, 1023, 1024):
            assert result.count(f"PASS n={n}: active lanes and unchanged tail") == 3
        return
    env["LD_LIBRARY_PATH"] = f"{output}:{simulator}:{cann / 'lib64'}:{env.get('LD_LIBRARY_PATH', '')}"
    # The matching profiler worker supplies simulator initialization/injection.
    # Direct ACL startup and the top-level msprof dispatcher may probe a real
    # device even when the simulator runtime library is on LD_LIBRARY_PATH.
    worker = cann / "tools/msopprof/bin/msopprof"
    assert worker.is_file(), f"Missing matching simulator worker: {worker}"
    executable = output / "check_vf"
    # The worker rejects application directories writable by another user/group.
    output.chmod(output.stat().st_mode & ~0o022)
    executable.chmod(executable.stat().st_mode & ~0o022)
    result = run(
        output,
        "simulator",
        [
            worker,
            "simulator",
            f"--application={executable}",
            "--soc-version=Ascend950PR_9599",
            "--launch-count=9",
            "--timeout=2",
            f"--output={output / 'profiler'}",
        ],
        env,
    )
    for n in (0, 1, 63, 64, 65, 127, 128, 1023, 1024):
        assert f"PASS n={n}: active lanes and unchanged tail" in result


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, f"--confcutdir={SOURCE}", "-v"]))
