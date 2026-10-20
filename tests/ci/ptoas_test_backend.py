# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Select PTOAS for existing CI suites, including fresh Python child processes.

This is an experimental test adapter, installed only into a CI job's interpreter.
It never substitutes an EmitC implementation for a failing VPTO compilation.
PYTEST_DONT_REWRITE: this bootstrap is imported before pytest starts.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sysconfig
import traceback
from contextvars import ContextVar
from pathlib import Path
from typing import Any
from uuid import uuid4

from compare_ptoas_backends import adapter_declarations, binary_comparison, elf_sections

_MARKER = "// PYPTO_PTOAS_AUDIT "
_LINK_OBJECTS: ContextVar[tuple[str, ...]] = ContextVar("ptoas_audit_link_objects", default=())
_COMMANDS: ContextVar[list[list[str]] | None] = ContextVar("ptoas_audit_commands", default=None)
_state = {"installed": False, "record_tests": True}


def _write_json(path: Path, record: Any) -> None:
    path.write_text(json.dumps(record, indent=2) + "\n")


def backend_flags(flags: list[str] | None, backend: str) -> list[str]:
    """Select exactly one backend without dropping planner or architecture flags."""
    return [
        *(flag for flag in flags or [] if not flag.startswith("--pto-backend=")),
        f"--pto-backend={backend}",
    ]


def metadata(source: str) -> dict[str, str] | None:
    """Find the module's durable adapter metadata inside the generated wrapper."""
    records = [json.loads(line[len(_MARKER) :]) for line in source.splitlines() if line.startswith(_MARKER)]
    if not records:
        return None
    if len(records) != 1:
        raise RuntimeError("Expected exactly one PTOAS audit module in a kernel wrapper")
    return records[0]


def _tool(compiler: Any, name: str) -> str:
    assert compiler.sdk.ccec is not None, "VPTO object extraction requires the CANN device toolchain"
    return str(Path(compiler.sdk.ccec.cxx_path).with_name(name))


def _run_tool(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    if result.returncode:
        raise RuntimeError(f"PTOAS audit command failed: {command}\n{result.stderr}")
    return result.stdout


def _platform(flags: list[str]) -> str:
    for index, flag in enumerate(flags):
        arch = flags[index + 1] if flag == "--pto-arch" else flag.removeprefix("--pto-arch=")
        if flag == "--pto-arch" or flag.startswith("--pto-arch="):
            return "a5" if arch == "a5" else "a2a3"
    raise RuntimeError(f"PTOAS audit requires an explicit target architecture, got {flags}")


def _configure_process(backend: str) -> str:
    # The backend is absent from production cache keys. A job-specific epoch
    # prevents reuse of another backend's generated sources or binaries, even
    # for tests that explicitly enable the persistent cache.
    generation_file = Path(os.environ["PYPTO_PTOAS_AUDIT_DIR"]) / "generation.json"
    generation = json.loads(generation_file.read_text()) if generation_file.exists() else "local"
    salt = (
        f"ptoas-audit-{backend}-{os.environ.get('GITHUB_RUN_ID', 'local')}-"
        f"{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}-{generation}-"
        + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    )
    epoch = os.environ.get("PYPTO_CACHE_EPOCH", "")
    if not epoch.endswith(salt):
        os.environ["PYPTO_CACHE_EPOCH"] = f"{epoch}:{salt}"
    plugins = os.environ.get("PYTEST_PLUGINS", "").split(",")
    if "ptoas_test_backend" not in plugins:
        os.environ["PYTEST_PLUGINS"] = ",".join(filter(None, [*plugins, "ptoas_test_backend"]))
    return salt


def install() -> None:
    """Patch actual PTOAS and device compilation for the entire test process."""
    backend = os.environ.get("PYPTO_PTOAS_TEST_BACKEND", "")
    if not backend or _state["installed"]:
        return
    if backend not in ("emitc", "vpto"):
        raise ValueError(f"PYPTO_PTOAS_TEST_BACKEND must be emitc or vpto, got {backend!r}")

    from pypto.backend import pto_backend  # noqa: PLC0415
    from pypto.runtime.debug.pto_rebuild import _splice_body  # noqa: PLC0415
    from pypto.runtime.kernel_compiler import KernelCompiler  # noqa: PLC0415
    from pypto.runtime.pto_isa import ensure_pto_isa_root  # noqa: PLC0415

    root = Path(os.environ["PYPTO_PTOAS_AUDIT_DIR"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    salt = _configure_process(backend)
    _write_json(root / f"activated-{os.getpid()}.json", {"backend": backend, "epoch": salt})
    original_ptoas = pto_backend._run_ptoas
    original_compile = KernelCompiler.compile_incore
    original_run = KernelCompiler._run

    def selected_ptoas(pto_path: str, output_path: str, ptoas_flags: list[str] | None = None) -> None:
        directory = root / f"module-{uuid4().hex}"
        directory.mkdir()
        shutil.copyfile(pto_path, directory / "input.pto")
        flags = backend_flags(ptoas_flags, backend)
        record: dict[str, Any] = {"backend": backend, "flags": flags, "passed": False}
        try:
            if backend == "vpto":
                os.environ["PTO_ISA_ROOT"] = ensure_pto_isa_root()
                fat = directory / "vpto.fat.o"
                record["stage"] = "vpto"
                original_ptoas(pto_path, str(fat), flags)
            record["stage"] = "emitc-reference" if backend == "vpto" else "emitc"
            # Preserve the full EmitC implementation for the independent binary
            # baseline. VPTO only borrows its function declarations below.
            original_ptoas(pto_path, output_path, backend_flags(ptoas_flags, "emitc"))
            reference = Path(output_path).read_text()
            (directory / "emitc.cpp").write_text(reference)
            if backend == "vpto":
                record["stage"] = "native-adapter"
                compiler = KernelCompiler(_platform(flags))
                native = directory / "vpto.device.o"
                _run_tool(
                    [
                        _tool(compiler, "llvm-objcopy"),
                        f"--dump-section=__aicore_rel_binary={native}",
                        str(fat),
                    ]
                )
                symbols = _run_tool([_tool(compiler, "llvm-objdump"), "-t", str(native)])
                declarations = adapter_declarations(reference, symbols)
                _write_json(directory / "declarations.json", declarations)
            info = {"directory": str(directory), "backend": backend}
            Path(output_path).write_text(f"{reference}\n{_MARKER}{json.dumps(info)}\n")
            record["passed"] = True
        except Exception:
            record["error"] = traceback.format_exc()
            raise
        finally:
            _write_json(directory / "ptoas.json", record)

    def recording_run(self: Any, cmd: list[str], output: Path, label: str) -> bytes:
        command = cmd
        if label == "Incore-link":
            command = [*command, *_LINK_OBJECTS.get()]
        commands = _COMMANDS.get()
        if commands is not None:
            commands.append(command)
        return original_run(self, command, output, label)

    def compile_incore(self: Any, source_path: str, *args: Any, **kwargs: Any) -> bytes:
        source = Path(source_path).resolve()
        info = metadata(source.read_text())
        if info is None:
            # Hand-written/external kernels never invoke PTOAS. Record them as
            # such, rather than claiming they exercise the selected backend.
            _write_json(root / f"external-{uuid4().hex}.json", {"source": str(source)})
            return original_compile(self, source_path, *args, **kwargs)
        directory = Path(info["directory"]) / f"kernel-{uuid4().hex}"
        directory.mkdir()
        shutil.copyfile(source, directory / source.name)
        record: dict[str, Any] = {
            "source": str(source),
            "platform": self.platform,
            "backend": backend,
            "passed": False,
        }
        commands: list[list[str]] = []
        commands_token = _COMMANDS.set(commands)
        try:
            if backend == "vpto" and self.platform.endswith("sim"):
                raise RuntimeError(
                    "VPTO emitted a native AICore object; the CPU simulator "
                    "cannot execute it. EmitC fallback is disabled."
                )
            # Compile both implementations at the same source path. The test's
            # own generated wrapper remains untouched, including for concurrent
            # test compilation and persistent-cache publication.
            wrapper = directory / source.name
            emitc = original_compile(self, str(wrapper), *args, **kwargs)
            (directory / "emitc.elf").write_bytes(emitc)
            binary = emitc
            if backend == "vpto":
                module_dir = Path(info["directory"])
                declarations = json.loads((module_dir / "declarations.json").read_text())
                _splice_body(wrapper, "\n".join(declarations.values()))
                link_token = _LINK_OBJECTS.set((str(module_dir / "vpto.device.o"),))
                try:
                    binary = original_compile(self, str(wrapper), *args, **kwargs)
                finally:
                    _LINK_OBJECTS.reset(link_token)
                (directory / "vpto.elf").write_bytes(binary)
                record["comparison"] = binary_comparison(emitc, binary)
            for name, data in (("emitc", emitc), (backend, binary)):
                (directory / f"{name}.bin").write_bytes(elf_sections(data, 4)[".text"][2])
            record["passed"] = True
            return binary
        except Exception:
            record["error"] = traceback.format_exc()
            raise
        finally:
            _COMMANDS.reset(commands_token)
            record["commands"] = commands
            _write_json(directory / "result.json", record)

    pto_backend._run_ptoas = selected_ptoas
    KernelCompiler._run = recording_run
    KernelCompiler.compile_incore = compile_incore
    _state["installed"] = True


def bootstrap() -> None:
    """Fail closed: Python normally prints and ignores errors in .pth imports."""
    if os.environ.get("PYPTO_PTOAS_AUDIT_READY") != "1":
        return
    try:
        install()
    except BaseException:
        traceback.print_exc()
        os._exit(86)


def pytest_collection_finish(session: Any) -> None:
    """Retain the actual existing test inventory, including parametrized ids."""
    root = Path(os.environ["PYPTO_PTOAS_AUDIT_DIR"])
    _write_json(root / f"collected-{uuid4().hex}.json", {"nodeids": [item.nodeid for item in session.items]})


def pytest_configure(config: Any) -> None:
    """Avoid double-counting reports forwarded from xdist workers."""
    _state["record_tests"] = hasattr(config, "workerinput") or not config.getoption("numprocesses", default=0)


def pytest_runtest_logreport(report: Any) -> None:
    """Keep failures and skips explicit instead of counting compile calls as tests."""
    if not _state["record_tests"]:
        return
    root = Path(os.environ["PYPTO_PTOAS_AUDIT_DIR"])
    _write_json(
        root / f"test-{uuid4().hex}.json",
        {
            "nodeid": report.nodeid,
            "when": report.when,
            "outcome": report.outcome,
            "longrepr": str(report.longrepr) if report.longrepr else None,
        },
    )


def main() -> None:
    """Install the CI-only bootstrap or summarize attempts across processes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "summary"))
    args = parser.parse_args()
    if args.action == "install":
        configured = os.environ.get("PYPTO_PTOAS_AUDIT_DIR")
        if configured:
            root = Path(configured).resolve()
            build = Path.cwd().resolve() / "build"
            if not root.is_relative_to(build) or root == build:
                raise ValueError(f"Audit output must be a dedicated directory under {build}, got {root}")
            if root.exists():
                shutil.rmtree(root)
            root.mkdir(parents=True)
            _write_json(root / "generation.json", uuid4().hex)
        target = Path(sysconfig.get_paths()["purelib"]) / "zz_pypto_ptoas_audit.pth"
        ci_path = str(Path(__file__).resolve().parent)
        target.write_text(
            f"import sys; sys.path.insert(0, {ci_path!r}); "
            "import ptoas_test_backend; ptoas_test_backend.bootstrap()\n"
        )
        github_env = os.environ.get("GITHUB_ENV")
        if github_env:
            with open(github_env, "a") as environment:
                environment.write("PYPTO_PTOAS_AUDIT_READY=1\n")
        print(f"Installed PTOAS test backend bootstrap: {target}")
    else:
        root = Path(os.environ["PYPTO_PTOAS_AUDIT_DIR"])
        modules = [json.loads(p.read_text()) for p in root.glob("module-*/ptoas.json")]
        kernels = [json.loads(p.read_text()) for p in root.glob("module-*/kernel-*/result.json")]
        report: dict[str, Any] = {
            "backend": os.environ["PYPTO_PTOAS_TEST_BACKEND"],
            "ptoas_attempts": len(modules),
            "ptoas_passed": sum(r["passed"] for r in modules),
            "kernel_attempts": len(kernels),
            "kernel_passed": sum(r["passed"] for r in kernels),
            "external_kernels": len(list(root.glob("external-*.json"))),
            "comparisons": [r["comparison"] for r in kernels if "comparison" in r],
        }
        collected = [json.loads(p.read_text()) for p in root.glob("collected-*.json")]
        tests = [json.loads(p.read_text()) for p in root.glob("test-*.json")]
        report["unique_collected_tests"] = len({n for r in collected for n in r["nodeids"]})
        report["test_calls"] = {
            outcome: sum(r["when"] == "call" and r["outcome"] == outcome for r in tests)
            for outcome in ("passed", "failed", "skipped")
        }
        report["setup_skips"] = sum(r["when"] == "setup" and r["outcome"] == "skipped" for r in tests)
        report["setup_teardown_failures"] = sum(
            r["when"] != "call" and r["outcome"] == "failed" for r in tests
        )
        root.mkdir(parents=True, exist_ok=True)
        _write_json(root / "summary.json", report)
        print(json.dumps(report, indent=2))
        if not modules:
            raise SystemExit("No PTOAS invocations were recorded; this suite has no backend coverage.")


if __name__ == "__main__":
    main()
