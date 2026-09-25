# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Bounded build identities for the default JIT cache policy.

Published build/version identifiers cover native installations. Python package
sources are hashed once per process. This deliberately does not audit sysroots,
dynamic dependencies, or same-build local patches; use the content policy or a
new cache epoch for those. No disk memo of filesystem timestamps is involved.
"""

import ast
import hashlib
import importlib
import os
import struct
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

from pypto._identity import ComponentInputs, ToolchainInputs, _file_digest, digest_record


def native_build_id(path: Path) -> tuple[str, str]:
    """Read a complete GNU ELF build ID; hash contents when none is available."""
    with path.open("rb") as stream:
        header = stream.read(64)
        if len(header) == 64 and header[:5] == b"\x7fELF\x02" and header[5] in (1, 2):
            endian = "<" if header[5] == 1 else ">"
            offset = struct.unpack_from(f"{endian}Q", header, 32)[0]
            size, count = struct.unpack_from(f"{endian}HH", header, 54)
            length = os.fstat(stream.fileno()).st_size
            if size >= 56 and 0 < count < 4096 and offset + size * count <= length:
                for index in range(count):
                    stream.seek(offset + size * index)
                    program = stream.read(56)
                    if struct.unpack_from(f"{endian}I", program)[0] != 4:  # PT_NOTE
                        continue
                    start = struct.unpack_from(f"{endian}Q", program, 8)[0]
                    extent = struct.unpack_from(f"{endian}Q", program, 32)[0]
                    if extent > 1024 * 1024 or start + extent > length:
                        continue
                    stream.seek(start)
                    notes = stream.read(extent)
                    position = 0
                    while position + 12 <= len(notes):
                        namesz, descsz, kind = struct.unpack_from(f"{endian}III", notes, position)
                        position += 12
                        name = notes[position : position + namesz]
                        position += (namesz + 3) & ~3
                        end = position + descsz
                        if end > len(notes):
                            break
                        if kind == 3 and name == b"GNU\0" and descsz:
                            return ("gnu-build-id", notes[position:end].hex())
                        position += (descsz + 3) & ~3
    return ("sha256", _file_digest(path)[1])


def _walk_error(error: OSError) -> None:
    raise error


@lru_cache(maxsize=64)
def python_sources(root: Path) -> str:
    """Hash Python sources and bundled templates, excluding generated assets.

    Paths and bytes both participate. Installation files are immutable within
    a process, just as imported modules are; a fresh process detects edits.
    """
    records = []
    base = os.fspath(root)
    for directory, dirs, files in os.walk(base, onerror=_walk_error):
        dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", ".git", "_assets"))
        for name in dirs:
            if os.path.islink(os.path.join(directory, name)):
                raise ValueError(f"Python package has an unsupported directory symlink: {directory}/{name}")
        for filename in sorted(files):
            if filename.endswith((".py", ".in")):
                path = os.path.join(directory, filename)
                with open(path, "rb") as stream:
                    before = os.fstat(stream.fileno())
                    content = hashlib.sha256(stream.read()).hexdigest()
                    after = os.fstat(stream.fileno())
                current = os.stat(path)

                def stamp(info: os.stat_result) -> tuple[int, ...]:
                    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

                if stamp(before) != stamp(after) or stamp(after) != stamp(current):
                    raise ValueError(f"Python source changed while reading: {path}")
                records.append((path[len(base) + 1 :], content))
    if not records:
        raise ValueError(f"Python package has no sources: {root}")
    return digest_record((base, records))


def _module_path(name: str) -> Path:
    module = importlib.import_module(name)
    origin = getattr(module, "__file__", None)
    if origin is None:
        raise ValueError(f"Module has no installation path: {name}")
    return Path(origin).resolve(strict=True)


def _python_package(name: str) -> list[tuple[str, str]]:
    root = _module_path(name).parent
    roots = {root}
    for key, module in tuple(sys.modules.items()):
        if key.startswith(f"{name}."):
            origin = getattr(module, "__file__", None)
            if origin and str(origin).endswith(".py"):
                if not os.path.abspath(origin).startswith(f"{root}{os.sep}"):
                    path = Path(origin).resolve(strict=True)
                    anchor = next((p for p in path.parents if p.name == name), None)
                    if anchor is None:
                        raise ValueError(f"Unsupported external Python import redirect: {key}")
                    roots.add(anchor)
    return [(str(p), python_sources(p)) for p in sorted(roots)]


# Resolve in the launcher's interpreter, with script-mode sys.path[0]. Do not
# import ptoas: its __init__ loads the compiler and costs hundreds of ms.
_PTOAS_PROBE = """
import sys
startup = tuple(sys.modules.items())
sys.path[0] = sys.argv[1]
import importlib.util, os, site
stdlib = os.path.abspath(os.path.dirname(site.__file__))
site_roots = [os.path.abspath(root) for root in site.getsitepackages()]
if site.ENABLE_USER_SITE:
    site_roots.append(os.path.abspath(site.getusersitepackages()))
hook_roots = sorted(set(site_roots + [os.path.abspath(root) for root in sys.path if root]))
hooks = []
for root in hook_roots:
    if not os.path.isdir(root):
        continue
    with os.scandir(root) as entries:
        hooks.extend(entry.path for entry in entries if entry.name.endswith('.pth') and entry.is_file())
hooks = sorted(set(hooks))
modules = []
unresolved = []
for name, module in startup:
    origin = getattr(module, '__file__', None)
    if not origin:
        spec_origin = getattr(getattr(module, '__spec__', None), 'origin', None)
        if spec_origin not in (None, 'built-in', 'frozen'):
            unresolved.append(name)
        for directory in getattr(module, '__path__', ()):
            path = os.path.abspath(directory)
            if not path.startswith(stdlib + os.sep) or any(
                path.startswith(root + os.sep) for root in site_roots
            ):
                unresolved.append(name)
        continue
    path = os.path.abspath(origin)
    if path.startswith(stdlib + os.sep) and not any(path.startswith(root + os.sep) for root in site_roots):
        if name not in ('sitecustomize', 'usercustomize'):
            continue
    if os.path.isfile(path):
        modules.append((name, path))
    else:
        unresolved.append(name)
ptoas = importlib.util.find_spec('ptoas')
numpy = importlib.util.find_spec('numpy')
print(repr({
    'ptoas': ptoas.origin if ptoas is not None else None,
    'numpy': numpy.origin if numpy is not None else None,
    'startup_hooks': hooks,
    'startup_modules': modules,
    'startup_path': sys.path,
    'startup_unresolved': sorted(set(unresolved)),
}))
"""


def _ptoas_identity(selected: str) -> Any:
    from ._toolchain import _console_interpreter, _run  # noqa: PLC0415

    launcher = Path(selected).absolute()
    with launcher.open("rb") as stream:
        if stream.read(4) == b"\x7fELF":
            return (str(launcher.resolve()), native_build_id(launcher))
    try:
        interpreter = _console_interpreter(launcher)
    except ValueError as exc:
        raise ValueError(f"Unsupported PTOAS launcher for build identity: {launcher}") from exc
    selected_modules = ast.literal_eval(_run([str(interpreter), "-c", _PTOAS_PROBE, str(launcher.parent)]))
    origin = selected_modules.get("ptoas")
    if not isinstance(origin, str):
        raise ValueError(f"Cannot resolve PTOAS package from {launcher}")
    package = Path(origin).resolve(strict=True).parent
    if (package / "_online").exists():
        raise ValueError(f"PTOAS online build has no stable published identity: {package}")
    numpy_identity = _numpy_wheel_identity(selected_modules.get("numpy"), interpreter)
    natives = sorted(package.glob("_core*.so"))
    if not natives:
        raise ValueError(f"PTOAS wheel has no native compiler module: {package}")
    metadata = sorted(package.parent.glob("ptoas-*.dist-info/METADATA"))
    if len(metadata) != 1:
        raise ValueError(f"PTOAS wheel metadata is unavailable or ambiguous: {package}")
    return (
        str(launcher),
        _file_digest(launcher)[1],
        str(interpreter),
        _file_digest(metadata[0])[1],
        [(str(p), native_build_id(p)) for p in natives],
        numpy_identity,
        _startup_identity(selected_modules),
    )


def _startup_identity(selected_modules: dict[str, Any]) -> tuple[Any, ...]:
    """Identify files that shaped the selected interpreter before PTOAS imports."""
    hooks = selected_modules.get("startup_hooks")
    modules = selected_modules.get("startup_modules")
    search_path = selected_modules.get("startup_path")
    unresolved = selected_modules.get("startup_unresolved")
    if (
        not isinstance(hooks, list)
        or not isinstance(modules, list)
        or not isinstance(search_path, list)
        or unresolved != []
        or len(hooks) > 256
        or len(modules) > 512
        or len(search_path) > 512
        or any(type(path) is not str for path in (*hooks, *search_path))
    ):
        raise ValueError("PTOAS interpreter startup evidence is unavailable")
    loaded = []
    for entry in modules:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2 or any(type(v) is not str for v in entry):
            raise ValueError(f"Invalid PTOAS startup module evidence: {entry!r}")
        loaded.append(tuple(entry))
    paths = {Path(path) for path in hooks}
    paths.update(Path(path) for _, path in loaded)
    files = [(str(path), _file_digest(path)[1]) for path in sorted(paths)]
    return (hooks, loaded, search_path, files)


def _numpy_wheel_identity(numpy_origin: Any, interpreter: Path) -> tuple[Any, ...]:
    """Identify the NumPy wheel selected by PTOAS without importing it."""
    if not isinstance(numpy_origin, str):
        raise ValueError(f"Selected PTOAS interpreter cannot identify NumPy: {interpreter}")
    numpy_package = Path(numpy_origin).resolve(strict=True).parent
    if numpy_package.name != "numpy":
        raise ValueError(f"Unsupported selected NumPy package layout: {numpy_origin}")
    numpy_metadata = sorted(numpy_package.parent.glob("numpy-*.dist-info"))
    if len(numpy_metadata) != 1 or not all(
        (numpy_metadata[0] / name).is_file() for name in ("METADATA", "WHEEL", "RECORD")
    ):
        raise ValueError(f"Selected NumPy wheel identity is unavailable: {numpy_package}")
    return (
        str(numpy_package),
        [(name, _file_digest(numpy_metadata[0] / name)[1]) for name in ("METADATA", "WHEEL", "RECORD")],
    )


def _reported(value: Any) -> ComponentInputs:
    # Tag the policy so these identities cannot alias content inventories.
    return ComponentInputs(
        unavailable_reason=None,
        reported_version=digest_record(("build-identity-v1", os.environ.get("PYPTO_CACHE_EPOCH"), value)),
    )


def discover_builds(compiler: Callable[[], Any], ptoas: str, runtime_name: str) -> ToolchainInputs:
    """Overlap the external interpreter probe with local package reads."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        assembler = pool.submit(_ptoas_identity, ptoas)
        sources = pool.submit(_python_package, "pypto")
        selected = compiler()
        components = _local_builds(selected, runtime_name, sources.result())
        return ToolchainInputs(*components[:3], _reported(assembler.result()), components[3])


def _compiler_identity(selected: str) -> tuple[Any, ...]:
    from ._toolchain import _driver_executed, _executable, _invocable, _run  # noqa: PLC0415

    invoked = _invocable(selected)

    # GCC selects these programs independently of its driver. Query the same
    # invocation used for compilation so wrapper and PATH selection are honored.
    def selected_program(name: str) -> tuple[str, str, tuple[str, str]]:
        answer = _run([str(invoked), f"-print-prog-name={name}"]).strip()
        if not answer or "\n" in answer:
            raise ValueError(f"Cannot identify GCC's selected {name}: {invoked}")
        path = _executable(answer)
        return (name, str(path), native_build_id(path))

    with ThreadPoolExecutor(max_workers=6) as pool:
        executed = pool.submit(_run, [str(invoked), "-E", "-x", "c++", "-v", os.devnull])
        version = pool.submit(_run, [str(invoked), "--version"])
        programs = list(pool.map(selected_program, ("cc1plus", "collect2", "as", "ld")))
        reported_version = version.result().strip()
        driver = _driver_executed(executed.result(), invoked)
    invoked_native = invoked.resolve(strict=True)
    invoked_id = native_build_id(invoked_native)
    driver_id = invoked_id if driver == invoked_native else native_build_id(driver)
    return (str(invoked), reported_version, invoked_id, driver_id, programs)


def _local_builds(
    compiler: Any, runtime_name: str, pypto_sources: list[tuple[str, str]]
) -> tuple[ComponentInputs, ...]:
    """Identify the actual imports, selected tools and resolver-selected ISA."""
    from ._toolchain import _cann_install_version, _run  # noqa: PLC0415

    if sys.platform != "linux":
        raise ValueError(f"Unsupported build identity platform: {sys.platform}")
    native = _module_path("pypto.pypto_core")
    runtime_native = _module_path("_task_interface")
    runtime_module = importlib.import_module("_task_interface")
    revision = getattr(runtime_module, "__build_commit__", "")
    root = compiler.project_root
    # Source builds compile orchestration helpers from the runtime checkout;
    # wheels ship those sources under _assets and record their build revision.
    if (root / ".git").exists():
        revision = _run(["git", "-C", str(root), "rev-parse", "HEAD"]).strip()
        dirty = _run(["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"])
        if dirty.strip():
            raise ValueError(f"Runtime source checkout has uncommitted changes: {root}")
    if not revision:
        raise ValueError(f"Runtime source build revision is unavailable: {root}")
    metadata_path = root / "build/lib/pto_isa_build.json"
    runtime_build = metadata_path.read_bytes() if metadata_path.is_file() else None
    device: list[Any] = [_compiler_identity(compiler._orchestration_toolchain(runtime_name).cxx_path)]
    if compiler.platform.endswith("sim"):
        device.append(_compiler_identity(compiler.sdk.gxx15.cxx_path))
    else:
        ccec = Path(compiler.sdk.ccec.cxx_path).resolve(strict=True)
        cann_layout = tuple(p.name for p in ccec.parents[:3]) == ("bin", "bisheng_compiler", "tools")
        cann_version = _cann_install_version(ccec.parents[3]) if cann_layout else ""
        if not cann_version:
            raise ValueError(f"CANN installation build version is unavailable: {ccec}")
        device.append((str(ccec), cann_version))
        linker = Path(compiler.sdk.ccec.linker_path).resolve(strict=True)
        device.append((str(linker), native_build_id(linker)))
    return (
        _reported(
            (
                pypto_sources,
                str(native),
                native_build_id(native),
                sys.version,
                sys.implementation.cache_tag,
            )
        ),
        _reported(
            (
                revision,
                str(root),
                _python_package("simpler"),
                _python_package("simpler_setup"),
                str(runtime_native),
                native_build_id(runtime_native),
                runtime_build,
            )
        ),
        # This is the effective version selected by the resolver, not an extra
        # compatibility gate. A hit needs neither a checkout nor a git status.
        _reported(importlib.import_module("simpler_setup.pto_isa").read_pto_isa_pin(root / "pto_isa.pin")),
        _reported(device),
    )
