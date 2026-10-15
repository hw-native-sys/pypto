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
from email.parser import BytesParser
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

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
    records: list[tuple[str, str]] = []
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
# Only site directories process .pth files. A directory added by a path-only
# .pth entry contributes imports, but does not execute its own .pth files.
hook_roots = sorted(set(site_roots))
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
package_names = ('ptoas', 'ptodsl', 'TileOps', 'SoftOps')
packages = {}
for name in package_names:
    spec = importlib.util.find_spec(name)
    packages[name] = None if spec is None else (
        spec.origin,
        list(getattr(sys.modules.get(name), '__path__', spec.submodule_search_locations) or ()),
    )
loaded_packages = []
for name, module in tuple(sys.modules.items()):
    if name.partition('.')[0] in package_names:
        loaded_packages.append(
            (name, getattr(module, '__file__', None), list(getattr(module, '__path__', ())))
        )
print(repr({
    'interpreter_version': sys.version,
    'interpreter_cache_tag': sys.implementation.cache_tag,
    'ptoas': ptoas.origin if ptoas is not None else None,
    'numpy': numpy.origin if numpy is not None else None,
    'ptoas_packages': packages,
    'loaded_ptoas_packages': loaded_packages,
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
    version = selected_modules.get("interpreter_version")
    cache_tag = selected_modules.get("interpreter_cache_tag")
    if not isinstance(version, str) or not version or not isinstance(cache_tag, str) or not cache_tag:
        raise ValueError(f"PTOAS interpreter version/ABI evidence is unavailable: {interpreter}")
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
    metadata = sorted(package.parent.glob("ptoas-*.dist-info"))
    if len(metadata) != 1 or not all(
        (metadata[0] / name).is_file() for name in ("METADATA", "WHEEL", "RECORD")
    ):
        raise ValueError(f"PTOAS wheel metadata is unavailable or ambiguous: {package}")
    _validate_wheel_requirements(metadata[0] / "METADATA", ["numpy"])
    packages = _ptoas_package_identity(selected_modules, package)
    return (
        str(launcher),
        _file_digest(launcher)[1],
        (str(interpreter), version, cache_tag, native_build_id(interpreter)),
        [(name, _file_digest(metadata[0] / name)[1]) for name in ("METADATA", "WHEEL", "RECORD")],
        packages,
        [(str(p), native_build_id(p)) for p in natives],
        numpy_identity,
        _startup_identity(selected_modules),
    )


def _ptoas_package_identity(selected_modules: dict[str, Any], package: Path) -> tuple[Any, ...]:
    """Reject package redirects outside the wheel covered by its RECORD."""
    names = ("ptoas", "ptodsl", "TileOps", "SoftOps")
    packages = selected_modules.get("ptoas_packages")
    loaded = selected_modules.get("loaded_ptoas_packages")
    if not isinstance(packages, dict) or not isinstance(loaded, list):
        raise ValueError("PTOAS package origin evidence is unavailable")
    packages = cast(dict[str, object], packages)
    loaded = cast(list[object], loaded)
    if set(packages) != set(names):
        raise ValueError("PTOAS package origin evidence is unavailable")
    records: list[tuple[str, str, list[str]]] = []
    for name in names:
        entry = packages[name]
        expected = package.parent / name
        if not isinstance(entry, (tuple, list)):
            raise ValueError(f"PTOAS package origin evidence is unavailable: {name}")
        fields = cast(tuple[object, ...] | list[object], entry)
        if len(fields) != 2 or not isinstance(fields[0], str) or not isinstance(fields[1], list):
            raise ValueError(f"PTOAS package origin evidence is unavailable: {name}")
        paths = cast(list[object], fields[1])
        if (
            len(paths) != 1
            or not isinstance(paths[0], str)
            or Path(fields[0]).resolve(strict=True) != expected / "__init__.py"
            or Path(paths[0]).resolve(strict=True) != expected
        ):
            raise ValueError(f"PTOAS wheel uses an unsupported package origin or external redirect: {name}")
        records.append((name, fields[0], [paths[0]]))
    loaded_records: list[tuple[str, str, list[str]]] = []
    for entry in loaded:
        if not isinstance(entry, (tuple, list)):
            raise ValueError("PTOAS loaded package origin evidence is unavailable")
        fields = cast(tuple[object, ...] | list[object], entry)
        if (
            len(fields) != 3
            or not isinstance(fields[0], str)
            or not isinstance(fields[1], str)
            or not isinstance(fields[2], list)
            or fields[0].partition(".")[0] not in names
        ):
            raise ValueError("PTOAS loaded package origin evidence is unavailable")
        paths = cast(list[object], fields[2])
        if any(not isinstance(path, str) for path in paths):
            raise ValueError("PTOAS loaded package path evidence is unavailable")
        paths = cast(list[str], paths)
        root = package.parent / fields[0].partition(".")[0]
        if root not in Path(fields[1]).resolve(strict=True).parents or any(
            not Path(path).resolve(strict=True).is_relative_to(root) for path in paths
        ):
            raise ValueError(f"PTOAS wheel uses an external import redirect: {fields[0]}")
        loaded_records.append((fields[0], fields[1], paths))
    return (records, loaded_records)


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
    ):
        raise ValueError("PTOAS interpreter startup evidence is unavailable")
    hooks = cast(list[object], hooks)
    modules = cast(list[object], modules)
    search_path = cast(list[object], search_path)
    if (
        len(hooks) > 256
        or len(modules) > 512
        or len(search_path) > 512
        or any(type(path) is not str for path in (*hooks, *search_path))
    ):
        raise ValueError("PTOAS interpreter startup evidence is unavailable")
    hooks = cast(list[str], hooks)
    search_path = cast(list[str], search_path)
    loaded: list[tuple[str, str]] = []
    for entry in modules:
        if not isinstance(entry, (list, tuple)):
            raise ValueError(f"Invalid PTOAS startup module evidence: {entry!r}")
        fields = cast(list[object] | tuple[object, ...], entry)
        if len(fields) != 2 or type(fields[0]) is not str or type(fields[1]) is not str:
            raise ValueError(f"Invalid PTOAS startup module evidence: {entry!r}")
        loaded.append((fields[0], fields[1]))
    # Hash observed startup inputs without rejecting executable hooks. Inputs
    # loaded only in script mode require explicit cache invalidation; the -c
    # probe cannot discover arbitrary conditional startup dependencies.
    paths = {Path(path) for path in hooks}
    paths.update(Path(path) for _, path in loaded)
    files = [(str(path), _file_digest(path)[1]) for path in sorted(paths)]
    return (hooks, loaded, search_path, files)


def _validate_wheel_requirements(metadata: Path, expected: list[str]) -> None:
    """Reject dependencies outside the installed-wheel adapter's fixed scope."""
    with metadata.open("rb") as stream:
        requirements = BytesParser().parse(stream, headersonly=True).get_all("Requires-Dist", [])
    if requirements != expected:
        raise ValueError(
            f"Unsupported wheel dependency declarations for {metadata.parent.name}: {requirements}"
        )


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
    _validate_wheel_requirements(numpy_metadata[0] / "METADATA", [])
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


def _compiler_identity(selected: str, cwd: Path | None = None) -> tuple[Any, ...]:
    from ._toolchain import _driver_executed, _executable, _invocable, _run  # noqa: PLC0415

    context = {} if cwd is None else {"cwd": cwd}
    invoked = _invocable(selected, **context)

    # GCC selects these programs independently of its driver. Query the same
    # invocation used for compilation so wrapper and PATH selection are honored.
    def selected_program(name: str) -> tuple[str, str, tuple[str, str]]:
        answer = _run([str(invoked), f"-print-prog-name={name}"], **context).strip()
        if not answer or "\n" in answer:
            raise ValueError(f"Cannot identify GCC's selected {name}: {invoked}")
        path = _executable(answer, **context)
        return (name, str(path), native_build_id(path))

    with ThreadPoolExecutor(max_workers=6) as pool:
        executed = pool.submit(_run, [str(invoked), "-E", "-x", "c++", "-v", os.devnull], **context)
        version = pool.submit(_run, [str(invoked), "--version"], **context)
        programs = list(pool.map(selected_program, ("cc1plus", "collect2", "as", "ld")))
        reported_version = version.result().strip()
        driver = _executable(str(_driver_executed(executed.result(), invoked, **context)), **context)
    invoked_native = invoked.resolve(strict=True)
    invoked_id = native_build_id(invoked_native)
    driver_id = invoked_id if driver == invoked_native else native_build_id(driver)
    return (str(invoked), reported_version, invoked_id, driver_id, programs)


def _local_builds(
    compiler: Any, runtime_name: str, pypto_sources: list[tuple[str, str]]
) -> tuple[ComponentInputs, ComponentInputs, ComponentInputs, ComponentInputs]:
    """Identify the actual imports, selected tools and resolver-selected ISA."""
    from ._toolchain import _cann_install_version, _executable, _run  # noqa: PLC0415

    if sys.platform != "linux":
        raise ValueError(f"Unsupported build identity platform: {sys.platform}")
    interpreter = Path(sys.executable).resolve(strict=True)
    native = _module_path("pypto.pypto_core")
    runtime_native = _module_path("_task_interface")
    runtime_module = importlib.import_module("_task_interface")
    revision = getattr(runtime_module, "__build_commit__", "")
    root = Path(compiler.project_root).resolve(strict=True)
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
    device: list[Any] = [_compiler_identity(compiler._orchestration_toolchain(runtime_name).cxx_path, root)]
    if compiler.platform.endswith("sim"):
        device.append(_compiler_identity(compiler.sdk.gxx15.cxx_path, root))
    else:
        # Installation metadata cannot identify an arbitrary launcher's dispatch target.
        ccec = _executable(compiler.sdk.ccec.cxx_path, cwd=root)
        cann_layout = tuple(p.name for p in ccec.parents[:3]) == ("bin", "bisheng_compiler", "tools")
        cann_version = _cann_install_version(ccec.parents[3]) if cann_layout else ""
        if not cann_version:
            raise ValueError(f"CANN installation build version is unavailable: {ccec}")
        device.append((str(ccec), cann_version))
        # A launcher's bytes do not identify the linker it dispatches to.
        linker = _executable(compiler.sdk.ccec.linker_path, cwd=root)
        device.append((str(linker), native_build_id(linker)))
    return (
        _reported(
            (
                pypto_sources,
                str(native),
                native_build_id(native),
                sys.version,
                sys.implementation.cache_tag,
                str(interpreter),
                native_build_id(interpreter),
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
