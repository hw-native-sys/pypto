# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Conservative Linux inventories for the compiler paths used by JIT.

Unknown launchers and compiler layouts are unavailable, never weak identities.
Installed files are immutable until process exit; mutable application inputs
are handled separately. Discovery is cached by effective tool selection.
"""

import ast
import importlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from pypto._identity import (
    ComponentInputs,
    ContentRoot,
    InstallationIdentityCache,
    ToolchainIdentity,
    ToolchainInputs,
)
from pypto.backend._ptoas_locate import find_ptoas_binary

_identity_cache = InstallationIdentityCache()
_discovery_lock = threading.Lock()
_identities: dict[tuple[Any, ...], ToolchainIdentity] = {}


def _run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
    return result.stdout + result.stderr


def _executable(name: str) -> Path:
    selected = shutil.which(name)
    if selected is None:
        raise ValueError(f"Compiler executable is unavailable: {name}")
    path = Path(selected).resolve(strict=True)
    with path.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise ValueError(f"Unsupported compiler launcher (requires dependency adapter): {selected}")
    return path


def _elf_inputs(path: Path, library_path: str | None = None) -> set[Path]:
    """ldd reports the loader's transitive resolution, including the interpreter."""
    with path.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise ValueError(f"Expected an ELF installation input: {path}")
    environment = dict(os.environ)
    if library_path is not None:
        environment["LD_LIBRARY_PATH"] = library_path
    result = subprocess.run(
        ["ldd", str(path)], env=environment, capture_output=True, text=True, timeout=30, check=False
    )
    output = result.stdout + result.stderr
    if "not found" in output:
        raise ValueError(f"Unresolved native dependencies for {path}: {output.strip()}")
    if result.returncode and not any(s in output for s in ("statically linked", "not a dynamic executable")):
        raise ValueError(f"Cannot discover native dependencies for {path}: {output.strip()}")
    return {
        path,
        *(
            Path(p).resolve(strict=True)
            for p in re.findall(r"^\s*(?:[^\n]*=>\s*)?(/[^\n]+?)\s+\(0x[0-9a-f]+\)", output, re.MULTILINE)
        ),
    }


def _component(paths: set[Path]) -> ComponentInputs:
    # Parents already enumerate child contents. Retain logical paths in the
    # inventory; resolving every root would lose compiler selection aliases.
    ordered = sorted(paths, key=lambda p: (len(p.parts), str(p)))
    roots: list[Path] = []
    for path in ordered:
        path.resolve(strict=True)
        if not any(parent in path.parents for parent in roots):
            roots.append(path)
    return ComponentInputs(tuple(ContentRoot(p) for p in roots), unavailable_reason=None)


def _package(name: str) -> set[Path]:
    module = importlib.import_module(name)
    filename = getattr(module, "__file__", None)
    if filename is None:
        raise ValueError(f"Compiler module has no inspectable installation: {name}")
    root = Path(filename).resolve().parent
    paths = {root}
    for imported_name, imported in tuple(sys.modules.items()):
        if imported_name == name or imported_name.startswith(f"{name}."):
            origin = getattr(imported, "__file__", None)
            if origin is not None:
                selected = Path(origin).resolve(strict=True)
                if root not in selected.parents:
                    raise ValueError(f"Compiler package uses an external import redirect: {imported_name}")
    for native in root.rglob("*.so"):
        paths.update(_elf_inputs(native))
    return paths


def _include_roots(output: str, executable: Path) -> set[Path]:
    try:
        includes = output.split("#include <...> search starts here:", 1)[1].split("End of search list.", 1)[0]
    except IndexError as exc:
        raise ValueError(f"Cannot discover implicit C++ include roots: {executable}") from exc
    return {Path(line.strip()).resolve(strict=True) for line in includes.splitlines() if line.strip()}


def _gcc_inputs(executable: Path) -> set[Path]:
    if "clang" in _run([str(executable), "--version"]).lower():
        raise ValueError(f"Unsupported host compiler resource layout: {executable}")
    paths = _elf_inputs(executable)
    for program in ("cc1plus", "collect2", "as", "ld"):
        selected = _run([str(executable), f"-print-prog-name={program}"]).strip()
        paths.update(_elf_inputs(_executable(selected)))
    libgcc = Path(_run([str(executable), "-print-libgcc-file-name"]).strip()).resolve(strict=True)
    paths.add(libgcc.parent)  # GCC specs, plugins, startup objects, resources.
    output = _run([str(executable), "-E", "-x", "c++", "-v", os.devnull])
    paths.update(_include_roots(output, executable))
    paths.update(_gcc_link_inputs(executable))
    return paths


def _gcc_link_inputs(executable: Path) -> set[Path]:
    """Resolve the actual driver's shared-library link inputs."""
    paths: set[Path] = set()
    # Ask the actual GCC driver for its shared-library link command. -###
    # prints commands without executing compilation/linking. This inventories
    # selected startup objects, plugins and default libraries, without treating
    # every unrelated library installed on the machine as an input.
    plan = _run(
        [str(executable), "-###", "-shared", "-fPIC", "-pthread", "-x", "c++", os.devnull, "-o", os.devnull]
    )
    link_args = None
    for line in plan.splitlines():
        tokens = shlex.split(line)
        if tokens and Path(tokens[0]).name in ("collect2", "ld"):
            link_args = tokens
    if link_args is None:
        raise ValueError(f"Cannot discover the GCC linker invocation: {executable}")
    for argument in link_args[1:]:
        if argument.startswith("-l"):
            stem = argument[2:]
            for filename in (f"lib{stem}.so", f"lib{stem}.a"):
                resolved = _run([str(executable), f"-print-file-name={filename}"]).strip()
                if resolved != filename:
                    paths.add(Path(resolved).resolve(strict=True))
                    break
            else:
                raise ValueError(f"Cannot resolve linker input {argument} for {executable}")
        elif argument.startswith("/") and argument != os.devnull and Path(argument).is_file():
            paths.add(Path(argument).resolve(strict=True))
        elif argument.startswith("-plugin-opt=/"):
            paths.add(Path(argument.split("=", 1)[1]).resolve(strict=True))
    sysroot_value = _run([str(executable), "-print-sysroot"]).strip()
    sysroot = Path(sysroot_value).resolve(strict=True) if sysroot_value else None
    return _linker_script_inputs(paths, sysroot)


def _linker_script_names(text: str, script: Path) -> list[str]:
    """Read the supported implicit-script grammar, rejecting untracked inputs.

    Only absolute INPUT/GROUP dependencies (including AS_NEEDED) and format
    declarations are supported. Relative names, -l, SEARCH_DIR, INCLUDE, and
    other commands need the linker's complete search state, so fail closed.
    """
    tokens = re.findall(r'/\*.*?\*/|\#[^\n]*|"[^"\\]*"|[(),;]|[^\s(),;"]+|\S', text, re.DOTALL)
    if any(token.startswith("/*") and not token.endswith("*/") for token in tokens):
        raise ValueError(f"Unterminated linker script comment: {script}")
    tokens = [token for token in tokens if not token.startswith(("/*", "#"))]
    names: list[str] = []
    position = 0

    def arguments(dependencies: bool) -> None:
        nonlocal position
        if position >= len(tokens) or tokens[position] != "(":
            raise ValueError(f"Unsupported linker script syntax: {script}")
        position += 1
        while position < len(tokens) and tokens[position] != ")":
            token = tokens[position]
            position += 1
            if token == ",":
                continue
            if dependencies and token == "AS_NEEDED":
                arguments(True)
                continue
            name = token.removeprefix('"').removesuffix('"')
            if token in ("(", ";", '"') or any(char in name for char in ('"', "\\", "#", "*")):
                raise ValueError(f"Unsupported linker script token {token!r}: {script}")
            if dependencies:
                if not Path(name).is_absolute():
                    raise ValueError(
                        f"Linker script input requires search-path resolution: {token!r} in {script}"
                    )
                names.append(name)
        if position >= len(tokens):
            raise ValueError(f"Unterminated linker script command: {script}")
        position += 1

    while position < len(tokens):
        command = tokens[position]
        position += 1
        if command == ";":
            continue
        if command not in ("INPUT", "GROUP", "OUTPUT_FORMAT", "OUTPUT_ARCH"):
            raise ValueError(f"Unsupported linker script command {command!r}: {script}")
        arguments(command in ("INPUT", "GROUP"))
    return names


def _linker_script_inputs(paths: set[Path], sysroot: Path | None) -> set[Path]:
    """Expand supported scripts; never accept dependencies we cannot resolve."""
    paths = set(paths)
    pending = [p for p in paths if p.is_file()]
    seen = set(pending)
    while pending:
        script = pending.pop()
        with script.open("rb") as stream:
            prefix = stream.read(8)
            if prefix.startswith((b"\x7fELF", b"!<arch>")):
                continue
            content = prefix + stream.read()
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Unsupported linker input format: {script}") from exc
        for name in _linker_script_names(text, script):
            dependency = Path(name)
            if sysroot is not None and sysroot in script.resolve().parents:
                dependency = sysroot / name.lstrip("/")
            dependency = dependency.resolve(strict=True)
            if not dependency.is_file():
                raise ValueError(f"Expected a linker input file: {dependency} in {script}")
            paths.add(dependency)
            if dependency not in seen:
                seen.add(dependency)
                pending.append(dependency)
    return paths


_CONSOLE_BODIES = (
    """import re
import sys
from ptoas._cli import main
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\\.pyw|\\.exe)?$', '', sys.argv[0])
    sys.exit(main())
""",
    """import sys
from ptoas._cli import main
if __name__ == '__main__':
    if sys.argv[0].endswith('-script.pyw'):
        sys.argv[0] = sys.argv[0][:-11]
    elif sys.argv[0].endswith('.exe'):
        sys.argv[0] = sys.argv[0][:-4]
    sys.exit(main())
""",
)
_CONSOLE_TREES = frozenset(ast.dump(ast.parse(body)) for body in _CONSOLE_BODIES)

# Execute only after validating the standard console-script grammar. Preserve
# the shebang path: resolving its symlink first would discard the active venv.
# Match script-mode sys.path[0], then inventory the selected installed wheels,
# their declared dependency and the interpreter's startup inputs. No compiler
# stage is invoked by this probe.
_WHEEL_PROBE = """
import sys
sys.path[0] = sys.argv[1]
import importlib.metadata as metadata
import json
from pathlib import Path
import sysconfig
import ptoas._cli
import ptoas._core
import ptodsl
import TileOps
import SoftOps
import numpy

roots = set()
for name in ('ptoas', 'numpy'):
    distribution = metadata.distribution(name)
    requirements = distribution.requires or []
    if requirements != (['numpy'] if name == 'ptoas' else []):
        raise ValueError('Unsupported wheel dependency declarations for ' + name + ': ' + str(requirements))
    if name == 'ptoas' and Path(distribution.locate_file('ptoas/_online')).exists():
        raise ValueError('Online-built PTOAS extensions require a separate dependency adapter')
    if name == 'ptoas' and not any(
        entry.group == 'console_scripts' and entry.name == 'ptoas'
        and entry.value == 'ptoas._cli:main' for entry in distribution.entry_points
    ):
        raise ValueError('PTOAS wheel does not declare the selected entry point')
    files = distribution.files
    if not files or not any(str(p).endswith('.dist-info/WHEEL') for p in files):
        raise ValueError('Expected an installed wheel with a file inventory: ' + name)
    for file in files:
        if str(file).endswith('.pyc'):
            continue
        selected = file if file.parts[0] == '..' else Path(file.parts[0])
        roots.add(str(distribution.locate_file(selected).absolute()))
    packages = ('ptoas', 'ptodsl', 'TileOps', 'SoftOps') if name == 'ptoas' else ('numpy',)
    for package in packages:
        expected = Path(distribution.locate_file(package)).resolve(strict=True)
        for loaded_name, loaded in tuple(sys.modules.items()):
            if loaded_name == package or loaded_name.startswith(package + '.'):
                origin = getattr(loaded, '__file__', None)
                if origin and expected not in Path(origin).resolve(strict=True).parents:
                    raise ValueError('Wheel uses an external import redirect: ' + loaded_name)

# Include startup hooks and other already-loaded modules outside the wheel
# roots (e.g. a sitecustomize module or a .pth-installed startup helper).
for module in tuple(sys.modules.values()):
    for directory in getattr(module, '__path__', ()):
        roots.add(str(Path(directory).absolute()))
    origin = getattr(module, '__file__', None)
    if origin and Path(origin).is_file():
        roots.add(str(Path(origin).absolute()))
for directory in sys.path:
    if directory and Path(directory).is_dir():
        roots.update(str(p.absolute()) for p in Path(directory).glob('*.pth'))
venv_config = Path(sys.prefix) / 'pyvenv.cfg'
if venv_config.is_file():
    roots.add(str(venv_config))
print(json.dumps({'roots': sorted(roots), 'stdlib': sysconfig.get_path('stdlib')}))
"""


def _console_interpreter(launcher: Path) -> Path:
    source = launcher.read_text()
    first = source.splitlines()[0]
    # Absolute, argument-free shebangs from pip/uv; shell trampolines and
    # arbitrary Python launchers require their own dependency adapters.
    if not first.startswith("#!/") or len(shlex.split(first[2:])) != 1:
        raise ValueError(f"Unsupported PTOAS console-script shebang: {launcher}")
    try:
        tree = ast.dump(ast.parse(source))
    except SyntaxError as exc:
        raise ValueError(f"Unsupported PTOAS console-script grammar: {launcher}") from exc
    if tree not in _CONSOLE_TREES:
        raise ValueError(f"Unsupported PTOAS console-script grammar: {launcher}")
    interpreter = Path(first[2:])
    _executable(str(interpreter))  # Require an actual ELF interpreter.
    return interpreter


def _wheel_inputs(launcher: Path) -> set[Path]:
    interpreter = _console_interpreter(launcher)
    try:
        output = _run([str(interpreter), "-c", _WHEEL_PROBE, str(launcher.parent)])
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"Cannot inventory PTOAS wheel at {launcher}: {exc.stderr.strip()}") from exc
    inventory = json.loads(output)
    paths = {launcher, interpreter, *(Path(p) for p in inventory["roots"])}
    stdlib = Path(inventory["stdlib"])
    paths.update(
        p for p in stdlib.iterdir() if p.name not in ("site-packages", "dist-packages", "__pycache__")
    )
    paths.update(_elf_inputs(interpreter.resolve(strict=True)))
    # Reduce nested roots before visiting native extensions and bundled ELF
    # libraries, including wheel-specific directories such as numpy.libs.
    for root in _component(paths).roots:
        candidates = root.path.rglob("*") if root.path.is_dir() else (root.path,)
        for native in candidates:
            if native.is_file() and ".so" in native.name:
                with native.open("rb") as stream:
                    if stream.read(4) == b"\x7fELF":
                        paths.update(_elf_inputs(native.resolve(strict=True)))
    return paths


def _ptoas_inputs(launcher: Path, ancestors: frozenset[Path] = frozenset()) -> set[Path]:
    """Inventory ELF releases and the packaged-CPython launcher grammar.

    Only literal exec forwarding and the release's root/interpreter/loader
    assignments are accepted. Arbitrary shell execution is not analyzed.
    """
    launcher = launcher.resolve(strict=True)
    if launcher in ancestors:
        raise ValueError(f"PTOAS launcher cycle: {launcher}")
    with launcher.open("rb") as stream:
        is_elf = stream.read(4) == b"\x7fELF"
    if is_elf:
        root = launcher.parent.parent if launcher.parent.name == "bin" else launcher.parent
        paths = _elf_inputs(launcher)
        paths.update(child for child in (root / "lib", root / "share", root / "include") if child.exists())
        return paths
    with launcher.open() as stream:
        shebang = stream.readline()
    if shebang.startswith("#!/") and "python" in Path(shebang[2:].strip()).name:
        return _wheel_inputs(launcher)
    lines = [
        line.strip()
        for line in launcher.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    forwarding = re.fullmatch(r'exec "([^"$]+)" "\$@"', "\n".join(lines))
    if forwarding:
        return {
            launcher,
            *_elf_inputs(_executable("bash")),
            *_ptoas_inputs(Path(forwarding[1]), ancestors | {launcher}),
        }
    if len(lines) != 5:
        raise ValueError(f"Unsupported PTOAS launcher dependency grammar: {launcher}")
    root_match = re.fullmatch(r'PTOAS_ROOT="([^"$]+)"', lines[0])
    python_match = re.fullmatch(r'PTOAS_PY="\$\{PTOAS_PYTHON:-([^"$]+)\}"', lines[1])
    if (
        root_match is None
        or python_match is None
        or lines[2:]
        != [
            "unset PYTHONHOME",
            'export LD_LIBRARY_PATH="${PTOAS_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"',
            'exec "$PTOAS_PY" "${PTOAS_ROOT}/bin/ptoas" "$@"',
        ]
    ):
        raise ValueError(f"Unsupported PTOAS launcher dependency grammar: {launcher}")
    root = Path(root_match[1]).resolve(strict=True)
    interpreter = _executable(os.environ.get("PTOAS_PYTHON") or python_match[1])
    wrapper = root / "bin/ptoas"
    if not (root / "ptoas/_cli.py").is_file() or not wrapper.is_file():
        raise ValueError(f"Incomplete packaged PTOAS tree: {root}")
    if (root / "ptoas/_online").exists():
        raise ValueError("Online-built PTOAS extensions require a separate dependency adapter")
    library_path = str(root / "lib") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    # Query interpreter resources without loading PTOAS or running a compiler.
    # Match the launcher's PYTHONHOME removal and library search environment.
    environment = dict(os.environ)
    environment.pop("PYTHONHOME", None)
    environment["LD_LIBRARY_PATH"] = library_path
    probe = subprocess.run(
        [
            str(interpreter),
            "-c",
            "import json,sys,sysconfig; print(json.dumps([sys.prefix,"
            "sysconfig.get_config_var('EXT_SUFFIX'),sysconfig.get_path('stdlib')]))",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    prefix, suffix, stdlib = json.loads(probe.stdout)
    prefix = Path(prefix).resolve(strict=True)
    if prefix not in interpreter.parents or not (root / "ptoas" / f"_core{suffix}").is_file():
        raise ValueError("PTOAS requires its compatible self-contained CPython installation")
    paths = {launcher, root, prefix, *_elf_inputs(_executable("bash"))}
    paths.update(_elf_inputs(interpreter, library_path))
    for directory in (root, Path(stdlib) / "lib-dynload"):
        for native in directory.rglob("*.so"):
            with native.open("rb") as stream:
                if stream.read(4) == b"\x7fELF":
                    paths.update(_elf_inputs(native, library_path))
    return paths


def _discover(compiler: Any, ptoas: str, runtime_name: str) -> ToolchainInputs:
    if sys.platform != "linux":
        raise ValueError(f"Unsupported dependency discovery platform: {sys.platform}")
    ptoas_paths = _ptoas_inputs(Path(ptoas))
    pypto = _package("pypto")
    pypto.update(_elf_inputs(Path(sys.executable).resolve()))
    stdlib = Path(sysconfig.get_path("stdlib"))
    # Top-level stdlib resources, excluding separately installed third-party
    # distributions. CPython and extension modules are installation inputs.
    pypto.update(
        p for p in stdlib.iterdir() if p.name not in ("site-packages", "dist-packages", "__pycache__")
    )
    for native in (stdlib / "lib-dynload").glob("*.so"):
        pypto.update(_elf_inputs(native))
    runtime = _package("simpler") | _package("simpler_setup")
    runtime.update({compiler.project_root / "src", compiler.project_root / "build/lib"})
    native_interface = importlib.import_module("_task_interface")
    native_filename = native_interface.__file__
    if native_filename is None:
        raise ValueError("Runtime task interface has no native installation path")
    runtime.update(_elf_inputs(Path(native_filename).resolve()))
    from pypto.runtime.pto_isa import ensure_pto_isa_root  # noqa: PLC0415

    isa = Path(ensure_pto_isa_root())
    includes, sources = compiler.get_orchestration_cache_inputs(runtime_name)
    runtime.update(Path(p).resolve(strict=True) for p in sources)
    runtime.update(
        Path(p).resolve(strict=True)
        for p in (
            *includes,
            *compiler.get_kernel_include_dirs(runtime_name),
            *compiler.get_incore_include_dirs(),
        )
        if Path(p).exists()
    )
    orchestration = compiler._orchestration_toolchain(runtime_name)
    device = _gcc_inputs(_executable(orchestration.cxx_path))
    if compiler.platform.endswith("sim"):
        device.update(_gcc_inputs(_executable(compiler.gxx15.cxx_path)))
    else:
        ccec = _executable(compiler.ccec.cxx_path)
        device.update(_elf_inputs(ccec))
        device.update(_elf_inputs(_executable(compiler.ccec.linker_path)))
        # CANN's BiSheng installation contains its resource headers, device
        # libraries and subprograms; the SDK supplies AscendC headers as well.
        if ccec.parent.name != "bin" or ccec.parent.parent.name != "bisheng_compiler":
            raise ValueError(f"Unsupported CCEC installation layout: {ccec}")
        device.add(ccec.parent.parent)
        for core_type in ("aiv", "aic"):
            flags = [flag for flag in compiler.ccec.get_compile_flags(core_type=core_type) if flag != "-c"]
            output = _run([str(ccec), *flags, "-E", "-v", os.devnull])
            device.update(_include_roots(output, ccec))
    return ToolchainInputs(
        _component(pypto), _component(runtime), _component({isa}), _component(ptoas_paths), _component(device)
    )


@lru_cache(maxsize=32)
def _compiler(platform: str, path: str | None, sdk: str | None, sanitizers: str) -> Any:
    """Cache immutable compiler metadata by effective selection inputs."""
    from pypto.runtime.kernel_compiler import KernelCompiler  # noqa: PLC0415

    return KernelCompiler(platform)


def capture_toolchain(platform: str, runtime_name: str) -> ToolchainIdentity:
    """Resolve effective tools; return explicit unavailable evidence on failure."""
    try:
        # These mechanisms can redirect arbitrary implicit inputs. Supporting
        # them requires tracing their dependencies, not hashing their strings.
        unsupported = {
            "CPATH": os.environ.get("CPATH"),
            "CPLUS_INCLUDE_PATH": os.environ.get("CPLUS_INCLUDE_PATH"),
            "C_INCLUDE_PATH": os.environ.get("C_INCLUDE_PATH"),
            "COMPILER_PATH": os.environ.get("COMPILER_PATH"),
            "GCC_EXEC_PREFIX": os.environ.get("GCC_EXEC_PREFIX"),
            "LIBRARY_PATH": os.environ.get("LIBRARY_PATH"),
            "LD_PRELOAD": os.environ.get("LD_PRELOAD"),
            "LD_AUDIT": os.environ.get("LD_AUDIT"),
        }
        for name, value in unsupported.items():
            if value:
                raise ValueError(f"Implicit dependency override requires a cache adapter: {name}")
        from pypto.runtime.kernel_compiler import KernelCompiler  # noqa: PLC0415

        compiler = _compiler(
            platform, os.environ.get("PATH"), os.environ.get("ASCEND_HOME_PATH"), KernelCompiler._sanitizers
        )
        if compiler._sanitizers:
            raise ValueError("Sanitized compiler installations require a cache adapter")
        # Installed paths (including launcher symlinks) are immutable for the
        # process lifetime. Re-resolve on selection changes, including cwd for
        # relative search roots; do not probe every executable on an object hit.
        selected = (
            platform,
            runtime_name,
            str(compiler.project_root),
            os.getcwd(),
            os.environ.get("PATH"),
            os.environ.get("ASCEND_HOME_PATH"),
            os.environ.get("PTOAS_ROOT"),
            os.environ.get("LD_LIBRARY_PATH"),
            os.environ.get("PTOAS_PYTHON"),
            os.environ.get("PYTHONPATH"),
            os.environ.get("PYTHONHOME"),
            os.environ.get("PYTHONUSERBASE"),
            os.environ.get("PYTHONNOUSERSITE"),
            os.environ.get("PYTHONSAFEPATH"),
            os.environ.get("PYTHONOPTIMIZE"),
        )
        with _discovery_lock:
            identity = _identities.get(selected)
            if identity is None:
                ptoas = find_ptoas_binary()
                if ptoas is None:
                    raise ValueError("PTOAS is unavailable")
                inputs = _discover(compiler, ptoas, runtime_name)
                identity = _identity_cache.capture(inputs)
                if identity.usable:
                    _identities[selected] = identity
            return identity
    except Exception as exc:
        # Discovery is optional cache evidence, including adapter/schema drift.
        # Actual compilation runs outside this boundary and still propagates errors.
        missing = ComponentInputs(unavailable_reason=str(exc))
        return _identity_cache.capture(ToolchainInputs(missing, missing, missing, missing, missing))
