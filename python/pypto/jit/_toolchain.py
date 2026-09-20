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
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
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
# ldd is subprocess-bound, not CPU-bound; this only caps how many run at once.
_LDD_WORKERS = 32
_identities: dict[tuple[Any, ...], ToolchainIdentity] = {}


# Every probe below reads its answer out of a tool's diagnostic output, and
# those strings are translated: on a non-English host gcc prints its own
# rendering of "#include <...> search starts here:", which no marker here
# matches. Pin the C locale for the probes rather than teach every parser
# every translation.
_C_LOCALE = {"LC_ALL": "C", "LANG": "C", "LANGUAGE": ""}


def _run(command: list[str]) -> str:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
        env={**os.environ, **_C_LOCALE},
    )
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


def _invocable(name: str) -> Path:
    """Return the compiler as the build invokes it, without resolving it.

    A compiler on PATH may be a wrapper that dispatches on argv[0]: ccache
    installs a directory of symlinks, one per compiler name, every one of them
    pointing at the single ccache binary, and decides which compiler to run
    from the name it was called by. Resolving that symlink first discards the
    name, and the wrapper then answers ``-print-prog-name`` about *itself* --
    ccache rejects the option outright -- so discovery fails on a host whose
    PATH puts those shims first, which is the ordinary state of a build
    machine. Invoking the path the build invokes keeps the dispatch intact and
    reaches the real compiler underneath.

    The file behind the name still has to be an ELF image; a wrapper written
    as a shell script needs its own dependency adapter, exactly as before.
    """
    selected = shutil.which(name)
    if selected is None:
        raise ValueError(f"Compiler executable is unavailable: {name}")
    path = Path(selected)
    with path.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise ValueError(f"Unsupported compiler launcher (requires dependency adapter): {selected}")
    return path


def _elf_inputs(path: Path, library_path: str | None = None) -> set[Path]:
    """ldd reports the loader's transitive resolution, including the interpreter."""
    with path.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise ValueError(f"Expected an ELF installation input: {path}")
    environment = {**os.environ, **_C_LOCALE}
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


def _elf_inputs_many(natives: Iterable[Path], library_path: str | None = None) -> set[Path]:
    """Resolve several ELF dependency closures at once.

    Each ``ldd`` is an independent subprocess whose wait releases the GIL, so a
    bounded thread pool turns discovery's dominant serial cost into roughly one
    round trip. Failure behaves as the serial loop did: ``map`` re-raises the
    first worker error, and a partial inventory is never returned.
    """
    ordered = list(natives)
    if not ordered:
        return set()

    # Reproduce each caller's own call shape: a site with no library path called
    # _elf_inputs with one argument, and a stub standing in for it may accept
    # only that one.
    def resolve(native: Path) -> set[Path]:
        return _elf_inputs(native) if library_path is None else _elf_inputs(native, library_path)

    if len(ordered) == 1:
        return resolve(ordered[0])
    paths: set[Path] = set()
    with ThreadPoolExecutor(min(_LDD_WORKERS, len(ordered))) as pool:
        for closure in pool.map(resolve, ordered):
            paths.update(closure)
    return paths


def _component(paths: set[Path]) -> ComponentInputs:
    # Parents already enumerate child contents. Retain logical paths in the
    # inventory; resolving every root would lose compiler selection aliases.
    # Ancestors sort first, so a containing root has already been accepted by the
    # time a descendant is tested: walking the candidate's own parents replaces
    # rescanning every accepted root. resolve(strict=True) discarded its result,
    # so it only asserted that the path exists; os.stat raises for the same
    # missing, dangling and symlink-loop cases without the per-component
    # readlink walk.
    ordered = sorted(paths, key=lambda p: (len(p.parts), str(p)))
    roots: list[Path] = []
    accepted: set[Path] = set()
    for path in ordered:
        os.stat(path)
        if accepted.isdisjoint(path.parents):
            roots.append(path)
            accepted.add(path)
    return ComponentInputs(tuple(ContentRoot(p) for p in roots), unavailable_reason=None)


def _package(name: str) -> set[Path]:
    """Inventory every tree a package actually loads from.

    An editable install splits one package across two: scikit-build-core maps
    the Python sources to the checkout and leaves the built extensions as real
    files under ``site-packages/<name>``, which is the layout the documented
    ``pip install -e`` workflow produces. Refusing that split does not make the
    identity safer -- it makes the whole toolchain unavailable, and with it the
    persistent cache -- so a submodule loading from outside the first tree adds
    *its* tree instead.

    A redirect is only accepted when it still lands inside a directory named
    for the package: that is the build system placing the package's own files,
    and every later import from it is covered because the whole directory is
    inventoried. A redirect anywhere else is still refused, because nothing
    bounds what it would drag in.
    """
    module = importlib.import_module(name)
    filename = getattr(module, "__file__", None)
    if filename is None:
        raise ValueError(f"Compiler module has no inspectable installation: {name}")
    roots = {Path(filename).resolve().parent}
    for imported_name, imported in tuple(sys.modules.items()):
        if imported_name == name or imported_name.startswith(f"{name}."):
            origin = getattr(imported, "__file__", None)
            if origin is None:
                continue
            selected = Path(origin).resolve(strict=True)
            if any(root == selected.parent or root in selected.parents for root in roots):
                continue
            anchor = next((parent for parent in selected.parents if parent.name == name), None)
            if anchor is None:
                raise ValueError(f"Compiler package uses an external import redirect: {imported_name}")
            roots.add(anchor)
    paths = set(roots)
    for root in roots:
        paths.update(_elf_inputs_many(root.rglob("*.so")))
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
    # pip 26 onwards. Same shim, expressed with str.removesuffix.
    """import sys
from ptoas._cli import main
if __name__ == '__main__':
    sys.argv[0] = sys.argv[0].removesuffix('.exe')
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
    natives = []
    for root in _component(paths).roots:
        candidates = root.path.rglob("*") if root.path.is_dir() else (root.path,)
        for native in candidates:
            if native.is_file() and ".so" in native.name:
                with native.open("rb") as stream:
                    if stream.read(4) == b"\x7fELF":
                        natives.append(native.resolve(strict=True))
    paths.update(_elf_inputs_many(natives))
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
    environment = {**os.environ, **_C_LOCALE}
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
    natives = []
    for directory in (root, Path(stdlib) / "lib-dynload"):
        for native in directory.rglob("*.so"):
            with native.open("rb") as stream:
                if stream.read(4) == b"\x7fELF":
                    natives.append(native)
    paths.update(_elf_inputs_many(natives, library_path))
    return paths


def _unaccounted_checkout_state(checkout: Path) -> str:
    """Report everything in ``checkout`` that its committed revision does not cover.

    ``--ignored`` is the point: the resolver's own cleanliness check omits
    ignored paths, so a generated file sitting in the tree leaves both the
    revision and that check unchanged. Anything reported here -- ignored,
    untracked or modified -- means the revision no longer describes the bytes.
    An unusable git answer reports itself rather than passing as clean.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--ignored"],
            cwd=checkout,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"git status is unavailable: {exc}"
    if result.returncode != 0:
        return f"git status failed: {result.stderr.strip()[:200]}"
    return result.stdout.strip()


def _pto_isa_component(isa_root: Path) -> ComponentInputs:
    """Identify the ISA checkout by the pin its own resolution already verified.

    ``ensure_pto_isa_root`` returns a checkout only after proving it is clean
    and at the pinned commit -- git objects are content-addressed, so a clean
    tree at ``HEAD == pin`` *is* the pinned tree -- and re-clones it from the
    pin otherwise, never checking out over a dirty tree. Reading the same
    ~5.8k files again re-proves what that resolution established.

    That resolution decides cleanliness with ``git status --porcelain``, which
    omits paths the checkout ignores -- a build writing generated files into the
    ISA tree would not disturb it. Re-asking with ``--ignored`` closes that gap
    for 18ms against the 0.71s the content read costs, and covers the tracked
    state again at the same time, so anything at all in the tree that git does
    not account for sends this component back to its contents.

    Falls back to the content inventory whenever the revision cannot be read or
    the tree cannot be accounted for, so neither weakens anything.
    """
    from simpler_setup.pto_isa import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        get_pto_isa_head,
    )

    revision = get_pto_isa_head(str(isa_root))
    if not revision or _unaccounted_checkout_state(isa_root) != "":
        return _component({isa_root})
    return ComponentInputs(unavailable_reason=None, verified_revision=revision)


def _ptoas_component(ptoas_bin: str) -> ComponentInputs:
    """Identify the assembler by the version it reports about itself.

    Unlike PTO-ISA, nothing proves this installation's bytes: ptoas is an
    external tree selected by ``PTOAS_ROOT``, its releases carry no manifest
    the installer checks, and the sha256 in ``toolchain/versions.env`` names
    the downloaded wheel rather than anything reachable from the unpacked
    tree. This identity therefore rests on a deployment property -- that ptoas
    arrives as an unmodified published build -- and not on evidence PyPTO can
    check. A rebuild or a patch applied in place under an unchanged version is
    invisible here, where the full inventory would have caught it.

    The probe's complete output is the identity, not the number parsed out of
    it: the parser keeps only the numeric part, so a dev build's suffix -- the
    one marker that separates it from the release it came from -- would
    otherwise be discarded. check_ptoas_version already runs this probe once
    per executable, so no extra process is started, and an assembler it
    rejects never reaches this point.

    Falls back to the content inventory whenever the probe fails.
    """
    from pypto.backend._ptoas_locate import check_ptoas_version  # noqa: PLC0415

    try:
        reported = check_ptoas_version(ptoas_bin).strip()
    except RuntimeError:
        reported = ""
    if not reported:
        return _component(_ptoas_inputs(Path(ptoas_bin)))
    return ComponentInputs(unavailable_reason=None, reported_version=reported)


_CANN_INSTALL_INFO = "ascend_toolkit_install.info"


def _cann_install_version(cann_root: Path) -> str:
    """Return the build the CANN installation states about itself, or "".

    ``innerversion`` is preferred over ``version``: it carries the vendor's
    build number, so two builds of one release are distinguishable, while the
    release number alone is not. Exactly one install-info *file* must be
    present -- several would mean this is not a single installation -- but an
    installation reaches its own through more than one path, so candidates are
    counted by the file they name: CANN ships `arm64-linux` as a symlink to
    `aarch64-linux`, and both spell the same inode. The value must be
    non-empty, or the caller reads the contents instead.
    """
    try:
        found = {path.resolve(strict=True) for path in cann_root.glob(f"*/{_CANN_INSTALL_INFO}")}
    except OSError:
        return ""
    if len(found) != 1:
        return ""
    fields: dict[str, str] = {}
    try:
        for line in found.pop().read_text().splitlines():
            key, separator, value = line.partition("=")
            if separator:
                fields[key.strip()] = value.strip()
    except OSError:
        return ""
    return fields.get("innerversion") or fields.get("version") or ""


def _outside(paths: set[Path], install_root: Path) -> set[Path]:
    """Return the paths an installation does not own."""
    return {p for p in paths if p != install_root and install_root not in p.parents}


def _discover(compiler: Any, ptoas: str, runtime_name: str) -> ToolchainInputs:
    """Collect the compiler, linker, SDK and PTO assembler inputs for cache identity."""
    if sys.platform != "linux":
        raise ValueError(f"Unsupported dependency discovery platform: {sys.platform}")
    ptoas_component = _ptoas_component(ptoas)
    pypto = _package("pypto")
    pypto.update(_elf_inputs(Path(sys.executable).resolve()))
    stdlib = Path(sysconfig.get_path("stdlib"))
    # Top-level stdlib resources, excluding separately installed third-party
    # distributions. CPython and extension modules are installation inputs.
    pypto.update(
        p for p in stdlib.iterdir() if p.name not in ("site-packages", "dist-packages", "__pycache__")
    )
    pypto.update(_elf_inputs_many((stdlib / "lib-dynload").glob("*.so")))
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
    device = _gcc_inputs(_invocable(orchestration.cxx_path))
    cann_root: Path | None = None
    if compiler.platform.endswith("sim"):
        device.update(_gcc_inputs(_invocable(compiler.sdk.gxx15.cxx_path)))
    else:
        ccec = _executable(compiler.sdk.ccec.cxx_path)
        device.update(_elf_inputs(ccec))
        device.update(_elf_inputs(_executable(compiler.sdk.ccec.linker_path)))
        # CANN's BiSheng installation contains its resource headers, device
        # libraries and subprograms; the SDK supplies AscendC headers as well.
        if ccec.parent.name != "bin" or ccec.parent.parent.name != "bisheng_compiler":
            raise ValueError(f"Unsupported CCEC installation layout: {ccec}")
        device.add(ccec.parent.parent)
        # <cann>/tools/bisheng_compiler/bin/ccec -- the layout checked above
        # fixes the first two levels, so require the third before trusting it
        # to name the installation whose stated build covers these files.
        if ccec.parent.parent.parent.name == "tools":
            cann_root = ccec.parents[3]
        for core_type in ("aiv", "aic"):
            flags = [
                flag for flag in compiler.sdk.ccec.get_compile_flags(core_type=core_type) if flag != "-c"
            ]
            output = _run([str(ccec), *flags, "-E", "-v", os.devnull])
            device.update(_include_roots(output, ccec))
    # The device compiler runs on the host, so its inputs come from two
    # sources: the CANN installation, which states its own build, and files the
    # host OS provides, which state nothing. Cover each with the evidence it
    # actually has rather than letting one version speak for both. Without a
    # usable CANN version every input is read, as before.
    cann_version = _cann_install_version(cann_root) if cann_root is not None else ""
    if cann_root is not None and cann_version:
        device_component = replace(_component(_outside(device, cann_root)), reported_version=cann_version)
    else:
        device_component = _component(device)
    return ToolchainInputs(
        _component(pypto),
        _component(runtime),
        _pto_isa_component(isa),
        ptoas_component,
        device_component,
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
