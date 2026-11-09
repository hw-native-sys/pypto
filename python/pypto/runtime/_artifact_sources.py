# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit packaging of generated configurations and supported extern sources."""

import ast
import gzip
import hashlib
import io
import os
import tarfile
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from pypto.jit._artifact_manifest import BuildKind, encode_manifest

from . import _vpto
from ._extern_includes import UnsupportedArtifactInput, literal_includes
from ._prebuilt import _encode_signature, _relative_file, chip_directories

if TYPE_CHECKING:
    from pypto.ir.distributed_compiled_program import DistributedCompiledProgram

# Keep the producer compatible with R1 startup_artifacts without importing the
# optional runtime/compiler into parent-only artifact packaging. The per-file
# ceiling is deliberately conservative; the other limits match the receiver.
_CLUSTER_MAX_FILE_BYTES = 64 * 1024 * 1024
_CLUSTER_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
_CLUSTER_MAX_UNPACKED_BYTES = 256 * 1024 * 1024
_CLUSTER_MAX_MEMBERS = 4096


def _read_cluster_source(directory: Path, path: str, remaining: int) -> bytes:
    """Bound every read before parsing or adding bytes to the source closure."""
    if remaining < 0:
        raise ValueError("Cluster source unpacked content exceeds size limit")
    source = _relative_file(directory, path)
    limit = min(_CLUSTER_MAX_FILE_BYTES, remaining)
    with source.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > _CLUSTER_MAX_FILE_BYTES:
        raise ValueError(f"Cluster source member exceeds file size limit: {path}")
    if len(data) > remaining:
        raise ValueError(f"Cluster source unpacked content exceeds size limit: {path}")
    return data


def read_kernel_config(path: Path) -> ModuleType:
    """Execute trusted generated configuration without reading or writing bytecode."""
    module = ModuleType("_pypto_artifact_config")
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def _external_path(path: Path) -> Path:
    """Reject links whose lexical include topology cannot be preserved by copying."""
    # Check before normalizing '..': link/../file can have different semantics
    # from lexical parent traversal. Seed paths already have a canonical root.
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise UnsupportedArtifactInput(
                f"Symbolic links in extern inputs require private compilation: {path}"
            )
    return path.resolve(strict=True)


def _external_files(source: Path, include_dirs: tuple[Path, ...]) -> dict[Path, bytes]:
    files: dict[Path, bytes] = {}

    def visit(path: Path) -> None:
        path = _external_path(path)
        if path in files:
            return
        data = path.read_bytes()
        files[path] = data
        for delimiter, name in literal_includes(data):
            if "\ufffd" in name:
                raise UnsupportedArtifactInput(
                    f"Non-UTF8 extern include name requires private compilation: {path}"
                )
            if Path(name).is_absolute() or "\\" in name:
                raise UnsupportedArtifactInput(f"Artifact extern includes must be relative: {path}: {name}")
            candidates = (path.parent / name,) if delimiter == '"' else ()
            candidates += tuple(root / name for root in include_dirs)
            dependency = next((p for p in candidates if p.is_file()), None)
            if dependency is not None:
                visit(dependency)
            elif delimiter == '"':
                raise UnsupportedArtifactInput(f"Unresolved local extern include: {path}: {name}")
            # Unresolved angle includes belong to the separately identified SDK.

    visit(source)
    return files


def _package_external(root: Path, kernel: dict[str, Any], index: int) -> None:
    source = Path(kernel["source"]).absolute()
    includes = tuple(Path(p).absolute() for p in (kernel.get("extra_include_dirs") or ()))
    # Resolve the shared packaging root once: workspace/home/automount aliases
    # above it do not change topology. Links below it still require fallback.
    lexical_root = Path(os.path.commonpath([source.parent, *includes]))
    canonical_root = lexical_root.resolve(strict=True)
    source = _external_path(canonical_root / source.relative_to(lexical_root))
    includes = tuple(_external_path(canonical_root / p.relative_to(lexical_root)) for p in includes)
    files = _external_files(source, includes)
    common = Path(os.path.commonpath([str(p.parent) for p in files] + [str(p) for p in includes]))
    destination = root / "extern" / str(index)
    for path, data in files.items():
        target = destination / path.relative_to(common)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    relocated = destination / source.relative_to(common)
    kernel["source"] = str(relocated.relative_to(root))
    relocated_includes = [destination / path.relative_to(common) for path in includes]
    for path in relocated_includes:
        path.mkdir(parents=True, exist_ok=True)
    kernel["extra_include_dirs"] = [str(path.relative_to(root)) for path in relocated_includes]


def _local_source(root: Path, source: str) -> str:
    path = Path(source).resolve(strict=True)
    if root not in path.parents:
        raise ValueError(f"Generated source is outside the artifact: {source}")
    relative = str(path.relative_to(root))
    _relative_file(root, relative)
    return relative


def package_generated_sources(directory: Path, kind: BuildKind) -> None:
    """Make a private generated tree self-contained before GENERATED publication.

    Literal local extern include graphs are supported. UnsupportedArtifactInput
    identifies packaging limitations for caller-owned private fallback; other
    failures propagate unchanged. SDK angle includes remain
    supplied by the explicitly identified toolchain. This mutates only the
    caller's private tree, never an ArtifactHandle's published directory.
    """
    directory = directory.resolve()
    if (directory / "artifact_manifest.json").exists():
        raise ValueError("Cannot package sources inside a published artifact")
    for chip in chip_directories(directory, kind).values():
        config = read_kernel_config(chip / "kernel_config.py")
        kernels = [dict(k) for k in config.KERNELS]
        for index, kernel in enumerate(kernels):
            if kernel.get("external", False):
                _package_external(chip, kernel, index)
            else:
                kernel["source"] = _local_source(chip, kernel["source"])
            kernel["signature"] = _encode_signature(kernel.get("signature", []))
            helpers = _vpto.source_paths(kernel.get("vpto_sources"))
            if helpers:
                packaged: list[str] = []
                for ordinal, source in enumerate(helpers):
                    path = Path(source).resolve(strict=True)
                    target = (
                        path
                        if path.is_relative_to(chip.resolve())
                        else chip / "vpto" / str(index) / str(ordinal) / path.name
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if path != target:
                        target.write_bytes(path.read_bytes())
                    packaged.append(_local_source(chip, str(target)))
                kernel["vpto_sources"] = packaged
        orchestration = dict(config.ORCHESTRATION)
        orchestration["source"] = _local_source(chip, orchestration["source"])
        orchestration["signature"] = _encode_signature(orchestration.get("signature", []))
        runtime_config = getattr(config, "RUNTIME_CONFIG", {})
        # Reject non-serializable build inputs instead of persisting repr(object).
        metadata = encode_manifest(
            {"kernels": kernels, "orchestration": orchestration, "runtime": runtime_config}
        )
        (chip / "kernel_config.json").write_bytes(metadata)
        text = (
            "from pathlib import Path\n"
            "from simpler.task_interface import ArgDirection as _D\n"
            "_ROOT_DIR = Path(__file__).parent\n"
            f"KERNELS = {kernels!r}\n"
            f"ORCHESTRATION = {orchestration!r}\n"
            f"RUNTIME_CONFIG = {runtime_config!r}\n"
            "for _entry in [*KERNELS, ORCHESTRATION]:\n"
            "    _entry['source'] = str(_ROOT_DIR / _entry['source'])\n"
            "    _entry['signature'] = [getattr(_D, s) for s in _entry['signature']]\n"
            "    if 'extra_include_dirs' in _entry:\n"
            "        _entry['extra_include_dirs'] = [\n"
            "            str(_ROOT_DIR / p) for p in _entry['extra_include_dirs']]\n"
            "    if 'vpto_sources' in _entry:\n"
            "        _entry['vpto_sources'] = [\n"
            "            str(_ROOT_DIR / p) for p in (_entry['vpto_sources'] or [])]\n"
        )
        (chip / "kernel_config.py").write_text(text, encoding="utf-8")


def validate_generated_sources(directory: Path, kind: BuildKind) -> None:
    """Reject generated handles that still depend on an external source tree."""
    directory = directory.resolve()
    for chip in chip_directories(directory, kind).values():
        config = read_kernel_config(chip / "kernel_config.py")
        for entry in [*config.KERNELS, config.ORCHESTRATION]:
            _local_source(chip, entry["source"])
            for helper in _vpto.source_paths(entry.get("vpto_sources")):
                _local_source(chip, helper)
            for include in entry.get("extra_include_dirs") or ():
                # Empty include directories have no payload and are intentionally
                # omitted by ArtifactStore. A missing -I directory is harmless.
                path = Path(include).resolve()
                if path != chip and chip not in path.parents:
                    raise ValueError(f"External include directory is outside the artifact: {path}")
                if path.exists() and not path.is_dir():
                    raise ValueError(f"External include path is not a directory: {path}")


def _cluster_source_payloads(directory: Path, entries: list[dict[str, Any]], budget: int) -> dict[str, bytes]:
    """Read one validated source closure within the remaining session budget."""
    from pypto.ir._cluster_artifacts import source_layout  # noqa: PLC0415

    directory = directory.resolve()
    paths = {
        "__init__.py",
        "orchestration/__init__.py",
        "orchestration/cluster_orch.py",
        "orchestration/_distributed_support.py",
        "next_levels/__init__.py",
    }
    for entry in entries:
        prefix = entry["logical_path"]
        source_path = f"{prefix}/orchestration/host_orch.py"
        paths.update({f"{prefix}/__init__.py", f"{prefix}/orchestration/__init__.py", source_path})
        if (directory / prefix / "orchestration/_distributed_support.py").exists():
            paths.add(f"{prefix}/orchestration/_distributed_support.py")
        if entry["sub_worker_paths"]:
            paths.add(f"{prefix}/sub_workers/__init__.py")
            paths.update(entry["sub_worker_paths"].values())
    payloads = {}
    for path in sorted(paths):
        data = _read_cluster_source(directory, path, budget)
        payloads[path] = data
        budget -= len(data)
    # Bound the complete declared closure before any source reaches the AST
    # parser; metadata comparison then uses the exact bytes that will be sent.
    for entry in entries:
        source_path = f"{entry['logical_path']}/orchestration/host_orch.py"
        if source_layout(payloads[source_path].decode("utf-8"), entry["entry_name"]) != entry["layout"]:
            raise ValueError(f"Cluster entry source layout differs from cold metadata: {source_path}")
    for path, data in payloads.items():
        if not path.endswith(".py"):
            continue
        for node in ast.walk(ast.parse(data, filename=path)):
            imports = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            )
            if any(name == "pypto" or name.startswith("pypto.") for name in imports):
                raise ValueError(f"Cluster runtime source imports PyPTO: {path}")
    return payloads


def _encode_cluster_payloads(payloads: dict[str, bytes]) -> tuple[str, bytes]:
    """Own a plain USTAR closure and stream it into a bounded gzip sink."""
    if len(payloads) > _CLUSTER_MAX_MEMBERS:
        raise ValueError("Cluster source archive has too many members")
    if sum(map(len, payloads.values())) > _CLUSTER_MAX_UNPACKED_BYTES:
        raise ValueError("Cluster source unpacked content exceeds size limit")

    class BoundedPayload(io.BytesIO):
        def write(self, buffer: Any) -> int:
            # Enforce the receiver budget on every gzip write, including the
            # header and final trailer emitted by GzipFile.close().
            if self.tell() + len(buffer) > _CLUSTER_MAX_PAYLOAD_BYTES:
                raise ValueError("Cluster gzip payload exceeds size limit")
            return super().write(buffer)

    stream = BoundedPayload()
    # Stream tar records into compression; never allocate the expanded archive.
    # Empty filename and fixed mtime keep the gzip envelope deterministic too.
    with gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
            for path, data in sorted(payloads.items()):
                member = tarfile.TarInfo(path)
                member.size = len(data)
                member.mode = 0o644
                member.uid = member.gid = member.mtime = 0
                member.uname = member.gname = ""
                archive.addfile(member, io.BytesIO(data))
    bundle = stream.getvalue()
    return hashlib.sha256(bundle).hexdigest(), bundle


_SESSION_HELPER_EXPORTS = (
    "_ChipCallables",
    "_bind_embedded_invocation",
    "_copy_return_tensor",
    "make_tensor_arg",
    "add_chip_tensor_arg",
    "_submit_chip",
    "_decode_scalar",
    "_encode_scalar",
    "_slice_tensor",
    "_reshape_tensor",
    "_record_dispatch_program",
    "_resolve_chip_worker",
    "_reset_dfx_dispatch_state",
    "_dfx_rank_label",
)


def build_cluster_session_bundle(
    programs: list["DistributedCompiledProgram"], bindings: list[dict[str, Any]] | None = None
) -> tuple[str, bytes]:
    """Relocate program closures and package each L3 entry's runtime bindings.

    Input artifacts are immutable. Every helper copy is validated before a
    source-relative shim replaces it in this privately owned aggregate.
    """
    payloads = {"programs/__init__.py": b"", "orchestration/__init__.py": b""}
    for portable in (
        "_distributed_support.py",
        "_cluster_controller.py",
        "_cluster_channel.py",
    ):
        remaining = _CLUSTER_MAX_UNPACKED_BYTES - sum(len(data) for data in payloads.values())
        payloads["orchestration/" + portable] = _read_cluster_source(
            Path(__file__).parent, portable, remaining
        )
    helper = payloads["orchestration/_distributed_support.py"]
    for index, program in enumerate(programs):
        remaining = _CLUSTER_MAX_UNPACKED_BYTES - sum(len(data) for data in payloads.values())
        closure = program.l3_entries
        assert closure is not None
        source = _cluster_source_payloads(program.output_dir, closure["entries"], remaining)
        prefix = f"programs/p{index}"
        if bindings is not None:
            for entry in closure["entries"]:
                logical = entry["logical_path"]
                path = f"{logical}/orchestration/host_orch.py"
                source[path] += (
                    f"\n{entry['qualname']}._pypto_embedded_bindings = {bindings[index][logical]!r}\n"
                ).encode()
        for path, data in source.items():
            if path.endswith("/_distributed_support.py"):
                if data != helper:
                    raise ValueError(f"Cluster session helper contract mismatch: p{index}/{path}")
                # Each level moves toward the session root; no sys.modules
                # alias or import hook is needed for identical class objects.
                relative = "." * len((prefix + "/" + path).split("/"))
                data = (
                    "from "
                    + relative
                    + "orchestration._distributed_support import (\n    "
                    + ", ".join(_SESSION_HELPER_EXPORTS)
                    + ",\n)\n"
                ).encode()
            elif path.endswith("/orchestration/__init__.py") and path != "orchestration/__init__.py":
                if data != b"from ....orchestration import _distributed_support\n":
                    raise ValueError(f"Cluster session helper package shim contract mismatch: {path}")
            payloads[prefix + "/" + path] = data
    return _encode_cluster_payloads(payloads)
