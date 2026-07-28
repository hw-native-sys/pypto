# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Render an interactive HTML memory map from a PyPTO pass dump.

Draws on-chip memory as a two-dimensional map — address across, lifetime down —
with the IR source alongside, so a tile's live range can be read against the
code that produced it and a reuse decision is one glance rather than a
cross-reference between tables.

Input is a ``passes_dump/NN_after_<Pass>.py`` file. Those dumps are valid Python
(they are what ``@pl.program`` would accept back), so the tile bindings are
parsed with :mod:`ast` rather than regexes, and the dump's own line numbers give
the lifetime axis. Per-space capacities come from the :mod:`pypto.backend`
:class:`~pypto.backend.Backend` interface, which the dump does not record.
"""

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pypto import backend as _backend
from pypto import ir as _ir

#: Function types that own tile memory and therefore get a map.
COMPUTE_FUNC_TYPES = frozenset({"AIC", "AIV", "InCore"})

#: Left-to-right panel order. Spaces the backend reports but that are missing
#: here are appended rather than dropped, so a new memory space still maps.
SPACE_ORDER = ("Vec", "Mat", "Left", "Right", "Acc", "Bias")

#: Used when the dump's target architecture cannot be determined.
DEFAULT_BACKEND = "Ascend910B"

_TEMPLATE = Path(__file__).parent / "templates" / "memory_map.html"
_DATA_PLACEHOLDER = "/*__PYPTO_MEMORY_MAP_DATA__*/null"
_PASS_NAME_RE = re.compile(r"^\d+_after_|\.py$")
_TARGET_ARCH_RE = re.compile(r"pto\.target_arch\s*=\s*\"([^\"]+)\"")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Tile:
    """One tile binding: a named SSA value with a MemRef and a live range."""

    name: str
    space: str
    base: str
    offset: int
    size: int
    shape: list[int]
    dtype: str
    op: str
    start: int  # first dump line mentioning the name
    end: int  # last dump line mentioning the name


@dataclass
class Box(Tile):
    """A drawn rectangle: one or more alias tiles sharing a slot and lifetime."""

    aliases: list[str] = field(default_factory=list)
    view: bool = False
    conflict: bool = False


@dataclass
class SpaceUsage:
    """Per-memory-space totals for one function."""

    space: str
    hwm: int  # high-water mark: max(offset + size)
    limit: int
    tiles: int
    bases: int


@dataclass
class FunctionMap:
    """Everything the page needs to draw one compute function."""

    name: str
    ftype: str
    src_start: int
    src_end: int
    source: list[str]
    spaces: list[SpaceUsage] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)


@dataclass
class BackendChoice:
    """Which backend supplied the per-space capacities, and how it was picked."""

    name: str  # BackendType member name, e.g. "Ascend910B"
    arch: str  # PTO target arch, e.g. "a2a3"
    detected: bool  # False when nothing identified the target and a default was assumed


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _dotted_name(node: ast.AST) -> str | None:
    """Flatten an attribute chain: ``pl.Mem.Vec`` -> ``"pl.Mem.Vec"``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _int_value(node: ast.AST) -> int | None:
    """Fold ``4096``, ``-1`` and ``pl.const(4096, pl.INT64)`` to an int."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _int_value(node.operand)
        return None if inner is None else -inner
    if isinstance(node, ast.Call) and (_dotted_name(node.func) or "").endswith(".const") and node.args:
        return _int_value(node.args[0])
    return None


def _find_memref(annotation: ast.AST) -> tuple[str, int, int] | None:
    """Extract ``(base, offset, size)`` from ``pl.MemRef(base, pl.const(off), size)``."""
    for node in ast.walk(annotation):
        if not isinstance(node, ast.Call):
            continue
        if not (_dotted_name(node.func) or "").endswith(".MemRef") or len(node.args) < 3:
            continue
        base_node = node.args[0]
        if isinstance(base_node, ast.Name):
            base = base_node.id
        elif isinstance(base_node, ast.Constant):
            base = str(base_node.value)
        else:
            continue
        offset, size = _int_value(node.args[1]), _int_value(node.args[2])
        if offset is not None and size is not None:
            return base, offset, size
    return None


def _find_space(annotation: ast.AST) -> str | None:
    """Extract the space name from a ``pl.Mem.<Space>`` attribute."""
    for node in ast.walk(annotation):
        parts = (_dotted_name(node) or "").split(".")
        if len(parts) >= 2 and parts[-2] == "Mem":
            return parts[-1]
    return None


def _annotation_elements(annotation: ast.AST) -> list[ast.expr]:
    """Return the subscript elements of ``pl.Tile[a, b, c]``."""
    if not isinstance(annotation, ast.Subscript):
        return []
    node = annotation.slice
    return list(node.elts) if isinstance(node, ast.Tuple) else [node]


def _is_tile_annotation(annotation: ast.AST) -> bool:
    return isinstance(annotation, ast.Subscript) and (_dotted_name(annotation.value) or "").endswith(".Tile")


def _shape_of(annotation: ast.AST) -> list[int]:
    elements = _annotation_elements(annotation)
    if not elements or not isinstance(elements[0], ast.List):
        return []
    dims: list[int] = []
    for element in elements[0].elts:
        value = _int_value(element)
        if value is None:
            return []
        dims.append(value)
    return dims


def _dtype_of(annotation: ast.AST) -> str:
    """Pick the ``pl.<DTYPE>`` element out of a tile annotation."""
    for element in _annotation_elements(annotation)[1:3]:
        name = _dotted_name(element)
        if name and name.count(".") == 1 and name.split(".")[1].isupper():
            return name.split(".")[1]
    return ""


def _function_kwargs(func: ast.FunctionDef) -> dict[str, str]:
    """Read the ``@pl.function(type=..., level=...)`` keywords as plain strings."""
    out: dict[str, str] = {}
    for decorator in func.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        if not (_dotted_name(decorator.func) or "").endswith(".function"):
            continue
        for keyword in decorator.keywords:
            if keyword.arg is None:
                continue
            dotted = _dotted_name(keyword.value)
            out[keyword.arg] = dotted.split(".")[-1] if dotted else ast.unparse(keyword.value)
    return out


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _collect_tiles(func: ast.FunctionDef) -> dict[str, Tile]:
    """Collect every tile binding in a function, keyed by SSA name."""
    tiles: dict[str, Tile] = {}
    loads: list[ast.Name] = []
    for stmt in ast.walk(func):
        if isinstance(stmt, ast.Name) and isinstance(stmt.ctx, ast.Load):
            loads.append(stmt)  # applied below, once every binding is known
        if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
            continue
        if not _is_tile_annotation(stmt.annotation):
            continue
        memref = _find_memref(stmt.annotation)
        space = _find_space(stmt.annotation)
        if memref is None or space is None or space == "DDR":
            continue

        base, offset, size = memref
        end_line = stmt.end_lineno or stmt.lineno
        existing = tiles.get(stmt.target.id)
        if existing is not None:
            # Same name rebound in a sibling if/else arm — one live range covers both.
            existing.start = min(existing.start, stmt.lineno)
            existing.end = max(existing.end, end_line)
            continue

        producer = ""
        if isinstance(stmt.value, ast.Call):
            producer = (_dotted_name(stmt.value.func) or "").removeprefix("pl.")
        tiles[stmt.target.id] = Tile(
            name=stmt.target.id,
            space=space,
            base=base,
            offset=offset,
            size=size,
            shape=_shape_of(stmt.annotation),
            dtype=_dtype_of(stmt.annotation),
            op=producer,
            start=stmt.lineno,
            end=end_line,
        )

    # Extend each live range to the last line that reads the name.
    for node in loads:
        tile = tiles.get(node.id)
        if tile is not None:
            tile.start = min(tile.start, node.lineno)
            tile.end = max(tile.end, node.lineno)
    return tiles


def parse_dump(path: Path) -> list[FunctionMap]:
    """Parse a pass dump into one :class:`FunctionMap` per compute function.

    Args:
        path: A ``passes_dump/NN_after_<Pass>.py`` file.

    Returns:
        Maps for every ``AIC`` / ``AIV`` / ``InCore`` function that binds at
        least one tile, in declaration order. Functions without tile memory
        (``Group``, ``Orchestration``) are skipped.

    Raises:
        SyntaxError: The dump is not parseable Python.
    """
    text = path.read_text()
    lines = text.splitlines()
    tree = ast.parse(text, filename=str(path))

    maps: list[FunctionMap] = []
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        for func in (n for n in cls.body if isinstance(n, ast.FunctionDef)):
            ftype = _function_kwargs(func).get("type", "")
            if ftype not in COMPUTE_FUNC_TYPES:
                continue
            tiles = _collect_tiles(func)
            if not tiles:
                continue
            start = func.decorator_list[0].lineno if func.decorator_list else func.lineno
            end = func.end_lineno or func.lineno
            maps.append(
                FunctionMap(
                    name=func.name,
                    ftype=ftype,
                    src_start=start,
                    src_end=end,
                    source=lines[start - 1 : end],
                    boxes=build_boxes(list(tiles.values())),
                )
            )
    return maps


# ---------------------------------------------------------------------------
# Box construction
# ---------------------------------------------------------------------------


def build_boxes(tiles: list[Tile]) -> list[Box]:
    """Fold alias tiles into drawable boxes and classify their overlaps.

    A phi/yield chain rebinds the same storage under a new SSA name on every
    iteration; drawing each rebind separately buries the picture under a stack
    of identical rectangles. Tiles sharing a slot are therefore merged when
    their live ranges touch. Tiles sharing a slot whose live ranges are
    *disjoint* stay separate — that is memory reuse, the thing worth seeing.

    Args:
        tiles: All tile bindings of one function.

    Returns:
        Boxes sorted by (space order, address, first line).
    """
    slots: dict[tuple[str, str, int, int], list[Tile]] = {}
    for tile in tiles:
        slots.setdefault((tile.space, tile.base, tile.offset, tile.size), []).append(tile)

    runs: list[list[Tile]] = []
    for members in slots.values():
        members.sort(key=lambda t: t.start)
        slot_runs: list[list[Tile]] = []
        run_end = 0
        for tile in members:
            if slot_runs and tile.start <= run_end:
                slot_runs[-1].append(tile)
                run_end = max(run_end, tile.end)
            else:
                slot_runs.append([tile])
                run_end = tile.end
        runs.extend(slot_runs)

    boxes = [_merge(run) for run in runs]
    # Sorted before classification so _classify_overlaps sees each space's boxes
    # in address order and can stop scanning at the first non-overlapping one.
    boxes.sort(key=lambda b: (_space_rank(b.space), b.offset, b.start))
    _classify_overlaps(boxes)
    return boxes


def _merge(run: list[Tile]) -> Box:
    """Fold a run of alias tiles into one box. `run` is sorted by `start`."""
    head = run[0]
    return Box(
        **{**asdict(head), "end": max(t.end for t in run)},
        aliases=[t.name for t in run[1:]],
    )


def _space_rank(space: str) -> int:
    return SPACE_ORDER.index(space) if space in SPACE_ORDER else len(SPACE_ORDER)


def _group_by_space(boxes: list[Box]) -> dict[str, list[Box]]:
    """Bucket boxes by memory space, preserving their incoming order."""
    grouped: dict[str, list[Box]] = {}
    for box in boxes:
        grouped.setdefault(box.space, []).append(box)
    return grouped


def _classify_overlaps(boxes: list[Box]) -> None:
    """Mark boxes that overlap in both address and lifetime.

    Two boxes on the *same* base are a view of one allocation
    (``tile.slice`` / ``transpose_view``) — expected, and the narrower one is
    flagged ``view``. An overlap across *different* bases means two independent
    allocations share bytes while both are live, which is an allocator bug;
    both sides are flagged ``conflict``.
    """
    for group in _group_by_space(boxes).values():
        for i, lhs in enumerate(group):
            for rhs in group[i + 1 :]:
                if rhs.offset >= lhs.offset + lhs.size:
                    break  # sorted by offset — nothing further can overlap lhs
                if lhs.start > rhs.end or rhs.start > lhs.end:
                    continue
                if lhs.base == rhs.base:
                    narrower = rhs if rhs.size <= lhs.size else lhs
                    narrower.view = True
                else:
                    lhs.conflict = rhs.conflict = True


# ---------------------------------------------------------------------------
# Space limits, from the backend
# ---------------------------------------------------------------------------


def backend_names() -> list[str]:
    """Every configurable :class:`BackendType` member name."""
    return list(_backend.BackendType.__members__)


def backend_instance(backend_name: str) -> _backend.Backend:
    """Look up a backend singleton by :class:`BackendType` member name.

    Args:
        backend_name: A :class:`BackendType` member name, e.g. ``"Ascend910B"``.

    Returns:
        The backend singleton.

    Raises:
        ValueError: ``backend_name`` is not a known backend.
    """
    backend_type = _backend.BackendType.__members__.get(backend_name)
    if backend_type is None:
        raise ValueError(f"unknown backend {backend_name!r}; known backends: {', '.join(backend_names())}")
    return _backend.get_backend_instance(backend_type)


def backend_limits(backend_name: str) -> dict[str, int]:
    """Read the capacity of every on-chip memory space a backend describes.

    Walking the SoC rather than querying a fixed list of spaces means a backend
    that carries an extra space (Ascend950 has ``Bias``) gets a real capacity
    instead of falling back to its own high-water mark, which would render that
    panel permanently full.

    Args:
        backend_name: A :class:`BackendType` member name, e.g. ``"Ascend910B"``.

    Returns:
        ``{space: capacity_bytes}``, excluding off-chip ``DDR``.

    Raises:
        ValueError: ``backend_name`` is not a known backend.
    """
    limits: dict[str, int] = {}
    for die in backend_instance(backend_name).soc.die_counts:
        for cluster in die.cluster_counts:
            for core in cluster.core_counts:
                for mem in core.mems:
                    if mem.mem_type != _ir.MemorySpace.DDR:
                        limits[mem.mem_type.name] = mem.mem_size
    return limits


def detect_target_arch(dump: Path) -> str | None:
    """Read the PTO target arch out of the case's generated ``.pto`` files.

    Args:
        dump: The pass dump; its case directory is the dump's grandparent.

    Returns:
        The ``pto.target_arch`` attribute (e.g. ``"a2a3"``), or ``None`` when
        the case has no ``ptoas/`` output to read it from.
    """
    ptoas_dir = dump.parent.parent / "ptoas"
    if not ptoas_dir.is_dir():
        return None
    for pto in sorted(ptoas_dir.glob("*.pto")):
        match = _TARGET_ARCH_RE.search(pto.read_text(errors="replace"))
        if match:
            return match.group(1)
    return None


def resolve_backend(dump: Path, requested: str | None = None) -> BackendChoice:
    """Decide which backend's capacities to scale the panels by.

    An explicit request wins. Otherwise the case's ``.pto`` target arch is
    matched against each backend's own ``get_pto_target_arch()``, so the mapping
    stays in step with the backends rather than duplicating a table here. With
    neither, :data:`DEFAULT_BACKEND` is assumed and flagged as such on the page.

    Args:
        dump: The pass dump being rendered.
        requested: An explicit :class:`BackendType` member name, or ``None``.

    Returns:
        The chosen backend, its arch, and whether it was actually determined.

    Raises:
        ValueError: ``requested`` is not a known backend.
    """

    def arch_of(name: str) -> str:
        return backend_instance(name).get_handler().get_pto_target_arch()

    if requested is not None:
        return BackendChoice(name=requested, arch=arch_of(requested), detected=True)

    arch = detect_target_arch(dump)
    if arch is not None:
        for name in backend_names():
            if arch_of(name) == arch:
                return BackendChoice(name=name, arch=arch, detected=True)

    return BackendChoice(name=DEFAULT_BACKEND, arch=arch_of(DEFAULT_BACKEND), detected=False)


def is_unallocated(function: FunctionMap) -> bool:
    """Report whether a function's MemRefs still lack assigned addresses.

    Before ``AllocateMemoryAddr`` every MemRef sits at offset 0, so the whole
    function collapses onto address 0 — every base overlaps every other one and
    the map is all false conflicts. One base per space at offset 0 is a real
    (trivial) layout, so the tell is *two or more bases* sharing offset 0.
    """
    bases_at_zero: dict[str, set[str]] = {}
    for box in function.boxes:
        if box.offset != 0:
            return False
        bases_at_zero.setdefault(box.space, set()).add(box.base)
    return any(len(bases) > 1 for bases in bases_at_zero.values())


def _summarize_spaces(boxes: list[Box], limits: dict[str, int]) -> list[SpaceUsage]:
    ordered: list[SpaceUsage] = []
    grouped = _group_by_space(boxes)
    for space in sorted(grouped, key=_space_rank):
        group = grouped[space]
        hwm = max(b.offset + b.size for b in group)
        ordered.append(
            SpaceUsage(
                space=space,
                hwm=hwm,
                # A space the backend does not describe still needs a panel
                # scale; its own high-water mark is the only honest one.
                limit=limits.get(space) or hwm,
                tiles=len(group),
                bases=len({b.base for b in group}),
            )
        )
    return ordered


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def pass_name_of(dump: Path) -> str:
    """``32_after_AllocateMemoryAddr.py`` -> ``AllocateMemoryAddr``."""
    return _PASS_NAME_RE.sub("", dump.name)


def render(dump: Path, backend_name: str | None = None) -> str:
    """Build the self-contained HTML page for a pass dump.

    Args:
        dump: A ``passes_dump/NN_after_<Pass>.py`` file.
        backend_name: Explicit :class:`BackendType` member name whose capacities
            scale the panels. ``None`` detects it from the case's ``.pto``
            output, falling back to :data:`DEFAULT_BACKEND`.

    Returns:
        A complete HTML document with no external references.

    Raises:
        ValueError: The dump binds no tiles, its MemRefs have no assigned
            addresses yet, or ``backend_name`` is unknown.
    """
    functions = parse_dump(dump)
    if not functions:
        raise ValueError(
            f"{dump}: no AIC/AIV/InCore function binds a tile MemRef. "
            f"Use a dump at or after AllocateMemoryAddr — earlier passes have no MemRef addresses."
        )

    unallocated = [f.name for f in functions if is_unallocated(f)]
    if unallocated:
        raise ValueError(
            f"{dump}: every MemRef in {', '.join(unallocated)} still sits at offset 0, "
            f"so addresses have not been assigned yet. "
            f"Use a dump at or after AllocateMemoryAddr."
        )

    choice = resolve_backend(dump, backend_name)
    limits = backend_limits(choice.name)
    for function in functions:
        function.spaces = _summarize_spaces(function.boxes, limits)

    payload = {
        "dump": dump.name,
        "pass_name": pass_name_of(dump),
        "backend": asdict(choice),
        "functions": [asdict(f) for f in functions],
    }
    # `</` would close the host <script> element early.
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return _TEMPLATE.read_text().replace(_DATA_PLACEHOLDER, data)


def resolve_dump(path: Path, pass_name: str) -> Path:
    """Accept either a dump file or a ``build_output/<case>/`` directory.

    Args:
        path: A dump ``.py`` file, a ``passes_dump/`` directory, or a case
            directory containing one.
        pass_name: Pass whose dump to pick when ``path`` is a directory.

    Returns:
        The resolved dump file.

    Raises:
        FileNotFoundError: No matching dump exists under ``path``.
    """
    if path.is_file():
        return path
    candidates = [path, path / "passes_dump"]
    for directory in candidates:
        matches = sorted(directory.glob(f"*_after_{pass_name}.py")) if directory.is_dir() else []
        if matches:
            return matches[-1]
    raise FileNotFoundError(
        f"no '*_after_{pass_name}.py' dump under {path}. "
        f"Compile with dump_passes enabled, or pass the dump file directly."
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="memory_map",
        description="Render an interactive HTML memory map from a PyPTO pass dump.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="a passes_dump/*.py file, or a build_output/<case>/ directory containing one",
    )
    parser.add_argument(
        "-p",
        "--pass-name",
        default="AllocateMemoryAddr",
        help="pass to pick when PATH is a directory (default: %(default)s)",
    )
    parser.add_argument("-o", "--output", type=Path, help="output .html (default: next to the dump)")
    parser.add_argument(
        "-b",
        "--backend",
        choices=backend_names(),
        help="backend whose memory capacities scale the panels "
        "(default: detected from the case's .pto target arch)",
    )
    args = parser.parse_args(argv)

    try:
        dump = resolve_dump(args.path, args.pass_name)
        page = render(dump, args.backend)
    except (FileNotFoundError, ValueError, SyntaxError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    output = args.output or dump.with_suffix(".memory_map.html")
    output.write_text(page)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
