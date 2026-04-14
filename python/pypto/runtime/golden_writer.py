# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Golden script writer for the PyPTO runtime module.

Generates a ``golden.py`` file compatible with Simpler's CodeRunner from a list
of :class:`TensorSpec` objects and a user-supplied golden function.

:func:`write_golden` materialises all tensor data via
:meth:`TensorSpec.create_tensor` and saves them as ``.pt`` files under
``data/in/`` and ``data/out/`` subdirectories co-located with ``golden.py``.
An explicit ``data_dir`` can redirect storage to any path.  When the target
directory already contains the required files they are reused.

Storage rules by tensor type:

- **Pure input** (``is_output=False``): saved to ``data/in/{name}.pt`` only.
- **Pure output** (``is_output=True``, ``init_value=None``): saved to
  ``data/out/{name}.pt`` only; ``generate_inputs`` initialises with
  ``torch.zeros``.
- **Inout** (``is_output=True``, ``init_value`` set): saved to both
  ``data/in/{name}.pt`` (initial value) and ``data/out/{name}.pt``
  (golden result); ``generate_inputs`` loads from ``data/in/``.

Generated file format (data-file mode)::

    from pathlib import Path
    import torch

    _DATA_DIR = Path(__file__).parent / "data"

    __outputs__ = ["out"]
    RTOL = 1e-5
    ATOL = 1e-5

    def generate_inputs(params):
        query = torch.load(_DATA_DIR / "in" / "query.pt", weights_only=True)
        out   = torch.zeros((128,), dtype=torch.float32)
        return [
            ("query", query),
            ("out", out),
        ]

    def compute_golden(tensors, params):
        tensors["out"][:] = tensors["query"].float().sum(dim=-1, keepdim=True)
"""

import ast
import builtins
import inspect
import math
import re
import textwrap
import types
from collections.abc import Callable
from pathlib import Path

import torch

from .tensor_spec import SCALAR_CTYPE_MAP, ScalarSpec, TensorSpec

_AUTO_IMPORT_LINES = {
    "struct": "import struct",
}

_KNOWN_FACTORIES: dict[Callable, str] = {
    torch.randn: "torch.randn",
    torch.rand: "torch.rand",
    torch.zeros: "torch.zeros",
    torch.ones: "torch.ones",
}


def write_golden(
    tensor_specs: list[TensorSpec],
    golden_fn: Callable,
    output_path: Path,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    scalar_specs: list[ScalarSpec] | None = None,
    data_dir: Path | str | None = None,
) -> Path:
    """Generate and write a ``golden.py`` file for Simpler's CodeRunner.

    All tensor data is materialised and saved as ``.pt`` files under
    ``data/in/`` (inputs and inout initial values) and ``data/out/``
    (golden outputs computed by *golden_fn*).  By default these directories
    are created alongside the generated ``golden.py``; an explicit *data_dir*
    redirects storage elsewhere.

    When *data_dir* is provided the generated ``golden.py`` always references
    that directory.  If the directory already contains the required ``.pt``
    files they are reused; otherwise the directory is created and data files
    are generated there.

    Args:
        tensor_specs: Ordered list of tensor specifications matching the program's
            parameter list.
        golden_fn: A callable with signature ``(tensors, params)`` that computes
            expected outputs in-place (writes to ``tensors[output_name]``).
        output_path: Destination path for the generated file.
        rtol: Relative tolerance used by CodeRunner for result comparison.
        atol: Absolute tolerance used by CodeRunner for result comparison.
        scalar_specs: Optional list of scalar parameter specifications.  Scalar
            TaskArg entries appear after all tensor entries in the generated list.
        data_dir: Root directory for ``in/`` and ``out/`` data subdirectories.
            If the directory already contains data it is reused without
            regeneration.  If it does not exist it is created and data is
            generated there.  The generated ``golden.py`` always references
            this path.  When ``None`` (default), data is saved to
            ``<output_path>/../data/``.

    Returns:
        The resolved ``output_path`` after writing.
    """
    output_path = Path(output_path)
    if data_dir is not None:
        resolved_dir = Path(data_dir).resolve()
    else:
        resolved_dir = (output_path.parent / "data").resolve()

    if not _data_dir_has_files(resolved_dir, tensor_specs):
        tensors = _materialize_tensors(tensor_specs)

        in_data = {
            s.name: tensors[s.name] for s in tensor_specs if not s.is_output or s.init_value is not None
        }
        _save_data_files(in_data, resolved_dir / "in")

        golden_fn(tensors, None)
        out_data = {s.name: tensors[s.name] for s in tensor_specs if s.is_output}
        _save_data_files(out_data, resolved_dir / "out")

    content = generate_golden_source(
        tensor_specs,
        golden_fn,
        rtol,
        atol,
        scalar_specs=scalar_specs,
        data_dir=resolved_dir if data_dir is not None else None,
        use_data_files=True,
    )
    output_path.write_text(content, encoding="utf-8")
    return output_path


def generate_golden_source(
    tensor_specs: list[TensorSpec],
    golden_fn: Callable | None,
    rtol: float,
    atol: float,
    *,
    compute_golden_src: str | None = None,
    scalar_specs: list[ScalarSpec] | None = None,
    data_dir: Path | None = None,
    use_data_files: bool = False,
) -> str:
    """Build the full content of golden.py as a string.

    Args:
        tensor_specs: Ordered list of tensor specifications.
        golden_fn: A callable whose source is extracted to produce ``compute_golden``.
            Must be provided when *compute_golden_src* is ``None``.
        rtol: Relative tolerance written into the generated file.
        atol: Absolute tolerance written into the generated file.
        compute_golden_src: Pre-extracted ``compute_golden`` function source.
            When provided, *golden_fn* is ignored and this string is used directly.
            Use this when the source comes from a method that requires caller-side
            transformation (e.g. stripping ``self`` from the signature).
        scalar_specs: Optional list of scalar TaskArg specifications.  Entries are
            placed after all tensor entries in the returned list, matching the
            TaskArg slot order produced by orchestration codegen.
        data_dir: When set, the generated ``_DATA_DIR`` constant uses this
            absolute path.  When ``None`` and *use_data_files* is ``True``,
            a portable ``Path(__file__).parent / "data"`` expression is used.
        use_data_files: When ``True``, the generated ``generate_inputs``
            loads all tensors from ``.pt`` files via ``torch.load``.
            Defaults to ``False`` for backward compatibility with callers
            that rely on inline expressions.

    Returns:
        Full Python source for ``golden.py`` as a string.
    """
    emit_data_dir = use_data_files or data_dir is not None

    scalars = scalar_specs or []
    output_names = [spec.name for spec in tensor_specs if spec.is_output]

    # compute_golden: use caller-supplied source or extract from golden_fn
    if compute_golden_src is None:
        if golden_fn is None:
            raise ValueError("Either golden_fn or compute_golden_src must be provided")
        compute_golden_src = _extract_compute_golden(golden_fn)

    imports = _compute_golden_imports(compute_golden_src)
    if scalars:
        imports.append("import ctypes")
    if emit_data_dir:
        imports.append("from pathlib import Path")
    imports.append("import torch")

    # Pre-compute init expressions so that helper function preambles (e.g. for
    # callable init_values) are collected before we start building the output.
    preambles: dict[str, str] = {}
    init_exprs = [_init_expr(spec, preambles, use_data_dir=emit_data_dir) for spec in tensor_specs]

    lines: list[str] = [
        '"""',
        "Auto-generated golden script — do not edit by hand.",
        "",
        "Generated by pypto.runtime.golden_writer",
        '"""',
        "",
        *imports,
    ]

    if emit_data_dir:
        lines.append("")
        if data_dir is not None:
            escaped = str(data_dir).replace("\\", "\\\\")
            lines.append(f'_DATA_DIR = Path("{escaped}")')
        else:
            lines.append('_DATA_DIR = Path(__file__).parent / "data"')

    lines.extend(
        [
            "",
            f"__outputs__ = {output_names!r}",
            f"RTOL = {rtol}",
            f"ATOL = {atol}",
        ]
    )

    # Helper functions referenced by init expressions (e.g. copied from
    # callable init_values).
    for preamble in preambles.values():
        lines.append("")
        lines.append("")
        lines.extend(preamble.splitlines())

    lines.append("")
    lines.append("def generate_inputs(params):")
    lines.append('    """Generate inputs as a list of (name, value) tuples."""')

    # Tensor variable declarations
    for spec, expr in zip(tensor_specs, init_exprs, strict=True):
        lines.append(f"    {spec.name} = {expr}")

    # Scalar variable declarations
    for spec in scalars:
        ctype_ctor = SCALAR_CTYPE_MAP[spec.ctype]
        lines.append(f"    {spec.name} = {ctype_ctor}({spec.value!r})")

    lines.append("")
    lines.append("    return [")
    for spec in tensor_specs:
        lines.append(f'        ("{spec.name}", {spec.name}),')
    for spec in scalars:
        lines.append(f'        ("{spec.name}", {spec.name}),')
    lines.append("    ]")
    lines.append("")
    lines.append("")

    lines.extend(compute_golden_src.splitlines())
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _data_dir_has_files(data_dir: Path, tensor_specs: list[TensorSpec]) -> bool:
    """Return ``True`` if *data_dir* already contains all required ``.pt`` files.

    Checks ``in/`` for non-pure-output tensors and ``out/`` for output tensors.
    """
    in_dir = data_dir / "in"
    out_dir = data_dir / "out"
    for spec in tensor_specs:
        if spec.is_output:
            if not out_dir.is_dir() or not (out_dir / f"{spec.name}.pt").exists():
                return False
            if spec.init_value is not None and (
                not in_dir.is_dir() or not (in_dir / f"{spec.name}.pt").exists()
            ):
                return False
        elif not in_dir.is_dir() or not (in_dir / f"{spec.name}.pt").exists():
            return False
    return True


def _materialize_tensors(tensor_specs: list[TensorSpec]) -> dict[str, torch.Tensor]:
    """Create concrete tensors from specs and return them keyed by name."""
    return {spec.name: spec.create_tensor() for spec in tensor_specs}


def _save_data_files(data_files: dict[str, torch.Tensor], data_dir: Path) -> None:
    """Save materialised tensors to ``data_dir/{name}.pt``."""
    if not data_files:
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, tensor in data_files.items():
        torch.save(tensor, data_dir / f"{name}.pt")


def _compute_golden_imports(compute_golden_src: str) -> list[str]:
    """Return import lines required by the generated compute_golden body."""
    return [
        import_line
        for module_name, import_line in _AUTO_IMPORT_LINES.items()
        if re.search(rf"\b{module_name}\.", compute_golden_src)
    ]


def _init_expr(
    spec: TensorSpec,
    preambles: dict[str, str],
    *,
    use_data_dir: bool = False,
) -> str:
    """Return the Python expression (string) used to initialise this tensor in golden.py.

    When *use_data_dir* is ``True``, returns a ``torch.load(...)`` expression
    referencing a ``.pt`` file via the ``_DATA_DIR`` constant.

    When ``False`` (legacy mode), callable init_values that are not built-in
    factories have their source extracted and appended to *preambles* for
    emission before ``generate_inputs``.
    """
    if use_data_dir:
        if spec.is_output and spec.init_value is None:
            dtype_str = _torch_dtype_str(spec.dtype)
            shape_str = repr(tuple(spec.shape))
            return f"torch.zeros({shape_str}, dtype={dtype_str})"
        return f'torch.load(_DATA_DIR / "in" / "{spec.name}.pt", weights_only=True)'

    dtype_str = _torch_dtype_str(spec.dtype)
    shape_str = repr(tuple(spec.shape))

    # Outputs are always zero-initialised; their content is set by compute_golden.
    if spec.is_output or spec.init_value is None:
        return f"torch.zeros({shape_str}, dtype={dtype_str})"

    iv = spec.init_value

    if isinstance(iv, (int, float)):
        return f"torch.full({shape_str}, {iv!r}, dtype={dtype_str})"

    if isinstance(iv, torch.Tensor):
        return _tensor_literal_expr(iv, shape_str, dtype_str)

    if callable(iv):
        # Try to extract source (works for named functions with valid identifiers).
        expr = _extract_callable_expr(iv, preambles)
        if expr is not None:
            return f"torch.as_tensor({expr}, dtype={dtype_str})"
        # Fallback for C builtins (torch.randn etc.) whose source is not
        # available via inspect.
        factory_name = _KNOWN_FACTORIES.get(iv)
        if factory_name is not None:
            return f"{factory_name}({shape_str}, dtype={dtype_str})"
        raise ValueError(
            f"Callable init_value {iv!r} for tensor {spec.name!r} is not supported by "
            "golden_writer. Use a scalar, a torch.Tensor, a named function, "
            "or one of: torch.randn, torch.rand, torch.zeros, torch.ones."
        )

    raise TypeError(f"Unsupported init_value type {type(iv)!r} for tensor {spec.name!r}")


def _tensor_literal_expr(tensor: torch.Tensor, shape_str: str, dtype_str: str) -> str:
    """Generate an inline expression for a concrete torch.Tensor init_value."""
    if tensor.numel() == 0:
        return f"torch.zeros({shape_str}, dtype={dtype_str})"

    # Constant tensor
    if torch.all(tensor == tensor.flatten()[0]):
        val = tensor.flatten()[0].item()
        if val == 0.0:
            return f"torch.zeros({shape_str}, dtype={dtype_str})"
        if val == 1.0:
            return f"torch.ones({shape_str}, dtype={dtype_str})"
        return f"torch.full({shape_str}, {val!r}, dtype={dtype_str})"

    # Identity matrix
    if tensor.ndim == 2 and tensor.shape[0] == tensor.shape[1]:
        if torch.allclose(tensor.float(), torch.eye(tensor.shape[0])):
            return f"torch.eye({tensor.shape[0]}, dtype={dtype_str})"

    # Tensor small enough to inline as a list literal
    if tensor.numel() <= 100:
        return f"torch.tensor({tensor.tolist()!r}, dtype={dtype_str})"

    raise ValueError(
        f"Tensor init_value for {dtype_str} has {tensor.numel()} elements, too large to "
        "inline as a literal. Use a named function as init_value instead, e.g.:\n"
        "    def make_tensor():\n"
        "        return torch.arange(0, 2048, dtype=torch.int32)\n"
        '    TensorSpec("name", shape, dtype, init_value=make_tensor)'
    )


def _extract_callable_expr(fn: Callable, preambles: dict[str, str]) -> str | None:
    """Extract source from a callable and return an expression for golden.py.

    Copies the full function definition (with any closure constants) into
    *preambles* and returns ``fn_name()`` as the call expression.  Returns
    ``None`` if source extraction fails (e.g. C builtins) or if the callable
    name is not a valid Python identifier (e.g. lambdas whose ``__name__``
    is ``<lambda>``).
    """
    name = getattr(fn, "__name__", None)
    if name is None or not name.isidentifier():
        return None

    # Already extracted (e.g. multiple tensors share the same init function).
    if name in preambles:
        return f"{name}()"

    try:
        source = inspect.getsource(fn)
    except (TypeError, OSError):
        return None

    # Include closure constants so the function body can reference captured variables.
    closure_lines = _extract_closure_constants(fn)
    parts: list[str] = []
    if closure_lines:
        parts.append("\n".join(closure_lines))
    parts.append(textwrap.dedent(source))
    preambles[name] = "\n\n".join(parts)
    return f"{name}()"


def _torch_dtype_str(dtype: torch.dtype) -> str:
    """Return the string representation of a torch dtype (e.g. 'torch.float32')."""
    _map: dict[torch.dtype, str] = {
        torch.float32: "torch.float32",
        torch.float16: "torch.float16",
        torch.bfloat16: "torch.bfloat16",
        torch.float64: "torch.float64",
        torch.int8: "torch.int8",
        torch.int16: "torch.int16",
        torch.int32: "torch.int32",
        torch.int64: "torch.int64",
        torch.bool: "torch.bool",
    }
    result = _map.get(dtype)
    if result is None:
        raise ValueError(f"Unsupported dtype {dtype!r} in TensorSpec")
    return result


def _extract_compute_golden(golden_fn: Callable) -> str:
    """Extract source of *golden_fn* and rename it to ``compute_golden``.

    The resulting string is a properly-indented top-level function definition
    named ``compute_golden`` with the same body as *golden_fn*.

    When *golden_fn* is a bound method (detected via ``inspect.ismethod``), four
    additional transformations are applied, handling both single-line and
    multiline signatures:

    - The ``def name(`` is renamed to ``def compute_golden(``.
    - The ``self`` parameter is stripped (single-line: via regex; multiline: the
      standalone ``self,`` continuation line is dropped).
    - The return type annotation (``-> ...:``) is removed from the closing
      signature line, since ``compute_golden`` in golden.py is untyped.
    - Simple ``self.<attr>`` references in the function body are replaced with
      literal values from the bound instance.

    Args:
        golden_fn: User-provided golden function or bound method.
            Standalone functions must have signature ``(tensors, params)``.
            Bound methods must have signature ``(self, tensors, params)``.

    Returns:
        Source code for ``compute_golden(tensors, params)`` as a string.
    """
    try:
        source = inspect.getsource(golden_fn)
    except (TypeError, OSError) as exc:
        raise RuntimeError(
            f"Cannot extract source code from golden function {golden_fn!r}. "
            "Ensure it is defined in a .py file (not in a REPL or dynamically)."
        ) from exc

    # Remove extra leading indentation (e.g. if defined inside a class or function)
    source = textwrap.dedent(source)

    is_method = inspect.ismethod(golden_fn)
    original_name = golden_fn.__name__
    lines = source.splitlines()
    renamed_lines: list[str] = []
    in_sig = False  # True while still inside a multiline def signature

    for line in lines:
        if not in_sig and line.lstrip().startswith(f"def {original_name}("):
            new_line = line.replace(f"def {original_name}(", "def compute_golden(", 1)
            if is_method:
                # Strip self from the def line: handles "(self, ...)" and "(self,"
                new_line = re.sub(r"\(self,\s*", "(", new_line, count=1)
                new_line = new_line.replace("(self)", "()", 1)
            if new_line.rstrip().endswith(":"):
                # Single-line signature — strip return annotation if present
                if is_method and "->" in new_line:
                    new_line = new_line.split("->")[0].rstrip() + ":"
            else:
                in_sig = True  # Signature continues on subsequent lines
            renamed_lines.append(new_line)
        elif in_sig:
            if is_method and line.strip() in ("self,", "self"):
                continue  # Drop standalone self parameter line
            new_line = line
            if new_line.rstrip().endswith(":"):
                # Last line of a multiline signature
                if is_method and "->" in new_line:
                    new_line = new_line.split("->")[0].rstrip() + ":"
                in_sig = False
            renamed_lines.append(new_line)
        else:
            renamed_lines.append(line)

    source = "\n".join(renamed_lines)
    if is_method:
        source = _inline_bound_self_attributes(source, golden_fn.__self__)

    # Inject closure variable values as constants above the function definition.
    closure_lines = _extract_closure_constants(golden_fn)
    if closure_lines:
        source = "\n".join(closure_lines) + "\n\n" + source

    return source


def _inline_bound_self_attributes(source: str, bound_instance: object) -> str:
    """Replace literal-compatible ``self.<attr>`` references with bound values."""
    attr_names = set(re.findall(r"\bself\.([A-Za-z_][A-Za-z0-9_]*)\b", source))
    for attr_name in sorted(attr_names):
        if not hasattr(bound_instance, attr_name):
            raise RuntimeError(f"Bound golden method references missing attribute self.{attr_name}")

        attr_value = getattr(bound_instance, attr_name)
        literal_value = repr(attr_value)
        try:
            ast.literal_eval(literal_value)
        except (SyntaxError, ValueError) as exc:
            raise RuntimeError(
                "Bound golden method references unsupported non-literal attribute "
                f"self.{attr_name}={attr_value!r}"
            ) from exc

        source = re.sub(rf"\bself\.{attr_name}\b", literal_value, source)

    return source


def _extract_closure_constants(fn: Callable) -> list[str]:
    """Extract closure and global variable bindings as top-level constant assignments.

    Handles two categories of captured variables:

    1. **Closure variables** (free variables from an enclosing scope).
    2. **Global constants** referenced by the function that are simple scalar
       values (not builtins, modules, or callables).

    Only supports int, float, str, bool, and None values. Unsupported types
    are silently skipped (the generated code may still fail at runtime if those
    variables are actually used).
    """
    lines: list[str] = []
    seen: set[str] = set()
    _SIMPLE_TYPES = (int, float, str, bool, type(None))

    def _safe_repr(value: object) -> str | None:
        """Return a repr that is valid Python, or None for non-finite floats."""
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return repr(value)

    # 1. Closure (free) variables
    closure = getattr(fn, "__closure__", None)
    freevars = getattr(getattr(fn, "__code__", None), "co_freevars", ())
    if closure and freevars:
        for name, cell in zip(freevars, closure, strict=True):
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if isinstance(value, _SIMPLE_TYPES):
                literal = _safe_repr(value)
                if literal is not None:
                    lines.append(f"{name} = {literal}")
                    seen.add(name)

    # 2. Global constants — names referenced via LOAD_GLOBAL that resolve to
    #    simple scalar values in the function's __globals__.
    code = getattr(fn, "__code__", None)
    fn_globals = getattr(fn, "__globals__", {})
    if code and fn_globals:
        builtin_names = set(dir(builtins))
        for name in code.co_names:
            if name in seen or name in builtin_names:
                continue
            value = fn_globals.get(name)
            if value is None and name not in fn_globals:
                continue  # Name not in globals at all
            if isinstance(value, (types.ModuleType, type)):
                continue
            if callable(value):
                continue
            if isinstance(value, _SIMPLE_TYPES):
                literal = _safe_repr(value)
                if literal is not None:
                    lines.append(f"{name} = {literal}")
                    seen.add(name)

    return lines
