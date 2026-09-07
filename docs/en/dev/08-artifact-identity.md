# Artifact Identity Foundations

The internal `pypto._identity` module provides content hashing and deterministic
record encoding for the persistent JIT cache proposed in [RFC #2653](https://github.com/hw-native-sys/pypto/issues/2653).
It is not connected to JIT dispatch or artifact lookup. Existing in-process
cache behavior and dispatch cost are unchanged.

This is the first part of the identity milestone. Automatic inventories of the
selected compilers, SDK resources, and dynamic dependencies remain a follow-up.
The runtime's existing compiler-version tokens are not complete content
identities and must not be promoted to persistent-cache keys.

## Typed records

`encode_record()` uses a versioned, type-tagged encoding. `digest_record()`
returns its full SHA-256 digest. Supported values are `None`, booleans, integers,
floats, strings, bytes, lists, tuples, and dictionaries with string keys.
Dictionary insertion order does not affect the result; sequence order does.

```python
from pypto._identity import digest_record

assert digest_record(True) != digest_record(1)
assert digest_record(1) != digest_record(1.0)
assert digest_record(0.0) != digest_record(-0.0)
assert digest_record({"rows": 32, "cols": 64}) == digest_record({"cols": 64, "rows": 32})
```

Floats retain their IEEE-754 bits, including signed zero and NaN payloads.
String values and dictionary keys preserve Python code points, distinguishing
non-BMP characters from explicit surrogate pairs before JSON serialization.
Unsupported objects, non-string dictionary keys, and cycles raise errors;
there is no `str()`/`repr()` fallback. Adapters must explicitly normalize enum,
path, and configuration values, preserving their semantic types. Bump the
identity schema when changing this encoding.

## File and directory inputs

`ContentRoot` captures an absolute path when constructed, preserving `..` so
the filesystem resolves preceding symlinks correctly. `fingerprint_content()`
hashes the bytes of each supplied file and recursively enumerates directory
inputs in sorted order. Root order and boundaries are retained. Paths remain
part of identity until the compiler supplies stable source-location and include
path mapping; identical content at different paths may therefore miss.

Directory inventories exclude `.git`, `__pycache__`, `.pyc`, and `.pyo` metadata.
All other resources contribute. `fingerprint_extra_sources()` filters directory
inputs to Python sources; directly supplied files contribute regardless of
extension. Empty additional-source lists are valid, but an empty required
installation inventory is unavailable.

Symlinks contribute resolved paths and target content. Broken links, directory
cycles, non-regular files, unreadable inputs, and detected read-time changes
produce an unavailable digest with a reason. Errors are not silently treated as
empty files or omitted dependencies. File metadata helps detect races but never
substitutes for content in an identity. This is not an atomic filesystem
snapshot: later integration must revalidate mutable source inputs before
publication, and installation inputs must remain immutable within a process.

Additional application sources are reread for each request. An application's
extra fingerprint supplements those inputs; it cannot replace missing files
or toolchain evidence.

## Installation inventories and missing evidence

`ToolchainInputs` has five required components: PyPTO, runtime, PTO-ISA, ptoas,
and the device/orchestration toolchain. A `ComponentInputs` inventory starts
unavailable, even when some file paths are known. A dependency-aware adapter
must establish completeness before clearing `unavailable_reason`.

The adapter must account for actual imported code, native libraries, compiler
resources, dynamic dependencies, headers, SDK/sysroot inputs, and link inputs.
It must share resolution with the compilation path, including ptoas launchers
and effective compiler selection. A file hash establishes that file's identity;
it does not prove that the file is the complete dependency set.

`InstallationIdentityCache.capture()` reports every unavailable component. Its
result has `usable=False` and `digest=None` if any component is incomplete or
unreadable. There is no shared `UNKNOWN` key. An extra application fingerprint
cannot turn this result into a usable toolchain identity.

Successful component reads are memoized by their complete resolved inventory,
with synchronization for concurrent threads. Changed selection must produce a
new inventory. Failed reads are retried rather than cached indefinitely.
Replacing code, libraries, or tools at the same installation paths requires a
process restart. Application source refresh never uses this memoization.

## Environment classification

`python/pypto/_environment.json` records environment inputs and their rationale:

| Category | Required treatment at integration |
| -------- | --------------------------------- |
| `semantic` | Resolve effective precedence and include the value in compilation inputs. |
| `tool_resolution` | Identify the selected tools and dependency contents, not just the search-path strings. |
| `fresh_request` | Honor requested compilation/check/output behavior before cache lookup. |
| `nonsemantic` | Exclude only with the recorded justification, such as terminal formatting or logging. |

The registry is an audit inventory, not a blanket hash of the environment or
an implementation of those policies. Existing diagnostic bypass behavior is
described in [JIT functions](language/03-functions.md#compile-options-and-diagnostic-requests).

`tests/lint/check_environment_inputs.py` runs in pre-commit without loading the
native extension. It checks Python reads and C++ `getenv`/`secure_getenv` calls
under `python/pypto`, `python/bindings`, `src`, and `include`. Recognized Python
forms include imports, aliases, module string constants, mapping reads, and
bulk reads. Aliases are scoped to their lexical context; ambiguous module
constant assignments, including control-flow writes, remain dynamic.
The hook uses Python 3.10. Dynamic reads need an exact file/function exception and a reason;
that exception cannot hide a new literal variable. Unused exceptions fail.

This static check does not certify dependencies inside downstream tools or
arbitrary reflective Python code. The registry also records non-`PYPTO_*`
inputs such as `PATH`, compiler include/library search variables, loader
injection, and locale; the toolchain adapters must account for these before
claiming complete identity. Unsupported dependency discovery must remain
unavailable when persistent lookup is eventually connected.
