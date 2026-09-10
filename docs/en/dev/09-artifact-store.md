# Immutable Artifact Store

The internal `pypto.jit.artifact_cache` module implements the storage milestone
of [RFC #2653](https://github.com/hw-native-sys/pypto/issues/2653). It provides
validated manifests, per-key locking, immutable publication, and private-build
fallback. The [opt-in JIT integration](10-jit-cache.md) connects it to ordinary
compilation, execution and warmup. The runtime adapter below owns device-stage
promotion and read-only loading. [Artifact identities](08-artifact-identity.md)
cover dependency contents rather than compiler version strings.

## Adapter contract

`ArtifactKey` requires a usable `ToolchainIdentity` with the current identity
schema, plus full SHA-256 source and specialization digests. Its record retains
every environment component digest and both request digests; its full digest
also includes the artifact schema. Missing identities and truncated or malformed
digests raise `ValueError`. This validates the representation, not the
completeness of the adapter's source/toolchain inventories.

`ArtifactSpec` identifies `GENERATED` or `BINARY_READY`, the single-chip or
distributed build kind, and a nonempty, unique list of required relative files.
The store includes a digest of the state, build kind, and sorted required-file
list in the slot path. A changed spec therefore selects a new slot and misses
cleanly, independently of the adapter's specialization digest. Existing specs
remain reusable. The adapter must enumerate all required source/configuration files for generated output,
and all binaries plus complete loader metadata for binary-ready output. Merely
labeling a directory `BINARY_READY` does not establish runtime readiness.

`ArtifactStore.get_or_build(key, spec, builder)` calls `builder(private_directory)`
only when a validated hit is unavailable. The builder returns an adapter-owned
value and must finish writing files before returning. Compiler exceptions,
including `OSError`, and missing/invalid build outputs propagate. Builders must
not recursively acquire the same key's lock.

`ArtifactBuild` reports `HIT`, `PUBLISHED`, or `PRIVATE`. Its optional `failure` is a
`BuildFailure` (`INVALID`, `STORAGE`, `LOCK`, or `PUBLICATION`), independent
of diagnostic wording. A read-only miss has no failure.
 Fresh builds retain both
the builder's value and `private_directory`, including after successful
publication: the value may still reference private files. The adapter owns
rebinding and eventual cleanup. Hits return an `ArtifactHandle` with no builder
value or private directory. Lookup reports `HIT`, `MISS`, `INVALID`, or
`STORAGE_ERROR`, with diagnostic reasons for invalid or unavailable storage.

Overlapping calls in one process with the same root, private root, read-only
policy, key, and spec share one in-flight operation, even across store instances.
This includes `INVALID`, `STORAGE_ERROR`, read-only misses, and lock/publication
failures. A private result shares its builder value and directory among waiters,
so matching builders must be interchangeable and their values safe to share.
Compiler errors, including cancellation, wake all waiters and propagate. Results
and errors are removed from the coordinator when the operation finishes;
subsequent private requests build again. This is not a private-object cache.

## Layout and validation

```text
<root>/
  locks/<key>.lock
  artifacts/<environment-digest>/<key>/<spec-digest>/
    <state>/artifact_manifest.json
    .tmp.<random>/
```

The state is `generated` or `ready`; each has its own spec digest. Each
published stage contains its payload beside `artifact_manifest.json`.
The completion marker contains the schema, full key and components, state,
build kind, required-file list, and a sorted inventory of every payload file's
relative path, byte size, SHA-256 digest, and owner-executable flag. It is bounded to
16 MiB. Readers verify the entire inventory against the exact request; no
timestamps substitute for content hashes. Unexpected files, duplicate JSON
fields, altered metadata, missing files, and malformed markers invalidate the
entry. Empty directories carry no artifact semantics.

Manifest paths are never used to open files: validation enumerates the actual
tree and compares its canonical record to the marker. Absolute, non-normalized,
parent-traversing, and backslash paths are rejected. Payload links, special
files, and symlinked cache descendants are rejected. The explicitly configured
root is resolved once to its canonical path. Only the owner's execute bit is
part of the artifact contract. Read/write bits and group/other execute bits are
access policy: changing them does not invalidate unchanged payloads. Payload
copies are owner-readable and owner-writable and preserve the owner execute bit;
group/other and setuid/setgid/sticky bits are not propagated to published files.

The root must have trusted writers: digests detect corruption, not malicious
replacement of executable code and its matching manifest. Writers must not
modify published entries or race readers with deletion. This protocol is not
an atomic filesystem snapshot or a defense against a hostile cache owner.

## Publication and recovery

1. Lookup reads and validates the requested stage without writing anything.
2. For every non-hit in writable mode, acquire `flock` on the persistent key
   lock when available and recheck. Both stages and all specs share that lock.
   Independent keys can build concurrently.
3. Build outside the cache root. Validate all required private output files.
4. Copy payload to a unique staging directory beside the final slot, using
   separate files rather than hardlinks. Revalidate the copy, sync payload files
   and directory entries, write the completion marker last, and sync it.
5. Publish with Linux `renameat2(RENAME_NOREPLACE)` and sync the parent directory.
   Even an existing empty destination is never replaced.

Staging, lock, or publication failures return the usable private build with a
reason. Failed staging is removed on a best-effort basis. A process crash may
leave private directories or `.tmp.*` directories; neither is a cache hit.
The kernel releases a dead process's lock. Lock files are never unlinked, so
waiting processes continue synchronizing on the same inode.

Invalid final slots are never repaired or overwritten online. Such requests
build privately until offline cleanup removes the invalid slot. Across processes,
file locking only eliminates duplicate compilation when the first process can
publish a reusable artifact. It serializes private builds when available, but
does not share their results. If storage or locking remains unavailable, distinct
processes still build independently. Cross-process private-result reuse would
require an additional shared fallback publication protocol, which is not
implemented here. If publication
succeeds but the final parent sync fails, the private result is retained and
the valid published slot is left intact. Unsupported no-replace rename or
unavailable writable storage also yields private output. Writers require a
Linux filesystem honoring `flock` and atomic no-replace rename; network
filesystems must establish those semantics before use.

## Sharing permissions

By default, reuse is limited to the publishing UID. A published stage inherits
the staging directory's `0700`; payload files use `0600`, or `0700` when owner
execution is required. Intermediate directories and completion markers follow
the process umask, but cannot make the enclosing `0700` stage traversable by
another UID. There is no mode option or umask-based override for stage/payload
permissions. Concurrent writers with different UIDs are not supported.

To expose prewarmed artifacts to other UIDs as read-only consumers, stop all
writers and consumers, set `CACHE_ROOT` to the intended cache root, and grant
read access explicitly. The following policy grants all local users read access;
use an administrator-managed group/ACL policy if that is too broad:

```bash
chmod a+rx,go-w -- "$CACHE_ROOT"
chmod -R a+rX,a-w -- "$CACHE_ROOT/artifacts"
```

Every ancestor of the root must also be searchable by the intended readers.
Consumers must use `readonly=True`; this does not grant them access to writer
locks or permission to publish. The commands preserve owner execute bits.
Adding or removing group/other execute bits, as with `a+rX` or `go-x`, also
preserves artifact identity; removing the owner execute bit does not.

## Read-only use and stage promotion

`ArtifactStore(..., readonly=True)` performs no cache-root writes, including
locks, indexes, or staging. A hit only reads its manifest and payload. A miss or
invalid entry builds in an explicitly supplied `private_root` outside the cache
root. Without one, hits still work but requests needing a build raise `OSError`.
The store does not probe temporary-directory candidates, which could themselves
be inside the cache root. If the selected private location is not writable, the
filesystem error propagates. The store never imports or executes
cached Python files; future loaders must independently avoid bytecode writes.

To promote generated output, a binary builder uses
`generated_handle.materialize(private_directory)` to copy the validated payload
into an empty directory outside the entire shared cache root, including when
the destination is reached through a symlink. It excludes the old marker and
uses no hardlinks. Copies are owner-writable even when the cached payload is
read-only. Binary compilation can modify the private files freely; publication
creates a separate `ready/` slot under its own spec digest and leaves
`generated/` intact. The runtime
adapter is responsible for path rebinding and complete binary/metadata coverage.

There is no online garbage collection, diagnostic index, global cache statistics,
or automatic stage preference in this layer. Cleanup is offline with all
consumers stopped. Later integration must select ready before generated output
and keep live object paths valid throughout their lifetime.

## Explicit runtime adapter

The internal `pypto.runtime._artifact_runtime` module bridges validated handles
and compiled programs. It is the runtime milestone of RFC #2653, not a public
cache configuration API. Callers must supply a complete `ArtifactKey`, a matching
input snapshot, and an `ArtifactSpec` listing all required generated files.
Do not substitute placeholder identity digests in production.

Before publishing `GENERATED`, the builder calls
`package_generated_sources(private_directory, build_kind)` from
`pypto.runtime._artifact_sources`. This normalizes generated configuration paths
and packages supported extern dependencies into the private tree. The store then
validates and publishes that self-contained tree.

```python
from pypto.runtime._artifact_runtime import bind_artifact, restore_artifact

# generated_handle is a validated ArtifactHandle for this compiled input snapshot.
# run_directory is a Path outside store.root, owned by this runtime session.
bind_artifact(compiled, store, generated_handle, run_directory)
compiled.load()  # Single-chip: promotes if needed, then assembles live callables.
ready_handle = compiled._artifact_runtime.handle

# A different process may look up the ready key/spec in a read-only ArtifactStore.
restored = restore_artifact(readonly_store, ready_handle, another_run_directory)
restored.load()  # Validates metadata and bytes; does not compile or execute.
```

For a distributed program, binding/restoration works the same way; its existing
runner or worker lazily requests every child callable. `compiled.load()` above
is the single-chip interface. The adapter currently supports one single-chip
build or a distributed parent with chip children. It rejects single-chip
multi-orchestration parents, whose current `from_dir` protocol cannot reconstruct
the parent without live IR. Persist supported individual children instead.

### Stage transition and loading

1. Validate the handle against its exact key and spec before executing any
   generated configuration. Compute the ready spec, listing the parent marker,
   every child marker, orchestration binary, and kernel binary. The generated
   spec must declare every required chip configuration: missing declared files
   fail store validation. Auxiliary directories without `kernel_config.py` are
   skipped, matching ordinary distributed replay.
2. Look up `BINARY_READY` through `ArtifactStore.get_or_build`. On a miss,
   materialize the generated handle into a private directory. The lock order is
   artifact key lock, then private runtime build lock.
3. Compile all chip builds with the existing runtime compiler, bypassing inherited
   mutable binary caches and source-adjacent binaries even when their legacy
   context stamp matches. A generated identity does not certify those bytes.
   Record the exact
   final kernel bytes handed to `CoreCallable.build` and orchestration bytes
   handed to `ChipCallable.build`. Compilation and assembly do not execute on a
   device. After releasing each private compiler lock, remove its `cache/`
   directory (including context stamps and locks) and generated source-adjacent
   `.o`/`.so` outputs. Keep sources, configuration, extern inputs, binary manifests,
   and one copy of each final binary under `prebuilt/`. Inherited cache/sidecar
   files are excluded from the ready spec. Only after all children succeed may
   the store publish ready output.
4. Load the versioned `binary_manifest.json` and validate every child before
   constructing any callable. Records include relative binary paths, sizes,
   SHA-256 digests, platform, runtime configuration, function IDs, signatures,
   and diagnostic names. Distributed parents list their complete chip set.
5. Reconstruct callables from bytes. Ready loading does not resolve PTO-ISA,
   construct a compiler, acquire a writable cache lock, rewrite headers, execute
   `kernel_config.py`, or write binary-context stamps or bytecode. It still
   requires the compatible runtime libraries identified by the supplied key.

The enclosing store manifest remains authoritative for the full identity and
payload inventory. A binary marker alone is insufficient to attach a handle.
Attachment/restoration hashes the complete payload once and reconstructs metadata
once. Loading reuses that verified inventory to check the inner binary sizes and
digests, without hashing the same bytes again. A directly constructed internal
`ArtifactRuntime` validates on its first load. Promotion validates the generated
copy and the new ready payload at their own boundaries; private fallback loading
checks binary digests directly. Store lookup validation is separate from attachment.
This validation is scoped to the attached object's lifetime, not a process-wide
cache: published files must remain unchanged and present until all users release
them. A new attachment validates again. Missing or corrupt handles fail at those
boundaries before execution. The adapter does not catch a
device execution error and retry the operation.

Storage/publication failures retain a usable private directory and live callables
for the adapter's lifetime. `handle` remains the generated handle in this case;
`directory` identifies the private binary output. Overlapping promotions share
the store's in-process operation, including private fallback. Across processes,
private results remain independent as described above. Private directories are
not automatically deleted; their owner must release all users before cleanup.

### Paths, diagnostics, and extern inputs

Attachment keeps a fresh compiled object's live IR. Restoring from persisted
metadata has `program is None`. Attachment rebinds its source path once, before
worker registration; later promotion changes only the runtime binary handle.
The compiled object's path and hash stay stable for worker registries. Ordinary
unattached `from_dir` replay retains its existing mutable compilation behavior.

DFX output, dependency capture, and swimlane conversion use the separate
`run_directory`. Each concurrent runtime session must choose its own directory.
Labels come from binary records; loading generated host orchestration avoids
Python bytecode caching. Published artifacts must remain alive and unchanged
while compiled objects or workers reference them.

Extern packaging supports recursively resolved literal local includes, retaining
relative include topology and ordered explicit include directories. The common
root of the source directory and explicit include directories is resolved once;
symlinked workspace/home ancestors are supported. Symbolic links below that root,
including links encountered before `..` traversal, require private compilation.
The scanner removes comments, joins escaped newlines, and ignores branches proven
inactive by literal `#if 0`/`#if 1` and their `#elif`/`#else` structure. Unknown
conditions conservatively scan all possible branches; this is not a full C
preprocessor. Active macro includes, absolute includes, and unresolved quoted
includes require private compilation. Non-UTF8 source bytes are copied unchanged;
replacement decoding is used only for include scanning.
Unresolved angle includes are supplied by the separately identified SDK/toolchain. Empty
include directories may disappear during publication; their missing `-I` paths
remain valid, and `extra_include_dirs=None` is normalized to an empty list. Other
file-bearing preprocessor or assembler constructs are not supported. The caller
must establish complete input identity before publication; this packager does
not discover a toolchain inventory or make an arbitrary C++ build hermetic.
`UnsupportedArtifactInput`, available from `pypto.runtime._artifact_sources`, is
a dedicated `ValueError` subclass for these packaging limitations. Adapters may
catch only this exception to bypass persistent publication and compile the
original input privately. Packaging mutates a private staging tree: discard that
tree on fallback. Filesystem failures and malformed input/configuration errors
remain distinct and propagate; do not treat every `ValueError` as a cache bypass.
This explicit adapter does not automatically invoke the ordinary compiler on a
packaging rejection. Once ready,
the supported artifact can relocate and load without its original extern tree.
