# Immutable Artifact Store

The internal `pypto.jit.artifact_cache` module implements the storage milestone
of [RFC #2653](https://github.com/hw-native-sys/pypto/issues/2653). It provides
validated manifests, per-key locking, immutable publication, and private-build
fallback. It is not connected to JIT dispatch. The complete automatic toolchain
inventories described in [Artifact Identity Foundations](08-artifact-identity.md),
runtime loader integration, public cache configuration, and warmup remain
separate milestones.

## Adapter contract

`ArtifactKey` requires a usable `ToolchainIdentity` with the current identity
schema, plus full SHA-256 source and specialization digests. Its record retains
every environment component digest and both request digests; its full digest
also includes the artifact schema. Missing identities and truncated or malformed
digests raise `ValueError`. This validates the representation, not the
completeness of the adapter's source/toolchain inventories.

`ArtifactSpec` identifies `GENERATED` or `BINARY_READY`, the single-chip or
distributed build kind, and a nonempty, unique list of required relative files.
The adapter must include its effective build kind in specialization identity.
It must enumerate all required source/configuration files for generated output,
and all binaries plus complete loader metadata for binary-ready output. Merely
labeling a directory `BINARY_READY` does not establish runtime readiness.

`ArtifactStore.get_or_build(key, spec, builder)` calls `builder(private_directory)`
only when a validated hit is unavailable. The builder returns an adapter-owned
value and must finish writing files before returning. Compiler exceptions,
including `OSError`, and missing/invalid build outputs propagate. Builders must
not recursively acquire the same key's lock.

`ArtifactBuild` reports `HIT`, `PUBLISHED`, or `PRIVATE`. Fresh builds retain both
the builder's value and `private_directory`, including after successful
publication: the value may still reference private files. The adapter owns
rebinding and eventual cleanup. Hits return an `ArtifactHandle` with no builder
value or private directory. Lookup reports `HIT`, `MISS`, `INVALID`, or
`STORAGE_ERROR`, with diagnostic reasons for invalid or unavailable storage.

## Layout and validation

```text
<root>/
  locks/<key>.lock
  artifacts/<environment-digest>/<key>/
    generated/artifact_manifest.json
    ready/artifact_manifest.json
    .tmp.<random>/
```

Each published stage contains its payload beside `artifact_manifest.json`.
The completion marker contains the schema, full key and components, state,
build kind, required-file list, and a sorted inventory of every payload file's
relative path, byte size, SHA-256 digest, and permission bits. It is bounded to
16 MiB. Readers verify the entire inventory against the exact request; no
timestamps substitute for content hashes. Unexpected files, duplicate JSON
fields, altered metadata, missing files, and malformed markers invalidate the
entry. Empty directories carry no artifact semantics.

Manifest paths are never used to open files: validation enumerates the actual
tree and compares its canonical record to the marker. Absolute, non-normalized,
parent-traversing, and backslash paths are rejected. Payload links, special
files, and symlinked cache descendants are rejected. The explicitly configured
root is resolved once to its canonical path. Payload permission bits are copied;
setuid/setgid/sticky bits are not propagated to published files.

The root must have trusted writers: digests detect corruption, not malicious
replacement of executable code and its matching manifest. Writers must not
modify published entries or race readers with deletion. This protocol is not
an atomic filesystem snapshot or a defense against a hostile cache owner.

## Publication and recovery

1. Lookup reads and validates the requested stage without writing anything.
2. On a writable miss, acquire `flock` on the persistent key lock and recheck.
   Both stages share that lock. Independent keys can build concurrently.
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
build privately until offline cleanup removes the invalid slot. If publication
succeeds but the final parent sync fails, the private result is retained and
the valid published slot is left intact. Unsupported no-replace rename or
unavailable writable storage also yields private output. Writers require a
Linux filesystem honoring `flock` and atomic no-replace rename; network
filesystems must establish those semantics before use.

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
into an empty private directory. It excludes the old marker and uses no
hardlinks. Binary compilation can modify the private files freely; publication
creates a separate `ready/` slot and leaves `generated/` intact. The runtime
adapter is responsible for path rebinding and complete binary/metadata coverage.

There is no online garbage collection, diagnostic index, global cache statistics,
or automatic stage preference in this layer. Cleanup is offline with all
consumers stopped. Later integration must select ready before generated output
and keep live object paths valid throughout their lifetime.
