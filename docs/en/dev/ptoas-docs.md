# Documentation updates


The site checks out the mirror's `master` branch at build time. The existing Docs
workflow publishes on `main` pushes and manual runs; a daily run also picks up
mirrored PTOAS changes. A failed build keeps the last successful deployment live.

For a local build, prepare the source once, then use the normal MkDocs commands:

```bash
git clone --depth 1 https://github.com/hw-native-sys/PTOAS.git .cache/ptoas-docs
pip install -r docs/requirements.txt
mkdocs build --strict
```

To update an existing source checkout, run
`git -C .cache/ptoas-docs pull --ff-only` before rebuilding.

See also the [PTOAS op status matrix](../../dev/ptoas-op-status.md) for the
operations currently emitted by PyPTO.
