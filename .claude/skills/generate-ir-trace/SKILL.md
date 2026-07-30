---
name: generate-ir-trace
description: Use when generating, inspecting, or sharing an interactive IR lowering trace from an existing PyPTO passes_dump directory or by running a PyPTO or pypto-lib case first.
---

# Generate IR Trace

Generate a report only after proving its worktree provenance, input identity, freshness, and standalone integrity. Keep the selected dump and delivered HTML intact.

## Choose the flow

- Use the **quick flow** for one exact `passes_dump/` directory.
- Use the **full flow** for a case, script, or command; it must produce a dump inside a fresh output root before conversion.
- Ask only if multiple materially different cases match. Preserve all user arguments.
- If a requested path is missing, stop. Never search for or substitute a historical dump.

Open one persistent Bash session and run every Bash block below, in order, in that same session. Do not dispatch fenced blocks as independent shell calls: the shared functions and flow variables are session state.

```bash
set -euo pipefail
WORKTREE="$(git rev-parse --show-toplevel)"
WORKTREE_PYTHONPATH="$WORKTREE/python${PYTHONPATH:+:$PYTHONPATH}"

assert_worktree_cli() {
  EXPECTED_CLI="$WORKTREE/python/pypto/tools/ir_trace/cli.py"
  CLI_PATH="$(PYTHONPATH="$WORKTREE_PYTHONPATH" python -c \
    'from pathlib import Path; import pypto.tools.ir_trace.cli as m; print(Path(m.__file__).resolve())')"
  test "$CLI_PATH" = "$EXPECTED_CLI" || {
    printf 'wrong IR trace implementation: %s (expected %s)\n' "$CLI_PATH" "$EXPECTED_CLI" >&2
    return 1
  }
}

wt_python() {
  assert_worktree_cli
  PYTHONPATH="$WORKTREE_PYTHONPATH" python "$@"
}
```

If the assertion fails because the worktree build is missing or stale, build it in place and retry. Never drop the assertion or fall back to an installed package:

```bash
test -f "$WORKTREE/build/CMakeCache.txt" || cmake -S "$WORKTREE" -B "$WORKTREE/build" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$WORKTREE/build" --parallel
assert_worktree_cli
```

Define the shared validation and conversion block once in the same session. `convert_dump` accepts an already-resolved dump plus output/context policy, initializes global `DUMP`, `REPORT`, and `CONTEXT`, and returns only after validation:

```bash
validate_report() {
  wt_python - "$1" <<'PY'
import json
import sys
from html.parser import HTMLParser
from pathlib import Path

class ResourceAudit(HTMLParser):
    def __init__(self):
        super().__init__()
        self.external = []
    def handle_starttag(self, tag, attrs):
        normalized = [(name.lower(), value or "") for name, value in attrs]
        if any(name == "src" and value for name, value in normalized):
            self.external.append(f"{tag}[src]")
        rels = [value for name, value in normalized if name == "rel"]
        has_href = any(name == "href" and value for name, value in normalized)
        if tag.lower() == "link" and has_href and any("stylesheet" in rel.lower().split() for rel in rels):
            self.external.append("link[rel=stylesheet]")

path = Path(sys.argv[1])
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit(f"missing or empty report: {path}")
text = path.read_text(encoding="utf-8")
marker = '<script id="trace-data" type="application/json">'
checks = (text.startswith("<!doctype html>"), text.rstrip().endswith("</html>"),
          "<style>" in text, marker in text, text.count("<script") >= 2)
if not all(checks):
    raise SystemExit("report is missing doctype, closing HTML, CSS, trace data, or JavaScript")
if not json.loads(text.split(marker, 1)[1].split("</script>", 1)[0]).get("passes"):
    raise SystemExit("report contains no pass trace data")
audit = ResourceAudit()
audit.feed(text)
if audit.external:
    raise SystemExit(f"report references external resources: {audit.external}")
print(f"validated self-contained report: {path.resolve()} ({path.stat().st_size} bytes)")
PY
}

convert_dump() {
  test "$#" -eq 4 || { printf 'convert_dump requires DUMP REPORT CONTEXT ALLOW_OVERWRITE\n' >&2; return 2; }
  DUMP="$1"
  REPORT_INPUT="$2"
  CONTEXT="$3"
  ALLOW_OVERWRITE="$4"
  test -d "$DUMP" || { printf 'missing requested passes_dump: %s\n' "$DUMP" >&2; return 1; }
  test "$DUMP" = "$(realpath -- "$DUMP")" || { printf 'DUMP must be resolved: %s\n' "$DUMP" >&2; return 1; }
  REPORT_DIR_INPUT="$(dirname -- "$REPORT_INPUT")"
  test -d "$REPORT_DIR_INPUT" || { printf 'missing report directory: %s\n' "$REPORT_DIR_INPUT" >&2; return 1; }
  REPORT="$(realpath -- "$REPORT_DIR_INPUT")/$(basename -- "$REPORT_INPUT")"
  case "$REPORT" in "$DUMP"/*) printf 'report must be outside passes_dump: %s\n' "$REPORT" >&2; return 1;; esac
  if test -e "$REPORT" && test "$ALLOW_OVERWRITE" -ne 1; then
    printf 'report already exists; choose a new path or approve overwrite: %s\n' "$REPORT" >&2
    return 1
  fi

  wt_python - "$DUMP" <<'PY'
import sys
from pathlib import Path
from pypto.tools.ir_trace.discovery import discover_snapshots
from pypto.tools.ir_trace.model import IRTraceError
dump = Path(sys.argv[1])
try:
    snapshots = discover_snapshots(dump)
except IRTraceError as error:
    raise SystemExit(f"invalid requested passes_dump: {error}") from error
print(f"validated {len(snapshots)} snapshots from {dump.resolve()}")
for snapshot in snapshots:
    print(snapshot.path.name)
PY
  wt_python -c 'from pypto.tools.ir_trace.cli import main; raise SystemExit(main())' \
    "$DUMP" --context "$CONTEXT" --output "$REPORT"
  validate_report "$REPORT"
  REPORT="$(realpath -- "$REPORT")"
}
```

## Quick flow: existing dump

Resolve only the exact requested dump, then call the shared block with explicit output/context values:

```bash
DUMP_INPUT='<requested-passes_dump>'
REPORT_INPUT='<requested-or-new-report.html>'
CONTEXT_INPUT=3
ALLOW_OVERWRITE=0  # Set to 1 only after explicit user approval.
test -d "$DUMP_INPUT" || { printf 'missing requested passes_dump: %s\n' "$DUMP_INPUT" >&2; exit 1; }
DUMP="$(realpath -- "$DUMP_INPUT")"
convert_dump "$DUMP" "$REPORT_INPUT" "$CONTEXT_INPUT" "$ALLOW_OVERWRITE"
```

## Full flow: run a case first

1. Locate the requested case and its documented invocation; do not hard-code a pypto-lib checkout or invent flags. Use `wt_python "$CASE_SCRIPT" --help` for a Python entry point so its import is asserted.
2. Confirm the invocation supports pass dumping and can route all outputs below an explicit root. If not (for example, a real-device path disables dumps), stop with that concrete blocker.
3. Create a fresh, empty provenance boundary:

   ```bash
   RUN_PARENT="$WORKTREE/build/ir-trace-runs"
   mkdir -p "$RUN_PARENT"
   RUN_ROOT="$(mktemp -d "$RUN_PARENT/run.XXXXXX")"
   RUN_MANIFEST="$(mktemp "$RUN_PARENT/manifest.XXXXXX")"
   find "$RUN_ROOT" -mindepth 1 -print0 > "$RUN_MANIFEST" || { printf 'failed to enumerate fresh run root: %s\n' "$RUN_ROOT" >&2; exit 1; }
   test ! -s "$RUN_MANIFEST" || { printf 'fresh run root is not empty: %s\n' "$RUN_ROOT" >&2; exit 1; }
   ```

4. Run the exact command in this session with all user arguments, pass dumping, and documented output setting pointed at `RUN_ROOT`. For a Python script use `wt_python "$CASE_SCRIPT" <arguments>`; for another launcher run `assert_worktree_cli` immediately before it and prefix `PYTHONPATH="$WORKTREE_PYTHONPATH"`. Require status `0` and record the expanded command.
5. Accept exactly one dump beneath the previously empty root, initialize report/context policy, and call the same conversion block:

   ```bash
   find "$RUN_ROOT" -type d -name passes_dump -print0 > "$RUN_MANIFEST" || { printf 'failed to enumerate fresh dumps: %s\n' "$RUN_ROOT" >&2; exit 1; }
   mapfile -d '' FRESH_DUMPS < "$RUN_MANIFEST"
   test "${#FRESH_DUMPS[@]}" -eq 1 || {
     printf 'expected exactly one fresh passes_dump under %s, found %s\n' "$RUN_ROOT" "${#FRESH_DUMPS[@]}" >&2
     printf '%s\n' "${FRESH_DUMPS[@]}" >&2
     exit 1
   }
   DUMP="$(realpath -- "${FRESH_DUMPS[0]}")"
   REPORT_INPUT='<requested-or-new-report.html>'
   CONTEXT_INPUT=3
   ALLOW_OVERWRITE=0  # Set to 1 only after explicit user approval.
   convert_dump "$DUMP" "$REPORT_INPUT" "$CONTEXT_INPUT" "$ALLOW_OVERWRITE"
   ```

Never select by latest modification time. Keep `RUN_ROOT` for provenance.

## Validate and hand off

`convert_dump` has already validated and resolved `REPORT`. If copying it, verify byte identity and validate the copy through the same worktree-asserting function:

```bash
COPY_INPUT='<requested-copy.html>'
COPY_ALLOW_OVERWRITE=0  # Set to 1 only after explicit user approval.
COPY_PARENT="$(realpath -- "$(dirname -- "$COPY_INPUT")")"
COPY="$COPY_PARENT/$(basename -- "$COPY_INPUT")"
if test -e "$COPY" && test "$COPY_ALLOW_OVERWRITE" -ne 1; then
  printf 'report copy already exists; choose a new path or approve overwrite: %s\n' "$COPY" >&2; exit 1
fi
cp -- "$REPORT" "$COPY"
cmp -s -- "$REPORT" "$COPY" || { printf 'report copy mismatch: %s\n' "$COPY" >&2; exit 1; }
validate_report "$COPY"
REPORT="$(realpath -- "$COPY")"
```

Hand off the complete file as a clickable absolute path. Report `WORKTREE`, `CLI_PATH`, full-flow command and `RUN_ROOT` when applicable, `DUMP` and snapshots, `CONTEXT`, validation result, and final `REPORT`.

## Common mistakes

- Running blocks in separate shells or invoking Python outside `wt_python`/the asserted worktree environment.
- Replacing a missing dump, choosing the newest dump, accepting zero/multiple dumps, reusing an HTML file, or skipping complete absolute-path validation.
