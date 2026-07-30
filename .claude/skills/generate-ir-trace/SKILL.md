---
name: generate-ir-trace
description: Use when generating, inspecting, or sharing an interactive IR lowering trace from an existing PyPTO passes_dump directory or by running a PyPTO or pypto-lib case first.
---

# Generate IR Trace

Generate a report only after proving its worktree provenance, input identity, freshness, and standalone integrity. Keep the selected dump and delivered HTML intact.

## Choose the flow

- Use the **quick flow** when the user supplies one exact `passes_dump/` directory.
- Use the **full flow** when the user supplies a case, script, or command; it must produce a dump inside a fresh output root before the quick flow begins.
- Ask only if multiple materially different cases match. Preserve all user arguments.
- If a requested path is missing, stop. Never search for or substitute a historical dump.

Run commands with Bash from the current PyPTO worktree. Establish and verify the current source first:

```bash
set -euo pipefail
WORKTREE="$(git rev-parse --show-toplevel)"
export PYTHONPATH="$WORKTREE/python${PYTHONPATH:+:$PYTHONPATH}"
CLI_PATH="$(python -c 'from pathlib import Path; import pypto.tools.ir_trace.cli as m; print(Path(m.__file__).resolve())')"
EXPECTED_CLI="$WORKTREE/python/pypto/tools/ir_trace/cli.py"
test "$CLI_PATH" = "$EXPECTED_CLI" || {
  printf 'wrong IR trace implementation: %s (expected %s)\n' "$CLI_PATH" "$EXPECTED_CLI" >&2
  exit 1
}
```

If that import fails because the worktree build is missing or stale, build it in place and retry. Never remove `PYTHONPATH` or fall back to an installed `pypto-ir-trace`:

```bash
test -f "$WORKTREE/build/CMakeCache.txt" || cmake -S "$WORKTREE" -B "$WORKTREE/build" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$WORKTREE/build" --parallel
```

## Quick flow: existing dump

Set `DUMP_INPUT` to the exact requested directory, `REPORT_INPUT` to an explicit output outside it, and `CONTEXT` to the requested value (default `3`). The report parent must exist. Refuse an existing report unless the user explicitly approved overwriting it.

```bash
DUMP_INPUT='<requested-passes_dump>'
REPORT_INPUT='<requested-or-new-report.html>'
CONTEXT=3
ALLOW_OVERWRITE=0  # Set to 1 only after explicit user approval.

test -d "$DUMP_INPUT" || { printf 'missing requested passes_dump: %s\n' "$DUMP_INPUT" >&2; exit 1; }
DUMP="$(realpath -- "$DUMP_INPUT")"
REPORT_DIR_INPUT="$(dirname -- "$REPORT_INPUT")"
test -d "$REPORT_DIR_INPUT" || { printf 'missing report directory: %s\n' "$REPORT_DIR_INPUT" >&2; exit 1; }
REPORT_PARENT="$(realpath -- "$REPORT_DIR_INPUT")"
REPORT="$REPORT_PARENT/$(basename -- "$REPORT_INPUT")"
case "$REPORT" in "$DUMP"/*) printf 'report must be outside passes_dump: %s\n' "$REPORT" >&2; exit 1;; esac
if test -e "$REPORT" && test "$ALLOW_OVERWRITE" -ne 1; then
  printf 'report already exists; choose a new path or approve overwrite: %s\n' "$REPORT" >&2
  exit 1
fi
```

Validate the exact dump and record its ordered snapshots before conversion:

```bash
python - "$DUMP" <<'PY'
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
```

Invoke the imported worktree module directly. This calls `main`; `python -m pypto.tools.ir_trace.cli` alone does not.

```bash
python -c 'from pypto.tools.ir_trace.cli import main; raise SystemExit(main())' \
  "$DUMP" --context "$CONTEXT" --output "$REPORT"
```

Any nonzero status is a blocker. Do not hand off a pre-existing or partial file after failure.

## Full flow: run a case first

1. Locate the requested case and its documented invocation with repository search and `--help`; do not hard-code a pypto-lib checkout or invent flags:

   ```bash
   rg -n -- '<case-name>|dump_passes|output_dir' "$CASE_ROOT"
   python "$CASE_SCRIPT" --help
   ```

2. Confirm the invocation supports pass dumping and can route all outputs below an explicit root. Configure/build the current worktree if the case requires it. If either property is unavailable (for example, a real-device path disables dumps), stop with that concrete blocker.
3. Create a fresh provenance boundary and prove it is empty:

   ```bash
   RUN_PARENT="$WORKTREE/build/ir-trace-runs"
   mkdir -p "$RUN_PARENT"
   RUN_ROOT="$(mktemp -d "$RUN_PARENT/run.XXXXXX")"
   test -z "$(find "$RUN_ROOT" -mindepth 1 -print -quit)"
   printf 'fresh run root: %s\n' "$RUN_ROOT"
   ```

4. Run the exact discovered command directly (never through `eval`) with all user arguments, current-worktree `PYTHONPATH`, pass dumping enabled, and its documented output setting pointed at `RUN_ROOT`. Record the expanded command and require status `0`.
5. Accept exactly one dump created beneath the previously empty root:

   ```bash
   mapfile -d '' FRESH_DUMPS < <(find "$RUN_ROOT" -type d -name passes_dump -print0)
   if test "${#FRESH_DUMPS[@]}" -ne 1; then
     printf 'expected exactly one fresh passes_dump under %s, found %s\n' \
       "$RUN_ROOT" "${#FRESH_DUMPS[@]}" >&2
     printf '%s\n' "${FRESH_DUMPS[@]}" >&2
     exit 1
   fi
   DUMP="$(realpath -- "${FRESH_DUMPS[0]}")"
   ```

6. Continue with the quick flow's exact-dump validation and report generation. Never select by latest modification time. Keep `RUN_ROOT` for provenance.

## Validate and hand off

Validate non-empty UTF-8 HTML, required inline CSS/JavaScript/data, parsed trace data, a closing document, and absence of external resource tags:

```bash
python - "$REPORT" <<'PY'
import json
import sys
from html.parser import HTMLParser
from pathlib import Path

class Audit(HTMLParser):
    external = []
    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("src") or (tag == "link" and "stylesheet" in values.get("rel", "").lower()):
            self.external.append(tag)

path = Path(sys.argv[1])
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit(f"missing or empty report: {path}")
text = path.read_text(encoding="utf-8")
marker = '<script id="trace-data" type="application/json">'
required = [text.startswith("<!doctype html>"), text.rstrip().endswith("</html>"), "<style>" in text,
            marker in text, text.count("<script") >= 2]
if not all(required):
    raise SystemExit("report is missing doctype, closing HTML, CSS, trace data, or JavaScript")
payload = json.loads(text.split(marker, 1)[1].split("</script>", 1)[0])
if not payload.get("passes"):
    raise SystemExit("report contains no pass trace data")
audit = Audit(); audit.feed(text)
if audit.external:
    raise SystemExit(f"report references external resources: {audit.external}")
print(f"validated self-contained report: {path.resolve()} ({path.stat().st_size} bytes)")
PY
REPORT="$(realpath -- "$REPORT")"
```

If copying to another requested location, verify byte identity and repeat the validation command on the copy:

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
REPORT="$(realpath -- "$COPY")"
```

Hand off the complete file as a clickable absolute path, not pasted HTML. Report the worktree and `CLI_PATH`, exact case command and `RUN_ROOT` for the full flow, resolved dump and snapshot list, converter command/context, validation result, and final `REPORT`.

## Common mistakes

- Using an installed console script or dropping worktree `PYTHONPATH` after an import failure.
- Replacing a missing requested dump, choosing the newest dump, or accepting zero/multiple full-flow dumps.
- Reusing or overwriting an HTML file without an explicit successful generation boundary.
- Checking only file existence instead of HTML markers, embedded JSON, inline resources, and copy integrity.
- Handing off a relative path, incomplete source excerpt, or report with missing provenance.
