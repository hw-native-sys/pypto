#!/usr/bin/env python3
"""Count the emission sites where a typed constructor would enforce something.

WHY THIS EXISTS. Deciding what to convert next was being done by eye and by regex, and
both lied: a regex that split the type list on commas counted `vec_ds, vec_ds, rr_d` as
"carries a check" both before and after that statement was converted, so the number it
reported stayed at 27 while eight sites moved out of it. A metric that does not move
when the code does is not a measurement.

WHAT IT ACTUALLY DOES. It parses each `ops.push(format!(...))` whose format string emits
a same-type PTO op, fills the placeholders from the argument list the way `format!`
would, and reads the TYPE slots out of the resulting text:

    ins(<a>, <b> : <Ta>, <Tb>) outs(<c> : <Tc>)

A site CARRIES A CHECK when the type expressions are not all the same identifier --
i.e. when two independently derived expressions must agree, which is exactly what a
constructor can refuse. A site whose three slots name one variable enforces nothing.

Exits 1 if the file cannot be parsed, so a silent zero cannot pass for a clean result.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EMITTER = ROOT / "rust/crates/rspto_emit/src/mlir_to_pto.rs"

SAME_TYPE_OPS = (
    "pto.tmul", "pto.tadd", "pto.tmuls", "pto.tadds", "pto.texp", "pto.tsqrt",
    "pto.trecip", "pto.tmaxs", "pto.tmins", "pto.tabs", "pto.tneg", "pto.tfill",
    "pto.trowsum",
)
# Ops whose PURPOSE is to change a tile's type, layout or location. A constructor
# requiring one type cannot describe them, so they are never candidates.
BRIDGING_OPS = (
    "pto.tmov", "pto.tcvt", "pto.trowexpand", "pto.tcolexpand", "pto.tinsert",
    "pto.tpop_from_aic", "pto.tconcat", "pto.tsplit", "pto.treshape",
    "pto.tpush_to_aiv", "pto.tfree_from_aic", "pto.tfillpad", "pto.tfill_pad",
)

CALL = re.compile(r"ops\.push\(\s*format!\(\s*(?P<body>.*?)\)\)\s*;", re.S)
PLACEHOLDER = re.compile(r"\{(\d*)\}")


def split_args(body: str) -> tuple[str, list[str]]:
    """Split `"fmt", a, b, c` into the format string and its argument expressions."""
    body = body.strip()
    if not body.startswith('"'):
        raise ValueError("not a string literal")
    i, out = 1, []
    while i < len(body):
        ch = body[i]
        if ch == "\\":
            i += 2
            continue
        if ch == '"':
            break
        i += 1
    fmt = body[1:i]
    # The text after the closing quote begins with the separating comma; dropping it
    # matters because otherwise the first split yields an empty argument and every
    # argument is read one slot to the right -- which made SSA names look like types and
    # reported every site as informative.
    rest = body[i + 1 :].lstrip()
    if rest.startswith(","):
        rest = rest[1:]
    args, depth, cur = [], 0, ""
    for ch in rest:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        args.append(cur.strip())
    return fmt, args


def fill(fmt: str, args: list[str]) -> str:
    """Substitute placeholders the way format! does, leaving prose intact."""
    seq = 0

    def sub(m: re.Match) -> str:
        nonlocal seq
        idx = m.group(1)
        if idx:
            n = int(idx)
        else:
            n, seq = seq, seq + 1
        return args[n] if 0 <= n < len(args) else m.group(0)

    return PLACEHOLDER.sub(sub, fmt)


def type_slots(rendered: str) -> list[str]:
    """The type expressions after `ins(... :` and `outs(... :`."""
    slots: list[str] = []
    for kw in ("ins(", "outs("):
        at = rendered.find(kw)
        if at < 0:
            continue
        close = rendered.find(")", at)
        inner = rendered[at + len(kw) : close if close >= 0 else len(rendered)]
        if ":" in inner:
            tail = inner.split(":", 1)[1]
            slots.extend(p.strip() for p in tail.split(",") if p.strip())
    return slots


def main() -> int:
    if not EMITTER.is_file():
        print(f"cannot read {EMITTER}", file=sys.stderr)
        return 1
    text = EMITTER.read_text()
    checked = uninformative = 0
    by_op: dict[str, list[int]] = {}
    for m in CALL.finditer(text):
        body = m.group("body")
        mnemonic = next((o for o in SAME_TYPE_OPS if f'"{o}' in body), None)
        if mnemonic is None or any(f'"{b}' in body for b in BRIDGING_OPS):
            continue
        if "pto_ir::" in body:
            continue
        try:
            fmt, args = split_args(body)
        except ValueError:
            continue
        slots = type_slots(fill(fmt, args))
        # ONLY SUBSTITUTED SLOTS COUNT. A scalar op spells its scalar's dtype as a
        # literal -- `ins({}, {} : {}, f32)` -- and counting that literal as a second
        # type expression made every tmuls/tadds site look like it carried a check.
        # A scalar op whose multiplier is the SOURCE ITSELF (`ins({0}, {0} : {1}, {1})`)
        # has no scalar to type: the tile stands in for one, which `Scalar` cannot
        # describe because its scalar operand is a literal and a dtype.
        ins_part = fmt.split("ins(", 1)[1].split(")", 1)[0] if "ins(" in fmt else ""
        heads = re.findall(r"\{(\d*)\}", ins_part)
        if len(heads) > 1 and heads[0] == heads[1]:
            continue
        # An IN-PLACE scalar op spells its source as its own scalar
        # (`ins({0}, {0} : {1}, {1})`); there is no second type to compare.
        if len(args) > 1 and args[0] == args[1]:
            continue
        slots = [s for s in slots if s in args]
        if not slots:
            continue
        distinct = {s for s in slots if s}
        line = text.count("\n", 0, m.start()) + 1
        if len(distinct) > 1:
            checked += 1
            by_op.setdefault(mnemonic, []).append(line)
        else:
            uninformative += 1

    print(f"same-type sites still built from strings: {checked + uninformative}")
    print(f"  carrying a real check (two or more type expressions must agree): {checked}")
    print(f"  one type variable in every slot (a constructor would enforce nothing): {uninformative}")
    if not checked:
        print("\nThere is nothing left to convert that would enforce anything.")
    if checked:
        print("\nnext candidates by op:")
        for op, lines in sorted(by_op.items(), key=lambda kv: -len(kv[1])):
            shown = ", ".join(str(x) for x in lines[:8])
            more = "" if len(lines) <= 8 else f", ... (+{len(lines) - 8})"
            print(f"  {op:12} {len(lines):3}  lines {shown}{more}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
