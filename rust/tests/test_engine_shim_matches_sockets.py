"""The shim's declarations must mangle to EXACTLY the engine's socket symbols.

`engine_shim.cpp` exposes Rust kernels under the engine's C++ names. `device.rs` binds by MANGLED
symbol, so a signature that differs in arity or type produces a different symbol and the engine
silently leaves that slot on the CPU -- which is the benign version. The harmful version is what
happened for thirty rounds: an artefact that matched the SHAPE but not the signature was counted as
serving a slot (iterations 45-46). This test pins the contract for the slots the shim claims.

It deliberately does NOT require every socket to be covered: the four `ds4_q8mv_*` sockets are 4-arg
Q8_0 with a companion plane and the w8a8 arms take five arguments, so covering them would be the
over-credit iteration 51 removed.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "rust"
DEVICE = ROOT / "ds4-rs-ascend" / "crates" / "engine_ascend" / "src" / "device.rs"
SHIM = ROOT / "tools" / "direct-invoke" / "engine_shim.cpp"

# Itanium mangling for the pointer types this engine uses: Pf = float*, Pa = signed char*.
CODE = {"float*": "Pf", "signed char*": "Pa"}


def mangle(name: str, params: list[str]) -> str:
    """Mangle `name(param types)` the way ccec/g++ does for these plain pointer signatures."""
    codes = [CODE[p] for p in params]
    # Substitutions: the second and later occurrence of a type is S_, S0_, S1_...
    seen: dict[str, int] = {}
    out = []
    for c in codes:
        if c in seen:
            out.append("S_" if seen[c] == 0 else f"S{seen[c] - 1}_")
        else:
            seen[c] = len(seen)
            out.append(c)
    return f"_Z{len(name)}{name}{''.join(out)}"


def shim_functions() -> dict[str, list[str]]:
    text = SHIM.read_text()
    fns: dict[str, list[str]] = {}
    for m in re.finditer(r"__global__\s+__aicore__\s+void\s+(\w+)\s*\(([^)]*)\)", text):
        name, raw = m.group(1), m.group(2)
        params = []
        for p in raw.split(","):
            p = p.strip()
            if not p:
                continue
            # "__gm__ float *x" -> "float*"
            t = p.replace("__gm__", "").strip()
            t = re.sub(r"\s*\*\s*\w+$", "*", t).strip()
            params.append(t)
        fns[name] = params
    return fns


def engine_symbols() -> dict[str, str]:
    text = DEVICE.read_text()
    out: dict[str, str] = {}
    # A comment block may sit between `name:` and `symbol:` (rope_halves has one), so allow
    # intervening lines rather than requiring adjacency -- otherwise the parser silently drops
    # sockets and the test reports a mismatch that does not exist.
    for m in re.finditer(r'name:\s*"(\w+)",(?:\n(?!\s*name:)[^\n]*){0,6}?\n\s*symbol:\s*"([^"]+)"', text):
        out[m.group(1)] = m.group(2)
    return out


def test_the_shim_mangles_to_the_engine_socket_symbols():
    fns = shim_functions()
    assert fns, "no shim functions parsed -- the regex or the file moved"
    syms = engine_symbols()
    assert syms, "no engine sockets parsed -- device.rs layout changed"

    checked = 0
    for name, params in fns.items():
        want = mangle(name, params)
        assert want in syms.values(), (
            f"shim function {name}({', '.join(params)}) mangles to {want}, "
            f"which is not a socket symbol in device.rs. Covered sockets: {sorted(syms.values())}"
        )
        checked += 1
    # The five slots the shim claims, by name.
    for slot in ("rms_norm_4096", "rms_norm_1024", "rms_norm_512", "hc_norm", "rope_halves"):
        assert slot in syms, f"{slot} vanished from device.rs"
    assert checked == 10, f"expected the shim to cover ten sockets, it declares {checked}"


def test_the_still_unimplemented_sockets_are_deliberately_uncovered():
    # A shim entry for these would compile and would be wrong: 4-argument Q8_0 with a companion
    # plane against 5-argument w8a8. Keeping them out is the point, so their absence is asserted.
    # `ds4_q8mv_*` LEFT THIS LIST when it was IMPLEMENTED (iteration 57-60): its four pointers are
    # (w, s, x, o) with pre-expanded scales, which a Rust kernel now computes exactly. What stays
    # out is what has no kernel behind it: packed formats (`hc_mv`, `mx4pk_*`), flash attention,
    # and the grouped MoE.
    fns = shim_functions()
    # `ds4_attn_sink_batched` LEFT this list when the certified one-head kernel plus a 64-head shim
    # loop was wired (iteration 89). What remains has no kernel behind it at all.
    for name in ("ds4_hc_mv", "ds4_mx4pk_4096_2048", "ds4_mx4pk_2048_4096",
                 "ds4_gmm_swiglu_quant_16"):
        assert name not in fns, (
            f"{name} is in the shim, but its socket is a packed/quantised format the w8a8 arms "
            f"do not implement -- that is the over-credit iterations 45-46 removed"
        )
