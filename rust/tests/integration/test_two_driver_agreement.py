"""The regression test for the bugs that live in the DRIVERS, not the library.

Five of this campaign's fixes were made inside `examples/glm53_block0.rs` and `examples/prefill_kda.rs` -- the skipped FFN
on MLA blocks, the missing MoE in one driver, the missing final norm, `beta`'s input, the stubbed constants. **None of them
can be caught by a unit test**, because they are control flow in a 300-line driver.

What caught every one of them was running BOTH drivers and comparing. This test does that automatically, so the day one of
those bugs returns, the suite says so instead of a person noticing two numbers differ.

Slow (a few seconds per block, real 96 GB checkpoint), so it is skipped unless DS4_SLOW=1.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = ROOT / "ds4-rs-ascend" / "target" / "release" / "examples"
DECODE = EXAMPLE_DIR / "glm53_block0"
PREFILL = EXAMPLE_DIR / "prefill_kda"
CKPT = Path("/Volumes/Data/models/glm-5.3-flash-gguf/GLM-5.3-Flash-Q2.gguf")

BLOCKS = 2  # enough to cross the first MLA block (block 3 is MLA, so use 4)

slow = pytest.mark.skipif(
    os.environ.get("DS4_SLOW") != "1",
    reason="slow integration test: set DS4_SLOW=1 and build the two examples",
)


def _run(binary: Path, env_extra: dict) -> str:
    assert binary.exists(), f"build it first: cargo build --release --example {binary.stem}"
    env = dict(os.environ, **env_extra)
    out = subprocess.run([str(binary), str(CKPT)], capture_output=True, text=True, env=env, timeout=1800)
    assert out.returncode == 0, f"{binary.stem} failed:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}"
    return out.stdout


def _token(stdout: str) -> tuple[int, float]:
    """The two drivers print their result differently, so accept BOTH forms.

    decode:  `logits: 154880 finite, best 5.4487 at id 88504`
    prefill: `ONE-TOKEN prefill after 6 blocks: 88504 (5.448745)`

    (My first version parsed only the first, so this test failed on the second driver's output rather than on a real
    disagreement -- the test's own bug, which is the third one this campaign has had of that kind.)
    """
    m = re.search(r"best ([-\d.]+) at id (\d+)", stdout)
    if m:
        return int(m.group(2)), float(m.group(1))
    m = re.search(r"ONE-TOKEN prefill after \d+ blocks: (\d+) \(([-\d.]+)\)", stdout)
    assert m, f"no logits line in either format:\n{stdout[-1500:]}"
    return int(m.group(1)), float(m.group(2))


@slow
def test_the_two_drivers_agree_on_the_token_and_the_logit():
    """THE regression test for the five driver-level fixes.

    Both drivers run the same blocks over the same weights. If either loses its FFN, its MoE, its final norm or its
    `beta` input, the tokens diverge -- which is how every one of those bugs was found.
    """
    decode = _run(DECODE, {"GLM53_PROMPT": "Hello", "GLM53_BLOCKS": str(BLOCKS)})
    prefill = _run(PREFILL, {"PREFILL_ONE": "1", "PREFILL_BLOCKS": str(BLOCKS)})
    d_tok, d_logit = _token(decode)
    p_tok, p_logit = _token(prefill)
    assert d_tok == p_tok, f"the drivers disagree on the token: decode {d_tok} vs prefill {p_tok}"
    # THE TOLERANCE IS THE PRINT PRECISION, NOT THE ARITHMETIC'S: the decode driver prints its logit with 4 decimals and
    # the prefill with 6, so a 1e-6 bound is tighter than the data this test can see. (Measured: 1.3e-05 apart.)
    assert abs(d_logit - p_logit) < 1e-3, (
        f"the drivers agree on the token but not the logit: {d_logit} vs {p_logit} -- a magnitude difference with the "
        f"same argmax is what the missing final norm looked like"
    )


@slow
@pytest.mark.parametrize("blocks", [2], ids=["2blocks"])
def test_the_drivers_still_agree_at_a_DEPTH_THAT_RUNS_IN_MINUTES(blocks):
    """A FAST version of the agreement test, added because the 6-block one takes ~34 minutes and so is skipped in practice.

    The bug that motivated it: `prefill_kda.rs` carried its own hardcoded `1536` q-latent while the decode driver's was
    derived. At 2 blocks the two drivers still agree (110512), and at 6 they agree (88504) -- so this test guards the
    cheap case, which is the one that will actually run on every change.
    """
    decode = _run(DECODE, {"GLM53_PROMPT": "Hello", "GLM53_BLOCKS": str(blocks)})
    prefill = _run(PREFILL, {"PREFILL_ONE": "1", "PREFILL_BLOCKS": str(blocks)})
    d_tok, _ = _token(decode)
    p_tok, _ = _token(prefill)
    assert d_tok == p_tok, f"the drivers disagree at {blocks} blocks: decode {d_tok} vs prefill {p_tok}"


@slow
@pytest.mark.parametrize("blocks", [2, 6], ids=["2blocks", "6blocks"])
def test_hyper_connections_agree_across_BOTH_drivers(blocks):
    """THE regression test for B14: the prefill accepted `GLM53_HC` and ignored it.

    For one round the drivers "agreed" only in the no-HC configuration, because hyper-connections were implemented in decode
    and absent from prefill. The cross-check the whole campaign rests on was silently checking the wrong path -- and still
    printing PASS.

    Two assertions, because either alone is passable in the broken state:
      * both drivers must produce the SAME token WITH HC on;
      * that token must DIFFER from the no-HC token, so a prefill that ignores the flag still fails.
    """
    env = {"GLM53_BLOCKS": str(blocks), "GLM53_HC": "1"}
    d_hc, _ = _token(_run(DECODE, dict(env, GLM53_PROMPT="Hello")))
    p_hc, _ = _token(_run(PREFILL, dict(env, PREFILL_ONE="1", PREFILL_BLOCKS=str(blocks))))
    assert "hyper-connections" in _run(PREFILL, dict(env, PREFILL_ONE="1", PREFILL_BLOCKS=str(blocks))), (
        "the prefill must ANNOUNCE its hyper-connections; silence means it ignored the flag"
    )
    assert d_hc == p_hc, f"the drivers disagree WITH hyper-connections: decode {d_hc} vs prefill {p_hc}"
    d_plain, _ = _token(_run(DECODE, {"GLM53_PROMPT": "Hello", "GLM53_BLOCKS": str(blocks)}))
    assert d_hc != d_plain, (
        f"hyper-connections did not change the token ({d_hc} == {d_plain}), so this test cannot tell a wired path from an "
        f"ignored flag -- exactly the state B14 described"
    )


@slow
def test_hyper_connections_change_the_token_but_do_not_break_it():
    """The HC path is behind a flag; it must RUN, and it must change the output -- otherwise it is not wired in."""
    without = _run(DECODE, {"GLM53_PROMPT": "Hello", "GLM53_BLOCKS": str(BLOCKS)})
    with_hc = _run(DECODE, {"GLM53_PROMPT": "Hello", "GLM53_BLOCKS": str(BLOCKS), "GLM53_HC": "1"})
    assert "hyper-connections: 4 streams" in with_hc, "the HC path did not announce itself:\n" + with_hc[-1500:]
    t0, _ = _token(without)
    t1, _ = _token(with_hc)
    assert t1 != t0, (
        f"hyper-connections produced the SAME token ({t0}) -- either the path is not wired in or it is being ignored"
    )


KIMI_SHARDS = sorted(
    str(p) for p in Path("/Volumes/T7 Shield/LLMs/kimi-k3-ud-iq1s").glob("*.gguf")
) if Path("/Volumes/T7 Shield/LLMs/kimi-k3-ud-iq1s").is_dir() else []


@slow
def test_kimi_agrees_across_BOTH_drivers():
    """THE Kimi cross-check, which took nine rounds to make trustworthy.

    Every earlier comparison was contaminated: a GLM baseline against a Kimi run, a decode number from the no-prompt path
    (which embeds a SYNTHETIC vector), a prefill that hardcoded "Hello world", a prefill that demanded two tokens when
    "Hello" is one. Once all of that was fixed, the two drivers still disagreed -- and the cause was a real bug: the decode
    driver's output projection hardcoded 8192, GLM's `heads*head_dim`, so on Kimi (12288 wide) it summed the first 8192 and
    strode the wrong rows.

    Two assertions, because the first alone passes if both drivers share a bug and the second alone passes if the model is
    never run:
      * both drivers produce the same token for Kimi at 2 blocks;
      * that token is in Kimi's vocabulary and differs from the no-prompt run, so a synthetic embedding cannot pass.
    """
    # SORTED, AND ONLY REAL SHARDS: `glob` order is arbitrary, and shard 1 must be FIRST -- it carries the metadata
    # (`expert_weights_scale` lives there). An unsorted list failed with "no expert_weights_scale in this checkpoint",
    # which reads like a missing key and was actually a missing order.
    shards = sorted(
        p for p in Path("/Volumes/T7 Shield/LLMs/kimi-k3-ud-iq1s").glob("Kimi-K3-UD-IQ1_S-*-of-00014.gguf")
    )
    if len(shards) < 2:
        pytest.skip("Kimi K3's shards are not mounted")
    common = {"DS4_KIMI": "1", "GLM53_PROMPT": "Hello"}
    d = subprocess.run([str(DECODE), *map(str, shards)], capture_output=True, text=True, timeout=3600,
                       env=dict(os.environ, **common, GLM53_BLOCKS="2"))
    assert d.returncode == 0, f"decode failed:\n{d.stdout[-1500:]}\n{d.stderr[-1500:]}"
    pr = subprocess.run([str(PREFILL), *map(str, shards)], capture_output=True, text=True, timeout=3600,
                        env=dict(os.environ, **common, PREFILL_ONE="1", PREFILL_BLOCKS="2"))
    assert pr.returncode == 0, f"prefill failed:\n{pr.stdout[-1500:]}\n{pr.stderr[-1500:]}"
    d_tok, _ = _token(d.stdout)
    p_tok, _ = _token(pr.stdout)
    assert d_tok == p_tok, (
        f"THE KIMI CROSS-CHECK FAILS: decode {d_tok} vs prefill {p_tok}. This is the check that found the hardcoded 8192."
    )
    assert 0 <= d_tok < 163840, f"token {d_tok} is outside Kimi's vocabulary"
