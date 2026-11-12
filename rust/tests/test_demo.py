# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The demo must cover every tool, and must not be able to lie about its length.

A demo rots faster than code: a renamed flag or a moved corpus file turns a
segment into a wall of errors, and nobody notices until it is being recorded.
These checks run without a toolchain and without recording anything -- each
scene has a dry mode that executes no commands -- so they can sit in the normal
suite.

The length check is the one that matters. The request is for roughly five
minutes per segment, and a script CAN be made to print any number it likes. So
the projection is recomputed here from the scene itself, and the driver is
required to measure what it actually recorded rather than assume.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "rust" / "demo"

# The five utilities the demo exists to show.
TOOLS = ["rspto", "ptoas-rs", "tile-search", "pto-to-ascendc", "ds4-rs-ascend"]

TARGET_S = 300
# Checked at the DRIVER'S calibrated speed, not at 1.0 -- a scene is written to
# be recorded at its own pacing, and judging it at unity measures a
# configuration nobody records. The window is generous because the projection
# excludes real command runtime, so a scene projecting a little under target
# still lands near it; what this rules out is one that could not fill the slot
# at all, or one padded to twice it.
MIN_PROJECTED_S = 210
MAX_PROJECTED_S = 480


def driver_speed(scene: str) -> str:
    """The pacing the driver will actually record this scene at."""
    text = (DEMO / "run.sh").read_text(encoding="utf-8")
    m = re.search(rf"{re.escape(scene)}\)\s*printf '([0-9.]+)'", text)
    assert m, f"the driver has no pacing entry for {scene}"
    return m.group(1)


def dry_run(scene: str, speed: str = "1.0") -> str:
    r = subprocess.run(
        ["bash", str(DEMO / "scenes" / f"{scene}.sh")],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DEMO_DRY": "1",
            "DEMO_SPEED": speed,
            "HOME": str(Path.home()),
        },
    )
    assert r.returncode == 0, f"{scene} dry run failed: {r.stderr[-800:]}"
    return r.stdout


@pytest.mark.parametrize("tool", TOOLS)
def test_every_tool_has_a_scene(tool):
    assert (DEMO / "scenes" / f"{tool}.sh").is_file(), f"no demo scene for {tool}"


def test_there_are_no_scenes_for_tools_that_do_not_exist():
    """The other direction: a scene for a removed tool would record errors."""
    scenes = {p.stem for p in (DEMO / "scenes").glob("*.sh")}
    assert scenes == set(TOOLS), f"scenes and tools disagree: {scenes ^ set(TOOLS)}"


@pytest.mark.parametrize("tool", TOOLS)
def test_each_scene_exercises_its_own_tool(tool):
    text = (DEMO / "scenes" / f"{tool}.sh").read_text(encoding="utf-8")
    assert f"tool {tool}" in text or f'tool "{tool}"' in text, (
        f"{tool}'s scene never resolves the {tool} binary"
    )
    assert "source" in text and "lib.sh" in text, f"{tool}'s scene does not use the shared helpers"


@pytest.mark.parametrize("tool", TOOLS)
def test_each_scene_projects_a_narratable_length(tool):
    out = dry_run(tool, driver_speed(tool))
    m = re.search(r"projected_seconds\t(\d+)", out)
    assert m, f"{tool} does not report a projected duration"
    projected = int(m.group(1))
    assert MIN_PROJECTED_S <= projected <= MAX_PROJECTED_S, (
        f"{tool} projects {projected}s, outside [{MIN_PROJECTED_S}, {MAX_PROJECTED_S}] "
        f"for a {TARGET_S}s target"
    )


@pytest.mark.parametrize("tool", TOOLS)
def test_each_scene_emits_timestamped_narration_cues(tool):
    out = dry_run(tool)
    cues = [line for line in out.splitlines() if re.match(r"^\d\d:\d\d\t", line)]
    assert len(cues) >= 12, f"{tool} has only {len(cues)} narration cue(s)"

    # Monotonic, or a voice artist cannot follow them.
    def secs(line):
        mm, ss = line.split("\t")[0].split(":")
        return int(mm) * 60 + int(ss)

    stamps = [secs(c) for c in cues]
    assert stamps == sorted(stamps), f"{tool}'s cues are not in time order"


@pytest.mark.parametrize("tool", TOOLS)
def test_speed_scales_the_whole_segment(tool):
    """One knob has to move the runtime, or the five-minute target is untunable."""
    slow = int(re.search(r"projected_seconds\t(\d+)", dry_run(tool, "2.0")).group(1))
    fast = int(re.search(r"projected_seconds\t(\d+)", dry_run(tool, "1.0")).group(1))
    assert slow > fast * 1.8, f"{tool}: DEMO_SPEED does not scale the segment"


def test_the_driver_knows_every_scene_and_paces_each_one():
    text = (DEMO / "run.sh").read_text(encoding="utf-8")
    for tool in TOOLS:
        assert tool in text, f"the driver does not list {tool}"
    # A pacing entry per scene, or one segment records at the wrong length.
    for tool in TOOLS:
        assert re.search(rf"{re.escape(tool)}\)\s*printf", text), f"the driver has no pacing entry for {tool}"


def test_the_driver_measures_rather_than_assumes_the_length():
    """The property that keeps the five-minute claim honest."""
    text = (DEMO / "run.sh").read_text(encoding="utf-8")
    assert "cast_seconds" in text, "the driver never measures a recording"
    assert "OUTSIDE TOLERANCE" in text, "the driver never reports a length it cannot meet"
    # Cast v3 stores deltas; summing is the whole correctness of the measurement.
    assert "total +=" in text, "the driver does not SUM the cast's event deltas"


def test_the_gif_step_does_not_clip_the_narration_pauses():
    """The bug this exists for, and it was silent.

    `agg --idle-time-limit` defaults to FIVE SECONDS. In a narrated demo the
    pauses are the content -- they are where the voice-over goes -- and the
    default clipped every hold longer than that. A 290-second recording rendered
    as a 233-second GIF, and the narration marks no longer lined up with the
    picture. Nothing failed; the file was simply wrong.
    """
    text = (DEMO / "run.sh").read_text(encoding="utf-8")
    m = re.search(r"--idle-time-limit\s+(\d+)", text)
    assert m, "the gif step does not set --idle-time-limit, so agg clips every long pause"
    assert int(m.group(1)) >= 60, (
        f"--idle-time-limit is {m.group(1)}s; the longest narration hold exceeds that"
    )


def test_the_mux_does_not_let_the_audio_truncate_the_picture():
    """The third time this pipeline lost time, and the worst of the three.

    `-shortest` stops at the shorter stream. A draft voice-over is far shorter
    than its segment, so it cut a 306-second recording to a 178-second file --
    AFTER the mp4 step had verified the silent version as correct. The published
    artefact was the muxed one, so a passing check sat next to a wrong output.
    The audio is padded with silence instead, and the published file is the one
    that gets verified.
    """
    text = (DEMO / "run.sh").read_text(encoding="utf-8")
    assert "apad" in text, "the mux does not pad the audio, so it can truncate the video"
    assert "WARNING: narrated mp4 is" in text, "the mux does not verify the file it publishes"


def test_the_mp4_step_preserves_the_gif_timing_and_checks_itself():
    """The same failure one stage later: ffmpeg re-times a variable-delay GIF
    unless told not to, which shortens the video all over again."""
    text = (DEMO / "run.sh").read_text(encoding="utf-8")
    assert "fps_mode passthrough" in text, "the mp4 step lets ffmpeg re-time the stream"
    assert "WARNING: mp4 is" in text, "the mp4 step does not verify its own duration"


@pytest.mark.parametrize("tool", TOOLS)
def test_published_media_matches_its_recording(tool):
    """Measured, when the media and the tools are both present.

    Separate from the source-level checks above because those always run and
    this cannot: media is produced by a recording step nobody runs in CI. When
    it IS here, it has to agree with the cast -- otherwise the five-minute claim
    and the narration marks are both wrong.
    """

    gif = DEMO / "media" / f"{tool}.gif"
    mp4 = DEMO / "media" / f"{tool}.mp4"
    cast = DEMO / "out" / f"{tool}.cast"
    if not (gif.is_file() and mp4.is_file() and cast.is_file()):
        pytest.skip(f"{tool}: no published media and recording on this machine")
    if not shutil.which("ffprobe"):
        pytest.skip("ffprobe is not installed")

    total = 0.0
    with cast.open() as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            try:
                total += float(__import__("json").loads(line)[0])
            except Exception:
                pass

    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "csv=p=0",
            str(mp4),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    video = float(out.stdout.strip())
    # 8%: GIF frame delays are quantised to centiseconds, and over a few
    # thousand frames that rounding accumulates -- the measured spread across
    # these five segments is -1.0% to -6.0%. The failures this guards against
    # are not subtle: a clipped pause or a truncating mux shows up at -40%.
    assert abs(video - total) <= 0.08 * total, (
        f"{tool}: the mp4 is {video:.0f}s against a {total:.0f}s recording. "
        "Something in the gif/mp4 chain is dropping time -- check "
        "agg --idle-time-limit and ffmpeg -fps_mode."
    )


def test_recordings_are_not_committed():
    """A five-minute terminal GIF is tens of megabytes."""
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "rust/demo/out/" in ignore
    # Working artefacts stay in demo/out and are ignored. demo/media is the
    # PUBLISHED set the READMEs reference, so it is committed deliberately.
    for pat in ("*.cast", "*.aiff"):
        found = [p for p in DEMO.rglob(pat) if "out" not in p.parts]
        assert not found, f"working artefacts committed outside rust/demo/out: {found}"


def test_every_scene_has_a_cue_sheet_on_disk():
    """Generated, but committed: the cue sheet is the narration deliverable."""
    for tool in TOOLS:
        f = DEMO / "narration" / f"{tool}.txt"
        assert f.is_file(), f"no cue sheet for {tool} -- run `rust/demo/run.sh cues`"
        text = f.read_text(encoding="utf-8")
        assert re.search(r"^\d\d:\d\d\t", text, re.MULTILINE), f"{tool}'s cue sheet has no marks"
        assert "DEMO_SPEED" in text, f"{tool}'s cue sheet does not say which pacing it is for"


def test_the_demo_carries_no_markdown():
    """One README for the whole tree, by instruction."""
    assert not list(DEMO.rglob("*.md")), "the demo added markdown"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
