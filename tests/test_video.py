from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from padel.cli import app
from padel.video import VideoError, probe


def write_video(path: Path, fps: float, frames: int, size=(64, 48)) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), i % 255, dtype=np.uint8))
    writer.release()
    return path


def test_probe_reads_metadata(tmp_path):
    info = probe(write_video(tmp_path / "clip.avi", fps=60, frames=120))
    assert (info.width, info.height) == (64, 48)
    assert info.fps == pytest.approx(60)
    assert info.frame_count == 120
    assert info.duration_s == pytest.approx(2.0)


def test_probe_missing_file(tmp_path):
    with pytest.raises(VideoError):
        probe(tmp_path / "missing.mp4")


def test_info_command(tmp_path):
    video = write_video(tmp_path / "clip.avi", fps=60, frames=60)
    result = CliRunner().invoke(app, ["info", str(video)])
    assert result.exit_code == 0
    assert "64x48" in result.output
    assert "60.00 fps" in result.output
    assert "Uwaga" not in result.output


def test_info_warns_on_low_fps(tmp_path):
    video = write_video(tmp_path / "clip.avi", fps=30, frames=30)
    result = CliRunner().invoke(app, ["info", str(video)])
    assert result.exit_code == 0
    assert "Uwaga" in result.output


def test_info_missing_file_exits_nonzero(tmp_path):
    result = CliRunner().invoke(app, ["info", str(tmp_path / "missing.mp4")])
    assert result.exit_code == 1
