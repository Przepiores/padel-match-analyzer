"""Video file access."""

from dataclasses import dataclass
from pathlib import Path

import cv2


class VideoError(Exception):
    """Raised when a video file cannot be opened or read."""


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


def probe(path: Path) -> VideoInfo:
    """Read basic metadata of a video file."""
    if not path.is_file():
        raise VideoError(f"Plik nie istnieje: {path}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError(f"Nie można otworzyć wideo: {path}")
        return VideoInfo(
            path=path,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()
