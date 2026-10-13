"""Keep four fixed player identities across frames.

Identities are seeded once (on the first frame, by the user) and then each
frame's person detections are matched to the four slots by distance of the
feet point, so a player keeps the same name when detections drop out or the
detector's own track ids change.
"""

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass(frozen=True)
class Person:
    bbox: BBox
    # Visible wrist keypoints (image px). Pose models miss far players, so
    # this is often empty for them.
    wrists: tuple[tuple[float, float], ...] = ()
    # Normalised colour histogram of the torso, to keep partners apart when
    # they cross (see `torso_histogram`).
    hist: np.ndarray | None = None


def torso_histogram(frame: np.ndarray, b: BBox) -> np.ndarray | None:
    """Hue-saturation histogram of the shirt area (20-55% of the box height)."""
    x1, y1, x2, y2 = (int(round(v)) for v in b)
    h = y2 - y1
    crop = frame[max(0, y1 + int(0.2 * h)) : max(0, y1 + int(0.55 * h)), max(0, x1) : max(0, x2)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def feet(b: BBox) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, b[3]


@dataclass
class PlayerSlot:
    name: str
    team: str
    bbox: BBox
    last_seen: int = 0
    hist: np.ndarray | None = None


class PlayerSlots:
    def __init__(
        self, slots: list[PlayerSlot], max_jump_px: float = 160.0, appearance_px: float = 150.0
    ):
        self.slots = slots
        self.max_jump_px = max_jump_px
        # Cost of a completely different shirt, in pixels of feet distance.
        self.appearance_px = appearance_px

    def update(
        self,
        frame: int,
        detections: list[Person],
        allowed: Callable[[PlayerSlot, BBox], bool] | None = None,
    ) -> dict[int, int]:
        """Match detections to slots; returns slot index -> detection index.

        Cost is the feet distance, plus a size change term, plus how different
        the shirt looks, so partners who cross keep their names.
        `allowed(slot, bbox)` can forbid pairs outright, e.g. a near-side slot
        taking a person on the far side of the net.
        """
        if not detections:
            return {}
        cost = np.full((len(self.slots), len(detections)), 1e9)
        for i, s in enumerate(self.slots):
            sx, sy = feet(s.bbox)
            h = s.bbox[3] - s.bbox[1]
            # Allow larger jumps for players unseen for a while.
            gate = self.max_jump_px * (1 + 0.1 * min(frame - s.last_seen, 30))
            for j, p in enumerate(detections):
                d = p.bbox
                dx, dy = feet(d)
                dist = np.hypot(dx - sx, dy - sy)
                size = abs((d[3] - d[1]) - h) / max(h, 1)
                if dist <= gate and (allowed is None or allowed(s, d)):
                    cost[i, j] = (
                        dist + 50 * size + self.appearance_px * _hist_distance(s.hist, p.hist)
                    )
        rows, cols = linear_sum_assignment(cost)
        out = {}
        for i, j in zip(rows, cols, strict=True):
            if cost[i, j] < 1e9:
                slot, p = self.slots[i], detections[j]
                slot.bbox = p.bbox
                slot.last_seen = frame
                if p.hist is not None:
                    slot.hist = p.hist if slot.hist is None else 0.9 * slot.hist + 0.1 * p.hist
                out[int(i)] = int(j)
        return out


def _hist_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    return float(
        cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
    )
