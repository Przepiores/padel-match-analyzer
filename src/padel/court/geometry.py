"""Court model and image <-> court mapping.

Court coordinates are metres: x across the court (0..10), y along it (0..20),
y=0 at the near baseline, net at y=10. The homography is valid only for points
on the floor.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

COURT_WIDTH_M = 10.0
COURT_LENGTH_M = 20.0
NET_Y_M = COURT_LENGTH_M / 2

# Corner order used everywhere: near-left, near-right, far-right, far-left.
CORNERS_M = np.array(
    [[0, 0], [COURT_WIDTH_M, 0], [COURT_WIDTH_M, COURT_LENGTH_M], [0, COURT_LENGTH_M]],
    dtype=np.float64,
)


@dataclass(frozen=True)
class Court:
    image_to_court: np.ndarray  # 3x3 homography
    court_to_image: np.ndarray

    @classmethod
    def from_points(
        cls, image_px: list[tuple[float, float]], court_m: list[tuple[float, float]]
    ) -> "Court":
        """Build from any >= 4 floor landmarks (no 3 collinear) with known court
        positions, e.g. far corners, net post bases, service line T. Near
        corners are often out of frame, so they are not required."""
        src = np.asarray(image_px, dtype=np.float64)
        dst = np.asarray(court_m, dtype=np.float64)
        if src.shape != dst.shape or src.ndim != 2 or len(src) < 4:
            raise ValueError("Potrzeba co najmniej 4 par punktów (obraz, kort).")
        h, _ = cv2.findHomography(src, dst, method=0)
        if h is None:
            raise ValueError("Punkty są zdegenerowane (np. 3 na jednej prostej).")
        return cls(image_to_court=h, court_to_image=np.linalg.inv(h))

    @classmethod
    def from_corners(cls, corners_px: list[tuple[float, float]]) -> "Court":
        """Build from the 4 floor corners in image pixels (near-left, near-right,
        far-right, far-left)."""
        return cls.from_points(corners_px, CORNERS_M.tolist())

    @classmethod
    def from_json(cls, path: Path, scale: float = 1.0) -> "Court":
        """Load `{"points": [{"image": [x, y], "court": [x_m, y_m]}, ...]}`;
        `scale` rescales image coordinates when frames are resized."""
        pts = json.loads(path.read_text())["points"]
        return cls.from_points(
            [(p["image"][0] * scale, p["image"][1] * scale) for p in pts],
            [tuple(p["court"]) for p in pts],
        )

    def to_court(self, x: float, y: float) -> tuple[float, float]:
        p = self.image_to_court @ np.array([x, y, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def to_image(self, x_m: float, y_m: float) -> tuple[float, float]:
        p = self.court_to_image @ np.array([x_m, y_m, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def contains(self, x: float, y: float, margin_m: float = 0.0) -> bool:
        """Whether an image point on the floor lies inside the court (+ margin)."""
        cx, cy = self.to_court(x, y)
        return (
            -margin_m <= cx <= COURT_WIDTH_M + margin_m
            and -margin_m <= cy <= COURT_LENGTH_M + margin_m
        )

    def ball_region(self, width: int, height: int, margin_px: int = 30) -> np.ndarray:
        """Mask of image pixels where the ball can be: the column above the
        court. Each row allows the x-range between the side lines at that row,
        or at the net row for rows above it (a ball in the air is drawn above
        its floor point). Removes reflections in the side glass and the
        stands at the image edges."""
        net_y = self.to_image(COURT_WIDTH_M / 2, NET_Y_M)[1]
        mask = np.zeros((height, width), np.uint8)
        for y in range(height):
            yy = max(float(y), net_y)
            xl = self._side_x(0.0, yy)
            xr = self._side_x(COURT_WIDTH_M, yy)
            x1, x2 = max(0, int(xl - margin_px)), min(width, int(xr + margin_px))
            mask[y, x1:x2] = 1
        return mask

    def _side_x(self, x_m: float, y_img: float) -> float:
        """Image x of a side line (court x = x_m) at image row y_img."""
        (xa, ya), (xb, yb) = self.to_image(x_m, 0.0), self.to_image(x_m, COURT_LENGTH_M)
        return xa + (xb - xa) * (y_img - ya) / (yb - ya)

    def is_near_side(self, x: float, y: float) -> bool:
        """Whether a floor point is on the near (camera) half of the court."""
        return self.to_court(x, y)[1] < NET_Y_M

    def outline_px(self) -> list[np.ndarray]:
        """Court lines in image pixels for drawing: outline, net, service lines."""
        service = 6.95
        lines_m = [
            CORNERS_M.tolist() + [CORNERS_M[0].tolist()],
            [[0, NET_Y_M], [COURT_WIDTH_M, NET_Y_M]],
            [[0, NET_Y_M - service], [COURT_WIDTH_M, NET_Y_M - service]],
            [[0, NET_Y_M + service], [COURT_WIDTH_M, NET_Y_M + service]],
            [[COURT_WIDTH_M / 2, NET_Y_M - service], [COURT_WIDTH_M / 2, NET_Y_M + service]],
        ]
        return [
            np.array([self.to_image(x, y) for x, y in line], dtype=np.int32) for line in lines_m
        ]
