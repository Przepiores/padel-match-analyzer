"""Per-frame ball candidates from motion and colour.

A padel ball is a small yellow-green blob that moves. Three-frame differencing
keeps only pixels that changed both before and after the current frame, which
removes static yellow things (logos, clothes of standing people) and the
"ghost" a moving object leaves in a single difference image.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ColorRange:
    hue: tuple[int, int]  # OpenCV hue scale 0..180
    min_saturation: int
    min_value: int

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        h, s, v = cv2.split(hsv)
        return (
            (h >= self.hue[0])
            & (h <= self.hue[1])
            & (s >= self.min_saturation)
            & (v >= self.min_value)
        )


@dataclass(frozen=True)
class CandidateConfig:
    diff_threshold: int = 18
    # Measured on real footage: a sharp ball is hue 28-40, saturation 160+;
    # skin, caps and racket faces sit at hue 15-27 or saturation < 160.
    strong: ColorRange = ColorRange((27, 50), 140, 80)
    # A motion-blurred ball loses saturation; such blobs only extend
    # trajectories that strong candidates started.
    weak: ColorRange = ColorRange((22, 55), 70, 80)
    min_area: int = 2
    max_area: int = 220
    max_aspect: float = 4.0


@dataclass(frozen=True)
class Candidate:
    x: float
    y: float
    area: int
    strong: bool = True
    contrast: float = 0.0  # brightest core pixel minus surrounding ring (V channel)


def ball_candidates(
    prev: np.ndarray,
    cur: np.ndarray,
    nxt: np.ndarray,
    cfg: CandidateConfig = CandidateConfig(),
    ignore_mask: np.ndarray | None = None,
) -> list[Candidate]:
    """Find moving yellow blobs in `cur` (BGR frames of equal size)."""
    g_prev, g_cur, g_next = (cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in (prev, cur, nxt))
    moving = (cv2.absdiff(g_cur, g_prev) > cfg.diff_threshold) & (
        cv2.absdiff(g_next, g_cur) > cfg.diff_threshold
    )

    hsv = cv2.cvtColor(cur, cv2.COLOR_BGR2HSV)
    strong = moving & cfg.strong.mask(hsv)
    mask = (moving & cfg.weak.mask(hsv)).astype(np.uint8) * 255
    if ignore_mask is not None:
        mask[ignore_mask > 0] = 0
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, hgt, area = stats[i]
        if not cfg.min_area <= area <= cfg.max_area:
            continue
        if max(w, hgt) / max(1, min(w, hgt)) > cfg.max_aspect:
            continue
        cx, cy = centroids[i]
        is_strong = bool(strong[y : y + hgt, x : x + w][labels[y : y + hgt, x : x + w] == i].any())
        out.append(Candidate(float(cx), float(cy), int(area), is_strong))
    return out


@dataclass(frozen=True)
class BackgroundConfig:
    diff_threshold: int = 14  # vs frames t-k and t+k
    fg_threshold: int = 18  # vs the temporal-median background
    min_area: int = 2
    max_area: int = 70  # larger blobs are players, rackets, people outside
    max_aspect: float = 4.0
    strong: ColorRange = ColorRange((27, 50), 140, 80)
    # A pale, blurred ball over the beige back wall: yellowish and brighter
    # than what is behind it.
    weak: ColorRange = ColorRange((18, 55), 30, 100)
    min_brightening: int = 8


def background_candidates(
    prev: np.ndarray,
    cur: np.ndarray,
    nxt: np.ndarray,
    background: np.ndarray,
    cfg: BackgroundConfig = BackgroundConfig(),
) -> list[Candidate]:
    """Small blobs in `cur` that differ from frames t-k, t+k and from a
    temporal-median background.

    With k of a few frames a ball moving only 1-2 px per frame (flying
    towards the camera) still clears its own size between t-k and t, which
    consecutive-frame differencing misses. The background test removes
    compression flicker on static texture, which changes between frames but
    not against the median.
    """
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in (prev, cur, nxt)]
    moving = (cv2.absdiff(g[1], g[0]) > cfg.diff_threshold) & (
        cv2.absdiff(g[2], g[1]) > cfg.diff_threshold
    )
    fg = moving & (cv2.absdiff(cur, background).max(axis=2) > cfg.fg_threshold)
    hsv = cv2.cvtColor(cur, cv2.COLOR_BGR2HSV)
    strong = cfg.strong.mask(hsv)
    v_bg = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.int16)
    weak = cfg.weak.mask(hsv) & ((hsv[:, :, 2].astype(np.int16) - v_bg) >= cfg.min_brightening)

    mask = cv2.dilate(fg.astype(np.uint8), np.ones((3, 3), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not cfg.min_area <= area <= cfg.max_area:
            continue
        if max(w, h) / max(1, min(w, h)) > cfg.max_aspect:
            continue
        sel = labels[y : y + h, x : x + w] == i
        is_strong = bool(strong[y : y + h, x : x + w][sel].any())
        is_weak = bool(weak[y : y + h, x : x + w][sel].any())
        if is_strong or is_weak:
            cx, cy = centroids[i]
            out.append(
                Candidate(
                    float(cx), float(cy), int(area), is_strong, dot_contrast(hsv[:, :, 2], cx, cy)
                )
            )
    return out


_YY, _XX = np.mgrid[-8:9, -8:9]
_R = np.hypot(_XX, _YY)
_CORE, _RING = _R <= 1.5, (_R >= 4) & (_R <= 7)


def dot_contrast(v: np.ndarray, x: float, y: float) -> float:
    """How much a small bright dot stands out: brightest core pixel minus the
    median of a ring 4-7 px away. A pale ball over the back wall scores
    60-120, compression noise on the same wall mostly below 35."""
    xi, yi = round(x), round(y)
    h, w = v.shape
    if not (8 <= xi < w - 8 and 8 <= yi < h - 8):
        patch = np.pad(v, 8, mode="edge")[yi : yi + 17, xi : xi + 17]
    else:
        patch = v[yi - 8 : yi + 9, xi - 8 : xi + 9]
    return float(patch[_CORE].max()) - float(np.median(patch[_RING]))


class RollingBackground:
    """Temporal median of frames sampled around the current one.

    Feed frames in order with `push`; `background(i)` is available once frame
    i + half_window has been pushed. Medians are recomputed every `refresh`
    frames, which is plenty for a fixed camera.
    """

    def __init__(self, half_window: int = 30, step: int = 6, refresh: int = 6):
        self.half, self.step, self.refresh = half_window, step, refresh
        self.frames: dict[int, np.ndarray] = {}
        self._cache: tuple[int, np.ndarray] | None = None

    def push(self, index: int, frame: np.ndarray) -> None:
        self.frames[index] = frame
        for k in [k for k in self.frames if k < index - 2 * self.half]:
            del self.frames[k]

    def background(self, index: int) -> np.ndarray:
        anchor = index - index % self.refresh
        if self._cache is None or self._cache[0] != anchor:
            idx = [
                k
                for k in range(anchor - self.half, anchor + self.half + 1, self.step)
                if k in self.frames
            ]
            stack = np.stack([self.frames[k] for k in idx])
            self._cache = (anchor, np.median(stack, axis=0).astype(np.uint8))
        return self._cache[1]
