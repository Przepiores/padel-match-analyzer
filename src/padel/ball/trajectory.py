"""Turn noisy per-frame ball candidates into one ball trajectory.

1. Link candidates into tracklets: seed on triplets with consistent motion,
   extend with a constant-velocity prediction.
2. Keep tracklets that are long enough.
3. Pick a set of non-overlapping tracklets covering the most frames
   (weighted interval scheduling): there is only one ball.
4. Fill short gaps by linear interpolation.
"""

from dataclasses import dataclass, field

import numpy as np

from padel.ball.candidates import Candidate


@dataclass(frozen=True)
class TrajectoryConfig:
    max_speed_px: float = 28.0  # per frame; faster jumps are not the ball
    min_speed_px: float = 0.7  # slower seeds are static noise, not a ball
    max_step: int = 2  # frames a seed may skip between its points
    seed_residual_px: float = 3.0  # triplet must be nearly constant-velocity
    extend_gate_px: float = 4.0  # + extend_gate_rel * speed
    extend_gate_rel: float = 0.35
    max_gap: int = 4  # frames a tracklet may skip while being extended
    min_len: int = 6  # detections needed for a tracklet to count
    # A tracklet counts if it has strong (sharp, saturated) detections or if
    # its points are clear bright dots (pale ball over the back wall).
    min_strong: int = 2
    min_median_contrast: float = 35.0
    max_interp_gap: int = 10  # frames of gap filled by interpolation


@dataclass
class Tracklet:
    points: dict[int, tuple[float, float]] = field(default_factory=dict)
    strong: int = 0

    @property
    def first(self) -> int:
        return min(self.points)

    @property
    def last(self) -> int:
        return max(self.points)

    def add(self, frame: int, x: float, y: float) -> None:
        self.points[frame] = (x, y)


@dataclass(frozen=True)
class BallPoint:
    x: float
    y: float
    interpolated: bool


def link_tracklets(
    candidates: dict[int, list[Candidate]], cfg: TrajectoryConfig = TrajectoryConfig()
) -> list[Tracklet]:
    """Seed-and-extend tracking.

    Seeds are triplets of candidates in nearby frames that move at a plausible,
    nearly constant velocity. Each seed is grown forwards and backwards by
    predicting the next position from the current velocity. Candidates used by
    an accepted tracklet are not reused, and the longest tracklets win.
    """
    frames = sorted(candidates)
    used: set[tuple[int, int]] = set()
    seeds = []
    for fa in frames:
        for ia, a in enumerate(candidates[fa]):
            for d1 in range(1, cfg.max_step + 1):
                for ib, b in enumerate(candidates.get(fa + d1, [])):
                    v1 = ((b.x - a.x) / d1, (b.y - a.y) / d1)
                    sp = np.hypot(*v1)
                    if not cfg.min_speed_px <= sp <= cfg.max_speed_px:
                        continue
                    for d2 in range(1, cfg.max_step + 1):
                        fc = fa + d1 + d2
                        for ic, c in enumerate(candidates.get(fc, [])):
                            px, py = b.x + v1[0] * d2, b.y + v1[1] * d2
                            res = np.hypot(c.x - px, c.y - py)
                            if res <= cfg.seed_residual_px + 0.15 * sp * d2:
                                seeds.append((res, [(fa, ia), (fa + d1, ib), (fc, ic)]))
    seeds.sort(key=lambda s: s[0])

    tracklets = []
    for _, seed in seeds:
        if any(k in used for k in seed):
            continue
        members = _extend(seed, candidates, used, cfg)
        if len(members) < cfg.min_len:
            continue
        t = Tracklet()
        for f, i in sorted(members):
            c = candidates[f][i]
            t.add(f, c.x, c.y)
            t.strong += c.strong
        contrast = float(np.median([candidates[f][i].contrast for f, i in members]))
        if t.strong < cfg.min_strong and contrast < cfg.min_median_contrast:
            continue
        used.update(members)
        tracklets.append(t)
    return tracklets


def _extend(seed, candidates, used, cfg: TrajectoryConfig) -> list[tuple[int, int]]:
    members = list(seed)
    for direction in (1, -1):
        while True:
            chain = members if direction == 1 else members[::-1]
            (f1, i1), (f2, i2) = chain[-2], chain[-1]
            p1, p2 = candidates[f1][i1], candidates[f2][i2]
            dt = f2 - f1
            v = ((p2.x - p1.x) / dt, (p2.y - p1.y) / dt)
            sp = np.hypot(*v)
            best = None
            for gap in range(1, cfg.max_gap + 1):
                f = f2 + direction * gap
                px, py = p2.x + v[0] * gap * direction, p2.y + v[1] * gap * direction
                gate = (cfg.extend_gate_px + cfg.extend_gate_rel * sp) * gap
                for i, c in enumerate(candidates.get(f, [])):
                    if (f, i) in used:
                        continue
                    d = np.hypot(c.x - px, c.y - py)
                    if d <= gate and (best is None or d < best[0]):
                        best = (d, (f, i))
                if best is not None:
                    break
            if best is None:
                break
            if direction == 1:
                members.append(best[1])
            else:
                members.insert(0, best[1])
    return members


def select_tracklets(tracklets: list[Tracklet]) -> list[Tracklet]:
    """Non-overlapping subset maximising the number of covered detections."""
    ts = sorted(tracklets, key=lambda t: t.last)
    ends = [t.last for t in ts]
    best = [0] * (len(ts) + 1)
    take = [False] * len(ts)
    prev_idx = []
    for i, t in enumerate(ts):
        # Last tracklet ending strictly before this one starts.
        p = int(np.searchsorted(ends, t.first, side="left"))
        prev_idx.append(p)
        with_t = best[p] + len(t.points)
        if with_t > best[i]:
            best[i + 1] = with_t
            take[i] = True
        else:
            best[i + 1] = best[i]
    chosen = []
    i = len(ts)
    while i > 0:
        if take[i - 1] and best[i] == best[prev_idx[i - 1]] + len(ts[i - 1].points):
            chosen.append(ts[i - 1])
            i = prev_idx[i - 1]
        else:
            i -= 1
    return sorted(chosen, key=lambda t: t.first)


def build_trajectory(
    candidates: dict[int, list[Candidate]], cfg: TrajectoryConfig = TrajectoryConfig()
) -> dict[int, BallPoint]:
    chosen = select_tracklets(link_tracklets(candidates, cfg))
    points: dict[int, BallPoint] = {}
    for t in chosen:
        for f, (x, y) in t.points.items():
            points[f] = BallPoint(x, y, interpolated=False)

    frames = sorted(points)
    for a, b in zip(frames, frames[1:], strict=False):
        gap = b - a
        if 1 < gap <= cfg.max_interp_gap:
            pa, pb = points[a], points[b]
            for k in range(1, gap):
                r = k / gap
                points[a + k] = BallPoint(
                    pa.x + r * (pb.x - pa.x), pa.y + r * (pb.y - pa.y), interpolated=True
                )
    return dict(sorted(points.items()))
