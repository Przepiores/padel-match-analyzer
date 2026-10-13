import numpy as np

from padel.ball.candidates import Candidate
from padel.ball.trajectory import build_trajectory


def arc(frames, x0=100.0, y0=300.0, vx=6.0, vy=-9.0, g=0.5):
    return {f: (x0 + vx * f, y0 + vy * f + 0.5 * g * f * f) for f in frames}


def test_follows_ball_and_ignores_static_and_random_noise():
    rng = np.random.default_rng(0)
    truth = arc(range(30))
    cands = {}
    for f, (x, y) in truth.items():
        noise = [Candidate(*rng.uniform(0, 640, 2), 5, contrast=60.0) for _ in range(3)]
        static = Candidate(400.0, 50.0, 5, contrast=60.0)  # a flickering spot
        cands[f] = [Candidate(x, y, 5, strong=True, contrast=60.0), static, *noise]
    traj = build_trajectory(cands)
    hits = [f for f, (x, y) in truth.items() if f in traj and abs(traj[f].x - x) < 1e-6]
    assert len(hits) >= 25
    assert all(abs(p.x - 400) > 1 or abs(p.y - 50) > 1 for p in traj.values())


def test_short_gaps_are_interpolated():
    truth = arc(range(30))
    cands = {
        f: [Candidate(x, y, 5, strong=True)] for f, (x, y) in truth.items() if f not in (12, 13)
    }
    traj = build_trajectory(cands)
    assert traj[12].interpolated and traj[13].interpolated
    assert abs(traj[12].x - truth[12][0]) < 1.0


def test_weak_low_contrast_track_is_rejected():
    truth = arc(range(30))
    cands = {f: [Candidate(x, y, 5, strong=False, contrast=10.0)] for f, (x, y) in truth.items()}
    assert build_trajectory(cands) == {}
