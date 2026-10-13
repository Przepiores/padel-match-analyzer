import numpy as np

from padel.ball.candidates import RollingBackground, background_candidates

BLUE = (200, 90, 30)  # BGR court colour
BALL = (40, 230, 230)  # BGR yellow


def frame_with_ball(pos):
    f = np.zeros((120, 160, 3), np.uint8)
    f[:] = BLUE
    if pos is not None:
        x, y = pos
        f[y - 1 : y + 2, x - 1 : x + 2] = BALL
    return f


def test_moving_ball_found_static_yellow_ignored():
    bg = RollingBackground(half_window=12, step=3, refresh=1)
    frames = {}
    for i in range(30):
        f = frame_with_ball((20 + 4 * i, 60))
        f[10:13, 10:13] = BALL  # a static yellow logo
        frames[i] = f
        bg.push(i, f)
    k, i = 3, 15
    cands = background_candidates(frames[i - k], frames[i], frames[i + k], bg.background(i))
    assert len(cands) == 1
    assert abs(cands[0].x - (20 + 4 * i)) <= 1 and abs(cands[0].y - 60) <= 1
    assert cands[0].strong
