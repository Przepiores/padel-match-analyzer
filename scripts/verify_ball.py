"""Contact sheet for checking ball tracking by eye.

Crops a window around the tracked ball position in randomly chosen frames
(upscaled, with a crosshair that leaves the centre free). If tracking is right,
the ball sits in the middle of every tile.

    uv run python scripts/verify_ball.py VIDEO OUT_DIR/ball.csv sheet.jpg [--n 48]
"""

import argparse
import csv
import random

import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("video")
p.add_argument("ball_csv")
p.add_argument("out")
p.add_argument("--n", type=int, default=48)
p.add_argument("--half", type=int, default=28)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--interpolated", action="store_true", help="sample interpolated points")
a = p.parse_args()

with open(a.ball_csv) as fh:
    rows = [r for r in csv.DictReader(fh) if int(r["interpolated"]) == a.interpolated]
random.Random(a.seed).shuffle(rows)
rows = sorted(rows[: a.n], key=lambda r: int(r["frame"]))
cap = cv2.VideoCapture(a.video)
tiles = []
h, z = a.half, 4
for r in rows:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["frame"]))
    ok, f = cap.read()
    x, y = round(float(r["x"])), round(float(r["y"]))
    pad = cv2.copyMakeBorder(f, h, h, h, h, cv2.BORDER_CONSTANT)
    crop = pad[y : y + 2 * h, x : x + 2 * h]
    t = cv2.resize(crop, (2 * h * z, 2 * h * z), interpolation=cv2.INTER_NEAREST)
    c = h * z
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(t, (c + dx * 14, c + dy * 14), (c + dx * 30, c + dy * 30), (0, 0, 255), 2)
    cv2.putText(t, r["frame"], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    tiles.append(t)
while len(tiles) % 8:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(a.out, np.vstack([np.hstack(tiles[i : i + 8]) for i in range(0, len(tiles), 8)]))
print(f"{len(rows)} kafelków -> {a.out}")
