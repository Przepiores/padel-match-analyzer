"""Sheet of frames where the tracker has NO ball, to check recall by eye.

Shows the court area of randomly chosen frames without a ball position
(upscaled 2x). If the ball is visible in many of them, recall is low.

    uv run python scripts/verify_missing.py VIDEO OUT_DIR/ball.csv START END sheet.jpg
"""

import argparse
import csv
import random

import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("video")
p.add_argument("ball_csv")
p.add_argument("start", type=int, help="first frame of the analysed range")
p.add_argument("end", type=int)
p.add_argument("out")
p.add_argument("--n", type=int, default=8)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--crop", default="60,40,580,330", help="x1,y1,x2,y2 of the court area")
a = p.parse_args()

with open(a.ball_csv) as fh:
    have = {int(r["frame"]) for r in csv.DictReader(fh)}
missing = [f for f in range(a.start, a.end) if f not in have]
random.Random(a.seed).shuffle(missing)
x1, y1, x2, y2 = map(int, a.crop.split(","))
cap = cv2.VideoCapture(a.video)
tiles = []
for f in sorted(missing[: a.n]):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, frame = cap.read()
    size = ((x2 - x1) * 2, (y2 - y1) * 2)
    t = cv2.resize(frame[y1:y2, x1:x2], size, interpolation=cv2.INTER_CUBIC)
    cv2.putText(t, str(f), (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    tiles.append(t)
while len(tiles) % 2:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(a.out, np.vstack([np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]))
print(f"{len(missing)} klatek bez piłki, pokazano {min(a.n, len(missing))}")
