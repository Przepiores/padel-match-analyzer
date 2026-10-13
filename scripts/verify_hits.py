"""Contact sheet of detected hits for checking them by eye.

One tile per hit: the court area at the hit frame (and 4 frames later, side
by side), the ball circled and the assigned player written on top.

    uv run python scripts/verify_hits.py VIDEO OUT_DIR/events.csv sheet.jpg [--first 0 --n 8]
"""

import argparse
import csv

import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("video")
p.add_argument("events_csv")
p.add_argument("out")
p.add_argument("--first", type=int, default=0, help="index of the first hit to show")
p.add_argument("--n", type=int, default=8)
p.add_argument("--crop", default="40,100,600,340", help="x1,y1,x2,y2 of the court area")
a = p.parse_args()

with open(a.events_csv) as fh:
    hits = [r for r in csv.DictReader(fh) if r["type"] == "hit"][a.first : a.first + a.n]
x1, y1, x2, y2 = map(int, a.crop.split(","))
cap = cv2.VideoCapture(a.video)
tiles = []
for i, h in enumerate(hits, start=a.first):
    pair = []
    for d in (0, 4):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(h["frame"]) + d)
        ok, frame = cap.read()
        if d == 0:
            cv2.circle(frame, (round(float(h["x"])), round(float(h["y"]))), 7, (0, 0, 255), 1)
        pair.append(frame[y1:y2, x1:x2])
    t = np.hstack(pair)
    label = f"#{i} {h['frame']} {h['player']}"
    cv2.putText(t, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(t, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    tiles.append(t)
cv2.imwrite(a.out, np.vstack(tiles))
print(f"{len(hits)} uderzeń -> {a.out}")
