"""Compare detected hits with hand-labelled ones.

    uv run python scripts/eval_hits.py OUT_DIR/events.csv examples/yt_hits_gt.csv

The ground-truth file has `window,first,last` rows (ranges where labels are
complete) and `hit,frame,player` rows; lines starting with # are comments.
A detection matches a label within --tol frames; each is used once.

`check,frame,player` rows instead verify single detections (a hand-judged
sample of hits): the detection nearest to that frame should name that player.
"""

import argparse
import csv

p = argparse.ArgumentParser()
p.add_argument("events_csv")
p.add_argument("gt_csv")
p.add_argument("--tol", type=int, default=12)
a = p.parse_args()

windows, truth, checks = [], [], []
with open(a.gt_csv) as fh:
    for row in csv.reader(line for line in fh if line.strip() and not line.startswith("#")):
        if row[0] == "window":
            windows.append((int(row[1]), int(row[2])))
        elif row[0] == "hit":
            truth.append((int(row[1]), row[2]))
        elif row[0] == "check":
            checks.append((int(row[1]), row[2]))

if checks:
    with open(a.events_csv) as fh:
        hits = [(int(r["frame"]), r["player"]) for r in csv.DictReader(fh) if r["type"] == "hit"]
    ok = 0
    for f, who in checks:
        near = [h for h in hits if abs(h[0] - f) <= a.tol]
        got = min(near, key=lambda h: abs(h[0] - f))[1] if near else "(brak)"
        ok += got == who
        print(f"  {f:>6} {who:<12} -> {got:<12} {'OK' if got == who else 'ŹLE'}")
    print(f"\nPrzypisanie gracza poprawne: {ok}/{len(checks)} ({ok / len(checks):.0%})")
    raise SystemExit


def in_window(f: int) -> bool:
    return any(lo <= f <= hi for lo, hi in windows)


with open(a.events_csv) as fh:
    found = [
        (int(r["frame"]), r["player"])
        for r in csv.DictReader(fh)
        if r["type"] == "hit" and in_window(int(r["frame"]))
    ]

used: set[int] = set()
player_ok = team_ok = 0
rows = []
for f, who in truth:
    best = min(
        (i for i, (g, _) in enumerate(found) if i not in used and abs(g - f) <= a.tol),
        key=lambda i: abs(found[i][0] - f),
        default=None,
    )
    if best is None:
        rows.append(f"  {f:>6} {who:<12} -> pominięte")
        continue
    used.add(best)
    g, got = found[best]
    mark = "OK" if got == who else ("drużyna OK" if got[0] == who[0] else "ZŁY GRACZ")
    player_ok += got == who
    team_ok += got[0] == who[0]
    rows.append(f"  {f:>6} {who:<12} -> {g:>6} {got:<12} {mark}")
false = [found[i] for i in range(len(found)) if i not in used]
print("\n".join(rows))
for g, got in false:
    print(f"  fałszywe: {g} {got}")
n = len(truth)
print(f"\nUderzenia w prawdzie: {n}, wykryte w oknach: {len(found)}")
print(f"Wykryte, właściwy gracz:  {player_ok}/{n} ({player_ok / n:.0%})")
print(f"Wykryte, właściwa drużyna: {team_ok}/{n} ({team_ok / n:.0%})")
print(f"Fałszywe alarmy: {len(false)}")
