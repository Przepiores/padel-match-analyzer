# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

M0 (skeleton, `padel info`) is done. `padel demo` is an end-to-end prototype of stages 2–4 (players, ball, hits) used to validate the approach on real footage before the staged pipeline is built. `PLAN.md` (in Polish) holds the agreed design and milestones M0–M7. Build to it, and update it when a decision changes. The project owner writes in Polish. Code identifiers are in English; user-facing text (CLI output, docs) is in Polish.

## Commands

```
uv sync                              # install core deps into .venv
uv sync --extra vision               # + ultralytics/torch, needed by `padel demo`
uv run padel info <video>            # run the CLI
uv run padel demo <video> --calib <calibration.json> --start 30 --duration 120 -o data/demo
uv run pytest                        # all tests
uv run pytest tests/test_video.py::test_info_command   # single test
uv run ruff check . && uv run ruff format --check .    # lint + format check
```

Tests build tiny synthetic videos and trajectories; never commit real match footage. Heavy deps (ultralytics, torch) live in the `vision` extra and are imported lazily inside `padel.demo`, so the core and the tests run without them. YOLO weights download into `models/` on first use.

`padel demo` caches the slow detection pass in `<out>/detections.pkl`; `--reuse --no-video` re-runs players, trajectory and hits from it in about a second, so iterate on `ball/trajectory.py` / `events/hits.py` that way. Changes to `ball/candidates.py` or `detect_people` need a full run (~12 min for 2 min of 360p video on CPU).

Evaluation tools (the demo clip itself is not in git: https://www.youtube.com/watch?v=Oq2YPJNNEDk, which the user downloaded and uploaded to `data/yt_clip.mp4` because YouTube is blocked in the cloud environment; court points are in `examples/yt_calibration.json`):
- `scripts/eval_hits.py <out>/events.csv examples/yt_hits_gt.csv`: hits against two hand-labelled rallies. These rallies were also used for tuning, so treat the score as optimistic; `examples/yt_hits_sample.csv` (`check` rows) is a hand-judged random sample of detections.
- `scripts/verify_ball.py`, `scripts/verify_missing.py` and `scripts/verify_hits.py` build contact sheets for checking by eye: ball precision, ball recall and hit attribution. Use them after any change: aggregate counts alone hid most regressions while tuning.

## What this is

An offline Python CLI (`padel`) that analyses a padel match video from one fixed camera (≥60 fps, on an NVIDIA GPU). It tracks the ball, counts hits per player and splits the match into rallies. For each rally it proposes a point winner with a confidence score. The user confirms or corrects proposals in a keyboard-driven OpenCV review window, and the score is computed only from confirmed points.

## Architecture

The pipeline runs in stages. Each stage writes its output into the per-video output directory, and a re-run skips any stage whose output already exists. Detection is slow, while the hit and rally heuristics get re-tuned often, so tuning should never force re-detection.

`calibration.json → players.jsonl → ball.csv → hits.jsonl → rallies.json → review.json → score / stats / annotated.mp4`

Constraints that span modules:

- **Floor-only homography.** The court homography maps image points to metres on the 10×20 m court. It is valid only for points on the floor (player feet, ball bounces), never for the ball in flight: it is 2D only, with no ball height.
- **Player identity follows the track.** Identity is assigned manually on the first frame and follows the track, not the court side, because teams swap sides.
- **Pluggable ball detector.** The plan puts ball detection behind a `BallDetector` interface (fine-tuned YOLO11, TrackNetV3 later). The prototype uses classical detection instead (`ball/candidates.py`): frame t is compared with t±3 and with a temporal-median background, colour is checked in two tiers (strong: saturated ball-yellow; weak: pale but brighter than the background), and tracking (`ball/trajectory.py`) seeds on triplets with consistent motion and extends them. Weak-only tracklets must be clear bright dots (`dot_contrast`). Stock COCO "sports ball" detection was tried and finds mostly balls lying on the floor.
- **Alternating teams resolve depth.** In a single 2D view a big near player's reach box covers the far court, so geometry alone often assigns a far hit to a near player. `events/hits.py` assigns hits in time order and forbids the team that hit last within a rally, prefers a far player when near and far are both in reach, and picks between partners by the wrist (YOLO11-pose) closest to the ball. Hits are also searched at the ends of trajectory pieces, because the ball usually disappears at contact. Direction changes explained by gravity alone (`is_free_flight`) are ignored.
- **Hit context is recorded.** Every hit stores its context: pose keypoints for ±15 frames, ball velocity before and after, and the player's court position. Hit-type classification (M7) can then be added without re-processing video.
- **Pure scoring engine.** The scoring engine is pure Python with no computer-vision dependencies. It uses advantage scoring, a tie-break at 6:6 and best of 3 sets. Golden point and super tie-break are config flags, off by default. The score is always recomputed from the list of confirmed points, never mutated incrementally.
- **Evaluation set.** A held-out set (ball labels, hit list and points from one match) backs `padel eval`. Never train on it.
- **Large files stay out of git.** Videos, labels and outputs live in `data/`, model weights in `models/`, and both are gitignored.
