# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

M0 (skeleton, `padel info`) is done. `PLAN.md` (in Polish) holds the agreed design and milestones M0–M7. Build to it, and update it when a decision changes. The project owner writes in Polish. Code identifiers are in English; user-facing text (CLI output, docs) is in Polish.

## Commands

```
uv sync                              # install deps into .venv
uv run padel info <video>            # run the CLI
uv run pytest                        # all tests
uv run pytest tests/test_video.py::test_info_command   # single test
uv run ruff check . && uv run ruff format --check .    # lint + format check
```

Tests build tiny synthetic videos with `cv2.VideoWriter` in `tmp_path`; never commit real match footage. Heavy deps (ultralytics, torch) are added only in the milestone that needs them.

## What this is

An offline Python CLI (`padel`) that analyses a padel match video from one fixed camera (≥60 fps, on an NVIDIA GPU). It tracks the ball, counts hits per player and splits the match into rallies. For each rally it proposes a point winner with a confidence score. The user confirms or corrects proposals in a keyboard-driven OpenCV review window, and the score is computed only from confirmed points.

## Architecture

The pipeline runs in stages. Each stage writes its output into the per-video output directory, and a re-run skips any stage whose output already exists. Detection is slow, while the hit and rally heuristics get re-tuned often, so tuning should never force re-detection.

`calibration.json → players.jsonl → ball.csv → hits.jsonl → rallies.json → review.json → score / stats / annotated.mp4`

Constraints that span modules:

- **Floor-only homography.** The court homography maps image points to metres on the 10×20 m court. It is valid only for points on the floor (player feet, ball bounces), never for the ball in flight: it is 2D only, with no ball height.
- **Player identity follows the track.** Identity is assigned manually on the first frame and follows the track, not the court side, because teams swap sides.
- **Pluggable ball detector.** Ball detection sits behind a `BallDetector` interface: a fine-tuned YOLO11 first, TrackNetV3 as a possible later swap.
- **Alternating teams.** Padel teams must hit alternately. Hit detection uses this to reject wall bounces that were misread as hits.
- **Hit context is recorded.** Every hit stores its context: pose keypoints for ±15 frames, ball velocity before and after, and the player's court position. Hit-type classification (M7) can then be added without re-processing video.
- **Pure scoring engine.** The scoring engine is pure Python with no computer-vision dependencies. It uses advantage scoring, a tie-break at 6:6 and best of 3 sets. Golden point and super tie-break are config flags, off by default. The score is always recomputed from the list of confirmed points, never mutated incrementally.
- **Evaluation set.** A held-out set (ball labels, hit list and points from one match) backs `padel eval`. Never train on it.
- **Large files stay out of git.** Videos, labels and outputs live in `data/`, model weights in `models/`, and both are gitignored.
