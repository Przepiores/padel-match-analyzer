"""End-to-end prototype: ball tracking + hits per player on a video fragment.

This is the throwaway-friendly version of stages 2-4 of PLAN.md, run in one
go to see how far the approach gets on real footage. It needs the `vision`
extra (ultralytics).
"""

import csv
import json
import pickle
import shutil
import subprocess
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from padel.ball.candidates import (
    BackgroundConfig,
    Candidate,
    RollingBackground,
    background_candidates,
)
from padel.ball.trajectory import build_trajectory
from padel.court.geometry import Court
from padel.events.hits import BallEvent, HitConfig, detect_events
from padel.players.slots import BBox, Person, PlayerSlot, PlayerSlots, feet, torso_histogram
from padel.video import probe

PERSON = 0
K_DIFF = 3  # frames between the frames compared when looking for motion
TEAM_COLORS = {"A": (255, 160, 40), "B": (60, 60, 255)}  # BGR


@dataclass
class DemoConfig:
    start_s: float = 0.0
    duration_s: float = 120.0
    proc_width: int = 1280
    detect_every: int = 2
    model: str = "yolo11s.pt"
    pose_model: str = "yolo11s-pose.pt"
    imgsz: int = 960
    main_view_factor: float = 4.0
    reuse: bool = False  # reuse detections.pkl from a previous run
    video: bool = True  # render annotated.mp4


def load_model(name: str, models_dir: Path):
    from ultralytics import YOLO

    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / name
    if not path.exists():
        from ultralytics.utils.downloads import attempt_download_asset

        downloaded = Path(attempt_download_asset(name))
        shutil.move(str(downloaded), path)
    return YOLO(str(path))


def frame_signature(frame: np.ndarray) -> np.ndarray:
    """Coarse colour layout used to tell the main camera view from cut-aways."""
    small = cv2.resize(frame, (32, 18), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32).ravel()


def main_view_mask(signatures: np.ndarray, factor: float) -> np.ndarray:
    """Frames whose layout is close to the dominant one (the main camera);
    intros, replays and close-ups are far off it. Within the main view the
    distance comes only from moving players, so a multiple of its median
    separates the two cleanly."""
    ref = np.median(signatures, axis=0)
    dist = np.abs(signatures - ref).mean(axis=1) / 255.0
    # Re-centre on frames that already look like the main view.
    ref = np.median(signatures[dist <= np.median(dist)], axis=0)
    dist = np.abs(signatures - ref).mean(axis=1) / 255.0
    return dist < factor * np.median(dist)


def run_demo(video: Path, out: Path, calibration: Path, cfg: DemoConfig) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    info = probe(video)
    scale = min(1.0, cfg.proc_width / info.width)
    size = (round(info.width * scale), round(info.height * scale))
    court = Court.from_json(calibration, scale)
    first = round(cfg.start_s * info.fps)
    count = min(round(cfg.duration_s * info.fps), info.frame_count - first)

    # Detection is the slow part (minutes per minute of video on a CPU); the
    # rest is re-tuned often, so detections are cached in the output dir.
    cache = out / "detections.pkl"
    key = (str(video), first, count, size, calibration.read_text())
    det = None
    if cfg.reuse and cache.exists():
        with open(cache, "rb") as fh:
            det = pickle.load(fh)
        if det.get("key") != key:
            print("  detections.pkl dotyczy innego fragmentu, liczę od nowa", flush=True)
            det = None
    if det is None:
        det = detect(video, court, first, count, scale, size, cfg)
        det["key"] = key
        with open(cache, "wb") as fh:
            pickle.dump(det, fh)
    count, main, candidates, persons = det["count"], det["main"], det["candidates"], det["persons"]

    # Players --------------------------------------------------------------
    slots, players = track_players(persons, main, court)
    teams = [s.team for s in slots.slots]
    names = [s.name for s in slots.slots]

    # Ball + events --------------------------------------------------------
    ball = build_trajectory(candidates)
    ball = {f: p for f, p in ball.items() if main[f]}
    events = detect_events(ball, players, teams, info.fps, HitConfig())
    hits = [e for e in events if e.player is not None]

    hit_counts = Counter(names[e.player] for e in hits)
    stats = {
        "video": str(video),
        "start_s": cfg.start_s,
        "frames": count,
        "fps": info.fps,
        "main_view_frames": int(main[:count].sum()),
        "ball_frames_detected": sum(not p.interpolated for p in ball.values()),
        "ball_frames_interpolated": sum(p.interpolated for p in ball.values()),
        "hits_per_player": {n: hit_counts.get(n, 0) for n in names},
        "hits_per_team": dict(Counter(teams[e.player] for e in hits)),
        "bounces": len(events) - len(hits),
    }
    write_outputs(out, ball, events, names, info.fps, first, stats)
    write_players(out / "players.csv", players, names, first)

    # Pass 2: annotated video ---------------------------------------------
    if not cfg.video:
        return stats
    render(
        video,
        out,
        first,
        count,
        scale,
        size,
        info.fps,
        court,
        ball,
        players,
        events,
        names,
        teams,
        main,
    )
    return stats


def detect(
    video: Path,
    court: Court,
    first: int,
    count: int,
    scale: float,
    size: tuple[int, int],
    cfg: DemoConfig,
) -> dict:
    """People (every `detect_every` frames) and filtered ball candidates."""
    model = load_model(cfg.model, Path("models"))
    pose_model = load_model(cfg.pose_model, Path("models"))
    bg_cfg = BackgroundConfig()
    bg = RollingBackground()
    # Pass 1: detections ---------------------------------------------------
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    candidates: dict[int, list[Candidate]] = {}
    persons: dict[int, list[Person]] = {}
    signatures = []
    for i in range(count):
        ok, frame = cap.read()
        if not ok:
            count = i
            break
        if scale < 1:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        signatures.append(frame_signature(frame))
        bg.push(i, frame)

        if i % cfg.detect_every == 0:
            persons[i] = detect_people(model, pose_model, frame, court, cfg.imgsz)

        # The background for frame j needs frames up to j + half window.
        j = i - bg.half
        if j >= K_DIFF:
            candidates.setdefault(j, []).extend(
                background_candidates(
                    bg.frames[j - K_DIFF],
                    bg.frames[j],
                    bg.frames[j + K_DIFF],
                    bg.background(j),
                    bg_cfg,
                )
            )
        if i % 300 == 0:
            print(f"  analiza: klatka {i}/{count}", flush=True)
    cap.release()
    for j in range(max(K_DIFF, count - bg.half), count - K_DIFF):
        candidates.setdefault(j, []).extend(
            background_candidates(
                bg.frames[j - K_DIFF],
                bg.frames[j],
                bg.frames[j + K_DIFF],
                bg.background(j),
                bg_cfg,
            )
        )

    main = main_view_mask(np.array(signatures), cfg.main_view_factor)
    region = court.ball_region(*size)
    hot = hot_spots(candidates, size)
    candidates = {
        f: drop_on_people(
            [c for c in cs if region[int(c.y), int(c.x)] and not hot[int(c.y), int(c.x)]],
            [p.bbox for p in persons.get(f) or persons.get(f - 1) or []],
        )
        for f, cs in candidates.items()
        if f < count and main[f]
    }
    return {"count": count, "main": main, "candidates": candidates, "persons": persons}


def detect_people(model, pose_model, frame, court: Court, imgsz: int) -> list[Person]:
    """People standing on the court, with wrists where the pose model sees them.

    The plain detector finds far players (~30 px tall) that the pose model
    misses, so boxes come from the detector and wrists are attached from the
    best-overlapping pose box.
    """
    det = model.predict(frame, imgsz=imgsz, conf=0.3, classes=[PERSON], verbose=False)[0]
    boxes = [
        tuple(map(float, b))
        for b in det.boxes.xyxy.tolist()
        if court.contains(*feet(b), margin_m=1.5)
    ]
    pose = pose_model.predict(frame, imgsz=imgsz, conf=0.3, verbose=False)[0]
    poses = []
    for b, kp in zip(pose.boxes.xyxy.tolist(), pose.keypoints.data.tolist(), strict=True):
        wrists = tuple((k[0], k[1]) for k in (kp[9], kp[10]) if k[2] >= 0.5)
        poses.append((b, wrists))
    people = []
    for b in boxes:
        best = max(poses, key=lambda pw: iou(b, pw[0]), default=None)
        wrists = best[1] if best is not None and iou(b, best[0]) >= 0.5 else ()
        people.append(Person(b, wrists, torso_histogram(frame, b)))
    return people


def iou(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def hot_spots(
    candidates: dict[int, list[Candidate]],
    size: tuple[int, int],
    cell: int = 4,
    max_share: float = 0.05,
) -> np.ndarray:
    """Pixels where candidates keep appearing (compression flicker on static
    texture, a flag in the stands). The ball rarely passes the same spot in
    more than a few percent of frames."""
    w, h = size
    counts = np.zeros((h // cell + 1, w // cell + 1), np.float32)
    for cs in candidates.values():
        for r, q in {(int(c.y) // cell, int(c.x) // cell) for c in cs}:
            counts[r, q] += 1
    counts = cv2.blur(counts, (3, 3)) * 9
    hot = counts >= max_share * max(1, len(candidates))
    return cv2.resize(hot.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)


def drop_on_people(cands: list[Candidate], people: list[BBox]) -> list[Candidate]:
    """Drop candidates that are really parts of players.

    Bright yellow-green shoes move like a ball, so anything on the feet goes
    (a ball at the feet is rarely a hit). Far players are only ~30 px tall and
    their arms or shirts pass the weak colour test, so weak candidates inside a
    player box go too; strong (saturated ball-yellow) ones are kept.
    """
    out = []
    for c in cands:
        keep = True
        for x1, y1, x2, y2 in people:
            w, h = x2 - x1, y2 - y1
            on_feet = (
                x1 - 0.15 * w <= c.x <= x2 + 0.15 * w and y2 - 0.15 * h <= c.y <= y2 + 0.05 * h
            )
            inside = x1 <= c.x <= x2 and y1 <= c.y <= y2
            if on_feet or (inside and not c.strong):
                keep = False
                break
        if keep:
            out.append(c)
    return out


def track_players(
    persons: dict[int, list[Person]], main: np.ndarray, court: Court
) -> tuple[PlayerSlots, dict[int, dict[int, Person]]]:
    """Seed four identities on the first main-view frame with 2+2 players
    (near team A, far team B, named by left/right) and follow them.

    A pair only takes people on its own side of the net. In a highlights
    clip that mixes games (players change ends, a player steps out of frame)
    an unconstrained matcher hands a near identity to a far player; with the
    constraint, team A always means "the pair nearer the camera".
    """
    seed = None
    for f in sorted(persons):
        if not main[f]:
            continue
        boxes = [p.bbox for p in persons[f]]
        near = [d for d in boxes if court.is_near_side(*feet(d))]
        far = [d for d in boxes if not court.is_near_side(*feet(d))]
        if len(near) == 2 and len(far) == 2:
            seed = (
                f,
                [p for p in persons[f] if p.bbox in near],
                [p for p in persons[f] if p.bbox in far],
            )
            break
    if seed is None:
        raise RuntimeError("Nie znaleziono klatki z 2 graczami po każdej stronie siatki.")
    f0, near, far = seed
    slots = []
    for team, pair in (("A", near), ("B", far)):
        left, right = sorted(pair, key=lambda p: (p.bbox[0] + p.bbox[2]) / 2)
        slots.append(PlayerSlot(f"{team}1 (lewy)", team, left.bbox, f0, left.hist))
        slots.append(PlayerSlot(f"{team}2 (prawy)", team, right.bbox, f0, right.hist))
    tracker = PlayerSlots(slots)

    def same_side(slot: PlayerSlot, bbox: BBox) -> bool:
        return court.is_near_side(*feet(bbox)) == (slot.team == "A")

    # A player missed by the detector for a moment (racket swing, overlap)
    # keeps the last box for up to `hold` frames.
    hold = 15
    players: dict[int, dict[int, Person]] = {}
    last: dict[int, tuple[int, Person]] = {}
    for f in range(f0, len(main)):
        if not main[f]:
            last = {}
            continue
        if f in persons:
            dets = persons[f]
            for slot, j in tracker.update(f, dets, same_side).items():
                last[slot] = (f, dets[j])
        players[f] = {slot: p for slot, (seen, p) in last.items() if f - seen <= hold}
    return tracker, players


def write_outputs(out, ball, events, names, fps, first, stats):
    with open(out / "ball.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "time_s", "x", "y", "interpolated"])
        for f, p in ball.items():
            w.writerow(
                [
                    first + f,
                    f"{(first + f) / fps:.3f}",
                    f"{p.x:.1f}",
                    f"{p.y:.1f}",
                    int(p.interpolated),
                ]
            )
    with open(out / "events.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "time_s", "type", "player", "x", "y", "angle_deg"])
        for e in events:
            w.writerow(
                [
                    first + e.frame,
                    f"{(first + e.frame) / fps:.3f}",
                    "hit" if e.player is not None else "bounce",
                    names[e.player] if e.player is not None else "",
                    f"{e.x:.1f}",
                    f"{e.y:.1f}",
                    f"{e.angle_deg:.0f}",
                ]
            )
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))


def write_players(path: Path, players: dict[int, dict[int, Person]], names, first) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "player", "x1", "y1", "x2", "y2", "wrists"])
        for f, people in players.items():
            for slot, p in people.items():
                wrists = ";".join(f"{x:.0f}:{y:.0f}" for x, y in p.wrists)
                w.writerow([first + f, names[slot], *(f"{v:.0f}" for v in p.bbox), wrists])


def render(
    video, out, first, count, scale, size, fps, court, ball, players, events, names, teams, main
):
    # Small sources are drawn at 2x so labels stay readable.
    k = 2 if size[0] < 960 else 1
    out_size = (size[0] * k, size[1] * k)

    def pt(x: float, y: float) -> tuple[int, int]:
        return round(x * k), round(y * k)

    raw = out / "annotated_raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    hits_so_far = [0] * len(names)
    events_by_frame = {e.frame: e for e in events}
    flash: list[tuple[int, BallEvent]] = []
    trail: deque[tuple[int, int]] = deque(maxlen=14)
    for f in range(count):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_CUBIC)
        if main[f]:
            for line in court.outline_px():
                cv2.polylines(frame, [line * k], False, (255, 255, 255), 1, cv2.LINE_AA)
        if (e := events_by_frame.get(f)) is not None:
            flash.append((f, e))
            if e.player is not None:
                hits_so_far[e.player] += 1
        for slot, person in players.get(f, {}).items():
            color = TEAM_COLORS[teams[slot]]
            x1, y1, x2, y2 = person.bbox
            cv2.rectangle(frame, pt(x1, y1), pt(x2, y2), color, 2)
            label_at = (pt(x1, y1)[0], max(16, pt(x1, y1)[1] - 8))
            text(frame, f"{names[slot]}: {hits_so_far[slot]}", label_at, color, 0.6)
        p = ball.get(f)
        if p is not None:
            trail.append(pt(p.x, p.y))
            for a, b in zip(trail, list(trail)[1:], strict=False):
                cv2.line(frame, a, b, (0, 255, 255), 2, cv2.LINE_AA)
            color = (0, 165, 255) if p.interpolated else (0, 255, 0)
            cv2.circle(frame, pt(p.x, p.y), 7 * k, color, 2, cv2.LINE_AA)
        else:
            trail.clear()
        flash = [(g, e) for g, e in flash if f - g < round(fps * 0.6)]
        for _, e in flash:
            if e.player is not None:
                cv2.circle(frame, pt(e.x, e.y), 16 * k, (0, 0, 255), 3, cv2.LINE_AA)
                x, y = pt(e.x, e.y)
                text(frame, f"HIT {names[e.player]}", (x + 18 * k, y), (0, 0, 255), 0.8)
            else:
                cv2.circle(frame, pt(e.x, e.y), 10 * k, (255, 255, 255), 2, cv2.LINE_AA)
        draw_panel(frame, names, teams, hits_so_far, (first + f) / fps)
        writer.write(frame)
    cap.release()
    writer.release()
    to_h264(raw, out / "annotated.mp4")


def text(frame, label: str, at: tuple[int, int], color, size: float) -> None:
    """Text with a dark outline, readable on any background."""
    for c, w in (((0, 0, 0), 4), (color, 1 if size < 0.7 else 2)):
        cv2.putText(frame, label, at, cv2.FONT_HERSHEY_SIMPLEX, size, c, w, cv2.LINE_AA)


def draw_panel(frame, names, teams, counts, t):
    """Hit counters in the top-right corner (above the back wall, no play there)."""
    w, line = 300, 26
    x0, y0 = frame.shape[1] - w - 10, 10
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + 40 + line * len(names)), (0, 0, 0), -1)
    text(frame, f"Uderzenia    t={t:6.1f} s", (x0 + 10, y0 + 24), (255, 255, 255), 0.6)
    for i, (n, c) in enumerate(zip(names, counts, strict=True)):
        text(frame, f"{n:<12} {c:3d}", (x0 + 10, y0 + 52 + line * i), TEAM_COLORS[teams[i]], 0.6)


def to_h264(src: Path, dst: Path) -> None:
    """Re-encode to H.264 so the video plays in browsers; keep mp4v otherwise."""
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        src.rename(dst)
        return
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            "-preset",
            "veryfast",
            str(dst),
        ],
        check=True,
    )
    src.unlink()
