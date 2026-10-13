"""Hit detection from the ball trajectory and player boxes.

A hit is a sharp change of the ball's direction (or speed) while the ball is
within racket reach of a player. Direction changes away from players are
bounces (floor or glass). Padel teams must hit alternately, so two "hits" by
the same team in quick succession mean one was really a bounce or a serve
bounce; the later one is kept.
"""

from dataclasses import dataclass

import numpy as np

from padel.ball.trajectory import BallPoint
from padel.players.slots import BBox, Person


@dataclass(frozen=True)
class HitConfig:
    k: int = 3  # frames on each side used to measure direction
    min_angle_deg: float = 35.0
    min_speed_px: float = 2.0  # per frame; slower = ball (nearly) still
    speed_ratio: float = 2.2  # speed-up that counts even without turning
    nms_frames: int = 8
    # Racket reach around the body box, in player heights (an arm plus a
    # racket is about half a body), so tiny far players get a usable reach.
    reach_x: float = 0.55
    reach_top: float = 0.35
    rally_gap_s: float = 2.5  # no hit for this long: a new rally starts
    wrist_reach: float = 0.35  # ball this close to a wrist (in player heights)
    # Overriding the alternation (a hit was probably missed): the ball is this
    # close to a far player's body, and the near choice has no wrist this close.
    missed_hit_far_dist: float = 0.35
    missed_hit_wrist: float = 0.5
    same_team_window_s: float = 1.2
    max_boundary_gap: int = 45  # frames between trajectory pieces still joined
    min_boundary_angle_deg: float = 50.0


@dataclass(frozen=True)
class BallEvent:
    frame: int
    x: float
    y: float
    angle_deg: float
    player: int | None  # slot index for hits, None for bounces


def direction_changes(
    ball: dict[int, BallPoint], cfg: HitConfig = HitConfig()
) -> list[tuple[int, float]]:
    """Frames where the trajectory turns sharply, with the turn angle."""
    k = cfg.k
    raw = []
    for f, p in ball.items():
        a, b = ball.get(f - k), ball.get(f + k)
        if a is None or b is None:
            continue
        vin = np.array([p.x - a.x, p.y - a.y]) / k
        vout = np.array([b.x - p.x, b.y - p.y]) / k
        sin, sout = np.linalg.norm(vin), np.linalg.norm(vout)
        if max(sin, sout) < cfg.min_speed_px:
            continue
        if min(sin, sout) < 1e-6:
            angle = 180.0
        else:
            cos = float(np.dot(vin, vout) / (sin * sout))
            angle = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        # Only a sudden speed-up counts: a racket accelerates the ball, while
        # slowing down is gravity near the top of an arc or a wall.
        ratio = sout / max(sin, 1e-6)
        if is_free_flight(vin, vout):
            continue
        if angle >= cfg.min_angle_deg or ratio >= cfg.speed_ratio:
            raw.append((f, angle + (30.0 if ratio >= cfg.speed_ratio else 0.0)))

    # Non-maximum suppression: one event per turn.
    raw.sort(key=lambda e: -e[1])
    kept: list[tuple[int, float]] = []
    for f, score in raw:
        if all(abs(f - g) > cfg.nms_frames for g, _ in kept):
            kept.append((f, score))
    return sorted(kept)


def is_free_flight(vin: np.ndarray, vout: np.ndarray, tol: float = 1.5) -> bool:
    """Whether a change of velocity is explained by gravity alone.

    In free flight the sideways motion stays the same and the downward image
    velocity only grows (image y grows downwards): the top of an arc and a
    falling ball speeding up both look like sharp turns for a slow ball, but
    neither is a hit. A racket or a wall changes the sideways motion or sends
    a falling ball back up.
    """
    sideways_kept = vin[0] * vout[0] >= 0 and abs(vout[0] - vin[0]) < tol
    return bool(sideways_kept and vout[1] >= vin[1] - 0.5)


def boundary_changes(
    ball: dict[int, BallPoint], cfg: HitConfig = HitConfig()
) -> list[tuple[int, float, bool]]:
    """Events hidden in gaps of the trajectory: (frame, angle, needs_player).

    At contact the ball is often hidden by the player or the racket, so the
    trajectory breaks exactly where the hit is. Around each gap:
    - if the piece after it leaves in a clearly different direction, the turn
      happened in the gap (reported at the end of the first piece);
    - the ends of both pieces are hit candidates too, kept only if a player
      is within reach there (the ball vanished into, or came out of, a racket).
    """
    frames = sorted(ball)
    pieces: list[list[int]] = []
    for f in frames:
        if pieces and f - pieces[-1][-1] == 1:
            pieces[-1].append(f)
        else:
            pieces.append([f])
    out: list[tuple[int, float, bool]] = []
    k = cfg.k
    for a, b in zip(pieces, pieces[1:], strict=False):
        if b[0] - a[-1] > cfg.max_boundary_gap:
            continue
        out.append((a[-1], 0.0, True))
        out.append((b[0], 0.0, True))
        if len(a) <= k or len(b) <= k:
            continue
        pa0, pa1 = ball[a[-1 - k]], ball[a[-1]]
        pb0, pb1 = ball[b[0]], ball[b[k]]
        vin = np.array([pa1.x - pa0.x, pa1.y - pa0.y])
        vout = np.array([pb1.x - pb0.x, pb1.y - pb0.y])
        nin, nout = np.linalg.norm(vin), np.linalg.norm(vout)
        if min(nin, nout) < 1e-6:
            continue
        angle = float(np.degrees(np.arccos(np.clip(np.dot(vin, vout) / (nin * nout), -1, 1))))
        if angle >= cfg.min_boundary_angle_deg:
            out.append((a[-1], angle, False))
    return out


def reach_distance(x: float, y: float, b: BBox, cfg: HitConfig) -> float:
    """Distance from a point to the player's reach box (0 if inside), in units
    of the player's height so near and far players are treated alike."""
    h = b[3] - b[1]
    x1, x2 = b[0] - cfg.reach_x * h, b[2] + cfg.reach_x * h
    y1, y2 = b[1] - cfg.reach_top * h, b[3]
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return float(np.hypot(dx, dy) / max(h, 1))


def detect_events(
    ball: dict[int, BallPoint],
    players: dict[int, dict[int, Person]],
    teams: list[str],
    fps: float,
    cfg: HitConfig = HitConfig(),
) -> list[BallEvent]:
    """Classify direction changes into hits (with player) and bounces.

    `players` maps frame -> {slot index: person}; `teams[slot]` is its team.
    """
    candidates = [(f, angle, False) for f, angle in direction_changes(ball, cfg)]
    candidates += boundary_changes(ball, cfg)

    def people_at(f: int) -> dict[int, Person]:
        return players.get(f) or _nearest_frame(players, f)

    # One candidate per moment: those with a player in reach first, then the
    # sharper turn.
    candidates.sort(
        key=lambda c: (
            assign_player(ball[c[0]].x, ball[c[0]].y, people_at(c[0]), cfg) is None,
            -c[1],
        )
    )
    kept: list[tuple[int, float, bool]] = []
    for c in candidates:
        if all(abs(c[0] - k[0]) > cfg.nms_frames for k in kept):
            kept.append(c)
    kept.sort()

    # Assign in time order. Within a rally the teams must alternate, so the
    # team that hit last cannot hit next: this resolves the near/far
    # ambiguity of a 2D image where a near player's reach covers the far court.
    events: list[BallEvent] = []
    last_team, last_frame = None, -(10**9)
    for f, angle, needs_player in kept:
        p = ball[f]
        in_rally = f - last_frame <= cfg.rally_gap_s * fps
        forbid = last_team if in_rally else None
        player = assign_player(p.x, p.y, people_at(f), cfg, teams, forbid)
        if needs_player and player is None:
            continue
        events.append(BallEvent(f, p.x, p.y, angle, player))
        if player is not None:
            last_team, last_frame = teams[player], f
    return enforce_alternation(events, teams, fps, cfg)


def assign_player(
    x: float,
    y: float,
    people: dict[int, Person],
    cfg: HitConfig,
    teams: list[str] | None = None,
    forbid_team: str | None = None,
) -> int | None:
    """Player who hit a ball at image point (x, y), or None for a bounce.

    Players of `forbid_team` (the team that hit last in this rally) are
    excluded, unless the evidence says a hit was missed in between (see
    below). A ball right at a player's wrist is theirs. Otherwise, since in
    2D a big near player's reach box also covers the far court, a far player
    in reach wins over near ones. Among the rest, the closest wrist wins
    (partners at the net often both reach for it); without wrists, the
    closest reach box.
    """
    in_reach = [(s, p) for s, p in people.items() if reach_distance(x, y, p.bbox, cfg) <= 0.15]
    allowed = [
        (s, p)
        for s, p in in_reach
        if forbid_team is None or teams is None or teams[s] != forbid_team
    ]
    best = _best_player(x, y, allowed, cfg)
    if best is None or forbid_team is None:
        return best

    # A missed hit breaks the alternation. If the ball sits right next to a
    # far player of the "forbidden" team while the allowed choice is a near
    # player who only reaches it with the edge of a big box (no wrist close),
    # trust the image over the alternation.
    b = people[best]
    b_h = b.bbox[3] - b.bbox[1]
    b_d = body_distance(x, y, b.bbox)
    if _wrist_distance(x, y, b) <= cfg.missed_hit_wrist:
        return best
    for slot, p in in_reach:
        h = p.bbox[3] - p.bbox[1]
        d = body_distance(x, y, p.bbox)
        if (
            teams is not None
            and teams[slot] == forbid_team
            and b_h > 1.6 * h
            and d <= cfg.missed_hit_far_dist
            and b_d > d
        ):
            return slot
    return best


def _best_player(
    x: float, y: float, in_reach: list[tuple[int, Person]], cfg: HitConfig
) -> int | None:
    if not in_reach:
        return None
    heights = {slot: p.bbox[3] - p.bbox[1] for slot, p in in_reach}

    # A wrist right at the ball is the strongest evidence there is (pose is
    # found for near players only, e.g. an overhead whose racket overlaps a
    # far player in the image).
    at_wrist = [
        (_wrist_distance(x, y, p), s)
        for s, p in in_reach
        if _wrist_distance(x, y, p) <= cfg.wrist_reach
    ]
    if at_wrist:
        return min(at_wrist)[1]

    min_h = min(heights.values())
    far = [(s, p) for s, p in in_reach if heights[s] <= 1.6 * min_h]
    if len(far) < len(in_reach):
        in_reach = far

    def score(item: tuple[int, Person]) -> float:
        slot, p = item
        w = _wrist_distance(x, y, p)
        return w if w < float("inf") else 1.0 + reach_distance(x, y, p.bbox, cfg)

    return min(in_reach, key=score)[0]


def _wrist_distance(x: float, y: float, p: Person) -> float:
    """Distance from the ball to the closest wrist, in player heights."""
    if not p.wrists:
        return float("inf")
    h = max(p.bbox[3] - p.bbox[1], 1.0)
    return min(float(np.hypot(x - wx, y - wy)) for wx, wy in p.wrists) / h


def body_distance(x: float, y: float, b: BBox) -> float:
    """Distance from a point to a player's box (0 inside), in player heights."""
    h = b[3] - b[1]
    dx = max(b[0] - x, 0, x - b[2])
    dy = max(b[1] - y, 0, y - b[3])
    return float(np.hypot(dx, dy) / max(h, 1))


def enforce_alternation(
    events: list[BallEvent], teams: list[str], fps: float, cfg: HitConfig
) -> list[BallEvent]:
    out: list[BallEvent] = []
    last_hit_idx: int | None = None
    window = cfg.same_team_window_s * fps
    for e in events:
        if e.player is not None and last_hit_idx is not None:
            prev = out[last_hit_idx]
            same_team = teams[prev.player] == teams[e.player]
            if same_team and e.frame - prev.frame <= window:
                # Earlier one becomes a bounce; the later contact is the hit.
                out[last_hit_idx] = BallEvent(prev.frame, prev.x, prev.y, prev.angle_deg, None)
        out.append(e)
        if e.player is not None:
            last_hit_idx = len(out) - 1
    return out


def _nearest_frame(
    players: dict[int, dict[int, Person]], f: int, span: int = 3
) -> dict[int, Person]:
    for d in range(1, span + 1):
        for g in (f - d, f + d):
            if players.get(g):
                return players[g]
    return {}
