from padel.ball.trajectory import BallPoint
from padel.events.hits import HitConfig, assign_player, detect_events, is_free_flight
from padel.players.slots import Person

FPS = 30.0
TEAMS = ["A", "A", "B", "B"]


def line(f0, f1, start, v):
    return {
        f: BallPoint(start[0] + v[0] * (f - f0), start[1] + v[1] * (f - f0), False)
        for f in range(f0, f1)
    }


def players_const(people, frames):
    return {f: dict(people) for f in frames}


def test_reversal_next_to_player_is_a_hit_for_that_player():
    # Ball comes towards the near-right player and goes back up.
    ball = line(0, 20, (300, 100), (5, 8)) | line(20, 40, (400, 260), (-4, -9))
    near_right = Person((380, 200, 440, 320), wrists=((395, 255),))
    near_left = Person((100, 200, 160, 320))
    people = {0: near_left, 1: near_right}
    events = detect_events(ball, players_const(people, range(40)), TEAMS, FPS)
    hits = [e for e in events if e.player is not None]
    assert [e.player for e in hits] == [1]
    assert abs(hits[0].frame - 20) <= 3


def test_free_flight_arc_is_not_an_event():
    ball = {f: BallPoint(100 + 2 * f, 200 - 6 * f + 0.4 * f * f, False) for f in range(40)}
    assert detect_events(ball, {}, TEAMS, FPS) == []


def test_is_free_flight():
    assert is_free_flight([2.0, -3.0], [2.0, 1.0])  # top of an arc
    assert not is_free_flight([2.0, 4.0], [2.0, -4.0])  # bounce: falling turns into rising
    assert not is_free_flight([3.0, 1.0], [-3.0, 1.0])  # sideways reversal


def test_far_player_in_reach_beats_big_near_player():
    near = Person((200, 150, 300, 400))  # big, its reach covers the far court
    far = Person((260, 100, 275, 140))
    assert assign_player(268, 110, {0: near, 2: far}, HitConfig()) == 2


def test_closest_wrist_decides_between_partners():
    left = Person((150, 180, 250, 280), wrists=((237, 233),))
    right = Person((317, 183, 404, 269), wrists=((322, 207),))
    assert assign_player(291, 185, {0: left, 1: right}, HitConfig()) == 1


def test_team_that_hit_last_cannot_hit_again():
    left = Person((150, 180, 250, 280))
    right = Person((260, 180, 360, 280))
    people = {0: left, 1: right}
    assert assign_player(255, 220, people, HitConfig(), TEAMS, forbid_team="A") is None


def test_wrist_at_ball_beats_far_player_rule():
    # Near player's overhead: the racket hand is up where a far player stands.
    near = Person((200, 150, 300, 400), wrists=((268, 112),))
    far = Person((260, 100, 275, 140))
    assert assign_player(268, 110, {0: near, 2: far}, HitConfig()) == 0


def test_missed_hit_lets_far_player_hit_again():
    # Last hit was team B, but the ball is right at a far B player and the
    # near A player only reaches it with the edge of a big box: a hit by A
    # was missed, so B takes it.
    near = Person((440, 172, 498, 360), wrists=((451, 262),))
    far = Person((345, 143, 361, 189))
    people = {1: near, 3: far}
    assert assign_player(337, 161, people, HitConfig(), TEAMS, forbid_team="B") == 3


def test_alternation_holds_when_near_player_is_close():
    near = Person((356, 179, 429, 270), wrists=((424, 201),))
    far = Person((356, 144, 379, 189))
    people = {1: near, 3: far}
    assert assign_player(385, 177, people, HitConfig(), TEAMS, forbid_team="B") == 1
