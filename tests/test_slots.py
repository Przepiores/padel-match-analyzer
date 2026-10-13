import numpy as np

from padel.players.slots import Person, PlayerSlot, PlayerSlots

WHITE = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
DARK = np.array([0.0, 0.0, 0.0, 1.0], np.float32)


def test_partners_keep_names_when_they_cross():
    left = PlayerSlot("B1", "B", (100, 100, 120, 140), 0, WHITE)
    right = PlayerSlot("B2", "B", (200, 100, 220, 140), 0, DARK)
    slots = PlayerSlots([left, right])
    # They meet in the middle and swap places; position alone would swap names.
    for f, (xw, xd) in enumerate([(140, 180), (155, 165), (165, 155), (180, 140)], start=1):
        dets = [
            Person((xd, 100, xd + 20, 140), hist=DARK),
            Person((xw, 100, xw + 20, 140), hist=WHITE),
        ]
        slots.update(f, dets)
    assert slots.slots[0].bbox[0] == 180  # white player, now on the right
    assert slots.slots[1].bbox[0] == 140


def test_allowed_forbids_pairs():
    s = PlayerSlots([PlayerSlot("A1", "A", (100, 100, 120, 140))])
    assert s.update(1, [Person((105, 100, 125, 140))], allowed=lambda slot, b: False) == {}
