"""Rule checks on synthetic people, using the real thresholds from configs/config.yaml."""
import copy

import numpy as np

from src.config import load_config
from src.rules import Aid, Person, RuleEngine

FPS = 10
BASE = load_config()


def engine(zones=None, **overrides):
    cfg = copy.deepcopy(BASE)
    cfg["zones"] = zones or []
    for key, val in overrides.items():
        section, name = key.split("__")
        cfg["rules"][section][name] = val
    return RuleEngine(cfg)


def person(tid, cx, top, height=200, lying=False, hand_up=False):
    """Synthetic person with nose, shoulders, wrists, hips and ankles."""
    k, c = np.zeros((17, 2)), np.zeros(17)
    if lying:
        y = top + 20
        k[0] = [cx - 80, y]
        k[[5, 6]] = [cx - 60, y]
        k[[9, 10]] = [cx - 40, y + 5]
        k[[11, 12]] = [cx, y]
        k[[15, 16]] = [cx + 80, y]
        box = np.array([cx - 90, top, cx + 90, top + 40])
    else:
        k[0] = [cx, top + 0.08 * height]
        k[[5, 6]] = [cx, top + 0.2 * height]
        k[[9, 10]] = [cx, top - 0.05 * height] if hand_up else [cx, top + 0.45 * height]
        k[[11, 12]] = [cx, top + 0.5 * height]
        k[[15, 16]] = [cx, top + height]
        box = np.array([cx - 30, top - (0.1 * height if hand_up else 0), cx + 30, top + height])
    c[[0, 5, 6, 9, 10, 11, 12, 15, 16]] = 0.9
    return Person(tid, box, k, c)


def chair_for(p):
    x1, y1, x2, y2 = p.box
    return Aid("wheelchair", np.array([x1, y1 + 0.5 * (y2 - y1), x2 + 10, y2]), 0.8)


def cane_for(p):
    x1, y1, x2, y2 = p.box
    return Aid("white_cane", np.array([x2 - 5, y1 + 0.4 * (y2 - y1), x2 + 15, y2]), 0.7)


def run(eng, frames):
    """frames: list of (people, aids). Returns the list of event kinds."""
    out = []
    for i, (people, aids) in enumerate(frames):
        out += [e.kind for e in eng.update(people, aids, t=i / FPS)]
    return out


def seconds(s):
    return int(s * FPS)


# ---- restricted zone ----------------------------------------------------
ZONE = [{"name": "Track area", "type": "restricted",
         "points": [[0, 400], [640, 400], [640, 480], [0, 480]]}]


def test_zone_entry():
    outside = [([person(1, 100, 100)], [])] * 5
    inside = [([person(1, 100, 230)], [])] * 10      # ankles at y=430
    assert run(engine(ZONE), outside + inside) == ["zone_entry"]


def test_zone_brief_touch_ignored():
    outside = [([person(1, 100, 100)], [])] * 5
    inside = [([person(1, 100, 230)], [])] * 2
    assert run(engine(ZONE), outside + inside + outside) == []


# ---- fall ------------------------------------------------------------------
def test_fall_detected():
    standing = [([person(1, 300, 100)], [])] * 5
    down = [([person(1, 300, 260, lying=True)], [])] * seconds(2.5)
    assert run(engine(), standing + down) == ["fall"]


def test_quick_recovery_is_not_a_fall():
    standing = [([person(1, 300, 100)], [])] * 5
    down = [([person(1, 300, 260, lying=True)], [])] * 3
    assert run(engine(), standing + down + standing) == []


def test_walking_is_not_a_fall():
    assert run(engine(), [([person(1, 100 + 5 * i, 100)], []) for i in range(40)]) == []


# ---- mobility aids -----------------------------------------------------------
def test_wheelchair_detected_once():
    p = person(7, 300, 100)
    assert run(engine(), [([p], [chair_for(p)])] * seconds(3)) == ["aid_detected"]


def test_aid_far_from_person_not_linked():
    p = person(7, 100, 100)
    far = Aid("wheelchair", np.array([500, 200, 600, 300]), 0.8)
    assert run(engine(), [([p], [far])] * seconds(3)) == []


def test_aid_survives_missed_detections():
    p = person(7, 300, 100)
    frames = [([p], [chair_for(p)] if i % 3 else []) for i in range(seconds(3))]
    assert run(engine(), frames) == ["aid_detected"]


# ---- raised hand ---------------------------------------------------------------
def test_help_request():
    frames = [([person(1, 300, 100)], [])] * 5 + [([person(1, 300, 100, hand_up=True)], [])] * seconds(3)
    assert run(engine(), frames) == ["help_request"]


def test_short_wave_ignored():
    frames = [([person(1, 300, 100, hand_up=True)], [])] * seconds(1) + [([person(1, 300, 100)], [])] * 5
    assert run(engine(), frames) == []


# ---- stuck -------------------------------------------------------------------------
def test_wheelchair_user_stuck():
    p = person(3, 300, 100)
    kinds = run(engine(), [([p], [chair_for(p)])] * seconds(30))
    assert kinds == ["aid_detected", "stuck"]


def test_no_stuck_in_seating_area():
    seat = [{"name": "Benches", "type": "seating", "points": [[200, 250], [400, 250], [400, 350], [200, 350]]}]
    p = person(3, 300, 100)
    assert run(engine(seat), [([p], [chair_for(p)])] * seconds(30)) == ["aid_detected"]


def test_no_stuck_while_being_helped():
    p, helper = person(3, 300, 100), person(4, 360, 100)
    assert run(engine(), [([p, helper], [chair_for(p)])] * seconds(30)) == ["aid_detected"]


def test_person_without_aid_never_stuck():
    assert run(engine(), [([person(3, 300, 100)], [])] * seconds(30)) == []


# ---- barrier ---------------------------------------------------------------------------
ESCALATOR = [{"name": "Escalator", "type": "barrier", "points": [[400, 250], [640, 250], [640, 400], [400, 400]]}]


def test_wheelchair_at_escalator():
    frames = []
    for i in range(seconds(6)):
        p = person(5, 200 + 6 * i, 100)        # walks right into the escalator zone
        frames.append(([p], [chair_for(p)]))
    assert run(engine(ESCALATOR), frames) == ["aid_detected", "barrier"]


def test_no_barrier_alert_without_aid():
    frames = [([person(5, 200 + 6 * i, 100)], []) for i in range(seconds(6))]
    assert run(engine(ESCALATOR), frames) == []


# ---- hazard ------------------------------------------------------------------------------
STAIRS = [{"name": "Stairs", "type": "hazard", "points": [[600, 0], [640, 0], [640, 480], [600, 480]]}]


def test_cane_user_walking_toward_stairs():
    frames = []
    for i in range(seconds(6)):
        p = person(6, 150 + 7 * i, 100)
        frames.append(([p], [cane_for(p)]))
    assert run(engine(STAIRS), frames) == ["aid_detected", "hazard"]


def test_cane_user_walking_away_is_fine():
    frames = []
    for i in range(seconds(4)):
        p = person(6, 300 - 7 * i, 100)
        frames.append(([p], [cane_for(p)]))
    assert run(engine(STAIRS), frames) == ["aid_detected"]


# ---- disoriented ----------------------------------------------------------------------------
def test_cane_user_disoriented():
    frames = []
    for i in range(seconds(15)):
        x = 300 + 80 * np.sin(i / 6)           # slow back-and-forth
        p = person(8, x, 100)
        frames.append(([p], [cane_for(p)]))
    assert "disoriented" in run(engine(), frames)


def test_cane_user_walking_straight_is_fine():
    frames = []
    for i in range(seconds(15)):
        p = person(8, 50 + 3 * i, 100)
        frames.append(([p], [cane_for(p)]))
    assert run(engine(), frames) == ["aid_detected"]
