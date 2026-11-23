import numpy as np

from src.rules import Aid, Person, RuleEngine

CFG = {
    "rules": {
        "keypoint_conf": 0.3,
        "zone": {"min_frames": 3},
        "fall": {"window_sec": 1.0, "drop_ratio": 0.3, "torso_angle_deg": 55, "down_sec": 1.0},
        "aid": {"min_frames": 3, "min_overlap": 0.3},
    },
    "zones": {"restricted": [[0, 400], [640, 400], [640, 480], [0, 480]]},
}


def make_person(tid, cx, top, height, lying=False):
    """Synthetic person: upright or lying, with shoulder, hip and ankle keypoints."""
    kps = np.zeros((17, 2))
    conf = np.zeros(17)
    if lying:
        y = top + 20
        kps[[5, 6]] = [cx - 60, y]
        kps[[11, 12]] = [cx, y]
        kps[[15, 16]] = [cx + 80, y]
        box = np.array([cx - 90, top, cx + 90, top + 40])
    else:
        kps[[5, 6]] = [cx, top + 0.2 * height]
        kps[[11, 12]] = [cx, top + 0.5 * height]
        kps[[15, 16]] = [cx, top + height]
        box = np.array([cx - 30, top, cx + 30, top + height])
    conf[[5, 6, 11, 12, 15, 16]] = 0.9
    return Person(tid, box, kps, conf)


def run(engine, frames, fps=10):
    events = []
    for i, people in enumerate(frames):
        events += engine.update(*people, t=i / fps)
    return [e.kind for e in events]


def test_zone_entry_after_min_frames():
    eng = RuleEngine(CFG)
    outside = [([make_person(1, 100, 100, 200)], [])] * 3
    inside = [([make_person(1, 100, 230, 200)], [])] * 5   # ankles at y=430
    assert run(eng, outside + inside) == ["zone_entry"]


def test_fall_detected_when_person_stays_down():
    eng = RuleEngine(CFG)
    standing = [([make_person(1, 300, 100, 200)], [])] * 5
    down = [([make_person(1, 300, 260, 200, lying=True)], [])] * 15
    assert run(eng, standing + down) == ["fall"]


def test_no_fall_when_person_gets_up_quickly():
    eng = RuleEngine(CFG)
    standing = [([make_person(1, 300, 100, 200)], [])] * 5
    down = [([make_person(1, 300, 260, 200, lying=True)], [])] * 3
    assert run(eng, standing + down + standing) == []


def test_no_fall_when_walking():
    eng = RuleEngine(CFG)
    walk = [([make_person(1, 100 + 5 * i, 100, 200)], []) for i in range(30)]
    assert run(eng, walk) == []


def test_wheelchair_linked_to_person():
    eng = RuleEngine(CFG)
    p = make_person(7, 300, 100, 200)
    chair = Aid("wheelchair", np.array([270, 200, 340, 310]), 0.8)
    kinds = run(eng, [([p], [chair])] * 5)
    assert kinds == ["assistance:wheelchair"]


def test_far_aid_not_linked():
    eng = RuleEngine(CFG)
    p = make_person(7, 100, 100, 200)
    chair = Aid("wheelchair", np.array([500, 200, 600, 300]), 0.8)
    assert run(eng, [([p], [chair])] * 5) == []
