"""Rule engine: turns per-frame detections into events.

Events: "zone_entry", "fall", and "assistance:<aid_class>".
All rules work per tracked person (track ID), so each person is judged on
their own history across frames.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Point, Polygon

# COCO keypoint indices
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_ANKLE, R_ANKLE = 15, 16


@dataclass
class Person:
    track_id: int
    box: np.ndarray            # [x1, y1, x2, y2]
    keypoints: np.ndarray      # (17, 2)
    kp_conf: np.ndarray        # (17,)


@dataclass
class Aid:
    cls_name: str
    box: np.ndarray            # [x1, y1, x2, y2]
    conf: float


@dataclass
class Event:
    track_id: int
    kind: str
    box: np.ndarray
    t: float


@dataclass
class TrackState:
    history: deque = field(default_factory=lambda: deque(maxlen=120))  # (t, hip_y, angle, height)
    zone_frames: int = 0
    aid_frames: dict = field(default_factory=dict)
    fall_candidate_t: float | None = None
    fall_reported: bool = False
    last_seen: float = 0.0


def _mid(kps, conf, a, b, thr):
    pts = [kps[i] for i in (a, b) if conf[i] >= thr]
    if not pts:
        return None
    return np.mean(pts, axis=0)


def foot_point(p: Person, thr: float) -> np.ndarray:
    """Midpoint of the ankles; falls back to the bottom-center of the box."""
    ankle = _mid(p.keypoints, p.kp_conf, L_ANKLE, R_ANKLE, thr)
    if ankle is not None:
        return ankle
    x1, _, x2, y2 = p.box
    return np.array([(x1 + x2) / 2, y2])


def torso_angle(p: Person, thr: float) -> float | None:
    """Angle of the shoulder->hip line from vertical, in degrees (0 = upright)."""
    sh = _mid(p.keypoints, p.kp_conf, L_SHOULDER, R_SHOULDER, thr)
    hp = _mid(p.keypoints, p.kp_conf, L_HIP, R_HIP, thr)
    if sh is None or hp is None:
        return None
    dx, dy = hp - sh
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))


def overlap_ratio(inner: np.ndarray, outer: np.ndarray) -> float:
    """Fraction of `inner` box area that lies inside `outer` box."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


class RuleEngine:
    def __init__(self, cfg: dict):
        r = cfg["rules"]
        self.kp_thr = r["keypoint_conf"]
        self.zone_cfg = r["zone"]
        self.fall_cfg = r["fall"]
        self.aid_cfg = r["aid"]
        pts = cfg.get("zones", {}).get("restricted") or []
        self.zone = Polygon(pts) if len(pts) >= 3 else None
        self.tracks: dict[int, TrackState] = {}

    def update(self, people: list[Person], aids: list[Aid], t: float) -> list[Event]:
        events: list[Event] = []
        linked = self._link_aids(people, aids)

        for p in people:
            st = self.tracks.setdefault(p.track_id, TrackState())
            st.last_seen = t
            events += self._zone(p, st, t)
            events += self._fall(p, st, t)
            events += self._aids(p, st, linked.get(p.track_id, set()), t)

        self._forget_old(t)
        return events

    # ---- zone ---------------------------------------------------------
    def _zone(self, p: Person, st: TrackState, t: float) -> list[Event]:
        if self.zone is None:
            return []
        fx, fy = foot_point(p, self.kp_thr)
        if self.zone.contains(Point(fx, fy)):
            st.zone_frames += 1
        else:
            st.zone_frames = 0
        if st.zone_frames == self.zone_cfg["min_frames"]:
            return [Event(p.track_id, "zone_entry", p.box, t)]
        return []

    # ---- fall ---------------------------------------------------------
    def _fall(self, p: Person, st: TrackState, t: float) -> list[Event]:
        c = self.fall_cfg
        angle = torso_angle(p, self.kp_thr)
        hip = _mid(p.keypoints, p.kp_conf, L_HIP, R_HIP, self.kp_thr)
        x1, y1, x2, y2 = p.box
        w, h = x2 - x1, y2 - y1
        if angle is None:
            # no torso keypoints: approximate with box shape
            angle = 80.0 if w > 1.2 * h else 10.0
        hip_y = hip[1] if hip is not None else (y1 + y2) / 2
        st.history.append((t, hip_y, angle, h))

        is_down = angle >= c["torso_angle_deg"]

        # Step 1: sudden drop of the hip within the window, compared with standing height
        if st.fall_candidate_t is None and not st.fall_reported:
            past = [s for s in st.history if t - s[0] <= c["window_sec"]]
            if len(past) >= 3:
                start = past[0]
                standing_h = max(s[3] for s in past)
                drop = hip_y - start[1]
                if start[2] < c["torso_angle_deg"] and drop >= c["drop_ratio"] * standing_h and is_down:
                    st.fall_candidate_t = t

        # Step 2: confirm the person stays down
        if st.fall_candidate_t is not None:
            if not is_down:
                st.fall_candidate_t = None          # got up quickly: not a fall
            elif t - st.fall_candidate_t >= c["down_sec"]:
                st.fall_candidate_t = None
                st.fall_reported = True
                return [Event(p.track_id, "fall", p.box, t)]

        # allow a new fall to be detected once the person is upright again
        if st.fall_reported and angle < c["torso_angle_deg"] * 0.5:
            st.fall_reported = False
        return []

    # ---- mobility aids ------------------------------------------------
    def _link_aids(self, people: list[Person], aids: list[Aid]) -> dict[int, set[str]]:
        """Attach each aid box to the person it overlaps most."""
        linked: dict[int, set[str]] = {}
        for a in aids:
            best_id, best = None, 0.0
            for p in people:
                r = overlap_ratio(a.box, self._expand(p.box))
                if r > best:
                    best_id, best = p.track_id, r
            if best_id is not None and best >= self.aid_cfg["min_overlap"]:
                linked.setdefault(best_id, set()).add(a.cls_name)
        return linked

    @staticmethod
    def _expand(box: np.ndarray, k: float = 0.15) -> np.ndarray:
        # canes and wheelchairs stick out of the person box, so widen it a little
        x1, y1, x2, y2 = box
        dw, dh = (x2 - x1) * k, (y2 - y1) * k
        return np.array([x1 - dw, y1 - dh, x2 + dw, y2 + dh])

    def _aids(self, p: Person, st: TrackState, classes: set[str], t: float) -> list[Event]:
        events = []
        for name in list(st.aid_frames):
            if name not in classes:
                st.aid_frames[name] = max(0, st.aid_frames[name] - 1)
        for name in classes:
            st.aid_frames[name] = st.aid_frames.get(name, 0) + 1
            if st.aid_frames[name] == self.aid_cfg["min_frames"]:
                events.append(Event(p.track_id, f"assistance:{name}", p.box, t))
        return events

    def _forget_old(self, t: float, ttl: float = 10.0) -> None:
        for tid in [k for k, s in self.tracks.items() if t - s.last_seen > ttl]:
            del self.tracks[tid]
