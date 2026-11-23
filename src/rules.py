"""Rule engine: turns per-frame detections into events.

Every rule works per tracked person (track ID) on that person's own history.
Distances are measured relative to the person's body height in pixels, so the
same thresholds work near and far from the camera.

Event kinds
-----------
aid_detected   a mobility aid is linked to a person                (info)
stuck          aid user has not moved for a while                  (medium)
barrier        wheelchair / walking frame in a barrier zone        (medium)
disoriented    aid user moving slowly with many direction changes  (medium)
hazard         white cane user inside or approaching a hazard zone (high)
help_request   hand raised above the head for a few seconds        (high)
fall           sudden drop, torso horizontal, stays down           (high)
zone_entry     anyone inside a restricted zone                     (high)
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Point, Polygon

# COCO keypoint indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_ANKLE, R_ANKLE = 15, 16

ZONE_TYPES = ("restricted", "hazard", "barrier", "seating")


@dataclass
class Person:
    track_id: int
    box: np.ndarray            # [x1, y1, x2, y2]
    keypoints: np.ndarray      # (17, 2)
    kp_conf: np.ndarray        # (17,)


@dataclass
class Aid:
    cls_name: str
    box: np.ndarray
    conf: float


@dataclass
class Event:
    track_id: int
    kind: str
    box: np.ndarray
    t: float
    detail: str = ""           # zone name or aid class
    who: str = ""              # mobility aid of the person, if any


@dataclass
class Zone:
    name: str
    type: str
    poly: Polygon

    @classmethod
    def from_cfg(cls, z: dict) -> "Zone":
        if z["type"] not in ZONE_TYPES:
            raise ValueError(f"zone '{z['name']}': type must be one of {ZONE_TYPES}")
        return cls(z["name"], z["type"], Polygon(z["points"]))


class Hold:
    """Measures how long a condition has been true, tolerating short gaps
    (missed detections, a keypoint dropping out for a frame or two)."""

    def __init__(self, gap: float = 0.5):
        self.gap = gap
        self.start: float | None = None
        self.last: float | None = None
        self.fired = False

    def update(self, cond: bool, t: float) -> float:
        if cond:
            if self.start is None or (self.last is not None and t - self.last > self.gap):
                self.start, self.fired = t, False
            self.last = t
            return t - self.start
        if self.last is not None and t - self.last > self.gap:
            self.start, self.fired = None, False
        return 0.0

    def trigger(self, cond: bool, t: float, need: float) -> bool:
        """True once per episode, when the condition has held for `need` seconds."""
        held = self.update(cond, t)
        if cond and held >= need and not self.fired:
            self.fired = True
            return True
        return False


@dataclass
class TrackState:
    pose: deque = field(default_factory=lambda: deque(maxlen=90))     # (t, hip_y, angle, height)
    path: deque = field(default_factory=lambda: deque(maxlen=400))    # (t, x, y) foot point
    heights: deque = field(default_factory=lambda: deque(maxlen=60))
    holds: dict = field(default_factory=dict)
    aids: set = field(default_factory=set)        # confirmed mobility aids
    fall_candidate_t: float | None = None
    fall_reported: bool = False
    helped: bool = False
    last_seen: float = 0.0

    def hold(self, key: str, gap: float = 0.5) -> Hold:
        if key not in self.holds:
            self.holds[key] = Hold(gap)
        return self.holds[key]

    @property
    def height(self) -> float:
        return float(np.median(self.heights)) if self.heights else 1.0

    @property
    def who(self) -> str:
        for a in ("white_cane", "wheelchair", "walking_frame", "crutches"):
            if a in self.aids:
                return a
        return ""


# ---- geometry helpers --------------------------------------------------
def _mid(kps, conf, a, b, thr):
    pts = [kps[i] for i in (a, b) if conf[i] >= thr]
    return np.mean(pts, axis=0) if pts else None


def foot_point(p: Person, thr: float) -> np.ndarray:
    """Midpoint of the ankles; falls back to the bottom-center of the box."""
    ankle = _mid(p.keypoints, p.kp_conf, L_ANKLE, R_ANKLE, thr)
    if ankle is not None:
        return ankle
    x1, _, x2, y2 = p.box
    return np.array([(x1 + x2) / 2, y2])


def torso_angle(p: Person, thr: float) -> float | None:
    """Angle of the shoulder-to-hip line from vertical, in degrees (0 = upright)."""
    sh = _mid(p.keypoints, p.kp_conf, L_SHOULDER, R_SHOULDER, thr)
    hp = _mid(p.keypoints, p.kp_conf, L_HIP, R_HIP, thr)
    if sh is None or hp is None:
        return None
    dx, dy = hp - sh
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))


def hand_raised(p: Person, thr: float) -> bool:
    """A wrist clearly above the head (nose, or shoulders if the face is not visible)."""
    k, c = p.keypoints, p.kp_conf
    h = p.box[3] - p.box[1]
    if c[NOSE] >= thr:
        ref = k[NOSE][1] - 0.05 * h
    else:
        sh = _mid(k, c, L_SHOULDER, R_SHOULDER, thr)
        if sh is None:
            return False
        ref = sh[1] - 0.2 * h
    return any(c[w] >= thr and k[w][1] < ref for w in (L_WRIST, R_WRIST))


def overlap_ratio(inner: np.ndarray, outer: np.ndarray) -> float:
    """Fraction of the `inner` box area that lies inside the `outer` box."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


def expand(box: np.ndarray, k: float = 0.15) -> np.ndarray:
    x1, y1, x2, y2 = box
    dw, dh = (x2 - x1) * k, (y2 - y1) * k
    return np.array([x1 - dw, y1 - dh, x2 + dw, y2 + dh])


# ---- engine ------------------------------------------------------------
class RuleEngine:
    def __init__(self, cfg: dict):
        self.r = cfg["rules"]
        self.kp_thr = self.r["keypoint_conf"]
        self.zones = [Zone.from_cfg(z) for z in (cfg.get("zones") or [])]
        self.tracks: dict[int, TrackState] = {}

    def update(self, people: list[Person], aids: list[Aid], t: float) -> list[Event]:
        events: list[Event] = []
        linked = self._link_aids(people, aids)
        feet = {p.track_id: foot_point(p, self.kp_thr) for p in people}

        for p in people:
            st = self.tracks.setdefault(p.track_id, TrackState())
            st.last_seen = t
            st.heights.append(p.box[3] - p.box[1])
            st.path.append((t, *feet[p.track_id]))

            ev: list[Event] = []
            ev += self._aids(p, st, linked.get(p.track_id, set()), t)
            ev += self._zones(p, st, feet[p.track_id], t)
            ev += self._fall(p, st, t)
            ev += self._help_request(p, st, t)
            if st.aids:
                self._update_helped(p, st, feet, t)
                ev += self._stuck(p, st, feet[p.track_id], t)
                ev += self._disoriented(p, st, t)
            for e in ev:
                e.who = st.who
            events += ev

        self._forget_old(t)
        return events

    def zone_at(self, xy: np.ndarray, ztype: str) -> Zone | None:
        pt = Point(*xy)
        return next((z for z in self.zones if z.type == ztype and z.poly.contains(pt)), None)

    # ---- mobility aids -------------------------------------------------
    def _link_aids(self, people: list[Person], aids: list[Aid]) -> dict[int, set[str]]:
        """Attach each aid box to the person it overlaps most."""
        linked: dict[int, set[str]] = {}
        for a in aids:
            best_id, best = None, 0.0
            for p in people:
                r = overlap_ratio(a.box, expand(p.box))
                if r > best:
                    best_id, best = p.track_id, r
            if best_id is not None and best >= self.r["aid"]["min_overlap"]:
                linked.setdefault(best_id, set()).add(a.cls_name)
        return linked

    def _aids(self, p: Person, st: TrackState, classes: set[str], t: float) -> list[Event]:
        c = self.r["aid"]
        out = []
        for name in set(classes) | {k[4:] for k in st.holds if k.startswith("aid:")}:
            if st.hold(f"aid:{name}", gap=c["gap_sec"]).trigger(name in classes, t, c["confirm_sec"]):
                if name not in st.aids:
                    st.aids.add(name)
                    out.append(Event(p.track_id, "aid_detected", p.box, t, detail=name))
        return out

    # ---- zones ---------------------------------------------------------
    def _zones(self, p: Person, st: TrackState, foot: np.ndarray, t: float) -> list[Event]:
        c = self.r["zones"]
        out = []
        pt = Point(*foot)
        h = st.height
        for z in self.zones:
            inside = z.poly.contains(pt)
            if z.type == "restricted":
                if st.hold(f"zone:{z.name}").trigger(inside, t, c["restricted_sec"]):
                    out.append(Event(p.track_id, "zone_entry", p.box, t, detail=z.name))

            elif z.type == "barrier":
                cond = inside and bool(st.aids & set(c["barrier_aids"]))
                if st.hold(f"barrier:{z.name}").trigger(cond, t, c["barrier_sec"]):
                    out.append(Event(p.track_id, "barrier", p.box, t, detail=z.name))

            elif z.type == "hazard":
                cond = False
                if st.aids & set(c["hazard_aids"]):
                    d_now = pt.distance(z.poly)
                    past = [s for s in st.path if t - s[0] >= 1.0]
                    d_before = Point(past[-1][1], past[-1][2]).distance(z.poly) if past else d_now
                    approaching = d_now < c["hazard_approach"] * h and d_before - d_now > 0.1 * h
                    cond = inside or approaching
                if st.hold(f"hazard:{z.name}", gap=1.0).trigger(cond, t, c["hazard_sec"]):
                    out.append(Event(p.track_id, "hazard", p.box, t, detail=z.name))
        return out

    # ---- fall ----------------------------------------------------------
    def _fall(self, p: Person, st: TrackState, t: float) -> list[Event]:
        c = self.r["fall"]
        angle = torso_angle(p, self.kp_thr)
        hip = _mid(p.keypoints, p.kp_conf, L_HIP, R_HIP, self.kp_thr)
        x1, y1, x2, y2 = p.box
        w, h = x2 - x1, y2 - y1
        if angle is None:
            angle = 80.0 if w > 1.2 * h else 10.0      # no torso keypoints: use box shape
        hip_y = hip[1] if hip is not None else (y1 + y2) / 2
        st.pose.append((t, hip_y, angle, h))
        is_down = angle >= c["torso_angle_deg"]

        # 1. sudden hip drop inside the window, relative to standing height
        if st.fall_candidate_t is None and not st.fall_reported:
            past = [s for s in st.pose if t - s[0] <= c["window_sec"]]
            if len(past) >= 3:
                start = past[0]
                standing_h = max(s[3] for s in past)
                drop = hip_y - start[1]
                if start[2] < c["torso_angle_deg"] and drop >= c["drop_ratio"] * standing_h and is_down:
                    st.fall_candidate_t = t

        # 2. confirm the person stays down
        if st.fall_candidate_t is not None:
            if not is_down:
                st.fall_candidate_t = None                 # got up quickly: not a fall
            elif t - st.fall_candidate_t >= c["down_sec"]:
                st.fall_candidate_t = None
                st.fall_reported = True
                return [Event(p.track_id, "fall", p.box, t)]

        if st.fall_reported and angle < c["torso_angle_deg"] * 0.5:
            st.fall_reported = False                       # upright again: can detect a new fall
        return []

    # ---- raised hand -----------------------------------------------------
    def _help_request(self, p: Person, st: TrackState, t: float) -> list[Event]:
        c = self.r["help_request"]
        if not c["enabled"] or (c["aid_users_only"] and not st.aids):
            return []
        if st.hold("hand", gap=0.4).trigger(hand_raised(p, self.kp_thr), t, c["hold_sec"]):
            return [Event(p.track_id, "help_request", p.box, t)]
        return []

    # ---- being helped ----------------------------------------------------
    def _update_helped(self, p: Person, st: TrackState, feet: dict, t: float) -> None:
        c = self.r["helper"]
        me = feet[p.track_id]
        near = any(np.linalg.norm(xy - me) < c["distance"] * st.height
                   for tid, xy in feet.items() if tid != p.track_id)
        hold = st.hold("helper", gap=c["release_sec"])
        st.helped = hold.update(near, t) >= c["hold_sec"]

    # ---- stuck -----------------------------------------------------------
    def _stuck(self, p: Person, st: TrackState, foot: np.ndarray, t: float) -> list[Event]:
        c = self.r["stuck"]
        win = [s for s in st.path if t - s[0] <= c["window_sec"]]
        covered = bool(win) and t - win[0][0] >= 0.9 * c["window_sec"]
        cond = False
        if covered and not st.helped and self.zone_at(foot, "seating") is None:
            pts = np.array([[s[1], s[2]] for s in win])
            spread = np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1))
            cond = spread < c["max_move"] * st.height
        if st.hold("stuck", gap=1.0).trigger(cond, t, 0.0):
            return [Event(p.track_id, "stuck", p.box, t)]
        return []

    # ---- disoriented -----------------------------------------------------
    def _disoriented(self, p: Person, st: TrackState, t: float) -> list[Event]:
        c = self.r["disoriented"]
        win = [s for s in st.path if t - s[0] <= c["window_sec"]]
        cond = False
        if win and t - win[0][0] >= 0.9 * c["window_sec"] and not st.helped:
            # sample every 0.5 s so detection jitter does not inflate the path length
            pts, last_t = [], -1e9
            for s in win:
                if s[0] - last_t >= 0.5:
                    pts.append((s[1], s[2]))
                    last_t = s[0]
            pts = np.array(pts)
            if len(pts) >= 4:
                steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
                length = steps.sum()
                net = np.linalg.norm(pts[-1] - pts[0])
                h = st.height
                cond = (length > c["min_path"] * h
                        and length / max(net, 1e-6) > c["tortuosity"]
                        and length / (win[-1][0] - win[0][0]) < c["max_speed"] * h)
        if st.hold("disoriented", gap=1.0).trigger(cond, t, 0.0):
            return [Event(p.track_id, "disoriented", p.box, t)]
        return []

    def _forget_old(self, t: float, ttl: float = 10.0) -> None:
        for tid in [k for k, s in self.tracks.items() if t - s.last_seen > ttl]:
            del self.tracks[tid]
