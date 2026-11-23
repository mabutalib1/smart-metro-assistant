"""Alert manager: severity levels, messages, cooldown, logging and snapshots."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .rules import Event

LEVEL = {
    "aid_detected": "info",
    "stuck": "medium",
    "barrier": "medium",
    "disoriented": "medium",
    "hazard": "high",
    "help_request": "high",
    "fall": "high",
    "zone_entry": "high",
}
LEVEL_RANK = {"info": 0, "medium": 1, "high": 2}

WHO = {
    "wheelchair": "Wheelchair user",
    "white_cane": "White cane user",
    "crutches": "Passenger with crutches",
    "walking_frame": "Passenger with walking frame",
    "": "Passenger",
}


def describe(e: Event, gate: str) -> str:
    who = WHO.get(e.who, "Passenger")
    if e.kind == "aid_detected":
        return f"{WHO.get(e.detail, 'Passenger')} at {gate}, may need assistance"
    if e.kind == "stuck":
        return f"{who} has not moved for a while at {gate}"
    if e.kind == "barrier":
        return f"{who} at {e.detail}, guide to the elevator ({gate})"
    if e.kind == "disoriented":
        return f"{who} seems lost at {gate}"
    if e.kind == "hazard":
        return f"{who} approaching {e.detail} at {gate}"
    if e.kind == "help_request":
        return f"{who} is asking for help at {gate}"
    if e.kind == "fall":
        return f"Fall detected at {gate}, send help now"
    if e.kind == "zone_entry":
        return f"Person inside {e.detail} at {gate}"
    return f"{e.kind} at {gate}"


@dataclass
class Alert:
    time: str
    t: float                  # stream time in seconds (video time for files)
    camera: str
    gate: str
    track_id: int
    event: str
    level: str
    message: str
    snapshot: str | None = None


class AlertManager:
    def __init__(self, cfg: dict):
        a = cfg["alerts"]
        self.cooldown = a["cooldown_sec"]
        self.camera = cfg["camera"]["id"]
        self.gate = cfg["camera"]["gate"]
        self.log_file = Path(a["log_file"]) if a.get("log_file") else None
        self.save_snapshots = a.get("save_snapshots", False)
        self.snapshot_dir = Path(a.get("snapshot_dir", "outputs/snapshots"))
        self.blur = a.get("blur_faces", True)
        self.last: dict[tuple[int, str], float] = {}
        self.history: list[Alert] = []
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if self.save_snapshots:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def handle(self, events: list[Event], frame: np.ndarray | None = None) -> list[Alert]:
        out = []
        for e in sorted(events, key=lambda e: -LEVEL_RANK[LEVEL.get(e.kind, "medium")]):
            key = (e.track_id, e.kind)
            if key in self.last and e.t - self.last[key] < self.cooldown:
                continue
            self.last[key] = e.t
            now = datetime.now()
            alert = Alert(
                time=now.strftime("%Y-%m-%d %H:%M:%S"),
                t=round(e.t, 2),
                camera=self.camera,
                gate=self.gate,
                track_id=e.track_id,
                event=e.kind,
                level=LEVEL.get(e.kind, "medium"),
                message=describe(e, self.gate),
            )
            if self.save_snapshots and frame is not None:
                alert.snapshot = self._snapshot(frame, e, now)
            if self.log_file:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(alert), ensure_ascii=False) + "\n")
            self.history.append(alert)
            out.append(alert)
        return out

    def _snapshot(self, frame: np.ndarray, e: Event, now: datetime) -> str:
        img = frame.copy()
        if self.blur:
            img = blur_head(img, e.box)
        x1, y1, x2, y2 = map(int, e.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        path = self.snapshot_dir / f"{now:%Y%m%d_%H%M%S}_{e.track_id}_{e.kind}.jpg"
        cv2.imwrite(str(path), img)
        return str(path)


def blur_head(img: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Blur the top part of the person box (approximate head region) for privacy."""
    x1, y1, x2, y2 = map(int, box)
    hx1, hy1 = max(0, x1), max(0, y1)
    hx2, hy2 = max(0, x2), max(0, y1 + int(0.22 * (y2 - y1)))
    roi = img[hy1:hy2, hx1:hx2]
    if roi.size:
        img[hy1:hy2, hx1:hx2] = cv2.GaussianBlur(roi, (51, 51), 0)
    return img
