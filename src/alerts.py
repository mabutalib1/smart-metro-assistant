"""Alert manager: one alert per person per event type, with a cooldown."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .rules import Event

MESSAGES = {
    "zone_entry": "Person in restricted zone",
    "fall": "Fall detected, send help",
    "assistance:wheelchair": "Wheelchair user needs assistance",
    "assistance:white_cane": "White cane user needs assistance",
    "assistance:crutches": "Passenger with crutches may need assistance",
    "assistance:walking_frame": "Passenger with walking frame may need assistance",
}

SEVERITY = {"fall": "high", "zone_entry": "high"}


@dataclass
class Alert:
    time: str
    camera: str
    gate: str
    track_id: int
    event: str
    severity: str
    message: str
    snapshot: str | None = None


class AlertManager:
    def __init__(self, cfg: dict):
        a = cfg["alerts"]
        self.cooldown = a["cooldown_sec"]
        self.camera = cfg["camera"]["id"]
        self.gate = cfg["camera"]["gate"]
        self.log_file = Path(a["log_file"])
        self.save_snapshots = a.get("save_snapshots", False)
        self.snapshot_dir = Path(a.get("snapshot_dir", "outputs/snapshots"))
        self.blur = a.get("blur_faces", True)
        self.last: dict[tuple[int, str], float] = {}
        self.history: list[Alert] = []
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if self.save_snapshots:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def handle(self, events: list[Event], frame: np.ndarray | None = None) -> list[Alert]:
        out = []
        for e in events:
            key = (e.track_id, e.kind)
            if key in self.last and e.t - self.last[key] < self.cooldown:
                continue
            self.last[key] = e.t
            now = datetime.now()
            alert = Alert(
                time=now.strftime("%Y-%m-%d %H:%M:%S"),
                camera=self.camera,
                gate=self.gate,
                track_id=e.track_id,
                event=e.kind,
                severity=SEVERITY.get(e.kind, "medium"),
                message=f"{MESSAGES.get(e.kind, e.kind)} at {self.gate}",
            )
            if self.save_snapshots and frame is not None:
                alert.snapshot = self._snapshot(frame, e, now)
            self._log(alert)
            self.history.append(alert)
            out.append(alert)
        return out

    def _snapshot(self, frame: np.ndarray, e: Event, now: datetime) -> str:
        img = frame.copy()
        if self.blur:
            img = blur_heads(img, e.box)
        x1, y1, x2, y2 = map(int, e.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        path = self.snapshot_dir / f"{now:%Y%m%d_%H%M%S}_{e.track_id}_{e.kind.replace(':', '-')}.jpg"
        cv2.imwrite(str(path), img)
        return str(path)

    def _log(self, alert: Alert) -> None:
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(alert), ensure_ascii=False) + "\n")


def blur_heads(img: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Blur the top part of the person box (approximate head region) for privacy."""
    x1, y1, x2, y2 = map(int, box)
    h = y2 - y1
    hx1, hy1, hx2, hy2 = max(0, x1), max(0, y1), max(0, x2), max(0, y1 + int(0.22 * h))
    roi = img[hy1:hy2, hx1:hx2]
    if roi.size:
        img[hy1:hy2, hx1:hx2] = cv2.GaussianBlur(roi, (51, 51), 0)
    return img
