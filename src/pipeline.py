"""Main pipeline: video frame -> pose + tracking -> aid detection -> rules -> alerts.

Run:
    python -m src.pipeline --source path/to/video.mp4 --show
    python -m src.pipeline --source 0 --show          # webcam
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO

from .alerts import Alert, AlertManager
from .config import load_config
from .rules import Aid, Person, RuleEngine

COLORS = {"zone_entry": (0, 0, 255), "fall": (0, 0, 255)}
AID_COLOR = (255, 160, 0)


class StationPipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        m = cfg["models"]
        self.pose = YOLO(m["pose"])
        self.aids = YOLO(m["aids"]) if m.get("aids") else None
        self.rules = RuleEngine(cfg)
        self.alerts = AlertManager(cfg)
        self.zone = np.array(cfg.get("zones", {}).get("restricted") or [], dtype=np.int32)
        self.flags: dict[int, str] = {}   # track_id -> latest event, for drawing

    def process(self, frame: np.ndarray, t: float) -> tuple[np.ndarray, list[Alert]]:
        m = self.cfg["models"]
        res = self.pose.track(frame, persist=True, tracker=m["tracker"], conf=m["person_conf"],
                              classes=[0], device=m.get("device"), verbose=False)[0]
        people = self._people(res)
        aids = self._aids(frame) if self.aids else []

        events = self.rules.update(people, aids, t)
        for e in events:
            self.flags[e.track_id] = e.kind
        alerts = self.alerts.handle(events, frame)
        return self._draw(frame, people, aids), alerts

    @staticmethod
    def _people(res) -> list[Person]:
        if res.boxes is None or res.boxes.id is None:
            return []
        boxes = res.boxes.xyxy.cpu().numpy()
        ids = res.boxes.id.int().cpu().numpy()
        kps = res.keypoints.xy.cpu().numpy()
        conf = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else np.ones(kps.shape[:2])
        return [Person(int(i), b, k, c) for i, b, k, c in zip(ids, boxes, kps, conf)]

    def _aids(self, frame: np.ndarray) -> list[Aid]:
        m = self.cfg["models"]
        r = self.aids.predict(frame, conf=m["aid_conf"], device=m.get("device"), verbose=False)[0]
        names = r.names
        return [Aid(names[int(c)], b, float(s)) for b, c, s in
                zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]

    def _draw(self, frame: np.ndarray, people: list[Person], aids: list[Aid]) -> np.ndarray:
        img = frame.copy()
        if len(self.zone) >= 3:
            overlay = img.copy()
            cv2.fillPoly(overlay, [self.zone], (0, 215, 255))
            img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
            cv2.polylines(img, [self.zone], True, (0, 215, 255), 2)
        for a in aids:
            x1, y1, x2, y2 = map(int, a.box)
            cv2.rectangle(img, (x1, y1), (x2, y2), AID_COLOR, 2)
            cv2.putText(img, a.cls_name, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, AID_COLOR, 1)
        active = {p.track_id for p in people}
        for p in people:
            flag = self.flags.get(p.track_id)
            color = COLORS.get(flag, (0, 200, 255) if flag else (0, 200, 0))
            x1, y1, x2, y2 = map(int, p.box)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"ID {p.track_id}" + (f" | {flag}" if flag else "")
            cv2.putText(img, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        self.flags = {k: v for k, v in self.flags.items() if k in active}
        return img


def open_source(src):
    return cv2.VideoCapture(int(src) if str(src).isdigit() else src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", default=None, help="video path, webcam index or RTSP URL")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save", default=None, help="write annotated video to this path")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    source = args.source if args.source is not None else cfg["camera"]["source"]
    pipe = StationPipeline(cfg)
    cap = open_source(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    is_file = not str(source).isdigit() and not str(source).startswith("rtsp")
    writer, idx, t0 = None, 0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps if is_file else time.time() - t0   # video time keeps results repeatable
        out, alerts = pipe.process(frame, t)
        for a in alerts:
            print(f"[{a.time}] {a.severity.upper():6} {a.message} (ID {a.track_id})")
        if args.save:
            if writer is None:
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(out)
        if args.show:
            cv2.imshow("Smart Metro Station Assistant", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
