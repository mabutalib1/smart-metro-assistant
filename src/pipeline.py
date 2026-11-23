"""Main pipeline: frame -> pose + tracking -> mobility-aid detection -> rules -> alerts.

Run:
    python -m src.pipeline --source path/to/video.mp4 --show
    python -m src.pipeline --source 0 --show --save outputs/demo.mp4
"""
from __future__ import annotations

import argparse
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

from .alerts import LEVEL, Alert, AlertManager
from .config import load_config
from .rules import Aid, Person, RuleEngine

# BGR colours
LEVEL_COLOR = {"info": (230, 160, 40), "medium": (0, 165, 255), "high": (0, 0, 255)}
ZONE_COLOR = {"restricted": (0, 0, 220), "hazard": (0, 215, 255),
              "barrier": (200, 120, 0), "seating": (120, 180, 0)}
OK_COLOR = (60, 190, 60)
AID_COLOR = (255, 160, 0)


class StationPipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        m = cfg["models"]
        self.pose = YOLO(m["pose"])
        self.aid_model = YOLO(m["aids"]) if m.get("aids") else None
        self.rules = RuleEngine(cfg)
        self.alerts = AlertManager(cfg)
        self.flags: dict[int, tuple[str, float]] = {}   # track_id -> (event, time) for drawing
        self._times: deque = deque(maxlen=30)
        self.fps = 0.0

    def process(self, frame: np.ndarray, t: float) -> tuple[np.ndarray, list[Alert]]:
        t0 = time.perf_counter()
        m = self.cfg["models"]
        res = self.pose.track(frame, persist=True, tracker=m["tracker"], conf=m["person_conf"],
                              classes=[0], device=m.get("device"), verbose=False)[0]
        people = self._people(res)
        aids = self._aids(frame) if self.aid_model else []

        events = self.rules.update(people, aids, t)
        for e in events:
            self.flags[e.track_id] = (e.kind, t)
        alerts = self.alerts.handle(events, frame)

        self._times.append(time.perf_counter() - t0)
        self.fps = len(self._times) / max(sum(self._times), 1e-6)
        return self._draw(frame, people, aids, t), alerts

    @staticmethod
    def _people(res) -> list[Person]:
        if res.boxes is None or res.boxes.id is None:
            return []
        boxes = res.boxes.xyxy.cpu().numpy()
        ids = res.boxes.id.int().cpu().numpy()
        kps = res.keypoints.xy.cpu().numpy()
        conf = (res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None
                else np.ones(kps.shape[:2]))
        return [Person(int(i), b, k, c) for i, b, k, c in zip(ids, boxes, kps, conf)]

    def _aids(self, frame: np.ndarray) -> list[Aid]:
        m = self.cfg["models"]
        r = self.aid_model.predict(frame, conf=m["aid_conf"], device=m.get("device"), verbose=False)[0]
        return [Aid(r.names[int(c)], b, float(s)) for b, c, s in
                zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]

    # ---- drawing ---------------------------------------------------------
    def _draw(self, frame, people, aids, t) -> np.ndarray:
        img = frame.copy()
        overlay = img.copy()
        for z in self.rules.zones:
            pts = np.array(z.poly.exterior.coords, np.int32)
            cv2.fillPoly(overlay, [pts], ZONE_COLOR[z.type])
        img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)
        for z in self.rules.zones:
            pts = np.array(z.poly.exterior.coords, np.int32)
            cv2.polylines(img, [pts], True, ZONE_COLOR[z.type], 2)
            x, y = pts.min(axis=0)
            _label(img, z.name, (int(x) + 4, int(y) + 18), ZONE_COLOR[z.type])

        for a in aids:
            x1, y1, x2, y2 = map(int, a.box)
            cv2.rectangle(img, (x1, y1), (x2, y2), AID_COLOR, 1)

        for p in people:
            st = self.rules.tracks.get(p.track_id)
            flag = self.flags.get(p.track_id)
            if flag and t - flag[1] > 15:          # flags fade after 15 s
                flag = None
            color = LEVEL_COLOR[LEVEL[flag[0]]] if flag else OK_COLOR
            x1, y1, x2, y2 = map(int, p.box)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            text = f"ID {p.track_id}"
            if st and st.who:
                text += f" {st.who.replace('_', ' ')}"
            if st and st.helped:
                text += " (helped)"
            if flag:
                text += f" | {flag[0].replace('_', ' ')}"
            _label(img, text, (x1, max(18, y1 - 6)), color)

        _label(img, f"{self.cfg['camera']['gate']}  |  {self.fps:4.1f} FPS", (10, 24), (40, 40, 40))
        return img


def _label(img, text, org, color):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    x, y = org
    cv2.rectangle(img, (x - 2, y - h - 6), (x + w + 4, y + 4), color, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def open_source(src):
    return cv2.VideoCapture(int(src) if str(src).isdigit() else src)


def is_live(src) -> bool:
    s = str(src)
    return s.isdigit() or s.startswith(("rtsp://", "http://", "https://"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", default=None, help="video path, webcam index or stream URL")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save", default=None, help="write the annotated video to this path")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    source = args.source if args.source is not None else cfg["camera"]["source"]
    pipe = StationPipeline(cfg)
    cap = open_source(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    live = is_live(source)
    writer, idx, t0 = None, 0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.time() - t0 if live else idx / fps   # video time keeps results repeatable
        out, alerts = pipe.process(frame, t)
        for a in alerts:
            print(f"[{a.t:7.1f}s] {a.level.upper():6} {a.message} (ID {a.track_id})")
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
