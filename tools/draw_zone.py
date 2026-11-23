"""Draw a zone on a frame from the camera and print it as YAML for configs/config.yaml.

Usage:
    python tools/draw_zone.py --source path/to/video.mp4 --name "Stairs" --type hazard
    python tools/draw_zone.py --source 0 --name "Escalator" --type barrier --frame 50

Left click: add point   Right click: undo   Enter: print YAML   S: save frame as PNG   Esc: quit
Zone types: restricted, hazard, barrier, seating (see configs/config.yaml).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.rules import ZONE_TYPES  # noqa: E402

points: list[list[int]] = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN and points:
        points.pop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--name", default="New zone")
    ap.add_argument("--type", default="restricted", choices=ZONE_TYPES)
    ap.add_argument("--frame", type=int, default=0, help="frame number to draw on")
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Could not read a frame from the source.")

    existing = load_config().get("zones") or []
    cv2.namedWindow("zone")
    cv2.setMouseCallback("zone", on_mouse)
    while True:
        img = frame.copy()
        for z in existing:                       # zones already in the config, in grey
            cv2.polylines(img, [np.array(z["points"], np.int32)], True, (160, 160, 160), 1)
        if points:
            pts = np.array(points, np.int32)
            cv2.polylines(img, [pts], len(points) > 2, (0, 215, 255), 2)
            for p in points:
                cv2.circle(img, tuple(p), 4, (0, 0, 255), -1)
        cv2.putText(img, f"{args.name} ({args.type})  points: {len(points)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
        cv2.imshow("zone", img)
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and len(points) >= 3:
            print(f'  - name: "{args.name}"\n    type: {args.type}\n    points: {points}')
        elif key in (ord("s"), ord("S")):
            cv2.imwrite("outputs/zone_frame.png", img)
            print("saved outputs/zone_frame.png")
        elif key == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
