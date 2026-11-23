"""Click points on a camera frame to define the restricted zone polygon.

Usage:
    python tools/draw_zone.py --source path/to/video.mp4

Left click: add point. Right click: undo. Enter: print the YAML. Esc: quit.
Paste the printed points under zones.restricted in configs/config.yaml.
"""
import argparse

import cv2
import numpy as np

points = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN and points:
        points.pop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    args = ap.parse_args()
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Could not read a frame from the source.")

    cv2.namedWindow("zone")
    cv2.setMouseCallback("zone", on_mouse)
    while True:
        img = frame.copy()
        if points:
            pts = np.array(points, np.int32)
            cv2.polylines(img, [pts], len(points) > 2, (0, 215, 255), 2)
            for p in points:
                cv2.circle(img, tuple(p), 4, (0, 0, 255), -1)
        cv2.imshow("zone", img)
        key = cv2.waitKey(20) & 0xFF
        if key == 13:
            print("zones:\n  restricted:")
            for x, y in points:
                print(f"    - [{x}, {y}]")
        elif key == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
