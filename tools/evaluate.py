"""Event-level evaluation on labeled test clips.

Measures what matters for staff: how many real incidents are caught, how many
false alarms are raised, and how long an alert takes.

Labels CSV (one row per real incident, including one aid_detected row per
mobility-aid user in the clip; see docs/data_collection.md):
    clip,event,start,end
    fall_01.mp4,fall,3.2,6.0
    cane_stairs.mp4,hazard,4.0,8.5
    normal_walk.mp4,,,                 <- clip with no incidents (for false alarms)

Usage:
    python tools/evaluate.py --clips data/test_clips --labels data/test_clips/labels.csv
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.pipeline import StationPipeline  # noqa: E402

TOLERANCE = 2.0   # seconds an alert may come after the labeled end


def read_labels(path: Path) -> dict[str, list[dict]]:
    clips: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            clip = row["clip"].strip()
            clips[clip]
            if row.get("event", "").strip():
                clips[clip].append({"event": row["event"].strip(),
                                    "start": float(row["start"]), "end": float(row["end"])})
    return clips


def run_clip(cfg: dict, path: Path) -> tuple[list, float, float]:
    """Returns (alerts, clip duration in seconds, processing FPS)."""
    pipe = StationPipeline(cfg)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    idx, t_start = 0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pipe.process(frame, idx / fps)
        idx += 1
    cap.release()
    elapsed = time.perf_counter() - t_start
    return pipe.alerts.history, idx / fps, idx / max(elapsed, 1e-6)


def match(alerts: list, truth: list[dict]):
    """Greedy match: each labeled incident takes the first alert of the same kind in its window."""
    used, rows = set(), []
    for g in truth:
        hit = None
        for i, a in enumerate(alerts):
            if i in used or a.event != g["event"]:
                continue
            if g["start"] - 0.5 <= a.t <= g["end"] + TOLERANCE:
                hit = i
                break
        if hit is not None:
            used.add(hit)
        rows.append((g, alerts[hit] if hit is not None else None))
    false_alarms = [a for i, a in enumerate(alerts) if i not in used]
    return rows, false_alarms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="folder with the test videos")
    ap.add_argument("--labels", required=True, help="labels CSV")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="outputs/evaluation.csv")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    cfg = copy.deepcopy(cfg)
    cfg["alerts"]["save_snapshots"] = False
    cfg["alerts"]["log_file"] = None

    labels = read_labels(Path(args.labels))
    stats = defaultdict(lambda: {"incidents": 0, "caught": 0, "delays": [], "false_alarms": 0})
    total_minutes, fps_list, detail = 0.0, [], []

    for clip, truth in labels.items():
        path = Path(args.clips) / clip
        if not path.exists():
            print(f"missing clip: {path}")
            continue
        alerts, duration, fps = run_clip(cfg, path)
        total_minutes += duration / 60
        fps_list.append(fps)
        rows, false_alarms = match(alerts, truth)
        for g, a in rows:
            s = stats[g["event"]]
            s["incidents"] += 1
            if a:
                s["caught"] += 1
                s["delays"].append(max(0.0, a.t - g["start"]))
            detail.append([clip, g["event"], g["start"], g["end"], a.t if a else "", "caught" if a else "missed"])
        for a in false_alarms:
            stats[a.event]["false_alarms"] += 1
            detail.append([clip, a.event, "", "", a.t, "false alarm"])
        print(f"{clip:30} {len(truth)} incidents, {len(alerts)} alerts, {fps:5.1f} FPS")

    print(f"\n{'Event':14} {'Incidents':>9} {'Caught':>7} {'Recall':>7} {'False':>6} {'FA/min':>7} {'Delay s':>8}")
    for event, s in sorted(stats.items()):
        recall = s["caught"] / s["incidents"] if s["incidents"] else float("nan")
        delay = sum(s["delays"]) / len(s["delays"]) if s["delays"] else float("nan")
        fa_min = s["false_alarms"] / total_minutes if total_minutes else 0.0
        print(f"{event:14} {s['incidents']:9} {s['caught']:7} {recall:7.2f} {s['false_alarms']:6} {fa_min:7.2f} {delay:8.2f}")
    if fps_list:
        print(f"\nTotal video: {total_minutes:.1f} min | mean processing speed: {sum(fps_list) / len(fps_list):.1f} FPS")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip", "event", "label_start", "label_end", "alert_time", "result"])
        w.writerows(detail)
    print(f"Details written to {out}")


if __name__ == "__main__":
    main()
