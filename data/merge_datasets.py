"""Download the Roboflow datasets and merge them into one YOLO dataset.

Target classes: wheelchair, crutches, walking_frame, white_cane.
Person boxes are dropped: people come from the pretrained pose model.

Usage:
    export ROBOFLOW_API_KEY=your_key
    python data/merge_datasets.py                 # download + merge
    python data/merge_datasets.py --skip-download # merge already downloaded folders

Check each dataset's version number on its Roboflow page and edit SOURCES if needed.
Any source class not in CLASS_MAP is printed and dropped, so you can extend the map.
"""
from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "merged"

TARGET = ["wheelchair", "crutches", "walking_frame", "white_cane"]

# source class name (lowercase) -> target class
CLASS_MAP = {
    "wheelchair": "wheelchair",
    "wheel chair": "wheelchair",
    "push_wheelchair": "wheelchair",
    "peoplewithwheelchair": "wheelchair",
    "crutches": "crutches",
    "crutch": "crutches",
    "walking_frame": "walking_frame",
    "walkingframe": "walking_frame",
    "walker": "walking_frame",
    "rollator": "walking_frame",
    "white_cane": "white_cane",
    "white cane": "white_cane",
    "whitecane": "white_cane",
    "white-cane": "white_cane",
}

# (workspace, project, version)
SOURCES = [
    ("mobilityaids", "wheelchair-detection-hh3io", 7),
    # ("disable-people-on-crutches", "mobility-aids-jcnno-1cyew", 1),
    ("elishian", "dpi_cane", 3),
    ("visionaid-7mkcb", "blind-guide-ai", 1),
]

SPLITS = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}


def download():
    from roboflow import Roboflow

    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("Set ROBOFLOW_API_KEY first (free account at roboflow.com).")
    rf = Roboflow(api_key=key)
    RAW.mkdir(parents=True, exist_ok=True)
    for ws, proj, ver in SOURCES:
        dest = RAW / proj
        if (dest / "data.yaml").exists():
            print(f"skip {proj} (already downloaded)")
            continue
        if dest.exists():
            shutil.rmtree(dest)          # remove a half-finished download
        print(f"downloading {ws}/{proj} v{ver}")
        rf.workspace(ws).project(proj).version(ver).download("yolov8", location=str(dest))


def merge():
    if OUT.exists():
        shutil.rmtree(OUT)
    counts = {s: Counter() for s in ("train", "valid", "test")}
    unmapped = Counter()

    for ds in sorted(p for p in RAW.iterdir() if p.is_dir()):
        meta = ds / "data.yaml"
        if not meta.exists():
            print(f"no data.yaml in {ds}, skipping")
            continue
        names = yaml.safe_load(meta.read_text())["names"]
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        remap = {}
        for i, n in enumerate(names):
            target = CLASS_MAP.get(str(n).strip().lower())
            if target:
                remap[i] = TARGET.index(target)
            else:
                unmapped[f"{ds.name}:{n}"] += 1

        for split_dir in ds.iterdir():
            split = SPLITS.get(split_dir.name)
            if not split or not (split_dir / "labels").exists():
                continue
            (OUT / split / "images").mkdir(parents=True, exist_ok=True)
            (OUT / split / "labels").mkdir(parents=True, exist_ok=True)
            for n, lbl in enumerate(sorted((split_dir / "labels").glob("*.txt"))):
                lines = []
                for row in lbl.read_text().splitlines():
                    parts = row.split()
                    if len(parts) < 5:
                        continue
                    c = int(parts[0])
                    if c in remap:
                        lines.append(" ".join([str(remap[c])] + parts[1:]))
                        counts[split][TARGET[remap[c]]] += 1
                img = next((split_dir / "images").glob(lbl.stem + ".*"), None)
                if img is None:
                    continue
                stem = f"{ds.name[:20]}_{split}_{n:06d}"   # short names avoid Windows path limits
                shutil.copy(img, OUT / split / "images" / f"{stem}{img.suffix}")
                (OUT / split / "labels" / f"{stem}.txt").write_text("\n".join(lines))

    data = {"path": str(OUT), "train": "train/images", "val": "valid/images",
            "names": {i: n for i, n in enumerate(TARGET)}}
    if (OUT / "test" / "images").exists():
        data["test"] = "test/images"
    (OUT / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

    print("\nBoxes per class:")
    for split, c in counts.items():
        print(f"  {split:5}: " + ", ".join(f"{n}={c[n]}" for n in TARGET))
    if unmapped:
        print("\nDropped source classes (add to CLASS_MAP if needed):")
        for k in sorted(unmapped):
            print(f"  {k}")
    print(f"\nWrote {OUT / 'data.yaml'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()
    if not args.skip_download:
        download()
    merge()
