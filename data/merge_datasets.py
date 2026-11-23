"""Download the Roboflow datasets and merge them into one YOLO dataset.

Target classes: wheelchair, crutches, walking_frame, white_cane.
Person boxes are dropped: people come from the pretrained pose model.

Usage:
    export ROBOFLOW_API_KEY=your_key
    python data/merge_datasets.py                 # download + merge
    python data/merge_datasets.py --skip-download # merge already downloaded folders

The latest published version of each dataset is used. Datasets without a published
version are skipped. Images are re-split 80/15/5 per dataset with a fixed seed.
Any source class not in CLASS_MAP is printed and dropped, so you can extend the map.
"""
from __future__ import annotations

import argparse
import os
import random
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
    "person_wheelchair": "wheelchair",
    "person_crutches": "crutches",
    "person_walking_frame": "walking_frame",
    "crutches": "crutches",
    "crutch": "crutches",
    "walking_frame": "walking_frame",
    "walkingframe": "walking_frame",
    "walker": "walking_frame",
    "rollator": "walking_frame",
    "white_cane": "white_cane",
    "white cane": "white_cane",
    "whitecane": "white_cane",
    "white_canes": "white_cane",
    "white-canes": "white_cane",
    "white-cane": "white_cane",
}

# Class names that only make sense inside one dataset: dataset folder -> {source name: target}
DATASET_CLASS_MAP = {
    "whitecanes": {"0": "white_cane"},     # single unnamed class in a white cane dataset
}

# (workspace, project, version). version None = latest published version.
# A dataset that cannot be downloaded (no published version, removed, ...) is skipped.
SOURCES = [
    ("mobilityaids", "wheelchair-detection-hh3io", 7),
    # ("disable-people-on-crutches", "mobility-aids-jcnno-1cyew", 1),   # no published version
    ("elishian", "dpi_cane", 3),
    ("visionaid-7mkcb", "blind-guide-ai", 1),
    ("project-4v0dw", "white_cane_6", None),
    ("tra-nguyen1", "white-cane-3", None),
    ("miso", "whitecanes", None),
]

SPLIT_RATIO = (0.80, 0.15, 0.05)    # train / valid / test, applied per dataset
MAX_NEGATIVES = 0.15                # images with no aid boxes, as a share of images with boxes
SEED = 42

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
        try:
            project = rf.workspace(ws).project(proj)
            if ver is None:
                versions = project.versions()
                if not versions:
                    print(f"SKIP {ws}/{proj}: no published version (fork it in Roboflow to use it)")
                    continue
                ver = max(int(str(v.version).rstrip("/").split("/")[-1]) for v in versions)
            print(f"downloading {ws}/{proj} v{ver}")
            project.version(ver).download("yolov8", location=str(dest))
        except Exception as e:  # keep going with the other datasets
            print(f"SKIP {ws}/{proj}: {e}")
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)


def merge():
    if OUT.exists():
        shutil.rmtree(OUT)
    rng = random.Random(SEED)
    counts = {s: Counter() for s in ("train", "valid", "test")}
    images = {s: 0 for s in ("train", "valid", "test")}
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
            key = str(n).strip().lower()
            target = DATASET_CLASS_MAP.get(ds.name, {}).get(key) or CLASS_MAP.get(key)
            if target:
                remap[i] = TARGET.index(target)
            else:
                unmapped[f"{ds.name}:{n}"] += 1

        # collect every labeled image from all of the dataset's splits
        positives, negatives = [], []
        for split_dir in ds.iterdir():
            if split_dir.name not in SPLITS or not (split_dir / "labels").exists():
                continue
            for lbl in sorted((split_dir / "labels").glob("*.txt")):
                img = next((split_dir / "images").glob(lbl.stem + ".*"), None)
                if img is None:
                    continue
                lines = []
                for row in lbl.read_text().splitlines():
                    parts = row.split()
                    if len(parts) >= 5 and int(parts[0]) in remap:
                        lines.append(" ".join([str(remap[int(parts[0])])] + parts[1:]))
                (positives if lines else negatives).append((img, lines))

        # keep a limited number of images without aids (they teach the model what is NOT an aid)
        rng.shuffle(negatives)
        items = positives + negatives[: int(len(positives) * MAX_NEGATIVES)]
        rng.shuffle(items)

        # our own split, so every dataset contributes to train, valid and test
        n_train = int(len(items) * SPLIT_RATIO[0])
        n_valid = int(len(items) * SPLIT_RATIO[1])
        parts = {"train": items[:n_train], "valid": items[n_train:n_train + n_valid],
                 "test": items[n_train + n_valid:]}
        for split, chunk in parts.items():
            (OUT / split / "images").mkdir(parents=True, exist_ok=True)
            (OUT / split / "labels").mkdir(parents=True, exist_ok=True)
            for n, (img, lines) in enumerate(chunk):
                stem = f"{ds.name[:20]}_{split}_{n:06d}"   # short names avoid Windows path limits
                shutil.copy(img, OUT / split / "images" / f"{stem}{img.suffix}")
                (OUT / split / "labels" / f"{stem}.txt").write_text("\n".join(lines))
                for line in lines:
                    counts[split][TARGET[int(line.split()[0])]] += 1
            images[split] += len(chunk)
        print(f"{ds.name:32} {len(positives):6} images with aids, {len(negatives):6} without")

    data = {"path": str(OUT), "train": "train/images", "val": "valid/images", "test": "test/images",
            "names": {i: n for i, n in enumerate(TARGET)}}
    (OUT / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

    print("\nBoxes per class:")
    for split, c in counts.items():
        print(f"  {split:5} ({images[split]:5} images): " + ", ".join(f"{n}={c[n]}" for n in TARGET))
    if unmapped:
        print("\nDropped source classes (add to CLASS_MAP if needed):")
        for k in sorted(unmapped):
            print(f"  {k}")
    if counts["train"]["white_cane"] < 1000:
        print("\nNote: few white cane boxes. Add your own labeled images (docs/data_collection.md).")
    print(f"\nWrote {OUT / 'data.yaml'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()
    if not args.skip_download:
        download()
    merge()
