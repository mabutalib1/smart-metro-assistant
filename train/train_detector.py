"""Fine-tune YOLO on the merged mobility-aid dataset.

Usage (Kaggle / Colab GPU recommended):
    python train/train_detector.py --model yolo11s.pt --epochs 100
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "merged" / "data.yaml"))
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=20,
        project=str(ROOT / "runs"),
        name="aids",
        exist_ok=True,
    )

    metrics = model.val(data=args.data)
    print("\nPer-class mAP50-95:")
    for i, name in model.names.items():
        print(f"  {name:14} {metrics.box.maps[i]:.3f}")

    best = ROOT / "runs" / "aids" / "weights" / "best.pt"
    dest = ROOT / "models" / "aids_best.pt"
    dest.parent.mkdir(exist_ok=True)
    shutil.copy(best, dest)
    print(f"\nSaved {dest}. Set models.aids: models/aids_best.pt in configs/config.yaml")


if __name__ == "__main__":
    main()
