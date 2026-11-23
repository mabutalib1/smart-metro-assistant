# Smart Metro Station Assistant

An AI layer for existing station cameras that spots passengers who need help or are in danger, and alerts staff with the exact location.

## Problem

Stations already have extensive CCTV coverage, but people watch the feeds and cannot follow every screen at once. A wheelchair user or a blind passenger may wait without help, a passenger who falls may not be seen for minutes, and unsafe behavior near stairs or restricted areas is caught late.

## Solution

The system runs on the existing camera feeds and sends graded alerts such as **"White cane user approaching Stairs at Gate 3"**, so staff go straight to the right place.

| Level | Situation | Example alert |
| --- | --- | --- |
| Info | Wheelchair, white cane, crutches or walking frame user detected | Wheelchair user at Gate 3, may need assistance |
| Medium | Aid user has not moved for a while (outside seating areas) | Wheelchair user has not moved for a while at Gate 3 |
| Medium | Wheelchair / walking frame user at an escalator or narrow gate | Wheelchair user at Escalator, guide to the elevator |
| Medium | Aid user moving slowly with many direction changes | White cane user seems lost at Gate 3 |
| High | White cane user inside or walking toward stairs / platform edge | White cane user approaching Stairs at Gate 3 |
| High | Anyone holding a hand above their head | Passenger is asking for help at Gate 3 |
| High | Fall: sudden drop, body horizontal, stays down | Fall detected at Gate 3, send help now |
| High | Anyone inside a restricted zone | Person inside Restricted area at Gate 3 |

If another person stays next to an aid user, the system marks them as being helped and stops the medium alerts.

## How it works

```
Camera ──► YOLO11-pose + ByteTrack ──► Rule engine ──► Alert manager ──► Dashboard
           (people, keypoints, IDs)        ▲            (levels, one alert
                                           │             per person, cooldown,
           YOLO11 mobility-aid detector ───┘             blurred snapshots)
           (wheelchair, white cane, crutches, walking frame)
```

- **Pose model** (pretrained, no training): finds each person and 17 body keypoints.
- **Tracker**: gives each person a stable ID, so every rule judges one person's history.
- **Mobility-aid detector** (trained by us): each aid box is linked to the person it overlaps most.
- **Rules** use distances relative to body height, so the same thresholds work near and far from the camera. All thresholds are in `configs/config.yaml`.
- **Zones** are drawn per camera: `restricted`, `hazard`, `barrier`, `seating`.
- **Privacy**: frames are processed in memory. Only alert snapshots are saved, with the head region blurred.

## Project structure

```
configs/config.yaml          thresholds, zones, camera and gate name
src/pipeline.py              detection, tracking, rules, alerts, drawing
src/rules.py                 all detection rules
src/alerts.py                alert levels, messages, cooldown, log, snapshots
app/dashboard.py             Streamlit staff dashboard
data/merge_datasets.py       download Roboflow datasets and merge into one YOLO dataset
train/train_detector.py      fine-tune the mobility-aid detector
notebooks/train_kaggle.ipynb training on a free Kaggle GPU
tools/draw_zone.py           draw zones on a camera frame
tools/evaluate.py            event-level evaluation on labeled clips
tests/test_rules.py          rule checks on synthetic data
docs/data_collection.md      how to record and label our own data
docs/demo_guide.md           live demo setup and script
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests             # 20 rule checks
```

## Usage

**Run on a video or webcam.** Fall, zone and raised-hand alerts work right away with the pretrained pose model:

```bash
python -m src.pipeline --source path/to/video.mp4 --show
python -m src.pipeline --source 0 --show --save outputs/demo.mp4
```

**Draw the zones** for your camera and paste the printed YAML under `zones:` in the config:

```bash
python tools/draw_zone.py --source path/to/video.mp4 --name "Stairs" --type hazard
```

**Train the mobility-aid detector:**

```bash
export ROBOFLOW_API_KEY=your_key
python data/merge_datasets.py
python train/train_detector.py --model yolo11s.pt --epochs 100
```

or run `notebooks/train_kaggle.ipynb` on Kaggle. Then set `models.aids: models/aids_best.pt` in the config.

**Dashboard:**

```bash
streamlit run app/dashboard.py
```

**Evaluate** on our labeled test clips (see `docs/data_collection.md`):

```bash
python tools/evaluate.py --clips data/test_clips --labels data/test_clips/labels.csv
```

## Datasets

| Dataset | Used for |
| --- | --- |
| [Roboflow: Wheelchair Detection](https://universe.roboflow.com/mobilityaids/wheelchair-detection-hh3io) | Wheelchairs |
| [Roboflow: Mobility Aids](https://universe.roboflow.com/disable-people-on-crutches/mobility-aids-jcnno-1cyew) | Wheelchairs, crutches, walking frames |
| [Roboflow: DPI_Cane](https://universe.roboflow.com/elishian/dpi_cane) | White canes |
| [Roboflow: blind-guide-ai](https://universe.roboflow.com/visionaid-7mkcb/blind-guide-ai) | White canes |
| [Freiburg MobilityAids](http://mobility-aids.informatik.uni-freiburg.de/) | Extra indoor images (manual download) |
| [GMDCSA24](https://doi.org/10.5281/zenodo.13354453) | Testing fall detection |
| [OmniFall](https://huggingface.co/datasets/simplexsigil2/omnifall) | Fall detection benchmark |
| Our own recordings | High-angle cane images and labeled test clips |

## Results

| Metric | Value |
| --- | --- |
| Detector mAP50 (all classes) | _to fill after training_ |
| White cane mAP50 | _to fill_ |
| Fall recall / false alarms per minute | _to fill from tools/evaluate.py_ |
| Mean alert delay | _to fill_ |
| Processing speed | _to fill_ |

## Limitations

- Public datasets are mostly eye level, while station cameras are mounted high. Our own high-angle images reduce this gap.
- White cane data is limited, so cane detection is weaker than wheelchair detection.
- Dense crowds cause occlusion and can break track IDs.
- The system cannot know for certain that someone needs help. It points staff to likely cases, and staff decide.
