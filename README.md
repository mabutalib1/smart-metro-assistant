# Smart Metro Station Assistant

An AI layer for existing station cameras that detects passengers who need help or are in danger and alerts staff with the exact location.

## Problem

Metro stations already have extensive CCTV coverage, but the feeds are watched by people. Staff cannot follow every screen at once, so a wheelchair user or a blind passenger may wait without help, a passenger who falls may not be seen for minutes, and unsafe behavior near restricted areas is caught late.

## Solution

The system runs on the existing camera feeds and raises an alert such as **"Wheelchair user needs assistance at Gate 3"** so staff go straight to the right place.

| Situation | Alert |
| --- | --- |
| Wheelchair user | Assistance needed at the gate |
| White cane user | Assistance needed at the gate |
| Person falls | Fall detected, send help |
| Person enters a restricted zone | Safety warning |

## How it works

```
Camera ──► YOLO-pose + ByteTrack ──► Rule engine ──► Alert manager ──► Dashboard
                (people + IDs)          ▲              (one alert per
                                        │               person, cooldown)
           Mobility-aid detector ───────┘
           (wheelchair, cane, crutches, walking frame)
```

- **Restricted zone:** the midpoint of the ankle keypoints must stay inside the zone polygon for several frames.
- **Fall:** the hip drops sharply within about one second, the torso turns close to horizontal, and the person stays down.
- **Mobility aid:** an aid box is linked to the person it overlaps most, and must stay linked for several frames.
- **Privacy:** frames are processed in memory. Only alert snapshots are saved, with the head region blurred.

## Project structure

```
configs/config.yaml        thresholds, zone polygon, camera and gate name
src/pipeline.py            main loop: detection, tracking, rules, alerts, drawing
src/rules.py               zone, fall and mobility-aid logic
src/alerts.py              alert cooldown, logging, snapshots
data/merge_datasets.py     download Roboflow datasets and merge into one YOLO dataset
train/train_detector.py    fine-tune the mobility-aid detector
tools/draw_zone.py         click the zone polygon on a camera frame
app/dashboard.py           Streamlit staff dashboard
tests/test_rules.py        rule checks on synthetic data
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**1. Run on a video or webcam** (fall and zone detection work right away with the pretrained pose model):

```bash
python -m src.pipeline --source path/to/video.mp4 --show
python -m src.pipeline --source 0 --show --save outputs/demo.mp4
```

**2. Set the zone** for your camera, then paste the points into `configs/config.yaml`:

```bash
python tools/draw_zone.py --source path/to/video.mp4
```

**3. Prepare the dataset** (free API key from roboflow.com):

```bash
export ROBOFLOW_API_KEY=your_key
python data/merge_datasets.py
```

**4. Train the mobility-aid detector** (GPU recommended, Kaggle or Colab):

```bash
python train/train_detector.py --model yolo11s.pt --epochs 100
```

Then set `models.aids: models/aids_best.pt` in the config.

**5. Open the dashboard:**

```bash
streamlit run app/dashboard.py
```

**Tests:**

```bash
python -m pytest tests
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

## Evaluation

- Detector: mAP50 and mAP50-95 per class on the validation split.
- Events: recall (incidents caught / total) and false alarms per minute on our own recorded test clips.
- Alert delay in seconds and frames per second on the demo machine.

## Limitations

- Public datasets are mostly eye-level; station cameras are mounted high. We add our own high-angle images to close this gap.
- White cane data is limited, so cane detection is weaker than wheelchair detection.
- Dense crowds cause occlusion and can break track IDs.
