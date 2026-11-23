# Data collection and labeling

Public datasets are mostly shot at eye level, while station cameras look down from the ceiling. Our own data closes that gap and gives us honest test results.

## 1. Training images (white cane focus)

Goal: 200–300 labeled images, mostly white canes, plus some wheelchairs.

- Mount the phone or camera high, 2.5–3 m if possible, angled down like a CCTV camera.
- Vary distance, background, lighting and clothing. Several team members should take turns.
- Hold the cane in different positions: sweeping, resting, held across the body.
- Include hard negatives: umbrellas, walking sticks, selfie sticks, broom handles.
- Record short videos and extract every 10th frame instead of taking photos one by one:

```bash
mkdir -p frames && ffmpeg -i cane_01.mp4 -vf "select=not(mod(n\,10))" -vsync vfr frames/cane_01_%04d.jpg
```

**Labeling:** upload to a Roboflow project, draw boxes with class names exactly `white_cane`, `wheelchair`, `crutches`, `walking_frame`, then export in YOLOv8 format into `data/raw/own_images/` and run:

```bash
python data/merge_datasets.py --skip-download
```

## 2. Test clips (for evaluation)

Goal: 10–20 clips of 20–60 seconds, recorded with the same high camera setup. Never use these for training.

| Clip | What happens | Label rows |
| --- | --- | --- |
| fall_01..03 | Person walks, falls onto a mat, stays down | `fall` |
| fake_fall_01 | Person sits down quickly / bends to tie a shoe | none (false alarm test) |
| hand_01 | Person raises a hand for 3+ seconds | `help_request` |
| zone_01 | Person crosses into the restricted zone | `zone_entry` |
| wheelchair_01 | Wheelchair user enters and waits 30 s | `aid_detected`, `stuck` |
| wheelchair_escalator | Wheelchair user moves to the escalator zone | `aid_detected`, `barrier` |
| cane_stairs | White cane user walks toward the stairs zone | `aid_detected`, `hazard` |
| cane_lost | White cane user moves back and forth slowly | `aid_detected`, `disoriented` |
| helped_01 | Wheelchair user with a helper standing next to them | `aid_detected` only |
| normal_01..03 | People walking normally, no incidents | none |

Safety: falls only onto a thick mat, slowly and controlled. Nobody goes near real tracks, stairs or escalators; mark those zones on the floor with tape.

## 3. Labels file

Put the clips in `data/test_clips/` and create `data/test_clips/labels.csv`. One row per incident per person. Times are seconds from the start of the clip; watch the clip in VLC to read them.

```csv
clip,event,start,end
fall_01.mp4,fall,4.2,9.0
fake_fall_01.mp4,,,
wheelchair_01.mp4,aid_detected,0.0,5.0
wheelchair_01.mp4,stuck,25.0,40.0
cane_stairs.mp4,aid_detected,0.0,5.0
cane_stairs.mp4,hazard,6.0,10.0
normal_01.mp4,,,
```

- `start` is when the situation begins (the fall starts, the person reaches the zone).
- `end` is the latest time an alert is still useful.
- A clip with no incidents gets one row with empty event, start and end.

Run the evaluation:

```bash
python tools/evaluate.py --clips data/test_clips --labels data/test_clips/labels.csv
```

Remember to draw the zones for the test camera position first (`tools/draw_zone.py`).
