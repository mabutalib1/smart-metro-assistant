# Demo guide

A 5-minute live demo that shows every alert level.

## Setup (30 minutes before)

1. Mount the camera high (tripod at full height or taped to a shelf), angled down at a 3 × 3 m floor area.
2. Tape lines on the floor: one strip as the **Restricted area**, one area as **Stairs** (hazard), one as **Escalator** (barrier).
3. Put a thick mat inside the frame for the fall.
4. Draw the zones for this exact camera position and paste them into `configs/config.yaml`:

```bash
python tools/draw_zone.py --source 0 --name "Restricted area" --type restricted
python tools/draw_zone.py --source 0 --name "Stairs" --type hazard
python tools/draw_zone.py --source 0 --name "Escalator" --type barrier
```

5. Start the dashboard and check the FPS shown under the video (aim for 10+):

```bash
streamlit run app/dashboard.py
```

6. Record a full backup run with `python -m src.pipeline --source 0 --save outputs/backup.mp4` in case the live demo fails.

## Script

| Step | Action | Expected alert |
| --- | --- | --- |
| 1 | A team member walks across the frame normally | None (green box) |
| 2 | They step into the Restricted area | HIGH: Person inside Restricted area |
| 3 | They raise a hand for 3 seconds | HIGH: Passenger is asking for help |
| 4 | They lie down on the mat and stay there | HIGH: Fall detected, send help now |
| 5 | Another member enters with a wheelchair | INFO: Wheelchair user, may need assistance |
| 6 | The wheelchair moves into the Escalator zone | MEDIUM: Wheelchair user at Escalator, guide to the elevator |
| 7 | A member with a white cane walks toward the Stairs zone | HIGH: White cane user approaching Stairs |
| 8 | A second person stands next to the wheelchair user | Box shows "(helped)", no stuck alert |

## Talking points

- It runs on cameras the station already has; no new hardware.
- One alert per person per event, so staff are not flooded.
- Alerts are graded: info, medium, high. Staff decide; the system only points them to the right place.
- Privacy: frames stay in memory; only alert snapshots are saved, with the head blurred.
- Results from `tools/evaluate.py` on our own test clips: recall, false alarms per minute, alert delay.

## If something goes wrong

| Problem | Fix |
| --- | --- |
| Low FPS | Use `yolo11n` models and set `device: "0"` if a GPU is available |
| Person not detected | More light; make sure the whole body is in frame |
| Fall not detected | Fall must be fast; lower `rules.fall.drop_ratio` to 0.25 |
| Cane not detected | Hold the cane clearly visible; lower `models.aid_conf` to 0.25 |
| Live demo fails | Play `outputs/backup.mp4` through the dashboard (Video file) |
