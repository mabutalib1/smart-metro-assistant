"""Staff dashboard: live annotated video + alert feed.

Run from the project root:
    streamlit run app/dashboard.py
"""
import sys
import tempfile
import time
from pathlib import Path

import cv2
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.pipeline import StationPipeline, open_source  # noqa: E402

st.set_page_config(page_title="Smart Metro Station Assistant", layout="wide")
st.title("Smart Metro Station Assistant")

cfg = load_config()
with st.sidebar:
    st.header("Camera")
    cfg["camera"]["gate"] = st.text_input("Gate name", cfg["camera"]["gate"])
    mode = st.radio("Source", ["Webcam", "Video file"])
    source = None
    if mode == "Webcam":
        source = st.number_input("Webcam index", 0, 5, 0)
    else:
        up = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
        if up:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix)
            tmp.write(up.read())
            source = tmp.name
    run = st.toggle("Run", value=False)

col_video, col_alerts = st.columns([3, 2])
video_slot = col_video.empty()
col_alerts.subheader("Alerts")
alert_slot = col_alerts.empty()

ICON = {"high": "🔴", "medium": "🟠"}


def render_alerts(history):
    if not history:
        alert_slot.info("No alerts yet.")
        return
    lines = [f"{ICON.get(a.severity, '')} **{a.time[11:]}**  {a.message}  (ID {a.track_id})"
             for a in reversed(history[-20:])]
    alert_slot.markdown("\n\n".join(lines))


if run and source is not None:
    pipe = StationPipeline(cfg)
    cap = open_source(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    is_file = mode == "Video file"
    idx, t0 = 0, time.time()
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps if is_file else time.time() - t0
        out, alerts = pipe.process(frame, t)
        video_slot.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)
        if alerts:
            for a in alerts:
                st.toast(a.message)
        render_alerts(pipe.alerts.history)
        idx += 1
    cap.release()
    st.success("Stream ended.")
else:
    video_slot.info("Choose a source and switch on Run.")
    render_alerts([])
