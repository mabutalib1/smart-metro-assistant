"""Staff dashboard: live annotated video, alert counters and alert feed.

Run from the project root:
    streamlit run app/dashboard.py
"""
import json
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.pipeline import StationPipeline, is_live, open_source  # noqa: E402

st.set_page_config(page_title="Smart Metro Station Assistant", layout="wide")

LEVEL_STYLE = {
    "high": ("#d93025", "HIGH"),
    "medium": ("#e8710a", "MEDIUM"),
    "info": ("#1a73e8", "INFO"),
}

st.markdown("""
<style>
.alert {border-left: 5px solid; padding: 6px 10px; margin-bottom: 6px; border-radius: 4px;
        background: rgba(128,128,128,0.08); font-size: 0.92rem;}
.alert b {font-size: 0.75rem; letter-spacing: 0.04em;}
.alert small {opacity: 0.7;}
</style>
""", unsafe_allow_html=True)

st.title("Smart Metro Station Assistant")

cfg = load_config()
with st.sidebar:
    st.header("Camera")
    cfg["camera"]["gate"] = st.text_input("Gate name", cfg["camera"]["gate"])
    mode = st.radio("Source", ["Video file", "Webcam", "Stream URL"])
    source = None
    if mode == "Webcam":
        source = int(st.number_input("Webcam index", 0, 5, 0))
    elif mode == "Stream URL":
        source = st.text_input("RTSP / HTTP URL") or None
    else:
        up = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
        if up:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix)
            tmp.write(up.read())
            source = tmp.name

    st.header("Alerts")
    show_levels = st.multiselect("Show levels", ["high", "medium", "info"], ["high", "medium", "info"])
    aids_model = cfg["models"].get("aids")
    if not aids_model or not Path(aids_model).exists():
        st.warning("No mobility-aid model configured: only fall, zone and raised-hand alerts are active.")
        cfg["models"]["aids"] = None
    run = st.toggle("Run", value=False)

col_video, col_alerts = st.columns([3, 2])
video_slot = col_video.empty()
status_slot = col_video.empty()
with col_alerts:
    c1, c2, c3 = st.columns(3)
    counters = {"high": c1.empty(), "medium": c2.empty(), "info": c3.empty()}
    st.subheader("Alert feed")
    feed_slot = st.empty()


def render(history):
    for level, slot in counters.items():
        slot.metric(LEVEL_STYLE[level][1].title(), sum(a.level == level for a in history))
    items = [a for a in reversed(history) if a.level in show_levels][:25]
    if not items:
        feed_slot.info("No alerts yet.")
        return
    html = []
    for a in items:
        color, tag = LEVEL_STYLE[a.level]
        html.append(f'<div class="alert" style="border-color:{color}"><b style="color:{color}">{tag}</b> '
                    f'{a.message}<br><small>{a.time[11:]} · person ID {a.track_id}</small></div>')
    feed_slot.markdown("".join(html), unsafe_allow_html=True)


if run and source is not None:
    pipe = StationPipeline(cfg)
    cap = open_source(source)
    if not cap.isOpened():
        st.error(f"Cannot open source: {source}")
        st.stop()
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    live = is_live(source)
    idx, t0 = 0, time.time()
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        t = time.time() - t0 if live else idx / fps
        out, alerts = pipe.process(frame, t)
        video_slot.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), width="stretch")
        status_slot.caption(f"Processing at {pipe.fps:.1f} FPS · people tracked: {len(pipe.rules.tracks)}")
        for a in alerts:
            if a.level == "high":
                st.toast(a.message)
        render(pipe.alerts.history)
        idx += 1
    cap.release()
    st.success("Stream ended.")
    log = "\n".join(json.dumps(asdict(a), ensure_ascii=False) for a in pipe.alerts.history)
    st.download_button("Download alert log", log, file_name="alerts.jsonl")
else:
    video_slot.info("Choose a source in the sidebar and switch on Run.")
    render([])
