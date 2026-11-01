"""WasteSep dashboard: detection recordings, spectral readings, the session
reference and a live camera view.

    pip install "streamlit>=1.50"
    streamlit run dashboard.py --server.address 0.0.0.0    # then open http://<pi>:8501

Start main.py from the same directory, so both use the same logs/,
detection_recordings/ and session_reference.json (or point the sidebar field,
or WASTESEP_RUN_DIR, at main.py's directory). The Live tab shows the frames
main.py writes while its Orbbec source is open; the camera itself can only be
opened by one process.
"""

import base64
import io
import os
import time

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

import contamination_config as cfg
import dashboard_data as data
from capture import LIVE_FRAME_PATH

HERE = os.path.dirname(os.path.abspath(__file__))

# Reference palette of the dataviz skill: categorical slots 1-2 (validated
# light and dark) and the fixed status steps, which always carry a text label.
SERIES = {"light": ("#2a78d6", "#eb6834"), "dark": ("#3987e5", "#d95926")}
INK = {"light": "#0b0b0b", "dark": "#f0efec"}
STATUS = {"clean": "#0ca30c", "undetermined": "#fab219", "veto": "#d03b3b"}
ITEM, CLEAN = "this item", "clean reference"

LIVE_REFRESH_S = 0.2
LIVE_STALE_S = 3.0
LIVE_PREVIEW_WIDTH = 960      # px; frames are inlined in every update, so keep them small
LIVE_PREVIEW_QUALITY = 75
LIVE_MAX_HEIGHT_VH = 75       # the frame never makes the Live tab scroll

STATE_LABEL = {True: "⛔ contaminated", False: "✅ clean", None: "⚠️ undetermined"}


def theme():
    kind = getattr(getattr(st.context, "theme", None), "type", None)
    return "dark" if kind == "dark" else "light"


@st.cache_data(show_spinner=False)
def _jsonl(path, stamp):
    return data.read_jsonl(path)


def load_jsonl(path):
    """Re-read a log only when it has changed."""
    try:
        s = os.stat(path)
        stamp = (s.st_mtime_ns, s.st_size)
    except FileNotFoundError:
        stamp = None
    return _jsonl(path, stamp)


def fmt(value, spec, missing="—"):
    return missing if value is None else format(value, spec)


# --- Charts ---------------------------------------------------------------------------

EXCLUDED = [ch for ch in data.WAVELENGTH_NM if ch not in cfg.LEVEL_CHANNELS]


def spectrum_chart(rows, legend=True, show_excluded=False):
    """Mean calibrated counts per channel against wavelength. The line joins
    the 13 level channels; the excluded ones, when shown, are faded points
    over a shaded band (they can be far off scale, so they are opt-in)."""
    df = pd.DataFrame(rows)
    if not show_excluded:
        df = df[df["in_level"]]
    x_scale = alt.Scale(domain=[400, 950 if show_excluded else 740], nice=False)
    colour = alt.Scale(domain=[ITEM, CLEAN], range=list(SERIES[theme()]))
    base = alt.Chart(df).encode(
        x=alt.X("nm:Q", title="wavelength (nm)", scale=x_scale),
        y=alt.Y("counts:Q", title="calibrated counts (mean of readings)"),
        color=alt.Color("series:N", scale=colour,
                        legend=alt.Legend(title=None, orient="top") if legend else None))
    lines = base.transform_filter(alt.datum.in_level).mark_line(strokeWidth=2)
    points = base.mark_point(filled=True, size=64).encode(
        opacity=alt.condition(alt.datum.in_level, alt.value(1.0), alt.value(0.35)),
        tooltip=[alt.Tooltip("series:N"), alt.Tooltip("channel:N"),
                 alt.Tooltip("nm:Q", title="nm"), alt.Tooltip("counts:Q", format=",.1f"),
                 alt.Tooltip("in_level:N", title="in level")])
    layers = [lines, points]
    if show_excluded:
        band = pd.DataFrame([{"lo": min(data.WAVELENGTH_NM[ch] for ch in EXCLUDED) - 10, "hi": 950,
                              "label": "not in level ({})".format(" ".join(EXCLUDED))}])
        shade = alt.Chart(band).mark_rect(opacity=0.08, color="gray").encode(
            x=alt.X("lo:Q", scale=x_scale), x2="hi:Q")
        shade_label = alt.Chart(band).mark_text(align="left", baseline="top", dx=4, dy=4,
                                                color="gray").encode(
            x=alt.X("lo:Q", scale=x_scale), y=alt.value(0), text="label:N")
        layers = [shade, shade_label] + layers
    return alt.layer(*layers).properties(height=320)


def d_chart(d, substrate):
    """Where d falls against the substrate's clean / undetermined / veto bands."""
    span = max(0.3, abs(d) * 1.2)
    # An explicit 0-1 y range: without a y channel the plot collapses to a sliver.
    y = alt.Y("y0:Q", scale=alt.Scale(domain=[0, 1]), axis=None)
    x_scale = alt.Scale(domain=[-span, span], nice=False)
    zones = pd.DataFrame([
        {"lo": -span if lo is None else lo, "hi": span if hi is None else hi, "zone": zone,
         "y0": 0, "y1": 1}
        for lo, hi, zone in data.d_zones(substrate)])
    bands = alt.Chart(zones).mark_rect(opacity=0.35).encode(
        x=alt.X("lo:Q", scale=x_scale, title="d = (clean_ref − level) / clean_ref"),
        x2="hi:Q", y=y, y2="y1:Q",
        color=alt.Color("zone:N", scale=alt.Scale(domain=list(STATUS), range=list(STATUS.values())),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("zone:N"), alt.Tooltip("lo:Q", format="+.3f"),
                 alt.Tooltip("hi:Q", format="+.3f")])
    point = pd.DataFrame([{"d": d, "y0": 0, "y1": 1, "label_y": 0.5}])
    ink = INK[theme()]
    marker = alt.Chart(point).mark_rule(strokeWidth=3, color=ink).encode(
        x=alt.X("d:Q", scale=x_scale), y=y, y2="y1:Q",
        tooltip=alt.Tooltip("d:Q", format="+.3f"))
    label = alt.Chart(point).mark_text(align="right" if d > 0 else "left", dx=-8 if d > 0 else 8,
                                       color=ink, fontWeight="bold").encode(
        x=alt.X("d:Q", scale=x_scale), y=alt.Y("label_y:Q", scale=alt.Scale(domain=[0, 1]), axis=None),
        text=alt.Text("d:Q", format="+.3f"))
    # Streamlit autosizes with "fit": this height includes the legend and axis.
    return alt.layer(bands, marker, label).properties(height=150)


def history_chart(history):
    df = pd.DataFrame([{"taken": pd.to_datetime(data.local_time(r["timestamp"])),
                        "clean_ref_level": r["clean_ref_level"]} for r in history])
    colour = SERIES[theme()][1]
    base = alt.Chart(df).encode(
        x=alt.X("taken:T", title="calibration time", axis=alt.Axis(format="%m-%d %H:%M")),
        y=alt.Y("clean_ref_level:Q", title="clean_ref_level",
                scale=alt.Scale(zero=False)))
    lines = base.mark_line(strokeWidth=2, color=colour)
    points = base.mark_point(filled=True, size=64, color=colour).encode(
        tooltip=[alt.Tooltip("taken:T", format="%Y-%m-%d %H:%M"),
                 alt.Tooltip("clean_ref_level:Q", format=",.1f")])
    config_level = cfg.CONTAM_MODEL[history[-1]["substrate"]][0]
    rule = pd.DataFrame([{"level": config_level, "label": "config {:.1f}".format(config_level)}])
    config_rule = alt.Chart(rule).mark_rule(strokeDash=[4, 4], color="gray").encode(y="level:Q")
    config_label = alt.Chart(rule).mark_text(align="left", dx=4, dy=-6, color="gray").encode(
        y="level:Q", x=alt.value(0), text="label:N")
    return (config_rule + config_label + lines + points).properties(height=260)


# --- Tabs ---------------------------------------------------------------------------------

def video_status(recording):
    if recording is None:
        return "—"
    if recording["mp4"]:
        return "✓"
    return "encoding" if recording["pending"] else "—"


def detections_tab(p):
    spectral_records = load_jsonl(p["spectral"])
    dets = data.detections(load_jsonl(p["decisions"]), spectral_records,
                           data.recordings(p["recordings"]))
    if not dets:
        st.info("No detections logged yet in {}.".format(p["decisions"]))
        return

    cols = st.columns(4)
    cols[0].metric("Detections", len(dets))
    cols[1].metric("GREEN", sum(d.get("stream") == "green" for d in dets))
    cols[2].metric("CLEAR", sum(d.get("stream") == "clear" for d in dets))
    cols[3].metric("Contamination vetoes", sum(d.get("reason") == "contamination_veto" for d in dets))

    table = pd.DataFrame([{
        "time": data.local_time(d.get("timestamp")),
        "request_id": d.get("request_id"),
        "labels": ", ".join(d.get("labels") or []) or "—",
        "stream": (d.get("stream") or "").upper(),
        "reason": d.get("reason"),
        "d": (d.get("spectral") or {}).get("d"),
        "decision_s": d.get("decision_s"),
        "vlm_s": d.get("vlm_s"),
        "video": video_status(d["recording"]),
    } for d in dets])
    st.caption("Newest first. Select a row to inspect it; the newest is shown otherwise.")
    event = st.dataframe(
        table, hide_index=True, on_select="rerun", selection_mode="single-row", key="detections",
        height=min(38 + 35 * len(table), 300),
        column_config={
            "d": st.column_config.NumberColumn(format="%+.3f"),
            "decision_s": st.column_config.NumberColumn("decision (s)", format="%.2f"),
            "vlm_s": st.column_config.NumberColumn("vlm (s)", format="%.2f"),
        })
    rows = event.selection.rows if event is not None else []
    st.divider()
    show_detection(dets[rows[0] if rows else 0], spectral_records)


def show_detection(det, spectral_records):
    st.subheader("{} · {}".format(data.local_time(det.get("timestamp")), det.get("request_id")))
    left, right = st.columns([3, 2])
    with left:
        recording = det["recording"]
        if recording and recording["mp4"]:
            st.video(recording["mp4"])
        elif recording and recording["pending"]:
            st.info("This recording is still encoding (if it stays like this, the encode "
                    "failed and the .mjpeg was kept).")
        else:
            st.caption("No recording for this detection.")
    with right:
        stream = (det.get("stream") or "?").upper()
        angle = det.get("angle")
        st.markdown("### {} {}".format(stream, "" if angle is None else "({:+d}°)".format(angle)))
        st.markdown("reason: `{}`".format(det.get("reason")))
        st.markdown("labels: {}".format(", ".join(det.get("labels") or []) or "none"))
        a, b = st.columns(2)
        a.metric("Decision", fmt(det.get("decision_s"), ".2f") + " s")
        b.metric("VLM", fmt(det.get("vlm_s"), ".2f") + " s")
        timing = (det.get("spectral") or {}).get("timing")
        if timing:
            a.metric("Sensor conversion", "{:.2f} s".format(timing["conversion_s"]))
            b.metric("Sensor readout", "{:.2f} s".format(timing["readout_s"]))
            st.caption("over {} reading(s)".format(timing["readings"]))
    spectral_section(det, spectral_records)


def spectral_section(det, spectral_records):
    st.markdown("#### Spectral reading")
    record = data.final_read(det["spectral_log"])
    if record is None:
        st.caption("No spectral read was logged for this detection.")
        return
    if det.get("spectral") is None:
        st.caption("The read started with the frame but the labels decided without it "
                   "(`{}`); it ended as `{}`.".format(det.get("reason"), record.get("reason")))
        if record.get("reason") == "read_cancelled":
            return

    st.markdown("**{}** · gate reason `{}`".format(
        STATE_LABEL.get(record.get("state"), "?"), record.get("reason")))
    cols = st.columns(4)
    cols[0].metric("d", fmt(record.get("d"), "+.3f"))
    cols[1].metric("level", fmt(record.get("level"), ",.1f"))
    cols[2].metric("clean_ref_level", fmt(record.get("clean_ref_level"), ",.1f"))
    cols[3].metric("max CV", fmt(data.read_max_cv(record), ".4f"),
                   help="limit {}; above it the item moved mid-read".format(cfg.REPLICATE_CV_MAX))

    substrate = record.get("substrate")
    if record.get("d") is not None and substrate in cfg.CONTAM_MODEL:
        st.altair_chart(d_chart(record["d"], substrate), width="stretch")

    if record.get("calibrated"):
        rows = data.spectrum_rows(data.channel_means(record["calibrated"]), ITEM)
        clean = data.reference_spectrum(spectral_records,
                                        {"timestamp": record.get("reference_timestamp")})
        rows += data.spectrum_rows(clean, CLEAN)
        excluded = st.checkbox("Show excluded channels ({})".format(" ".join(EXCLUDED)),
                               key="excluded_" + det.get("request_id", ""))
        st.altair_chart(spectrum_chart(rows, show_excluded=excluded), width="stretch")
        if not clean:
            st.caption("The clean spectrum of this read's calibration is not in the log, "
                       "so only the item is drawn.")
        st.dataframe(pd.DataFrame(data.replicate_rows(record)), hide_index=True, column_config={
            "level": st.column_config.NumberColumn(format="%.1f"),
            "conversion_s": st.column_config.NumberColumn("conversion (s)", format="%.2f"),
            "readout_s": st.column_config.NumberColumn("readout (s)", format="%.2f"),
        })
        with st.expander("Per-channel values"):
            raw = data.channel_means(record.get("raw"))
            st.dataframe(pd.DataFrame([
                dict(row, raw=raw.get(row["channel"])) for row in
                data.spectrum_rows(data.channel_means(record["calibrated"]), ITEM)]).drop(
                columns="series"), hide_index=True)
    failures = [r for r in det["spectral_log"] if r.get("event") == "sensor_failure"]
    if failures:
        with st.expander("{} failed attempt(s)".format(len(failures))):
            for r in failures:
                st.code(r.get("error", "?"))


def reference_tab(p):
    reference = data.read_reference(p["reference"])
    spectral_records = load_jsonl(p["spectral"])
    if reference is None:
        st.warning("No session reference at {}. Run `python3 main.py --calibrate`; until then "
                   "every GREEN-class item routes CLEAR.".format(p["reference"]))
    else:
        age = data.reference_age_h(reference)
        if age is not None and 0 <= age <= cfg.REFERENCE_MAX_AGE_H:
            st.success("✅ Within its age limit: taken {} ({:.1f} h ago, limit {} h)".format(
                data.local_time(reference.get("timestamp")), age, cfg.REFERENCE_MAX_AGE_H))
        else:
            st.error("⛔ Expired: taken {} ({} h ago, limit {} h). main.py will not use it; "
                     "run --calibrate.".format(data.local_time(reference.get("timestamp")),
                                               fmt(age, ".1f", "?"), cfg.REFERENCE_MAX_AGE_H))
        level = reference.get("clean_ref_level")
        config_level = reference.get("config_clean_ref_level")
        cols = st.columns(5)
        cols[0].metric("Substrate", reference.get("substrate", "?"))
        cols[1].metric("clean_ref_level", fmt(level, ",.1f"),
                       delta=None if level is None or not config_level else
                       "{:+.1%} vs config".format(level / config_level - 1),
                       delta_color="off")
        cols[2].metric("Placement CV", fmt(reference.get("seating_cv"), ".4f", "n/a"),
                       help="{} placement(s)".format(len(reference.get("seating_levels") or [])))
        dark, white = reference.get("dark_raw_level"), reference.get("white_raw_level")
        cols[3].metric("White / dark (raw)", "—" if not dark or white is None
                       else "{:,.0f}x".format(white / dark),
                       help="calibration refuses below {}x".format(cfg.WHITE_DARK_RATIO_MIN))
        cols[4].metric("White flatness", fmt(reference.get("white_flatness"), ".3f"),
                       help="compare across sessions; a rising value suggests a soiled tile")
        clean = data.reference_spectrum(spectral_records, reference)
        if clean:
            st.markdown("#### Clean spectrum (mean of the placements)")
            excluded = st.checkbox("Show excluded channels ({})".format(" ".join(EXCLUDED)),
                                   key="excluded_reference")
            st.altair_chart(spectrum_chart(data.spectrum_rows(clean, CLEAN), legend=False,
                                           show_excluded=excluded),
                            width="stretch")
        with st.expander("session_reference.json"):
            st.json(reference)

    history = data.calibrations(spectral_records)
    if history:
        st.markdown("#### Calibration history")
        st.altair_chart(history_chart(history), width="stretch")
        st.dataframe(pd.DataFrame([{
            "taken": data.local_time(r.get("timestamp")),
            "substrate": r.get("substrate"),
            "clean_ref_level": r.get("clean_ref_level"),
            "placements": len(r.get("seating_levels") or []),
            "placement_cv": r.get("seating_cv"),
            "white_flatness": r.get("white_flatness"),
        } for r in reversed(history)]), hide_index=True)


def live_tab():
    if not st.toggle("Live preview", key="live"):
        st.caption("Shows what the camera sees while main.py runs with the Orbbec source "
                   "(it writes the newest frame to {}).".format(LIVE_FRAME_PATH))
        return
    live_view()


@st.fragment(run_every=LIVE_REFRESH_S)
def live_view():
    try:
        modified = os.stat(LIVE_FRAME_PATH).st_mtime
        with open(LIVE_FRAME_PATH, "rb") as f:
            jpeg = f.read()
    except FileNotFoundError:
        st.warning("No live frames. Start main.py with the Orbbec source.")
        return
    age = time.time() - modified
    if age > LIVE_STALE_S:
        st.warning("The last frame is {:.0f} s old: is main.py still running?".format(age))
    st.html(live_frame_html(jpeg))
    st.caption("frame age {:.2f} s".format(age))


def live_frame_html(jpeg):
    """The frame inlined as a data URI. st.image serves each frame from a
    short-lived /media URL that the next refresh deletes; over a slow path
    (tailscale serve, a relay) the browser asks for it too late and gets a
    404. Inlined, the frame arrives with the page update itself. Downscaled
    first to keep each update small.

    The frame sits in a box sized from numbers alone (width capped by the
    window, height from the aspect ratio), never from the image: an image's
    box is empty until it decodes, and that instant of a shorter page makes
    the browser clamp the scroll position to the top on every refresh. The
    cap keeps the whole frame on screen, so the Live tab needs no scrolling."""
    width, height = 16, 9
    try:
        with Image.open(io.BytesIO(jpeg)) as img:
            width, height = img.size
            if img.width > LIVE_PREVIEW_WIDTH:
                width, height = LIVE_PREVIEW_WIDTH, round(img.height * LIVE_PREVIEW_WIDTH / img.width)
                buf = io.BytesIO()
                img.convert("RGB").resize((width, height)).save(
                    buf, format="JPEG", quality=LIVE_PREVIEW_QUALITY)
                jpeg = buf.getvalue()
    except OSError:
        pass   # caught mid-replace or corrupt: send as is, the next refresh replaces it
    return ('<div style="position:relative;margin:0 auto;aspect-ratio:{w}/{h};'
            'width:min(100%,calc({vh}vh*{w}/{h}))">'
            '<img src="data:image/jpeg;base64,{b64}" width="{w}" height="{h}" alt="live camera frame" '
            'style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;border-radius:4px">'
            '</div>').format(
        b64=base64.b64encode(jpeg).decode("ascii"), w=width, h=height, vh=LIVE_MAX_HEIGHT_VH)


def main():
    st.set_page_config(page_title="WasteSep", layout="wide")
    run_dir = st.sidebar.text_input("main.py directory", os.environ.get("WASTESEP_RUN_DIR", HERE))
    p = data.paths(run_dir)
    st.sidebar.button("Refresh", width="stretch")
    with st.sidebar.expander("Files", expanded=not all(os.path.exists(x) for x in p.values())):
        for name, path in p.items():
            st.caption("{} {}: `{}`".format("✓" if os.path.exists(path) else "✗ missing", name, path))

    st.title("WasteSep")
    detections, reference, live = st.tabs(["Detections", "Reference", "Live"])
    with detections:
        detections_tab(p)
    with reference:
        reference_tab(p)
    with live:
        live_tab()


main()
