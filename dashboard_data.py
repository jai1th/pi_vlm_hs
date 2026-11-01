"""Data behind dashboard.py: the decision and spectral logs, the per-detection
recordings and the session reference, joined by request_id.

Everything here only reads files that main.py writes, so the dashboard can
run next to it. Paths are those main.py uses, resolved against `run_dir` (the
directory main.py is started from).
"""

import datetime as dt
import json
import os
import statistics

import contamination_config as cfg
from spectral_node import level_of, max_cv
from storage import UTC_FORMAT

# AS7265x channel -> centre wavelength (nm).
WAVELENGTH_NM = {
    "A": 410, "B": 435, "C": 460, "D": 485, "E": 510, "F": 535,
    "G": 560, "H": 585, "R": 610, "I": 645, "S": 680, "J": 705,
    "T": 730, "U": 760, "V": 810, "W": 860, "K": 900, "L": 940,
}


def paths(run_dir):
    return {
        "decisions": os.path.join(run_dir, cfg.DECISION_LOG_PATH),
        "spectral": os.path.join(run_dir, cfg.SPECTRAL_LOG_PATH),
        "reference": os.path.join(run_dir, cfg.SESSION_REFERENCE_PATH),
        "recordings": os.path.join(run_dir, "detection_recordings"),
    }


def read_jsonl(path):
    """Records of a JSON-lines log. A line main.py is still writing (or any
    malformed line) is skipped rather than failing the page."""
    records = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except ValueError:
                    continue
    except FileNotFoundError:
        pass
    return records


def read_reference(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def recordings(recordings_dir):
    """request_id -> {"mp4": path or None, "pending": bool}. pending is a .mjpeg
    with no .mp4 yet: still encoding, or an encode that failed."""
    found = {}
    try:
        names = os.listdir(recordings_dir)
    except FileNotFoundError:
        return found
    for name in names:
        stem, ext = os.path.splitext(name)
        if ext not in (".mp4", ".mjpeg") or "_" not in stem:
            continue
        entry = found.setdefault(stem.split("_", 1)[1], {"mp4": None, "pending": False})
        if ext == ".mp4":
            entry["mp4"] = os.path.join(recordings_dir, name)
        else:
            entry["pending"] = True
    for entry in found.values():
        entry["pending"] = entry["pending"] and entry["mp4"] is None
    return found


def local_time(utc_text):
    """'2026-10-31T19:32:53Z' -> local '2026-10-31 15:32:53'; unparseable text as-is."""
    try:
        t = dt.datetime.strptime(utc_text, UTC_FORMAT).replace(tzinfo=dt.timezone.utc)
    except (TypeError, ValueError):
        return utc_text or ""
    return t.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def detections(decision_records, spectral_records, recordings_by_id):
    """One entry per routed frame, newest first: the decision record plus its
    spectral log records ("spectral_log", oldest first) and its recording."""
    by_request = {}
    for r in spectral_records:
        if r.get("request_id"):
            by_request.setdefault(r["request_id"], []).append(r)
    out = []
    for d in decision_records:
        rid = d.get("request_id")
        out.append(dict(d, spectral_log=by_request.get(rid, []),
                        recording=recordings_by_id.get(rid)))
    out.reverse()
    return out


def final_read(spectral_log):
    """The last 'read' record of one request (the reading the gate returned)."""
    reads = [r for r in spectral_log if r.get("event") == "read"]
    return reads[-1] if reads else None


def channel_means(calibrated):
    """channel -> mean over the replicates of one acquisition."""
    if not calibrated:
        return {}
    return {ch: statistics.mean(rep[ch] for rep in calibrated) for ch in calibrated[0]}


def spectrum_rows(means, series):
    """Chart rows, one per channel, ordered by wavelength."""
    return sorted(
        ({"series": series, "channel": ch, "nm": WAVELENGTH_NM[ch], "counts": value,
          "in_level": ch in cfg.LEVEL_CHANNELS}
         for ch, value in means.items() if ch in WAVELENGTH_NM),
        key=lambda row: row["nm"])


def replicate_rows(record):
    """Per conversion: its 13-channel level and, when logged, its timing."""
    calibrated = record.get("calibrated") or []
    conversion = record.get("conversion_s") or []
    readout = record.get("readout_s") or []
    rows = []
    for i, rep in enumerate(calibrated):
        rows.append({
            "reading": i + 1,
            "level": level_of([rep]),
            "conversion_s": conversion[i] if i < len(conversion) else None,
            "readout_s": readout[i] if i < len(readout) else None,
        })
    return rows


def read_max_cv(record):
    calibrated = record.get("calibrated") or []
    return max_cv(calibrated) if len(calibrated) > 1 else None


def calibrations(spectral_records):
    """Every completed calibration, oldest first."""
    return [r for r in spectral_records if r.get("event") == "calibration_result"]


def reference_spectrum(spectral_records, reference):
    """Mean clean spectrum over the placements of the calibration that produced
    `reference`, or {} if that calibration is not in the log. A placement that
    was reseated is logged more than once; its last block is the accepted one."""
    if not reference:
        return {}
    end = None
    for i, r in enumerate(spectral_records):
        if r.get("event") == "calibration_result" and r.get("timestamp") == reference.get("timestamp"):
            end = i
    if end is None:
        return {}
    seatings = {}
    for r in reversed(spectral_records[:end]):
        event = r.get("event", "")
        if event == "calibration_dark":
            break
        if event.startswith("calibration_seating_") and "error" not in r:
            seatings.setdefault(event, r)
    per_seating = [channel_means(r.get("calibrated")) for r in seatings.values()]
    per_seating = [m for m in per_seating if m]
    if not per_seating:
        return {}
    return {ch: statistics.mean(m[ch] for m in per_seating) for ch in per_seating[0]}


def reference_age_h(reference, now=None):
    try:
        taken = dt.datetime.strptime(reference["timestamp"], UTC_FORMAT).replace(
            tzinfo=dt.timezone.utc)
    except (KeyError, TypeError, ValueError):
        return None
    now = now or dt.datetime.now(dt.timezone.utc)
    return (now - taken).total_seconds() / 3600


def d_zones(substrate):
    """(lo, hi, zone) bands of d for the chart, from the substrate's thresholds."""
    _, veto, undetermined = cfg.CONTAM_MODEL[substrate]
    return [
        (None, -veto, "veto"),
        (-veto, -undetermined, "undetermined"),
        (-undetermined, undetermined, "clean"),
        (undetermined, veto, "undetermined"),
        (veto, None, "veto"),
    ]
