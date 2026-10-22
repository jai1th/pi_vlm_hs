"""Decision node: class -> stream, gated by the spectral contamination veto.

    VLM (proposer)    -> object label + bbox
    spectral (gate)   -> contamination tri-state (spectral_node.py)
    decision node     -> owns ALL routing logic (this file)

route() is pure and deterministic: the SpectralReading is passed in, the sensor
is never called inside it, and every failure mode routes CLEAR.

Direction follows servo_test.py (Left key = negative angle, Right = positive):
  CLEAR bin = left  = -90 deg
  GREEN bin = right = +90 deg
The diverter rests at 0. Every routed frame swings to its bin, holds 1 s and
returns to 0 (see servo.py); an empty frame does not move it.

The model outputs 9 classes. togo_cup spans both bins in the original routing
split (Paper cup, Foam cup -> CLEAR; Disposable plastic cup -> GREEN) and
follows the majority -> CLEAR.
"""

import time

from contamination_config import DECISION_LOG_PATH, GATE_NOT_IMPLEMENTED, SUBSTRATE_OF
from storage import append_jsonl, utc_timestamp

GREEN = "green"
CLEAR = "clear"

STREAM_ANGLE = {
    CLEAR: -90,   # left
    GREEN: +90,   # right
}

# togo_box is GREEN for the POC on the assumption of CARTON stock. The class also spans
# EPS foam and PET, which have different clean baselines; a foam box would be given the
# carton reference and produce a meaningless d. Showcase with carton boxes only.
ROUTING = {
    "togo_box":       GREEN,
    "plastic_bottle": GREEN,
    "paper":          GREEN,
    "togo_cup":       CLEAR,
    "cup_lid":        CLEAR,
    "straw":          CLEAR,
    "tissue":         CLEAR,
    "wrapper":        CLEAR,
    "glass_bottle":   CLEAR,
}


def class_route(objects):
    """The class-only part of route(). Returns (stream, reason) when the labels
    alone decide, or None when the frame needs the spectral gate.

    There is one physical object per frame, but the VLM may draw several boxes
    with conflicting labels for it; the most conservative label wins.
    """
    streams = {ROUTING.get(o["label"]) for o in objects}
    if not streams:
        return CLEAR, "no_detection"
    if None in streams:
        return CLEAR, "unknown_label"
    if CLEAR in streams:
        return CLEAR, "class_landfill"   # also covers a GREEN/CLEAR label conflict
    if any(o["label"] in GATE_NOT_IMPLEMENTED for o in objects):
        return CLEAR, "gate_not_implemented"
    return None


def substrate_of(objects):
    """Substrate to read, from the label map only. Labels that disagree on a
    substrate give None (uncharacterized) rather than a guess."""
    substrates = {SUBSTRATE_OF.get(o["label"]) for o in objects}
    return substrates.pop() if len(substrates) == 1 else None


def route(objects, reading=None):
    """(stream, reason) for one frame. `reading` is required exactly when
    class_route() returns None.

    TODO: the VLM's per-object `contaminated` flag would apply only where no
    spectral model exists. It is dead until the server passes it through (it is
    dropped in vlm_server detect.py) and the model has seen contamination
    positives. Never fuse it with the spectral state.
    """
    decided = class_route(objects)
    if decided is not None:
        return decided
    if reading is None:
        raise ValueError("this frame needs a spectral reading")
    if reading.state is None:
        return CLEAR, reading.reason
    if reading.state:
        return CLEAR, "contamination_veto"
    return GREEN, "class_recyclable_clean"


def act_on_result(result, spectral, servo=None, t0=None, log_path=DECISION_LOG_PATH):
    """Route one frame, log the decision and move the servo; servo=None only
    prints (dry run). An empty frame is logged but never actuates: in --loop
    it is the common case, and there is nothing to divert.
    Returns (stream, reason).

    The spectral read happens only when the classes leave it undecided, so
    CLEAR-routing classes never pay its latency. t0 is a time.monotonic() value
    taken at the Enter press; the elapsed time to the decision (spectral read
    included) is printed before the servo moves, so actuation is not included.
    """
    objects = result.get("objects", [])
    labels = [o["label"] for o in objects]
    request_id = result.get("request_id", "?")

    reading = None
    if class_route(objects) is None:
        reading = spectral.read_contamination(substrate_of(objects), request_id=request_id)
    stream, reason = route(objects, reading)
    angle = STREAM_ANGLE[stream]
    elapsed = None if t0 is None else time.monotonic() - t0

    append_jsonl(log_path, {
        "timestamp": utc_timestamp(),
        "request_id": request_id,
        "n_detections": len(objects),
        "labels": labels,
        "stream": stream,
        "angle": angle,
        "reason": reason,
        "spectral": None if reading is None else {
            "state": reading.state, "d": reading.d, "level": reading.level,
            "reason": reading.reason},
        "reference_timestamp": (spectral.reference or {}).get("timestamp"),
        "decision_s": elapsed,
    })

    took = "" if elapsed is None else " [decision in {:.2f} s]".format(elapsed)
    if not objects:
        print("[{}] no objects detected: no action{}".format(request_id, took))
        return stream, reason

    gate = "" if reading is None or reading.d is None else " d={:+.3f}".format(reading.d)
    print("[{}] {} detection(s): {} -> {} {:+d} deg ({}){}{}, hold 1 s, home".format(
        request_id, len(objects), ", ".join(labels),
        stream.upper(), angle, reason, gate, took))
    if servo is not None:
        servo.swing(angle)
    return stream, reason
