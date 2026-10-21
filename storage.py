"""Save captured frames and detection results, named by request ID."""

import json
import os
import time


def save_frame(image, save_dir, request_id):
    """Write the raw capture as <timestamp>_<request_id>.jpg; return the path."""
    os.makedirs(save_dir, exist_ok=True)
    stem = "{}_{}".format(time.strftime("%Y%m%d-%H%M%S"), request_id)
    path = os.path.join(save_dir, stem + ".jpg")
    image.convert("RGB").save(path, format="JPEG", quality=95)
    return path


def save_result(result, frame_path):
    """Write the detection JSON next to the frame (same name, .json)."""
    path = os.path.splitext(frame_path)[0] + ".json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    return path


UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def utc_timestamp():
    return time.strftime(UTC_FORMAT, time.gmtime())


def append_jsonl(path, record):
    """Append one record as a JSON line, creating the directory if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
