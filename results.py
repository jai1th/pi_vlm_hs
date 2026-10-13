"""Turn a PiVLMClient.detect_* result into class names and printable text."""

import json


def extract_classes(result):
    """Class labels in the order the server returned them (duplicates kept)."""
    return [obj["label"] for obj in result.get("objects", [])]


def format_detections(result):
    objects = result.get("objects", [])
    if not objects:
        return "No objects detected."
    lines = []
    for i, obj in enumerate(objects, 1):
        x1, y1, x2, y2 = obj["bbox_source_px"]
        lines.append("{}. {:<15} box=({:.0f},{:.0f})-({:.0f},{:.0f})".format(
            i, obj["label"], x1, y1, x2, y2))
    return "\n".join(lines)


def print_result(result, as_json=False):
    if as_json:
        print(json.dumps(result, indent=2))
        return
    classes = extract_classes(result)
    print("[{}] {} object(s): {}".format(
        result.get("request_id", "?"), len(classes), ", ".join(classes) or "none"))
    print(format_detections(result))
    print("round trip: {} ms".format(result.get("client_elapsed_ms")))
