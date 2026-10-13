"""Pi entry point: capture -> VLM server -> print detected classes.

  python3 main.py --source orbbec --camera-ip 192.168.1.10   # press Enter to capture, q to quit
  python3 main.py --source orbbec --camera-ip ... --once
  python3 main.py --source orbbec --camera-ip ... --save-dir captures
  python3 main.py --source file --image test.jpg          # no camera needed
  python3 main.py --source orbbec --camera-ip ... --loop --interval 2

Modules: capture.py (frames), vlm_pi_client.py (HTTP + preprocessing),
results.py (class extraction and printing), storage.py (--save-dir).
"""

import argparse
import sys
import time
import uuid

from capture import CaptureError, make_source
from results import print_result
from storage import save_frame, save_result
from vlm_pi_client import DEFAULT_SERVER, PiVLMClient, VLMClientError


def parse_args():
    p = argparse.ArgumentParser(description="Capture a frame and detect waste classes via the VLM server")
    p.add_argument("--source", choices=("orbbec", "file"), default="orbbec")
    p.add_argument("--camera-ip", help="Gemini 335Le IP address (source=orbbec)")
    p.add_argument("--camera-port", type=int, default=8090)
    p.add_argument("--image", help="Image path (source=file)")
    p.add_argument("--config", default="config/preprocess.json")
    p.add_argument("--server", default=DEFAULT_SERVER)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--loop", action="store_true", help="Capture continuously until Ctrl+C (no Enter needed)")
    mode.add_argument("--once", action="store_true", help="Capture a single frame and exit")
    p.add_argument("--interval", type=float, default=1.0, help="Seconds between --loop iterations")
    p.add_argument("--save-dir", help="Save each raw frame (and its result JSON) here")
    p.add_argument("--json", action="store_true", help="Print the full result as JSON")
    return p.parse_args()


def run_once(source, client, as_json, save_dir=None):
    image = source.grab()
    request_id = "cap-{}".format(uuid.uuid4().hex[:12])
    frame_path = save_frame(image, save_dir, request_id) if save_dir else None
    result = client.detect_pil(image, request_id=request_id)
    print_result(result, as_json=as_json)
    if frame_path:
        save_result(result, frame_path)


def main():
    args = parse_args()
    try:
        source = make_source(args.source, args.image, args.camera_ip, args.camera_port)
        client = PiVLMClient(args.config, args.server)
    except (CaptureError, ValueError, OSError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2

    try:
        source.open()
        client.check_ready()
        if args.once or args.loop:
            while True:
                try:
                    run_once(source, client, args.json, args.save_dir)
                except (VLMClientError, CaptureError) as exc:
                    print("error: {}".format(exc), file=sys.stderr)
                    if args.once:
                        return 1
                if args.once:
                    return 0
                time.sleep(args.interval)

        # Default: one capture per Enter press.
        print("Ready. Press Enter to capture, q + Enter (or Ctrl+C) to quit.")
        while True:
            try:
                line = input("> ")
            except EOFError:
                return 0
            if line.strip().lower() in ("q", "quit", "exit"):
                return 0
            try:
                run_once(source, client, args.json, args.save_dir)
            except (VLMClientError, CaptureError) as exc:
                print("error: {}".format(exc), file=sys.stderr)
    except (VLMClientError, CaptureError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        source.close()
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
