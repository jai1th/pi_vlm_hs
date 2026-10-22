"""Pi entry point: capture -> VLM server -> print detected classes.

  python3 main.py --source orbbec --camera-ip 192.168.1.10   # press Enter to capture, q to quit
  python3 main.py --source orbbec --camera-ip ... --once
  python3 main.py --source orbbec --camera-ip ... --save-dir captures
  python3 main.py --source file --image test.jpg          # no camera needed
  python3 main.py --source orbbec --camera-ip ... --loop --interval 2
  python3 main.py --check-spectral                        # bench test of the AS7265x readback
  python3 main.py --calibrate                             # session spectral reference, then exit
  python3 main.py --source file --image test.jpg --no-spectral --dry-run

Run --calibrate at the start of each session: without a usable
session_reference.json every GREEN-class item routes CLEAR (no_session_reference).

Modules: capture.py (frames), vlm_pi_client.py (HTTP + preprocessing),
actions.py (decision node: class + spectral gate -> stream), spectral_node.py (AS7265x
gate), contamination_config.py (veto constants), servo.py (pigpio servo),
results.py (JSON print), storage.py (--save-dir, JSONL logs).
"""

import argparse
import sys
import time
import uuid

from capture import CaptureError, make_source
from actions import act_on_result
from contamination_config import CONTAM_MODEL, SESSION_REFERENCE_PATH
from results import print_result
from servo import DEFAULT_SPEED_DPS, SERVO_PIN, Servo, ServoError
from spectral_node import (As7265xSensor, SpectralError, SpectralNode, calibrate_and_save,
                           load_reference)
from storage import save_frame, save_result
from vlm_pi_client import DEFAULT_SERVER, PiVLMClient, VLMClientError


def parse_args():
    p = argparse.ArgumentParser(description="Capture a frame and detect waste classes via the VLM server")
    p.add_argument("--source", choices=("orbbec", "file"), default="orbbec")
    p.add_argument("--camera-ip", help="Camera IP; omit to let the SDK discover the camera (source=orbbec)")
    p.add_argument("--list-devices", action="store_true", help="List discoverable Orbbec devices and exit")
    p.add_argument("--camera-port", type=int, default=8090)
    p.add_argument("--image", help="Image path (source=file)")
    p.add_argument("--config", default="config/preprocess.json")
    p.add_argument("--server", default=DEFAULT_SERVER)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--loop", action="store_true", help="Capture continuously until Ctrl+C (no Enter needed)")
    mode.add_argument("--once", action="store_true", help="Capture a single frame and exit")
    p.add_argument("--interval", type=float, default=1.0, help="Seconds between --loop iterations")
    p.add_argument("--save-dir", help="Save each raw frame (and its result JSON) here")
    p.add_argument("--json", action="store_true", help="Also print the full result as JSON")
    p.add_argument("--dry-run", action="store_true", help="Print the servo action but do not move the servo")
    p.add_argument("--servo-speed", type=float, default=DEFAULT_SPEED_DPS, help="Servo speed in deg/s (default: medium)")
    p.add_argument("--servo-pin", type=int, default=SERVO_PIN, help="GPIO (BCM) pin of the servo signal")
    p.add_argument("--calibrate", action="store_true",
                   help="Run the interactive spectral calibration, write --reference and exit")
    p.add_argument("--substrate", choices=sorted(CONTAM_MODEL), default="carton",
                   help="Substrate to calibrate (with --calibrate)")
    p.add_argument("--reference", default=SESSION_REFERENCE_PATH,
                   help="Session spectral reference written by --calibrate")
    p.add_argument("--check-spectral", action="store_true",
                   help="Bench test: does the register readback read correctly and leave the bulb lit? Then exit")
    p.add_argument("--no-spectral", action="store_true",
                   help="Run without the AS7265x (same as a missing reference: GREEN classes route CLEAR)")
    return p.parse_args()


def run_once(source, client, spectral, servo, as_json, save_dir=None, t0=None):
    if t0 is None:
        t0 = time.monotonic()
    image = source.grab()
    request_id = "cap-{}".format(uuid.uuid4().hex[:12])
    frame_path = save_frame(image, save_dir, request_id) if save_dir else None
    result = client.detect_pil(image, request_id=request_id)
    if as_json:
        print_result(result, as_json=True)
    act_on_result(result, spectral, servo, t0)
    if frame_path:
        save_result(result, frame_path)


def warn_no_reference(why):
    print("=" * 72, file=sys.stderr)
    print("WARNING: no usable spectral reference: {}".format(why), file=sys.stderr)
    print("         every GREEN-class item will route CLEAR (no_session_reference)", file=sys.stderr)
    print("=" * 72, file=sys.stderr)


def open_spectral(no_spectral, reference_path):
    """Start the AS7265x and load the session reference. Never prompts. Every
    failure leaves a node without a reference, which routes conservatively."""
    if no_spectral:
        warn_no_reference("--no-spectral")
        return SpectralNode()
    sensor = As7265xSensor()
    try:
        sensor.open()
        live = sensor.read_settings()
    except Exception as exc:
        sensor.close()
        warn_no_reference("AS7265x unavailable: {}".format(exc))
        return SpectralNode()
    reference, why = load_reference(reference_path, live)
    if reference is None:
        warn_no_reference(why)
    else:
        print("spectral reference: {} clean_ref_level={:.1f} taken {}".format(
            reference["substrate"], reference["clean_ref_level"], reference["timestamp"]))
    return SpectralNode(sensor, reference)


def calibrate(substrate, reference_path):
    sensor = As7265xSensor()
    try:
        sensor.open()
        calibrate_and_save(SpectralNode(sensor), substrate, reference_path)
    except SpectralError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("calibration aborted; {} not written".format(reference_path), file=sys.stderr)
        return 1
    finally:
        sensor.close()
    print("wrote {}".format(reference_path))
    return 0


def check_spectral():
    sensor = As7265xSensor()
    try:
        sensor.open()
        ok = SpectralNode(sensor).check_bulb_readback()
    except (KeyboardInterrupt, EOFError):
        return 1
    except Exception as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1
    finally:
        sensor.close()
    print("check-spectral: {}".format("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    args = parse_args()
    if args.list_devices:
        from capture import list_devices
        list_devices()
        return 0
    if args.check_spectral:
        return check_spectral()
    if args.calibrate:
        return calibrate(args.substrate, args.reference)
    try:
        source = make_source(args.source, args.image, args.camera_ip, args.camera_port)
        client = PiVLMClient(args.config, args.server)
        servo = None if args.dry_run else Servo(pin=args.servo_pin, speed_dps=args.servo_speed)
    except (CaptureError, ValueError, OSError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2

    spectral = None
    try:
        source.open()
        client.check_ready()
        spectral = open_spectral(args.no_spectral, args.reference)
        if servo:
            servo.open()
        if args.once or args.loop:
            while True:
                try:
                    run_once(source, client, spectral, servo, args.json, args.save_dir)
                except (VLMClientError, CaptureError, ServoError) as exc:
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
            t0 = time.monotonic()  # Enter pressed
            if line.strip().lower() in ("q", "quit", "exit"):
                return 0
            try:
                run_once(source, client, spectral, servo, args.json, args.save_dir, t0)
            except (VLMClientError, CaptureError, ServoError) as exc:
                print("error: {}".format(exc), file=sys.stderr)
    except (VLMClientError, CaptureError, ServoError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        source.close()
        client.close()
        if spectral and spectral.sensor:
            spectral.sensor.close()
        if servo:
            servo.close()


if __name__ == "__main__":
    raise SystemExit(main())
