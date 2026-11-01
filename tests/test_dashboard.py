"""Dashboard data joins, the camera frame thread, and a dashboard render smoke test.

    python3 -m unittest discover -s tests -t . -v
"""

import contextlib
import io
import json
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

from PIL import Image

import contamination_config as cfg
import dashboard_data as data
from actions import act_on_result
from capture import CaptureError, OrbbecCapture
from spectral_node import SpectralNode, calibrate_and_save, load_reference
from tests.test_veto import FakeSensor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_session(run_dir):
    """Logs, reference and recordings as main.py writes them: a calibration,
    an oiled togo_box (veto, recorded) and a straw (cancelled read, encoding)."""
    p = data.paths(run_dir)
    os.makedirs(os.path.dirname(p["spectral"]), exist_ok=True)
    os.makedirs(p["recordings"], exist_ok=True)
    quiet = contextlib.redirect_stdout(io.StringIO())
    with quiet:
        cal = SpectralNode(FakeSensor(level=5400, dark=4.5), log_path=p["spectral"],
                           sleep=lambda s: None)
        calibrate_and_save(cal, "carton", p["reference"], prompt=lambda _: "")
    reference, _ = load_reference(p["reference"], {"gain": cfg.GAIN,
                                                   "integration_cycles": cfg.INTEGRATION_CYCLES,
                                                   "white_current": cfg.WHITE_BULB_CURRENT})
    node = SpectralNode(FakeSensor(level=2700), reference, log_path=p["spectral"],
                        sleep=lambda s: None)
    for request_id, label in (("cap-oiled", "togo_box"), ("cap-straw", "straw")):
        result = {"request_id": request_id, "client_elapsed_ms": 3710.0, "objects": [
            {"label": label, "bbox_canvas": [0, 0, 1, 1], "bbox_source_px": [0, 0, 1, 1]}]}
        node.start_read(request_id)
        with quiet:
            act_on_result(result, node, None, time.monotonic() - 7.1, log_path=p["decisions"])
        node.cancel()
        node.wait()
    open(os.path.join(p["recordings"], "20261031-120000_cap-oiled.mp4"), "wb").close()
    open(os.path.join(p["recordings"], "20261031-120010_cap-straw.mjpeg"), "wb").close()
    return p


class DashboardDataTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.p = write_session(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def detections(self):
        return data.detections(data.read_jsonl(self.p["decisions"]),
                               data.read_jsonl(self.p["spectral"]),
                               data.recordings(self.p["recordings"]))

    def test_partial_line_is_skipped(self):
        with open(self.p["decisions"], "a") as f:
            f.write('{"request_id": "cap-half')
        self.assertEqual(len(data.read_jsonl(self.p["decisions"])), 2)

    def test_detections_newest_first_with_reads_and_videos(self):
        straw, oiled = self.detections()
        self.assertEqual(oiled["request_id"], "cap-oiled")
        self.assertEqual(oiled["reason"], "contamination_veto")
        self.assertTrue(oiled["recording"]["mp4"].endswith("cap-oiled.mp4"))
        self.assertAlmostEqual(data.final_read(oiled["spectral_log"])["d"], 0.5)
        self.assertEqual(straw["reason"], "class_landfill")
        self.assertEqual(straw["recording"], {"mp4": None, "pending": True})

    def test_replicate_rows_carry_timing(self):
        oiled = self.detections()[1]
        rows = data.replicate_rows(data.final_read(oiled["spectral_log"]))
        self.assertEqual([r["reading"] for r in rows], [1, 2, 3])
        self.assertEqual(rows[0]["conversion_s"], 0.4)
        self.assertAlmostEqual(rows[0]["level"], 2700)

    def test_spectrum_rows_ordered_by_wavelength(self):
        oiled = self.detections()[1]
        rows = data.spectrum_rows(data.channel_means(data.final_read(oiled["spectral_log"])["calibrated"]),
                                  "item")
        self.assertEqual(len(rows), 18)
        self.assertEqual([r["nm"] for r in rows], sorted(r["nm"] for r in rows))
        self.assertEqual(sum(r["in_level"] for r in rows), len(cfg.LEVEL_CHANNELS))

    def test_reference_spectrum_uses_the_accepted_placements(self):
        records = data.read_jsonl(self.p["spectral"])
        reference = data.read_reference(self.p["reference"])
        clean = data.reference_spectrum(records, reference)
        self.assertAlmostEqual(clean["A"], 5400)
        # A reseated placement: the earlier, rejected block must not count.
        result_at = max(i for i, r in enumerate(records) if r.get("event") == "calibration_result")
        rejected = dict(records[result_at - 1], calibrated=[{ch: 99999.0 for ch in data.WAVELENGTH_NM}] * 3)
        records.insert(result_at - 1, rejected)
        self.assertAlmostEqual(data.reference_spectrum(records, reference)["A"], 5400)

    def test_unknown_reference_has_no_spectrum(self):
        records = data.read_jsonl(self.p["spectral"])
        self.assertEqual(data.reference_spectrum(records, {"timestamp": "2020-01-01T00:00:00Z"}), {})

    def test_d_zones_follow_the_config(self):
        _, veto, undetermined = cfg.CONTAM_MODEL["carton"]
        zones = data.d_zones("carton")
        self.assertEqual(zones[2], (-undetermined, undetermined, "clean"))
        self.assertEqual(zones[4], (veto, None, "veto"))


# --- Camera frame thread -------------------------------------------------------------------

class FakeOb:
    class OBFormat:
        MJPG, RGB, BGR = "MJPG", "RGB", "BGR"


class FakeColour:
    def __init__(self, n):
        buf = io.BytesIO()
        Image.new("RGB", (32, 24), (n % 256, 0, 0)).save(buf, format="JPEG")
        self.data = buf.getvalue()

    def get_data(self):
        return self.data

    def get_format(self):
        return FakeOb.OBFormat.MJPG

    def get_width(self):
        return 32

    def get_height(self):
        return 24


class FakeFrames:
    def __init__(self, n):
        self.colour = FakeColour(n)

    def get_color_frame(self):
        return self.colour


class FakePipeline:
    def __init__(self, period=0.005, fail_after=None):
        self.period = period
        self.fail_after = fail_after
        self.sent = 0
        self.stopped = False

    def wait_for_frames(self, timeout_ms):
        time.sleep(self.period)
        if self.fail_after is not None and self.sent >= self.fail_after:
            raise RuntimeError("network camera dropped")
        self.sent += 1
        return FakeFrames(self.sent)

    def stop(self):
        self.stopped = True


class FrameThreadTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.live = os.path.join(self._tmp.name, "live.jpg")

    def tearDown(self):
        self._tmp.cleanup()

    def camera(self, pipeline):
        cam = OrbbecCapture(None, live_path=self.live, timeout_ms=1000)
        cam._ob = FakeOb
        cam._pipeline = pipeline
        cam._start_pump()
        return cam

    def test_grab_waits_for_fresh_frames(self):
        pipeline = FakePipeline()
        cam = self.camera(pipeline)
        try:
            time.sleep(0.05)
            before = cam._seq
            image = cam.grab()
            self.assertEqual(image.size, (32, 24))
            self.assertGreater(cam._seq, before + cam.skip_frames)
        finally:
            cam.close()
        self.assertTrue(pipeline.stopped)

    def test_next_jpeg_never_repeats_a_frame(self):
        cam = self.camera(FakePipeline())
        try:
            seqs = []
            for _ in range(5):
                self.assertIsNotNone(cam.next_jpeg())
                seqs.append(cam._jpeg_seq)
            self.assertEqual(seqs, sorted(set(seqs)))
        finally:
            cam.close()

    def test_live_frame_written_and_removed_on_close(self):
        cam = self.camera(FakePipeline())
        try:
            deadline = time.monotonic() + 2
            while not os.path.exists(self.live) and time.monotonic() < deadline:
                time.sleep(0.01)
            with Image.open(self.live) as img:
                self.assertEqual(img.size, (32, 24))
        finally:
            cam.close()
        self.assertFalse(os.path.exists(self.live))

    def test_stream_failure_is_a_capture_error(self):
        cam = self.camera(FakePipeline(fail_after=2))
        try:
            time.sleep(0.05)
            with self.assertRaises(CaptureError):
                cam.grab()
        finally:
            cam.close()


# --- Dashboard render ------------------------------------------------------------------------

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None


@unittest.skipIf(AppTest is None, "streamlit not installed")
class DashboardRenderTests(unittest.TestCase):
    def render(self, run_dir):
        with mock.patch.dict(os.environ, {"WASTESEP_RUN_DIR": run_dir}):
            app = AppTest.from_file(os.path.join(ROOT, "dashboard.py"), default_timeout=60)
            app.run()
        return app

    def test_renders_a_session_without_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_session(tmp)
            app = self.render(tmp)
        self.assertEqual(list(app.exception), [])
        self.assertIn("cap-straw", app.subheader[0].value)   # newest shown by default
        self.assertIn("Within its age limit", app.success[0].value)

    def test_renders_an_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = self.render(tmp)
        self.assertEqual(list(app.exception), [])
        self.assertIn("No detections logged yet", app.info[0].value)
        self.assertIn("No session reference", app.warning[0].value)


if __name__ == "__main__":
    unittest.main()
