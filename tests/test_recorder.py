"""Per-detection MP4 recording against a fake camera.

    python3 -m unittest discover -s tests -t . -v
"""

import contextlib
import io
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest import mock

from PIL import Image

import main
from recorder import Recorder, RecorderError
from spectral_node import SpectralNode


def jpeg(colour):
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), colour).save(buf, format="JPEG")
    return buf.getvalue()


class FakeCamera:
    """Streams a small JPEG every `period` seconds, like the Orbbec MJPG stream."""

    def __init__(self, period=0.01, frames=True):
        self.period = period
        self.frames = frames
        self.sent = 0

    def grab(self):
        return Image.new("RGB", (64, 48), (90, 90, 90))

    def next_jpeg(self, timeout_ms=200):
        time.sleep(self.period)
        if not self.frames:
            return None
        self.sent += 1
        return jpeg((self.sent * 7 % 256, 40, 200))


class FakeClient:
    def __init__(self, labels):
        self.labels = labels

    def detect_pil(self, image, request_id=None):
        return {"request_id": request_id, "client_elapsed_ms": 50.0, "objects": [
            {"label": label, "bbox_canvas": [0, 0, 10, 10], "bbox_source_px": [0, 0, 10, 10]}
            for label in self.labels]}


class RecordingServo:
    """Checks the recorder is still running for the whole swing."""

    def __init__(self, recorder):
        self.recorder = recorder
        self.recording_during_swing = None

    def swing(self, angle):
        time.sleep(0.05)
        self.recording_during_swing = self.recorder.recording


def ok_ffmpeg(cmd, **kwargs):
    return subprocess.CompletedProcess(cmd, 0, "", "")


class RecorderTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.out = io.StringIO()

    def tearDown(self):
        self._tmp.cleanup()

    def record(self, camera, seconds=0.1, run=ok_ffmpeg):
        """One start/stop with ffmpeg mocked; returns (mp4 path, ffmpeg calls)."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return run(cmd, **kwargs)

        with mock.patch("recorder.shutil.which", return_value="/usr/bin/ffmpeg"), \
                mock.patch("recorder.subprocess.run", fake_run), \
                contextlib.redirect_stdout(self.out), contextlib.redirect_stderr(self.out):
            recorder = Recorder(camera, self.tmp)
            recorder.start("cap-test")
            time.sleep(seconds)
            mp4 = recorder.stop()
            recorder.close()
        return mp4, calls


class RecorderTests(RecorderTestCase):
    def test_records_and_encodes_at_the_measured_rate(self):
        mp4, calls = self.record(FakeCamera(period=0.01))
        self.assertTrue(mp4.startswith(self.tmp) and mp4.endswith("_cap-test.mp4"))
        self.assertEqual(len(calls), 1)
        fps = float(calls[0][calls[0].index("-framerate") + 1])
        self.assertGreater(fps, 20)
        self.assertLess(fps, 110)
        self.assertEqual(os.listdir(self.tmp), [], "the .mjpeg is deleted after a good encode")

    def test_no_frames_no_video(self):
        mp4, calls = self.record(FakeCamera(frames=False))
        self.assertIsNone(mp4)
        self.assertEqual(calls, [])
        self.assertEqual(os.listdir(self.tmp), [])

    def test_failed_encode_keeps_the_mjpeg(self):
        def bad(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, "", "codec missing")
        self.record(FakeCamera(), run=bad)
        self.assertEqual([os.path.splitext(f)[1] for f in os.listdir(self.tmp)], [".mjpeg"])
        self.assertIn("codec missing", self.out.getvalue())

    def test_missing_ffmpeg(self):
        with mock.patch("recorder.shutil.which", return_value=None):
            with self.assertRaises(RecorderError):
                Recorder(FakeCamera(), self.tmp)

    def test_file_source_cannot_record(self):
        with mock.patch("recorder.shutil.which", return_value="/usr/bin/ffmpeg"):
            with self.assertRaises(RecorderError):
                Recorder(object(), self.tmp)

    def test_stop_when_not_recording(self):
        with mock.patch("recorder.shutil.which", return_value="/usr/bin/ffmpeg"):
            self.assertIsNone(Recorder(FakeCamera(), self.tmp).stop())

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_real_ffmpeg_writes_an_mp4(self):
        with contextlib.redirect_stdout(self.out):
            recorder = Recorder(FakeCamera(period=0.02), self.tmp)
            recorder.start("cap-test")
            time.sleep(0.3)
            mp4 = recorder.stop()
            recorder.close()
        self.assertEqual(os.listdir(self.tmp), [os.path.basename(mp4)])
        self.assertGreater(os.path.getsize(mp4), 0)


class RunOnceRecordingTests(RecorderTestCase):
    def run_frame(self, labels):
        """run_once with a fake camera, VLM and servo; cwd is the temp dir so
        the decision log lands there."""
        camera = FakeCamera()
        spectral = SpectralNode(log_path=os.path.join(self.tmp, "spectral.jsonl"))
        cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            with mock.patch("recorder.shutil.which", return_value="/usr/bin/ffmpeg"), \
                    mock.patch("recorder.subprocess.run", ok_ffmpeg), \
                    contextlib.redirect_stdout(self.out), contextlib.redirect_stderr(self.out):
                recorder = Recorder(camera, os.path.join(self.tmp, "videos"))
                servo = RecordingServo(recorder)
                main.run_once(camera, FakeClient(labels), spectral, servo, False, recorder=recorder)
                recorder.close()
        finally:
            os.chdir(cwd)
        return recorder, servo

    def test_records_until_the_servo_is_home(self):
        recorder, servo = self.run_frame(["straw"])
        self.assertIs(servo.recording_during_swing, True)
        self.assertFalse(recorder.recording)
        self.assertIn("video: ", self.out.getvalue())

    def test_empty_frame_stops_at_the_decision(self):
        recorder, servo = self.run_frame([])
        self.assertIsNone(servo.recording_during_swing)   # the servo never moved
        self.assertFalse(recorder.recording)


if __name__ == "__main__":
    unittest.main()
