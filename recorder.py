"""Record each detection from the Orbbec colour stream to MP4.

main.run_once starts a recording right after the detection frame is grabbed
and stops it once act_on_result returns, i.e. once the servo is back home.
While it runs, a thread writes frames as they arrive, undecoded, to
<stem>.mjpeg, so the Pi spends nothing on video during the detection. After
stop(), ffmpeg transcodes that to H.264 <stem>.mp4 in the background at low
priority and deletes the .mjpeg. The frame rate is measured from arrival
times, so the video plays back in real time.

Needs ffmpeg on the Pi: sudo apt install ffmpeg
"""

import os
import shutil
import subprocess
import sys
import threading
import time

RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection_recordings")


class RecorderError(Exception):
    pass


class Recorder:
    def __init__(self, source, out_dir=RECORDINGS_DIR):
        if shutil.which("ffmpeg") is None:
            raise RecorderError("ffmpeg not found (sudo apt install ffmpeg)")
        if not hasattr(source, "next_jpeg"):
            raise RecorderError("the {} source cannot record video".format(type(source).__name__))
        self.source = source
        self.out_dir = out_dir
        self._stop = threading.Event()
        self._thread = None
        self._encoders = []

    @property
    def recording(self):
        return self._thread is not None

    def start(self, request_id):
        os.makedirs(self.out_dir, exist_ok=True)
        self._stem = os.path.join(
            self.out_dir, "{}_{}".format(time.strftime("%Y%m%d-%H%M%S"), request_id))
        self._file = open(self._stem + ".mjpeg", "wb")
        self._frames = 0
        self._first = self._last = None
        self._error = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="recorder")
        self._thread.start()

    def _run(self):
        try:
            while not self._stop.is_set():
                jpeg = self.source.next_jpeg()
                if jpeg is None:
                    continue
                self._file.write(jpeg)
                self._last = time.monotonic()
                if self._first is None:
                    self._first = self._last
                self._frames += 1
        except Exception as exc:
            self._error = exc

    def stop(self):
        """End the recording and start its MP4 encode in the background.
        Returns the .mp4 path, or None when nothing usable was recorded.
        Safe to call when not recording."""
        if self._thread is None:
            return None
        self._stop.set()
        self._thread.join()
        self._thread = None
        self._file.close()
        mjpeg, mp4 = self._stem + ".mjpeg", self._stem + ".mp4"
        if self._error is not None:
            print("warning: recording stopped early: {}".format(self._error), file=sys.stderr)
        if self._frames < 2 or self._last <= self._first:
            os.remove(mjpeg)
            print("warning: no video for {} ({} frame(s) recorded)".format(mp4, self._frames),
                  file=sys.stderr)
            return None
        duration = self._last - self._first
        fps = (self._frames - 1) / duration
        print("    video: {} ({} frames, {:.2f} s, {:.1f} fps; encoding)".format(
            mp4, self._frames, duration, fps))
        encoder = threading.Thread(target=self._encode, args=(mjpeg, mp4, fps), name="encoder")
        encoder.start()
        self._encoders.append(encoder)
        return mp4

    @staticmethod
    def _encode(mjpeg, mp4, fps):
        # start_new_session: Ctrl+C in the terminal must not kill an encode in progress.
        result = subprocess.run(
            ["nice", "-n", "10", "ffmpeg", "-y", "-loglevel", "error",
             "-f", "mjpeg", "-framerate", "{:.3f}".format(fps), "-i", mjpeg,
             "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", mp4],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            start_new_session=True)
        if result.returncode == 0:
            os.remove(mjpeg)
        else:
            print("warning: ffmpeg could not encode {} (kept {}): {}".format(
                mp4, mjpeg, result.stderr.strip()), file=sys.stderr)

    def close(self):
        """Stop any recording and wait for pending encodes to finish."""
        self.stop()
        pending = [t for t in self._encoders if t.is_alive()]
        if pending:
            print("finishing {} video encode(s)...".format(len(pending)))
        for t in pending:
            t.join()
        self._encoders = []
