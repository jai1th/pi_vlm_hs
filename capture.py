"""Frame sources for the Pi. Every source returns a PIL RGB image.

Returning PIL RGB (not a BGR array) avoids the silent B/R swap described in
preprocess.encode_for_detect: nothing downstream has to guess the colour order.

Sources:
  OrbbecCapture  Orbbec Gemini 335Le over Ethernet (needs pyorbbecsdk)
  FileCapture    a JPEG/PNG on disk, for testing without the camera

While open, OrbbecCapture also writes the newest frame LIVE_FPS times a
second to LIVE_FRAME_PATH (in RAM under /dev/shm), which is what the
dashboard's Live tab shows: the camera can only be opened by one process.
"""

import io
import os
import sys
import tempfile
import threading
import time

from PIL import Image

LIVE_FRAME_PATH = os.path.join(
    "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir(), "wastesep_live.jpg")
LIVE_FPS = 8


class CaptureError(Exception):
    pass


class FileCapture:
    def __init__(self, path):
        self.path = path

    def open(self):
        pass

    def grab(self):
        with Image.open(self.path) as img:
            img.load()
            return img.copy()

    def close(self):
        pass


class OrbbecCapture:
    """Colour stream from a Gemini 335Le. With ip=None the SDK discovers it;
    with an ip, create_net_device(ip, port) is used (the Femto Mega path).

    pyorbbecsdk is imported lazily so the rest of the Pi code works without it.
    """

    def __init__(self, ip, port=8090, timeout_ms=2000, skip_frames=3, live_path=LIVE_FRAME_PATH):
        self.ip = ip
        self.port = port
        self.timeout_ms = timeout_ms
        self.skip_frames = skip_frames
        self.live_path = live_path
        self._ob = None
        self._ctx = None
        self._pipeline = None
        self._cond = threading.Condition()
        self._seq = 0                 # frames received so far
        self._latest = None           # (data, format, width, height) of the newest
        self._jpeg_seq = 0            # last frame handed out by next_jpeg()
        self._pump_error = None
        self._stop = threading.Event()
        self._pump = None

    def open(self):
        try:
            import pyorbbecsdk as ob
        except ImportError as exc:
            raise CaptureError("pyorbbecsdk is not installed: {}".format(exc)) from exc
        self._ob = ob

        try:
            # Keep the Context alive for as long as the pipeline runs.
            self._ctx = ctx = ob.Context()
            if self.ip:
                device = ctx.create_net_device(self.ip, self.port)
            else:
                # Let the SDK discover the camera on the network (GVCP broadcast).
                devices = ctx.query_devices()
                if devices.get_count() == 0:
                    raise CaptureError("no Orbbec device discovered on the network")
                device = devices.get_device_by_index(0)
            pipeline = ob.Pipeline(device)
            config = ob.Config()
            profiles = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
            # Prefer MJPG (small on the wire, decoded by Pillow); else the default.
            try:
                profile = profiles.get_video_stream_profile(0, 0, ob.OBFormat.MJPG, 0)
            except Exception:
                profile = profiles.get_default_video_stream_profile()
            config.enable_stream(profile)
            pipeline.start(config)
        except Exception as exc:
            raise CaptureError(
                "Could not start Orbbec camera at {}:{}: {}".format(self.ip, self.port, exc)
            ) from exc
        self._pipeline = pipeline
        self._clear_live()   # leftovers of a run that was killed
        self._start_pump()

    def _start_pump(self):
        self._stop.clear()
        self._pump = threading.Thread(target=self._pump_frames, name="camera", daemon=True)
        self._pump.start()

    def _pump_frames(self):
        """Read every frame as it arrives and keep the newest. grab(), the
        recorder and the live preview all take frames from here, so nothing
        else calls wait_for_frames and no stale frames queue up."""
        last_live = 0.0
        try:
            while not self._stop.is_set():
                frames = self._pipeline.wait_for_frames(200)
                color = None if frames is None else frames.get_color_frame()
                if color is None:
                    continue
                frame = (bytes(color.get_data()), color.get_format(),
                         color.get_width(), color.get_height())
                with self._cond:
                    self._seq += 1
                    self._latest = frame
                    self._cond.notify_all()
                now = time.monotonic()
                if self.live_path and now - last_live >= 1 / LIVE_FPS:
                    last_live = now
                    self._write_live(frame)
        except Exception as exc:
            with self._cond:
                self._pump_error = exc
                self._cond.notify_all()

    def _write_live(self, frame):
        """Replace the live-preview JPEG atomically, so a reader never sees half a file."""
        tmp = self.live_path + ".tmp"
        try:
            with open(tmp, "wb") as f:
                f.write(self._jpeg_of(frame))
            os.replace(tmp, self.live_path)
        except OSError as exc:
            print("warning: live preview disabled: {}".format(exc), file=sys.stderr)
            self.live_path = None

    def _frame_after(self, seq, timeout_s):
        """(seq, frame) for the newest frame once the count passes `seq`, or None on timeout."""
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while self._seq <= seq:
                if self._pump_error is not None:
                    raise CaptureError("camera stream stopped: {}".format(self._pump_error))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            return self._seq, self._latest

    def grab(self):
        if self._pipeline is None:
            raise CaptureError("camera not open; call open() first")
        # Same freshness as before the frame thread: skip a few frames that may
        # have been exposed before the call and use the next one.
        with self._cond:
            start = self._seq
        got = self._frame_after(start + self.skip_frames, self.timeout_ms / 1000)
        if got is None:
            raise CaptureError("timed out waiting for a frame")
        return self._to_image(got[1])

    def next_jpeg(self, timeout_ms=200):
        """The next colour frame as JPEG bytes, or None if none arrived in
        timeout_ms. MJPG frames pass through undecoded; RGB/BGR frames are
        encoded here, which is slower and may drop frames on the Pi."""
        if self._pipeline is None:
            raise CaptureError("camera not open; call open() first")
        got = self._frame_after(self._jpeg_seq, timeout_ms / 1000)
        if got is None:
            return None
        self._jpeg_seq, frame = got
        return self._jpeg_of(frame)

    def _jpeg_of(self, frame):
        if frame[1] == self._ob.OBFormat.MJPG:
            return frame[0]
        buf = io.BytesIO()
        self._to_image(frame).save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def _to_image(self, frame):
        ob = self._ob
        data, fmt, w, h = frame
        if fmt == ob.OBFormat.MJPG:
            return Image.open(io.BytesIO(data)).convert("RGB")
        if fmt == ob.OBFormat.RGB:
            return Image.frombuffer("RGB", (w, h), data, "raw", "RGB", 0, 1)
        if fmt == ob.OBFormat.BGR:
            return Image.frombuffer("RGB", (w, h), data, "raw", "BGR", 0, 1)
        raise CaptureError("unsupported colour format {}; request MJPG, RGB or BGR".format(fmt))

    def close(self):
        self._stop.set()
        if self._pump is not None:
            self._pump.join(timeout=2)
            self._pump = None
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            finally:
                self._pipeline = None
        self._clear_live()   # the dashboard then reports no live feed

    def _clear_live(self):
        """Delete the live-preview frame and any half-written .tmp: no frame
        outlives the run that wrote it."""
        if not self.live_path:
            return
        for path in (self.live_path, self.live_path + ".tmp"):
            try:
                os.remove(path)
            except OSError:
                pass


def list_devices():
    """Print every Orbbec device the SDK can discover (USB and network)."""
    import pyorbbecsdk as ob
    ctx = ob.Context()  # must outlive the device list
    devices = ctx.query_devices()
    n = devices.get_count()
    print("{} device(s) found".format(n))
    for i in range(n):
        try:
            info = devices.get_device_by_index(i).get_device_info()
            print("  [{}] {} serial={} ip={}".format(
                i, info.get_name(), info.get_serial_number(),
                getattr(info, "get_ip_address", lambda: "?")()))
        except Exception as exc:
            print("  [{}] found, but could not read its info: {}".format(i, exc))


def make_source(kind, image=None, camera_ip=None, camera_port=8090):
    if kind == "file":
        if not image:
            raise CaptureError("--image is required with --source file")
        return FileCapture(image)
    if kind == "orbbec":
        return OrbbecCapture(camera_ip, camera_port)
    raise CaptureError("unknown source {!r}".format(kind))
