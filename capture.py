"""Frame sources for the Pi. Every source returns a PIL RGB image.

Returning PIL RGB (not a BGR array) avoids the silent B/R swap described in
preprocess.encode_for_detect: nothing downstream has to guess the colour order.

Sources:
  OrbbecCapture  Orbbec Gemini 335Le over Ethernet (needs pyorbbecsdk)
  FileCapture    a JPEG/PNG on disk, for testing without the camera
"""

import io

from PIL import Image


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
    """Colour stream from a Gemini 335Le addressed by IP (default port 8090).

    pyorbbecsdk is imported lazily so the rest of the Pi code works without it.
    """

    def __init__(self, ip, port=8090, timeout_ms=2000):
        self.ip = ip
        self.port = port
        self.timeout_ms = timeout_ms
        self._ob = None
        self._pipeline = None

    def open(self):
        try:
            import pyorbbecsdk as ob
        except ImportError as exc:
            raise CaptureError("pyorbbecsdk is not installed: {}".format(exc)) from exc
        self._ob = ob

        try:
            ctx = ob.Context()
            device = ctx.create_net_device(self.ip, self.port)
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

    def grab(self):
        if self._pipeline is None:
            raise CaptureError("camera not open; call open() first")
        ob = self._ob
        frames = self._pipeline.wait_for_frames(self.timeout_ms)
        if frames is None:
            raise CaptureError("timed out waiting for a frame")
        color = frames.get_color_frame()
        if color is None:
            raise CaptureError("frame set had no colour frame")

        data = bytes(color.get_data())
        fmt = color.get_format()
        w, h = color.get_width(), color.get_height()
        if fmt == ob.OBFormat.MJPG:
            return Image.open(io.BytesIO(data)).convert("RGB")
        if fmt == ob.OBFormat.RGB:
            return Image.frombuffer("RGB", (w, h), data, "raw", "RGB", 0, 1)
        if fmt == ob.OBFormat.BGR:
            return Image.frombuffer("RGB", (w, h), data, "raw", "BGR", 0, 1)
        raise CaptureError("unsupported colour format {}; request MJPG, RGB or BGR".format(fmt))

    def close(self):
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            finally:
                self._pipeline = None


def make_source(kind, image=None, camera_ip=None, camera_port=8090):
    if kind == "file":
        if not image:
            raise CaptureError("--image is required with --source file")
        return FileCapture(image)
    if kind == "orbbec":
        if not camera_ip:
            raise CaptureError("--camera-ip is required with --source orbbec")
        return OrbbecCapture(camera_ip, camera_port)
    raise CaptureError("unknown source {!r}".format(kind))
