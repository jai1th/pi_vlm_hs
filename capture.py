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
    """Colour stream from a Gemini 335Le. With ip=None the SDK discovers it;
    with an ip, create_net_device(ip, port) is used (the Femto Mega path).

    pyorbbecsdk is imported lazily so the rest of the Pi code works without it.
    """

    def __init__(self, ip, port=8090, timeout_ms=2000, skip_frames=3):
        self.ip = ip
        self.port = port
        self.timeout_ms = timeout_ms
        self.skip_frames = skip_frames
        self._ob = None
        self._ctx = None
        self._pipeline = None

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

    def grab(self):
        if self._pipeline is None:
            raise CaptureError("camera not open; call open() first")
        ob = self._ob
        # The pipeline streams while we wait for Enter, so the first frames
        # returned can be stale. Skip a few and use the newest.
        for _ in range(self.skip_frames):
            self._pipeline.wait_for_frames(self.timeout_ms)
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
