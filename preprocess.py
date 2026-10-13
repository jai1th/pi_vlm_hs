"""The single definition of how an image becomes model input (SPEC.md §6.5).

Shared by the dataset/training side, the FastAPI service and the Pi client.
The transform reproduced here is the training loader
(``WasteSepGRPODataset.__getitem__``, quoted in SPEC.md §6.5.2)::

    img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
    img = img.resize((self.max_image_size, self.max_image_size), Image.LANCZOS)

and the label-space normalisation is ``build_taco_jsonl.coco_to_qwen_corners``
(§6.5.2, §6.5.8)::

    round(x1 * 1000 / img_w), round(y1 * 1000 / img_h),
    round(x2 * 1000 / img_w), round(y2 * 1000 / img_h)

Constraints (§6.5.7): this file is copied to the Pi as-is, so it imports only
the standard library and Pillow (no numpy, cv2, torch or ``vlm_server``
modules), and it is held to Python 3.8 syntax.

Every parameter value (size, filter, mode, crop, JPEG quality, pad colour)
comes from ``config/preprocess.json`` through :meth:`PreprocessConfig.load`.
None of them appears as a literal here. The numeric bounds below are the
validation ranges from §6.5.6, not configuration values.
"""

import base64
import hashlib
import io
import json
import math
from collections.abc import Mapping as _AbcMapping
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from PIL import Image, ImageOps

__all__ = [
    "COORD_SPACE",
    "PreprocessConfig",
    "PreprocessResult",
    "preprocess_image",
    "preprocess_path",
    "preprocess_bytes",
    "boxes_source_px_to_canvas",
    "boxes_canvas_to_source_px",
    "assert_compatible",
    "encode_jpeg",
    "encode_for_detect",
]

COORD_SPACE = 1000  # Qwen3-VL native output range. Not configurable.

# ---------------------------------------------------------------------------
# Validation ranges (§6.5.6). These are the spec's bounds, not config values.
# ---------------------------------------------------------------------------
_SCHEMA_VERSION = 1
_TARGET_SIZE_MIN, _TARGET_SIZE_MAX = 64, 2048
_JPEG_QUALITY_MIN, _JPEG_QUALITY_MAX = 90, 95
_CHANNEL_MIN, _CHANNEL_MAX = 0, 255
_ASPECT_MODES = ("stretch", "letterbox")

# Pillow >= 9.1 has Image.Resampling; older Pillow (possible on the Pi) has the
# same integer constants directly on Image.
_RS = getattr(Image, "Resampling", Image)
_RESAMPLE_FILTERS = {
    "lanczos": int(_RS.LANCZOS),
    "bicubic": int(_RS.BICUBIC),
    "bilinear": int(_RS.BILINEAR),
}

# Fields compared by assert_compatible() and hashed by fingerprint() (§6.5.7).
# crop_box is deliberately excluded (training uncropped / serving cropped is a
# legitimate pairing); pad_color is unused under stretch; jpeg_quality is
# transport only.
_COMPAT_FIELDS = ("target_size", "aspect_mode", "resample", "apply_exif_transpose")

_FIELDS = (
    "schema_version",
    "target_size",
    "aspect_mode",
    "pad_color",
    "crop_box",
    "apply_exif_transpose",
    "resample",
    "jpeg_quality",
)

_GEOMETRY_FIELDS = (
    "source_size",
    "crop_box",
    "scale_x",
    "scale_y",
    "pad_x",
    "pad_y",
    "canvas_size",
)


def _field_error(field, msg):
    # type: (str, str) -> ValueError
    return ValueError("preprocess config field '{}': {}".format(field, msg))


def _is_int(v):
    # type: (Any) -> bool
    """True for real ints. bool is an int subclass in Python and is rejected."""
    return isinstance(v, int) and not isinstance(v, bool)


def _is_tbd(v):
    # type: (Any) -> bool
    if isinstance(v, str):
        return v.strip().upper() == "TBD"
    if isinstance(v, (list, tuple)):
        return any(_is_tbd(x) for x in v)
    return False


def _int_tuple(field, value, length, what="preprocess config"):
    # type: (str, Any, int, str) -> Tuple[int, ...]
    def err(msg):
        # type: (str) -> ValueError
        return ValueError("{} field '{}': {}".format(what, field, msg))

    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise err("must be a list of {} ints, got {!r}".format(length, value))
    for x in value:
        if not _is_int(x):
            raise err("must contain only ints, got {!r} in {!r}".format(x, value))
    return tuple(value)


def _reject_duplicate_keys(pairs):
    # type: (List[Tuple[str, Any]]) -> Dict[str, Any]
    out = {}  # type: Dict[str, Any]
    for k, v in pairs:
        if k in out:
            raise ValueError(
                "preprocess config field '{}': duplicate key in JSON".format(k)
            )
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class PreprocessConfig:
    """The image transform, loaded from ``config/preprocess.json`` (§6.5.6).

    There are no defaults: every field must be supplied. Values are validated
    on construction; :meth:`load` additionally rejects missing, unknown,
    duplicate and ``"TBD"`` keys before construction.
    """

    schema_version: int
    target_size: int
    aspect_mode: str  # "stretch" | "letterbox"
    pad_color: Tuple[int, int, int]  # letterbox only; unused under stretch
    crop_box: Optional[Tuple[int, int, int, int]]  # source px, pre-resize
    apply_exif_transpose: bool
    resample: str  # "lanczos" | "bicubic" | "bilinear"
    jpeg_quality: int  # Pi-side encode quality

    def __post_init__(self):
        # type: () -> None
        for name in _FIELDS:
            if _is_tbd(getattr(self, name)):
                raise _field_error(name, "value is 'TBD'; fill it in from source")

        if not _is_int(self.schema_version) or self.schema_version != _SCHEMA_VERSION:
            raise _field_error(
                "schema_version",
                "must be {}, got {!r}".format(_SCHEMA_VERSION, self.schema_version),
            )

        if not _is_int(self.target_size) or not (
            _TARGET_SIZE_MIN <= self.target_size <= _TARGET_SIZE_MAX
        ):
            raise _field_error(
                "target_size",
                "must be an int in {}-{}, got {!r}".format(
                    _TARGET_SIZE_MIN, _TARGET_SIZE_MAX, self.target_size
                ),
            )

        if not isinstance(self.aspect_mode, str) or self.aspect_mode not in _ASPECT_MODES:
            raise _field_error(
                "aspect_mode",
                "must be one of {}, got {!r}".format(list(_ASPECT_MODES), self.aspect_mode),
            )

        pad = _int_tuple("pad_color", self.pad_color, 3)
        for c in pad:
            if not (_CHANNEL_MIN <= c <= _CHANNEL_MAX):
                raise _field_error(
                    "pad_color",
                    "each value must be in {}-{}, got {!r}".format(
                        _CHANNEL_MIN, _CHANNEL_MAX, self.pad_color
                    ),
                )
        self.pad_color = pad  # type: ignore[assignment]

        if self.crop_box is not None:
            cb = _int_tuple("crop_box", self.crop_box, 4)
            if not (cb[2] > cb[0] and cb[3] > cb[1]):
                raise _field_error(
                    "crop_box",
                    "must satisfy x2 > x1 and y2 > y1, got {!r}".format(self.crop_box),
                )
            self.crop_box = cb  # type: ignore[assignment]

        if not isinstance(self.apply_exif_transpose, bool):
            raise _field_error(
                "apply_exif_transpose",
                "must be true or false, got {!r}".format(self.apply_exif_transpose),
            )

        if not isinstance(self.resample, str) or self.resample not in _RESAMPLE_FILTERS:
            raise _field_error(
                "resample",
                "must be one of {}, got {!r}".format(
                    sorted(_RESAMPLE_FILTERS), self.resample
                ),
            )

        if not _is_int(self.jpeg_quality) or not (
            _JPEG_QUALITY_MIN <= self.jpeg_quality <= _JPEG_QUALITY_MAX
        ):
            raise _field_error(
                "jpeg_quality",
                "must be an int in {}-{}, got {!r}".format(
                    _JPEG_QUALITY_MIN, _JPEG_QUALITY_MAX, self.jpeg_quality
                ),
            )

    def resample_filter(self):
        # type: () -> int
        """The Pillow filter constant for :attr:`resample`."""
        return _RESAMPLE_FILTERS[self.resample]

    def to_dict(self):
        # type: () -> Dict[str, Any]
        """JSON-safe dict in file order (tuples as lists)."""
        d = asdict(self)
        d["pad_color"] = list(self.pad_color)
        d["crop_box"] = None if self.crop_box is None else list(self.crop_box)
        return dict((k, d[k]) for k in _FIELDS)

    def save(self, path):
        # type: (Any) -> None
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")

    @classmethod
    def from_dict(cls, d):
        # type: (Mapping[str, Any]) -> PreprocessConfig
        """Strict construction from a parsed JSON object (same rules as load)."""
        if not isinstance(d, _AbcMapping):
            raise ValueError(
                "preprocess config: top level must be a JSON object, got {}".format(
                    type(d).__name__
                )
            )
        unknown = sorted(k for k in d if k not in _FIELDS)
        if unknown:
            raise ValueError(
                "preprocess config: unknown field(s) {}".format(
                    ", ".join("'{}'".format(k) for k in unknown)
                )
            )
        missing = [k for k in _FIELDS if k not in d]
        if missing:
            # A missing key is an error, never a default (§6.5.6).
            raise ValueError(
                "preprocess config: missing field(s) {}".format(
                    ", ".join("'{}'".format(k) for k in missing)
                )
            )
        for k in _FIELDS:
            if _is_tbd(d[k]):
                raise _field_error(k, "value is 'TBD'; fill it in from source")
        return cls(**dict((k, d[k]) for k in _FIELDS))

    @classmethod
    def load(cls, path):
        # type: (Any) -> PreprocessConfig
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        try:
            d = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as e:
            raise ValueError("preprocess config {}: invalid JSON: {}".format(path, e))
        try:
            return cls.from_dict(d)
        except ValueError as e:
            raise ValueError("{} ({})".format(e, path))

    def fingerprint(self):
        # type: () -> str
        """12-hex sha256 over the compat fields only, sorted-key JSON (§6.5.7).

        Excludes crop_box, pad_color, jpeg_quality and schema_version.
        """
        d = dict((k, getattr(self, k)) for k in _COMPAT_FIELDS)
        blob = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def assert_compatible(train_cfg, serve_cfg):
    # type: (PreprocessConfig, PreprocessConfig) -> None
    """Raise ValueError listing every compat-field mismatch (§6.5.7)."""
    mismatches = []
    for k in _COMPAT_FIELDS:
        a, b = getattr(train_cfg, k), getattr(serve_cfg, k)
        if a != b:
            mismatches.append("{}: train={!r} serve={!r}".format(k, a, b))
    if mismatches:
        raise ValueError(
            "preprocess configs incompatible: " + "; ".join(mismatches)
        )


# ---------------------------------------------------------------------------
# Image transform
# ---------------------------------------------------------------------------
@dataclass
class PreprocessResult:
    image: Image.Image  # square target_size x target_size RGB
    source_size: Tuple[int, int]  # post-EXIF, pre-crop
    crop_box: Tuple[int, int, int, int]  # applied crop, source px
    scale_x: float  # crop px -> canvas px, x axis
    scale_y: float  # crop px -> canvas px, y axis
    pad_x: int  # 0 under stretch
    pad_y: int  # 0 under stretch
    canvas_size: int

    def to_geometry(self):
        # type: () -> Dict[str, Any]
        """JSON-safe geometry (everything but the image), for the Pi to retain."""
        return {
            "source_size": list(self.source_size),
            "crop_box": list(self.crop_box),
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "pad_x": self.pad_x,
            "pad_y": self.pad_y,
            "canvas_size": self.canvas_size,
        }

    @staticmethod
    def geometry_from(d):
        # type: (Mapping[str, Any]) -> Dict[str, Any]
        """Validate a dict from :meth:`to_geometry` and restore tuples.

        The returned dict is accepted wherever a PreprocessResult is accepted
        by the box transforms.
        """
        if not isinstance(d, _AbcMapping):
            raise ValueError("geometry must be a JSON object")
        unknown = sorted(k for k in d if k not in _GEOMETRY_FIELDS)
        missing = [k for k in _GEOMETRY_FIELDS if k not in d]
        if unknown or missing:
            raise ValueError(
                "geometry: unknown {} / missing {}".format(unknown, missing)
            )
        g = {
            "source_size": _int_tuple("source_size", d["source_size"], 2, "geometry"),
            "crop_box": _int_tuple("crop_box", d["crop_box"], 4, "geometry"),
            "pad_x": d["pad_x"],
            "pad_y": d["pad_y"],
            "canvas_size": d["canvas_size"],
            "scale_x": d["scale_x"],
            "scale_y": d["scale_y"],
        }
        for k in ("pad_x", "pad_y", "canvas_size"):
            if not _is_int(g[k]):
                raise ValueError("geometry field '{}' must be an int".format(k))
        for k in ("scale_x", "scale_y"):
            v = g[k]
            if not (_is_int(v) or isinstance(v, float)) or not (
                math.isfinite(v) and v > 0
            ):
                raise ValueError(
                    "geometry field '{}' must be a positive number".format(k)
                )
            g[k] = float(v)
        return g


def preprocess_image(img, cfg):
    # type: (Image.Image, PreprocessConfig) -> PreprocessResult
    """EXIF transpose -> convert("RGB") -> crop -> resize (§6.5.5).

    Returns a new image; never mutates ``img``. With the trained config and
    ``crop_box: null`` the pixels are byte-identical to the training loader.
    """
    if not isinstance(img, Image.Image):
        raise TypeError("preprocess_image expects a PIL.Image.Image")
    if not isinstance(cfg, PreprocessConfig):
        raise TypeError("preprocess_image expects a PreprocessConfig")

    if cfg.apply_exif_transpose:
        img = ImageOps.exif_transpose(img)  # returns a copy; input untouched
    img = img.convert("RGB")
    source_size = (img.width, img.height)

    if cfg.crop_box is None:
        crop = (0, 0, source_size[0], source_size[1])
    else:
        crop = cfg.crop_box
        if crop[0] < 0 or crop[1] < 0 or crop[2] > source_size[0] or crop[3] > source_size[1]:
            # PIL would silently pad out-of-bounds regions with black.
            raise ValueError(
                "crop_box {!r} exceeds the EXIF-transposed source size {!r}".format(
                    crop, source_size
                )
            )
        img = img.crop(crop)
    crop_w, crop_h = crop[2] - crop[0], crop[3] - crop[1]

    size = cfg.target_size
    flt = cfg.resample_filter()
    if cfg.aspect_mode == "stretch":
        out = img.resize((size, size), flt)
        scale_x = size / crop_w
        scale_y = size / crop_h
        pad_x = pad_y = 0
    else:  # letterbox (ablation only)
        scale = size / max(crop_w, crop_h)
        nw = max(1, int(round(crop_w * scale)))
        nh = max(1, int(round(crop_h * scale)))
        resized = img.resize((nw, nh), flt)
        out = Image.new("RGB", (size, size), cfg.pad_color)
        pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
        out.paste(resized, (pad_x, pad_y))
        scale_x = scale_y = scale

    return PreprocessResult(
        image=out,
        source_size=source_size,
        crop_box=tuple(crop),  # type: ignore[arg-type]
        scale_x=scale_x,
        scale_y=scale_y,
        pad_x=pad_x,
        pad_y=pad_y,
        canvas_size=size,
    )


def preprocess_path(path, cfg):
    # type: (Any, PreprocessConfig) -> PreprocessResult
    with Image.open(path) as img:
        return preprocess_image(img, cfg)


def preprocess_bytes(data, cfg):
    # type: (bytes, PreprocessConfig) -> PreprocessResult
    with Image.open(io.BytesIO(data)) as img:
        return preprocess_image(img, cfg)


# ---------------------------------------------------------------------------
# Box transforms (§6.5.8). Boxes are [x1, y1, x2, y2] corners.
# ---------------------------------------------------------------------------
_Geom = Union[PreprocessResult, Mapping[str, Any]]


def _geometry(r, cfg):
    # type: (_Geom, PreprocessConfig) -> Dict[str, Any]
    if isinstance(r, PreprocessResult):
        g = PreprocessResult.geometry_from(r.to_geometry())
    else:
        g = PreprocessResult.geometry_from(r)
    if g["canvas_size"] != cfg.target_size:
        raise ValueError(
            "geometry canvas_size {} does not match config target_size {}".format(
                g["canvas_size"], cfg.target_size
            )
        )
    if cfg.crop_box is not None and g["crop_box"] != cfg.crop_box:
        raise ValueError(
            "geometry crop_box {!r} does not match config crop_box {!r}".format(
                g["crop_box"], cfg.crop_box
            )
        )
    return g


def _box4(b):
    # type: (Any) -> List[float]
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        raise ValueError("box must be [x1, y1, x2, y2], got {!r}".format(b))
    for v in b:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError("box coordinates must be finite numbers, got {!r}".format(b))
    return list(b)


def _clip(v, lo, hi):
    # type: (Any, Any, Any) -> Any
    return lo if v < lo else hi if v > hi else v


def boxes_source_px_to_canvas(boxes, r, cfg):
    # type: (Sequence[Sequence[float]], _Geom, PreprocessConfig) -> List[Optional[List[int]]]
    """Source px -> canvas 0-1000 ints. ``None`` for boxes degenerate after clamping.

    Steps: minus crop origin -> clip to crop bounds -> normalise per axis.
    Clamping happens before scaling. Uses Python's built-in ``round``
    (banker's rounding). Under stretch with ``crop_box: null`` this is
    exactly ``round(x * COORD_SPACE / w), round(y * COORD_SPACE / h)``.
    """
    g = _geometry(r, cfg)
    ox, oy, ex, ey = g["crop_box"]
    cw, ch = ex - ox, ey - oy
    out = []  # type: List[Optional[List[int]]]
    for b in boxes:
        x1, y1, x2, y2 = _box4(b)
        # Clip in source px to the crop rectangle, then subtract the origin.
        x1 = _clip(x1, ox, ex) - ox
        x2 = _clip(x2, ox, ex) - ox
        y1 = _clip(y1, oy, ey) - oy
        y2 = _clip(y2, oy, ey) - oy
        if x2 <= x1 or y2 <= y1:
            out.append(None)
            continue
        if cfg.aspect_mode == "stretch":
            # Same expression shape and evaluation order as the builder:
            # round(x * 1000 / w). Under crop null, ox == oy == 0.
            out.append([
                round(x1 * COORD_SPACE / cw),
                round(y1 * COORD_SPACE / ch),
                round(x2 * COORD_SPACE / cw),
                round(y2 * COORD_SPACE / ch),
            ])
        else:
            # letterbox: the model sees the padded square, so normalise
            # against the canvas after scale + pad.
            s = g["canvas_size"]
            out.append([
                round((g["pad_x"] + x1 * g["scale_x"]) * COORD_SPACE / s),
                round((g["pad_y"] + y1 * g["scale_y"]) * COORD_SPACE / s),
                round((g["pad_x"] + x2 * g["scale_x"]) * COORD_SPACE / s),
                round((g["pad_y"] + y2 * g["scale_y"]) * COORD_SPACE / s),
            ])
    return out


def boxes_canvas_to_source_px(boxes, r, cfg):
    # type: (Sequence[Sequence[float]], _Geom, PreprocessConfig) -> List[List[float]]
    """Canvas 0-1000 -> source px floats, clamped to the crop rectangle.

    Runs on the Pi, with the PreprocessResult (or its persisted geometry)
    retained from the request. The server never calls this (§6.5.7).
    """
    g = _geometry(r, cfg)
    ox, oy, ex, ey = g["crop_box"]
    cw, ch = ex - ox, ey - oy
    out = []  # type: List[List[float]]
    for b in boxes:
        x1, y1, x2, y2 = _box4(b)
        if cfg.aspect_mode == "stretch":
            xs = [ox + v * cw / COORD_SPACE for v in (x1, x2)]
            ys = [oy + v * ch / COORD_SPACE for v in (y1, y2)]
        else:
            s = g["canvas_size"]
            xs = [ox + (v * s / COORD_SPACE - g["pad_x"]) / g["scale_x"] for v in (x1, x2)]
            ys = [oy + (v * s / COORD_SPACE - g["pad_y"]) / g["scale_y"] for v in (y1, y2)]
        out.append([
            float(_clip(xs[0], ox, ex)),
            float(_clip(ys[0], oy, ey)),
            float(_clip(xs[1], ox, ex)),
            float(_clip(ys[1], oy, ey)),
        ])
    return out


# ---------------------------------------------------------------------------
# Pi-side encode (§8.7)
# ---------------------------------------------------------------------------
def encode_jpeg(img, cfg):
    # type: (Image.Image, PreprocessConfig) -> bytes
    """JPEG bytes of a PIL **RGB** image at ``cfg.jpeg_quality``, encoded by Pillow.

    Same colour-order rules as :func:`encode_for_detect`. Used by
    ``scripts/make_pi_fixtures.py`` so fixtures are encoded exactly as the Pi
    encodes.
    """
    if not isinstance(img, Image.Image):
        raise TypeError("encode_jpeg expects a PIL.Image.Image (RGB)")
    if img.mode != "RGB":
        raise ValueError("encode_jpeg expects an RGB image, got mode {!r}".format(img.mode))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=cfg.jpeg_quality)
    return buf.getvalue()


def encode_for_detect(img, cfg):
    # type: (Image.Image, PreprocessConfig) -> str
    """Return the ``data:image/jpeg;base64,...`` URI for ``POST /detect``.

    ``img`` must be a **PIL RGB image**, normally ``preprocess_image(...).image``.

    Colour order (SPEC.md §8.7) -- the highest-risk silent failure:

    * BGR arrays from ``cv2`` / ``camera_node.frame_to_bgr()`` must go through
      ``cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)`` *before*
      ``Image.fromarray(...)``.
    * Never use ``cv2.imencode`` on the RGB array: it assumes BGR and swaps
      the channels again. Encode with Pillow (this function).

    A B/R swap does not crash anything; detection quality just drops. This
    function cannot detect a swap; the caller must get the order right.
    """
    data = encode_jpeg(img, cfg)
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
