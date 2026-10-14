"""RGB at every model boundary, with explicit JPEG storage conventions.

Unmarked UniVTAC files passed RGB directly to OpenCV's BGR encoder. Their
decoded arrays already have sensor RGB order. New files use standard JPEGs.
"""
import cv2
import numpy as np

ENCODING_ATTR = "image_encoding"
LEGACY_ENCODING = "legacy_rgb_as_bgr_jpeg"
STANDARD_ENCODING = "standard_jpeg"


def image_encoding(dataset):
    value = dataset.attrs.get(ENCODING_ATTR, LEGACY_ENCODING)
    return value.decode() if isinstance(value, bytes) else value


def decode_rgb(encoded, encoding=LEGACY_ENCODING):
    if encoding not in (LEGACY_ENCODING, STANDARD_ENCODING):
        raise ValueError(f"Unknown image encoding: {encoding!r}")
    raw = np.frombuffer(encoded, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode JPEG image")
    if encoding == STANDARD_ENCODING:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def read_rgb(dataset, step):
    return decode_rgb(dataset[step], image_encoding(dataset))


def encode_rgb(image):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected uint8 RGB [H,W,3], got {image.shape}, {image.dtype}")
    success, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError("Cannot encode RGB image")
    return encoded.tobytes()


def require_rgb_model(metadata):
    """Reject unversioned/BGR weights instead of silently changing their inputs."""
    if metadata.get("color_order") != "RGB":
        raise ValueError("Model requires color_order=RGB metadata; retrain with the RGB pipeline")
