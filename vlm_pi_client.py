"""Raspberry Pi -> WasteSep VLM server HTTP client.

Copy the APPROVED server src/vlm_server/preprocess.py next to this file.
Copy the server config/preprocess.json to the Pi and supply --config.

Uses requests + Pillow for file input; detect_bgr() additionally needs OpenCV.
Server handles prompting, text parsing, class validation, and retries.
"""

import argparse
import base64
import io
import json
import time
import uuid

import requests
from PIL import Image

from preprocess import PreprocessConfig, boxes_canvas_to_source_px, preprocess_image


DEFAULT_SERVER = "http://100.106.158.16:8000"
EXPECTED_MODEL = "qwen3vl4b-grpo-g4"
EXPECTED_PROMPT_HASH = "5c537b428856"


class VLMClientError(Exception):
    def __init__(self, message, status=None, code=None, request_id=None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.request_id = request_id


class PiVLMClient:
    def __init__(self, config_path, server=DEFAULT_SERVER):
        self.server = server.rstrip("/")
        self.config = PreprocessConfig.load(config_path)
        self.session = requests.Session()
        # Tailnet traffic should not accidentally go through a system HTTP proxy.
        self.session.trust_env = False

    def close(self):
        self.session.close()

    def check_ready(self):
        """Verify model readiness and preprocessing compatibility before sending images."""
        try:
            response = self.session.get(self.server + "/ready", timeout=(2, 4))
        except requests.RequestException as exc:
            raise VLMClientError("Could not reach VLM server: {}".format(exc)) from exc

        if response.status_code != 200:
            raise VLMClientError(
                "VLM server not ready (HTTP {})".format(response.status_code),
                status=response.status_code,
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise VLMClientError("/ready returned non-JSON response") from exc

        if data.get("status") != "ready":
            raise VLMClientError("Unexpected readiness state: {!r}".format(data.get("status")))
        if data.get("model_id") != EXPECTED_MODEL:
            raise VLMClientError("Unexpected model: {!r}".format(data.get("model_id")))

        server_fp = (data.get("preprocess") or {}).get("fingerprint")
        local_fp = self.config.fingerprint()
        if server_fp != local_fp:
            raise VLMClientError(
                "Preprocessing mismatch: Pi={} server={}. Stop inference and sync configs."
                .format(local_fp, server_fp)
            )

        # The server checks its training prompt at startup. Also check its
        # reported hash when present in /ready.
        prompt_data = data.get("prompt") or {}
        server_prompt_hash = (
            data.get("prompt_hash")
            or prompt_data.get("hash")
            or prompt_data.get("sha256_12")
        )
        if server_prompt_hash is not None and server_prompt_hash != EXPECTED_PROMPT_HASH:
            raise VLMClientError(
                "Unexpected detection prompt hash: {!r}".format(server_prompt_hash)
            )
        return data

    def detect_pil(self, image, request_id=None):
        """Send one image and return server detections plus source-pixel boxes."""
        request_id = request_id or "cap-{}".format(uuid.uuid4().hex[:12])
        started = time.monotonic()

        # Use the exact training-compatible transform copied from the server.
        transformed = preprocess_image(image, self.config)
        jpeg = io.BytesIO()
        transformed.image.save(jpeg, format="JPEG", quality=self.config.jpeg_quality)
        data_uri = "data:image/jpeg;base64," + base64.b64encode(jpeg.getvalue()).decode("ascii")

        payload = {
            "image": data_uri,
            "preprocess_fingerprint": self.config.fingerprint(),
        }

        try:
            response = self.session.post(
                self.server + "/detect",
                json=payload,
                headers={"X-Request-ID": request_id},
                # Requests uses connect and socket-read timeouts, not a strict
                # end-to-end deadline; keep the call off the ROS callback thread.
                timeout=(2, 10),
            )
        except requests.Timeout as exc:
            raise VLMClientError("VLM request timed out", code="client_timeout", request_id=request_id) from exc
        except requests.RequestException as exc:
            raise VLMClientError("VLM connection failed: {}".format(exc), code="connection_error", request_id=request_id) from exc

        try:
            result = response.json()
        except ValueError as exc:
            raise VLMClientError(
                "VLM returned non-JSON response (HTTP {})".format(response.status_code),
                status=response.status_code, request_id=request_id,
            ) from exc

        if response.status_code != 200:
            err = result.get("error", {}) if isinstance(result, dict) else {}
            raise VLMClientError(
                "VLM error HTTP {} / {}: {}".format(
                    response.status_code, err.get("code", "unknown"), err.get("message", "")
                ),
                status=response.status_code,
                code=err.get("code", "unknown"),
                request_id=result.get("request_id", request_id) if isinstance(result, dict) else request_id,
            )

        if not isinstance(result, dict) or not isinstance(result.get("objects"), list):
            raise VLMClientError("Malformed success response", code="bad_response", request_id=request_id)

        objects = result["objects"]
        try:
            boxes = boxes_canvas_to_source_px(
                [obj["bbox"] for obj in objects], transformed, self.config
            )
            source_objects = [
                {"label": obj["label"], "bbox_canvas": obj["bbox"], "bbox_source_px": box}
                for obj, box in zip(objects, boxes)
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise VLMClientError("Malformed detection object: {}".format(exc),
                                 code="bad_response", request_id=request_id) from exc

        return {
            "request_id": result.get("request_id", request_id),
            "objects": source_objects,
            "metrics": result.get("metrics", {}),
            "client_elapsed_ms": round((time.monotonic() - started) * 1000, 1),
        }

    def detect_file(self, image_path, request_id=None):
        with Image.open(image_path) as image:
            return self.detect_pil(image, request_id=request_id)

    def detect_bgr(self, frame_bgr, request_id=None):
        """Use with existing Orbbec/ROS camera_node.frame_to_bgr() output."""
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.detect_pil(Image.fromarray(rgb), request_id=request_id)


def main():
    parser = argparse.ArgumentParser(description="Send a Pi image to WasteSep /detect")
    parser.add_argument("--image", required=True, help="Path to a camera JPEG or PNG")
    parser.add_argument("--config", default="config/preprocess.json", help="Copied server preprocess.json")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--request-id", default=None)
    args = parser.parse_args()

    client = PiVLMClient(args.config, args.server)
    try:
        client.check_ready()
        result = client.detect_file(args.image, request_id=args.request_id)
        print(json.dumps(result, indent=2))
    except VLMClientError as exc:
        print(json.dumps({"error": str(exc), "code": exc.code, "status": exc.status,
                          "request_id": exc.request_id}, indent=2))
        return 1
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
