import subprocess
import time
import logging
import os
import tempfile
from typing import Optional, Tuple, List

import numpy as np

logger = logging.getLogger(__name__)


class ADBDevice:
    BASE_RESOLUTION = (1280, 720)

    def __init__(self, adb_path: str = "adb", serial: Optional[str] = None):
        self.adb_path = adb_path
        self.serial = serial
        self._screen_resolution: Optional[Tuple[int, int]] = None

    def _run(self, args: List[str], timeout: int = 30) -> subprocess.CompletedProcess:
        cmd = [self.adb_path]
        if self.serial:
            cmd.extend(["-s", self.serial])
        cmd.extend(args)
        logger.debug("ADB command: %s", " ".join(cmd))
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def connect(self, address: Optional[str] = None) -> bool:
        if address:
            result = self._run(["connect", address])
            if "connected" in result.stdout.lower() or "already connected" in result.stdout.lower():
                self.serial = address
                logger.info("Connected to %s", address)
                return True
            logger.error("Failed to connect to %s: %s", address, result.stderr)
            return False

        result = self._run(["devices"])
        lines = result.stdout.strip().split("\n")[1:]
        devices = [line.split("\t")[0] for line in lines if "\tdevice" in line]
        if not devices:
            logger.error("No ADB devices found")
            return False
        if len(devices) == 1:
            self.serial = devices[0]
            logger.info("Auto-selected device: %s", self.serial)
            return True
        logger.warning("Multiple devices found: %s, using first: %s", devices, devices[0])
        self.serial = devices[0]
        return True

    def disconnect(self):
        if self.serial:
            self._run(["disconnect", self.serial])
            logger.info("Disconnected from %s", self.serial)

    def is_connected(self) -> bool:
        result = self._run(["devices"])
        return self.serial in result.stdout if self.serial else False

    def get_screen_resolution(self) -> Tuple[int, int]:
        if self._screen_resolution:
            return self._screen_resolution
        result = self._run(["shell", "wm", "size"])
        output = result.stdout.strip()
        if "Physical size:" in output:
            parts = output.split("Physical size:")[1].strip().split("x")
            self._screen_resolution = (int(parts[0]), int(parts[1]))
        else:
            self._screen_resolution = self.BASE_RESOLUTION
            logger.warning("Failed to get resolution, using default %s", self.BASE_RESOLUTION)
        return self._screen_resolution

    def screenshot(self) -> np.ndarray:
        import cv2

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()

        try:
            remote_path = "/data/local/tmp/maaptn_screenshot.png"
            self._run(["shell", "screencap", "-p", remote_path])
            self._run(["pull", remote_path, tmp_path])
            self._run(["shell", "rm", remote_path])

            img = cv2.imread(tmp_path)
            if img is None:
                raise RuntimeError("Failed to read screenshot")

            resolution = self.get_screen_resolution()
            if (img.shape[1], img.shape[0]) != resolution:
                img = cv2.resize(img, resolution)

            return img
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def screenshot_raw(self) -> np.ndarray:
        result = self._run(["shell", "screencap"], timeout=10)
        if result.returncode != 0:
            return self.screenshot()

        import cv2

        raw_data = result.stdout.encode("latin1") if isinstance(result.stdout, str) else result.stdout
        try:
            header = np.frombuffer(raw_data[:12], dtype="<i4")
            width, height = int(header[0]), int(header[1])
            pixel_format = int(header[2])
            if pixel_format != 1:
                return self.screenshot()

            pixels = np.frombuffer(raw_data[12:], dtype=np.uint8)
            expected_size = width * height * 4
            if len(pixels) < expected_size:
                return self.screenshot()

            img = pixels[:expected_size].reshape(height, width, 4)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            return img
        except Exception:
            return self.screenshot()

    def tap(self, x: int, y: int, duration: int = 100):
        self._run(["shell", "input", "tap", str(x), str(y)])
        logger.debug("Tap (%d, %d)", x, y)

    def long_press(self, x: int, y: int, duration: int = 1000):
        self._run(["shell", "input", "swipe", str(x), str(y), str(x), str(y), str(duration)])
        logger.debug("Long press (%d, %d) for %dms", x, y, duration)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: int = 300):
        self._run(["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration)])
        logger.debug("Swipe (%d,%d) -> (%d,%d)", x1, y1, x2, y2)

    def input_text(self, text: str):
        self._run(["shell", "input", "text", text])
        logger.debug("Input text: %s", text)

    def press_key(self, keycode: int):
        self._run(["shell", "input", "keyevent", str(keycode)])
        logger.debug("Press key: %d", keycode)

    def start_app(self, package: str):
        self._run(["shell", "am", "start", "-n", package])
        logger.info("Start app: %s", package)

    def stop_app(self, package: str):
        self._run(["shell", "am", "force-stop", package])
        logger.info("Stop app: %s", package)

    def is_app_running(self, package: str) -> bool:
        result = self._run(["shell", "pidof", package])
        return bool(result.stdout.strip())

    def get_android_id(self) -> str:
        result = self._run(["shell", "settings", "get", "secure", "android_id"])
        return result.stdout.strip()

    def scale_coords(self, x: int, y: int) -> Tuple[int, int]:
        resolution = self.get_screen_resolution()
        sx = int(x * resolution[0] / self.BASE_RESOLUTION[0])
        sy = int(y * resolution[1] / self.BASE_RESOLUTION[1])
        return sx, sy

    def tap_scaled(self, x: int, y: int, duration: int = 100):
        sx, sy = self.scale_coords(x, y)
        self.tap(sx, sy, duration)

    def swipe_scaled(self, x1: int, y1: int, x2: int, y2: int, duration: int = 300):
        sx1, sy1 = self.scale_coords(x1, y1)
        sx2, sy2 = self.scale_coords(x2, y2)
        self.swipe(sx1, sy1, sx2, sy2, duration)

    def wait_for_screen_stable(self, threshold: float = 0.95, timeout: int = 10, interval: float = 0.5) -> bool:
        import cv2

        start = time.time()
        prev = self.screenshot()
        while time.time() - start < timeout:
            time.sleep(interval)
            curr = self.screenshot()
            result = cv2.matchTemplate(prev, curr, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val >= threshold:
                logger.debug("Screen stable (similarity=%.3f)", max_val)
                return True
            prev = curr
        logger.warning("Screen not stable after %ds", timeout)
        return False
