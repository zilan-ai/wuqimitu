import logging
import os
from typing import Optional, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_OCR_ENGINE = None


def get_ocr_engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        try:
            from paddleocr import PaddleOCR

            _OCR_ENGINE = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
            logger.info("PaddleOCR engine initialized")
        except ImportError:
            logger.warning("PaddleOCR not installed, OCR recognition unavailable")
            _OCR_ENGINE = False
    return _OCR_ENGINE if _OCR_ENGINE is not False else None


class OCRRecognizer:
    def __init__(self):
        self._engine = None

    def _ensure_engine(self):
        if self._engine is None:
            self._engine = get_ocr_engine()
        return self._engine is not None

    def recognize(
        self,
        screenshot: np.ndarray,
        expected: Optional[List[str]] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
        threshold: float = 0.3,
        only_rec: bool = False,
    ) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        if not self._ensure_engine():
            logger.error("OCR engine not available")
            return []

        search_img = screenshot
        offset_x, offset_y = 0, 0
        if roi:
            x, y, w, h = roi
            search_img = screenshot[y : y + h, x : x + w]
            offset_x, offset_y = x, y

        if only_rec:
            result = self._engine.ocr(search_img, det=False, cls=True)
        else:
            result = self._engine.ocr(search_img, cls=True)

        if not result or not result[0]:
            return []

        results = []
        for line in result[0]:
            if only_rec:
                text = line[0]
                confidence = line[1]
                box = (offset_x, offset_y, search_img.shape[1], search_img.shape[0])
            else:
                box_pts = line[0]
                text = line[1][0]
                confidence = line[1][1]
                x_min = int(min(p[0] for p in box_pts)) + offset_x
                y_min = int(min(p[1] for p in box_pts)) + offset_y
                x_max = int(max(p[0] for p in box_pts)) + offset_x
                y_max = int(max(p[1] for p in box_pts)) + offset_y
                box = (x_min, y_min, x_max - x_min, y_max - y_min)

            if confidence < threshold:
                continue

            if expected:
                import re

                matched = False
                for exp in expected:
                    if re.search(exp, text):
                        matched = True
                        break
                if not matched:
                    continue

            results.append((text, confidence, box))

        return results

    def find_text(
        self,
        screenshot: np.ndarray,
        text: str,
        roi: Optional[Tuple[int, int, int, int]] = None,
        threshold: float = 0.3,
    ) -> Optional[Tuple[int, int, int, int, float]]:
        results = self.recognize(screenshot, expected=[text], roi=roi, threshold=threshold)
        if not results:
            return None
        _, confidence, box = results[0]
        return (*box, confidence)
