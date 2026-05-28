import logging
import os
from typing import Optional, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class TemplateMatcher:
    def __init__(self, template_dir: str, default_threshold: float = 0.7):
        self.template_dir = template_dir
        self.default_threshold = default_threshold
        self._cache = {}

    def load_template(self, name: str) -> Optional[np.ndarray]:
        if name in self._cache:
            return self._cache[name]

        path = os.path.join(self.template_dir, name)
        if not os.path.exists(path):
            for ext in [".png", ".jpg", ".bmp"]:
                alt_path = path + ext
                if os.path.exists(alt_path):
                    path = alt_path
                    break
            else:
                logger.warning("Template not found: %s", name)
                return None

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("Failed to load template: %s", path)
            return None

        self._cache[name] = img
        return img

    def clear_cache(self):
        self._cache.clear()

    def match(
        self,
        screenshot: np.ndarray,
        template_name: str,
        threshold: Optional[float] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
        method: int = cv2.TM_CCOEFF_NORMED,
    ) -> Optional[Tuple[int, int, int, int, float]]:
        template = self.load_template(template_name)
        if template is None:
            return None

        search_img = screenshot
        offset_x, offset_y = 0, 0
        if roi:
            x, y, w, h = roi
            search_img = screenshot[y : y + h, x : x + w]
            offset_x, offset_y = x, y

        if template.shape[0] > search_img.shape[0] or template.shape[1] > search_img.shape[1]:
            logger.warning("Template %s is larger than search area", template_name)
            return None

        result = cv2.matchTemplate(search_img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            match_val = 1 - min_val
            top_left = min_loc
        else:
            match_val = max_val
            top_left = max_loc

        thresh = threshold if threshold is not None else self.default_threshold
        if match_val < thresh:
            logger.debug("Template %s not matched (score=%.3f < %.3f)", template_name, match_val, thresh)
            return None

        x, y = top_left[0] + offset_x, top_left[1] + offset_y
        w, h = template.shape[1], template.shape[0]
        logger.debug("Template %s matched at (%d,%d,%d,%d) score=%.3f", template_name, x, y, w, h, match_val)
        return (x, y, w, h, match_val)

    def match_all(
        self,
        screenshot: np.ndarray,
        template_name: str,
        threshold: Optional[float] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
        method: int = cv2.TM_CCOEFF_NORMED,
        max_count: int = 10,
    ) -> List[Tuple[int, int, int, int, float]]:
        template = self.load_template(template_name)
        if template is None:
            return []

        search_img = screenshot
        offset_x, offset_y = 0, 0
        if roi:
            x, y, w, h = roi
            search_img = screenshot[y : y + h, x : x + w]
            offset_x, offset_y = x, y

        if template.shape[0] > search_img.shape[0] or template.shape[1] > search_img.shape[1]:
            return []

        result = cv2.matchTemplate(search_img, template, method)
        thresh = threshold if threshold is not None else self.default_threshold

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            result = 1 - result

        matches = []
        locations = np.where(result >= thresh)
        for pt in zip(*locations[::-1]):
            x, y = pt[0] + offset_x, pt[1] + offset_y
            w, h = template.shape[1], template.shape[0]
            score = float(result[pt[1], pt[0]])
            matches.append((x, y, w, h, score))

        matches = self._nms(matches, 0.3)
        matches.sort(key=lambda m: m[4], reverse=True)
        return matches[:max_count]

    @staticmethod
    def _nms(matches: List[Tuple[int, int, int, int, float]], iou_threshold: float) -> List[Tuple[int, int, int, int, float]]:
        if not matches:
            return []

        boxes = np.array([[m[0], m[1], m[0] + m[2], m[1] + m[3]] for m in matches])
        scores = np.array([m[4] for m in matches])

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return [matches[i] for i in keep]


class FeatureMatcher:
    def __init__(self, template_dir: str, default_count: int = 4):
        self.template_dir = template_dir
        self.default_count = default_count
        self._sift = cv2.SIFT_create()
        self._bf = cv2.BFMatcher(cv2.NORM_L2)
        self._cache = {}

    def load_template(self, name: str) -> Optional[np.ndarray]:
        if name in self._cache:
            return self._cache[name]
        path = os.path.join(self.template_dir, name)
        if not os.path.exists(path):
            for ext in [".png", ".jpg", ".bmp"]:
                alt_path = path + ext
                if os.path.exists(alt_path):
                    path = alt_path
                    break
            else:
                return None
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        self._cache[name] = img
        return img

    def match(
        self,
        screenshot: np.ndarray,
        template_name: str,
        min_count: Optional[int] = None,
        ratio: float = 0.6,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[Tuple[int, int, int, int, int]]:
        template = self.load_template(template_name)
        if template is None:
            return None

        search_img = screenshot
        offset_x, offset_y = 0, 0
        if roi:
            x, y, w, h = roi
            search_img = screenshot[y : y + h, x : x + w]
            offset_x, offset_y = x, y

        kp1, des1 = self._sift.detectAndCompute(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), None)
        kp2, des2 = self._sift.detectAndCompute(cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY), None)

        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            return None

        matches = self._bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good.append(m)

        count = min_count if min_count is not None else self.default_count
        if len(good) < count:
            logger.debug("Feature %s: %d good matches < %d required", template_name, len(good), count)
            return None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            return None

        h, w = template.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, M)

        x = int(transformed[:, 0, 0].min()) + offset_x
        y = int(transformed[:, 0, 1].min()) + offset_y
        bw = int(transformed[:, 0, 0].max() - transformed[:, 0, 0].min())
        bh = int(transformed[:, 0, 1].max() - transformed[:, 0, 1].min())

        logger.debug("Feature %s matched at (%d,%d,%d,%d) with %d points", template_name, x, y, bw, bh, len(good))
        return (x, y, bw, bh, len(good))


class ColorMatcher:
    @staticmethod
    def match(
        screenshot: np.ndarray,
        lower: List[int],
        upper: List[int],
        method: int = cv2.COLOR_BGR2HSV,
        min_count: int = 1,
        roi: Optional[Tuple[int, int, int, int]] = None,
        connected: bool = False,
    ) -> Optional[Tuple[int, int, int, int, int]]:
        search_img = screenshot
        offset_x, offset_y = 0, 0
        if roi:
            x, y, w, h = roi
            search_img = screenshot[y : y + h, x : x + w]
            offset_x, offset_y = x, y

        converted = cv2.cvtColor(search_img, method)
        mask = cv2.inRange(converted, np.array(lower), np.array(upper))

        if connected:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num_labels <= 1:
                return None
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            x, y, w, h = (
                stats[largest, cv2.CC_STAT_LEFT],
                stats[largest, cv2.CC_STAT_TOP],
                stats[largest, cv2.CC_STAT_WIDTH],
                stats[largest, cv2.CC_STAT_HEIGHT],
            )
            count = int(stats[largest, cv2.CC_STAT_AREA])
        else:
            count = int(cv2.countNonZero(mask))
            if count < min_count:
                return None
            coords = np.where(mask > 0)
            if len(coords[0]) == 0:
                return None
            x = int(coords[1].min()) + offset_x
            y = int(coords[0].min()) + offset_y
            w = int(coords[1].max() - coords[1].min())
            h = int(coords[0].max() - coords[0].min())

        if count < min_count:
            return None

        return (x, y, w, h, count)
