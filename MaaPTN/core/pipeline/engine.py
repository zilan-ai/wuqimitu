import json
import logging
import os
import time
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.device.adb import ADBDevice
from core.recognition.matcher import TemplateMatcher, FeatureMatcher, ColorMatcher
from core.recognition.ocr import OCRRecognizer

logger = logging.getLogger(__name__)


class RecognitionType(str, Enum):
    DIRECT_HIT = "DirectHit"
    TEMPLATE_MATCH = "TemplateMatch"
    FEATURE_MATCH = "FeatureMatch"
    COLOR_MATCH = "ColorMatch"
    OCR = "OCR"
    CUSTOM = "Custom"


class ActionType(str, Enum):
    DO_NOTHING = "DoNothing"
    CLICK = "Click"
    LONG_PRESS = "LongPress"
    SWIPE = "Swipe"
    START_APP = "StartApp"
    STOP_APP = "StopApp"
    STOP_TASK = "StopTask"
    CUSTOM = "Custom"


class PipelineNode:
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.recognition = RecognitionType(config.get("recognition", "DirectHit"))
        self.action = ActionType(config.get("action", "DoNothing"))
        self.next_nodes: List[str] = []
        self.on_error: List[str] = []
        self.rate_limit: int = config.get("rate_limit", 1000)
        self.timeout: int = config.get("timeout", 20000)
        self.pre_delay: int = config.get("pre_delay", 200)
        self.post_delay: int = config.get("post_delay", 200)
        self.pre_wait_freezes: int = config.get("pre_wait_freezes", 0)
        self.post_wait_freezes: int = config.get("post_wait_freezes", 0)
        self.inverse: bool = config.get("inverse", False)
        self.enabled: bool = config.get("enabled", True)
        self.max_hit: int = config.get("max_hit", 0)
        self.repeat: int = config.get("repeat", 1)
        self.repeat_delay: int = config.get("repeat_delay", 0)
        self.hit_count: int = 0
        self._raw_config = config

        self._parse_next(config.get("next", []))
        self._parse_on_error(config.get("on_error", []))

        self.recognition_params = self._extract_recognition_params()
        self.action_params = self._extract_action_params()

    def _parse_next(self, next_list):
        for item in next_list:
            if isinstance(item, str):
                self.next_nodes.append(item)
            elif isinstance(item, dict):
                self.next_nodes.append(item.get("name", ""))

    def _parse_on_error(self, error_list):
        if isinstance(error_list, str):
            self.on_error = [error_list]
        elif isinstance(error_list, list):
            for item in error_list:
                if isinstance(item, str):
                    self.on_error.append(item)
                elif isinstance(item, dict):
                    self.on_error.append(item.get("name", ""))

    def _extract_recognition_params(self) -> Dict[str, Any]:
        params = {}
        skip_keys = {
            "recognition", "action", "next", "on_error", "rate_limit", "timeout",
            "pre_delay", "post_delay", "pre_wait_freezes", "post_wait_freezes",
            "inverse", "enabled", "max_hit", "repeat", "repeat_delay",
        }
        action_keys = {
            "target", "target_offset", "duration", "begin", "begin_offset",
            "end", "end_offset", "contact", "pressure", "package", "input_text",
            "key", "exec", "args", "detach", "custom_action", "custom_action_param",
            "custom_recognition", "custom_recognition_param",
        }
        for k, v in self._raw_config.items():
            if k not in skip_keys and k not in action_keys:
                params[k] = v
        return params

    def _extract_action_params(self) -> Dict[str, Any]:
        params = {}
        action_keys = {
            "target", "target_offset", "duration", "begin", "begin_offset",
            "end", "end_offset", "contact", "pressure", "package", "input_text",
            "key", "exec", "args", "detach", "custom_action", "custom_action_param",
        }
        for k in action_keys:
            if k in self._raw_config:
                params[k] = self._raw_config[k]
        return params

    def is_exhausted(self) -> bool:
        if self.max_hit <= 0:
            return False
        return self.hit_count >= self.max_hit


class PipelineEngine:
    PTN_PACKAGE = "com.zigzagame.ptn/com.unity3d.player.UnityPlayerActivity"

    def __init__(self, device: ADBDevice, resource_dir: str):
        self.device = device
        self.resource_dir = resource_dir
        self.template_dir = os.path.join(resource_dir, "template", "image")

        self.template_matcher = TemplateMatcher(self.template_dir)
        self.feature_matcher = FeatureMatcher(self.template_dir)
        self.color_matcher = ColorMatcher()
        self.ocr_recognizer = OCRRecognizer()

        self.nodes: Dict[str, PipelineNode] = {}
        self._running = False
        self._custom_recognizers = {}
        self._custom_actions = {}
        self._last_screenshot: Optional[np.ndarray] = None
        self._last_box: Optional[Tuple[int, int, int, int]] = None
        self._node_boxes: Dict[str, Tuple[int, int, int, int]] = {}

    def load_pipeline(self, pipeline_path: str):
        if os.path.isdir(pipeline_path):
            for root, _, files in os.walk(pipeline_path):
                for f in sorted(files):
                    if f.endswith(".json") and not f.startswith("."):
                        self._load_json(os.path.join(root, f))
        elif os.path.isfile(pipeline_path):
            self._load_json(pipeline_path)
        else:
            logger.error("Pipeline path not found: %s", pipeline_path)

    def _load_json(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for name, config in data.items():
                if name.startswith("$"):
                    continue
                self.nodes[name] = PipelineNode(name, config)
            logger.info("Loaded pipeline: %s (%d nodes)", path, len(data))
        except Exception as e:
            logger.error("Failed to load pipeline %s: %s", path, e)

    def register_custom_recognizer(self, name: str, func):
        self._custom_recognizers[name] = func

    def register_custom_action(self, name: str, func):
        self._custom_actions[name] = func

    def run_task(self, entry: str, max_rounds: int = 1000) -> bool:
        if entry not in self.nodes:
            logger.error("Entry node not found: %s", entry)
            return False

        self._running = True
        current = entry
        rounds = 0

        logger.info("Task started: %s", entry)

        while self._running and rounds < max_rounds:
            node = self.nodes.get(current)
            if node is None:
                logger.error("Node not found: %s", current)
                break

            if not node.enabled or node.is_exhausted():
                logger.debug("Node %s skipped (disabled or exhausted)", current)
                break

            success, next_node = self._execute_node(node)
            rounds += 1

            if not success:
                if node.on_error:
                    current = node.on_error[0]
                    logger.info("Node %s error, going to %s", node.name, current)
                    continue
                else:
                    logger.warning("Node %s failed with no error handler", node.name)
                    break

            if next_node:
                current = next_node
            else:
                if not node.next_nodes:
                    logger.info("Task completed: %s (no more next nodes)", entry)
                    break

        self._running = False
        logger.info("Task finished: %s (rounds=%d)", entry, rounds)
        return True

    def stop(self):
        self._running = False

    def _execute_node(self, node: PipelineNode) -> Tuple[bool, Optional[str]]:
        logger.debug("Executing node: %s", node.name)

        if node.pre_wait_freezes > 0:
            self.device.wait_for_screen_stable(timeout=node.pre_wait_freezes // 1000)

        if node.pre_delay > 0:
            time.sleep(node.pre_delay / 1000.0)

        for i in range(node.repeat):
            if not self._running:
                return False, None

            box = self._recognize(node)
            if box is None and node.recognition != RecognitionType.DIRECT_HIT:
                if node.inverse:
                    box = (0, 0, 0, 0)
                else:
                    if i > 0:
                        logger.warning("Repeat %d/%d recognition failed for %s", i + 1, node.repeat, node.name)
                    return False, None

            if node.inverse and box is not None and node.recognition != RecognitionType.DIRECT_HIT:
                logger.debug("Node %s inverse matched, skipping action", node.name)
                node.hit_count += 1
                return True, self._find_next(node)

            if box:
                self._last_box = box
                self._node_boxes[node.name] = box

            action_success = self._execute_action(node, box)
            if not action_success and node.action != ActionType.DO_NOTHING:
                return False, None

            node.hit_count += 1

            if i < node.repeat - 1:
                if node.repeat_delay > 0:
                    time.sleep(node.repeat_delay / 1000.0)

        if node.post_wait_freezes > 0:
            self.device.wait_for_screen_stable(timeout=node.post_wait_freezes // 1000)

        if node.post_delay > 0:
            time.sleep(node.post_delay / 1000.0)

        self._last_screenshot = self.device.screenshot()

        return True, self._find_next(node)

    def _recognize(self, node: PipelineNode) -> Optional[Tuple[int, int, int, int]]:
        params = node.recognition_params
        roi = params.get("roi")
        if roi and isinstance(roi, list) and len(roi) == 4:
            roi = tuple(roi)
        elif isinstance(roi, str):
            if roi in self._node_boxes:
                roi = self._node_boxes[roi]
            else:
                roi = None

        screenshot = self._last_screenshot if self._last_screenshot is not None else self.device.screenshot()
        self._last_screenshot = screenshot

        if node.recognition == RecognitionType.DIRECT_HIT:
            if roi:
                return tuple(roi)
            return None

        elif node.recognition == RecognitionType.TEMPLATE_MATCH:
            template = params.get("template", "")
            threshold = params.get("threshold", 0.7)
            method = params.get("method", cv2.TM_CCOEFF_NORMED)
            if isinstance(template, list):
                for tpl in template:
                    result = self.template_matcher.match(screenshot, tpl, threshold, roi, method)
                    if result:
                        return result[:4]
                return None
            result = self.template_matcher.match(screenshot, template, threshold, roi, method)
            return result[:4] if result else None

        elif node.recognition == RecognitionType.FEATURE_MATCH:
            template = params.get("template", "")
            min_count = params.get("count", 4)
            ratio = params.get("ratio", 0.6)
            result = self.feature_matcher.match(screenshot, template, min_count, ratio, roi)
            return result[:4] if result else None

        elif node.recognition == RecognitionType.COLOR_MATCH:
            lower = params.get("lower", [0, 0, 0])
            upper = params.get("upper", [255, 255, 255])
            method = params.get("method", cv2.COLOR_BGR2HSV)
            min_count = params.get("count", 1)
            connected = params.get("connected", False)
            result = self.color_matcher.match(screenshot, lower, upper, method, min_count, roi, connected)
            return result[:4] if result else None

        elif node.recognition == RecognitionType.OCR:
            expected = params.get("expected")
            threshold = params.get("threshold", 0.3)
            only_rec = params.get("only_rec", False)
            results = self.ocr_recognizer.recognize(screenshot, expected, roi, threshold, only_rec)
            if results:
                return results[0][2]
            return None

        elif node.recognition == RecognitionType.CUSTOM:
            custom_name = params.get("custom_recognition", "")
            if custom_name in self._custom_recognizers:
                return self._custom_recognizers[custom_name](screenshot, roi, params)
            logger.error("Custom recognizer not found: %s", custom_name)
            return None

        return None

    def _execute_action(self, node: PipelineNode, box: Optional[Tuple[int, int, int, int]]) -> bool:
        params = node.action_params

        if node.action == ActionType.DO_NOTHING:
            return True

        elif node.action == ActionType.CLICK:
            target = self._resolve_target(params, box)
            if target is None:
                return False
            x, y = self._box_center(target)
            offset = params.get("target_offset", [0, 0, 0, 0])
            x += offset[0]
            y += offset[1]
            self.device.tap_scaled(x, y)
            return True

        elif node.action == ActionType.LONG_PRESS:
            target = self._resolve_target(params, box)
            if target is None:
                return False
            x, y = self._box_center(target)
            offset = params.get("target_offset", [0, 0, 0, 0])
            x += offset[0]
            y += offset[1]
            duration = params.get("duration", 1000)
            self.device.long_press(x, y, duration)
            return True

        elif node.action == ActionType.SWIPE:
            begin = self._resolve_begin(params, box)
            end = self._resolve_end(params, box)
            if begin is None or end is None:
                return False
            bx, by = self._box_center(begin)
            ex, ey = self._box_center(end)
            duration = params.get("duration", 300)
            self.device.swipe_scaled(bx, by, ex, ey, duration)
            return True

        elif node.action == ActionType.START_APP:
            package = params.get("package", self.PTN_PACKAGE)
            self.device.start_app(package)
            time.sleep(3)
            return True

        elif node.action == ActionType.STOP_APP:
            package = params.get("package", "com.zigzagame.ptn")
            self.device.stop_app(package)
            time.sleep(1)
            return True

        elif node.action == ActionType.STOP_TASK:
            self._running = False
            return True

        elif node.action == ActionType.CUSTOM:
            custom_name = params.get("custom_action", "")
            if custom_name in self._custom_actions:
                self._custom_actions[custom_name](self.device, box, params)
                return True
            logger.error("Custom action not found: %s", custom_name)
            return False

        return True

    def _resolve_target(self, params: Dict[str, Any], box: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        target = params.get("target", True)
        if target is True:
            return box
        elif isinstance(target, list):
            if len(target) == 2:
                return (target[0], target[1], 1, 1)
            elif len(target) == 4:
                return tuple(target)
        elif isinstance(target, str):
            if target in self._node_boxes:
                return self._node_boxes[target]
        return box

    def _resolve_begin(self, params: Dict[str, Any], box: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        begin = params.get("begin", True)
        if begin is True:
            return box
        elif isinstance(begin, list):
            if len(begin) == 2:
                return (begin[0], begin[1], 1, 1)
            elif len(begin) == 4:
                return tuple(begin)
        elif isinstance(begin, str):
            if begin in self._node_boxes:
                return self._node_boxes[begin]
        return box

    def _resolve_end(self, params: Dict[str, Any], box: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        end = params.get("end", True)
        if end is True:
            return box
        elif isinstance(end, list):
            if len(end) == 2:
                return (end[0], end[1], 1, 1)
            elif len(end) == 4:
                return tuple(end)
        elif isinstance(end, str):
            if end in self._node_boxes:
                return self._node_boxes[end]
        return box

    @staticmethod
    def _box_center(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
        return box[0] + box[2] // 2, box[1] + box[3] // 2

    def _find_next(self, node: PipelineNode) -> Optional[str]:
        if not node.next_nodes:
            return None

        start_time = time.time()
        timeout_sec = node.timeout / 1000.0

        while self._running:
            screenshot = self.device.screenshot()
            self._last_screenshot = screenshot

            for next_name in node.next_nodes:
                next_node = self.nodes.get(next_name)
                if next_node is None or not next_node.enabled or next_node.is_exhausted():
                    continue

                box = self._recognize(next_node)
                if box is not None:
                    self._node_boxes[next_name] = box
                    logger.debug("Next node matched: %s", next_name)
                    return next_name

            elapsed = time.time() - start_time
            if elapsed >= timeout_sec:
                logger.warning("Next node timeout for %s (%.1fs)", node.name, timeout_sec)
                return None

            time.sleep(node.rate_limit / 1000.0)

        return None
