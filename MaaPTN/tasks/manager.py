import logging
import time
from typing import Dict, List, Optional

from core.device.adb import ADBDevice
from core.pipeline.engine import PipelineEngine
from config.settings import Config

logger = logging.getLogger(__name__)


class TaskManager:
    TASK_PIPELINE_MAP = {
        "daily_reward": ("DailyRewardEntry", "daily.json"),
        "mail": ("MailEntry", "daily.json"),
        "dispatch": ("DispatchEntry", "daily.json"),
        "visit_friend": ("FriendEntry", "daily.json"),
        "stamina_farm": ("StaminaFarmEntry", "stamina_farm.json"),
        "nightmare": ("NightmareEntry", "nightmare.json"),
    }

    DAILY_TASK_ORDER = [
        "daily_reward",
        "mail",
        "dispatch",
        "visit_friend",
        "stamina_farm",
        "nightmare",
    ]

    def __init__(self, device: ADBDevice, config: Config, resource_dir: str):
        self.device = device
        self.config = config
        self.resource_dir = resource_dir
        self.engine = PipelineEngine(device, resource_dir)
        self._results: Dict[str, bool] = {}

    def load_pipelines(self):
        pipeline_dir = self.config.get("pipeline_dir", "tasks")
        if pipeline_dir and not pipeline_dir.startswith("/"):
            pipeline_dir = self._resolve_path(pipeline_dir)
        self.engine.load_pipeline(pipeline_dir)

    def _resolve_path(self, path: str) -> str:
        import os
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base, path)

    def run_all_daily(self):
        logger.info("=" * 50)
        logger.info("Starting daily tasks for 无期迷途")
        logger.info("=" * 50)

        self._ensure_game_running()

        loaded_pipelines = set()
        for task_name in self.DAILY_TASK_ORDER:
            if not self.config.get(f"tasks.{task_name}", False):
                logger.info("Task %s is disabled, skipping", task_name)
                continue

            if task_name not in self.TASK_PIPELINE_MAP:
                logger.warning("Unknown task: %s", task_name)
                continue

            entry, pipeline_file = self.TASK_PIPELINE_MAP[task_name]

            if pipeline_file not in loaded_pipelines:
                pipeline_path = self._resolve_path(
                    os.path.join(self.config.get("pipeline_dir", "tasks"), pipeline_file)
                )
                if os.path.exists(pipeline_path):
                    self.engine.nodes.clear()
                    self.engine.load_pipeline(pipeline_path)
                    loaded_pipelines.add(pipeline_file)
                else:
                    logger.error("Pipeline file not found: %s", pipeline_path)
                    continue

            logger.info("-" * 30)
            logger.info("Running task: %s", task_name)
            logger.info("-" * 30)

            try:
                success = self.engine.run_task(entry)
                self._results[task_name] = success
                if success:
                    logger.info("Task %s completed successfully", task_name)
                else:
                    logger.warning("Task %s failed", task_name)
            except Exception as e:
                logger.error("Task %s error: %s", task_name, e)
                self._results[task_name] = False

            delay = self.config.get("delays.between_tasks", 2.0)
            time.sleep(delay)

        self._print_summary()

    def run_task(self, task_name: str) -> bool:
        if task_name not in self.TASK_PIPELINE_MAP:
            logger.error("Unknown task: %s", task_name)
            return False

        entry, pipeline_file = self.TASK_PIPELINE_MAP[task_name]
        pipeline_path = self._resolve_path(
            os.path.join(self.config.get("pipeline_dir", "tasks"), pipeline_file)
        )

        if os.path.exists(pipeline_path):
            self.engine.nodes.clear()
            self.engine.load_pipeline(pipeline_path)
        else:
            logger.error("Pipeline file not found: %s", pipeline_path)
            return False

        self._ensure_game_running()

        logger.info("Running single task: %s", task_name)
        success = self.engine.run_task(entry)
        self._results[task_name] = success
        return success

    def _ensure_game_running(self):
        ptn_package = "com.zigzagame.ptn"
        if not self.device.is_app_running(ptn_package):
            logger.info("Game not running, starting...")
            self.device.start_app(PipelineEngine.PTN_PACKAGE)
            time.sleep(5)
            self.device.wait_for_screen_stable(timeout=30)

    def _print_summary(self):
        logger.info("=" * 50)
        logger.info("Daily Tasks Summary")
        logger.info("=" * 50)
        for task, success in self._results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            logger.info("  %s: %s", task, status)
        logger.info("=" * 50)

    @property
    def results(self) -> Dict[str, bool]:
        return self._results
