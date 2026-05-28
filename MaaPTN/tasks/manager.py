import logging
import time
from typing import Dict, List, Optional

from core.device.adb import ADBDevice
from core.pipeline.engine import PipelineEngine
from config.settings import Config

logger = logging.getLogger(__name__)


class TaskManager:
    TASK_PIPELINE_MAP = {
        "login": ("Login_EnterGame", "daily.json"),
        "emotion_check": ("EmotionCheck_Entry", "daily.json"),
        "login_push": ("LoginPush_CheckNoMore", "daily.json"),
        "assistant_all": ("AssistantMode_Enter", "daily.json"),
        "nightingale_stamina": ("Assistant_NightingaleStamina", "daily.json"),
        "nightingale_supervision": ("Assistant_NightingaleSupervision", "daily.json"),
        "nightingale_supervision_event": ("Assistant_NightingaleSupervisionEvent", "daily.json"),
        "banquet_memory_storm": ("Assistant_BanquetMemoryStorm", "daily.json"),
        "nightingale_dispatch": ("Assistant_NightingaleDispatch", "daily.json"),
        "nightingale_secret_alliance": ("Assistant_NightingaleSecretAlliance", "daily.json"),
        "banquet_abyss": ("Assistant_BanquetAbyss", "daily.json"),
        "nightingale_free_gift": ("Assistant_NightingaleFreeGift", "daily.json"),
        "nightingale_friend_point": ("Assistant_NightingaleFriendPoint", "daily.json"),
    }

    DAILY_TASK_ORDER = [
        "login",
        "emotion_check",
        "login_push",
        "assistant_all",
        "nightingale_stamina",
        "nightingale_supervision",
        "nightingale_supervision_event",
        "banquet_memory_storm",
        "nightingale_dispatch",
        "nightingale_secret_alliance",
        "banquet_abyss",
        "nightingale_free_gift",
        "nightingale_friend_point",
    ]

    TASK_DISPLAY_NAMES = {
        "login": "登录游戏",
        "emotion_check": "情绪检测",
        "login_push": "登录推送处理",
        "assistant_all": "进入助手模式",
        "nightingale_stamina": "夜莺助手-体力专属补给",
        "nightingale_supervision": "夜莺助手-监管系统物资",
        "nightingale_supervision_event": "夜莺助手-监管事件",
        "banquet_memory_storm": "夜宴助手-记忆风暴",
        "nightingale_dispatch": "夜莺助手-派遣任务",
        "nightingale_secret_alliance": "夜莺助手-秘盟捐赠",
        "banquet_abyss": "夜宴助手-浊暗之阱",
        "nightingale_free_gift": "夜莺助手-免费礼包",
        "nightingale_friend_point": "夜莺助手-友情点",
    }

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

    def run_full_daily(self):
        logger.info("=" * 60)
        logger.info("  无期迷途 - 自动化日常助手")
        logger.info("  MaaPTN - Path To Nowhere Daily Automation")
        logger.info("=" * 60)

        self._ensure_game_running()

        pipeline_path = self._resolve_path(
            os.path.join(self.config.get("pipeline_dir", "tasks"), "daily.json")
        )
        if not os.path.exists(pipeline_path):
            logger.error("Pipeline file not found: %s", pipeline_path)
            return

        self.engine.nodes.clear()
        self.engine.load_pipeline(pipeline_path)

        logger.info("-" * 60)
        logger.info("  开始执行完整日常流程")
        logger.info("  流程: 登录 → 情绪检测 → 退出 → 推送处理 → 助手模式")
        logger.info("-" * 60)

        try:
            success = self.engine.run_task("Login_EnterGame")
            self._results["full_daily"] = success
        except Exception as e:
            logger.error("Daily task error: %s", e)
            self._results["full_daily"] = False

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

        display_name = self.TASK_DISPLAY_NAMES.get(task_name, task_name)
        logger.info("Running task: %s (%s)", display_name, task_name)
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
        logger.info("=" * 60)
        logger.info("  日常任务执行结果")
        logger.info("=" * 60)
        for task, success in self._results.items():
            display = self.TASK_DISPLAY_NAMES.get(task, task)
            status = "✓ 成功" if success else "✗ 失败"
            logger.info("  %-20s %s", display, status)
        logger.info("=" * 60)

    @property
    def results(self) -> Dict[str, bool]:
        return self._results
