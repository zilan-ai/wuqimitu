#!/usr/bin/env python3
import argparse
import logging
import os
import sys

from config.settings import Config
from core.device.adb import ADBDevice
from core.pipeline.engine import PipelineEngine
from tasks.manager import TaskManager
from utils.logger import setup_logger


def get_base_dir():
    return os.path.dirname(os.path.abspath(__file__))


def main():
    base_dir = get_base_dir()
    default_config = os.path.join(base_dir, "config", "config.json")

    parser = argparse.ArgumentParser(
        description="MaaPTN - 无期迷途自动化日常助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                          运行所有启用的日常任务
  python main.py --task daily_reward      只运行领取日常奖励
  python main.py --task mail              只运行领取邮件
  python main.py --task dispatch          只运行派遣
  python main.py --task stamina_farm      只运行体力刷副本
  python main.py --task nightmare         只运行恶梦本
  python main.py --list                   列出可用任务
  python main.py --config config.json     指定配置文件
  python main.py --serial 127.0.0.1:5555  指定设备序列号
        """,
    )

    parser.add_argument("--config", default=default_config, help="配置文件路径")
    parser.add_argument("--serial", default=None, help="ADB设备序列号")
    parser.add_argument("--adb-path", default="adb", help="ADB可执行文件路径")
    parser.add_argument("--address", default=None, help="ADB连接地址 (如 127.0.0.1:5555)")
    parser.add_argument("--task", default=None, help="运行指定任务 (不指定则运行全部)")
    parser.add_argument("--list", action="store_true", help="列出可用任务")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    parser.add_argument("--log-to-file", action="store_true", help="将日志写入文件")
    parser.add_argument("--debug-screenshot", default=None, help="使用截图文件调试 (不连接设备)")

    args = parser.parse_args()

    config = Config(args.config)

    if args.serial:
        config.set("serial", args.serial)
    if args.adb_path:
        config.set("adb_path", args.adb_path)
    if args.address:
        config.set("adb_address", args.address)

    setup_logger(
        level=args.log_level or config.get("log_level", "INFO"),
        log_to_file=args.log_to_file or config.get("log_to_file", False),
        log_dir=os.path.join(base_dir, config.get("log_dir", "logs")),
    )

    logger = logging.getLogger(__name__)

    if args.list:
        logger.info("可用任务列表:")
        for task_name, (entry, pipeline) in TaskManager.TASK_PIPELINE_MAP.items():
            enabled = config.get(f"tasks.{task_name}", False)
            status = "启用" if enabled else "禁用"
            logger.info("  %-15s [%s] - 入口: %s, 文件: %s", task_name, status, entry, pipeline)
        return

    resource_dir = os.path.join(base_dir, config.get("resource_dir", "resource"))

    if args.debug_screenshot:
        logger.info("调试模式: 使用截图 %s", args.debug_screenshot)
        from core.recognition.matcher import TemplateMatcher
        import cv2

        matcher = TemplateMatcher(os.path.join(resource_dir, "template", "image"))
        img = cv2.imread(args.debug_screenshot)
        if img is None:
            logger.error("无法读取截图: %s", args.debug_screenshot)
            return

        logger.info("截图尺寸: %dx%d", img.shape[1], img.shape[0])
        logger.info("正在测试模板匹配...")

        import glob
        template_dir = os.path.join(resource_dir, "template", "image")
        if os.path.exists(template_dir):
            for tpl_file in glob.glob(os.path.join(template_dir, "*.png")):
                tpl_name = os.path.relpath(tpl_file, template_dir)
                result = matcher.match(img, tpl_name, threshold=0.5)
                if result:
                    x, y, w, h, score = result
                    logger.info("  匹配: %s -> (%d,%d,%d,%d) score=%.3f", tpl_name, x, y, w, h, score)
        return

    device = ADBDevice(adb_path=config.get("adb_path", "adb"), serial=config.get("serial"))

    if args.address:
        if not device.connect(args.address):
            logger.error("无法连接到设备: %s", args.address)
            return
    else:
        if not device.connect():
            logger.error("未找到可用设备，请确保设备已连接并开启USB调试")
            return

    logger.info("设备已连接: %s (%s)", device.serial, device.get_screen_resolution())

    manager = TaskManager(device, config, resource_dir)
    manager.load_pipelines()

    try:
        if args.task:
            success = manager.run_task(args.task)
            sys.exit(0 if success else 1)
        else:
            manager.run_all_daily()
    except KeyboardInterrupt:
        logger.info("用户中断，正在停止...")
        manager.engine.stop()
    finally:
        device.disconnect()


if __name__ == "__main__":
    main()
