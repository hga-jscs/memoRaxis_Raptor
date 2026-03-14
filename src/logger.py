# -*- coding: utf-8 -*-
"""统一日志系统

日志自动写入 log/YYYYMMDD-HHMMSS.log，屏幕仅输出 ERROR 级别简短提示。
同时输出结构化事件日志 log/YYYYMMDD-HHMMSS.events.jsonl，便于 token 成本分析。
全局单例，禁止各脚本自建。
"""

from __future__ import annotations

import json
import logging
import shlex
import sys
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

_logger: Optional[logging.Logger] = None
_log_file_path: Optional[Path] = None
_event_file_path: Optional[Path] = None
_event_lock = threading.Lock()
_trace_ctx: ContextVar[Dict[str, Any]] = ContextVar("trace_ctx", default={})


def _init_event_file() -> None:
    """初始化结构化事件日志文件。"""
    global _event_file_path
    if _event_file_path is not None:
        return

    log_dir = Path(__file__).parent.parent / "log"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    _event_file_path = log_dir / f"{timestamp}.events.jsonl"


def _init_logger() -> logging.Logger:
    """初始化日志系统"""
    global _log_file_path

    log_dir = Path(__file__).parent.parent / "log"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    _log_file_path = log_dir / f"{timestamp}.log"

    logger = logging.getLogger("memoRaxis")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = logging.FileHandler(_log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.ERROR)
    console_formatter = logging.Formatter("[错误] %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    cmd = " ".join([shlex.quote(sys.executable)] + [shlex.quote(arg) for arg in sys.argv])
    logger.info("启动命令: %s", cmd)
    logger.info("日志系统初始化完成，日志文件: %s", _log_file_path)
    _init_event_file()
    if _event_file_path is not None:
        logger.info("事件日志初始化完成，事件文件: %s", _event_file_path)

    return logger


def get_logger() -> logging.Logger:
    """获取全局日志实例"""
    global _logger
    if _logger is None:
        _logger = _init_logger()
    return _logger


@contextmanager
def bind_trace(**kwargs: Any):
    """绑定执行上下文，用于事件日志自动注入 run/adaptor/stage 等字段。"""
    ctx = dict(_trace_ctx.get())
    ctx.update({k: v for k, v in kwargs.items() if v is not None})
    token = _trace_ctx.set(ctx)
    try:
        yield
    finally:
        _trace_ctx.reset(token)


def get_trace_context() -> Dict[str, Any]:
    """获取当前 trace 上下文。"""
    return dict(_trace_ctx.get())


def log_event(event_type: str, **fields: Any) -> None:
    """写入结构化事件日志（JSONL）。

    说明：
    - 每行一条事件，方便 pandas / duckdb 聚合分析。
    - 自动附加 bind_trace 提供的上下文字段。
    """
    _init_event_file()
    if _event_file_path is None:
        return

    event = dict(_trace_ctx.get())
    event.update(fields)
    event["event_type"] = event_type
    event.setdefault("timestamp", datetime.now().isoformat())

    with _event_lock:
        with open(_event_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def get_log_file_path() -> Optional[Path]:
    """获取当前日志文件路径"""
    return _log_file_path


def get_event_file_path() -> Optional[Path]:
    """获取当前事件日志文件路径。"""
    return _event_file_path
