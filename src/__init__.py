# -*- coding: utf-8 -*-
"""Agent 推理范式适配器模块"""

from .logger import get_logger
from .config import Config, get_config
from .memory_interface import Evidence, BaseMemorySystem, MockMemory
from .llm_interface import BaseLLMClient, MockLLMClient, OpenAIClient
from .adaptors import SingleTurnAdaptor, IterativeAdaptor, PlanAndActAdaptor

try:
    from .simple_memory import SimpleRAGMemory
except Exception:
    SimpleRAGMemory = None

__all__ = [
    "get_logger",
    "Config",
    "get_config",
    "Evidence",
    "BaseMemorySystem",
    "MockMemory",
    "BaseLLMClient",
    "MockLLMClient",
    "OpenAIClient",
    "SingleTurnAdaptor",
    "IterativeAdaptor",
    "PlanAndActAdaptor",
]

if SimpleRAGMemory is not None:
    __all__.append("SimpleRAGMemory")