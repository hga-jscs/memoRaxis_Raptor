#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""驱动脚本。

该脚本用于演示三种推理范式适配器（R1/R2/R3）的完整调用链：
1) 初始化日志、配置、记忆系统与 LLM 客户端；
2) 在同一个测试任务上分别运行三个适配器；
3) 输出回答与关键运行指标，便于快速对比行为差异。

为了便于调试，脚本会在控制台输出结构化分段信息。
"""

import sys
from pathlib import Path

# 确保 src 目录在路径中
sys.path.insert(0, str(Path(__file__).parent))

from src.adaptors import IterativeAdaptor, PlanAndActAdaptor, SingleTurnAdaptor
from src.config import get_config
from src.llm_interface import MockLLMClient, OpenAIClient
from src.logger import get_log_file_path, get_logger
from src.simple_memory import SimpleRAGMemory


def print_result(adaptor_name: str, result):
    """打印单个适配器执行结果。

    Args:
        adaptor_name: 适配器展示名称。
        result: 适配器运行结果对象（包含答案、步骤数、token 等指标）。
    """
    print(f"\n{'='*60}")
    print(f"适配器: {adaptor_name}")
    print(f"{'='*60}")
    print(f"回答: {result.answer}")
    print(f"\n--- 运行指标 ---")
    print(f"执行步数: {result.steps_taken}")
    print(f"Token 消耗: {result.token_consumption}")
    if result.replan_count > 0:
        print(f"重规划次数: {result.replan_count}")
    print(f"收集证据数: {len(result.evidence_collected)}")


def main():
    """程序主入口。

    流程说明：
    - 优先初始化日志，确保后续异常可追踪；
    - 初始化记忆系统与 LLM 客户端（失败时降级到 Mock）；
    - 串行执行 R1/R2/R3，输出可视化分段结果。
    """
    # 初始化日志
    logger = get_logger()
    logger.info("程序启动")

    # 初始化组件
    # 使用真实的 PostgreSQL + pgvector 记忆系统
    try:
        memory = SimpleRAGMemory()
        print("Using SimpleRAGMemory (PostgreSQL + Vector)")
    except Exception as e:
        logger.error(f"Failed to init SimpleRAGMemory: {e}")
        print(f"Failed to init SimpleRAGMemory: {e}")
        return
    
    # 从配置文件读取 LLM 参数
    config = get_config()
    llm_conf = config.llm

    try:
        llm = OpenAIClient(
            api_key=llm_conf.get("api_key"),
            base_url=llm_conf.get("base_url"),
            model=llm_conf.get("model"),
        )
        logger.info("使用 OpenAI 客户端: model=%s", llm_conf.get("model"))
    except ImportError:
        logger.warning("未安装 openai 库，使用 MockLLMClient")
        llm = MockLLMClient()
    except Exception as e:
        logger.error("OpenAI 客户端初始化失败: %s，使用 MockLLMClient", e)
        llm = MockLLMClient()

    # 测试任务
    test_task = "什么是深度学习？它与机器学习有什么关系？"

    print("\n" + "="*60)
    print("Agent 推理范式适配器演示")
    print("="*60)
    print("[DEBUG] 已完成组件初始化，开始运行适配器..." )
    print(f"测试任务: {test_task}")

    # R1: SingleTurnAdaptor
    logger.info("开始测试 SingleTurnAdaptor")
    llm.reset_stats()
    r1 = SingleTurnAdaptor(llm_client=llm, memory_system=memory)
    result1 = r1.run(test_task)
    print_result("R1 SingleTurnAdaptor (一次检索直接生成)", result1)

    # R2: IterativeAdaptor
    logger.info("开始测试 IterativeAdaptor")
    llm.reset_stats()
    r2 = IterativeAdaptor(llm_client=llm, memory_system=memory, max_iterations=5)
    result2 = r2.run(test_task)
    print_result("R2 IterativeAdaptor (迭代式检索-判断循环)", result2)

    # R3: PlanAndActAdaptor
    logger.info("开始测试 PlanAndActAdaptor")
    llm.reset_stats()
    r3 = PlanAndActAdaptor(llm_client=llm, memory_system=memory, max_additions=2)
    result3 = r3.run(test_task)
    print_result("R3 PlanAndActAdaptor (先规划再执行)", result3)

    # 输出日志文件位置
    log_path = get_log_file_path()
    print(f"\n{'='*60}")
    print(f"日志文件: {log_path}")
    print("="*60)

    logger.info("程序结束")


if __name__ == "__main__":
    main()
