import pandas as pd
import re
from pathlib import Path
from typing import Any, Dict, List

from src.logger import get_logger

logger = get_logger()


def parse_instance_indices(idx_str: str) -> List[int]:
    """
    解析索引范围字符串。
    支持格式: "0", "0-5", "1,3,5", "0-2,5"
    """
    indices = set()
    parts = idx_str.split(",")
    for part in parts:
        part = part.strip()
        if "-" in part:
            try:
                start, end = map(int, part.split("-"))
                indices.update(range(start, end + 1))
            except ValueError:
                logger.warning(f"Invalid range format: {part}")
        else:
            try:
                indices.add(int(part))
            except ValueError:
                logger.warning(f"Invalid index format: {part}")
    return sorted(list(indices))


def load_benchmark_data(file_path: str, instance_idx: int) -> Dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    try:
        df = pd.read_parquet(str(path))
        if instance_idx >= len(df):
            raise IndexError(f"Instance index {instance_idx} out of range (total {len(df)})")
        data = df.iloc[instance_idx].to_dict()
        logger.info(f"Loaded instance {instance_idx} from {path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def split_text_exact(context: str, target_chunks: int) -> List[str]:
    """
    按字符数将全文尽量均匀地切成“恰好 target_chunks 份”。
    这是固定 chunk 数成本实验专用模式，优先保证 chunk 数可控。
    """
    text = (context or "").strip()
    if not text:
        return []

    target_chunks = int(target_chunks)
    if target_chunks <= 0:
        raise ValueError("target_chunks must be a positive integer")
    if target_chunks == 1:
        return [text]

    n = len(text)
    target_chunks = min(target_chunks, n)

    base = n // target_chunks
    remainder = n % target_chunks

    chunks: List[str] = []
    start = 0
    for i in range(target_chunks):
        end = start + base + (1 if i < remainder else 0)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end

    logger.info(
        "Chunking Strategy: Exact text split by target_chunks. Result: %s chunks.",
        len(chunks),
    )
    return chunks


def group_units_exact(units: List[str], target_chunks: int, joiner: str = "\n") -> List[str]:
    """
    将原子单元（如 facts / dialogues / lines）按顺序压成恰好 target_chunks 个 chunk。
    若原子单元数不足，则退回到全文等分模式。
    """
    clean_units = [u.strip() for u in units if isinstance(u, str) and u.strip()]
    if not clean_units:
        return []

    target_chunks = int(target_chunks)
    if target_chunks <= 0:
        raise ValueError("target_chunks must be a positive integer")
    if target_chunks == 1:
        return [joiner.join(clean_units)]

    if len(clean_units) < target_chunks:
        return split_text_exact(joiner.join(clean_units), target_chunks)

    lengths = [len(u) for u in clean_units]
    chunks: List[str] = []
    start = 0

    for remaining_chunks in range(target_chunks, 0, -1):
        if remaining_chunks == 1:
            end = len(clean_units)
        else:
            remaining_lengths = lengths[start:]
            target_chars = sum(remaining_lengths) / remaining_chunks

            cur = 0
            end = start
            max_end = len(clean_units) - (remaining_chunks - 1)

            while end < max_end:
                next_len = lengths[end]
                if cur > 0 and cur + next_len > target_chars:
                    break
                cur += next_len
                end += 1

            if end == start:
                end = start + 1

        chunk = joiner.join(clean_units[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start = end

    logger.info(
        "Chunking Strategy: Exact grouped units by target_chunks. Result: %s chunks.",
        len(chunks),
    )
    return chunks


def chunk_context(
    context: str,
    chunk_size: int = 850,
    overlap: int = 50,
    target_chunks: int | None = None,
) -> List[str]:
    """
    将长 Context 切分为文档片段。
    策略优先级：
    1. 若 target_chunks 非空：进入固定 chunk 数实验模式，直接等分全文。
    2. 否则优先尝试 "Document N:" 标记切分。
    3. 如果没有标记，使用固定长度滑动窗口切分。
    """
    if target_chunks is not None:
        return split_text_exact(context, target_chunks)

    regex_chunks = re.split(r"Document \d+:\n", context)
    valid_regex_chunks = [c.strip() for c in regex_chunks if len(c.strip()) > 10]

    if len(valid_regex_chunks) > 1:
        logger.info(
            "Chunking Strategy: Regex split ('Document N:'). Result: %s chunks.",
            len(valid_regex_chunks),
        )
        return valid_regex_chunks

    logger.info("Chunking Strategy: Fallback to Fixed-size Sliding Window.")
    chunks = []
    start = 0
    text_len = len(context)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = context[start:end]
        chunks.append(chunk)
        if end == text_len:
            break
        start += chunk_size - overlap

    logger.info(f"Result: {len(chunks)} chunks (size={chunk_size}, overlap={overlap}).")
    return chunks
