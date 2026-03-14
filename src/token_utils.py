# -*- coding: utf-8 -*-
"""Token 估算工具。

当上游 API usage 缺失时，使用 tokenizer（优先 tiktoken）给出可复现估算值。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def _get_encoding(model: str):
    try:
        import tiktoken

        try:
            return tiktoken.encoding_for_model(model)
        except Exception:
            return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def estimate_text_tokens(text: str, model: str) -> int:
    enc = _get_encoding(model)
    if enc is None:
        # 退化估算：1 token ~= 4 chars
        return max(1, len(text or "") // 4)
    return len(enc.encode(text or ""))


def estimate_messages_tokens(messages: List[Dict[str, Any]], model: str) -> int:
    payload = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return estimate_text_tokens(payload, model)
