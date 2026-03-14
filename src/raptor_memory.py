# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from src.config import get_config
from src.logger import bind_trace, get_trace_context, log_event
from src.memory_interface import BaseMemorySystem, Evidence
from src.token_utils import estimate_messages_tokens, estimate_text_tokens

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_RAPTOR = PROJECT_ROOT / "third_party" / "raptor"
if THIRD_PARTY_RAPTOR.exists() and str(THIRD_PARTY_RAPTOR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_RAPTOR))

try:
    from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
    from raptor.EmbeddingModels import BaseEmbeddingModel
    from raptor.QAModels import BaseQAModel
    from raptor.SummarizationModels import BaseSummarizationModel
except Exception as e:
    raise ImportError(
        "未找到 RAPTOR 依赖。请先安装官方 RAPTOR，并确保它位于 third_party/raptor，"
        "或已加入 PYTHONPATH。"
    ) from e


class _NoQAModel(BaseQAModel):
    def answer_question(self, *args, **kwargs):
        return ""


class _CompatEmbeddingModel(BaseEmbeddingModel):
    def __init__(self):
        conf = get_config().embedding
        self.provider = conf.get("provider", "openai_compat")
        self.base_url = conf.get("base_url")
        self.api_key = conf.get("api_key")
        self.model = conf.get("model", "text-embedding-3-small")

        if self.provider != "ark_multimodal":
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self._client = None

    def create_embedding(self, text: str):
        """创建 embedding，并将 token/耗时写入结构化事件日志。"""
        text = text.replace("\n", " ")
        ctx = get_trace_context()
        stage = ctx.get("stage", "unknown")
        token_bucket = ctx.get("token_bucket", "unknown")
        start = datetime.now(timezone.utc)

        prompt_tokens = None
        total_tokens = None
        estimated_tokens = None
        token_source = "api_usage"

        if self.provider == "ark_multimodal":
            url = (self.base_url or "").rstrip("/") + "/embeddings/multimodal"
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + str(self.api_key),
            }
            payload = {"model": self.model, "input": [{"type": "text", "text": text}]}
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            body = r.json()
            usage = body.get("usage", {}) if isinstance(body, dict) else {}
            prompt_tokens = usage.get("prompt_tokens")
            total_tokens = usage.get("total_tokens")
            embedding = body["data"]["embedding"]
        else:
            resp = self._client.embeddings.create(input=text, model=self.model)
            usage = getattr(resp, "usage", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)
            embedding = resp.data[0].embedding

        if total_tokens is None:
            estimated_tokens = estimate_text_tokens(text, self.model)
            prompt_tokens = estimated_tokens
            total_tokens = estimated_tokens
            token_source = "tokenizer_estimate"

        end = datetime.now(timezone.utc)
        log_event(
            "model_call",
            stage=stage,
            token_bucket=token_bucket,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            total_tokens=total_tokens,
            estimated_tokens=estimated_tokens,
            token_source=token_source,
            start_time=start.isoformat(),
            end_time=end.isoformat(),
            latency_ms=(end - start).total_seconds() * 1000,
            success=True,
        )
        return embedding


class _CompatSummarizationModel(BaseSummarizationModel):
    def __init__(self):
        conf = get_config().llm
        from openai import OpenAI

        self._client = OpenAI(api_key=conf.get("api_key"), base_url=conf.get("base_url"))
        self.model = conf.get("model")

    def summarize(self, context, max_tokens=180):
        # 1) 防止 max_tokens 意外为 0/None
        try:
            max_tokens = int(max_tokens)
        except Exception:
            max_tokens = 180
        if max_tokens < 32:
            max_tokens = 32

        start = datetime.now(timezone.utc)
        messages = [
            {"role": "system", "content": "You are a careful summarizer."},
            {
                "role": "user",
                "content": f"Summarize the following. Keep key entities, dates, decisions:\n{context}",
            },
        ]
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens,
        )

        content = ""
        try:
            content = resp.choices[0].message.content or ""
        except Exception:
            content = ""

        if not content.strip():
            try:
                raw = resp.model_dump() if hasattr(resp, "model_dump") else resp.dict()
                msg = raw.get("choices", [{}])[0].get("message", {})
                c = msg.get("content", "")

                if isinstance(c, str):
                    content = c
                elif isinstance(c, list):
                    buf = []
                    for part in c:
                        if isinstance(part, dict):
                            buf.append(part.get("text") or part.get("content") or "")
                        elif isinstance(part, str):
                            buf.append(part)
                    content = "".join(buf)
            except Exception:
                pass

        if not content.strip():
            content = (context or "")[:2000]

        usage = getattr(resp, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
            estimated_tokens = None
            token_source = "api_usage"
        else:
            prompt_tokens = estimate_messages_tokens(messages, self.model)
            completion_tokens = estimate_text_tokens(content, self.model)
            total_tokens = int(prompt_tokens) + int(completion_tokens)
            estimated_tokens = total_tokens
            token_source = "tokenizer_estimate"

        end = datetime.now(timezone.utc)
        log_event(
            "model_call",
            stage="ingest.summarization.parent",
            token_bucket="ingest_summarization",
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_tokens=estimated_tokens,
            token_source=token_source,
            start_time=start.isoformat(),
            end_time=end.isoformat(),
            latency_ms=(end - start).total_seconds() * 1000,
            success=True,
        )

        return content


class RaptorTreeMemory(BaseMemorySystem):
    def __init__(self, tree_path: Optional[str] = None, tb_num_layers: int = 3):
        self._buffer: List[str] = []
        emb = _CompatEmbeddingModel()
        summ = _CompatSummarizationModel()

        self._config = RetrievalAugmentationConfig(
            embedding_model=emb,
            summarization_model=summ,
            qa_model=_NoQAModel(),
            tb_num_layers=tb_num_layers,
            tb_max_tokens=200,
            tb_summarization_length=120,
            tr_threshold=0.5,
            tr_top_k=5,
        )
        self._ra = RetrievalAugmentation(config=self._config, tree=tree_path)

    def add_memory(self, data: str, metadata: Dict[str, Any]) -> None:
        self._buffer.append(data)

    def build_tree(self) -> None:
        if self._ra.tree is not None:
            return
        text = "\n\n".join(self._buffer)

        start = datetime.now(timezone.utc)
        with bind_trace(stage="ingest.embedding.leaf", token_bucket="ingest_embedding"):
            self._ra.add_documents(text)
        end = datetime.now(timezone.utc)

        tree = self._ra.tree
        layer_to_nodes = getattr(tree, "layer_to_nodes", {}) or {}
        all_nodes = getattr(tree, "all_nodes", {}) or {}
        log_event(
            "tree_build",
            stage="ingest.tree_build",
            tree_layers=len(layer_to_nodes),
            node_count=len(all_nodes),
            start_time=start.isoformat(),
            end_time=end.isoformat(),
            latency_ms=(end - start).total_seconds() * 1000,
            success=True,
        )

    def save_tree(self, path: str) -> None:
        self.build_tree()
        self._ra.save(path)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        max_tokens: int = 3500,
        stage: str = "infer.retrieve",
    ) -> List[Evidence]:
        """执行检索并记录检索元信息。

        说明：
        - 显式暴露 max_tokens，便于做 budget sweep。
        - 去除后置 query embedding 打分，避免污染推理成本统计。
        """
        self.build_tree()

        with bind_trace(
            stage=stage,
            token_bucket="infer_retrieval_embedding",
            query=query,
            top_k=top_k,
        ):
            context, layer_info = self._ra.retrieve(
                question=query,
                top_k=top_k,
                max_tokens=max_tokens,
                collapse_tree=True,
                return_layer_information=True,
            )

        evidences: List[Evidence] = []
        for item in layer_info:
            idx = int(item["node_index"])
            layer = int(item["layer_number"])
            node = self._ra.tree.all_nodes[idx]
            evidences.append(
                Evidence(
                    content=node.text,
                    metadata={
                        "source": "RAPTOR",
                        "node_index": idx,
                        "layer": layer,
                        "score": None,
                        "score_source": "raptor_default",
                    },
                )
            )

        if not evidences and isinstance(context, str) and context.strip():
            evidences.append(
                Evidence(
                    content=context,
                    metadata={
                        "source": "RAPTOR",
                        "node_index": -1,
                        "layer": -1,
                        "score": 0.0,
                        "score_source": "fallback_context",
                    },
                )
            )

        tree = self._ra.tree
        layer_to_nodes = getattr(tree, "layer_to_nodes", {}) or {}
        all_nodes = getattr(tree, "all_nodes", {}) or {}
        retrieved_layers = sorted(
            {int(item.get("layer_number", -1)) for item in layer_info if "layer_number" in item}
        )
        log_event(
            "retrieval",
            stage=stage,
            query=query,
            top_k=top_k,
            tree_layers=len(layer_to_nodes),
            node_count=len(all_nodes),
            retrieved_layers=retrieved_layers,
            retrieved_node_count=len(layer_info),
            success=True,
        )

        return evidences

    def reset(self) -> None:
        self._buffer.clear()
        self._ra = RetrievalAugmentation(config=self._config, tree=None)
