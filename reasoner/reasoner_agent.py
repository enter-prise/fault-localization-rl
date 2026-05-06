# reasoner/reasoner_agent.py

import re
from typing import Dict, List, Tuple, Optional, Any

from utils.llm import LLMWrapper


class ReasonerAgent:
    """
    混合版 Reasoner：
    1. 规则打分
    2. LLM 对 top-k 候选做语义选择
    3. 融合成最终重排结果

    兼容用途：
    - rerank(query, candidates) -> List[(node_id, score)]
    - choose_best(query, candidates) -> (node_id, score) / (None, 0.0)
    - reason(query, candidates) -> Dict
    
    输入 candidates 支持两种格式：
    1. List[Tuple[str, float]]: [(node_id, score), ...]
    2. List[Dict]: [{"entity_id": xxx, "relevance_score": xxx}, ...]
    """

    def __init__(
        self,
        graph,
        retriever=None,
        model_name: str = "qwen2.5-coder:32b",
        use_llm: bool = True,
    ):
        self.graph = graph
        self.retriever = retriever
        self.use_llm = use_llm

        # 注意：LLMWrapper 参数名是 model，不是 model_name
        self.llm = LLMWrapper(model=model_name) if use_llm else None

        self.last_trace = None

    # ------------------------------------------------------------------
    # Basic text utilities
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        text = str(text).lower()
        text = text.replace("_", " ")
        text = text.replace("/", " ")
        text = re.sub(r"[^a-zA-Z0-9\.\-\s]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        return text.split()

    def _overlap_score(self, query: str, text: str) -> float:
        q_tokens = set(self._tokenize(query))
        t_tokens = set(self._tokenize(text))
        if not q_tokens or not t_tokens:
            return 0.0
        return len(q_tokens.intersection(t_tokens)) / max(len(q_tokens), 1)

    # ------------------------------------------------------------------
    # Graph / node helpers
    # ------------------------------------------------------------------

    def _get_node(self, node_id: str):
        if node_id not in self.graph.nodes:
            raise KeyError(f"Node id not found in graph: {node_id}")
        return self.graph.nodes[node_id]

    def _safe_attr(self, obj: Any, attr: str, default: Any = "") -> Any:
        return getattr(obj, attr, default)

    def _type_bonus(self, node_type: str) -> float:
        if node_type == "function":
            return 0.30
        if node_type == "class":
            return 0.15
        if node_type == "file":
            return 0.05
        return 0.0

    def _neighbor_bonus(self, node_id: str) -> float:
        try:
            neighbors = self.graph.get_neighbors(node_id)
            degree = len(neighbors) if neighbors is not None else 0
        except Exception:
            degree = 0
        return min(degree / 20.0, 1.0) * 0.10

    # ------------------------------------------------------------------
    # Candidate normalization (关键修改)
    # ------------------------------------------------------------------
    
    def _normalize_candidates(self, candidates: List[Any]) -> List[Tuple[str, float]]:
        """
        将不同格式的 candidates 统一转换为 (node_id, score) 元组列表
        
        支持格式：
        1. List[Tuple[str, float]]: [(node_id, score), ...]
        2. List[Dict]: [{"entity_id": xxx, "relevance_score": xxx}, ...]
        """
        if not candidates:
            return []
        
        normalized = []
        for item in candidates:
            # 格式1: 字典格式
            if isinstance(item, dict):
                # 尝试多种可能的键名
                node_id = item.get("entity_id") or item.get("node_id") or item.get("id")
                # 尝试多种可能的分数键名
                score = item.get("relevance_score") or item.get("score") or item.get("retrieval_score") or 0.5
                if node_id:
                    try:
                        normalized.append((str(node_id), float(score)))
                    except (ValueError, TypeError):
                        normalized.append((str(node_id), 0.5))
            # 格式2: 元组或列表格式
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                try:
                    normalized.append((str(item[0]), float(item[1])))
                except (ValueError, TypeError):
                    normalized.append((str(item[0]), 0.5))
            # 格式3: 只有 node_id 的字符串
            elif isinstance(item, str):
                normalized.append((item, 0.5))
            else:
                # 未知格式，尝试转换为字符串
                try:
                    normalized.append((str(item), 0.5))
                except Exception:
                    pass
        
        return normalized

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def score_candidate(self, query: str, node_id: str, retrieval_score: float) -> float:
        node = self._get_node(node_id)

        node_name = self._safe_attr(node, "name", "")
        node_type = self._safe_attr(node, "type", "")
        node_path = self._safe_attr(node, "file_path", "")
        node_doc = self._safe_attr(node, "doc", "")
        node_code = self._safe_attr(node, "code", "")

        name_overlap = self._overlap_score(query, node_name)
        path_overlap = self._overlap_score(query, node_path)
        doc_overlap = self._overlap_score(query, node_doc)
        code_overlap = self._overlap_score(query, str(node_code)[:300])

        score = 0.0

        # retrieval score 主权重
        score += min(float(retrieval_score) / 20.0, 1.0) * 0.50

        # 文本/语义重合
        score += name_overlap * 0.25
        score += path_overlap * 0.20
        score += doc_overlap * 0.15
        score += code_overlap * 0.10

        # 类型与图结构启发
        score += self._type_bonus(node_type)
        score += self._neighbor_bonus(node_id)

        return score

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(self, query: str, candidates: List[Tuple[str, float]]) -> str:
        lines = []

        for i, (node_id, score) in enumerate(candidates):
            node = self._get_node(node_id)

            node_name = self._safe_attr(node, "name", "")
            node_type = self._safe_attr(node, "type", "")
            node_path = self._safe_attr(node, "file_path", "")
            node_doc = self._safe_attr(node, "doc", "")
            node_code = self._safe_attr(node, "code", "")

            code_preview = str(node_code)[:200].replace("\n", " ")
            doc_preview = str(node_doc)[:120].replace("\n", " ")

            lines.append(
                f"{i}. "
                f"name={node_name} | "
                f"type={node_type} | "
                f"path={node_path} | "
                f"rerank_score={round(score, 4)} | "
                f"doc_preview={doc_preview} | "
                f"code_preview={code_preview}"
            )

        prompt = f"""
You are a software fault localization reasoning assistant.

Task:
Given a bug description and several candidate code entities, choose the MOST likely bug location.

Bug description:
{query}

Candidate nodes:
{chr(10).join(lines)}

Selection rule:
Prefer the candidate that is most likely to be the actual bug root cause, based on:
1. semantic relevance to the bug description,
2. suspiciousness from code context,
3. whether the entity looks like the place where the faulty behavior originates.

Return EXACTLY in this format:
INDEX: <number>
REASON: <short reason>
""".strip()

        return prompt

    def _parse_llm_choice(self, text: str, num_candidates: int) -> Optional[int]:
        if not text:
            return None
        match = re.search(r"INDEX\s*:\s*(\d+)", text, re.IGNORECASE)
        if not match:
            return None
        idx = int(match.group(1))
        if 0 <= idx < num_candidates:
            return idx
        return None

    def _extract_llm_reason(self, text: str) -> str:
        if not text:
            return ""
        match = re.search(r"REASON\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    # ------------------------------------------------------------------
    # Main rerank logic
    # ------------------------------------------------------------------

    def rerank(self, query: str, candidates: List[Any]) -> List[Tuple[str, float]]:
        """
        输入:
            query: bug / issue 描述
            candidates: 支持两种格式：
                1. [(node_id, retrieval_score), ...]  # 元组列表
                2. [{"entity_id": xxx, "relevance_score": xxx}, ...]  # 字典列表

        输出:
            [(node_id, fused_score), ...]  按降序排列
        """
        if not candidates:
            self.last_trace = {
                "mode": "empty",
                "raw_output": None,
                "chosen_index": None,
                "llm_reason": "",
                "top_after_rerank": [],
            }
            return []

        # 关键修改：统一转换为标准格式
        normalized_candidates = self._normalize_candidates(candidates)
        
        if not normalized_candidates:
            self.last_trace = {
                "mode": "normalization_failed",
                "raw_output": None,
                "chosen_index": None,
                "llm_reason": "Failed to normalize candidates",
                "top_after_rerank": [],
            }
            return []

        rescored = []
        score_details = []

        for node_id, retrieval_score in normalized_candidates:
            try:
                base_score = self.score_candidate(query, node_id, retrieval_score)
                rescored.append((node_id, base_score))
                score_details.append(
                    {
                        "node_id": node_id,
                        "retrieval_score": retrieval_score,
                        "base_score": base_score,
                    }
                )
            except Exception as e:
                score_details.append(
                    {
                        "node_id": node_id,
                        "retrieval_score": retrieval_score,
                        "base_score": None,
                        "error": str(e),
                    }
                )

        rescored.sort(key=lambda x: x[1], reverse=True)

        llm_raw_output = None
        llm_choice_idx = None
        llm_reason = ""

        if self.use_llm and self.llm is not None and rescored:
            top_candidates = rescored[: min(5, len(rescored))]
            prompt = self._build_prompt(query, top_candidates)

            try:
                llm_raw_output = self.llm.generate(prompt)
                llm_choice_idx = self._parse_llm_choice(llm_raw_output, len(top_candidates))
                llm_reason = self._extract_llm_reason(llm_raw_output)

                if llm_choice_idx is not None:
                    boosted = []
                    for i, (node_id, score) in enumerate(top_candidates):
                        if i == llm_choice_idx:
                            score += 0.75
                        boosted.append((node_id, score))

                    remaining = rescored[len(top_candidates):]
                    rescored = boosted + remaining
                    rescored.sort(key=lambda x: x[1], reverse=True)
            except Exception as e:
                llm_raw_output = f"LLM call failed: {e}"
                llm_choice_idx = None
                llm_reason = "LLM failed; kept rule-based reranking."

        self.last_trace = {
            "mode": "llm_hybrid" if self.use_llm else "rule_only",
            "raw_output": llm_raw_output,
            "chosen_index": llm_choice_idx,
            "llm_reason": llm_reason,
            "score_details": score_details,
            "top_after_rerank": rescored[:5],
        }

        return rescored

    def choose_best(self, query: str, candidates: List[Any]):
        """
        返回:
            (node_id, score)
            若为空则返回 (None, 0.0)
            
        支持两种 candidates 格式
        """
        if not candidates:
            return None, 0.0

        reranked = self.rerank(query, candidates)
        if not reranked:
            return None, 0.0

        return reranked[0]

    # ------------------------------------------------------------------
    # Compatibility API for env / pipeline
    # ------------------------------------------------------------------

    def reason(self, query: str, candidates: List[Any]) -> Dict:
        """
        提供给 rl_env 或主流程调用的统一接口。

        返回结构化结果，例如：
        {
            "best_node_id": "...",
            "best_score": 1.23,
            "ranked_candidates": [...],
            "trace": {...}
        }
        
        支持两种 candidates 格式
        """
        reranked = self.rerank(query, candidates)

        if not reranked:
            return {
                "best_node_id": None,
                "best_score": 0.0,
                "ranked_candidates": [],
                "trace": self.last_trace,
            }

        best_node_id, best_score = reranked[0]

        return {
            "best_node_id": best_node_id,
            "best_score": best_score,
            "ranked_candidates": reranked,
            "trace": self.last_trace,
        }