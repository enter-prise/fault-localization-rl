import re
from typing import Dict, List, Optional, Any
from utils.llm import LLMWrapper


class VerifierAgent:
    """
    混合版 debate / verifier：
    1. 规则支持/反对证据
    2. LLM 生成支持/反对/结论
    3. 合并后给出 verdict

    兼容用途：
    - debate(query, candidate_node_id, bug_node_id=None) -> Dict
    - verify(...) -> 兼容 rl_env / 主流程
    """

    def __init__(
        self,
        graph,
        model_name: str = "qwen2.5-coder:32b",
        use_llm: bool = True,
    ):
        self.graph = graph
        self.use_llm = use_llm

        # 注意：LLMWrapper 参数名是 model，不是 model_name
        self.llm = LLMWrapper(model=model_name) if use_llm else None

        self.last_trace = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_node(self, node_id: str):
        if node_id not in self.graph.nodes:
            raise KeyError(f"Node id not found in graph: {node_id}")
        return self.graph.nodes[node_id]

    def _safe_attr(self, obj: Any, attr: str, default: Any = "") -> Any:
        return getattr(obj, attr, default)

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

    def _overlap(self, a: str, b: str) -> float:
        a_tokens = set(self._tokenize(a))
        b_tokens = set(self._tokenize(b))
        if not a_tokens or not b_tokens:
            return 0.0
        return len(a_tokens.intersection(b_tokens)) / max(len(a_tokens), 1)

    # ------------------------------------------------------------------
    # Prompt / parsing
    # ------------------------------------------------------------------

    def _build_prompt(self, query: str, candidate_node_id: str) -> str:
        node = self._get_node(candidate_node_id)

        node_name = self._safe_attr(node, "name", "")
        node_type = self._safe_attr(node, "type", "")
        node_path = self._safe_attr(node, "file_path", "")
        node_doc = self._safe_attr(node, "doc", "")
        node_code = self._safe_attr(node, "code", "")

        code_preview = str(node_code)[:400]

        prompt = f"""
You are a software debugging debate agent.

Task:
Evaluate whether the candidate node is likely to be a genuine bug location.

Bug description:
{query}

Candidate node:
Name: {node_name}
Type: {node_type}
Path: {node_path}
Doc: {node_doc}
Code Preview:
{code_preview}

Return EXACTLY in this format:
SUPPORT: <one short paragraph>
OPPOSE: <one short paragraph>
VERDICT: <yes or no>
""".strip()

        return prompt

    def _parse_llm_debate(self, text: str):
        if not text:
            return "", "", False

        support_match = re.search(
            r"SUPPORT\s*:\s*(.*?)(?:OPPOSE\s*:|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        oppose_match = re.search(
            r"OPPOSE\s*:\s*(.*?)(?:VERDICT\s*:|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        verdict_match = re.search(
            r"VERDICT\s*:\s*(yes|no)",
            text,
            re.IGNORECASE,
        )

        support_text = support_match.group(1).strip() if support_match else ""
        oppose_text = oppose_match.group(1).strip() if oppose_match else ""
        verdict = verdict_match.group(1).strip().lower() == "yes" if verdict_match else False

        return support_text, oppose_text, verdict

    # ------------------------------------------------------------------
    # Main debate logic
    # ------------------------------------------------------------------

    def debate(
        self,
        query: str,
        candidate_node_id: str,
        bug_node_id: Optional[str] = None,
    ) -> Dict:
        """
        真实验证接口。

        参数:
            query: bug / issue 描述
            candidate_node_id: 候选节点
            bug_node_id: 训练/评估时可选的 oracle 节点

        返回:
            {
                "candidate_node_id": ...,
                "candidate_name": ...,
                "support_score": ...,
                "oppose_score": ...,
                "verdict": ...,
                "confidence": ...,
                "evidence": [...],
                "llm_raw_output": ...
            }
        """
        node = self._get_node(candidate_node_id)

        node_name = self._safe_attr(node, "name", "")
        node_type = self._safe_attr(node, "type", "")
        node_path = self._safe_attr(node, "file_path", "")
        node_doc = self._safe_attr(node, "doc", "")
        node_code = self._safe_attr(node, "code", "")

        support = 0.0
        oppose = 0.0
        evidence = []

        # ---------- 规则证据 ----------
        name_overlap = self._overlap(query, node_name)
        path_overlap = self._overlap(query, node_path)
        doc_overlap = self._overlap(query, node_doc)
        code_overlap = self._overlap(query, str(node_code)[:300])

        support += name_overlap * 0.40
        support += path_overlap * 0.30
        support += doc_overlap * 0.20
        support += code_overlap * 0.10

        if name_overlap > 0:
            evidence.append(f"name overlap={round(name_overlap, 4)}")
        if path_overlap > 0:
            evidence.append(f"path overlap={round(path_overlap, 4)}")
        if doc_overlap > 0:
            evidence.append(f"doc overlap={round(doc_overlap, 4)}")
        if code_overlap > 0:
            evidence.append(f"code overlap={round(code_overlap, 4)}")

        if node_type == "function":
            support += 0.10
            evidence.append("candidate is a function node")
        elif node_type == "class":
            support += 0.05
            evidence.append("candidate is a class node")
        elif node_type == "file":
            oppose += 0.02
            evidence.append("candidate is only a file-level node")
        else:
            oppose += 0.05
            evidence.append(f"candidate type={node_type} is weakly informative")

        try:
            neighbors = self.graph.get_neighbors(candidate_node_id)
            degree = len(neighbors) if neighbors is not None else 0
        except Exception:
            degree = 0

        if degree == 0:
            oppose += 0.10
            evidence.append("candidate has no graph neighbors")
        else:
            support += min(degree / 20.0, 1.0) * 0.05
            evidence.append(f"candidate graph degree={degree}")

        # 训练 / benchmark 评估时可用
        if bug_node_id is not None:
            if candidate_node_id == bug_node_id:
                support += 0.60
                evidence.append("candidate matches oracle bug node")
            else:
                oppose += 0.10
                evidence.append("candidate does not match oracle bug node")

        # ---------- LLM 辩论 ----------
        llm_raw_output = None
        llm_support_text = ""
        llm_oppose_text = ""
        llm_verdict = None

        if self.use_llm and self.llm is not None:
            prompt = self._build_prompt(query, candidate_node_id)

            try:
                llm_raw_output = self.llm.generate(prompt)

                llm_support_text, llm_oppose_text, llm_verdict = self._parse_llm_debate(llm_raw_output)

                if llm_support_text:
                    evidence.append(f"llm_support: {llm_support_text[:120]}")
                    support += 0.20

                if llm_oppose_text:
                    evidence.append(f"llm_oppose: {llm_oppose_text[:120]}")
                    oppose += 0.10

                if llm_verdict is True:
                    support += 0.20
                    evidence.append("llm verdict=yes")
                elif llm_verdict is False:
                    oppose += 0.10
                    evidence.append("llm verdict=no")
            except Exception as e:
                llm_raw_output = f"LLM call failed: {e}"
                evidence.append("llm call failed; verdict falls back to rule-based scoring")

        verdict = "accept" if support >= oppose else "reject"
        confidence = round(support / (support + oppose), 2) if (support + oppose) > 0 else 0.0

        return {
            "candidate_node_id": candidate_node_id,
            "candidate_name": node_name,
            "candidate_type": node_type,
            "candidate_path": node_path,
            "support_score": round(support, 4),
            "oppose_score": round(oppose, 4),
            "verdict": verdict,
            "confidence": confidence,
            "evidence": evidence,
            "llm_raw_output": llm_raw_output,
        }

    # ------------------------------------------------------------------
    # Compatibility API
    # ------------------------------------------------------------------

    def verify(
        self,
        candidate_node_id: str,
        bug_node_id: Optional[str] = None,
        query: Optional[str] = None,
    ):
        """
        兼容接口，适配不同调用场景：

        1) RL 训练场景：
            verify(candidate_node_id, bug_node_id)
            -> bool

        2) 真实推理场景：
            verify(candidate_node_id, query="...")
            -> Dict

        说明：
        - 如果传了 bug_node_id 且没传 query，按训练逻辑返回 bool
        - 如果传了 query，就走真实 debate 逻辑返回 Dict
        """
        # 训练/评估兼容：直接用 oracle 判断
        if bug_node_id is not None and query is None:
            return candidate_node_id == bug_node_id

        # 真实推理：必须有 query
        if query is None:
            raise ValueError("verify() in real inference mode requires `query`.")

        return self.debate(query=query, candidate_node_id=candidate_node_id, bug_node_id=bug_node_id)