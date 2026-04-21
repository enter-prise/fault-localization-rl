from typing import Dict, List, Tuple, Optional, Any

from retrieval.vector_store import SimpleVectorStore


class Retriever:
    """
    混合检索器：
    1. BM25 检索
    2. 轻量语义检索（SimpleVectorStore）
    3. 混合打分融合

    目标：
    - 保持和你原系统接口尽量兼容
    - 支持：
        retrieve(query)
        retrieve_with_score(query, top_k=...)
        build_index(graph)

    默认策略：
    hybrid_score = alpha * bm25_norm + beta * semantic_norm + bonuses
    """

    def __init__(
        self,
        bm25,
        alpha: float = 0.6,
        beta: float = 0.4,
    ):
        self.bm25 = bm25
        self.alpha = alpha
        self.beta = beta

        self.graph = None
        self.vector_store = SimpleVectorStore()
        self.documents: Dict[str, str] = {}
        self.node_name_map: Dict[str, str] = {}
        self._built = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _safe_attr(self, obj, attr: str, default=""):
        return getattr(obj, attr, default)

    def _build_document_for_node(self, node) -> str:
        """
        把图节点转成用于检索的文档文本
        """
        name = self._safe_attr(node, "name", "") or ""
        node_type = self._safe_attr(node, "type", "") or ""
        file_path = self._safe_attr(node, "file_path", "") or ""
        doc = self._safe_attr(node, "doc", "") or ""
        code = self._safe_attr(node, "code", "") or ""

        parts = [
            name,
            node_type,
            file_path,
            doc,
            str(code)[:500],
        ]

        return " ".join([str(x) for x in parts if x]).strip()

    def _normalize_scores(self, results: List[Tuple[str, float]]) -> Dict[str, float]:
        """
        把一组分数线性归一化到 [0,1]
        """
        if not results:
            return {}

        scores = [float(score) for _, score in results]
        max_score = max(scores)
        min_score = min(scores)

        normalized = {}
        if max_score == min_score:
            for doc_id, _ in results:
                normalized[doc_id] = 1.0
            return normalized

        for doc_id, score in results:
            normalized[doc_id] = (float(score) - min_score) / (max_score - min_score)

        return normalized

    def _keyword_bonus(self, query: str, doc_text: str) -> float:
        """
        轻量加成：如果 query 中关键字和 name/path/doc 有明显重合，增加一点分
        """
        if not query or not doc_text:
            return 0.0

        q_tokens = set(str(query).lower().split())
        d_tokens = set(str(doc_text).lower().split())

        if not q_tokens or not d_tokens:
            return 0.0

        overlap = len(q_tokens.intersection(d_tokens)) / max(len(q_tokens), 1)
        return min(overlap, 1.0) * 0.1

    # ------------------------------------------------------------------
    # Build index
    # ------------------------------------------------------------------

    def build_index(self, graph):
        self.graph = graph
        self.documents = {}
        self.node_name_map = {}

        for node_id, node in graph.nodes.items():
            node_id = str(node_id)
            self.documents[node_id] = self._build_document_for_node(node)
            self.node_name_map[node_id] = self._safe_attr(node, "name", "") or ""

        self.vector_store.build(self.documents)
        self._built = True

    # ------------------------------------------------------------------
    # BM25 adapters
    # ------------------------------------------------------------------

    def _bm25_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        兼容不同 BM25 实现。

        尝试以下可能接口：
        1. bm25.retrieve(query, top_k=top_k)
        2. bm25.search(query, top_k=top_k)
        3. bm25.get_scores(query) -> Dict or List
        """
        # 1) retrieve
        if hasattr(self.bm25, "retrieve"):
            try:
                results = self.bm25.retrieve(query, top_k=top_k)
                return [(str(doc_id), float(score)) for doc_id, score in results[:top_k]]
            except TypeError:
                try:
                    results = self.bm25.retrieve(query)
                    return [(str(doc_id), float(score)) for doc_id, score in results[:top_k]]
                except Exception:
                    pass
            except Exception:
                pass

        # 2) search
        if hasattr(self.bm25, "search"):
            try:
                results = self.bm25.search(query, top_k=top_k)
                return [(str(doc_id), float(score)) for doc_id, score in results[:top_k]]
            except TypeError:
                try:
                    results = self.bm25.search(query)
                    return [(str(doc_id), float(score)) for doc_id, score in results[:top_k]]
                except Exception:
                    pass
            except Exception:
                pass

        # 3) get_scores
        if hasattr(self.bm25, "get_scores"):
            try:
                raw_scores = self.bm25.get_scores(query)
                if isinstance(raw_scores, dict):
                    results = [(str(doc_id), float(score)) for doc_id, score in raw_scores.items()]
                    results.sort(key=lambda x: x[1], reverse=True)
                    return results[:top_k]

                if isinstance(raw_scores, list):
                    # 假设顺序与 self.documents.keys() 一一对应
                    doc_ids = list(self.documents.keys())
                    results = []
                    for i, score in enumerate(raw_scores):
                        if i < len(doc_ids):
                            results.append((str(doc_ids[i]), float(score)))
                    results.sort(key=lambda x: x[1], reverse=True)
                    return results[:top_k]
            except Exception:
                pass

        # 兜底：没有可用 BM25 接口
        return []

    # ------------------------------------------------------------------
    # Public retrieval API
    # ------------------------------------------------------------------

    def retrieve_with_score(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        返回混合检索结果：
            [{"entity_id": node_id, "entity_name": node_name, "entity_type": node_type, 
            "file_path": node_path, "code_snippet": code_snippet, "match_source": "bm25"|"semantic", 
            "relevance_score": score}]
        """
        if not self._built:
            raise ValueError("Retriever index not built. Call build_index(graph) first.")

        bm25_results = self._bm25_search(query, top_k=max(top_k * 3, 20))
        semantic_results = self.vector_store.search(query, top_k=max(top_k * 3, 20))

        bm25_norm = self._normalize_scores(bm25_results)
        semantic_norm = self._normalize_scores(semantic_results)

        all_doc_ids = set(bm25_norm.keys()) | set(semantic_norm.keys())
        fused_results = []

        for doc_id in all_doc_ids:
            bm25_score = bm25_norm.get(doc_id, 0.0)
            semantic_score = semantic_norm.get(doc_id, 0.0)

            hybrid_score = self.alpha * bm25_score + self.beta * semantic_score

            doc_text = self.documents.get(doc_id, "")
            hybrid_score += self._keyword_bonus(query, doc_text)

            # 修复：使用 getattr 而不是 .get()，因为 node 是对象
            node = self.graph.nodes.get(doc_id)
            if node is not None:
                node_type = getattr(node, "type", "")
                if node_type == "function":
                    hybrid_score += 0.05
                elif node_type == "class":
                    hybrid_score += 0.03
                elif node_type == "file":
                    hybrid_score += 0.01
                
                # 修复：使用 getattr 获取属性
                file_path = getattr(node, "file_path", "")
                code = getattr(node, "code", "")
                code_snippet = str(code)[:400] if code else ""
            else:
                node_type = ""
                file_path = ""
                code_snippet = ""

            fused_results.append({
                "entity_id": doc_id,
                "entity_name": self.node_name_map.get(doc_id, ""),
                "entity_type": node_type,
                "file_path": file_path,
                "code_snippet": code_snippet,
                "match_source": "hybrid",
                "relevance_score": round(hybrid_score, 4)
            })

        fused_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return fused_results[:top_k]

    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        results = self.retrieve_with_score(query, top_k=top_k)
        return [result["entity_id"] for result in results]

    # ------------------------------------------------------------------
    # Optional analysis utilities
    # ------------------------------------------------------------------

    def retrieve_detailed(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        返回更详细的检索分解，便于调试和论文展示
        """
        if not self._built:
            raise ValueError("Retriever index not built. Call build_index(graph) first.")

        bm25_results = self._bm25_search(query, top_k=max(top_k * 3, 20))
        semantic_results = self.vector_store.search(query, top_k=max(top_k * 3, 20))

        bm25_norm = self._normalize_scores(bm25_results)
        semantic_norm = self._normalize_scores(semantic_results)

        all_doc_ids = set(bm25_norm.keys()) | set(semantic_norm.keys())
        detailed = []

        for doc_id in all_doc_ids:
            bm25_score = bm25_norm.get(doc_id, 0.0)
            semantic_score = semantic_norm.get(doc_id, 0.0)
            doc_text = self.documents.get(doc_id, "")

            keyword_bonus = self._keyword_bonus(query, doc_text)

            type_bonus = 0.0
            node_name = ""
            node_type = ""
            node_path = ""

            if self.graph is not None and doc_id in self.graph.nodes:
                node = self.graph.nodes[doc_id]
                # 修复：使用 getattr 而不是 .get()
                node_name = self._safe_attr(node, "name", "")
                node_type = self._safe_attr(node, "type", "")
                node_path = self._safe_attr(node, "file_path", "")

                if node_type == "function":
                    type_bonus = 0.05
                elif node_type == "class":
                    type_bonus = 0.03
                elif node_type == "file":
                    type_bonus = 0.01

            hybrid_score = self.alpha * bm25_score + self.beta * semantic_score + keyword_bonus + type_bonus

            detailed.append({
                "node_id": doc_id,
                "node_name": node_name,
                "node_type": node_type,
                "file_path": node_path,
                "bm25_score": round(float(bm25_score), 4),
                "semantic_score": round(float(semantic_score), 4),
                "keyword_bonus": round(float(keyword_bonus), 4),
                "type_bonus": round(float(type_bonus), 4),
                "hybrid_score": round(float(hybrid_score), 4),
            })

        detailed.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return detailed[:top_k]