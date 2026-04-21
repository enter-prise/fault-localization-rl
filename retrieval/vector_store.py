# retrieval/vector_store.py

import math
import re
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any


class SimpleVectorStore:
    """
    轻量级语义检索实现（不依赖外部 embedding 模型）。

    设计目标：
    1. 用 token + tf-idf 风格向量近似实现轻量语义匹配
    2. 后续可以很容易替换成 sentence-transformers / OpenAI embedding / bge 等真实向量模型

    核心思想：
    - 把每个节点文本转为稀疏 token 权重向量
    - 用 cosine similarity 做相似度检索
    """

    def __init__(self):
        self.documents: Dict[str, str] = {}
        self.doc_vectors: Dict[str, Dict[str, float]] = {}
        self.idf: Dict[str, float] = {}
        self.doc_norms: Dict[str, float] = {}
        self.avg_doc_len: float = 0.0
        self._built = False

    # ------------------------------------------------------------------
    # Text processing
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []

        text = str(text).lower()
        text = text.replace("_", " ")
        text = text.replace("/", " ")
        text = text.replace("\\", " ")
        text = text.replace(".", " ")
        text = re.sub(r"[^a-zA-Z0-9\-\s]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return []

        return text.split()

    def _compute_idf(self, tokenized_docs: Dict[str, List[str]]) -> Dict[str, float]:
        num_docs = max(len(tokenized_docs), 1)
        df = Counter()

        for _, tokens in tokenized_docs.items():
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] += 1

        idf = {}
        for token, freq in df.items():
            # 平滑版 idf
            idf[token] = math.log((1 + num_docs) / (1 + freq)) + 1.0

        return idf

    def _vectorize_tokens(self, tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
        if not tokens:
            return {}

        tf = Counter(tokens)
        total = sum(tf.values()) or 1

        vec = {}
        for token, count in tf.items():
            tf_weight = count / total
            idf_weight = idf.get(token, 1.0)
            vec[token] = tf_weight * idf_weight

        return vec

    def _norm(self, vec: Dict[str, float]) -> float:
        return math.sqrt(sum(v * v for v in vec.values()))

    def _cosine(self, vec_a: Dict[str, float], vec_b: Dict[str, float], norm_a: Optional[float] = None, norm_b: Optional[float] = None) -> float:
        if not vec_a or not vec_b:
            return 0.0

        if norm_a is None:
            norm_a = self._norm(vec_a)
        if norm_b is None:
            norm_b = self._norm(vec_b)

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        # 遍历较小向量，提高效率
        if len(vec_a) > len(vec_b):
            vec_a, vec_b = vec_b, vec_a

        dot = 0.0
        for token, value in vec_a.items():
            if token in vec_b:
                dot += value * vec_b[token]

        return dot / (norm_a * norm_b)

    # ------------------------------------------------------------------
    # Build index
    # ------------------------------------------------------------------

    def build(self, documents: Dict[str, str]):
        """
        documents:
            {
                node_id: "text for indexing",
                ...
            }
        """
        self.documents = {str(k): (v or "") for k, v in documents.items()}

        tokenized_docs = {}
        total_len = 0

        for doc_id, text in self.documents.items():
            tokens = self._tokenize(text)
            tokenized_docs[doc_id] = tokens
            total_len += len(tokens)

        self.avg_doc_len = total_len / max(len(tokenized_docs), 1)
        self.idf = self._compute_idf(tokenized_docs)

        self.doc_vectors = {}
        self.doc_norms = {}

        for doc_id, tokens in tokenized_docs.items():
            vec = self._vectorize_tokens(tokens, self.idf)
            self.doc_vectors[doc_id] = vec
            self.doc_norms[doc_id] = self._norm(vec)

        self._built = True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        返回：
            [(doc_id, similarity_score), ...]
        """
        if not self._built:
            raise ValueError("SimpleVectorStore has not been built. Call build(documents) first.")

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        query_vec = self._vectorize_tokens(query_tokens, self.idf)
        query_norm = self._norm(query_vec)

        results = []
        for doc_id, doc_vec in self.doc_vectors.items():
            score = self._cosine(
                query_vec,
                doc_vec,
                norm_a=query_norm,
                norm_b=self.doc_norms.get(doc_id, None),
            )
            if score > 0:
                results.append((doc_id, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_vector(self, doc_id: str) -> Optional[Dict[str, float]]:
        return self.doc_vectors.get(str(doc_id))

    def has_doc(self, doc_id: str) -> bool:
        return str(doc_id) in self.documents