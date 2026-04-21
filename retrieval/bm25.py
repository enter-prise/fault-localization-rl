# retrieval/bm25.py

import math
import re
from collections import Counter
from typing import Dict, List, Tuple, Union


class SimpleBM25:
    """
    轻量 BM25 检索器

    支持接口：
    - retrieve(query, top_k=5) -> List[(doc_id, score)]
    - search(query, top_k=5) -> List[(doc_id, score)]
    - get_scores(query) -> Dict[doc_id, score]

    设计目标：
    1. 和 Retriever 保持兼容
    2. 对 fault localization 的代码文本更稳
    3. 不依赖第三方库
    """
# 构造函数：初始化 BM25 模型，接收语料库字典以及两个核心参数 k1（词频饱和度）和 b（长度惩罚）
    def __init__(self, corpus: Dict[str, str], k1: float = 1.5, b: float = 0.75):
        """
        corpus: {doc_id: text}
        """
        self.k1 = k1   # 保存 k1 参数，控制词频在得分中的饱和程度（默认1.5）
        self.b = b   # 保存 b 参数，控制文档长度对得分惩罚的力度（默认0.75）


        self.corpus: Dict[str, str] = {str(doc_id): (text or "") for doc_id, text in corpus.items()}

        self.doc_tokens: Dict[str, List[str]] = {}
        self.doc_len: Dict[str, int] = {}
        self.term_freqs: Dict[str, Counter] = {}  # 初始化字典：存储每个文档中各词元的词频（TF）
        self.doc_freqs: Counter = Counter()  # 初始化计数器：存储文档频率（DF），统计每个词元在多少个文档中出现过
        self.N = len(self.corpus) #记录文档总数
        self.avgdl = 0.0

        self._build()

    # ------------------------------------------------------------------
    # Tokenization
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

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self):
        self.doc_tokens = {}
        self.doc_len = {}
        self.term_freqs = {}
        self.doc_freqs = Counter()

        self.N = len(self.corpus)
        total_len = 0

        for doc_id, text in self.corpus.items():
            tokens = self._tokenize(text)
            self.doc_tokens[doc_id] = tokens
            self.doc_len[doc_id] = len(tokens)
            total_len += len(tokens)

            tf = Counter(tokens)
            self.term_freqs[doc_id] = tf

            for term in tf:
                self.doc_freqs[term] += 1

        self.avgdl = total_len / self.N if self.N > 0 else 0.0

    # ------------------------------------------------------------------
    # BM25 scoring
    # ------------------------------------------------------------------

    def _idf(self, term: str) -> float:
        df = self.doc_freqs.get(term, 0)
        if df == 0:
            return 0.0

        # 标准平滑版 BM25 IDF
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def _score(self, query_terms: List[str], doc_id: str) -> float:
        if not query_terms:
            return 0.0

        score = 0.0
        dl = self.doc_len.get(doc_id, 0)
        tf = self.term_freqs.get(doc_id, Counter())

        for term in query_terms:
            freq = tf.get(term, 0)
            if freq == 0:
                continue

            idf = self._idf(term)

            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl + 1e-9))

            score += idf * (numerator / (denominator + 1e-9))

        return float(score)

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------

    def get_scores(self, query: str) -> Dict[str, float]:
        """
        返回所有文档的 BM25 分数
        """
        query_terms = self._tokenize(query)
        if not query_terms:
            return {}

        scores: Dict[str, float] = {}
        for doc_id in self.corpus:
            score = self._score(query_terms, doc_id)
            if score > 0:
                scores[doc_id] = float(score)

        return scores

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        返回 top-k 检索结果
        """
        scores = self.get_scores(query)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        与 retrieve 保持同义接口，方便 Retriever 自动兼容
        """
        return self.retrieve(query, top_k=top_k)

    # ------------------------------------------------------------------
    # Optional utilities
    # ------------------------------------------------------------------

    def add_documents(self, new_corpus: Dict[str, str]):
        """
        增量加入新文档后重建索引
        """
        for doc_id, text in new_corpus.items():
            self.corpus[str(doc_id)] = text or ""
        self._build()

    def get_document(self, doc_id: str) -> str:
        return self.corpus.get(str(doc_id), "")

    def __len__(self):
        return self.N