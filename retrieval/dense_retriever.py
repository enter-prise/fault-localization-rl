# retrieval/dense_retriever.py

"""
真正的 Dense Retrieval 实现（基于 Sentence-BERT）
提供语义级别的代码检索能力
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import pickle
import os
import hashlib


class DenseRetriever:
    """
    基于 Sentence-BERT 的稠密向量检索器
    真正的语义理解，能够理解代码的语义含义
    
    特点：
    1. 使用神经网络将文本转换为稠密向量
    2. 支持批量编码，提高效率
    3. 支持保存/加载索引，避免重复构建
    4. 自动检测 GPU 加速
    5. 支持本地模型路径（无需联网）
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ):
        """
        初始化 Dense Retriever
        
        Args:
            model_name: Sentence-BERT 模型名称（联网下载）
                推荐选项:
                - "all-MiniLM-L6-v2": 轻量级，384维，速度快（推荐）
                - "all-mpnet-base-v2": 效果好，768维，稍慢
                - "microsoft/codebert-base": 代码专用模型
            model_path: 本地模型路径（优先使用，无需联网）
            device: 运行设备 ("cuda" 或 "cpu")
            cache_dir: 模型缓存目录
        """
        # 优先使用本地路径
        if model_path is not None and os.path.exists(model_path):
            self.model_name = model_path
            self.use_local = True
            print(f"Using local model from: {model_path}")
        else:
            self.model_name = model_name or "all-MiniLM-L6-v2"
            self.use_local = False
            print(f"Using model: {self.model_name} (will download if needed)")
            
        self.device = device
        self.cache_dir = cache_dir
        
        self.documents: Dict[str, str] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.embedding_dim: Optional[int] = None
        self._built = False
        self._model = None
    
    @property
    def model(self):
        """懒加载模型，只在需要时加载"""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self):
        """加载 Sentence-BERT 模型"""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # 检查 GPU 可用性
            if self.device == "cuda" and not torch.cuda.is_available():
                print("Warning: CUDA not available, falling back to CPU")
                self.device = "cpu"
            
            print(f"Loading dense retriever model: {self.model_name}")
            print(f"Device: {self.device}")
            
            # 加载模型（支持本地路径）
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_dir,
            )
            
            self.embedding_dim = self._model.get_sentence_embedding_dimension()
            print(f"Model loaded, embedding dimension: {self.embedding_dim}")
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import sentence-transformers: {e}\n"
                "Please install: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
    def _build_document_text(self, doc_id: str, metadata: Optional[Dict] = None) -> str:
        """
        构建用于编码的文档文本
        可以添加元信息来提高检索质量
        """
        text = self.documents.get(doc_id, "")
        
        # 如果有元信息，可以增强文本
        if metadata:
            enhanced_parts = []
            if metadata.get("name"):
                enhanced_parts.append(f"Function: {metadata['name']}")
            if metadata.get("type"):
                enhanced_parts.append(f"Type: {metadata['type']}")
            if metadata.get("file_path"):
                enhanced_parts.append(f"File: {metadata['file_path']}")
            enhanced_parts.append(text)
            return " | ".join(enhanced_parts)
        
        return text
    
    def build(
        self, 
        documents: Dict[str, str],
        metadata: Optional[Dict[str, Dict]] = None,
        batch_size: int = 32,
        show_progress: bool = True,
    ):
        """
        构建索引：为所有文档生成 embedding
        
        Args:
            documents: {doc_id: document_text}
            metadata: {doc_id: metadata_dict} 可选的元信息
            batch_size: 批处理大小
            show_progress: 是否显示进度条
        """
        self.documents = {str(k): (v or "") for k, v in documents.items()}
        
        print(f"Building dense index for {len(self.documents)} documents...")
        
        # 准备文本列表
        doc_ids = list(self.documents.keys())
        texts = []
        for doc_id in doc_ids:
            meta = metadata.get(doc_id) if metadata else None
            text = self._build_document_text(doc_id, meta)
            texts.append(text)
        
        # 批量编码
        print(f"Encoding documents (batch_size={batch_size})...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # 归一化，便于余弦相似度计算
            convert_to_numpy=True,
        )
        
        # 存储 embeddings
        for doc_id, emb in zip(doc_ids, embeddings):
            self.embeddings[doc_id] = emb
        
        self._built = True
        print(f"Built dense index with {len(self.embeddings)} documents")
        print(f"Embedding shape: {embeddings.shape}")
    
    def search(
        self, 
        query: str, 
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        检索最相似的 top_k 文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            threshold: 相似度阈值，低于此值的结果将被过滤
        
        Returns:
            [(doc_id, similarity_score), ...]
        """
        if not self._built:
            raise ValueError(
                "DenseRetriever not built. Call build() first."
            )
        
        if not query or not query.strip():
            return []
        
        # 编码查询
        query_emb = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        
        # 计算相似度
        results = []
        for doc_id, doc_emb in self.embeddings.items():
            # 余弦相似度（向量已归一化，所以直接点积）
            similarity = float(np.dot(query_emb, doc_emb))
            
            if similarity >= threshold:
                results.append((doc_id, similarity))
        
        # 排序并返回 top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def search_batch(
        self,
        queries: List[str],
        top_k: int = 10,
        batch_size: int = 32,
    ) -> List[List[Tuple[str, float]]]:
        """
        批量检索多个查询
        
        Args:
            queries: 查询文本列表
            top_k: 每个查询返回结果数量
            batch_size: 批处理大小
        
        Returns:
            [ [(doc_id, score), ...], ... ]
        """
        if not self._built:
            raise ValueError("DenseRetriever not built. Call build() first.")
        
        # 批量编码查询
        query_embs = self.model.encode(
            queries,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        
        # 收集所有文档 embeddings
        doc_ids = list(self.embeddings.keys())
        doc_embeddings = np.array([self.embeddings[did] for did in doc_ids])
        
        # 批量计算相似度
        all_results = []
        for query_emb in query_embs:
            similarities = np.dot(doc_embeddings, query_emb)
            
            # 获取 top_k 索引
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                score = float(similarities[idx])
                if score > 0:
                    results.append((doc_ids[idx], score))
            
            all_results.append(results)
        
        return all_results
    
    def similarity(self, doc_id: str, query: str) -> float:
        """
        计算单个文档与查询的相似度
        """
        if not self._built:
            raise ValueError("DenseRetriever not built. Call build() first.")
        
        if doc_id not in self.embeddings:
            return 0.0
        
        query_emb = self.model.encode(query, normalize_embeddings=True)
        doc_emb = self.embeddings[doc_id]
        return float(np.dot(query_emb, doc_emb))
    
    def save(self, path: str):
        """
        保存索引到磁盘
        
        Args:
            path: 保存路径（不含扩展名）
        """
        if not self._built:
            raise ValueError("Cannot save: index not built")
        
        save_path = f"{path}.dense_index.pkl"
        
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved dense index to {save_path}")
        return save_path
    
    def load(self, path: str):
        """
        从磁盘加载索引
        
        Args:
            path: 保存路径（不含扩展名）
        """
        load_path = f"{path}.dense_index.pkl"
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Index file not found: {load_path}")
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.documents = data['documents']
        self.embeddings = data['embeddings']
        self.model_name = data['model_name']
        self.embedding_dim = data['embedding_dim']
        
        # 重新加载模型
        self._load_model()
        
        self._built = True
        print(f"Loaded dense index from {load_path}")
        print(f"Loaded {len(self.embeddings)} documents")
    
    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict] = None):
        """
        动态添加单个文档（会重新编码）
        """
        if not self._built:
            raise ValueError("Index not built")
        
        # 构建增强文本
        enhanced_text = self._build_document_text(doc_id, metadata) if metadata else text
        
        # 编码
        embedding = self.model.encode(
            enhanced_text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        
        self.documents[doc_id] = text
        self.embeddings[doc_id] = embedding
        
        print(f"Added document: {doc_id}")
    
    def remove_document(self, doc_id: str):
        """
        移除文档
        """
        if doc_id in self.documents:
            del self.documents[doc_id]
        if doc_id in self.embeddings:
            del self.embeddings[doc_id]
        print(f"Removed document: {doc_id}")
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取检索器信息
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dim": self.embedding_dim,
            "num_documents": len(self.documents),
            "built": self._built,
        }
    
    def __len__(self) -> int:
        return len(self.documents)
    
    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self.documents


class DenseRetrieverFactory:
    """
    Dense Retriever 工厂类
    方便创建不同配置的检索器
    """
    
    # 预定义模型配置
    MODELS = {
        "fast": {
            "name": "all-MiniLM-L6-v2",
            "dim": 384,
            "description": "Fast, lightweight model",
        },
        "balanced": {
            "name": "all-mpnet-base-v2",
            "dim": 768,
            "description": "Good balance of speed and quality",
        },
        "code": {
            "name": "microsoft/codebert-base",
            "dim": 768,
            "description": "Code-specific model",
        },
    }
    
    @classmethod
    def create(cls, model_type: str = "fast", model_path: Optional[str] = None, device: str = "cuda", **kwargs):
        """
        创建 Dense Retriever
        
        Args:
            model_type: "fast", "balanced", "code"
            model_path: 本地模型路径（优先使用）
            device: "cuda" or "cpu"
            **kwargs: 其他参数
        """
        # 如果提供了本地路径，优先使用
        if model_path and os.path.exists(model_path):
            return DenseRetriever(model_path=model_path, device=device, **kwargs)
        
        if model_type not in cls.MODELS:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available: {list(cls.MODELS.keys())}")
        
        model_name = cls.MODELS[model_type]["name"]
        return DenseRetriever(model_name=model_name, device=device, **kwargs)
    
    @classmethod
    def list_models(cls):
        """列出所有可用模型"""
        print("\nAvailable Dense Retrieval Models:")
        print("-" * 50)
        for model_type, info in cls.MODELS.items():
            print(f"  {model_type}: {info['name']}")
            print(f"      {info['description']}")
            print(f"      Dimension: {info['dim']}")
        print("-" * 50)


# 快速测试代码
if __name__ == "__main__":
    print("Testing DenseRetriever...")
    
    # 创建测试文档
    test_documents = {
        "doc1": "null pointer exception when accessing database connection",
        "doc2": "function to handle user authentication with JWT token",
        "doc3": "memory leak in cache implementation",
        "doc4": "database connection pool configuration error",
    }
    
    # 创建检索器（使用本地模型路径）
    # 优先检查本地模型是否存在
    local_model_path = "./dense_retriever_model"
    if os.path.exists(local_model_path):
        print(f"Using local model from: {local_model_path}")
        retriever = DenseRetriever(model_path=local_model_path, device="cpu")
    else:
        print("Local model not found, using default model")
        retriever = DenseRetriever(model_name="all-MiniLM-L6-v2", device="cpu")
    
    # 构建索引
    retriever.build(test_documents, show_progress=False)
    
    # 测试检索
    test_queries = [
        "database connection error",
        "authentication token",
        "memory issue",
    ]
    
    print("\n" + "=" * 60)
    print("Search Results:")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.search(query, top_k=2)
        for doc_id, score in results:
            print(f"  {doc_id}: {test_documents[doc_id][:50]}... (score={score:.4f})")
    
    print("\n✅ DenseRetriever test passed!")