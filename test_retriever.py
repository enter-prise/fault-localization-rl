#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from retrieval.dense_retriever import DenseRetriever
from graph.builder import GraphBuilder

def main():
    # 加载模型
    dense = DenseRetriever(model_path='./dense_retriever_model', device='cpu')
    
    # 构建图
    builder = GraphBuilder()
    graph = builder.build_from_repo('data_storage/repos/astropy')
    
    # 构建文档
    corpus = {}
    for node_id, node in graph.nodes.items():
        name = getattr(node, 'name', '') or ''
        doc = getattr(node, 'doc', '') or ''
        file_path = getattr(node, 'file_path', '') or ''
        code = getattr(node, 'code', '') or ''
        corpus[node_id] = f'{name} {doc} {file_path} {str(code)[:300]}'.strip()
    
    dense.build(corpus, show_progress=False)
    query = 'null pointer exception when accessing database'
    results = dense.search(query, top_k=10)
    
    print('Dense 检索 Top-10:')
    for i, (node_id, score) in enumerate(results[:10]):
        node = graph.nodes.get(node_id)
        name = getattr(node, 'name', '')
        print(f'{i+1}. {name}: {score:.4f}')

if __name__ == '__main__':
    main()